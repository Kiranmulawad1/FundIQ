"""Retrieval pipeline — orchestrates dense + sparse + RRF + reranker.

Three modes share one entry point so callers can A/B compare them on the
same query without writing branching code:

  RetrievalMode.DENSE          dense (pgvector cosine) only
  RetrievalMode.HYBRID         dense + sparse + RRF fusion
  RetrievalMode.HYBRID_RERANK  dense + sparse + RRF + cross-encoder rerank

Same input contract, same output contract, different ranking quality.
This is the seam the thesis eval (Phase 5C) iterates over.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import Grant
from app.models.base import GrantPortal
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.sparse import sparse_search

if TYPE_CHECKING:
    from app.rag.cache import SemanticCache
    from app.rag.hyde import HyDEService

logger = get_logger(__name__)

DENSE_CANDIDATES = 50
SPARSE_CANDIDATES = 50
RERANK_INPUT_K = 50  # top-50 from RRF go to the cross-encoder
DEFAULT_FINAL_K = 5


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class _DenseHit:
    """Internal — dense-leg result with cosine similarity in [0, 1]."""

    grant_id: uuid.UUID
    similarity: float


@dataclass(slots=True)
class RetrievedHit:
    grant: Grant
    final_score: float
    # Provenance — what produced this rank? Lets the API surface and the
    # eval harness reason about retrieval quality by stage.
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float | None
    rerank_score: float | None


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    hits: list[RetrievedHit]
    mode: RetrievalMode
    dense_count: int
    sparse_count: int
    rrf_input_count: int
    rerank_input_count: int
    elapsed_ms: int
    # Phase 5B additions — provenance for the new pipeline knobs.
    used_hyde: bool = False
    hypotheticals: list[str] | None = None
    cache_hit: bool = False
    cached_for_query: str | None = None


# ---------------------------------------------------------------------------
# Dependencies (kept as protocols to keep the pipeline testable)
# ---------------------------------------------------------------------------
class EmbedderLike(Protocol):
    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class RerankerLike(Protocol):
    async def score_pairs(self, query: str, passages: list[str]) -> list[float]: ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RetrievalPipeline:
    """Stateless — pass the embedder + reranker per call."""

    def __init__(
        self,
        *,
        embedder: EmbedderLike,
        reranker: RerankerLike | None,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(
        self,
        session: AsyncSession,
        *,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID_RERANK,
        limit: int = DEFAULT_FINAL_K,
        portal: GrantPortal | None = None,
        country: str | None = None,
        use_hyde: bool = False,
        hyde_service: HyDEService | None = None,
        cache: SemanticCache | None = None,
    ) -> RetrievalResult:
        if mode is RetrievalMode.HYBRID_RERANK and self._reranker is None:
            msg = "HYBRID_RERANK mode requires a reranker; none was injected."
            raise ValueError(msg)
        if use_hyde and hyde_service is None:
            msg = "use_hyde=True requires a hyde_service; none was injected."
            raise ValueError(msg)

        started = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Embed the original query (used for cache lookup AND for the
        #    sparse leg, regardless of HyDE).
        # -------------------------------------------------------------
        qvecs = await self._embedder.embed_passages([f"query: {query}"])
        original_qvec = qvecs[0]

        # -------------------------------------------------------------
        # 1a. Cache lookup — only for HYBRID_RERANK without HyDE.
        #     Why exclude HyDE: the cache key is derived from the query
        #     embedding, which is the *original* vector before HyDE fires.
        #     Without this exclusion, a HyDE-on request would hit a
        #     HyDE-off cached entry and silently skip Gemini, lying about
        #     `used_hyde` in the response. Caching HyDE results properly
        #     requires keying on (embedding, use_hyde) — deferred to 5C.
        #     Other modes (dense / hybrid) are already fast enough that
        #     the Redis round-trip would dominate their latency.
        # -------------------------------------------------------------
        cache_eligible = (
            cache is not None
            and mode is RetrievalMode.HYBRID_RERANK
            and not use_hyde
        )
        if cache_eligible:
            assert cache is not None
            cached = await cache.lookup(original_qvec)
            if cached is not None:
                hits = await self._rehydrate_cached_hits(session, cached.result_blob)
                meta = cached.result_blob.get("meta", {})
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "rag.retrieve.cache_hit",
                    mode=mode.value,
                    query=query[:120],
                    elapsed_ms=elapsed_ms,
                )
                return RetrievalResult(
                    hits=hits,
                    mode=mode,
                    dense_count=int(meta.get("dense_count", 0)),
                    sparse_count=int(meta.get("sparse_count", 0)),
                    rrf_input_count=int(meta.get("rrf_input_count", 0)),
                    rerank_input_count=int(meta.get("rerank_input_count", 0)),
                    elapsed_ms=elapsed_ms,
                    used_hyde=bool(meta.get("used_hyde", False)),
                    hypotheticals=meta.get("hypotheticals"),
                    cache_hit=True,
                    cached_for_query=cached.cached_for_query,
                )

        # -------------------------------------------------------------
        # 1b. HyDE — if enabled, generate hypotheticals and mean-pool
        #     their embeddings to form the DENSE query vector. The sparse
        #     leg keeps the original query string (keyword signal matters).
        # -------------------------------------------------------------
        hypotheticals: list[str] | None = None
        if use_hyde:
            assert hyde_service is not None
            hypotheticals = await hyde_service.generate_hypotheticals(query)
            if hypotheticals and len(hypotheticals) > 0:
                hvecs = await self._embedder.embed_passages(
                    [f"passage: {h}" for h in hypotheticals]
                )
                qvec = _mean_pool_normalised(hvecs)
            else:
                # HyDE returned nothing usable — fall back to the original vector.
                qvec = original_qvec
        else:
            qvec = original_qvec

        # -------------------------------------------------------------
        # 2. Fetch dense candidates always (it's cheap thanks to HNSW)
        # -------------------------------------------------------------
        if mode is RetrievalMode.DENSE:
            dense_hits = await self._dense_search(
                session, qvec=qvec, limit=limit, portal=portal, country=country,
            )
            sparse_hits: list = []  # unused in DENSE mode
        else:
            # Sequential — a single AsyncSession is NOT safe for concurrent
            # operations (SQLAlchemy raises "session is provisioning a new
            # connection"). Both queries are indexed (HNSW + GIN trigram)
            # so the latency cost of sequential is small. If we ever need
            # parallelism, the fix is to pass two distinct sessions in.
            dense_hits = await self._dense_search(
                session, qvec=qvec, limit=DENSE_CANDIDATES,
                portal=portal, country=country,
            )
            sparse_hits = await sparse_search(
                session, query=query, limit=SPARSE_CANDIDATES,
                portal=portal, country=country,
            )

        # -------------------------------------------------------------
        # 3. Choose ranking strategy by mode
        # -------------------------------------------------------------
        if mode is RetrievalMode.DENSE:
            ranked_ids = [h.grant_id for h in dense_hits[:limit]]
            ranking_meta = {
                "dense_ranks": {h.grant_id: i for i, h in enumerate(dense_hits)},
                "sparse_ranks": {},
                "rrf_scores": {},
                "rerank_scores": {},
            }
            rrf_input_count = 0
            rerank_input_count = 0
        else:
            dense_ids = [h.grant_id for h in dense_hits]
            sparse_ids = [h.grant_id for h in sparse_hits]
            fused = reciprocal_rank_fusion(dense_ids, sparse_ids)
            rrf_input_count = len(fused)

            if mode is RetrievalMode.HYBRID:
                ranked_ids = [f.grant_id for f in fused[:limit]]
                ranking_meta = {
                    "dense_ranks": {f.grant_id: f.dense_rank for f in fused if f.dense_rank is not None},
                    "sparse_ranks": {f.grant_id: f.sparse_rank for f in fused if f.sparse_rank is not None},
                    "rrf_scores": {f.grant_id: f.rrf_score for f in fused},
                    "rerank_scores": {},
                }
                rerank_input_count = 0
            else:
                # HYBRID_RERANK: take top-50 from RRF, rerank with cross-encoder
                assert self._reranker is not None
                rerank_input = fused[:RERANK_INPUT_K]
                rerank_input_count = len(rerank_input)
                rerank_ids = [f.grant_id for f in rerank_input]
                grants_by_id = await self._fetch_grants(session, rerank_ids)
                ordered_grants = [grants_by_id[gid] for gid in rerank_ids if gid in grants_by_id]
                # Score each candidate by query × (title + summary) — body is too long
                passages = [f"{g.title}\n{g.summary}" for g in ordered_grants]
                scores = await self._reranker.score_pairs(query, passages)
                # Re-pair, sort by reranker score descending
                paired = sorted(
                    zip([g.id for g in ordered_grants], scores, strict=False),
                    key=lambda x: x[1],
                    reverse=True,
                )
                ranked_ids = [gid for gid, _ in paired[:limit]]
                rerank_scores = {gid: s for gid, s in paired}
                ranking_meta = {
                    "dense_ranks": {f.grant_id: f.dense_rank for f in fused if f.dense_rank is not None},
                    "sparse_ranks": {f.grant_id: f.sparse_rank for f in fused if f.sparse_rank is not None},
                    "rrf_scores": {f.grant_id: f.rrf_score for f in fused},
                    "rerank_scores": rerank_scores,
                }

        # -------------------------------------------------------------
        # 4. Fetch the actual Grant rows for the final ranked list
        # -------------------------------------------------------------
        grants_by_id = await self._fetch_grants(session, ranked_ids)

        # -------------------------------------------------------------
        # 5. Build typed result
        # -------------------------------------------------------------
        final_hits: list[RetrievedHit] = []
        for gid in ranked_ids:
            g = grants_by_id.get(gid)
            if g is None:
                continue
            rerank_score = ranking_meta["rerank_scores"].get(gid)
            rrf_score = ranking_meta["rrf_scores"].get(gid)
            if rerank_score is not None:
                final_score = float(rerank_score)
            elif rrf_score is not None:
                final_score = float(rrf_score)
            else:
                # Pure dense mode — use cosine sim from the typed dense hits.
                final_score = next(
                    (h.similarity for h in dense_hits if h.grant_id == gid),
                    0.0,
                )
            # The ranking_meta sub-dicts are populated from FusedHit fields
            # (int | None at runtime) but the literal-dict construction in
            # branches 156-209 makes Pyright infer them too widely. The
            # ignores below are scoped to that inference gap only.
            final_hits.append(
                RetrievedHit(
                    grant=g,
                    final_score=final_score,
                    dense_rank=ranking_meta["dense_ranks"].get(gid),  # type: ignore[arg-type]
                    sparse_rank=ranking_meta["sparse_ranks"].get(gid),  # type: ignore[arg-type]
                    rrf_score=float(rrf_score) if rrf_score is not None else None,
                    rerank_score=float(rerank_score) if rerank_score is not None else None,
                )
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "rag.retrieve",
            mode=mode.value,
            query=query[:120],
            use_hyde=use_hyde,
            dense_n=len(dense_hits),
            sparse_n=len(sparse_hits),
            rrf_n=rrf_input_count,
            rerank_n=rerank_input_count,
            returned=len(final_hits),
            elapsed_ms=elapsed_ms,
        )

        # -------------------------------------------------------------
        # Cache STORE — only for HYBRID_RERANK and only on cache miss.
        # We store the serialisable result blob (grant_ids + scores), not
        # the SQL-bound Grant objects.
        # -------------------------------------------------------------
        if cache_eligible:
            assert cache is not None
            await cache.store(
                query=query,
                query_embedding=original_qvec,
                result_blob={
                    "hits": [
                        {
                            "grant_id": str(h.grant.id),
                            "final_score": h.final_score,
                            "dense_rank": h.dense_rank,
                            "sparse_rank": h.sparse_rank,
                            "rrf_score": h.rrf_score,
                            "rerank_score": h.rerank_score,
                        }
                        for h in final_hits
                    ],
                    "meta": {
                        "dense_count": len(dense_hits),
                        "sparse_count": len(sparse_hits),
                        "rrf_input_count": rrf_input_count,
                        "rerank_input_count": rerank_input_count,
                        "used_hyde": use_hyde,
                        "hypotheticals": hypotheticals,
                    },
                },
            )

        return RetrievalResult(
            hits=final_hits,
            mode=mode,
            dense_count=len(dense_hits),
            sparse_count=len(sparse_hits),
            rrf_input_count=rrf_input_count,
            rerank_input_count=rerank_input_count,
            elapsed_ms=elapsed_ms,
            used_hyde=use_hyde,
            hypotheticals=hypotheticals,
            cache_hit=False,
            cached_for_query=None,
        )

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def _dense_search(
        session: AsyncSession,
        *,
        qvec: list[float],
        limit: int,
        portal: GrantPortal | None,
        country: str | None,
    ) -> list[_DenseHit]:
        qstr = "[" + ",".join(f"{x:.7f}" for x in qvec) + "]"
        conditions = ["deleted_at IS NULL", "embedding IS NOT NULL"]
        params: dict[str, object] = {"limit": limit}
        if portal is not None:
            conditions.append("portal = :portal")
            params["portal"] = portal.value.upper()
        if country is not None:
            conditions.append("country = :country")
            params["country"] = country.upper()
        sql = text(
            f"""
            SELECT id, (1.0 - (embedding <=> '{qstr}'::vector))::float8 AS similarity
            FROM grants
            WHERE {" AND ".join(conditions)}
            ORDER BY embedding <=> '{qstr}'::vector
            LIMIT :limit
            """
        )
        rows = (await session.execute(sql, params)).mappings().all()
        return [
            _DenseHit(grant_id=r["id"], similarity=float(r["similarity"]))
            for r in rows
        ]

    @staticmethod
    async def _fetch_grants(
        session: AsyncSession, ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, Grant]:
        if not ids:
            return {}
        from sqlalchemy import select

        rows = (
            await session.execute(select(Grant).where(Grant.id.in_(ids)))  # type: ignore[attr-defined]
        ).scalars().all()
        return {g.id: g for g in rows}

    @classmethod
    async def _rehydrate_cached_hits(
        cls,
        session: AsyncSession,
        result_blob: dict[str, object],
    ) -> list[RetrievedHit]:
        """Turn a serialised cache blob back into `RetrievedHit`s.

        Cache stores grant IDs + scores; we look up the live Grant rows so
        the response always reflects the current title/summary/etc. (a
        scrape might have updated content since the cache entry was
        stored). The scores are preserved as-is.
        """
        raw_hits = result_blob.get("hits", [])
        if not isinstance(raw_hits, list):
            return []
        ids: list[uuid.UUID] = []
        for h in raw_hits:
            if not isinstance(h, dict):
                continue
            raw_id = h.get("grant_id")
            if isinstance(raw_id, str):
                try:
                    ids.append(uuid.UUID(raw_id))
                except ValueError:
                    continue
        grants_by_id = await cls._fetch_grants(session, ids)
        out: list[RetrievedHit] = []
        for h in raw_hits:
            if not isinstance(h, dict):
                continue
            raw_id = h.get("grant_id")
            if not isinstance(raw_id, str):
                continue
            try:
                gid = uuid.UUID(raw_id)
            except ValueError:
                continue
            g = grants_by_id.get(gid)
            if g is None:
                # Grant was deleted since the cache entry was stored. Skip.
                continue
            out.append(
                RetrievedHit(
                    grant=g,
                    final_score=float(h.get("final_score", 0.0) or 0.0),
                    dense_rank=_opt_int(h.get("dense_rank")),
                    sparse_rank=_opt_int(h.get("sparse_rank")),
                    rrf_score=_opt_float(h.get("rrf_score")),
                    rerank_score=_opt_float(h.get("rerank_score")),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _mean_pool_normalised(vectors: list[list[float]]) -> list[float]:
    """Mean-pool a list of unit-normalised embeddings, renormalise the result.

    HyDE mean-pools the per-hypothetical embeddings; that vector still
    needs to be unit-norm so pgvector's `<=>` operator gives a clean
    cosine ranking. Empty input returns a zero vector — caller should
    have fallen back to the original query embedding before calling this.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    summed = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            summed[i] += v[i]
    n = float(len(vectors))
    meaned = [s / n for s in summed]
    norm = sum(x * x for x in meaned) ** 0.5
    if norm == 0.0:
        return meaned
    return [x / norm for x in meaned]


def _opt_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _opt_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

"""Public read endpoints for the grants corpus.

Three concerns, separated:
  - GET /grants            list + filter + paginate + sort
  - GET /grants/{id}       full detail (includes body)
  - POST /grants/search    hybrid retrieval (dense + sparse + RRF + reranker)

The search endpoint is the seam Phase 5 grows into. Modes are toggleable
per request so the eval harness can A/B compare retrieval strategies on
the same query without writing branching code.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from sqlalchemy import desc, func, select
from sqlalchemy.sql import ColumnElement

from app.api.deps import SessionDep
from app.core.logging import get_logger
from app.models import Grant
from app.models.base import GrantPortal, GrantStatus, Sector
from app.rag.pipeline import RetrievalMode, RetrievedHit
from app.schemas.grants import (
    Citation,
    GrantDetail,
    GrantListItem,
    GrantListResponse,
    GrantSearchHit,
    GrantSearchRequest,
    GrantSearchResponse,
    PageMeta,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/grants", tags=["grants"])


SortKey = Literal["created_at", "deadline", "funding_max"]


@router.get(
    "",
    response_model=GrantListResponse,
    summary="List grants with optional filters",
)
async def list_grants(
    session: SessionDep,
    portal: Annotated[GrantPortal | None, Query()] = None,
    status_: Annotated[GrantStatus | None, Query(alias="status")] = None,
    sector: Annotated[Sector | None, Query()] = None,
    country: Annotated[str | None, Query(min_length=2, max_length=2)] = None,
    federal_state: Annotated[str | None, Query(max_length=64)] = None,
    min_funding: Annotated[float | None, Query(ge=0)] = None,
    max_funding: Annotated[float | None, Query(ge=0)] = None,
    sort: Annotated[SortKey, Query()] = "created_at",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> GrantListResponse:
    """Filtered + paginated list. Returns compact items; for full body use
    `/grants/{id}`.
    """
    conditions: list[ColumnElement[bool]] = [Grant.deleted_at.is_(None)]  # type: ignore[attr-defined]
    if portal is not None:
        conditions.append(Grant.portal == portal)  # type: ignore[arg-type]
    if status_ is not None:
        conditions.append(Grant.status == status_)  # type: ignore[arg-type]
    if sector is not None:
        conditions.append(Grant.sector == sector)  # type: ignore[arg-type]
    if country is not None:
        conditions.append(Grant.country == country)  # type: ignore[arg-type]
    if federal_state is not None:
        conditions.append(Grant.federal_state == federal_state)  # type: ignore[arg-type]
    if min_funding is not None:
        conditions.append(Grant.funding_max_eur >= min_funding)  # type: ignore[operator,arg-type]
    if max_funding is not None:
        conditions.append(Grant.funding_max_eur <= max_funding)  # type: ignore[operator,arg-type]

    sort_col: ColumnElement
    if sort == "deadline":
        # nulls last for ascending deadline (open-ended programs go to bottom).
        sort_col = Grant.deadline.asc().nulls_last()  # type: ignore[attr-defined]
    elif sort == "funding_max":
        sort_col = desc(Grant.funding_max_eur).nulls_last()  # type: ignore[arg-type]
    else:
        sort_col = desc(Grant.created_at)  # type: ignore[arg-type]

    base = select(Grant).where(*conditions)
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(base.order_by(sort_col).offset(offset).limit(limit))
    ).scalars().all()

    items = [_to_list_item(g) for g in rows]
    return GrantListResponse(
        items=items,
        page=PageMeta(total=total, limit=limit, offset=offset, returned=len(items)),
    )


@router.get(
    "/{grant_id}",
    response_model=GrantDetail,
    summary="Full grant detail (includes body, eligibility, citations)",
)
async def get_grant(
    session: SessionDep,
    grant_id: Annotated[uuid.UUID, Path()],
) -> GrantDetail:
    g = await session.get(Grant, grant_id)
    if g is None or g.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Grant {grant_id} not found")
    return _to_detail(g)


@router.post(
    "/search",
    response_model=GrantSearchResponse,
    summary="Hybrid semantic search (dense + sparse + RRF + cross-encoder reranker)",
)
async def search_grants(
    request: Request,
    session: SessionDep,
    body: GrantSearchRequest,
) -> GrantSearchResponse:
    """Three retrieval modes share this endpoint:
      `dense`         — pgvector cosine over multilingual-e5-large.
      `hybrid`        — dense + sparse pg_trgm, fused via Reciprocal Rank Fusion.
      `hybrid_rerank` — hybrid → top-50 → BGE-reranker-v2-m3 cross-encoder.

    Phase 5B adds HyDE query rewriting + semantic cache atop this pipeline.
    """
    from app.rag.pipeline import RetrievalPipeline

    embedder = _resolve_embedder(request)
    reranker = _resolve_reranker(request) if body.mode is RetrievalMode.HYBRID_RERANK else None
    hyde_service = _resolve_hyde(request) if body.use_hyde else None
    cache = _resolve_cache(request) if body.mode is RetrievalMode.HYBRID_RERANK else None

    pipeline = RetrievalPipeline(embedder=embedder, reranker=reranker)
    result = await pipeline.retrieve(
        session,
        query=body.query,
        mode=body.mode,
        limit=body.limit,
        portal=body.portal,
        country=body.country,
        use_hyde=body.use_hyde,
        hyde_service=hyde_service,
        cache=cache,
    )

    hits = [_to_search_hit(h) for h in result.hits]
    return GrantSearchResponse(
        query=body.query,
        mode=result.mode,
        hits=hits,
        elapsed_ms=result.elapsed_ms,
        dense_count=result.dense_count,
        sparse_count=result.sparse_count,
        rrf_input_count=result.rrf_input_count,
        rerank_input_count=result.rerank_input_count,
        used_hyde=result.used_hyde,
        hypotheticals=result.hypotheticals,
        cache_hit=result.cache_hit,
        cached_for_query=result.cached_for_query,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_list_item(g: Grant) -> GrantListItem:
    return GrantListItem(
        id=g.id,
        portal=g.portal,
        status=g.status,
        title=g.title,
        title_en=g.title_en,
        summary=g.summary,
        sector=g.sector,
        country=g.country,
        federal_state=g.federal_state,
        funding_min_eur=float(g.funding_min_eur) if g.funding_min_eur is not None else None,
        funding_max_eur=float(g.funding_max_eur) if g.funding_max_eur is not None else None,
        deadline=g.deadline,
        opens_at=g.opens_at,
        source_url=g.source_url,
        source_doc_id=g.source_doc_id,
        created_at=g.created_at,
        updated_at=g.updated_at,
    )


def _to_detail(g: Grant) -> GrantDetail:
    return GrantDetail(
        **_to_list_item(g).model_dump(),
        body=g.body,
        summary_en=g.summary_en,
        eligibility=g.eligibility,
        metadata=g.metadata_,
    )


def _to_search_hit(hit: RetrievedHit) -> GrantSearchHit:
    """Materialise a pipeline `RetrievedHit` into the API response shape.

    This is the only place that knows how to assemble `GrantSearchHit` +
    citation from a Grant + provenance metadata. Keeping it in one place
    means future ranking modes (HyDE, sub-doc chunks) only touch this
    function, not every caller.
    """
    g = hit.grant
    base = _to_list_item(g).model_dump()
    return GrantSearchHit(
        **base,
        final_score=hit.final_score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        rrf_score=hit.rrf_score,
        rerank_score=hit.rerank_score,
        citation=Citation(
            grant_id=g.id,
            source_doc_id=g.source_doc_id,
            source_url=g.source_url,
            portal=g.portal,
            title=g.title,
        ),
    )


def _resolve_embedder(request: Request):  # type: ignore[no-untyped-def]
    """Reuse the scheduler's embedder when running; otherwise build one."""
    existing = getattr(request.app.state, "scheduler_embedder", None)
    if existing is not None:
        return existing

    # Lazy build — torch import is heavy.
    from app.services.embedding import EmbeddingService

    embedder = EmbeddingService(redis=getattr(request.app.state, "redis", None))
    # Cache on app.state so subsequent requests share it.
    request.app.state.scheduler_embedder = embedder
    return embedder


def _resolve_reranker(request: Request):  # type: ignore[no-untyped-def]
    """Reuse the process-level reranker when present, else build + cache one.

    Lazy because the model is ~2.3GB. Cached on app.state so we pay the
    load cost at most once per process.
    """
    existing = getattr(request.app.state, "reranker", None)
    if existing is not None:
        return existing

    from app.rag.reranker import RerankerService

    reranker = RerankerService()
    request.app.state.reranker = reranker
    return reranker


def _resolve_hyde(request: Request):  # type: ignore[no-untyped-def]
    """Process-level HyDE service. Reuses a single httpx client."""
    existing = getattr(request.app.state, "hyde", None)
    if existing is not None:
        return existing

    from app.rag.hyde import HyDEService

    hyde = HyDEService()
    request.app.state.hyde = hyde
    return hyde


def _resolve_cache(request: Request):  # type: ignore[no-untyped-def]
    """Semantic query cache backed by the lifespan-managed Redis client.

    Returns None if Redis is somehow absent (e.g. tests that bypass
    lifespan), so the pipeline's `cache_eligible` check stays correct.
    """
    existing = getattr(request.app.state, "rag_cache", None)
    if existing is not None:
        return existing
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return None

    from app.rag.cache import SemanticCache

    cache = SemanticCache(redis=redis)
    request.app.state.rag_cache = cache
    return cache

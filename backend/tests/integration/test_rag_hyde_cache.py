"""Integration tests for Phase 5B: HyDE + semantic cache.

Stubs Gemini and the reranker so the suite stays fast and offline. The
embedder is a deterministic stub; the cache uses the real Redis container
since `redis-py` mocking is more painful than it's worth and Redis is
already running in our test stack.
"""

from __future__ import annotations

import math
import uuid
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus
from app.rag.cache import SemanticCache
from app.rag.pipeline import RetrievalMode, RetrievalPipeline


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _vec(seed: int) -> list[float]:
    """Deterministic unit-norm vector — `seed` controls which direction."""
    base = (seed % 7) + 1
    raw = [float(base + i % 3) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _StubEmbedder:
    """Embeds passages deterministically: the first digit in the text picks
    the seed, defaulting to `default_seed`."""

    def __init__(self, *, default_seed: int = 2) -> None:
        self.default_seed = default_seed
        self.call_count = 0

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        out: list[list[float]] = []
        for t in texts:
            seed = self.default_seed
            for c in t:
                if c.isdigit():
                    seed = int(c)
                    break
            out.append(_vec(seed))
        return out


class _StubReranker:
    """Scores passages containing the query string higher than others."""

    def __init__(self) -> None:
        self.call_count = 0

    async def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        self.call_count += 1
        return [10.0 if query.lower() in p.lower() else 1.0 for p in passages]


class _StubHyDE:
    """Returns canned hypotheticals + tracks call count."""

    def __init__(self, descriptions: list[str] | None = None) -> None:
        self.descriptions = descriptions or [
            "Hypothetical 1 about founder stipends.",
            "Hypothetical 2 about academic founders.",
            "Hypothetical 3 about DeepTech research grants.",
        ]
        self.call_count = 0

    async def generate_hypotheticals(self, query: str, *, n: int = 3) -> list[str]:
        self.call_count += 1
        return self.descriptions[:n]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
async def _seed_grants(session: AsyncSession) -> list[Grant]:
    grants: list[Grant] = []
    for i in range(1, 4):
        g = Grant(
            portal=GrantPortal.EXIST,
            title=f"HyDETest grant {i} — query word",
            summary=f"Summary {i}",
            body=f"Body {i}",
            status=GrantStatus.OPEN,
            country="DE",
            funding_max_eur=Decimal("100000") * i,
            source_url=f"https://hyde-test.example/{i}",
            source_doc_id=f"hyde-test-{i}",
            source_hash=f"hash-hyde-{i}",
            embedding=_vec(i),
        )
        session.add(g)
        grants.append(g)
    await session.flush()
    for g in grants:
        await session.refresh(g)
    await session.commit()
    return grants


@pytest.fixture
async def fresh_cache(client) -> SemanticCache:  # type: ignore[no-untyped-def]
    """Returns a SemanticCache backed by the app's real Redis, cleared per test."""
    redis = client._transport.app.state.redis  # type: ignore[attr-defined]
    cache = SemanticCache(redis=redis, max_entries=50)
    await cache.clear()
    yield cache
    await cache.clear()


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_pipeline_use_hyde_calls_service_and_re_embeds(
    db_session: AsyncSession,
) -> None:
    await _seed_grants(db_session)
    embedder = _StubEmbedder(default_seed=2)
    hyde = _StubHyDE()
    pipeline = RetrievalPipeline(embedder=embedder, reranker=_StubReranker())

    result = await pipeline.retrieve(
        db_session,
        query="founder stipend question",
        mode=RetrievalMode.HYBRID_RERANK,
        limit=3,
        use_hyde=True,
        hyde_service=hyde,  # type: ignore[arg-type]
    )

    assert hyde.call_count == 1
    assert result.used_hyde is True
    assert result.hypotheticals is not None
    assert len(result.hypotheticals) == 3
    # Embedder is called twice: once for the original query, once for the
    # 3 hypothetical passages (in a single batched call).
    assert embedder.call_count == 2


@pytest.mark.integration
async def test_pipeline_use_hyde_without_service_raises(
    db_session: AsyncSession,
) -> None:
    pipeline = RetrievalPipeline(embedder=_StubEmbedder(), reranker=_StubReranker())
    with pytest.raises(ValueError, match="hyde_service"):
        await pipeline.retrieve(
            db_session,
            query="x",
            mode=RetrievalMode.HYBRID_RERANK,
            limit=3,
            use_hyde=True,
        )


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_pipeline_cache_miss_then_hit_short_circuits(
    db_session: AsyncSession,
    fresh_cache: SemanticCache,
) -> None:
    await _seed_grants(db_session)
    embedder = _StubEmbedder(default_seed=2)
    reranker = _StubReranker()
    pipeline = RetrievalPipeline(embedder=embedder, reranker=reranker)

    # First call: cache miss → reranker invoked, result stored.
    r1 = await pipeline.retrieve(
        db_session,
        query="cache test query",
        mode=RetrievalMode.HYBRID_RERANK,
        limit=3,
        cache=fresh_cache,
    )
    assert r1.cache_hit is False
    assert reranker.call_count == 1

    # Second call with the *same* query: cache hit → reranker NOT invoked.
    r2 = await pipeline.retrieve(
        db_session,
        query="cache test query",
        mode=RetrievalMode.HYBRID_RERANK,
        limit=3,
        cache=fresh_cache,
    )
    assert r2.cache_hit is True
    assert r2.cached_for_query == "cache test query"
    assert reranker.call_count == 1  # unchanged → cache short-circuited
    # The cached hit IDs match the original ranked output.
    assert [h.grant.id for h in r2.hits] == [h.grant.id for h in r1.hits]


@pytest.mark.integration
async def test_pipeline_cache_only_engages_on_hybrid_rerank(
    db_session: AsyncSession,
    fresh_cache: SemanticCache,
) -> None:
    """DENSE and HYBRID modes skip the cache entirely — they're already fast."""
    await _seed_grants(db_session)
    pipeline = RetrievalPipeline(embedder=_StubEmbedder(), reranker=None)

    # Two DENSE calls — never cached.
    for _ in range(2):
        r = await pipeline.retrieve(
            db_session, query="x", mode=RetrievalMode.DENSE, limit=3, cache=fresh_cache,
        )
        assert r.cache_hit is False

    # Cache is still empty (ZSET has no members).
    cached = await fresh_cache.lookup(_vec(2))
    assert cached is None


@pytest.mark.integration
async def test_semantic_cache_lookup_uses_cosine_threshold(
    fresh_cache: SemanticCache,
) -> None:
    """A query embedding within sim>0.95 of a stored one should hit."""
    # Store a result against a known embedding.
    base_vec = _vec(2)
    await fresh_cache.store(
        query="stored query",
        query_embedding=base_vec,
        result_blob={"hits": [], "meta": {}},
    )

    # Exact lookup should hit.
    hit = await fresh_cache.lookup(base_vec)
    assert hit is not None
    assert hit.cached_for_query == "stored query"

    # An orthogonal vector should miss.
    orth = [0.0] * EMBEDDING_DIM
    orth[0] = 1.0
    miss = await fresh_cache.lookup(orth)
    assert miss is None


@pytest.mark.integration
async def test_semantic_cache_evicts_when_full(fresh_cache: SemanticCache) -> None:
    """ZSET trim keeps only the newest `max_entries` (3 for this small test)."""
    cache = SemanticCache(redis=fresh_cache._redis, max_entries=3)  # type: ignore[attr-defined]
    await cache.clear()

    for i in range(5):
        v = _vec(i)
        await cache.store(
            query=f"q-{i}",
            query_embedding=v,
            result_blob={"hits": [], "meta": {"i": i}},
        )

    # Only 3 entries remain in the index after ZREMRANGEBYRANK.
    idx_count = await cache._redis.zcard(cache._redis.connection_pool.connection_kwargs and "rag:cache:index" or "rag:cache:index")  # type: ignore[attr-defined]
    # Simpler equivalent:
    idx_count = await fresh_cache._redis.zcard("rag:cache:index")  # type: ignore[attr-defined]
    assert idx_count == 3

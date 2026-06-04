"""Integration tests for the hybrid retrieval pipeline.

We test the three modes against real Postgres (with HNSW + GIN trigram
indexes) but stub the embedder and reranker so the suite stays fast and
doesn't load multi-gigabyte models.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus
from app.rag.pipeline import RetrievalMode, RetrievalPipeline


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _StubEmbedder:
    """Deterministic, seed-controlled embeddings."""

    def __init__(self, *, default_seed: int = 1) -> None:
        self.default_seed = default_seed

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(self._seed_for(t)) for t in texts]

    def _seed_for(self, text: str) -> int:
        # Heuristic — match against test grants seeded with the same digit
        # in their title (see fixture below).
        for c in text:
            if c.isdigit():
                return int(c)
        return self.default_seed

    @staticmethod
    def _vec(seed: int) -> list[float]:
        base = (seed % 7) + 1
        return [1.0 / (EMBEDDING_DIM * base) ** 0.5] * EMBEDDING_DIM


class _StubReranker:
    """Returns higher scores for passages that contain `query`."""

    def __init__(self) -> None:
        self.call_count = 0

    async def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        self.call_count += 1
        return [10.0 if query.lower() in p.lower() else 1.0 for p in passages]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _vec(seed: int) -> list[float]:
    base = (seed % 7) + 1
    return [1.0 / (EMBEDDING_DIM * base) ** 0.5] * EMBEDDING_DIM


async def _seed_three_grants(session: AsyncSession) -> Sequence[Grant]:
    grants: list[Grant] = []
    for i in range(1, 4):
        g = Grant(
            portal=GrantPortal.EXIST,
            title=f"PipelineTest grant {i} — stipend exists",
            summary=f"Body for pipeline test {i}.",
            body=f"Long body {i}.",
            status=GrantStatus.OPEN,
            country="DE",
            funding_max_eur=Decimal("100000") * i,
            source_url=f"https://pipeline-test.example/{i}",
            source_doc_id=f"pipeline-test-{i}",
            source_hash=f"hash-pipeline-{i}",
            embedding=_vec(i),
        )
        session.add(g)
        grants.append(g)
    await session.flush()
    for g in grants:
        await session.refresh(g)
    await session.commit()
    return grants


# ---------------------------------------------------------------------------
# Pipeline directly
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_pipeline_dense_mode_returns_hits(db_session: AsyncSession) -> None:
    await _seed_three_grants(db_session)
    pipeline = RetrievalPipeline(embedder=_StubEmbedder(default_seed=2), reranker=None)
    result = await pipeline.retrieve(
        db_session, query="2", mode=RetrievalMode.DENSE, limit=3,
    )
    assert result.mode is RetrievalMode.DENSE
    assert len(result.hits) >= 1
    # Dense mode populates dense_rank but not rrf/rerank.
    top = result.hits[0]
    assert top.dense_rank is not None
    assert top.rrf_score is None
    assert top.rerank_score is None
    assert 0.0 <= top.final_score <= 1.0


@pytest.mark.integration
async def test_pipeline_hybrid_mode_populates_rrf_provenance(
    db_session: AsyncSession,
) -> None:
    await _seed_three_grants(db_session)
    pipeline = RetrievalPipeline(embedder=_StubEmbedder(default_seed=2), reranker=None)
    # Use a query that the sparse leg can match against title (`PipelineTest`).
    result = await pipeline.retrieve(
        db_session, query="PipelineTest", mode=RetrievalMode.HYBRID, limit=3,
    )
    assert result.mode is RetrievalMode.HYBRID
    assert result.rrf_input_count >= 1
    top = result.hits[0]
    assert top.rrf_score is not None
    assert top.rerank_score is None


@pytest.mark.integration
async def test_pipeline_hybrid_rerank_uses_reranker(db_session: AsyncSession) -> None:
    await _seed_three_grants(db_session)
    reranker = _StubReranker()
    pipeline = RetrievalPipeline(
        embedder=_StubEmbedder(default_seed=2), reranker=reranker,
    )
    result = await pipeline.retrieve(
        db_session, query="PipelineTest", mode=RetrievalMode.HYBRID_RERANK, limit=3,
    )
    assert reranker.call_count == 1
    assert result.rerank_input_count >= 1
    top = result.hits[0]
    assert top.rerank_score is not None


@pytest.mark.integration
async def test_pipeline_hybrid_rerank_raises_without_reranker(
    db_session: AsyncSession,
) -> None:
    pipeline = RetrievalPipeline(embedder=_StubEmbedder(), reranker=None)
    with pytest.raises(ValueError, match="reranker"):
        await pipeline.retrieve(
            db_session, query="x", mode=RetrievalMode.HYBRID_RERANK, limit=3,
        )


# ---------------------------------------------------------------------------
# /grants/search end-to-end (HTTP)
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_grants_search_hybrid_mode_via_http(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_three_grants(db_session)
    from app.api.routes import grants as grants_route

    monkeypatch.setattr(grants_route, "_resolve_embedder", lambda _r: _StubEmbedder(default_seed=2))

    r = await client.post(
        "/grants/search",
        json={"query": "PipelineTest grant", "limit": 3, "mode": "hybrid"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "hybrid"
    assert body["rrf_input_count"] >= 1
    assert body["rerank_input_count"] == 0
    for hit in body["hits"]:
        assert hit["citation"]["grant_id"] == hit["id"]
        assert hit["citation"]["source_url"].startswith("https://pipeline-test")


@pytest.mark.integration
async def test_grants_search_hybrid_rerank_via_http(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_three_grants(db_session)
    from app.api.routes import grants as grants_route

    monkeypatch.setattr(grants_route, "_resolve_embedder", lambda _r: _StubEmbedder(default_seed=2))
    monkeypatch.setattr(grants_route, "_resolve_reranker", lambda _r: _StubReranker())

    r = await client.post(
        "/grants/search",
        json={"query": "PipelineTest grant", "limit": 3, "mode": "hybrid_rerank"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "hybrid_rerank"
    assert body["rerank_input_count"] >= 1
    top = body["hits"][0]
    # The stub reranker scores higher for passages containing the query string.
    assert top["rerank_score"] is not None

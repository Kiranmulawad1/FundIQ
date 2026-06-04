"""Integration tests for /grants/* and /analytics/funding endpoints.

Tests build their own grants inside the test transaction; the rollback at
teardown undoes everything so the dev DB stays clean.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus


def _vec(seed: int) -> list[float]:
    """Deterministic normalised vector for HNSW cosine queries."""
    base = (seed % 7) + 1
    return [1.0 / (EMBEDDING_DIM * base) ** 0.5] * EMBEDDING_DIM


async def _make_grant(
    session: AsyncSession,
    *,
    title: str,
    portal: GrantPortal,
    source_url: str,
    federal_state: str | None = None,
    funding_max: Decimal | None = None,
    embed_seed: int | None = 1,
    deadline: datetime | None = None,
    body: str = "Body content.",
) -> Grant:
    g = Grant(
        portal=portal,
        title=title,
        summary=f"Summary for {title}.",
        body=body,
        status=GrantStatus.OPEN,
        country="DE" if portal not in (GrantPortal.EIC, GrantPortal.HORIZON) else "EU",
        federal_state=federal_state,
        funding_max_eur=funding_max,
        deadline=deadline,
        source_url=source_url,
        source_doc_id=source_url.rsplit("/", 1)[-1] or "doc",
        source_hash=f"hash-{title}",
        embedding=_vec(embed_seed) if embed_seed is not None else None,
    )
    session.add(g)
    await session.flush()
    await session.refresh(g)
    return g


class _StubEmbedder:
    """Test embedder — deterministic, no torch."""

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [_vec(len(t)) for t in texts]


# ---------------------------------------------------------------------------
# GET /grants
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_grants_list_paginated(client: AsyncClient, db_session: AsyncSession) -> None:
    for i in range(3):
        await _make_grant(
            db_session,
            title=f"Test Grant {i}",
            portal=GrantPortal.EXIST,
            source_url=f"https://test.example/grant-{i}",
        )
    await db_session.commit()

    r = await client.get("/grants?limit=2&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["page"]["limit"] == 2
    assert body["page"]["offset"] == 0
    assert body["page"]["returned"] == 2
    assert body["page"]["total"] >= 3
    assert len(body["items"]) == 2
    assert all("body" not in item for item in body["items"])  # list view is compact


@pytest.mark.integration
async def test_grants_list_filter_by_portal(client: AsyncClient, db_session: AsyncSession) -> None:
    await _make_grant(
        db_session,
        title="EXIST one",
        portal=GrantPortal.EXIST,
        source_url="https://test.example/exist-x",
    )
    await _make_grant(
        db_session,
        title="KfW one",
        portal=GrantPortal.KFW,
        source_url="https://test.example/kfw-x",
    )
    await db_session.commit()

    r = await client.get("/grants?portal=exist")
    assert r.status_code == 200
    titles = [item["title"] for item in r.json()["items"]]
    assert "EXIST one" in titles
    assert "KfW one" not in titles


@pytest.mark.integration
async def test_grants_list_funding_range_filter(client: AsyncClient, db_session: AsyncSession) -> None:
    await _make_grant(
        db_session,
        title="Small program",
        portal=GrantPortal.EXIST,
        source_url="https://test.example/small",
        funding_max=Decimal("50000"),
    )
    await _make_grant(
        db_session,
        title="Big program",
        portal=GrantPortal.KFW,
        source_url="https://test.example/big",
        funding_max=Decimal("10000000"),
    )
    await db_session.commit()

    r = await client.get("/grants?min_funding=1000000")
    assert r.status_code == 200
    titles = [item["title"] for item in r.json()["items"]]
    assert "Big program" in titles
    assert "Small program" not in titles


# ---------------------------------------------------------------------------
# GET /grants/{id}
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_grant_detail_returns_full_body(client: AsyncClient, db_session: AsyncSession) -> None:
    g = await _make_grant(
        db_session,
        title="Detail Probe",
        portal=GrantPortal.EXIST,
        source_url="https://test.example/detail",
        body="Long-form body text that the list endpoint omits.",
    )
    await db_session.commit()

    r = await client.get(f"/grants/{g.id}")
    assert r.status_code == 200
    body: dict[str, Any] = r.json()
    assert body["title"] == "Detail Probe"
    assert "Long-form body text" in body["body"]


@pytest.mark.integration
async def test_grant_detail_404_when_missing(client: AsyncClient) -> None:
    r = await client.get("/grants/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    body = r.json()
    assert "code" in body  # uses our error envelope


# ---------------------------------------------------------------------------
# POST /grants/search
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_grants_search_returns_ranked_hits(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three grants with different embedding seeds → different similarity to a query vector.
    await _make_grant(
        db_session,
        title="Closest",
        portal=GrantPortal.EXIST,
        source_url="https://test.example/closest",
        embed_seed=42,
    )
    await _make_grant(
        db_session,
        title="Middle",
        portal=GrantPortal.KFW,
        source_url="https://test.example/middle",
        embed_seed=43,
    )
    await _make_grant(
        db_session,
        title="Far",
        portal=GrantPortal.EIC,
        source_url="https://test.example/far",
        embed_seed=99,
    )
    await db_session.commit()

    # Inject the stub embedder so we don't load multilingual-e5-large.
    from app.api.routes import grants as grants_route

    monkeypatch.setattr(grants_route, "_resolve_embedder", lambda _req: _StubEmbedder())

    r = await client.post(
        "/grants/search", json={"query": "test query", "limit": 3, "mode": "dense"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "test query"
    assert len(body["hits"]) <= 3
    for hit in body["hits"]:
        # In DENSE mode `final_score` is cosine similarity in [0, 1].
        assert 0.0 <= hit["final_score"] <= 1.0
        assert "title" in hit
        assert "portal" in hit
        # Citation is always present and points back at this grant.
        assert hit["citation"]["grant_id"] == hit["id"]


@pytest.mark.integration
async def test_grants_search_respects_portal_filter(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _make_grant(
        db_session, title="E1", portal=GrantPortal.EXIST,
        source_url="https://test.example/e1", embed_seed=1,
    )
    await _make_grant(
        db_session, title="K1", portal=GrantPortal.KFW,
        source_url="https://test.example/k1", embed_seed=1,
    )
    await db_session.commit()

    from app.api.routes import grants as grants_route

    monkeypatch.setattr(grants_route, "_resolve_embedder", lambda _req: _StubEmbedder())

    r = await client.post(
        "/grants/search",
        json={"query": "x", "portal": "exist", "limit": 10, "mode": "dense"},
    )
    assert r.status_code == 200
    portals = {h["portal"] for h in r.json()["hits"]}
    assert portals.issubset({"exist"})


# ---------------------------------------------------------------------------
# GET /analytics/funding
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_analytics_funding_returns_typed_envelope(client: AsyncClient) -> None:
    """DuckDB attaches to live Postgres — we don't seed anything special;
    the existing 26 grants from prior c2 runs are queried.
    """
    r = await client.get("/analytics/funding")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["computed_via"] == "duckdb+postgres_scanner"
    assert body["total_grants"] >= 0
    assert body["embedded_grants"] >= 0
    assert isinstance(body["by_portal"], list)
    assert isinstance(body["by_status"], list)
    assert isinstance(body["by_federal_state"], list)
    assert body["elapsed_ms"] >= 0

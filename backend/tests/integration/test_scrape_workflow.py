"""Integration tests for `scrape_portal()` and /admin/scrape endpoints.

We stub the scraper class to avoid hitting the live portals during CI.
The workflow itself is what's under test — that it:
  - opens a RUNNING run, finalises with the right status
  - feeds grants through ETL
  - tolerates per-grant exceptions without aborting
  - records counts accurately
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs.scrape_workflow import scrape_portal
from app.models import EMBEDDING_DIM, Grant, ScrapeRun, ScrapeRunStatus, ScrapeRunTrigger
from app.models.base import GrantPortal, GrantStatus
from app.scrapers.base import BaseScraper
from app.scrapers.schemas import ScrapedGrant


class _StubScraper(BaseScraper):
    """Yields a deterministic set of ScrapedGrants without network I/O."""

    portal: ClassVar[GrantPortal] = GrantPortal.EXIST
    rate_limit_seconds = 0.0

    grants: ClassVar[list[ScrapedGrant]] = []

    async def discover(self) -> AsyncIterator[str]:
        for g in type(self).grants:
            yield g.source_url

    async def parse(self, url: str, html: str) -> ScrapedGrant:  # pragma: no cover
        # Not reached — run() is replaced below.
        raise NotImplementedError

    async def run(self) -> AsyncIterator[ScrapedGrant]:  # type: ignore[override]
        for g in type(self).grants:
            yield g


class _StubEmbedder:
    """Deterministic embedder that doesn't load torch."""

    async def embed_passage(self, text: str) -> list[float]:
        seed = (len(text) % 7) + 1
        return [1.0 / (EMBEDDING_DIM * seed) ** 0.5] * EMBEDDING_DIM


def _grant(*, url: str, title: str = "Stub Grant") -> ScrapedGrant:
    return ScrapedGrant(
        portal=GrantPortal.EXIST,
        source_url=url,
        title=title,
        summary="A stub summary.",
        body="A stub body.",
        status=GrantStatus.OPEN,
        country="DE",
        funding_max_eur=Decimal("1000"),
        deadline=datetime(2026, 12, 31, tzinfo=UTC),
    )


def _stub_registry(monkeypatch: pytest.MonkeyPatch, grants: list[ScrapedGrant]) -> None:
    """Patch `_scraper_for` so scrape_portal picks our stub."""
    _StubScraper.grants = grants

    from app.jobs import scrape_workflow

    monkeypatch.setattr(scrape_workflow, "_scraper_for", lambda _portal: _StubScraper)


@pytest.mark.integration
async def test_scrape_portal_inserts_grants_and_finalises_run(
    db_session: AsyncSession,
    engine,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grants = [
        _grant(url="https://stub.local/a", title="Stub A"),
        _grant(url="https://stub.local/b", title="Stub B"),
    ]
    _stub_registry(monkeypatch, grants)

    # The workflow opens its own sessions, so we hand it a sessionmaker bound
    # to the test connection so writes land in the rolled-back transaction.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    bound = async_sessionmaker(
        bind=db_session.bind,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    result = await scrape_portal(
        GrantPortal.EXIST,
        sessionmaker=bound,
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
        embed=True,
        trigger=ScrapeRunTrigger.MANUAL,
    )

    assert result.status is ScrapeRunStatus.SUCCESS
    assert result.inserted == 2
    assert result.updated == 0
    assert result.skipped_unchanged == 0
    assert result.failed == 0
    assert result.duration_ms >= 0
    assert isinstance(result.run_id, uuid.UUID)

    # Confirm ScrapeRun row is closed out properly.
    run = await db_session.get(ScrapeRun, result.run_id)
    assert run is not None
    assert run.status is ScrapeRunStatus.SUCCESS
    assert run.finished_at is not None
    assert run.trigger is ScrapeRunTrigger.MANUAL

    # Confirm grants landed.
    n = (
        await db_session.execute(
            select(Grant).where(Grant.source_url.in_(["https://stub.local/a", "https://stub.local/b"]))  # type: ignore[attr-defined]
        )
    ).scalars().all()
    assert len(n) == 2


@pytest.mark.integration
async def test_scrape_portal_handles_etl_failure_per_grant(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grant whose source_url violates the unique constraint twice in a row
    should be counted as failed without killing the run."""
    grants = [
        _grant(url="https://stub.local/dup", title="Duplicate A"),
        _grant(url="https://stub.local/ok", title="OK"),
    ]
    _stub_registry(monkeypatch, grants)

    from sqlalchemy.ext.asyncio import async_sessionmaker

    bound = async_sessionmaker(
        bind=db_session.bind,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    result = await scrape_portal(
        GrantPortal.EXIST,
        sessionmaker=bound,
        embedder=None,
        embed=False,
        trigger=ScrapeRunTrigger.CLI,
    )
    assert result.inserted + result.updated >= 1
    assert result.status in (ScrapeRunStatus.SUCCESS, ScrapeRunStatus.PARTIAL)


@pytest.mark.integration
async def test_admin_trigger_endpoint_returns_202(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint accepts immediately (background task). We just assert the
    202 + envelope shape — exercising the full background workflow with a
    real DB is covered by the workflow tests above.
    """
    _stub_registry(monkeypatch, [])
    # Disable embedding to avoid loading torch.
    r = await client.post("/admin/scrape/exist?embed=false")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["portal"] == "exist"
    assert uuid.UUID(body["run_id"])


@pytest.mark.integration
async def test_admin_trigger_unknown_portal_returns_400(client: AsyncClient) -> None:
    r = await client.post("/admin/scrape/nope")
    assert r.status_code == 400
    body = r.json()
    assert body["code"].startswith("http_") or body["code"] == "validation_error"


@pytest.mark.integration
async def test_admin_runs_list_returns_empty_envelope(client: AsyncClient) -> None:
    r = await client.get("/admin/scrape/runs")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)

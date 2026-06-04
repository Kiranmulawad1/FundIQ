"""Integration tests for grant enrichment.

Stubs the Gemini agent client so the suite stays fast + offline. Uses
the same SAVEPOINT-bound session as the rest of the integration suite
so the writes don't leak into the real DB.
"""

from __future__ import annotations

import math
import uuid
from decimal import Decimal
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus, Sector
from app.schemas.enrichment import CURRENT_ENRICHMENT_VERSION, GrantEnrichment


# ---------------------------------------------------------------------------
# Stub LLM — minimal surface: respond_as for GrantEnrichment only.
# ---------------------------------------------------------------------------
class _StubLLM:
    def __init__(self, *, enrichment: GrantEnrichment | None = None) -> None:
        self._enrichment = enrichment or GrantEnrichment(
            sector=Sector.DEEPTECH,
            secondary_sectors=[Sector.HARDWARE],
            federal_state=None,
            target_groups=["academic founders", "research-based startups"],
            eligibility_criteria=[
                "Founder must be affiliated with a German university.",
                "Project must have an R&D component.",
            ],
            funding_phases=["idea phase", "pre-seed"],
            funding_form="stipend",
            application_notes="Rolling applications; verify enrolment status.",
        )
        self.calls = 0

    async def respond_as(self, model_cls: type, **_kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        if model_cls is GrantEnrichment:
            self.calls += 1
            return self._enrichment
        raise AssertionError(f"Unexpected model_cls: {model_cls}")

    async def __aexit__(self, *_: object) -> None:
        return None


def _vec(seed: int) -> list[float]:
    base = (seed % 7) + 1
    raw = [float(base + i % 3) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


async def _seed_grant(session: AsyncSession, **overrides: Any) -> Grant:
    g = Grant(
        portal=GrantPortal.EXIST,
        title="EnrichTest grant",
        summary="Summary text.",
        body="Body discussing academic founder eligibility and R&D requirements.",
        status=GrantStatus.OPEN,
        country="DE",
        funding_max_eur=Decimal("100000"),
        source_url="https://enrich-test.example/1",
        source_doc_id="enrich-test-1",
        source_hash="hash-enrich-1",
        embedding=_vec(1),
    )
    for k, v in overrides.items():
        setattr(g, k, v)
    session.add(g)
    await session.flush()
    await session.refresh(g)
    await session.commit()
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_enrich_single_grant_writes_structured_fields(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grant = await _seed_grant(db_session)
    app = client._transport.app  # type: ignore[attr-defined]
    stub = _StubLLM()
    app.state.agent_llm = stub

    r = await client.post(f"/admin/grants/{grant.id}/enrich")
    assert r.status_code == 200, r.text
    assert stub.calls == 1

    refreshed = await db_session.get(Grant, grant.id)
    assert refreshed is not None
    # Sector got populated because grant.sector was None.
    assert refreshed.sector is Sector.DEEPTECH
    # Eligibility JSONB carries the full structured payload + version stamp.
    elig = refreshed.eligibility
    assert elig["funding_form"] == "stipend"
    assert "academic founders" in elig["target_groups"]
    assert elig["enrichment_version"] == CURRENT_ENRICHMENT_VERSION
    assert "enriched_at" in elig


@pytest.mark.integration
async def test_enrich_preserves_existing_sector(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """The scraper's value wins — enrichment never overwrites a populated
    sector / federal_state.
    """
    grant = await _seed_grant(db_session, sector=Sector.HEALTH)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.agent_llm = _StubLLM(
        enrichment=GrantEnrichment(
            sector=Sector.DEEPTECH,  # different from existing
            target_groups=["x"],
        ),
    )

    await client.post(f"/admin/grants/{grant.id}/enrich")
    refreshed = await db_session.get(Grant, grant.id)
    assert refreshed is not None
    # Untouched — scraper-provided value is authoritative.
    assert refreshed.sector is Sector.HEALTH


@pytest.mark.integration
async def test_bulk_enrich_is_idempotent(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Second bulk call with the same version should skip every grant.

    The integration DB already has scraped grants from earlier suites;
    we assert on the delta between two bulk runs rather than on exact
    totals so the test is isolation-safe.
    """
    for i in range(3):
        await _seed_grant(
            db_session,
            title=f"Grant {i}",
            source_url=f"https://enrich-test.example/idem-{i}",
            source_doc_id=f"enrich-idem-{i}",
            source_hash=f"hash-idem-{i}",
        )
    app = client._transport.app  # type: ignore[attr-defined]
    stub = _StubLLM()
    app.state.agent_llm = stub

    r1 = await client.post("/admin/grants/enrich?per_call_delay_seconds=0")
    assert r1.status_code == 200
    body1 = r1.json()
    # Our 3 seeded grants must be in the enriched count. Other grants may
    # arrive pre-enriched from a prior live run — be isolation-tolerant.
    assert body1["enriched"] >= 3
    assert stub.calls == body1["enriched"]
    # Invariant: every grant ends in one of the buckets.
    assert body1["total"] == body1["enriched"] + body1["skipped"] + body1["failed"]

    # Second call — every previously-enriched grant must now be skipped.
    r2 = await client.post("/admin/grants/enrich?per_call_delay_seconds=0")
    body2 = r2.json()
    assert body2["enriched"] == 0
    assert body2["skipped"] >= body1["enriched"]
    assert stub.calls == body1["enriched"]  # no new LLM calls


@pytest.mark.integration
async def test_bulk_enrich_force_re_runs(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    app = client._transport.app  # type: ignore[attr-defined]
    stub = _StubLLM()
    app.state.agent_llm = stub

    body1 = (await client.post("/admin/grants/enrich")).json()
    first_calls = stub.calls
    assert first_calls == body1["enriched"]

    body2 = (await client.post("/admin/grants/enrich?force=true&per_call_delay_seconds=0")).json()
    # force=true re-runs every grant — even ones the first call skipped
    # (because they were pre-enriched at the current version).
    assert body2["enriched"] == body1["enriched"] + body1["skipped"]
    assert body2["skipped"] == 0
    assert stub.calls == first_calls + body2["enriched"]


@pytest.mark.integration
async def test_enrich_unknown_grant_returns_404(client: AsyncClient) -> None:
    r = await client.post(f"/admin/grants/{uuid.uuid4()}/enrich")
    assert r.status_code == 404

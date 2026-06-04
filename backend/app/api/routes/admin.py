"""Admin endpoints — cost dashboard, scrape control, run history.

Phase 1: cost stub.
Phase 2 (c3): scrape control + run history.
Phase 5+: agent traces, eval results dashboard.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.api.deps import SessionDep
from app.core.auth import current_user
from app.core.logging import get_logger
from app.jobs.scrape_workflow import scrape_portal
from app.models import ScrapeRun, ScrapeRunStatus, ScrapeRunTrigger
from app.models.base import GrantPortal
from app.services.grant_enrichment import apply_enrichment, bulk_enrich, enrich_grant

logger = get_logger(__name__)

# Every admin endpoint requires a real authenticated user. Setting the
# dependency at the router level inherits to all child routes — no need
# to add `_: CurrentUserDep` to every handler. When CLERK_SECRET_KEY is
# unset (local dev / tests) `current_user` returns the synthetic dev
# user, so the test suite still passes without monkey-patching auth.
router = APIRouter(dependencies=[Depends(current_user)])


# ---------------------------------------------------------------------------
# Cost stub (Phase 1)
# ---------------------------------------------------------------------------
class TokenUsage(BaseModel):
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usd_cost: float = Field(ge=0)


class CostsResponse(BaseModel):
    status: Literal["stub"] = "stub"
    period_start: datetime
    period_end: datetime
    total_usd: float
    by_model: list[TokenUsage]
    note: str = "Real cost rollup lands in Phase 5 (Logfire/LangSmith integration)."


@router.get(
    "/costs",
    response_model=CostsResponse,
    summary="Token cost dashboard (stub — Phase 5 wires real data)",
)
async def costs() -> CostsResponse:
    now = datetime.now(UTC)
    return CostsResponse(
        period_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
        period_end=now,
        total_usd=0.0,
        by_model=[],
    )


# ---------------------------------------------------------------------------
# Scrape control (Phase 2 c3)
# ---------------------------------------------------------------------------
class ScrapeTriggerResponse(BaseModel):
    accepted: Literal[True] = True
    portal: GrantPortal
    run_id: uuid.UUID
    detail: str = "Scrape started in the background. Poll /admin/scrape/runs."


class ScrapeRunView(BaseModel):
    id: uuid.UUID
    portal: GrantPortal
    trigger: ScrapeRunTrigger
    status: ScrapeRunStatus
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    inserted: int
    updated: int
    skipped_unchanged: int
    failed: int
    embedded: bool
    error: str | None
    error_type: str | None


class ScrapeRunsResponse(BaseModel):
    runs: list[ScrapeRunView]


_VALID_PORTALS = {p.value for p in GrantPortal}


@router.post(
    "/scrape/{portal}",
    response_model=ScrapeTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a scrape for one portal (runs in background)",
)
async def trigger_scrape(
    request: Request,
    portal: Annotated[str, Path(description="Portal slug, e.g. 'exist'")],
    background_tasks: BackgroundTasks,
    embed: Annotated[bool, Query()] = True,
) -> ScrapeTriggerResponse:
    """Fire-and-forget — returns 202 immediately, the run materialises in
    `scrape_runs` and can be observed via `/admin/scrape/runs`.
    """
    if portal not in _VALID_PORTALS:
        msg = f"Unknown portal {portal!r}. Valid: {sorted(_VALID_PORTALS)}"
        raise HTTPException(status_code=400, detail=msg)
    portal_enum = GrantPortal(portal)

    sessionmaker = request.app.state.sessionmaker
    embedder = getattr(request.app.state, "scheduler_embedder", None)
    if embed and embedder is None:
        # Build a one-off embedder for manual triggers when the scheduler is off.
        from app.services.embedding import EmbeddingService

        embedder = EmbeddingService(redis=request.app.state.redis)

    # Pre-allocate a run id so the caller has something to poll on.
    pre_run_id = uuid.uuid4()

    async def _runner() -> None:
        try:
            result = await scrape_portal(
                portal_enum,
                sessionmaker=sessionmaker,
                embedder=embedder if embed else None,
                embed=embed,
                trigger=ScrapeRunTrigger.MANUAL,
            )
            logger.info(
                "admin.scrape.complete",
                portal=portal_enum.value,
                run_id=str(result.run_id),
                status=result.status.value,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("admin.scrape.crashed", portal=portal_enum.value, error_type=type(e).__name__)

    background_tasks.add_task(_runner)

    return ScrapeTriggerResponse(portal=portal_enum, run_id=pre_run_id)


# ---------------------------------------------------------------------------
# Grant enrichment (LLM backfill of sector / eligibility / funding_form)
# ---------------------------------------------------------------------------
class EnrichResultView(BaseModel):
    grant_id: str
    status: Literal["enriched", "skipped", "failed"]
    detail: str = ""


class EnrichBulkResponse(BaseModel):
    total: int
    enriched: int
    skipped: int
    failed: int
    results: list[EnrichResultView]


def _resolve_llm(request: Request):  # type: ignore[no-untyped-def]
    """Reuse the process-level GeminiAgentClient set up by the agents
    surface. Constructed lazily on first call there so the slot may be
    None if no agent request has been made yet — build one here in that
    case and cache it on app.state.
    """
    from app.agents.llm import GeminiAgentClient

    existing = getattr(request.app.state, "agent_llm", None)
    if existing is not None:
        return existing
    llm = GeminiAgentClient()
    request.app.state.agent_llm = llm
    return llm


@router.post(
    "/grants/enrich",
    response_model=EnrichBulkResponse,
    summary="Backfill sector / eligibility / funding_form on all grants via LLM",
)
async def enrich_grants_bulk(
    request: Request,
    session: SessionDep,
    force: Annotated[bool, Query(description="Re-enrich grants already at the current version")] = False,
    limit: Annotated[int | None, Query(ge=1, le=500, description="Cap how many grants to process")] = None,
    per_call_delay_seconds: Annotated[
        float,
        Query(ge=0, le=60, description="Sleep between Gemini calls; default paces under free-tier 10 RPM."),
    ] = 7.0,
) -> EnrichBulkResponse:
    """Idempotent: skips grants already at CURRENT_ENRICHMENT_VERSION
    unless `force=true`. Run from /docs after a scrape to populate the
    structured fields that the agent graph relies on.
    """
    llm = _resolve_llm(request)
    results = await bulk_enrich(
        session, llm=llm, force=force, limit=limit,
        per_call_delay_seconds=per_call_delay_seconds,
    )
    return EnrichBulkResponse(
        total=len(results),
        enriched=sum(1 for r in results if r.status == "enriched"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        failed=sum(1 for r in results if r.status == "failed"),
        results=[EnrichResultView(**asdict(r)) for r in results],
    )


@router.post(
    "/grants/{grant_id}/enrich",
    response_model=EnrichResultView,
    summary="Enrich one grant (useful for testing / one-off retries)",
)
async def enrich_one_grant(
    request: Request,
    session: SessionDep,
    grant_id: Annotated[uuid.UUID, Path()],
) -> EnrichResultView:
    from app.models import Grant
    grant = await session.get(Grant, grant_id)
    if grant is None or grant.deleted_at is not None:
        raise HTTPException(status_code=404, detail=f"Grant {grant_id} not found")
    llm = _resolve_llm(request)
    try:
        enrichment = await enrich_grant(grant, llm=llm)
    except Exception as e:  # noqa: BLE001
        return EnrichResultView(grant_id=str(grant_id), status="failed", detail=str(e)[:300])
    apply_enrichment(grant, enrichment)
    await session.commit()
    return EnrichResultView(grant_id=str(grant_id), status="enriched")


# ---------------------------------------------------------------------------
# Grant re-embedding (run once after switching embedding providers)
# ---------------------------------------------------------------------------
class ReEmbedResponse(BaseModel):
    total: int
    embedded: int
    failed: int
    elapsed_ms: int


@router.post(
    "/grants/re-embed",
    response_model=ReEmbedResponse,
    summary="Re-compute every Grant.embedding via the current EmbeddingService.",
)
async def re_embed_grants(
    request: Request,
    session: SessionDep,
    limit: Annotated[int | None, Query(ge=1, le=10_000, description="Cap how many grants to re-embed.")] = None,
    per_call_delay_seconds: Annotated[
        float,
        Query(ge=0, le=60, description="Sleep between Gemini calls; default paces under free tier."),
    ] = 0.3,
) -> ReEmbedResponse:
    """Re-embed the entire corpus through the active EmbeddingService.

    When the embedder swapped from `multilingual-e5-large` (1024-dim,
    local) to Gemini Embedding (1024-dim, hosted) the vector space
    changed even though the dimension didn't — pre-existing rows still
    have e5 vectors and no longer match the query embedding. Run this
    endpoint once to overwrite them, then the search/agent surface
    works again.

    Idempotent enough — running twice just produces identical vectors
    (modulo Gemini's tiny per-call noise) at the cost of more API
    calls. Use `limit` to do incremental partial runs while you debug.
    """
    import asyncio
    import time as _time

    from app.models import Grant

    embedder = _resolve_embedder(request)

    started = _time.perf_counter()
    stmt = select(Grant).where(Grant.deleted_at.is_(None))  # type: ignore[attr-defined]
    if limit is not None:
        stmt = stmt.limit(limit)
    rows = (await session.execute(stmt)).scalars().all()

    embedded = 0
    failed = 0
    for i, grant in enumerate(rows):
        # Same embedding text the ETL uses (title + summary + body) so
        # we don't drift from the scrape-time vector definition.
        try:
            vec = await embedder.embed_passage(
                f"passage: {grant.title}\n\n{grant.summary or ''}\n\n{grant.body or ''}",
            )
            grant.embedding = vec
            embedded += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("re_embed.failed", grant_id=str(grant.id), error=str(e)[:200])
            failed += 1
        if per_call_delay_seconds > 0 and i + 1 < len(rows):
            await asyncio.sleep(per_call_delay_seconds)

    await session.commit()
    elapsed_ms = int((_time.perf_counter() - started) * 1000)
    logger.info(
        "re_embed.done",
        total=len(rows), embedded=embedded, failed=failed, elapsed_ms=elapsed_ms,
    )
    return ReEmbedResponse(total=len(rows), embedded=embedded, failed=failed, elapsed_ms=elapsed_ms)


def _resolve_embedder(request: Request):  # type: ignore[no-untyped-def]
    """Reuse the lifespan-built EmbeddingService when present; otherwise
    construct one. Cached on app.state for subsequent admin calls.
    """
    existing = getattr(request.app.state, "scheduler_embedder", None)
    if existing is not None:
        return existing
    from app.services.embedding import EmbeddingService

    redis = getattr(request.app.state, "redis", None)
    embedder = EmbeddingService(redis=redis)
    request.app.state.scheduler_embedder = embedder
    return embedder


@router.get(
    "/scrape/runs",
    response_model=ScrapeRunsResponse,
    summary="Recent scrape runs (default: last 50, newest first)",
)
async def list_scrape_runs(
    session: SessionDep,
    portal: Annotated[str | None, Query(description="Filter by portal slug")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> ScrapeRunsResponse:
    # SQLModel surfaces field annotations as their runtime types, which
    # confuses Pyright into thinking `started_at` is a plain `datetime`.
    # At runtime it's a SQLAlchemy `InstrumentedAttribute`. The runtime
    # behaviour is correct; the type checker is wrong here.
    stmt = select(ScrapeRun).order_by(desc(ScrapeRun.started_at)).limit(limit)  # type: ignore[arg-type]
    if portal is not None:
        if portal not in _VALID_PORTALS:
            raise HTTPException(status_code=400, detail=f"Unknown portal {portal!r}")
        stmt = stmt.where(ScrapeRun.portal == GrantPortal(portal))  # type: ignore[arg-type]
    rows = (await session.execute(stmt)).scalars().all()
    return ScrapeRunsResponse(runs=[ScrapeRunView.model_validate(r, from_attributes=True) for r in rows])

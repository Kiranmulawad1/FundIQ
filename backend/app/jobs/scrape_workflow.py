"""Canonical scraper workflow.

`scrape_portal(...)` is the **single entry point** for every scraping
execution in the system. CLI, scheduler, and admin endpoints all call
this function. Keeping it singular means:
  - one place owns "how a scrape behaves end-to-end"
  - the ScrapeRun row shape is identical regardless of trigger source
  - the Hatchet migration in Phase 3 wraps this function and nothing else

Signature shape is deliberately Hatchet-compatible: a single coroutine
taking primitive inputs and returning a primitive result. When we migrate,
the diff is a `@hatchet.workflow` decorator and a worker boot file — the
business logic stays here untouched.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models import (
    ScrapeRun,
    ScrapeRunStatus,
    ScrapeRunTrigger,
)
from app.models.base import GrantPortal
from app.services.grant_etl import GrantETL

if TYPE_CHECKING:
    from app.scrapers.base import BaseScraper
    from app.services.embedding import EmbeddingService

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ScrapeRunResult:
    """Returned to every caller. Stable shape — used in API responses + tests."""

    run_id: uuid.UUID
    portal: GrantPortal
    status: ScrapeRunStatus
    inserted: int
    updated: int
    skipped_unchanged: int
    failed: int
    duration_ms: int
    error: str | None = None


def _scraper_for(portal: GrantPortal) -> type[BaseScraper]:
    """Lazy import — keeps the workflow module free of heavyweight scraper
    imports until a scrape is actually executed. Also avoids a circular
    import (scrapers reference models which can reference workflows in tests).
    """
    from app.scrapers.portals.bayernkapital import BayernKapitalScraper
    from app.scrapers.portals.eic import EICScraper
    from app.scrapers.portals.exist import ExistScraper
    from app.scrapers.portals.horizon import HorizonScraper
    from app.scrapers.portals.kfw import KfWScraper
    from app.scrapers.portals.lbank import LBankScraper
    from app.scrapers.portals.nrwbank import NRWBankScraper

    registry: dict[GrantPortal, type[BaseScraper]] = {
        GrantPortal.EXIST: ExistScraper,
        GrantPortal.KFW: KfWScraper,
        GrantPortal.EIC: EICScraper,
        GrantPortal.NRW: NRWBankScraper,
        GrantPortal.BW: LBankScraper,
        GrantPortal.BAYERN: BayernKapitalScraper,
        GrantPortal.HORIZON: HorizonScraper,
    }
    if portal not in registry:
        msg = f"No scraper registered for portal {portal.value!r}"
        raise ValueError(msg)
    return registry[portal]


async def scrape_portal(
    portal: GrantPortal,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingService | None = None,
    embed: bool = True,
    trigger: ScrapeRunTrigger = ScrapeRunTrigger.SCHEDULED,
) -> ScrapeRunResult:
    """Scrape one portal end-to-end and record a ScrapeRun row.

    Per-grant failures are tolerated (logged + counted, run continues).
    Top-level failures (scraper construction, DB outage) flip the run
    status to FAILED and re-raise; the caller decides what to do.

    `sessionmaker` is injected so callers can wire to dev/test pools.
    `embedder` is shared across multiple scrape_portal calls so the
    e5-large model loads once per scheduler boot, not once per portal.
    """
    started_at = datetime.now(UTC)
    monotonic_start = time.monotonic()
    run = ScrapeRun(
        portal=portal,
        trigger=trigger,
        status=ScrapeRunStatus.RUNNING,
        started_at=started_at,
        embedded=embed,
    )

    async with sessionmaker() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)
    run_id = run.id

    etl = GrantETL(embedder=embedder if embed else None)
    counts = {"inserted": 0, "updated": 0, "skipped_unchanged": 0, "failed": 0}
    top_level_error: BaseException | None = None

    logger.info(
        "scrape_workflow.start",
        run_id=str(run_id),
        portal=portal.value,
        trigger=trigger.value,
        embed=embed,
    )

    try:
        scraper_cls = _scraper_for(portal)
        async with scraper_cls() as scraper:
            async for grant in scraper.run():
                try:
                    async with sessionmaker() as session:
                        result = await etl.upsert(session, grant, embed=embed)
                        await session.commit()
                    counts[result.action] = counts.get(result.action, 0) + 1
                except Exception as e:  # noqa: BLE001 - one bad grant must not kill the run
                    counts["failed"] += 1
                    logger.warning(
                        "scrape_workflow.grant_failed",
                        run_id=str(run_id),
                        portal=portal.value,
                        source_url=grant.source_url,
                        error_type=type(e).__name__,
                        error=str(e)[:200],
                    )
    except BaseException as e:
        top_level_error = e

    duration_ms = int((time.monotonic() - monotonic_start) * 1000)

    if top_level_error is not None:
        status = ScrapeRunStatus.FAILED
    elif counts["failed"] > 0 and (counts["inserted"] + counts["updated"]) > 0:
        status = ScrapeRunStatus.PARTIAL
    elif counts["failed"] > 0:
        status = ScrapeRunStatus.FAILED
    else:
        status = ScrapeRunStatus.SUCCESS

    error_text = repr(top_level_error)[:1000] if top_level_error is not None else None
    error_type = type(top_level_error).__name__ if top_level_error is not None else None

    async with sessionmaker() as session:
        existing = await session.get(ScrapeRun, run_id)
        if existing is not None:
            existing.status = status
            existing.finished_at = datetime.now(UTC)
            existing.duration_ms = duration_ms
            existing.inserted = counts["inserted"]
            existing.updated = counts["updated"]
            existing.skipped_unchanged = counts["skipped_unchanged"]
            existing.failed = counts["failed"]
            existing.error = error_text
            existing.error_type = error_type
            await session.commit()

    logger.info(
        "scrape_workflow.finish",
        run_id=str(run_id),
        portal=portal.value,
        status=status.value,
        duration_ms=duration_ms,
        **counts,
    )

    if top_level_error is not None and not isinstance(top_level_error, Exception):
        # Re-raise KeyboardInterrupt / SystemExit so cancellations propagate.
        raise top_level_error

    return ScrapeRunResult(
        run_id=run_id,
        portal=portal,
        status=status,
        inserted=counts["inserted"],
        updated=counts["updated"],
        skipped_unchanged=counts["skipped_unchanged"],
        failed=counts["failed"],
        duration_ms=duration_ms,
        error=error_text,
    )

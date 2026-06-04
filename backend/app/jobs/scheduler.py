"""APScheduler wrapper — schedules daily scrapes for every wired portal.

Why APScheduler and not Hatchet (for now):
  Hatchet's self-hosted stack needs ~6 containers (server + engine + queue
  + DBs + dashboard). For the current phase — one developer, no production
  traffic, 26 grants in DB — APScheduler is the right amount of machinery:
  one Python process, in-memory job store, zero new infra. The workflow
  function in `scrape_workflow.py` is Hatchet-shaped so the swap is
  mechanical when durability + distributed retries actually matter.

Scheduling policy:
  - Daily runs.
  - Stagger portals by one minute starting at 03:01 UTC. Hitting all seven
    government sites simultaneously every night would be impolite.
  - Hours configurable via env (`SCHEDULER_BASE_HOUR_UTC`) so we don't
    collide with other automation.

Embedder reuse:
  The scheduler instantiates ONE `EmbeddingService`. APScheduler runs every
  job in the same event loop, so all daily scrapes share the cached model —
  ~700MB loaded once per process boot instead of seven times.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.logging import get_logger
from app.models import ScrapeRunTrigger
from app.models.base import GrantPortal

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.services.embedding import EmbeddingService

logger = get_logger(__name__)

# Order matches the scrape-stagger order. Portals listed first run earliest.
SCHEDULED_PORTALS: tuple[GrantPortal, ...] = (
    GrantPortal.EXIST,
    GrantPortal.KFW,
    GrantPortal.NRW,
    GrantPortal.BW,
    GrantPortal.BAYERN,
    GrantPortal.EIC,
    GrantPortal.HORIZON,
)


def _base_hour() -> int:
    raw = os.environ.get("SCHEDULER_BASE_HOUR_UTC", "3")
    try:
        h = int(raw)
        if 0 <= h <= 23:
            return h
    except ValueError:
        pass
    logger.warning("scheduler.invalid_base_hour", value=raw, fallback=3)
    return 3


def create_scheduler(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingService | None,
) -> AsyncIOScheduler:
    """Build (but do NOT start) an AsyncIOScheduler with one job per portal.

    Caller is expected to:
        scheduler = create_scheduler(...)
        scheduler.start()
        ...
        scheduler.shutdown(wait=False)

    Lifespan owns start/stop; this function is pure construction so it's
    trivially unit-testable.
    """
    # Local import — keeps the embedding service (and its torch transitive
    # dep) out of the scheduler module's import graph until really used.
    from app.jobs.scrape_workflow import scrape_portal

    scheduler = AsyncIOScheduler(timezone="UTC")
    base_hour = _base_hour()

    for offset, portal in enumerate(SCHEDULED_PORTALS):
        minute = (1 + offset) % 60
        hour = (base_hour + (1 + offset) // 60) % 24
        trigger = CronTrigger(hour=hour, minute=minute, timezone="UTC")

        async def _job(p: GrantPortal = portal) -> None:
            try:
                await scrape_portal(
                    p,
                    sessionmaker=sessionmaker,
                    embedder=embedder,
                    embed=embedder is not None,
                    trigger=ScrapeRunTrigger.SCHEDULED,
                )
            except Exception as e:  # noqa: BLE001 - keep scheduler alive
                logger.exception(
                    "scheduler.job.failed",
                    portal=p.value,
                    error_type=type(e).__name__,
                )

        scheduler.add_job(
            _job,
            trigger=trigger,
            id=f"scrape_{portal.value}",
            name=f"Daily scrape — {portal.value}",
            replace_existing=True,
            max_instances=1,        # never overlap a slow run with the next day's
            coalesce=True,          # if we missed an hour, run once not many times
            misfire_grace_time=3600,
        )
        logger.info(
            "scheduler.job.registered",
            portal=portal.value,
            cron=f"{minute} {hour} * * * UTC",
        )

    return scheduler

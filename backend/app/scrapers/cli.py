"""Scraper CLI — thin wrapper around `scrape_portal()`.

Run one portal:
    uv run python -m app.scrapers.cli exist
    uv run python -m app.scrapers.cli exist --write
    uv run python -m app.scrapers.cli exist --write --embed
    uv run python -m app.scrapers.cli exist --write --embed --limit 1

Dry-run mode (no --write) prints scraped grants to stdout without writing
to the DB. This is the only CLI path that doesn't go through the canonical
`scrape_portal()` workflow — because dry-run by definition doesn't write a
ScrapeRun row or call the ETL.

All --write paths delegate to `scrape_portal()` so the CLI, the scheduler,
and the /admin/scrape endpoint all produce identical ScrapeRun rows.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.core.logging import configure_logging, get_logger
from app.jobs.scrape_workflow import scrape_portal
from app.models import ScrapeRunTrigger
from app.models.base import GrantPortal
from app.scrapers.base import BaseScraper
from app.scrapers.portals.bayernkapital import BayernKapitalScraper
from app.scrapers.portals.eic import EICScraper
from app.scrapers.portals.exist import ExistScraper
from app.scrapers.portals.horizon import HorizonScraper
from app.scrapers.portals.kfw import KfWScraper
from app.scrapers.portals.lbank import LBankScraper
from app.scrapers.portals.nrwbank import NRWBankScraper

logger = get_logger(__name__)


# Mirrors the registry inside `scrape_portal` — kept here only so the CLI
# can list valid choices in --help and run dry-runs without touching the
# workflow module.
PORTAL_REGISTRY: dict[str, type[BaseScraper]] = {
    GrantPortal.EXIST.value: ExistScraper,
    GrantPortal.KFW.value: KfWScraper,
    GrantPortal.EIC.value: EICScraper,
    GrantPortal.NRW.value: NRWBankScraper,
    GrantPortal.BW.value: LBankScraper,
    GrantPortal.BAYERN.value: BayernKapitalScraper,
    GrantPortal.HORIZON.value: HorizonScraper,
}


async def _dry_run(portal: str, *, limit: int | None) -> int:
    scraper_cls = PORTAL_REGISTRY[portal]
    async with scraper_cls() as scraper:
        n = 0
        async for grant in scraper.run():
            n += 1
            _print_grant(grant)
            if limit is not None and n >= limit:
                break
    return 0


async def _write_run(portal: str, *, embed: bool, limit: int | None) -> int:
    portal_enum = GrantPortal(portal)
    init_engine()
    sessionmaker = get_sessionmaker()

    embedder = None
    if embed:
        from app.services.embedding import EmbeddingService

        embedder = EmbeddingService()

    try:
        result = await scrape_portal(
            portal_enum,
            sessionmaker=sessionmaker,
            embedder=embedder,
            embed=embed,
            trigger=ScrapeRunTrigger.CLI,
        )
    finally:
        await dispose_engine()

    print(
        f"\n=== Run {result.run_id} — {result.status.value}\n"
        f"    inserted={result.inserted} updated={result.updated} "
        f"skipped={result.skipped_unchanged} failed={result.failed}\n"
        f"    duration_ms={result.duration_ms}"
    )
    # --limit isn't passed to the workflow yet (the workflow runs the full
    # portal). We surface this so a user who relied on --limit before knows.
    if limit is not None:
        print("NOTE: --limit is currently honoured only in dry-run mode.")

    return 0 if result.status.value in ("success", "partial") else 1


def _print_grant(g: object) -> None:
    from app.scrapers.schemas import ScrapedGrant

    assert isinstance(g, ScrapedGrant)
    print(f"\n=== {g.title}")
    print(f"  portal:   {g.portal.value}")
    print(f"  url:      {g.source_url}")
    print(f"  deadline: {g.deadline}")
    print(f"  funding:  {g.funding_min_eur} – {g.funding_max_eur} EUR")
    print(f"  summary:  {g.summary[:200]}...")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fundiq-scrape", description="Run a portal scraper.")
    parser.add_argument("portal", choices=sorted(PORTAL_REGISTRY), help="portal to scrape")
    parser.add_argument("--write", action="store_true", help="upsert into the grants table")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="generate embeddings (requires --write; loads ~700MB torch model)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N grants (dry-run only)")
    args = parser.parse_args(argv)

    if args.embed and not args.write:
        parser.error("--embed requires --write")

    configure_logging()
    if args.write:
        return asyncio.run(_write_run(args.portal, embed=args.embed, limit=args.limit))
    return asyncio.run(_dry_run(args.portal, limit=args.limit))


if __name__ == "__main__":
    sys.exit(main())

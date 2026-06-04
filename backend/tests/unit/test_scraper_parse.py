"""EXIST scraper unit tests — runs against fixture HTML, no network.

`gruendungsstipendium_real.html` is a snapshot of
https://exist.de/programm/exist-gruendungsstipendium/ captured during
Phase 2A.1. Re-snapshot when EXIST changes structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.exist import ExistScraper

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "scrapers" / "exist"
REAL_FIXTURE = FIXTURE_DIR / "gruendungsstipendium_real.html"


@pytest.mark.unit
async def test_exist_parses_real_gruendungsstipendium_page() -> None:
    html = REAL_FIXTURE.read_text(encoding="utf-8")
    url = "https://exist.de/programm/exist-gruendungsstipendium/"

    async with ExistScraper() as scraper:
        grant = await scraper.parse(url, html)

    assert grant.portal is GrantPortal.EXIST
    assert grant.source_url == url
    assert grant.source_doc_id == "exist-gruendungsstipendium"
    assert grant.country == "DE"

    # Title: WordPress renders this as `exist Gründungs­stipendium` with a
    # soft hyphen — our normalizer strips it, so the result is clean.
    assert "Gründungsstipendium" in grant.title
    assert "­" not in grant.title  # soft hyphen U+00AD must be gone

    # Summary: first substantive paragraph mentions the target audience.
    assert any(
        token in grant.summary.lower()
        for token in ("gründungsinteressierte", "studierende", "absolvent")
    )

    # Body: nav/script/footer stripped; substantive content remains.
    assert "Gründungsstipendium" in grant.body
    assert "<script" not in grant.body  # safety
    assert len(grant.body) > 500

    # Funding + deadline live in PDFs we don't parse yet (Phase 2B).
    # The page itself has no concrete amounts or dates → None is correct,
    # and the status falls back to ROLLING when no deadline is found.
    assert grant.status is GrantStatus.ROLLING


@pytest.mark.unit
async def test_exist_content_hash_stable() -> None:
    html = REAL_FIXTURE.read_text(encoding="utf-8")
    url = "https://exist.de/programm/exist-gruendungsstipendium/"
    async with ExistScraper() as scraper:
        g1 = await scraper.parse(url, html)
        g2 = await scraper.parse(url, html)
    assert g1.content_hash() == g2.content_hash()
    assert len(g1.content_hash()) == 64  # sha256 hex


@pytest.mark.unit
async def test_exist_embedding_text_starts_with_title() -> None:
    html = REAL_FIXTURE.read_text(encoding="utf-8")
    url = "https://exist.de/programm/exist-gruendungsstipendium/"
    async with ExistScraper() as scraper:
        g = await scraper.parse(url, html)
    et = g.embedding_text()
    assert et.startswith(g.title)


@pytest.mark.unit
def test_program_urls_use_new_wordpress_structure() -> None:
    """Guard against regression to the legacy TYPO3 URLs that 301-redirect."""
    from app.scrapers.portals.exist import PROGRAM_URLS

    assert all(u.startswith("https://exist.de/programm/") for u in PROGRAM_URLS)
    assert all(u.endswith("/") for u in PROGRAM_URLS)
    assert len(PROGRAM_URLS) == 4

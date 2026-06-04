"""EIC scraper unit tests against a fixture HTML snapshot."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.eic import EICScraper

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "scrapers"
    / "eic"
    / "accelerator_real.html"
)


@pytest.mark.unit
async def test_eic_parses_accelerator_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en"
    async with EICScraper() as scraper:
        grant = await scraper.parse(url, html)

    assert grant.portal is GrantPortal.EIC
    assert grant.source_url == url
    assert grant.source_doc_id == "eic-accelerator"
    assert grant.country == "EU"  # EU-wide pseudo-code
    assert grant.status is GrantStatus.OPEN

    # Title from ECL component, no suffix.
    assert grant.title == "EIC Accelerator"

    # Summary is the first substantive paragraph — the "what is" intro.
    assert "EIC Accelerator" in grant.summary
    assert any(t in grant.summary.lower() for t in ("start-ups", "smes", "support"))

    # Body cleaned (no scripts/forms/nav).
    assert "<script" not in grant.body
    assert len(grant.body) > 1000  # EIC pages are long

    # Funding: Accelerator famously offers up to €10M equity + €2.5M grant.
    # We expect parse to surface at least the 10M ceiling.
    assert grant.funding_max_eur is not None
    assert grant.funding_max_eur >= Decimal("10000000")


@pytest.mark.unit
async def test_eic_content_hash_stable() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en"
    async with EICScraper() as scraper:
        g1 = await scraper.parse(url, html)
        g2 = await scraper.parse(url, html)
    assert g1.content_hash() == g2.content_hash()


@pytest.mark.unit
def test_eic_program_urls_are_four_funding_instruments() -> None:
    from app.scrapers.portals.eic import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 4
    assert all(u.startswith("https://eic.ec.europa.eu/eic-funding-opportunities/") for u in PROGRAM_URLS)
    assert all(u.endswith("_en") for u in PROGRAM_URLS)

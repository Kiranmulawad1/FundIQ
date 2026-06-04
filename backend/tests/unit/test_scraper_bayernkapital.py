"""Bayern Kapital scraper unit tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.bayernkapital import BayernKapitalScraper

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "scrapers"
    / "bayernkapital"
    / "early_stage_real.html"
)


@pytest.mark.unit
async def test_bayern_kapital_parses_early_stage_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://bayernkapital.de/fuer-gruender/early-stage/"
    async with BayernKapitalScraper() as scraper:
        g = await scraper.parse(url, html)

    assert g.portal is GrantPortal.BAYERN
    assert g.country == "DE"
    assert g.federal_state == "Bayern"
    assert g.status is GrantStatus.OPEN
    assert g.source_doc_id == "early-stage"
    # Title comes from <title> element, not the marketing h1.
    assert "Early Stage" in g.title
    assert "Bayern Kapital" in g.title

    # Body should mention the funds.
    assert "Seedfonds" in g.body or "Innovationsfonds" in g.body

    # Largest ticket on the early-stage page is "bis zu 8 Millionen Euro".
    assert g.funding_max_eur is not None
    assert g.funding_max_eur >= Decimal("2500000")


@pytest.mark.unit
def test_bayern_kapital_program_urls() -> None:
    from app.scrapers.portals.bayernkapital import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 2
    assert all(u.startswith("https://bayernkapital.de/fuer-gruender/") for u in PROGRAM_URLS)

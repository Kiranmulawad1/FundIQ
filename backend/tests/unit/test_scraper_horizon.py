"""Horizon Europe scraper unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.horizon import HorizonScraper

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "scrapers"
    / "horizon"
    / "horizon_europe_real.html"
)


@pytest.mark.unit
async def test_horizon_parses_main_programme_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = (
        "https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/"
        "funding-programmes-and-open-calls/horizon-europe_en"
    )
    async with HorizonScraper() as scraper:
        g = await scraper.parse(url, html)

    assert g.portal is GrantPortal.HORIZON
    assert g.country == "EU"
    assert g.status is GrantStatus.OPEN
    assert g.title == "Horizon Europe"
    assert g.source_doc_id == "horizon-europe"
    assert "research and innovation" in g.summary.lower()
    assert len(g.body) > 1000


@pytest.mark.unit
def test_horizon_program_urls() -> None:
    from app.scrapers.portals.horizon import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 4
    assert all("horizon-europe" in u for u in PROGRAM_URLS)
    assert all(u.endswith("_en") for u in PROGRAM_URLS)

"""L-Bank (Baden-Württemberg) scraper unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.lbank import LBankScraper

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "scrapers" / "lbank" / "guw_bw_real.html"
)


@pytest.mark.unit
async def test_lbank_parses_guw_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://www.l-bank.de/produkte/wirtschaftsfoerderung/guw-bw.html"
    async with LBankScraper() as scraper:
        g = await scraper.parse(url, html)

    assert g.portal is GrantPortal.BW
    assert g.country == "DE"
    assert g.federal_state == "Baden-Württemberg"
    assert g.status is GrantStatus.OPEN
    assert g.source_doc_id == "guw-bw"

    # Title contains the program name; soft hyphens stripped.
    assert "GuW-BW" in g.title or "Gründungs" in g.title
    assert "­" not in g.title  # soft hyphen U+00AD must be gone

    assert "KMU" in g.summary or "Förderdarlehen" in g.summary
    assert len(g.body) > 500


@pytest.mark.unit
def test_lbank_program_urls() -> None:
    from app.scrapers.portals.lbank import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 4
    assert all(u.startswith("https://www.l-bank.de/produkte/") for u in PROGRAM_URLS)

"""NRW.BANK scraper unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.nrwbank import NRWBankScraper

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "scrapers"
    / "nrwbank"
    / "gruendungszuschuss_15086_real.html"
)


@pytest.mark.unit
async def test_nrwbank_parses_gruendungszuschuss_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://www.nrwbank.de/de/foerderung/foerderprodukte/15086/gruendungszuschuss.html"
    async with NRWBankScraper() as scraper:
        g = await scraper.parse(url, html)

    assert g.portal is GrantPortal.NRW
    assert g.country == "DE"
    assert g.federal_state == "Nordrhein-Westfalen"
    assert g.status is GrantStatus.OPEN
    assert g.source_doc_id == "15086"
    assert g.title == "Gründungszuschuss"

    # NRW.BANK meta descriptions are editor-curated → the preferred summary.
    assert "Existenzgründerinnen" in g.summary or "Existenzgründer" in g.summary
    assert len(g.body) > 500
    assert "<script" not in g.body


@pytest.mark.unit
async def test_nrwbank_content_hash_stable() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://www.nrwbank.de/de/foerderung/foerderprodukte/15086/x.html"
    async with NRWBankScraper() as scraper:
        g1 = await scraper.parse(url, html)
        g2 = await scraper.parse(url, html)
    assert g1.content_hash() == g2.content_hash()


@pytest.mark.unit
def test_nrwbank_program_urls_use_numeric_id_pattern() -> None:
    from app.scrapers.portals.nrwbank import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 4
    for u in PROGRAM_URLS:
        assert u.startswith("https://www.nrwbank.de/de/foerderung/foerderprodukte/")
        assert u.endswith(".html")

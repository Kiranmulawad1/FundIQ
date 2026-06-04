"""KfW scraper unit tests against fixture HTML.

`startgeld_067_real.html` is a snapshot of
https://www.kfw.de/inlandsfoerderung/Unternehmen/Gründen-Nachfolgen/Förderprodukte/ERP-Gründerkredit-Startgeld-(067)/
captured during Phase 2C.1. Re-snapshot when KfW changes layout.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.models.base import GrantPortal, GrantStatus
from app.scrapers.portals.kfw import KfWScraper

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "scrapers"
    / "kfw"
    / "startgeld_067_real.html"
)


@pytest.mark.unit
async def test_kfw_parses_startgeld_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = (
        "https://www.kfw.de/inlandsfoerderung/Unternehmen/Gründen-Nachfolgen/"
        "Förderprodukte/ERP-Gründerkredit-Startgeld-(067)/"
    )
    async with KfWScraper() as scraper:
        grant = await scraper.parse(url, html)

    assert grant.portal is GrantPortal.KFW
    assert grant.country == "DE"
    assert grant.source_url == url
    assert grant.status is GrantStatus.OPEN  # KfW loans are rolling
    assert grant.deadline is None

    # Title strips the "Kredit Nr.067" product-number prefix.
    assert grant.title == "ERP-Gründerkredit – StartGeld"

    # Summary skips the boilerplate CTA paragraph and picks the substantive one.
    assert "ERP-Gründerkredit – StartGeld" in grant.summary
    assert "200.000" in grant.summary or "Unternehmen" in grant.summary
    assert "wenigen Klicks" not in grant.summary.lower()

    # Body cleanly stripped — no scripts/footer/etc.
    assert "<script" not in grant.body
    assert len(grant.body) > 500

    # Funding ceiling: the page advertises "bis zu 200.000 Euro Kredit".
    assert grant.funding_max_eur == Decimal("200000")

    # source_doc_id retains the program number for stable identity.
    assert grant.source_doc_id == "ERP-Gründerkredit-Startgeld-(067)"


@pytest.mark.unit
async def test_kfw_content_hash_stable() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    url = "https://www.kfw.de/dummy/"
    async with KfWScraper() as scraper:
        g1 = await scraper.parse(url, html)
        g2 = await scraper.parse(url, html)
    assert g1.content_hash() == g2.content_hash()
    assert len(g1.content_hash()) == 64


@pytest.mark.unit
def test_program_urls_are_four_founder_products() -> None:
    from app.scrapers.portals.kfw import PROGRAM_URLS

    assert len(PROGRAM_URLS) == 4
    assert all(u.startswith("https://www.kfw.de/inlandsfoerderung/Unternehmen/") for u in PROGRAM_URLS)
    assert all("Förderprodukte" in u for u in PROGRAM_URLS)

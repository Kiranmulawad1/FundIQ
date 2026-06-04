"""NRW.BANK scraper — North Rhine-Westphalia state development bank.

NRW.BANK is NRW's state-owned development bank and aggregates ~161
Förderprodukte across state, federal, and EU programs through one
catalog. This scraper picks 4 founder/SME-relevant products that
demonstrate the variety:
  - Gründungszuschuss             (founder subsidy, qualitative funding)
  - High-Tech Gründerfonds (HTGF) (federal equity vehicle, listed by NRW)
  - ZIM                           (federal SME innovation programme)
  - NRW.BANK Mittelstandsfonds    (state-level SME fund)

What's notable about NRW.BANK as a portal:
  Many products listed here are not run by NRW.BANK directly — HTGF,
  ZIM, EXIST are federal. NRW.BANK acts as a regional discovery layer.
  We still ingest from this URL (it's the authoritative regional view).
  This means the same program may appear under multiple portals
  (EXIST direct, EXIST via NRW.BANK) — that's fine; users find what
  they find. Dedup-by-content if it ever becomes an issue.

URL structure:
  https://www.nrwbank.de/de/foerderung/foerderprodukte/<id>/<slug>.html
  The numeric `<id>` is stable — we keep it in source_doc_id.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import ClassVar

from bs4 import BeautifulSoup, Tag

from app.core.logging import get_logger
from app.models.base import GrantPortal, GrantStatus
from app.scrapers.base import BaseScraper
from app.scrapers.normalize import normalize_whitespace, parse_amount_eur
from app.scrapers.schemas import ScrapedGrant

logger = get_logger(__name__)

NRW_BASE = "https://www.nrwbank.de"

PROGRAM_URLS: list[str] = [
    f"{NRW_BASE}/de/foerderung/foerderprodukte/15086/gruendungszuschuss.html",
    f"{NRW_BASE}/de/foerderung/foerderprodukte/15134/high-tech-gruenderfonds-htgf.html",
    f"{NRW_BASE}/de/foerderung/foerderprodukte/15171/zentrales-innovationsprogramm-mittelstand-zim.html",
    f"{NRW_BASE}/de/foerderung/foerderprodukte/15206/nrwbank-mittelstandsfonds.html",
]

_TITLE_SUFFIX = " - NRW.BANK"


class NRWBankScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.NRW
    rate_limit_seconds = 1.5

    async def discover(self) -> AsyncIterator[str]:
        for url in PROGRAM_URLS:
            yield url

    async def parse(self, url: str, html: str) -> ScrapedGrant:
        soup = BeautifulSoup(html, "lxml")

        title = self._extract_title(soup)
        summary = self._extract_summary(soup)
        body = self._extract_body(soup)
        funding_max = self._extract_funding_max(soup)
        source_doc_id = self._extract_doc_id(url)

        return ScrapedGrant(
            portal=self.portal,
            source_url=url,
            source_doc_id=source_doc_id,
            title=title,
            summary=summary,
            body=body,
            status=GrantStatus.OPEN,
            country="DE",
            federal_state="Nordrhein-Westfalen",
            funding_max_eur=funding_max,
        )

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_doc_id(url: str) -> str:
        """Keep the stable numeric ID — `15086/gruendungszuschuss.html`
        becomes `15086`. The slug after may change; the ID does not.
        """
        parts = url.rstrip("/").rsplit("/", 2)
        # parts = [..., '15086', 'gruendungszuschuss.html']
        if len(parts) >= 2 and parts[-2].isdigit():
            return parts[-2]
        return parts[-1].removesuffix(".html")

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        for selector in ("h1.text-break", "main h1", "h1"):
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return normalize_whitespace(el.get_text())
        if soup.title and soup.title.string:
            raw = normalize_whitespace(soup.title.string)
            return raw.removesuffix(_TITLE_SUFFIX).strip()
        return "NRW.BANK product (title unavailable)"

    @staticmethod
    def _extract_summary(soup: BeautifulSoup) -> str:
        # Meta description is curated by editors on NRW.BANK pages and is
        # consistently the cleanest one-sentence summary available.
        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = meta.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_whitespace(content)

        main = soup.select_one("main") or soup
        for p in main.find_all("p"):
            text = normalize_whitespace(p.get_text())
            if len(text) >= 60:
                return text
        return "(no summary available)"

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        main = soup.select_one("main") or soup.body or soup
        for tag in main.find_all(
            ["script", "style", "nav", "footer", "header", "form", "noscript"]
        ):
            tag.decompose()
        return normalize_whitespace(main.get_text(separator="\n"))

    @staticmethod
    def _extract_funding_max(soup: BeautifulSoup) -> Decimal | None:
        """NRW.BANK pages label monetary ceilings under 'Förderhöhe' or
        'Förderbetrag'. We anchor the scan there; per-amount cues like
        'bis zu' also pick up the right values when 'Förderhöhe' is
        absent (some products are qualitative, e.g. Gründungszuschuss).
        """
        text = soup.get_text(separator=" ")
        cues = ("Förderhöhe", "Förderbetrag", "Kreditbetrag", "bis zu", "maximal")
        amounts: list[Decimal] = []
        for cue in cues:
            start = 0
            while True:
                idx = text.find(cue, start)
                if idx == -1:
                    break
                window = text[idx : idx + 100]
                amt = parse_amount_eur(window)
                if amt is not None and amt >= Decimal("1000"):
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 30:
                    break
        if not amounts:
            return None
        return max(amounts)

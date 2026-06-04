"""L-Bank scraper — Baden-Württemberg state development bank.

L-Bank is the state-owned development bank of Baden-Württemberg. Their
catalog is mostly housing/agriculture but has a small set of strong
startup/SME programs:
  - Start-up BW Pre-Seed         (state pre-seed equity)
  - GuW-BW                       (Gründungs- und Wachstumsfinanzierung)
  - Innovationsfinanzierung      (innovation financing)
  - InnoGrowth BW                (growth innovation programme)

Page structure (L-Bank uses a custom CMS):
  - h1 with classes `.a-heading.m-heading-composition__headline` carries
    the program name; soft hyphens are present in long titles
    (e.g. "Wachstums­finanzierung") — our normalizer strips them.
  - Meta description is editor-curated and is the cleanest summary.
  - <main> wraps the body.
  - Funding amounts use German formatting in plain prose.
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

LBANK_BASE = "https://www.l-bank.de"

PROGRAM_URLS: list[str] = [
    f"{LBANK_BASE}/produkte/unternehmensfinanzierung/start-up-bw-pre-seed.html",
    f"{LBANK_BASE}/produkte/wirtschaftsfoerderung/guw-bw.html",
    f"{LBANK_BASE}/produkte/wirtschaftsfoerderung/innovationsfinanzierung.html",
    f"{LBANK_BASE}/produkte/unternehmensfinanzierung/innogrowth-bw.html",
]

_TITLE_SUFFIX = " | L-Bank"


class LBankScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.BW
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
        source_doc_id = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")

        return ScrapedGrant(
            portal=self.portal,
            source_url=url,
            source_doc_id=source_doc_id,
            title=title,
            summary=summary,
            body=body,
            status=GrantStatus.OPEN,
            country="DE",
            federal_state="Baden-Württemberg",
            funding_max_eur=funding_max,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        for selector in (
            "h1.m-heading-composition__headline",
            "h1.a-heading--main-small",
            "main h1",
            "h1",
        ):
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return normalize_whitespace(el.get_text())
        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string).removesuffix(_TITLE_SUFFIX).strip()
        return "L-Bank product (title unavailable)"

    @staticmethod
    def _extract_summary(soup: BeautifulSoup) -> str:
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
        text = soup.get_text(separator=" ")
        cues = ("bis zu", "maximal", "Förderhöhe", "Höchstbetrag", "Kredithöhe")
        amounts: list[Decimal] = []
        for cue in cues:
            start = 0
            while True:
                idx = text.find(cue, start)
                if idx == -1:
                    break
                window = text[idx : idx + 100]
                amt = parse_amount_eur(window)
                if amt is not None and amt >= Decimal("10000"):
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 30:
                    break
        if not amounts:
            return None
        return max(amounts)

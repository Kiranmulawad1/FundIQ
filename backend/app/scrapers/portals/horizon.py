"""Horizon Europe scraper.

Horizon Europe is the EU's flagship research-and-innovation framework
(€95.5bn for 2021-2027). It's structured as three pillars with multiple
sub-programmes; we scrape the main programme page plus three high-level
sub-programmes that are most relevant to startup users:
  - Horizon Europe (overall programme overview)
  - Marie Skłodowska-Curie Actions  (talent + mobility)
  - EU Missions                     (5 mission-driven calls)
  - European Innovation Ecosystems  (ecosystem-building actions)

Why not the funding-tenders portal:
  The "real" call-level catalog lives at ec.europa.eu/info/funding-tenders/
  opportunities/... but that's a SPA — heavy JS, no useful pre-rendered
  HTML. The CMS pages we scrape here are programme-level overviews;
  per-call detail discovery requires Playwright (Phase 2D).

Page structure:
  Same Europa ECL CMS as EIC. Identical selectors. The two scrapers
  share a structural pattern; we keep them separate so each portal's
  URL list, title-cleaning rules, and field overrides stay isolated.
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

H_BASE = "https://research-and-innovation.ec.europa.eu"
H_PATH = "/funding/funding-opportunities/funding-programmes-and-open-calls"

PROGRAM_URLS: list[str] = [
    f"{H_BASE}{H_PATH}/horizon-europe_en",
    f"{H_BASE}{H_PATH}/horizon-europe/marie-sklodowska-curie-actions_en",
    f"{H_BASE}{H_PATH}/horizon-europe/eu-missions-horizon-europe_en",
    f"{H_BASE}{H_PATH}/horizon-europe/european-innovation-ecosystems_en",
]

_TITLE_SUFFIXES = (
    " - Research and innovation - European Commission",
    " - European Commission",
)


class HorizonScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.HORIZON
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
        source_doc_id = url.rstrip("/").rsplit("/", 1)[-1].removesuffix("_en")

        return ScrapedGrant(
            portal=self.portal,
            source_url=url,
            source_doc_id=source_doc_id,
            title=title,
            summary=summary,
            body=body,
            status=GrantStatus.OPEN,
            country="EU",
            funding_max_eur=funding_max,
        )

    @classmethod
    def _extract_title(cls, soup: BeautifulSoup) -> str:
        for selector in ("h1.ecl-page-header__title", "main h1", "h1"):
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return normalize_whitespace(el.get_text())
        if soup.title and soup.title.string:
            raw = normalize_whitespace(soup.title.string)
            for sfx in _TITLE_SUFFIXES:
                if raw.endswith(sfx):
                    return raw[: -len(sfx)].strip()
            return raw
        return "Horizon Europe programme (title unavailable)"

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
            if len(text) >= 50:
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
        """Per-project ceilings on Horizon programme pages — when they
        appear at all. Anchored on the same per-project cues we use for
        EIC; programme-level budgets (€95.5bn etc.) are excluded.
        """
        text = soup.get_text(separator=" ")
        cues = ("up to", "below", "maximum", "lump sum")
        amounts: list[Decimal] = []
        for cue in cues:
            start = 0
            while True:
                idx = text.lower().find(cue, start)
                if idx == -1:
                    break
                window = text[idx : idx + 80]
                amt = parse_amount_eur(window)
                if amt is not None and amt >= Decimal("100000"):
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 30:
                    break
        if not amounts:
            return None
        return max(amounts)

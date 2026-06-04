"""EIC (European Innovation Council) scraper.

EIC is the EU's flagship startup-funding programme under Horizon Europe.
Four core funding instruments are scraped here:
  - EIC Accelerator     (€2.5M grant + €10M equity; flagship for SMEs/startups)
  - EIC Pathfinder      (early-stage, breakthrough research, TRL 1-4)
  - EIC Transition      (proof-of-concept → market, TRL 4-6)
  - EIC Pre-Accelerator (countries with lower innovation performance)

URL discovery:
  Programs live under https://eic.ec.europa.eu/eic-funding-opportunities/
  and were discovered from the hub page at /eic-funding-opportunities_en.
  These 4 paths have been stable since the EIC site launched in 2021.
  When EIC publishes a new instrument (e.g. a future "EIC Scaleup"), this
  list will need updating — caught by a scrape diff in Phase 2D.

Page structure (Europa Component Library / Drupal):
  - h1.ecl-page-header__title carries the clean program name.
  - <main id="main-content"> wraps everything; <article> is its
    practical equivalent.
  - First substantive <p> is a clean intro paragraph.
  - Funding amounts are quoted as "€X million", "EUR X million",
    "Up to € N million" — covered by our parser.

Country handling:
  EIC is EU-wide, not country-specific. We use the pseudo-ISO code "EU"
  in `country` so consumers can distinguish EU-wide programmes from
  member-state ones without a separate flag.
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

EIC_BASE = "https://eic.ec.europa.eu"

PROGRAM_URLS: list[str] = [
    f"{EIC_BASE}/eic-funding-opportunities/eic-accelerator_en",
    f"{EIC_BASE}/eic-funding-opportunities/eic-pathfinder_en",
    f"{EIC_BASE}/eic-funding-opportunities/eic-transition_en",
    f"{EIC_BASE}/eic-funding-opportunities/eic-pre-accelerator_en",
]

# Common <title>-tag site suffixes appended by the Europa CMS — strip when
# falling back to the title element.
_TITLE_SUFFIXES = (
    " - European Innovation Council - European Commission",
    " - European Innovation Council",
    " - European Commission",
)


class EICScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.EIC
    rate_limit_seconds = 1.5
    # EU pages serve English by default; we still send Accept-Language
    # via the base class (de+en), Europa server respects ?lang=_en path suffix.

    async def discover(self) -> AsyncIterator[str]:
        for url in PROGRAM_URLS:
            yield url

    async def parse(self, url: str, html: str) -> ScrapedGrant:
        soup = BeautifulSoup(html, "lxml")

        title = self._extract_title(soup)
        summary = self._extract_summary(soup)
        body = self._extract_body(soup)
        funding_min, funding_max = self._extract_funding_range(soup)
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
            funding_min_eur=funding_min,
            funding_max_eur=funding_max,
        )

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------
    @classmethod
    def _extract_title(cls, soup: BeautifulSoup) -> str:
        # ECL component class is highly stable on Europa pages.
        for selector in ("h1.ecl-page-header__title", "main h1", "h1"):
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return normalize_whitespace(el.get_text())
        if soup.title and soup.title.string:
            raw = normalize_whitespace(soup.title.string)
            for suffix in _TITLE_SUFFIXES:
                if raw.endswith(suffix):
                    return raw[: -len(suffix)].strip()
            return raw
        return "EIC programme (title unavailable)"

    @staticmethod
    def _extract_summary(soup: BeautifulSoup) -> str:
        main = soup.select_one("main") or soup
        for p in main.find_all("p"):
            text = normalize_whitespace(p.get_text())
            if len(text) >= 50:
                return text
        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = meta.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_whitespace(content)
        return "(no summary available)"

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
        for tag in main.find_all(
            ["script", "style", "nav", "footer", "header", "form", "noscript"]
        ):
            tag.decompose()
        return normalize_whitespace(main.get_text(separator="\n"))

    # Cues that signal a per-project funding ceiling. Scanning anchored to
    # these phrases avoids picking up programme-level budgets ("overall
    # budget in 2026 is €220 million"), which are unrelated to the amount
    # any individual applicant can receive.
    _PER_PROJECT_CUES: ClassVar[tuple[str, ...]] = (
        "up to",
        "below",
        "maximum",
        "lump sum",
        "grant component",
    )

    @classmethod
    def _extract_funding_range(
        cls, soup: BeautifulSoup
    ) -> tuple[Decimal | None, Decimal | None]:
        """Return (min, max) of credible *per-project* funding amounts.

        EIC pages quote both per-project ceilings ("Up to €10 million",
        "Grant component below €2.5 million") AND programme-level budgets
        ("overall budget in 2026 is €220 million"). The former is what
        users care about; the latter is operational. We anchor the scan
        to per-project phrases.
        """
        text = soup.get_text(separator=" ")
        amounts: list[Decimal] = []
        for cue in cls._PER_PROJECT_CUES:
            start = 0
            while True:
                idx = text.lower().find(cue, start)
                if idx == -1:
                    break
                # Look 60 chars *after* the cue, where the amount lives.
                window = text[idx : idx + 80]
                amt = parse_amount_eur(window)
                if amt is not None and amt >= Decimal("100000"):
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 50:
                    break
        if not amounts:
            return (None, None)
        sorted_amounts = sorted(amounts)
        return (sorted_amounts[0], sorted_amounts[-1])

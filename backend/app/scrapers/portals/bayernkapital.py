"""Bayern Kapital scraper — Bavarian state venture-capital arm.

Bayern Kapital is the Bavarian state's primary equity investor in tech
startups. They manage several funds (Seedfonds Bayern, Innovationsfonds,
ScaleUp Fund, Wachstumsfonds Bayern) — but their public site groups them
under two umbrella pages we treat as the canonical program entities:
  - Early Stage  (Seedfonds + Innovationsfonds, ticket sizes up to ~€8M)
  - Later Stage  (Wachstumsfonds, larger growth-equity tickets)

Choice of granularity:
  Bayern Kapital's UX presents the funds inline on the umbrella pages,
  not as separate detail URLs. Forcing one-URL-per-fund would require
  parsing inline <h2> sections — fragile against design changes. The
  umbrella treatment loses a little fidelity (the embedding folds
  multiple funds into one record) but keeps the scraper robust.

Title source:
  The visible <h1> is a marketing tagline ("Vom Start weg in besten
  Händen"). The <title> element is informative and stable
  ("Bayern Kapital - Early Stage"). We use <title>.
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

BAYKAP_BASE = "https://bayernkapital.de"

PROGRAM_URLS: list[str] = [
    f"{BAYKAP_BASE}/fuer-gruender/early-stage/",
    f"{BAYKAP_BASE}/fuer-gruender/later-stage/",
]


class BayernKapitalScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.BAYERN
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
        source_doc_id = url.rstrip("/").rsplit("/", 1)[-1]

        return ScrapedGrant(
            portal=self.portal,
            source_url=url,
            source_doc_id=source_doc_id,
            title=title,
            summary=summary,
            body=body,
            status=GrantStatus.OPEN,
            country="DE",
            federal_state="Bayern",
            funding_max_eur=funding_max,
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        # <h1> is marketing copy; <title> is informative.
        if soup.title and soup.title.string:
            raw = normalize_whitespace(soup.title.string)
            return raw
        h1 = soup.select_one("h1")
        if h1:
            return normalize_whitespace(h1.get_text())
        return "Bayern Kapital (title unavailable)"

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
            if len(text) >= 80:
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
        """Bayern Kapital quotes ticket sizes as "bis zu N Mio. Euro".
        Anchor on "bis zu" and "Ticketgröße" to grab the per-fund caps.
        """
        text = soup.get_text(separator=" ")
        cues = ("bis zu", "Ticketgröße", "maximal", "Beteiligung von")
        amounts: list[Decimal] = []
        for cue in cues:
            start = 0
            while True:
                idx = text.find(cue, start)
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

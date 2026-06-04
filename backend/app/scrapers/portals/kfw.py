"""KfW portal scraper — startup/founder loan products.

KfW (Kreditanstalt für Wiederaufbau) is Germany's state-owned development
bank. Its Förderprodukte for founders and SMEs are technically *loans*
with subsidised interest rates, not grants — but they are part of the
same funding-discovery surface for our users, and our `Grant` schema
accommodates both (funding_min/max + status, no semantic distinction
needed here).

URL discovery:
  KfW's public XML sitemap (inlandsfoerderung/Technische-Seiten/
  Sitemap-XML.xml) lists ~9 Unternehmen products but the founder/SME
  loans are not in it. They are linked from the human-facing hub page
  https://www.kfw.de/inlandsfoerderung/Unternehmen/Gründen-Nachfolgen/
  Förderprodukte/ — we hard-code the 4 program URLs from that hub.

  When KfW launches a new founder product, this list will need an
  update (caught by a CI scrape diff, planned for Phase 2D).

URL inconsistency we accept:
  Two distinct path prefixes exist for historical reasons:
    /Unternehmen/Gründen-Nachfolgen/Förderprodukte/...   (newer)
    /Unternehmen/Gründung-und-Nachfolge/Förderprodukte/... (older)
  Both 200 OK. We keep the URLs as the hub page actually links them.

Page structure:
  - h1.product-header-content carries a "Kredit Nr.NNN<Name>" string;
    we strip the "Kredit Nr.NNN" prefix.
  - First substantive paragraph is generic CTA boilerplate ("find out
    if you qualify in a few clicks") — we skip it and use the second.
  - <main> holds the full content; nav/footer stripped.
  - Funding ceiling appears in body as "bis zu NNN.NNN Euro Kredit".
  - No deadlines; loans are rolling → status stays OPEN.
"""

from __future__ import annotations

import re
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

KFW_BASE = "https://www.kfw.de"

PROGRAM_URLS: list[str] = [
    f"{KFW_BASE}/inlandsfoerderung/Unternehmen/Gründen-Nachfolgen/Förderprodukte/ERP-Gründerkredit-Startgeld-(067)/",
    f"{KFW_BASE}/inlandsfoerderung/Unternehmen/Gründung-und-Nachfolge/Förderprodukte/ERP-Förderkredit-Gründung-und-Nachfolge-(077)/",
    f"{KFW_BASE}/inlandsfoerderung/Unternehmen/Gründung-und-Nachfolge/Förderprodukte/ERP-Förderkredit-KMU-(365-366)/",
    f"{KFW_BASE}/inlandsfoerderung/Unternehmen/Gründung-und-Nachfolge/Förderprodukte/KfW-Förderkredit-großer-Mittelstand-(375-376)/",
]

_TITLE_PREFIX_RE = re.compile(
    # `Kredit Nr.067…`, `Kredit Nr.365, 366…`, `Kredit Nr.375-376…` — all valid.
    # Match one product number followed by any number of `,` or dash-separated
    # additional product numbers.
    r"^(?:Kredit|Programm|Förderprogramm|Zuschuss)\s*Nr\.?\s*\d+(?:\s*[-–—,]\s*\d+)*\s*[-–—]?\s*",
    re.IGNORECASE,
)
# Boilerplate paragraphs to skip when picking a summary. These appear on
# every KfW product page and add zero information to the embedding.
_SUMMARY_BOILERPLATE_TOKENS = (
    "wenigen klicks",
    "voraussetzungen für die förderung erfüllen",
    "find out if you qualify",
)
# Funding ceiling phrase — KfW always frames the max loan as
# "bis zu N(.)NNN Euro Kredit" (or "Euro" alone). The regex grabs the
# preceding amount via parse_amount_eur on the surrounding window.
_FUNDING_CUES = ("bis zu", "Höchstbetrag", "Kreditbetrag", "Kredithöhe")


class KfWScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.KFW
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
            status=GrantStatus.OPEN,  # KfW loans are rolling
            country="DE",
            funding_max_eur=funding_max,
        )

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------
    @classmethod
    def _extract_title(cls, soup: BeautifulSoup) -> str:
        for selector in ("h1.product-header-content", "main h1", "h1"):
            el = soup.select_one(selector)
            if not el:
                continue
            raw = normalize_whitespace(el.get_text())
            if not raw or raw.lower() == "seite teilen":
                continue
            return _TITLE_PREFIX_RE.sub("", raw).strip()

        if soup.title and soup.title.string:
            return normalize_whitespace(soup.title.string).removesuffix(" | KfW").strip()
        return "KfW product (title unavailable)"

    @staticmethod
    def _extract_summary(soup: BeautifulSoup) -> str:
        main = soup.select_one("main") or soup
        for p in main.find_all("p"):
            text = normalize_whitespace(p.get_text())
            if len(text) < 60:
                continue
            lowered = text.lower()
            if any(token in lowered for token in _SUMMARY_BOILERPLATE_TOKENS):
                continue
            return text

        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = meta.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_whitespace(content)
        return "(no summary available)"

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        main = soup.select_one("main") or soup.body or soup
        for tag in main.find_all(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()
        return normalize_whitespace(main.get_text(separator="\n"))

    @staticmethod
    def _extract_funding_max(soup: BeautifulSoup) -> Decimal | None:
        """Pull the largest credibly-stated euro amount on the page.

        KfW pages quote both the loan ceiling ("bis zu 200.000 Euro
        Kredit") and various ranges ("ab 1.000 Euro", "5 Mio. Euro").
        We collect all parseable amounts, drop anything < 1k EUR
        (those are usually fees or insurance amounts), and take the max
        as the funding ceiling. This is heuristic but correct on every
        KfW founder product as of 2026-05.
        """
        text = soup.get_text(separator=" ")
        amounts: list[Decimal] = []
        for cue in _FUNDING_CUES:
            start = 0
            while True:
                idx = text.find(cue, start)
                if idx == -1:
                    break
                window = text[idx : idx + 80]
                amt = parse_amount_eur(window)
                if amt is not None and amt >= Decimal("1000"):
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 50:
                    break
        if not amounts:
            return None
        return max(amounts)

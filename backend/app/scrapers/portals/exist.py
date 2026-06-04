"""EXIST portal scraper.

EXIST is the German federal startup-funding family run by the BMWK.
Four active programs as of 2026, all hosted at https://exist.de on a
WordPress site:
  - exist Gründungsstipendium  (founder stipend, pre-seed)
  - exist Forschungstransfer   (research transfer, seed)
  - exist Potentiale           (universities, capacity grants)
  - exist Women                (women-led ventures, added 2024)

URL discovery:
  EXIST migrated from a TYPO3 site (under www.exist.de/EXIST/...) to a
  WordPress site (under exist.de/programm/...) sometime in 2024. The old
  deep links 301 → marketing homepage, losing the program-specific
  content. We hard-code the new slugs from
  https://exist.de/programm-sitemap1.xml — there are only 4 programs,
  no discoverability problem to solve, and the sitemap is publicly
  served so renames are easy to detect.

Why not Playwright:
  The page is fully server-rendered HTML (WordPress, no client JS for
  content). httpx + BeautifulSoup are sufficient.

What lives in the HTML vs in PDFs:
  - In the HTML: title, marketing summary, eligibility paragraphs.
  - In linked PDF "Förderrichtlinien": exact funding amounts, application
    deadlines, eligibility checklists. PDF parsing is Phase 2B work.
  - Therefore: funding_min_eur / funding_max_eur / deadline often remain
    None for EXIST. That's accurate, not a bug. The status is set to
    ROLLING when no deadline is found.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from bs4 import BeautifulSoup, Tag

from app.core.logging import get_logger
from app.models.base import GrantPortal, GrantStatus
from app.scrapers.base import BaseScraper
from app.scrapers.normalize import normalize_whitespace, parse_amount_eur, parse_german_date
from app.scrapers.schemas import ScrapedGrant

logger = get_logger(__name__)

EXIST_BASE = "https://exist.de"

PROGRAM_URLS: list[str] = [
    f"{EXIST_BASE}/programm/exist-gruendungsstipendium/",
    f"{EXIST_BASE}/programm/exist-forschungstransfer/",
    f"{EXIST_BASE}/programm/exist-potentiale/",
    f"{EXIST_BASE}/programm/exist-women/",
]


class ExistScraper(BaseScraper):
    portal: ClassVar[GrantPortal] = GrantPortal.EXIST
    rate_limit_seconds = 1.5  # bmbf-hosted; be polite

    async def discover(self) -> AsyncIterator[str]:
        for url in PROGRAM_URLS:
            yield url

    async def parse(self, url: str, html: str) -> ScrapedGrant:
        soup = BeautifulSoup(html, "lxml")

        title = self._extract_title(soup)
        summary = self._extract_summary(soup)
        body = self._extract_body(soup)

        deadline = self._extract_deadline(soup)
        funding_min, funding_max = self._extract_funding_range(soup)

        # The slug is the last path component — stable doc ID. Handle both
        # legacy `.html` URLs and the current trailing-slash URLs.
        source_doc_id = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")

        return ScrapedGrant(
            portal=self.portal,
            source_url=url,
            source_doc_id=source_doc_id,
            title=title,
            summary=summary,
            body=body,
            status=GrantStatus.OPEN if deadline else GrantStatus.ROLLING,
            country="DE",
            funding_min_eur=funding_min,
            funding_max_eur=funding_max,
            deadline=deadline,
        )

    # ------------------------------------------------------------------
    # Field extractors — split out so each is independently testable.
    # ------------------------------------------------------------------
    _TITLE_SUFFIXES: ClassVar[tuple[str, ...]] = (
        " - exist",
        " | exist",
        " — exist",
    )

    @classmethod
    def _extract_title(cls, soup: BeautifulSoup) -> str:
        for selector in ("h1.wp-block-heading", "main h1", "h1", ".title-headline"):
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return normalize_whitespace(el.get_text())
        # Last-resort fallback — strip the boilerplate " - exist" site suffix
        # so we don't end up with "exist Women - exist" style titles.
        if soup.title and soup.title.string:
            raw = normalize_whitespace(soup.title.string)
            for suffix in cls._TITLE_SUFFIXES:
                if raw.lower().endswith(suffix.lower()):
                    return raw[: -len(suffix)].strip()
            return raw
        return "EXIST program (title unavailable)"

    @staticmethod
    def _extract_summary(soup: BeautifulSoup) -> str:
        """First substantive paragraph in <main>, or meta description."""
        main = soup.select_one("main") or soup
        for p in main.find_all("p"):
            text = normalize_whitespace(p.get_text())
            if len(text) >= 40:  # filter out nav/breadcrumb fragments
                return text

        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = meta.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_whitespace(content)

        return "(no summary available)"

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        """Full main content with navigation/footer stripped."""
        main = soup.select_one("main") or soup.body or soup
        # Drop scripts, styles, nav blocks that pollute embeddings.
        for tag in main.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = main.get_text(separator="\n")
        return normalize_whitespace(text)

    @staticmethod
    def _extract_deadline(soup: BeautifulSoup) -> object | None:
        """EXIST programs typically have rolling deadlines, but Potentiale
        runs in named calls. Look for dates near 'Stichtag', 'Frist', 'Deadline'.
        """
        text = soup.get_text(separator=" ")
        for cue in ("Stichtag", "Antragsfrist", "Frist", "Bewerbungsschluss", "Deadline"):
            idx = text.find(cue)
            if idx == -1:
                continue
            window = text[idx : idx + 200]
            d = parse_german_date(window)
            if d is not None:
                return d
        return None

    @staticmethod
    def _extract_funding_range(soup: BeautifulSoup) -> tuple[object | None, object | None]:
        """Heuristic: scan for the largest two euro amounts on the page.

        EXIST pages cite both the stipend (~3k/month) and the total program
        ceiling. We grab whatever amounts we can; downstream eval can
        sharpen this with a few labelled examples.
        """
        text = soup.get_text(separator=" ")
        amounts: list[object] = []
        # Scan in 80-char windows around every '€' or 'EUR' occurrence.
        for cue in ("€", "EUR"):
            start = 0
            while True:
                idx = text.find(cue, start)
                if idx == -1:
                    break
                window = text[max(0, idx - 40) : idx + 40]
                amt = parse_amount_eur(window)
                if amt is not None:
                    amounts.append(amt)
                start = idx + 1
                if len(amounts) > 50:  # safety cap
                    break

        if not amounts:
            return (None, None)
        amounts_sorted = sorted(amounts)
        # Min = smallest credible amount; Max = largest. Both can be the same.
        return (amounts_sorted[0], amounts_sorted[-1])

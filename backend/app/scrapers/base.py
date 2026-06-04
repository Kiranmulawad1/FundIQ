"""Abstract base class for portal scrapers.

Subclass responsibilities (kept minimal on purpose):
  - declare `portal` (which `GrantPortal` enum value)
  - implement `discover()` → AsyncIterator[str] of detail-page URLs
  - implement `parse(url, html)` → ScrapedGrant

The base owns: HTTP fetching, retry, rate limiting, error containment.
A failure on one URL must not poison the rest of the run.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.logging import get_logger
from app.models.base import GrantPortal
from app.scrapers.schemas import ScrapedGrant

logger = get_logger(__name__)


class FetchError(Exception):
    """Raised when fetching a URL fails after all retries."""


class BaseScraper(abc.ABC):
    """All portal scrapers extend this.

    Lifecycle:
      async with ExistScraper() as s:
          async for grant in s.run():
              ...
    """

    # Subclasses MUST set this.
    portal: ClassVar[GrantPortal]

    # Tunable knobs — sensible defaults; overridden per portal as needed.
    rate_limit_seconds: float = 1.0
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    user_agent: str = (
        "FundIQ-Scraper/0.1 "
        "(research; contact: kiranmulawad1@gmail.com)"
    )

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> BaseScraper:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent, "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"},
                timeout=self.request_timeout_seconds,
                follow_redirects=True,
                http2=False,  # some EU gov sites still botch HTTP/2
            )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Subclass surface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def discover(self) -> AsyncIterator[str]:
        """Yield grant detail-page URLs for this portal."""

    @abc.abstractmethod
    async def parse(self, url: str, html: str) -> ScrapedGrant:
        """Extract a ScrapedGrant from a fetched HTML page."""

    # ------------------------------------------------------------------
    # Infra
    # ------------------------------------------------------------------
    async def fetch(self, url: str) -> str:
        """Fetch a URL with retry + exponential backoff + jitter.

        Retries on transport errors and 5xx; never retries on 4xx (no point).
        """
        if self._client is None:
            msg = "Scraper must be used as `async with ExistScraper() as s: ...`"
            raise RuntimeError(msg)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                r = await self._client.get(url)
                # Only retry on 5xx; 4xx should bubble up unmodified.
                if 500 <= r.status_code < 600:
                    r.raise_for_status()
                if r.status_code >= 400:
                    msg = f"HTTP {r.status_code} for {url}"
                    raise FetchError(msg)
                return r.text

        # Unreachable; AsyncRetrying with reraise=True either returns or raises.
        msg = f"fetch exhausted retries for {url}"
        raise FetchError(msg)

    async def run(self) -> AsyncIterator[ScrapedGrant]:
        """Yield ScrapedGrant for every URL we successfully fetch + parse.

        Per-URL errors are logged and skipped, never aborted. One bad page
        does not stop the rest of a portal run.
        """
        portal_name = self.portal.value
        n_yielded = 0
        n_failed = 0

        async for url in self.discover():
            try:
                html = await self.fetch(url)
                grant = await self.parse(url, html)
                yield grant
                n_yielded += 1
                logger.info("scraper.parsed", portal=portal_name, url=url, title=grant.title[:80])
            except (FetchError, httpx.HTTPError, ValueError) as e:
                n_failed += 1
                logger.warning(
                    "scraper.failed",
                    portal=portal_name,
                    url=url,
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )
            await asyncio.sleep(self.rate_limit_seconds)

        logger.info(
            "scraper.run.complete",
            portal=portal_name,
            yielded=n_yielded,
            failed=n_failed,
        )

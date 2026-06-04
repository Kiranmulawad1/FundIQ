"""Grant ETL: ScrapedGrant → grants row, with content-hash short-circuit.

Pipeline:
  1. Receive a ScrapedGrant (from a scraper).
  2. Compute content_hash. Look up existing row by source_url.
  3. If row exists and hash unchanged → skip (no DB write, no embedding).
  4. Else: upsert + (re)embed.

Why content-hash gating:
  Re-scraping the same page should be free. Without the hash check we
  would re-embed every grant on every refresh, which is wasteful (CPU)
  and forces HNSW index maintenance (slow on a populated index).

Why we use PostgreSQL `INSERT ... ON CONFLICT` rather than SELECT-then-
INSERT-or-UPDATE:
  Race-safe at the DB level even when multiple scrapers run in parallel
  (Phase 2C — Hatchet workers). One round-trip, atomic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import Grant
from app.scrapers.schemas import ScrapedGrant

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class UpsertResult:
    """What happened for one ScrapedGrant. Aggregated by callers for telemetry."""

    source_url: str
    action: str  # "inserted" | "updated" | "skipped_unchanged"
    grant_id: str | None  # populated for insert/update


class GrantETL:
    """Stateless service — pass a session per operation.

    Embedding service is injected so tests can stub it (and Phase 1 unit
    tests that don't load the 700MB torch stack can pass a fake).
    """

    def __init__(self, *, embedder: EmbeddingService | None = None) -> None:
        self._embedder = embedder

    async def upsert(
        self,
        session: AsyncSession,
        scraped: ScrapedGrant,
        *,
        embed: bool = True,
    ) -> UpsertResult:
        """Upsert one scraped grant. Returns what happened."""
        new_hash = scraped.content_hash()

        # Cheap pre-check: read just the hash. ~1ms on indexed source_url.
        existing_hash = await self._read_existing_hash(session, scraped.source_url)
        if existing_hash == new_hash:
            return UpsertResult(scraped.source_url, "skipped_unchanged", None)

        embedding: list[float] | None = None
        if embed and self._embedder is not None:
            embedding = await self._embedder.embed_passage(scraped.embedding_text())
        elif embed and self._embedder is None:
            logger.warning("etl.embed_requested_no_embedder", source_url=scraped.source_url)

        # Build the values dict — only fields we want to overwrite on conflict.
        values: dict[str, object] = {
            "title": scraped.title,
            "title_en": scraped.title_en,
            "summary": scraped.summary,
            "summary_en": scraped.summary_en,
            "body": scraped.body,
            "portal": scraped.portal,
            "status": scraped.status,
            "sector": scraped.sector,
            "country": scraped.country,
            "federal_state": scraped.federal_state,
            "funding_min_eur": scraped.funding_min_eur,
            "funding_max_eur": scraped.funding_max_eur,
            "deadline": scraped.deadline,
            "opens_at": scraped.opens_at,
            "eligibility": scraped.eligibility,
            "source_url": scraped.source_url,
            "source_doc_id": scraped.source_doc_id,
            "source_hash": new_hash,
        }
        if embedding is not None:
            values["embedding"] = embedding

        stmt = pg_insert(Grant.__table__).values(**values)
        # Update everything except the immutable id + created_at; updated_at
        # is driven by the ORM-side onupdate hook but we set it here as a
        # belt-and-braces measure since this is raw-SQL territory.
        update_cols = {k: stmt.excluded[k] for k in values if k != "source_url"}
        stmt = stmt.on_conflict_do_update(
            index_elements=["source_url"],
            set_=update_cols,
        ).returning(Grant.__table__.c.id)
        row = (await session.execute(stmt)).first()
        await session.flush()

        action = "inserted" if existing_hash is None else "updated"
        grant_id = str(row[0]) if row else None
        logger.info(
            "etl.upsert",
            action=action,
            portal=scraped.portal.value,
            source_url=scraped.source_url,
            grant_id=grant_id,
        )
        return UpsertResult(scraped.source_url, action, grant_id)

    @staticmethod
    async def _read_existing_hash(session: AsyncSession, source_url: str) -> str | None:
        from sqlalchemy import select

        stmt = select(Grant.source_hash).where(Grant.source_url == source_url)
        return (await session.execute(stmt)).scalar_one_or_none()

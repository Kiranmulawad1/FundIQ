"""ETL integration: ScrapedGrant → grants row, with stub embedder.

We stub the embedder so this test runs fast and doesn't load
sentence-transformers / torch. EmbeddingService has its own tests under
unit/ (Phase 2B).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus
from app.scrapers.schemas import ScrapedGrant
from app.services.grant_etl import GrantETL


class _StubEmbedder:
    """Deterministic fake. Returns a 1024-dim vector that depends on input length."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed_passage(self, text: str) -> list[float]:
        self.call_count += 1
        # Normalised vector so HNSW cosine ops don't choke.
        seed = (len(text) % 7) + 1
        return [1.0 / (EMBEDDING_DIM * seed) ** 0.5] * EMBEDDING_DIM


def _sample_grant(*, url: str = "https://example.gov/exist/abc", title: str = "Sample Grant") -> ScrapedGrant:
    return ScrapedGrant(
        portal=GrantPortal.EXIST,
        source_url=url,
        source_doc_id="abc",
        title=title,
        summary="A short summary of the program.",
        body="Long body about eligibility and conditions.",
        status=GrantStatus.OPEN,
        country="DE",
        funding_min_eur=Decimal("2500"),
        funding_max_eur=Decimal("30000"),
        deadline=datetime(2026, 12, 31, tzinfo=UTC),
    )


@pytest.mark.integration
async def test_first_upsert_inserts_row_with_embedding(db_session: AsyncSession) -> None:
    embedder = _StubEmbedder()
    etl = GrantETL(embedder=embedder)  # type: ignore[arg-type]

    result = await etl.upsert(db_session, _sample_grant(), embed=True)
    assert result.action == "inserted"
    assert embedder.call_count == 1

    row = (
        await db_session.execute(
            select(Grant).where(Grant.source_url == "https://example.gov/exist/abc")
        )
    ).scalar_one()
    assert row.title == "Sample Grant"
    assert row.source_hash is not None
    assert len(row.source_hash) == 64
    assert row.embedding is not None  # populated by the stub embedder


@pytest.mark.integration
async def test_second_upsert_with_unchanged_content_skips(db_session: AsyncSession) -> None:
    embedder = _StubEmbedder()
    etl = GrantETL(embedder=embedder)  # type: ignore[arg-type]

    grant = _sample_grant()
    first = await etl.upsert(db_session, grant, embed=True)
    assert first.action == "inserted"

    second = await etl.upsert(db_session, grant, embed=True)
    assert second.action == "skipped_unchanged"
    assert embedder.call_count == 1  # NOT re-embedded


@pytest.mark.integration
async def test_changed_content_updates_and_re_embeds(db_session: AsyncSession) -> None:
    embedder = _StubEmbedder()
    etl = GrantETL(embedder=embedder)  # type: ignore[arg-type]

    await etl.upsert(db_session, _sample_grant(title="Old Title"), embed=True)
    result = await etl.upsert(db_session, _sample_grant(title="New Title"), embed=True)

    assert result.action == "updated"
    assert embedder.call_count == 2  # re-embedded on content change

    row = (
        await db_session.execute(
            select(Grant).where(Grant.source_url == "https://example.gov/exist/abc")
        )
    ).scalar_one()
    assert row.title == "New Title"


@pytest.mark.integration
async def test_upsert_without_embed_inserts_without_vector(db_session: AsyncSession) -> None:
    etl = GrantETL(embedder=None)

    result = await etl.upsert(db_session, _sample_grant(), embed=False)
    assert result.action == "inserted"

    row = (
        await db_session.execute(
            select(Grant).where(Grant.source_url == "https://example.gov/exist/abc")
        )
    ).scalar_one()
    assert row.embedding is None  # untouched

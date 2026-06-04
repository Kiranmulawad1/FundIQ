"""Grant table — scraped funding opportunities.

Schema notes:
  - `embedding` is `vector(1024)` to match multilingual-e5-large.
    Created via pgvector; HNSW index added in the first migration.
  - `eligibility` JSONB is the structured rule set that the Scorer
    agent diffs against startup profiles.
  - Soft-delete via `deleted_at`. Grants are referenced by historical
    applications and roadmaps — never hard-delete.
  - `source_url` is unique (per portal) — dedup key for the scraper.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Numeric, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import (
    GrantPortal,
    GrantStatus,
    Sector,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    empty_dict,
)

EMBEDDING_DIM = 1024  # multilingual-e5-large


class Grant(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "grants"
    __table_args__ = (
        UniqueConstraint("source_url", name="uq_grants_source_url"),
        Index("ix_grants_portal_status", "portal", "status"),
        Index("ix_grants_deadline", "deadline"),
        Index("ix_grants_sector_status", "sector", "status"),
    )

    title: str = Field(max_length=500, index=True)
    title_en: str | None = Field(default=None, max_length=500)
    summary: str
    summary_en: str | None = None
    body: str

    portal: GrantPortal = Field(index=True)
    status: GrantStatus = Field(default=GrantStatus.OPEN, index=True)
    sector: Sector | None = Field(default=None, index=True)
    country: str = Field(default="DE", max_length=2)
    federal_state: str | None = Field(default=None, max_length=64)

    funding_min_eur: Decimal | None = Field(
        default=None,
        sa_column=Column(Numeric(14, 2)),
    )
    funding_max_eur: Decimal | None = Field(
        default=None,
        sa_column=Column(Numeric(14, 2)),
    )

    deadline: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )
    opens_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )

    eligibility: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    metadata_: dict[str, Any] = Field(
        default_factory=empty_dict,
        alias="metadata",
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

    embedding: Any | None = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIM)),
    )

    source_url: str = Field(max_length=1000, index=True)
    source_doc_id: str | None = Field(default=None, max_length=255, index=True)
    source_hash: str | None = Field(default=None, max_length=64)

    deleted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )

    def __repr__(self) -> str:
        return f"<Grant id={self.id} portal={self.portal} title={self.title[:40]!r}>"

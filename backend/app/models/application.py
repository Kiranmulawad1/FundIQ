"""GrantApplication — a startup's draft/submitted application to a grant.

Schema notes:
  - `section_scores` JSONB stores the LoRA Analyzer's per-section scores
    + flagged weak arguments. Shape evolves with the model.
  - `citations` JSONB stores `[{claim, source_doc_id, paragraph_id, url}]`
    — preserves the citation chain required by the grounding checker.
  - `pdf_path` is an object-storage key, not a filesystem path. (Storage
    backend chosen in Phase 5 when scrapers land.)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import (
    ApplicationStatus,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    empty_dict,
)


class GrantApplication(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "grant_applications"
    __table_args__ = (
        Index("ix_apps_startup_status", "startup_id", "status"),
        Index("ix_apps_grant", "grant_id"),
    )

    startup_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("startups.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    grant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("grants.id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        ),
    )

    status: ApplicationStatus = Field(default=ApplicationStatus.DRAFT)
    submitted_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )

    pdf_path: str | None = Field(default=None, max_length=1000)
    overall_score: float | None = Field(default=None, ge=0, le=10)

    section_scores: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )
    notes: str | None = None

"""ScrapeRun — record of one portal-scrape execution.

Captures enough state for an admin dashboard, retry logic, and the eventual
Hatchet migration audit trail:
  - which portal, when, how long, what happened
  - counts (inserted, updated, skipped, failed)
  - structured error on failure
  - trigger source (scheduled vs manual) so we can distinguish ops actions
    from cron noise in metrics
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import GrantPortal, TimestampMixin, UUIDPrimaryKeyMixin, empty_dict


class ScrapeRunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"  # some grants failed but the run completed
    FAILED = "failed"


class ScrapeRunTrigger(StrEnum):
    SCHEDULED = "scheduled"  # cron fire
    MANUAL = "manual"        # /admin/scrape/{portal}
    CLI = "cli"              # python -m app.scrapers.cli


class ScrapeRun(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "scrape_runs"
    __table_args__ = (
        Index("ix_scrape_runs_portal_started", "portal", "started_at"),
        Index("ix_scrape_runs_status", "status"),
    )

    portal: GrantPortal = Field(index=True)
    trigger: ScrapeRunTrigger = Field(default=ScrapeRunTrigger.SCHEDULED)
    status: ScrapeRunStatus = Field(default=ScrapeRunStatus.RUNNING, index=True)

    started_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )
    finished_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )
    duration_ms: int | None = Field(default=None, ge=0)

    # Counts. Sum of (inserted + updated + skipped + failed) is the total
    # ScrapedGrant items the scraper yielded.
    inserted: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped_unchanged: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)

    # Whether embeddings were generated this run. False during dry-run or
    # when --embed is omitted.
    embedded: bool = Field(default=False)

    error: str | None = Field(default=None, description="Top-level exception text.")
    error_type: str | None = Field(default=None, max_length=128)

    # Free-form audit blob — keeps options like the trigger source IP,
    # prompt version that was active, etc.
    metadata_: dict[str, Any] = Field(
        default_factory=empty_dict,
        alias="metadata",
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

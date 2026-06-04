"""Startup table — the user's company profile.

Schema notes:
  - `profile` is JSONB: flexible field for evolving interview data
    (team bios, traction metrics, IP, runway). Stable subfields get
    promoted to columns later if query patterns demand it.
  - `frs_scores` is JSONB rather than 6 columns so adding/removing
    FRS dimensions during the thesis experiments doesn't require
    a migration.
  - `(sector, stage)` composite index — primary filter combination in
    the Researcher agent's retrieval pre-filter.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import (
    Sector,
    StartupStage,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    empty_dict,
)


class Startup(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "startups"
    __table_args__ = (
        Index("ix_startups_sector_stage", "sector", "stage"),
        Index("ix_startups_owner_user_id", "owner_user_id"),
    )

    owner_user_id: str = Field(index=True, description="Clerk user ID of the owner.")
    name: str = Field(max_length=200)
    sector: Sector
    stage: StartupStage
    country: str = Field(default="DE", max_length=2)
    federal_state: str | None = Field(default=None, max_length=64)
    website: str | None = Field(default=None, max_length=500)

    profile: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    frs_scores: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )

    def __repr__(self) -> str:
        return f"<Startup id={self.id} name={self.name!r} stage={self.stage}>"


def startup_pk_type() -> type[uuid.UUID]:
    """Helper for typed FKs in other models."""
    return uuid.UUID

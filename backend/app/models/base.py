"""Shared SQLModel mixins and enums.

Design notes:
  - UUIDv7 primary keys: time-ordered, B-tree friendly. Falls back to uuid4
    on Python <3.13. (3.13 added `uuid.uuid7` natively.)
  - All datetimes are timezone-aware (UTC). Naïve datetimes have caused
    every grant-deadline off-by-one bug in this codebase's lineage.
  - `created_at`/`updated_at` set via Python defaults, not DB triggers,
    so they round-trip through ORM identity-map caching cleanly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


def _uuid7() -> uuid.UUID:
    """Time-ordered UUID. Uses stdlib uuid7 on 3.13+, falls back to uuid4."""
    factory = getattr(uuid, "uuid7", None)
    if callable(factory):
        return factory()  # type: ignore[no-any-return]
    return uuid.uuid4()


def utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin(SQLModel):
    """Adds `created_at` and `updated_at`. Mix into every table."""

    created_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
        nullable=False,
        index=True,
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
        nullable=False,
        sa_column_kwargs={"onupdate": utcnow},
    )


class UUIDPrimaryKeyMixin(SQLModel):
    id: uuid.UUID = Field(default_factory=_uuid7, primary_key=True, index=True)


# ---------------------------------------------------------------------------
# Enums — string-valued so they're readable in SQL and JSON.
# DB-side CHECK constraints are added in migrations.
# ---------------------------------------------------------------------------
class StartupStage(StrEnum):
    IDEA = "idea"
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    GROWTH = "growth"


class Sector(StrEnum):
    DEEPTECH = "deeptech"
    CLEANTECH = "cleantech"
    HEALTH = "health"
    BIOTECH = "biotech"
    SAAS = "saas"
    HARDWARE = "hardware"
    AI_ML = "ai_ml"
    FINTECH = "fintech"
    OTHER = "other"


class GrantPortal(StrEnum):
    BMBF = "bmbf"
    EXIST = "exist"
    KFW = "kfw"
    EIC = "eic"
    HORIZON = "horizon"
    BAYERN = "bayern"
    NRW = "nrw"
    BW = "bw"


class GrantStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UPCOMING = "upcoming"
    ROLLING = "rolling"


class ApplicationStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class AlertChannel(StrEnum):
    EMAIL = "email"
    SLACK = "slack"
    IN_APP = "in_app"


# Re-export for convenience (avoid `from sqlmodel import Field` in callers).
__all__: list[str] = [
    "AlertChannel",
    "ApplicationStatus",
    "GrantPortal",
    "GrantStatus",
    "Sector",
    "StartupStage",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "utcnow",
]


def empty_dict() -> dict[str, Any]:
    """Default factory for JSONB fields. (SQLModel chokes on `default={}`.)"""
    return {}

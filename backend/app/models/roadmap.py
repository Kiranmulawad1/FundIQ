"""FundingRoadmap — sequenced 12–18 month grant plan from the Planner agent.

Schema notes:
  - `plan` JSONB holds the structured roadmap (ordered grant nodes,
    target dates, dependencies, cash runway alignment).
  - `constraints` JSONB stores the user's stated constraints (max
    burn, must-include grants, blackout dates).
  - One *active* roadmap per startup at a time, enforced via partial
    unique index in migrations: `WHERE is_active`.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin, empty_dict


class FundingRoadmap(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "funding_roadmaps"
    __table_args__ = (Index("ix_roadmaps_startup", "startup_id"),)

    startup_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("startups.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )

    horizon_months: int = Field(default=12, ge=3, le=36)
    is_active: bool = Field(default=True, index=True)

    plan: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    constraints: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )

    # Provenance — which prompt version + agent run generated this roadmap.
    prompt_version: str | None = Field(default=None, max_length=32)
    agent_session_id: uuid.UUID | None = None

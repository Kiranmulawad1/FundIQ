"""AgentSession — one user-facing conversation/run across the 7-agent graph.

Schema notes:
  - `state` JSONB is the LangGraph state snapshot (checkpoint). Enables
    pause/resume + replay for the reasoning theater.
  - `conversation_history` JSONB is the user-visible chat log; kept
    separate from `state` because we render it directly in the UI.
  - `short_term_memory` / `long_term_memory_refs` / `episodic_memory`
    cover the three memory tiers. Long-term refs point to embeddings
    stored elsewhere (pgvector chunks); we keep IDs here for replay.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin, empty_dict


class AgentSession(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "agent_sessions"
    __table_args__ = (
        Index("ix_sessions_startup", "startup_id"),
        Index("ix_sessions_user", "owner_user_id"),
    )

    owner_user_id: str = Field(index=True)
    startup_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey("startups.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )

    title: str | None = Field(default=None, max_length=200)
    is_active: bool = Field(default=True, index=True)

    state: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )

    short_term_memory: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    long_term_memory_refs: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )
    episodic_memory: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
    )

"""UserFeedback — thumbs up/down + comment on an agent output.

This table directly feeds the eval set: thumbs-down events with
comments become new gold-set candidates after review (Phase 9).
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index
from sqlmodel import Column, Field, SQLModel

from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class UserFeedback(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "user_feedback"
    __table_args__ = (
        Index("ix_feedback_session", "session_id"),
        Index("ix_feedback_agent_thumbs", "agent_id", "thumbs_up"),
    )

    session_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("agent_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    owner_user_id: str = Field(index=True)
    agent_id: str = Field(max_length=64)
    thumbs_up: bool
    comment: str | None = None
    output_ref: str | None = Field(
        default=None,
        max_length=128,
        description="Identifier of the specific output being rated (e.g. message ID).",
    )

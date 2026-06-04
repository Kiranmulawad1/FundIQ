"""Alert — proactive grant match notification (Feature 6).

Schema notes:
  - One row per (startup, grant) match. `match_score` is the FRS-weighted
    similarity used by the alert worker.
  - `notification_sent_at` nullable — we record the match immediately
    and stamp the delivery later from the worker (so we can dedupe and
    retry without double-sending).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, UniqueConstraint
from sqlmodel import Column, Field, SQLModel

from app.models.base import AlertChannel, TimestampMixin, UUIDPrimaryKeyMixin


class Alert(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("startup_id", "grant_id", "channel", name="uq_alerts_triple"),
        Index("ix_alerts_pending", "notification_sent_at"),
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
            ForeignKey("grants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    channel: AlertChannel = Field(default=AlertChannel.EMAIL)
    match_score: float = Field(ge=0, le=1)
    notification_sent_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # type: ignore[arg-type]
    )
    failure_reason: str | None = None

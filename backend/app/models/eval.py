"""EvalResult — one LLM-as-judge scoring against a gold-set item.

Schema notes:
  - One row per (gold_set_item, agent, prompt_version) invocation. The
    CI gate aggregates across rows for regression detection.
  - `scores` JSONB carries the 4-dim rubric: faithfulness, relevance,
    completeness, citation_accuracy (each 0–10).
  - `gold_set_item_id` is a stable string identifier from the JSONL file,
    not an FK — gold set is file-versioned, not DB-versioned.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin, empty_dict


class EvalResult(UUIDPrimaryKeyMixin, TimestampMixin, SQLModel, table=True):
    __tablename__ = "eval_results"
    __table_args__ = (
        Index("ix_eval_agent_prompt", "agent_id", "prompt_version"),
        Index("ix_eval_gold_item", "gold_set_item_id"),
        Index("ix_eval_run", "run_id"),
    )

    run_id: uuid.UUID = Field(index=True, description="Groups all results from one eval run.")
    gold_set_item_id: str = Field(max_length=128, index=True)
    agent_id: str = Field(max_length=64)
    prompt_version: str = Field(max_length=32)
    model: str = Field(max_length=64, description="LLM that produced the output under judgement.")
    judge_model: str = Field(max_length=64, default="gpt-4o")

    scores: dict[str, float] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    aggregate_score: float = Field(ge=0, le=10)

    output: dict[str, Any] = Field(
        default_factory=empty_dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    rationale: str | None = None

    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

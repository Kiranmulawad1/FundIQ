"""API schemas for /agents/* — wrap the agent graph's I/O for the public
surface and add a trace block with stage timings.

The internal `AgentState` is a LangGraph TypedDict; this module turns the
relevant slices into a stable Pydantic shape the frontend can rely on.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.agents.state import CandidateScore, CriticFinding, GrantRecommendation


class StartupProfileInput(BaseModel):
    """Optional pre-conditioning for the Planner.

    Anything the founder filled in on /profile gets forwarded with each
    recommend call. All fields are optional; the Planner blends what's
    here with the explicit query, biasing extraction toward the profile.
    Pre-Clerk: this rides the request body. Post-Clerk: we'll persist to
    the `startups` table and look it up by `owner_user_id`.
    """

    name: str | None = Field(default=None, max_length=200)
    sector: str | None = Field(default=None, max_length=64)
    stage: str | None = Field(default=None, max_length=64)
    country: str | None = Field(default=None, min_length=2, max_length=2)
    federal_state: str | None = Field(default=None, max_length=64)
    funding_target_eur: int | None = Field(default=None, ge=0)
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Free-text context (team, traction, IP, etc.).",
    )


class AgentRecommendRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    session_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Stable per-browser UUID. If omitted, the server returns a "
            "fresh one in the response. Pre-Clerk: this is the anonymous "
            "user identity. Post-Clerk: this still scopes the chat; "
            "owner_user_id will be backfilled at first sign-in."
        ),
    )
    startup_profile: StartupProfileInput | None = Field(
        default=None,
        description=(
            "Optional structured profile that pre-conditions the Planner. "
            "When present, the Planner trusts it for facts and the query "
            "only needs to add what's specific to this question."
        ),
    )


class AgentTrace(BaseModel):
    """Stage-level observability for the frontend / eval harness."""

    rewritten_query: str
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    planner_ms: int = Field(ge=0)
    retrieval_ms: int = Field(ge=0)
    scorer_ms: int = Field(default=0, ge=0)
    writer_ms: int = Field(ge=0)
    critic_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    planner_rationale: str = ""
    scores: list[CandidateScore] = Field(
        default_factory=list,
        description=(
            "Per-candidate eligibility judgement from the Scorer. Empty when "
            "the Scorer fell back or there were no candidates to score."
        ),
    )
    critic_pass: bool = Field(
        default=True,
        description="True when the Critic found no issues with the Writer's response.",
    )
    critic_summary: str = Field(
        default="",
        description="One-sentence Critic verdict.",
    )
    critic_findings: list[CriticFinding] = Field(
        default_factory=list,
        description="Specific Critic findings. Empty when critic_pass is True.",
    )
    writer_attempts: int = Field(
        default=1,
        ge=1,
        description=(
            "How many times the Writer ran. >1 means the Critic rejected the "
            "first attempt and the Writer was re-run with the findings as feedback."
        ),
    )


class AgentRecommendResponse(BaseModel):
    session_id: uuid.UUID = Field(
        description="The session this recommendation belongs to. Echo back on subsequent calls to grow the same chat.",
    )
    summary: str
    recommendations: list[GrantRecommendation] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    trace: AgentTrace


class AgentConversationEntry(BaseModel):
    """One Q-and-A turn persisted into AgentSession.conversation_history."""

    ts: str = Field(description="ISO-8601 UTC timestamp of the turn.")
    query: str
    summary: str
    recommendations: list[GrantRecommendation] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    trace: AgentTrace


class AgentSessionResponse(BaseModel):
    """GET /agents/sessions/{id} response — full chat history for replay."""

    session_id: uuid.UUID
    history: list[AgentConversationEntry] = Field(default_factory=list)
    is_active: bool = True

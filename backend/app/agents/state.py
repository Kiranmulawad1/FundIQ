"""LangGraph state + I/O schemas for the recommend graph.

Graph shape (Phase 6 MVP):

    START → planner → retriever → writer → END

Each node mutates a slice of `AgentState`. State stays in-process for now
— persisting it into `AgentSession.state` (JSONB) is a follow-up that
needs auth wired up first.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, TypedDict

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.base import GrantPortal, Sector, StartupStage

# `Sector` is already imported above for PlannerFacts; we re-use it on
# CandidateGrant to surface the enrichment-derived primary sector.


# ---------------------------------------------------------------------------
# Planner output — structured facts extracted from the user's free-text query.
# Kept tolerant: every field is optional because the user may give us very
# little (e.g. "money for AI"). The Writer reasons over whatever's present.
# ---------------------------------------------------------------------------
class PlannerFacts(BaseModel):
    sector: Sector | None = Field(default=None)
    stage: StartupStage | None = Field(default=None)
    country: str | None = Field(
        default=None,
        description="Two-letter ISO code (DE / EU) or None.",
        min_length=2,
        max_length=2,
    )
    federal_state: str | None = Field(default=None, max_length=64)
    funding_target_eur: int | None = Field(default=None, ge=0)

    def to_filter_kwargs(self) -> dict[str, Any]:
        """Filters the existing RAG pipeline accepts (country only today).

        Sector / stage / federal_state / funding-target are NOT pre-filters
        because the corpus doesn't have those fields populated cleanly yet
        (Phase 2D + data enrichment). The Writer still uses them to reason.
        """
        kw: dict[str, Any] = {}
        if self.country:
            kw["country"] = self.country.upper()
        return kw


class PlannerOutput(BaseModel):
    rewritten_query: str = Field(
        description="Query optimised for dense + sparse retrieval.",
        min_length=1,
    )
    facts: PlannerFacts = Field(default_factory=PlannerFacts)
    rationale: str = Field(
        default="",
        description="Optional short explanation of why these facts were extracted.",
    )


# ---------------------------------------------------------------------------
# Retriever output — flattened candidate shape so it survives JSON
# serialisation when we later persist to AgentSession.state.
# ---------------------------------------------------------------------------
class CandidateGrant(BaseModel):
    grant_id: uuid.UUID
    title: str
    portal: GrantPortal
    source_url: str
    source_doc_id: str | None = None
    summary: str
    body_excerpt: str = Field(
        description="First ~600 chars of body, fed to the Writer for grounding.",
    )
    country: str | None = None
    federal_state: str | None = None
    funding_min_eur: float | None = None
    funding_max_eur: float | None = None
    deadline: str | None = None  # ISO string, None if rolling
    # Structured enrichment fields populated by app.services.grant_enrichment.
    # Empty objects when the grant hasn't been enriched yet; the prompts
    # tolerate the missing data and fall back to body_excerpt parsing.
    sector: Sector | None = None
    eligibility: dict[str, Any] = Field(default_factory=dict)
    # Provenance from the retrieval pipeline.
    final_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Scorer output — structured eligibility judgement per candidate.
#
# The Scorer is the 4th agent in the graph (between Retriever and Writer).
# It produces a typed judgement the Writer can cite, so the final
# `fit` labels and `caveats` don't come from the Writer's improvised
# intuition — they trace back to a deliberate eligibility decision.
# ---------------------------------------------------------------------------
class CandidateScore(BaseModel):
    grant_id: uuid.UUID
    eligibility_score: int = Field(ge=0, le=100)
    fit_label: Literal["high", "medium", "low"]
    strengths: list[str] = Field(
        default_factory=list,
        description="1-3 concrete reasons this grant matches the founder.",
        max_length=5,
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="1-3 things that could disqualify or weaken the fit.",
        max_length=5,
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="1-2 things the Planner couldn't determine that would help us judge better.",
        max_length=5,
    )

    # Tolerant validators — Gemini occasionally returns "Medium", numeric
    # strings, or floats. Without normalisation a single bad field would
    # invalidate the entire ScorerOutput and we'd lose all judgements.

    @field_validator("fit_label", mode="before")
    @classmethod
    def _normalize_fit_label(cls, v: Any) -> str:
        if v is None:
            return "low"
        s = str(v).lower().strip()
        if s in {"high", "h"} or s.startswith("high"):
            return "high"
        if s in {"medium", "med", "m"} or s.startswith("med"):
            return "medium"
        return "low"

    @field_validator("eligibility_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> int:
        try:
            n = round(float(v))
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, n))


class ScorerOutput(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Critic output — quality review of the Writer's response.
#
# Phase 1: one-shot review. The findings ride along on the response so the
# UI can show "Quality review" and the user (or a downstream eval harness)
# can audit groundedness. A Phase 2 evolution turns `overall_pass=False`
# into a conditional edge back to the Writer with the findings as
# feedback — for now it's purely advisory.
# ---------------------------------------------------------------------------
class CriticFinding(BaseModel):
    type: Literal[
        "citation_faithfulness",
        "fit_alignment",
        "caveat_omission",
        "language_mismatch",
        "profile_misuse",
        "other",
    ]
    severity: Literal["high", "medium", "low"]
    grant_id: uuid.UUID | None = Field(
        default=None,
        description="The recommendation this finding refers to, or null if general.",
    )
    message: str = Field(
        max_length=500,
        description="Plain-English explanation of the issue.",
    )

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v: Any) -> str:
        if v is None:
            return "low"
        s = str(v).lower().strip()
        if s.startswith("high") or s in {"critical", "blocker"}:
            return "high"
        if s.startswith("med") or s.startswith("mod"):
            return "medium"
        return "low"


class CriticOutput(BaseModel):
    overall_pass: bool = Field(
        description="Did the Writer's output meet quality standards?",
    )
    summary: str = Field(
        default="",
        max_length=400,
        description="One-sentence overall assessment.",
    )
    findings: list[CriticFinding] = Field(
        default_factory=list,
        description="Specific issues found; empty when overall_pass is True.",
        max_length=10,
    )


# ---------------------------------------------------------------------------
# Writer output — the final answer.
# ---------------------------------------------------------------------------
class GrantRecommendation(BaseModel):
    grant_id: uuid.UUID
    grant_title: str
    portal: GrantPortal
    source_url: str
    fit: Literal["high", "medium", "low"]
    rationale: str = Field(
        description="2-3 sentences citing concrete reasons.",
        max_length=1200,
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Things the user should verify before applying.",
    )


class WriterOutput(BaseModel):
    summary: str = Field(
        description="1-2 sentence overview of what we found.",
        max_length=600,
    )
    recommendations: list[GrantRecommendation] = Field(default_factory=list)
    questions_for_user: list[str] = Field(
        default_factory=list,
        description="Up to 3 clarifying questions if the query is ambiguous.",
    )


# ---------------------------------------------------------------------------
# Graph state — TypedDict required by LangGraph.
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # Input (always set on entry)
    query: str
    # Optional saved startup profile (forwarded from the request).
    startup_profile: dict[str, Any]

    # Planner outputs
    planner: PlannerOutput
    planner_ms: int

    # Retriever outputs
    candidates: list[CandidateGrant]
    retrieval_ms: int

    # Scorer outputs
    scorer: ScorerOutput
    scorer_ms: int

    # Writer outputs
    writer: WriterOutput
    writer_ms: int
    # Bumped each time the Writer runs. Starts at 1; the conditional edge
    # after the Critic can route back to writer_node when the Critic
    # rejected the previous attempt — see graph.py and Writer node.
    writer_attempts: int

    # Critic outputs
    critic: CriticOutput
    critic_ms: int

    # Final error envelope — set if any node aborts.
    error: str | None

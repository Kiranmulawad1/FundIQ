"""Scorer node — judges each candidate's eligibility before the Writer
sees them.

Input:  AgentState["planner"] (PlannerOutput with structured facts)
        AgentState["candidates"] (list[CandidateGrant])
Output: AgentState["scorer"] (ScorerOutput) + scorer_ms

The Scorer makes ONE Gemini call that returns a per-candidate score:

    [
      { grant_id, eligibility_score: 0-100, fit_label, strengths, concerns, missing_info },
      ...
    ]

Why a separate node instead of asking the Writer to score implicitly:
  - The Writer's `fit` label drifts when its primary job is prose. By
    splitting the judgement out, we get a deterministic, structured
    eligibility decision that the Writer can faithfully cite.
  - The score + strengths/concerns become the Writer's source of truth
    for caveats — no more "Writer guessed plausible warnings".
  - It opens the door to non-Writer consumers later (a dashboard, an
    eval harness) that want a typed eligibility view per grant.

Failure mode: if Gemini errors or returns invalid JSON, we fall back to
an empty ScorerOutput. The Writer then runs as before — it just doesn't
get the structured judgement to lean on. Functionally a no-op.
"""

from __future__ import annotations

import json as _json
import time
from typing import TYPE_CHECKING

from app.agents.llm import AgentLLMError
from app.agents.state import AgentState, CandidateGrant, ScorerOutput
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient
    from app.agents.state import PlannerOutput

logger = get_logger(__name__)

SCORER_PROMPT = """\
You are the Eligibility Scorer in a multi-agent grant-recommendation system.
Given the Planner's structured facts about the founder, and a shortlist of
candidate grant programmes retrieved by the Retriever, produce a typed
eligibility judgement FOR EACH candidate.

Founder facts (may be partial — null fields mean "not specified"):
{planner_json}

Candidates:
{candidates_json}

For EACH candidate, return:
  - grant_id (must match the candidate's grant_id exactly)
  - eligibility_score: integer 0-100
      90-100: clearly designed for this founder's situation
      70-89:  strong fit, minor caveats
      50-69:  plausible fit, real gaps to clarify
      30-49:  stretch fit, would need exceptional case
      0-29:   poor fit, only include if no better candidate exists
  - fit_label: "high" | "medium" | "low"  (must align with the score)
  - strengths: 1-3 concrete reasons this grant matches the founder's facts
  - concerns: 1-3 things that might disqualify or weaken the fit
  - missing_info: 0-2 things the Planner couldn't determine that would help
                  us judge better (e.g. "Is the founder a current student?")

STRICT:
  - Score every candidate in the list (do not skip any).
  - Use grant_ids from the candidate list — never invent new ones.
  - Match the founder's language only in `strengths`/`concerns`/`missing_info`;
    keep field names English.

Return ONLY a JSON object of this exact shape:
{{
  "scores": [
    {{
      "grant_id": "UUID string",
      "eligibility_score": integer,
      "fit_label": "high" | "medium" | "low",
      "strengths": ["string", ...],
      "concerns": ["string", ...],
      "missing_info": ["string", ...]
    }}
  ]
}}
No prose before or after. No markdown fences."""


def render_candidates_for_scorer(candidates: list[CandidateGrant]) -> str:
    """Trimmed candidate view — Scorer doesn't need scores/ranks.

    Includes the enrichment-derived `sector` + `eligibility` block when
    populated so the Scorer reasons against structured criteria instead
    of re-extracting from the body_excerpt on every call.
    """
    out = []
    for i, c in enumerate(candidates, start=1):
        item: dict[str, object] = {
            "rank": i,
            "grant_id": str(c.grant_id),
            "title": c.title,
            "portal": c.portal.value,
            "country": c.country,
            "federal_state": c.federal_state,
            "funding_min_eur": c.funding_min_eur,
            "funding_max_eur": c.funding_max_eur,
            "summary": c.summary,
            "body_excerpt": c.body_excerpt,
        }
        if c.sector is not None:
            item["sector"] = c.sector.value
        if c.eligibility:
            # Strip the bookkeeping keys — the Scorer doesn't need them.
            item["eligibility"] = {
                k: v for k, v in c.eligibility.items()
                if k not in {"enrichment_version", "enriched_at"}
            }
        out.append(item)
    return _json.dumps(out, ensure_ascii=False, indent=2)


def build_scorer_prompt(*, planner: PlannerOutput, candidates: list[CandidateGrant]) -> str:
    return SCORER_PROMPT.format(
        planner_json=planner.model_dump_json(indent=2),
        candidates_json=render_candidates_for_scorer(candidates),
    )


async def scorer_node(
    state: AgentState,
    *,
    llm: GeminiAgentClient,
) -> AgentState:
    candidates: list[CandidateGrant] = state["candidates"]
    planner = state["planner"]
    started = time.perf_counter()

    if not candidates:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {"scorer": ScorerOutput(scores=[]), "scorer_ms": elapsed_ms}

    prompt = build_scorer_prompt(planner=planner, candidates=candidates)
    try:
        out = await llm.respond_as(
            ScorerOutput,
            prompt=prompt,
            temperature=0.2,  # deterministic judgement matters here
            # 4096 truncates the JSON mid-list when there are 6-8 candidates
            # with rich strengths/concerns. 8192 gives generous headroom on
            # Gemini 2.5 Flash's free tier and unused tokens are free.
            max_output_tokens=8192,
        )
        out = _enforce_grounded_scores(out, candidates)
    except AgentLLMError as e:
        logger.warning("agents.scorer.fallback", error=str(e)[:200])
        out = ScorerOutput(scores=[])

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "agents.scorer.done",
        elapsed_ms=elapsed_ms,
        score_count=len(out.scores),
        candidate_count=len(candidates),
    )
    return {"scorer": out, "scorer_ms": elapsed_ms}


def _enforce_grounded_scores(
    out: ScorerOutput,
    candidates: list[CandidateGrant],
) -> ScorerOutput:
    """Drop any score whose grant_id isn't in the candidate set — Gemini
    occasionally hallucinates IDs even when forbidden.
    """
    valid = {c.grant_id for c in candidates}
    kept = [s for s in out.scores if s.grant_id in valid]
    if len(kept) != len(out.scores):
        logger.warning(
            "agents.scorer.hallucinated_scores_dropped",
            kept=len(kept),
            total=len(out.scores),
        )
    return out.model_copy(update={"scores": kept})

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
from app.core.prompts import CompiledPrompt, PromptFetchError, get_prompt

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient
    from app.agents.state import PlannerOutput

logger = get_logger(__name__)

# Prompt lives in Langfuse under name "scorer".


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


def build_scorer_prompt(
    *, planner: PlannerOutput, candidates: list[CandidateGrant],
) -> CompiledPrompt:
    return get_prompt("scorer").compile(
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

    try:
        compiled = build_scorer_prompt(planner=planner, candidates=candidates)
        out = await llm.respond_as(
            ScorerOutput,
            prompt=compiled.text,
            prompt_handle=compiled.langfuse_handle,
            temperature=0.2,  # deterministic judgement matters here
            # 4096 truncates the JSON mid-list when there are 6-8 candidates
            # with rich strengths/concerns. 8192 gives generous headroom on
            # Gemini 2.5 Flash's free tier and unused tokens are free.
            max_output_tokens=8192,
        )
        out = _enforce_grounded_scores(out, candidates)
    except (AgentLLMError, PromptFetchError) as e:
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

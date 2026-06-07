"""Critic node — quality review of the Writer's response.

Input:  AgentState["query"], state["planner"], state["candidates"],
        state["scorer"], state["writer"], (optional) state["startup_profile"]
Output: AgentState["critic"] (CriticOutput) + critic_ms

The Critic is advisory in Phase 1: it ships a list of findings but does
not loop back to the Writer. Findings surface in the response trace so
the UI can render a "Quality review" panel and a future eval harness
can count Critic regressions.

Why a Critic at all when we already have `_enforce_groundedness`:
  - The groundedness guard only catches the cheap case (grant_id not in
    candidate set). The Critic looks at semantic faithfulness: does the
    rationale's claim actually appear in the grant's body excerpt? Does
    the Writer's `fit` match the Scorer's `fit_label`? Did it drop a
    Scorer concern from the caveats?
  - These are LLM-judgment calls the deterministic guard can't make.

Failure mode: if Gemini fails, we degrade to a `overall_pass=True,
findings=[]` "review unavailable" output. The user still gets the
Writer's answer — we just don't surface a review for that turn.
"""

from __future__ import annotations

import json as _json
import time
from typing import TYPE_CHECKING

from app.agents.llm import AgentLLMError
from app.agents.state import (
    AgentState,
    CandidateGrant,
    CriticOutput,
    ScorerOutput,
    WriterOutput,
)
from app.core.logging import get_logger
from app.core.prompts import CompiledPrompt, PromptFetchError, get_prompt

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient
    from app.agents.state import PlannerOutput

logger = get_logger(__name__)


# Prompt lives in Langfuse under name "critic".


def _render_candidates_for_critic(candidates: list[CandidateGrant]) -> str:
    """Same trimmed shape the Scorer sees, plus the body excerpt so the
    Critic can check rationale claims against actual ground truth.
    """
    out = []
    for i, c in enumerate(candidates, start=1):
        out.append({
            "rank": i,
            "grant_id": str(c.grant_id),
            "title": c.title,
            "portal": c.portal.value,
            "country": c.country,
            "federal_state": c.federal_state,
            "summary": c.summary,
            "body_excerpt": c.body_excerpt,
        })
    return _json.dumps(out, ensure_ascii=False, indent=2)


def _render_profile_block(profile: dict[str, object] | None) -> str:
    if not profile:
        return ""
    filled = {k: v for k, v in profile.items() if v not in (None, "", [])}
    if not filled:
        return ""
    lines = ["Saved startup profile (use to check profile_misuse):"]
    for k, v in filled.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n\n"


def build_critic_prompt(
    *,
    query: str,
    planner: "PlannerOutput",
    candidates: list[CandidateGrant],
    scorer: ScorerOutput,
    writer: WriterOutput,
    startup_profile: dict[str, object] | None,
) -> CompiledPrompt:
    return get_prompt("critic").compile(
        query=query,
        profile_block=_render_profile_block(startup_profile),
        planner_json=planner.model_dump_json(indent=2),
        candidates_json=_render_candidates_for_critic(candidates),
        scorer_json=scorer.model_dump_json(indent=2),
        writer_json=writer.model_dump_json(indent=2),
    )


async def critic_node(
    state: AgentState,
    *,
    llm: GeminiAgentClient,
) -> AgentState:
    candidates: list[CandidateGrant] = state["candidates"]
    planner = state["planner"]
    writer = state["writer"]
    scorer = state.get("scorer") or ScorerOutput(scores=[])
    profile_raw = state.get("startup_profile")
    profile = profile_raw if isinstance(profile_raw, dict) else None
    query = state["query"]
    started = time.perf_counter()

    if not writer.recommendations or not candidates:
        # Nothing to audit — Writer either failed gracefully or had no
        # candidates to work with. Skip the LLM call entirely.
        out = CriticOutput(
            overall_pass=True,
            summary="No recommendations to review.",
            findings=[],
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {"critic": out, "critic_ms": elapsed_ms}

    try:
        compiled = build_critic_prompt(
            query=query,
            planner=planner,
            candidates=candidates,
            scorer=scorer,
            writer=writer,
            startup_profile=profile,
        )
        out = await llm.respond_as(
            CriticOutput,
            prompt=compiled.text,
            prompt_handle=compiled.langfuse_handle,
            temperature=0.1,  # judgement should be deterministic
            # The findings list can stretch when the Writer is bad; 4096
            # is plenty since each finding is bounded to 500 chars.
            max_output_tokens=4096,
        )
        out = _enforce_grounded_findings(out, candidates)
    except (AgentLLMError, PromptFetchError) as e:
        logger.warning("agents.critic.fallback", error=str(e)[:200])
        out = CriticOutput(
            overall_pass=True,  # advisory only — don't block the user
            summary=f"Critic unavailable: {str(e)[:150]}",
            findings=[],
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "agents.critic.done",
        elapsed_ms=elapsed_ms,
        overall_pass=out.overall_pass,
        finding_count=len(out.findings),
    )
    return {"critic": out, "critic_ms": elapsed_ms}


def _enforce_grounded_findings(
    out: CriticOutput,
    candidates: list[CandidateGrant],
) -> CriticOutput:
    """Drop findings whose `grant_id` doesn't match any candidate.

    The Critic occasionally cites hallucinated grant_ids when summarising;
    we drop those for the same reason we drop hallucinated recommendations.
    Findings with `grant_id=None` (general findings) always pass.
    """
    valid = {c.grant_id for c in candidates}
    kept: list = []
    for f in out.findings:
        if f.grant_id is None or f.grant_id in valid:
            kept.append(f)
    if len(kept) != len(out.findings):
        logger.warning(
            "agents.critic.hallucinated_findings_dropped",
            kept=len(kept),
            total=len(out.findings),
        )
    return out.model_copy(update={"findings": kept})

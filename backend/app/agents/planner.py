"""Planner node — turn a free-text founder question into structured facts.

Input:  AgentState["query"]
Output: AgentState["planner"] (PlannerOutput) + planner_ms

The Planner does TWO things in one Gemini call:
  1. Rewrite the query into a retrieval-friendly form (drops first person,
     adds disambiguating terms inferred from context).
  2. Extract optional structured facts (sector, stage, country, funding
     target) that the Writer can reason over.

Why one call instead of two? It's cheaper, and the LLM benefits from
keeping both tasks in one context — the rewrite improves when it knows
the structured facts it just extracted.

Failure mode: if the LLM call fails, we fall back to using the original
query as the rewritten one and emit an empty fact set. The graph then
runs Retriever + Writer on degraded but non-broken state, so the user
still gets a useful response.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.agents.llm import AgentLLMError
from app.agents.state import AgentState, PlannerFacts, PlannerOutput
from app.core.logging import get_logger
from app.core.prompts import PromptFetchError, get_prompt

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient

logger = get_logger(__name__)

# Prompt lives in Langfuse under name "planner". Fetched via
# app.core.prompts.get_prompt() at request time.


def _render_profile_block(profile: dict[str, object] | None) -> str:
    """If the request provided a saved startup profile, render it as
    context for the Planner. Empty when no profile, so the prompt stays
    identical to the no-profile case (no whitespace drift).
    """
    if not profile:
        return ""
    # Drop null / empty values so the prompt only carries facts the
    # founder actually filled in.
    filled = {k: v for k, v in profile.items() if v not in (None, "", [])}
    if not filled:
        return ""
    lines = ["Known startup profile (trust these unless the question contradicts them):"]
    for k, v in filled.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n\n"


async def planner_node(
    state: AgentState,
    *,
    llm: GeminiAgentClient,
) -> AgentState:
    """LangGraph node — returns the state slice it owns."""
    query = state["query"]
    profile_raw = state.get("startup_profile")
    profile = profile_raw if isinstance(profile_raw, dict) else None
    started = time.perf_counter()
    try:
        compiled = get_prompt("planner").compile(
            query=query,
            profile_block=_render_profile_block(profile),
        )
        out = await llm.respond_as(
            PlannerOutput,
            prompt=compiled.text,
            prompt_handle=compiled.langfuse_handle,
            temperature=0.3,
            # 512 occasionally truncates the JSON mid-string on Gemini 2.5 Flash
            # when the rationale runs long — 1024 buys generous headroom.
            max_output_tokens=1024,
        )
    except (AgentLLMError, PromptFetchError) as e:
        logger.warning("agents.planner.fallback", error=str(e)[:200])
        # Fallback: use the original query, no facts.
        out = PlannerOutput(
            rewritten_query=query,
            facts=PlannerFacts(),
            rationale=f"Planner unavailable ({e!s}); falling back to raw query.",
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "agents.planner.done",
        elapsed_ms=elapsed_ms,
        country=out.facts.country,
        sector=out.facts.sector.value if out.facts.sector else None,
        stage=out.facts.stage.value if out.facts.stage else None,
        funding_target=out.facts.funding_target_eur,
    )
    return {"planner": out, "planner_ms": elapsed_ms}

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

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient

logger = get_logger(__name__)

PLANNER_PROMPT = """\
You are the Planner in a multi-agent system that recommends EU and German
startup funding grants. The user is a founder asking a question in free text.

Your job:
  1. Rewrite their question into a concise retrieval query (DE or EN to match
     their language), suitable for semantic search over grant programme
     descriptions. If a known startup profile is provided below, weave the
     most query-relevant profile facts (sector, stage, federal_state,
     funding_target) into the rewritten query so semantic search benefits
     from them.
  2. Combine the known profile (if any) with whatever the question itself
     implies, and return the merged structured facts. Treat profile facts
     as authoritative unless the question contradicts them. If the question
     doesn't add or contradict anything, just echo the profile values.

Allowed values:
  sector: one of [deeptech, cleantech, health, biotech, saas, hardware,
                  fintech, other] or null
  stage: one of [idea, seed, growth] or null (idea = pre-revenue / pre-product,
         seed = early traction, growth = scaling)
  country: "DE" for Germany-specific programmes, "EU" for EU-wide, or null
  federal_state: one of the 16 German Länder (e.g. "Bayern", "Baden-Württemberg",
                 "Berlin", "Nordrhein-Westfalen") or null
  funding_target_eur: integer EUR amount the founder wants, or null

{profile_block}Founder question:
{query}

Return ONLY a JSON object with this exact shape:
{{
  "rewritten_query": "string — the search query",
  "facts": {{
    "sector": "deeptech" | "cleantech" | "health" | "biotech" | "saas" | "hardware" | "fintech" | "other" | null,
    "stage": "idea" | "seed" | "growth" | null,
    "country": "DE" | "EU" | null,
    "federal_state": "string" | null,
    "funding_target_eur": integer | null
  }},
  "rationale": "string — one short sentence explaining your fact extraction"
}}
No prose before or after. No markdown fences."""


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
        out = await llm.respond_as(
            PlannerOutput,
            prompt=PLANNER_PROMPT.format(
                query=query,
                profile_block=_render_profile_block(profile),
            ),
            temperature=0.3,
            # 512 occasionally truncates the JSON mid-string on Gemini 2.5 Flash
            # when the rationale runs long — 1024 buys generous headroom.
            max_output_tokens=1024,
        )
    except AgentLLMError as e:
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

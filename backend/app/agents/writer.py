"""Writer node — synthesises the final recommendation.

Input:  AgentState["query"], AgentState["planner"], AgentState["candidates"]
Output: AgentState["writer"] (WriterOutput) + writer_ms

Guard-rails baked into the prompt:
  - Only cite the grants we provide. If none fit, say so.
  - Use the planner's structured facts to reason about eligibility, but
    don't assert facts about the founder we don't have.
  - Output strictly the JSON schema we feed back in.

If Gemini fails entirely we degrade to a deterministic "here are the top
N retrieved grants" response so the user always gets something. The
trace still shows the failure so the UI / logs surface it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.agents.llm import AgentLLMError
from app.agents.state import (
    AgentState,
    CandidateGrant,
    CriticFinding,
    GrantRecommendation,
    ScorerOutput,
    WriterOutput,
)
from app.core.logging import get_logger
from app.core.prompts import CompiledPrompt, PromptFetchError, get_prompt

if TYPE_CHECKING:
    from app.agents.llm import GeminiAgentClient
    from app.agents.state import PlannerOutput


logger = get_logger(__name__)


# Prompt lives in Langfuse under name "writer".


async def writer_node(
    state: AgentState,
    *,
    llm: GeminiAgentClient,
) -> AgentState:
    candidates: list[CandidateGrant] = state["candidates"]
    planner = state["planner"]
    scorer = state.get("scorer") or ScorerOutput(scores=[])
    # On a retry, the previous Critic's findings are forwarded via state.
    # On the first attempt, both fields are absent; treat them as empty.
    critic = state.get("critic")
    critic_feedback = (
        critic.findings if (critic and not critic.overall_pass) else None
    )
    attempt = state.get("writer_attempts", 0) + 1
    query = state["query"]
    started = time.perf_counter()

    if not candidates:
        out = WriterOutput(
            summary=(
                "No grants matched your question with strong confidence. "
                "Try a more specific query, or broaden the geography / sector."
            ),
            recommendations=[],
            questions_for_user=[],
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("agents.writer.empty_candidates", elapsed_ms=elapsed_ms)
        return {"writer": out, "writer_ms": elapsed_ms, "writer_attempts": attempt}

    try:
        compiled = build_writer_prompt(
            query=query,
            planner=planner,
            candidates=candidates,
            scorer=scorer,
            critic_feedback=critic_feedback,
        )
        out = await llm.respond_as(
            WriterOutput,
            prompt=compiled.text,
            prompt_handle=compiled.langfuse_handle,
            temperature=0.3,
            # 4096 occasionally truncates mid-string when the Writer expands
            # caveats across all 8 candidates. 8192 is well within Gemini
            # 2.5 Flash's free-tier per-call cap and eliminates truncation
            # in practice. Output billing is per-token so unused headroom
            # is free.
            max_output_tokens=8192,
        )
        out = _enforce_groundedness(out, candidates)
    except (AgentLLMError, PromptFetchError) as e:
        logger.warning("agents.writer.fallback", error=str(e)[:200])
        out = _deterministic_fallback(candidates, error=str(e))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "agents.writer.done",
        elapsed_ms=elapsed_ms,
        attempt=attempt,
        recommendation_count=len(out.recommendations),
    )
    return {"writer": out, "writer_ms": elapsed_ms, "writer_attempts": attempt}


def _render_retry_block(feedback: list[CriticFinding] | None) -> str:
    """Render Critic findings as a retry directive that primes the
    Writer to address each issue. Empty when no retry is in flight.
    """
    if not feedback:
        return ""
    lines = [
        "You are RETRYING. A previous attempt was reviewed by the Critic and",
        "the following issues were flagged. Address EACH one in your output:",
    ]
    for f in feedback:
        gid = f" (grant_id={f.grant_id})" if f.grant_id else ""
        lines.append(f"  - [{f.severity}] {f.type}{gid}: {f.message}")
    return "\n".join(lines) + "\n\n"


def build_writer_prompt(
    *,
    query: str,
    planner: "PlannerOutput",
    candidates: list[CandidateGrant],
    scorer: ScorerOutput,
    critic_feedback: list[CriticFinding] | None = None,
) -> CompiledPrompt:
    """Public helper so the streaming code path in api/routes/agents.py
    can reuse the exact same prompt the batch writer_node uses.

    `scorer` may have an empty `scores` list — that's fine; the prompt
    tells the Writer to fall back to its own judgement in that case.
    `critic_feedback` is non-None only on the retry attempt: the Writer
    is asked to address each finding by name.
    """
    return get_prompt("writer").compile(
        retry_block=_render_retry_block(critic_feedback),
        query=query,
        planner_json=planner.model_dump_json(indent=2),
        candidates_json=_render_candidates(candidates),
        scorer_json=scorer.model_dump_json(indent=2),
    )


def enforce_groundedness(
    out: WriterOutput, candidates: list[CandidateGrant],
) -> WriterOutput:
    """Public alias for _enforce_groundedness — same behaviour."""
    return _enforce_groundedness(out, candidates)


def deterministic_writer_fallback(
    candidates: list[CandidateGrant], *, error: str,
) -> WriterOutput:
    """Public alias for _deterministic_fallback — same behaviour."""
    return _deterministic_fallback(candidates, error=error)


def _render_candidates(candidates: list[CandidateGrant]) -> str:
    """Compact JSON representation fed into the prompt.

    We deliberately drop fields the Writer doesn't need (final_score,
    dense_rank, etc.) — they invite "noise" reasoning about ranking that
    we want the Writer to ignore.
    """
    out = []
    for i, c in enumerate(candidates, start=1):
        item: dict[str, object] = {
            "rank": i,
            "grant_id": str(c.grant_id),
            "title": c.title,
            "portal": c.portal.value,
            "source_url": c.source_url,
            "source_doc_id": c.source_doc_id,
            "country": c.country,
            "federal_state": c.federal_state,
            "funding_min_eur": c.funding_min_eur,
            "funding_max_eur": c.funding_max_eur,
            "deadline": c.deadline,
            "summary": c.summary,
            "body_excerpt": c.body_excerpt,
        }
        if c.sector is not None:
            item["sector"] = c.sector.value
        if c.eligibility:
            item["eligibility"] = {
                k: v for k, v in c.eligibility.items()
                if k not in {"enrichment_version", "enriched_at"}
            }
        out.append(item)
    import json as _json
    return _json.dumps(out, ensure_ascii=False, indent=2)


def _enforce_groundedness(
    out: WriterOutput,
    candidates: list[CandidateGrant],
) -> WriterOutput:
    """Drop any recommendation whose grant_id isn't in the candidate set.

    The prompt forbids hallucinated grants but Gemini occasionally
    confabulates an ID. Cheap defence-in-depth.
    """
    valid_ids = {c.grant_id for c in candidates}
    kept = [r for r in out.recommendations if r.grant_id in valid_ids]
    if len(kept) != len(out.recommendations):
        logger.warning(
            "agents.writer.hallucinated_recs_dropped",
            kept=len(kept),
            total=len(out.recommendations),
        )
    return out.model_copy(update={"recommendations": kept})


def _deterministic_fallback(
    candidates: list[CandidateGrant],
    *,
    error: str,
) -> WriterOutput:
    """When Gemini is unavailable, surface the top-3 retrieval hits as
    medium-fit recommendations with a transparent caveat.
    """
    recs = [
        GrantRecommendation(
            grant_id=c.grant_id,
            grant_title=c.title,
            portal=c.portal,
            source_url=c.source_url,
            fit="medium",
            rationale=(
                "Retrieved as a strong semantic match; the Writer model was "
                "unavailable so this recommendation comes from the retrieval "
                "pipeline only — verify fit manually."
            ),
            caveats=[
                "Writer agent fallback: rationale was not LLM-generated.",
            ],
        )
        for c in candidates[:3]
    ]
    return WriterOutput(
        summary=(
            f"Writer model unavailable ({error[:120]}). Returning the top "
            "retrieval matches without LLM-curated rationale."
        ),
        recommendations=recs,
        questions_for_user=[],
    )

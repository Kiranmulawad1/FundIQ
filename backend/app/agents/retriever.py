"""Retriever node — wraps the existing RAG pipeline so the agent graph
benefits from the same dense + sparse + RRF + reranker stack used by
/grants/search.

Input: AgentState["planner"] (PlannerOutput with rewritten_query + facts)
Output: AgentState["candidates"] (list[CandidateGrant]) + retrieval_ms

We default to HYBRID_RERANK mode. The eval (Phase 5C) showed plain dense
is competitive on this corpus, but the reranker shines on vague / multi-
intent queries — exactly what we expect agent queries to be. HyDE stays
OFF here because the Planner already rewrote the query; layered rewriting
would over-fit.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.agents.state import AgentState, CandidateGrant
from app.core.logging import get_logger
from app.rag.pipeline import RetrievalMode, RetrievalPipeline

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)

# How many candidates the Writer sees. Keep small — context for the Writer
# stays cheap and the LLM does better with 6-8 grants than with 20.
DEFAULT_CANDIDATE_LIMIT = 8
BODY_EXCERPT_CHARS = 600


async def retriever_node(
    state: AgentState,
    *,
    session: AsyncSession,
    pipeline: RetrievalPipeline,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> AgentState:
    """LangGraph node — runs hybrid_rerank retrieval over the planner's
    rewritten query, then flattens results into JSON-safe CandidateGrant
    objects.
    """
    planner = state["planner"]
    facts = planner.facts
    started = time.perf_counter()

    result = await pipeline.retrieve(
        session,
        query=planner.rewritten_query,
        mode=RetrievalMode.HYBRID_RERANK,
        limit=limit,
        **facts.to_filter_kwargs(),
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    candidates: list[CandidateGrant] = []
    for h in result.hits:
        g = h.grant
        body = (g.body or "").strip()
        candidates.append(
            CandidateGrant(
                grant_id=g.id,
                title=g.title,
                portal=g.portal,
                source_url=g.source_url,
                source_doc_id=g.source_doc_id,
                summary=g.summary or "",
                body_excerpt=body[:BODY_EXCERPT_CHARS] + ("…" if len(body) > BODY_EXCERPT_CHARS else ""),
                country=g.country,
                federal_state=g.federal_state,
                funding_min_eur=float(g.funding_min_eur) if g.funding_min_eur is not None else None,
                funding_max_eur=float(g.funding_max_eur) if g.funding_max_eur is not None else None,
                deadline=g.deadline.isoformat() if g.deadline is not None else None,
                sector=g.sector,
                eligibility=g.eligibility or {},
                final_score=h.final_score,
                dense_rank=h.dense_rank,
                sparse_rank=h.sparse_rank,
                rerank_score=h.rerank_score,
            )
        )

    logger.info(
        "agents.retriever.done",
        elapsed_ms=elapsed_ms,
        rewritten_query=planner.rewritten_query[:120],
        candidate_count=len(candidates),
        retrieval_elapsed_ms=result.elapsed_ms,
        dense_count=result.dense_count,
        sparse_count=result.sparse_count,
    )
    return {"candidates": candidates, "retrieval_ms": elapsed_ms}

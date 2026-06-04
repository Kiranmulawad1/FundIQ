"""LangGraph wiring for the Phase 6 recommend graph.

    START → planner → retriever → writer → END

Linear today; the file is structured so adding nodes (eligibility scorer,
critic, memory) is a matter of registering them with `StateGraph` and
inserting an edge.

The compiled graph is constructed PER REQUEST because each invocation
needs a fresh DB session and pipeline. The Gemini client can be shared
across requests (it owns an httpx connection pool) — passed in by the
route handler.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.agents.critic import critic_node
from app.agents.planner import planner_node
from app.agents.retriever import retriever_node
from app.agents.scorer import scorer_node
from app.agents.state import AgentState
from app.agents.writer import writer_node

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.agents.llm import GeminiAgentClient
    from app.rag.pipeline import RetrievalPipeline


def build_graph(
    *,
    session: AsyncSession,
    pipeline: RetrievalPipeline,
    llm: GeminiAgentClient,
):  # type: ignore[no-untyped-def]
    """Compile the recommend graph with the right dependencies bound to
    each node. Returns a `CompiledStateGraph` ready to `ainvoke()`.
    """
    # Node names are suffixed `_node` to avoid clashing with same-named
    # fields on `AgentState` (LangGraph forbids the overlap).
    g: StateGraph = StateGraph(AgentState)

    g.add_node("planner_node", partial(planner_node, llm=llm))
    g.add_node("retriever_node", partial(retriever_node, session=session, pipeline=pipeline))
    g.add_node("scorer_node", partial(scorer_node, llm=llm))
    g.add_node("writer_node", partial(writer_node, llm=llm))
    g.add_node("critic_node", partial(critic_node, llm=llm))

    g.add_edge(START, "planner_node")
    g.add_edge("planner_node", "retriever_node")
    g.add_edge("retriever_node", "scorer_node")
    g.add_edge("scorer_node", "writer_node")
    g.add_edge("writer_node", "critic_node")
    # Conditional edge: if the Critic rejected the first attempt, loop
    # back to the Writer with the findings as feedback. Cap at 1 retry
    # (writer_attempts >= 2) to avoid runaway loops on adversarial inputs.
    g.add_conditional_edges(
        "critic_node",
        _should_retry_writer,
        {"retry": "writer_node", "end": END},
    )

    return g.compile()


def _should_retry_writer(state) -> str:  # type: ignore[no-untyped-def]
    """Route after Critic. Returns "retry" (loop back to writer_node) or
    "end" (terminate the graph). One retry max.
    """
    critic = state.get("critic")
    if not critic or critic.overall_pass:
        return "end"
    attempts = state.get("writer_attempts", 1)
    if attempts >= 2:
        return "end"
    return "retry"

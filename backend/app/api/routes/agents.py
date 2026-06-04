"""POST /agents/recommend — the Phase 6 multi-agent entrypoint.

Construction:
  - Reuse the request-scoped DB session (SessionDep).
  - Reuse app.state's embedder + reranker (built lazily by /grants/search
    helpers — agents shouldn't pay the cold-start tax twice).
  - Reuse a process-level GeminiAgentClient on app.state, lazy-built.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Path, Request, Response
from sse_starlette.sse import EventSourceResponse

from app.agents.graph import build_graph
from app.agents.llm import AgentLLMError, GeminiAgentClient
from app.agents.critic import critic_node
from app.agents.planner import planner_node
from app.agents.retriever import retriever_node
from app.agents.scorer import scorer_node
from app.agents.state import AgentState, CriticOutput, ScorerOutput, WriterOutput
from app.agents.writer import (
    build_writer_prompt,
    deterministic_writer_fallback,
    enforce_groundedness,
)
from app.core.auth import AuthenticatedUser
from app.api.deps import OptionalUserDep, SessionDep
from app.core.logging import get_logger
from app.models.session import AgentSession
from app.rag.pipeline import RetrievalPipeline
from app.schemas.agents import (
    AgentConversationEntry,
    AgentRecommendRequest,
    AgentRecommendResponse,
    AgentSessionResponse,
    AgentTrace,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])

# Anonymous sessions carry this fixed owner_user_id prefix so we can
# migrate them to a real Clerk user ID on first sign-in via a single
# UPDATE WHERE owner_user_id LIKE 'anon-...'. With Clerk auth wired in,
# every authenticated request transparently migrates its session over
# (see `_resolve_owner_for_session`).
_ANON_PREFIX = "anon"


def _anon_owner_id(session_id: uuid.UUID) -> str:
    return f"{_ANON_PREFIX}-{session_id}"


def _is_anon_owner(owner_user_id: str) -> bool:
    return owner_user_id.startswith(f"{_ANON_PREFIX}-")


def _user_id_for_writes(
    user: AuthenticatedUser | None,
    session_id: uuid.UUID,
) -> str:
    """Pick the owner_user_id to use when creating a fresh AgentSession row.

    Signed-in users get their real Clerk user id; anonymous users get a
    per-session sentinel so we can migrate it later.
    """
    return user.id if user is not None else _anon_owner_id(session_id)


@router.post(
    "/recommend",
    response_model=AgentRecommendResponse,
    summary="Free-text query → planner → retriever → writer → grant recommendations",
)
async def recommend(
    request: Request,
    session: SessionDep,
    body: AgentRecommendRequest,
    user: OptionalUserDep,
) -> AgentRecommendResponse:
    started = time.perf_counter()

    embedder = _resolve_embedder(request)
    reranker = _resolve_reranker(request)
    llm = _resolve_agent_llm(request)
    pipeline = RetrievalPipeline(embedder=embedder, reranker=reranker)

    # Resolve session FIRST so a graph failure doesn't strand a half-built
    # conversation. New sessions get a fresh UUID; existing ones round-trip
    # the caller's id.
    session_id = body.session_id or uuid.uuid4()

    graph = build_graph(session=session, pipeline=pipeline, llm=llm)
    initial: dict[str, object] = {"query": body.query}
    if body.startup_profile is not None:
        initial["startup_profile"] = body.startup_profile.model_dump(mode="json")
    state: AgentState = await graph.ainvoke(initial)  # type: ignore[assignment]

    planner = state["planner"]
    writer = state["writer"]
    candidates = state.get("candidates", [])

    total_ms = int((time.perf_counter() - started) * 1000)
    critic_for_trace = state.get("critic") or CriticOutput(overall_pass=True, summary="", findings=[])
    trace = AgentTrace(
        rewritten_query=planner.rewritten_query,
        extracted_facts=planner.facts.model_dump(mode="json"),
        planner_ms=state.get("planner_ms", 0),
        retrieval_ms=state.get("retrieval_ms", 0),
        scorer_ms=state.get("scorer_ms", 0),
        writer_ms=state.get("writer_ms", 0),
        critic_ms=state.get("critic_ms", 0),
        total_ms=total_ms,
        candidate_count=len(candidates),
        planner_rationale=planner.rationale,
        scores=list(state.get("scorer", ScorerOutput(scores=[])).scores) if state.get("scorer") else [],
        critic_pass=critic_for_trace.overall_pass,
        critic_summary=critic_for_trace.summary,
        critic_findings=list(critic_for_trace.findings),
        writer_attempts=state.get("writer_attempts", 1),
    )

    entry = AgentConversationEntry(
        ts=datetime.now(UTC).isoformat(),
        query=body.query,
        summary=writer.summary,
        recommendations=writer.recommendations,
        questions_for_user=writer.questions_for_user,
        trace=trace,
    )
    await _append_history(session, session_id=session_id, entry=entry, user=user)

    logger.info(
        "agents.recommend.done",
        session_id=str(session_id),
        total_ms=total_ms,
        recommendation_count=len(writer.recommendations),
        rewritten_query=planner.rewritten_query[:120],
    )
    return AgentRecommendResponse(
        session_id=session_id,
        summary=writer.summary,
        recommendations=writer.recommendations,
        questions_for_user=writer.questions_for_user,
        trace=trace,
    )


@router.post(
    "/recommend/stream",
    summary="Streaming variant of /recommend — SSE stage events + final 'done' payload.",
)
async def recommend_stream(
    request: Request,
    session: SessionDep,
    body: AgentRecommendRequest,
    user: OptionalUserDep,
) -> EventSourceResponse:
    """SSE endpoint for incremental UX. Same agent graph as POST /recommend,
    but emits a `stage` event each time a node starts/finishes so the
    frontend can show real progress instead of a fake cycling loader.

    Event stream:
        event: stage  data: {stage, status:"start", elapsed_ms}
        event: stage  data: {stage, status:"done",  elapsed_ms, ...stage_payload}
        ...
        event: done   data: AgentRecommendResponse  (same shape as batch endpoint)

    On any failure mid-stream:
        event: error  data: {message}
    """
    embedder = _resolve_embedder(request)
    reranker = _resolve_reranker(request)
    llm = _resolve_agent_llm(request)
    pipeline = RetrievalPipeline(embedder=embedder, reranker=reranker)
    session_id = body.session_id or uuid.uuid4()

    return EventSourceResponse(
        _stream_recommend(
            db=session,
            llm=llm,
            pipeline=pipeline,
            query=body.query,
            session_id=session_id,
            startup_profile=(
                body.startup_profile.model_dump(mode="json")
                if body.startup_profile is not None
                else None
            ),
            user=user,
        ),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=AgentSessionResponse,
    summary="Fetch a chat session's full history for replay.",
)
async def get_session_history(
    db: SessionDep,
    session_id: Annotated[uuid.UUID, Path()],
    user: OptionalUserDep,
) -> AgentSessionResponse:
    row = await db.get(AgentSession, session_id)
    if row is None or not row.is_active:
        # Don't 404 — empty history is a legitimate "fresh session" state
        # from the client's perspective. Returning 200 keeps the FE happy.
        return AgentSessionResponse(session_id=session_id, history=[], is_active=True)
    # Ownership check: if the row belongs to a real Clerk user, only
    # that user can read it. Anonymous rows (owner_user_id="anon-…")
    # stay readable by anyone who knows the session_id — same surface
    # the unauthenticated path always had.
    if not _is_anon_owner(row.owner_user_id):
        if user is None or row.owner_user_id != user.id:
            return AgentSessionResponse(session_id=session_id, history=[], is_active=True)
    return AgentSessionResponse(
        session_id=session_id,
        history=[AgentConversationEntry.model_validate(e) for e in row.conversation_history],
        is_active=row.is_active,
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Soft-delete a session (clears chat history).",
)
async def delete_session(
    db: SessionDep,
    session_id: Annotated[uuid.UUID, Path()],
    user: OptionalUserDep,
) -> Response:
    row = await db.get(AgentSession, session_id)
    if row is not None:
        # Owner check: same rule as the GET handler. Anonymous rows are
        # accessible to anyone who knows the id; user-owned rows only to
        # the matching authenticated user.
        if not _is_anon_owner(row.owner_user_id):
            if user is None or row.owner_user_id != user.id:
                # Don't reveal the row's existence — same 204 path.
                return Response(status_code=204)
        row.is_active = False
        row.conversation_history = []
        await db.commit()
    # Idempotent: deleting a non-existent session is a no-op.
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Streaming generator — runs nodes manually so we can emit a per-stage event
# between them. LangGraph's astream() works too but yields opaque state
# deltas; the manual sequence is clearer and three nodes is small enough
# that duplicating the orchestration here doesn't drift from graph.py.
# ---------------------------------------------------------------------------
def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """sse-starlette dict format."""
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False, default=str),
    }


async def _stream_recommend(
    *,
    db,  # type: ignore[no-untyped-def]
    llm: GeminiAgentClient,
    pipeline: RetrievalPipeline,
    query: str,
    session_id: uuid.UUID,
    startup_profile: dict[str, Any] | None = None,
    user: AuthenticatedUser | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    started = time.perf_counter()

    def now_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    state: AgentState = {"query": query}
    if startup_profile:
        state["startup_profile"] = startup_profile
    try:
        # ---- Planner -----------------------------------------------------
        yield _sse("stage", {"stage": "planner", "status": "start", "elapsed_ms": now_ms()})
        state.update(await planner_node(state, llm=llm))
        planner = state["planner"]
        yield _sse("stage", {
            "stage": "planner",
            "status": "done",
            "elapsed_ms": now_ms(),
            "planner_ms": state.get("planner_ms", 0),
            "rewritten_query": planner.rewritten_query,
            "extracted_facts": planner.facts.model_dump(mode="json"),
            "rationale": planner.rationale,
        })

        # ---- Retriever ---------------------------------------------------
        yield _sse("stage", {"stage": "retriever", "status": "start", "elapsed_ms": now_ms()})
        state.update(await retriever_node(state, session=db, pipeline=pipeline))
        candidates = state.get("candidates", [])
        yield _sse("stage", {
            "stage": "retriever",
            "status": "done",
            "elapsed_ms": now_ms(),
            "retrieval_ms": state.get("retrieval_ms", 0),
            "candidate_count": len(candidates),
        })

        # ---- Scorer ------------------------------------------------------
        yield _sse("stage", {"stage": "scorer", "status": "start", "elapsed_ms": now_ms()})
        await asyncio.sleep(0)
        state.update(await scorer_node(state, llm=llm))
        scorer = state.get("scorer") or ScorerOutput(scores=[])
        yield _sse("stage", {
            "stage": "scorer",
            "status": "done",
            "elapsed_ms": now_ms(),
            "scorer_ms": state.get("scorer_ms", 0),
            "score_count": len(scorer.scores),
            "fit_labels": [s.fit_label for s in scorer.scores],
        })

        # ---- Writer + Critic loop ---------------------------------------
        # Run Writer → Critic once. If Critic rejects (overall_pass=False),
        # run them BOTH again with the Critic's findings forwarded as
        # feedback to the Writer. Cap at 1 retry.
        writer: WriterOutput | None = None
        critic: CriticOutput | None = None
        for attempt in (1, 2):
            is_retry = attempt > 1
            # ---- Writer (streaming) -------------------------------------
            yield _sse("stage", {
                "stage": "writer",
                "status": "start",
                "elapsed_ms": now_ms(),
                "attempt": attempt,
                "retry": is_retry,
            })
            # Surrender control once so the start event reaches the client
            # before the long Writer call begins.
            await asyncio.sleep(0)

            writer_started = time.perf_counter()
            if not candidates:
                # No candidates — skip the LLM call and use the
                # deterministic empty-state response.
                writer = WriterOutput(
                    summary=(
                        "No grants matched your question with strong confidence. "
                        "Try a more specific query, or broaden the geography / sector."
                    ),
                    recommendations=[],
                    questions_for_user=[],
                )
            else:
                feedback = (
                    critic.findings if (is_retry and critic and not critic.overall_pass)
                    else None
                )
                prompt = build_writer_prompt(
                    query=query,
                    planner=planner,
                    candidates=candidates,
                    scorer=scorer,
                    critic_feedback=feedback,
                )
                accumulated = ""
                try:
                    async for chunk in llm.stream_text(
                        prompt=prompt,
                        temperature=0.3,
                        max_output_tokens=8192,
                    ):
                        accumulated += chunk
                        yield _sse("writer_delta", {"text": chunk, "attempt": attempt})

                    writer = WriterOutput.model_validate_json(accumulated)
                    writer = enforce_groundedness(writer, candidates)
                except (AgentLLMError, ValueError) as e:
                    logger.warning(
                        "agents.writer.stream.fallback",
                        error_type=type(e).__name__,
                        error=str(e)[:200],
                        attempt=attempt,
                    )
                    writer = deterministic_writer_fallback(candidates, error=str(e))

            writer_ms = int((time.perf_counter() - writer_started) * 1000)
            # Accumulate across retries — total Writer wall-clock matters
            # for the trace, not just the last attempt's slice.
            state["writer"] = writer
            state["writer_ms"] = state.get("writer_ms", 0) + writer_ms
            state["writer_attempts"] = attempt

            yield _sse("stage", {
                "stage": "writer",
                "status": "done",
                "elapsed_ms": now_ms(),
                "writer_ms": writer_ms,
                "attempt": attempt,
                "recommendation_count": len(writer.recommendations),
            })

            # ---- Critic ---------------------------------------------------
            yield _sse("stage", {
                "stage": "critic",
                "status": "start",
                "elapsed_ms": now_ms(),
                "attempt": attempt,
            })
            await asyncio.sleep(0)
            critic_started = time.perf_counter()
            critic_result = await critic_node(state, llm=llm)
            state.update(critic_result)
            critic = state.get("critic") or CriticOutput(
                overall_pass=True, summary="", findings=[],
            )
            critic_ms_this = int((time.perf_counter() - critic_started) * 1000)
            state["critic_ms"] = state.get("critic_ms", 0) + critic_ms_this

            yield _sse("stage", {
                "stage": "critic",
                "status": "done",
                "elapsed_ms": now_ms(),
                "critic_ms": critic_ms_this,
                "attempt": attempt,
                "overall_pass": critic.overall_pass,
                "finding_count": len(critic.findings),
            })

            # Loop continuation: bail on pass, or after one retry.
            if critic.overall_pass or attempt >= 2:
                break

        assert writer is not None  # the loop always assigns
        assert critic is not None

        # ---- Finalise + persist -----------------------------------------
        total_ms = now_ms()
        trace = AgentTrace(
            rewritten_query=planner.rewritten_query,
            extracted_facts=planner.facts.model_dump(mode="json"),
            planner_ms=state.get("planner_ms", 0),
            retrieval_ms=state.get("retrieval_ms", 0),
            writer_ms=state.get("writer_ms", 0),
            total_ms=total_ms,
            candidate_count=len(candidates),
            planner_rationale=planner.rationale,
        )
        entry = AgentConversationEntry(
            ts=datetime.now(UTC).isoformat(),
            query=query,
            summary=writer.summary,
            recommendations=writer.recommendations,
            questions_for_user=writer.questions_for_user,
            trace=trace,
        )
        await _append_history(db, session_id=session_id, entry=entry, user=user)

        response = AgentRecommendResponse(
            session_id=session_id,
            summary=writer.summary,
            recommendations=writer.recommendations,
            questions_for_user=writer.questions_for_user,
            trace=trace,
        )
        logger.info(
            "agents.recommend.stream.done",
            session_id=str(session_id),
            total_ms=total_ms,
        )
        yield _sse("done", response.model_dump(mode="json"))
    except AgentLLMError as e:
        logger.warning("agents.recommend.stream.llm_error", error=str(e)[:200])
        yield _sse("error", {"message": str(e)[:500]})
    except Exception as e:  # noqa: BLE001 — pass any other failure through SSE
        logger.exception("agents.recommend.stream.failed")
        yield _sse("error", {"message": f"{type(e).__name__}: {str(e)[:300]}"})


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
async def _append_history(
    db,  # type: ignore[no-untyped-def]
    *,
    session_id: uuid.UUID,
    entry: AgentConversationEntry,
    user: AuthenticatedUser | None = None,
) -> None:
    """Upsert: create the session row on first turn, append on subsequent turns.

    `conversation_history` is JSONB so we just round-trip the list. For a
    high-traffic future we'd want a `messages` child table with FK, but
    that's premature optimisation at single-digit-turns-per-session scale.

    Ownership migration: if the row was created anonymously and we now
    have a real authenticated user, transfer ownership in-place. That's
    how a chat the user started signed-out gets attached to their Clerk
    identity the moment they sign in.
    """
    row = await db.get(AgentSession, session_id)
    serialised = entry.model_dump(mode="json")
    if row is None:
        row = AgentSession(
            id=session_id,
            owner_user_id=_user_id_for_writes(user, session_id),
            title=entry.query[:80],
            is_active=True,
            conversation_history=[serialised],
        )
        db.add(row)
    else:
        # Migrate anon → real user on first turn after sign-in.
        if user is not None and _is_anon_owner(row.owner_user_id):
            logger.info(
                "agents.session.migrated",
                session_id=str(session_id),
                from_owner=row.owner_user_id,
                to_user=user.id,
            )
            row.owner_user_id = user.id
        # SQLAlchemy doesn't detect in-place list mutation on JSONB, so
        # rebind the attribute to a new list to force a change.
        row.conversation_history = [*row.conversation_history, serialised]
        if not row.title:
            row.title = entry.query[:80]
        row.is_active = True
    await db.commit()


# ---------------------------------------------------------------------------
# Resolvers — keep agents and /grants/search using the same shared objects.
# Each resolver mirrors the conventions in api/routes/grants.py.
# ---------------------------------------------------------------------------
def _resolve_embedder(request: Request):  # type: ignore[no-untyped-def]
    existing = getattr(request.app.state, "scheduler_embedder", None)
    if existing is not None:
        return existing
    from app.services.embedding import EmbeddingService

    embedder = EmbeddingService(redis=getattr(request.app.state, "redis", None))
    request.app.state.scheduler_embedder = embedder
    return embedder


def _resolve_reranker(request: Request):  # type: ignore[no-untyped-def]
    existing = getattr(request.app.state, "reranker", None)
    if existing is not None:
        return existing
    from app.rag.reranker import RerankerService

    reranker = RerankerService()
    request.app.state.reranker = reranker
    return reranker


def _resolve_agent_llm(request: Request) -> GeminiAgentClient:
    existing = getattr(request.app.state, "agent_llm", None)
    if existing is not None:
        return existing
    llm = GeminiAgentClient()
    # Enter the async context once; we'll close it in main.py's lifespan.
    # __aenter__ here is fine because GeminiAgentClient.__aenter__ doesn't
    # need to await anything beyond constructing httpx.AsyncClient.
    request.app.state.agent_llm = llm
    return llm

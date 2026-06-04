"""Integration test for Phase 6 — POST /agents/recommend.

Stubs the Gemini agent client so the suite stays fast + offline. Uses the
same SAVEPOINT-bound session + real Redis as the rest of the integration
suite so the retrieval leg is exercised end-to-end (e5 embeddings, dense
+ sparse + RRF — reranker stubbed because BGE is 2.3GB).
"""

from __future__ import annotations

import json
import math
import uuid
from decimal import Decimal
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import (
    CandidateScore,
    CriticFinding,
    CriticOutput,
    PlannerFacts,
    PlannerOutput,
    ScorerOutput,
    WriterOutput,
)
from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus


# ---------------------------------------------------------------------------
# Stubs — installed on app.state.* before the test runs the endpoint.
# ---------------------------------------------------------------------------
def _vec(seed: int) -> list[float]:
    base = (seed % 7) + 1
    raw = [float(base + i % 3) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _StubEmbedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = 2
            for c in t:
                if c.isdigit():
                    seed = int(c)
                    break
            out.append(_vec(seed))
        return out


class _StubReranker:
    async def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        return [10.0 if query.lower() in p.lower() else 1.0 for p in passages]


class _StubAgentLLM:
    """Returns canned PlannerOutput / WriterOutput per call.

    Tracks invocations so tests can assert each node ran exactly once.
    """

    def __init__(
        self,
        *,
        planner_out: PlannerOutput,
        writer_out: WriterOutput,
        scorer_out: ScorerOutput | None = None,
        critic_out: CriticOutput | None = None,
    ) -> None:
        self._planner_out = planner_out
        self._writer_out = writer_out
        # Default to empty scores so existing tests that don't care about
        # the Scorer don't have to set one up — Writer's prompt tells the
        # model to fall back when scores are empty.
        self._scorer_out = scorer_out if scorer_out is not None else ScorerOutput(scores=[])
        # Default Critic verdict: pass with no findings. Tests that care
        # about the Critic supply their own.
        self._critic_out = critic_out if critic_out is not None else CriticOutput(
            overall_pass=True, summary="Stubbed critic — pass.", findings=[],
        )
        self.planner_calls = 0
        self.scorer_calls = 0
        self.writer_calls = 0
        self.critic_calls = 0

    async def respond_as(self, model_cls: type, **_kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        if model_cls is PlannerOutput:
            self.planner_calls += 1
            return self._planner_out
        if model_cls is ScorerOutput:
            self.scorer_calls += 1
            return self._scorer_out
        if model_cls is WriterOutput:
            self.writer_calls += 1
            return self._writer_out
        if model_cls is CriticOutput:
            self.critic_calls += 1
            return self._critic_out
        raise AssertionError(f"Unexpected model_cls: {model_cls}")

    async def stream_text(self, **_kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        """Stub the streaming path by chunking the writer's canned JSON.

        Yields fixed-size slices so the test can assert that writer_delta
        events arrive and that the final assembled buffer parses into a
        WriterOutput identically to the batch path.
        """
        self.writer_calls += 1
        payload = self._writer_out.model_dump_json()
        chunk_size = 24
        for i in range(0, len(payload), chunk_size):
            yield payload[i : i + chunk_size]

    async def __aexit__(self, *_: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------
async def _seed_grants(session: AsyncSession) -> list[Grant]:
    grants: list[Grant] = []
    for i in range(1, 4):
        g = Grant(
            portal=GrantPortal.EXIST,
            title=f"AgentTest grant {i} — founder stipend",
            summary=f"Summary {i} about academic founder stipend",
            body=f"Body {i} explaining eligibility and process.",
            status=GrantStatus.OPEN,
            country="DE",
            funding_max_eur=Decimal("100000") * i,
            source_url=f"https://agent-test.example/{i}",
            source_doc_id=f"agent-test-{i}",
            source_hash=f"hash-agent-{i}",
            embedding=_vec(i),
        )
        session.add(g)
        grants.append(g)
    await session.flush()
    for g in grants:
        await session.refresh(g)
    await session.commit()
    return grants


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_recommend_happy_path(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grants = await _seed_grants(db_session)

    planner_out = PlannerOutput(
        rewritten_query="founder stipend academic spinoff",
        facts=PlannerFacts(country="DE"),
        rationale="Founder asked about academic spinoffs in Germany.",
    )
    writer_out = WriterOutput(
        summary="Two EXIST programmes fit best.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Direct match for academic founder stipend.",
                "caveats": ["Verify enrolment status."],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=["What stage is your prototype at?"],
    )
    stub_llm = _StubAgentLLM(planner_out=planner_out, writer_out=writer_out)

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post(
        "/agents/recommend",
        json={"query": "Stipendium für meine akademische Ausgründung"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Shape
    assert "summary" in body
    assert "recommendations" in body
    assert "questions_for_user" in body
    assert "trace" in body

    # Recommendations grounded to seeded grants
    assert len(body["recommendations"]) == 1
    rec = body["recommendations"][0]
    assert rec["grant_id"] == str(grants[0].id)
    assert rec["fit"] == "high"

    # Trace stage timings populated
    trace = body["trace"]
    assert trace["rewritten_query"] == "founder stipend academic spinoff"
    assert trace["extracted_facts"]["country"] == "DE"
    assert trace["candidate_count"] >= 1
    assert trace["planner_ms"] >= 0
    assert trace["retrieval_ms"] >= 0
    assert trace["writer_ms"] >= 0
    assert trace["total_ms"] >= 0

    # Each LLM-driven node fired exactly once
    assert stub_llm.planner_calls == 1
    assert stub_llm.writer_calls == 1


@pytest.mark.integration
async def test_recommend_surfaces_scorer_judgement(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """The Scorer's per-candidate judgement should land on the response
    trace, and the Scorer should be called exactly once.
    """
    grants = await _seed_grants(db_session)

    planner_out = PlannerOutput(
        rewritten_query="founder stipend",
        facts=PlannerFacts(country="DE"),
        rationale="DE founder stipend.",
    )
    scorer_out = ScorerOutput(scores=[
        CandidateScore(
            grant_id=grants[0].id,
            eligibility_score=92,
            fit_label="high",
            strengths=["Direct EXIST-flavoured match."],
            concerns=["Verify academic affiliation."],
            missing_info=["Is the founder a current student?"],
        ),
    ])
    writer_out = WriterOutput(
        summary="One strong fit.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Cited the Scorer's strength.",
                "caveats": ["Verify academic affiliation."],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    stub_llm = _StubAgentLLM(
        planner_out=planner_out, writer_out=writer_out, scorer_out=scorer_out,
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post("/agents/recommend", json={"query": "Stipendium"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert stub_llm.scorer_calls == 1
    trace = body["trace"]
    assert trace["scorer_ms"] >= 0
    assert len(trace["scores"]) == 1
    score = trace["scores"][0]
    assert score["grant_id"] == str(grants[0].id)
    assert score["eligibility_score"] == 92
    assert score["fit_label"] == "high"
    assert "Direct EXIST-flavoured match." in score["strengths"]


@pytest.mark.integration
async def test_recommend_forwards_startup_profile_to_planner(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """`startup_profile` in the request body should land in AgentState
    so the Planner prompt can use it. We assert by reading state via the
    stub: the Planner is the only LLM consumer of profile today, so we
    don't add a separate planner call counter — we just confirm the
    request shape is accepted and the pipeline still produces a valid
    response.
    """
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants)

    r = await client.post(
        "/agents/recommend",
        json={
            "query": "Wir suchen Förderung.",
            "startup_profile": {
                "name": "Foo Robotics",
                "sector": "deeptech",
                "stage": "idea",
                "country": "DE",
                "federal_state": "Bayern",
                "funding_target_eur": 150000,
                "description": "Spinoff from TU Munich.",
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    assert body["trace"]["rewritten_query"]


@pytest.mark.integration
async def test_recommend_rejects_invalid_profile_country(
    client: AsyncClient,
) -> None:
    """Bad ISO code on the profile should 422, not silently strip."""
    r = await client.post(
        "/agents/recommend",
        json={
            "query": "needs funding",
            "startup_profile": {"country": "DEU"},  # 3 chars, min/max=2
        },
    )
    assert r.status_code == 422


@pytest.mark.integration
async def test_recommend_surfaces_critic_findings(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """The Critic's verdict + findings should land on the trace, and the
    Critic should be called exactly once.
    """
    grants = await _seed_grants(db_session)

    planner_out = PlannerOutput(
        rewritten_query="founder stipend",
        facts=PlannerFacts(country="DE"),
        rationale="",
    )
    writer_out = WriterOutput(
        summary="One rec.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Direct match.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    critic_out = CriticOutput(
        overall_pass=False,
        summary="Writer omitted a caveat the Scorer flagged.",
        findings=[
            CriticFinding(
                type="caveat_omission",
                severity="medium",
                grant_id=grants[0].id,
                message="Scorer noted: 'Verify academic affiliation', but Writer's caveats are empty.",
            ),
        ],
    )
    stub_llm = _StubAgentLLM(
        planner_out=planner_out, writer_out=writer_out, critic_out=critic_out,
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post("/agents/recommend", json={"query": "Stipendium"})
    assert r.status_code == 200, r.text
    body = r.json()

    # The stub returns overall_pass=False every time, so the retry loop
    # fires once (writer 2× / critic 2×) before bailing at the cap.
    assert stub_llm.critic_calls == 2
    assert stub_llm.writer_calls == 2
    trace = body["trace"]
    assert trace["critic_pass"] is False
    assert "omitted" in trace["critic_summary"]
    assert len(trace["critic_findings"]) == 1
    assert trace["critic_findings"][0]["type"] == "caveat_omission"
    assert trace["critic_findings"][0]["severity"] == "medium"
    assert trace["critic_findings"][0]["grant_id"] == str(grants[0].id)
    assert trace["critic_ms"] >= 0


@pytest.mark.integration
async def test_writer_retries_when_critic_rejects(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """When the Critic returns overall_pass=False on the first attempt,
    the graph re-runs the Writer with the findings as feedback. Trace
    `writer_attempts` ends at 2.
    """
    grants = await _seed_grants(db_session)

    planner_out = PlannerOutput(
        rewritten_query="founder stipend",
        facts=PlannerFacts(country="DE"),
        rationale="",
    )
    writer_out = WriterOutput(
        summary="One rec.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Direct match.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    critic_out = CriticOutput(
        overall_pass=False,
        summary="Writer omitted a Scorer-flagged caveat.",
        findings=[
            CriticFinding(
                type="caveat_omission",
                severity="medium",
                grant_id=grants[0].id,
                message="Add the academic-affiliation caveat.",
            ),
        ],
    )
    stub_llm = _StubAgentLLM(
        planner_out=planner_out, writer_out=writer_out, critic_out=critic_out,
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post("/agents/recommend", json={"query": "Stipendium"})
    assert r.status_code == 200, r.text
    body = r.json()

    # Writer ran twice; Critic also ran twice (verdict still flagged on the
    # second attempt because the stub returns the same critic_out each time).
    assert stub_llm.writer_calls == 2
    assert stub_llm.critic_calls == 2
    assert body["trace"]["writer_attempts"] == 2


@pytest.mark.integration
async def test_writer_caps_at_one_retry(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Even when the Critic keeps flagging, the Writer must not loop
    forever — the conditional edge bails after writer_attempts >= 2.
    """
    grants = await _seed_grants(db_session)

    planner_out = PlannerOutput(
        rewritten_query="founder stipend",
        facts=PlannerFacts(country="DE"),
        rationale="",
    )
    writer_out = WriterOutput(
        summary="Same response every time.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Match.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    critic_out = CriticOutput(
        overall_pass=False,
        summary="Always reject.",
        findings=[
            CriticFinding(
                type="other",
                severity="medium",
                grant_id=None,
                message="Reject this.",
            ),
        ],
    )
    stub_llm = _StubAgentLLM(
        planner_out=planner_out, writer_out=writer_out, critic_out=critic_out,
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post("/agents/recommend", json={"query": "Stipendium"})
    assert r.status_code == 200
    assert stub_llm.writer_calls == 2  # capped at 2, not infinite
    assert r.json()["trace"]["writer_attempts"] == 2


@pytest.mark.integration
async def test_critic_drops_hallucinated_finding_grant_ids(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Findings referencing a grant_id that's not in the candidate set
    should be dropped at the node boundary."""
    grants = await _seed_grants(db_session)
    fake_id = uuid.uuid4()

    planner_out = PlannerOutput(
        rewritten_query="founder stipend", facts=PlannerFacts(country="DE"), rationale="",
    )
    writer_out = WriterOutput(
        summary="One rec.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Match.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    critic_out = CriticOutput(
        overall_pass=False,
        summary="Mixed real + hallucinated finding.",
        findings=[
            CriticFinding(
                type="citation_faithfulness",
                severity="high",
                grant_id=grants[0].id,
                message="Real finding on a real grant.",
            ),
            CriticFinding(
                type="citation_faithfulness",
                severity="high",
                grant_id=fake_id,
                message="Finding on a fabricated grant_id.",
            ),
            CriticFinding(
                type="other",
                severity="low",
                grant_id=None,
                message="General finding stays.",
            ),
        ],
    )
    stub_llm = _StubAgentLLM(
        planner_out=planner_out, writer_out=writer_out, critic_out=critic_out,
    )
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post("/agents/recommend", json={"query": "founder stipend"})
    assert r.status_code == 200, r.text
    body = r.json()
    finding_ids = [f["grant_id"] for f in body["trace"]["critic_findings"]]
    assert str(grants[0].id) in finding_ids
    assert str(fake_id) not in finding_ids
    assert None in finding_ids  # the general finding


@pytest.mark.integration
async def test_recommend_drops_hallucinated_grant_ids(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Writer groundedness guard: a rec referencing a UUID not in the
    candidate set should be silently dropped at the node boundary."""
    grants = await _seed_grants(db_session)
    fake_id = uuid.uuid4()

    planner_out = PlannerOutput(
        rewritten_query="founder stipend",
        facts=PlannerFacts(country="DE"),
        rationale="",
    )
    writer_out = WriterOutput(
        summary="Mixed real + hallucinated.",
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Real grant.",
                "caveats": [],
            },
            {
                "grant_id": str(fake_id),
                "grant_title": "Fabricated grant",
                "portal": "exist",
                "source_url": "https://fake.example/",
                "fit": "high",
                "rationale": "This grant doesn't exist.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    stub_llm = _StubAgentLLM(planner_out=planner_out, writer_out=writer_out)

    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm

    r = await client.post(
        "/agents/recommend",
        json={"query": "Stipendium"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    rec_ids = {rec["grant_id"] for rec in body["recommendations"]}
    assert str(grants[0].id) in rec_ids
    assert str(fake_id) not in rec_ids


@pytest.mark.integration
async def test_recommend_rejects_short_query(client: AsyncClient) -> None:
    r = await client.post("/agents/recommend", json={"query": "x"})
    assert r.status_code == 422  # Pydantic min_length=3


# ---------------------------------------------------------------------------
# Session persistence — POST creates, GET fetches, DELETE clears.
# ---------------------------------------------------------------------------
async def _install_stubs(
    client: AsyncClient,
    grants: list,  # type: ignore[type-arg]
    summary: str = "Two EXIST programmes fit best.",
) -> _StubAgentLLM:
    planner_out = PlannerOutput(
        rewritten_query="founder stipend academic spinoff",
        facts=PlannerFacts(country="DE"),
        rationale="DE founder stipend extracted.",
    )
    writer_out = WriterOutput(
        summary=summary,
        recommendations=[
            {
                "grant_id": str(grants[0].id),
                "grant_title": grants[0].title,
                "portal": grants[0].portal.value,
                "source_url": grants[0].source_url,
                "fit": "high",
                "rationale": "Direct match.",
                "caveats": [],
            },
        ],  # type: ignore[arg-type]
        questions_for_user=[],
    )
    stub_llm = _StubAgentLLM(planner_out=planner_out, writer_out=writer_out)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = stub_llm
    return stub_llm


@pytest.mark.integration
async def test_recommend_creates_session_when_id_omitted(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants)

    r = await client.post("/agents/recommend", json={"query": "founder stipend"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    session_id = body["session_id"]
    # Valid UUID and persisted (GET should return the entry we just wrote)
    fetched = await client.get(f"/agents/sessions/{session_id}")
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["session_id"] == session_id
    assert payload["is_active"] is True
    assert len(payload["history"]) == 1
    assert payload["history"][0]["query"] == "founder stipend"
    assert payload["history"][0]["summary"]


@pytest.mark.integration
async def test_recommend_appends_to_existing_session(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants)

    r1 = await client.post(
        "/agents/recommend",
        json={"query": "first turn"},
    )
    session_id = r1.json()["session_id"]

    # Reuse stubs (call counter resets when test starts but the stub is
    # idempotent — same canned response each call).
    r2 = await client.post(
        "/agents/recommend",
        json={"query": "second turn", "session_id": session_id},
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id

    fetched = await client.get(f"/agents/sessions/{session_id}")
    history = fetched.json()["history"]
    assert len(history) == 2
    assert history[0]["query"] == "first turn"
    assert history[1]["query"] == "second turn"


@pytest.mark.integration
async def test_get_unknown_session_returns_empty_history(
    client: AsyncClient,
) -> None:
    fake_id = uuid.uuid4()
    r = await client.get(f"/agents/sessions/{fake_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == str(fake_id)
    assert body["history"] == []


@pytest.mark.integration
async def test_delete_session_clears_history_and_deactivates(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants)

    r1 = await client.post("/agents/recommend", json={"query": "to be cleared"})
    session_id = r1.json()["session_id"]

    delete_r = await client.delete(f"/agents/sessions/{session_id}")
    assert delete_r.status_code == 204

    fetched = await client.get(f"/agents/sessions/{session_id}")
    payload = fetched.json()
    # Soft-deleted sessions report empty + still-active so the FE can
    # treat them as a fresh chat without surfacing the deletion.
    assert payload["history"] == []
    assert payload["is_active"] is True


@pytest.mark.integration
async def test_delete_unknown_session_is_idempotent(client: AsyncClient) -> None:
    fake_id = uuid.uuid4()
    r = await client.delete(f"/agents/sessions/{fake_id}")
    assert r.status_code == 204


# ---------------------------------------------------------------------------
# Streaming endpoint — SSE stage events + final `done` payload.
# ---------------------------------------------------------------------------
def _parse_sse_events(raw: str) -> list[dict[str, str]]:
    """Tiny SSE parser — splits on blank lines, extracts event + data.

    sse-starlette emits CRLF per the SSE spec; normalise so we can split
    on `\\n\\n` regardless of the producer.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    events: list[dict[str, str]] = []
    for chunk in raw.split("\n\n"):
        if not chunk.strip():
            continue
        event = "message"
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            events.append({"event": event, "data": "\n".join(data_lines)})
    return events


@pytest.mark.integration
async def test_stream_emits_stage_events_then_done(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants, summary="Streamed summary.")

    # httpx's AsyncClient buffers SSE responses but still exposes the full
    # body; for an integration test that's all we need. Real browsers see
    # tokens as they arrive thanks to the EventSourceResponse.
    r = await client.post(
        "/agents/recommend/stream",
        json={"query": "founder stipend"},
    )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers.get("content-type", "")

    events = _parse_sse_events(r.text)
    event_names = [e["event"] for e in events]

    # All five stages emit start + done; one final `done` carries the response.
    assert event_names.count("stage") == 10  # 5 starts + 5 dones
    assert event_names.count("done") == 1
    assert "error" not in event_names

    # The stages must arrive in the right order.
    stage_payloads = [json.loads(e["data"]) for e in events if e["event"] == "stage"]
    order = [(p["stage"], p["status"]) for p in stage_payloads]
    assert order == [
        ("planner", "start"),
        ("planner", "done"),
        ("retriever", "start"),
        ("retriever", "done"),
        ("scorer", "start"),
        ("scorer", "done"),
        ("writer", "start"),
        ("writer", "done"),
        ("critic", "start"),
        ("critic", "done"),
    ]

    # Final `done` carries the same response shape as the batch endpoint.
    done_payload = json.loads(next(e["data"] for e in events if e["event"] == "done"))
    assert done_payload["summary"] == "Streamed summary."
    assert "session_id" in done_payload
    assert len(done_payload["recommendations"]) == 1
    assert done_payload["recommendations"][0]["grant_id"] == str(grants[0].id)
    # Trace stage timings echoed on the response.
    assert done_payload["trace"]["candidate_count"] >= 1
    assert done_payload["trace"]["rewritten_query"]


@pytest.mark.integration
async def test_stream_emits_writer_delta_events(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Streaming path replaces the batch Writer with chunked Gemini output;
    the server should emit `writer_delta` events for each chunk and the
    final `done` event should still carry a fully-validated WriterOutput.
    """
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants, summary="Streamed Writer summary.")

    r = await client.post(
        "/agents/recommend/stream",
        json={"query": "founder stipend"},
    )
    assert r.status_code == 200
    events = _parse_sse_events(r.text)

    delta_events = [json.loads(e["data"]) for e in events if e["event"] == "writer_delta"]
    assert len(delta_events) >= 2  # canned writer JSON splits into multiple chunks
    # Re-assembling the chunks must reconstruct the writer's canned JSON.
    reassembled = "".join(d["text"] for d in delta_events)
    parsed = json.loads(reassembled)
    assert parsed["summary"] == "Streamed Writer summary."

    # Final `done` still ships a structured response with the same fields.
    done_payload = json.loads(next(e["data"] for e in events if e["event"] == "done"))
    assert done_payload["summary"] == "Streamed Writer summary."
    assert len(done_payload["recommendations"]) == 1


@pytest.mark.integration
async def test_stream_persists_to_session_history(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Streaming endpoint should write the conversation entry just like
    the batch endpoint — otherwise the next page load loses the chat."""
    grants = await _seed_grants(db_session)
    await _install_stubs(client, grants)

    r = await client.post(
        "/agents/recommend/stream",
        json={"query": "streamed query for persistence"},
    )
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    done = json.loads(next(e["data"] for e in events if e["event"] == "done"))
    session_id = done["session_id"]

    fetched = await client.get(f"/agents/sessions/{session_id}")
    assert fetched.status_code == 200
    history = fetched.json()["history"]
    assert len(history) == 1
    assert history[0]["query"] == "streamed query for persistence"

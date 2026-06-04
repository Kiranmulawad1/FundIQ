"""Auth + anon→user migration tests for the /agents/* surface.

The test environment leaves `CLERK_SECRET_KEY` unset, which makes the
`current_user` dependency return a synthetic dev user for ANY request
that sends an Authorization header — verification is bypassed for local
dev. To exercise the multi-user ownership checks we need to inject
distinct identities; we do that via FastAPI's `dependency_overrides`
mechanism, replacing `optional_user` per test.
"""

from __future__ import annotations

import math
import uuid
from decimal import Decimal
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import (
    CriticOutput,
    PlannerFacts,
    PlannerOutput,
    ScorerOutput,
    WriterOutput,
)
from app.core.auth import AuthenticatedUser, optional_user
from app.models import EMBEDDING_DIM, Grant
from app.models.base import GrantPortal, GrantStatus


# ---------------------------------------------------------------------------
# Local stubs (kept independent of test_agents_recommend.py to avoid the
# implicit coupling that comes with sharing _StubAgentLLM at module scope).
# ---------------------------------------------------------------------------
def _vec(seed: int) -> list[float]:
    base = (seed % 7) + 1
    raw = [float(base + i % 3) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _StubEmbedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [_vec(2) for _ in texts]


class _StubReranker:
    async def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        return [1.0 for _ in passages]


class _StubAgentLLM:
    async def respond_as(self, model_cls: type, **_kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        if model_cls is PlannerOutput:
            return PlannerOutput(
                rewritten_query="q",
                facts=PlannerFacts(country="DE"),
                rationale="",
            )
        if model_cls is ScorerOutput:
            return ScorerOutput(scores=[])
        if model_cls is WriterOutput:
            return WriterOutput(
                summary="ok", recommendations=[], questions_for_user=[],
            )
        if model_cls is CriticOutput:
            return CriticOutput(overall_pass=True, summary="", findings=[])
        raise AssertionError(f"Unexpected model_cls: {model_cls}")

    async def __aexit__(self, *_: object) -> None:
        return None


async def _seed_grant(session: AsyncSession) -> Grant:
    g = Grant(
        portal=GrantPortal.EXIST,
        title="AuthTest grant",
        summary="Summary",
        body="Body",
        status=GrantStatus.OPEN,
        country="DE",
        funding_max_eur=Decimal("100000"),
        source_url="https://auth-test.example/1",
        source_doc_id="auth-test-1",
        source_hash="hash-auth-1",
        embedding=_vec(1),
    )
    session.add(g)
    await session.flush()
    await session.refresh(g)
    await session.commit()
    return g


def _override_user(app, user: AuthenticatedUser | None) -> None:  # type: ignore[no-untyped-def]
    """Pin `optional_user` to a specific identity (or None for anonymous).

    Cleared automatically by the `client` fixture's teardown.
    """
    async def _dep() -> AuthenticatedUser | None:
        return user
    app.dependency_overrides[optional_user] = _dep


def _install_stubs(client: AsyncClient) -> None:
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.scheduler_embedder = _StubEmbedder()
    app.state.reranker = _StubReranker()
    app.state.agent_llm = _StubAgentLLM()


# ---------------------------------------------------------------------------
# Anonymous baseline — proves the pre-auth behaviour is preserved
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_anonymous_recommend_creates_anon_owned_session(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]
    _override_user(app, None)

    r = await client.post("/agents/recommend", json={"query": "test query"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    from app.models.session import AgentSession
    row = await db_session.get(AgentSession, uuid.UUID(sid))
    assert row is not None
    # Anonymous sessions still carry the anon-<uuid> sentinel.
    assert row.owner_user_id == f"anon-{sid}"


# ---------------------------------------------------------------------------
# Authenticated path
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_authenticated_recommend_creates_user_owned_session(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]
    _override_user(app, AuthenticatedUser(id="user_abc", email="abc@example.com"))

    r = await client.post("/agents/recommend", json={"query": "test query"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    from app.models.session import AgentSession
    row = await db_session.get(AgentSession, uuid.UUID(sid))
    assert row is not None
    # New session created directly with the Clerk user id.
    assert row.owner_user_id == "user_abc"


# ---------------------------------------------------------------------------
# Mid-chat migration — the headline feature
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_anon_session_migrates_to_user_on_sign_in(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Turn 1 is anonymous; turn 2 is the same session_id but the user
    has signed in. The session row's owner_user_id transfers to the
    real Clerk identity and the chat history is preserved.
    """
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]

    # Turn 1: anonymous.
    _override_user(app, None)
    r1 = await client.post("/agents/recommend", json={"query": "turn one"})
    sid = r1.json()["session_id"]

    from app.models.session import AgentSession
    row = await db_session.get(AgentSession, uuid.UUID(sid))
    assert row is not None
    assert row.owner_user_id == f"anon-{sid}"
    assert len(row.conversation_history) == 1

    # Turn 2: same session, now signed in.
    _override_user(app, AuthenticatedUser(id="user_after_signin"))
    r2 = await client.post(
        "/agents/recommend",
        json={"query": "turn two", "session_id": sid},
    )
    assert r2.status_code == 200, r2.text

    # Refresh from DB — owner transferred, history grew to 2.
    await db_session.refresh(row)
    assert row.owner_user_id == "user_after_signin"
    assert len(row.conversation_history) == 2


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_user_cannot_read_another_users_session(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]

    _override_user(app, AuthenticatedUser(id="user_alice"))
    sid = (await client.post("/agents/recommend", json={"query": "alice"})).json()["session_id"]

    # Bob comes in with the same session id (e.g. guessed UUID).
    _override_user(app, AuthenticatedUser(id="user_bob"))
    r = await client.get(f"/agents/sessions/{sid}")
    assert r.status_code == 200
    # Empty history — the route hides the row from a non-owner instead
    # of 404-ing (avoids leaking that the id exists).
    body = r.json()
    assert body["history"] == []


@pytest.mark.integration
async def test_user_cannot_delete_another_users_session(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]

    _override_user(app, AuthenticatedUser(id="user_alice"))
    sid = (await client.post("/agents/recommend", json={"query": "alice"})).json()["session_id"]

    _override_user(app, AuthenticatedUser(id="user_bob"))
    r = await client.delete(f"/agents/sessions/{sid}")
    assert r.status_code == 204

    # Alice can still read her session — Bob's delete was a no-op.
    _override_user(app, AuthenticatedUser(id="user_alice"))
    body = (await client.get(f"/agents/sessions/{sid}")).json()
    assert len(body["history"]) == 1


# ---------------------------------------------------------------------------
# Anonymous sessions stay readable by anyone who has the id (current contract)
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_anon_session_is_readable_by_anyone(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_grant(db_session)
    _install_stubs(client)
    app = client._transport.app  # type: ignore[attr-defined]

    _override_user(app, None)
    sid = (await client.post("/agents/recommend", json={"query": "anon"})).json()["session_id"]

    # A signed-in user with the id can still read it (Clerk never owned
    # it; we don't 404 anonymous rows on cross-identity access).
    _override_user(app, AuthenticatedUser(id="user_random"))
    r = await client.get(f"/agents/sessions/{sid}")
    assert r.status_code == 200
    assert len(r.json()["history"]) == 1

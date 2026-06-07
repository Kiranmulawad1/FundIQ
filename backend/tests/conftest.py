"""Test fixtures.

Strategy:
  - Single async engine per test session (NullPool — no pool teardown races).
  - One outer transaction per test, rolled back in teardown. Each test sees a
    pristine DB without re-running DDL.
  - `get_session` dependency overridden to yield the same transactional
    session the test asserts against — the route's writes and the test's
    reads share one transaction.
  - `ENVIRONMENT=test` forced before `Settings` is constructed so the
    rate-limit middleware is bypassed and the engine uses NullPool.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

# Force test environment BEFORE any app import — Settings reads env on import.
os.environ["ENVIRONMENT"] = "test"
os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep test output quiet

# Clear Clerk creds so `current_user` takes the dev-fallback branch
# (synthetic dev user, no JWKS round-trip). Setting these to empty
# strings overrides any `.env` value because Settings is configured
# with `env_ignore_empty=True` — empty becomes equivalent to unset.
# Suite-level tests that need distinct identities use
# `dependency_overrides` instead (see tests/integration/test_agents_auth.py).
for _var in (
    "CLERK_SECRET_KEY",
    "CLERK_PUBLISHABLE_KEY",
    "CLERK_JWKS_URL",
    # Provider keys — individual tests opt in via monkeypatch.setenv when
    # they want to assert the present-key behaviour. Blanking here keeps
    # `.env` values from leaking into the missing-key assertions.
    "COHERE_API_KEY",
    "GEMINI_API_KEY",
):
    os.environ[_var] = ""

# Pop ALEMBIC_DATABASE_URL so test_config.py can exercise the derive-from-
# DATABASE_URL code path. In CI we set ALEMBIC_DATABASE_URL explicitly for
# the alembic-upgrade step, but the unit test wants to assert that the
# model_validator computes it when absent. Pydantic-settings reads process
# env even with _env_file=None, so the only reliable way is to remove it
# from the environment at conftest load.
os.environ.pop("ALEMBIC_DATABASE_URL", None)


# Register stub prompts for the agent nodes. Production fetches these from
# Langfuse; tests can't hit the network and shouldn't depend on Langfuse
# being seeded. The stubs are intentionally minimal — agent code mostly
# mocks the LLM call itself, but if it ever actually formats the prompt
# (e.g. integration smoke), the placeholders are valid.
def _register_prompt_stubs() -> None:
    from app.core.prompts import set_test_override

    set_test_override("planner", "PLANNER {{profile_block}} {{query}}")
    set_test_override("scorer", "SCORER {{planner_json}} {{candidates_json}}")
    set_test_override(
        "writer",
        "WRITER {{retry_block}} {{query}} {{planner_json}} {{candidates_json}} {{scorer_json}}",
    )
    set_test_override(
        "critic",
        "CRITIC {{query}} {{profile_block}} {{planner_json}} {{candidates_json}} "
        "{{scorer_json}} {{writer_json}}",
    )
    set_test_override("hyde", "HYDE {{query}}")
    set_test_override(
        "enrichment",
        "ENRICHMENT {{title}} {{summary}} {{body_excerpt}} {{excerpt_chars}}",
    )


_register_prompt_stubs()


@pytest.fixture(scope="session")
def settings():  # type: ignore[no-untyped-def]
    """Cached settings, with the lru_cache cleared so env overrides take effect."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def engine(settings):  # type: ignore[no-untyped-def]
    """Per-test async engine.

    Function-scoped (not session-scoped) because asyncpg binds its connections
    to the event loop that created them, and pytest-asyncio creates a new
    event loop per test function. A session-scoped engine would attach
    connections to the session loop and fail with "got Future attached to a
    different loop" on first test use.
    """
    eng = create_async_engine(
        settings.database_url,
        poolclass=NullPool,
        future=True,
    )
    yield eng
    await eng.dispose()


@pytest.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:  # type: ignore[no-untyped-def]
    """Per-test transactional session. All writes rolled back on teardown.

    Pattern: open a connection, begin an outer transaction, bind a session
    to that connection. The session's commits/rollbacks turn into SAVEPOINTs
    inside the outer transaction (SQLAlchemy `join_transaction_mode="create_savepoint"`).
    Teardown rolls back the outer transaction, undoing every write.
    """
    async with engine.connect() as conn:
        outer = await conn.begin()
        sessionmaker = async_sessionmaker(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        async with sessionmaker() as session:
            try:
                yield session
            finally:
                await session.close()
        await outer.rollback()


@pytest.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:  # type: ignore[no-untyped-def]
    """In-process ASGI client. Shares the test's transaction via dependency override.

    We also point `app.state.sessionmaker` at a SAVEPOINT-bound sessionmaker
    so background tasks (e.g. /admin/scrape's scrape_portal workflow) write
    into the same transaction the test rolls back, preventing leakage into
    the real DB.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.core.db import get_session
    from app.main import create_app

    app = create_app()

    async def _override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            # Replace the real sessionmaker with one bound to the test's
            # connection so workflow-internal writes also rollback.
            app.state.sessionmaker = async_sessionmaker(
                bind=db_session.bind,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )
            yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def freeze_request_id(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin a request_id for deterministic log assertions."""
    rid = "test-req-0001"
    from app.core import middleware

    monkeypatch.setattr(middleware, "_new_request_id", lambda: rid)
    return rid


def make_startup(**overrides: Any) -> dict[str, Any]:
    """Builder for test Startup rows. Override any field with kwargs."""
    base: dict[str, Any] = {
        "owner_user_id": "user_test",
        "name": "Acme Robotics",
        "sector": "deeptech",
        "stage": "pre_seed",
        "country": "DE",
    }
    base.update(overrides)
    return base

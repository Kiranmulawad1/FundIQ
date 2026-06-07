"""Langfuse no-op behaviour.

The whole point of `core.observability` is that callers can sprinkle
`record_generation(...)` / `trace_request(...)` everywhere without
guarding on whether Langfuse is configured. These tests pin that
contract: with no keys, the helpers must be silent + side-effect-free.
"""

from __future__ import annotations

import pytest

from app.core import observability


@pytest.fixture(autouse=True)
def _reset_langfuse_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a fresh init on each test so we control state."""
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_initialised", False)


def test_init_with_no_credentials_does_not_construct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    from app.core.config import get_settings

    get_settings.cache_clear()

    observability.init_langfuse()
    assert observability._client is None
    assert observability._initialised is True


def test_record_generation_without_client_is_silent() -> None:
    # Must not raise — the caller has no `if` to guard it.
    observability.record_generation(
        name="x",
        model="gemini-2.5-flash",
        input="hi",
        output="ok",
    )


def test_trace_request_context_manager_without_client_yields_none() -> None:
    with observability.trace_request("any.name") as trace:
        assert trace is None


def test_cost_estimate_matches_pricing_constants() -> None:
    # 1M input + 1M output tokens.
    cost = observability.estimate_gemini_flash_cost_usd(1_000_000, 1_000_000)
    # 0.075 + 0.30 = 0.375 USD; allow small float wobble.
    assert abs(cost - 0.375) < 1e-6


@pytest.mark.asyncio
async def test_shutdown_without_client_does_not_raise() -> None:
    await observability.shutdown_langfuse()  # type: ignore[func-returns-value]

"""Unit tests for the 429-rate-limit backoff helpers in app.agents.llm.

These tests use a fake httpx transport that returns a programmable
sequence of responses so we can verify the retry logic without depending
on a real network or sleeping for the actual fallback intervals.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from app.agents import llm as llm_module
from app.agents.llm import (
    MAX_RATE_LIMIT_RETRIES,
    RATE_LIMIT_FALLBACK_DELAYS,
    _parse_retry_after,
    _post_with_rate_limit_retry,
)


# ---------------------------------------------------------------------------
# _parse_retry_after — pure function, easy to pin down.
# ---------------------------------------------------------------------------
def test_parse_retry_after_returns_seconds_for_int_string() -> None:
    assert _parse_retry_after("15") == 15.0


def test_parse_retry_after_returns_seconds_for_float_string() -> None:
    assert _parse_retry_after("12.5") == 12.5


def test_parse_retry_after_clamps_to_cap() -> None:
    # MAX_RETRY_AFTER_SECONDS = 90; anything larger gets clamped.
    assert _parse_retry_after("3600") == 90.0


def test_parse_retry_after_rejects_negative() -> None:
    assert _parse_retry_after("-5") is None


def test_parse_retry_after_rejects_http_date() -> None:
    # We deliberately don't parse the HTTP-date form — caller falls back
    # to its own delay sequence.
    assert _parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT") is None


def test_parse_retry_after_handles_none() -> None:
    assert _parse_retry_after(None) is None


# ---------------------------------------------------------------------------
# _post_with_rate_limit_retry — programmable transport.
# ---------------------------------------------------------------------------
class _Scripted429Transport(httpx.AsyncBaseTransport):
    """Returns a pre-programmed sequence of (status_code, headers) pairs."""

    def __init__(self, responses: Iterator[tuple[int, dict[str, str]]]) -> None:
        self._responses = responses
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        status, headers = next(self._responses)
        return httpx.Response(status_code=status, headers=headers, content=b"{}")


@pytest.mark.asyncio
async def test_post_returns_immediately_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """No 429 → no retry, one call."""
    # Patch asyncio.sleep to fail loudly if we accidentally sleep on a
    # success path; this is a stronger assertion than counting calls.
    monkeypatch.setattr(llm_module.asyncio, "sleep", _fail_sleep)

    transport = _Scripted429Transport(iter([(200, {})]))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await _post_with_rate_limit_retry(
            client, "http://t/x", json_payload={"a": 1},
        )
    assert r.status_code == 200
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_post_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the server sends Retry-After: 7, we sleep exactly 7 seconds
    before retrying.
    """
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)

    transport = _Scripted429Transport(iter([
        (429, {"Retry-After": "7"}),
        (200, {}),
    ]))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await _post_with_rate_limit_retry(
            client, "http://t/x", json_payload={"a": 1},
        )

    assert r.status_code == 200
    assert transport.calls == 2
    assert sleep_calls == [7.0]


@pytest.mark.asyncio
async def test_post_falls_back_to_default_delays_without_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Retry-After header → use RATE_LIMIT_FALLBACK_DELAYS in order."""
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)

    transport = _Scripted429Transport(iter([
        (429, {}),
        (429, {}),
        (200, {}),
    ]))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await _post_with_rate_limit_retry(
            client, "http://t/x", json_payload={"a": 1},
        )

    assert transport.calls == 3
    # First two sleeps use the first two delays in RATE_LIMIT_FALLBACK_DELAYS.
    assert sleep_calls == [
        RATE_LIMIT_FALLBACK_DELAYS[0],
        RATE_LIMIT_FALLBACK_DELAYS[1],
    ]


@pytest.mark.asyncio
async def test_post_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After MAX_RATE_LIMIT_RETRIES retries the helper returns the last
    429 unchanged — it does NOT raise. The caller (which then calls
    `r.raise_for_status()`) decides whether to fail.
    """
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)

    # All responses 429.
    transport = _Scripted429Transport(iter(
        [(429, {})] * (MAX_RATE_LIMIT_RETRIES + 1),
    ))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await _post_with_rate_limit_retry(
            client, "http://t/x", json_payload={"a": 1},
        )

    assert r.status_code == 429
    # Total calls = initial + MAX retries
    assert transport.calls == MAX_RATE_LIMIT_RETRIES + 1
    # We sleep MAX_RATE_LIMIT_RETRIES times (one per retry).
    assert len(sleep_calls) == MAX_RATE_LIMIT_RETRIES


@pytest.mark.asyncio
async def test_post_retries_on_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient 5xx (Gemini "service unavailable") is retried the same
    way as 429. Regression: a real 503 in production fell through to the
    degraded fallback because this path was 429-only.
    """
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)

    transport = _Scripted429Transport(iter([
        (503, {}),
        (200, {}),
    ]))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await _post_with_rate_limit_retry(
            client, "http://t/x", json_payload={"a": 1},
        )

    assert r.status_code == 200
    assert transport.calls == 2
    assert sleep_calls == [RATE_LIMIT_FALLBACK_DELAYS[0]]


async def _fail_sleep(_seconds: float) -> None:
    raise AssertionError("asyncio.sleep should not be called on the success path")

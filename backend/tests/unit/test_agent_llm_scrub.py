"""Regression test: agent LLM error messages must NEVER leak the Gemini key.

httpx renders the full request URL when raising HTTP errors; that URL
includes our `?key=AIza…` query param. Earlier versions of the agent
graph surfaced this in the planner's fallback rationale — a silent leak
into logs and API responses. The `_scrub` helper exists to prevent that
specific incident from recurring.
"""

from __future__ import annotations

from app.agents.llm import _scrub


def test_scrub_redacts_query_string_key() -> None:
    raw = (
        "Client error '429 Too Many Requests' for url "
        "'https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent?key=AIzaSyABC123_XYZ-456'"
    )
    out = _scrub(raw)
    assert "AIzaSyABC123" not in out
    assert "?key=<redacted>" in out


def test_scrub_handles_ampersand_form() -> None:
    raw = "POST https://x.test/path?foo=bar&key=AIzaSyDeadBeef baz"
    out = _scrub(raw)
    assert "AIzaSyDeadBeef" not in out
    assert "?key=<redacted>" in out


def test_scrub_passes_through_when_no_key() -> None:
    raw = "Internal connection error"
    assert _scrub(raw) == raw

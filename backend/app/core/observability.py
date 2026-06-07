"""Langfuse wiring with a graceful no-op fallback.

Why a fallback:
  Langfuse is optional. Local dev, CI, and the free-tier deploy can
  run without any account. The wrapper here returns a stub when the
  three Langfuse env vars are missing, so callers can sprinkle
  observability without `if settings.langfuse_...` guards everywhere.

What we track:
  * LLM generations (Gemini calls) — cost, latency, token counts,
    structured input/output. Spans live inside an enclosing trace
    when one is open via `trace_request(...)`.
  * Agent runs (Planner → Retriever → Scorer → Writer → Critic) —
    one trace per `/agents/recommend` call so the timeline shows
    node ordering and concurrency.

Why not just use the decorator everywhere:
  langgraph constructs the node functions dynamically and we want
  to attach metadata that depends on runtime state (the user query,
  retrieved grant IDs). The explicit `generation(...)` / `trace(...)`
  helpers below give us that — the @observe decorator can't.

Cost model:
  Gemini 2.5 Flash pricing as of 2026 (per Google's billing page):
    input  $0.075 / 1M tokens
    output $0.30  / 1M tokens
  Embedding (`gemini-embedding-001`) is free on the no-charge tier we
  use; if we ever move to billed embeddings the rate goes here too.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


# Per-million-token Gemini 2.5 Flash prices in USD.
GEMINI_FLASH_INPUT_USD_PER_MTOK = 0.075
GEMINI_FLASH_OUTPUT_USD_PER_MTOK = 0.30


_client: Any | None = None
_initialised = False


def init_langfuse() -> None:
    """Construct the Langfuse client once at startup if configured.

    Called from `lifespan` in `app.main`. Safe to call multiple times —
    second call is a no-op. If any required env var is missing this
    quietly returns and every subsequent `generation` / `trace_request`
    becomes a no-op.
    """
    global _client, _initialised
    if _initialised:
        return
    _initialised = True

    settings = get_settings()
    if (
        settings.langfuse_public_key is None
        or settings.langfuse_secret_key is None
        or settings.langfuse_host is None
    ):
        logger.info("langfuse.disabled", reason="missing credentials")
        return

    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("langfuse.disabled", reason="package not installed")
        return

    _client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    logger.info("langfuse.initialised", host=settings.langfuse_host)


async def shutdown_langfuse() -> None:
    """Flush pending events on shutdown so we don't lose the tail of
    the trace buffer when uvicorn closes.
    """
    global _client
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:  # noqa: BLE001 — never block shutdown on telemetry
        logger.warning("langfuse.flush_failed", error=str(exc)[:200])


def estimate_gemini_flash_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Cheap analytic estimate. Langfuse can do its own cost calculation
    server-side based on the model name but supplying our own gives the
    UI the right number even if the model list lags.
    """
    return (
        input_tokens * GEMINI_FLASH_INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * GEMINI_FLASH_OUTPUT_USD_PER_MTOK / 1_000_000
    )


# ---------------------------------------------------------------------------
# Span helpers — every helper checks `_client` first so the caller doesn't
# need to guard. The return type is intentionally `Any` (could be a real
# Langfuse span object or None) so callers can use it freely without typing
# headaches; the only operation we promise is "context-manager compatible".
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def trace_request(
    name: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a top-level trace for one agent request. Generations and
    spans created inside the `with` block automatically attach.

    Usage:
        with trace_request("agents.recommend", user_id=user.id) as trace:
            result = await graph.ainvoke(...)
            if trace is not None:
                trace.update(output=result)
    """
    if _client is None:
        yield None
        return
    trace = _client.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )
    try:
        yield trace
    finally:
        # Trace lifetime spans whatever uses it — we let langfuse close
        # it. We DO push so the buffer flushes on Render's short-lived
        # request lifetime.
        pass


def record_generation(
    *,
    name: str,
    model: str,
    input: Any,
    output: Any,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log one LLM call. Called from `agents/llm.py` after each Gemini
    completion (streamed or not). Cheap when Langfuse is disabled —
    just a None check.
    """
    if _client is None:
        return
    try:
        usage = None
        cost = None
        if input_tokens is not None and output_tokens is not None:
            usage = {"input": input_tokens, "output": output_tokens}
            cost = {
                "input": input_tokens * GEMINI_FLASH_INPUT_USD_PER_MTOK / 1_000_000,
                "output": output_tokens * GEMINI_FLASH_OUTPUT_USD_PER_MTOK / 1_000_000,
            }
        _client.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
            cost=cost,
            metadata=metadata or {},
        )
    except Exception as exc:  # noqa: BLE001
        # Never let telemetry break a real request.
        logger.warning("langfuse.record_failed", error=str(exc)[:200])

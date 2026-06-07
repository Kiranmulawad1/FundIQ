"""Shared httpx POST retry helper with bounded backoff.

Used by `services.embedding`, `rag.reranker`, and `agents.llm`.

Why one helper instead of three local copies:
  Before this module each service had its own ad-hoc retry. They diverged:
  the embedder retried 429+5xx, the reranker retried nothing at all,
  the agent LLM retried 429 only. When Gemini briefly 503'd in production
  the Writer node fell straight through to the degraded fallback message
  even though a single retry would have succeeded. Centralising the
  policy here keeps them in lockstep — any change to the retry budget
  or status list lands everywhere at once.

Policy:
  Three retries with exponential-jittered backoff: 1.5s, 4s, 10s
  (+ up to 0.5s jitter each). Honours `Retry-After` when the upstream
  sends it (Gemini's quota responses do; Cohere doesn't). Worst-case
  wait ≈ 15.5s, which sits inside the typical request budget without
  pushing the user past the streaming-spinner threshold.

Statuses we retry:
  429 (rate limit), 500/502/503/504 (transient upstream). 4xx other
  than 429 means the request is wrong — we don't retry those.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

import httpx

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)

# 3 retries → worst-case 1.5 + 4 + 10 ≈ 15.5s of sleep, plus jitter.
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.5, 4.0, 10.0)
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# If the upstream's Retry-After value exceeds this we give up. Quota
# is probably exhausted and the caller is better off failing fast than
# hanging on a single request for minutes.
MAX_RETRY_AFTER_SECONDS = 90.0


def _retry_after_seconds(header: str | None) -> float | None:
    """Parse a Retry-After header. Spec allows int seconds or HTTP-date;
    we only support the seconds form. Returns None if unusable, capped
    at MAX_RETRY_AFTER_SECONDS otherwise.
    """
    if not header:
        return None
    try:
        seconds = float(header)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


async def post_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    label: str,
    json: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST with bounded retries on 429/5xx.

    `label` is the structlog event prefix (e.g. `"embedding.gemini"`)
    so retry events show up under a consistent name per caller.

    Returns the first successful response. Raises:
      * `httpx.HTTPStatusError` for non-retryable 4xx (preserving the
        usual httpx behaviour after retries are exhausted, the final
        response is also raised on)
      * `httpx.RequestError` subclasses (network errors) — these are
        not retried; the caller usually wraps them in a service-level
        error class.
    """
    last_exc: httpx.HTTPStatusError | None = None
    # Iterate over (delay, attempt_index). The trailing `None` denotes
    # "this is the last attempt, don't sleep — raise on error."
    for attempt, base_delay in enumerate((*RETRY_DELAYS_SECONDS, None)):
        try:
            r = await client.post(url, json=dict(json) if json else None, headers=dict(headers) if headers else None)
        except httpx.RequestError:
            # Network-level errors (timeouts, DNS, connection reset). We
            # could retry these too, but they usually mean something the
            # caller wants to surface — and httpx already has its own
            # transport-level retry settings.
            raise

        if r.status_code not in RETRY_STATUSES or base_delay is None:
            # Either a success/non-retryable response, or we're out of
            # attempts. Let the caller see whatever came back.
            r.raise_for_status()
            return r

        sleep_seconds = (
            _retry_after_seconds(r.headers.get("retry-after"))
            or base_delay + random.uniform(0, 0.5)
        )
        logger.warning(
            f"{label}.retry",
            attempt=attempt + 1,
            status=r.status_code,
            sleep_seconds=round(sleep_seconds, 2),
        )
        # Consume the response so the connection can be released.
        await r.aclose()
        await asyncio.sleep(sleep_seconds)

    # Unreachable in normal flow — the final attempt either returns or
    # raises inside the loop. Keeps mypy happy and guards against future
    # refactors silently breaking the contract.
    if last_exc is not None:
        raise last_exc
    msg = f"{label}: retry loop exhausted with no response"
    raise RuntimeError(msg)

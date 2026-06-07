"""Shared Gemini JSON-mode client for the agent graph.

The HyDE service (app/rag/hyde.py) speaks Gemini too, but its surface is
specific to "generate 3 hypothetical descriptions". The agent graph needs
a generic "ask Gemini for a JSON object that matches this Pydantic model"
shape — and to share one httpx client across Planner + Writer to avoid
per-call connection setup.

Why duplicate the call code instead of importing HyDE?
  - HyDE has a fixed prompt template and result shape baked in.
  - The agent graph wants a typed `respond_as(model_cls, prompt)` surface.
  - The httpx client lifecycle is the same shape but the contract differs.
A future refactor can unify these once the Writer agent stabilises.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, TYPE_CHECKING, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger

# httpx error messages embed the request URL — which includes our Gemini
# API key as a query param. Scrub before logging or surfacing to callers.
_KEY_PATTERN = re.compile(r"[?&]key=[A-Za-z0-9_\-]+")


def _scrub(text: str) -> str:
    return _KEY_PATTERN.sub("?key=<redacted>", text)


def _parse_retry_after(value: str | None) -> float | None:
    """Read a `Retry-After` header. The spec allows two forms — delta
    seconds (integer) or HTTP-date. We only support the seconds form;
    if the server sends a date we ignore it and fall back to our own
    delay sequence.
    """
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


async def _post_with_rate_limit_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_payload: Mapping[str, Any],
) -> httpx.Response:
    """POST with retries on 429 AND transient 5xx, up to MAX_RATE_LIMIT_RETRIES.

    Returns the final response (which may still be a non-2xx — the caller
    is responsible for raise_for_status).

    Why we now retry 5xx too:
      Until 2026-06 this helper only retried 429 (rate limit). When
      Gemini briefly 503'd a Writer call in production the agent fell
      straight through to the degraded retrieval-only fallback, even
      though a single retry would have succeeded. Treating 5xx the same
      as 429 closes that gap — both indicate transient upstream pain
      that the next attempt is likely to clear.
    """
    last_response: httpx.Response | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        last_response = await client.post(url, json=json_payload)
        if last_response.status_code not in _RETRYABLE_STATUSES:
            return last_response
        if attempt >= MAX_RATE_LIMIT_RETRIES:
            break
        delay = (
            _parse_retry_after(last_response.headers.get("Retry-After"))
            or RATE_LIMIT_FALLBACK_DELAYS[min(attempt, len(RATE_LIMIT_FALLBACK_DELAYS) - 1)]
        )
        logger.warning(
            "agent.llm.transient_failure",
            attempt=attempt + 1,
            status=last_response.status_code,
            sleeping_s=delay,
        )
        await asyncio.sleep(delay)
    assert last_response is not None
    return last_response


async def _sleep_for_rate_limit_only(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_payload: Mapping[str, Any],
) -> None:
    """Pre-flight 429 probe for streaming endpoints.

    Sends one cheap HEAD-like POST to detect rate limiting before opening
    the streaming context. If 429, sleeps per Retry-After / fallback
    sequence and retries; passes through silently on non-429 responses
    (the caller's `.stream(...)` then reissues the request).

    We can't reuse the response body because the streaming client needs
    its own connection — this helper exists purely to gate on quota
    before allocating that long-lived connection.
    """
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        # Use a HEAD-like probe by sending the POST and immediately
        # reading status. httpx will buffer at most one chunk — cheap.
        probe = await client.post(url, json=json_payload)
        try:
            if probe.status_code not in _RETRYABLE_STATUSES:
                return
            if attempt >= MAX_RATE_LIMIT_RETRIES:
                return
            delay = (
                _parse_retry_after(probe.headers.get("Retry-After"))
                or RATE_LIMIT_FALLBACK_DELAYS[
                    min(attempt, len(RATE_LIMIT_FALLBACK_DELAYS) - 1)
                ]
            )
            logger.warning(
                "agent.llm.stream_transient_failure",
                attempt=attempt + 1,
                status=probe.status_code,
                sleeping_s=delay,
            )
            await asyncio.sleep(delay)
        finally:
            await probe.aclose()

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# Rate-limit backoff knobs. Gemini's free tier is 10 RPM / 250k TPM /
# 250 RPD; bursting hits 429 surprisingly easily during bulk enrichment
# or back-to-back agent calls. We honour `Retry-After` when the server
# sends it; otherwise we fall back to exponential delays (5s, 15s, 45s)
# capped at MAX_RATE_LIMIT_RETRIES attempts.
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_FALLBACK_DELAYS = (5.0, 15.0, 45.0)
# Statuses that a retry will plausibly clear. 4xx other than 429 are the
# caller's bug — surfacing those immediately avoids masking real errors.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
# Above this Retry-After value we give up — the daily quota is probably
# exhausted and the caller's better off failing fast than blocking for
# 10+ minutes inside one request.
MAX_RETRY_AFTER_SECONDS = 90.0
GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent"
)
DEFAULT_TIMEOUT_SECONDS = 30.0  # Writer prompts are long; planner is short.
# Streaming Writer can take much longer than batch (we want to allow
# every byte to arrive even on a slow link).
STREAM_TIMEOUT_SECONDS = 120.0

T = TypeVar("T", bound=BaseModel)


class AgentLLMError(RuntimeError):
    """Raised when Gemini fails or returns output we can't parse."""


class GeminiAgentClient:
    """One httpx client per process; reused by Planner + Writer.

    Use as an async context manager when constructing standalone, or pass
    in your own client (e.g. for tests).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout_seconds

    async def __aenter__(self) -> GeminiAgentClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream_text(
        self,
        *,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
        json_mode: bool = True,
        prompt_handle: object | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks from Gemini's streamGenerateContent SSE.

        Used by the Writer node in the streaming graph path so the UI can
        render the response as it's being generated instead of waiting
        15-30s for the full call.

        When `json_mode=True`, Gemini still produces JSON-shaped output —
        the consumer is expected to assemble the full text and parse at
        the end. The frontend can extract the `summary` field
        progressively via regex on the in-flight buffer.

        Raises `AgentLLMError` on missing API key or transport failure.
        """
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise AgentLLMError("GEMINI_API_KEY is not configured.")
        api_key = settings.gemini_api_key.get_secret_value()

        gen_config: dict[str, object] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            "generationConfig": gen_config,
        }

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=STREAM_TIMEOUT_SECONDS)
            self._owns_client = True

        url = f"{GEMINI_STREAM_URL}?alt=sse&key={api_key}"
        # Buffers for the trailing Langfuse record_generation call. Even
        # though we yield chunks live, we accumulate the full text so the
        # Langfuse UI can show the complete response.
        full_text_parts: list[str] = []
        last_usage: dict[str, int] = {}
        try:
            # Sleep + retry on 429 BEFORE the streaming context opens. We
            # can't gracefully retry mid-stream (already yielded chunks
            # to the caller), so the backoff only protects the initial
            # request — the most common failure mode anyway.
            await _sleep_for_rate_limit_only(self._client, url, json_payload=payload)
            async with self._client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    # SSE event lines come as "data: {...}". Comments
                    # (`:` prefix) and blank lines are ignored.
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # Gemini occasionally splits a chunk mid-JSON; we
                        # drop those rather than try to reassemble — the
                        # next chunk usually re-emits the full delta.
                        logger.debug("agent.llm.stream_chunk_bad_json", raw=data[:200])
                        continue
                    # Gemini puts usage on the final chunk; keep the last
                    # one we see so we record real numbers, not None.
                    usage = chunk.get("usageMetadata")
                    if isinstance(usage, dict):
                        last_usage = usage
                    text = _extract_text(chunk)
                    if text:
                        full_text_parts.append(text)
                        yield text
        except httpx.HTTPError as e:
            scrubbed = _scrub(str(e))
            logger.warning(
                "agent.llm.gemini_stream_failed",
                error_type=type(e).__name__,
                error=scrubbed[:200],
            )
            raise AgentLLMError(f"Gemini stream failed: {scrubbed}") from e

        # Stream completed cleanly — record one Langfuse generation with
        # the full assembled text. We do this in the success path only so
        # partial-failure streams don't pollute the trace.
        from app.core.observability import record_generation

        record_generation(
            name="gemini.stream_text",
            model="gemini-2.5-flash",
            input=prompt,
            output="".join(full_text_parts),
            input_tokens=last_usage.get("promptTokenCount"),
            output_tokens=last_usage.get("candidatesTokenCount"),
            metadata={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "json_mode": json_mode,
                "streaming": True,
            },
            prompt_handle=prompt_handle,
        )

    async def respond_as(
        self,
        model_cls: type[T],
        *,
        prompt: str,
        temperature: float = 0.4,
        max_output_tokens: int = 2048,
        prompt_handle: object | None = None,
    ) -> T:
        """Ask Gemini for JSON parsable into `model_cls`.

        Raises `AgentLLMError` on missing API key, transport failure, or
        unparseable output — callers handle those at node boundaries.
        """
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise AgentLLMError("GEMINI_API_KEY is not configured.")
        api_key = settings.gemini_api_key.get_secret_value()

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": max_output_tokens,
            },
        }

        # Lazy-create + reuse the httpx client across calls so we keep the
        # TLS/HTTP connection pool warm. Lifespan closes via __aexit__.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True

        try:
            r = await _post_with_rate_limit_retry(
                self._client,
                f"{GEMINI_GENERATE_URL}?key={api_key}",
                json_payload=payload,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            scrubbed = _scrub(str(e))
            logger.warning(
                "agent.llm.gemini_failed",
                error_type=type(e).__name__,
                error=scrubbed[:200],
            )
            raise AgentLLMError(f"Gemini call failed: {scrubbed}") from e

        data = r.json()
        text = _extract_text(data)
        if not text:
            raise AgentLLMError("Gemini returned empty content.")

        # Record the generation to Langfuse before parsing — the raw text
        # is the most useful trace artefact even when JSON parsing fails.
        from app.core.observability import record_generation

        usage = data.get("usageMetadata") or {}
        record_generation(
            name=f"gemini.respond_as.{model_cls.__name__}",
            model="gemini-2.5-flash",
            input=prompt,
            output=text,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            metadata={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "schema": model_cls.__name__,
            },
            prompt_handle=prompt_handle,
        )

        try:
            obj = json.loads(_strip_fences(text))
        except json.JSONDecodeError as e:
            logger.warning("agent.llm.invalid_json", raw=text[:300])
            raise AgentLLMError(f"Gemini returned non-JSON: {e}") from e

        try:
            return model_cls.model_validate(obj)
        except ValidationError as e:
            logger.warning(
                "agent.llm.schema_mismatch",
                model=model_cls.__name__,
                errors=str(e)[:300],
            )
            raise AgentLLMError(
                f"Gemini output failed schema validation: {e}"
            ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_text(data: dict[str, object]) -> str:
    """Pull text out of the Gemini response envelope."""
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return ""
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict):
            t = p.get("text")
            if isinstance(t, str):
                chunks.append(t)
    return "\n".join(chunks).strip()


def _strip_fences(text: str) -> str:
    """Defensive — Gemini's JSON mode usually skips fences but old prompts may not."""
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

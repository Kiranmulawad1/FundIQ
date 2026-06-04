"""ASGI middleware: request_id injection and Redis-backed rate limiting.

Implemented as pure ASGI rather than Starlette's BaseHTTPMiddleware so that
streaming responses (SSE for the agent reasoning theater) are not buffered.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import Environment, get_settings
from app.core.logging import get_logger, request_id_ctx, user_id_ctx

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

REQUEST_ID_HEADER = b"x-request-id"


class RequestIDMiddleware:
    """Read or generate a request ID, set it in contextvars, echo it in the response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = _header(scope, REQUEST_ID_HEADER)
        request_id = inbound or _new_request_id()
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "http.request",
                method=scope.get("method"),
                path=scope.get("path"),
                elapsed_ms=round(elapsed_ms, 2),
            )
            request_id_ctx.reset(token)


class RateLimitMiddleware:
    """Sliding-window rate limit. Per-user when authenticated, per-IP otherwise.

    Redis is resolved from `scope["app"].state.redis` at request time so that
    the lifespan-managed connection pool is used (no separate pool per
    middleware instance).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        per_minute: int,
        burst: int,
    ) -> None:
        self.app = app
        self.per_minute = per_minute
        self.burst = burst
        self.window_seconds = 60
        self.settings = get_settings()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.settings.environment is Environment.TEST:
            await self.app(scope, receive, send)
            return

        # Skip health checks — uptime probes shouldn't consume tokens.
        path = scope.get("path", "")
        if path in ("/health", "/healthz", "/ready"):
            await self.app(scope, receive, send)
            return

        redis: Redis | None = getattr(scope["app"].state, "redis", None)
        if redis is None:
            # Lifespan hasn't initialised Redis yet — fail open rather than block.
            await self.app(scope, receive, send)
            return

        identity = user_id_ctx.get() or _client_ip(scope)
        key = f"ratelimit:{identity}"
        now = int(time.time())

        # ZSET sliding window: add this hit, drop entries older than the window, count.
        async with redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, now - self.window_seconds)
            pipe.zadd(key, {f"{now}:{uuid.uuid4().hex}": now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds + 1)
            _, _, count, _ = await pipe.execute()

        limit = self.per_minute + self.burst
        if count > limit:
            retry_after = str(self.window_seconds).encode("ascii")
            headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", retry_after),
                (b"x-ratelimit-limit", str(limit).encode("ascii")),
                (b"x-ratelimit-remaining", b"0"),
            ]
            body = (
                b'{"code":"rate_limited","message":"Too many requests.",'
                b'"request_id":"' + (request_id_ctx.get() or "").encode("ascii") + b'"}'
            )
            await send({"type": "http.response.start", "status": 429, "headers": headers})
            await send({"type": "http.response.body", "body": body})
            logger.warning("ratelimit.exceeded", identity=identity, count=count, limit=limit)
            return

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_request_id() -> str:
    # Time-prefixed for human-sortability in logs; uuid suffix for uniqueness.
    return f"{int(time.time() * 1000):x}-{uuid.uuid4().hex[:12]}"


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            try:
                return v.decode("ascii")
            except UnicodeDecodeError:
                return None
    return None


def _client_ip(scope: Scope) -> str:
    # Trust X-Forwarded-For only if running behind a known proxy (set in deployment).
    fwd = _header(scope, b"x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"

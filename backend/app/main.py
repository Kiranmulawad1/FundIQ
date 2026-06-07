"""FastAPI application entry point.

Construction order matters:
  1. configure_logging() — so import-time logs render correctly.
  2. lifespan(): init engine, redis, JWKS, optional Logfire.
  3. Middleware stack (RequestID -> CORS -> RateLimit).
  4. Exception handlers.
  5. Routers.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.api.routes import admin, agents, analytics, grants, health
from app.core.auth import fetch_jwks_eagerly
from app.core.config import Environment, get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RateLimitMiddleware, RequestIDMiddleware
from app.jobs.scheduler import create_scheduler

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info(
        "app.startup",
        env=settings.environment.value,
        version=settings.app_version,
    )

    init_engine()
    app.state.sessionmaker = get_sessionmaker()
    app.state.redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    await app.state.redis.ping()
    logger.info("redis.connected", url=settings.redis_url)

    await fetch_jwks_eagerly()

    logfire_token = (
        settings.logfire_token.get_secret_value() if settings.logfire_token else ""
    )
    if logfire_token:
        import logfire

        logfire.configure(
            token=logfire_token,
            service_name=settings.app_name,
            environment=settings.environment.value,
        )
        logfire.instrument_fastapi(app, capture_headers=False)
        logger.info("logfire.instrumented")

    # Langfuse — separate observability sink dedicated to LLM call traces.
    # No-op if any of the three Langfuse env vars is unset.
    from app.core.observability import init_langfuse

    init_langfuse()

    # Scheduler:
    #   - off in tests (deterministic test runs; cron firing during tests
    #     would write to scrape_runs and hit the live network)
    #   - off by default in development unless SCHEDULER_ENABLED=true
    #     (most dev sessions don't want a 03:00 UTC scrape ambushing them)
    #   - on in staging/production
    scheduler_enabled = (
        settings.environment is not Environment.TEST
        and (
            settings.is_production
            or os.environ.get("SCHEDULER_ENABLED", "").lower() in ("1", "true", "yes")
        )
    )
    # `WARM_MODELS` used to pre-load BGE + e5 into RAM. We moved both
    # to API-backed services (Gemini Embedding + Cohere Rerank), so
    # there are no torch models to warm and the env flag is a no-op.
    # Existing deploy configs that still set WARM_MODELS=true are
    # silently ignored.
    app.state.scheduler = None
    app.state.scheduler_embedder = None
    # Reranker (Phase 5A), HyDE service + semantic cache (Phase 5B),
    # agent LLM client (Phase 6) are lazily constructed on first use,
    # unless WARM_MODELS triggers eager construction below. Slots
    # initialised here so resolver helpers can `getattr(..., None)`
    # cleanly.
    app.state.reranker = None
    app.state.hyde = None
    app.state.rag_cache = None
    app.state.agent_llm = None

    # Construct one EmbeddingService at lifespan and share it with the
    # scheduler. Both the embedder and the reranker are API-backed now —
    # constructors are cheap (no model load), so we no longer guard them
    # behind WARM_MODELS / scheduler_enabled.
    from app.rag.reranker import RerankerService
    from app.services.embedding import EmbeddingService

    embedder = EmbeddingService(redis=app.state.redis)
    reranker = RerankerService()
    app.state.scheduler_embedder = embedder
    app.state.reranker = reranker

    if scheduler_enabled:
        scheduler = create_scheduler(
            sessionmaker=app.state.sessionmaker,
            embedder=embedder,
        )
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("scheduler.started", jobs=len(scheduler.get_jobs()))
    else:
        logger.info("scheduler.disabled", reason="test/dev — set SCHEDULER_ENABLED=true to enable")

    try:
        yield
    finally:
        logger.info("app.shutdown")
        if app.state.scheduler is not None:
            app.state.scheduler.shutdown(wait=False)
            logger.info("scheduler.stopped")
        # HyDEService owns an httpx client when constructed standalone;
        # closing it explicitly avoids "Unclosed client session" warnings.
        if app.state.hyde is not None:
            await app.state.hyde.__aexit__(None, None, None)
        # GeminiAgentClient lazily constructs its httpx client on first
        # call. Same teardown story.
        if app.state.agent_llm is not None:
            await app.state.agent_llm.__aexit__(None, None, None)
        # Flush Langfuse buffer before the worker exits — otherwise the
        # tail of the LLM-trace timeline gets dropped on Render restarts.
        from app.core.observability import shutdown_langfuse

        await shutdown_langfuse()
        await app.state.redis.aclose()
        await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FundIQ API",
        version=settings.app_version,
        description="AI funding intelligence for EU and German startups.",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # Middleware stack — order matters. Starlette wraps in reverse, so the
    # last `add_middleware` runs first. We want RequestID outermost.
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.rate_limit_per_minute,
        burst=settings.rate_limit_burst,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router, tags=["system"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])
    app.include_router(grants.router)        # prefix + tags set on the router itself
    app.include_router(analytics.router)     # prefix + tags set on the router itself
    app.include_router(agents.router)        # prefix + tags set on the router itself

    return app


app = create_app()

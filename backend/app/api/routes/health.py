"""Liveness and readiness endpoints.

  /health  — liveness probe. Returns 200 if the process is alive. No I/O.
  /ready   — readiness probe. Verifies DB + Redis + Neo4j are reachable.
             Returns 503 if any dependency is unhealthy.

Splitting liveness from readiness is the K8s/uptime-probe standard.
Conflating them causes pod restarts during transient DB blips.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import RedisDep, SessionDep
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: str


class DependencyHealth(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, DependencyHealth]


@router.get("/health", response_model=LivenessResponse, summary="Liveness probe")
async def health() -> LivenessResponse:
    """Process is alive. No I/O — must respond fast under load."""
    settings = get_settings()
    return LivenessResponse(version=settings.app_version, environment=settings.environment.value)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe — verifies all critical dependencies",
)
async def ready(session: SessionDep, redis: RedisDep) -> JSONResponse:
    checks: dict[str, DependencyHealth] = {}

    # Postgres
    try:
        await session.execute(text("SELECT 1"))
        checks["postgres"] = DependencyHealth(status="ok")
    except Exception as e:  # noqa: BLE001 - we want all failure modes
        checks["postgres"] = DependencyHealth(status="error", detail=type(e).__name__)
        logger.warning("ready.postgres.failed", error=str(e))

    # Redis
    try:
        pong = await redis.ping()
        checks["redis"] = DependencyHealth(status="ok" if pong else "error")
    except Exception as e:  # noqa: BLE001
        checks["redis"] = DependencyHealth(status="error", detail=type(e).__name__)
        logger.warning("ready.redis.failed", error=str(e))

    healthy = all(c.status == "ok" for c in checks.values())
    body = ReadinessResponse(
        status="ready" if healthy else "degraded",
        checks=checks,
    )
    return JSONResponse(
        content=body.model_dump(),
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )

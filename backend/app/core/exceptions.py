"""Domain exceptions and unified FastAPI error handlers.

Every error response shares the same envelope:
    {"code": "<stable_string>", "message": "...", "request_id": "...", "details": {...}}

Clients match on `code`. Tracebacks never leak to clients — they go to logs only.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException  # catches both FastAPI and Starlette

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------
class FundIQError(Exception):
    """Root domain exception. All custom errors inherit from this."""

    code: str = "internal_error"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(FundIQError):
    code = "not_found"
    http_status = status.HTTP_404_NOT_FOUND


class ValidationError(FundIQError):
    code = "validation_error"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class AuthError(FundIQError):
    code = "unauthorized"
    http_status = status.HTTP_401_UNAUTHORIZED


class ForbiddenError(FundIQError):
    code = "forbidden"
    http_status = status.HTTP_403_FORBIDDEN


class RateLimitedError(FundIQError):
    code = "rate_limited"
    http_status = status.HTTP_429_TOO_MANY_REQUESTS


class ExternalServiceError(FundIQError):
    """Upstream (OpenAI, Neon, Neo4j, Hatchet) failure. Retryable upstream of the API."""

    code = "external_service_error"
    http_status = status.HTTP_502_BAD_GATEWAY


class GuardrailViolation(FundIQError):
    """Prompt injection / PII / grounding failure. Always logged with severity."""

    code = "guardrail_violation"
    http_status = status.HTTP_400_BAD_REQUEST


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _envelope(
    *,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id_ctx.get(),
    }
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


async def _fundiq_error_handler(_: Request, exc: FundIQError) -> JSONResponse:
    logger.warning(
        "domain.error",
        code=exc.code,
        message=exc.message,
        http_status=exc.http_status,
        details=exc.details,
    )
    return _envelope(
        code=exc.code,
        message=exc.message,
        status_code=exc.http_status,
        details=exc.details or None,
    )


async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _envelope(
        code=f"http_{exc.status_code}",
        message=exc.detail,
        status_code=exc.status_code,
    )


async def _validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return _envelope(
        code="validation_error",
        message="Request validation failed.",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"errors": exc.errors()},
    )


async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled.exception", exc_type=type(exc).__name__)
    return _envelope(
        code="internal_error",
        message="An internal error occurred. Reference the request_id when reporting.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire handlers into the FastAPI app. Called from `main.py`."""
    app.add_exception_handler(FundIQError, _fundiq_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)

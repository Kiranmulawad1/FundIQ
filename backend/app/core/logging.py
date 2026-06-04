"""Structured JSON logging with request-scoped context.

Every log line carries `request_id`, `agent_id`, and `session_id` when set in
the current `contextvars` scope. Stdlib `logging` is routed through structlog
so third-party libraries (uvicorn, sqlalchemy, httpx) produce the same JSON
shape as our own code — one log pipeline, one schema.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import Environment, LogLevel, get_settings

# ---------------------------------------------------------------------------
# Context vars: set on request entry, read by the context processor.
# ---------------------------------------------------------------------------
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
agent_id_ctx: ContextVar[str | None] = ContextVar("agent_id", default=None)
session_id_ctx: ContextVar[str | None] = ContextVar("session_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)


def _inject_context(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Add request-scoped context vars to every log line, when present."""
    for key, ctx in (
        ("request_id", request_id_ctx),
        ("agent_id", agent_id_ctx),
        ("session_id", session_id_ctx),
        ("user_id", user_id_ctx),
    ):
        value = ctx.get()
        if value is not None:
            event_dict[key] = value
    return event_dict


def _drop_color_message_key(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Uvicorn re-injects the message with ANSI codes under `color_message`. Drop it."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """Configure structlog + stdlib logging. Idempotent — safe to call from lifespan."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.value)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _inject_context,
        _drop_color_message_key,
    ]

    renderer: Processor
    if settings.environment is Environment.DEVELOPMENT:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid duplicate handlers if called twice (tests, autoreload).
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Tame the noisy ones. SQLAlchemy is INFO-noisy; httpx logs every request.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(
            logging.WARNING if settings.log_level is LogLevel.INFO else log_level
        )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Public factory. Prefer this over `logging.getLogger`."""
    return structlog.stdlib.get_logger(name)

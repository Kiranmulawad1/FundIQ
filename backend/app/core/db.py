"""Async database engine, session factory, and FastAPI session dependency.

One engine per process, created during lifespan startup, disposed on shutdown.
Session-per-request via `get_session` Depends — rolls back on exception,
commits on success. `expire_on_commit=False` so response models can read
attributes after the transaction closes without triggering lazy loads.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import Environment, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    if _engine is None:  # pragma: no cover - guarded by lifespan
        msg = "Database engine not initialised. Call init_engine() in app lifespan."
        raise RuntimeError(msg)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:  # pragma: no cover
        msg = "Sessionmaker not initialised. Call init_engine() in app lifespan."
        raise RuntimeError(msg)
    return _sessionmaker


def init_engine() -> AsyncEngine:
    """Create the engine + sessionmaker. Called once from the FastAPI lifespan."""
    global _engine, _sessionmaker  # noqa: PLW0603 - module-level singleton by design

    settings = get_settings()

    pool_kwargs: dict[str, object] = {}
    if settings.environment is Environment.TEST:
        pool_kwargs["poolclass"] = NullPool
    else:
        pool_kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_pre_ping=True,
            pool_recycle=1800,  # Neon idle timeout safety
        )

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        future=True,
        **pool_kwargs,
    )
    _sessionmaker = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )
    logger.info("db.engine.initialised", env=settings.environment.value)
    return _engine


async def dispose_engine() -> None:
    """Close all pooled connections. Called from lifespan shutdown."""
    global _engine, _sessionmaker  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        logger.info("db.engine.disposed")
    _engine = None
    _sessionmaker = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, commits on success, rolls back on error."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

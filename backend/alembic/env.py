"""Alembic environment.

Runs sync (psycopg). The runtime app uses asyncpg — two drivers, same DB.

Key behaviours:
  - URL pulled from `Settings.alembic_database_url` (env-driven).
  - `target_metadata = SQLModel.metadata` after importing `app.models`,
    so autogenerate sees every table.
  - `render_item` hook teaches autogenerate to emit `pgvector.Vector(...)`
    and `sqlmodel.AutoString` correctly (default Alembic strips them).
  - `compare_type` + `compare_server_default` enabled — default Alembic
    silently misses Numeric precision changes and JSONB default drift.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# Import models package to populate SQLModel.metadata.
from app import models  # noqa: F401 - registers tables
from app.core.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
assert settings.alembic_database_url is not None  # validated in Settings
config.set_main_option("sqlalchemy.url", settings.alembic_database_url)

target_metadata = SQLModel.metadata


def _render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Teach autogenerate to render pgvector + sqlmodel types correctly."""
    if type_ == "type":
        # pgvector
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError:
            Vector = None  # type: ignore[assignment,misc]
        if Vector is not None and isinstance(obj, Vector):
            autogen_context.imports.add("import pgvector.sqlalchemy")
            return f"pgvector.sqlalchemy.Vector({obj.dim})"

        # SQLModel's AutoString — alias for sa.String
        from sqlmodel.sql.sqltypes import AutoString

        if isinstance(obj, AutoString):
            return "sa.String()"

    return False  # fall through to default rendering


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=_render_item,
        compare_type=True,
        compare_server_default=True,
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=_render_item,
            compare_type=True,
            compare_server_default=True,
            include_schemas=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

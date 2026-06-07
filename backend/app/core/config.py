"""Typed application configuration loaded from environment variables.

All env access in the codebase flows through `get_settings()`. There is no
`os.getenv()` anywhere else — this is the single configuration seam, which
makes tests trivial (override the cached singleton) and prevents secrets
from being read at arbitrary import sites.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file_candidates() -> tuple[str, ...]:
    """Locate `.env` whether the process runs from repo root or from `backend/`.

    pydantic-settings resolves `env_file` relative to CWD, which differs between
    `uv run uvicorn ...` (root) and `uv run alembic ...` (backend/). We hand it
    every plausible location so the search succeeds in both contexts.
    """
    here = Path(__file__).resolve()
    # Walk up to find the repo root (marker: pyproject.toml).
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return (str(parent / ".env"), ".env")
    return (".env",)


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings. Validated once on import; fail-fast on missing secrets."""

    model_config = SettingsConfigDict(
        env_file=_env_file_candidates(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    @field_validator(
        # Optional secrets (SecretStr | None).
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "cohere_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "hf_token",
        "langsmith_api_key",
        "logfire_token",
        "clerk_secret_key",
        "hatchet_client_token",
        # Optional plain strings — same trap: `FOO=` in .env reaches Python
        # as the empty string, not None, and downstream `is not None` checks
        # fire spuriously (e.g. fetching JWKS from "" → httpx ProtocolError).
        "clerk_publishable_key",
        "clerk_jwks_url",
        "langfuse_host",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        """Treat `FOO=` in .env (empty value) as None for optional fields."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # ---------------- App ----------------
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    app_port: int = 8000
    app_name: str = "fundiq"
    app_version: str = "0.1.0"

    # ---------------- CORS ----------------
    cors_origins: str = "http://localhost:5173"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ---------------- LLM providers ----------------
    # OpenAI is the master-plan default but optional in dev — we route through
    # Gemini when OpenAI is unset (cheaper free tier, strong German performance).
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    # Cohere — used for the API-backed reranker after we moved off
    # BAAI/bge-reranker-v2-m3 to stop hauling 2.3GB of torch into the
    # production image. Free tier covers 1000 rerank calls/month.
    cohere_api_key: SecretStr | None = None

    # ---------------- HuggingFace ----------------
    hf_token: SecretStr | None = None

    # ---------------- Observability ----------------
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "fundiq"
    langsmith_tracing: bool = False
    logfire_token: SecretStr | None = None
    # Langfuse — LLM tracing + cost tracking. Three values are required
    # for the SDK to send anything; if any of them is unset the
    # `core.observability` module short-circuits to a no-op. Free cloud
    # tier covers 50k events/month which is comfortable headroom for the
    # portfolio demo.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    # ---------------- Database (Neon Postgres + pgvector) ----------------
    database_url: str = Field(..., description="Async SQLAlchemy URL (asyncpg driver).")
    alembic_database_url: str | None = Field(
        default=None,
        description="Sync URL for Alembic (psycopg). Derived from database_url if unset.",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_echo: bool = False

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """Accept `postgres://` and `postgresql://`, normalize to asyncpg driver."""
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql://", 1)
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if not v.startswith("postgresql+asyncpg://"):
            msg = f"DATABASE_URL must use postgresql/postgresql+asyncpg scheme, got: {v[:30]}..."
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _derive_alembic_url(self) -> Self:
        if self.alembic_database_url is None:
            self.alembic_database_url = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql+psycopg://", 1
            )
        return self

    # ---------------- Neo4j ----------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("fundiq_dev_neo4j")

    # ---------------- Redis ----------------
    redis_url: str = "redis://localhost:6379/0"

    # ---------------- Clerk auth ----------------
    clerk_publishable_key: str | None = None
    clerk_secret_key: SecretStr | None = None
    clerk_jwks_url: str | None = None  # e.g. https://<your-app>.clerk.accounts.dev/.well-known/jwks.json

    # ---------------- Hatchet ----------------
    hatchet_client_token: SecretStr | None = None

    # ---------------- Rate limiting ----------------
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 20

    # ---------------- Convenience flags ----------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_test(self) -> bool:
        return self.environment is Environment.TEST

    @model_validator(mode="after")
    def _require_secrets_in_production(self) -> Self:
        """Production cannot boot without auth wired up.

        Logfire is intentionally NOT required — app/main.py already
        guards its instrumentation behind `if logfire_token`, so the
        app degrades cleanly to structlog-only logging. Forcing a
        Logfire token would block free-tier deploys that don't pay
        for observability.
        """
        if self.environment is Environment.PRODUCTION:
            missing: list[str] = []
            if self.clerk_secret_key is None:
                missing.append("CLERK_SECRET_KEY")
            if self.clerk_jwks_url is None:
                missing.append("CLERK_JWKS_URL")
            if missing:
                msg = f"Missing required production secrets: {', '.join(missing)}"
                raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Tests override via `get_settings.cache_clear()`."""
    return Settings()  # type: ignore[call-arg]  # populated from env

"""Unit tests for `app.core.config.Settings`.

We pass `_env_file=None` to bypass real `.env` loading — otherwise the
project's `.env` shadows the test's env vars (monkeypatch can't unset
values that live in a file).
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Environment, Settings


def _make(**env: str) -> Settings:
    """Construct Settings directly from env vars, skipping any .env file."""
    base = {"OPENAI_API_KEY": "sk-test", "DATABASE_URL": "postgresql+asyncpg://u:p@h/db"}
    base.update({k.upper(): v for k, v in env.items()})
    return Settings(_env_file=None, **{k.lower(): v for k, v in base.items()})  # type: ignore[arg-type]


@pytest.mark.unit
def test_empty_string_optional_fields_become_none() -> None:
    """Empty `FOO=` in .env must coerce to None for BOTH SecretStr and plain str.

    Plain-string fields like CLERK_JWKS_URL caused noisy startup warnings
    when treated as truthy empty strings (httpx tried to GET "").
    """
    s = _make(
        # SecretStr optionals
        LOGFIRE_TOKEN="",
        LANGSMITH_API_KEY="   ",  # whitespace-only counts as empty
        CLERK_SECRET_KEY="",
        # Plain str | None optionals
        CLERK_JWKS_URL="",
        CLERK_PUBLISHABLE_KEY="",
    )
    assert s.logfire_token is None
    assert s.langsmith_api_key is None
    assert s.clerk_secret_key is None
    assert s.clerk_jwks_url is None
    assert s.clerk_publishable_key is None
    assert isinstance(s.openai_api_key, SecretStr)
    assert s.openai_api_key.get_secret_value() == "sk-test"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("postgres://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgresql+asyncpg://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
    ],
)
def test_database_url_normalizes_to_asyncpg(inp: str, expected: str) -> None:
    s = _make(DATABASE_URL=inp)
    assert s.database_url == expected


@pytest.mark.unit
def test_alembic_url_derived_from_database_url() -> None:
    s = _make(DATABASE_URL="postgresql://u:p@h/db")
    assert s.alembic_database_url == "postgresql+psycopg://u:p@h/db"


@pytest.mark.unit
def test_invalid_database_scheme_rejected() -> None:
    with pytest.raises(ValidationError, match="postgresql"):
        _make(DATABASE_URL="mysql://u:p@h/db")


@pytest.mark.unit
def test_production_requires_secrets() -> None:
    with pytest.raises(ValidationError, match="CLERK_SECRET_KEY"):
        _make(ENVIRONMENT="production")


@pytest.mark.unit
def test_cors_origins_parsed_as_list() -> None:
    s = _make(CORS_ORIGINS="http://a.com, http://b.com,  http://c.com")
    assert s.cors_origin_list == ["http://a.com", "http://b.com", "http://c.com"]


@pytest.mark.unit
def test_is_test_flag() -> None:
    s = _make(ENVIRONMENT="test")
    assert s.is_test is True
    assert s.environment is Environment.TEST

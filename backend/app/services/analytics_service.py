"""DuckDB-backed analytics service.

We attach DuckDB to our Postgres database read-only via the
`postgres_scanner` extension and run columnar analytical queries
in-process. This gives us:
  - DuckDB's vectorised execution + columnar perf without a separate ETL
  - Single source of truth — no Parquet snapshots to keep in sync
  - Zero new infra; the DuckDB binary ships inside the Python package

When the corpus crosses tens of thousands of rows, we may want a daily
Parquet snapshot for further speedups (Postgres scanner still has to
fetch row data over the wire). For now this is the right level of
abstraction.

Concurrency note:
  DuckDB connections are not thread-safe. We open one connection per
  query, wrapped in `asyncio.to_thread` so the FastAPI event loop stays
  responsive while DuckDB executes synchronously.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import duckdb

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class FundingAnalytics:
    total_grants: int
    embedded_grants: int
    by_portal: list[dict[str, Any]]
    by_status: list[dict[str, Any]]
    by_federal_state: list[dict[str, Any]]
    funding_global_min: float | None
    funding_global_max: float | None
    funding_global_avg: float | None


# ---------------------------------------------------------------------------
# Postgres DSN translation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _postgres_dsn_for_duckdb() -> str:
    """DuckDB's postgres_scanner wants a libpq-style DSN string.

    Settings.alembic_database_url is sync `postgresql+psycopg://...`. We
    strip the SA driver prefix to get a clean DSN.
    """
    settings = get_settings()
    src = settings.alembic_database_url or settings.database_url
    # Strip any SA driver suffix.
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://"):
        if src.startswith(prefix):
            return "postgresql://" + src[len(prefix) :]
    return src


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
_FUNDING_QUERY = """
WITH base AS (
    SELECT * FROM pg.public.grants WHERE deleted_at IS NULL
),
totals AS (
    SELECT
        COUNT(*) AS total_grants,
        COUNT(embedding) AS embedded_grants,
        MIN(funding_max_eur)::DOUBLE AS funding_global_min,
        MAX(funding_max_eur)::DOUBLE AS funding_global_max,
        AVG(funding_max_eur)::DOUBLE AS funding_global_avg
    FROM base
),
by_portal AS (
    SELECT
        portal::TEXT AS portal,
        COUNT(*) AS n,
        COUNT(funding_max_eur) AS n_with_funding_max,
        MIN(funding_max_eur)::DOUBLE AS funding_min,
        MAX(funding_max_eur)::DOUBLE AS funding_max,
        AVG(funding_max_eur)::DOUBLE AS funding_avg
    FROM base
    GROUP BY portal
),
by_status AS (
    SELECT status::TEXT AS status, COUNT(*) AS n
    FROM base
    GROUP BY status
),
by_state AS (
    SELECT federal_state, COUNT(*) AS n
    FROM base
    WHERE federal_state IS NOT NULL
    GROUP BY federal_state
)
SELECT
    (SELECT to_json(t) FROM totals t)                                    AS totals,
    (SELECT json_group_array(to_json(p)) FROM by_portal p)               AS by_portal,
    (SELECT json_group_array(to_json(s)) FROM by_status s)               AS by_status,
    (SELECT json_group_array(to_json(fs)) FROM by_state fs)              AS by_federal_state
"""


def _execute_funding_analytics_sync() -> FundingAnalytics:
    """Sync DuckDB query — wrap in `asyncio.to_thread` from async callers."""
    import json

    dsn = _postgres_dsn_for_duckdb()
    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL postgres_scanner")
        con.execute("LOAD postgres_scanner")
        con.execute(f"ATTACH '{dsn}' AS pg (TYPE POSTGRES, READ_ONLY)")
        row = con.execute(_FUNDING_QUERY).fetchone()
    finally:
        con.close()

    if row is None:
        return FundingAnalytics(
            total_grants=0,
            embedded_grants=0,
            by_portal=[],
            by_status=[],
            by_federal_state=[],
            funding_global_min=None,
            funding_global_max=None,
            funding_global_avg=None,
        )

    totals = json.loads(row[0]) if row[0] else {}
    by_portal_raw = json.loads(row[1]) if row[1] else []
    by_status_raw = json.loads(row[2]) if row[2] else []
    by_state_raw = json.loads(row[3]) if row[3] else []

    def _str_to_dict_list(items: list[Any]) -> list[dict[str, Any]]:
        # DuckDB's json_group_array(to_json(t)) returns a list of JSON strings.
        return [json.loads(item) if isinstance(item, str) else item for item in items]

    return FundingAnalytics(
        total_grants=int(totals.get("total_grants", 0)),
        embedded_grants=int(totals.get("embedded_grants", 0)),
        by_portal=_str_to_dict_list(by_portal_raw),
        by_status=_str_to_dict_list(by_status_raw),
        by_federal_state=_str_to_dict_list(by_state_raw),
        funding_global_min=_opt_float(totals.get("funding_global_min")),
        funding_global_max=_opt_float(totals.get("funding_global_max")),
        funding_global_avg=_opt_float(totals.get("funding_global_avg")),
    )


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def compute_funding_analytics() -> FundingAnalytics:
    """Async entrypoint — runs the sync DuckDB query on a worker thread."""
    return await asyncio.to_thread(_execute_funding_analytics_sync)

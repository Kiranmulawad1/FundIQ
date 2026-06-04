"""Sparse retrieval leg — pg_trgm fuzzy keyword search.

Why pg_trgm and not BM25:
  - Postgres has no native BM25 (would need an extension like ParadeDB).
  - Trigram similarity is built-in, supported by the GIN index we already
    created in migration 6f72ff401213, and works well on German + English
    text out of the box.
  - For the thesis, we ship trigram as the baseline sparse leg and treat
    BM25 as a Phase 5C experiment (sparse-only A/B). The fusion logic
    (RRF) is unchanged regardless of which sparse method we use.

Query semantics:
  - We score each grant by the MAX of trigram similarity over its title
    and summary. Body is excluded because it's long and dilutes the
    signal — title + summary carry the discriminative tokens.
  - We use `similarity()` (returns [0,1]) and order by it.
  - Returned rows are not yet ranked by RRF — that's the fusion layer's job.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.base import GrantPortal

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SparseHit:
    """Sparse-leg result. `score` is trigram similarity in [0, 1]."""

    grant_id: uuid.UUID
    score: float


async def sparse_search(
    session: AsyncSession,
    *,
    query: str,
    limit: int = 50,
    portal: GrantPortal | None = None,
    country: str | None = None,
) -> list[SparseHit]:
    """Return up to `limit` grants ranked by pg_trgm similarity.

    The `%` operator + GIN index gives us index-backed candidate
    selection; `similarity()` gives us the score for ranking. We use
    GREATEST(title_sim, summary_sim) so a strong hit on either field
    is sufficient to surface.
    """
    if not query.strip():
        return []

    conditions = ["deleted_at IS NULL"]
    params: dict[str, object] = {"q": query, "limit": limit}

    if portal is not None:
        conditions.append("portal = :portal")
        params["portal"] = portal.value.upper()
    if country is not None:
        conditions.append("country = :country")
        params["country"] = country.upper()

    # The trigram `%` operator is what the GIN index serves. asyncpg uses
    # `$N` parameter style so the `%` doesn't need pyformat-style escape.
    # If we ever swap the driver back to psycopg pyformat, this string must
    # change to `%%` here.
    sql = text(
        f"""
        SELECT
            id,
            GREATEST(similarity(title, :q), similarity(summary, :q)) AS score
        FROM grants
        WHERE {' AND '.join(conditions)}
          AND (title % :q OR summary % :q)
        ORDER BY score DESC
        LIMIT :limit
        """
    )

    rows = (await session.execute(sql, params)).mappings().all()
    return [SparseHit(grant_id=r["id"], score=float(r["score"])) for r in rows]

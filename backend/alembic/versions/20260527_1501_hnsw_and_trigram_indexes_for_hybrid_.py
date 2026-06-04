"""hnsw and trigram indexes for hybrid retrieval

Revision ID: 6f72ff401213
Revises: 550747571357
Create Date: 2026-05-27 15:01:31.611829+00:00

Why this is a hand-written migration:
  Alembic's autogenerate inspects SQLAlchemy column metadata, but neither
  pgvector HNSW indexes nor pg_trgm GIN indexes are expressible via the
  ORM-level `Index(...)` API — they need explicit `USING hnsw (...)` /
  `USING gin (...)` clauses with operator-class arguments. So we emit the
  raw SQL here.

What this migration adds:
  1. HNSW index on `grants.embedding` with cosine ops.
     Parameters per ADR-001: m=16, ef_construction=64 (pgvector defaults).
     Justification: HNSW outperforms IVFFlat on read-heavy workloads at our
     expected scale (~10k–100k grants) and doesn't require a pre-trained
     centroid table.

  2. GIN trigram index on `grants.title`.
  3. GIN trigram index on `grants.summary`.
     Justification: the sparse leg of hybrid retrieval (Phase 5) ranks by
     trigram similarity. Without a GIN index this is a seq-scan per query.

What's deferred to Phase 5:
  - Tsvector full-text search with German + English dictionaries. Trigram
    is the baseline; FTS gets A/B-tested against it on the gold set.
  - HNSW index on startup embeddings (no startup embedding column yet).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6f72ff401213"
down_revision: str | Sequence[str] | None = "550747571357"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Dense leg: HNSW cosine index on 1024-dim multilingual-e5-large vectors.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_grants_embedding_hnsw
        ON grants
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # Sparse leg: trigram GIN indexes for fuzzy keyword search on title + summary.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_grants_title_trgm "
        "ON grants USING gin (title gin_trgm_ops);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_grants_summary_trgm "
        "ON grants USING gin (summary gin_trgm_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_grants_summary_trgm;")
    op.execute("DROP INDEX IF EXISTS ix_grants_title_trgm;")
    op.execute("DROP INDEX IF EXISTS ix_grants_embedding_hnsw;")

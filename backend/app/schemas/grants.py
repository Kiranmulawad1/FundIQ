"""API response schemas for /grants/* endpoints.

These are deliberately separate from the SQLModel `Grant` table model:
  - they never include the 1024-dim `embedding` vector (heavy + useless to
    the client; consumers query through /grants/search)
  - they normalise `Decimal` to `float` for JSON-safe transport
  - they expose only fields the public is allowed to see (no `source_hash`,
    no `deleted_at`)

Splitting list-view from detail-view keeps `GET /grants` responses
compact — the `body` field can be tens of kilobytes per grant.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.base import GrantPortal, GrantStatus, Sector


class GrantListItem(BaseModel):
    """Compact view for list responses — body and full eligibility omitted."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    portal: GrantPortal
    status: GrantStatus
    title: str
    title_en: str | None = None
    summary: str
    sector: Sector | None = None
    country: str
    federal_state: str | None = None
    funding_min_eur: float | None = None
    funding_max_eur: float | None = None
    deadline: datetime | None = None
    opens_at: datetime | None = None
    source_url: str
    source_doc_id: str | None = None
    created_at: datetime
    updated_at: datetime


class GrantDetail(GrantListItem):
    """Full view — adds body, eligibility, full metadata."""

    body: str
    summary_en: str | None = None
    eligibility: dict[str, Any] = Field(default_factory=dict)
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        alias="metadata",
    )


class PageMeta(BaseModel):
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    returned: int = Field(ge=0)


class GrantListResponse(BaseModel):
    items: list[GrantListItem]
    page: PageMeta


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
from app.rag.pipeline import RetrievalMode  # re-export so callers don't dive deep


class Citation(BaseModel):
    """Provenance pointer for one retrieved grant.

    For now, one citation per grant (no body chunking yet). Phase 5B will
    refine this to paragraph-level pointers needed for the Writer agent's
    claim-grounding.
    """

    grant_id: uuid.UUID
    source_doc_id: str | None = None
    source_url: str
    portal: GrantPortal
    title: str


class GrantSearchRequest(BaseModel):
    """Hybrid retrieval request.

    `mode` controls the ranking strategy:
      dense          — pgvector cosine only (baseline)
      hybrid         — dense + sparse trigram, fused via RRF
      hybrid_rerank  — hybrid + BGE cross-encoder reranker (highest quality)

    `use_hyde` is orthogonal to `mode`. When true, Gemini generates 3
    hypothetical grant descriptions and the mean-pool of their embeddings
    becomes the dense-leg query vector. Useful for vague queries.
    """

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    use_hyde: bool = Field(
        default=False,
        description="Apply HyDE query rewriting before dense retrieval.",
    )
    portal: GrantPortal | None = None
    country: str | None = Field(default=None, min_length=2, max_length=2)


class GrantSearchHit(GrantListItem):
    final_score: float = Field(
        description=(
            "Score on whichever scale `mode` produced. "
            "Cosine in DENSE; RRF score in HYBRID; reranker logit in HYBRID_RERANK. "
            "Always: higher is more relevant. Compare ranks across hits, not magnitudes across modes."
        ),
    )
    dense_rank: int | None = Field(default=None, description="Rank in dense leg, 0-indexed.")
    sparse_rank: int | None = Field(default=None, description="Rank in sparse leg, 0-indexed.")
    rrf_score: float | None = None
    rerank_score: float | None = None
    citation: Citation


class GrantSearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[GrantSearchHit]
    elapsed_ms: int
    # Provenance counts — useful for the eval harness + admin dashboards.
    dense_count: int
    sparse_count: int
    rrf_input_count: int
    rerank_input_count: int
    # Phase 5B provenance.
    used_hyde: bool = False
    hypotheticals: list[str] | None = None
    cache_hit: bool = False
    cached_for_query: str | None = None

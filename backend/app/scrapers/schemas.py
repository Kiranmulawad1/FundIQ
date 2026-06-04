"""Output schema for scrapers.

`ScrapedGrant` is what every scraper produces. It deliberately mirrors
`Grant` (the DB model) but stays Pydantic-only — no SQLModel coupling.
That lets us:
  - hand a ScrapedGrant to the ETL layer without dragging in a DB session
  - validate scraper output in isolation (unit tests against fixtures)
  - serialize for queueing (Hatchet workers, in Phase 2C)

The content hash is the dedup signal — ETL upserts only re-embed when the
hash changes, which keeps HNSW maintenance and embedding costs sane.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.base import GrantPortal, GrantStatus, Sector


class ScrapedGrant(BaseModel):
    """Raw output from a scraper, before normalization + DB upsert."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    portal: GrantPortal
    source_url: str = Field(min_length=10, max_length=1000)
    source_doc_id: str | None = Field(default=None, max_length=255)

    title: str = Field(min_length=1, max_length=500)
    title_en: str | None = Field(default=None, max_length=500)
    summary: str = Field(min_length=1)
    summary_en: str | None = None
    body: str = Field(min_length=1)

    status: GrantStatus = GrantStatus.OPEN
    sector: Sector | None = None
    country: str = Field(default="DE", min_length=2, max_length=2)
    federal_state: str | None = Field(default=None, max_length=64)

    funding_min_eur: Decimal | None = Field(default=None, ge=0)
    funding_max_eur: Decimal | None = Field(default=None, ge=0)

    deadline: datetime | None = None
    opens_at: datetime | None = None

    eligibility: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash of the content fields that drive re-embedding.

        Excludes status/deadline/funding amounts on purpose — those change
        often without changing the semantic content, and we don't want to
        re-embed every time a deadline rolls forward.
        """
        material = "|".join(
            [
                self.title,
                self.summary,
                self.body,
                self.sector.value if self.sector else "",
                self.country,
                self.federal_state or "",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def embedding_text(self) -> str:
        """The exact string fed to the embedding model.

        Centralised here so the ETL, the eval harness, and the alert
        worker all embed grants the same way. Order matters: title first
        (most informative tokens), then summary, then body — early tokens
        get higher attention weight.
        """
        return f"{self.title}\n\n{self.summary}\n\n{self.body}"

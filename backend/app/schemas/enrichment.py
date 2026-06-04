"""LLM grant-enrichment output schema.

Scrapers ship structured fields where the source provides them, but most
portal pages bury sector / eligibility / target groups / funding form
inside prose body text. This schema is what an LLM extracts from that
body so the Planner can pre-filter, the Scorer can judge against typed
criteria, and the Writer can cite specific eligibility points instead of
re-parsing the body every turn.

The enrichment writes back into existing Grant columns + a structured
`eligibility` JSONB sub-document. We never overwrite fields the scraper
populated — only fill the ones it left empty. Re-running is idempotent
via the `enrichment_version` stamp.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.base import Sector

# Bump when the prompt/schema changes incompatibly. A grant whose stored
# `enrichment_version` is lower than this gets re-enriched on bulk runs.
CURRENT_ENRICHMENT_VERSION = 1


FundingForm = Literal["grant", "loan", "equity", "stipend", "mixed", "other"]


class GrantEnrichment(BaseModel):
    """The structured fields an LLM derives from a Grant's body text."""

    # Sector classification — primary plus secondary. The Planner uses the
    # primary one for filtering; the Scorer can also reason over the
    # secondary list when a grant supports multiple verticals.
    sector: Sector | None = Field(
        default=None,
        description="Best-guess primary sector. Null if the grant is sector-agnostic.",
    )
    secondary_sectors: list[Sector] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 additional sectors the grant explicitly supports.",
    )

    # Geography — fill only when clearly regional. Federal-level grants
    # leave this null so the existing `federal_state` column doesn't get
    # over-attributed.
    federal_state: str | None = Field(
        default=None,
        max_length=64,
        description="German Land if the grant is regional, else null.",
    )

    # Who can apply
    target_groups: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="1-6 short labels, e.g. 'academic founders', 'SMEs', 'female founders'.",
    )

    # What they need to demonstrate
    eligibility_criteria: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="1-8 specific requirements (one sentence each).",
    )

    # Stage support
    funding_phases: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="e.g. ['pre-seed', 'seed'] or ['idea phase', 'growth'].",
    )

    funding_form: FundingForm = Field(
        default="other",
        description="Primary instrument: grant / loan / equity / stipend / mixed / other.",
    )

    application_notes: str = Field(
        default="",
        max_length=600,
        description=(
            "Caveats about deadlines, cofinancing requirements, application "
            "rhythm, or other practical points the user should know up front."
        ),
    )

    def to_eligibility_dict(self) -> dict[str, Any]:
        """The JSONB blob written into `Grant.eligibility`.

        Includes `enrichment_version` + `enriched_at` so future re-runs
        can identify which grants need re-processing without ad-hoc
        bookkeeping.
        """
        return {
            "target_groups": self.target_groups,
            "criteria": self.eligibility_criteria,
            "funding_phases": self.funding_phases,
            "funding_form": self.funding_form,
            "secondary_sectors": [s.value for s in self.secondary_sectors],
            "application_notes": self.application_notes,
            "enrichment_version": CURRENT_ENRICHMENT_VERSION,
            "enriched_at": datetime.now(UTC).isoformat(),
        }

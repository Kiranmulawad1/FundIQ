"""Analytics response schemas — typed shapes for the funding dashboard."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PortalCount(BaseModel):
    portal: str
    n: int = Field(ge=0)
    n_with_funding_max: int = Field(ge=0)
    funding_min: float | None = None
    funding_max: float | None = None
    funding_avg: float | None = None


class StatusCount(BaseModel):
    status: str
    n: int = Field(ge=0)


class FederalStateCount(BaseModel):
    federal_state: str
    n: int = Field(ge=0)


class FundingAnalyticsResponse(BaseModel):
    total_grants: int = Field(ge=0)
    embedded_grants: int = Field(
        ge=0,
        description="Subset of grants with a populated embedding vector.",
    )
    by_portal: list[PortalCount]
    by_status: list[StatusCount]
    by_federal_state: list[FederalStateCount]
    funding_global_min: float | None = None
    funding_global_max: float | None = None
    funding_global_avg: float | None = None
    computed_via: str = Field(default="duckdb+postgres_scanner")
    elapsed_ms: int = Field(ge=0)

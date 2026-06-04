"""Public funding-market analytics endpoint.

Backed by DuckDB attached to Postgres via `postgres_scanner`. At 26 grants
this is overkill; the architecture pays off once we cross thousands of
rows (Phase 2D when foerderdatenbank.de + BMFTR unlock).
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.analytics import (
    FederalStateCount,
    FundingAnalyticsResponse,
    PortalCount,
    StatusCount,
)
from app.services.analytics_service import compute_funding_analytics

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get(
    "/funding",
    response_model=FundingAnalyticsResponse,
    summary="Funding-corpus analytics (DuckDB on Postgres)",
)
async def funding_analytics() -> FundingAnalyticsResponse:
    started = time.perf_counter()
    data = await compute_funding_analytics()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return FundingAnalyticsResponse(
        total_grants=data.total_grants,
        embedded_grants=data.embedded_grants,
        by_portal=[PortalCount(**p) for p in data.by_portal],
        by_status=[StatusCount(**s) for s in data.by_status],
        by_federal_state=[FederalStateCount(**fs) for fs in data.by_federal_state],
        funding_global_min=data.funding_global_min,
        funding_global_max=data.funding_global_max,
        funding_global_avg=data.funding_global_avg,
        elapsed_ms=elapsed_ms,
    )

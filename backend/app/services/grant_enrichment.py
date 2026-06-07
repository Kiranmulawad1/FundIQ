"""LLM enrichment of Grant rows.

Most scrapers ship a title + summary + body and leave the structured
columns (sector, eligibility, funding_form) empty because the upstream
pages encode that information in prose. This service runs each grant
through Gemini once to derive a typed `GrantEnrichment` and writes the
result back, so the agent graph stops re-extracting from raw text on
every request.

Two paths:
  - `enrich_grant(grant, llm)`: pure function — returns the enrichment.
  - `apply_enrichment(grant, enrichment)`: mutates the Grant row,
    respecting fields the scraper already populated (never overwrite
    sector or federal_state if they're already set).

`bulk_enrich(session, llm)`: iterates grants and orchestrates both. By
default it skips grants whose `eligibility` already carries the current
`enrichment_version` — idempotent so a partial run can resume cleanly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.agents.llm import AgentLLMError
from app.core.logging import get_logger
from app.core.prompts import PromptFetchError, get_prompt
from app.models import Grant
from app.schemas.enrichment import CURRENT_ENRICHMENT_VERSION, GrantEnrichment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.agents.llm import GeminiAgentClient


logger = get_logger(__name__)

# Cap how much body text we ship to Gemini. The first ~3500 chars
# almost always carry the eligibility section; anything more inflates
# token cost without changing the extraction.
BODY_EXCERPT_CHARS = 3500


# Prompt lives in Langfuse under name "enrichment".


@dataclass(slots=True)
class EnrichResult:
    grant_id: str
    status: str          # "enriched" | "skipped" | "failed"
    detail: str = ""


async def enrich_grant(
    grant: Grant,
    *,
    llm: GeminiAgentClient,
) -> GrantEnrichment:
    """Call Gemini for a single grant. Raises AgentLLMError on transport
    or schema failure — callers decide whether to swallow or propagate.
    """
    body = (grant.body or "").strip()
    compiled = get_prompt("enrichment").compile(
        title=grant.title,
        summary=grant.summary or "",
        body_excerpt=body[:BODY_EXCERPT_CHARS],
        excerpt_chars=BODY_EXCERPT_CHARS,
    )
    return await llm.respond_as(
        GrantEnrichment,
        prompt=compiled.text,
        prompt_handle=compiled.langfuse_handle,
        temperature=0.2,
        # 8 lists × ~5 items × ~50 chars ≈ 2000 chars of payload; 4096
        # tokens leaves comfortable headroom for application_notes.
        max_output_tokens=4096,
    )


def apply_enrichment(grant: Grant, enrichment: GrantEnrichment) -> None:
    """Write the enrichment back into the Grant row in-place.

    Sector and federal_state get populated only when currently null —
    never overwrite a value the scraper provided authoritatively.
    Eligibility is replaced wholesale because we're the only producer
    of that JSONB shape today.
    """
    if grant.sector is None and enrichment.sector is not None:
        grant.sector = enrichment.sector
    if grant.federal_state is None and enrichment.federal_state:
        grant.federal_state = enrichment.federal_state
    grant.eligibility = enrichment.to_eligibility_dict()


def _is_already_enriched(grant: Grant) -> bool:
    """Idempotency check: did we run at the current version already?"""
    elig = grant.eligibility or {}
    version = elig.get("enrichment_version")
    return isinstance(version, int) and version >= CURRENT_ENRICHMENT_VERSION


async def bulk_enrich(
    session: AsyncSession,
    *,
    llm: GeminiAgentClient,
    force: bool = False,
    limit: int | None = None,
    per_call_delay_seconds: float = 7.0,
) -> list[EnrichResult]:
    """Enrich every active grant. Idempotent by default — pass force=True
    to re-run grants already at CURRENT_ENRICHMENT_VERSION.

    `per_call_delay_seconds` paces the calls to stay under Gemini's free
    tier RPM (10/min for gemini-2.5-flash). The first call doesn't wait;
    every subsequent call sleeps before firing.
    """
    stmt = select(Grant).where(Grant.deleted_at.is_(None))  # type: ignore[attr-defined]
    if limit is not None:
        stmt = stmt.limit(limit)
    grants = (await session.execute(stmt)).scalars().all()

    results: list[EnrichResult] = []
    needs_delay = False
    for grant in grants:
        gid = str(grant.id)
        if not force and _is_already_enriched(grant):
            results.append(EnrichResult(grant_id=gid, status="skipped", detail="already current"))
            continue
        if needs_delay and per_call_delay_seconds > 0:
            await asyncio.sleep(per_call_delay_seconds)
        needs_delay = True
        try:
            enrichment = await enrich_grant(grant, llm=llm)
        except (AgentLLMError, PromptFetchError) as e:
            logger.warning("enrich.failed", grant_id=gid, error=str(e)[:200])
            results.append(EnrichResult(grant_id=gid, status="failed", detail=str(e)[:300]))
            continue
        apply_enrichment(grant, enrichment)
        results.append(EnrichResult(grant_id=gid, status="enriched"))

    await session.commit()
    logger.info(
        "enrich.bulk_done",
        total=len(grants),
        enriched=sum(1 for r in results if r.status == "enriched"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        failed=sum(1 for r in results if r.status == "failed"),
    )
    return results

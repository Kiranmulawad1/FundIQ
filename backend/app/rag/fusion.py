"""Reciprocal Rank Fusion — combine multiple ranked lists into one.

Why RRF and not weighted score fusion:
  Dense (pgvector cosine) and sparse (pg_trgm similarity) scores live on
  fundamentally different scales. Cosine similarity is roughly Gaussian
  around 0.7-0.85 for our corpus; trigram similarity for matching tokens
  is typically 0.2-0.5. A naïve `0.5 * dense + 0.5 * sparse` would either
  hide every sparse-only hit or be drowned by a single high-cosine
  outlier. RRF only uses the *rank* in each list, so the scoring scales
  cancel out.

Standard k=60 is from the original RRF paper (Cormack et al. 2009).
Lower k makes top ranks dominate more aggressively.

Returns a list ordered by combined RRF score, descending.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(slots=True, frozen=True)
class FusedHit:
    grant_id: uuid.UUID
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


def reciprocal_rank_fusion(
    dense_ranking: Iterable[uuid.UUID],
    sparse_ranking: Iterable[uuid.UUID],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Merge two ranked grant-id lists by RRF.

    Inputs are already-ranked sequences (best-first). Ties are broken by
    insertion order, which matches dense-first behaviour when both legs
    return the same id at the same rank.
    """
    dense_ranks: dict[uuid.UUID, int] = {}
    for rank, gid in enumerate(dense_ranking):
        if gid not in dense_ranks:
            dense_ranks[gid] = rank

    sparse_ranks: dict[uuid.UUID, int] = {}
    for rank, gid in enumerate(sparse_ranking):
        if gid not in sparse_ranks:
            sparse_ranks[gid] = rank

    all_ids = set(dense_ranks) | set(sparse_ranks)
    fused: list[FusedHit] = []
    for gid in all_ids:
        score = 0.0
        d_rank = dense_ranks.get(gid)
        s_rank = sparse_ranks.get(gid)
        if d_rank is not None:
            score += 1.0 / (k + d_rank + 1)
        if s_rank is not None:
            score += 1.0 / (k + s_rank + 1)
        fused.append(
            FusedHit(grant_id=gid, rrf_score=score, dense_rank=d_rank, sparse_rank=s_rank)
        )

    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    return fused

"""Retrieval metrics — precision@K, recall@K, nDCG@K, MRR.

Pure functions over a graded relevance map:

    gains[doc_id] = 2   # ideal match
    gains[doc_id] = 1   # acceptable / relevant
    gains[doc_id] = 0   # not relevant (or absent from dict)

`precision@K` and `recall@K` treat any positive gain as a hit (lenient).
`nDCG@K` uses the graded values directly. MRR is reported in two flavours:
strict (ideal-only, gain >= 2) and lenient (any positive gain).
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def precision_at_k(ranked: list[str], gains: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if gains.get(d, 0) > 0)
    return hits / k


def recall_at_k(ranked: list[str], gains: Mapping[str, int], k: int) -> float:
    total_positive = sum(1 for v in gains.values() if v > 0)
    if total_positive == 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if gains.get(d, 0) > 0)
    return hits / total_positive


def dcg_at_k(ranked: list[str], gains: Mapping[str, int], k: int) -> float:
    return sum(
        gains.get(d, 0) / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
    )


def ndcg_at_k(ranked: list[str], gains: Mapping[str, int], k: int) -> float:
    ideal_order = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_order))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, gains, k) / idcg


def mrr(ranked: list[str], gains: Mapping[str, int], *, strict: bool = False) -> float:
    """Reciprocal rank of the first positive hit.

    `strict=True` only counts gain >= 2 (ideal); otherwise any positive gain.
    Returns 0.0 when no hit is found in `ranked`.
    """
    threshold = 2 if strict else 1
    for i, d in enumerate(ranked):
        if gains.get(d, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def build_gains(ideal: list[str], relevant: list[str]) -> dict[str, int]:
    """Merge ideal + relevant lists into a graded-gain map.

    `ideal` wins ties — if a doc appears in both lists it's scored as ideal.
    """
    gains: dict[str, int] = {d: 1 for d in relevant}
    for d in ideal:
        gains[d] = 2
    return gains

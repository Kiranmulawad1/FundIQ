"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import uuid

import pytest

from app.rag.fusion import DEFAULT_RRF_K, FusedHit, reciprocal_rank_fusion


def _uuids(n: int) -> list[uuid.UUID]:
    return [uuid.UUID(int=i + 1) for i in range(n)]


@pytest.mark.unit
def test_rrf_single_list_orders_by_rank() -> None:
    ids = _uuids(3)
    fused = reciprocal_rank_fusion(ids, [])
    assert [h.grant_id for h in fused] == ids
    # All sparse ranks are None when sparse list is empty.
    assert all(h.sparse_rank is None for h in fused)


@pytest.mark.unit
def test_rrf_top_of_both_lists_beats_top_of_one() -> None:
    a, b, c = _uuids(3)
    # `a` is #1 dense; `b` is #1 sparse. They tie. `c` is #2 in both.
    fused = reciprocal_rank_fusion([a, c], [b, c])
    by_id = {h.grant_id: h.rrf_score for h in fused}
    # c appears at rank 1 in both → its score should beat a (only #0 dense)
    # and b (only #0 sparse) — combined RRF wins over single-leg top hits.
    assert by_id[c] > by_id[a]
    assert by_id[c] > by_id[b]


@pytest.mark.unit
def test_rrf_score_uses_k_plus_rank_plus_one() -> None:
    """`score = 1 / (k + rank + 1)` per the standard formula. Rank is 0-indexed."""
    (a,) = _uuids(1)
    fused = reciprocal_rank_fusion([a], [])
    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1.0 / (DEFAULT_RRF_K + 0 + 1))


@pytest.mark.unit
def test_rrf_dedups_duplicate_ids_within_one_ranking() -> None:
    a, b = _uuids(2)
    # Dense list contains `a` twice — only its first appearance should count.
    fused = reciprocal_rank_fusion([a, a, b], [])
    ids = [h.grant_id for h in fused]
    assert ids.count(a) == 1
    assert ids.count(b) == 1


@pytest.mark.unit
def test_rrf_records_provenance_ranks() -> None:
    a, b = _uuids(2)
    fused = reciprocal_rank_fusion([a, b], [b])
    by_id = {h.grant_id: h for h in fused}
    assert by_id[a].dense_rank == 0
    assert by_id[a].sparse_rank is None
    assert by_id[b].dense_rank == 1
    assert by_id[b].sparse_rank == 0


@pytest.mark.unit
def test_rrf_k_parameter_changes_top_concentration() -> None:
    """Lower k makes the top rank dominate — useful intuition guard."""
    (a, b) = _uuids(2)
    # With k=0: rank 0 → score=1.0, rank 1 → score=0.5.
    low_k = {h.grant_id: h.rrf_score for h in reciprocal_rank_fusion([a, b], [], k=0)}
    # With k=1000: rank 0 → score≈1/1001, rank 1 → score≈1/1002 — much flatter.
    high_k = {h.grant_id: h.rrf_score for h in reciprocal_rank_fusion([a, b], [], k=1000)}

    low_ratio = low_k[a] / low_k[b]
    high_ratio = high_k[a] / high_k[b]
    assert low_ratio > high_ratio  # top-rank dominance shrinks as k grows


@pytest.mark.unit
def test_rrf_returned_type_is_fused_hit() -> None:
    a, = _uuids(1)
    fused = reciprocal_rank_fusion([a], [a])
    assert isinstance(fused[0], FusedHit)
    assert fused[0].grant_id == a

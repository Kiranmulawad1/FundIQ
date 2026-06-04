"""Unit tests for ml/evals/metrics.py.

The metrics drive the thesis numbers — they need to be obviously correct,
not just "passing on toy data". Each test pins down a specific property
(top-K boundary, graded vs lenient, MRR strict mode, etc.) so a regression
points straight at the cause.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ml/ is a sibling of backend/, not on sys.path by default. Plumb it in.
_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from evals.metrics import (  # noqa: E402  — sys.path mutation above
    build_gains,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@pytest.fixture
def gains() -> dict[str, int]:
    # a, b ideal; c, d relevant; e, f, g unrelated.
    return build_gains(ideal=["a", "b"], relevant=["c", "d"])


def test_build_gains_ideal_wins_ties() -> None:
    g = build_gains(ideal=["a"], relevant=["a", "b"])
    assert g["a"] == 2
    assert g["b"] == 1


def test_precision_at_k_lenient_counts_relevant_too(gains: dict[str, int]) -> None:
    # top-5 = a, c, x, y, b → 3 positive hits / 5.
    p = precision_at_k(["a", "c", "x", "y", "b"], gains, k=5)
    assert p == pytest.approx(3 / 5)


def test_precision_at_k_zero_when_no_hits(gains: dict[str, int]) -> None:
    assert precision_at_k(["x", "y", "z"], gains, k=3) == 0.0


def test_recall_at_k_uses_total_positive_as_denominator(gains: dict[str, int]) -> None:
    # gains has 4 positive entries (a, b, c, d). Top-3 catches 2.
    assert recall_at_k(["a", "c", "x"], gains, k=3) == pytest.approx(2 / 4)


def test_recall_at_k_caps_at_one(gains: dict[str, int]) -> None:
    # Top-10 catches all 4 positives.
    r = recall_at_k(["a", "b", "c", "d", "x", "y", "z", "w", "v", "u"], gains, k=10)
    assert r == pytest.approx(1.0)


def test_dcg_uses_graded_gains(gains: dict[str, int]) -> None:
    # Position 0 → gain/log2(2) = gain/1.
    # Position 1 → gain/log2(3).
    # rank: a (2), c (1), x (0) → 2/log2(2) + 1/log2(3) + 0
    expected = 2 / math.log2(2) + 1 / math.log2(3)
    assert dcg_at_k(["a", "c", "x"], gains, k=3) == pytest.approx(expected)


def test_ndcg_is_one_when_perfectly_ranked(gains: dict[str, int]) -> None:
    # Ideal order by gain: a(2), b(2), c(1), d(1).
    assert ndcg_at_k(["a", "b", "c", "d"], gains, k=4) == pytest.approx(1.0)


def test_ndcg_penalises_misranking(gains: dict[str, int]) -> None:
    # Same items, worst plausible order — but only on positions 0..3.
    worse = ndcg_at_k(["d", "c", "b", "a"], gains, k=4)
    best = ndcg_at_k(["a", "b", "c", "d"], gains, k=4)
    assert 0 < worse < best


def test_ndcg_returns_zero_when_no_positives() -> None:
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_mrr_lenient_treats_relevant_as_hit(gains: dict[str, int]) -> None:
    # First positive is at index 1 ("c", relevant) → 1/2.
    assert mrr(["x", "c", "a"], gains, strict=False) == pytest.approx(0.5)


def test_mrr_strict_only_counts_ideal(gains: dict[str, int]) -> None:
    # Same input, strict: "c" is relevant not ideal — first ideal is "a" at index 2.
    assert mrr(["x", "c", "a"], gains, strict=True) == pytest.approx(1 / 3)


def test_mrr_zero_when_no_hit(gains: dict[str, int]) -> None:
    assert mrr(["x", "y", "z"], gains, strict=False) == 0.0


def test_precision_handles_zero_k(gains: dict[str, int]) -> None:
    assert precision_at_k(["a"], gains, k=0) == 0.0

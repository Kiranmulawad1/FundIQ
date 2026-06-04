"""Phase 5C eval harness — sweep retrieval configurations against the gold set.

Runs every (query, mode, hyde) cell directly against `RetrievalPipeline` —
no FastAPI lifespan, no HTTP. Cache is intentionally bypassed (`cache=None`)
so each config gets a fresh ranking.

Output:
  ml/evals/results/runs/<ts>.jsonl    per-(query, config) rows with metrics
  ml/evals/results/<ts>_report.md     aggregated comparison table

Run from the repo root:
  uv run python -m evals.run_evals --gold ml/evals/gold_set.jsonl
  uv run python -m evals.run_evals --gold ml/evals/gold_set.jsonl --modes dense hybrid
  uv run python -m evals.run_evals --skip-hyde       # skip Gemini-dependent cells
  uv run python -m evals.run_evals --skip-rerank     # skip BGE-loading cells

The script lives under ml/evals/ so it ships with the dataset; we sys.path
the backend root in so `app.*` imports resolve without an editable install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Path plumbing — let `app.*` and `evals.*` both import cleanly.
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_ML_ROOT = _REPO_ROOT / "ml"
for p in (_BACKEND_ROOT, _ML_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Set a real-but-non-production env so config validation passes; this script
# is read-only against the live DB but should never act like prod.
os.environ.setdefault("ENVIRONMENT", "development")

from redis.asyncio import Redis  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import dispose_engine, get_sessionmaker, init_engine  # noqa: E402
from app.models import Grant  # noqa: E402
from app.rag.hyde import HyDEService  # noqa: E402
from app.rag.pipeline import RetrievalMode, RetrievalPipeline  # noqa: E402
from app.rag.reranker import RerankerService  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402

from evals.metrics import (  # noqa: E402
    build_gains,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class GoldQuery:
    qid: str
    lang: str
    query: str
    ideal: list[str]
    relevant: list[str]
    rationale: str

    @property
    def all_positive(self) -> set[str]:
        return set(self.ideal) | set(self.relevant)


@dataclass(slots=True, frozen=True)
class EvalConfig:
    mode: RetrievalMode
    use_hyde: bool

    @property
    def label(self) -> str:
        return f"{self.mode.value}{'+hyde' if self.use_hyde else ''}"


@dataclass(slots=True)
class QueryResult:
    qid: str
    config_label: str
    mode: str
    use_hyde: bool
    elapsed_ms: int
    ranked: list[str]  # source_doc_id order
    precision_5: float
    recall_10: float
    ndcg_10: float
    mrr_lenient: float
    mrr_strict: float


# -----------------------------------------------------------------------------
# Gold set
# -----------------------------------------------------------------------------
def load_gold_set(path: Path) -> list[GoldQuery]:
    queries: list[GoldQuery] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append(
                GoldQuery(
                    qid=obj["qid"],
                    lang=obj["lang"],
                    query=obj["query"],
                    ideal=list(obj["ideal"]),
                    relevant=list(obj.get("relevant", [])),
                    rationale=obj.get("rationale", ""),
                )
            )
    return queries


# -----------------------------------------------------------------------------
# Eval driver
# -----------------------------------------------------------------------------
TOP_K_FETCH = 10  # we always ask for 10 so recall@10 + nDCG@10 are computable.


async def _resolve_doc_id_map(session) -> tuple[dict, dict]:  # type: ignore[no-untyped-def]
    """Build bidirectional source_doc_id ↔ grant_id maps from the live corpus."""
    rows = (
        await session.execute(
            select(Grant.id, Grant.source_doc_id).where(
                Grant.deleted_at.is_(None),  # type: ignore[attr-defined]
                Grant.source_doc_id.is_not(None),  # type: ignore[attr-defined]
            )
        )
    ).all()
    id_to_doc = {r.id: r.source_doc_id for r in rows}
    doc_to_id = {r.source_doc_id: r.id for r in rows}
    return id_to_doc, doc_to_id


async def _run_one(
    *,
    pipeline: RetrievalPipeline,
    session,  # type: ignore[no-untyped-def]
    query: GoldQuery,
    config: EvalConfig,
    hyde_service: HyDEService | None,
    id_to_doc: dict,
) -> QueryResult:
    t0 = time.perf_counter()
    result = await pipeline.retrieve(
        session,
        query=query.query,
        mode=config.mode,
        limit=TOP_K_FETCH,
        use_hyde=config.use_hyde,
        hyde_service=hyde_service if config.use_hyde else None,
        cache=None,  # honesty: each config produces a fresh ranking.
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    ranked_doc_ids: list[str] = []
    for h in result.hits:
        doc = id_to_doc.get(h.grant.id)
        if doc is not None:
            ranked_doc_ids.append(doc)

    gains = build_gains(query.ideal, query.relevant)
    return QueryResult(
        qid=query.qid,
        config_label=config.label,
        mode=config.mode.value,
        use_hyde=config.use_hyde,
        elapsed_ms=elapsed_ms,
        ranked=ranked_doc_ids,
        precision_5=precision_at_k(ranked_doc_ids, gains, 5),
        recall_10=recall_at_k(ranked_doc_ids, gains, 10),
        ndcg_10=ndcg_at_k(ranked_doc_ids, gains, 10),
        mrr_lenient=mrr(ranked_doc_ids, gains, strict=False),
        mrr_strict=mrr(ranked_doc_ids, gains, strict=True),
    )


def _aggregate(rows: list[QueryResult]) -> dict[str, dict[str, float]]:
    """Group by config_label, average each metric."""
    by_cfg: dict[str, list[QueryResult]] = {}
    for r in rows:
        by_cfg.setdefault(r.config_label, []).append(r)

    agg: dict[str, dict[str, float]] = {}
    for label, items in by_cfg.items():
        agg[label] = {
            "n": float(len(items)),
            "precision@5": statistics.mean(r.precision_5 for r in items),
            "recall@10": statistics.mean(r.recall_10 for r in items),
            "nDCG@10": statistics.mean(r.ndcg_10 for r in items),
            "MRR_lenient": statistics.mean(r.mrr_lenient for r in items),
            "MRR_strict": statistics.mean(r.mrr_strict for r in items),
            "p50_ms": statistics.median(r.elapsed_ms for r in items),
            "mean_ms": statistics.mean(r.elapsed_ms for r in items),
        }
    return agg


def _write_report(
    *,
    report_path: Path,
    runs_path: Path,
    queries: list[GoldQuery],
    rows: list[QueryResult],
    agg: dict[str, dict[str, float]],
) -> None:
    """Render a human-readable markdown comparison table."""
    lines: list[str] = []
    lines.append("# RAG retrieval eval — Phase 5C")
    lines.append("")
    lines.append(f"- Gold queries: **{len(queries)}**")
    lines.append(f"- Configurations: **{len(agg)}**")
    lines.append(f"- Raw rows: `{runs_path.name}`")
    lines.append("")
    lines.append("## Aggregate metrics (mean across queries)")
    lines.append("")
    header = (
        "| Config | precision@5 | recall@10 | nDCG@10 | MRR (lenient) | "
        "MRR (strict) | p50 (ms) | mean (ms) |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.append(header)
    lines.append(sep)
    # Sort by nDCG desc so the table reads "best on top".
    for label, m in sorted(agg.items(), key=lambda kv: kv[1]["nDCG@10"], reverse=True):
        lines.append(
            f"| `{label}` | "
            f"{m['precision@5']:.3f} | {m['recall@10']:.3f} | {m['nDCG@10']:.3f} | "
            f"{m['MRR_lenient']:.3f} | {m['MRR_strict']:.3f} | "
            f"{m['p50_ms']:.0f} | {m['mean_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## Per-query winners (highest nDCG@10)")
    lines.append("")
    lines.append("| qid | query | winner | nDCG@10 |")
    lines.append("| --- | --- | --- | ---: |")
    by_qid: dict[str, list[QueryResult]] = {}
    for r in rows:
        by_qid.setdefault(r.qid, []).append(r)
    for q in queries:
        items = by_qid.get(q.qid, [])
        if not items:
            continue
        winner = max(items, key=lambda r: r.ndcg_10)
        truncated = q.query if len(q.query) <= 60 else q.query[:57] + "..."
        lines.append(
            f"| {q.qid} | {truncated} | `{winner.config_label}` | {winner.ndcg_10:.3f} |"
        )

    report_path.write_text("\n".join(lines) + "\n")


async def _async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    gold_path = Path(args.gold).resolve()
    if not gold_path.exists():
        print(f"gold set not found: {gold_path}", file=sys.stderr)
        return 2

    queries = load_gold_set(gold_path)
    print(f"[eval] loaded {len(queries)} queries from {gold_path}")

    # Build the config matrix.
    selected_modes: list[RetrievalMode] = []
    for m in args.modes:
        try:
            selected_modes.append(RetrievalMode(m))
        except ValueError:
            print(f"unknown mode: {m}", file=sys.stderr)
            return 2

    configs: list[EvalConfig] = []
    for m in selected_modes:
        configs.append(EvalConfig(mode=m, use_hyde=False))
        # HyDE is most informative on hybrid_rerank; skip it on DENSE-only to
        # save time unless explicitly requested.
        if not args.skip_hyde and m is RetrievalMode.HYBRID_RERANK:
            configs.append(EvalConfig(mode=m, use_hyde=True))

    print(f"[eval] configs: {[c.label for c in configs]}")

    # Bootstrap infra (no FastAPI).
    init_engine()
    sessionmaker = get_sessionmaker()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    embedder = EmbeddingService(redis=redis)
    reranker = (
        None if args.skip_rerank else RerankerService()
    )
    hyde_service: HyDEService | None = None
    if not args.skip_hyde:
        hyde_service = HyDEService()
        await hyde_service.__aenter__()

    pipeline = RetrievalPipeline(embedder=embedder, reranker=reranker)

    rows: list[QueryResult] = []
    started = time.perf_counter()
    try:
        async with sessionmaker() as session:
            id_to_doc, _doc_to_id = await _resolve_doc_id_map(session)
            print(f"[eval] corpus: {len(id_to_doc)} grants resolved")

            for qi, q in enumerate(queries, 1):
                for c in configs:
                    if c.use_hyde and args.skip_hyde:
                        continue
                    if c.mode is RetrievalMode.HYBRID_RERANK and args.skip_rerank:
                        continue
                    try:
                        r = await _run_one(
                            pipeline=pipeline,
                            session=session,
                            query=q,
                            config=c,
                            hyde_service=hyde_service,
                            id_to_doc=id_to_doc,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[eval] {q.qid} {c.label} FAILED: {e}", file=sys.stderr)
                        continue
                    rows.append(r)
                    print(
                        f"[eval] {qi}/{len(queries)} {q.qid} {c.label:24s} "
                        f"P@5={r.precision_5:.2f} R@10={r.recall_10:.2f} "
                        f"nDCG={r.ndcg_10:.2f} MRR*={r.mrr_strict:.2f} "
                        f"({r.elapsed_ms}ms)"
                    )
    finally:
        if hyde_service is not None:
            await hyde_service.__aexit__(None, None, None)
        await redis.aclose()
        await dispose_engine()

    elapsed = time.perf_counter() - started
    print(f"[eval] {len(rows)} rows in {elapsed:.1f}s")

    if not rows:
        print("[eval] no rows captured — nothing to report.", file=sys.stderr)
        return 1

    # Write outputs.
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = _ML_ROOT / "evals" / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_path = runs_dir / f"{ts}.jsonl"
    with runs_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    agg = _aggregate(rows)
    report_path = results_dir / f"{ts}_report.md"
    _write_report(
        report_path=report_path,
        runs_path=runs_path,
        queries=queries,
        rows=rows,
        agg=agg,
    )
    print(f"[eval] wrote {runs_path}")
    print(f"[eval] wrote {report_path}")

    # Console summary so CI / interactive runs see the headline numbers.
    print("\nAggregate (mean across queries):")
    print(f"{'config':24s} {'P@5':>6s} {'R@10':>6s} {'nDCG':>6s} {'MRR*':>6s} {'p50':>6s}")
    for label, m in sorted(agg.items(), key=lambda kv: kv[1]["nDCG@10"], reverse=True):
        print(
            f"{label:24s} "
            f"{m['precision@5']:>6.3f} {m['recall@10']:>6.3f} "
            f"{m['nDCG@10']:>6.3f} {m['MRR_strict']:>6.3f} "
            f"{m['p50_ms']:>6.0f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5C retrieval eval sweep")
    parser.add_argument(
        "--gold",
        default=str(_ML_ROOT / "evals" / "gold_set.jsonl"),
        help="Path to gold_set.jsonl",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dense", "hybrid", "hybrid_rerank"],
        help="Retrieval modes to sweep",
    )
    parser.add_argument(
        "--skip-hyde",
        action="store_true",
        help="Skip configs that use HyDE (saves Gemini calls)",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip hybrid_rerank (saves BGE model load)",
    )
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())

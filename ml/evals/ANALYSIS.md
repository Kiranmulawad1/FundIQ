# Phase 5C — retrieval eval analysis

Companion to `results/<ts>_report.md`. Numbers below come from the
2026-05-31 sweep (`20260531T190812Z`): 25 gold queries × 4 configurations
= 100 rows.

## Headline numbers

| Config              | precision@5 | recall@10 | nDCG@10   | MRR (strict) | p50 (ms) |
| ------------------- | ----------: | --------: | --------: | -----------: | -------: |
| `dense`             |       0.552 |     0.873 | **0.854** |    **0.891** |       76 |
| `hybrid`            |       0.552 |     0.873 |     0.843 |        0.871 |   **26** |
| `hybrid_rerank`     |       0.544 |     0.836 |     0.805 |        0.765 |   36 478 |
| `hybrid_rerank+hyde`|       0.544 |     0.836 |     0.805 |        0.765 |   74 902 |

## Three findings worth defending

### 1. Dense alone is competitive on a 26-grant corpus

DENSE wins nDCG@10 (0.854) and MRR_strict (0.891). HYBRID matches its
precision/recall and costs ~3× less wall time (26 ms p50). HYBRID_RERANK
underperforms DENSE on every metric except a few high-affinity queries
where the cross-encoder finds a better top-1 (q03, q09, q16, q17, q25).

This is consistent with the literature: reranker gains are most visible
when the first-stage retriever returns many noisy candidates that need
to be sorted by a stronger model. With 26 grants and `RERANK_INPUT_K=50`
the cross-encoder sees the entire corpus — there is no noise to push
down. Dense cosine over `multilingual-e5-large` is already strong enough
on a corpus this small.

**Implication for the thesis:** the reranker's value is conditional on
corpus size. The eval should be re-run once Phase 2D adds BMFTR +
foerderdatenbank.de (~400+ grants) — the ranking is expected to invert
at that point.

### 2. HyDE produces identical rankings to non-HyDE under HYBRID_RERANK

`hybrid_rerank` and `hybrid_rerank+hyde` give bit-for-bit identical
top-5 on **0/25** queries differ between the two configurations.

Mechanism:
- The corpus has 26 grants.
- `DENSE_CANDIDATES = SPARSE_CANDIDATES = RERANK_INPUT_K = 50`.
- Both legs return the entire corpus; RRF fuses 26 unique IDs.
- The cross-encoder scores `(original_query_string, title + summary)` for
  all 26 candidates.
- HyDE shifts the dense *embedding*, not the *query string* the
  reranker sees. With the rerank input window already containing every
  grant, the dense leg's ordering becomes irrelevant.

So HyDE is functioning correctly (it generates hypotheticals, embeds
them, mean-pools them — confirmed in 5B smoke), but at this corpus size
its effect is masked by the reranker. The same eval against `dense` or
`hybrid` modes would show movement (those configs return top-K directly
from the embedding ranking).

**Implication for the thesis:** HyDE's benefit is bounded by
`P(some dense-relevant grant is OUT of top-RERANK_INPUT_K without HyDE)`.
That probability is 0 here. Re-run at ≥150 grants for a meaningful
HyDE evaluation.

### 3. Tail queries — where structural recall fails

Three queries deserve attention:

- **q17** "money for a new tech company" — dense nDCG=0.54 / strict MRR=0.10.
  Vague query, broad gold-set of 7 candidates. The reranker actually
  *helps* here (nDCG climbs to 0.64), which matches expectation: vague
  queries are where lexical cues are weakest and the cross-encoder's
  full attention pays off most.
- **q20** "Research-based startup funding in Germany" — rerank
  *demotes* the correct answers, dropping nDCG from 0.69 (dense) to 0.44.
  The reranker apparently prefers EU-level instruments over the EXIST
  family for this English phrasing. Worth labelling as a known
  failure mode in the thesis.
- **q22** "Equity investment from public investor for German startup"
  — same pattern: dense 0.71, rerank 0.54. The reranker's title+summary
  passage doesn't carry the "public investor" signal as cleanly as the
  embeddings do.

## Operational caveats

- **Latency**: rerank costs ~36 s p50 on CPU because the M-series MPS
  backend isn't available in this venv — measure again with `torch`
  built against Metal before treating this number as ceiling.
- **HyDE cost**: ~75 s p50 = ~38 s of Gemini round-trip per query plus
  the rerank. Acceptable in eval; will need streaming + caching in prod.
- **Cache bypass**: `cache=None` in every run, by design. Production
  numbers should show p50 < 50 ms for repeat queries thanks to the
  Phase 5B semantic cache.

## What this run does NOT measure

- **Recall floor**: we sweep K∈{5,10}. We do not measure whether top-50
  reliably contains the ideal answer — though by construction it does in
  this corpus (RRF fuses 26 grants out of 26).
- **Citation faithfulness**: the Writer agent (Phase 7) has not run
  yet, so we cannot measure end-to-end answer correctness — only
  retrieval ordering.
- **Cross-lingual gap**: half the queries are German, half English.
  We have not stratified the metrics by language; a follow-up should.

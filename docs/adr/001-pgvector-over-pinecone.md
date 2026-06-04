# ADR-001: Use pgvector (in Neon Postgres) instead of a dedicated vector DB

- **Status:** Accepted
- **Date:** 2026-05-25
- **Deciders:** Kiran Mulawad
- **Phase impact:** Foundation (Phase 1), RAG pipeline (Phase 5)

---

## Context

FundIQ's Researcher agent retrieves grants by hybrid search: dense embedding
similarity (multilingual-e5-large, 1024-dim) combined with BM25-style sparse
keyword matching, then reranked with a cross-encoder. The dense leg of that
pipeline needs a vector store.

Constraints driving the decision:

1. **Polyglot persistence already exists.** We commit to Neon Postgres for
   relational state (startups, grants, applications, sessions), Neo4j for the
   grant dependency graph, and DuckDB for analytics. Adding a fourth data
   system has real operational cost.
2. **Hybrid search is non-negotiable.** Dense-only retrieval misses
   keyword-precise queries ("EXIST stipend amount"). The reranker mitigates
   recall failure but cannot fix retrieval misses.
3. **Strong transactional coupling between grants and embeddings.** When a
   scraper re-pulls a grant and the body changes, the row update *and* the
   re-embed should be one atomic transaction. Cross-system writes are a
   reliability hazard.
4. **Solo-founder operability.** I'm running this. Each additional service is
   another set of credentials to rotate, another dashboard to monitor,
   another bill to track.
5. **Thesis cost ceiling.** The project must run on free/low tiers through
   the writeup. ~$200/month vector DB bills are not in scope.
6. **EU data residency.** Grant text + (eventually) startup PII land in this
   store. Hosting must be EU-region by default with no extra setup.

---

## Decision

**Use the `pgvector` extension on Neon Postgres as the vector store for grant
embeddings (1024-dim, multilingual-e5-large). Index with HNSW
(`m=16, ef_construction=64`) using cosine distance.**

Concretely:
- `grants.embedding vector(1024)` column.
- Index: `CREATE INDEX ix_grants_embedding ON grants USING hnsw (embedding vector_cosine_ops);`
- Hybrid retrieval combines pgvector cosine similarity with a
  `pg_trgm`-backed sparse keyword score, fused via reciprocal rank
  fusion in application code.
- BGE-reranker-v2-m3 reranks the fused top-50 to top-5.

---

## Alternatives Considered

### Pinecone (managed serverless vector DB)
- **Pros:** Best-in-class ANN performance; serverless billing; mature SDK.
- **Cons:**
  - Separate system → cross-system write coordination for grant updates.
  - $0.096/M reads + $4.50/M writes scales beyond free quickly when the
    reranker is fed top-50 candidates per query.
  - Hybrid search is "supported" via a separate sparse index — two indexes
    to maintain and keep in sync.
  - EU region available but additive setup.
- **Rejected because:** the operational and integration cost outweighs the
  ANN speed advantage at our scale (target: ~10k grants, ~1k startup
  embeddings).

### Weaviate Cloud / Qdrant Cloud
- **Pros:** Native hybrid search; OSS lineage; good DX.
- **Cons:** Free tiers expire/are restrictive. Self-hosting means another
  container in compose, another set of backups, another upgrade cadence.
- **Rejected because:** same operability cost as Pinecone without the ANN
  edge that justified Pinecone's price.

### ChromaDB / LanceDB (embedded)
- **Pros:** Zero infrastructure; perfect for local dev.
- **Cons:** Designed for embedded / single-process. No transactional
  coupling with the grant row in Postgres. Production deployments either
  bolt on a server-mode that's underbaked, or live in the API process and
  break on horizontal scale.
- **Rejected because:** doesn't survive past one API replica.

### Pure dense in Postgres + no pgvector (cube extension / array math)
- **Pros:** Smallest dep surface.
- **Cons:** No real ANN index; sequential scan over ~10k+ rows at query
  time; CPU-bound. Falls over at scale.
- **Rejected because:** the entire reason we want a vector store is the
  ANN index.

---

## Consequences

### Positive
- **One transactional store for grants + embeddings.** Scraper writes
  `(grant_row, embedding)` in a single transaction. No two-phase commit,
  no orphaned embeddings.
- **One credential, one dashboard, one backup story.** Neon handles PITR.
- **Neon branching** gives us instant copy-on-write dev/test branches with
  embeddings already populated. This is a significant DX win for the eval
  harness (Phase 9).
- **Hybrid search natively in SQL.** `pgvector` cosine score JOIN
  `pg_trgm` similarity score → single query, single round-trip.
- **EU-resident by Neon default.** No compliance bureaucracy.
- **Free dev tier, predictable production cost.** Neon scales-to-zero
  pricing on dev; production is a flat compute hour spend.

### Negative / Trade-offs
- **HNSW index build time grows non-linearly.** At ~100k grants the
  initial index build will take minutes. Mitigation: build offline during
  scheduled refresh; production reads use the previous index until the
  new one is warm.
- **Updates to embedded columns force HNSW maintenance.** pgvector's HNSW
  rebuilds the affected node on each update. Mitigation: batch
  re-embeddings during off-hours via Hatchet.
- **No managed hybrid-search abstraction.** We write the fusion logic
  ourselves. Mitigation: it's ~30 lines of SQL + RRF code, tested with a
  fixed gold set. We'd write similar glue against any vector DB.
- **Migration path if we outgrow it.** If/when grant volume exceeds
  500k rows, we'll either (a) move embeddings to a dedicated store while
  keeping the grant row in Postgres, or (b) move the cluster to Neon's
  paid tier with read replicas. Either path is doable without rewriting
  the retrieval API — it's behind the `app/rag/` interface.

---

## Validation Plan

- **Phase 5 thesis eval:** measure `precision@5` and `recall@10` on the
  100-item gold set for: (dense-only pgvector), (hybrid pgvector + trgm),
  (hybrid + BGE-reranker). Document in `EVALS.md`.
- **Latency budget:** p95 retrieval ≤ 200 ms at 10k grants. If exceeded,
  revisit by adding a Redis-backed query-embedding cache before going to
  pgvector. (Caching is also planned in Phase 5 regardless.)

---

## References

- pgvector HNSW docs: https://github.com/pgvector/pgvector#hnsw
- Neon vector benchmarks: https://neon.tech/blog/pgvector-hnsw-comparison
- "Retrieval that scales" (Anthropic): https://www.anthropic.com/research/contextual-retrieval

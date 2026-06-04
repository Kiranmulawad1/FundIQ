"""Cross-encoder reranker backed by Cohere's Rerank API.

History:
  Versions ≤2.0 used `BAAI/bge-reranker-v2-m3` locally via
  sentence-transformers. Same story as the embedder — 2.3GB of torch
  doesn't fit on a free-tier host. We switched to Cohere's hosted
  rerank model so the runtime stays slim.

API surface (unchanged):
  `score_pairs(query, passages)` keeps the same signature and the same
  "higher score = more relevant" contract. Callers (the retrieval
  pipeline and its tests) need no edits — the protocol is stable.

Scoring scale:
  Cohere returns scores in [0, 1] (probability-like). BGE returned raw
  logits in roughly [-10, 10]. The absolute scale is different, but the
  pipeline only uses these scores for sorting + top-K selection — the
  monotonicity is preserved, and the trace strip in the UI surfaces
  whichever value the model produced.

Cost guard:
  Cohere's free tier covers 1000 rerank calls/month. We bound the input
  batch at COHERE_MAX_DOCS (1000 per call by Cohere policy, but the
  RAG pipeline only sends top-50 candidates anyway). The retrieval
  pipeline's `cache_eligible` short-circuit also kicks in for repeat
  queries, so steady-state cost stays well under that budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
MODEL_NAME = "rerank-multilingual-v3.0"
DEFAULT_TIMEOUT_SECONDS = 30.0
# Cohere accepts up to 1000 docs per call. The retrieval pipeline only
# rerank's top-K candidates (defaults to 50) so we never approach this
# cap, but cap it explicitly to fail loud if a caller ever overdoes it.
COHERE_MAX_DOCS = 1000


class RerankerError(RuntimeError):
    """Raised when Cohere fails or returns output we can't interpret."""


class RerankerService:
    """API-backed reranker. One instance per process, share freely.

    No model load — the constructor is cheap. The HTTP client is created
    lazily on first call and shared across subsequent calls so the TLS
    connection pool stays warm.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout_seconds

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Return Cohere relevance scores aligned to the input `passages`
        order. Higher = more relevant.

        Empty passage list → empty score list (skip the API call). This
        mirrors the previous BGE behaviour and the tests rely on it.
        """
        if not passages:
            return []
        if len(passages) > COHERE_MAX_DOCS:
            msg = f"Cohere rerank: {len(passages)} docs exceeds cap {COHERE_MAX_DOCS}"
            raise RerankerError(msg)

        settings = get_settings()
        if settings.cohere_api_key is None:
            msg = "COHERE_API_KEY is not configured."
            raise RerankerError(msg)
        api_key = settings.cohere_api_key.get_secret_value()

        client = await self._ensure_client()
        payload = {
            "model": MODEL_NAME,
            "query": query,
            "documents": passages,
            # `top_n=len(passages)` makes Cohere return scores for ALL
            # docs, not just the top few. Without this the response only
            # carries the highest-ranked candidates, which breaks our
            # "score per passage in input order" contract.
            "top_n": len(passages),
        }
        try:
            r = await client.post(
                COHERE_RERANK_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(
                "rerank.cohere_failed",
                error_type=type(e).__name__,
                # Cohere's URL doesn't carry the key in the query string
                # so we don't need to scrub — but we still cap the
                # message length to keep log lines tidy.
                error=str(e)[:200],
            )
            raise RerankerError(f"Cohere rerank failed: {e}") from e

        body = r.json()
        # Response shape: {"results": [{"index": i, "relevance_score": f}, ...]}
        # Results come back sorted descending by relevance — we need to
        # un-sort them into the caller's original passage order.
        scores: list[float] = [0.0] * len(passages)
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(passages):
            msg = f"Cohere rerank returned unexpected shape: {body!r}"
            raise RerankerError(msg)
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(idx, int) or not isinstance(score, (int, float)):
                msg = f"Cohere rerank result missing fields: {item!r}"
                raise RerankerError(msg)
            if 0 <= idx < len(scores):
                scores[idx] = float(score)
        return scores

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

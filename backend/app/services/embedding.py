"""Multilingual embedding service backed by Gemini's Embedding API.

History:
  Versions ≤2.0 used `intfloat/multilingual-e5-large` locally via
  sentence-transformers. That worked great on a laptop but made the
  production image ~2.5GB and required ~2.2GB RAM at runtime — neither
  fits on a free-tier host. We swapped to Gemini's API so the backend
  becomes a thin Python layer (~150MB image, ~256MB RAM).

API surface (unchanged):
  `embed_passage(text)` and `embed_passages(texts)` — same signatures,
  same return shapes (1024-dim list[float], normalised). Stub embedders
  in tests stay valid without changes.

Caller convention (preserved):
  The retriever already routes queries through `embed_passages` with a
  "query: " prefix; the ETL passes passages without a prefix. e5
  documented this convention; Gemini doesn't, but the API does support
  task-type hints. We detect the e5 prefix here, map it to the right
  Gemini taskType, then strip it before sending — zero caller-side
  changes were needed during the refactor.

Caching:
  Same Redis cache shape, but the key prefix changed (`embed:gemini-v1:…`)
  so the e5 cache from the local-model era doesn't masquerade as fresh
  embeddings of a different vector space.

Dim contract:
  We request `outputDimensionality=1024` so embeddings line up with the
  existing pgvector(1024) column + HNSW index. Switching dimensions
  later would require a migration + re-embed.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

import httpx
import orjson

from app.core.http_retry import post_with_backoff

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis


logger = get_logger(__name__)

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 1024
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:batchEmbedContents"
)

# Gemini's API supports several task hints. We use these two for retrieval:
TaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]
DEFAULT_TIMEOUT_SECONDS = 30.0
# Conservative batch size — Gemini's batchEmbedContents accepts up to
# 100 per call, but smaller batches keep latency bounded for the
# retriever's single-query path and amortise nicely across the ETL.
MAX_BATCH = 32


class EmbeddingService:
    """API-backed embedder with the same public surface the rest of the
    codebase already calls into. Lazy httpx client; safe to share one
    instance across the whole process (it has a Redis cache, no GIL hot
    path).
    """

    def __init__(
        self,
        *,
        redis: Redis | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._redis = redis
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout_seconds

    async def aclose(self) -> None:
        """Mirror the GeminiAgentClient surface so the lifespan can close
        owned httpx clients cleanly.
        """
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Cache + prefix helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(text: str, task_type: TaskType) -> str:
        # Include task_type in the key — a query and a passage with
        # identical text still produce different vectors under Gemini.
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:gemini-v1:{task_type}:{h}"

    @staticmethod
    def _classify(text: str) -> tuple[str, TaskType]:
        """Detect the e5-style instruction prefix and map to Gemini's
        taskType. Strips the prefix from the returned text since
        Gemini's API doesn't want it.
        """
        if text.startswith("query: "):
            return text[len("query: ") :], "RETRIEVAL_QUERY"
        if text.startswith("passage: "):
            return text[len("passage: ") :], "RETRIEVAL_DOCUMENT"
        # No prefix → assume the caller is embedding a document (ETL,
        # eval harness). RETRIEVAL_QUERY would also work but document
        # is the more conservative default.
        return text, "RETRIEVAL_DOCUMENT"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def embed_passage(self, text: str) -> list[float]:
        """Embed a single document. Most ETL callers want this."""
        out = await self.embed_passages([text])
        return out[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Batch embed. Cache-first, then any misses go to Gemini in
        chunks of MAX_BATCH.
        """
        if not texts:
            return []

        # Classify + cache lookup. We honour the same prefix convention
        # the rest of the code uses, so the cache key is stable.
        classified = [self._classify(t) for t in texts]
        cache_keys = [self._cache_key(stripped, tt) for stripped, tt in classified]

        result: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []

        if self._redis is not None:
            cached = await self._redis.mget(cache_keys)
            for i, raw in enumerate(cached):
                if raw is not None:
                    result[i] = orjson.loads(raw)
                else:
                    misses.append(i)
        else:
            misses = list(range(len(texts)))

        if misses:
            client = await self._ensure_client()
            api_key = self._api_key()
            url = f"{GEMINI_EMBED_URL}?key={api_key}"

            # Walk misses in MAX_BATCH chunks. Each Gemini request needs
            # its `taskType` set per-content; we keep the original index
            # so we can stitch results back into `result` in order.
            for chunk_start in range(0, len(misses), MAX_BATCH):
                chunk = misses[chunk_start : chunk_start + MAX_BATCH]
                requests = [
                    {
                        "model": f"models/{MODEL_NAME}",
                        "content": {"parts": [{"text": classified[i][0]}]},
                        "taskType": classified[i][1],
                        "outputDimensionality": EMBEDDING_DIM,
                    }
                    for i in chunk
                ]
                payload = {"requests": requests}
                r = await post_with_backoff(
                    client, url, json=payload, label="embedding.gemini"
                )
                body = r.json()
                embeddings = body.get("embeddings", [])
                if len(embeddings) != len(chunk):
                    msg = (
                        f"Gemini returned {len(embeddings)} embeddings "
                        f"for {len(chunk)} requests"
                    )
                    raise RuntimeError(msg)

                # Gemini batchEmbedContents returns vectors NOT normalised
                # by default when outputDimensionality is set. We
                # normalise here so cosine similarity stays equivalent
                # to dot product — matches the pgvector index assumption.
                for offset, idx in enumerate(chunk):
                    values = embeddings[offset].get("values") or embeddings[offset].get("value")
                    if not isinstance(values, list):
                        msg = f"Gemini embedding payload missing values: {embeddings[offset]!r}"
                        raise RuntimeError(msg)
                    vec = _l2_normalise([float(v) for v in values])
                    result[idx] = vec

            # Write-back to cache.
            if self._redis is not None:
                pipe = self._redis.pipeline(transaction=False)
                for idx in misses:
                    pipe.set(
                        cache_keys[idx],
                        orjson.dumps(result[idx]),
                        ex=CACHE_TTL_SECONDS,
                    )
                await pipe.execute()

        # Every slot is populated by this point.
        return [r for r in result if r is not None]

    # ------------------------------------------------------------------
    # Internal HTTP plumbing
    # ------------------------------------------------------------------
    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

    @staticmethod
    def _api_key() -> str:
        settings = get_settings()
        if settings.gemini_api_key is None:
            msg = "GEMINI_API_KEY is not configured."
            raise RuntimeError(msg)
        return settings.gemini_api_key.get_secret_value()


def _l2_normalise(v: list[float]) -> list[float]:
    """Project to the unit hypersphere so cosine ≡ dot product.

    pgvector's HNSW index over a `vector_cosine_ops` opclass also tolerates
    unnormalised vectors, but the existing corpus + agent cache assume
    unit-norm — we keep that invariant.
    """
    import math

    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]

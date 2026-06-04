"""Semantic query cache for the retrieval pipeline.

How it works:
  1. Embed the incoming query (this is already done before retrieval).
  2. Compute cosine similarity vs every cached query embedding.
  3. If any cached embedding has sim > 0.95 → return its stored result.
  4. Miss → run full pipeline, then store this (embedding, result) pair.

Why sim > 0.95 (not exact match):
  Queries like "Stipendium für Doktoranden" and "Förderung für Promovierte"
  should both hit the same cached result. e5 cosine similarity above 0.95
  means the two queries are essentially paraphrases.

Why scope to HYBRID_RERANK only:
  Dense (~30ms) and hybrid (~50ms) are already faster than a Redis round
  trip + sim scan. Reranked retrieval (~3-8s with HyDE) is the only call
  worth caching.

Eviction:
  Redis ZSET with timestamp scores → LRU via ZREMRANGEBYRANK when size
  exceeds CACHE_MAX_ENTRIES. No TTL on per-entry SET keys is also fine
  (the ZSET membership is the source of truth), but we set EX to bound
  worst-case footprint if the ZSET gets wiped.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson

from app.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

INDEX_KEY = "rag:cache:index"
ENTRY_KEY_PREFIX = "rag:cache:entry:"

DEFAULT_SIM_THRESHOLD = 0.95
CACHE_MAX_ENTRIES = 200
DEFAULT_TTL_SECONDS = 86400  # 24h


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """Lightweight serialisable cache slot. The retrieval result is opaque
    JSON-able dict — we don't import the full RetrievalResult to avoid a
    circular import."""

    embedding: list[float]
    result_blob: dict[str, Any]
    cached_for_query: str
    cached_at: float
    cache_key: str


class SemanticCache:
    """Redis-backed semantic cache. Single instance per process is fine."""

    def __init__(
        self,
        *,
        redis: Redis,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        max_entries: int = CACHE_MAX_ENTRIES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._redis = redis
        self._sim_threshold = sim_threshold
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    @staticmethod
    def _key_for(embedding: list[float]) -> str:
        # Hash on quantised bytes so essentially-equal vectors collide.
        # We don't rely on this alone for sim — the explicit scan still
        # runs — but it gives stable, short cache keys.
        quantised = bytes(int((x + 1.0) * 32767) % 65536 // 256 for x in embedding[:32])
        return f"{ENTRY_KEY_PREFIX}{hashlib.sha256(quantised).hexdigest()[:24]}"

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        # Both are unit-normalised by EmbeddingService → dot is cosine.
        return sum(x * y for x, y in zip(a, b, strict=False))

    async def lookup(self, query_embedding: list[float]) -> CacheEntry | None:
        """Return the best-matching cached entry if any has sim > threshold."""
        keys = await self._redis.zrevrange(INDEX_KEY, 0, self._max_entries - 1)
        if not keys:
            return None

        best: tuple[float, CacheEntry] | None = None
        for raw_key in keys:
            key = raw_key if isinstance(raw_key, str) else raw_key.decode()
            raw = await self._redis.get(key)
            if raw is None:
                # ZSET entry pointed at an evicted/expired key — clean up.
                await self._redis.zrem(INDEX_KEY, key)
                continue
            try:
                blob = orjson.loads(raw)
                cached_emb = blob["embedding"]
            except (orjson.JSONDecodeError, KeyError):
                await self._redis.zrem(INDEX_KEY, key)
                continue
            sim = self._cosine(query_embedding, cached_emb)
            if sim >= self._sim_threshold and (best is None or sim > best[0]):
                best = (
                    sim,
                    CacheEntry(
                        embedding=cached_emb,
                        result_blob=blob.get("result", {}),
                        cached_for_query=blob.get("query", ""),
                        cached_at=float(blob.get("cached_at", 0)),
                        cache_key=key,
                    ),
                )

        if best is not None:
            logger.info("rag.cache.hit", sim=round(best[0], 3), key=best[1].cache_key)
            return best[1]
        return None

    async def store(
        self,
        *,
        query: str,
        query_embedding: list[float],
        result_blob: dict[str, Any],
    ) -> str:
        """Persist (query, embedding, result). Returns the cache key used."""
        key = self._key_for(query_embedding)
        now = time.time()
        blob = orjson.dumps(
            {
                "query": query,
                "embedding": query_embedding,
                "result": result_blob,
                "cached_at": now,
                "cache_id": uuid.uuid4().hex,
            }
        )
        await self._redis.set(key, blob, ex=self._ttl)
        await self._redis.zadd(INDEX_KEY, {key: now})
        # Trim oldest entries beyond max_entries. ZREMRANGEBYRANK uses
        # ascending rank; we keep the newest `max_entries`.
        if self._max_entries > 0:
            await self._redis.zremrangebyrank(INDEX_KEY, 0, -(self._max_entries + 1))
        logger.info("rag.cache.store", key=key, query=query[:120])
        return key

    async def clear(self) -> int:
        """Drop everything. Returns the number of entries deleted."""
        keys = await self._redis.zrange(INDEX_KEY, 0, -1)
        if keys:
            await self._redis.delete(*keys)
        await self._redis.delete(INDEX_KEY)
        return len(keys)

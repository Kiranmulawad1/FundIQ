"""Unit tests for the Gemini-backed EmbeddingService.

We use a scripted httpx transport instead of patching `client.post`
directly — that catches both the request shape AND the URL/headers, and
mirrors the same pattern the LLM-client backoff tests use.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from app.services.embedding import (
    EMBEDDING_DIM,
    MODEL_NAME,
    EmbeddingService,
)


# ---------------------------------------------------------------------------
# Programmable transport
# ---------------------------------------------------------------------------
class _CapturingTransport(httpx.AsyncBaseTransport):
    """Returns a fixed JSON body and remembers each request that came in."""

    def __init__(self, response_body: dict[str, Any], status_code: int = 200) -> None:
        self._body = response_body
        self._status = status_code
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "url": str(request.url),
                "json": json.loads(request.content.decode("utf-8")) if request.content else None,
            },
        )
        return httpx.Response(
            status_code=self._status,
            content=json.dumps(self._body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


def _fake_embeddings(n: int, dim: int = EMBEDDING_DIM) -> dict[str, Any]:
    """Build a plausible-ish batchEmbedContents response. Vectors are all
    1/sqrt(dim) so they normalise cleanly.
    """
    base = 1.0 / (dim**0.5)
    return {
        "embeddings": [
            {"values": [base + (i * 0.0001) for _ in range(dim)]} for i in range(n)
        ],
    }


# ---------------------------------------------------------------------------
# Prefix classifier — pure
# ---------------------------------------------------------------------------
def test_classify_query_prefix_maps_to_retrieval_query() -> None:
    e = EmbeddingService()
    stripped, task = e._classify("query: pre-seed grant Bayern")
    assert stripped == "pre-seed grant Bayern"
    assert task == "RETRIEVAL_QUERY"


def test_classify_passage_prefix_maps_to_retrieval_document() -> None:
    e = EmbeddingService()
    stripped, task = e._classify("passage: this is the grant body")
    assert stripped == "this is the grant body"
    assert task == "RETRIEVAL_DOCUMENT"


def test_classify_bare_text_defaults_to_document() -> None:
    e = EmbeddingService()
    stripped, task = e._classify("no prefix here")
    assert stripped == "no prefix here"
    assert task == "RETRIEVAL_DOCUMENT"


# ---------------------------------------------------------------------------
# Cache key includes task_type so query/passage of the same text differ
# ---------------------------------------------------------------------------
def test_cache_key_distinguishes_task_types() -> None:
    k_q = EmbeddingService._cache_key("hello", "RETRIEVAL_QUERY")
    k_d = EmbeddingService._cache_key("hello", "RETRIEVAL_DOCUMENT")
    assert k_q != k_d
    assert k_q.startswith("embed:gemini-v1:RETRIEVAL_QUERY:")
    assert k_d.startswith("embed:gemini-v1:RETRIEVAL_DOCUMENT:")


# ---------------------------------------------------------------------------
# HTTP path — no Redis, no real Gemini
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_embed_passages_sends_batch_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call → one POST to batchEmbedContents with per-text taskType
    set to RETRIEVAL_DOCUMENT (no prefix on input).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "ci-dummy")
    # Force settings to re-read so the override takes effect.
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport(_fake_embeddings(2))
    client = httpx.AsyncClient(transport=transport, base_url="http://t")

    svc = EmbeddingService(client=client)
    vecs = await svc.embed_passages(["first body", "second body"])

    assert len(vecs) == 2
    assert len(vecs[0]) == EMBEDDING_DIM
    # The vectors should be unit-norm (cosine ≡ dot for the index).
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6

    # Single POST captured; the per-content payload uses RETRIEVAL_DOCUMENT.
    assert len(transport.requests) == 1
    body = transport.requests[0]["json"]
    assert "requests" in body
    assert len(body["requests"]) == 2
    for req in body["requests"]:
        assert req["model"] == f"models/{MODEL_NAME}"
        assert req["taskType"] == "RETRIEVAL_DOCUMENT"
        assert req["outputDimensionality"] == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_embed_passages_query_prefix_switches_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport(_fake_embeddings(1))
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = EmbeddingService(client=client)

    await svc.embed_passages(["query: stipend academic spinoff"])

    req = transport.requests[0]["json"]["requests"][0]
    assert req["taskType"] == "RETRIEVAL_QUERY"
    # The "query: " prefix must NOT leak into Gemini's input — e5
    # convention is internal to the cache + classifier.
    assert req["content"]["parts"][0]["text"] == "stipend academic spinoff"


@pytest.mark.asyncio
async def test_embed_empty_input_returns_empty_list_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty `texts` list must not allocate an httpx client OR hit
    the network. Important for tests + cold-start latency.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport(_fake_embeddings(0))
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = EmbeddingService(client=client)

    out = await svc.embed_passages([])
    assert out == []
    assert transport.requests == []


@pytest.fixture(autouse=True)
def _restore_settings_cache() -> Any:
    """Each test mutates GEMINI_API_KEY via monkeypatch; flush the
    lru_cache so the next test starts from a clean Settings object.
    """
    yield
    from app.core.config import get_settings

    get_settings.cache_clear()
    os.environ.pop("GEMINI_API_KEY", None)

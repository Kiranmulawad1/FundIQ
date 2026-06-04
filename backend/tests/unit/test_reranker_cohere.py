"""Unit tests for the Cohere-backed RerankerService.

Same pattern as the embedder tests — scripted httpx transport instead
of a real Cohere account.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from app.rag.reranker import (
    COHERE_MAX_DOCS,
    MODEL_NAME,
    RerankerError,
    RerankerService,
)


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_body: dict[str, Any], status_code: int = 200) -> None:
        self._body = response_body
        self._status = status_code
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content.decode("utf-8")) if request.content else None,
            },
        )
        return httpx.Response(
            status_code=self._status,
            content=json.dumps(self._body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


# ---------------------------------------------------------------------------
# Happy path: scores arrive in passage-order, not Cohere's sorted order.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_score_pairs_realigns_to_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    # Cohere returns results sorted by relevance DESC. Passage 2 is the
    # winner (score 0.9), passage 0 second (0.6), passage 1 last (0.1).
    transport = _CapturingTransport({
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.6},
            {"index": 1, "relevance_score": 0.1},
        ],
    })
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = RerankerService(client=client)

    scores = await svc.score_pairs(
        "query: founder stipend",
        ["first", "second", "third"],
    )

    # Returned list must align with the INPUT passage order, NOT
    # Cohere's sorted order — this is the whole point of the un-sort.
    assert scores == [0.6, 0.1, 0.9]


@pytest.mark.asyncio
async def test_score_pairs_uses_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "sk-test-secret")
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport({
        "results": [{"index": 0, "relevance_score": 0.5}],
    })
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = RerankerService(client=client)
    await svc.score_pairs("q", ["one"])

    headers = transport.requests[0]["headers"]
    assert headers.get("authorization") == "Bearer sk-test-secret"


@pytest.mark.asyncio
async def test_score_pairs_requests_top_n_equal_to_input_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `top_n=len(passages)` Cohere only returns the top few
    candidates — we'd be left with default 0.0 scores for the rest and
    silently degrade retrieval quality.
    """
    monkeypatch.setenv("COHERE_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport({
        "results": [
            {"index": i, "relevance_score": 0.5} for i in range(5)
        ],
    })
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = RerankerService(client=client)

    await svc.score_pairs("q", [f"p{i}" for i in range(5)])
    body = transport.requests[0]["json"]
    assert body["top_n"] == 5
    assert body["model"] == MODEL_NAME


# ---------------------------------------------------------------------------
# Empty input skips the API entirely.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_passages_returns_empty_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    transport = _CapturingTransport({"results": []})
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    svc = RerankerService(client=client)

    out = await svc.score_pairs("q", [])
    assert out == []
    assert transport.requests == []


# ---------------------------------------------------------------------------
# Hard cap (defence in depth — pipeline already caps at top-50)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_over_cap_input_raises_reranker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "ci-dummy")
    from app.core.config import get_settings

    get_settings.cache_clear()

    svc = RerankerService()
    with pytest.raises(RerankerError, match="exceeds cap"):
        await svc.score_pairs("q", ["p"] * (COHERE_MAX_DOCS + 1))


# ---------------------------------------------------------------------------
# Missing key fails fast (no HTTP attempt)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_api_key_raises_reranker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setting to "" (not delenv) so pydantic-settings doesn't fall
    # through to .env — which on a developer machine probably has the
    # real key. The _empty_str_to_none validator turns "" into None.
    monkeypatch.setenv("COHERE_API_KEY", "")
    from app.core.config import get_settings

    get_settings.cache_clear()

    svc = RerankerService()
    with pytest.raises(RerankerError, match="COHERE_API_KEY"):
        await svc.score_pairs("q", ["one"])


@pytest.fixture(autouse=True)
def _restore_settings_cache() -> Any:
    yield
    from app.core.config import get_settings

    get_settings.cache_clear()
    os.environ.pop("COHERE_API_KEY", None)

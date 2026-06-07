"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Vague user queries ("we do AI for healthcare, need ~200k") embed poorly
against real grant descriptions because they share little vocabulary.
HyDE fixes this by asking an LLM to *imagine* what a matching grant
would say, embedding those hypothetical descriptions, and using the
mean of those embeddings as the dense-leg query vector.

Why mean-pool over concatenation:
  The HyDE paper tests both. Mean-pool is preferred at retrieval time
  because:
    1. One dense query, one HNSW lookup (cost stays O(log n)).
    2. Concatenation requires a longer-context embedding, which e5
       handles but produces noisier vectors for short queries.
    3. Mean of normalised vectors stays unit-norm (cosine semantics hold).

Why Gemini 2.5 Flash:
  - Free tier sufficient for thesis-scale eval traffic.
  - Strong multilingual performance on German grant text.
  - Single REST call; no new SDK needed (httpx already in deps).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.prompts import PromptFetchError, get_prompt

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Prompt lives in Langfuse under name "hyde".

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 12.0


class HyDEError(RuntimeError):
    """Raised when Gemini fails or returns unparseable output."""


class HyDEService:
    """Generates hypothetical grant descriptions via Gemini.

    Stateless; share one instance per process.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._model = model
        self._timeout = timeout_seconds

    async def __aenter__(self) -> HyDEService:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_hypotheticals(self, query: str, *, n: int = 3) -> list[str]:
        """Call Gemini and return up to `n` hypothetical grant descriptions.

        Falls back to returning `[query]` (which downstream still works via
        the dense leg) if Gemini fails — HyDE is an enhancement, not a
        precondition.
        """
        settings = get_settings()
        if settings.gemini_api_key is None:
            logger.warning("hyde.no_gemini_key")
            return [query]

        try:
            compiled = get_prompt("hyde").compile(query=query)
        except PromptFetchError as e:
            logger.warning("hyde.prompt_unavailable", error=str(e)[:200])
            return [query]

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": compiled.text}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "responseMimeType": "application/json",
                "maxOutputTokens": 800,
            },
        }
        api_key = settings.gemini_api_key.get_secret_value()
        url = f"{GEMINI_GENERATE_URL}?key={api_key}"

        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        owns = self._client is None
        try:
            try:
                r = await client.post(url, json=payload)
                r.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("hyde.gemini_failed", error_type=type(e).__name__, error=str(e)[:200])
                return [query]

            data = r.json()
            text = self._extract_text(data)
            descriptions = self._parse_descriptions(text)
            if not descriptions:
                logger.warning("hyde.empty_response", raw=text[:300])
                return [query]
            logger.info("hyde.generated", count=len(descriptions), query=query[:120])
            return descriptions[:n]
        finally:
            if owns:
                await client.aclose()

    @staticmethod
    def _extract_text(data: dict[str, object]) -> str:
        """Pull the model's text out of the Gemini response envelope."""
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        first = candidates[0]
        if not isinstance(first, dict):
            return ""
        content = first.get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            return ""
        chunks: list[str] = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(chunks).strip()

    @staticmethod
    def _parse_descriptions(text: str) -> list[str]:
        """Robust JSON parse. Falls back to line-split on parse failures."""
        if not text:
            return []
        # Strip code fences just in case despite responseMimeType.
        stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            # Best-effort line split.
            return [
                line.strip().strip("-•*").strip()
                for line in stripped.splitlines()
                if len(line.strip()) > 20
            ]
        if isinstance(obj, dict):
            descs = obj.get("descriptions")
            if isinstance(descs, list):
                return [str(d).strip() for d in descs if isinstance(d, str) and d.strip()]
        if isinstance(obj, list):
            return [str(d).strip() for d in obj if isinstance(d, str) and d.strip()]
        return []

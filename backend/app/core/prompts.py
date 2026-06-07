"""Langfuse-backed prompt registry.

The prompts for every agent node (Planner, Scorer, Writer, Critic, HyDE)
live in Langfuse so they can be iterated without a redeploy. This module
is the thin shim agents talk to.

Why no fallback templates in this repo:
  The whole point of moving prompts to Langfuse is to make them the
  source of truth. Keeping a fallback copy here would invite drift
  (someone tweaks Langfuse, forgets to update the file, the two diverge
  silently). If Langfuse is unreachable AND the prompt hasn't been
  cached yet, the agent fails loudly with `PromptFetchError` and the
  existing degraded-fallback path returns retrieval-only results to
  the user.

Caching:
  Langfuse's SDK caches each prompt by name for 60 seconds by default,
  so a healthy traffic pattern reads from network once per minute. We
  also cache the wrapper object so we don't reconstruct it on every
  call.

Tests:
  The `_overrides` dict lets unit + integration tests inject prompt
  strings without hitting the network. CI sets `_TEST_PROMPTS` via
  conftest before any agent code runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.core.observability import _client as _langfuse_client

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PromptFetchError(RuntimeError):
    """Raised when Langfuse can't return a prompt and no test override
    is registered. Callers (agent nodes) catch this to fall back to a
    graceful "model unavailable" response rather than 500ing the request.
    """


@dataclass
class CompiledPrompt:
    """Result of fetching + rendering. Carries the Langfuse handle so the
    generation trace can link back to the exact prompt version.
    """

    text: str
    langfuse_handle: Any | None = None  # the underlying Langfuse Prompt object


# Per-process cache of fetched-prompt wrappers. Langfuse caches the
# prompt content itself; this avoids reconstructing the wrapper.
_cache: dict[str, Any] = {}

# Test injection. Keys are prompt names ("planner", "scorer", ...),
# values are raw template strings. When set, `get_prompt(name).compile(**vars)`
# uses the override and skips Langfuse entirely.
_overrides: dict[str, str] = {}


def set_test_override(name: str, template: str) -> None:
    """Register a fake prompt for unit/integration tests. Tests do this
    in conftest before any agent code imports — the override sticks for
    the rest of the process unless cleared.
    """
    _overrides[name] = template


def clear_test_overrides() -> None:
    _overrides.clear()


def get_prompt(name: str, *, label: str = "production") -> _PromptHandle:
    """Return a handle that can `.compile(**variables) -> CompiledPrompt`.

    Lazily fetches from Langfuse on first call per process; subsequent
    calls hit the SDK cache (60s TTL by default).

    Raises `PromptFetchError` if Langfuse is unconfigured/unreachable
    AND no test override is registered.
    """
    if name in _overrides:
        return _PromptHandle(name=name, langfuse_prompt=None, override=_overrides[name])

    cache_key = f"{name}:{label}"
    if cache_key in _cache:
        return _cache[cache_key]

    if _langfuse_client is None:
        msg = (
            f"Cannot fetch prompt '{name}': Langfuse is not configured. "
            f"Set LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST or register a test "
            f"override via set_test_override()."
        )
        raise PromptFetchError(msg)

    try:
        prompt_obj = _langfuse_client.get_prompt(name, label=label)
    except Exception as exc:  # noqa: BLE001 — Langfuse SDK raises a variety
        msg = f"Langfuse get_prompt failed for '{name}': {exc}"
        logger.warning("prompts.fetch_failed", name=name, error=str(exc)[:200])
        raise PromptFetchError(msg) from exc

    handle = _PromptHandle(name=name, langfuse_prompt=prompt_obj, override=None)
    _cache[cache_key] = handle
    return handle


@dataclass
class _PromptHandle:
    name: str
    langfuse_prompt: Any | None
    override: str | None

    def compile(self, **variables: Any) -> CompiledPrompt:
        """Render the template. Langfuse uses `{{name}}` (mustache); the
        local override path supports the same syntax for symmetry.
        """
        if self.override is not None:
            text = _render_mustache(self.override, variables)
            return CompiledPrompt(text=text, langfuse_handle=None)

        assert self.langfuse_prompt is not None  # type-narrow; get_prompt ensures
        text: str = self.langfuse_prompt.compile(**variables)
        return CompiledPrompt(text=text, langfuse_handle=self.langfuse_prompt)


def _render_mustache(template: str, variables: dict[str, Any]) -> str:
    """Minimal mustache substitution for the test-override path. Does
    NOT support nested constructs — agents only use plain `{{name}}`.
    """
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out

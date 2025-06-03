"""A caching wrapper that memoizes another backend's responses.

Repeated prompts (common when re-running caption/QA pipelines over the same
clips) are served from an in-memory cache instead of hitting the wrapped
backend again.
"""

from __future__ import annotations

from auricle.llm.base import GenerationResult, LLMBackend


class CachingBackend(LLMBackend):
    """Wrap ``backend`` and remember one response per distinct prompt.

    The cache key includes ``max_new_tokens`` so the same prompt requested
    with different budgets is stored separately. ``name`` mirrors the wrapped
    backend so logging stays honest.
    """

    def __init__(self, backend: LLMBackend):
        self.name = backend.name
        self._backend = backend
        self._cache: dict[tuple[str, int], GenerationResult] = {}
        self._hits = 0

    @property
    def hits(self) -> int:
        """Number of generate calls served from the cache."""
        return self._hits

    @property
    def cache_size(self) -> int:
        """Number of distinct prompts currently cached."""
        return len(self._cache)

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        key = (prompt, max_new_tokens)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        result = self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        self._cache[key] = result
        return result

    def clear(self) -> None:
        """Drop all cached responses."""
        self._cache.clear()
        self._hits = 0

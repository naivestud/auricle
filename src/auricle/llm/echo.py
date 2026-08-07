"""A deterministic backend that parrots the prompt back.

Useful for wiring tests, offline demos, and dry runs where no real model
should be loaded. It still honours ``max_new_tokens`` by truncating on word
boundaries, so prompt construction can be exercised end to end.
"""

from __future__ import annotations

from auricle.llm.base import GenerationResult, LLMBackend


class EchoBackend(LLMBackend):
    """Returns the prompt (or a prefix of it) unchanged."""

    name = "echo"

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        words = prompt.split()
        text = " ".join(words[:max_new_tokens])
        if self.prefix:
            text = f"{self.prefix}{text}"
        return GenerationResult(text=text, backend=self.name, prompt=prompt)

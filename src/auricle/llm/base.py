"""The backend protocol shared by all language-model integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The text a backend produced, with enough context to log or debug it."""

    text: str
    backend: str
    prompt: str = field(default="", repr=False)


class LLMBackend(ABC):
    """Anything that can turn a prompt into text.

    Concrete backends set the ``name`` class attribute and implement
    :meth:`generate`. Backends are expected to be cheap to construct and
    safe to reuse across calls.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        """Produce a completion for ``prompt``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"

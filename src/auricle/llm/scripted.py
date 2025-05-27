"""A deterministic backend that replays a fixed list of responses.

Handy for tests and demos that need more control than :class:`EchoBackend`
offers: the exact reply to each successive prompt is known ahead of time.
"""

from __future__ import annotations

from collections.abc import Sequence

from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.registry import register_backend


@register_backend
class ScriptedBackend(LLMBackend):
    """Return canned responses in order, one per ``generate`` call.

    When responses run out, ``cycle=True`` (the default) wraps around to the
    start; otherwise the final response is repeated. An empty script yields
    empty strings.
    """

    name = "scripted"

    def __init__(self, responses: Sequence[str] = (), cycle: bool = True):
        self.responses = tuple(responses)
        self.cycle = cycle
        self._calls = 0

    @property
    def calls(self) -> int:
        """Number of ``generate`` calls served so far."""
        return self._calls

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        if not self.responses:
            text = ""
        elif self.cycle:
            text = self.responses[self._calls % len(self.responses)]
        else:
            text = self.responses[min(self._calls, len(self.responses) - 1)]
        self._calls += 1
        return GenerationResult(text=text, backend=self.name, prompt=prompt)

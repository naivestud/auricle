#!/usr/bin/env python
"""Register a custom LLM backend and use it for audio captioning.

Backends implement a single ``generate`` method and announce themselves with
the ``@register_backend`` decorator; after that they are selectable by name
anywhere a backend is accepted.

    python examples/custom_backend.py
"""

from __future__ import annotations

import numpy as np

from auricle import AuricleModel, caption_audio, get_backend
from auricle.llm import GenerationResult, LLMBackend, register_backend


@register_backend
class ReverseBackend(LLMBackend):
    """A toy backend that reverses the prompt's words.

    Stands in for any real integration: replace ``generate`` with a call to
    your model server and the rest of the pipeline is unchanged.
    """

    name = "reverse"

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        words = prompt.split()[:max_new_tokens]
        return GenerationResult(text=" ".join(reversed(words)), backend=self.name, prompt=prompt)


def synth_clip(seconds: float = 1.5, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.4 * np.sin(2 * np.pi * 520.0 * t)).astype(np.float32)


def main() -> None:
    model = AuricleModel.tiny()
    backend = get_backend("reverse")  # look up the backend we just registered
    caption = caption_audio(model, backend, synth_clip(), sample_rate=16_000)
    print("Caption from the 'reverse' backend:")
    print(f"  {caption[:200]}")


if __name__ == "__main__":
    main()

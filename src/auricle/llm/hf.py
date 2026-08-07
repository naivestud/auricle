"""Optional HuggingFace transformers backend.

Requires the ``hf`` extra::

    pip install "auricle[hf]"
"""

from __future__ import annotations

from auricle.errors import BackendError
from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.registry import register_backend


@register_backend
class HuggingFaceBackend(LLMBackend):
    """Generate text with a local ``transformers`` text-generation pipeline."""

    name = "huggingface"

    def __init__(self, model: str, device: str | None = None, **pipeline_kwargs):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise BackendError(
                "the huggingface backend needs transformers; install with `pip install auricle[hf]`"
            ) from exc
        self.model = model
        self._pipeline = pipeline("text-generation", model=model, device=device, **pipeline_kwargs)

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"].strip()
        return GenerationResult(text=text, backend=self.name, prompt=prompt)

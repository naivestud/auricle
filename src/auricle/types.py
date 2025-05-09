"""Shared structural types used across the package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from auricle.asr.vocab import CharVocabulary
    from auricle.encoder.config import EncoderConfig


@runtime_checkable
class ModelLike(Protocol):
    """The surface of :class:`auricle.model.AuricleModel` helpers rely on.

    Pipelines, streaming, evaluation and checkpoint utilities only need
    this subset, so anything that implements it — a wrapper, a quantized
    variant, a test double — can stand in for the real model.
    """

    config: EncoderConfig
    vocab: CharVocabulary

    def transcribe(self, waveform: torch.Tensor) -> list[str]:
        """Greedy-decode ``waveform`` into a list of transcripts."""

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialisable parameter dictionary (torch.nn.Module interface)."""

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Load parameters from ``state_dict`` (torch.nn.Module interface)."""

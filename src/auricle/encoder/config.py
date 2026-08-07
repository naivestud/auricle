"""Configuration for the whisper-style audio encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(kw_only=True)
class EncoderConfig:
    """Hyperparameters for the audio encoder.

    Defaults are a compact configuration meant for tests and
    experimentation; scale ``d_model``/``n_layers`` up for real training.
    """

    n_mels: int = 80
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    max_frames: int = 1500
    dropout: float = 0.0

    @classmethod
    def tiny(cls) -> EncoderConfig:
        """A tiny configuration that runs in milliseconds on CPU."""
        return cls(d_model=64, n_layers=2, n_heads=4, max_frames=500)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

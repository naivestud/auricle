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

    @classmethod
    def small(cls) -> EncoderConfig:
        """A small configuration for single-GPU experiments."""
        return cls(d_model=256, n_layers=4, n_heads=4, max_frames=1500)

    @classmethod
    def base(cls) -> EncoderConfig:
        """A base configuration approaching whisper-small capacity."""
        return cls(d_model=512, n_layers=8, n_heads=8, max_frames=1500)

    def validate(self) -> None:
        """Raise ``ValueError`` if any hyperparameter is inconsistent."""
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be positive and divide d_model ({self.d_model})"
            )
        if self.ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {self.ff_mult}")
        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

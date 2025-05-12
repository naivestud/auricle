"""Synthetic test signals.

Deterministic signal generators for fixtures, examples and property tests.
All generators return mono float32 arrays with values in ``[-1, 1]``.
"""

from __future__ import annotations

import numpy as np

from auricle.constants import SAMPLE_RATE


def silence(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A fully silent signal."""
    n = int(round(duration_seconds * sample_rate))
    return np.zeros(n, dtype=np.float32)


def sine(
    duration_seconds: float,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """A pure tone at ``frequency`` hertz."""
    if frequency < 0:
        raise ValueError(f"frequency must be non-negative, got {frequency}")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
    t = np.arange(int(round(duration_seconds * sample_rate)), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def chirp(
    duration_seconds: float,
    start_frequency: float = 100.0,
    end_frequency: float = 4000.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """A linear frequency sweep from ``start_frequency`` to ``end_frequency``."""
    if start_frequency < 0 or end_frequency < 0:
        raise ValueError("frequencies must be non-negative")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rate = (end_frequency - start_frequency) / duration_seconds if duration_seconds > 0 else 0.0
    phase = 2.0 * np.pi * (start_frequency * t + 0.5 * rate * t * t)
    return (amplitude * np.sin(phase)).astype(np.float32)


def white_noise(
    duration_seconds: float,
    amplitude: float = 0.25,
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Uniform white noise scaled to ``amplitude``. Seeded for reproducibility."""
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
    rng = np.random.default_rng(seed)
    n = int(round(duration_seconds * sample_rate))
    noise = rng.uniform(-1.0, 1.0, size=n)
    return (amplitude * noise).astype(np.float32)

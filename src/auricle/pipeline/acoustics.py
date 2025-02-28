"""Cheap acoustic summary statistics used in captioning prompts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcousticSummary:
    """Scalar descriptors of a clip, cheap enough to compute on any machine."""

    duration_seconds: float
    rms: float
    peak: float
    zero_crossing_rate: float
    centroid_hz: float

    def describe(self) -> str:
        return (
            f"duration={self.duration_seconds:.1f}s rms={self.rms:.3f} "
            f"peak={self.peak:.3f} zcr={self.zero_crossing_rate:.3f} "
            f"centroid~{self.centroid_hz:.0f}Hz"
        )


def summarize(samples: np.ndarray, sample_rate: int) -> AcousticSummary:
    """Compute an :class:`AcousticSummary` for mono float samples."""
    samples = np.asarray(samples, dtype=np.float32)
    n = len(samples)
    if n == 0:
        return AcousticSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    crossings = np.count_nonzero(np.diff(np.signbit(samples)))
    zcr = crossings / (n - 1) if n > 1 else 0.0

    power = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total = power.sum()
    centroid = float((freqs * power).sum() / total) if total > 0 else 0.0

    return AcousticSummary(n / sample_rate, rms, peak, zcr, centroid)

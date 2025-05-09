"""Sample-rate conversion helpers."""

from __future__ import annotations

import numpy as np


def resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample with linear interpolation.

    Good enough for feature extraction; not a high-quality resampler.
    """
    if orig_sr == target_sr:
        return samples.astype(np.float32)
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    duration = len(samples) / orig_sr
    n_out = int(round(duration * target_sr))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, duration, len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)

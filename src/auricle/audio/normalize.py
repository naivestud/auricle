"""Gain normalisation helpers."""

from __future__ import annotations

import numpy as np


def normalize_peak(samples: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Scale audio so its loudest sample reaches ``target``.

    Silent input is returned unchanged; a negative or zero target is an
    error. The output dtype is float32.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        return samples
    return samples * (target / peak)


def normalize_rms(samples: np.ndarray, target: float = 0.1) -> np.ndarray:
    """Scale audio so its RMS level reaches ``target``.

    Useful for levelling clips before feature extraction. Silent input is
    returned unchanged.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples**2)))
    if rms == 0.0:
        return samples
    return samples * (target / rms)

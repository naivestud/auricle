"""Log-mel spectrogram features."""

from __future__ import annotations

import numpy as np

from auricle.constants import N_FFT, N_MELS, SAMPLE_RATE


def hz_to_mel(hz: float) -> float:
    """Convert a frequency in hertz to the mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert a mel-scale value back to hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Triangular mel-spaced filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    if f_max is None:
        f_max = sample_rate / 2.0

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freqs = n_fft // 2 + 1
    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            filters[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            filters[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return filters

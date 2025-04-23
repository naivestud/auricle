"""Log-mel spectrogram features."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from auricle.constants import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert a frequency in hertz to the mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
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


class LogMelSpectrogram(nn.Module):
    """Log-mel spectrogram with whisper-style magnitude scaling.

    Input waveforms are 16 kHz mono; output shape is
    ``(batch, n_mels, frames)`` where ``frames = n_samples // hop_length``.
    """

    window: torch.Tensor
    mel_filters: torch.Tensor

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(mel_filterbank(n_mels, n_fft, sample_rate)),
            persistent=False,
        )

    @property
    def n_mels(self) -> int:
        return self.mel_filters.shape[0]

    def n_frames_for(self, n_samples: int) -> int:
        """Number of spectrogram frames produced by ``n_samples`` samples.

        The STFT uses center padding and the final frame is dropped, which
        yields exactly ``n_samples // hop_length`` frames — 3000 for a full
        30 s window at 16 kHz, matching the whisper convention.
        """
        return max(0, n_samples // self.hop_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel = self.mel_filters @ magnitudes
        # Clamp before the log so fully silent audio stays finite.
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=-1, keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

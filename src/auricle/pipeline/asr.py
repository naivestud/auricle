"""Offline transcription of whole utterances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from auricle.audio.resample import resample_linear
from auricle.audio.wav import read_wav
from auricle.constants import SAMPLE_RATE
from auricle.errors import SampleRateError

__all__ = ["resample_linear", "to_waveform", "transcribe"]


def to_waveform(
    audio: str | Path | np.ndarray | torch.Tensor, sample_rate: int | None = None
) -> torch.Tensor:
    """Normalise ``audio`` to a 16 kHz float32 mono tensor.

    Accepts a WAV path, a numpy array or a torch tensor. Arrays and tensors
    require ``sample_rate`` unless they are already at 16 kHz.
    """
    if isinstance(audio, (str, Path)):
        samples, sr = read_wav(audio)
    elif isinstance(audio, torch.Tensor):
        samples = audio.detach().cpu().numpy().astype(np.float32)
        sr = sample_rate if sample_rate is not None else SAMPLE_RATE
    else:
        samples = np.asarray(audio, dtype=np.float32)
        sr = sample_rate if sample_rate is not None else SAMPLE_RATE

    if samples.ndim != 1:
        raise SampleRateError(f"expected mono audio, got shape {samples.shape}")
    if sr != SAMPLE_RATE:
        samples = resample_linear(samples, sr, SAMPLE_RATE)
    return torch.from_numpy(np.ascontiguousarray(samples))


def transcribe(
    model, audio: str | Path | np.ndarray | torch.Tensor, sample_rate: int | None = None
) -> str:
    """Transcribe a whole utterance and return the text."""
    waveform = to_waveform(audio, sample_rate)
    texts = model.transcribe(waveform)
    return texts[0].strip()

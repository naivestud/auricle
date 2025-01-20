"""Reading WAV audio with only the standard library and numpy.

Keeping dependencies out of audio loading means the eval tooling can run on
machines without any ML stack installed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from auricle.errors import UnsupportedFormatError


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file.

    Returns a tuple ``(samples, sample_rate)`` where ``samples`` is a float32
    mono array with values in ``[-1, 1)``. Multi-channel files are averaged
    down to mono.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise UnsupportedFormatError(
            f"expected 16-bit PCM audio, got {sampwidth * 8}-bit (file: {path})"
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate

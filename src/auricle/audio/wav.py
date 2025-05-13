"""Reading WAV audio with only the standard library and numpy.

Keeping dependencies out of audio loading means the eval tooling can run on
machines without any ML stack installed.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from auricle.errors import AudioFormatError, UnsupportedFormatError


@dataclass(frozen=True, slots=True)
class WavInfo:
    """Header metadata of a WAV file, read without decoding samples."""

    n_channels: int
    sampwidth: int
    sample_rate: int
    n_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def bit_depth(self) -> int:
        return self.sampwidth * 8


def read_wav_info(path: str | Path) -> WavInfo:
    """Read only the WAV header: channels, sample rate and length.

    Much cheaper than :func:`read_wav` when callers only need metadata —
    e.g. building manifests or skipping clips that are too long.
    """
    with wave.open(str(path), "rb") as wf:
        return WavInfo(
            n_channels=wf.getnchannels(),
            sampwidth=wf.getsampwidth(),
            sample_rate=wf.getframerate(),
            n_frames=wf.getnframes(),
        )


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


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples in ``[-1, 1]`` as a 16-bit PCM WAV file."""
    path = Path(path)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise AudioFormatError(f"write_wav expects mono audio, got shape {samples.shape}")

    pcm = np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

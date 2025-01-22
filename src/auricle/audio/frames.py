"""Split audio into frames and overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    """A half-open sample range ``[start, end)`` into an audio array."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def take(self, samples: np.ndarray) -> np.ndarray:
        """Return the slice of ``samples`` covered by this chunk."""
        return samples[self.start : self.end]


def frame_starts(n_samples: int, frame: int, hop: int) -> list[int]:
    """Start positions of ``frame``-sized windows spaced ``hop`` apart.

    Windows that would extend past the end of the audio are dropped, so a
    caller that needs to cover trailing samples must pad beforehand.
    """
    if frame <= 0:
        raise ValueError(f"frame must be positive, got {frame}")
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    if n_samples < frame:
        return []
    return list(range(0, n_samples - frame + 1, hop))


def overlap_chunks(n_samples: int, chunk: int, overlap: int = 0) -> list[Chunk]:
    """Plan chunks of ``chunk`` samples with ``overlap`` shared samples.

    The chunks always cover ``[0, n_samples)``; the final chunk may be
    shorter than ``chunk``.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if overlap < 0 or overlap >= chunk:
        raise ValueError(f"overlap must be in [0, chunk), got {overlap}")
    if n_samples <= 0:
        return []

    step = chunk - overlap
    chunks: list[Chunk] = []
    for start in range(0, n_samples, step):
        end = min(start + chunk, n_samples)
        chunks.append(Chunk(start, end))
        if end == n_samples:
            break
    return chunks

"""Chunk scheduling for streaming audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auricle.constants import SAMPLE_RATE


@dataclass(frozen=True)
class StreamChunk:
    """A chunk emitted by the scheduler with an absolute sample offset."""

    samples: np.ndarray
    start: int


class StreamScheduler:
    """Accumulates audio and emits fixed-size overlapping chunks.

    Every emitted chunk is ``chunk_seconds`` long except possibly the final
    flushed chunk, and overlaps the previous one by ``overlap_seconds`` so
    words on a boundary are seen twice by the encoder.
    """

    def __init__(
        self,
        chunk_seconds: float = 2.0,
        overlap_seconds: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
    ):
        if chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")
        if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
            raise ValueError(
                f"overlap_seconds must be in [0, chunk_seconds), got {overlap_seconds}"
            )
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples
        self._buffer = np.zeros(0, dtype=np.float32)
        self._emitted = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, samples: np.ndarray) -> list[StreamChunk]:
        """Feed samples and return every chunk that becomes ready."""
        self._buffer = np.concatenate([self._buffer, np.asarray(samples, dtype=np.float32)])
        chunks: list[StreamChunk] = []
        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[: self.chunk_samples]
            chunks.append(StreamChunk(chunk.copy(), self._emitted))
            self._buffer = self._buffer[self.step_samples :]
            self._emitted += self.step_samples
        return chunks

    def flush(self) -> StreamChunk | None:
        """Emit the trailing partial chunk, if any audio remains."""
        if len(self._buffer) == 0:
            return None
        chunk = StreamChunk(self._buffer.copy(), self._emitted)
        self._buffer = np.zeros(0, dtype=np.float32)
        return chunk

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._emitted = 0

"""Chunk scheduling for streaming audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auricle.constants import SAMPLE_RATE


@dataclass(frozen=True, slots=True)
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
        # A growable buffer with a write cursor. Pre-allocating and appending
        # in place avoids re-copying the whole backlog on every push, which a
        # naive ``np.concatenate`` would do.
        self._buffer = np.zeros(self.chunk_samples, dtype=np.float32)
        self._n = 0
        self._emitted = 0

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return (
            f"<StreamScheduler chunk={self.chunk_samples} "
            f"overlap={self.overlap_samples} buffered={self._n}>"
        )

    @property
    def pending_seconds(self) -> float:
        """Seconds of audio buffered but not yet emitted as a chunk."""
        return self._n / self.sample_rate

    @property
    def samples_emitted(self) -> int:
        """Absolute sample offset of the next chunk's start."""
        return self._emitted

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= len(self._buffer):
            return
        capacity = max(needed, 2 * len(self._buffer))
        grown = np.zeros(capacity, dtype=np.float32)
        grown[: self._n] = self._buffer[: self._n]
        self._buffer = grown

    def push(self, samples: np.ndarray) -> list[StreamChunk]:
        """Feed samples and return every chunk that becomes ready."""
        samples = np.asarray(samples, dtype=np.float32)
        self._ensure_capacity(self._n + samples.size)
        self._buffer[self._n : self._n + samples.size] = samples
        self._n += samples.size

        chunks: list[StreamChunk] = []
        while self._n >= self.chunk_samples:
            chunks.append(StreamChunk(self._buffer[: self.chunk_samples].copy(), self._emitted))
            # Slide the not-yet-emitted tail to the front of the buffer.
            remaining = self._n - self.step_samples
            self._buffer[:remaining] = self._buffer[self.step_samples : self._n]
            self._n = remaining
            self._emitted += self.step_samples
        return chunks

    def flush(self) -> StreamChunk | None:
        """Emit the trailing partial chunk, if any audio remains.

        If the remainder is no longer than the overlap, it was already fully
        contained in the previous chunk, so re-decoding it would only repeat
        text; in that case nothing is emitted.
        """
        if self._n == 0 or self._n <= self.overlap_samples:
            return None
        chunk = StreamChunk(self._buffer[: self._n].copy(), self._emitted)
        self._n = 0
        return chunk

    def reset(self) -> None:
        self._n = 0
        self._emitted = 0

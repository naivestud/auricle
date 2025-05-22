"""Streaming ASR over chunked audio."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from auricle.streaming.merge import merge_transcripts
from auricle.streaming.scheduler import StreamScheduler
from auricle.types import ModelLike


@dataclass(slots=True)
class StreamStats:
    """Counters describing the work a :class:`StreamingASR` has done."""

    samples_fed: int = 0
    chunks_decoded: int = 0
    decode_seconds: float = 0.0

    @property
    def audio_seconds(self) -> float:
        return self.samples_fed / 16_000.0

    @property
    def real_time_factor(self) -> float | None:
        """Decode time over audio duration; ``None`` before any audio is fed.

        Values below 1.0 mean decoding runs faster than real time.
        """
        if self.samples_fed == 0:
            return None
        return self.decode_seconds / self.audio_seconds


class StreamingASR:
    """Incrementally transcribe audio as it arrives.

    Feed audio in arbitrary block sizes with :meth:`feed`; the scheduler
    turns it into fixed-overlap encoder windows and the transcript grows
    monotonically. Call :meth:`finalize` once the stream ends.
    """

    def __init__(self, model: ModelLike, chunk_seconds: float = 2.0, overlap_seconds: float = 0.5):
        self.model = model
        self.scheduler = StreamScheduler(chunk_seconds, overlap_seconds)
        self._committed = ""
        self.stats = StreamStats()

    @property
    def text(self) -> str:
        """The transcript committed so far."""
        return self._committed

    def feed(self, samples: np.ndarray) -> str:
        """Feed a block of audio and return the committed transcript."""
        self.stats.samples_fed += int(len(samples))
        for chunk in self.scheduler.push(samples):
            self._integrate(chunk.samples)
        return self._committed

    def finalize(self) -> str:
        """Flush the trailing partial chunk and return the final transcript."""
        tail = self.scheduler.flush()
        if tail is not None:
            self._integrate(tail.samples)
        return self._committed

    def process(self, blocks: Iterable[np.ndarray]) -> Iterator[str]:
        """Consume an iterable of audio blocks, yielding the transcript."""
        for block in blocks:
            yield self.feed(block)
        yield self.finalize()

    def reset(self) -> None:
        self.scheduler.reset()
        self._committed = ""
        self.stats = StreamStats()

    def _integrate(self, samples: np.ndarray) -> None:
        waveform = torch.from_numpy(np.ascontiguousarray(samples))
        start = time.perf_counter()
        hypothesis = self.model.transcribe(waveform)[0]
        self.stats.decode_seconds += time.perf_counter() - start
        self.stats.chunks_decoded += 1
        self._committed = merge_transcripts(self._committed, hypothesis)

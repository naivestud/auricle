"""Streaming ASR over chunked audio."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch

from auricle.streaming.merge import merge_transcripts
from auricle.streaming.scheduler import StreamScheduler


class StreamingASR:
    """Incrementally transcribe audio as it arrives.

    Feed audio in arbitrary block sizes with :meth:`feed`; the scheduler
    turns it into fixed-overlap encoder windows and the transcript grows
    monotonically. Call :meth:`finalize` once the stream ends.
    """

    def __init__(self, model, chunk_seconds: float = 2.0, overlap_seconds: float = 0.5):
        self.model = model
        self.scheduler = StreamScheduler(chunk_seconds, overlap_seconds)
        self._committed = ""

    @property
    def text(self) -> str:
        """The transcript committed so far."""
        return self._committed

    def feed(self, samples: np.ndarray) -> str:
        """Feed a block of audio and return the committed transcript."""
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

    def _integrate(self, samples: np.ndarray) -> None:
        waveform = torch.from_numpy(np.ascontiguousarray(samples))
        hypothesis = self.model.transcribe(waveform)[0]
        self._committed = merge_transcripts(self._committed, hypothesis)

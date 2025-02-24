import numpy as np
import pytest

from auricle.streaming.scheduler import StreamScheduler


def test_no_chunk_until_full():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.25)
    assert sched.push(np.zeros(8_000, dtype=np.float32)) == []
    chunks = sched.push(np.zeros(8_000, dtype=np.float32))
    assert len(chunks) == 1
    assert chunks[0].samples.shape == (16_000,)


def test_chunk_offsets_step_by_stride():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.25)
    audio = np.zeros(60_000, dtype=np.float32)
    chunks = sched.push(audio)
    starts = [c.start for c in chunks]
    step = sched.step_samples
    assert starts == [0, step, 2 * step, 3 * step]


def test_push_and_flush_cover_stream():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.25)
    audio = np.arange(35_000, dtype=np.float32)
    chunks = sched.push(audio)
    tail = sched.flush()
    assert tail is not None
    assert chunks[0].start == 0
    # The final flushed chunk reaches the end of the stream.
    assert tail.start + len(tail.samples) == len(audio)


def test_flush_empty_returns_none():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.0)
    sched.push(np.zeros(16_000, dtype=np.float32))
    assert sched.flush() is None


def test_zero_overlap_tiles_exactly():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.0)
    chunks = sched.push(np.zeros(32_000, dtype=np.float32))
    assert [c.start for c in chunks] == [0, 16_000]
    assert sched.flush() is None


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        StreamScheduler(chunk_seconds=1.0, overlap_seconds=1.0)
    with pytest.raises(ValueError):
        StreamScheduler(chunk_seconds=1.0, overlap_seconds=-0.1)


def test_reset_clears_state():
    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.0)
    sched.push(np.zeros(10_000, dtype=np.float32))
    sched.reset()
    assert len(sched) == 0
    assert sched.flush() is None

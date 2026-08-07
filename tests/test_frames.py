import numpy as np
import pytest

from auricle.audio.frames import Chunk, frame_starts, overlap_chunks


def test_frame_starts_basic():
    assert frame_starts(10, frame=4, hop=2) == [0, 2, 4, 6]


def test_frame_starts_drops_trailing_partial():
    # 11 samples: window [8, 12) would overflow, so only starts 0, 2, 4, 6.
    assert frame_starts(11, frame=4, hop=2) == [0, 2, 4, 6]


def test_frame_starts_too_short():
    assert frame_starts(3, frame=4, hop=2) == []


@pytest.mark.parametrize("frame,hop", [(0, 2), (4, 0), (-1, 2)])
def test_frame_starts_rejects_bad_sizes(frame, hop):
    with pytest.raises(ValueError):
        frame_starts(10, frame=frame, hop=hop)


def test_overlap_chunks_cover_audio():
    chunks = overlap_chunks(100, chunk=30, overlap=10)
    assert chunks[0].start == 0
    assert chunks[-1].end == 100
    # No gap between consecutive chunks.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start <= prev.end


def test_overlap_chunks_single_chunk():
    assert overlap_chunks(10, chunk=30) == [Chunk(0, 10)]


def test_overlap_chunks_empty():
    assert overlap_chunks(0, chunk=30) == []


def test_overlap_chunks_exact_multiple():
    chunks = overlap_chunks(60, chunk=30)
    assert chunks == [Chunk(0, 30), Chunk(30, 60)]


@pytest.mark.parametrize("chunk,overlap", [(0, 0), (30, -1), (30, 30), (30, 40)])
def test_overlap_chunks_rejects_bad_args(chunk, overlap):
    with pytest.raises(ValueError):
        overlap_chunks(100, chunk=chunk, overlap=overlap)


def test_chunk_take():
    samples = np.arange(10)
    assert np.array_equal(Chunk(2, 5).take(samples), np.array([2, 3, 4]))

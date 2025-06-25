import numpy as np
import torch

from auricle.model import AuricleModel
from auricle.streaming.asr import StreamingASR
from auricle.streaming.merge import merge_transcripts


def test_merge_empty_committed():
    assert merge_transcripts("", "hello world") == "hello world"


def test_merge_empty_hypothesis():
    assert merge_transcripts("hello world", "") == "hello world"


def test_merge_no_overlap_concatenates():
    assert merge_transcripts("hello", "world") == "hello world"


def test_merge_dedupes_overlapping_words():
    committed = "the quick brown fox"
    hypothesis = "brown fox jumps over"
    assert merge_transcripts(committed, hypothesis) == "the quick brown fox jumps over"


def test_merge_full_overlap_no_growth():
    committed = "a b c"
    hypothesis = "a b c"
    assert merge_transcripts(committed, hypothesis) == "a b c"


def test_merge_respects_window():
    # Overlap longer than the window is not detected, so it concatenates.
    committed = " ".join(f"w{i}" for i in range(20))
    hypothesis = committed  # full repeat, but window defaults to 12 words
    merged = merge_transcripts(committed, hypothesis)
    assert len(merged.split()) > 20


class FakeModel:
    """Returns a fixed transcript regardless of input."""

    def __init__(self, text):
        self._text = text

    def transcribe(self, waveform):
        return [self._text]


def test_streaming_grows_and_is_stable():
    asr = StreamingASR(FakeModel("alpha beta gamma"), chunk_seconds=1.0, overlap_seconds=0.25)
    audio = np.zeros(40_000, dtype=np.float32)
    texts = list(asr.process([audio[:20_000], audio[20_000:]]))
    # Every yielded transcript is a prefix-stable growth of the final one.
    final = texts[-1]
    for t in texts:
        assert final.startswith(t)
    assert "alpha beta gamma" in final


def test_streaming_reset_clears():
    asr = StreamingASR(FakeModel("x"), chunk_seconds=1.0, overlap_seconds=0.0)
    asr.feed(np.zeros(20_000, dtype=np.float32))
    asr.reset()
    assert asr.text == ""


def test_streaming_with_real_model_deterministic():
    torch.manual_seed(11)
    model = AuricleModel.tiny()
    audio = np.random.default_rng(3).standard_normal(32_000).astype(np.float32)

    a = StreamingASR(model, chunk_seconds=1.0, overlap_seconds=0.25)
    b = StreamingASR(model, chunk_seconds=1.0, overlap_seconds=0.25)
    assert a.feed(audio) == b.feed(audio)
    assert a.finalize() == b.finalize()


def test_streaming_independent_of_push_block_size():
    # The scheduler cuts chunks from cumulative samples, so how the caller
    # slices the push() calls must not change the final transcript.
    torch.manual_seed(17)
    model = AuricleModel.tiny()
    audio = np.random.default_rng(5).standard_normal(40_000).astype(np.float32)

    results = []
    for block in (4_000, 7_000, 40_000):
        asr = StreamingASR(model, chunk_seconds=1.0, overlap_seconds=0.25)
        for start in range(0, len(audio), block):
            asr.feed(audio[start : start + block])
        results.append(asr.finalize())

    assert results[0] == results[1] == results[2]


def test_stats_track_samples_and_chunks():
    asr = StreamingASR(FakeModel("x"), chunk_seconds=1.0, overlap_seconds=0.0)
    assert asr.stats.samples_fed == 0
    assert asr.stats.real_time_factor is None

    asr.feed(np.zeros(32_000, dtype=np.float32))
    asr.finalize()

    assert asr.stats.samples_fed == 32_000
    assert asr.stats.audio_seconds == 2.0
    assert asr.stats.chunks_decoded == 2
    assert asr.stats.decode_seconds >= 0.0
    assert asr.stats.real_time_factor is not None


def test_stats_reset_with_stream():
    asr = StreamingASR(FakeModel("x"), chunk_seconds=1.0, overlap_seconds=0.0)
    asr.feed(np.zeros(16_000, dtype=np.float32))
    asr.reset()
    assert asr.stats.samples_fed == 0
    assert asr.stats.chunks_decoded == 0


def test_merge_whitespace_only_hypothesis_keeps_committed():
    assert merge_transcripts("hello", "   ") == "hello"


def test_merge_whitespace_only_committed_returns_hypothesis():
    assert merge_transcripts("   ", "hello") == "hello"


def test_merge_custom_window_detects_longer_overlap():
    committed = "one two three four five"
    hypothesis = "four five six seven"
    # Default window is plenty here, but shrinking it below 2 misses the overlap.
    assert merge_transcripts(committed, hypothesis, window=1) == (
        "one two three four five four five six seven"
    )
    assert merge_transcripts(committed, hypothesis, window=5) == "one two three four five six seven"


def test_scheduler_push_after_flush_continues():
    from auricle.streaming.scheduler import StreamScheduler

    sched = StreamScheduler(chunk_seconds=1.0, overlap_seconds=0.0)
    sched.push(np.zeros(20_000, dtype=np.float32))
    tail = sched.flush()
    assert tail is not None
    end_after_flush = tail.start + len(tail.samples)
    # Feeding more audio after a flush keeps absolute offsets monotonic.
    chunks = sched.push(np.zeros(16_000, dtype=np.float32))
    assert len(chunks) == 1
    assert chunks[0].start >= end_after_flush - sched.chunk_samples


def test_streaming_finalize_is_idempotent():
    asr = StreamingASR(FakeModel("alpha"), chunk_seconds=1.0, overlap_seconds=0.0)
    asr.feed(np.zeros(20_000, dtype=np.float32))
    first = asr.finalize()
    second = asr.finalize()  # nothing left to flush
    assert first == second

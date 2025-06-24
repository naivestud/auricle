from pathlib import Path

import pytest
import torch

from auricle.llm import EchoBackend
from auricle.model import AuricleModel
from auricle.pipeline.acoustics import AcousticSummary, summarize
from auricle.pipeline.caption import build_caption_prompt, caption_audio

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(5)
    return AuricleModel.tiny()


def test_summarize_tone():
    import numpy as np

    t = np.arange(16_000) / 16_000.0
    tone = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    summary = summarize(tone, 16_000)
    assert summary.duration_seconds == pytest.approx(1.0)
    assert summary.rms > 0
    # Spectral centroid of a 1 kHz tone should be near 1 kHz.
    assert 800 < summary.centroid_hz < 1200


def test_summarize_empty():
    import numpy as np

    summary = summarize(np.zeros(0, dtype=np.float32), 16_000)
    assert summary == AcousticSummary(0.0, 0.0, 0.0, 0.0, 0.0)


def test_summarize_single_sample():
    import numpy as np

    summary = summarize(np.array([0.5], dtype=np.float32), 16_000)
    assert summary.duration_seconds == pytest.approx(1 / 16_000)
    assert summary.rms == pytest.approx(0.5)
    assert summary.peak == pytest.approx(0.5)
    assert summary.zero_crossing_rate == 0.0  # no pair to compare


def test_summarize_full_scale_sine_rms():
    import numpy as np

    t = np.arange(16_000) / 16_000.0
    full = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    summary = summarize(full, 16_000)
    # RMS of a unit sine is 1/sqrt(2).
    assert summary.rms == pytest.approx(2**-0.5, rel=1e-2)
    assert summary.peak == pytest.approx(1.0, rel=1e-2)


def test_describe_contains_every_field():
    summary = AcousticSummary(1.5, 0.25, 0.9, 0.1, 1200.0)
    text = summary.describe()
    assert "duration=1.5s" in text
    assert "rms=0.250" in text
    assert "peak=0.900" in text
    assert "zcr=0.100" in text
    assert "centroid~1200Hz" in text


def test_build_prompt_handles_empty_transcript():
    prompt = build_caption_prompt("duration=1.0s", "")
    assert "<none>" in prompt


def test_caption_audio_with_echo(model):
    backend = EchoBackend()
    caption = caption_audio(model, backend, FIXTURES / "tone_1s.wav")
    # The echo backend returns the prompt, which includes the acoustic summary.
    assert "duration=" in caption
    assert "Caption:" in caption


def test_caption_audio_deterministic(model):
    backend = EchoBackend()
    a = caption_audio(model, backend, FIXTURES / "sweep_2s.wav")
    b = caption_audio(model, backend, FIXTURES / "sweep_2s.wav")
    assert a == b

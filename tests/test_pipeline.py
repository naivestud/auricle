from pathlib import Path

import numpy as np
import pytest
import torch

from auricle.errors import SampleRateError
from auricle.model import AuricleModel
from auricle.pipeline.asr import (
    resample_linear,
    to_waveform,
    transcribe,
    transcribe_with_confidence,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(7)
    return AuricleModel.tiny()


def test_transcribe_from_path(model):
    text = transcribe(model, FIXTURES / "tone_1s.wav")
    assert isinstance(text, str)


def test_transcribe_deterministic(model):
    a = transcribe(model, FIXTURES / "sweep_2s.wav")
    b = transcribe(model, FIXTURES / "sweep_2s.wav")
    assert a == b


def test_to_waveform_from_array(model):
    samples = np.zeros(8_000, dtype=np.float32)
    waveform = to_waveform(samples, sample_rate=16_000)
    assert waveform.dtype == torch.float32
    assert waveform.shape == (8_000,)


def test_to_waveform_resamples(model):
    samples = np.zeros(4_000, dtype=np.float32)
    waveform = to_waveform(samples, sample_rate=8_000)
    assert waveform.shape == (8_000,)


def test_to_waveform_rejects_stereo():
    with pytest.raises(SampleRateError):
        to_waveform(np.zeros((100, 2), dtype=np.float32), sample_rate=16_000)


def test_resample_linear_identity():
    samples = np.random.default_rng(0).standard_normal(1_000).astype(np.float32)
    out = resample_linear(samples, 16_000, 16_000)
    assert np.array_equal(out, samples)


def test_resample_linear_length():
    samples = np.ones(8_000, dtype=np.float32)
    assert resample_linear(samples, 8_000, 16_000).shape == (16_000,)
    assert resample_linear(samples, 16_000, 8_000).shape == (4_000,)


def test_resample_linear_empty():
    assert resample_linear(np.zeros(0, dtype=np.float32), 8_000, 16_000).shape == (0,)


def test_transcribe_with_confidence_returns_bounded_score(model):
    text, confidence = transcribe_with_confidence(model, FIXTURES / "tone_1s.wav")
    assert text == transcribe(model, FIXTURES / "tone_1s.wav")
    assert 0.0 <= confidence <= 1.0


def test_transcribe_with_confidence_accepts_array(model):
    samples = np.zeros(8_000, dtype=np.float32)
    text, confidence = transcribe_with_confidence(model, samples, sample_rate=16_000)
    assert isinstance(text, str)
    assert 0.0 <= confidence <= 1.0


def test_model_transcribe_with_confidence_batch():
    torch.manual_seed(3)
    model = AuricleModel.tiny()
    pairs = model.transcribe_with_confidence(torch.randn(2, 4_000))
    assert len(pairs) == 2
    texts = model.transcribe(torch.randn(2, 4_000))
    assert len(texts) == len(pairs)

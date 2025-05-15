import numpy as np
import pytest

from auricle.audio.normalize import normalize_peak, normalize_rms


def test_normalize_peak_reaches_target():
    samples = np.array([0.1, -0.25, 0.05], dtype=np.float32)
    out = normalize_peak(samples, target=1.0)
    assert np.max(np.abs(out)) == pytest.approx(1.0)


def test_normalize_peak_custom_target():
    samples = np.array([0.5, -0.5], dtype=np.float32)
    out = normalize_peak(samples, target=0.25)
    assert np.max(np.abs(out)) == pytest.approx(0.25)


def test_normalize_peak_preserves_shape_and_sign():
    samples = np.array([0.2, -0.4, 0.0], dtype=np.float32)
    out = normalize_peak(samples)
    assert out.shape == samples.shape
    assert out[1] < 0


def test_normalize_peak_silent_unchanged():
    samples = np.zeros(8, dtype=np.float32)
    assert np.array_equal(normalize_peak(samples), samples)


def test_normalize_peak_rejects_bad_target():
    with pytest.raises(ValueError):
        normalize_peak(np.ones(4, dtype=np.float32), target=0.0)


def test_normalize_rms_reaches_target():
    rng = np.random.default_rng(0)
    samples = (0.01 * rng.standard_normal(16_000)).astype(np.float32)
    out = normalize_rms(samples, target=0.1)
    rms = float(np.sqrt(np.mean(out**2)))
    assert rms == pytest.approx(0.1, rel=1e-5)


def test_normalize_rms_silent_unchanged():
    samples = np.zeros(16, dtype=np.float32)
    assert np.array_equal(normalize_rms(samples), samples)


def test_normalize_rms_empty():
    out = normalize_rms(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,)


def test_normalize_rms_rejects_bad_target():
    with pytest.raises(ValueError):
        normalize_rms(np.ones(4, dtype=np.float32), target=-0.5)

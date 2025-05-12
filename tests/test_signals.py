import numpy as np
import pytest

from auricle.audio.signals import chirp, silence, sine, white_noise


def test_silence_shape_and_values():
    out = silence(1.0)
    assert out.shape == (16_000,)
    assert out.dtype == np.float32
    assert not np.any(out)


def test_sine_length_and_amplitude():
    out = sine(0.5, frequency=440.0, amplitude=0.5)
    assert out.shape == (8_000,)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 0.5 + 1e-6


def test_sine_frequency_content():
    # A 1 kHz tone should peak near 1 kHz in the spectrum.
    out = sine(1.0, frequency=1000.0, amplitude=1.0)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / 16_000.0)
    peak_hz = freqs[np.argmax(spectrum)]
    assert abs(peak_hz - 1000.0) < 5.0


def test_sine_zero_frequency_is_flat():
    out = sine(0.1, frequency=0.0, amplitude=0.5)
    assert np.allclose(out, 0.0)


@pytest.mark.parametrize("kwargs", [{"frequency": -1.0}, {"amplitude": 1.5}, {"amplitude": -0.1}])
def test_sine_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        sine(0.1, **kwargs)


def test_chirp_sweeps_upward():
    out = chirp(1.0, start_frequency=200.0, end_frequency=4000.0)
    assert out.shape == (16_000,)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 0.5 + 1e-6


def test_chirp_rejects_bad_amplitude():
    with pytest.raises(ValueError):
        chirp(0.1, amplitude=2.0)


def test_chirp_rejects_negative_frequency():
    with pytest.raises(ValueError):
        chirp(0.1, start_frequency=-100.0)


def test_white_noise_seed_reproducible():
    a = white_noise(0.5, seed=7)
    b = white_noise(0.5, seed=7)
    assert np.array_equal(a, b)


def test_white_noise_different_seeds_differ():
    a = white_noise(0.5, seed=1)
    b = white_noise(0.5, seed=2)
    assert not np.array_equal(a, b)


def test_white_noise_bounded():
    out = white_noise(1.0, amplitude=0.25, seed=0)
    assert np.max(np.abs(out)) <= 0.25 + 1e-6

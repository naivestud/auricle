import numpy as np
import pytest

from auricle.audio.wav import WavInfo, read_wav, read_wav_info, write_wav
from auricle.errors import AudioFormatError, UnsupportedFormatError


def test_write_then_read_roundtrip(tmp_path):
    rate = 16_000
    t = np.linspace(0.0, 1.0, rate, endpoint=False, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    path = tmp_path / "tone.wav"
    write_wav(path, tone, rate)
    samples, sr = read_wav(path)

    assert sr == rate
    assert samples.shape == tone.shape
    assert samples.dtype == np.float32
    # 16-bit quantization bounds the roundtrip error.
    assert np.max(np.abs(samples - tone)) < 1.0 / 32768.0 + 1e-6


def test_read_empty_wav(tmp_path):
    path = tmp_path / "empty.wav"
    write_wav(path, np.zeros(0, dtype=np.float32), 16_000)
    samples, sr = read_wav(path)
    assert sr == 16_000
    assert samples.shape == (0,)


def test_write_rejects_multidimensional(tmp_path):
    with pytest.raises(AudioFormatError):
        write_wav(tmp_path / "bad.wav", np.zeros((2, 8), dtype=np.float32), 16_000)


def test_read_rejects_wrong_bit_depth(tmp_path):
    import wave

    path = tmp_path / "pcm8.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8-bit, unsupported
        wf.setframerate(16_000)
        wf.writeframes(b"\x00" * 16)

    with pytest.raises(UnsupportedFormatError):
        read_wav(path)


def test_read_stereo_downmixes(tmp_path):
    import wave

    rate = 8_000
    frames = np.zeros((rate, 2), dtype=np.int16)
    frames[:, 0] = 32767
    frames[:, 1] = -32768
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames.tobytes())

    samples, sr = read_wav(path)
    assert sr == rate
    assert samples.shape == (rate,)
    # Average of full-scale positive and negative channels is ~0.
    assert np.max(np.abs(samples)) < 1.0 / 32768.0 + 1e-6


def test_write_odd_length_roundtrip(tmp_path):
    path = tmp_path / "odd.wav"
    tone = np.ones(101, dtype=np.float32) * 0.25
    write_wav(path, tone, 16_000)
    samples, _ = read_wav(path)
    assert samples.shape == (101,)


def test_write_clips_out_of_range(tmp_path):
    path = tmp_path / "clip.wav"
    loud = np.array([2.0, -2.0], dtype=np.float32)
    write_wav(path, loud, 16_000)
    samples, _ = read_wav(path)
    assert samples.max() <= 32767.0 / 32768.0
    assert samples.min() >= -1.0


def test_read_wav_info_matches_decode(tmp_path):
    path = tmp_path / "info.wav"
    tone = np.zeros(24_000, dtype=np.float32)
    write_wav(path, tone, 16_000)

    info = read_wav_info(path)
    assert isinstance(info, WavInfo)
    assert info.n_channels == 1
    assert info.sample_rate == 16_000
    assert info.n_frames == 24_000
    assert info.duration_seconds == pytest.approx(1.5)
    assert info.bit_depth == 16

    samples, sr = read_wav(path)
    assert len(samples) == info.n_frames
    assert sr == info.sample_rate


def test_read_wav_info_empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    write_wav(path, np.zeros(0, dtype=np.float32), 8_000)
    info = read_wav_info(path)
    assert info.n_frames == 0
    assert info.duration_seconds == 0.0
    assert info.sample_rate == 8_000


def test_write_rejects_nan(tmp_path):
    bad = np.array([0.1, float("nan"), 0.2], dtype=np.float32)
    with pytest.raises(AudioFormatError, match="non-finite"):
        write_wav(tmp_path / "nan.wav", bad, 16_000)


def test_write_rejects_inf(tmp_path):
    bad = np.array([0.1, float("inf")], dtype=np.float32)
    with pytest.raises(AudioFormatError, match="non-finite"):
        write_wav(tmp_path / "inf.wav", bad, 16_000)

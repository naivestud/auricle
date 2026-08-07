import numpy as np
import torch

from auricle.audio.spectrogram import LogMelSpectrogram, hz_to_mel, mel_filterbank, mel_to_hz


def test_mel_conversions_roundtrip():
    for hz in [0.0, 100.0, 440.0, 4000.0, 8000.0]:
        assert abs(mel_to_hz(hz_to_mel(hz)) - hz) < 1e-6


def test_filterbank_shape_and_support():
    fb = mel_filterbank(n_mels=40, n_fft=400, sample_rate=16_000)
    assert fb.shape == (40, 201)
    assert fb.min() >= 0.0
    # Every filter has some nonzero support.
    assert (fb.sum(axis=1) > 0).all()


def test_filterbank_peak_within_band():
    fb = mel_filterbank(n_mels=10, n_fft=400, sample_rate=16_000)
    for row in fb:
        if row.max() > 0:
            assert row.max() <= 1.0 + 1e-6


def test_spectrogram_shape():
    spec = LogMelSpectrogram(n_mels=40)
    waveform = torch.randn(16_000)  # 1 second
    out = spec(waveform)
    assert out.shape == (1, 40, 100)  # 1s at 10ms hop -> 100 frames


def test_spectrogram_batch():
    spec = LogMelSpectrogram(n_mels=40)
    waveform = torch.randn(3, 8_000)
    out = spec(waveform)
    assert out.shape == (3, 40, 50)


def test_spectrogram_deterministic():
    spec = LogMelSpectrogram(n_mels=40)
    waveform = torch.randn(4_800)
    a = spec(waveform)
    b = spec(waveform)
    assert torch.equal(a, b)


def test_n_frames_for():
    spec = LogMelSpectrogram()
    assert spec.n_frames_for(16_000) == 100
    assert spec.n_frames_for(0) == 0
    assert spec.n_frames_for(159) == 0


def test_n_frames_exact_hop_multiple():
    # A full 30 s whisper window must produce exactly 3000 frames.
    spec = LogMelSpectrogram()
    assert spec.n_frames_for(480_000) == 3000
    out = spec(torch.randn(3_200))  # 0.2 s = 20 frames
    assert out.shape[-1] == 20


def test_spectrogram_silence_is_finite():
    spec = LogMelSpectrogram(n_mels=40)
    out = spec(torch.zeros(16_000))
    assert torch.isfinite(out).all()


def test_spectrogram_concentrates_energy():
    # A 440 Hz tone should put most energy in the low mel bands.
    spec = LogMelSpectrogram(n_mels=40)
    t = np.arange(16_000) / 16_000.0
    tone = torch.tensor(0.5 * np.sin(2 * np.pi * 440.0 * t), dtype=torch.float32)
    out = spec(tone)[0]
    band_energy = out.mean(dim=1)
    low = band_energy[:10].mean()
    high = band_energy[25:].mean()
    assert low > high

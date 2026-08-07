"""Shared constants for audio processing.

auricle works exclusively with 16 kHz mono audio internally, mirroring the
conventions of whisper-style encoders.
"""

SAMPLE_RATE = 16_000
"""Reference sample rate. Everything is resampled to this before modeling."""

N_FFT = 400
"""FFT window size in samples (25 ms at 16 kHz)."""

HOP_LENGTH = 160
"""Hop between STFT windows in samples (10 ms at 16 kHz)."""

WINDOW_SECONDS = 0.025
HOP_SECONDS = 0.01

N_MELS = 80
"""Number of mel bands in the spectrogram."""

CHUNK_SECONDS = 30.0
"""Maximum audio window a single encoder pass consumes."""

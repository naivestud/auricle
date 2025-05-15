"""Audio I/O and framing utilities."""

from auricle.audio.normalize import normalize_peak, normalize_rms, trim_silence
from auricle.audio.resample import resample_linear
from auricle.audio.signals import chirp, silence, sine, white_noise
from auricle.audio.spectrogram import LogMelSpectrogram
from auricle.audio.wav import WavInfo, read_wav, read_wav_info, write_wav

__all__ = [
    "LogMelSpectrogram",
    "WavInfo",
    "chirp",
    "normalize_peak",
    "normalize_rms",
    "read_wav",
    "read_wav_info",
    "resample_linear",
    "silence",
    "sine",
    "trim_silence",
    "white_noise",
    "write_wav",
]

"""Automatic speech recognition: vocabulary and decoding."""

from auricle.asr.ctc import CTCHead, greedy_decode
from auricle.asr.vocab import CharVocabulary

__all__ = ["CTCHead", "CharVocabulary", "greedy_decode"]

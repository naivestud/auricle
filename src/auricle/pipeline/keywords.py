"""Keyword spotting over transcribed audio.

A lightweight alternative to wake-word models: transcribe the clip, then
search the normalised text for any of a set of keywords or phrases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from auricle.eval.metrics import normalize_text
from auricle.pipeline.asr import to_waveform, transcribe_waveform
from auricle.types import ModelLike


def find_keywords(transcript: str, keywords: list[str]) -> list[str]:
    """Return the keywords present in ``transcript``.

    Both sides are lowercased and stripped of punctuation before matching, so
    ``"Don't stop"`` matches the keyword ``"dont stop"``. Order follows the
    ``keywords`` list; a keyword appears at most once.
    """
    if not keywords:
        raise ValueError("keywords must not be empty")
    haystack = normalize_text(transcript)
    found = []
    for keyword in keywords:
        needle = normalize_text(keyword)
        if needle and needle in haystack:
            found.append(keyword)
    return found


def spot_keywords(
    model: ModelLike,
    audio: str | Path | np.ndarray | torch.Tensor,
    keywords: list[str],
    sample_rate: int | None = None,
) -> list[str]:
    """Transcribe ``audio`` and return which ``keywords`` it contains."""
    waveform = to_waveform(audio, sample_rate)
    transcript = transcribe_waveform(model, waveform)
    return find_keywords(transcript, keywords)

"""Evaluation metrics for speech recognition."""

from __future__ import annotations

import string
from collections.abc import Sequence

_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Apostrophes survive so contractions stay one word.
    """
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(text.split())


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance between two sequences."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def _score_words(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        raise ValueError("reference is empty after normalisation")
    return edit_distance(ref_words, hyp_words), len(ref_words)


def _score_chars(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        raise ValueError("reference is empty after normalisation")
    return edit_distance(ref_chars, hyp_chars), len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over reference length.

    Both sides are normalised before scoring. WER can exceed 1.0 when the
    hypothesis has many insertions.
    """
    distance, length = _score_words(reference, hypothesis)
    return distance / length


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, ignoring spaces after normalisation."""
    distance, length = _score_chars(reference, hypothesis)
    return distance / length

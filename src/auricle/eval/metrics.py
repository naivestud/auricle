"""Evaluation metrics for speech recognition."""

from __future__ import annotations

import string
from collections.abc import Sequence

import numpy as np

_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Apostrophes survive so contractions stay one word.
    """
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(text.split())


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance between two sequences.

    Identical and empty inputs short-circuit, and the row updates are
    vectorised with numpy, which is noticeably faster than a pure-Python
    double loop on transcript-length sequences.
    """
    if reference == hypothesis:
        return 0
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    n = len(hypothesis)
    if n == 0:
        return len(reference)

    hyp = np.asarray(list(hypothesis))
    positions = np.arange(n + 1, dtype=np.int64)
    previous = positions.copy()
    for i, ref_item in enumerate(reference, start=1):
        cost = (hyp != ref_item).astype(np.int64)
        # Deletion vs substitution for positions 1..n; position 0 costs i.
        base = np.minimum(previous[:-1] + cost, previous[1:] + 1)
        base_ext = np.concatenate((np.array([i], dtype=np.int64), base))
        # Insertion propagates left-to-right; resolve the recurrence with a
        # prefix minimum of (base - position), then add the position back.
        prefix = np.minimum.accumulate(base_ext - positions)
        previous = positions + prefix
    return int(previous[-1])


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


def ser(references: list[str], hypotheses: list[str]) -> float:
    """Sentence error rate: fraction of utterances with any word error.

    Unlike WER, a single-word mistake weighs the same as a fully garbled
    sentence, which is useful when whole-utterance correctness matters more
    than edit count. Raises ``ValueError`` on length mismatch or empty input.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"need equal-length lists, got {len(references)} references "
            f"and {len(hypotheses)} hypotheses"
        )
    if not references:
        raise ValueError("ser() requires at least one sentence pair")
    wrong = sum(1 for ref, hyp in zip(references, hypotheses, strict=True) if wer(ref, hyp) > 0.0)
    return wrong / len(references)

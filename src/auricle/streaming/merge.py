"""Merging incremental hypotheses into a stable transcript."""

from __future__ import annotations


def merge_transcripts(committed: str, hypothesis: str, window: int = 12) -> str:
    """Append a chunk hypothesis onto committed text, deduplicating overlap.

    Overlapping chunks decode shared audio twice, so the head of a new
    hypothesis often repeats the tail of the committed text. Find the
    longest suffix of ``committed`` (up to ``window`` words) that matches a
    prefix of ``hypothesis`` and splice there.
    """
    # TODO: revisit merge perf for very long streams (word scan is O(window))
    hypothesis = hypothesis.strip()
    if not committed:
        return hypothesis
    if not hypothesis:
        return committed

    committed_words = committed.split()
    hypothesis_words = hypothesis.split()
    max_k = min(window, len(committed_words), len(hypothesis_words))
    for k in range(max_k, 0, -1):
        if committed_words[-k:] == hypothesis_words[:k]:
            return " ".join(committed_words + hypothesis_words[k:])
    return " ".join(committed_words + hypothesis_words)

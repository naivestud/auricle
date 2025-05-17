"""CTC decoding: greedy and prefix beam search."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from auricle.asr.vocab import CharVocabulary

_NEG_INF = float("-inf")


def collapse_tokens(tokens: torch.Tensor, vocab: CharVocabulary) -> str:
    """Collapse a 1-D token sequence: merge repeats, drop blanks, map to text."""
    out: list[int] = []
    prev = None
    for token in tokens.tolist():
        if token != prev and token != vocab.BLANK:
            out.append(token)
        prev = token
    return vocab.decode(out)


def greedy_decode(logits: torch.Tensor, vocab: CharVocabulary) -> list[str]:
    """Greedy CTC decode a ``(batch, time, vocab_size)`` logit tensor."""
    best = logits.argmax(dim=-1)
    return [collapse_tokens(row, vocab) for row in best]


def greedy_decode_with_confidence(
    logits: torch.Tensor, vocab: CharVocabulary
) -> list[tuple[str, float]]:
    """Greedy decode that also reports mean frame confidence.

    Confidence is the average softmax probability of the argmax token at each
    frame, in ``[0, 1]`` — a cheap proxy for how decisive the encoder was. A
    fully blank sequence still averages over frames, so it reflects silence
    rather than collapsing to zero length.
    """
    probs = torch.softmax(logits, dim=-1)
    best_probs, best = probs.max(dim=-1)
    results: list[tuple[str, float]] = []
    for row, prob_row in zip(best, best_probs, strict=False):
        confidence = float(prob_row.mean()) if prob_row.numel() else 0.0
        results.append((collapse_tokens(row, vocab), confidence))
    return results


def _logsumexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def _beam_search_one(log_probs: np.ndarray, vocab: CharVocabulary, beam_width: int) -> str:
    """Prefix beam search over a single ``(time, vocab_size)`` log-prob matrix."""
    blank = vocab.BLANK
    # beams maps a prefix (token tuple) to a pair of log probabilities:
    # (ending in blank, ending in a non-blank).
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _NEG_INF)}

    for t in range(log_probs.shape[0]):
        frame = log_probs[t]
        # Keep only the most promising prefixes before expanding.
        ranked = sorted(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True)[
            :beam_width
        ]

        new_beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [_NEG_INF, _NEG_INF])
        for prefix, (pb, pnb) in ranked:
            p_total = _logsumexp(pb, pnb)
            for token in range(frame.shape[0]):
                lp = float(frame[token])
                if token == blank:
                    new_beams[prefix][0] = _logsumexp(new_beams[prefix][0], p_total + lp)
                    continue

                if prefix and token == prefix[-1]:
                    # Repeating the last character only extends the prefix when
                    # the previous path ended in a blank; otherwise CTC merges it.
                    extended = prefix + (token,)
                    new_beams[extended][1] = _logsumexp(new_beams[extended][1], pb + lp)
                    new_beams[prefix][1] = _logsumexp(new_beams[prefix][1], pnb + lp)
                else:
                    extended = prefix + (token,)
                    new_beams[extended][1] = _logsumexp(new_beams[extended][1], p_total + lp)

        beams = {prefix: (scores[0], scores[1]) for prefix, scores in new_beams.items()}

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))
    return vocab.decode(list(best[0]))


def beam_decode(logits: torch.Tensor, vocab: CharVocabulary, beam_width: int = 10) -> list[str]:
    """CTC prefix beam search over a ``(batch, time, vocab_size)`` logit tensor.

    Wider beams search more alignments and can beat greedy decoding on noisy
    frames, at the cost of roughly ``beam_width`` times the work per frame.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    log_probs = torch.log_softmax(logits, dim=-1).numpy()
    return [_beam_search_one(item, vocab, beam_width) for item in log_probs]

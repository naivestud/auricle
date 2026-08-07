"""CTC projection head and greedy decoding."""

from __future__ import annotations

import torch
from torch import nn

from auricle.asr.vocab import CharVocabulary


class CTCHead(nn.Module):
    """Projects encoder features to per-frame vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, features):
        """``features`` has shape ``(batch, time, d_model)``.

        Returns logits of shape ``(batch, time, vocab_size)``.
        """
        return self.proj(features)


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

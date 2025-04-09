"""CTC projection head."""

from __future__ import annotations

from torch import nn


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

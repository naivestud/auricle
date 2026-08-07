"""Multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must divide d_model ({d_model})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``(batch, time, d_model)``; output has the same shape."""
        b, t, c = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, t, c)
        return self.out_proj(y)

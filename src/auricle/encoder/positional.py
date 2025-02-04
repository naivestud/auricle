"""Sinusoidal positional embeddings (Vaswani et al., 2017)."""

from __future__ import annotations

import math

import torch


def sinusoidal_positional_embedding(max_len: int, d_model: int) -> torch.Tensor:
    """Return a fixed ``(max_len, d_model)`` positional embedding table."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal embeddings, got {d_model}")
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(positions * div)
    table[:, 1::2] = torch.cos(positions * div)
    return table

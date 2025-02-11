"""Whisper-style convolutional frontend."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class ConvFrontend(nn.Module):
    """Two 1-D convolutions that halve the frame rate once.

    Takes a mel spectrogram ``(batch, n_mels, frames)`` and returns
    ``(batch, frames // 2, d_model)``.
    """

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

    def forward(self, mel):
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        return x.transpose(1, 2)

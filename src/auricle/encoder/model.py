"""The full whisper-style audio encoder."""

from __future__ import annotations

import torch
from torch import nn

from auricle.encoder.config import EncoderConfig
from auricle.encoder.frontend import ConvFrontend
from auricle.encoder.positional import sinusoidal_positional_embedding
from auricle.encoder.transformer import TransformerEncoder


class WhisperStyleEncoder(nn.Module):
    """Mel spectrogram in, contextual features out.

    Architecture mirrors the whisper audio tower: a two-layer convolutional
    frontend halves the frame rate, sinusoidal positions are added, and a
    pre-norm transformer mixes information across time.
    """

    positional: torch.Tensor

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.frontend = ConvFrontend(config.n_mels, config.d_model)
        self.blocks = TransformerEncoder(
            config.d_model, config.n_layers, config.n_heads, config.ff_mult, config.dropout
        )
        positional = sinusoidal_positional_embedding(config.max_frames, config.d_model)
        self.register_buffer("positional", positional, persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel`` has shape ``(batch, n_mels, frames)``.

        Returns ``(batch, frames // 2, d_model)``.
        """
        h = self.frontend(mel)
        h = h + self.positional[: h.shape[1]].unsqueeze(0)
        return self.blocks(h)

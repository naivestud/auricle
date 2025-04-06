"""Pre-norm transformer encoder blocks."""

from __future__ import annotations

from torch import nn

from auricle.encoder.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """Pre-norm block: self-attention followed by a positionwise feed-forward."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerEncoder(nn.Module):
    """A stack of :class:`EncoderBlock` with a final layer norm."""

    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

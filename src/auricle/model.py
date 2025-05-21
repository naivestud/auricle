"""Top-level speech recognition model."""

from __future__ import annotations

import torch
from torch import nn

from auricle.asr.ctc import CTCHead
from auricle.asr.decode import greedy_decode
from auricle.asr.vocab import CharVocabulary
from auricle.audio.spectrogram import LogMelSpectrogram
from auricle.encoder.config import EncoderConfig
from auricle.encoder.model import WhisperStyleEncoder


class AuricleModel(nn.Module):
    """Waveform to CTC logits: log-mel, whisper-style encoder, CTC head."""

    def __init__(self, config: EncoderConfig, vocab: CharVocabulary | None = None):
        super().__init__()
        self.config = config
        self.vocab = vocab if vocab is not None else CharVocabulary()
        self.spectrogram = LogMelSpectrogram(n_mels=config.n_mels)
        self.encoder = WhisperStyleEncoder(config)
        self.head = CTCHead(config.d_model, len(self.vocab))

    @classmethod
    def tiny(cls) -> AuricleModel:
        """A randomly initialised tiny model, useful for tests and demos."""
        return cls(EncoderConfig.tiny())

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total number of scalar parameters, optionally only trainable ones."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def summary(self) -> str:
        """A one-line description of the architecture and its size."""
        cfg = self.config
        return (
            f"AuricleModel(d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
            f"n_heads={cfg.n_heads}, n_mels={cfg.n_mels}, "
            f"vocab={len(self.vocab)}, params={self.count_parameters():,})"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute CTC logits for ``waveform`` of shape ``(batch, samples)``."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.spectrogram(waveform)
        # The stride-2 frontend halves the frame count; drop a trailing odd
        # frame so the output length is always frames // 2.
        if mel.shape[-1] % 2 == 1:
            mel = mel[..., :-1]
        features = self.encoder(mel)
        return self.head(features)

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> list[str]:
        """Greedy-decode ``waveform`` into a list of transcripts."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(waveform)
            return greedy_decode(logits, self.vocab)
        finally:
            self.train(was_training)

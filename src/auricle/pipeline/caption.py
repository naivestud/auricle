"""Audio captioning: describe a clip in one sentence via an LLM backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from auricle.constants import SAMPLE_RATE
from auricle.llm.base import LLMBackend
from auricle.pipeline.acoustics import summarize
from auricle.pipeline.asr import to_waveform
from auricle.types import ModelLike

CAPTION_PROMPT = """\
You are an audio captioning system. Write one concise caption for an audio clip.

Acoustic summary: {summary}
Transcript (may be empty): {transcript}

Caption:"""


def build_caption_prompt(summary: str, transcript: str) -> str:
    return CAPTION_PROMPT.format(summary=summary, transcript=transcript or "<none>")


def caption_audio(
    model: ModelLike,
    backend: LLMBackend,
    audio: str | Path | np.ndarray | torch.Tensor,
    sample_rate: int | None = None,
) -> str:
    """Caption an audio clip.

    The encoder provides a transcript; lightweight acoustic statistics give
    the backend a sense of the non-speech content. The backend produces the
    final caption.
    """
    waveform = to_waveform(audio, sample_rate)
    summary = summarize(waveform.numpy(), SAMPLE_RATE)
    transcript = model.transcribe(waveform)[0].strip()
    prompt = build_caption_prompt(summary.describe(), transcript)
    return backend.generate(prompt).text.strip()

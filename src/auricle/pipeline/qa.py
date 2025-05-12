"""Speech question answering: ask a question about spoken content."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from auricle.llm.base import LLMBackend
from auricle.pipeline.asr import to_waveform, transcribe_waveform
from auricle.types import ModelLike

QA_PROMPT = """\
Answer the question using only the information in the audio transcript.

Transcript: {transcript}

Question: {question}

Answer:"""


def build_qa_prompt(transcript: str, question: str) -> str:
    return QA_PROMPT.format(transcript=transcript or "<none>", question=question)


def answer_question(
    model: ModelLike,
    backend: LLMBackend,
    audio: str | Path | np.ndarray | torch.Tensor,
    question: str,
    sample_rate: int | None = None,
) -> str:
    """Transcribe ``audio`` and ask ``backend`` a question about it."""
    waveform = to_waveform(audio, sample_rate)
    transcript = transcribe_waveform(model, waveform)
    prompt = build_qa_prompt(transcript, question)
    return backend.generate(prompt).text.strip()

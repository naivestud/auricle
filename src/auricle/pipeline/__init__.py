"""High-level pipelines built on top of the encoder."""

from auricle.pipeline.asr import transcribe
from auricle.pipeline.caption import caption_audio
from auricle.pipeline.qa import answer_question

__all__ = ["answer_question", "caption_audio", "transcribe"]

"""High-level pipelines built on top of the encoder."""

from auricle.pipeline.asr import transcribe
from auricle.pipeline.caption import caption_audio

__all__ = ["caption_audio", "transcribe"]

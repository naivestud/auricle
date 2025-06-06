"""High-level pipelines built on top of the encoder."""

from auricle.pipeline.asr import transcribe, transcribe_with_confidence
from auricle.pipeline.caption import caption_audio
from auricle.pipeline.keywords import find_keywords, spot_keywords
from auricle.pipeline.qa import answer_question

__all__ = [
    "answer_question",
    "caption_audio",
    "find_keywords",
    "spot_keywords",
    "transcribe",
    "transcribe_with_confidence",
]

"""auricle: streaming speech understanding toolkit.

Offline transcription, streaming ASR, audio captioning and speech question
answering on top of a whisper-style encoder with pluggable LLM backends.
"""

from auricle.checkpoint import load_checkpoint, save_checkpoint
from auricle.encoder import EncoderConfig
from auricle.eval import cer, ser, wer
from auricle.llm import EchoBackend, LLMBackend, get_backend
from auricle.model import AuricleModel
from auricle.pipeline import answer_question, caption_audio, transcribe
from auricle.streaming import StreamingASR

__version__ = "0.3.0"

__all__ = [
    "AuricleModel",
    "EchoBackend",
    "EncoderConfig",
    "LLMBackend",
    "StreamingASR",
    "__version__",
    "answer_question",
    "caption_audio",
    "cer",
    "get_backend",
    "load_checkpoint",
    "save_checkpoint",
    "ser",
    "transcribe",
    "wer",
]

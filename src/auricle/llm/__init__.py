"""Pluggable language-model backends."""

from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.echo import EchoBackend
from auricle.llm.openai_compat import OpenAICompatBackend
from auricle.llm.registry import available_backends, get_backend, register_backend

__all__ = [
    "EchoBackend",
    "GenerationResult",
    "LLMBackend",
    "OpenAICompatBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]

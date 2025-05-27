"""Pluggable language-model backends."""

from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.echo import EchoBackend
from auricle.llm.openai_compat import OpenAICompatBackend
from auricle.llm.registry import available_backends, get_backend, register_backend
from auricle.llm.scripted import ScriptedBackend

__all__ = [
    "EchoBackend",
    "GenerationResult",
    "LLMBackend",
    "OpenAICompatBackend",
    "ScriptedBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]

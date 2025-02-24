"""Backend registry: look up an LLM backend by name."""

from __future__ import annotations

from auricle.errors import BackendNotFoundError
from auricle.llm.base import LLMBackend

_REGISTRY: dict[str, type[LLMBackend]] = {}


def register_backend(cls: type[LLMBackend]) -> type[LLMBackend]:
    """Class decorator that registers a backend under its ``name``."""
    if not cls.name or cls.name == "base":
        raise ValueError(f"{cls.__name__} must set a non-default `name`")
    _REGISTRY[cls.name] = cls
    return cls


def available_backends() -> list[str]:
    """Names of all registered backends, sorted for stable output."""
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs) -> LLMBackend:
    """Instantiate the backend registered under ``name``.

    The optional HuggingFace backend is imported lazily so that installing
    ``transformers`` stays opt-in.
    """
    if name not in _REGISTRY and name == "huggingface":
        try:
            import auricle.llm.hf  # noqa: F401  (registers itself)
        except ImportError:
            pass

    if name not in _REGISTRY:
        known = ", ".join(available_backends()) or "<none>"
        raise BackendNotFoundError(f"no backend named {name!r} (known: {known})")
    return _REGISTRY[name](**kwargs)

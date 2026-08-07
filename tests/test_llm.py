import pytest

from auricle.errors import BackendNotFoundError
from auricle.llm import EchoBackend, available_backends, get_backend
from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.registry import register_backend


def test_echo_returns_prompt():
    backend = EchoBackend()
    result = backend.generate("hello world", max_new_tokens=10)
    assert result.text == "hello world"
    assert result.backend == "echo"
    assert result.prompt == "hello world"


def test_echo_truncates_to_max_new_tokens():
    backend = EchoBackend()
    result = backend.generate("one two three four", max_new_tokens=2)
    assert result.text == "one two"


def test_echo_prefix():
    backend = EchoBackend(prefix="[caption] ")
    assert backend.generate("a b").text.startswith("[caption]")


def test_registry_lists_echo():
    assert "echo" in available_backends()


def test_get_backend_echo():
    backend = get_backend("echo")
    assert isinstance(backend, EchoBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(BackendNotFoundError):
        get_backend("does-not-exist")


def test_register_requires_name():
    with pytest.raises(ValueError):

        @register_backend
        class NoName(LLMBackend):
            def generate(self, prompt, *, max_new_tokens=128):
                return GenerationResult(text="", backend=self.name)


def test_custom_backend_roundtrip():
    @register_backend
    class Shout(LLMBackend):
        name = "shout"

        def generate(self, prompt, *, max_new_tokens=128):
            return GenerationResult(text=prompt.upper(), backend=self.name)

    assert get_backend("shout").generate("hi").text == "HI"

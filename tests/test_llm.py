import pytest

from auricle.errors import BackendNotFoundError
from auricle.llm import (
    CachingBackend,
    EchoBackend,
    ScriptedBackend,
    available_backends,
    get_backend,
)
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


def test_scripted_returns_responses_in_order():
    backend = ScriptedBackend(responses=["first", "second"])
    assert backend.generate("a").text == "first"
    assert backend.generate("b").text == "second"
    assert backend.calls == 2


def test_scripted_cycles_by_default():
    backend = ScriptedBackend(responses=["x", "y"])
    assert [backend.generate("p").text for _ in range(4)] == ["x", "y", "x", "y"]


def test_scripted_repeats_last_when_not_cycling():
    backend = ScriptedBackend(responses=["x", "y"], cycle=False)
    assert [backend.generate("p").text for _ in range(3)] == ["x", "y", "y"]


def test_scripted_empty_script_is_empty_string():
    backend = ScriptedBackend()
    assert backend.generate("hi").text == ""


def test_scripted_registered_and_constructible():
    assert "scripted" in available_backends()
    backend = get_backend("scripted", responses=["only"])
    assert backend.generate("q").text == "only"


def test_caching_reuses_response_for_same_prompt():
    inner = ScriptedBackend(responses=["a", "b", "c"])
    cached = CachingBackend(inner)
    assert cached.generate("same").text == "a"
    assert cached.generate("same").text == "a"  # served from cache
    assert cached.hits == 1
    assert inner.calls == 1


def test_caching_keys_include_max_new_tokens():
    inner = ScriptedBackend(responses=["short", "longer"])
    cached = CachingBackend(inner)
    assert cached.generate("p", max_new_tokens=8).text == "short"
    assert cached.generate("p", max_new_tokens=16).text == "longer"
    assert cached.cache_size == 2


def test_caching_mirrors_wrapped_name():
    cached = CachingBackend(EchoBackend())
    assert cached.name == "echo"


def test_caching_clear_resets():
    cached = CachingBackend(ScriptedBackend(responses=["x", "y"]))
    cached.generate("p")
    cached.generate("p")
    assert cached.hits == 1
    cached.clear()
    assert cached.cache_size == 0
    assert cached.hits == 0
    assert cached.generate("p").text == "y"

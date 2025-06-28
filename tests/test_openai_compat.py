import pytest

from auricle.errors import BackendError
from auricle.llm.openai_compat import OpenAICompatBackend


def _fake_transport(response):
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        return response

    return transport, calls


def test_generate_uses_chat_completions_shape():
    response = {"choices": [{"message": {"content": "a cat purring"}}]}
    transport, calls = _fake_transport(response)
    backend = OpenAICompatBackend(model="tiny-llama", base_url="http://srv/v1", transport=transport)

    result = backend.generate("Describe this audio.", max_new_tokens=32)

    assert result.text == "a cat purring"
    assert result.backend == "openai"
    call = calls[0]
    assert call["url"] == "http://srv/v1/chat/completions"
    assert call["payload"]["model"] == "tiny-llama"
    assert call["payload"]["messages"] == [{"role": "user", "content": "Describe this audio."}]
    assert call["payload"]["max_tokens"] == 32
    assert call["headers"]["Content-Type"] == "application/json"


def test_trailing_slash_in_base_url():
    transport, calls = _fake_transport({"choices": [{"message": {"content": "x"}}]})
    backend = OpenAICompatBackend(model="m", base_url="http://srv/v1/", transport=transport)
    backend.generate("hi")
    assert calls[0]["url"] == "http://srv/v1/chat/completions"


def test_no_auth_header_without_api_key():
    transport, calls = _fake_transport({"choices": [{"message": {"content": "x"}}]})
    backend = OpenAICompatBackend(model="m", api_key="", transport=transport)
    backend.generate("hi")
    assert "Authorization" not in calls[0]["headers"]


def test_api_key_sent_as_bearer():
    transport, calls = _fake_transport({"choices": [{"message": {"content": "x"}}]})
    backend = OpenAICompatBackend(model="m", api_key="sekrit", transport=transport)
    backend.generate("hi")
    assert calls[0]["headers"]["Authorization"] == "Bearer sekrit"


def test_malformed_response_raises():
    transport, _ = _fake_transport({"unexpected": True})
    backend = OpenAICompatBackend(model="m", transport=transport)
    with pytest.raises(BackendError, match="malformed response"):
        backend.generate("hi")


def test_backend_registered_under_openai():
    from auricle.llm import get_backend

    backend = get_backend("openai", model="m")
    assert isinstance(backend, OpenAICompatBackend)


def _flaky_transport(failures: int, response: dict):
    calls = {"n": 0}

    def transport(url, payload, headers, timeout):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise BackendError("transient failure")
        return response

    return transport, calls


def test_retry_recovers_from_transient_errors():
    ok = {"choices": [{"message": {"content": "recovered"}}]}
    transport, calls = _flaky_transport(failures=2, response=ok)
    sleeps: list[float] = []
    backend = OpenAICompatBackend(
        model="m", transport=transport, max_retries=3, backoff_seconds=0.5, sleep=sleeps.append
    )
    result = backend.generate("hi")
    assert result.text == "recovered"
    assert calls["n"] == 3  # two failures + one success
    assert sleeps == [0.5, 1.0]  # exponential backoff


def test_retry_exhausted_raises():
    ok = {"choices": [{"message": {"content": "x"}}]}
    transport, calls = _flaky_transport(failures=10, response=ok)
    backend = OpenAICompatBackend(
        model="m", transport=transport, max_retries=2, sleep=lambda _: None
    )
    with pytest.raises(BackendError, match="transient"):
        backend.generate("hi")
    assert calls["n"] == 3  # initial + 2 retries


def test_no_retry_by_default():
    ok = {"choices": [{"message": {"content": "x"}}]}
    transport, calls = _flaky_transport(failures=10, response=ok)
    backend = OpenAICompatBackend(model="m", transport=transport)
    with pytest.raises(BackendError):
        backend.generate("hi")
    assert calls["n"] == 1


def test_rejects_negative_retries():
    with pytest.raises(ValueError):
        OpenAICompatBackend(model="m", max_retries=-1)

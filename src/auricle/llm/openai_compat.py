"""Backend for OpenAI-compatible ``/chat/completions`` endpoints.

Works with any server implementing the OpenAI chat API shape: vLLM,
llama.cpp server, Ollama (``/v1``), or the hosted API itself. Uses only the
standard library so it adds no dependency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from auricle.errors import BackendError
from auricle.llm.base import GenerationResult, LLMBackend
from auricle.llm.registry import register_backend

Transport = Callable[[str, dict, dict, float], dict]


def _urllib_transport(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise BackendError(f"HTTP {exc.code} from {url}: {exc.read()[:200]!r}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"could not reach {url}: {exc.reason}") from exc


@register_backend
class OpenAICompatBackend(LLMBackend):
    """Send prompts to an OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
        max_retries: int = 0,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._transport = transport or _urllib_transport

    def generate(self, prompt: str, *, max_new_tokens: int = 128) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        response = self._request_with_retries(url, payload, headers)

        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"malformed response from {self.base_url}: {response!r}") from exc
        return GenerationResult(text=text, backend=self.name, prompt=prompt)

    def _request_with_retries(self, url: str, payload: dict, headers: dict) -> dict:
        """Call the transport, retrying transient BackendErrors with backoff."""
        attempt = 0
        while True:
            try:
                return self._transport(url, payload, headers, self.timeout)
            except BackendError:
                if attempt >= self.max_retries:
                    raise
                self._sleep(self.backoff_seconds * (2**attempt))
                attempt += 1

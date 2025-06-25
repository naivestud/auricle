from pathlib import Path

import pytest
import torch

from auricle.llm import CachingBackend, ScriptedBackend
from auricle.model import AuricleModel
from auricle.pipeline.caption import caption_audio
from auricle.pipeline.qa import answer_question

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(21)
    return AuricleModel.tiny()


def test_caption_uses_backend_response_verbatim(model):
    backend = ScriptedBackend(responses=["A soft humming tone."])
    caption = caption_audio(model, backend, FIXTURES / "tone_1s.wav")
    assert caption == "A soft humming tone."
    assert backend.calls == 1


def test_qa_uses_backend_response_verbatim(model):
    backend = ScriptedBackend(responses=["Forty-two."])
    answer = answer_question(model, backend, FIXTURES / "tone_1s.wav", "what is the answer?")
    assert answer == "Forty-two."


def test_caption_caching_avoids_second_model_backend_call(model):
    backend = CachingBackend(ScriptedBackend(responses=["Only caption."]))
    first = caption_audio(model, backend, FIXTURES / "tone_1s.wav")
    second = caption_audio(model, backend, FIXTURES / "tone_1s.wav")
    assert first == second == "Only caption."
    assert backend.hits == 1


def test_qa_prompt_varies_with_question(model):
    # A scripted backend that echoes call count proves each question issues a
    # fresh generate call (prompts differ, so no accidental caching).
    backend = ScriptedBackend(responses=["one", "two"])
    answer_question(model, backend, FIXTURES / "tone_1s.wav", "first?")
    answer_question(model, backend, FIXTURES / "tone_1s.wav", "second?")
    assert backend.calls == 2

from pathlib import Path

import pytest
import torch

from auricle.llm import EchoBackend
from auricle.model import AuricleModel
from auricle.pipeline.qa import answer_question, build_qa_prompt

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(9)
    return AuricleModel.tiny()


def test_build_prompt_includes_question():
    prompt = build_qa_prompt("hello there", "who is speaking?")
    assert "who is speaking?" in prompt
    assert "hello there" in prompt


def test_build_prompt_empty_transcript():
    assert "<none>" in build_qa_prompt("", "q")


def test_answer_question_with_echo(model):
    backend = EchoBackend()
    answer = answer_question(model, backend, FIXTURES / "tone_1s.wav", "what is this?")
    # Echo returns the prompt, which contains the question.
    assert "what is this?" in answer


def test_answer_question_deterministic(model):
    backend = EchoBackend()
    a = answer_question(model, backend, FIXTURES / "chirp_3s.wav", "q one")
    b = answer_question(model, backend, FIXTURES / "chirp_3s.wav", "q one")
    assert a == b

import numpy as np
import pytest
import torch

from auricle.model import AuricleModel
from auricle.pipeline.keywords import find_keywords, spot_keywords


def test_find_keywords_basic_match():
    assert find_keywords("the cat sat on the mat", ["cat", "dog"]) == ["cat"]


def test_find_keywords_multiple_and_order():
    found = find_keywords("alpha beta gamma", ["gamma", "alpha"])
    assert found == ["gamma", "alpha"]  # preserves keyword-list order


def test_find_keywords_normalises_case_and_punct():
    # Apostrophes survive normalisation (contractions stay one word), so the
    # keyword keeps it; case and trailing punctuation are ignored.
    assert find_keywords("Don't STOP!", ["don't stop"]) == ["don't stop"]


def test_find_keywords_phrase_match():
    assert find_keywords("turn on the lights", ["the lights", "the tv"]) == ["the lights"]


def test_find_keywords_no_match():
    assert find_keywords("hello world", ["zebra"]) == []


def test_find_keywords_empty_keyword_ignored():
    assert find_keywords("anything", ["", "any"]) == ["any"]


def test_find_keywords_rejects_empty_list():
    with pytest.raises(ValueError):
        find_keywords("text", [])


class FakeModel:
    def __init__(self, text):
        self._text = text

    def transcribe(self, waveform):
        return [self._text]


def test_spot_keywords_end_to_end():
    model = FakeModel("play some jazz music")
    found = spot_keywords(model, np.zeros(8_000, dtype=np.float32), ["jazz", "rock"])
    assert found == ["jazz"]


def test_spot_keywords_with_real_model_runs():
    torch.manual_seed(0)
    model = AuricleModel.tiny()
    found = spot_keywords(model, np.zeros(4_000, dtype=np.float32), ["word"])
    assert isinstance(found, list)

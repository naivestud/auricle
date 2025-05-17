import pytest
import torch

from auricle.asr.ctc import CTCHead
from auricle.asr.decode import (
    beam_decode,
    collapse_tokens,
    greedy_decode,
    greedy_decode_with_confidence,
)
from auricle.asr.vocab import CharVocabulary


def test_head_shape():
    head = CTCHead(d_model=16, vocab_size=5)
    out = head(torch.randn(2, 10, 16))
    assert out.shape == (2, 10, 5)


def test_collapse_merges_repeats():
    vocab = CharVocabulary(chars=("a", "b"))
    # a a _ b b b a -> "aba": runs merge, blank ends the b run, final a starts fresh
    tokens = torch.tensor([1, 1, 0, 2, 2, 2, 1])
    assert collapse_tokens(tokens, vocab) == "aba"


def test_collapse_blank_separates_same_char():
    vocab = CharVocabulary(chars=("a",))
    tokens = torch.tensor([1, 0, 1])
    assert collapse_tokens(tokens, vocab) == "aa"


def test_collapse_all_blank():
    vocab = CharVocabulary(chars=("a",))
    tokens = torch.tensor([0, 0, 0])
    assert collapse_tokens(tokens, vocab) == ""


def test_greedy_decode_batch():
    vocab = CharVocabulary(chars=("a", "b"))
    # Build logits that argmax to a known token path per batch item.
    logits = torch.full((2, 3, 3), -10.0)
    # item 0: tokens [1, 1, 2] -> "ab"
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 10.0
    logits[0, 2, 2] = 10.0
    # item 1: tokens [0, 2, 2] -> "b"
    logits[1, 0, 0] = 10.0
    logits[1, 1, 2] = 10.0
    logits[1, 2, 2] = 10.0

    assert greedy_decode(logits, vocab) == ["ab", "b"]


def _one_hot_logits(token_path: list[int], vocab_size: int, confidence: float = 10.0):
    logits = torch.full((1, len(token_path), vocab_size), -confidence)
    for t, token in enumerate(token_path):
        logits[0, t, token] = confidence
    return logits


def test_beam_decode_matches_greedy_on_sharp_logits():
    vocab = CharVocabulary(chars=("a", "b"))
    # tokens [1, 0, 2, 2, 1] -> "aba"
    logits = _one_hot_logits([1, 0, 2, 2, 1], vocab_size=3)
    assert beam_decode(logits, vocab, beam_width=4) == greedy_decode(logits, vocab)


def test_beam_decode_handles_repeated_char_with_blank():
    vocab = CharVocabulary(chars=("a",))
    # tokens [1, 0, 1] -> "aa" (blank separates the two a's)
    logits = _one_hot_logits([1, 0, 1], vocab_size=2)
    assert beam_decode(logits, vocab, beam_width=5) == ["aa"]


def test_beam_decode_merges_repeats_without_blank():
    vocab = CharVocabulary(chars=("a",))
    # tokens [1, 1, 1] -> "a" (repeats collapse)
    logits = _one_hot_logits([1, 1, 1], vocab_size=2)
    assert beam_decode(logits, vocab, beam_width=5) == ["a"]


def test_beam_decode_all_blank_is_empty():
    vocab = CharVocabulary(chars=("a",))
    logits = _one_hot_logits([0, 0, 0], vocab_size=2)
    assert beam_decode(logits, vocab, beam_width=3) == [""]


def test_beam_decode_batch():
    vocab = CharVocabulary(chars=("a", "b"))
    logits = torch.full((2, 3, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 0] = 10.0
    logits[0, 2, 2] = 10.0  # "ab"
    logits[1, 0, 2] = 10.0
    logits[1, 1, 2] = 10.0
    logits[1, 2, 1] = 10.0  # "ba"
    assert beam_decode(logits, vocab, beam_width=4) == ["ab", "ba"]


def test_beam_width_one_is_valid():
    vocab = CharVocabulary(chars=("a",))
    logits = _one_hot_logits([1, 1, 1], vocab_size=2)
    assert beam_decode(logits, vocab, beam_width=1) == ["a"]


def test_beam_decode_rejects_zero_width():
    vocab = CharVocabulary(chars=("a",))
    logits = _one_hot_logits([1], vocab_size=2)
    with pytest.raises(ValueError):
        beam_decode(logits, vocab, beam_width=0)


def test_decode_with_confidence_sharp_is_high():
    vocab = CharVocabulary(chars=("a",))
    logits = _one_hot_logits([1, 1, 1], vocab_size=2, confidence=20.0)
    ((text, conf),) = greedy_decode_with_confidence(logits, vocab)
    assert text == "a"
    assert conf > 0.99


def test_decode_with_confidence_flat_is_low():
    vocab = CharVocabulary(chars=("a", "b"))
    # Uniform logits -> confidence is exactly 1/vocab_size.
    logits = torch.zeros(1, 4, 3)
    ((text, conf),) = greedy_decode_with_confidence(logits, vocab)
    assert conf == pytest.approx(1.0 / 3.0)


def test_decode_with_confidence_text_matches_greedy():
    vocab = CharVocabulary(chars=("a", "b"))
    logits = torch.randn(2, 6, 3)
    texts = greedy_decode(logits, vocab)
    pairs = greedy_decode_with_confidence(logits, vocab)
    assert [t for t, _ in pairs] == texts
    for _, conf in pairs:
        assert 0.0 <= conf <= 1.0

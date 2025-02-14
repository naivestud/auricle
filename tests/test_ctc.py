import torch

from auricle.asr.ctc import CTCHead, collapse_tokens, greedy_decode
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

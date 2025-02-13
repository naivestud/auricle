import pytest

from auricle.asr.vocab import CHARS, CharVocabulary


def test_default_vocab_size():
    vocab = CharVocabulary()
    # 26 letters + apostrophe + space + blank.
    assert len(vocab) == 29
    assert len(CHARS) == 28


def test_blank_is_zero():
    assert CharVocabulary.BLANK == 0


def test_encode_decode_roundtrip():
    vocab = CharVocabulary()
    text = "hello world"
    assert vocab.decode(vocab.encode(text)) == text


def test_encode_apostrophe():
    vocab = CharVocabulary()
    assert vocab.decode(vocab.encode("don't")) == "don't"


def test_encode_rejects_unknown_char():
    vocab = CharVocabulary()
    with pytest.raises(ValueError, match="not in the vocabulary"):
        vocab.encode("hello!")


def test_decode_skips_blank():
    vocab = CharVocabulary(chars=("a", "b"))
    ids = [0, 1, 0, 2, 0]
    assert vocab.decode(ids) == "ab"


def test_decode_rejects_unknown_id():
    vocab = CharVocabulary(chars=("a",))
    with pytest.raises(ValueError):
        vocab.decode([99])


def test_custom_chars():
    vocab = CharVocabulary(chars=("x", "y"))
    assert vocab.size == 3
    assert vocab.decode(vocab.encode("xy")) == "xy"

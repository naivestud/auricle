import pytest

from auricle.eval.metrics import cer, edit_distance, normalize_text, wer


def test_edit_distance_identical():
    assert edit_distance("abc", "abc") == 0


def test_edit_distance_substitution():
    assert edit_distance("abc", "axc") == 1


def test_edit_distance_insertion_deletion():
    assert edit_distance("abc", "abcd") == 1
    assert edit_distance("abcd", "abc") == 1


def test_edit_distance_empty():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "xy") == 2


def test_normalize_text():
    assert normalize_text("  Hello,   World! ") == "hello world"
    assert normalize_text("Don't STOP") == "don't stop"


def test_wer_perfect():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_wer_case_and_punctuation_insensitive():
    assert wer("The cat, sat.", "the cat sat") == 0.0


def test_wer_one_substitution():
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_insertions_can_exceed_one():
    assert wer("the cat", "the cat sat on the mat") == pytest.approx(2.0)


def test_wer_empty_reference_raises():
    with pytest.raises(ValueError):
        wer("", "anything")


def test_cer_basic():
    assert cer("cat", "cat") == 0.0
    assert cer("cat", "bat") == pytest.approx(1 / 3)


def test_cer_ignores_spaces():
    assert cer("the cat", "thecat") == 0.0


def test_wer_empty_hypothesis_is_one():
    assert wer("the cat sat", "") == pytest.approx(1.0)


def test_wer_lower_bound_is_length_difference():
    rng = __import__("random").Random(42)
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
    for _ in range(50):
        ref = [rng.choice(words) for _ in range(rng.randint(1, 12))]
        hyp = [rng.choice(words) for _ in range(rng.randint(0, 12))]
        error = wer(" ".join(ref), " ".join(hyp))
        assert error >= abs(len(ref) - len(hyp)) / len(ref) - 1e-9

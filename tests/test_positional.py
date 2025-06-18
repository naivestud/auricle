import pytest
import torch

from auricle.encoder.positional import sinusoidal_positional_embedding


def test_shape():
    table = sinusoidal_positional_embedding(max_len=16, d_model=8)
    assert table.shape == (16, 8)


def test_values_bounded():
    table = sinusoidal_positional_embedding(max_len=64, d_model=32)
    assert table.abs().max() <= 1.0 + 1e-6


def test_first_row_is_origin():
    table = sinusoidal_positional_embedding(max_len=4, d_model=8)
    # position 0: sin(0)=0 on even columns, cos(0)=1 on odd columns.
    assert torch.allclose(table[0, 0::2], torch.zeros(4))
    assert torch.allclose(table[0, 1::2], torch.ones(4))


def test_rows_are_distinct():
    table = sinusoidal_positional_embedding(max_len=32, d_model=16)
    # No two positions share an embedding.
    for i in range(32):
        for j in range(i + 1, 32):
            assert not torch.equal(table[i], table[j])


def test_relative_offset_depends_only_on_delta():
    # Sinusoids make <pe(p), pe(p+k)> a function of k; check the norm of the
    # difference is the same for two different p at the same k.
    table = sinusoidal_positional_embedding(max_len=64, d_model=16)
    d1 = (table[10] - table[5]).norm()
    d2 = (table[30] - table[25]).norm()
    assert torch.isclose(d1, d2, atol=1e-5)


def test_rejects_odd_d_model():
    with pytest.raises(ValueError):
        sinusoidal_positional_embedding(max_len=8, d_model=7)


def test_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        sinusoidal_positional_embedding(max_len=0, d_model=8)


def test_deterministic():
    a = sinusoidal_positional_embedding(max_len=16, d_model=8)
    b = sinusoidal_positional_embedding(max_len=16, d_model=8)
    assert torch.equal(a, b)

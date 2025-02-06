import pytest
import torch

from auricle.encoder.attention import MultiHeadAttention


def test_attention_preserves_shape():
    attn = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    assert attn(x).shape == (2, 10, 64)


def test_attention_deterministic_in_eval():
    attn = MultiHeadAttention(d_model=32, n_heads=2).eval()
    x = torch.randn(1, 5, 32)
    with torch.no_grad():
        assert torch.equal(attn(x), attn(x))


def test_attention_rejects_bad_head_count():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=32, n_heads=3)


def test_attention_mixes_positions():
    # Changing one position should change other positions' outputs too.
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=32, n_heads=2).eval()
    x = torch.randn(1, 4, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[0, 0] += 1.0
    y2 = attn(x2)
    diff = (y2 - y1).abs().sum(dim=-1)[0]
    assert (diff > 1e-4).all()

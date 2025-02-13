import pytest
import torch

from auricle.encoder import EncoderConfig, WhisperStyleEncoder


def test_tiny_config_defaults():
    cfg = EncoderConfig.tiny()
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.d_model % cfg.n_heads == 0


def test_config_dict_roundtrip():
    cfg = EncoderConfig.tiny()
    restored = EncoderConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_config_from_dict_ignores_unknown_keys():
    cfg = EncoderConfig.from_dict({"d_model": 32, "n_layers": 1, "bogus": True})
    assert cfg.d_model == 32
    assert cfg.n_layers == 1


def test_encoder_forward_shape():
    cfg = EncoderConfig.tiny()
    encoder = WhisperStyleEncoder(cfg).eval()
    mel = torch.randn(2, cfg.n_mels, 100)
    with torch.no_grad():
        out = encoder(mel)
    assert out.shape == (2, 50, cfg.d_model)


def test_encoder_deterministic_in_eval():
    cfg = EncoderConfig.tiny()
    encoder = WhisperStyleEncoder(cfg).eval()
    mel = torch.randn(1, cfg.n_mels, 60)
    with torch.no_grad():
        assert torch.equal(encoder(mel), encoder(mel))


def test_encoder_output_finite():
    cfg = EncoderConfig.tiny()
    encoder = WhisperStyleEncoder(cfg).eval()
    mel = torch.zeros(1, cfg.n_mels, 40)
    with torch.no_grad():
        out = encoder(mel)
    assert torch.isfinite(out).all()


def test_encoder_respects_max_frames():
    cfg = EncoderConfig.tiny()
    encoder = WhisperStyleEncoder(cfg)
    # max_frames=500 for tiny -> up to 1000 mel frames.
    mel = torch.randn(1, cfg.n_mels, 1000)
    with torch.no_grad():
        out = encoder(mel)
    assert out.shape == (1, 500, cfg.d_model)


def test_encoder_rejects_overlong_input():
    cfg = EncoderConfig.tiny()
    encoder = WhisperStyleEncoder(cfg)
    mel = torch.randn(1, cfg.n_mels, 1200)
    with pytest.raises((IndexError, RuntimeError)):
        encoder(mel)

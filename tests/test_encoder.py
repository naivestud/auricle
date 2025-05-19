import pytest
import torch

from auricle.encoder import EncoderConfig, WhisperStyleEncoder


def test_tiny_config_defaults():
    cfg = EncoderConfig.tiny()
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.d_model % cfg.n_heads == 0


def test_named_presets_scale_up():
    tiny, small, base = EncoderConfig.tiny(), EncoderConfig.small(), EncoderConfig.base()
    assert tiny.d_model < small.d_model < base.d_model
    assert tiny.n_layers < small.n_layers < base.n_layers
    for cfg in (tiny, small, base):
        cfg.validate()  # every shipped preset must be internally consistent


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_model": 0},
        {"n_layers": -1},
        {"n_heads": 3, "d_model": 64},  # does not divide
        {"n_heads": 0},
        {"ff_mult": 0},
        {"max_frames": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"n_mels": -80},
    ],
)
def test_config_validate_rejects_bad_values(overrides):
    cfg = EncoderConfig(**overrides)
    with pytest.raises(ValueError):
        cfg.validate()


def test_encoder_rejects_invalid_config():
    cfg = EncoderConfig(d_model=64, n_heads=5)
    with pytest.raises(ValueError):
        WhisperStyleEncoder(cfg)


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

import pytest
import torch

from auricle.checkpoint import load_checkpoint, save_checkpoint
from auricle.errors import CheckpointError
from auricle.model import AuricleModel


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = AuricleModel.tiny().eval()
    waveform = torch.randn(8_000)
    with torch.no_grad():
        expected = model(waveform.unsqueeze(0))

    directory = save_checkpoint(model, tmp_path / "ckpt")
    restored = load_checkpoint(directory).eval()

    assert restored.config == model.config
    assert restored.vocab.chars == model.vocab.chars
    with torch.no_grad():
        assert torch.allclose(restored(waveform.unsqueeze(0)), expected)


def test_checkpoint_preserves_transcript(tmp_path):
    torch.manual_seed(1)
    model = AuricleModel.tiny()
    waveform = torch.randn(4_000)
    expected = model.transcribe(waveform)

    restored = load_checkpoint(save_checkpoint(model, tmp_path / "ckpt"))
    assert restored.transcribe(waveform) == expected


def test_load_missing_directory(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "does-not-exist")


def test_load_rejects_directory_without_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path)


def test_load_rejects_corrupt_config(tmp_path):
    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "config.json").write_text("{ not json")
    (directory / "model.pt").write_bytes(b"junk")
    with pytest.raises(CheckpointError):
        load_checkpoint(directory)


def test_load_partial_config_fills_defaults(tmp_path):
    import json

    torch.manual_seed(4)
    model = AuricleModel.tiny()
    directory = save_checkpoint(model, tmp_path / "ckpt")

    # Drop optional keys; from_dict should fill them with defaults.
    config = json.loads((directory / "config.json").read_text())
    config.pop("ff_mult", None)
    config.pop("dropout", None)
    (directory / "config.json").write_text(json.dumps(config))

    restored = load_checkpoint(directory)
    assert restored.config.ff_mult == 4
    assert restored.config.dropout == 0.0


def test_save_writes_format_version(tmp_path):
    torch.manual_seed(5)
    model = AuricleModel.tiny()
    directory = save_checkpoint(model, tmp_path / "ckpt")
    payload = torch.load(directory / "model.pt", weights_only=True)
    assert payload["format_version"] == 1


def test_load_legacy_checkpoint_without_version(tmp_path):
    # A payload missing format_version is treated as v1 and still loads.
    torch.manual_seed(6)
    model = AuricleModel.tiny()
    directory = save_checkpoint(model, tmp_path / "ckpt")
    payload = torch.load(directory / "model.pt", weights_only=True)
    payload.pop("format_version")
    torch.save(payload, directory / "model.pt")
    assert load_checkpoint(directory).config == model.config


def test_load_rejects_newer_format(tmp_path):
    torch.manual_seed(7)
    model = AuricleModel.tiny()
    directory = save_checkpoint(model, tmp_path / "ckpt")
    payload = torch.load(directory / "model.pt", weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, directory / "model.pt")
    with pytest.raises(CheckpointError, match="newer than supported"):
        load_checkpoint(directory)

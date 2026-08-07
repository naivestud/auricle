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

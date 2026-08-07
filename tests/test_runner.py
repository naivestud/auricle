import json
from pathlib import Path

import pytest
import torch

from auricle.errors import ManifestError
from auricle.eval.manifest import load_manifest, save_manifest
from auricle.eval.runner import evaluate_manifest
from auricle.model import AuricleModel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(13)
    return AuricleModel.tiny()


def _write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_manifest_basic(tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(manifest, [{"audio": "a.wav", "text": "hello"}])
    items = load_manifest(manifest)
    assert len(items) == 1
    assert items[0].audio == "a.wav"
    assert items[0].text == "hello"


def test_load_manifest_skips_blank_lines(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"audio":"a.wav","text":"x"}\n\n{"audio":"b.wav","text":"y"}\n')
    assert len(load_manifest(manifest)) == 2


def test_load_manifest_missing_file(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.jsonl")


def test_load_manifest_bad_json(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("{not json}\n")
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_manifest(manifest)


def test_load_manifest_missing_keys(tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(manifest, [{"audio": "a.wav"}])
    with pytest.raises(ManifestError, match="'audio' and 'text'"):
        load_manifest(manifest)


def test_save_and_reload_roundtrip(tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(manifest, [{"audio": "a.wav", "text": "x"}, {"audio": "b.wav", "text": "y"}])
    items = load_manifest(manifest)
    out = tmp_path / "out.jsonl"
    save_manifest(items, out)
    assert load_manifest(out) == items


def test_evaluate_manifest_end_to_end(model, tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(
        manifest,
        [
            {"audio": str(FIXTURES / "tone_1s.wav"), "text": "a steady tone"},
            {"audio": str(FIXTURES / "sweep_2s.wav"), "text": "a rising sweep"},
        ],
    )
    items = load_manifest(manifest)
    report = evaluate_manifest(model, items)

    assert report.n_samples == 2
    assert report.wer_micro >= 0.0
    assert report.cer_micro >= 0.0
    assert len(report.per_sample) == 2
    assert {"n_samples", "wer_micro", "cer_micro", "per_sample"} <= set(report.to_dict())


def test_evaluate_manifest_with_root(model, tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(manifest, [{"audio": "tone_1s.wav", "text": "a tone"}])
    report = evaluate_manifest(model, load_manifest(manifest), root=FIXTURES)
    assert report.n_samples == 1

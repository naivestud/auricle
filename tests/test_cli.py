import json
from pathlib import Path

import pytest

from auricle.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "usage: auricle" in capsys.readouterr().out.lower()


def test_transcribe_fixture(capsys):
    rc = main(["transcribe", str(FIXTURES / "tone_1s.wav")])
    assert rc == 0
    out = capsys.readouterr().out
    assert isinstance(out, str)


def test_stream_fixture(capsys):
    rc = main(["stream", str(FIXTURES / "tone_1s.wav"), "--chunk-seconds", "0.5"])
    assert rc == 0


def test_caption_with_echo(capsys):
    rc = main(["caption", str(FIXTURES / "tone_1s.wav"), "--backend", "echo"])
    assert rc == 0
    assert "duration=" in capsys.readouterr().out


def test_ask_with_echo(capsys):
    rc = main(
        ["ask", str(FIXTURES / "tone_1s.wav"), "--question", "what is it", "--backend", "echo"]
    )
    assert rc == 0
    assert "what is it" in capsys.readouterr().out


def test_eval_writes_report(tmp_path, capsys):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"audio": str(FIXTURES / "tone_1s.wav"), "text": "tone"}) + "\n")
    out_path = tmp_path / "report.json"

    rc = main(["eval", str(manifest), "--out", str(out_path)])
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["n_samples"] == 1
    assert "wer_micro" in report

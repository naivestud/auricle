import json
from pathlib import Path

from auricle.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_backends_lists_registered(capsys):
    assert main(["backends"]) == 0
    names = capsys.readouterr().out.split()
    assert "echo" in names
    assert "openai" in names
    assert "scripted" in names


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "usage: auricle" in capsys.readouterr().out.lower()


def test_transcribe_fixture(capsys):
    rc = main(["transcribe", str(FIXTURES / "tone_1s.wav")])
    assert rc == 0
    out = capsys.readouterr().out
    assert isinstance(out, str)


def test_transcribe_json_output(capsys):
    rc = main(["transcribe", str(FIXTURES / "tone_1s.wav"), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audio"].endswith("tone_1s.wav")
    assert isinstance(payload["text"], str)


def test_stream_fixture(capsys):
    rc = main(
        [
            "stream",
            str(FIXTURES / "tone_1s.wav"),
            "--chunk-seconds",
            "0.5",
            "--overlap-seconds",
            "0.1",
        ]
    )
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


def test_info_prints_metadata(capsys):
    rc = main(["info", str(FIXTURES / "tone_1s.wav")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "duration:     1.000 s" in out
    assert "sample rate:  16000 Hz" in out
    assert "bit depth:    16" in out
    assert "acoustics:" in out


def test_info_missing_file_exits_nonzero(capsys):
    rc = main(["info", "/does/not/exist.wav"])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_transcribe_missing_file_exits_nonzero(capsys):
    rc = main(["transcribe", "/does/not/exist.wav"])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_caption_unknown_backend_exits_nonzero(capsys):
    rc = main(["caption", str(FIXTURES / "tone_1s.wav"), "--backend", "nope"])
    assert rc == 2
    assert "auricle:" in capsys.readouterr().err


def _save_checkpoint(tmp_path):
    import torch

    from auricle.checkpoint import save_checkpoint
    from auricle.model import AuricleModel

    torch.manual_seed(0)
    return save_checkpoint(AuricleModel.tiny(), tmp_path / "ckpt")


def test_transcribe_with_checkpoint(tmp_path, capsys):
    ckpt = _save_checkpoint(tmp_path)
    rc = main(["transcribe", str(FIXTURES / "tone_1s.wav"), "--checkpoint", str(ckpt)])
    assert rc == 0
    assert isinstance(capsys.readouterr().out, str)


def test_stream_with_checkpoint(tmp_path, capsys):
    ckpt = _save_checkpoint(tmp_path)
    rc = main(
        [
            "stream",
            str(FIXTURES / "tone_1s.wav"),
            "--chunk-seconds",
            "0.5",
            "--overlap-seconds",
            "0.1",
            "--checkpoint",
            str(ckpt),
        ]
    )
    assert rc == 0


def test_checkpoint_missing_dir_exits_nonzero(tmp_path, capsys):
    rc = main(["transcribe", str(FIXTURES / "tone_1s.wav"), "--checkpoint", str(tmp_path / "nope")])
    assert rc == 2
    assert "auricle:" in capsys.readouterr().err

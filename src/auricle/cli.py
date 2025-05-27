"""auricle command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auricle import __version__
from auricle.checkpoint import load_checkpoint
from auricle.errors import AuricleError
from auricle.model import AuricleModel


def _load_model(checkpoint: str | None):
    if checkpoint:
        return load_checkpoint(checkpoint)
    return AuricleModel.tiny()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auricle",
        description="Streaming speech understanding: ASR, captioning and speech QA.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="print the version and exit")

    def add_model_args(sub):
        sub.add_argument("--checkpoint", help="checkpoint directory to load weights from")

    transcribe = subparsers.add_parser("transcribe", help="transcribe a WAV file")
    transcribe.add_argument("audio", help="path to a WAV file")
    transcribe.add_argument("--json", action="store_true", help="emit a JSON object with the text")
    add_model_args(transcribe)

    stream = subparsers.add_parser("stream", help="stream a WAV file through the ASR")
    stream.add_argument("audio", help="path to a WAV file")
    stream.add_argument("--chunk-seconds", type=float, default=2.0)
    stream.add_argument("--overlap-seconds", type=float, default=0.5)
    add_model_args(stream)

    caption = subparsers.add_parser("caption", help="caption a WAV file")
    caption.add_argument("audio", help="path to a WAV file")
    caption.add_argument("--backend", default="echo", help="LLM backend name")
    add_model_args(caption)

    ask = subparsers.add_parser("ask", help="ask a question about a WAV file")
    ask.add_argument("audio", help="path to a WAV file")
    ask.add_argument("--question", required=True, help="question to ask")
    ask.add_argument("--backend", default="echo", help="LLM backend name")
    add_model_args(ask)

    eval_parser = subparsers.add_parser("eval", help="evaluate a model on a manifest")
    eval_parser.add_argument("manifest", help="path to a JSONL manifest")
    eval_parser.add_argument("--root", help="directory that relative audio paths resolve against")
    eval_parser.add_argument("--out", help="write the JSON report here (default: stdout)")
    add_model_args(eval_parser)

    info = subparsers.add_parser("info", help="print WAV metadata and an acoustic summary")
    info.add_argument("audio", help="path to a WAV file")

    return parser


def _run_transcribe(args: argparse.Namespace) -> int:
    import json

    from auricle.pipeline.asr import transcribe

    model = _load_model(args.checkpoint)
    text = transcribe(model, args.audio)
    if args.json:
        print(json.dumps({"audio": args.audio, "text": text}))
    else:
        print(text)
    return 0


def _run_stream(args: argparse.Namespace) -> int:
    from auricle.audio.wav import read_wav
    from auricle.streaming.asr import StreamingASR

    model = _load_model(args.checkpoint)
    samples, _ = read_wav(args.audio)
    asr = StreamingASR(model, args.chunk_seconds, args.overlap_seconds)

    block = int(args.chunk_seconds * asr.scheduler.sample_rate)
    for start in range(0, len(samples), block):
        print(asr.feed(samples[start : start + block]))
    print(asr.finalize())
    return 0


def _run_caption(args: argparse.Namespace) -> int:
    from auricle.llm import get_backend
    from auricle.pipeline.caption import caption_audio

    model = _load_model(args.checkpoint)
    backend = get_backend(args.backend)
    print(caption_audio(model, backend, args.audio))
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    from auricle.llm import get_backend
    from auricle.pipeline.qa import answer_question

    model = _load_model(args.checkpoint)
    backend = get_backend(args.backend)
    print(answer_question(model, backend, args.audio, args.question))
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    import json

    from auricle.eval.manifest import load_manifest
    from auricle.eval.runner import evaluate_manifest

    model = _load_model(args.checkpoint)
    report = evaluate_manifest(model, load_manifest(args.manifest), root=args.root)
    text = json.dumps(report.to_dict(), indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


def _run_info(args: argparse.Namespace) -> int:
    from auricle.audio.wav import read_wav, read_wav_info
    from auricle.constants import SAMPLE_RATE
    from auricle.pipeline.acoustics import summarize

    info = read_wav_info(args.audio)
    samples, _ = read_wav(args.audio)
    acoustic = summarize(samples, SAMPLE_RATE)

    print(f"file:         {args.audio}")
    print(f"duration:     {info.duration_seconds:.3f} s")
    print(f"sample rate:  {info.sample_rate} Hz")
    print(f"channels:     {info.n_channels}")
    print(f"bit depth:    {info.bit_depth}")
    print(f"acoustics:    {acoustic.describe()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "version":
            print(__version__)
            return 0
        if args.command == "transcribe":
            return _run_transcribe(args)
        if args.command == "stream":
            return _run_stream(args)
        if args.command == "caption":
            return _run_caption(args)
        if args.command == "ask":
            return _run_ask(args)
        if args.command == "eval":
            return _run_eval(args)
        if args.command == "info":
            return _run_info(args)
    except FileNotFoundError as exc:
        print(f"auricle: no such file: {exc.filename or args}", file=sys.stderr)
        return 2
    except AuricleError as exc:
        print(f"auricle: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

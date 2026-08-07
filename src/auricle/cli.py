"""auricle command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auricle import __version__
from auricle.checkpoint import load_checkpoint
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
    add_model_args(transcribe)

    return parser


def _run_transcribe(args) -> int:
    from auricle.pipeline.asr import transcribe

    model = _load_model(args.checkpoint)
    print(transcribe(model, args.audio))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "transcribe":
        return _run_transcribe(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

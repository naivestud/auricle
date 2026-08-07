"""auricle command line interface."""

from __future__ import annotations

import argparse
import sys

from auricle import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auricle",
        description="Streaming speech understanding: ASR, captioning and speech QA.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="print the version and exit")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

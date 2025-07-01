#!/usr/bin/env python
"""Benchmark transcription throughput as a real-time factor.

Feeds progressively longer synthetic clips through the offline pipeline and
reports seconds of audio processed per second of wall time. A real-time
factor below 1.0 means faster than real time.

    python examples/benchmark.py
    python examples/benchmark.py --durations 1 2 4 --checkpoint runs/exp1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from auricle import AuricleModel, load_checkpoint
from auricle.pipeline.asr import transcribe


def synth_clip(seconds: float, rate: int = 16_000) -> np.ndarray:
    rng = np.random.default_rng(int(seconds * 1000))
    return (0.2 * rng.standard_normal(int(seconds * rate))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--checkpoint", help="checkpoint directory with trained weights")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint) if args.checkpoint else AuricleModel.tiny()

    print(f"{'audio (s)':>10} {'wall (s)':>10} {'RTF':>8}")
    for seconds in args.durations:
        audio = synth_clip(seconds)
        start = time.perf_counter()
        transcribe(model, audio, sample_rate=16_000)
        wall = time.perf_counter() - start
        rtf = wall / seconds if seconds else float("inf")
        print(f"{seconds:>10.2f} {wall:>10.4f} {rtf:>8.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stream synthetic audio through the incremental ASR.

Demonstrates the feed/finalize loop. As with the other examples the default
model is untrained, so treat the text as a placeholder.

    python examples/streaming_demo.py
"""

from __future__ import annotations

import numpy as np

from auricle import AuricleModel, StreamingASR


def synth_stream(seconds: float = 4.0, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    sweep = 0.5 * np.sin(2 * np.pi * (220.0 + 330.0 * t) * t)
    return sweep.astype(np.float32)


def main() -> None:
    model = AuricleModel.tiny()
    asr = StreamingASR(model, chunk_seconds=1.0, overlap_seconds=0.25)

    audio = synth_stream()
    block = 4_000  # 0.25 s blocks, smaller than the chunk to exercise buffering

    for start in range(0, len(audio), block):
        text = asr.feed(audio[start : start + block])
        if text:
            print(f"\r{text}", end="", flush=True)

    print(f"\r{asr.finalize()}")


if __name__ == "__main__":
    main()

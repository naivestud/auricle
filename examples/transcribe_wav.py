#!/usr/bin/env python
"""Transcribe a WAV file.

Uses a randomly initialised tiny model by default, so the output text is
nonsense — point ``--checkpoint`` at a trained checkpoint for real results.

    python examples/transcribe_wav.py path/to/audio.wav
    python examples/transcribe_wav.py audio.wav --checkpoint runs/exp1
"""

from __future__ import annotations

import argparse

import numpy as np

from auricle import AuricleModel, load_checkpoint, transcribe


def synth_tone(seconds: float = 1.0, freq: float = 440.0, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="path to a WAV file (omit to use a synth tone)")
    parser.add_argument("--checkpoint", help="checkpoint directory with trained weights")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint) if args.checkpoint else AuricleModel.tiny()

    if args.audio:
        print(transcribe(model, args.audio))
    else:
        print("(no audio given, transcribing a synthetic 440 Hz tone)")
        print(repr(transcribe(model, synth_tone(), sample_rate=16_000)))


if __name__ == "__main__":
    main()

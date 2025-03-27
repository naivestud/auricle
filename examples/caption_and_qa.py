#!/usr/bin/env python
"""Caption an audio clip and answer a question about it.

Both tasks pipe the encoder's transcript plus acoustic statistics into an
LLM backend. The default backend is ``echo``, which just returns the prompt
— handy for seeing exactly what gets sent. Swap in a real backend:

    python examples/caption_and_qa.py --backend openai
"""

from __future__ import annotations

import argparse

import numpy as np

from auricle import AuricleModel, answer_question, caption_audio, get_backend


def synth_clip(seconds: float = 2.0, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.4 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="echo")
    args = parser.parse_args()

    model = AuricleModel.tiny()
    backend = get_backend(args.backend)
    audio = synth_clip()

    caption = caption_audio(model, backend, audio, sample_rate=16_000)
    answer = answer_question(model, backend, audio, "Is anyone speaking?", sample_rate=16_000)

    print("Caption:")
    print(f"  {caption[:200]}")
    print("Answer:")
    print(f"  {answer[:200]}")


if __name__ == "__main__":
    main()

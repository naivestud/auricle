#!/usr/bin/env python
"""Compare greedy and beam CTC decoding on the same audio.

Beam search considers multiple alignments and can produce a different (often
better) transcript than the single best frame path. With an untrained model
both outputs are meaningless, but the mechanics are identical to real use.

    python examples/beam_decode_demo.py
"""

from __future__ import annotations

import argparse

import torch

from auricle import AuricleModel
from auricle.asr.decode import beam_decode, greedy_decode, greedy_decode_with_confidence


def synth_clip(seconds: float = 2.0, rate: int = 16_000) -> torch.Tensor:
    t = torch.arange(int(seconds * rate), dtype=torch.float32) / rate
    return 0.5 * torch.sin(2 * torch.pi * 440.0 * t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args()

    model = AuricleModel.tiny().eval()
    waveform = synth_clip()

    with torch.no_grad():
        logits = model(waveform)

    greedy = greedy_decode(logits, model.vocab)[0]
    beam = beam_decode(logits, model.vocab, beam_width=args.beam_width)[0]
    ((text, confidence),) = greedy_decode_with_confidence(logits, model.vocab)

    print(f"greedy      : {greedy!r}")
    print(f"beam (w={args.beam_width}): {beam!r}")
    print(f"confidence  : {confidence:.3f}")


if __name__ == "__main__":
    main()

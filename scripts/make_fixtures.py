"""Generate the small WAV fixtures used by the test suite.

Run from the repository root:

    python scripts/make_fixtures.py

Fixtures are intentionally tiny (a few seconds of 16 kHz audio) so they can
live in git without a binary-store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from auricle.audio.io import write_wav

RATE = 16_000
HERE = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _tone(seconds: float, freq: float, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(RATE * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sweep(seconds: float, f0: float, f1: float, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(RATE * seconds), endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * seconds))
    return (amp * np.sin(phase)).astype(np.float32)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    write_wav(HERE / "tone_1s.wav", _tone(1.0, 440.0), RATE)
    write_wav(HERE / "silence_1s.wav", np.zeros(RATE, dtype=np.float32), RATE)
    write_wav(HERE / "sweep_2s.wav", _sweep(2.0, 220.0, 1760.0), RATE)
    write_wav(HERE / "chirp_3s.wav", _sweep(3.0, 110.0, 880.0), RATE)
    for p in sorted(HERE.glob("*.wav")):
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

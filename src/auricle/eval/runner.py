"""Run a model over a manifest and aggregate WER / CER."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from auricle.eval.manifest import ManifestItem
from auricle.eval.metrics import _score_chars, _score_words, cer, wer
from auricle.pipeline.asr import transcribe


@dataclass
class EvalReport:
    """Aggregated results over a manifest."""

    n_samples: int
    wer_micro: float
    cer_micro: float
    per_sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "wer_micro": round(self.wer_micro, 6),
            "cer_micro": round(self.cer_micro, 6),
            "per_sample": self.per_sample,
        }


def evaluate_manifest(
    model, items: list[ManifestItem], root: str | Path | None = None
) -> EvalReport:
    """Transcribe every item and score it against the reference.

    ``root`` is prepended to relative ``audio`` paths.
    """
    root_path = Path(root) if root is not None else None

    word_edits = word_total = 0
    char_edits = char_total = 0
    per_sample: list[dict] = []

    for item in items:
        audio_path = Path(item.audio)
        if root_path is not None and not audio_path.is_absolute():
            audio_path = root_path / audio_path
        hypothesis = transcribe(model, audio_path)
        # print(f"[debug] {item.audio}: {hypothesis!r}")

        w_edits, w_len = _score_words(item.text, hypothesis)
        c_edits, c_len = _score_chars(item.text, hypothesis)
        word_edits += w_edits
        word_total += w_len
        char_edits += c_edits
        char_total += c_len

        per_sample.append(
            {
                "audio": item.audio,
                "reference": item.text,
                "hypothesis": hypothesis,
                "wer": wer(item.text, hypothesis),
                "cer": cer(item.text, hypothesis),
            }
        )

    return EvalReport(
        n_samples=len(items),
        wer_micro=word_edits / word_total if word_total else 0.0,
        cer_micro=char_edits / char_total if char_total else 0.0,
        per_sample=per_sample,
    )

"""JSONL evaluation manifests.

A manifest is a file with one JSON object per line::

    {"audio": "path/to/clip.wav", "text": "the reference transcript"}

Relative ``audio`` paths are resolved against a root directory supplied at
evaluation time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from auricle.errors import ManifestError


@dataclass(frozen=True)
class ManifestItem:
    audio: str
    text: str


def load_manifest(path: str | Path) -> list[ManifestItem]:
    """Read a JSONL manifest into a list of items."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")

    items: list[ManifestItem] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(record, dict) or "audio" not in record or "text" not in record:
            raise ManifestError(f"{path}:{lineno}: needs 'audio' and 'text' keys")
        items.append(ManifestItem(str(record["audio"]), str(record["text"])))

    if not items:
        raise ManifestError(f"manifest is empty: {path}")
    return items


def save_manifest(items: list[ManifestItem], path: str | Path) -> None:
    """Write items back out as JSONL."""
    path = Path(path)
    lines = [json.dumps({"audio": item.audio, "text": item.text}) for item in items]
    path.write_text("\n".join(lines) + "\n")

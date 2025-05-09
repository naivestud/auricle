"""Checkpoint saving and loading.

A checkpoint is a directory holding a JSON config and a ``model.pt`` weight
file, in the spirit of huggingface-style ``save_pretrained`` layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from auricle.asr.vocab import CharVocabulary
from auricle.encoder.config import EncoderConfig
from auricle.errors import CheckpointError
from auricle.types import ModelLike

if TYPE_CHECKING:
    from auricle.model import AuricleModel

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.pt"


def save_checkpoint(model: ModelLike, directory: str | Path) -> Path:
    """Write ``model`` into ``directory`` (created if missing)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    config_text = json.dumps(model.config.to_dict(), indent=2, sort_keys=True)
    (directory / CONFIG_NAME).write_text(config_text + "\n")

    payload = {
        "state_dict": model.state_dict(),
        "vocab": list(model.vocab.chars),
    }
    torch.save(payload, directory / WEIGHTS_NAME)
    return directory


def load_checkpoint(directory: str | Path, map_location: str = "cpu") -> AuricleModel:
    """Load an :class:`auricle.model.AuricleModel` from ``directory``."""
    from auricle.model import AuricleModel  # local import avoids a cycle

    directory = Path(directory)
    config_path = directory / CONFIG_NAME
    weights_path = directory / WEIGHTS_NAME
    if not config_path.is_file() or not weights_path.is_file():
        raise CheckpointError(
            f"{directory} is not a checkpoint directory (expected {CONFIG_NAME} and {WEIGHTS_NAME})"
        )

    try:
        config = EncoderConfig.from_dict(json.loads(config_path.read_text()))
    except (json.JSONDecodeError, TypeError) as exc:
        raise CheckpointError(f"could not parse {config_path}: {exc}") from exc

    try:
        payload = torch.load(weights_path, map_location=map_location, weights_only=True)
    except Exception as exc:  # torch raises several error types across versions
        raise CheckpointError(f"could not load weights from {weights_path}: {exc}") from exc

    chars = payload.get("vocab")
    vocab = CharVocabulary(tuple(chars)) if chars else CharVocabulary()

    model = AuricleModel(config, vocab=vocab)
    model.load_state_dict(payload["state_dict"])
    return model

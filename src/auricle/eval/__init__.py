"""Evaluation tooling: metrics, manifests, and the eval runner."""

from auricle.eval.manifest import ManifestItem, load_manifest, save_manifest
from auricle.eval.metrics import cer, edit_distance, ser, wer
from auricle.eval.runner import EvalReport, evaluate_manifest

__all__ = [
    "EvalReport",
    "ManifestItem",
    "cer",
    "edit_distance",
    "evaluate_manifest",
    "load_manifest",
    "save_manifest",
    "ser",
    "wer",
]

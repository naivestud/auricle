"""Exception hierarchy for auricle."""

from __future__ import annotations


class AuricleError(Exception):
    """Base class for every error raised by auricle."""


class AudioFormatError(AuricleError):
    """The audio data is malformed in some way."""


class UnsupportedFormatError(AudioFormatError):
    """The container or encoding is not supported."""


class SampleRateError(AudioFormatError):
    """The sample rate does not match what the pipeline expects."""


class CheckpointError(AuricleError):
    """A checkpoint could not be read or is incompatible."""


class BackendError(AuricleError):
    """Something went wrong in an LLM backend."""


class BackendNotFoundError(BackendError):
    """No backend is registered under the requested name."""


class ManifestError(AuricleError):
    """An evaluation manifest is missing, malformed or empty."""

import pytest

from auricle import errors


def test_all_errors_subclass_base():
    for name in (
        "AudioFormatError",
        "UnsupportedFormatError",
        "SampleRateError",
        "CheckpointError",
        "BackendError",
        "BackendNotFoundError",
        "ManifestError",
    ):
        cls = getattr(errors, name)
        assert issubclass(cls, errors.AuricleError), name


def test_audio_format_subtypes():
    assert issubclass(errors.UnsupportedFormatError, errors.AudioFormatError)
    assert issubclass(errors.SampleRateError, errors.AudioFormatError)


def test_backend_subtypes():
    assert issubclass(errors.BackendNotFoundError, errors.BackendError)


def test_base_is_exception():
    assert issubclass(errors.AuricleError, Exception)


def test_catch_base_catches_all():
    for exc in (
        errors.AudioFormatError("x"),
        errors.CheckpointError("x"),
        errors.BackendNotFoundError("x"),
        errors.ManifestError("x"),
    ):
        with pytest.raises(errors.AuricleError):
            raise exc


def test_error_messages_preserved():
    exc = errors.SampleRateError("expected 16 kHz")
    assert str(exc) == "expected 16 kHz"

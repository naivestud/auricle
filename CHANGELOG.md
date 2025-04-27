# Changelog

All notable changes to auricle are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-08

First release of the core toolkit.

### Added
- Whisper-style audio encoder (log-mel spectrogram, conv frontend, pre-norm
  transformer) sized by `EncoderConfig`, with a `tiny()` preset.
- Character-level CTC head and greedy decoding.
- Offline transcription via `auricle.transcribe` and the `auricle transcribe`
  CLI subcommand.
- Checkpoint save/load (`save_checkpoint` / `load_checkpoint`).
- WAV I/O and framing helpers in `auricle.audio`.
- WER / CER metrics and an edit-distance implementation in `auricle.eval`.
- Test suite, ruff lint/format, and CI across Python 3.10–3.12.

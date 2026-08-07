# Changelog

All notable changes to auricle are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-08

### Added
- Pluggable LLM backends: `LLMBackend` protocol, a deterministic `echo`
  backend, an OpenAI-compatible HTTP backend, and an optional HuggingFace
  backend (`auricle[hf]`), all discoverable via `get_backend`.
- Audio captioning (`caption_audio`) and speech question answering
  (`answer_question`) pipelines.
- JSONL evaluation manifests and an aggregate runner
  (`evaluate_manifest` -> `EvalReport`) with micro WER / CER.
- `auricle caption`, `auricle ask`, and `auricle eval` CLI subcommands.
- Examples for transcription, streaming, and captioning/QA.

## [0.2.0] - 2026-08-08

### Added
- Streaming ASR: `StreamScheduler` for overlapping chunks and
  `StreamingASR.feed` / `finalize` for incremental decoding.
- `merge_transcripts` overlap deduplication, independent of how callers slice
  their pushes.
- `auricle stream` CLI subcommand.

### Fixed
- Trailing partial chunks no longer re-decode audio already covered by the
  previous chunk's overlap.

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

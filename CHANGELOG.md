# Changelog

All notable changes to auricle are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CTC prefix beam search (`beam_decode`) and greedy decoding with mean-frame
  confidence (`greedy_decode_with_confidence`, `transcribe_with_confidence`).
- Keyword spotting over transcripts (`find_keywords`, `spot_keywords`).
- Streaming throughput statistics (`StreamStats` with a real-time factor) and
  scheduler introspection (`pending_seconds`, `samples_emitted`, `repr`).
- Sentence error rate metric (`ser`) and per-sample timing / real-time factor
  in evaluation reports.
- LLM backend utilities: `ScriptedBackend` for deterministic tests,
  `CachingBackend` memoization, and retry with exponential backoff in the
  OpenAI-compatible backend.
- Audio helpers: `read_wav_info`, `normalize_peak`, `normalize_rms`,
  `trim_silence`, `resample_linear`, and deterministic signal generators
  (`sine`, `chirp`, `silence`, `white_noise`).
- `AuricleModel.count_parameters` / `.summary`, `EncoderConfig.small` /
  `.base` presets and `validate()`.
- CLI: `auricle info`, `auricle backends`, and `--json` output for
  `transcribe`.
- `py.typed` marker (PEP 561) and examples for beam decoding, benchmarking and
  custom backends.

### Changed
- Checkpoints record a `format_version`; loading a newer-than-supported
  version raises `CheckpointError`.
- Vocabulary `encode` reports the offending character position; added
  `encode_lenient`, `can_encode` and `blank_id`.
- `write_wav` rejects non-finite samples; WAV read errors surface as
  `UnsupportedFormatError`.

### Fixed
- The OpenAI backend omits the `Authorization` header when no API key is set.

### Performance
- Vectorised CTC token collapsing, edit distance, and attention (fused
  scaled-dot-product); streaming scheduler now appends to a growable buffer
  instead of re-copying the backlog on every push.

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

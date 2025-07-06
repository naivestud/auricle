# API reference

The public surface lives in `auricle` and its sub-packages. Everything
documented here is re-exported from the top level unless noted.

## Models and checkpoints

- `AuricleModel(config, vocab=None)` — waveform → CTC logits. Use
  `AuricleModel.tiny()` for a randomly initialised small model.
  - `.transcribe(waveform) -> list[str]` — greedy-decode a batch.
  - `.transcribe_with_confidence(waveform) -> list[tuple[str, float]]` — greedy
    decode plus a mean-frame confidence per item.
  - `.count_parameters(trainable_only=True) -> int` / `.summary() -> str` —
    size introspection.
- `EncoderConfig` — encoder hyper-parameters. Presets `tiny()`, `small()` and
  `base()` span a capacity range; `validate()` checks consistency and
  `to_dict()`/`from_dict()` serialise it.
- `save_checkpoint(model, directory)` / `load_checkpoint(directory)` — write or
  read a `config.json` + `model.pt` checkpoint directory. Payloads carry a
  `format_version`; loading a newer-than-supported version raises.

## Pipelines

- `transcribe(model, audio, sample_rate=None) -> str` — offline transcription
  of a path, array or tensor.
- `transcribe_with_confidence(model, audio, sample_rate=None) -> (str, float)`
  — transcription plus a confidence score (needs a concrete `AuricleModel`).
- `caption_audio(model, backend, audio, sample_rate=None) -> str` — caption a
  clip using an LLM backend.
- `answer_question(model, backend, audio, question, sample_rate=None) -> str`
  — ask a question about the spoken content.
- `find_keywords(transcript, keywords) -> list[str]` /
  `spot_keywords(model, audio, keywords, sample_rate=None) -> list[str]` —
  keyword spotting over the normalised transcript.

## Decoding and vocabulary

- `auricle.asr.greedy_decode(logits, vocab) -> list[str]` — argmax path decode.
- `auricle.asr.greedy_decode_with_confidence(logits, vocab)` — greedy decode
  with mean-frame confidence.
- `auricle.asr.beam_decode(logits, vocab, beam_width=10) -> list[str]` — CTC
  prefix beam search.
- `auricle.asr.collapse_tokens(tokens, vocab) -> str` — merge repeats and drop
  blanks.
- `CharVocabulary` — character ↔ id mapping; `encode`, `encode_lenient`,
  `decode`, `can_encode`, `blank_id`.

## Streaming

- `StreamingASR(model, chunk_seconds=2.0, overlap_seconds=0.5)`
  - `.feed(samples) -> str` — feed a block, get the committed transcript.
  - `.finalize() -> str` — flush the trailing chunk.
  - `.process(blocks) -> Iterator[str]` — convenience over an iterable.
  - `.stats -> StreamStats` — samples fed, chunks decoded, decode time and a
    `real_time_factor`.
- `StreamScheduler` — the chunk planner, if you need it directly; exposes
  `pending_seconds`, `samples_emitted` and `len()`.
- `merge_transcripts(committed, hypothesis, window=12)` — the
  overlap-deduplication step.

## LLM backends

- `LLMBackend` — the protocol; implement `generate(prompt, *, max_new_tokens)`.
- `EchoBackend`, `OpenAICompatBackend`, `HuggingFaceBackend` — bundled
  implementations (the last requires the `hf` extra).
- `ScriptedBackend(responses, cycle=True)` — replays canned responses, useful
  in tests.
- `CachingBackend(backend)` — memoizes another backend by prompt; exposes
  `hits`, `cache_size` and `clear()`.
- `OpenAICompatBackend(..., max_retries=0, backoff_seconds=0.5, sleep=...)` —
  optional retry with exponential backoff for transient failures.
- `get_backend(name, **kwargs)` — instantiate a backend by registered name.
- `register_backend` — class decorator to register a custom backend.

## Evaluation

- `wer(reference, hypothesis)` / `cer(reference, hypothesis)` — error rates
  after normalisation.
- `ser(references, hypotheses)` — sentence error rate over paired lists.
- `edit_distance(a, b)` — Levenshtein distance over sequences.
- `load_manifest(path)` / `save_manifest(items, path)` — JSONL manifests.
- `evaluate_manifest(model, items, root=None) -> EvalReport` — score a whole
  manifest. The report tracks `total_seconds`/`audio_seconds` and a
  `real_time_factor`; `to_dict()` serialises the result.

## Audio utilities

- `auricle.audio.read_wav(path)` / `write_wav(path, samples, rate)` — 16-bit
  PCM WAV I/O. `write_wav` rejects non-finite samples.
- `auricle.audio.read_wav_info(path) -> WavInfo` — header-only metadata
  (`duration_seconds`, `bit_depth`, channels, rate).
- `auricle.audio.resample_linear(samples, orig_sr, target_sr)` — linear
  interpolation resampler.
- `auricle.audio.normalize_peak` / `normalize_rms` / `trim_silence` — gain and
  silence helpers.
- `auricle.audio.sine` / `chirp` / `silence` / `white_noise` — deterministic
  test-signal generators.
- `auricle.audio.LogMelSpectrogram` — the feature extractor.

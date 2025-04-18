# API reference

The public surface lives in `auricle` and its sub-packages. Everything
documented here is re-exported from the top level unless noted.

## Models and checkpoints

- `AuricleModel(config, vocab=None)` — waveform → CTC logits. Use
  `AuricleModel.tiny()` for a randomly initialised small model.
  - `.transcribe(waveform) -> list[str]` — greedy-decode a batch.
- `EncoderConfig` — encoder hyper-parameters; `EncoderConfig.tiny()` gives a
  small preset, `to_dict()`/`from_dict()` serialise it.
- `save_checkpoint(model, directory)` / `load_checkpoint(directory)` — write
  or read a `config.json` + `model.pt` checkpoint directory.

## Pipelines

- `transcribe(model, audio, sample_rate=None) -> str` — offline transcription
  of a path, array or tensor.
- `caption_audio(model, backend, audio, sample_rate=None) -> str` — caption a
  clip using an LLM backend.
- `answer_question(model, backend, audio, question, sample_rate=None) -> str`
  — ask a question about the spoken content.

## Streaming

- `StreamingASR(model, chunk_seconds=2.0, overlap_seconds=0.5)`
  - `.feed(samples) -> str` — feed a block, get the committed transcript.
  - `.finalize() -> str` — flush the trailing chunk.
  - `.process(blocks) -> Iterator[str]` — convenience over an iterable.
- `StreamScheduler` — the chunk planner, if you need it directly.
- `merge_transcripts(committed, hypothesis)` — the overlap-deduplication step.

## LLM backends

- `LLMBackend` — the protocol; implement `generate(prompt, *, max_new_tokens)`.
- `EchoBackend`, `OpenAICompatBackend`, `HuggingFaceBackend` — bundled
  implementations (the last requires the `hf` extra).
- `get_backend(name, **kwargs)` — instantiate a backend by registered name.
- `register_backend` — class decorator to register a custom backend.

## Evaluation

- `wer(reference, hypothesis)` / `cer(reference, hypothesis)` — error rates
  after normalisation.
- `edit_distance(a, b)` — Levenshtein distance over sequences.
- `load_manifest(path)` / `save_manifest(items, path)` — JSONL manifests.
- `evaluate_manifest(model, items, root=None) -> EvalReport` — score a whole
  manifest; `EvalReport.to_dict()` serialises the result.

## Audio utilities

- `auricle.audio.read_wav(path)` / `write_wav(path, samples, rate)` — 16-bit
  PCM WAV I/O (in `auricle.audio.wav`).
- `auricle.audio.LogMelSpectrogram` — the feature extractor.

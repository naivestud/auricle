# auricle

Streaming speech understanding toolkit: ASR, audio captioning and speech
question answering on top of a whisper-style audio encoder with pluggable
LLM backends.

auricle implements the *architecture and tooling* — a from-scratch
whisper-style encoder, chunked streaming, caption/QA pipelines, an evaluation
suite and a CLI. Models are randomly initialised by default; point
`load_checkpoint` at trained weights to get real transcripts.

## Features

- **Whisper-style encoder** — log-mel spectrogram, conv frontend, pre-norm
  transformer, CTC head. Sized by a small `EncoderConfig`.
- **Streaming ASR** — feed audio in any block size; overlapping chunks are
  decoded and merged into a growing transcript.
- **Audio captioning** — transcript + acoustic statistics into an LLM backend.
- **Speech QA** — ask a question about spoken content.
- **Pluggable LLM backends** — `echo` (deterministic), OpenAI-compatible HTTP,
  optional HuggingFace. Register your own with a decorator.
- **Evaluation suite** — WER/CER, JSONL manifests, an aggregate report.
- **CLI** — `auricle transcribe | stream | caption | ask | eval`.

## Quickstart

```bash
pip install -e ".[dev]"

# offline transcription (tiny random model)
auricle transcribe tests/fixtures/tone_1s.wav

# streaming
auricle stream tests/fixtures/sweep_2s.wav --chunk-seconds 1 --overlap-seconds 0.25

# caption + QA with the deterministic echo backend
auricle caption tests/fixtures/tone_1s.wav --backend echo
auricle ask tests/fixtures/tone_1s.wav --question "what is this?"

# evaluate against a manifest
auricle eval manifest.jsonl --root data/ --out report.json
```

Python:

```python
from auricle import AuricleModel, transcribe, StreamingASR

model = AuricleModel.tiny()
print(transcribe(model, "audio.wav"))
```

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [API reference](docs/api-reference.md)
- [Design notes](docs/design-notes.md)

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
ruff check . && ruff format --check .
```

## License

MIT

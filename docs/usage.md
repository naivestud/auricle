# Usage

## Installation

```bash
pip install -e .
# optional extras
pip install -e ".[dev]"   # tests, lint, pre-commit
pip install -e ".[hf]"    # HuggingFace backend
```

auricle needs Python 3.10+ and PyTorch. On machines without a GPU install the
CPU build first to avoid pulling CUDA wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## CLI

```bash
auricle transcribe recording.wav
auricle stream recording.wav --chunk-seconds 2 --overlap-seconds 0.5
auricle caption recording.wav --backend echo
auricle ask recording.wav --question "what is being discussed?"
auricle eval manifest.jsonl --root data/ --out report.json
auricle version
```

All subcommands accept `--checkpoint path/to/dir` to load trained weights
instead of the randomly initialised tiny model.

## Python API

### Offline transcription

```python
from auricle import AuricleModel, transcribe

model = AuricleModel.tiny()          # or: load_checkpoint("runs/exp1")
text = transcribe(model, "audio.wav")
```

### Streaming

```python
from auricle import AuricleModel, StreamingASR

model = AuricleModel.tiny()
asr = StreamingASR(model, chunk_seconds=2.0, overlap_seconds=0.5)
for block in mic_blocks:             # float32 mono numpy arrays
    partial = asr.feed(block)
    print(partial, end="\r")
print(asr.finalize())
```

### Captioning and QA

```python
from auricle import caption_audio, answer_question, get_backend

backend = get_backend("echo")        # or "openai", "huggingface"
caption = caption_audio(model, backend, "audio.wav")
answer = answer_question(model, backend, "audio.wav", "Who is speaking?")
```

### Evaluation

```python
from auricle import load_manifest, evaluate_manifest

items = load_manifest("manifest.jsonl")   # {"audio": ..., "text": ...}
report = evaluate_manifest(model, items, root="data/")
print(report.wer_micro, report.cer_micro)
```

## Checkpoints

```python
from auricle import save_checkpoint, load_checkpoint

save_checkpoint(model, "runs/exp1")       # writes config.json + model.pt
model = load_checkpoint("runs/exp1")
```

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
auricle transcribe recording.wav --json        # machine-readable output
auricle stream recording.wav --chunk-seconds 2 --overlap-seconds 0.5
auricle caption recording.wav --backend echo
auricle ask recording.wav --question "what is being discussed?"
auricle eval manifest.jsonl --root data/ --out report.json
auricle info recording.wav                     # WAV metadata + acoustics
auricle backends                               # list registered LLM backends
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

#### Confidence and beam decoding

`transcribe_with_confidence` also returns a mean-frame confidence in `[0, 1]`,
useful for filtering low-quality transcripts. For noisier audio, CTC beam
search explores multiple alignments:

```python
from auricle import transcribe_with_confidence
from auricle.asr import beam_decode
from auricle.pipeline.asr import to_waveform
import torch

text, confidence = transcribe_with_confidence(model, "audio.wav")

waveform = to_waveform("audio.wav")
with torch.no_grad():
    logits = model(waveform)          # (batch, time, vocab)
beams = beam_decode(logits, model.vocab, beam_width=8)
```

#### Keyword spotting

```python
from auricle.pipeline import spot_keywords

found = spot_keywords(model, "audio.wav", ["play music", "stop"])
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

# Throughput bookkeeping is available once audio has flowed:
print(asr.stats.chunks_decoded, asr.stats.real_time_factor)
```

### Captioning and QA

```python
from auricle import caption_audio, answer_question, get_backend

backend = get_backend("echo")        # or "openai", "huggingface"
caption = caption_audio(model, backend, "audio.wav")
answer = answer_question(model, backend, "audio.wav", "Who is speaking?")
```

Backends compose. Wrap one to cache repeat prompts or retry transient
failures:

```python
from auricle.llm import CachingBackend
from auricle.llm.openai_compat import OpenAICompatBackend

backend = CachingBackend(
    OpenAICompatBackend(model="my-local-model", max_retries=3, backoff_seconds=0.5)
)
```

### Evaluation

```python
from auricle import load_manifest, evaluate_manifest

items = load_manifest("manifest.jsonl")   # {"audio": ..., "text": ...}
report = evaluate_manifest(model, items, root="data/")
print(report.wer_micro, report.cer_micro)
print(report.real_time_factor)            # transcription speed vs. audio length
```

## Checkpoints

```python
from auricle import save_checkpoint, load_checkpoint

save_checkpoint(model, "runs/exp1")       # writes config.json + model.pt
model = load_checkpoint("runs/exp1")
```

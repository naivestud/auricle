# Examples

Runnable scripts showing each pipeline. All of them default to a randomly
initialised tiny model, so the text they produce is meaningless until you
point them at a trained checkpoint (`--checkpoint` where supported) or a real
LLM backend (`--backend`).

| Script | Shows |
| --- | --- |
| `transcribe_wav.py` | Offline transcription of a WAV file |
| `streaming_demo.py` | Incremental `feed`/`finalize` streaming ASR |
| `caption_and_qa.py` | Audio captioning and speech question answering |

Run them from the repository root after an editable install:

```bash
pip install -e ".[dev]"
python examples/streaming_demo.py
```

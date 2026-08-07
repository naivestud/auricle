# Architecture

auricle is organised as a small pipeline: audio in, text out, with an LLM
backend optionally turning that text (plus a few acoustic statistics) into
captions and answers.

```
 waveform ── LogMelSpectrogram ── ConvFrontend ── TransformerEncoder ── CTCHead ── logits
                                                                                     │
                                                                     greedy CTC decode │
                                                                                     ▼
                                                                                  text
```

## Modules

| Package | Responsibility |
| --- | --- |
| `auricle.audio` | WAV I/O, framing helpers, the log-mel spectrogram |
| `auricle.encoder` | The whisper-style audio encoder (config, attention, transformer, conv frontend) |
| `auricle.asr` | Character vocabulary, CTC head and greedy decoding |
| `auricle.streaming` | Chunk scheduler and incremental `StreamingASR` |
| `auricle.llm` | Pluggable text backends (echo, OpenAI-compatible, HuggingFace) |
| `auricle.pipeline` | `transcribe`, `caption_audio`, `answer_question` |
| `auricle.eval` | WER/CER metrics, JSONL manifests, the evaluation runner |
| `auricle.cli` | The `auricle` command |

## Audio encoder

The encoder follows the shape of whisper's audio tower without its weights:

1. **Log-mel spectrogram.** 25 ms Hann windows, 10 ms hop, 80 mel bands,
   log-compressed and normalised to roughly `[0, 1]`.
2. **Convolutional frontend.** Two 1-D convolutions with GELU; the second has
   stride 2, halving the frame rate.
3. **Transformer.** Sinusoidal positions plus a stack of pre-norm blocks
   (self-attention + feed-forward), finished by a layer norm.
4. **CTC head.** A linear projection to the vocabulary, decoded greedily.

Everything is sized by an `EncoderConfig`; the `tiny()` preset keeps the whole
model small enough to run in milliseconds on a laptop CPU.

## Streaming

`StreamScheduler` turns arbitrary pushes of samples into fixed-size,
overlapping chunks. Each chunk is decoded independently and the hypothesis is
stitched onto the committed transcript by `merge_transcripts`, which dedupes
the overlapping words. Because chunk boundaries come from cumulative samples,
the transcript does not depend on how the caller slices their pushes.

## LLM backends

Captioning and question answering are deliberately thin: the encoder produces
a transcript and a few acoustic statistics, a prompt template combines them,
and any `LLMBackend` produces the final text. Backends register by name and
are looked up through `get_backend`, which is how the CLI's `--backend` flag
works. The bundled `echo` backend returns the prompt unchanged, which keeps
tests deterministic and offline.

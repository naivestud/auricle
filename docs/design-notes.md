# Design notes

A few of the choices in auricle, and the reasoning behind them.

## A whisper-style encoder, from scratch

The encoder mirrors the shape of whisper's audio tower (log-mel → two convs →
transformer) but is implemented from scratch and randomly initialised. The goal
is a clean, readable reference for the *architecture* and the surrounding
tooling (streaming, captioning, QA, evaluation), not a pretrained model.
Because every layer is ordinary PyTorch, you can swap in trained weights with
`load_checkpoint` and the rest of the stack keeps working.

## Character-level CTC for the ASR head

A character vocabulary with a CTC blank keeps the decoder trivially simple —
greedy collapse, no language model, no beam. That makes the streaming path
easy to reason about and test. The trade-off is accuracy: character CTC is a
weak baseline. Beam search over the CTC lattice is on the roadmap.

## Overlap-and-merge streaming

Each chunk is decoded independently with a shared overlap, then merged by
matching the tail of the committed transcript against the head of the new
hypothesis. This is simpler than carrying decoder state across chunks and is
robust to a chunk landing mid-word. The merge scans only the last `window`
words, which is cheap; for very long streams that scan is the first thing to
optimise.

## LLM backends behind a protocol

Captioning and QA are prompt templates plus a text generator. Keeping the
generator behind a small `LLMBackend` protocol means the same code runs
against a local model, an OpenAI-compatible server, or a deterministic echo
for tests. New backends are one class and a `@register_backend` decorator.

## Deterministic tests with an untrained model

Most tests seed a tiny random model and assert *properties* — shapes,
determinism, transcript growth, metric identities — rather than accuracy.
That keeps the suite hermetic and fast (a couple of seconds) while still
exercising the full forward path.

## stdlib-only audio I/O

WAV reading and writing use the standard library plus numpy, so the eval and
data tooling run on machines without any ML dependencies. Only the model path
needs torch.

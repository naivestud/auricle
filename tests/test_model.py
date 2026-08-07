import torch

from auricle.model import AuricleModel


def test_tiny_model_constructs():
    model = AuricleModel.tiny()
    assert model.config.d_model == 64
    assert len(model.vocab) == 29


def test_logits_shape():
    torch.manual_seed(0)
    model = AuricleModel.tiny().eval()
    waveform = torch.randn(16_000)  # 100 mel frames -> 50 feature frames
    with torch.no_grad():
        logits = model(waveform)
    assert logits.shape == (1, 50, len(model.vocab))


def test_odd_frame_count_is_trimmed():
    # 16_160 samples -> 101 mel frames; the odd trailing frame is dropped.
    torch.manual_seed(0)
    model = AuricleModel.tiny().eval()
    waveform = torch.randn(16_160)
    with torch.no_grad():
        logits = model(waveform)
    assert logits.shape == (1, 50, len(model.vocab))


def test_transcribe_returns_strings():
    torch.manual_seed(2)
    model = AuricleModel.tiny()
    texts = model.transcribe(torch.randn(8_000))
    assert len(texts) == 1
    assert isinstance(texts[0], str)
    for ch in texts[0]:
        assert ch in model.vocab.chars

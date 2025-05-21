import torch

from auricle.encoder.config import EncoderConfig
from auricle.model import AuricleModel


def test_tiny_model_constructs():
    model = AuricleModel.tiny()
    assert model.config.d_model == 64
    assert len(model.vocab) == 29


def test_count_parameters_positive():
    model = AuricleModel.tiny()
    n = model.count_parameters()
    assert n > 0
    assert model.count_parameters(trainable_only=False) == n


def test_count_parameters_ignores_frozen():
    model = AuricleModel.tiny()
    for p in model.head.parameters():
        p.requires_grad = False
    assert model.count_parameters(trainable_only=True) < model.count_parameters(
        trainable_only=False
    )


def test_summary_contains_key_facts():
    model = AuricleModel.tiny()
    text = model.summary()
    assert "d_model=64" in text
    assert "vocab=29" in text
    assert "params=" in text


def test_larger_config_has_more_parameters():
    small = AuricleModel(EncoderConfig.small())
    tiny = AuricleModel.tiny()
    assert small.count_parameters() > tiny.count_parameters()


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

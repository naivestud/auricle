import torch
from torch import nn

from auricle.encoder.config import EncoderConfig
from auricle.model import AuricleModel


def _ctc_loss(model: AuricleModel, waveform: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = model(waveform)  # (batch, time, vocab)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (time, batch, vocab)
    time = log_probs.shape[0]
    ctc = nn.CTCLoss(blank=model.vocab.BLANK, zero_infinity=True)
    return ctc(
        log_probs,
        target.unsqueeze(0),
        torch.tensor([time]),
        torch.tensor([target.shape[0]]),
    )


def test_ctc_loss_decreases_with_training():
    """The whole stack must be differentiable and able to overfit one clip."""
    torch.manual_seed(0)
    config = EncoderConfig(d_model=32, n_layers=1, n_heads=2, max_frames=200)
    model = AuricleModel(config)
    model.train()

    waveform = torch.randn(1, 8_000)
    target = torch.tensor(model.vocab.encode("hi"), dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    first = _ctc_loss(model, waveform, target).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = _ctc_loss(model, waveform, target)
        loss.backward()
        optimizer.step()
    last = _ctc_loss(model, waveform, target).item()

    assert last < first


def test_gradients_reach_the_frontend():
    torch.manual_seed(1)
    model = AuricleModel.tiny()
    model.train()
    waveform = torch.randn(1, 4_000)
    target = torch.tensor(model.vocab.encode("a"), dtype=torch.long)

    loss = _ctc_loss(model, waveform, target)
    loss.backward()

    assert model.encoder.frontend.conv1.weight.grad is not None
    assert model.head.proj.weight.grad is not None

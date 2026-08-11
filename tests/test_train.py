import numpy as np
import torch
from torch import nn

from eoml.train import predict


class IdentityLogits(nn.Module):
    """Treats each input row as raw logits, shape (N, num_classes)."""

    def forward(self, x):
        return x


def make_loader(batches):
    """A dataloader here is just an iterable of {"image", "label"} dicts."""
    return [
        {
            "image": torch.tensor(logits, dtype=torch.float32),
            "label": torch.tensor(labels),
        }
        for logits, labels in batches
    ]


# Two batches, five samples, logits chosen so MSP is known in closed form.
BATCHES = [
    ([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # argmax 0, MSP e / (e + 9)
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # argmax 0, MSP 0.1 (uniform)
     [0, 5]),
    ([[0, 0, 3, 0, 0, 0, 0, 0, 0, 0],   # argmax 2, MSP e**3 / (e**3 + 9)
      [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # argmax 0, MSP e**4 / (e**4 + 9)
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # argmax 0, MSP 0.1
     [2, 0, 7]),
]
EXP_LABELS = [0, 5, 2, 0, 7]
EXP_PREDS = [0, 0, 2, 0, 0]
EXP_CONF = [
    np.e / (np.e + 9),
    0.1,
    np.e**3 / (np.e**3 + 9),
    np.e**4 / (np.e**4 + 9),
    0.1,
]


def test_predict_default_returns_labels_and_preds():
    result = predict(IdentityLogits(), make_loader(BATCHES), device="cpu")
    assert len(result) == 2  # no confidence unless requested
    labels, preds = result
    assert labels.tolist() == EXP_LABELS
    assert preds.tolist() == EXP_PREDS


def test_predict_returns_confidence_when_requested():
    labels, preds, conf = predict(
        IdentityLogits(), make_loader(BATCHES), device="cpu", return_confidence=True
    )
    assert labels.tolist() == EXP_LABELS
    assert preds.tolist() == EXP_PREDS
    assert conf.shape == (len(EXP_LABELS),)
    assert np.allclose(conf, EXP_CONF)
    assert ((conf > 0) & (conf <= 1)).all()


def test_predict_preds_identical_across_flag():
    _, preds_a = predict(IdentityLogits(), make_loader(BATCHES), device="cpu")
    _, preds_b, _ = predict(
        IdentityLogits(), make_loader(BATCHES), device="cpu", return_confidence=True
    )
    assert np.array_equal(preds_a, preds_b)

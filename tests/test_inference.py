import numpy as np
import torch
import xarray as xr
from torch import nn

from eoml.data.eurosat import EUROSAT_BANDS
from eoml.data.scenes import L2A_BANDS
from eoml.inference import classify_scene, to_model_input


class ConstantModel(nn.Module):
    """Predicts class 2 for every chip."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 10)
        logits[:, 2] = 1.0
        return logits


def make_scene(height, width):
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 10000, size=(len(L2A_BANDS), height, width)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": list(L2A_BANDS), "y": np.arange(height), "x": np.arange(width)},
    )


def test_to_model_input_zero_fills_missing_bands():
    scene = make_scene(64, 64)
    stack = to_model_input(scene)
    assert stack.shape == (len(EUROSAT_BANDS), 64, 64)
    b10_index = EUROSAT_BANDS.index("B10")
    assert torch.all(stack[b10_index] == 0)
    assert torch.any(stack[0] != 0)


def test_classify_scene_shape_and_values():
    scene = make_scene(130, 200)  # 2 x 3 full chips, with edge remainders
    class_map = classify_scene(ConstantModel(), scene)
    assert class_map.shape == (2, 3)
    assert (class_map.values == 2).all()


def test_classify_scene_rejects_tiny_scene():
    scene = make_scene(32, 32)
    try:
        classify_scene(ConstantModel(), scene)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

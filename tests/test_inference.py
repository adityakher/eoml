import numpy as np
import torch
import xarray as xr
from torch import nn

from eoml.data.eurosat import EUROSAT_BANDS
from eoml.data.scenes import L2A_BANDS
from eoml.inference import chip_stats, classify_scene, to_model_input


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


def test_classify_scene_default_returns_single_array():
    result = classify_scene(ConstantModel(), make_scene(128, 128))
    assert isinstance(result, xr.DataArray)  # not a tuple


def test_classify_scene_returns_confidence_when_requested():
    scene = make_scene(130, 200)  # 2 x 3 chips
    class_map, confidence = classify_scene(ConstantModel(), scene, return_confidence=True)
    assert class_map.shape == (2, 3)
    assert confidence.shape == (2, 3)
    assert (class_map.values == 2).all()
    # ConstantModel logits are 1 for class 2, 0 for the other nine classes,
    # so max softmax probability is e / (e + 9) for every chip.
    expected = np.e / (np.e + 9)
    assert np.allclose(confidence.values, expected)
    assert ((confidence.values > 0) & (confidence.values <= 1)).all()


def test_confidence_shares_grid_with_class_map():
    scene = make_scene(128, 128)
    class_map, confidence = classify_scene(ConstantModel(), scene, return_confidence=True)
    assert confidence.dims == class_map.dims
    assert np.array_equal(confidence.y.values, class_map.y.values)
    assert np.array_equal(confidence.x.values, class_map.x.values)
    assert confidence.rio.crs == class_map.rio.crs


def make_class_map(n_rows, n_cols):
    """A bare class map with integer coords, standing in for classify_scene output."""
    return xr.DataArray(
        np.zeros((n_rows, n_cols), dtype=np.int64),
        dims=("y", "x"),
        coords={"y": np.arange(n_rows), "x": np.arange(n_cols)},
    )


def make_field(values):
    values = np.asarray(values, dtype=np.float32)
    h, w = values.shape
    return xr.DataArray(
        values, dims=("y", "x"), coords={"y": np.arange(h), "x": np.arange(w)}
    )


def test_chip_stats_constant_field():
    mean, std = chip_stats(make_field(np.full((8, 12), 5.0)), make_class_map(2, 3), chip_size=4)
    assert mean.shape == (2, 3)
    assert np.allclose(mean.values, 5.0)
    assert np.allclose(std.values, 0.0)


def test_chip_stats_per_chip_mean():
    cs, n_rows, n_cols = 2, 2, 2
    field = np.zeros((n_rows * cs, n_cols * cs), dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            field[i * cs : (i + 1) * cs, j * cs : (j + 1) * cs] = i * n_cols + j
    mean, std = chip_stats(make_field(field), make_class_map(n_rows, n_cols), chip_size=cs)
    assert mean.values.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert np.allclose(std.values, 0.0)


def test_chip_stats_within_chip_std():
    # one chip, values {0, 0, 1, 1}: mean 0.5, population std 0.5
    field = make_field(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    mean, std = chip_stats(field, make_class_map(1, 1), chip_size=2)
    assert np.isclose(mean.values[0, 0], 0.5)
    assert np.isclose(std.values[0, 0], 0.5)


def test_chip_stats_matches_classify_scene_tiling():
    # The load-bearing claim: chip_stats' numpy tiling reproduces
    # classify_scene's torch.unfold tiling, chip for chip.
    rng = np.random.default_rng(1)
    cs = 8
    height, width = cs * 2 + 3, cs * 3 + 5  # remainders on both axes
    field_vals = rng.normal(size=(height, width)).astype(np.float32)

    # classify_scene's exact chip-extraction path, applied to the single field
    t = torch.from_numpy(field_vals)[None]  # (1, H, W) -> treat field as one "band"
    n_rows, n_cols = height // cs, width // cs
    chips = (
        t.unfold(1, cs, cs)
        .unfold(2, cs, cs)
        .permute(1, 2, 0, 3, 4)
        .reshape(-1, 1, cs, cs)
    )
    unfold_mean = chips.mean(dim=(1, 2, 3)).reshape(n_rows, n_cols).numpy()

    mean, _ = chip_stats(make_field(field_vals), make_class_map(n_rows, n_cols), chip_size=cs)
    assert mean.shape == (n_rows, n_cols)
    assert np.allclose(mean.values, unfold_mean, atol=1e-5)


def test_chip_stats_aligns_with_real_class_map():
    # End to end: a class_map straight from classify_scene aligns with chip_stats.
    scene = make_scene(130, 200)
    class_map = classify_scene(ConstantModel(), scene)
    field = make_field(np.arange(130 * 200, dtype=np.float32).reshape(130, 200))
    field = field.assign_coords(y=scene.y.values, x=scene.x.values)
    mean, std = chip_stats(field, class_map)
    assert mean.shape == class_map.shape
    assert np.array_equal(mean.y.values, class_map.y.values)
    assert np.array_equal(mean.x.values, class_map.x.values)


def test_chip_stats_preserves_crs():
    scene = make_scene(128, 128)
    class_map = classify_scene(ConstantModel(), scene)
    field = make_field(np.zeros((128, 128), dtype=np.float32)).assign_coords(
        y=scene.y.values, x=scene.x.values
    )
    mean, _ = chip_stats(field, class_map)
    assert mean.rio.crs == class_map.rio.crs


def test_chip_stats_rejects_mismatched_grids():
    field = make_field(np.zeros((8, 8), dtype=np.float32))  # tiles to 2x2 at cs=4
    try:
        chip_stats(field, make_class_map(2, 3), chip_size=4)  # class_map claims 3 cols
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_chip_stats_rejects_non_2d_field():
    field = xr.DataArray(np.zeros((3, 8, 12), dtype=np.float32), dims=("band", "y", "x"))
    try:
        chip_stats(field, make_class_map(2, 3), chip_size=4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

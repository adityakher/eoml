"""Run the land cover classifier over arbitrary Sentinel-2 scenes.

This is the bridge between the EuroSAT-trained model (13-band L1C chips,
64x64 px) and scenes loaded from the Planetary Computer (12-band L2A
rasters of arbitrary size): order the bands, zero-fill B10, tile into
chips, classify, and reassemble a georeferenced class map.
"""

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import torch
import xarray as xr

from eoml.data.eurosat import EUROSAT_BANDS, get_preprocess

CHIP_SIZE = 64


def to_model_input(scene: xr.DataArray) -> torch.Tensor:
    """Reorder scene bands to EUROSAT_BANDS, zero-filling missing ones.

    L2A scenes lack B10 (cirrus), which the pretrained model expects as
    channel 9; a zero fill is a reasonable stand-in since B10 is near zero
    over clear land.
    """
    available = set(scene.band.values.tolist())
    height, width = scene.shape[-2:]
    stack = np.zeros((len(EUROSAT_BANDS), height, width), dtype=np.float32)
    for i, band in enumerate(EUROSAT_BANDS):
        if band in available:
            stack[i] = scene.sel(band=band).values
    return torch.from_numpy(stack)


def _chip_map(grid, y, x, crs, name):
    """Wrap a (n_rows, n_cols) chip grid as a georeferenced DataArray."""
    da = xr.DataArray(grid, coords={"y": y, "x": x}, dims=("y", "x"), name=name)
    if crs is not None:
        da.rio.write_crs(crs, inplace=True)
    return da


@torch.no_grad()
def classify_scene(
    model,
    scene: xr.DataArray,
    device: str = "cpu",
    chip_size: int = CHIP_SIZE,
    batch_size: int = 64,
    preprocess=None,
    return_confidence: bool = False,
) -> "xr.DataArray | tuple[xr.DataArray, xr.DataArray]":
    """Tile a scene into chips, classify each, and return a class map.

    Edge remainders smaller than chip_size are dropped. The result is a
    (rows, cols) DataArray of class indices whose coordinates are the chip
    centers on the scene's grid, carrying the scene's CRS.

    By default returns the class map alone. With return_confidence=True,
    returns (class_map, confidence), where confidence is the maximum softmax
    probability of the predicted class for each chip. That is the standard
    baseline signal for out-of-distribution inputs -- a chip unlike anything
    in EuroSAT (open ocean, cloud, bare rock) tends to score lower -- and the
    raw material for calibration analysis. Both maps share the chip grid,
    coordinates, and CRS.
    """
    if preprocess is None:
        preprocess = get_preprocess()

    data = to_model_input(scene)  # (13, H, W)
    _, height, width = data.shape
    n_rows, n_cols = height // chip_size, width // chip_size
    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"Scene ({height}x{width} px) is smaller than chip size {chip_size}; "
            "use a larger bbox."
        )

    # (n_rows * n_cols, bands, chip_size, chip_size)
    chips = (
        data.unfold(1, chip_size, chip_size)
        .unfold(2, chip_size, chip_size)
        .permute(1, 2, 0, 3, 4)
        .reshape(-1, data.shape[0], chip_size, chip_size)
    )

    model.eval()
    preds, confs = [], []
    for start in range(0, len(chips), batch_size):
        batch = preprocess(chips[start : start + batch_size].to(device))
        probs = model(batch).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    class_grid = torch.cat(preds).reshape(n_rows, n_cols).numpy()

    # Coordinates of chip centers on the original grid
    y = scene.y.values[: n_rows * chip_size].reshape(n_rows, chip_size).mean(axis=1)
    x = scene.x.values[: n_cols * chip_size].reshape(n_cols, chip_size).mean(axis=1)
    crs = scene.rio.crs
    class_map = _chip_map(class_grid, y, x, crs, name="class")
    if not return_confidence:
        return class_map

    conf_grid = torch.cat(confs).reshape(n_rows, n_cols).numpy()
    confidence = _chip_map(conf_grid, y, x, crs, name="confidence")
    return class_map, confidence


def _chip_grid(field: np.ndarray, n_rows: int, n_cols: int, chip_size: int) -> np.ndarray:
    """Reshape a pixel field into (n_rows, chip_size, n_cols, chip_size).

    Chip membership matches classify_scene's tiling exactly: non-overlapping
    chip_size x chip_size blocks in row-major order, edge remainders smaller
    than chip_size dropped. Reducing over axes (1, 3) collapses each chip in
    the same order as the class map. (classify_scene tiles with torch.unfold;
    this reproduces that ordering in numpy -- verified equivalent in the tests.)
    """
    cropped = field[: n_rows * chip_size, : n_cols * chip_size]
    return cropped.reshape(n_rows, chip_size, n_cols, chip_size)


def chip_stats(
    field: xr.DataArray,
    class_map: xr.DataArray,
    chip_size: int = CHIP_SIZE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Per-chip mean and standard deviation of a pixel field on the class grid.

    Aggregates a full-resolution field (e.g. NDVI) onto the coarse chip grid
    produced by classify_scene, so a learned per-chip label and a physical
    per-pixel index can be compared chip-for-chip instead of as scene-wide
    fractions. The mean answers "how vegetated is this chip"; the within-chip
    std answers "how spectrally mixed is it" -- the latter flags chips that
    straddle several land-cover parcels, which the classifier must still
    collapse to a single label.

    field must be a 2D (y, x) array at the same pixel resolution as the scene
    that produced class_map; class_map is the (n_rows, n_cols) output of
    classify_scene. Both returned arrays share class_map's coords and CRS, so
    they align with it directly (e.g. ``mean.where(class_map == idx)``).

    Standard deviation is population (ddof=0).
    """
    if field.ndim != 2:
        raise ValueError(f"field must be 2D (y, x); got shape {tuple(field.shape)}")

    n_rows, n_cols = class_map.shape
    fh, fw = field.shape[-2:]
    if fh // chip_size != n_rows or fw // chip_size != n_cols:
        raise ValueError(
            f"field grid ({fh}x{fw}) does not tile to class_map "
            f"({n_rows}x{n_cols}) at chip size {chip_size}; field and "
            "class_map must derive from the same scene."
        )

    grid = _chip_grid(np.asarray(field.values), n_rows, n_cols, chip_size)
    mean = class_map.copy(data=grid.mean(axis=(1, 3))).rename("chip_mean")
    std = class_map.copy(data=grid.std(axis=(1, 3))).rename("chip_std")
    return mean, std

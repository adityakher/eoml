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


@torch.no_grad()
def classify_scene(
    model,
    scene: xr.DataArray,
    device: str = "cpu",
    chip_size: int = CHIP_SIZE,
    batch_size: int = 64,
    preprocess=None,
) -> xr.DataArray:
    """Tile a scene into chips, classify each, and return a class map.

    Edge remainders smaller than chip_size are dropped. The result is a
    (rows, cols) DataArray of class indices whose coordinates are the chip
    centers on the scene's grid, carrying the scene's CRS.
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
    preds = []
    for start in range(0, len(chips), batch_size):
        batch = preprocess(chips[start : start + batch_size].to(device))
        preds.append(model(batch).argmax(1).cpu())
    class_grid = torch.cat(preds).reshape(n_rows, n_cols).numpy()

    # Coordinates of chip centers on the original grid
    y = scene.y.values[: n_rows * chip_size].reshape(n_rows, chip_size).mean(axis=1)
    x = scene.x.values[: n_cols * chip_size].reshape(n_cols, chip_size).mean(axis=1)
    class_map = xr.DataArray(class_grid, coords={"y": y, "x": x}, dims=("y", "x"))
    if scene.rio.crs is not None:
        class_map.rio.write_crs(scene.rio.crs, inplace=True)
    return class_map

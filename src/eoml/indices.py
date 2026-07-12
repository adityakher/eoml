"""Spectral indices computed from Sentinel-2 scenes."""

import numpy as np
import xarray as xr

# Thresholds between vegetation classes; values below the first bound are
# class 0, values >= the last bound are the final class.
NDVI_CLASS_BOUNDS = (0.0, 0.2, 0.4, 0.6)
NDVI_CLASS_LABELS = ("Water/Non-veg", "Bare/Urban", "Sparse", "Moderate", "Dense/Healthy")
NDVI_CLASS_COLORS = ("#2166ac", "#d6b56a", "#c2e699", "#4daf4a", "#006837")


def ndvi(scene: xr.DataArray) -> xr.DataArray:
    """NDVI from a scene containing B04 (red) and B08 (NIR)."""
    red = scene.sel(band="B04").astype(float)
    nir = scene.sel(band="B08").astype(float)
    result = (nir - red) / (nir + red)
    # Zero out division-by-zero artifacts (e.g. nodata areas)
    return result.where(np.isfinite(result), 0)


def classify_ndvi(ndvi_da: xr.DataArray, bounds=NDVI_CLASS_BOUNDS) -> xr.DataArray:
    """Threshold NDVI into vegetation classes 0..len(bounds)."""
    values = ndvi_da.values
    classes = np.zeros(values.shape, dtype=np.uint8)
    for i, bound in enumerate(bounds):
        classes[values >= bound] = i + 1
    return ndvi_da.copy(data=classes)

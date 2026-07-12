import numpy as np
import xarray as xr

from eoml.indices import classify_ndvi, ndvi


def make_scene(red, nir):
    data = np.stack([red, nir]).astype(float)
    return xr.DataArray(data, dims=("band", "y", "x"), coords={"band": ["B04", "B08"]})


def test_ndvi_values():
    red = np.array([[1000.0, 500.0]])
    nir = np.array([[3000.0, 500.0]])
    result = ndvi(make_scene(red, nir))
    assert np.allclose(result.values, [[0.5, 0.0]])


def test_ndvi_handles_zero_denominator():
    scene = make_scene(np.zeros((2, 2)), np.zeros((2, 2)))
    assert np.allclose(ndvi(scene).values, 0)


def test_classify_ndvi_thresholds():
    values = np.array([[-0.5, 0.1, 0.3, 0.5, 0.9]])
    da = xr.DataArray(values, dims=("y", "x"))
    classes = classify_ndvi(da)
    assert classes.values.tolist() == [[0, 1, 2, 3, 4]]

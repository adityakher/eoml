from types import SimpleNamespace

import numpy as np
import xarray as xr

from eoml.data.scenes import bbox_coverage, harmonize_to_eurosat_range, rank_scenes

BBOX = (0.0, 0.0, 1.0, 1.0)


def make_item(west, south, east, north, cloud=0.0, baseline=None):
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south],
        ]],
    }
    properties = {"eo:cloud_cover": cloud}
    if baseline is not None:
        properties["s2:processing_baseline"] = baseline
    return SimpleNamespace(geometry=geometry, properties=properties)


def test_bbox_coverage():
    assert bbox_coverage(make_item(-1, -1, 2, 2), BBOX) == 1.0
    assert abs(bbox_coverage(make_item(0, 0, 0.5, 1), BBOX) - 0.5) < 1e-9


def test_rank_scenes_prefers_coverage_over_cloud():
    sliver = make_item(0, 0, 0.1, 1, cloud=0.0)
    full_cloudy = make_item(-1, -1, 2, 2, cloud=15.0)
    assert rank_scenes([sliver, full_cloudy], BBOX)[0] is full_cloudy


def test_rank_scenes_breaks_coverage_ties_by_cloud():
    full_cloudy = make_item(-1, -1, 2, 2, cloud=15.0)
    full_clear = make_item(-2, -2, 3, 3, cloud=1.0)
    assert rank_scenes([full_cloudy, full_clear], BBOX)[0] is full_clear


def make_band(values):
    return xr.DataArray(np.array(values, dtype=np.uint16), dims=("y",))


def test_harmonize_subtracts_offset_for_new_baseline():
    da = harmonize_to_eurosat_range(make_band([0, 500, 1000, 1644, 3000]), make_item(0, 0, 1, 1, baseline="05.10"))
    # nodata (0) and sub-offset values clip to 0 instead of wrapping around
    assert da.values.tolist() == [0, 0, 0, 644, 2000]


def test_harmonize_noop_for_old_baseline():
    da = harmonize_to_eurosat_range(make_band([0, 1644]), make_item(0, 0, 1, 1, baseline="03.01"))
    assert da.values.tolist() == [0, 1644]


def test_harmonize_noop_when_baseline_missing():
    da = harmonize_to_eurosat_range(make_band([1644]), make_item(0, 0, 1, 1))
    assert da.values.tolist() == [1644]

"""Sentinel-2 scene search and loading via the Planetary Computer STAC API."""

import numpy as np
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
from shapely.geometry import box, shape

from eoml.data.eurosat import EUROSAT_BANDS

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# L2A products do not include B10 (cirrus); it is consumed by atmospheric
# correction. Inference zero-fills that channel (see eoml.inference).
L2A_BANDS = tuple(b for b in EUROSAT_BANDS if b != "B10")

# Radiometric offset added to L2A DN values in processing baseline >= 04.00
# (scenes processed after Jan 2022). EuroSAT-era data predates it.
BOA_OFFSET = 1000


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def bbox_coverage(item, bbox) -> float:
    """Fraction of bbox covered by the item's footprint (0..1)."""
    aoi = box(*bbox)
    return shape(item.geometry).intersection(aoi).area / aoi.area


def rank_scenes(items, bbox) -> list:
    """Best scene first: fullest bbox coverage, then lowest cloud cover.

    STAC returns any intersecting granule; at swath edges a scene may cover
    only a sliver of the bbox, and the nodata fill would dominate analysis.
    Coverage is rounded so near-identical footprints tie and cloud cover
    decides.
    """
    return sorted(
        items,
        key=lambda item: (
            -round(bbox_coverage(item, bbox), 2),
            item.properties["eo:cloud_cover"],
        ),
    )


def search_scenes(bbox, datetime, max_cloud_cover: float = 20.0, catalog=None) -> list:
    """Return Sentinel-2 L2A items intersecting bbox, best first.

    Ranked by bbox coverage, then cloud cover (see rank_scenes).

    Args:
        bbox: (west, south, east, north) in WGS84 lon/lat.
        datetime: STAC datetime range, e.g. "2024-06-01/2024-06-30".
        max_cloud_cover: maximum scene cloud cover percentage.
    """
    catalog = catalog or open_catalog()
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    return rank_scenes(search.item_collection(), bbox)


def harmonize_to_eurosat_range(da: xr.DataArray, item) -> xr.DataArray:
    """Remove the +1000 BOA offset from baseline >= 04.00 scenes.

    Keeps DN values on the same scale as EuroSAT-era products, which the
    classifier was trained on, and keeps NDVI uncompressed. Nodata (DN 0)
    and negative reflectances both clip back to 0.
    """
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is None or float(baseline) < 4.0:
        return da
    # Cast before subtracting: the raw data is uint16 and would wrap around.
    return (da.astype(np.int32) - BOA_OFFSET).clip(min=0)


def load_band(item, band: str, bbox=None) -> xr.DataArray:
    """Load a single band asset, optionally clipped to a WGS84 bbox.

    Values are harmonized to the pre-offset (EuroSAT-era) DN range.
    """
    raster = rioxarray.open_rasterio(item.assets[band].href)
    # open_rasterio is typed DataArray | Dataset | list[Dataset]; a
    # single-band COG asset always yields a DataArray.
    assert isinstance(raster, xr.DataArray)
    da = raster.squeeze("band", drop=True)
    if bbox is not None:
        da = da.rio.clip_box(*bbox, crs="EPSG:4326")
    return harmonize_to_eurosat_range(da, item)


def load_scene(item, bands=L2A_BANDS, bbox=None, reference_band: str = "B04") -> xr.DataArray:
    """Load bands from a STAC item into a single (band, y, x) DataArray.

    All bands are resampled onto the reference band's grid (10 m for B04),
    so mixed 10/20/60 m bands stack cleanly. Both NDVI and classification
    consume this same representation.
    """
    reference = load_band(item, reference_band, bbox=bbox)
    layers = []
    for band in bands:
        if band == reference_band:
            da = reference
        else:
            da = load_band(item, band, bbox=bbox)
            if da.shape != reference.shape:
                da = da.rio.reproject_match(reference)
        layers.append(da.astype(np.float32))
    scene = xr.concat(layers, dim="band").assign_coords(band=list(bands))
    scene.rio.write_crs(reference.rio.crs, inplace=True)
    return scene

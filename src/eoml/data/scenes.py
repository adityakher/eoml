"""Sentinel-2 scene search and loading via the Planetary Computer STAC API."""

import numpy as np
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr

from eoml.data.eurosat import EUROSAT_BANDS

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# L2A products do not include B10 (cirrus); it is consumed by atmospheric
# correction. Inference zero-fills that channel (see eoml.inference).
L2A_BANDS = tuple(b for b in EUROSAT_BANDS if b != "B10")


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(bbox, datetime, max_cloud_cover: float = 20.0, catalog=None) -> list:
    """Return Sentinel-2 L2A items intersecting bbox, least cloudy first.

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
    items = list(search.item_collection())
    return sorted(items, key=lambda item: item.properties["eo:cloud_cover"])


def load_band(item, band: str, bbox=None) -> xr.DataArray:
    """Load a single band asset, optionally clipped to a WGS84 bbox."""
    da = rioxarray.open_rasterio(item.assets[band].href).squeeze("band", drop=True)
    if bbox is not None:
        da = da.rio.clip_box(*bbox, crs="EPSG:4326")
    return da


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

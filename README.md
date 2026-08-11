# eoml

Land cover classification and NDVI analysis on Sentinel-2 imagery.

Two pieces, one scene-acquisition path:

- **Land cover classifier** — a ResNet-18 (Sentinel-2 MoCo pretrained weights,
  via torchgeo) fine-tuned on EuroSAT's 10 land cover classes.
- **NDVI analysis** — vegetation health maps from B04/B08 of any Sentinel-2 L2A
  scene, retrieved from the Microsoft Planetary Computer STAC API.

The bridge between them ([src/eoml/inference.py](src/eoml/inference.py)) runs
the trained classifier over arbitrary scenes: bands are resampled to a common
10 m grid, ordered to match EuroSAT, tiled into 64x64 chips, classified, and
reassembled into a georeferenced class map.

## Layout

```
src/eoml/
├── data/
│   ├── eurosat.py    # EuroSAT datasets, dataloaders, transforms, band/class constants
│   └── scenes.py     # STAC search + scene loading (Planetary Computer, Sentinel-2 L2A)
├── models.py         # classifier construction, checkpoint save/load
├── train.py          # training/evaluation loops
├── inference.py      # scene → chips → georeferenced class map
├── indices.py        # NDVI + threshold classification
├── viz.py            # confusion matrix, Grad-CAM, NDVI and class-map plots
└── cli.py            # entry points
notebooks/            # executed walkthrough of the full pipeline
tests/                # unit tests (no network/GPU needed)
```

## Install

```sh
pip install -e .[dev]
```

## Usage

Train the classifier (downloads EuroSAT on first run, saves a checkpoint):

```sh
eoml-train --epochs 3 --checkpoint artifacts/eurosat_resnet18.pth
```

Classify land cover in an arbitrary Sentinel-2 scene:

```sh
eoml-classify --bbox -120.5 36.5 -120.0 37.0 --datetime 2024-06-01/2024-06-30 ^
    --checkpoint artifacts/eurosat_resnet18.pth --output artifacts/class_map.png
```

NDVI analysis of the same area:

```sh
eoml-ndvi --bbox -120.5 36.5 -120.0 37.0 --datetime 2024-06-01/2024-06-30 ^
    --output artifacts/ndvi_analysis.png
```

## Notes on the EuroSAT ↔ L2A gap

EuroSAT's training chips are L1C; served scenes are L2A. Most of the resulting
differences are corrected in code (below); the largest radiometric one is left
as a documented systematic.

- **B10**: EuroSAT chips come from L1C products (13 bands); Planetary Computer
  serves L2A, where B10 (cirrus) is consumed by atmospheric correction.
  Inference zero-fills that channel — B10 is near zero over clear land, so this
  is a mild domain shift rather than a hard break.
- **Resolution**: L2A bands arrive at 10/20/60 m; `load_scene()` resamples
  everything onto the 10 m grid of the reference band (B04).
- **Normalization**: training and inference share the same `Normalize(0, 10000)`
  preprocessing from `eoml.data.eurosat.get_preprocess()`.
- **Scene selection**: STAC returns any granule intersecting the bbox — at
  swath edges that can be a sliver of nodata-padded scene. `search_scenes()`
  ranks by bbox coverage first, cloud cover second.
- **Radiometric offset**: L2A scenes with processing baseline >= 04.00
  (post-Jan 2022) carry a +1000 DN offset that EuroSAT-era data lacks;
  `load_band()` subtracts it so the classifier and NDVI see pre-offset values.
- **TOA vs BOA reflectance (uncorrected)**: EuroSAT's L1C chips are
  top-of-atmosphere reflectance; L2A is bottom-of-atmosphere (surface)
  reflectance, with Sen2Cor having removed the atmospheric path radiance the
  training data still carries. This is the largest radiometric difference
  between the two: on the order of hundreds of DN in the blue and visible bands
  (against the B10 zero-fill's ~0.001 normalized perturbation), tapering toward
  the NIR/SWIR, and it leaves served scenes darker than the model's training
  inputs, most so in the blue. It is **not** corrected here; an on-footprint fix
  would need L1C for the same scene from a second catalog (Element84 earth-search
  or the Copernicus Data Space). Left as a known systematic, it is a likely
  contributor to the confidence shift the walkthrough shows on real scenes.
  (NDVI is computed directly on L2A surface reflectance, so it is unaffected by
  this train/serve mismatch.)

EuroSAT downloads to `~/.cache/eoml/eurosat` by default (override with
`--root` or `get_datasets(root=...)`).

## Tests

```sh
pytest
```

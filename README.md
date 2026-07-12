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
notebooks/            # original exploratory notebooks
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

- **B10**: EuroSAT chips come from L1C products (13 bands); Planetary Computer
  serves L2A, where B10 (cirrus) is consumed by atmospheric correction.
  Inference zero-fills that channel — B10 is near zero over clear land, so this
  is a mild domain shift rather than a hard break.
- **Resolution**: L2A bands arrive at 10/20/60 m; `load_scene()` resamples
  everything onto the 10 m grid of the reference band (B04).
- **Normalization**: training and inference share the same `Normalize(0, 10000)`
  preprocessing from `eoml.data.eurosat.get_preprocess()`.

## Tests

```sh
pytest
```

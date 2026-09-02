# eoml

This project develops a Python framework for applying a benchmark-trained land cover classifier to operational Sentinel-2 imagery and characterizing how far its accuracy transfers off the benchmark. The framework fine-tunes a ResNet-18 (Sentinel-2 MoCo-pretrained weights, via torchgeo) on the ten-class EuroSAT dataset, then runs the trained model over full Sentinel-2 Level-2A scenes retrieved from the Microsoft Planetary Computer: bands are harmonized to the EuroSAT radiometric and spectral convention, tiled into 64×64 chips, classified with per-chip confidence, and reassembled into a georeferenced land cover map. An independent NDVI computation on the same scene provides a physics-based second opinion, and each chip's maximum softmax probability provides an out-of-distribution signal. The model can be used to produce land cover, NDVI, and per-chip confidence maps over arbitrary scenes, and to compare the learned and physical views chip by chip. The project is motivated by the gap between benchmark accuracy and operational reliability that governs whether a curated-dataset classifier can be trusted on real sensor feeds.

[Full study writeup.](https://adityakher.com/eoml.html)


### Installation
```bash
pip install -e .[dev]
```


### Example Usage
The command-line interface trains the classifier and runs both analyses on any scene.

Train the classifier (downloads EuroSAT on first run to `~/.cache/eoml/eurosat`, override with `--root`; saves a checkpoint):

```bash
eoml-train --epochs 3 --checkpoint artifacts/eurosat_resnet18.pth
```

Classify land cover in an arbitrary Sentinel-2 scene. The `--confidence` flag hatches low-confidence chips on the class map and saves a confidence map alongside it:

```bash
eoml-classify --bbox -120.5 36.6 -120.35 36.7 --datetime 2024-06-01/2024-06-30 --checkpoint artifacts/eurosat_resnet18.pth --output artifacts/class_map.png --confidence
```

NDVI analysis of the same area:

```bash
eoml-ndvi --bbox -120.5 36.6 -120.35 36.7 --datetime 2024-06-01/2024-06-30 --output artifacts/ndvi_analysis.png
```


### Layout
```
src/eoml/
├── data/
│   ├── eurosat.py    # EuroSAT datasets, dataloaders, transforms, band/class constants
│   └── scenes.py     # STAC search + scene loading (Planetary Computer, Sentinel-2 L2A)
├── models.py         # classifier construction, checkpoint save/load
├── train.py          # training/evaluation loops (with optional per-sample confidence)
├── inference.py      # scene → chips → georeferenced class map; per-chip confidence; NDVI-on-chip-grid
├── indices.py        # NDVI + threshold classification
├── viz.py            # confusion matrix, Grad-CAM, NDVI, class-map, confidence, scatter plots
└── cli.py            # entry points (eoml-train, eoml-classify, eoml-ndvi)
notebooks/            # tutorial walkthrough of the full pipeline
tests/                # unit tests (no network/GPU needed)
```


### Tutorial
A [tutorial notebook](notebooks/tutorial.ipynb) is included in this repository and walks through training and evaluating the classifier, retrieving a Sentinel-2 scene, and running the classifier and NDVI on two contrasting scenes — an agricultural scene in California's Central Valley and the Mediterranean coast near Malibu — with per-chip confidence throughout.


### Citation
If you reference this work, please cite:

A. Kher, "Benchmark Land-Cover Classification on Operational Sentinel-2 Imagery," 2026.
https://adityakher.com/eoml.html


### References
1. P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019, doi: [10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242).
2. A. J. Stewart, C. Robinson, I. A. Corley, A. Ortiz, J. M. Lavista Ferres, and A. Banerjee, "TorchGeo: Deep Learning With Geospatial Data," in *Proc. 30th Int. Conf. on Advances in Geographic Information Systems (SIGSPATIAL '22)*, 2022, doi: [10.1145/3557915.3560953](https://doi.org/10.1145/3557915.3560953).
3. K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum Contrast for Unsupervised Visual Representation Learning," in *IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2020, arXiv: [1911.05722](https://arxiv.org/abs/1911.05722).
4. D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," in *Int. Conf. on Learning Representations (ICLR)*, 2017, arXiv: [1610.02136](https://arxiv.org/abs/1610.02136).


### Tests
```bash
pytest
```


### License
This project is licensed under the MIT License.

"""EuroSAT dataset access, dataloaders, and transforms."""

import os
import tempfile

import kornia.augmentation as K
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT

# Sentinel-2 band order used by EuroSAT (and expected by the pretrained
# SENTINEL2_ALL_MOCO weights). Note B10 (cirrus) is present: EuroSAT is
# derived from L1C products, which still include it.
EUROSAT_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
)

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation",
    "Highway", "Industrial", "Pasture",
    "Permanent Crop", "Residential", "River", "Sea & Lake",
]

NUM_CLASSES = len(CLASS_NAMES)


def default_root() -> str:
    return os.path.join(tempfile.gettempdir(), "pytorch")


def get_preprocess():
    """Normalization applied to raw Sentinel-2 DN values before the model.

    Shared by training and scene inference so both see identical inputs.
    """
    return K.Normalize(mean=0.0, std=10000.0)


def get_augment():
    return K.ImageSequential(K.RandomHorizontalFlip(), K.RandomVerticalFlip())


def get_datasets(root: str | None = None, download: bool = True) -> dict:
    root = root or default_root()
    return {
        split: EuroSAT(root, split=split, download=download)
        for split in ("train", "val", "test")
    }


def get_dataloaders(
    datasets: dict,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 4,
) -> dict:
    common = {"num_workers": num_workers, "pin_memory": True}
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, **common),
        "val": DataLoader(datasets["val"], batch_size=eval_batch_size, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=eval_batch_size, shuffle=False, **common),
    }

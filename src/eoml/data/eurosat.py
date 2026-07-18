"""EuroSAT dataset access, dataloaders, and transforms."""

import shutil
from pathlib import Path

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
    # Durable, unlike the system temp dir: Windows/Linux cleanup tools can
    # reap temp files while leaving the directory tree, which corrupts the
    # dataset cache (see _clear_stale_extraction).
    return str(Path.home() / ".cache" / "eoml" / "eurosat")


def _clear_stale_extraction(root: str) -> None:
    """Remove a partially deleted EuroSAT extraction so torchgeo re-downloads.

    torchgeo only checks that the extraction directory exists, not that it
    has contents; a cache where the images were deleted but the tree
    survived would otherwise fail with an opaque FileNotFoundError deep in
    torchvision. Only removes the tree when it holds zero images, so intact
    data is never touched.
    """
    extraction = Path(root, EuroSAT.base_dir)
    if extraction.is_dir() and not any(extraction.rglob("*.tif")):
        shutil.rmtree(extraction)


def get_preprocess():
    """Normalization applied to raw Sentinel-2 DN values before the model.

    Shared by training and scene inference so both see identical inputs.
    """
    return K.Normalize(mean=0.0, std=10000.0)


def get_augment():
    return K.ImageSequential(K.RandomHorizontalFlip(), K.RandomVerticalFlip())


def get_datasets(root: str | None = None, download: bool = True) -> dict:
    root = root or default_root()
    _clear_stale_extraction(root)
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

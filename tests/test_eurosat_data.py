from pathlib import Path

from torchgeo.datasets import EuroSAT

from eoml.data.eurosat import _clear_stale_extraction


def make_extraction(root: Path) -> Path:
    extraction = root / EuroSAT.base_dir
    (extraction / "AnnualCrop").mkdir(parents=True)
    (extraction / "Forest").mkdir()
    return extraction


def test_clears_extraction_with_no_images(tmp_path):
    extraction = make_extraction(tmp_path)
    _clear_stale_extraction(str(tmp_path))
    assert not extraction.exists()


def test_keeps_extraction_with_images(tmp_path):
    extraction = make_extraction(tmp_path)
    (extraction / "Forest" / "Forest_1.tif").write_bytes(b"not a real tif")
    _clear_stale_extraction(str(tmp_path))
    assert (extraction / "Forest" / "Forest_1.tif").exists()


def test_noop_when_extraction_missing(tmp_path):
    _clear_stale_extraction(str(tmp_path))  # must not raise

"""Command line entry points: eoml-train, eoml-classify, eoml-ndvi."""

import argparse
from pathlib import Path


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _add_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84 lon/lat",
    )
    parser.add_argument(
        "--datetime",
        required=True,
        help="STAC datetime range, e.g. 2024-06-01/2024-06-30",
    )
    parser.add_argument("--max-cloud-cover", type=float, default=20.0)


def _find_scene(args):
    from eoml.data import scenes

    items = scenes.search_scenes(args.bbox, args.datetime, args.max_cloud_cover)
    if not items:
        raise SystemExit("No scenes found matching the search criteria.")
    item = items[0]
    print(
        f"Found {len(items)} scenes; using {item.id} "
        f"(cloud cover {item.properties['eo:cloud_cover']}%)"
    )
    return item


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def train_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the EuroSAT land cover classifier.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--root", default=None, help="EuroSAT data root (default: ~/.cache/eoml/eurosat)")
    parser.add_argument("--checkpoint", default="artifacts/eurosat_resnet18.pth")
    parser.add_argument(
        "--confusion-matrix", default=None, help="Optional path to save a test-set confusion matrix PNG"
    )
    args = parser.parse_args(argv)

    from eoml import models, train
    from eoml.data import eurosat

    device = _device()
    print(f"Using device: {device}")

    datasets = eurosat.get_datasets(args.root)
    dataloaders = eurosat.get_dataloaders(datasets, batch_size=args.batch_size)
    model = models.create_classifier()
    preprocess = eurosat.get_preprocess().to(device)
    augment = eurosat.get_augment().to(device)

    train.fit(model, dataloaders, device, epochs=args.epochs, lr=args.lr,
              preprocess=preprocess, augment=augment)

    test_acc = train.evaluate(model, dataloaders["test"], device, preprocess)
    print(f"Test accuracy: {test_acc:.0%}")

    _ensure_parent(args.checkpoint)
    models.save_classifier(model, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    if args.confusion_matrix:
        from eoml import viz

        y_true, y_pred = train.predict(model, dataloaders["test"], device, preprocess)
        _ensure_parent(args.confusion_matrix)
        viz.plot_confusion_matrix(y_true, y_pred, save_path=args.confusion_matrix)
        print(f"Saved confusion matrix to {args.confusion_matrix}")


def classify_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Classify land cover in a Sentinel-2 scene.")
    _add_scene_args(parser)
    parser.add_argument("--checkpoint", default="artifacts/eurosat_resnet18.pth")
    parser.add_argument("--output", default="artifacts/class_map.png")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    from eoml import inference, models, viz
    from eoml.data import scenes

    device = _device()
    item = _find_scene(args)
    print("Loading scene bands (this may take a while)...")
    scene = scenes.load_scene(item, bbox=args.bbox)
    model = models.load_classifier(args.checkpoint, device)
    class_map = inference.classify_scene(model, scene, device, batch_size=args.batch_size)

    _ensure_parent(args.output)
    viz.plot_class_map(class_map, title=f"Land cover — {item.id}", save_path=args.output)
    print(f"Saved class map to {args.output}")


def ndvi_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NDVI analysis of a Sentinel-2 scene.")
    _add_scene_args(parser)
    parser.add_argument("--output", default="artifacts/ndvi_analysis.png")
    args = parser.parse_args(argv)

    from eoml import indices, viz
    from eoml.data import scenes

    item = _find_scene(args)
    print("Loading scene bands...")
    scene = scenes.load_scene(item, bands=("B04", "B08"), bbox=args.bbox)
    ndvi_da = indices.ndvi(scene)
    print(
        f"NDVI range: {float(ndvi_da.min()):.2f} to {float(ndvi_da.max()):.2f}, "
        f"mean {float(ndvi_da.mean()):.2f}"
    )
    veg_classes = indices.classify_ndvi(ndvi_da)

    _ensure_parent(args.output)
    viz.plot_ndvi_analysis(
        ndvi_da, veg_classes, title=f"Sentinel-2 NDVI Analysis\n{item.id}", save_path=args.output
    )
    print(f"Saved NDVI analysis to {args.output}")

"""Plotting helpers: confusion matrix, Grad-CAM, NDVI maps, class and confidence maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from eoml.data.eurosat import CLASS_NAMES
from eoml.indices import NDVI_CLASS_COLORS, NDVI_CLASS_LABELS


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    accuracy = (np.asarray(y_true) == np.asarray(y_pred)).mean()
    ax.set_title(f"EuroSAT Classification — {accuracy:.1%} accuracy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def chip_to_rgb(image, rgb_band_indices=(3, 2, 1)):
    """CHW multispectral chip (EuroSAT band order) -> HWC RGB in [0, 1]."""
    rgb = np.asarray(image)[list(rgb_band_indices)].transpose(1, 2, 0)
    return ((rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.float32)


def plot_gradcam_grid(
    model, images, labels, device="cpu", n=4, class_names=CLASS_NAMES, preprocess=None, save_path=None
):
    """Show n chips beside their Grad-CAM overlays from the last conv layer."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
    for i in range(n):
        input_tensor = images[i].unsqueeze(0).to(device)
        if preprocess is not None:
            input_tensor = preprocess(input_tensor)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        pred = model(input_tensor).argmax(dim=1).item()

        rgb = chip_to_rgb(images[i])
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"True: {class_names[labels[i]]}")
        axes[i][1].imshow(overlay)
        axes[i][1].set_title(f"Pred: {class_names[pred]}")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ndvi_analysis(ndvi_da, veg_classes, title="Sentinel-2 NDVI Analysis", save_path=None):
    """Two-panel figure: NDVI heatmap and thresholded vegetation classes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im = axes[0].imshow(ndvi_da.values, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0].set_title("Normalized Difference Vegetation Index")
    fig.colorbar(im, ax=axes[0], label="NDVI", shrink=0.8)

    n_classes = len(NDVI_CLASS_LABELS)
    cmap = ListedColormap(NDVI_CLASS_COLORS)
    im2 = axes[1].imshow(veg_classes.values, cmap=cmap, vmin=0, vmax=n_classes - 1)
    axes[1].set_title("Vegetation Classification from NDVI")
    cbar = fig.colorbar(im2, ax=axes[1], ticks=range(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels(NDVI_CLASS_LABELS)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


LOW_CONFIDENCE_THRESHOLD = 0.5


def plot_class_map(
    class_map,
    class_names=CLASS_NAMES,
    confidence=None,
    low_conf_threshold=LOW_CONFIDENCE_THRESHOLD,
    title="Land cover classification",
    save_path=None,
):
    """Render a classify_scene() result with a class legend.

    If a per-chip confidence map is passed, chips whose max softmax
    probability falls below low_conf_threshold are hatched, marking where the
    model is guessing (e.g. open ocean or cloud, which resemble no EuroSAT
    class). confidence must share class_map's grid.
    """
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10", n_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(class_map.values, cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels(class_names)

    if confidence is not None:
        if confidence.shape != class_map.shape:
            raise ValueError(
                f"confidence grid {tuple(confidence.shape)} does not match "
                f"class_map {tuple(class_map.shape)}."
            )
        low = np.asarray(confidence.values) < low_conf_threshold
        for i, j in zip(*np.where(low)):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="xxx", edgecolor="black", linewidth=0.3,
                )
            )
        if low.any():
            proxy = Patch(
                facecolor="none", hatch="xxx", edgecolor="black",
                label=f"MSP < {low_conf_threshold:g}",
            )
            ax.legend(handles=[proxy], loc="lower right", fontsize=8, framealpha=0.9)
        title = f"{title} (hatched: low confidence)"

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confidence_map(
    confidence, title="Prediction confidence (max softmax prob)", save_path=None
):
    """Heatmap of per-chip max softmax probability from classify_scene."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confidence.values, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="max softmax probability", shrink=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confidence_histogram(
    reference_conf,
    scene_conf,
    reference_label="EuroSAT test",
    scene_label="Scene",
    threshold=None,
    bins=30,
    save_path=None,
):
    """Overlaid confidence (MSP) histograms: benchmark vs. real scene.

    Accepts numpy arrays or DataArrays; both are flattened and non-finite
    values dropped. Densities (not counts) are plotted so the two are
    comparable despite very different sample sizes -- the EuroSAT test set has
    thousands of chips, a single scene a few hundred. The leftward shift and
    heavier low-confidence tail of the scene distribution is the out-of-
    distribution signal.
    """
    def flat(x):
        v = np.asarray(getattr(x, "values", x)).ravel()
        return v[np.isfinite(v)]

    ref, scn = flat(reference_conf), flat(scene_conf)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ref, bins=bins, range=(0, 1), density=True, alpha=0.6, label=reference_label)
    ax.hist(scn, bins=bins, range=(0, 1), density=True, alpha=0.6, label=scene_label)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"threshold {threshold:g}")
    ax.set_xlabel("max softmax probability")
    ax.set_ylabel("density")
    ax.set_title("Prediction confidence: benchmark vs. real scene")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ndvi_class_scatter(
    class_map,
    ndvi_mean,
    ndvi_std,
    class_names=CLASS_NAMES,
    title="Per-chip NDVI vs. within-chip spread",
    save_path=None,
):
    """Scatter of chip-mean NDVI vs. within-chip NDVI std, colored by class.

    ndvi_mean and ndvi_std are the per-chip arrays from inference.chip_stats;
    class_map is the matching classify_scene output. Points low and to the left
    are spectrally bare chips the classifier still labels as some class (keying
    on geometry, not greenness); points high on the y axis are spectrally mixed
    chips collapsed to a single label.
    """
    cls_v = np.asarray(class_map.values)
    mean_v = np.asarray(ndvi_mean.values)
    std_v = np.asarray(ndvi_std.values)
    cmap = plt.get_cmap("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in np.unique(cls_v):
        m = cls_v == idx
        ax.scatter(
            mean_v[m], std_v[m], s=18, alpha=0.7,
            color=cmap(int(idx)), label=class_names[int(idx)],
        )
    ax.set_xlabel("chip-mean NDVI")
    ax.set_ylabel("within-chip NDVI σ")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=1.3, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

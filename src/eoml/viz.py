"""Plotting helpers: confusion matrix, Grad-CAM, NDVI maps, class maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
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
    axes[0].set_title("NDVI — Vegetation Health")
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


def plot_class_map(class_map, class_names=CLASS_NAMES, title="Land cover classification", save_path=None):
    """Render a classify_scene() result with a class legend."""
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10", n_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(class_map.values, cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels(class_names)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

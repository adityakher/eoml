"""Model construction and checkpoint I/O for the land cover classifier."""

import torch
from torch import nn
from torchgeo.models import ResNet18_Weights, resnet18

from eoml.data.eurosat import EUROSAT_BANDS, NUM_CLASSES


def create_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with Sentinel-2 MoCo pretrained weights and a fresh head.

    The MoCo weights are self-supervised and ship without a classification
    head, so the default 1000-class fc layer is random anyway; replace it
    with one sized for EuroSAT.
    """
    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO if pretrained else None
    # in_chans must be explicit: torchgeo only infers 13 channels from the
    # weights metadata, and load_classifier() builds without weights.
    model = resnet18(weights, in_chans=len(EUROSAT_BANDS))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_classifier(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_classifier(path: str, device: str = "cpu") -> nn.Module:
    model = create_classifier(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

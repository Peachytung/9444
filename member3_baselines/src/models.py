from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16


class ResNet50FCN(nn.Module):
    """Standard ResNet50 encoder with a simple FCN-32s-style head."""

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.classifier(self.encoder(x))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class VGG16FCN(nn.Module):
    """Standard VGG16 encoder with a simple FCN-32s-style head."""

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = vgg16(weights=None).features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.classifier(self.encoder(x))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def build_model(name: str, num_classes: int, dropout: float = 0.2) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"resnet50", "resnet50_fcn"}:
        return ResNet50FCN(num_classes, dropout=dropout)
    if key in {"vgg16", "vgg16_fcn"}:
        return VGG16FCN(num_classes, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

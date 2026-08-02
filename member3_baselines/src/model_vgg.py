"""Compatibility entry point for the Member 3 VGG16 baseline model."""

from __future__ import annotations

try:
    from .models import VGG16FCN
except ImportError:
    from models import VGG16FCN


def build_vgg16_fcn(num_classes: int = 103, dropout: float = 0.2) -> VGG16FCN:
    return VGG16FCN(num_classes=num_classes, dropout=dropout)


__all__ = ["VGG16FCN", "build_vgg16_fcn"]

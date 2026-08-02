"""Compatibility entry point for the Member 3 ResNet50 baseline model."""

from __future__ import annotations

try:
    from .models import ResNet50FCN
except ImportError:
    from models import ResNet50FCN


def build_resnet50_fcn(num_classes: int = 103, dropout: float = 0.2) -> ResNet50FCN:
    return ResNet50FCN(num_classes=num_classes, dropout=dropout)


__all__ = ["ResNet50FCN", "build_resnet50_fcn"]

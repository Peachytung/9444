from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    categories: list[str]
    num_classes: int


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    candidates = [
        root,
        root / "data",
        root / "UECFOODPIX" / "data",
        root / "UECFoodPIX",
    ]
    for candidate in candidates:
        if (candidate / "category.txt").is_file() and (candidate / "UECFoodPIX").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate UECFoodPIX under {root}. Expected category.txt and UECFoodPIX/."
    )


def load_categories(data_root: str | Path) -> list[str]:
    root = resolve_data_root(data_root)
    rows = root.joinpath("category.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    categories = ["background"]
    for row in rows[1:]:
        fields = row.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        expected_id = len(categories)
        category_id = int(fields[0])
        if category_id != expected_id:
            raise ValueError(f"Unexpected category id {category_id}; expected {expected_id}")
        categories.append(fields[1])
    if len(categories) != 103:
        raise ValueError(f"Expected 103 labels including background, found {len(categories)}")
    return categories


def read_ids(path: str | Path) -> list[str]:
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate image ids in {path}")
    return values


class UECFoodPixSegmentation(Dataset):
    """UECFoodPIX semantic-segmentation dataset.

    The archive stores the class id in the first channel of each RGB/RGBA PNG
    mask; remaining colour channels are zero (alpha, when present, is 255).
    """

    def __init__(
        self,
        data_root: str | Path,
        ids_file: str | Path,
        source_folder: str,
        image_size: int | Sequence[int] = 224,
        augment: bool = False,
    ) -> None:
        self.root = resolve_data_root(data_root)
        self.ids = read_ids(ids_file)
        self.source_folder = source_folder
        if source_folder not in {"train", "test"}:
            raise ValueError("source_folder must be 'train' or 'test'")
        self.image_dir = self.root / "UECFoodPIX" / source_folder / "img"
        self.mask_dir = self.root / "UECFoodPIX" / source_folder / "mask"
        if isinstance(image_size, int):
            self.image_width = int(image_size)
            self.image_height = int(image_size)
        else:
            if len(image_size) != 2:
                raise ValueError("image_size must be an int or a (width, height) pair")
            self.image_width = int(image_size[0])
            self.image_height = int(image_size[1])
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")
        self.augment = bool(augment)
        self.categories = load_categories(self.root)
        self.num_classes = len(self.categories)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        mask_path = self.mask_dir / f"{image_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask pair for id {image_id}")

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB").resize(
                (self.image_width, self.image_height), Image.Resampling.BILINEAR
            )
        with Image.open(mask_path) as raw_mask:
            mask = raw_mask.resize(
                (self.image_width, self.image_height), Image.Resampling.NEAREST
            )

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_array = mask_array.astype(np.int64, copy=False)
        if mask_array.size and (mask_array.min() < 0 or mask_array.max() >= self.num_classes):
            raise ValueError(
                f"Mask {mask_path} contains labels outside [0, {self.num_classes - 1}]"
            )
        return image_tensor, torch.from_numpy(mask_array.copy()).long()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    restored = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return restored.clamp(0, 1).permute(1, 2, 0).numpy()

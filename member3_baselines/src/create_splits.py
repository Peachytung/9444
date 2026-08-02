from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data_uecfoodpix import read_ids, resolve_data_root


def dominant_foreground_class(mask_path: Path) -> int:
    with Image.open(mask_path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    values, counts = np.unique(array, return_counts=True)
    foreground = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != 0]
    return max(foreground, key=lambda item: item[1])[0] if foreground else 0


def stratified_validation_split(
    ids: list[str], labels: dict[str, int], val_size: int, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < val_size < len(ids):
        raise ValueError("val_size must be between 1 and len(ids)-1")
    rng = random.Random(seed)
    groups: dict[int, list[str]] = defaultdict(list)
    for image_id in ids:
        groups[labels[image_id]].append(image_id)
    for values in groups.values():
        rng.shuffle(values)

    quotas: dict[int, int] = {}
    remainder: list[tuple[float, int]] = []
    for label, values in sorted(groups.items()):
        raw = val_size * len(values) / len(ids)
        count = min(len(values), math.floor(raw))
        if len(values) >= 2 and raw > 0 and count == 0:
            count = 1
        quotas[label] = count
        remainder.append((raw - math.floor(raw), label))

    allocated = sum(quotas.values())
    if allocated < val_size:
        for _, label in sorted(remainder, reverse=True):
            capacity = len(groups[label]) - quotas[label]
            if capacity <= 0:
                continue
            take = min(capacity, val_size - allocated)
            quotas[label] += take
            allocated += take
            if allocated == val_size:
                break
    elif allocated > val_size:
        for _, label in sorted(remainder):
            removable = max(0, quotas[label] - (1 if len(groups[label]) >= 2 else 0))
            take = min(removable, allocated - val_size)
            quotas[label] -= take
            allocated -= take
            if allocated == val_size:
                break
    if allocated != val_size:
        raise RuntimeError(f"Could not allocate exactly {val_size} validation samples")

    val_ids: list[str] = []
    train_ids: list[str] = []
    for label, values in sorted(groups.items()):
        count = quotas[label]
        val_ids.extend(values[:count])
        train_ids.extend(values[count:])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-free UECFoodPIX splits.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9444)
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    output = Path(args.output_dir).expanduser().resolve()
    official_train = read_ids(root / "train9000.txt")
    official_test = read_ids(root / "test1000.txt")
    if set(official_train) & set(official_test):
        raise ValueError("Official train and test ids overlap")

    labels: dict[str, int] = {}
    for index, image_id in enumerate(official_train, start=1):
        labels[image_id] = dominant_foreground_class(
            root / "UECFoodPIX" / "train" / "mask" / f"{image_id}.png"
        )
        if index % 1000 == 0:
            print(f"scanned {index}/{len(official_train)} masks", flush=True)

    train_ids, val_ids = stratified_validation_split(
        official_train, labels, val_size=args.val_size, seed=args.seed
    )
    train_path = output / "train_8000.txt"
    val_path = output / "val_1000.txt"
    test_path = output / "test_official_1000.txt"
    write_ids(train_path, train_ids)
    write_ids(val_path, val_ids)
    write_ids(test_path, official_test)

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Generated train and validation splits overlap")
    if set(train_ids) | set(val_ids) != set(official_train):
        raise RuntimeError("Generated train/validation splits do not cover official training ids")

    manifest = {
        "seed": args.seed,
        "strategy": "stratified by dominant non-background mask class",
        "data_policy": "validation is drawn only from official train9000; official test1000 is held out",
        "counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(official_test)},
        "dominant_class_counts": {
            "train": dict(sorted(Counter(labels[x] for x in train_ids).items())),
            "validation": dict(sorted(Counter(labels[x] for x in val_ids).items())),
        },
        "files": {
            train_path.name: sha256(train_path),
            val_path.name: sha256(val_path),
            test_path.name: sha256(test_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()


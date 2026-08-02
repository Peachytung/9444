from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_uecfoodpix import read_ids, resolve_data_root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import and validate the shared 3000/500 UECFoodPIX split JSON."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", default="splits")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    split_json = Path(args.split_json).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in payload["train_ids"]]
    val_ids = [str(value) for value in payload["val_ids"]]
    official_train = read_ids(root / "train9000.txt")
    official_test = read_ids(root / "test1000.txt")

    if payload.get("train_size") != 3000 or len(train_ids) != 3000:
        raise ValueError("The shared split must contain exactly 3000 training ids")
    if payload.get("val_size") != 500 or len(val_ids) != 500:
        raise ValueError("The shared split must contain exactly 500 validation ids")
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise ValueError("Duplicate ids found in the shared split")
    if set(train_ids) & set(val_ids):
        raise ValueError("Shared training and validation ids overlap")
    if not set(train_ids).issubset(official_train) or not set(val_ids).issubset(official_train):
        raise ValueError("Shared training/validation ids must be subsets of official train9000")
    if set(train_ids) & set(official_test) or set(val_ids) & set(official_test):
        raise ValueError("Official test ids leaked into the shared training/validation split")

    train_path = output / "train_3000.txt"
    val_path = output / "val_500.txt"
    test_path = output / "test_official_1000.txt"
    write_ids(train_path, train_ids)
    write_ids(val_path, val_ids)
    write_ids(test_path, official_test)

    manifest = {
        "source_json": split_json.name,
        "source_json_sha256": sha256(split_json),
        "train_seed": payload.get("train_seed"),
        "val_seed": payload.get("val_seed"),
        "strategy": "shared group split imported verbatim from supplied JSON",
        "data_policy": (
            "3000 training and 500 validation ids are subsets of official train9000; "
            "official test1000 is held out until final evaluation"
        ),
        "counts": {"train": 3000, "validation": 500, "test": len(official_test)},
        "unused_official_train_count": len(set(official_train) - set(train_ids) - set(val_ids)),
        "files": {
            train_path.name: sha256(train_path),
            val_path.name: sha256(val_path),
            test_path.name: sha256(test_path),
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

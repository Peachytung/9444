from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = [("resnet50_fcn", "ResNet50-FCN"), ("vgg16_fcn", "VGG16-FCN")]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed baseline results.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--output-dir", default="outputs/comparison")
    args = parser.parse_args()
    outputs = Path(args.outputs).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    full: dict[str, dict] = {}
    for key, label in MODELS:
        run_dir = outputs / key
        completed = read_json(run_dir / "COMPLETED.json")
        config = read_json(run_dir / "run_config.json")
        test = completed["test"]
        row = {
            "model": label,
            "parameters": config["parameter_count"],
            "best_epoch": completed["best_epoch"],
            "validation_mIoU": completed["best_validation_mean_iou"],
            "test_loss": test["loss"],
            "test_pixel_accuracy": test["pixel_accuracy"],
            "test_mIoU": test["mean_iou"],
            "test_foreground_mIoU": test["foreground_mean_iou"],
            "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall"],
            "test_macro_f1": test["macro_f1"],
        }
        rows.append(row)
        full[key] = {"label": label, "config": config, "results": completed}

    with (output_dir / "baseline_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "baseline_results.json").write_text(
        json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics = [
        ("test_pixel_accuracy", "Pixel accuracy"),
        ("test_mIoU", "mIoU"),
        ("test_foreground_mIoU", "Foreground mIoU"),
        ("test_macro_f1", "Macro F1"),
    ]
    x = list(range(len(metrics)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    for model_index, row in enumerate(rows):
        offsets = [value + (model_index - 0.5) * width for value in x]
        values = [float(row[key]) for key, _ in metrics]
        bars = axis.bar(offsets, values, width, label=row["model"])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title("UECFoodPIX baseline comparison (official held-out test set)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


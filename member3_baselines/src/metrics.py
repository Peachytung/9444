from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def update_confusion_matrix(
    confusion: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, num_classes: int
) -> None:
    prediction = prediction.detach().reshape(-1).to(torch.int64)
    target = target.detach().reshape(-1).to(torch.int64)
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + prediction[valid].clamp(0, num_classes - 1)
    bins = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bins.reshape(num_classes, num_classes).to(confusion.device)


def segmentation_metrics(confusion: torch.Tensor) -> dict[str, float]:
    cm = confusion.detach().double().cpu()
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    union = support + predicted - tp
    present = support > 0
    valid_iou = union > 0
    precision_valid = predicted > 0

    class_accuracy = torch.zeros_like(tp)
    class_accuracy[present] = tp[present] / support[present]
    iou = torch.zeros_like(tp)
    iou[valid_iou] = tp[valid_iou] / union[valid_iou]
    precision = torch.zeros_like(tp)
    precision[precision_valid] = tp[precision_valid] / predicted[precision_valid]
    recall = class_accuracy
    f1 = torch.zeros_like(tp)
    denom = precision + recall
    nonzero = denom > 0
    f1[nonzero] = 2 * precision[nonzero] * recall[nonzero] / denom[nonzero]

    foreground_present = present.clone()
    foreground_present[0] = False
    foreground_iou = valid_iou.clone()
    foreground_iou[0] = False
    total = cm.sum().clamp_min(1)
    return {
        "pixel_accuracy": float(tp.sum() / total),
        "mean_class_accuracy": float(class_accuracy[present].mean()) if present.any() else 0.0,
        "mean_iou": float(iou[valid_iou].mean()) if valid_iou.any() else 0.0,
        "foreground_mean_iou": float(iou[foreground_iou].mean()) if foreground_iou.any() else 0.0,
        "macro_precision": float(precision[foreground_present].mean()) if foreground_present.any() else 0.0,
        "macro_recall": float(recall[foreground_present].mean()) if foreground_present.any() else 0.0,
        "macro_f1": float(f1[foreground_present].mean()) if foreground_present.any() else 0.0,
        "present_classes": int(present.sum()),
    }


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], path: Path, model_label: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title=f"{model_label}: cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_pixel_accuracy"] for row in history], label="Pixel accuracy")
    axes[1].plot(epochs, [row["val_mean_iou"] for row in history], label="mIoU")
    axes[1].plot(epochs, [row["val_foreground_mean_iou"] for row in history], label="Foreground mIoU")
    axes[1].set(title=f"{model_label}: validation scores", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_outputs(
    confusion: torch.Tensor, categories: list[str], output_dir: Path, max_classes: int = 25
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion.detach().cpu().numpy().astype(np.int64)
    np.save(output_dir / "test_confusion_matrix.npy", cm)
    with (output_dir / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *categories])
        for label, row in zip(categories, cm):
            writer.writerow([label, *row.tolist()])

    support = cm.sum(axis=1)
    selected = np.argsort(support)[::-1][:max_classes]
    selected = selected[support[selected] > 0]
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    small = normalized[np.ix_(selected, selected)]
    labels = [categories[index] for index in selected]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    full_image = axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set(
        title="All 103 classes (row-normalized)",
        xlabel="Predicted class id",
        ylabel="True class id",
    )
    class_ticks = np.arange(0, len(categories), 10)
    axes[0].set_xticks(class_ticks)
    axes[0].set_yticks(class_ticks)
    fig.colorbar(full_image, ax=axes[0], fraction=0.046, pad=0.04)

    detail_image = axes[1].imshow(small, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set(
        title=f"Detail: {len(selected)} most-supported true classes",
        xlabel="Predicted",
        ylabel="True",
    )
    axes[1].set_xticks(np.arange(len(labels)), labels=labels, rotation=70, ha="right", fontsize=7)
    axes[1].set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    fig.colorbar(detail_image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Official test pixel confusion (rows normalized by true-class support)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(confusion: torch.Tensor, categories: list[str], path: Path) -> None:
    cm = confusion.detach().double().cpu()
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    union = support + predicted - tp
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "support_pixels", "precision", "recall", "iou"])
        for index, category in enumerate(categories):
            precision = float(tp[index] / predicted[index]) if predicted[index] > 0 else 0.0
            recall = float(tp[index] / support[index]) if support[index] > 0 else 0.0
            iou = float(tp[index] / union[index]) if union[index] > 0 else 0.0
            writer.writerow([index, category, int(support[index]), precision, recall, iou])


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

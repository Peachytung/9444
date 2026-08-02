from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from data_uecfoodpix import UECFoodPixSegmentation, denormalize_image
from metrics import (
    plot_history,
    save_confusion_outputs,
    save_history,
    save_json,
    save_per_class_metrics,
    segmentation_metrics,
    update_confusion_matrix,
)
from models import build_model, count_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UECFoodPIX CNN segmentation baseline.")
    parser.add_argument("--model", choices=["resnet50", "vgg16"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Backward-compatible square size; overrides --image-width/--image-height when set.",
    )
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9444)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--throughput-mode",
        action="store_true",
        default=Path("reproducibility/ENABLE_THROUGHPUT_MODE").is_file(),
        help=(
            "Use faster CUDA kernels, channels-last tensors, TF32, and "
            "deferred loss synchronization without changing the training protocol."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-epochs-this-run",
        type=int,
        default=None,
        help="Run at most this many additional epochs, saving a resumable partial result without test evaluation.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int, throughput_mode: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(throughput_mode)
    torch.backends.cudnn.deterministic = not throughput_mode
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(throughput_mode)
        torch.backends.cudnn.allow_tf32 = bool(throughput_mode)
        if throughput_mode:
            torch.set_float32_matmul_precision("high")


def seed_epoch(seed: int, epoch: int, train_loader: DataLoader) -> None:
    """Make each epoch deterministic while still changing order/augmentation after a resume."""
    epoch_seed = seed + epoch
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(epoch_seed)


def make_loader(
    dataset: UECFoodPixSegmentation,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    log_every: int,
    throughput_mode: bool = False,
) -> tuple[float, dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_images = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    start = time.time()
    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if throughput_mode and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.shape[0]
        total_loss += loss.detach().float() * batch_size
        total_images += batch_size
        update_confusion_matrix(confusion, logits.argmax(1), masks, num_classes)
        if log_every and step % log_every == 0:
            phase = "train" if training else "eval"
            elapsed = time.time() - start
            print(
                f"{phase} step={step}/{len(loader)} loss={float(loss.detach()):.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    mean_loss = float((total_loss / max(total_images, 1)).cpu())
    return mean_loss, segmentation_metrics(confusion), confusion.cpu()


def subset_dataset(dataset: UECFoodPixSegmentation, limit: int | None) -> UECFoodPixSegmentation:
    if limit is not None:
        dataset.ids = dataset.ids[:limit]
    return dataset


def palette(num_classes: int) -> np.ndarray:
    colours = np.zeros((num_classes, 3), dtype=np.float32)
    colours[0] = (0.05, 0.05, 0.05)
    for index in range(1, num_classes):
        hue = (index * 0.61803398875) % 1.0
        colours[index] = matplotlib.colors.hsv_to_rgb((hue, 0.72, 0.95))
    return colours


@torch.no_grad()
def save_qualitative_predictions(
    model: nn.Module,
    dataset: UECFoodPixSegmentation,
    device: torch.device,
    output_path: Path,
    count: int = 6,
) -> None:
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=int)
    colours = palette(dataset.num_classes)
    fig, axes = plt.subplots(len(indices), 3, figsize=(10.5, 3.1 * len(indices)))
    if len(indices) == 1:
        axes = np.asarray([axes])
    for row, index in enumerate(indices):
        image, target = dataset[int(index)]
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(1).squeeze(0).cpu().numpy()
        target_array = target.numpy()
        axes[row, 0].imshow(denormalize_image(image))
        axes[row, 1].imshow(colours[target_array])
        axes[row, 2].imshow(colours[prediction])
        axes[row, 0].set_ylabel(f"test id {dataset.ids[int(index)]}", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def environment_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "command": [sys.executable, *sys.argv],
    }
    if device.type == "cuda":
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count_visible": torch.cuda.device_count(),
            }
        )
    return info


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.image_size is not None:
        args.image_width = args.image_size
        args.image_height = args.image_size
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    seed_everything(args.seed, throughput_mode=args.throughput_mode)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    split_dir = Path(args.split_dir).expanduser().resolve()
    run_name = f"{args.model}_fcn"
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = subset_dataset(
        UECFoodPixSegmentation(
            args.data_root,
            split_dir / "train_3000.txt",
            "train",
            (args.image_width, args.image_height),
            True,
        ),
        args.limit_train,
    )
    val_set = subset_dataset(
        UECFoodPixSegmentation(
            args.data_root,
            split_dir / "val_500.txt",
            "train",
            (args.image_width, args.image_height),
            False,
        ),
        args.limit_val,
    )
    test_set = subset_dataset(
        UECFoodPixSegmentation(
            args.data_root,
            split_dir / "test_official_1000.txt",
            "test",
            (args.image_width, args.image_height),
            False,
        ),
        args.limit_test,
    )
    loaders = {
        "train": make_loader(train_set, args.batch_size, args.num_workers, True, args.seed),
        "validation": make_loader(val_set, args.batch_size, args.num_workers, False, args.seed + 1),
        "test": make_loader(test_set, args.batch_size, args.num_workers, False, args.seed + 2),
    }

    model = build_model(args.model, train_set.num_classes, dropout=args.dropout).to(device)
    if args.throughput_mode and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    # PyTorch 2.0.1 fused AdamW cannot reliably resume this FCN's mixed-layout
    # optimizer state under AMP. Keep standard AdamW; the material throughput
    # gains come from channels-last/TF32/cuDNN and avoiding per-batch syncs.
    use_fused_adamw = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_fused_adamw,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "num_classes": train_set.num_classes,
            "parameter_count": count_parameters(model),
            "split_counts": {
                "train": len(train_set),
                "validation": len(val_set),
                "test": len(test_set),
            },
            "pretrained_weights": False,
            "scheduler": None,
            "early_stopping": False,
            "class_weighting": False,
            "kernel_mode": "throughput" if args.throughput_mode else "deterministic",
            "channels_last": bool(args.throughput_mode and device.type == "cuda"),
            "fused_adamw": bool(use_fused_adamw),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "throughput_mode_start_epoch": None,
        }
    )
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_miou = -1.0
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        previous_config = checkpoint.get("config", {})
        for key in (
            "model",
            "image_width",
            "image_height",
            "batch_size",
            "lr",
            "weight_decay",
            "dropout",
            "seed",
        ):
            if key in previous_config and previous_config[key] != config[key]:
                raise ValueError(
                    f"Cannot resume with changed {key}: {previous_config[key]!r} != {config[key]!r}"
                )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # Optimizer.load_state_dict restores the old parameter-group options as
        # well as the moments.  A deterministic checkpoint therefore resets
        # ``fused`` to False unless we explicitly re-apply the requested
        # throughput setting after loading it.
        for parameter_group in optimizer.param_groups:
            parameter_group["fused"] = use_fused_adamw
            if use_fused_adamw:
                parameter_group["foreach"] = None
        if args.throughput_mode and device.type == "cuda":
            # The checkpoint was created with NCHW parameters. Preserve every
            # optimizer value while matching 4-D Adam state strides to the
            # channels-last parameters used by the throughput path.
            for parameter, state in optimizer.state.items():
                # Non-fused AdamW stores its scalar step counter on CPU.  The
                # PyTorch 2.0 fused CUDA implementation requires every state
                # tensor, including that counter, on the parameter's device.
                # It also initializes fused step counters as one-element
                # tensors, whereas non-fused AdamW checkpoints use scalars.
                step_tensor = state.get("step")
                if use_fused_adamw and isinstance(step_tensor, torch.Tensor):
                    state["step"] = step_tensor.reshape(1).to(parameter.device)
                if parameter.ndim != 4:
                    continue
                for state_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    state_tensor = state.get(state_key)
                    if isinstance(state_tensor, torch.Tensor) and state_tensor.ndim == 4:
                        state[state_key] = state_tensor.contiguous(
                            memory_format=torch.channels_last
                        )
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.throughput_mode:
            config["throughput_mode_start_epoch"] = (
                previous_config.get("throughput_mode_start_epoch") or start_epoch
            )
        best_path = output_dir / "best.pt"
        metrics_path = output_dir / "best_validation_metrics.json"
        if best_path.exists() and metrics_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_epoch = int(best_checkpoint["epoch"])
            best_val_miou = float(json.loads(metrics_path.read_text(encoding="utf-8"))["mean_iou"])
        print(f"resuming {run_name} at epoch {start_epoch}", flush=True)

    save_json(config, output_dir / "run_config.json")
    save_json(environment_info(device), output_dir / "environment.json")
    (output_dir / "model_architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    end_epoch = args.epochs
    if args.max_epochs_this_run is not None:
        if args.max_epochs_this_run < 1:
            raise ValueError("max-epochs-this-run must be positive")
        end_epoch = min(args.epochs, start_epoch + args.max_epochs_this_run - 1)

    for epoch in range(start_epoch, end_epoch + 1):
        seed_epoch(args.seed, epoch, loaders["train"])
        epoch_start = time.time()
        train_loss, train_scores, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            train_set.num_classes,
            args.amp,
            optimizer,
            scaler,
            args.log_every,
            args.throughput_mode,
        )
        with torch.no_grad():
            val_loss, val_scores, val_confusion = run_epoch(
                model,
                loaders["validation"],
                criterion,
                device,
                val_set.num_classes,
                args.amp,
                None,
                None,
                args.log_every,
                args.throughput_mode,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_scores.items()},
            **{f"val_{key}": value for key, value in val_scores.items()},
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        save_history(history, output_dir / "history.csv")
        plot_history(history, output_dir / "training_curves.png", run_name)
        checkpoint = {
            "model_name": args.model,
            "num_classes": train_set.num_classes,
            "categories": train_set.categories,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_scores["mean_iou"] > best_val_miou:
            best_epoch = epoch
            best_val_miou = val_scores["mean_iou"]
            torch.save(checkpoint, output_dir / "best.pt")
            torch.save(val_confusion, output_dir / "best_validation_confusion.pt")
            save_json(val_scores, output_dir / "best_validation_metrics.json")
        print(json.dumps(row, indent=2), flush=True)

    if end_epoch < args.epochs:
        save_json(
            {
                "status": "partial",
                "last_completed_epoch": end_epoch,
                "target_epochs": args.epochs,
                "test_evaluated": False,
            },
            output_dir / "PARTIAL.json",
        )
        print(
            json.dumps(
                {
                    "status": "partial",
                    "last_completed_epoch": end_epoch,
                    "target_epochs": args.epochs,
                    "test_evaluated": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    with torch.no_grad():
        test_loss, test_scores, test_confusion = run_epoch(
            model,
            loaders["test"],
            criterion,
            device,
            test_set.num_classes,
            args.amp,
            None,
            None,
            args.log_every,
            args.throughput_mode,
        )
    test_scores = {"loss": test_loss, **test_scores}
    results = {
        "model": run_name,
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_val_miou,
        "test": test_scores,
    }
    save_json(results, output_dir / "test_metrics.json")
    save_confusion_outputs(test_confusion, test_set.categories, output_dir)
    save_per_class_metrics(test_confusion, test_set.categories, output_dir / "test_per_class_metrics.csv")
    save_qualitative_predictions(model, test_set, device, output_dir / "test_predictions.png")
    save_json(
        {"status": "complete", "completed_unix_time": time.time(), **results},
        output_dir / "COMPLETED.json",
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

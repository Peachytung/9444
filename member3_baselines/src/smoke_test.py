from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_uecfoodpix import UECFoodPixSegmentation
from models import build_model, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify data and both baseline forward/backward passes.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    split_dir = Path(args.split_dir).expanduser().resolve()
    dataset = UECFoodPixSegmentation(
        args.data_root,
        split_dir / "train_3000.txt",
        "train",
        (args.image_width, args.image_height),
        True,
    )
    image, mask = dataset[0]
    print(f"dataset_size={len(dataset)} classes={dataset.num_classes}")
    print(f"image={tuple(image.shape)} mask={tuple(mask.shape)} labels=[{int(mask.min())}, {int(mask.max())}]")
    for name in ("resnet50", "vgg16"):
        model = build_model(name, dataset.num_classes, dropout=args.dropout).to(device)
        model.train()
        logits = model(image.unsqueeze(0).to(device))
        loss = torch.nn.functional.cross_entropy(logits, mask.unsqueeze(0).to(device))
        loss.backward()
        print(
            f"{name}: parameters={count_parameters(model)} output={tuple(logits.shape)} loss={float(loss):.4f}"
        )
        del model, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

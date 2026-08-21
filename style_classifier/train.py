from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    MeterStyleDataset,
    build_deduplicated_records,
    discover_styles,
    file_sha256,
    stratified_split,
)
from .manifest import parse_markdown_manifest, valid_unique_entries
from .model import build_model, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a visual M01-M12 meter-style classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../all_set"))
    parser.add_argument("--manifest", type=Path, default=Path("all_set/仪表盘读数标注.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/style_classifier"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--no-yolo-crop", action="store_true")
    return parser.parse_args()


def accuracy(model, loader, device) -> tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += criterion(logits, labels).item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total), loss_total / max(1, total)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    audit = parse_markdown_manifest(args.manifest, args.dataset_root)
    audit.write_json(args.output_dir / "manifest_audit.json")
    holdout_entries = valid_unique_entries(audit)
    holdout_paths = {Path(entry.effective_absolute_path) for entry in holdout_entries if entry.effective_absolute_path}
    holdout_hashes = {file_sha256(path) for path in holdout_paths}

    styles = discover_styles(args.dataset_root)
    if len(styles) < 2:
        raise RuntimeError(f"Need at least two styles, found: {styles}")
    records, dataset_audit = build_deduplicated_records(
        args.dataset_root, styles, excluded_hashes=holdout_hashes
    )
    (args.output_dir / "dataset_hash_audit.json").write_text(
        json.dumps(dataset_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_records, validation_records = stratified_split(records, args.validation_fraction, args.seed)
    train_transform, evaluation_transform = build_transforms(args.image_size)
    use_yolo_crop = not args.no_yolo_crop
    train_dataset = MeterStyleDataset(train_records, train_transform, use_yolo_crop)
    validation_dataset = MeterStyleDataset(validation_records, evaluation_transform, use_yolo_crop)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(styles), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_accuracy = -1.0
    history: list[dict] = []
    started = time.time()

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        for parameter in model.parameters():
            parameter.requires_grad = not frozen
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        learning_rate = 1e-3 if frozen else 2e-4
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4
        )
        model.train()
        epoch_total = epoch_correct = 0
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.numel()
            epoch_correct += (logits.argmax(1) == labels).sum().item()
            epoch_total += labels.numel()
        validation_accuracy, validation_loss = accuracy(model, validation_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "frozen_backbone": frozen,
            "train_accuracy": epoch_correct / max(1, epoch_total),
            "train_loss": epoch_loss / max(1, epoch_total),
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "styles": styles,
                    "image_size": args.image_size,
                    "use_yolo_crop": use_yolo_crop,
                    "seed": args.seed,
                    "validation_accuracy": validation_accuracy,
                },
                args.output_dir / "best.pt",
            )

    report = {
        "device": str(device),
        "styles": styles,
        "class_count": len(styles),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "held_out_manifest_unique_count": len(holdout_paths),
        "held_out_manifest_unique_hash_count": len(holdout_hashes),
        "dataset_hash_audit": dataset_audit,
        "train_style_counts": dict(Counter(record.style for record in train_records)),
        "validation_style_counts": dict(Counter(record.style for record in validation_records)),
        "best_validation_accuracy": best_accuracy,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

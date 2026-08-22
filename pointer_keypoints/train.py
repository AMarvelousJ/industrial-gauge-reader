from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


DEFAULTS = {
    "imgsz": 384,
    "epochs": 150,
    "patience": 30,
    "batch": 16,
    "seed": 20260822,
    "degrees": 12.0,
    "perspective": 0.0005,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.35,
    "fliplr": 0.0,
    "flipud": 0.0,
}


def train_pose_model(args: argparse.Namespace) -> Path:
    dataset_yaml = args.data.resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "pose":
        raise ValueError("the training base must be a pose model; the frozen meter detector is forbidden")
    config = {
        **DEFAULTS,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "seed": args.seed,
        "data": str(dataset_yaml),
        "device": args.device,
        "project": str(args.project),
        "name": args.name,
        "cache": False,
        "workers": args.workers,
        "pretrained": True,
        "plots": True,
        "deterministic": True,
        "exist_ok": False,
    }
    run_dir = args.project / args.name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing training run: {run_dir}")
    model.train(**config)
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "requested_training_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in config.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError(f"training finished without best weights: {weights}")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated pivot/tip pose model.")
    parser.add_argument("--data", type=Path, default=Path("dataset/pointer_keypoints_v1/dataset.yaml"))
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--project", type=Path, default=Path("runs/pointer_keypoints"))
    parser.add_argument("--name", default="industrial_single_pointer_v1")
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    weights = train_pose_model(parse_args())
    print(json.dumps({"status": "ok", "weights": str(weights.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

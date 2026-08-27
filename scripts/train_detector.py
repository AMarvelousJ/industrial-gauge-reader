"""Train a single-class YOLOv8 detector on visually diverse meter styles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="训练多样式、统一 meter 类别的 YOLOv8 检测器。")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "dataset" / "meter.yaml",
        help="由 prepare_dataset.py 生成的 Ultralytics 数据配置。",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="预训练权重或恢复训练时使用的 last.pt。",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="RTX 3060 Laptop 6GB 的稳妥默认值；显存不足时可降至 4。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='auto、cpu 或 CUDA 编号，例如 "0"。',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Windows 数据加载进程数。",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache",
        choices=("false", "disk", "ram"),
        default="false",
        help="缓存策略；默认关闭，以免在原始图片目录旁写入磁盘缓存。",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=project_root / "runs" / "detect",
    )
    parser.add_argument("--name", default="meter_yolov8n")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --model 指定的 last.pt 恢复优化器与轮次状态。",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="允许复用同名输出目录。",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"数据配置不存在：{data_path}\n"
            "请先运行：python scripts/prepare_dataset.py"
        )
    if args.resume and not Path(args.model).is_file():
        raise FileNotFoundError("使用 --resume 时，--model 必须指向现有的 last.pt。")

    device = resolve_device(args.device)
    cache: bool | str = False if args.cache == "false" else args.cache
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}；CUDA 运行时: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"训练设备: {device}")

    model = YOLO(args.model)
    train_options = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "cache": cache,
        "amp": True,
        "cos_lr": True,
        "close_mosaic": min(10, max(0, args.epochs // 10)),
        "plots": True,
        "save": True,
        "save_period": 10,
        "exist_ok": args.exist_ok,
    }
    if args.resume:
        train_options["resume"] = True

    try:
        model.train(**train_options)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "GPU 显存不足。请把 --batch 从 8 降为 4 或 2 后重试。"
            ) from exc
        raise

    save_dir = Path(model.trainer.save_dir)
    print(f"训练结果：{save_dir}")
    print(f"最佳权重：{save_dir / 'weights' / 'best.pt'}")
    print(f"末轮权重：{save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()

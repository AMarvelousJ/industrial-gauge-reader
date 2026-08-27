"""Detect meters, draw boxes, and export padded crops for a reader model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jfif", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="框出仪表并裁剪 ROI，供下一阶段读数识别模型使用。"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "meter_detection",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--max-det",
        type=int,
        default=20,
        help="每张图片最多保留的仪表框数量。",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="在检测框四周额外保留的相对边距，默认 5%%。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="当 source 为目录时递归搜索图片。",
    )
    parser.add_argument(
        "--save-annotated",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def collect_images(source: Path, recursive: bool) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式：{source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"输入图片或目录不存在：{source}")

    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"目录中没有找到受支持的图片：{source}")
    return images


def image_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def padded_box(
    xyxy: list[float],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x_padding = (x2 - x1) * padding
    y_padding = (y2 - y1) * padding
    left = max(0, int(x1 - x_padding))
    top = max(0, int(y1 - y_padding))
    right = min(image_width, int(x2 + x_padding + 0.9999))
    bottom = min(image_height, int(y2 + y_padding + 0.9999))
    return left, top, right, bottom


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    return names[class_id] if 0 <= class_id < len(names) else str(class_id)


def main() -> None:
    args = parse_args()
    if not 0 <= args.padding <= 0.5:
        raise ValueError("--padding 必须位于 [0, 0.5] 范围内。")
    if not args.weights.is_file():
        raise FileNotFoundError(f"模型权重不存在：{args.weights}")

    images = collect_images(args.source, args.recursive)
    output = args.output.resolve()
    crop_dir = output / "crops"
    annotated_dir = output / "annotated"
    crop_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights.resolve()))
    device = resolve_device(args.device)
    results = model.predict(
        source=[str(path) for path in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )

    manifest_path = output / "detections.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    image_count = 0
    detection_count = 0

    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as manifest:
        for expected_source_path, result in zip(images, results, strict=True):
            image_count += 1
            # Ultralytics assigns synthetic names such as image0.jpg when a list
            # is passed as source. The ordered input list is the reliable origin.
            source_path = expected_source_path.resolve()
            output_id = image_id(source_path)
            image = result.orig_img
            image_height, image_width = image.shape[:2]

            if args.save_annotated:
                annotated_path = annotated_dir / f"{output_id}.jpg"
                if not cv2.imwrite(str(annotated_path), result.plot()):
                    raise OSError(f"无法写入标框图片：{annotated_path}")

            boxes = result.boxes
            if boxes is None:
                continue

            xyxy_rows = boxes.xyxy.detach().cpu().tolist()
            confidence_rows = boxes.conf.detach().cpu().tolist()
            class_rows = boxes.cls.detach().cpu().tolist()
            order = sorted(
                range(len(xyxy_rows)),
                key=lambda index: confidence_rows[index],
                reverse=True,
            )

            for meter_index, box_index in enumerate(order, start=1):
                raw_box = xyxy_rows[box_index]
                confidence = float(confidence_rows[box_index])
                detected_class_id = int(class_rows[box_index])
                left, top, right, bottom = padded_box(
                    raw_box,
                    image_width=image_width,
                    image_height=image_height,
                    padding=args.padding,
                )
                if right <= left or bottom <= top:
                    continue

                crop_path = crop_dir / f"{output_id}_meter_{meter_index:03d}.jpg"
                crop = image[top:bottom, left:right]
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"无法写入仪表裁剪图：{crop_path}")

                record = {
                    "source": source_path.as_posix(),
                    "crop": crop_path.resolve().as_posix(),
                    "class_id": detected_class_id,
                    "class_name": class_name(result.names, detected_class_id),
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": [round(float(value), 2) for value in raw_box],
                    "crop_bbox_xyxy": [left, top, right, bottom],
                    "image_width": image_width,
                    "image_height": image_height,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                detection_count += 1

    temporary_manifest.replace(manifest_path)
    print(f"处理图片：{image_count} 张；检出仪表：{detection_count} 个")
    print(f"裁剪目录：{crop_dir}")
    if args.save_annotated:
        print(f"标框目录：{annotated_dir}")
    print(f"检测清单：{manifest_path}")


if __name__ == "__main__":
    main()

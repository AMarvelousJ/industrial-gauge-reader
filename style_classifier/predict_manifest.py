from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from .manifest import parse_markdown_manifest
from .model import build_transforms, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict every row in the markdown sample manifest.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/style_classifier/best.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../all_set"))
    parser.add_argument("--manifest", type=Path, default=Path("all_set/仪表盘读数标注.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/style_classifier/manifest_eval"))
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("runs/detect/meter_yolov8n_final/weights/best.pt"),
        help="Frozen user-supplied one-class YOLO detector; it is never trained here.",
    )
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--crop-padding", type=float, default=0.04)
    return parser.parse_args()


def detector_crop(
    detector: YOLO,
    image_path: Path,
    confidence: float,
    image_size: int,
    padding: float,
) -> tuple[Image.Image | None, dict]:
    results = detector.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=image_size,
        max_det=1,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, {"detector_confidence": None, "detector_box_xyxy": None}
    box = results[0].boxes[0]
    left, top, right, bottom = [float(value) for value in box.xyxy[0].detach().cpu().tolist()]
    detector_confidence = float(box.conf[0].detach().cpu().item())
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = right - left, bottom - top
    left = max(0, round(left - width * padding))
    top = max(0, round(top - height * padding))
    right = min(image.width, round(right + width * padding))
    bottom = min(image.height, round(bottom + height * padding))
    if right <= left or bottom <= top:
        return None, {"detector_confidence": detector_confidence, "detector_box_xyxy": None}
    return image.crop((left, top, right, bottom)), {
        "detector_confidence": round(detector_confidence, 6),
        "detector_box_xyxy": [left, top, right, bottom],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = parse_markdown_manifest(args.manifest, args.dataset_root)
    audit.write_json(args.output_dir / "manifest_audit.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, styles, checkpoint = load_checkpoint(args.checkpoint, device)
    if not args.detector_weights.is_file():
        raise FileNotFoundError(f"Frozen detector weights not found: {args.detector_weights}")
    detector = YOLO(str(args.detector_weights))
    _, transform = build_transforms(checkpoint.get("image_size", 224))
    results: list[dict] = []
    inference_cache: dict[str, dict] = {}
    for entry in audit.entries:
        row = {
            "line_number": entry.line_number,
            "relative_path": entry.relative_path,
            "effective_absolute_path": entry.effective_absolute_path,
            "reading": entry.reading,
            "expected_style": entry.expected_style,
            "exists": entry.exists,
            "is_duplicate": entry.is_duplicate,
            "first_line_number": entry.first_line_number,
            "suggestion": entry.suggestion,
            "suggestion_exists": entry.suggestion_exists,
            "resolution": entry.resolution,
            "status": "missing" if entry.effective_absolute_path is None else "ok",
            "detector_confidence": None,
            "detector_box_xyxy": None,
            "predicted_style": None,
            "confidence": None,
            "correct": None,
        }
        if entry.effective_absolute_path is not None:
            cache_key = entry.effective_absolute_path
            if cache_key not in inference_cache:
                image, detector_info = detector_crop(
                    detector,
                    Path(entry.effective_absolute_path),
                    args.detector_confidence,
                    args.detector_image_size,
                    args.crop_padding,
                )
                prediction_info = dict(detector_info)
                if image is None:
                    prediction_info.update(
                        status="detector_miss", predicted_style=None, confidence=None
                    )
                else:
                    tensor = transform(image).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        probabilities = model(tensor).softmax(1)[0]
                    confidence, prediction = probabilities.max(0)
                    prediction_info.update(
                        status="ok",
                        predicted_style=styles[prediction.item()],
                        confidence=round(confidence.item(), 6),
                    )
                inference_cache[cache_key] = prediction_info
            row.update(inference_cache[cache_key])
            row["correct"] = row["predicted_style"] == entry.expected_style
        results.append(row)

    available = [row for row in results if row["resolution"] != "missing"]
    unique_available = []
    seen: set[str] = set()
    for row in available:
        key = row["effective_absolute_path"]
        if key not in seen:
            seen.add(key)
            unique_available.append(row)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "frozen_detector_weights": str(args.detector_weights.resolve()),
        "detector_was_trained": False,
        "device": str(device),
        "row_accuracy": sum(bool(row["correct"]) for row in available) / max(1, len(available)),
        "unique_accuracy": sum(bool(row["correct"]) for row in unique_available) / max(1, len(unique_available)),
        "available_rows": len(available),
        "available_unique_images": len(unique_available),
        "missing_rows": sum(row["status"] == "missing" for row in results),
        "detector_miss_rows": sum(row["status"] == "detector_miss" for row in results),
        "detector_coverage": sum(row["status"] == "ok" for row in available) / max(1, len(available)),
        "repair_candidate_rows": sum(row["resolution"] == "repair_candidate" for row in results),
        "duplicate_rows": audit.duplicate_row_count,
        "predictions": results,
    }
    per_style: dict[str, dict] = {}
    for style in sorted({row["expected_style"] for row in unique_available}):
        rows = [row for row in unique_available if row["expected_style"] == style]
        per_style[style] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "count": len(rows),
            "recall": sum(bool(row["correct"]) for row in rows) / max(1, len(rows)),
        }
    report["per_style_recall"] = per_style
    report["macro_recall"] = sum(value["recall"] for value in per_style.values()) / max(1, len(per_style))
    (args.output_dir / "predictions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

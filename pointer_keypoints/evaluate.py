from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contract import circular_distance_degrees, clockwise_angle_from_top
from .inference import PointerKeypointEstimator


def evaluate_rows(rows: list[dict[str, Any]], confidence_threshold: float) -> dict[str, Any]:
    pivot_errors: list[float] = []
    tip_errors: list[float] = []
    angle_errors: list[float] = []
    accepted = 0
    for row in rows:
        if min(float(row["pivot_confidence"]), float(row["pointer_tip_confidence"])) < confidence_threshold:
            continue
        accepted += 1
        truth_pivot = np.asarray(row["truth_pivot"], dtype=float)
        truth_tip = np.asarray(row["truth_pointer_tip"], dtype=float)
        predicted_pivot = np.asarray(row["predicted_pivot"], dtype=float)
        predicted_tip = np.asarray(row["predicted_pointer_tip"], dtype=float)
        diameter = float(row["dial_diameter"])
        pivot_errors.append(float(np.linalg.norm(predicted_pivot - truth_pivot) / diameter))
        tip_errors.append(float(np.linalg.norm(predicted_tip - truth_tip) / diameter))
        angle_errors.append(
            circular_distance_degrees(
                clockwise_angle_from_top(predicted_pivot, predicted_tip),
                clockwise_angle_from_top(truth_pivot, truth_tip),
            )
        )
    def percentile(values: list[float], q: float) -> float | None:
        return None if not values else float(np.percentile(values, q))
    total = len(rows)
    return {
        "status": "ok" if total else "empty",
        "sample_count": total,
        "accepted_count": accepted,
        "coverage": accepted / total if total else 0.0,
        "confidence_threshold": confidence_threshold,
        "pivot_median_diameter_ratio": None if not pivot_errors else statistics.median(pivot_errors),
        "pointer_tip_median_diameter_ratio": None if not tip_errors else statistics.median(tip_errors),
        "angle_median_degrees": None if not angle_errors else statistics.median(angle_errors),
        "angle_p90_degrees": percentile(angle_errors, 90),
        "acceptance": {
            "pivot_median_within_3_percent": bool(pivot_errors and statistics.median(pivot_errors) <= 0.03),
            "tip_median_within_5_percent": bool(tip_errors and statistics.median(tip_errors) <= 0.05),
            "angle_median_within_2_degrees": bool(angle_errors and statistics.median(angle_errors) <= 2.0),
            "angle_p90_within_5_degrees": bool(angle_errors and percentile(angle_errors, 90) <= 5.0),
            "coverage_at_least_90_percent": bool(total and accepted / total >= 0.90),
        },
    }


def calibrate_threshold(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = [round(value / 100.0, 2) for value in range(10, 96, 5)]
    reports = [(threshold, evaluate_rows(rows, threshold)) for threshold in candidates]
    eligible = [item for item in reports if item[1]["coverage"] >= 0.90]
    pool = eligible or reports
    threshold, report = min(
        pool,
        key=lambda item: (
            float("inf") if item[1]["angle_p90_degrees"] is None else item[1]["angle_p90_degrees"],
            -item[1]["coverage"],
            -item[0],
        ),
    )
    return threshold, report


def _load_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("split") == split]


def run_inference(dataset_root: Path, manifest_path: Path, split: str, estimator: PointerKeypointEstimator) -> list[dict[str, Any]]:
    rows = []
    for record in _load_manifest(manifest_path, split):
        image = cv2.imread(str(dataset_root / record["image"]))
        if image is None:
            raise FileNotFoundError(dataset_root / record["image"])
        estimate = estimator.predict(image)
        height, width = image.shape[:2]
        predicted_pivot = estimate.pivot or (0.0, 0.0)
        predicted_tip = estimate.pointer_tip or (0.0, 0.0)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "meter_family": record["meter_family"],
                "truth_pivot": [record["pivot"][0] * width, record["pivot"][1] * height],
                "truth_pointer_tip": [record["pointer_tip"][0] * width, record["pointer_tip"][1] * height],
                "predicted_pivot": predicted_pivot,
                "predicted_pointer_tip": predicted_tip,
                "pivot_confidence": estimate.pivot_confidence or 0.0,
                "pointer_tip_confidence": estimate.pointer_tip_confidence or 0.0,
                "dial_diameter": min(width, height),
                "prediction_status": estimate.status,
                "rejection_reason": estimate.rejection_reason,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate the pivot/tip model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/pointer_keypoints_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/pointer_keypoints_v1/manifest.jsonl"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibrate and args.split != "val":
        raise SystemExit("threshold calibration is only permitted on the validation split")
    threshold = args.threshold
    if args.threshold_file:
        threshold = float(json.loads(args.threshold_file.read_text(encoding="utf-8"))["confidence_threshold"])
    estimator = PointerKeypointEstimator(args.weights, confidence_threshold=threshold)
    rows = run_inference(args.dataset_root, args.manifest, args.split, estimator)
    if args.calibrate:
        threshold, report = calibrate_threshold(rows)
    else:
        report = evaluate_rows(rows, threshold)
    payload = {"schema_version": "1.0", "split": args.split, "confidence_threshold": threshold, "report": report, "predictions": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from style_classifier.manifest import parse_markdown_manifest, valid_unique_entries

from .frozen_detector import FrozenGaugeDetector
from .geometry import analyze_pointer
from .ocr_mapping import OCRScaleReader, extract_scale_points, infer_reading
from .scale_fit import OCRNumberObservation, fit_scale_models


def normalize_unit(candidates: list[str], mapping: dict | None = None) -> str | None:
    aliases = {
        "mpa": "MPa",
        "bar": "bar",
        "psi": "psi",
        "pa": "Pa",
        "kpa": "kPa",
        "℃": "degC",
        "°c": "degC",
        "a": "A",
        "kg/cm²": "kgf/cm2",
        "kg/cm2": "kgf/cm2",
    }
    normalized_candidates = [aliases.get(candidate.strip().lower()) for candidate in candidates]
    mapping = mapping or {}
    texts = [str(item.get("text", "")).lower() for item in mapping.get("ocr", {}).get("items", [])]
    numeric_values = [float(point.get("value")) for point in mapping.get("scale_points", [])]
    fitted_values = [float(point.get("value")) for point in mapping.get("inliers", [])]
    if any("en13190" in text for text in texts):
        return "degC"
    if any("pascal" in text or "poscol" in text for text in texts):
        return "Pa"
    if (
        fitted_values
        and max(abs(value) for value in fitted_values) <= 1.0
        and any("japan" in text for text in texts)
    ):
        # The CKD G59D face prints MPa vertically; OCR often sees the nearby
        # MADE IN JAPAN text while missing the large rotated unit glyphs.
        return "MPa"
    if "MPa" in normalized_candidates and "psi" in normalized_candidates and numeric_values:
        return "psi" if max(numeric_values) > 20 else "MPa"
    # Prefer physically specific printed units; single letters such as A/V are
    # frequent OCR false positives in brand text.
    priority = ["MPa", "bar", "psi", "Pa", "kPa", "degC", "kgf/cm2", "A", "V"]
    for normalized in priority:
        if normalized not in normalized_candidates:
            continue
        if normalized == "kPa":
            # Evaluator canonical units omit kPa; conversion happens here.
            return "kPa"
        return normalized
    return None


def select_scale_consistent_pointer(geometry: dict, ocr: dict, center: tuple[float, float], radius: float) -> None:
    points, units = extract_scale_points(ocr, center, radius)
    if {"mpa", "psi"}.issubset({unit.lower() for unit in units}):
        outer_points = [point for point in points if point.value > 20]
        if len(outer_points) >= 3:
            points = outer_points
    observations = [
        OCRNumberObservation(
            value=point.value,
            angle=point.angle,
            radius_ratio=((point.x - center[0]) ** 2 + (point.y - center[1]) ** 2) ** 0.5 / radius,
            confidence=point.score,
        )
        for point in points
    ]
    fit = fit_scale_models(observations, allow_decreasing=False)
    if fit.status != "ok":
        geometry["scale_consistent_pointer_selection"] = {"status": fit.status}
        return
    primary_evidence = 1.0
    colored = geometry.get("colored_pointer_candidate") or {}
    if (
        str(geometry.get("pointer_method", "")).startswith("hsv_red_pointer")
        and not colored.get("detached_scale_marker")
        and float(colored.get("elongation", 0.0)) < 2.5
    ):
        primary_evidence = 0.2
    raw_candidates = [(geometry.get("angle_degrees_clockwise_from_top"), primary_evidence, geometry.get("pointer_method"))]
    raw_candidates += [
        (candidate["angle_degrees"], float(candidate["score"]), "hough_line_scale_checked")
        for candidate in geometry.get("line_candidates", [])[:15]
    ]
    raw_candidates += [
        (geometry.get("radial_scan", {}).get("angle_degrees"), geometry.get("radial_scan", {}).get("confidence", 0.0), "radial_scale_checked")
    ]
    ranked = []
    for angle, evidence_score, method in raw_candidates:
        if angle is None:
            continue
        for candidate_angle, reverse_penalty in ((float(angle) % 360, 0.0), ((float(angle) + 180) % 360, 0.08)):
            for model_index, model in enumerate(fit.models):
                mapped = fit.map_pointer(candidate_angle, scale_index=model_index, max_extrapolation_degrees=35)
                if mapped.status == "ok":
                    rank = float(evidence_score) + 0.35 * model.confidence_score - reverse_penalty - 0.01 * float(mapped.extrapolation_degrees or 0)
                    ranked.append((rank, candidate_angle, method + ("_reversed" if reverse_penalty else ""), model_index, mapped))
    if not ranked:
        geometry["scale_consistent_pointer_selection"] = {"status": "no_candidate_inside_scale_arc", "scale_fit": fit.as_dict()}
        return
    best = max(ranked, key=lambda item: item[0])
    geometry["angle_degrees_clockwise_from_top"] = round(best[1], 4)
    geometry["pointer_method"] = best[2]
    geometry["scale_consistent_pointer_selection"] = {
        "status": "selected",
        "rank": round(best[0], 6),
        "scale_index": best[3],
        "mapped_preview": best[4].as_dict(),
        "scale_fit": fit.as_dict(),
    }


def segmented_pointer_angle(segmenter: YOLO, crop: np.ndarray, center: tuple[float, float]) -> dict | None:
    result = segmenter.predict(crop, conf=0.25, imgsz=640, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return None
    candidates = []
    for polygon, class_id, confidence in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
        if int(class_id.detach().cpu().item()) != 0 or len(polygon) < 2:
            continue
        points = np.asarray(polygon, dtype=float)
        distances = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])
        tip = points[int(distances.argmax())]
        angle = (np.degrees(np.arctan2(tip[0] - center[0], -(tip[1] - center[1]))) % 360).item()
        mean = points.mean(axis=0)
        _, _, axes = np.linalg.svd(points - mean, full_matrices=False)
        principal_axis = axes[0]
        projected = (points - mean) @ principal_axis
        endpoint_a = mean + principal_axis * np.percentile(projected, 2)
        endpoint_b = mean + principal_axis * np.percentile(projected, 98)
        image_center = np.asarray([crop.shape[1] / 2.0, crop.shape[0] / 2.0])
        pivot, pca_tip = (
            (endpoint_a, endpoint_b)
            if np.linalg.norm(endpoint_a - image_center) <= np.linalg.norm(endpoint_b - image_center)
            else (endpoint_b, endpoint_a)
        )
        pca_angle = (
            np.degrees(np.arctan2(pca_tip[0] - pivot[0], -(pca_tip[1] - pivot[1]))) % 360
        ).item()
        minimum_center_distance = float(distances.min())
        candidates.append(
            {
                "angle": float(angle),
                "pca_angle": float(pca_angle),
                "pca_pivot": [round(float(pivot[0]), 3), round(float(pivot[1]), 3)],
                "pca_tip": [round(float(pca_tip[0]), 3), round(float(pca_tip[1]), 3)],
                "confidence": float(confidence.detach().cpu().item()),
                "tip": [round(float(tip[0]), 3), round(float(tip[1]), 3)],
                "max_radius": round(float(distances.max()), 3),
                "minimum_center_distance": round(minimum_center_distance, 3),
            }
        )
    return max(candidates, key=lambda item: (item["confidence"], item["max_radius"])) if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-YOLO + OpenCV pointer-angle baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../all_set"))
    parser.add_argument("--manifest", type=Path, default=Path("all_set/仪表盘读数标注.md"))
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("runs/detect/meter_yolov8n_final/weights/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/style_reader/baseline"))
    parser.add_argument(
        "--pointer-segmenter",
        type=Path,
        default=Path("third_party/Gauge-Pointer-Reading/scale_segment.pt"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 processes every unique manifest image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    audit = parse_markdown_manifest(args.manifest, args.dataset_root)
    entries = valid_unique_entries(audit)
    if args.limit > 0:
        entries = entries[: args.limit]
    detector = FrozenGaugeDetector(args.detector_weights)
    segmenter = YOLO(str(args.pointer_segmenter)) if args.pointer_segmenter.is_file() else None
    ocr_reader = OCRScaleReader()
    rows: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        image_path = Path(entry.effective_absolute_path)
        crop, detection = detector.crop(image_path)
        row = {
            "sample_id": f"RG-{index:03d}",
            "relative_path": entry.relative_path,
            "effective_path": str(image_path),
            "ground_truth_used_by_algorithm": False,
            "path_resolution": entry.resolution,
            "detector": detection,
            "geometry": None,
            "visualization": None,
        }
        if crop is not None:
            ocr = ocr_reader.recognize(crop)
            geometry, visualization = analyze_pointer(crop, [item["box"] for item in ocr["items"]])
            scale = float(geometry["analysis_scale"])
            circle = geometry["circle"]
            center_original = (circle["center_x"] / scale, circle["center_y"] / scale)
            radius_original = circle["radius"] / scale
            segmented = segmented_pointer_angle(segmenter, crop, center_original) if segmenter else None
            geometry["segmented_pointer"] = segmented
            if segmented is not None:
                center_distance_ratio = segmented["minimum_center_distance"] / max(radius_original, 1e-6)
                use_pca_angle = center_distance_ratio <= 0.25
                selected_angle = segmented["pca_angle"] if use_pca_angle else segmented["angle"]
                geometry["angle_degrees_clockwise_from_top"] = round(selected_angle, 4)
                geometry["pointer_method"] = (
                    "mit_scale_segment_pointer_mask_pca"
                    if use_pca_angle
                    else "mit_scale_segment_pointer_mask_circle_center"
                )
                geometry["pointer_confidence"] = round(segmented["confidence"], 6)
                geometry["segmented_pointer"]["minimum_center_distance_ratio"] = round(center_distance_ratio, 6)
                # Strongly perspective-distorted/rectangular dials can make a
                # circle detector land on the pointer shaft. In that case the
                # mask endpoint nearest the image center is the better pivot.
                aspect_ratio = max(crop.shape[:2]) / max(1, min(crop.shape[:2]))
                if aspect_ratio >= 1.35 and use_pca_angle:
                    center_original = tuple(float(value) for value in segmented["pca_pivot"])
                    radius_original = min(crop.shape[:2]) * 0.42
                    geometry["ocr_geometry_override"] = {
                        "method": "segmented_pointer_pivot_for_rectangular_dial",
                        "center": [round(center_original[0], 3), round(center_original[1], 3)],
                        "radius": round(float(radius_original), 3),
                    }
            else:
                select_scale_consistent_pointer(geometry, ocr, center_original, radius_original)
            reading = infer_reading(
                ocr_reader,
                crop,
                center_original,
                radius_original,
                geometry["angle_degrees_clockwise_from_top"],
                ocr=ocr,
            )
            geometry["reading_mapping"] = reading
            geometry["reading"] = reading["reading"]
            geometry["reading_status"] = reading["status"]
            output_name = f"{index:02d}_{image_path.stem}.jpg"
            output_path = visual_dir / output_name
            cv2.imwrite(str(output_path), visualization)
            row["geometry"] = geometry
            row["visualization"] = str(output_path.resolve())
        rows.append(row)
        print(json.dumps({"index": index, "path": entry.relative_path, "detector": detection["status"], "geometry": None if row["geometry"] is None else row["geometry"]["status"]}, ensure_ascii=False))

    summary = {
        "detector_weights": str(detector.weights),
        "detector_trained_or_modified": False,
        "pointer_segmenter": None if segmenter is None else str(args.pointer_segmenter.resolve()),
        "pointer_segmenter_trained_or_modified": False,
        "manifest_rows": audit.row_count,
        "manifest_unique_resolved": audit.resolved_unique_count,
        "processed_unique": len(rows),
        "detector_success": sum(row["detector"]["status"] == "ok" for row in rows),
        "angle_estimated": sum(bool(row["geometry"] and row["geometry"]["status"] == "angle_estimated") for row in rows),
        "actual_reading_available": sum(bool(row["geometry"] and row["geometry"]["reading"] is not None) for row in rows),
        "actual_reading_note": "Values come only from OCR scale fitting and pointer geometry; MD readings are metadata and never enter inference.",
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    predictions = []
    for row in rows:
        if row["detector"]["status"] != "ok":
            status, value, unit = "detector_miss", None, None
        else:
            geometry = row["geometry"] or {}
            mapping = geometry.get("reading_mapping") or {}
            value = mapping.get("reading")
            unit = normalize_unit(mapping.get("unit_candidates", []), mapping)
            if unit == "kPa" and value is not None:
                value, unit = float(value) / 1000.0, "MPa"
            status = "ok" if value is not None and unit is not None else "no_output"
            if status != "ok":
                value = unit = None
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "status": status,
                "value": value,
                "unit": unit,
                "confidence": None if row["geometry"] is None else row["geometry"].get("pointer_confidence"),
                "source_path": row["relative_path"],
                "diagnostics": {"detector": row["detector"], "geometry": row["geometry"]},
            }
        )
    (args.output_dir / "predictions.json").write_text(
        json.dumps({"schema_version": "1.0", "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pointer_keypoints.contract import KeypointEstimate
from pointer_keypoints.inference import PointerKeypointEstimator

from .dial_geometry import DialGeometry, estimate_dial_geometry
from .frozen_detector import FrozenGaugeDetector
from .geometry import CircleEstimate, analyze_pointer
from .ocr_mapping import OCRScaleReader, extract_scale_points, infer_reading, infer_tick_anchored_reading
from .pointer_semantics import PointerCandidate, candidates_from_geometry, select_primary_pointer
from .scale_fit import OCRNumberObservation, fit_scale_models


def load_image_list(path: Path, dataset_root: Path) -> list[dict]:
    """Load a truth-free inference list containing only IDs and image paths."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or not isinstance(payload.get("images"), list):
        raise ValueError("image list must use schema_version 1.0 and contain an images array")
    entries: list[dict] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(payload["images"], start=1):
        if not isinstance(raw, dict) or set(raw) != {"sample_id", "path"}:
            raise ValueError(f"image list item {index} may contain only sample_id and path")
        sample_id = str(raw["sample_id"]).strip()
        relative_path = str(raw["path"]).replace("\\", "/").strip()
        if not sample_id or not relative_path or sample_id in seen_ids or relative_path in seen_paths:
            raise ValueError(f"image list item {index} is empty or duplicated")
        absolute_path = (dataset_root / Path(relative_path)).resolve()
        root = dataset_root.resolve()
        try:
            absolute_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"image list item {index} escapes dataset root") from exc
        if not absolute_path.is_file():
            raise FileNotFoundError(f"image list item {index} not found: {absolute_path}")
        seen_ids.add(sample_id)
        seen_paths.add(relative_path)
        entries.append(
            {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "absolute_path": absolute_path,
            }
        )
    return entries


def normalization_policy(dial: DialGeometry) -> tuple[bool, str]:
    """Conservatively decide whether geometry rectification may affect reading."""
    if dial.geometry_type == "perspective_ellipse":
        axes = dial.source_boundary.axes or (1.0, 1.0)
        axis_ratio = max(axes) / max(min(axes), 1e-6)
        accepted = dial.confidence >= 0.80 and dial.reprojection_error <= 8.0 and axis_ratio >= 1.25
        return accepted, "strong_perspective_ellipse" if accepted else "ellipse_shadow_only"
    if dial.geometry_type == "rectangular_sector":
        accepted = dial.confidence >= 0.68 and dial.reprojection_error <= 14.0
        return accepted, "rectangular_sector" if accepted else "rectangle_shadow_only"
    return False, "legacy_round_reader"


def canonical_radius(dial: DialGeometry) -> float:
    pivot = np.asarray(dial.canonical_pivot, dtype=float)
    curve = np.asarray(dial.canonical_scale_curve, dtype=float)
    if len(curve) == 0:
        return min(dial.canonical_size) * 0.45
    scale_radius = float(np.median(np.linalg.norm(curve - pivot, axis=1)))
    return scale_radius / 0.90 if dial.geometry_type != "rectangular_sector" else scale_radius


def pointer_semantics(
    geometry: dict,
    segmented: dict | None,
    radius: float,
    keypoint: KeypointEstimate | None = None,
) -> None:
    candidates = candidates_from_geometry(geometry)
    if segmented is not None:
        distance_ratio = float(segmented.get("minimum_center_distance", radius)) / max(radius, 1e-6)
        extent_ratio = float(segmented.get("max_radius", 0.0)) / max(radius, 1e-6)
        candidates.append(
            PointerCandidate(
                candidate_id="segmented",
                angle_degrees=segmented.get("pca_angle") if distance_ratio <= 0.25 else segmented.get("angle"),
                confidence=float(segmented.get("confidence", 0.0)),
                source="mit_scale_segment_pointer_mask",
                pivot_distance_ratio=distance_ratio,
                extent_ratio=extent_ratio,
                diagnostics={"minimum_center_distance_ratio": round(distance_ratio, 6)},
            )
        )
    preferred = None
    if keypoint is not None and keypoint.pivot is not None and keypoint.pointer_tip is not None:
        preferred = PointerCandidate(
            candidate_id="learned_keypoints",
            angle_degrees=keypoint.angle_degrees_clockwise_from_top,
            confidence=float(keypoint.confidence or 0.0),
            source="pivot_tip_pose_model",
            pivot_connected=True,
            extent_ratio=keypoint.length_ratio,
            diagnostics={
                "pivot": list(keypoint.pivot),
                "pointer_tip": list(keypoint.pointer_tip),
                "coordinate_system": keypoint.coordinate_system,
                "status": keypoint.status,
            },
        )
        candidates.append(preferred)
    selection = select_primary_pointer(candidates, ambiguity_margin=0.02)
    geometry["pointer_candidates"] = [candidate.as_dict() for candidate in candidates]
    if keypoint is not None and keypoint.status == "accepted" and preferred is not None:
        geometry["pointer_selection"] = {
            "status": "selected",
            "angle_degrees": keypoint.angle_degrees_clockwise_from_top,
            "primary": preferred.as_dict(),
            "diagnostics": {
                "preferred_source": "validated_pivot_tip_pose_model",
                "fallback_selection": selection.as_dict(),
            },
        }
        geometry["selected_pointer_role"] = "measurement"
        geometry["status"] = "angle_estimated"
        geometry["angle_degrees_clockwise_from_top"] = keypoint.angle_degrees_clockwise_from_top
        geometry["pointer_confidence"] = round(float(keypoint.confidence or 0.0), 6)
        geometry["pointer_method"] = "semantic_main:pivot_tip_pose_model"
        return
    geometry["pointer_selection"] = selection.as_dict()
    geometry["selected_pointer_role"] = "measurement" if selection.status == "selected" else None
    if selection.status != "selected" or selection.primary is None:
        geometry["status"] = "pointer_not_found"
        geometry["angle_degrees_clockwise_from_top"] = None
        geometry["pointer_confidence"] = 0.0
        geometry["pointer_method"] = f"semantic_{selection.status}"
        return
    geometry["angle_degrees_clockwise_from_top"] = selection.angle_degrees
    geometry["pointer_confidence"] = round(float(selection.primary.confidence), 6)
    geometry["pointer_method"] = f"semantic_main:{selection.primary.source}"


def keypoint_in_working_space(
    estimate: KeypointEstimate,
    dial: DialGeometry,
    *,
    apply_normalization: bool,
    working_shape: tuple[int, ...],
) -> KeypointEstimate:
    if estimate.pivot is None or estimate.pointer_tip is None:
        return estimate
    if not apply_normalization:
        return estimate
    mapped = dial.source_to_canonical(np.asarray((estimate.pivot, estimate.pointer_tip), dtype=float))
    return estimate.with_points(
        mapped[0],
        mapped[1],
        coordinate_system="canonical",
        dial_diameter=min(working_shape[:2]),
    )


def draw_keypoint_overlay(image: np.ndarray, estimate: KeypointEstimate | None) -> None:
    if estimate is None or estimate.status != "accepted" or estimate.pivot is None or estimate.pointer_tip is None:
        return
    pivot = tuple(int(round(value)) for value in estimate.pivot)
    tip = tuple(int(round(value)) for value in estimate.pointer_tip)
    cv2.line(image, pivot, tip, (180, 40, 255), 3, cv2.LINE_AA)
    cv2.circle(image, pivot, 6, (180, 40, 255), 2, cv2.LINE_AA)
    cv2.circle(image, tip, 5, (180, 40, 255), 2, cv2.LINE_AA)


def source_pointer_overlay(dial: DialGeometry, angle: float | None, radius: float) -> dict | None:
    if angle is None:
        return None
    radians = np.deg2rad(float(angle))
    center = np.asarray(dial.canonical_pivot, dtype=float)
    tip = center + np.asarray([np.sin(radians), -np.cos(radians)]) * radius * 0.72
    mapped = dial.canonical_to_source(np.vstack((center, tip)))
    return {
        "center": [round(float(value), 3) for value in mapped[0]],
        "tip": [round(float(value), 3) for value in mapped[1]],
    }


def transform_ocr_to_canonical(ocr: dict, dial: DialGeometry) -> dict:
    """Preserve source OCR recognition while moving its geometry after warp."""
    items = []
    for item in ocr.get("items", []):
        box = np.asarray(item.get("box") or [], dtype=float)
        if box.size != 8:
            continue
        transformed = dial.source_to_canonical(box.reshape(4, 2))
        copied = dict(item)
        copied["source_box"] = item.get("box")
        copied["box"] = transformed.round(2).tolist()
        copied["center"] = transformed.mean(axis=0).round(3).tolist()
        copied["coordinate_transform"] = "source_to_canonical"
        copied["coordinate_space"] = "canonical"
        items.append(copied)
    return {
        **ocr,
        "items": items,
        "source_item_count": len(ocr.get("items", [])),
        "coordinate_transform": "source_to_canonical",
    }


def build_stage_diagnostics(geometry: dict) -> dict:
    dial = geometry.get("dial_geometry") or {}
    selection = geometry.get("pointer_selection") or {}
    tick = geometry.get("tick_mapping") or {}
    issues: list[str] = []
    if dial.get("geometry_type") == "roi_fallback":
        issues.append("dial_boundary_low_confidence")
    if selection.get("status") != "selected":
        issues.append("main_pointer_not_uniquely_selected")
    marker_count = len((selection.get("diagnostics") or {}).get("marker_candidates") or [])
    if marker_count:
        issues.append("detached_marker_excluded_from_primary_reading")
    if tick.get("status") == "ok" and not tick.get("trusted_for_reading", False):
        issues.append("physical_tick_mapping_shadow_only_unverified_major_tick_or_ring")
    elif tick.get("status") != "ok":
        issues.append("physical_tick_mapping_unavailable_legacy_scale_fallback_used")
    segmented = geometry.get("segmented_pointer") or {}
    if (
        dial.get("geometry_type") == "rectangular_sector"
        and float(segmented.get("minimum_center_distance_ratio", 0.0)) > 0.25
    ):
        issues.append("rectangular_virtual_pivot_uncertain")
    return {
        "geometry_type": dial.get("geometry_type"),
        "geometry_confidence": dial.get("confidence"),
        "geometry_reprojection_error": dial.get("reprojection_error"),
        "pointer_selection_status": selection.get("status"),
        "pointer_confidence": geometry.get("pointer_confidence"),
        "keypoint_status": (geometry.get("keypoint_model") or {}).get("status"),
        "tick_mapping_status": tick.get("status"),
        "issues": issues,
    }


def merge_tick_reading(legacy: dict, tick_reading: dict, *, trusted: bool) -> dict:
    """Use tick anchors only after the caller validates their coordinate frame."""
    if tick_reading.get("status") != "ok" or not trusted:
        reason = tick_reading.get("status", "missing")
        if tick_reading.get("status") == "ok" and not trusted:
            reason = "shadow_only_unverified_major_tick_or_ring"
        return {**legacy, "tick_fallback_reason": reason}
    return {
        **legacy,
        "legacy_mapping": legacy,
        "status": "ok",
        "reading": tick_reading.get("reading"),
        "method": tick_reading.get("method"),
        "tick_mapping": tick_reading.get("tick_mapping"),
        "pointer_mapping": tick_reading.get("pointer_mapping"),
    }


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
            # The public writer converts kPa to MPa for a compact canonical output.
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
    parser.add_argument("--dataset-root", type=Path, default=Path("all_set"))
    parser.add_argument("--image-list", type=Path, default=Path("docs/reading_images.json"))
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
    parser.add_argument("--pointer-keypoints", type=Path, help="Optional trained 2-keypoint pose weights")
    parser.add_argument("--keypoint-threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-threshold-file", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every unique manifest image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    entries = load_image_list(args.image_list, args.dataset_root)
    if args.limit > 0:
        entries = entries[: args.limit]
    detector = FrozenGaugeDetector(args.detector_weights)
    segmenter = YOLO(str(args.pointer_segmenter)) if args.pointer_segmenter.is_file() else None
    keypoint_threshold = args.keypoint_threshold
    if args.keypoint_threshold_file is not None:
        threshold_payload = json.loads(args.keypoint_threshold_file.read_text(encoding="utf-8"))
        keypoint_threshold = float(threshold_payload["confidence_threshold"])
    keypoint_estimator = (
        PointerKeypointEstimator(args.pointer_keypoints, confidence_threshold=keypoint_threshold)
        if args.pointer_keypoints is not None
        else None
    )
    ocr_reader = OCRScaleReader()
    rows: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        image_path = Path(entry["absolute_path"])
        crop, detection = detector.crop(image_path)
        row = {
            "sample_id": entry["sample_id"],
            "relative_path": entry["relative_path"],
            "effective_path": str(image_path),
            "ground_truth_used_by_algorithm": False,
            "path_resolution": "truth_free_image_list",
            "detector": detection,
            "geometry": None,
            "visualization": None,
        }
        if crop is not None:
            dial = estimate_dial_geometry(crop)
            apply_normalization, normalization_reason = normalization_policy(dial)
            working_crop = dial.warp_to_canonical(crop) if apply_normalization else crop
            keypoint = None
            if keypoint_estimator is not None:
                keypoint = keypoint_in_working_space(
                    keypoint_estimator.predict(crop),
                    dial,
                    apply_normalization=apply_normalization,
                    working_shape=working_crop.shape,
                )
            source_ocr = ocr_reader.recognize(crop)
            ocr = transform_ocr_to_canonical(source_ocr, dial) if apply_normalization else source_ocr
            override = None
            if apply_normalization:
                dial_radius = canonical_radius(dial)
                override = CircleEstimate(
                    center_x=float(dial.canonical_pivot[0]),
                    center_y=float(dial.canonical_pivot[1]),
                    radius=float(dial_radius),
                    method=f"dial_geometry:{dial.geometry_type}",
                    confidence=float(dial.confidence),
                )
            geometry, visualization = analyze_pointer(
                working_crop,
                [item["box"] for item in ocr["items"]],
                circle_override=override,
            )
            geometry["dial_geometry"] = dial.as_dict()
            geometry["normalization"] = {
                "applied": apply_normalization,
                "reason": normalization_reason,
                "working_coordinate_system": "canonical" if apply_normalization else "detector_crop",
            }
            geometry["coordinate_spaces"] = {
                "dial_geometry_source_boundary": "detector_crop",
                "dial_geometry_canonical": "canonical",
                "circle_pointer_tip_line_candidates": "analysis_image",
                "segmented_pointer": "canonical" if apply_normalization else "detector_crop",
                "keypoint_model": "canonical" if apply_normalization else "detector_crop",
                "ocr_items": "canonical" if apply_normalization else "detector_crop",
                "source_pointer_overlay": "detector_crop",
            }
            scale = float(geometry["analysis_scale"])
            circle = geometry["circle"]
            center_original = (circle["center_x"] / scale, circle["center_y"] / scale)
            radius_original = circle["radius"] / scale
            segmented = segmented_pointer_angle(segmenter, working_crop, center_original) if segmenter else None
            geometry["segmented_pointer"] = segmented
            geometry["keypoint_model"] = None if keypoint is None else keypoint.as_dict()
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
                aspect_ratio = max(working_crop.shape[:2]) / max(1, min(working_crop.shape[:2]))
                if aspect_ratio >= 1.35 and use_pca_angle:
                    center_original = tuple(float(value) for value in segmented["pca_pivot"])
                    radius_original = min(working_crop.shape[:2]) * 0.42
                    geometry["ocr_geometry_override"] = {
                        "method": "segmented_pointer_pivot_for_rectangular_dial",
                        "center": [round(center_original[0], 3), round(center_original[1], 3)],
                        "radius": round(float(radius_original), 3),
                    }
            else:
                select_scale_consistent_pointer(geometry, ocr, center_original, radius_original)
            pointer_semantics(geometry, segmented, radius_original, keypoint)
            geometry["geometry_source"] = (
                "pivot_tip_pose_model"
                if keypoint is not None and keypoint.status == "accepted"
                else "legacy_geometry_fallback"
            )
            if keypoint is not None and keypoint.status == "accepted" and keypoint.pivot is not None:
                center_original = tuple(float(value) for value in keypoint.pivot)
                geometry["ocr_geometry_override"] = {
                    "method": "validated_pivot_tip_pose_model",
                    "center": [round(value, 3) for value in center_original],
                    "radius": round(float(radius_original), 3),
                }
            tick_reading = infer_tick_anchored_reading(
                ocr,
                center_original,
                radius_original,
                geometry["angle_degrees_clockwise_from_top"],
                geometry.get("tick_angle_candidates") or [],
            )
            reading = infer_reading(
                ocr_reader,
                working_crop,
                center_original,
                radius_original,
                geometry["angle_degrees_clockwise_from_top"],
                ocr=ocr,
            )
            geometry["tick_mapping"] = tick_reading
            # Darkness peaks alone do not prove that OCR values belong to the
            # same major-tick ring/unit. Keep the new mapper observable but do
            # not let it replace the established scale mapping yet.
            tick_trusted = False
            tick_reading["trusted_for_reading"] = tick_trusted
            tick_reading["trust_reason"] = "major_tick_or_ring_evidence_not_yet_verified"
            reading = merge_tick_reading(reading, tick_reading, trusted=tick_trusted)
            geometry["reading_mapping"] = reading
            geometry["reading"] = reading["reading"]
            geometry["reading_status"] = reading["status"]
            geometry["stage_diagnostics"] = build_stage_diagnostics(geometry)
            if apply_normalization:
                if keypoint is not None and keypoint.status == "accepted" and keypoint.pivot and keypoint.pointer_tip:
                    mapped = dial.canonical_to_source(np.asarray((keypoint.pivot, keypoint.pointer_tip), dtype=float))
                    geometry["source_pointer_overlay"] = {
                        "center": [round(float(value), 3) for value in mapped[0]],
                        "tip": [round(float(value), 3) for value in mapped[1]],
                    }
                else:
                    geometry["source_pointer_overlay"] = source_pointer_overlay(
                        dial,
                        geometry["angle_degrees_clockwise_from_top"],
                        radius_original,
                    )
            draw_keypoint_overlay(visualization, keypoint)
            output_name = f"{index:02d}_{image_path.stem}.jpg"
            output_path = visual_dir / output_name
            cv2.imwrite(str(output_path), visualization)
            row["geometry"] = geometry
            row["visualization"] = str(output_path.resolve())
        rows.append(row)
        print(json.dumps({"index": index, "path": entry["relative_path"], "detector": detection["status"], "geometry": None if row["geometry"] is None else row["geometry"]["status"]}, ensure_ascii=False))

    summary = {
        "detector_weights": str(detector.weights),
        "detector_trained_or_modified": False,
        "pointer_segmenter": None if segmenter is None else str(args.pointer_segmenter.resolve()),
        "pointer_segmenter_trained_or_modified": False,
        "pointer_keypoints": None if keypoint_estimator is None else str(keypoint_estimator.weights),
        "pointer_keypoints_trained_or_modified_during_inference": False,
        "keypoint_confidence_threshold": None if keypoint_estimator is None else keypoint_threshold,
        "input_contract": str(args.image_list.resolve()),
        "input_contains_ground_truth": False,
        "manifest_rows": len(entries),
        "manifest_unique_resolved": len(entries),
        "processed_unique": len(rows),
        "detector_success": sum(row["detector"]["status"] == "ok" for row in rows),
        "angle_estimated": sum(bool(row["geometry"] and row["geometry"]["status"] == "angle_estimated") for row in rows),
        "keypoint_accepted": sum(
            bool(row["geometry"] and (row["geometry"].get("keypoint_model") or {}).get("status") == "accepted")
            for row in rows
        ),
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

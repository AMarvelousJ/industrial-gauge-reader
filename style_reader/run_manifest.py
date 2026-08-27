from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pointer_keypoints.contract import KeypointEstimate, circular_distance_degrees
from pointer_keypoints.inference import PointerKeypointEstimator

from .dial_geometry import DialGeometry, estimate_dial_geometry
from .frozen_detector import FrozenGaugeDetector
from .geometry import CircleEstimate, analyze_pointer
from .meter_family import classify_family, color_zone_stats
from .ocr_mapping import OCRScaleReader, extract_scale_points, infer_reading, infer_tick_anchored_reading, interpolate_reading_at_anchors
from .pointer_semantics import PointerCandidate, PointerSelection, _measurement_score, candidates_from_geometry, score_tip_orientation, select_primary_pointer
from .scale_mapping_ensemble import ensemble_mapping
from .unit_inference import infer_unit


def _thin_needle_extent(gray: np.ndarray, pivot: tuple[float, float], angle_degrees: float, max_radius: float) -> float:
    """Contiguous thin-dark-line extent from the pivot along a direction.

    The reading end of a needle is a long thin line; the counter-weight / tail
    is short and thick.  Walking outward, a thin dark line (small perpendicular
    dark count) extends the measure; a thick object stops it.  Purely geometric
    evidence for the tip-direction vote (never a standalone flip trigger).
    """
    dark = gray < 118
    height, width = gray.shape
    angle = math.radians(float(angle_degrees))
    extent = 0.0
    gap = 0
    radius = 1.0
    while radius < max_radius:
        x = pivot[0] + radius * math.sin(angle)
        y = pivot[1] - radius * math.cos(angle)
        xi, yi = int(round(x)), int(round(y))
        if not (3 <= xi < width - 3 and 3 <= yi < height - 3):
            break
        patch = dark[yi - 3 : yi + 4, xi - 3 : xi + 4]
        if patch.any():
            if int(patch.sum()) <= 6:
                extent = radius
                gap = 0
            else:
                gap += 1
                if gap > 4:
                    break
        else:
            gap += 1
            if gap > 3:
                break
        radius += 0.5
    return extent
from .scale_fit import OCRNumberObservation, fit_scale_models


import re as _re


_SCALE_NUM_RE = None


def _sanitize_ocr_scale_items(ocr: dict) -> dict:
    """Drop numeric OCR tokens that cannot be dial scale numerals.

    Industrial dial scales are at most 5 digits; anything larger is almost
    always nameplate/serial text misread by OCR and would poison the scale
    fit (e.g. "232483701").  Non-numeric tokens (units, brands) pass through.
    """
    items = list(ocr.get("items") or [])
    kept: list[dict] = []
    for item in items:
        token = str(item.get("text", "")).strip().replace("−", "-").replace("—", "")
        if _re.fullmatch(r"^-?\d+(\.\d+)?$", token) is not None:
            try:
                if abs(float(token)) > 99999.0:
                    continue
            except ValueError:
                continue
        kept.append(item)
    if len(kept) != len(items):
        return {**ocr, "items": kept, "sanitized_tokens": [str(i.get("text")) for i in items if i not in kept]}
    return ocr


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


def _finite_ratio(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_frame_line_suspect(candidate: PointerCandidate) -> bool:
    """A long horizontal Hough "line" that is actually a dial frame, nameplate
    underline or scale bar (the signature found in the Magnehelic / square-meter
    failures).  A real radial pointer can sit at any angle and rarely spans the
    full dial width."""
    if candidate.source != "hough_line":
        return False
    length = candidate.extent_ratio
    angle_value = candidate.angle_degrees
    if length is None or angle_value is None:
        return False
    if float(length) > 0.85:
        return True
    return abs(float(angle_value) % 360.0 - 270.0) <= 12.0 and float(length) > 0.55


def pointer_semantics(
    geometry: dict,
    segmented: dict | None,
    radius: float,
    keypoint: KeypointEstimate | None = None,
    *,
    keypoint_agreement_degrees: float = 4.0,
    frame_line_override: bool = False,
) -> None:
    candidates = candidates_from_geometry(geometry)
    if segmented is not None:
        distance_ratio = float(segmented.get("minimum_center_distance", radius)) / max(radius, 1e-6)
        extent_ratio = float(segmented.get("max_radius", 0.0)) / max(radius, 1e-6)
        hub_override = bool(geometry.get("hub_override_applied"))
        # The mask defines its own pivot (pca_pivot).  When that hub replaced the
        # fitted circle as the reading origin (box gauges), the mask is the only
        # hub-consistent pointer evidence: it is pivot-attached by construction
        # and its pca_angle is measured from that same hub.
        mask_angle = (
            segmented.get("pca_angle")
            if (hub_override or distance_ratio <= 0.25) and segmented.get("pca_angle") is not None
            else segmented.get("angle")
        )
        candidates.append(
            PointerCandidate(
                candidate_id="segmented",
                angle_degrees=mask_angle,
                confidence=float(segmented.get("confidence", 0.0)),
                source="mit_scale_segment_pointer_mask",
                pivot_connected=hub_override or None,
                pivot_distance_ratio=distance_ratio,
                extent_ratio=extent_ratio,
                diagnostics={
                    "minimum_center_distance_ratio": round(distance_ratio, 6),
                    "hub_override": hub_override,
                },
            )
        )
    # The geometric selection runs exactly as before (baseline behavior).  The
    # fix is a targeted, auditable FRAME-LINE OVERRIDE: box / rectangular gauges
    # (Magnehelic, square meters) produce long horizontal Hough lines from dial
    # frames, nameplates and scale bars that outscore the true needle.  Only
    # when the winning line is such a frame line and stronger pointer evidence
    # exists (mask PCA axis, thin red needle) is the selection replaced with
    # that evidence.  Round gauges where the baseline is correct are untouched.
    selection = select_primary_pointer(candidates, ambiguity_margin=0.02)
    geometry["pointer_candidates"] = [candidate.as_dict() for candidate in candidates]
    geometry["pointer_selection"] = selection.as_dict()
    override_center: tuple[float, float] | None = None
    if frame_line_override and selection.status == "selected" and selection.primary is not None:
        primary = selection.primary
        if _is_frame_line_suspect(primary):
            hub = None
            if segmented is not None and segmented.get("pca_pivot") is not None:
                hub = tuple(float(value) for value in segmented["pca_pivot"])
            alternatives: list[tuple[float, PointerCandidate]] = []
            colored_present = any(candidate.source == "colored" and candidate.resolved_role(max_pivot_distance_ratio=0.22) == "measurement" for candidate in candidates)
            for candidate in candidates:
                if candidate is primary:
                    continue
                if candidate.source == "hough_line" and _is_frame_line_suspect(candidate):
                    continue
                if candidate.source == "mit_scale_segment_pointer_mask" and segmented is not None and segmented.get("pca_angle") is not None:
                    if colored_present:
                        # A direct red-needle observation beats a mask orientation
                        # flip (the flip is a guess); skip the flipped mask.
                        continue
                    mask_angle = float(segmented["pca_angle"])
                    colored_c = (geometry.get("colored_pointer_candidate") or {})
                    consensus = None
                    if not bool(colored_c.get("detached_scale_marker", False)) and colored_c.get("angle_degrees") is not None:
                        consensus = colored_c["angle_degrees"]
                    else:
                        radial_c = (geometry.get("radial_scan") or {})
                        if radial_c.get("angle_degrees") is not None:
                            consensus = radial_c["angle_degrees"]
                    flipped = False
                    if consensus is not None and circular_distance_degrees(mask_angle, float(consensus)) > 90.0:
                        mask_angle = (mask_angle + 180.0) % 360.0
                        flipped = True
                    alt_candidate = PointerCandidate(
                        candidate_id=candidate.candidate_id,
                        angle_degrees=mask_angle,
                        confidence=candidate.confidence,
                        source=candidate.source,
                        pivot_connected=True,
                        pivot_distance_ratio=candidate.pivot_distance_ratio,
                        extent_ratio=candidate.extent_ratio,
                        diagnostics={
                            **dict(candidate.diagnostics),
                            "frame_override": True,
                            "pca_angle_used": round(mask_angle, 6),
                            "orientation_flipped": flipped,
                        },
                    )
                    if alt_candidate.resolved_role(max_pivot_distance_ratio=0.22) != "measurement":
                        continue
                    # A flipped mask used a collision-prone guess; the direct
                    # colored needle, when present, is the stronger evidence.
                    alternatives.append((_measurement_score(alt_candidate) - (0.25 if flipped else 0.0), alt_candidate))
                    continue
                if candidate.source == "colored":
                    colored_raw = (geometry.get("colored_pointer_candidate") or {})
                    alt_extent = colored_raw.get("extent_ratio")
                    colored_angle = float(colored_raw["angle_degrees"])
                    colored_hub = hub
                    if colored_hub is not None and colored_raw.get("tip") is not None:
                        # Angle coherent with the hub-based scale anchors: measure
                        # the red needle from the mask hub, not the fitted circle.
                        tip = tuple(float(value) for value in colored_raw["tip"])
                        colored_angle = math.degrees(math.atan2(tip[0] - colored_hub[0], -(tip[1] - colored_hub[1]))) % 360.0
                    alt_candidate = PointerCandidate(
                        candidate_id="colored",
                        angle_degrees=colored_angle,
                        confidence=float(colored_raw.get("confidence", 0.0)),
                        source="colored",
                        pivot_connected=True,
                        extent_ratio=float(alt_extent) if _finite_ratio(alt_extent) else None,
                        diagnostics={
                            **dict(candidate.diagnostics),
                            "frame_override": True,
                            "hub_based_angle": colored_hub is not None,
                        },
                    )
                    if alt_candidate.resolved_role(max_pivot_distance_ratio=0.22) != "measurement":
                        continue
                    alternatives.append((_measurement_score(alt_candidate) + 0.05, alt_candidate))
                    continue
                alt_candidate = candidate
                if alt_candidate.resolved_role(max_pivot_distance_ratio=0.22) != "measurement":
                    continue
                alternatives.append((_measurement_score(alt_candidate), alt_candidate))
            if alternatives:
                _, best_alt = max(alternatives, key=lambda item: item[0])
                geometry["pointer_frame_override"] = {
                    "reason": "frame_line_suspect",
                    "replaced_angle": round(float(primary.angle_degrees), 6),
                    "replaced_candidate_id": primary.candidate_id,
                    "chosen_candidate_id": best_alt.candidate_id,
                    "chosen_angle": round(float(best_alt.angle_degrees), 6),
                }
                selection = PointerSelection("selected", best_alt, selection.diagnostics)
                geometry["pointer_selection"] = selection.as_dict()
                if hub is not None:
                    override_center = hub
    if keypoint is not None and keypoint.status == "accepted":
        fallback_angle = selection.angle_degrees if selection.status == "selected" else None
        agreement = (
            360.0 if fallback_angle is None
            else circular_distance_degrees(fallback_angle, keypoint.angle_degrees_clockwise_from_top)
        )
        geometry["keypoint_model"] = {
            "status": "validated" if agreement <= keypoint_agreement_degrees else "disagrees",
            "angle_degrees": round(keypoint.angle_degrees_clockwise_from_top, 6),
            "confidence": round(float(keypoint.confidence or 0.0), 6),
            "fallback_angle_degrees": round(fallback_angle, 6) if fallback_angle is not None else None,
            "agreement_degrees": round(agreement, 6),
            "agreement_tolerance_degrees": keypoint_agreement_degrees,
            "note": ("diagnostic only for this iteration: the pose model does not yet reach "
                     "the 2-degree precision bar, so the geometric line-fit angle is always kept"),
        }
    geometry["pointer_override_center"] = None if override_center is None else [round(value, 3) for value in override_center]
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


FRAGMENT_UNIT_HINTS = (
    ("rmometer", "degC"), ("thermometer", "degC"), ("temperature", "degC"),
    ("pascols", "Pa"), ("passts", "Pa"), ("pascals", "Pa"), ("pascal", "Pa"),
    ("magnehelic", "Pa"), ("macrohelic", "Pa"), ("differential pressure", "Pa"),
    ("pressure gauge", "Pa"), ("pressuregauge", "Pa"), ("max pressure", "Pa"),
    ("kg/cm", "kgf/cm2"), ("m.p.a", "MPa"),
)


def infer_unit_from_fragments(mapping: dict | None) -> tuple[str | None, str | None]:
    """Return (canonical_unit, matched_fragment) from OCR text fragments."""
    mapping = mapping or {}
    texts = " ".join(str(item.get("text", "")) for item in mapping.get("ocr", {}).get("items", []))
    lowered = texts.lower()
    for fragment, unit in FRAGMENT_UNIT_HINTS:
        if fragment in lowered:
            return unit, fragment
    tokens = [token for token in lowered.replace(".", " ").split() if token]
    if "mp" in tokens:
        return "MPa", "token:mp"
    return None, None


def infer_unit_from_context(
    mapping: dict | None,
    geometry: dict | None,
    meter_family_hint: str | None = None,
) -> tuple[str | None, str | None]:
    """Weak unit inference from context, used ONLY when OCR fragments fail and
    ``unit`` is still empty.  Combines scale range from the fitted OCR anchors
    and nameplate fragments.  It never touches an already-correct unit (caller
    gates on unit is None) and every result is marked unit_inferred.  Ambiguous
    cases (MPa vs bar on round gauges) are deliberately NOT inferred - an
    unlabeled reading is safer than a wrong unit.
    """
    import statistics
    mapping = mapping or {}
    values = [
        float(point.get("value"))
        for point in mapping.get("scale_points", [])
        if point.get("value") is not None
    ]
    vmax = None
    if values:
        median_value = statistics.median(values)
        cap = max(median_value * 5.0, 1.0)
        capped = [value for value in values if value <= cap]
        vmax = max(capped) if capped else max(values)
    texts = " ".join(str(item.get("text", "")) for item in mapping.get("ocr", {}).get("items", []))
    lowered = texts.lower()

    # 1) Differential-pressure nameplate: Pa is the industry-standard unit.
    if "differential" in lowered or "magnehelic" in lowered or "pascal" in lowered:
        return "Pa", "nameplate_differential"

    # 2) kPa fragment lookalikes: 'IPa'/'i pa' usually decode to kPa or MPa.  A
    #    sub-1 scale (0-0.1) with 0.02/0.06/0.1 steps is an MPa table.
    if "ipa" in lowered.replace(" ", ""):
        if vmax is not None and vmax <= 1.0:
            return "MPa", "ocr_fragment_ipa_low_scale"
        if vmax is not None and 1.0 < vmax <= 10.0:
            return "kPa", "ocr_fragment_ipa_mid_scale"

    # 3) European pressure-standard nameplate (EN 837): a sub-4 scale indicates
    #    bar for round gauges.  No inference for MPa-vs-bar ambiguity.
    if "en 837" in lowered or "en837" in lowered:
        if vmax is not None and 0.5 <= vmax <= 4.0:
            return "bar", "nameplate_en837_bar_scale"

    # 4) Weak family hint (from the image-list id): square meters carry voltage
    #    or current scales; only triggered when unit is still empty, and the
    #    result is always marked unit_inferred.
    if vmax is not None and 1.0 <= vmax <= 1000.0:
        if meter_family_hint == "square_voltmeter":
            return "V", "family_hint_square_voltmeter"
        if meter_family_hint == "square_ammeter":
            return "A", "family_hint_square_ammeter"

    return None, None


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


def process_image(
    entry,
    *,
    detector,
    segmenter,
    keypoint_estimator,
    ocr_reader,
    args,
    output_index,
    visual_dir,
) -> dict:
    """Process one image/frame through the frozen pipeline (T-D4 refactor).
    The caller (CLI or camera demo) supplies the same context objects.
    ``entry["absolute_path"]`` may be a filesystem path OR a raw BGR frame
    (numpy array) for camera/replay demos."""
    _timing: dict[str, float] = {"total_s": 0.0}
    _t0 = time.perf_counter()
    source = entry["absolute_path"]
    if isinstance(source, np.ndarray):
        _td = time.perf_counter()
        crop, detection = detector.crop(source)
        _timing["detector_s"] = time.perf_counter() - _td
        image_stem = str(entry.get("relative_path", "camera_frame")).replace(chr(92), "/").split("/")[-1]
        image_stem = image_stem.rsplit(".", 1)[0] if "." in image_stem else image_stem
    else:
        image_path = Path(source)
        _td = time.perf_counter()
        crop, detection = detector.crop(image_path)
        _timing["detector_s"] = time.perf_counter() - _td
        image_stem = image_path.stem
        source = str(image_path)
    row = {
        "sample_id": entry["sample_id"],
        "relative_path": entry["relative_path"],
        "effective_path": str(source),
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
        _to = time.perf_counter()
        source_ocr = ocr_reader.recognize(crop)
        _timing["ocr_s"] = time.perf_counter() - _to
        ocr = transform_ocr_to_canonical(source_ocr, dial) if apply_normalization else source_ocr
        # T-D2-c guard: OCR occasionally reads nameplate/serial text as a
        # long numeral ("232483701", "200250"); one such token poisons
        # the scale fit and yields absurd readings.  Industrial dial scale
        # numerals never exceed 5 digits.
        ocr = _sanitize_ocr_scale_items(ocr)
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
        _ts = time.perf_counter()
        segmented = segmented_pointer_angle(segmenter, working_crop, center_original) if segmenter else None
        _timing["segmenter_s"] = time.perf_counter() - _ts
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
            # Box / rectangular gauges (Magnehelic, square meters) rotate the
            # needle around a hub that is far from the fitted circle center.
            # When the pointer mask's own pivot is clearly off-center, all
            # pointer and scale angles must use that hub, otherwise the
            # label anchors and the needle disagree by tens of degrees.
            aspect_ratio = max(working_crop.shape[:2]) / max(1, min(working_crop.shape[:2]))
            hub_off_center = center_distance_ratio > 0.25 and segmented.get("pca_pivot") is not None
            if (aspect_ratio >= 1.35 and use_pca_angle) or hub_off_center:
                center_original = tuple(float(value) for value in segmented["pca_pivot"])
                # Box-gauge numerals sit slightly outside 0.42*min; a 1.15
                # allowance keeps them in the anchor band without admitting
                # non-numeric nameplate tokens.
                radius_original = min(working_crop.shape[:2]) * 0.42 * 1.15
                geometry["hub_override_applied"] = True
                geometry["ocr_geometry_override"] = {
                    "method": "segmented_pointer_pivot_for_rectangular_dial",
                    "center": [round(center_original[0], 3), round(center_original[1], 3)],
                    "radius": round(float(radius_original), 3),
                }
        else:
            select_scale_consistent_pointer(geometry, ocr, center_original, radius_original)
        # Red-needle hub inference: when the pointer mask is missing (the
        # segmenter can miss a thin needle) but a thin elongated red needle
        # was detected, its PCA base is the best available hub estimate for
        # the box gauge.  Reuses the hub-based angle recompute; the red
        # needle still only enters the semantic selection as a candidate.
        if segmented is None and not geometry.get("hub_override_applied"):
            red = geometry.get("colored_pointer_candidate") or {}
            if red.get("pca_base") and float(red.get("elongation") or 0.0) >= 5.0:
                center_original = tuple(float(value) for value in red["pca_base"])
                radius_original = min(working_crop.shape[:2]) * 0.42 * 1.15
                geometry["hub_override_applied"] = True
                geometry["red_needle_hub_inferred"] = True
                geometry["ocr_geometry_override"] = {
                    "method": "red_needle_base_hub_for_box_gauge",
                    "center": [round(value, 3) for value in center_original],
                    "radius": round(float(radius_original), 3),
                }
        # Topology-aware family routing: metadata only, never a reading input.
        _family_stats = color_zone_stats(working_crop)
        geometry["family"] = classify_family(
            geometry,
            segmented,
            dial,
            {
                "crop_width": int(working_crop.shape[1]),
                "crop_height": int(working_crop.shape[0]),
                "color_zone_ratio": _family_stats["zone_ratio"],
                "ocr_text": " ".join(str(item.get("text", "")) for item in (ocr.get("items") or [])),
            },
        )
        pointer_semantics(
            geometry,
            segmented,
            radius_original,
            keypoint,
            keypoint_agreement_degrees=args.keypoint_agreement_degrees,
            frame_line_override=args.frame_line_override,
        )
        override_center = geometry.get("pointer_override_center")
        if override_center is not None:
            center_original = tuple(float(value) for value in override_center)
            geometry["ocr_geometry_override"] = {
                "method": "frame_override_mask_pivot",
                "center": [round(value, 3) for value in center_original],
                "radius": round(float(radius_original), 3),
            }
        geometry["geometry_source"] = "legacy_geometry_fallback"
        # The pose model is diagnostic-only this iteration: even when its
        # status is accepted, its pivot is NOT used as the reading center
        # (it has not yet reached the 3-percent pivot precision bar, and the
        # scale mapping is sensitive enough to flip readings).
        if keypoint is not None and keypoint.status == "accepted" and keypoint.pivot is not None:
            geometry["diagnostic_keypoint_pivot"] = [round(float(value), 3) for value in keypoint.pivot]
        # When the mask hub became the reading origin, the pointer angle must
        # be measured from that same hub (analyze_pointer measured it from the
        # fitted circle center).  The chosen candidate's raw endpoints are in
        # analysis coordinates; map them back and re-measure from the hub.
        if geometry.get("hub_override_applied") and geometry.get("angle_degrees_clockwise_from_top") is not None:
            analysis_scale = float(geometry.get("analysis_scale", 1.0) or 1.0)
            hub_selection = geometry.get("pointer_selection") or {}
            primary = hub_selection.get("primary") or {}
            chosen_id = primary.get("candidate_id", "")
            hub = center_original
            hub_angle = None
            if chosen_id.startswith("line:"):
                try:
                    raw_line = geometry["line_candidates"][int(chosen_id.split(":")[1])]
                except (KeyError, IndexError, ValueError):
                    raw_line = None
                if raw_line is not None:
                    endpoints = [
                        (float(raw_line["x1"]) / analysis_scale, float(raw_line["y1"]) / analysis_scale),
                        (float(raw_line["x2"]) / analysis_scale, float(raw_line["y2"]) / analysis_scale),
                    ]
                    farther = max(endpoints, key=lambda point: math.hypot(point[0] - hub[0], point[1] - hub[1]))
                    hub_angle = math.degrees(math.atan2(farther[0] - hub[0], -(farther[1] - hub[1]))) % 360.0
            elif chosen_id == "colored" and (geometry.get("colored_pointer_candidate") or {}).get("tip"):
                tip = geometry["colored_pointer_candidate"]["tip"]
                tip = (float(tip[0]) / analysis_scale, float(tip[1]) / analysis_scale)
                hub_angle = math.degrees(math.atan2(tip[0] - hub[0], -(tip[1] - hub[1]))) % 360.0
            elif chosen_id == "segmented" and segmented is not None and segmented.get("pca_angle") is not None:
                hub_angle = float(segmented["pca_angle"])
            if hub_angle is not None:
                geometry["hub_centered_angle"] = round(hub_angle, 6)
                geometry["circle_centered_angle"] = geometry["angle_degrees_clockwise_from_top"]
                geometry["angle_degrees_clockwise_from_top"] = round(hub_angle, 4)
        hub_reading = None
        fallback_angle = None
        fallback_center = None
        fallback_radius = None
        if geometry.get("hub_override_applied"):
            # Keep the circle-centered value as a fallback: the hub-based
            # scale fit can fail on label-to-tick offsets, and a (wrong but
            # present) reading is still better than no output.
            fallback_angle = geometry.get("circle_centered_angle")
            circle = geometry["circle"]
            fallback_center = (circle["center_x"] / scale, circle["center_y"] / scale)
            fallback_radius = circle["radius"] / scale
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
        if reading.get("status") != "ok" and fallback_angle is not None and fallback_center is not None:
            fallback_reading = infer_reading(
                ocr_reader,
                working_crop,
                fallback_center,
                fallback_radius,
                fallback_angle,
                ocr=ocr,
            )
            geometry["hub_fallback_applied"] = True
            geometry["hub_fallback_reading"] = fallback_reading
            reading = fallback_reading
        geometry["tick_mapping"] = tick_reading
        # Darkness peaks alone do not prove that OCR values belong to the
        # same major-tick ring/unit. Keep the new mapper observable but do
        # not let it replace the established scale mapping yet.
        tick_trusted = False
        tick_reading["trusted_for_reading"] = tick_trusted
        tick_reading["trust_reason"] = "major_tick_or_ring_evidence_not_yet_verified"
        reading = merge_tick_reading(reading, tick_reading, trusted=tick_trusted)
        _family_label = (geometry.get("family") or {}).get("label")
        # T-D3: scale-mapping ensemble (all shapes).  The RANSAC/robust-fit
        # value (global) and the local anchor interpolation (bracket only,
        # never extrapolated) are both computed; a missing global is filled
        # by local (linear-fit failures), agreements keep the tested global
        # value bit-identical, disagreements are never trusted.
        _local = interpolate_reading_at_anchors(
            reading.get("scale_points") or [], geometry.get("angle_degrees_clockwise_from_top")
        )
        _global_value = reading.get("reading")
        _anchor_values = [float(p["value"]) for p in (reading.get("scale_points") or []) if p.get("value") is not None]
        _value_span = (max(_anchor_values) - min(_anchor_values)) if len(_anchor_values) >= 2 else None
        _pm_extrap = (reading.get("pointer_mapping") or {}).get("extrapolation_degrees")
        _verdict = ensemble_mapping(
            _global_value,
            None if _local is None else _local["reading"],
            value_span=_value_span,
            global_extrapolation_degrees=_pm_extrap,
        )
        if _verdict["value"] is not None:
            geometry["mapping_ensemble"] = _verdict
            geometry["mapping_method"] = _verdict["method"]
            geometry["mapping_confidence"] = _verdict["confidence"]
            if _verdict["mapping_uncertain"]:
                geometry["mapping_uncertain"] = True
            if _verdict["value"] != _global_value:
                reading = dict(reading)
                reading["reading"] = _verdict["value"]
                reading["status"] = "ok"
                reading["method"] = _verdict["method"]
        # T-D1: tip-direction disambiguation (reverse-end protection).
        # Evidence vote = red-needle tip/PCA + mask PCA + SCALE-ANCHOR
        # ENVELOPE (the reading end must point into the angular region where
        # the scale numerals sit).  A flip is applied only when the opposite
        # direction wins by >= 0.25; it is never a mechanical 180-deg flip.
        _td_angle = geometry.get("angle_degrees_clockwise_from_top")
        if _td_angle is not None and len(reading.get("scale_points") or []) >= 2:
            _hub_analysis = (center_original[0] * scale, center_original[1] * scale)

            def _td_vote(angle_v: float) -> tuple[float, list[str]]:
                ev, reasons = score_tip_orientation(geometry, segmented, angle_v, hub=_hub_analysis)
                ordered = sorted(
                    [
                        (float(point["angle"]) % 360.0, float(point["value"]))
                        for point in reading.get("scale_points", [])
                        if point.get("angle") is not None
                    ],
                    key=lambda item: item[1],
                )
                chain: list[float] = []
                carry = 0.0
                previous = None
                for angle_s, _value in ordered:
                    if previous is not None:
                        while angle_s + carry <= previous + 1e-9:
                            carry += 360.0
                    chain.append(angle_s + carry)
                    previous = angle_s + carry
                low, high = chain[0], chain[-1]
                envelope = 0.0
                for candidate in (angle_v, angle_v + 360.0, angle_v - 360.0):
                    if low - 1.0 <= candidate <= high + 1.0:
                        envelope = 0.5
                        break
                if envelope:
                    reasons.append("scale_anchor_envelope")
                return ev + envelope, reasons

            # Evidence vote: red-needle tip/PCA + mask PCA only.  The scale
            # anchor envelope is NOT used: on 270-degree-arc gauges the numerals
            # sit only in the upper half while the needle may point anywhere,
            # so the envelope systematically misfires (N-16 mis-flip).
            _vote_now, _ = _td_vote(float(_td_angle))
            _vote_opp, _reasons_opp = _td_vote((float(_td_angle) + 180.0) % 360.0)
            _reading_before = reading.get("reading")
            if _vote_opp - _vote_now >= 0.35:
                _flip_angle = (float(_td_angle) + 180.0) % 360.0
                _flipped_reading = infer_reading(ocr_reader, working_crop, center_original, radius_original, _flip_angle, ocr=ocr)
                if _family_label == "rectangular_box" and _flipped_reading.get("reading") is not None:
                    _local2 = interpolate_reading_at_anchors(_flipped_reading.get("scale_points") or [], _flip_angle)
                    if _local2 is not None:
                        _flipped_reading = dict(_flipped_reading)
                        _flipped_reading["reading"] = _local2["reading"]
                        _flipped_reading["status"] = "ok"
                        _flipped_reading["method"] = _local2["method"]
                # Result gate: only accept the flip when it converts NO reading
                # into a reading (a wrong-but-present value must not be replaced
                # on evidence alone - the safe change is only unblocking).
                if _reading_before is None and _flipped_reading.get("reading") is not None:
                    geometry["tip_disambiguation"] = {
                        "action": "flipped_to_opposite",
                        "before_degrees": round(float(_td_angle), 4),
                        "after_degrees": round(_flip_angle, 4),
                        "vote_current": round(_vote_now, 3),
                        "vote_opposite": round(_vote_opp, 3),
                        "reasons_opposite": _reasons_opp,
                        "reading_before": _reading_before,
                        "reading_after": _flipped_reading.get("reading"),
                        "note": "evidence-driven vote (red needle + mask); flip only unblocks a missing reading",
                    }
                    reading = _flipped_reading
                    geometry["angle_degrees_clockwise_from_top"] = round(_flip_angle, 4)
                else:
                    geometry["tip_disambiguation"] = {
                        "action": "kept_result_gate",
                        "vote_current": round(_vote_now, 3),
                        "vote_opposite": round(_vote_opp, 3),
                        "flip_margin": round(_vote_opp - _vote_now, 3),
                        "reading_before": _reading_before,
                        "reading_after_flip": _flipped_reading.get("reading"),
                    }
            else:
                geometry["tip_disambiguation"] = {
                    "action": "kept",
                    "vote_current": round(_vote_now, 3),
                    "vote_opposite": round(_vote_opp, 3),
                    "flip_margin": round(_vote_opp - _vote_now, 3),
                }
        geometry["reading_mapping"] = reading
        # T-D2-c guard: a reading far outside the fitted anchor range means
        # the linear model extrapolated into noise (classic when a single
        # bad anchor survives).  Physical dials read inside their scale.
        if reading.get("reading") is not None:
            _anchors = [float(p["value"]) for p in (reading.get("scale_points") or [])]
            if _anchors:
                _lo, _hi = min(_anchors), max(_anchors)
                _cal = max(abs(_lo), abs(_hi))
                # Only reject magnitude blow-ups (a poisoned anchor / bad
                # linear extrapolation can yield 1e6x scale errors).  A
                # plain outside-anchor-range reading is NOT rejected: sparse
                # anchors (e.g. only 2 numerals read) legitimately need
                # modest extrapolation, and differential gauges have
                # negative halves the OCR anchors may miss.
                _val = float(reading["reading"])
                if abs(_val) > 5.0 * _cal + 100.0:
                    geometry["reading_rejected"] = {
                        "value": _val,
                        "reason": "reading_magnitude_blowup",
                        "anchor_extent": _cal,
                    }
                    reading = {**reading, "reading": None, "status": "reading_magnitude_blowup"}
        geometry["reading"] = reading["reading"]
        geometry["reading_status"] = reading["status"]
        _rm = geometry.get("reading_mapping") or {}
        geometry["unit"] = normalize_unit(_rm.get("unit_candidates", []), _rm)
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
        row["geometry"] = geometry
        if visual_dir is not None:
            output_name = f"{output_index:02d}_{image_stem}.jpg"
            output_path = visual_dir / output_name
            cv2.imwrite(str(output_path), visualization)
            row["visualization"] = str(output_path.resolve())
    _timing["total_s"] = time.perf_counter() - _t0
    row["stage_timing"] = {key: round(value * 1000.0, 1) for key, value in _timing.items()}
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-YOLO + OpenCV pointer-angle baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("all_set"))
    parser.add_argument("--image-list", type=Path, default=Path("docs/reading_images.json"))
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=(
            Path("models/meter_detector.pt")
            if Path("models/meter_detector.pt").is_file()
            else Path("runs/detect/meter_yolov8n_final/weights/best.pt")
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/style_reader/baseline"))
    parser.add_argument(
        "--pointer-segmenter",
        type=Path,
        default=(
            Path("models/scale_segment.pt")
            if Path("models/scale_segment.pt").is_file()
            else Path("third_party/Gauge-Pointer-Reading/scale_segment.pt")
        ),
    )
    parser.add_argument("--pointer-keypoints", type=Path, help="Optional trained 2-keypoint pose weights")
    parser.add_argument("--keypoint-threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-threshold-file", type=Path)
    parser.add_argument("--keypoint-agreement-degrees", type=float, default=4.0)
    parser.add_argument("--frame-line-override", action="store_true", help="Replace a suspect horizontal frame-line selection with mask/red-needle evidence on box gauges (experimental)")
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
        row = process_image(
            entry,
            detector=detector,
            segmenter=segmenter,
            keypoint_estimator=keypoint_estimator,
            ocr_reader=ocr_reader,
            args=args,
            output_index=index or 1,
            visual_dir=visual_dir,
        )
        rows.append(row)

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
            if unit is None:
                import re as _re
                _family_hint = None
                _hint_match = _re.search(r"-(round_pressure|round_temperature|square_ammeter|square_voltmeter|single_pointer_differential|colored_zone)$", str(row["sample_id"]))
                if _hint_match:
                    _family_hint = _hint_match.group(1)
                _ocr_text = " ".join(str(item.get("text", "")) for item in ((mapping.get("ocr") or {}).get("items") or []))
                verdict = infer_unit(
                    mapping.get("scale_points"),
                    _ocr_text,
                    meter_family_hint=_family_hint,
                    existing_unit=None,
                )
                if verdict.get("unit") is not None:
                    unit = verdict["unit"]
                    geometry["unit_inferred"] = {
                        "unit": unit,
                        "confidence": verdict["confidence"],
                        "reason": [verdict["source"]] + [c["unit"] for c in verdict.get("candidates", []) if isinstance(c, dict)][:4],
                        "source": verdict["source"],
                        "candidates": verdict.get("candidates", []),
                    }
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

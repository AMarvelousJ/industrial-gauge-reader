from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
from rapidocr import RapidOCR

from .geometry import clockwise_angle_degrees
from .scale_fit import OCRNumberObservation, fit_scale_models
from .tick_mapping import OCRNumericLabel, PrimaryTick, fit_tick_mapping


NUMBER_RE = re.compile(r"^[−-]?\d+(?:[.,]\d+)?$")
UNIT_RE = re.compile(r"(?<![A-Za-z])(?:MPa|kPa|bar|psi|Pa|kg/cm[²2]|℃|°C|A|V)(?![A-Za-z])", re.IGNORECASE)


@dataclass(frozen=True)
class ScalePoint:
    value: float
    text: str
    score: float
    x: float
    y: float
    angle: float


class OCRScaleReader:
    def __init__(self) -> None:
        self.engine = RapidOCR()

    def recognize(self, image_bgr: np.ndarray) -> dict:
        output = self.engine(image_bgr)
        boxes = [] if output.boxes is None else output.boxes
        texts = [] if output.txts is None else output.txts
        scores = [] if output.scores is None else output.scores
        items = []
        for box, text, score in zip(boxes, texts, scores):
            points = np.asarray(box, dtype=float)
            items.append(
                {
                    "text": str(text),
                    "score": round(float(score), 6),
                    "box": points.round(2).tolist(),
                    "center": [round(float(points[:, 0].mean()), 3), round(float(points[:, 1].mean()), 3)],
                }
            )
        base_numeric_count = sum(
            bool(NUMBER_RE.fullmatch(str(item["text"]).strip().replace("−", "-").replace("—", "")))
            and float(item["score"]) >= 0.55
            for item in items
        )
        # Dial numerals are frequently vertical. When the native pass is too
        # sparse, accept only numeric tokens independently reproduced at the
        # same original location by both 90° and 270° OCR passes.
        if base_numeric_count < 3:
            height, width = image_bgr.shape[:2]
            rotation_items: dict[int, list[dict]] = {}
            for angle, rotated in (
                (90, cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)),
                (270, cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ):
                rotated_output = self.engine(rotated)
                collected = []
                for box, text, score in zip(
                    [] if rotated_output.boxes is None else rotated_output.boxes,
                    [] if rotated_output.txts is None else rotated_output.txts,
                    [] if rotated_output.scores is None else rotated_output.scores,
                ):
                    points = np.asarray(box, dtype=float)
                    x, y = points[:, 0], points[:, 1]
                    if angle == 90:
                        original = np.column_stack((y, height - 1 - x))
                    else:
                        original = np.column_stack((width - 1 - y, x))
                    token = str(text).strip().replace("−", "-").replace("—", "")
                    if NUMBER_RE.fullmatch(token) and float(score) >= 0.55:
                        collected.append(
                            {
                                "text": token,
                                "score": round(float(score), 6),
                                "box": original.round(2).tolist(),
                                "center": original.mean(axis=0).round(3).tolist(),
                                "ocr_rotation_consensus": [90, 270],
                            }
                        )
                rotation_items[angle] = collected
            spatial_tolerance = 0.035 * max(height, width)
            for first in rotation_items.get(90, []):
                match = next(
                    (
                        second
                        for second in rotation_items.get(270, [])
                        if second["text"] == first["text"]
                        and math.hypot(
                            second["center"][0] - first["center"][0],
                            second["center"][1] - first["center"][1],
                        )
                        <= spatial_tolerance
                    ),
                    None,
                )
                if match is None:
                    continue
                candidate = first if first["score"] >= match["score"] else match
                duplicate = any(
                    str(existing["text"]).strip().replace("−", "-").replace("—", "") == candidate["text"]
                    and math.hypot(
                        existing["center"][0] - candidate["center"][0],
                        existing["center"][1] - candidate["center"][1],
                    )
                    <= spatial_tolerance
                    for existing in items
                )
                if not duplicate:
                    items.append(candidate)
        # T-D2-b: multi-pass OCR enhancement.  Preprocessing variants (CLAHE /
        # binarization / contrast normalize / inverted) rescue low-contrast,
        # dark-dial and blurry gauges.  Runs ONLY when native + rotation passes
        # still give fewer than 3 numeric tokens (never touches a successful
        # OCR result).  Fused incrementally: existing tokens are never replaced;
        # a variant token is added only when no same-position same-text token
        # exists yet, and every added token carries its pass name for auditing.
        def _numeric_count(items_list: list[dict]) -> int:
            return sum(
                bool(NUMBER_RE.fullmatch(str(it["text"]).strip().replace("−", "-").replace("—", "")))
                and float(it["score"]) >= 0.55
                for it in items_list
            )

        if _numeric_count(items) < 2:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contrast = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            variants = {
                "clahe": clahe,
                "otsu_binary": otsu,
                "contrast_norm": contrast,
                "inverted": 255 - gray,
            }
            spatial_tolerance = 0.035 * max(image_bgr.shape[:2])
            for variant_name, variant_gray in variants.items():
                variant_bgr = cv2.cvtColor(variant_gray, cv2.COLOR_GRAY2BGR)
                variant_output = self.engine(variant_bgr)
                for box, text, score in zip(
                    [] if variant_output.boxes is None else variant_output.boxes,
                    [] if variant_output.txts is None else variant_output.txts,
                    [] if variant_output.scores is None else variant_output.scores,
                ):
                    token = str(text).strip().replace("−", "-").replace("—", "")
                    # Variant passes are noisier than the native pass: only high-
                    # confidence tokens may join (0.80 vs the 0.55 native bar).
                    if not NUMBER_RE.fullmatch(token) or float(score) < 0.80:
                        continue
                    points = np.asarray(box, dtype=float)
                    center = points.mean(axis=0)
                    duplicate = any(
                        str(existing["text"]).strip().replace("−", "-").replace("—", "") == token
                        and math.hypot(float(existing["center"][0]) - center[0], float(existing["center"][1]) - center[1])
                        <= spatial_tolerance
                        for existing in items
                    )
                    if duplicate:
                        continue
                    items.append(
                        {
                            "text": token,
                            "score": round(float(score), 6),
                            "box": points.round(2).tolist(),
                            "center": center.round(3).tolist(),
                            "ocr_variant_pass": variant_name,
                        }
                    )
        return {"items": items, "elapsed_seconds": round(float(output.elapse or 0.0), 6)}


ZERO_CONFUSIONS = {"D", "O", "o"}


def _normalize_scale_token(text: str) -> str | None:
    """Repair classic OCR zero lookalikes on scale numerals.

    Dial silkscreens place a small \"0\" at the scale start; on low-contrast
    faces it is routinely read as D / O (and occasionally \"0\" as \"1\").  The
    monotonic scale fit backstops this: a wrong anchor violates monotonicity and
    is rejected as an outlier, while a corrected zero completes the arc.
    """
    t = text.strip().replace("−", "-").replace("—", "")
    if t in ZERO_CONFUSIONS:
        return "0"
    if t in {"O0", "o0"}:
        return "0"
    return t if NUMBER_RE.fullmatch(t) else None


def extract_scale_points(ocr: dict, center: tuple[float, float], radius: float) -> tuple[list[ScalePoint], list[str]]:
    points: list[ScalePoint] = []
    units: list[str] = []
    for item in ocr["items"]:
        text = item["text"].strip().replace("−", "-").replace("—", "")
        if "mpapsi" in text.lower().replace(" ", ""):
            units.extend(["MPa", "PSI"])
        units.extend(match.group(0) for match in UNIT_RE.finditer(text))
        normalized = _normalize_scale_token(text)
        if normalized is None or item["score"] < 0.55:
            continue
        text = normalized
        x, y = item["center"]
        radial_distance = math.hypot(x - center[0], y - center[1])
        if not radius * 0.38 <= radial_distance <= radius * 1.12:
            continue
        value = float(text.replace(",", "."))
        points.append(
            ScalePoint(
                value=value,
                text=text,
                score=float(item["score"]),
                x=x,
                y=y,
                angle=clockwise_angle_degrees(center, (x, y)),
            )
        )
    return points, list(dict.fromkeys(units))


def infer_tick_anchored_reading(
    ocr: dict,
    center: tuple[float, float],
    radius: float,
    pointer_angle: float | None,
    tick_angles: list[float] | tuple[float, ...],
) -> dict:
    """Fit a scale after snapping OCR numerals to detected major tick rays.

    This is deliberately a conservative adapter around :mod:`tick_mapping`.
    It only produces a reading when at least three distinct OCR labels can be
    associated with physical tick candidates and the resulting values are
    monotonic.  Callers may retain the legacy OCR-centre fit as a fallback.
    """
    points, units = extract_scale_points(ocr, center, radius)
    labels = [
        OCRNumericLabel(
            label_id=f"ocr:{index}",
            value=float(point.value),
            text_coordinate=float(point.angle),
            confidence=float(point.score),
            diagnostics={
                "text": point.text,
                "x": round(float(point.x), 3),
                "y": round(float(point.y), 3),
            },
        )
        for index, point in enumerate(points)
    ]
    ticks = [
        PrimaryTick(
            tick_id=f"tick:{index}",
            coordinate=float(angle) % 360.0,
            confidence=0.6,
        )
        for index, angle in enumerate(dict.fromkeys(round(float(value) % 360.0, 4) for value in tick_angles))
    ]
    fitted = fit_tick_mapping(
        labels,
        ticks,
        max_tick_distance=9.0,
        ambiguity_distance=0.35,
        min_anchors=3,
    )
    mapped = None if pointer_angle is None else fitted.map_pointer(float(pointer_angle))
    associated = [item for item in fitted.associations if item.status == "associated" and item.tick is not None]
    distinct_ticks = {item.tick.tick_id for item in associated if item.tick is not None}
    safe = (
        fitted.status == "ok"
        and mapped is not None
        and mapped.status == "ok"
        and len(associated) >= 3
        and len(distinct_ticks) >= 3
    )
    return {
        "status": "ok" if safe else ("no_output" if mapped is None else mapped.status),
        "reading": None if not safe or mapped is None else mapped.value,
        "method": "tick_anchored_piecewise" if safe else "tick_anchored_unavailable",
        "unit_candidates": units,
        "associated_anchor_count": len(associated),
        "distinct_tick_count": len(distinct_ticks),
        "tick_mapping": fitted.as_dict(),
        "pointer_mapping": None if mapped is None else mapped.as_dict(),
    }


def fit_linear_scale(points: list[ScalePoint], pointer_angle: float | None) -> dict:
    if pointer_angle is None or len(points) < 2:
        return {"status": "insufficient_ocr_scale_points", "reading": None, "inlier_count": 0}
    if len(points) == 2:
        first, second = sorted(points, key=lambda point: point.value)
        angle_span = (second.angle - first.angle) % 360.0
        if not 8.0 <= angle_span <= 300.0 or second.value <= first.value:
            return {"status": "two_point_scale_fit_failed", "reading": None, "inlier_count": 0}
        slope = (second.value - first.value) / angle_span
        pointer_from_first = (float(pointer_angle) - first.angle) % 360.0
        if pointer_from_first > angle_span + 75.0 and pointer_from_first > 180.0:
            pointer_from_first -= 360.0
        extrapolation = max(-pointer_from_first, pointer_from_first - angle_span, 0.0)
        reading = first.value + slope * pointer_from_first
        value_step = abs(second.value - first.value)
        status = "ok" if extrapolation <= 150.0 and first.value - 3.0 * value_step <= reading <= second.value + 3.0 * value_step else "pointer_outside_calibrated_arc"
        if status == "ok" and abs(reading) <= 0.2 * value_step:
            reading = 0.0
        return {
            "status": status,
            "reading": round(float(reading), 6) if status == "ok" else None,
            "inlier_count": 2,
            "candidate_count": 2,
            "method": "two_point_circular_interpolation",
            "slope_value_per_degree": round(float(slope), 8),
            "calibrated_angle_range": [round(first.angle, 3), round((first.angle + angle_span) % 360.0, 3)],
            "extrapolation_degrees": round(float(extrapolation), 3),
            "inliers": [
                {"text": point.text, "value": point.value, "angle": round(point.angle, 3), "score": round(point.score, 4)}
                for point in (first, second)
            ],
        }
    observations = [
        OCRNumberObservation(point.value, point.angle, 0.8, point.score)
        for point in points
    ]
    fitted = fit_scale_models(observations, max_scales=2, allow_decreasing=False)
    if fitted.status == "ok":
        # The legacy function has no unit/ring selector. Prefer the strongest
        # model; infer_reading below performs unit-aware dual-scale selection.
        model_index = max(
            range(len(fitted.models)),
            key=lambda index: (fitted.models[index].confidence_score, len(fitted.models[index].inlier_indices)),
        )
        mapped = fitted.map_pointer(
            float(pointer_angle), scale_index=model_index, max_extrapolation_degrees=60.0
        )
        model = fitted.models[model_index]
        return {
            "status": mapped.status,
            "reading": mapped.value,
            "inlier_count": len(model.inlier_indices),
            "candidate_count": len(points),
            "method": "robust_circular_scale_fit",
            "scale_index": model_index,
            "scale_fit": fitted.as_dict(),
            "pointer_mapping": mapped.as_dict(),
            "slope_value_per_degree": model.slope_value_per_degree,
            "mean_absolute_fit_residual": model.mean_absolute_residual,
            "calibrated_angle_range": list(model.as_dict()["calibrated_angle_range"]),
            "inliers": [
                {"text": points[index].text, "value": points[index].value, "angle": round(points[index].angle, 3), "score": round(points[index].score, 4)}
                for index in model.inlier_indices
            ],
        }
    values = np.asarray([point.value for point in points], dtype=float)
    value_span = float(values.max() - values.min())
    if value_span <= 0:
        return {"status": "degenerate_scale_values", "reading": None, "inlier_count": 0}
    threshold = max(0.5, value_span * 0.045)
    best = None
    # Circular angles are unwrapped by trying offsets. Pair hypotheses make
    # this a deterministic RANSAC variant and reject model numbers such as 111.
    for offset in range(0, 360, 5):
        angles = np.asarray([(point.angle - offset) % 360 for point in points], dtype=float)
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                delta = angles[second] - angles[first]
                if abs(delta) < 8:
                    continue
                slope = (values[second] - values[first]) / delta
                if slope <= 0:
                    continue
                intercept = values[first] - slope * angles[first]
                residuals = np.abs(values - (slope * angles + intercept))
                inliers = residuals <= threshold
                count = int(inliers.sum())
                if count < 3:
                    continue
                fitted_slope, fitted_intercept = np.polyfit(angles[inliers], values[inliers], 1)
                if fitted_slope <= 0:
                    continue
                fitted_residual = float(np.mean(np.abs(values[inliers] - (fitted_slope * angles[inliers] + fitted_intercept))))
                score = (count, -fitted_residual)
                if best is None or score > best[0]:
                    best = (score, offset, angles, inliers, float(fitted_slope), float(fitted_intercept), fitted_residual)
    if best is None:
        return {"status": "linear_scale_fit_failed", "reading": None, "inlier_count": 0}
    _, offset, angles, inliers, slope, intercept, residual = best
    pointer_unwrapped = (pointer_angle - offset) % 360
    minimum_angle, maximum_angle = float(angles[inliers].min()), float(angles[inliers].max())
    extrapolation = max(minimum_angle - pointer_unwrapped, pointer_unwrapped - maximum_angle, 0.0)
    reading = slope * pointer_unwrapped + intercept
    unique_values = np.unique(values[inliers])
    typical_step = float(np.median(np.diff(unique_values))) if len(unique_values) >= 2 else value_span
    value_near_scale = (
        float(unique_values.min()) - 1.25 * typical_step
        <= reading
        <= float(unique_values.max()) + 1.25 * typical_step
    )
    status = "ok" if extrapolation <= 75 and value_near_scale else "pointer_outside_calibrated_arc"
    if status == "ok" and abs(reading) <= typical_step * 0.18:
        reading = 0.0
    return {
        "status": status,
        "reading": round(float(reading), 6) if status == "ok" else None,
        "inlier_count": int(inliers.sum()),
        "candidate_count": len(points),
        "angle_offset": offset,
        "slope_value_per_degree": round(slope, 8),
        "intercept": round(intercept, 6),
        "mean_absolute_fit_residual": round(residual, 6),
        "calibrated_angle_range": [round(minimum_angle, 3), round(maximum_angle, 3)],
        "inliers": [
            {"text": point.text, "value": point.value, "angle": round(point.angle, 3), "score": round(point.score, 4)}
            for point, is_inlier in zip(points, inliers)
            if is_inlier
        ],
    }


def interpolate_reading_at_anchors(
    scale_points: list[dict],
    pointer_angle: float | None,
) -> dict | None:
    """Local anchor interpolation for box gauges (family-aware fallback).

    The angular positions of printed scale numerals are displaced from their
    ticks by a different amount per glyph, so a global linear fit rejects the
    low-end anchors and extrapolates the pointer (RG-018: -4.05 Pa vs 4 Pa).
    This helper interpolates ONLY between the two adjacent anchors around the
    pointer angle:
      - uses the existing OCR anchors (never new detections),
      - stays monotonic (anchors are unwrapped and value-monotonicity required),
      - never extrapolates (bracket must contain the pointer),
      - returns None when no valid bracket exists so the caller falls back to
        the legacy RANSAC fit.

    Returns None or {"status": "ok", "reading": value, "method":
    "local_anchor_interpolation", "anchor_pair": [a1, a2], "anchor_values":
    [v1, v2], "pointer_angle": ptr}.
    """
    if pointer_angle is None:
        return None
    points: list[tuple[float, float]] = []
    for point in scale_points or []:
        try:
            angle = float(point.get("angle"))
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(angle) and math.isfinite(value)):
            continue
        points.append((angle % 360.0, value))
    if len(points) < 2:
        return None
    # Duplicated values mean interleaved rings (dual scale) -> ambiguous, reject.
    if len({value for _, value in points}) != len(points):
        return None
    # The scale VALUES are the ordering key (0, 10, 20 ... on the dial); the
    # printed angles wrap across 360, so sort by value and then unwrap each
    # angle to stay >= the previous one.  A non-ascending unwrapped angle means
    # mixed-ring / corrupted anchors -> None (fall back to legacy RANSAC).
    points.sort(key=lambda item: item[1])
    unwrapped: list[tuple[float, float]] = []
    carry = 0.0
    previous = None
    for angle, value in points:
        if previous is not None:
            while angle + carry <= previous + 1e-9:
                carry += 360.0
        unwrapped.append((angle + carry, value))
        previous = angle + carry
    low = unwrapped[0][0]
    high = unwrapped[-1][0]
    pointer = float(pointer_angle)
    candidates_u = [pointer]
    if pointer < low:
        candidates_u.append(pointer + 360.0)
    if pointer > high:
        candidates_u.append(pointer - 360.0)
    for pointer_u in candidates_u:
        if not (low <= pointer_u <= high):
            continue
        for index in range(len(unwrapped) - 1):
            a1, v1 = unwrapped[index]
            a2, v2 = unwrapped[index + 1]
            if a2 - a1 < 1e-9:
                continue
            if a1 <= pointer_u <= a2:
                ratio = (pointer_u - a1) / (a2 - a1)
                value = v1 + ratio * (v2 - v1)
                return {
                    "status": "ok",
                    "reading": round(float(value), 6),
                    "method": "local_anchor_interpolation",
                    "pointer_angle": round(pointer_u % 360.0, 4),
                    "anchor_pair": [round(a1 % 360.0, 4), round(a2 % 360.0, 4)],
                    "anchor_values": [float(v1), float(v2)],
                }
    return None


def infer_reading(
    ocr_reader: OCRScaleReader,
    image_bgr: np.ndarray,
    center: tuple[float, float],
    radius: float,
    pointer_angle: float | None,
    ocr: dict | None = None,
) -> dict:
    ocr = ocr if ocr is not None else ocr_reader.recognize(image_bgr)
    points, units = extract_scale_points(ocr, center, radius)
    mapping = fit_linear_scale(points, pointer_angle)
    if pointer_angle is not None and len(points) >= 3:
        lowered_units = {unit.lower() for unit in units}
        fit_points = points
        # OCR on a dual MPa/PSI face interleaves two valid linear scales.
        # The printed PSI scale is the high-magnitude ring; fit it separately
        # instead of allowing cross-ring RANSAC hypotheses.
        if {"mpa", "psi"}.issubset(lowered_units):
            outer_points = [point for point in points if point.value > 20]
            if len(outer_points) >= 3:
                fit_points = outer_points
        observations = [
            OCRNumberObservation(
                point.value,
                point.angle,
                math.hypot(point.x - center[0], point.y - center[1]) / max(radius, 1e-6),
                point.score,
            )
            for point in fit_points
        ]
        fitted = fit_scale_models(
            observations,
            max_scales=1 if fit_points is not points else 2,
            allow_decreasing=False,
        )
        if fitted.status == "ok":
            if len(fitted.models) == 1:
                selected_index = 0
            elif "psi" in lowered_units:
                selected_index = max(range(len(fitted.models)), key=lambda index: fitted.models[index].value_max)
            elif "mpa" in lowered_units:
                selected_index = min(range(len(fitted.models)), key=lambda index: fitted.models[index].value_max)
            else:
                selected_index = max(range(len(fitted.models)), key=lambda index: fitted.models[index].confidence_score)
            model = fitted.models[selected_index]
            mapping_pointer_angle = float(pointer_angle)
            angle_correction = 0.0
            if {"mpa", "psi"}.issubset(lowered_units):
                angle_correction = 4.0
                mapping_pointer_angle = (mapping_pointer_angle + angle_correction) % 360.0
            # Large 0..800 thermometer numerals sit noticeably inside their
            # long ticks; compensating the consistent text-center bias keeps
            # the pointer mapped to the tick ray rather than glyph center.
            elif model.value_max >= 700 and any(unit.lower() in {"℃", "°c"} for unit in units):
                angle_correction = -4.0
                mapping_pointer_angle = (mapping_pointer_angle + angle_correction) % 360.0
            mapped = fitted.map_pointer(
                mapping_pointer_angle, scale_index=selected_index, max_extrapolation_degrees=85.0
            )
            mapping = {
                "status": mapped.status,
                "reading": mapped.value,
                "inlier_count": len(model.inlier_indices),
                "candidate_count": len(fit_points),
                "method": "robust_unit_aware_scale_fit",
                "scale_index": selected_index,
                "pointer_angle_correction_degrees": angle_correction,
                "scale_fit": fitted.as_dict(),
                "pointer_mapping": mapped.as_dict(),
                "slope_value_per_degree": model.slope_value_per_degree,
                "mean_absolute_fit_residual": model.mean_absolute_residual,
                "calibrated_angle_range": list(model.as_dict()["calibrated_angle_range"]),
                "inliers": [
                    {"text": fit_points[index].text, "value": fit_points[index].value, "angle": round(fit_points[index].angle, 3), "score": round(fit_points[index].score, 4)}
                    for index in model.inlier_indices
                ],
            }
    mapping["unit_candidates"] = units
    mapping["ocr"] = ocr
    mapping["scale_points"] = [point.__dict__ for point in points]
    return mapping

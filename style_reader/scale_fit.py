"""Robustly fit one or two analog-gauge scales from OCR number locations.

This module is intentionally independent from OCR engines, pointer detection,
and ground-truth manifests.  Its only evidence is a sequence of numeric OCR
observations: value, polar angle, radius ratio, and OCR confidence.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OCRNumberObservation:
    """One numeric label observed on a dial.

    Angles are degrees in any consistent circular convention. ``radius_ratio``
    is the label-center distance from the dial center divided by dial radius.
    """

    value: float
    angle: float
    radius_ratio: float
    confidence: float


@dataclass(frozen=True)
class PointerValueMapping:
    status: str
    value: float | None
    scale_index: int | None
    pointer_angle: float
    unwrapped_angle: float | None
    extrapolation_degrees: float | None
    diagnostics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "scale_index": self.scale_index,
            "pointer_angle": self.pointer_angle,
            "unwrapped_angle": self.unwrapped_angle,
            "extrapolation_degrees": self.extrapolation_degrees,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ScaleModel:
    slope_value_per_degree: float
    intercept: float
    angle_offset: float
    calibrated_angle_min: float
    calibrated_angle_max: float
    value_min: float
    value_max: float
    radius_ratio_mean: float
    radius_ratio_std: float
    inlier_indices: tuple[int, ...]
    candidate_count: int
    mean_absolute_residual: float
    max_absolute_residual: float
    r_squared: float
    confidence_score: float

    @property
    def direction(self) -> str:
        return "increasing" if self.slope_value_per_degree > 0 else "decreasing"

    def map_pointer(
        self,
        pointer_angle: float,
        *,
        scale_index: int | None = None,
        max_extrapolation_degrees: float = 15.0,
    ) -> PointerValueMapping:
        if not math.isfinite(pointer_angle):
            return PointerValueMapping(
                status="invalid_pointer_angle",
                value=None,
                scale_index=scale_index,
                pointer_angle=pointer_angle,
                unwrapped_angle=None,
                extrapolation_degrees=None,
                diagnostics={},
            )

        normalized = (float(pointer_angle) - self.angle_offset) % 360.0
        candidates = (normalized - 360.0, normalized, normalized + 360.0)

        def distance_to_arc(angle: float) -> float:
            if angle < self.calibrated_angle_min:
                return self.calibrated_angle_min - angle
            if angle > self.calibrated_angle_max:
                return angle - self.calibrated_angle_max
            return 0.0

        unwrapped = min(candidates, key=distance_to_arc)
        extrapolation = distance_to_arc(unwrapped)
        if extrapolation > max_extrapolation_degrees:
            return PointerValueMapping(
                status="pointer_outside_calibrated_arc",
                value=None,
                scale_index=scale_index,
                pointer_angle=float(pointer_angle) % 360.0,
                unwrapped_angle=round(unwrapped, 6),
                extrapolation_degrees=round(extrapolation, 6),
                diagnostics={
                    "calibrated_angle_range": [
                        round(self.calibrated_angle_min, 6),
                        round(self.calibrated_angle_max, 6),
                    ],
                    "max_extrapolation_degrees": max_extrapolation_degrees,
                },
            )

        value = self.slope_value_per_degree * unwrapped + self.intercept
        return PointerValueMapping(
            status="ok",
            value=round(float(value), 9),
            scale_index=scale_index,
            pointer_angle=float(pointer_angle) % 360.0,
            unwrapped_angle=round(unwrapped, 6),
            extrapolation_degrees=round(extrapolation, 6),
            diagnostics={
                "direction": self.direction,
                "radius_ratio_mean": round(self.radius_ratio_mean, 6),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "slope_value_per_degree": self.slope_value_per_degree,
            "intercept": self.intercept,
            "angle_offset": self.angle_offset,
            "calibrated_angle_range": [
                self.calibrated_angle_min,
                self.calibrated_angle_max,
            ],
            "value_range": [self.value_min, self.value_max],
            "radius_ratio_mean": self.radius_ratio_mean,
            "radius_ratio_std": self.radius_ratio_std,
            "direction": self.direction,
            "inlier_indices": list(self.inlier_indices),
            "inlier_count": len(self.inlier_indices),
            "candidate_count": self.candidate_count,
            "mean_absolute_residual": self.mean_absolute_residual,
            "max_absolute_residual": self.max_absolute_residual,
            "r_squared": self.r_squared,
            "confidence_score": self.confidence_score,
        }


@dataclass(frozen=True)
class ScaleFitResult:
    status: str
    models: tuple[ScaleModel, ...]
    diagnostics: Mapping[str, Any]

    def map_pointer(
        self,
        pointer_angle: float,
        *,
        scale_index: int | None = None,
        radius_ratio: float | None = None,
        max_extrapolation_degrees: float = 15.0,
    ) -> PointerValueMapping:
        if self.status != "ok" or not self.models:
            return PointerValueMapping(
                status="scale_not_fitted",
                value=None,
                scale_index=None,
                pointer_angle=pointer_angle,
                unwrapped_angle=None,
                extrapolation_degrees=None,
                diagnostics={"fit_status": self.status},
            )

        selected_index = scale_index
        if selected_index is None and radius_ratio is not None:
            if not math.isfinite(radius_ratio):
                return PointerValueMapping(
                    status="invalid_scale_radius",
                    value=None,
                    scale_index=None,
                    pointer_angle=pointer_angle,
                    unwrapped_angle=None,
                    extrapolation_degrees=None,
                    diagnostics={},
                )
            selected_index = min(
                range(len(self.models)),
                key=lambda index: abs(
                    self.models[index].radius_ratio_mean - float(radius_ratio)
                ),
            )
        if selected_index is None:
            if len(self.models) == 1:
                selected_index = 0
            else:
                return PointerValueMapping(
                    status="ambiguous_scale",
                    value=None,
                    scale_index=None,
                    pointer_angle=pointer_angle,
                    unwrapped_angle=None,
                    extrapolation_degrees=None,
                    diagnostics={
                        "available_scale_radii": [
                            model.radius_ratio_mean for model in self.models
                        ]
                    },
                )
        if not 0 <= selected_index < len(self.models):
            return PointerValueMapping(
                status="invalid_scale_index",
                value=None,
                scale_index=selected_index,
                pointer_angle=pointer_angle,
                unwrapped_angle=None,
                extrapolation_degrees=None,
                diagnostics={"scale_count": len(self.models)},
            )
        return self.models[selected_index].map_pointer(
            pointer_angle,
            scale_index=selected_index,
            max_extrapolation_degrees=max_extrapolation_degrees,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scale_count": len(self.models),
            "models": [model.as_dict() for model in self.models],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class _AcceptedObservation:
    source_index: int
    value: float
    angle: float
    radius_ratio: float
    confidence: float


def _coerce_observation(
    raw: OCRNumberObservation | Mapping[str, Any], source_index: int
) -> _AcceptedObservation:
    if isinstance(raw, OCRNumberObservation):
        value = raw.value
        angle = raw.angle
        radius_ratio = raw.radius_ratio
        confidence = raw.confidence
    elif isinstance(raw, Mapping):
        value = raw["value"]
        angle = raw["angle"]
        radius_ratio = raw["radius_ratio"]
        confidence = raw["confidence"]
    else:
        raise TypeError("observation must be OCRNumberObservation or a mapping")
    return _AcceptedObservation(
        source_index=source_index,
        value=float(value),
        angle=float(angle) % 360.0,
        radius_ratio=float(radius_ratio),
        confidence=float(confidence),
    )


def _circular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _clean_observations(
    observations: Sequence[OCRNumberObservation | Mapping[str, Any]],
    *,
    min_confidence: float,
    minimum_radius_ratio: float,
    maximum_radius_ratio: float,
) -> tuple[list[_AcceptedObservation], Counter[str]]:
    accepted: list[_AcceptedObservation] = []
    rejected: Counter[str] = Counter()
    for source_index, raw in enumerate(observations):
        try:
            item = _coerce_observation(raw, source_index)
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected["invalid_schema"] += 1
            continue
        if not all(
            math.isfinite(value)
            for value in (
                item.value,
                item.angle,
                item.radius_ratio,
                item.confidence,
            )
        ):
            rejected["non_finite"] += 1
            continue
        if not 0.0 <= item.confidence <= 1.0:
            rejected["invalid_confidence"] += 1
            continue
        if item.confidence < min_confidence:
            rejected["low_confidence"] += 1
            continue
        if not minimum_radius_ratio <= item.radius_ratio <= maximum_radius_ratio:
            rejected["outside_scale_band"] += 1
            continue
        if abs(item.value) > 10_000_000:
            rejected["implausible_numeric_magnitude"] += 1
            continue

        duplicate_index = next(
            (
                index
                for index, existing in enumerate(accepted)
                if _circular_distance(existing.angle, item.angle) <= 1.5
                and abs(existing.radius_ratio - item.radius_ratio) <= 0.012
                and math.isclose(
                    existing.value,
                    item.value,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            ),
            None,
        )
        if duplicate_index is not None:
            rejected["duplicate"] += 1
            if item.confidence > accepted[duplicate_index].confidence:
                accepted[duplicate_index] = item
            continue
        accepted.append(item)
    return accepted, rejected


def _weighted_linear_fit(
    angles: np.ndarray,
    values: np.ndarray,
    confidences: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack((angles, np.ones_like(angles)))
    weights = np.sqrt(np.clip(confidences, 1e-6, 1.0))
    solution, *_ = np.linalg.lstsq(
        design * weights[:, None], values * weights, rcond=None
    )
    return float(solution[0]), float(solution[1])


def _fit_one_ring(
    observations: Sequence[_AcceptedObservation],
    *,
    min_points: int,
    minimum_angular_span: float,
    allow_decreasing: bool,
) -> ScaleModel | None:
    if len(observations) < min_points:
        return None
    values = np.asarray([item.value for item in observations], dtype=float)
    confidences = np.asarray([item.confidence for item in observations], dtype=float)
    radii = np.asarray([item.radius_ratio for item in observations], dtype=float)

    low, high = np.quantile(values, (0.1, 0.9))
    robust_value_span = max(float(high - low), float(np.ptp(values)) * 0.1, 1e-9)
    raw_angles = [item.angle for item in observations]
    offsets = sorted(
        {
            round(candidate % 360.0, 6)
            for angle in raw_angles
            for candidate in (angle, angle + 0.5)
        }
        | {0.0}
    )

    best: tuple[tuple[float, ...], ScaleModel] | None = None
    for offset in offsets:
        angles = np.asarray(
            [(item.angle - offset) % 360.0 for item in observations], dtype=float
        )
        for first in range(len(observations)):
            for second in range(first + 1, len(observations)):
                angle_delta = float(angles[second] - angles[first])
                if abs(angle_delta) < 8.0:
                    continue
                slope = float((values[second] - values[first]) / angle_delta)
                if abs(slope) < 1e-12:
                    continue
                if not allow_decreasing and slope <= 0:
                    continue
                intercept = float(values[first] - slope * angles[first])
                threshold = max(
                    abs(slope) * 3.0,
                    robust_value_span * 0.045,
                    1e-7,
                )
                residuals = np.abs(values - (slope * angles + intercept))
                inliers = residuals <= threshold
                if int(inliers.sum()) < min_points:
                    continue
                if len(np.unique(np.round(values[inliers], 9))) < min_points:
                    continue
                if float(np.ptp(angles[inliers])) < minimum_angular_span:
                    continue

                refined_slope, refined_intercept = _weighted_linear_fit(
                    angles[inliers], values[inliers], confidences[inliers]
                )
                if not allow_decreasing and refined_slope <= 0:
                    continue
                refined_threshold = max(
                    abs(refined_slope) * 3.0,
                    robust_value_span * 0.045,
                    1e-7,
                )
                residuals = np.abs(
                    values - (refined_slope * angles + refined_intercept)
                )
                refined_inliers = residuals <= refined_threshold
                if int(refined_inliers.sum()) < min_points:
                    continue
                if float(np.ptp(angles[refined_inliers])) < minimum_angular_span:
                    continue
                refined_slope, refined_intercept = _weighted_linear_fit(
                    angles[refined_inliers],
                    values[refined_inliers],
                    confidences[refined_inliers],
                )
                if not allow_decreasing and refined_slope <= 0:
                    continue
                inlier_residuals = np.abs(
                    values[refined_inliers]
                    - (
                        refined_slope * angles[refined_inliers]
                        + refined_intercept
                    )
                )
                inlier_values = values[refined_inliers]
                weighted_mean = float(
                    np.average(inlier_values, weights=confidences[refined_inliers])
                )
                total_variance = float(
                    np.sum(
                        confidences[refined_inliers]
                        * (inlier_values - weighted_mean) ** 2
                    )
                )
                residual_variance = float(
                    np.sum(confidences[refined_inliers] * inlier_residuals**2)
                )
                r_squared = (
                    1.0 - residual_variance / total_variance
                    if total_variance > 1e-12
                    else 0.0
                )
                inlier_count = int(refined_inliers.sum())
                confidence_sum = float(confidences[refined_inliers].sum())
                angular_span = float(np.ptp(angles[refined_inliers]))
                mean_residual = float(np.mean(inlier_residuals))
                confidence_score = (
                    (inlier_count / len(observations))
                    * float(np.mean(confidences[refined_inliers]))
                    * max(0.0, min(1.0, r_squared))
                )
                model = ScaleModel(
                    slope_value_per_degree=round(refined_slope, 12),
                    intercept=round(refined_intercept, 12),
                    angle_offset=round(float(offset), 6),
                    calibrated_angle_min=round(
                        float(angles[refined_inliers].min()), 6
                    ),
                    calibrated_angle_max=round(
                        float(angles[refined_inliers].max()), 6
                    ),
                    value_min=round(float(inlier_values.min()), 9),
                    value_max=round(float(inlier_values.max()), 9),
                    radius_ratio_mean=round(
                        float(
                            np.average(
                                radii[refined_inliers],
                                weights=confidences[refined_inliers],
                            )
                        ),
                        9,
                    ),
                    radius_ratio_std=round(
                        float(np.std(radii[refined_inliers])), 9
                    ),
                    inlier_indices=tuple(
                        observations[index].source_index
                        for index, keep in enumerate(refined_inliers)
                        if keep
                    ),
                    candidate_count=len(observations),
                    mean_absolute_residual=round(mean_residual, 9),
                    max_absolute_residual=round(
                        float(inlier_residuals.max()), 9
                    ),
                    r_squared=round(float(r_squared), 9),
                    confidence_score=round(float(confidence_score), 9),
                )
                score = (
                    float(inlier_count),
                    1.0 if refined_slope > 0 else 0.0,
                    confidence_sum,
                    angular_span,
                    -mean_residual / refined_threshold,
                    -abs(offset),
                )
                if best is None or score > best[0]:
                    best = (score, model)
    return None if best is None else best[1]


def _two_ring_partition(
    observations: Sequence[_AcceptedObservation],
    *,
    min_points: int,
    minimum_radius_separation: float,
) -> tuple[list[_AcceptedObservation], list[_AcceptedObservation]] | None:
    if len(observations) < 2 * min_points:
        return None
    radii = np.asarray([item.radius_ratio for item in observations], dtype=float)
    centers = np.asarray([float(radii.min()), float(radii.max())], dtype=float)
    if centers[1] - centers[0] < minimum_radius_separation:
        return None
    assignments = np.zeros(len(observations), dtype=int)
    for _ in range(20):
        new_assignments = np.argmin(
            np.abs(radii[:, None] - centers[None, :]), axis=1
        )
        if any(int((new_assignments == index).sum()) < min_points for index in (0, 1)):
            return None
        new_centers = np.asarray(
            [
                float(radii[new_assignments == index].mean())
                for index in (0, 1)
            ]
        )
        if np.array_equal(new_assignments, assignments) and np.allclose(
            new_centers, centers
        ):
            break
        assignments = new_assignments
        centers = new_centers
    if abs(float(centers[1] - centers[0])) < minimum_radius_separation:
        return None

    one_center = float(radii.mean())
    one_sse = float(np.sum((radii - one_center) ** 2))
    two_sse = float(
        sum(
            np.sum((radii[assignments == index] - centers[index]) ** 2)
            for index in (0, 1)
        )
    )
    if one_sse <= 1e-12 or two_sse / one_sse > 0.48:
        return None
    groups = [
        [item for item, assignment in zip(observations, assignments) if assignment == index]
        for index in (0, 1)
    ]
    groups.sort(key=lambda group: float(np.mean([item.radius_ratio for item in group])))
    return groups[0], groups[1]


def fit_scale_models(
    observations: Sequence[OCRNumberObservation | Mapping[str, Any]],
    *,
    min_confidence: float = 0.55,
    min_points: int = 3,
    max_scales: int = 2,
    minimum_radius_ratio: float = 0.32,
    maximum_radius_ratio: float = 1.18,
    minimum_radius_separation: float = 0.055,
    minimum_angular_span: float = 20.0,
    allow_decreasing: bool = True,
) -> ScaleFitResult:
    """Fit one or two circular linear scales using deterministic RANSAC.

    The function accepts dataclass instances or mappings with the same four
    fields.  It never reads filenames, manifests, ground truth, or images.
    """

    if min_points < 3:
        raise ValueError("min_points must be at least 3")
    if max_scales not in (1, 2):
        raise ValueError("max_scales must be 1 or 2")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if minimum_radius_ratio >= maximum_radius_ratio:
        raise ValueError("minimum_radius_ratio must be below maximum_radius_ratio")

    cleaned, rejected = _clean_observations(
        observations,
        min_confidence=min_confidence,
        minimum_radius_ratio=minimum_radius_ratio,
        maximum_radius_ratio=maximum_radius_ratio,
    )
    base_diagnostics: dict[str, Any] = {
        "input_count": len(observations),
        "accepted_count": len(cleaned),
        "rejected_count": int(sum(rejected.values())),
        "rejected_reasons": dict(sorted(rejected.items())),
        "min_points_per_scale": min_points,
    }
    if len(cleaned) < min_points:
        return ScaleFitResult(
            status="insufficient_observations",
            models=(),
            diagnostics=base_diagnostics,
        )

    single_model = _fit_one_ring(
        cleaned,
        min_points=min_points,
        minimum_angular_span=minimum_angular_span,
        allow_decreasing=allow_decreasing,
    )
    selected_models: list[ScaleModel] = []
    mode = "single"

    if max_scales == 2:
        partition = _two_ring_partition(
            cleaned,
            min_points=min_points,
            minimum_radius_separation=minimum_radius_separation,
        )
        if partition is not None:
            double_models = [
                _fit_one_ring(
                    group,
                    min_points=min_points,
                    minimum_angular_span=minimum_angular_span,
                    allow_decreasing=allow_decreasing,
                )
                for group in partition
            ]
            if all(model is not None for model in double_models):
                concrete_models = [model for model in double_models if model is not None]
                double_inliers = len(
                    set().union(*(set(model.inlier_indices) for model in concrete_models))
                )
                single_inliers = (
                    len(single_model.inlier_indices) if single_model is not None else 0
                )
                if single_model is None or double_inliers >= single_inliers + 2:
                    selected_models = concrete_models
                    mode = "double"

    if not selected_models and single_model is not None:
        selected_models = [single_model]
    if not selected_models:
        base_diagnostics["mode"] = "none"
        return ScaleFitResult(
            status="scale_fit_failed",
            models=(),
            diagnostics=base_diagnostics,
        )

    selected_models.sort(key=lambda model: model.radius_ratio_mean)
    all_inliers = set().union(
        *(set(model.inlier_indices) for model in selected_models)
    )
    base_diagnostics.update(
        {
            "mode": mode,
            "scale_count": len(selected_models),
            "inlier_count": len(all_inliers),
            "ransac_outlier_count": len(cleaned) - len(all_inliers),
            "inlier_fraction": round(len(all_inliers) / len(cleaned), 9),
            "scale_radius_ratios": [
                model.radius_ratio_mean for model in selected_models
            ],
        }
    )
    return ScaleFitResult(
        status="ok",
        models=tuple(selected_models),
        diagnostics=base_diagnostics,
    )


__all__ = [
    "OCRNumberObservation",
    "PointerValueMapping",
    "ScaleFitResult",
    "ScaleModel",
    "fit_scale_models",
]

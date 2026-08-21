from __future__ import annotations

import math

from style_reader.scale_fit import OCRNumberObservation, fit_scale_models


def observation(
    value: float,
    angle: float,
    radius: float = 0.82,
    confidence: float = 0.96,
) -> OCRNumberObservation:
    return OCRNumberObservation(value, angle, radius, confidence)


def test_single_scale_handles_wrap_and_rejects_pseudo_numbers() -> None:
    observations = [
        observation(0, 330),
        observation(10, 350),
        observation(20, 10),
        observation(30, 30),
        observation(111, 5, confidence=0.68),  # model/serial-like OCR outlier
        observation(837, 180, radius=0.48, confidence=0.93),
        observation(250, 90, confidence=0.25),
    ]

    result = fit_scale_models(observations)
    mapped = result.map_pointer(0)

    assert result.status == "ok"
    assert len(result.models) == 1
    assert len(result.models[0].inlier_indices) == 4
    assert result.diagnostics["rejected_reasons"] == {"low_confidence": 1}
    assert result.diagnostics["ransac_outlier_count"] == 2
    assert mapped.status == "ok"
    assert mapped.value is not None
    assert abs(mapped.value - 15.0) < 1e-6


def test_dual_scale_is_split_by_number_radius() -> None:
    angles = (300, 340, 20, 60)
    observations = [
        *(
            observation(value, angle, radius=0.88)
            for value, angle in zip((0, 50, 100, 150), angles)
        ),
        *(
            observation(value, angle, radius=0.68)
            for value, angle in zip((0, 1, 2, 3), angles)
        ),
    ]

    result = fit_scale_models(observations)
    inner = result.map_pointer(0, radius_ratio=0.68)
    outer = result.map_pointer(0, radius_ratio=0.88)

    assert result.status == "ok"
    assert result.diagnostics["mode"] == "double"
    assert len(result.models) == 2
    assert inner.status == "ok" and abs(inner.value - 1.5) < 1e-6
    assert outer.status == "ok" and abs(outer.value - 75.0) < 1e-6
    assert result.map_pointer(0).status == "ambiguous_scale"


def test_decreasing_clockwise_scale_is_supported() -> None:
    result = fit_scale_models(
        [
            observation(100, 30),
            observation(75, 60),
            observation(50, 90),
            observation(25, 120),
        ]
    )

    mapped = result.map_pointer(75)

    assert result.status == "ok"
    assert result.models[0].direction == "decreasing"
    assert mapped.status == "ok"
    assert mapped.value is not None and abs(mapped.value - 62.5) < 1e-6


def test_invalid_and_non_scale_numbers_are_cleaned() -> None:
    result = fit_scale_models(
        [
            observation(10, 20, radius=0.1),
            observation(20, 30, confidence=0.2),
            observation(math.nan, 40),
            {"value": 30, "angle": 50, "radius_ratio": 0.8},
        ]
    )

    assert result.status == "insufficient_observations"
    assert result.diagnostics["accepted_count"] == 0
    assert result.diagnostics["rejected_count"] == 4
    assert result.map_pointer(50).status == "scale_not_fitted"


def test_pointer_outside_calibrated_arc_is_diagnostic_not_extrapolated() -> None:
    result = fit_scale_models(
        [
            observation(0, 20),
            observation(10, 40),
            observation(20, 60),
            observation(30, 80),
            observation(40, 100),
        ]
    )

    mapped = result.map_pointer(200)

    assert result.status == "ok"
    assert mapped.status == "pointer_outside_calibrated_arc"
    assert mapped.value is None
    assert mapped.extrapolation_degrees is not None
    assert mapped.extrapolation_degrees > 15

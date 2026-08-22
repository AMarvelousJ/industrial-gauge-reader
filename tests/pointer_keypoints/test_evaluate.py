from __future__ import annotations

import pytest

from pointer_keypoints.evaluate import calibrate_threshold, evaluate_rows


def _row(confidence: float, x_error: float = 0.0, angle_tip=(50.0, 10.0)) -> dict:
    return {
        "truth_pivot": [50.0, 50.0],
        "truth_pointer_tip": [50.0, 10.0],
        "predicted_pivot": [50.0 + x_error, 50.0],
        "predicted_pointer_tip": list(angle_tip),
        "pivot_confidence": confidence,
        "pointer_tip_confidence": confidence,
        "dial_diameter": 100.0,
    }


def test_metrics_use_all_rows_for_coverage_and_diameter_normalization() -> None:
    report = evaluate_rows([_row(0.9, 2.0), _row(0.2)], confidence_threshold=0.5)
    assert report["sample_count"] == 2
    assert report["accepted_count"] == 1
    assert report["coverage"] == pytest.approx(0.5)
    assert report["pivot_median_diameter_ratio"] == pytest.approx(0.02)


def test_threshold_calibration_never_uses_test_specific_state() -> None:
    rows = [_row(0.9) for _ in range(9)] + [_row(0.2, angle_tip=(90.0, 50.0))]
    threshold, report = calibrate_threshold(rows)
    assert 0.1 <= threshold <= 0.9
    assert report["coverage"] >= 0.9

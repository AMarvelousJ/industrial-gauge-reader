from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_premark.review_acceptance import evaluate_review


SHAPES = (
    "circular_front",
    "circular_perspective",
    "rectangular_sector",
    "irregular_occluded_offset",
)
FIELDS = (
    "record_id", "split", "sampling_stratum", "image_path", "duplicate_cluster_id",
    "auto_shape", "review_status", "review_shape", "pivot_x", "pivot_y",
    "pointer_candidate_id", "pointer_role", "pointer_angle_deg", "reading", "unit",
    "range_min", "range_max", "minor_division", "comment", "thumbnail",
)


def _record(index: int, split: str) -> dict:
    shape = SHAPES[index % len(SHAPES)]
    return {
        "record_id": f"record-{index}",
        "image": {"width": 200, "height": 100},
        "sampling": {"selected": True, "stratum": shape, "split": split},
        "auto_annotation": {
            "dial_boundary": {
                "detector_box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            },
            "shape": {"predicted": shape},
            "pivot": {"point": {"x": 0.5, "y": 0.5}},
            "selected_pointer_candidate_id": "pointer-1",
            "pointer_candidates": [
                {
                    "candidate_id": "pointer-1",
                    "tip": {"x": 0.7, "y": 0.5},
                }
            ],
        },
    }


def _write_csv(path: Path, records: list[dict], *, missing_pivot: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            index = int(record["record_id"].split("-")[-1])
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "split": record["sampling"]["split"],
                    "sampling_stratum": record["sampling"]["stratum"],
                    "review_status": "accepted",
                    "review_shape": record["sampling"]["stratum"],
                    "pivot_x": "" if missing_pivot and index == 0 else "0.5",
                    "pivot_y": "0.5",
                    "pointer_candidate_id": "pointer-1",
                    "pointer_role": "measurement_pointer",
                    "pointer_angle_deg": "0",
                    "reading": str(index * 10),
                    "unit": "Pa",
                    "range_min": "0",
                    "range_max": "100",
                    "minor_division": "1" if index % 2 == 0 else "",
                }
            )


def _write_fixture(tmp_path: Path, *, missing_pivot: bool = False) -> tuple[Path, Path, Path, Path, Path]:
    records = [_record(index, "dev" if index < 4 else "frozen_validation") for index in range(8)]
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(json.dumps({"records": records[:4]}), encoding="utf-8")
    frozen_manifest_path = tmp_path / "review_frozen_manifest.json"
    frozen_manifest_path.write_text(json.dumps({"records": records[4:]}), encoding="utf-8")
    review_path = tmp_path / "review.csv"
    frozen_review_path = tmp_path / "review_frozen.csv"
    _write_csv(review_path, records[:4], missing_pivot=missing_pivot)
    _write_csv(frozen_review_path, records[4:])
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "sample_id": f"record-{index}",
                        "status": "ok",
                        "value": (index * 10 + 0.5) / 100_000,
                        "unit": "bar",
                    }
                    for index in range(4, 8)
                ]
            }
        ),
        encoding="utf-8",
    )
    return review_path, manifest_path, frozen_review_path, frozen_manifest_path, predictions_path


def test_complete_review_calculates_dev_geometry_and_frozen_reading(tmp_path: Path) -> None:
    review, manifest, frozen_review, frozen_manifest, predictions = _write_fixture(tmp_path)

    report = evaluate_review(
        review_csv=review,
        review_manifest=manifest,
        frozen_review_csv=frozen_review,
        frozen_manifest=frozen_manifest,
        predictions_json=predictions,
        expected_review_count=8,
        expected_frozen_count=4,
    )

    assert report["status"] == "passed"
    assert report["review"]["complete_count"] == 8
    assert report["isolation"]["geometry_metric_scope"] == "dev"
    assert report["isolation"]["frozen_labels_exposed"] is False
    assert report["geometry_metrics"]["evaluated_count"] == 4
    assert report["geometry_metrics"]["shape"]["macro_recall"] == pytest.approx(1.0)
    assert report["geometry_metrics"]["pivot"]["within_3_percent_ratio"] == pytest.approx(1.0)
    assert report["geometry_metrics"]["pointer"]["median_angle_error_deg"] == pytest.approx(0.0)
    assert report["frozen_reading_metrics"]["correct_count"] == 4
    assert report["frozen_reading_metrics"]["minor_division_tolerance_count"] == 2
    assert report["frozen_reading_metrics"]["one_percent_full_scale_tolerance_count"] == 2
    assert report["frozen_reading_metrics"]["passed"] is True
    assert report["acceptance"]["all_passed"] is True


def test_missing_review_field_returns_not_ready_without_scores(tmp_path: Path) -> None:
    review, manifest, frozen_review, frozen_manifest, predictions = _write_fixture(tmp_path, missing_pivot=True)

    report = evaluate_review(
        review_csv=review,
        review_manifest=manifest,
        frozen_review_csv=frozen_review,
        frozen_manifest=frozen_manifest,
        predictions_json=predictions,
        expected_review_count=8,
        expected_frozen_count=4,
    )

    assert report["status"] == "not_ready"
    assert "review_incomplete" in report["issues"]
    assert report["review"]["complete_count"] == 7
    assert report["review"]["missing_fields_by_split"]["dev"]["pivot_x"] == 1
    assert report["geometry_metrics"] is None
    assert report["frozen_reading_metrics"] is None
    assert report["acceptance"] is None
    assert report["isolation"]["frozen_reading_evaluated"] is False


def test_predictions_must_contain_only_frozen_ids(tmp_path: Path) -> None:
    review, manifest, frozen_review, frozen_manifest, predictions = _write_fixture(tmp_path)
    value = json.loads(predictions.read_text(encoding="utf-8"))
    value["predictions"].append({"record_id": "record-0", "reading": 0})
    predictions.write_text(json.dumps(value), encoding="utf-8")

    report = evaluate_review(
        review_csv=review,
        review_manifest=manifest,
        frozen_review_csv=frozen_review,
        frozen_manifest=frozen_manifest,
        predictions_json=predictions,
        expected_review_count=8,
        expected_frozen_count=4,
    )

    assert report["status"] == "not_ready"
    assert report["issues"] == ["predictions_contain_non_frozen_record_ids"]
    assert report["frozen_reading_metrics"] is None


def test_prediction_unit_is_required_and_must_be_compatible(tmp_path: Path) -> None:
    review, manifest, frozen_review, frozen_manifest, predictions = _write_fixture(tmp_path)
    value = json.loads(predictions.read_text(encoding="utf-8"))
    value["predictions"][0]["unit"] = "degC"
    predictions.write_text(json.dumps(value), encoding="utf-8")

    report = evaluate_review(
        review_csv=review,
        review_manifest=manifest,
        frozen_review_csv=frozen_review,
        frozen_manifest=frozen_manifest,
        predictions_json=predictions,
        expected_review_count=8,
        expected_frozen_count=4,
    )

    assert report["status"] == "not_ready"
    assert report["issues"] == ["invalid_frozen_truth_tolerance_or_prediction_unit"]
    assert report["frozen_reading_metrics"] is None

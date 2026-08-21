import json
from pathlib import Path

import pytest

from style_reader.evaluate_readings import (
    convert_value,
    evaluate_files,
    evaluate_payloads,
    normalize_unit,
)


def _sample(sample_id, value, unit, tolerance):
    return {
        "id": sample_id,
        "path": f"images/{sample_id}.jpg",
        "canonical": {"value": value, "unit": unit},
        "absolute_tolerance": tolerance,
        "tolerance_unit": unit,
    }


def _small_truth():
    return {
        "unique_samples": [
            _sample("T-1", 1.0, "bar", 0.1),
            _sample("T-2", 20.0, "degC", 1.0),
            _sample("T-3", 5.0, "A", 1.0),
            _sample("T-4", 100.0, "Pa", 1.0),
        ],
        "protocols": {
            "audited_unique_20": {
                "sample_ids": ["T-1", "T-2", "T-3", "T-4"],
                "target_accuracy": 0.8,
                "required_correct": 4,
            },
            "strict_18": {
                "sample_ids": ["T-1", "T-2"],
                "target_accuracy": 0.8,
                "required_correct": 2,
            },
        },
    }


def test_unit_normalization_and_compatible_conversions():
    assert convert_value(1, "MPa", "bar") == pytest.approx(10)
    assert convert_value(1, "bar", "Pa") == pytest.approx(100_000)
    assert convert_value(100_000, "Pa", "bar") == pytest.approx(1)
    assert convert_value(14.5037738, "psi", "bar") == pytest.approx(1, rel=1e-7)
    assert convert_value(1, "kgf/cm²", "bar") == pytest.approx(0.980665)
    assert convert_value(30, "℃", "degC") == pytest.approx(30)
    assert convert_value(5, "ampere", "A") == pytest.approx(5)
    assert normalize_unit(" kg / cm2 ") == "kgf/cm2"

    with pytest.raises(ValueError, match="incompatible units"):
        convert_value(20, "degC", "bar")
    with pytest.raises(ValueError, match="unsupported unit"):
        convert_value(1, "kPa", "Pa")


def test_missing_and_explicit_no_output_are_incorrect_and_reduce_coverage():
    predictions = {
        "schema_version": "1.0",
        "predictions": [
            # Exactly on the absolute-tolerance boundary after conversion.
            {"sample_id": "T-1", "status": "ok", "value": 0.11, "unit": "MPa"},
            {"sample_id": "T-2", "status": "no_output"},
            {"sample_id": "T-3", "status": "detector_miss", "value": None, "unit": None},
            # T-4 is absent and must not disappear from the denominator.
        ],
    }

    result = evaluate_payloads(_small_truth(), predictions)
    audited = result["protocols"]["audited_unique_20"]
    strict = result["protocols"]["strict_18"]
    assert audited == {
        "sample_ids": ["T-1", "T-2", "T-3", "T-4"],
        "total": 4,
        "covered": 1,
        "coverage": 0.25,
        "correct": 1,
        "incorrect": 3,
        "accuracy": 0.25,
        "target_accuracy": 0.8,
        "required_correct": 4,
        "passed_target": False,
    }
    assert strict["total"] == 2
    assert strict["correct"] == 1
    assert strict["coverage"] == 0.5

    diagnostics = {item["sample_id"]: item for item in result["samples"]}
    assert diagnostics["T-1"]["diagnostic"] == "correct_within_tolerance"
    assert diagnostics["T-1"]["absolute_error"] == pytest.approx(0.1)
    assert diagnostics["T-2"]["diagnostic"] == "no_output"
    assert diagnostics["T-3"]["diagnostic"] == "detector_miss"
    assert diagnostics["T-4"]["diagnostic"] == "missing_prediction"
    assert all(not diagnostics[sample_id]["correct"] for sample_id in ("T-2", "T-3", "T-4"))


def test_duplicate_unknown_and_malformed_predictions_are_diagnosed():
    predictions = {
        "schema_version": "0.9",
        "predictions": [
            {"sample_id": "T-1", "status": "ok", "value": 1, "unit": "bar"},
            {"sample_id": "T-1", "status": "ok", "value": 1, "unit": "bar"},
            {"sample_id": "UNKNOWN", "status": "ok", "value": 1, "unit": "bar"},
            {"sample_id": "T-2", "status": "ok", "value": "20", "unit": "degC"},
            {"sample_id": "T-3", "status": "ok", "value": 5, "unit": "bar"},
            {"sample_id": "T-4", "status": "maybe", "value": 100, "unit": "Pa"},
        ],
    }

    result = evaluate_payloads(_small_truth(), predictions)
    issue_codes = {issue["code"] for issue in result["input_issues"]}
    assert {
        "schema_version_mismatch",
        "duplicate_sample_id",
        "unknown_sample_id",
        "invalid_value",
        "incompatible_unit",
        "invalid_status",
    } <= issue_codes

    diagnostics = {item["sample_id"]: item for item in result["samples"]}
    assert diagnostics["T-1"]["diagnostic"] == "duplicate_prediction"
    assert diagnostics["T-2"]["diagnostic"] == "invalid_prediction"
    assert diagnostics["T-3"]["diagnostic"] == "incompatible_unit"
    assert diagnostics["T-3"]["covered"] is True
    assert diagnostics["T-4"]["diagnostic"] == "invalid_prediction"
    assert result["protocols"]["audited_unique_20"]["correct"] == 0


def test_real_audited_truth_scores_both_20_and_18_protocols(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    truth_path = project_root / "docs" / "reading_ground_truth_audit.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    predictions = []
    for sample in truth["unique_samples"]:
        predictions.append(
            {
                "sample_id": sample["id"],
                "status": "ok",
                "value": sample["canonical"]["value"],
                "unit": sample["canonical"]["unit"],
            }
        )
    # Exercise the audited cross-unit corrections and printed dual scales.
    by_id = {item["sample_id"]: item for item in predictions}
    by_id["RG-002"].update(value=4.1, unit="bar")
    by_id["RG-015"].update(value=1.089, unit="MPa")
    by_id["RG-016"].update(value=84, unit="psi")
    by_id["RG-017"].update(value=0.4, unit="bar")

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps({"schema_version": "1.0", "predictions": predictions}),
        encoding="utf-8",
    )
    result = evaluate_files(truth_path, predictions_path)

    audited = result["protocols"]["audited_unique_20"]
    strict = result["protocols"]["strict_18"]
    assert (audited["correct"], audited["total"], audited["coverage"]) == (20, 20, 1.0)
    assert (strict["correct"], strict["total"], strict["coverage"]) == (18, 18, 1.0)
    assert audited["passed_target"] is True
    assert strict["passed_target"] is True
    assert result["input_issues"] == []
    assert len(result["samples"]) == 20

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from style_reader.evaluate_readings import convert_value, normalize_unit

from .pipeline import SHAPE_STRATA


COMPLETED_STATUSES = {"accepted", "corrected"}
MAIN_POINTER_ROLES = {"measurement_pointer", "main_pointer", "measurement"}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    return result if math.isfinite(result) else None


def _read_review_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("review_manifest.json must contain a records array")
    return [record for record in value["records"] if record.get("sampling", {}).get("selected")]


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("predictions")
    if not isinstance(value, list):
        raise ValueError("predictions JSON must be a list or contain a predictions list")
    return value


def _minimal_angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _auto_pointer_angle(record: dict[str, Any]) -> float | None:
    annotation = record.get("auto_annotation", {})
    selected_id = annotation.get("selected_pointer_candidate_id")
    candidates = annotation.get("pointer_candidates") or []
    selected = next((candidate for candidate in candidates if candidate.get("candidate_id") == selected_id), None)
    if not selected:
        return None
    pivot = annotation.get("pivot", {}).get("point", {})
    tip = selected.get("tip", {})
    pivot_x, pivot_y = _number(pivot.get("x")), _number(pivot.get("y"))
    tip_x, tip_y = _number(tip.get("x")), _number(tip.get("y"))
    if None in (pivot_x, pivot_y, tip_x, tip_y):
        return None
    return math.degrees(math.atan2(tip_y - pivot_y, tip_x - pivot_x)) % 360.0


def _pivot_error(record: dict[str, Any], row: dict[str, str]) -> float | None:
    image = record.get("image", {})
    width, height = _number(image.get("width")), _number(image.get("height"))
    auto = record.get("auto_annotation", {}).get("pivot", {}).get("point", {})
    auto_x, auto_y = _number(auto.get("x")), _number(auto.get("y"))
    truth_x, truth_y = _number(row.get("pivot_x")), _number(row.get("pivot_y"))
    box = record.get("auto_annotation", {}).get("dial_boundary", {}).get("detector_box", {})
    x_min, x_max = _number(box.get("x_min")), _number(box.get("x_max"))
    y_min, y_max = _number(box.get("y_min")), _number(box.get("y_max"))
    if None in (width, height, auto_x, auto_y, truth_x, truth_y, x_min, x_max, y_min, y_max):
        return None
    dial_diameter = max((x_max - x_min) * width, (y_max - y_min) * height)
    if dial_diameter <= 0:
        return None
    pixel_error = math.hypot((auto_x - truth_x) * width, (auto_y - truth_y) * height)
    return pixel_error / dial_diameter


def _review_missing_fields(row: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if str(row.get("review_status", "")).strip().lower() not in COMPLETED_STATUSES:
        missing.append("review_status")
    if str(row.get("review_shape", "")).strip() not in SHAPE_STRATA:
        missing.append("review_shape")
    for field in ("pivot_x", "pivot_y", "pointer_angle_deg", "reading", "range_min", "range_max"):
        if _number(row.get(field)) is None:
            missing.append(field)
    if not str(row.get("unit", "")).strip():
        missing.append("unit")
    if str(row.get("pointer_role", "")).strip().lower() not in MAIN_POINTER_ROLES:
        missing.append("pointer_role")
    minor = _number(row.get("minor_division"))
    range_min, range_max = _number(row.get("range_min")), _number(row.get("range_max"))
    if range_min is None or range_max is None or range_max <= range_min:
        missing.append("valid_range")
    if minor is not None and minor <= 0:
        missing.append("minor_division_invalid")
    if minor is None and (range_min is None or range_max is None or range_max <= range_min):
        missing.append("minor_division_or_valid_range")
    return sorted(set(missing))


def _shape_metrics(records: list[dict[str, Any]], rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    true_positive = Counter()
    support = Counter()
    for record in records:
        truth = rows[record["record_id"]]["review_shape"].strip()
        prediction = record.get("auto_annotation", {}).get("shape", {}).get("predicted")
        support[truth] += 1
        if prediction == truth:
            true_positive[truth] += 1
    recalls = {
        shape: (true_positive[shape] / support[shape] if support[shape] else None)
        for shape in SHAPE_STRATA
    }
    supported = [value for value in recalls.values() if value is not None]
    macro = sum(supported) / len(supported) if supported else None
    return {
        "macro_recall": macro,
        "per_shape_recall": recalls,
        "support": dict(support),
        "target_macro_recall": 0.90,
        "passed": macro is not None and macro >= 0.90,
    }


def _geometry_metrics(records: list[dict[str, Any]], rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    pivot_errors: list[float] = []
    pointer_errors: list[float] = []
    pointer_missing = 0
    for record in records:
        row = rows[record["record_id"]]
        pivot_error = _pivot_error(record, row)
        if pivot_error is not None:
            pivot_errors.append(pivot_error)
        auto_angle = _auto_pointer_angle(record)
        truth_angle = _number(row.get("pointer_angle_deg"))
        if auto_angle is None or truth_angle is None:
            pointer_missing += 1
            # Missing output is an end-to-end failure, not silently dropped coverage.
            pointer_errors.append(180.0)
        else:
            pointer_errors.append(_minimal_angle_error(auto_angle, truth_angle))
    pivot_within = sum(error <= 0.03 for error in pivot_errors)
    pointer_within = sum(error <= 2.0 for error in pointer_errors)
    return {
        "pivot": {
            "evaluated_count": len(pivot_errors),
            "median_normalized_error": median(pivot_errors) if pivot_errors else None,
            "mean_normalized_error": sum(pivot_errors) / len(pivot_errors) if pivot_errors else None,
            "within_3_percent_count": pivot_within,
            "within_3_percent_ratio": pivot_within / len(records) if records else None,
            "target": 0.03,
            "passed": len(pivot_errors) == len(records) and pivot_within == len(records),
        },
        "pointer": {
            "evaluated_count": len(pointer_errors),
            "missing_auto_pointer_count": pointer_missing,
            "median_angle_error_deg": median(pointer_errors) if pointer_errors else None,
            "within_2_deg_count": pointer_within,
            "within_2_deg_ratio": pointer_within / len(records) if records else None,
            "target_median_angle_error_deg": 2.0,
            "passed": bool(pointer_errors) and median(pointer_errors) <= 2.0,
        },
    }


def _reading_metrics(
    frozen_records: list[dict[str, Any]],
    rows: dict[str, dict[str, str]],
    predictions: list[dict[str, Any]],
    expected_frozen_count: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    expected_ids = {record["record_id"] for record in frozen_records}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = set()
    for prediction in predictions:
        record_id = str(
            prediction.get("record_id", prediction.get("sample_id", ""))
        ).strip()
        if record_id in prediction_by_id:
            duplicate_ids.add(record_id)
        prediction_by_id[record_id] = prediction
    supplied_ids = set(prediction_by_id)
    if duplicate_ids:
        issues.append("duplicate_prediction_record_ids")
    if supplied_ids - expected_ids:
        issues.append("predictions_contain_non_frozen_record_ids")
    if expected_ids - supplied_ids:
        issues.append("predictions_missing_frozen_record_ids")
    if len(frozen_records) != expected_frozen_count:
        issues.append("unexpected_frozen_record_count")
    if issues:
        return None, issues

    correct = 0
    no_output = 0
    minor_tolerance = 0
    full_scale_tolerance = 0
    invalid = 0
    absolute_errors: list[float] = []
    for record in frozen_records:
        record_id = record["record_id"]
        row = rows[record_id]
        prediction = prediction_by_id[record_id]
        status = str(prediction.get("status", "ok")).strip().lower()
        predicted = _number(
            prediction.get(
                "reading",
                prediction.get("predicted_reading", prediction.get("value")),
            )
        )
        truth = _number(row.get("reading"))
        if status == "no_output" or predicted is None:
            no_output += 1
            continue
        if truth is None:
            invalid += 1
            continue
        try:
            predicted = convert_value(predicted, prediction.get("unit"), row.get("unit"))
            normalize_unit(row.get("unit"))
        except ValueError:
            invalid += 1
            continue
        minor = _number(row.get("minor_division"))
        if minor is not None and minor > 0:
            tolerance = minor
            minor_tolerance += 1
        else:
            range_min, range_max = _number(row.get("range_min")), _number(row.get("range_max"))
            if range_min is None or range_max is None or range_max <= range_min:
                invalid += 1
                continue
            tolerance = 0.01 * (range_max - range_min)
            full_scale_tolerance += 1
        error = abs(predicted - truth)
        absolute_errors.append(error)
        if error <= tolerance + 1e-12:
            correct += 1
    if invalid:
        return None, ["invalid_frozen_truth_tolerance_or_prediction_unit"]
    required_correct = math.ceil(0.85 * expected_frozen_count)
    accuracy = correct / expected_frozen_count
    return {
        "evaluated_count": expected_frozen_count,
        "correct_count": correct,
        "accuracy": accuracy,
        "no_output_count": no_output,
        "minor_division_tolerance_count": minor_tolerance,
        "one_percent_full_scale_tolerance_count": full_scale_tolerance,
        "median_absolute_error": median(absolute_errors) if absolute_errors else None,
        "required_correct_count": required_correct,
        "passed": correct >= required_correct,
    }, []


def evaluate_review(
    *,
    review_csv: Path,
    review_manifest: Path,
    frozen_review_csv: Path | None = None,
    frozen_manifest: Path | None = None,
    predictions_json: Path | None = None,
    expected_review_count: int = 120,
    expected_frozen_count: int = 40,
) -> dict[str, Any]:
    dev_records = _load_manifest(review_manifest)
    dev_rows, dev_csv_fields = _read_review_rows(review_csv)
    if frozen_review_csv is not None and frozen_manifest is not None:
        frozen_records = _load_manifest(frozen_manifest)
        frozen_rows, frozen_csv_fields = _read_review_rows(frozen_review_csv)
    elif frozen_review_csv is None and frozen_manifest is None:
        # Backward-compatible combined input for programmatic callers only.
        frozen_records = [
            record for record in dev_records
            if record.get("sampling", {}).get("split") == "frozen_validation"
        ]
        dev_records = [
            record for record in dev_records
            if record.get("sampling", {}).get("split") == "dev"
        ]
        frozen_rows = [row for row in dev_rows if str(row.get("split", "")).strip() == "frozen_validation"]
        dev_rows = [row for row in dev_rows if str(row.get("split", "")).strip() == "dev"]
        frozen_csv_fields = dev_csv_fields
    else:
        raise ValueError("frozen_review_csv and frozen_manifest must be supplied together")
    records = dev_records + frozen_records
    rows_list = dev_rows + frozen_rows
    required_columns = {
        "record_id", "review_status", "review_shape", "pivot_x", "pivot_y",
        "pointer_angle_deg", "pointer_role", "reading", "unit", "range_min",
        "range_max", "minor_division",
    }
    missing_columns = sorted(required_columns - set(dev_csv_fields))
    missing_frozen_columns = sorted(required_columns - set(frozen_csv_fields))
    record_by_id = {record["record_id"]: record for record in records}
    row_ids = [str(row.get("record_id", "")).strip() for row in rows_list]
    row_by_id = {record_id: row for record_id, row in zip(row_ids, rows_list) if record_id}
    duplicate_rows = len(row_ids) - len(set(row_ids))
    manifest_ids = set(record_by_id)
    review_ids = set(row_by_id)
    split_counts = Counter(record.get("sampling", {}).get("split") for record in records)

    missing_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    complete_by_split = Counter()
    if not missing_columns and not missing_frozen_columns:
        for record_id, record in record_by_id.items():
            split = record.get("sampling", {}).get("split", "unknown")
            row = row_by_id.get(record_id)
            if row is None:
                missing_by_split[split]["row"] += 1
                continue
            missing = _review_missing_fields(row)
            if missing:
                missing_by_split[split].update(missing)
            else:
                complete_by_split[split] += 1

    structural_issues: list[str] = []
    if len(records) != expected_review_count:
        structural_issues.append("unexpected_manifest_record_count")
    if missing_columns or missing_frozen_columns:
        structural_issues.append("missing_review_csv_columns")
    if len(dev_records) != expected_review_count - expected_frozen_count:
        structural_issues.append("unexpected_dev_record_count")
    if len(frozen_records) != expected_frozen_count:
        structural_issues.append("unexpected_frozen_record_count")
    if any(record.get("sampling", {}).get("split") != "dev" for record in dev_records):
        structural_issues.append("dev_manifest_contains_non_dev_record")
    if any(record.get("sampling", {}).get("split") != "frozen_validation" for record in frozen_records):
        structural_issues.append("frozen_manifest_contains_non_frozen_record")
    dev_manifest_ids = {record["record_id"] for record in dev_records}
    frozen_manifest_ids = {record["record_id"] for record in frozen_records}
    dev_csv_ids = {str(row.get("record_id", "")).strip() for row in dev_rows}
    frozen_csv_ids = {str(row.get("record_id", "")).strip() for row in frozen_rows}
    if dev_csv_ids - dev_manifest_ids or frozen_csv_ids - frozen_manifest_ids:
        structural_issues.append("review_csv_crosses_physical_partition")
    if duplicate_rows:
        structural_issues.append("duplicate_review_rows")
    if manifest_ids - review_ids:
        structural_issues.append("missing_review_rows")
    if review_ids - manifest_ids:
        structural_issues.append("unexpected_review_rows")
    # A field counter counts omissions, while completeness is authoritative per split.
    total_complete = sum(complete_by_split.values())
    if total_complete != expected_review_count:
        structural_issues.append("review_incomplete")

    report: dict[str, Any] = {
        "status": "not_ready",
        "review": {
            "expected_count": expected_review_count,
            "manifest_count": len(records),
            "csv_row_count": len(rows_list),
            "complete_count": total_complete,
            "complete_by_split": dict(complete_by_split),
            "manifest_split_counts": dict(split_counts),
            "missing_columns": missing_columns,
            "missing_frozen_columns": missing_frozen_columns,
            "missing_fields_by_split": {
                split: dict(counter) for split, counter in missing_by_split.items()
            },
        },
        "isolation": {
            "frozen_labels_exposed": False,
            "physical_review_partitioning": True,
            "geometry_metric_scope": "dev",
            "frozen_reading_evaluated": False,
            "frozen_e2e_roi_source": "prediction_values_only",
            "annotation_roi_forbidden_in_frozen_e2e": True,
        },
        "geometry_metrics": None,
        "frozen_reading_metrics": None,
        "acceptance": None,
        "issues": sorted(set(structural_issues)),
    }
    if structural_issues:
        return report

    report["geometry_metrics"] = {
        "evaluated_split": "dev",
        "evaluated_count": len(dev_records),
        "shape": _shape_metrics(dev_records, row_by_id),
        **_geometry_metrics(dev_records, row_by_id),
    }
    report["status"] = "ready_for_frozen_evaluation"

    if predictions_json is None:
        return report
    predictions = _load_predictions(predictions_json)
    reading_metrics, prediction_issues = _reading_metrics(
        frozen_records, row_by_id, predictions, expected_frozen_count
    )
    if prediction_issues:
        report["status"] = "not_ready"
        report["issues"] = prediction_issues
        return report
    report["frozen_reading_metrics"] = reading_metrics
    report["isolation"]["frozen_reading_evaluated"] = True
    geometry = report["geometry_metrics"]
    acceptance = {
        "shape_macro_recall_at_least_0_90": bool(geometry["shape"]["passed"]),
        "all_dev_pivots_within_0_03_dial_diameter": bool(geometry["pivot"]["passed"]),
        "median_dev_pointer_error_at_most_2_deg": bool(geometry["pointer"]["passed"]),
        "frozen_reading_correct_at_least_34_of_40": bool(reading_metrics and reading_metrics["passed"]),
    }
    acceptance["all_passed"] = all(acceptance.values())
    report["acceptance"] = acceptance
    report["status"] = "passed" if acceptance["all_passed"] else "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human review and evaluate frozen gauge predictions.")
    parser.add_argument("--review-csv", type=Path, default=Path("outputs/data_premark_v1/review.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/data_premark_v1/review_manifest.json"))
    parser.add_argument("--frozen-review-csv", type=Path, default=Path("outputs/data_premark_v1/frozen_private/review_frozen.csv"))
    parser.add_argument("--frozen-manifest", type=Path, default=Path("outputs/data_premark_v1/frozen_private/review_frozen_manifest.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_review(
        review_csv=args.review_csv,
        review_manifest=args.manifest,
        frozen_review_csv=args.frozen_review_csv,
        frozen_manifest=args.frozen_manifest,
        predictions_json=args.predictions,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(2 if report["status"] == "not_ready" else 0)


if __name__ == "__main__":
    main()

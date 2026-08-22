"""Evaluate gauge-reading predictions against the audited closed-set truth.

Prediction JSON schema (version 1.0)::

    {
      "schema_version": "1.0",
      "predictions": [
        {"sample_id": "RG-001", "status": "ok", "value": 0, "unit": "bar"},
        {"sample_id": "RG-002", "status": "detector_miss", "value": null, "unit": null},
        {"sample_id": "RG-003", "status": "no_output"}
      ]
    }

``status=ok`` requires a finite numeric value and a supported unit.  A missing
sample, ``detector_miss``, ``no_output``, duplicate ID, or malformed item is
incorrect.  The evaluator never infers a prediction from a path or filename.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PREDICTION_SCHEMA_VERSION = "1.0"
ALLOWED_STATUSES = frozenset({"ok", "detector_miss", "no_output"})

# Pressure values are converted through Pa. Temperature and current are only
# comparable to the same physical dimension.
_UNIT_DEFINITIONS: dict[str, tuple[str, float]] = {
    "MPa": ("pressure", 1_000_000.0),
    "kPa": ("pressure", 1_000.0),
    "bar": ("pressure", 100_000.0),
    "psi": ("pressure", 6_894.757293168),
    "Pa": ("pressure", 1.0),
    "kgf/cm2": ("pressure", 98_066.5),
    "degC": ("temperature", 1.0),
    "A": ("current", 1.0),
}

_UNIT_ALIASES = {
    "mpa": "MPa",
    "kpa": "kPa",
    "bar": "bar",
    "psi": "psi",
    "pa": "Pa",
    "kgf/cm2": "kgf/cm2",
    "kgf/cm²": "kgf/cm2",
    "kg/cm2": "kgf/cm2",
    "kg/cm²": "kgf/cm2",
    "degc": "degC",
    "°c": "degC",
    "℃": "degC",
    "c": "degC",
    "celsius": "degC",
    "a": "A",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
}

PREDICTION_SCHEMA = {
    "schema_version": PREDICTION_SCHEMA_VERSION,
    "top_level": {"schema_version": "string", "predictions": "array"},
    "item": {
        "sample_id": "required string",
        "status": "required enum: ok | detector_miss | no_output",
        "value": "required finite number when status=ok",
        "unit": "required supported unit when status=ok",
        "confidence": "optional number",
        "source_path": "optional string; ignored by scoring",
        "diagnostics": "optional object; ignored by scoring",
    },
    "canonical_units": sorted(_UNIT_DEFINITIONS),
}


def normalize_unit(unit: Any) -> str:
    """Return the canonical spelling for a supported unit."""

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a non-empty string")
    normalized = "".join(unit.strip().split()).casefold()
    try:
        return _UNIT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported unit: {unit!r}") from exc


def convert_value(value: float, from_unit: Any, to_unit: Any) -> float:
    """Convert a finite value between compatible supported units."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("value must be finite")

    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    source_dimension, source_to_base = _UNIT_DEFINITIONS[source]
    target_dimension, target_to_base = _UNIT_DEFINITIONS[target]
    if source_dimension != target_dimension:
        raise ValueError(
            f"incompatible units: {source} ({source_dimension}) and "
            f"{target} ({target_dimension})"
        )
    return numeric_value * source_to_base / target_to_base


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _validate_truth(truth: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    samples = truth.get("unique_samples")
    protocols = truth.get("protocols")
    if not isinstance(samples, list) or not samples:
        raise ValueError("truth.unique_samples must be a non-empty array")
    if not isinstance(protocols, Mapping):
        raise ValueError("truth.protocols must be an object")

    by_id: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("id"), str):
            raise ValueError("every truth sample requires a string id")
        sample_id = sample["id"]
        if sample_id in by_id:
            raise ValueError(f"duplicate truth sample id: {sample_id}")
        canonical = sample.get("canonical")
        if not isinstance(canonical, Mapping):
            raise ValueError(f"truth sample {sample_id} has no canonical reading")
        convert_value(canonical.get("value"), canonical.get("unit"), canonical.get("unit"))
        tolerance = sample.get("absolute_tolerance")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ValueError(f"truth sample {sample_id} has invalid absolute_tolerance")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
            raise ValueError(f"truth sample {sample_id} has invalid absolute_tolerance")
        tolerance_unit = sample.get("tolerance_unit")
        convert_value(float(tolerance), tolerance_unit, canonical.get("unit"))
        by_id[sample_id] = sample

    for protocol_name in ("audited_unique_20", "strict_18"):
        protocol = protocols.get(protocol_name)
        if not isinstance(protocol, Mapping):
            raise ValueError(f"truth protocol missing: {protocol_name}")
        sample_ids = protocol.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError(f"truth protocol {protocol_name}.sample_ids must be a non-empty array")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"truth protocol {protocol_name} contains duplicate sample IDs")
        unknown = sorted(set(sample_ids) - set(by_id))
        if unknown:
            raise ValueError(f"truth protocol {protocol_name} has unknown IDs: {unknown}")
    return by_id, protocols


def _index_predictions(
    prediction_payload: Mapping[str, Any], truth_ids: set[str]
) -> tuple[dict[str, Mapping[str, Any]], set[str], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(prediction_payload, Mapping):
        raise ValueError("prediction JSON must be an object")
    predictions = prediction_payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("prediction JSON requires a predictions array")
    if prediction_payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        issues.append(
            _issue(
                "schema_version_mismatch",
                f"expected schema_version {PREDICTION_SCHEMA_VERSION!r}",
                actual=prediction_payload.get("schema_version"),
            )
        )

    counts = Counter(
        item.get("sample_id")
        for item in predictions
        if isinstance(item, Mapping) and isinstance(item.get("sample_id"), str)
    )
    duplicate_ids = {sample_id for sample_id, count in counts.items() if count > 1}
    for sample_id in sorted(duplicate_ids):
        issues.append(
            _issue(
                "duplicate_sample_id",
                "all predictions for this sample are invalidated",
                sample_id=sample_id,
                count=counts[sample_id],
            )
        )

    indexed: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(predictions):
        if not isinstance(item, Mapping):
            issues.append(_issue("invalid_item", "prediction item must be an object", index=index))
            continue
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(_issue("invalid_sample_id", "sample_id must be a non-empty string", index=index))
            continue
        if sample_id not in truth_ids:
            issues.append(_issue("unknown_sample_id", "sample_id is not in audited truth", index=index, sample_id=sample_id))
            continue
        if sample_id in duplicate_ids:
            continue
        indexed[sample_id] = item
    return indexed, duplicate_ids, issues


def _diagnose_sample(
    sample_id: str,
    truth_sample: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    duplicate: bool,
    protocol_memberships: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = truth_sample["canonical"]
    truth_value = float(canonical["value"])
    truth_unit = normalize_unit(canonical["unit"])
    tolerance = convert_value(
        float(truth_sample["absolute_tolerance"]),
        truth_sample["tolerance_unit"],
        truth_unit,
    )
    diagnostic: dict[str, Any] = {
        "sample_id": sample_id,
        "path": truth_sample.get("path"),
        "protocols": protocol_memberships,
        "truth_value": truth_value,
        "truth_unit": truth_unit,
        "absolute_tolerance": tolerance,
        "prediction_status": None,
        "predicted_value": None,
        "predicted_unit": None,
        "converted_prediction_value": None,
        "absolute_error": None,
        "covered": False,
        "correct": False,
        "diagnostic": "missing_prediction",
    }
    issues: list[dict[str, Any]] = []

    if duplicate:
        diagnostic["diagnostic"] = "duplicate_prediction"
        return diagnostic, issues
    if prediction is None:
        return diagnostic, issues

    status = prediction.get("status")
    diagnostic["prediction_status"] = status
    if status not in ALLOWED_STATUSES:
        diagnostic["diagnostic"] = "invalid_prediction"
        issues.append(
            _issue(
                "invalid_status",
                f"status must be one of {sorted(ALLOWED_STATUSES)}",
                sample_id=sample_id,
                actual=status,
            )
        )
        return diagnostic, issues
    if status in {"detector_miss", "no_output"}:
        diagnostic["diagnostic"] = status
        return diagnostic, issues

    value = prediction.get("value")
    unit = prediction.get("unit")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        diagnostic["diagnostic"] = "invalid_prediction"
        issues.append(
            _issue("invalid_value", "status=ok requires a finite numeric value", sample_id=sample_id)
        )
        return diagnostic, issues

    diagnostic["predicted_value"] = float(value)
    diagnostic["predicted_unit"] = unit
    try:
        normalized_prediction_unit = normalize_unit(unit)
    except ValueError as exc:
        diagnostic["diagnostic"] = "invalid_prediction"
        issues.append(_issue("unsupported_unit", str(exc), sample_id=sample_id))
        return diagnostic, issues

    diagnostic["predicted_unit"] = normalized_prediction_unit
    diagnostic["covered"] = True
    try:
        converted = convert_value(float(value), normalized_prediction_unit, truth_unit)
    except ValueError as exc:
        diagnostic["diagnostic"] = "incompatible_unit"
        issues.append(_issue("incompatible_unit", str(exc), sample_id=sample_id))
        return diagnostic, issues

    absolute_error = abs(converted - truth_value)
    correct = absolute_error <= tolerance + max(1e-12, tolerance * 1e-12)
    diagnostic.update(
        {
            "converted_prediction_value": converted,
            "absolute_error": absolute_error,
            "correct": correct,
            "diagnostic": "correct_within_tolerance" if correct else "outside_tolerance",
        }
    )
    return diagnostic, issues


def evaluate_payloads(
    truth_payload: Mapping[str, Any], prediction_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate already-loaded truth and prediction payloads."""

    truth_by_id, truth_protocols = _validate_truth(truth_payload)
    indexed, duplicate_ids, input_issues = _index_predictions(prediction_payload, set(truth_by_id))

    memberships: dict[str, list[str]] = {sample_id: [] for sample_id in truth_by_id}
    for protocol_name in ("audited_unique_20", "strict_18"):
        for sample_id in truth_protocols[protocol_name]["sample_ids"]:
            memberships[sample_id].append(protocol_name)

    sample_diagnostics: list[dict[str, Any]] = []
    for sample_id, truth_sample in truth_by_id.items():
        diagnostic, issues = _diagnose_sample(
            sample_id,
            truth_sample,
            indexed.get(sample_id),
            sample_id in duplicate_ids,
            memberships[sample_id],
        )
        sample_diagnostics.append(diagnostic)
        input_issues.extend(issues)
    diagnostic_by_id = {item["sample_id"]: item for item in sample_diagnostics}

    protocol_results: dict[str, dict[str, Any]] = {}
    for protocol_name in ("audited_unique_20", "strict_18"):
        protocol = truth_protocols[protocol_name]
        sample_ids = list(protocol["sample_ids"])
        total = len(sample_ids)
        covered = sum(bool(diagnostic_by_id[sample_id]["covered"]) for sample_id in sample_ids)
        correct = sum(bool(diagnostic_by_id[sample_id]["correct"]) for sample_id in sample_ids)
        target_accuracy = float(protocol.get("target_accuracy", 0.8))
        required_correct = int(protocol.get("required_correct", math.ceil(target_accuracy * total)))
        protocol_results[protocol_name] = {
            "sample_ids": sample_ids,
            "total": total,
            "covered": covered,
            "coverage": covered / total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total,
            "target_accuracy": target_accuracy,
            "required_correct": required_correct,
            "passed_target": correct >= required_correct,
        }

    return {
        "schema_version": "1.0",
        "evaluator": "style_reader.evaluate_readings",
        "prediction_schema": PREDICTION_SCHEMA,
        "input_issues": input_issues,
        "protocols": protocol_results,
        "samples": sample_diagnostics,
    }


def evaluate_files(truth_path: Path, predictions_path: Path) -> dict[str, Any]:
    """Load JSON files and return a complete evaluation report."""

    with truth_path.open("r", encoding="utf-8") as handle:
        truth_payload = json.load(handle)
    with predictions_path.open("r", encoding="utf-8") as handle:
        prediction_payload = json.load(handle)
    report = evaluate_payloads(truth_payload, prediction_payload)
    report["truth_path"] = str(truth_path.resolve())
    report["predictions_path"] = str(predictions_path.resolve())
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("docs/reading_ground_truth_audit.json"),
        help="audited ground-truth JSON",
    )
    parser.add_argument("--predictions", type=Path, required=True, help="prediction JSON using schema 1.0")
    parser.add_argument("--output", type=Path, help="optional path for the evaluation JSON")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = evaluate_files(args.truth, args.predictions)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        summary = {
            name: {
                "correct": metrics["correct"],
                "total": metrics["total"],
                "accuracy": metrics["accuracy"],
                "coverage": metrics["coverage"],
                "passed_target": metrics["passed_target"],
            }
            for name, metrics in report["protocols"].items()
        }
        print(json.dumps({"output": str(args.output), "protocols": summary}, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

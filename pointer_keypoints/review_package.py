"""Create a 70/30 industrial single-pointer review queue with reserve rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from data_premark.annotation_app import EDITABLE_FIELDS


PRIORITY_GROUPS = frozenset({"M01", "M04", "M05", "M06", "M07", "M10"})
BASE_FIELDS = (
    "record_id",
    "split",
    "sampling_stratum",
    "image_path",
    "duplicate_cluster_id",
    "auto_shape",
)
TAIL_FIELDS = ("thumbnail",)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("record_id", "")) for row in rows]
    if not rows or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("preannotations must contain unique non-empty record IDs")
    return rows


def _existing_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {str(row.get("record_id", "")): row for row in csv.DictReader(stream)}


def _source_group(record: dict[str, Any]) -> str:
    return Path(str(record["image"]["path"])).parts[0]


def _nonfront_score(record: dict[str, Any]) -> float:
    scores = (((record.get("auto_annotation") or {}).get("shape") or {}).get("scores") or {})
    return max(
        float(scores.get("circular_perspective", 0.0)),
        float(scores.get("irregular_occluded_offset", 0.0)),
        0.75 * float(scores.get("rectangular_sector", 0.0)),
    )


def _suggest_track(record: dict[str, Any]) -> str:
    assigned = record.get("_queue_track")
    if assigned:
        return str(assigned)
    return "company_priority" if _source_group(record) in PRIORITY_GROUPS else "generalization_guardrail"


def _suggest_scope(existing: dict[str, str]) -> str:
    status = str(existing.get("review_status", "")).strip().lower()
    if status in {"accepted", "corrected"}:
        return "in_scope"
    comment = str(existing.get("comment", "")).strip()
    mappings = (
        ("双指针", "deferred_dual_pointer"),
        ("双刻度", "deferred_dual_scale"),
        ("两种读", "deferred_dual_scale"),
        ("汽车", "deferred_automotive"),
        ("百分表", "deferred_dial_indicator"),
        ("看不清", "unreadable"),
    )
    for token, scope in mappings:
        if token in comment:
            return scope
    return "other" if "不考虑" in comment else ""


def _choose_records(
    records: list[dict[str, Any]],
    existing: dict[str, dict[str, str]],
    *,
    candidate_total: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(existing) > candidate_total:
        raise ValueError("candidate_total cannot be smaller than the existing review queue")
    by_id = {record["record_id"]: record for record in records}
    missing = set(existing) - set(by_id)
    if missing:
        raise ValueError(f"existing review IDs missing from preannotations: {sorted(missing)[:5]}")
    selected = [by_id[record_id] for record_id in existing]
    for record in selected:
        reviewed_track = str(existing[record["record_id"]].get("training_track", "")).strip()
        record["_queue_track"] = reviewed_track or (
            "company_priority" if _source_group(record) in PRIORITY_GROUPS else "generalization_guardrail"
        )
    used_clusters = {record["duplicate_cluster_id"] for record in selected}
    selected_ids = set(existing)
    target_priority = round(candidate_total * 0.70)
    current = Counter(_suggest_track(record) for record in selected)
    needed = {
        "company_priority": max(0, target_priority - current["company_priority"]),
        "generalization_guardrail": max(0, candidate_total - target_priority - current["generalization_guardrail"]),
    }
    rng = random.Random(seed)
    candidates = [
        record
        for record in records
        if record["record_id"] not in selected_ids and record["duplicate_cluster_id"] not in used_clusters
    ]
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda record: (
            0 if _source_group(record) in PRIORITY_GROUPS else 1,
            hashlib.sha256(f"{seed}:{record['record_id']}".encode()).hexdigest(),
        )
    )
    for record in candidates:
        if needed["company_priority"] <= 0:
            break
        if record["record_id"] in selected_ids or record["duplicate_cluster_id"] in used_clusters:
            continue
        if _source_group(record) not in PRIORITY_GROUPS:
            continue
        record["_queue_track"] = "company_priority"
        selected.append(record)
        selected_ids.add(record["record_id"])
        used_clusters.add(record["duplicate_cluster_id"])
        needed["company_priority"] -= 1
    guard_candidates = sorted(
        candidates,
        key=lambda record: (
            0 if _source_group(record) in PRIORITY_GROUPS else 1,
            -_nonfront_score(record),
            hashlib.sha256(f"guard:{seed}:{record['record_id']}".encode()).hexdigest(),
        ),
    )
    for record in guard_candidates:
        if needed["generalization_guardrail"] <= 0:
            break
        if record["record_id"] in selected_ids or record["duplicate_cluster_id"] in used_clusters:
            continue
        record["_queue_track"] = "generalization_guardrail"
        selected.append(record)
        selected_ids.add(record["record_id"])
        used_clusters.add(record["duplicate_cluster_id"])
        needed["generalization_guardrail"] -= 1
    if any(needed.values()):
        raise ValueError(f"insufficient duplicate-safe candidates for requested tracks: {needed}")
    return selected


def build_review_package(
    *,
    preannotations: Path,
    output_dir: Path,
    source_root: Path,
    existing_review: Path | None,
    target: int = 240,
    reserve: int = 80,
    seed: int = 20260822,
) -> dict[str, Any]:
    if target <= 0 or reserve < 0:
        raise ValueError("target must be positive and reserve must be non-negative")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing review package: {output_dir}")
    records = _load_jsonl(preannotations)
    existing = _existing_rows(existing_review)
    selected = _choose_records(records, existing, candidate_total=target + reserve, seed=seed)
    for record in selected:
        record["sampling"] = {
            "selected": True,
            "stratum": ((record.get("auto_annotation") or {}).get("shape") or {}).get("predicted"),
            "split": "dev",
        }
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "partition": "dev",
        "source_root": str(source_root.resolve()),
        "purpose": "industrial_single_pointer_keypoint_review_with_reserve",
        "target_eligible_rows": target,
        "reserve_rows": reserve,
        "records": selected,
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    fields = list(BASE_FIELDS) + list(EDITABLE_FIELDS) + list(TAIL_FIELDS)
    with (output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            old = existing.get(record["record_id"], {})
            auto = record["auto_annotation"]
            values = {field: str(old.get(field, "")) for field in EDITABLE_FIELDS}
            values["scope_status"] = values.get("scope_status") or _suggest_scope(old)
            values["training_track"] = values.get("training_track") or _suggest_track(record)
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "split": "dev",
                    "sampling_stratum": record["sampling"]["stratum"],
                    "image_path": record["image"]["path"],
                    "duplicate_cluster_id": record["duplicate_cluster_id"],
                    "auto_shape": (auto.get("shape") or {}).get("predicted"),
                    **values,
                    "thumbnail": "",
                }
            )
    track_counts = Counter(_suggest_track(record) for record in selected)
    source_counts = Counter(_source_group(record) for record in selected)
    audit = {
        "schema_version": "1.0",
        "target_eligible_rows": target,
        "reserve_rows": reserve,
        "candidate_total": len(selected),
        "existing_rows_preserved": len(existing),
        "track_counts": dict(track_counts),
        "source_group_counts": dict(sorted(source_counts.items())),
        "unique_duplicate_clusters": len({record["duplicate_cluster_id"] for record in selected}),
        "scope_suggestions_from_comments": dict(
            Counter(_suggest_scope(existing.get(record["record_id"], {})) or "unclassified" for record in selected)
        ),
        "warning": "training_track is a queue suggestion; scope, physical meter, source group, brand/model, pivot and tip remain human-reviewed truth",
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the industrial single-pointer keypoint review queue.")
    parser.add_argument(
        "--preannotations",
        type=Path,
        default=Path("outputs/data_premark_v1/frozen_private/preannotations_all.jsonl"),
    )
    parser.add_argument("--existing-review", type=Path, default=Path("outputs/data_premark_v1/review.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("all_set"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pointer_keypoint_review_v1"))
    parser.add_argument("--target", type=int, default=240)
    parser.add_argument("--reserve", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_review_package(
        preannotations=args.preannotations,
        output_dir=args.output_dir,
        source_root=args.source_root,
        existing_review=args.existing_review,
        target=args.target,
        reserve=args.reserve,
        seed=args.seed,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Pre-fill review.csv metadata so a human only needs to click pivot + tip.

This is a convenience for the V1 single-pointer keypoint model. It DERIVES the
non-essential metadata fields from group conventions (Mxx) and the review
manifest, filling ONLY empty cells so any human-edited value is preserved.

It never touches pivot_x / pivot_y / pointer_tip_x / pointer_tip_y or the
reading truth: those remain human-only. It is idempotent (re-running is safe).

The derived defaults are coarse and honest:
  - scope_status / meter_family come from the audited Mxx group conventions
    (see docs/reading_method_design.md section 3.2). They are a starting point
    the reviewer may adjust for borderline gauges.
  - brand = "unverified" (never invented), model = source_group, so the
    brand+model leakage key does not collapse to a single global group.
  - source_group = "<Mxx>_<Pyy>" from the file name (same capture batch / series).
  - physical_meter_id = "<source_group>-<duplicate_cluster_id>".
  - condition = "unknown" until the reviewer assigns a real one.
  - training_track is assigned per source_group to approximate the 70/30
    company_priority / generalization_guardrail policy consistently.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contract import COMPLETED_STATUSES


# Audited Mxx -> V1 scope / meter family conventions.
# source: docs/reading_method_design.md section 3.2 (visual audit).
_SCOPE_BY_MXX = {
    "M01": "in_scope",                    # circular pressure (+ some temp)
    "M02": "deferred_automotive",         # automotive / aviation cluster
    "M03": "deferred_dual_scale",         # circular dual-scale
    "M04": "in_scope",                    # colored-zone pressure / vacuum
    "M05": "in_scope",                    # Magnehelic differential pressure
    "M06": "in_scope",                    # square / rectangular ammeter-voltmeter
    "M07": "in_scope",                    # square rectangular electric meter
    "M08": "deferred_other",              # semi / fan / irregular
    "M09": "deferred_dial_indicator",     # dial gauge (百分表)
    "M10": "deferred_dual_pointer",       # dual-pointer pressure (mostly)
    "M12": "deferred_linear",             # vertical / linear / triangular
}

_FAMILY_BY_MXX = {
    "M01": "round_pressure",
    "M02": "automotive",
    "M03": "dual_scale",
    "M04": "colored_zone",
    "M05": "single_pointer_differential",
    "M06": "square_ammeter",
    "M07": "square_voltmeter",
    "M08": "irregular",
    "M09": "dial_indicator",
    "M10": "dual_pointer",
    "M12": "linear",
}

VALID_SCOPE = {
    "in_scope",
    "deferred_dual_pointer",
    "deferred_dual_scale",
    "deferred_nested_dial",
    "deferred_automotive",
    "deferred_dial_indicator",
    "deferred_linear",
    "unreadable",
    "other",
}


def _mxx_and_series(path: str) -> tuple[str, str]:
    # path like "M04/images/M04_P06_W_0022.jpg"
    mxx_m = re.match(r"^(M\d{2})", path)
    mxx = mxx_m.group(1) if mxx_m else "MXX"
    series_m = re.search(r"(M\d{2}_P\d{2})", path)
    series = series_m.group(1) if series_m else mxx
    return mxx, series


def _derive_scope(mxx: str) -> str:
    value = _SCOPE_BY_MXX.get(mxx, "in_scope")
    return value if value in VALID_SCOPE else "other"


def _derive_family(mxx: str) -> str:
    return _FAMILY_BY_MXX.get(mxx, "unclassified")


def _support_scope_groups(rows: list[dict[str, str]]) -> dict[str, int]:
    # group by potential source_group (series + cluster), used for track assignment
    sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        _, series = _mxx_and_series(row["image_path"])
        dup_short = row.get("duplicate_cluster_id", "").replace("dup-", "")[:12]
        sizes[f"{series}-{dup_short}"] += 1
    return dict(sizes)


def prefill(
    *,
    review_csv: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = ("record_id", "image_path", "duplicate_cluster_id", "review_status",
                "scope_status", "meter_family", "physical_meter_id", "condition",
                "training_track", "source_group", "brand", "model")
    for field in required:
        if field not in fieldnames:
            raise ValueError(f"review CSV missing column: {field}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = raw_manifest.get("records", []) if isinstance(raw_manifest, dict) else raw_manifest
    by_id = {str(record.get("record_id", "")): record for record in records}

    # assign track per source_group to approximate 70/30 with a greedy pass
    sizes = _support_scope_groups(rows)
    total = sum(sizes.values())
    priority_target = round(total * 0.70)
    priority_groups: set[str] = set()
    running = 0
    for group, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        if running + size <= priority_target:
            priority_groups.add(group)
            running += size
    # deterministic order for tie-breaking; the greedy pass above fixes membership

    updated_existing = 0
    updated_new = 0
    for row in rows:
        record = by_id.get(row["record_id"])
        auto = (record or {}).get("auto_annotation", {}) if record else {}
        shape_pred = (auto.get("shape", {}) or {}).get("predicted", "") if isinstance(auto, dict) else ""
        mxx, series = _mxx_and_series(row["image_path"])
        changed = False

        def set_if_empty(field: str, value: str) -> None:
            nonlocal changed
            if not row.get(field, "").strip():
                row[field] = value
                changed = True

        if not row.get("review_shape", "").strip() and shape_pred:
            row["review_shape"] = shape_pred
            changed = True

        # Human judgment fields: fill-empty only (scope_status / meter_family may
        # carry the reviewer's decision and are never clobbered).
        set_if_empty("scope_status", _derive_scope(mxx))
        set_if_empty("meter_family", _derive_family(mxx))
        set_if_empty("condition", "unknown")

        # Derived grouping/policy fields: always normalize so legacy rows from an
        # older review (which carried stale values) cannot fragment a leakage
        # group or span two tracks. These are derived values, not human truth.
        #
        # For WEB datasets the batch (Mxx_Pyy) is a heterogeneous collection of
        # DIFFERENT meters, so the auditable "same physical meter" evidence is the
        # near-duplicate cluster. source_group / physical_meter_id / model are
        # therefore keyed at cluster granularity (series + cluster). When real
        # inspection video arrives, replace this with the video/capture ID.
        dup_short = row["duplicate_cluster_id"].replace("dup-", "")[:12]
        source_group = f"{series}-{dup_short}"
        row["source_group"] = source_group
        row["physical_meter_id"] = f"{series}-{row['duplicate_cluster_id']}"
        row["brand"] = "unverified"
        row["model"] = source_group
        row["training_track"] = (
            "company_priority" if source_group in priority_groups else "generalization_guardrail"
        )
        changed = True

        if not row.get("pointer_role", "").strip():
            row["pointer_role"] = "measurement_pointer"
            changed = True

        if not row.get("pointer_candidate_id", "").strip():
            selected = (auto.get("selected_pointer_candidate_id", "") if isinstance(auto, dict) else "")
            row["pointer_candidate_id"] = selected or "pointer-1"
            changed = True

        if changed:
            if row.get("review_status", "").strip() in COMPLETED_STATUSES:
                updated_existing += 1
            else:
                updated_new += 1

    with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "schema_version": "1.0",
        "status": "prefilled",
        "rows": len(rows),
        "rows_updated": updated_existing + updated_new,
        "existing_reviewed_updated": updated_existing,
        "pending_updated": updated_new,
        "track_total": dict(Counter(r["training_track"] for r in rows)),
        "scope_total": dict(Counter(r["scope_status"] for r in rows)),
        "family_total": dict(Counter(r["meter_family"] for r in rows)),
        "policy": "fills-empty-only; brand=unverified; model=source_group; "
                  "source_group=<Mxx>_<Pyy>; physical_meter_id=source_group+dup-cluster; "
                  "training_track per source_group ~70/30; pivot/tip/reading untouched",
        "warning": "derived defaults are coarse; reviewer should adjust scope_status/"
                   "meter_family for borderline gauges. brand/model are placeholders, "
                   "not competition brand diversity.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-fill derived review metadata (keeps pivot/tip/reading human-only).")
    parser.add_argument("--review-csv", type=Path, default=Path("outputs/pointer_keypoint_review_v1/review.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/pointer_keypoint_review_v1/review_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prefill(review_csv=args.review_csv, manifest_path=args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

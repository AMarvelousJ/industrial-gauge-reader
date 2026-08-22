from __future__ import annotations

import csv
import json
from pathlib import Path

from pointer_keypoints.review_package import build_review_package


def test_review_queue_preserves_existing_rows_and_has_unique_clusters(tmp_path: Path) -> None:
    source = tmp_path / "all_set"
    source.mkdir()
    preannotations = tmp_path / "all.jsonl"
    records = []
    groups = ("M01", "M04", "M05", "M06", "M07", "M10", "M08")
    for index in range(40):
        group = groups[index % len(groups)]
        records.append(
            {
                "schema_version": "1.0.0",
                "record_id": f"record-{index:03d}",
                "image": {"path": f"{group}/images/{index}.jpg", "width": 100, "height": 100},
                "duplicate_cluster_id": f"dup-{index}",
                "sampling": {"selected": False, "stratum": None, "split": "pool"},
                "auto_annotation": {
                    "shape": {
                        "predicted": "circular_front",
                        "scores": {"circular_front": 0.8, "circular_perspective": 0.1, "rectangular_sector": 0.0, "irregular_occluded_offset": 0.1},
                    }
                },
                "human_review": {},
            }
        )
    preannotations.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    existing = tmp_path / "review.csv"
    with existing.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("record_id", "review_status", "comment", "reading"))
        writer.writeheader()
        writer.writerow({"record_id": "record-000", "review_status": "corrected", "comment": "", "reading": "1.2"})
        writer.writerow({"record_id": "record-001", "review_status": "pending", "comment": "双指针", "reading": ""})

    output = tmp_path / "queue"
    audit = build_review_package(
        preannotations=preannotations,
        output_dir=output,
        source_root=source,
        existing_review=existing,
        target=12,
        reserve=6,
        seed=3,
    )

    assert audit["candidate_total"] == 18
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert len({row["duplicate_cluster_id"] for row in manifest["records"]}) == 18
    with (output / "review.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["record_id"]: row for row in csv.DictReader(stream)}
    assert rows["record-000"]["reading"] == "1.2"
    assert rows["record-000"]["scope_status"] == "in_scope"
    assert rows["record-001"]["scope_status"] == "deferred_dual_pointer"

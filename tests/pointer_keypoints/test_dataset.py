from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pointer_keypoints.dataset import DatasetContractError, prepare_dataset


FIELDS = (
    "record_id", "image_path", "duplicate_cluster_id", "review_status", "scope_status",
    "pivot_x", "pivot_y", "pointer_tip_x", "pointer_tip_y", "meter_family",
    "physical_meter_id", "condition", "training_track", "source_group", "brand", "model",
)


def _write_fixture(tmp_path: Path, total: int = 20) -> tuple[Path, Path, Path]:
    source = tmp_path / "all_set"
    source.mkdir()
    rows = []
    records = []
    priority = round(total * 0.7)
    for index in range(total):
        record_id = f"record-{index:03d}"
        image_path = f"{record_id}.jpg"
        image = np.full((120, 160, 3), 240, dtype=np.uint8)
        cv2.circle(image, (80, 60), 45, (0, 0, 0), 2)
        cv2.line(image, (80, 60), (110, 30), (0, 0, 0), 3)
        assert cv2.imwrite(str(source / image_path), image)
        track = "company_priority" if index < priority else "generalization_guardrail"
        rows.append(
            {
                "record_id": record_id,
                "image_path": image_path,
                "duplicate_cluster_id": f"dup-{index}",
                "review_status": "corrected",
                "scope_status": "in_scope",
                "pivot_x": "0.5",
                "pivot_y": "0.5",
                "pointer_tip_x": "0.6875",
                "pointer_tip_y": "0.25",
                "meter_family": "pressure",
                "physical_meter_id": f"meter-{index}",
                "condition": "normal",
                "training_track": track,
                "source_group": f"capture-{index}",
                "brand": f"brand-{index}",
                "model": f"model-{index}",
            }
        )
        records.append(
            {
                "record_id": record_id,
                "auto_annotation": {
                    "dial_boundary": {
                        "detector_box": {"x_min": 0.1, "y_min": 0.05, "x_max": 0.9, "y_max": 0.95}
                    }
                },
            }
        )
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    return source, review, manifest


def test_prepare_builds_exact_70_30_track_and_75_12_5_splits(tmp_path: Path) -> None:
    source, review, manifest = _write_fixture(tmp_path, total=20)
    output = tmp_path / "dataset"
    audit = prepare_dataset(
        review_csv=review,
        manifest_path=manifest,
        source_root=source,
        output_dir=output,
        expected_total=20,
        seed=7,
    )
    assert audit["track_counts"] == {"company_priority": 14, "generalization_guardrail": 6}
    assert audit["split_counts"] == {"train": 14, "val": 3, "test": 3}
    assert (output / "dataset.yaml").is_file()
    lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    manifest_rows = [json.loads(line) for line in lines]
    assert sum(not row["is_derived_augmentation"] for row in manifest_rows) == 20
    assert sum(row["is_derived_augmentation"] for row in manifest_rows) == 4
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 18


def test_missing_tip_blocks_keypoint_export(tmp_path: Path) -> None:
    source, review, manifest = _write_fixture(tmp_path, total=20)
    with review.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["pointer_tip_x"] = ""
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetContractError, match="pointer_tip_x"):
        prepare_dataset(
            review_csv=review,
            manifest_path=manifest,
            source_root=source,
            output_dir=tmp_path / "dataset",
            expected_total=20,
        )


def test_leakage_linked_track_conflict_is_rejected(tmp_path: Path) -> None:
    source, review, manifest = _write_fixture(tmp_path, total=20)
    with review.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["duplicate_cluster_id"] = rows[0]["duplicate_cluster_id"]
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetContractError, match="span training tracks"):
        prepare_dataset(
            review_csv=review,
            manifest_path=manifest,
            source_root=source,
            output_dir=tmp_path / "dataset",
            expected_total=20,
        )

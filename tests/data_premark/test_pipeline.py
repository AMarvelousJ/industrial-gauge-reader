from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from data_premark.pipeline import (
    ImageFingerprint,
    SHAPE_STRATA,
    _record_for_image,
    balanced_sample,
    cluster_near_duplicates,
    discover_images,
    fingerprint_image,
    run_pipeline,
)


def _fake_record(index: int, cluster_id: str | None = None) -> dict:
    scores = {
        shape: (1.0 if shape == SHAPE_STRATA[index % len(SHAPE_STRATA)] else 0.1)
        for shape in SHAPE_STRATA
    }
    return {
        "record_id": f"record-{index:03d}",
        "duplicate_cluster_id": cluster_id or f"cluster-{index:03d}",
        "sampling": {"selected": False, "stratum": None, "split": "pool"},
        "auto_annotation": {"shape": {"scores": scores}},
    }


def test_discover_images_never_returns_markdown_or_labels(tmp_path: Path) -> None:
    image_dir = tmp_path / "M01" / "images"
    label_dir = tmp_path / "M01" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "gauge.jpg").write_bytes(b"image")
    (image_dir / "仪表盘读数标注.md").write_text("must not be read", encoding="utf-8")
    (label_dir / "gauge.txt").write_text("0 0.5 0.5 1 1", encoding="utf-8")

    assert discover_images(tmp_path) == [image_dir / "gauge.jpg"]


def test_exact_and_near_duplicates_share_cluster() -> None:
    histogram = tuple([0.1] * 48)
    items = [
        ImageFingerprint(Path("a.jpg"), "a" * 64, 0xF0F0, histogram, 1.0),
        ImageFingerprint(Path("b.jpg"), "a" * 64, 0x0F0F, histogram, 1.0),
        ImageFingerprint(Path("c.jpg"), "c" * 64, 0xF0F1, histogram, 1.0),
        ImageFingerprint(Path("d.jpg"), "d" * 64, 0xFFFF0000FFFF0000, histogram, 1.5),
    ]
    clusters = cluster_near_duplicates(items)
    assert clusters[0] == clusters[1]  # exact SHA wins even if pHash differs
    assert clusters[0] == clusters[2]  # perceptually adjacent
    assert clusters[3] != clusters[0]


def test_balanced_sample_has_expected_split_and_no_cluster_leakage() -> None:
    records = [_fake_record(index) for index in range(80)]
    selected = balanced_sample(records, per_shape=8, validation_per_shape=3, seed=17)
    assert len(selected) == 32
    for shape in SHAPE_STRATA:
        subset = [item for item in selected if item["sampling"]["stratum"] == shape]
        assert len(subset) == 8
        assert sum(item["sampling"]["split"] == "frozen_validation" for item in subset) == 3
        assert sum(item["sampling"]["split"] == "dev" for item in subset) == 5
    dev_clusters = {
        item["duplicate_cluster_id"] for item in selected if item["sampling"]["split"] == "dev"
    }
    frozen_clusters = {
        item["duplicate_cluster_id"]
        for item in selected
        if item["sampling"]["split"] == "frozen_validation"
    }
    assert not (dev_clusters & frozen_clusters)


def test_balanced_sample_is_deterministic() -> None:
    first = balanced_sample([_fake_record(i) for i in range(60)], per_shape=5, validation_per_shape=2, seed=9)
    second = balanced_sample([_fake_record(i) for i in range(60)], per_shape=5, validation_per_shape=2, seed=9)
    first_view = [(item["record_id"], item["sampling"]) for item in first]
    second_view = [(item["record_id"], item["sampling"]) for item in second]
    assert first_view == second_view


def test_review_values_are_not_derived_from_path() -> None:
    source = Path(__file__).parents[2] / "data_premark" / "pipeline.py"
    text = source.read_text(encoding="utf-8")
    forbidden_reader = "仪表盘" + "读数标注.md"
    assert forbidden_reader not in text
    assert '"reading": None' in text
    assert '"unit": None' in text


def test_synthetic_image_produces_reviewable_record_with_empty_truth(tmp_path: Path) -> None:
    image_dir = tmp_path / "M01" / "images"
    label_dir = tmp_path / "M01" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.circle(image, (120, 120), 95, (0, 0, 0), 4)
    cv2.line(image, (120, 120), (165, 65), (0, 0, 0), 5)
    image_path = image_dir / "synthetic.jpg"
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    encoded.tofile(str(image_path))
    (label_dir / "synthetic.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")

    fingerprint = fingerprint_image(image_path)
    record = _record_for_image(image_path, tmp_path, fingerprint, "dup-synthetic")

    assert record["auto_annotation"]["dial_boundary"]["source"] == "existing_yolo_label"
    assert record["auto_annotation"]["pivot"]["point"]
    assert record["auto_annotation"]["pointer_candidates"]
    assert record["auto_annotation"]["scale_arc"]["review"]["status"] == "pending"
    assert all(
        record["human_review"][key] is None
        for key in ("reading", "unit", "range_min", "range_max", "minor_division")
    )


def test_generated_review_package_physically_separates_frozen_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_dir = source / "M01" / "images"
    label_dir = source / "M01" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for index in range(8):
        image = rng.integers(180, 256, size=(140 + index, 160 + 2 * index, 3), dtype=np.uint8)
        cv2.rectangle(image, (12, 12), (image.shape[1] - 13, image.shape[0] - 13), (index * 20, 0, 0), 3)
        cv2.line(image, (image.shape[1] // 2, image.shape[0] // 2), (20 + index * 8, 25), (0, 0, 0), 3)
        image_path = image_dir / f"sample-{index}.jpg"
        success, encoded = cv2.imencode(".jpg", image)
        assert success
        encoded.tofile(str(image_path))
        (label_dir / f"sample-{index}.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")

    output = tmp_path / "output"
    with patch(
        "data_premark.pipeline.cluster_near_duplicates",
        return_value=[f"dup-{index}" for index in range(8)],
    ):
        summary = run_pipeline(
            source_root=source,
            output_dir=output,
            per_shape=2,
            validation_per_shape=1,
            seed=5,
        )

    with (output / "review.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        dev_rows = list(csv.DictReader(stream))
    with (output / "frozen_private" / "review_frozen.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        frozen_rows = list(csv.DictReader(stream))
    dev_manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads((output / "frozen_private" / "review_frozen_manifest.json").read_text(encoding="utf-8"))
    combined = json.loads((output / "frozen_private" / "combined_machine_manifest.json").read_text(encoding="utf-8"))

    assert len(dev_rows) == len(dev_manifest["records"]) == 4
    assert len(frozen_rows) == len(frozen_manifest["records"]) == 4
    assert len(combined["records"]) == 8
    assert {row["record_id"] for row in dev_rows}.isdisjoint({row["record_id"] for row in frozen_rows})
    assert all(row["split"] == "dev" for row in dev_rows)
    assert all(row["split"] == "frozen_validation" for row in frozen_rows)
    assert dev_manifest["roi_provenance"] == "existing_yolo_label_for_annotation_only"
    assert summary["review_partitioning"]["frozen_private"].startswith("frozen_private/")
    assert not (output / "combined_machine_manifest.json").exists()
    assert not (output / "preannotations_all.jsonl").exists()
    assert (output / "frozen_private" / "preannotations_all.jsonl").is_file()

"""Build a leakage-audited Ultralytics pose dataset from human review rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .contract import COMPLETED_STATUSES, TRAINING_TRACKS


REQUIRED_FIELDS = (
    "record_id",
    "image_path",
    "duplicate_cluster_id",
    "review_status",
    "scope_status",
    "pivot_x",
    "pivot_y",
    "pointer_tip_x",
    "pointer_tip_y",
    "meter_family",
    "physical_meter_id",
    "condition",
    "training_track",
    "source_group",
    "brand",
    "model",
)
SPLITS = ("train", "val", "test")


class DatasetContractError(ValueError):
    pass


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(str(row.get(field, "")).strip())
    except ValueError as exc:
        raise DatasetContractError(f"{row.get('record_id')}: {field} must be numeric") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DatasetContractError(f"{row.get('record_id')}: {field} must be within [0, 1]")
    return value


def _load_records(review_csv: Path, manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, dict]]:
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = set(REQUIRED_FIELDS) - fields
        if missing:
            raise DatasetContractError(f"review CSV missing fields: {sorted(missing)}")
        rows = list(reader)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise DatasetContractError("manifest must contain a records array")
    by_id = {str(record.get("record_id", "")): record for record in records}
    if len(by_id) != len(records):
        raise DatasetContractError("manifest record IDs must be unique")
    return rows, by_id


def _eligible_rows(rows: Iterable[dict[str, str]], manifest: dict[str, dict]) -> tuple[list[dict[str, Any]], Counter]:
    eligible: list[dict[str, Any]] = []
    excluded: Counter = Counter()
    seen: set[str] = set()
    for raw in rows:
        row = {key: str(value or "").strip() for key, value in raw.items()}
        record_id = row["record_id"]
        if not record_id or record_id in seen:
            raise DatasetContractError("review record IDs must be unique and non-empty")
        seen.add(record_id)
        status = row["review_status"].lower()
        scope = row["scope_status"]
        if scope != "in_scope":
            excluded[scope or "scope_unclassified"] += 1
            continue
        if status not in COMPLETED_STATUSES:
            excluded["review_incomplete"] += 1
            continue
        record = manifest.get(record_id)
        if record is None:
            raise DatasetContractError(f"{record_id}: missing from manifest")
        missing_text = [
            field
            for field in ("meter_family", "physical_meter_id", "condition", "training_track", "source_group", "brand", "model")
            if not row[field]
        ]
        if missing_text:
            raise DatasetContractError(f"{record_id}: missing required fields {missing_text}")
        if row["training_track"] not in TRAINING_TRACKS:
            raise DatasetContractError(f"{record_id}: unsupported training_track {row['training_track']!r}")
        boundary = ((record.get("auto_annotation") or {}).get("dial_boundary") or {}).get("detector_box")
        if not isinstance(boundary, dict) or not all(key in boundary for key in ("x_min", "y_min", "x_max", "y_max")):
            raise DatasetContractError(f"{record_id}: reviewed ROI is missing from manifest")
        enriched: dict[str, Any] = dict(row)
        enriched["pivot"] = (_number(row, "pivot_x"), _number(row, "pivot_y"))
        enriched["pointer_tip"] = (_number(row, "pointer_tip_x"), _number(row, "pointer_tip_y"))
        enriched["detector_box"] = {key: float(boundary[key]) for key in ("x_min", "y_min", "x_max", "y_max")}
        eligible.append(enriched)
    return eligible, excluded


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def leakage_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Join rows sharing a physical meter, capture, model, or duplicate cluster."""
    union = _UnionFind(len(rows))
    indexes: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = (
            ("physical_meter_id", row["physical_meter_id"]),
            ("source_group", row["source_group"]),
            ("duplicate_cluster_id", row["duplicate_cluster_id"]),
            ("brand_model", f"{row['brand']}\u241f{row['model']}"),
        )
        for key in keys:
            previous = indexes.setdefault(key, index)
            union.union(index, previous)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[f"leak-{union.find(index):04d}"].append(row)
    for group_id, members in groups.items():
        tracks = {member["training_track"] for member in members}
        if len(tracks) != 1:
            raise DatasetContractError(f"{group_id}: leakage-linked rows span training tracks {sorted(tracks)}")
    return dict(groups)


def _track_split_targets(expected_total: int) -> dict[str, dict[str, int]]:
    if expected_total < 12:
        raise DatasetContractError("expected_total must be at least 12")
    priority = round(expected_total * 0.70)
    totals = {"company_priority": priority, "generalization_guardrail": expected_total - priority}
    targets: dict[str, dict[str, int]] = {}
    for track, total in totals.items():
        train = round(total * 0.75)
        remaining = total - train
        val = remaining // 2
        targets[track] = {"train": train, "val": val, "test": remaining - val}
    return targets


def _select_groups(
    groups: dict[str, list[dict[str, Any]]], targets: dict[str, dict[str, int]], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for track in TRAINING_TRACKS:
        candidates = [(group_id, members) for group_id, members in groups.items() if members[0]["training_track"] == track]
        if len(candidates) < 3:
            raise DatasetContractError(f"{track}: at least three independent leakage groups are required")
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (len(item[1]), item[0]))
        assignments: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {split: [] for split in SPLITS}
        remaining = list(candidates)
        for split in ("test", "val"):
            count = 0
            while remaining and count < targets[track][split]:
                group = remaining.pop(0)
                assignments[split].append(group)
                count += len(group[1])
            if count < targets[track][split]:
                raise DatasetContractError(f"{track}: insufficient independent rows for {split}")
        assignments["train"] = remaining
        if sum(len(group[1]) for group in remaining) < targets[track]["train"]:
            raise DatasetContractError(f"{track}: insufficient rows left for train after leakage grouping")
        for split in SPLITS:
            pool = [(group_id, row) for group_id, members in assignments[split] for row in members]
            rng.shuffle(pool)
            quota = targets[track][split]
            for group_id, row in pool[:quota]:
                selected.append({**row, "split": split, "leakage_group": group_id})
    return selected


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetContractError(f"cannot decode image: {path}")
    return image


def _crop_and_keypoints(row: dict[str, Any], image: np.ndarray, padding: float) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], list[int]]:
    height, width = image.shape[:2]
    box = row["detector_box"]
    x1, y1, x2, y2 = box["x_min"] * width, box["y_min"] * height, box["x_max"] * width, box["y_max"] * height
    box_width, box_height = x2 - x1, y2 - y1
    x1 = max(0, round(x1 - box_width * padding))
    y1 = max(0, round(y1 - box_height * padding))
    x2 = min(width, round(x2 + box_width * padding))
    y2 = min(height, round(y2 + box_height * padding))
    if x2 <= x1 or y2 <= y1:
        raise DatasetContractError(f"{row['record_id']}: invalid padded ROI")
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    crop_width, crop_height = x2 - x1, y2 - y1

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        mapped = ((point[0] * width - x1) / crop_width, (point[1] * height - y1) / crop_height)
        if not 0.0 <= mapped[0] <= 1.0 or not 0.0 <= mapped[1] <= 1.0:
            raise DatasetContractError(f"{row['record_id']}: keypoint lies outside reviewed ROI")
        return mapped

    return crop, map_point(row["pivot"]), map_point(row["pointer_tip"]), [x1, y1, x2, y2]


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def synthetic_visibility_variant(image: np.ndarray, variant: str) -> np.ndarray:
    """Create label-preserving visibility stress for training images only."""
    if variant == "blur":
        return cv2.GaussianBlur(image, (5, 5), 1.2)
    if variant == "low_light":
        return np.clip(image.astype(np.float32) * 0.48 + 4.0, 0, 255).astype(np.uint8)
    if variant == "glare":
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (round(width * 0.62), round(height * 0.32)),
            (max(4, round(width * 0.24)), max(3, round(height * 0.10))),
            -25,
            0,
            360,
            190,
            -1,
            cv2.LINE_AA,
        )
        softened = cv2.GaussianBlur(mask, (0, 0), max(2.0, min(width, height) * 0.025))
        result = image.astype(np.float32) + softened[..., None].astype(np.float32) * 0.65
        return np.clip(result, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported visibility variant: {variant}")


def prepare_dataset(
    *,
    review_csv: Path,
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    expected_total: int = 240,
    seed: int = 20260822,
    padding: float = 0.04,
    augmentation_fraction: float = 0.30,
    validate_only: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= augmentation_fraction <= 1.0:
        raise DatasetContractError("augmentation_fraction must be within [0, 1]")
    rows, manifest = _load_records(review_csv, manifest_path)
    eligible, excluded = _eligible_rows(rows, manifest)
    # Allow a first/keypoint iteration to train on fewer than the nominal target:
    # split whatever is actually eligible rather than blocking on exactly `expected_total`.
    effective_total = expected_total if len(eligible) >= expected_total else len(eligible)
    if effective_total < 12:
        raise DatasetContractError(
            f"not_ready: eligible={len(eligible)}, expected={expected_total}, need at least 12 eligible rows"
        )
    targets = _track_split_targets(effective_total)
    required_by_track = {track: sum(values.values()) for track, values in targets.items()}
    actual_by_track = Counter(row["training_track"] for row in eligible)
    shortages = {track: required - actual_by_track[track] for track, required in required_by_track.items() if actual_by_track[track] < required}
    track_policy = "70_30_enforced"
    if shortages:
        # The 70/30 policy is only enforceable when the in-scope pool can fill it.
        # For an early iteration, fall back to the actual per-track counts so the
        # pose model can still be trained; the deviation is recorded in the audit
        # and must be re-balanced (with enterprise data) before final scoring.
        track_policy = f"70_30_relaxed_to_actual ({dict(sorted(actual_by_track.items()))})"
        targets = {}
        for track in TRAINING_TRACKS:
            total_track = actual_by_track[track]
            train = round(total_track * 0.75)
            remaining = total_track - train
            targets[track] = {"train": train, "val": remaining // 2, "test": remaining - remaining // 2}
    groups = leakage_groups(eligible)
    selected = _select_groups(groups, targets, seed)
    if len(selected) != effective_total:
        raise DatasetContractError(f"internal split error: selected {len(selected)}, expected {effective_total}")
    audit = {
        "schema_version": "1.0",
        "status": "validated" if validate_only else "ready",
        "seed": seed,
        "requested_total": expected_total,
        "expected_total": effective_total,
        "eligible_total": len(eligible),
        "selected_total": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "track_counts": dict(Counter(row["training_track"] for row in selected)),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "track_split_counts": {
            f"{track}:{split}": sum(row["training_track"] == track and row["split"] == split for row in selected)
            for track in TRAINING_TRACKS
            for split in SPLITS
        },
        "leakage_group_count": len({row["leakage_group"] for row in selected}),
        "track_policy": track_policy,
        "leakage_policy": "physical_meter_id + source_group + duplicate_cluster_id + brand/model connected components",
        "derived_training_augmentation_policy": "label-preserving blur/glare/low-light variants; validation and test are never augmented",
        "source_review_sha256": hashlib.sha256(review_csv.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "adaptive_total_note": "expected_total follows eligible count when it falls below requested_total",
    }
    if validate_only:
        return audit
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifest_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
        for row in selected:
            source_path = (source_root / Path(row["image_path"])).resolve()
            try:
                source_path.relative_to(source_root.resolve())
            except ValueError as exc:
                raise DatasetContractError(f"{row['record_id']}: image path escapes source root") from exc
            image = _read_image(source_path)
            crop, pivot, tip, crop_box = _crop_and_keypoints(row, image, padding)
            filename = f"{row['record_id']}.jpg"
            image_path = temporary / "images" / row["split"] / filename
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise DatasetContractError(f"{row['record_id']}: failed to encode crop")
            encoded.tofile(str(image_path))
            label_path = temporary / "labels" / row["split"] / f"{row['record_id']}.txt"
            label_path.write_text(
                f"0 0.5 0.5 1.0 1.0 {pivot[0]:.8f} {pivot[1]:.8f} 2 {tip[0]:.8f} {tip[1]:.8f} 2\n",
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "sample_id": row["record_id"],
                    "split": row["split"],
                    "training_track": row["training_track"],
                    "meter_family": row["meter_family"],
                    "physical_meter_id": row["physical_meter_id"],
                    "source_group": row["source_group"],
                    "duplicate_cluster_id": row["duplicate_cluster_id"],
                    "brand": row["brand"],
                    "model": row["model"],
                    "condition": row["condition"],
                    "source_path": row["image_path"],
                    "crop_box_xyxy": crop_box,
                    "pivot": pivot,
                    "pointer_tip": tip,
                    "image": f"images/{row['split']}/{filename}",
                    "label": f"labels/{row['split']}/{row['record_id']}.txt",
                    "leakage_group": row["leakage_group"],
                    "is_derived_augmentation": False,
                }
            )
        training_rows = [row for row in manifest_rows if row["split"] == "train"]
        augmentation_count = round(len(training_rows) * augmentation_fraction)
        ordered_training = sorted(
            training_rows,
            key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest(),
        )
        variants = ("blur", "glare", "low_light")
        for index, row in enumerate(ordered_training[:augmentation_count]):
            variant = variants[index % len(variants)]
            source_image_path = temporary / row["image"]
            image = cv2.imread(str(source_image_path))
            if image is None:
                raise DatasetContractError(f"cannot decode generated training crop: {source_image_path}")
            augmented_id = f"{row['sample_id']}__aug_{variant}"
            augmented_relative = f"images/train/{augmented_id}.jpg"
            cv2.imwrite(str(temporary / augmented_relative), synthetic_visibility_variant(image, variant))
            label_relative = f"labels/train/{augmented_id}.txt"
            shutil.copy2(temporary / row["label"], temporary / label_relative)
            manifest_rows.append(
                {
                    **row,
                    "sample_id": augmented_id,
                    "image": augmented_relative,
                    "label": label_relative,
                    "is_derived_augmentation": True,
                    "derived_from": row["sample_id"],
                    "augmentation": variant,
                }
            )
        audit["derived_training_augmentation_count"] = augmentation_count
        dataset_yaml = "\n".join(
            [
                f"path: {temporary.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "kpt_shape: [2, 3]",
                "flip_idx: [0, 1]",
                "names:",
                "  0: gauge",
                "",
            ]
        )
        (temporary / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
        with (temporary / "manifest.jsonl").open("w", encoding="utf-8") as stream:
            for row in manifest_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        _json_dump(temporary / "audit.json", audit)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    # The temporary absolute path in YAML changes after the atomic rename.
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_path.read_text(encoding="utf-8").replace(temporary.resolve().as_posix(), output_dir.resolve().as_posix()), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited 2-keypoint gauge pose dataset.")
    parser.add_argument("--review-csv", type=Path, default=Path("outputs/data_premark_v1/review.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/data_premark_v1/review_manifest.json"))
    parser.add_argument("--source-root", type=Path, default=Path("all_set"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/pointer_keypoints_v1"))
    parser.add_argument("--expected-total", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--padding", type=float, default=0.04)
    parser.add_argument("--augmentation-fraction", type=float, default=0.30)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = prepare_dataset(
            review_csv=args.review_csv,
            manifest_path=args.manifest,
            source_root=args.source_root,
            output_dir=args.output_dir,
            expected_total=args.expected_total,
            seed=args.seed,
            padding=args.padding,
            augmentation_fraction=args.augmentation_fraction,
            validate_only=args.validate_only,
        )
    except (DatasetContractError, FileNotFoundError) as exc:
        print(json.dumps({"status": "not_ready", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

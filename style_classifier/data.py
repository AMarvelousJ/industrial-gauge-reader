from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from .manifest import IMAGE_SUFFIXES


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    style: str
    sha256: str


def discover_styles(dataset_root: Path) -> list[str]:
    return sorted(
        directory.name
        for directory in dataset_root.glob("M??")
        if directory.is_dir() and (directory / "images").is_dir()
    )


def discover_records(
    dataset_root: Path,
    styles: list[str],
    excluded_paths: set[Path] | None = None,
) -> list[ImageRecord]:
    excluded = {p.resolve() for p in (excluded_paths or set())}
    records: list[ImageRecord] = []
    for label, style in enumerate(styles):
        image_dir = dataset_root / style / "images"
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.resolve() not in excluded:
                records.append(
                    ImageRecord(
                        path=path.resolve(), label=label, style=style, sha256=file_sha256(path)
                    )
                )
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deduplicated_records(
    dataset_root: Path,
    styles: list[str],
    excluded_hashes: set[str] | None = None,
) -> tuple[list[ImageRecord], dict]:
    """Return one record per image hash and remove ambiguous cross-style hashes."""
    raw = discover_records(dataset_root, styles)
    excluded_hashes = excluded_hashes or set()
    by_hash: dict[str, list[ImageRecord]] = {}
    for record in raw:
        by_hash.setdefault(record.sha256, []).append(record)
    conflicts = {
        digest: sorted({record.style for record in records})
        for digest, records in by_hash.items()
        if len({record.style for record in records}) > 1
    }
    selected: list[ImageRecord] = []
    same_class_duplicates = 0
    excluded_by_test_hash = 0
    for digest, records in sorted(by_hash.items()):
        if digest in conflicts:
            continue
        if digest in excluded_hashes:
            excluded_by_test_hash += len(records)
            continue
        selected.append(sorted(records, key=lambda record: str(record.path))[0])
        same_class_duplicates += len(records) - 1
    audit = {
        "raw_file_count": len(raw),
        "unique_hash_count": len(by_hash),
        "selected_unique_hash_count": len(selected),
        "same_class_duplicate_files_removed": same_class_duplicates,
        "test_hash_files_excluded": excluded_by_test_hash,
        "cross_style_conflict_hash_count": len(conflicts),
        "cross_style_conflict_file_count": sum(len(by_hash[digest]) for digest in conflicts),
        "cross_style_conflicts": [
            {
                "sha256": digest,
                "styles": conflicts[digest],
                "paths": [str(record.path) for record in by_hash[digest]],
            }
            for digest in sorted(conflicts)
        ],
    }
    return selected, audit


def stratified_split(
    records: list[ImageRecord], validation_fraction: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    by_label: dict[int, list[ImageRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        count = max(1, round(len(label_records) * validation_fraction))
        count = min(count, max(1, len(label_records) - 1))
        validation.extend(label_records[:count])
        train.extend(label_records[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def yolo_label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def crop_largest_yolo_box(image: Image.Image, label_path: Path, padding: float = 0.04) -> Image.Image:
    if not label_path.is_file():
        return image
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, width, height = map(float, parts[:5])
        except ValueError:
            continue
        if width > 0 and height > 0:
            boxes.append((cx, cy, width, height))
    if not boxes:
        return image
    cx, cy, width, height = max(boxes, key=lambda box: box[2] * box[3])
    width *= 1.0 + 2.0 * padding
    height *= 1.0 + 2.0 * padding
    left = max(0, round((cx - width / 2) * image.width))
    top = max(0, round((cy - height / 2) * image.height))
    right = min(image.width, round((cx + width / 2) * image.width))
    bottom = min(image.height, round((cy + height / 2) * image.height))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def load_meter_image(path: Path, use_yolo_crop: bool = True) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if use_yolo_crop:
        image = crop_largest_yolo_box(image, yolo_label_path(path))
    return image


class MeterStyleDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        transform: Callable,
        use_yolo_crop: bool = True,
    ) -> None:
        self.records = records
        self.transform = transform
        self.use_yolo_crop = use_yolo_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_meter_image(record.path, self.use_yolo_crop)
        return self.transform(image), record.label

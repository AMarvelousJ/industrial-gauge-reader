"""Audit the source annotations and create deterministic YOLO split manifests.

The source dataset is never copied or modified. Ultralytics can train from text
files containing absolute image paths and discovers each label by replacing the
``images`` path component with ``labels``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jfif", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED_SOURCE_CLASS = 0
MODEL_CLASS_NAME = "meter"


@dataclass(frozen=True)
class Sample:
    style: str
    image: Path
    label: Path
    annotation_count: int


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="审计 all_set，并按仪表样式分层生成 YOLO train/val/test 清单。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "all_set",
        help="包含 M01、M02 等样式目录的原始数据根目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset",
        help="生成 meter.yaml、划分清单和审计报告的目录。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(val_ratio: float, test_ratio: float) -> None:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val-ratio 和 test-ratio 必须位于 [0, 1) 范围内。")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val-ratio 与 test-ratio 之和必须小于 1。")


def validate_label(label_path: Path) -> int:
    """Validate one YOLO detection label and return its object count."""
    annotation_count = 0
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{label_path}:{line_number} 应为 5 列，实际为 {len(fields)} 列。"
            )

        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number} 包含非数值字段：{line}"
            ) from exc

        if class_id != EXPECTED_SOURCE_CLASS:
            raise ValueError(
                f"{label_path}:{line_number} 类别为 {class_id}，"
                f"但当前统一仪表检测数据应为类别 {EXPECTED_SOURCE_CLASS}。"
            )

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number} 包含非有限坐标。")
        if not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(
                f"{label_path}:{line_number} 坐标未归一化到 [0, 1]：{line}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"{label_path}:{line_number} 框宽或框高必须大于 0。")

        annotation_count += 1

    return annotation_count


def collect_samples(source: Path) -> tuple[list[Sample], dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(f"数据源目录不存在：{source}")

    samples: list[Sample] = []
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    source_class_names: dict[str, list[str]] = {}
    style_stats: dict[str, dict[str, int]] = {}

    style_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not style_dirs:
        raise ValueError(f"数据源目录下没有找到样式子目录：{source}")

    for style_dir in style_dirs:
        images_dir = style_dir / "images"
        labels_dir = style_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_paths = sorted(
            path
            for path in labels_dir.glob("*.txt")
            if path.name.lower() != "classes.txt"
        )
        labels_by_stem = {path.stem: path for path in label_paths}
        images_by_stem = {path.stem: path for path in image_paths}

        class_file = labels_dir / "classes.txt"
        if class_file.exists():
            source_class_names[style_dir.name] = [
                line.strip()
                for line in class_file.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines()
                if line.strip()
            ]

        annotation_total = 0
        paired_count = 0
        for image_path in image_paths:
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                missing_labels.append(image_path.resolve().as_posix())
                continue

            annotation_count = validate_label(label_path)
            annotation_total += annotation_count
            paired_count += 1
            samples.append(
                Sample(
                    style=style_dir.name,
                    image=image_path.resolve(),
                    label=label_path.resolve(),
                    annotation_count=annotation_count,
                )
            )

        for label_path in label_paths:
            if label_path.stem not in images_by_stem:
                orphan_labels.append(label_path.resolve().as_posix())

        style_stats[style_dir.name] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "paired": paired_count,
            "annotations": annotation_total,
            "excluded_missing_label": len(image_paths) - paired_count,
        }

    if not samples:
        raise ValueError("没有找到任何有效的图片/标签配对。")
    if orphan_labels:
        preview = "\n".join(orphan_labels[:10])
        raise ValueError(f"发现无对应图片的标签文件：\n{preview}")

    audit = {
        "source": source.resolve().as_posix(),
        "model_class": {"0": MODEL_CLASS_NAME},
        "source_class_names": source_class_names,
        "style_stats": style_stats,
        "valid_images": len(samples),
        "annotations": sum(sample.annotation_count for sample in samples),
        "excluded_images_without_labels": missing_labels,
        "orphan_labels": orphan_labels,
    }
    return samples, audit


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    """Split every style independently so all known styles occur in each split."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.style].append(sample)

    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for style, style_samples in sorted(grouped.items()):
        shuffled = list(style_samples)
        random.Random(f"{seed}:{style}").shuffle(shuffled)
        count = len(shuffled)

        val_count = max(1, round(count * val_ratio)) if val_ratio else 0
        test_count = max(1, round(count * test_ratio)) if test_ratio else 0
        while val_count + test_count >= count:
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                break

        splits["val"].extend(shuffled[:val_count])
        splits["test"].extend(shuffled[val_count : val_count + test_count])
        splits["train"].extend(shuffled[val_count + test_count :])

    for split_index, split_name in enumerate(("train", "val", "test")):
        random.Random(seed + split_index).shuffle(splits[split_name])
    return splits


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def write_outputs(
    project_root: Path,
    output: Path,
    splits: dict[str, list[Sample]],
    audit: dict[str, object],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_style_counts: dict[str, dict[str, int]] = {}
    for split_name, split_samples_for_name in splits.items():
        lines = [sample.image.as_posix() for sample in split_samples_for_name]
        content = "\n".join(lines) + ("\n" if lines else "")
        write_text_atomic(split_dir / f"{split_name}.txt", content)

        counts: dict[str, int] = defaultdict(int)
        for sample in split_samples_for_name:
            counts[sample.style] += 1
        split_style_counts[split_name] = dict(sorted(counts.items()))

    try:
        output_relative = output.resolve().relative_to(project_root.resolve())
    except ValueError:
        output_relative = output.resolve()

    yaml_text = (
        f"path: {json.dumps(project_root.resolve().as_posix(), ensure_ascii=False)}\n"
        f"train: {json.dumps((output_relative / 'splits/train.txt').as_posix(), ensure_ascii=False)}\n"
        f"val: {json.dumps((output_relative / 'splits/val.txt').as_posix(), ensure_ascii=False)}\n"
        f"test: {json.dumps((output_relative / 'splits/test.txt').as_posix(), ensure_ascii=False)}\n"
        "names:\n"
        f"  0: {MODEL_CLASS_NAME}\n"
    )
    write_text_atomic(output / "meter.yaml", yaml_text)

    audit["split"] = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": {name: len(items) for name, items in splits.items()},
        "style_counts": split_style_counts,
    }
    report_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    write_text_atomic(output / "audit_report.json", report_text)


def main() -> None:
    args = parse_args()
    validate_ratios(args.val_ratio, args.test_ratio)
    project_root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    output = args.output.resolve()

    samples, audit = collect_samples(source)
    splits = split_samples(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_outputs(
        project_root=project_root,
        output=output,
        splits=splits,
        audit=audit,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    split_counts = ", ".join(
        f"{name}={len(items)}" for name, items in splits.items()
    )
    excluded_count = len(audit["excluded_images_without_labels"])
    print(f"有效图片：{len(samples)}；标注框：{audit['annotations']}")
    print(f"分层划分：{split_counts}")
    print(f"因缺少标签而排除：{excluded_count} 张")
    print(f"数据配置：{output / 'meter.yaml'}")
    print(f"审计报告：{output / 'audit_report.json'}")


if __name__ == "__main__":
    main()

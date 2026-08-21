"""Compatibility audit for MaomaoMAo-17/Gauge-Pointer-Reading.

This does not train either model and does not copy the upstream geometry code.
It measures whether the pretrained MIT segmentation checkpoint produces both
required masks on our manifest images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from style_classifier.manifest import parse_markdown_manifest, valid_unique_entries

from .frozen_detector import FrozenGaugeDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("../all_set"))
    parser.add_argument("--manifest", type=Path, default=Path("all_set/仪表盘读数标注.md"))
    parser.add_argument("--detector", type=Path, default=Path("runs/detect/meter_yolov8n_final/weights/best.pt"))
    parser.add_argument("--segmenter", type=Path, default=Path("third_party/Gauge-Pointer-Reading/scale_segment.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/style_reader/third_party_segmentation_audit.json"))
    args = parser.parse_args()

    detector = FrozenGaugeDetector(args.detector)
    segmenter = YOLO(str(args.segmenter))
    entries = valid_unique_entries(parse_markdown_manifest(args.manifest, args.dataset_root))
    rows = []
    for entry in entries:
        crop, detection = detector.crop(Path(entry.effective_absolute_path))
        pointer_count = scale_count = 0
        if crop is not None:
            result = segmenter.predict(crop, conf=0.25, imgsz=640, verbose=False)[0]
            classes = [] if result.boxes is None else [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
            mask_count = 0 if result.masks is None else len(result.masks)
            if mask_count:
                pointer_count = classes.count(0)
                scale_count = classes.count(1)
        rows.append(
            {
                "path": entry.relative_path,
                "detector_status": detection["status"],
                "pointer_masks": pointer_count,
                "scale_masks": scale_count,
                "direct_upstream_geometry_usable": pointer_count > 0 and scale_count > 0,
            }
        )
    report = {
        "upstream": "https://github.com/MaomaoMAo-17/Gauge-Pointer-Reading",
        "license": "MIT",
        "segmenter_trained_or_modified": False,
        "image_count": len(rows),
        "pointer_coverage": sum(row["pointer_masks"] > 0 for row in rows) / len(rows),
        "scale_coverage": sum(row["scale_masks"] > 0 for row in rows) / len(rows),
        "direct_geometry_coverage": sum(row["direct_upstream_geometry_usable"] for row in rows) / len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


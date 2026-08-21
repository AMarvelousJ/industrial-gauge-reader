from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def format_value(value: object, unit: object) -> str:
    if value is None:
        return "no_output"
    return f"{float(value):.3f} {unit}".rstrip("0").rstrip(".")


def fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    left = (width - resized.shape[1]) // 2
    top = (height - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-image gauge-reading report and contact sheet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.output_dir / "results.json").read_text(encoding="utf-8"))
    predictions = json.loads((args.output_dir / "predictions.json").read_text(encoding="utf-8"))["predictions"]
    evaluation = json.loads((args.output_dir / "evaluation.json").read_text(encoding="utf-8"))
    prediction_by_id = {row["sample_id"]: row for row in predictions}
    evaluation_by_id = {row["sample_id"]: row for row in evaluation["samples"]}

    tile_width, tile_height, image_height = 420, 470, 370
    tiles = []
    markdown_rows = []
    for row in results["results"]:
        sample_id = row["sample_id"]
        prediction = prediction_by_id[sample_id]
        scored = evaluation_by_id[sample_id]
        source = cv2.imread(row["effective_path"], cv2.IMREAD_COLOR)
        left, top, right, bottom = row["detector"]["box_xyxy"]
        crop = source[top:bottom, left:right].copy()
        geometry = row.get("geometry") or {}
        if geometry:
            analysis_scale = float(geometry.get("analysis_scale") or 1.0)
            override = geometry.get("ocr_geometry_override")
            if override:
                center = tuple(float(value) for value in override["center"])
                radius = float(override["radius"])
            else:
                circle = geometry["circle"]
                center = (circle["center_x"] / analysis_scale, circle["center_y"] / analysis_scale)
                radius = circle["radius"] / analysis_scale
            angle = geometry.get("angle_degrees_clockwise_from_top")
            if angle is not None:
                radians = math.radians(float(angle))
                tip = (
                    round(center[0] + radius * 0.72 * math.sin(radians)),
                    round(center[1] - radius * 0.72 * math.cos(radians)),
                )
                cv2.line(crop, (round(center[0]), round(center[1])), tip, (255, 0, 255), max(3, round(radius * 0.012)))
                cv2.circle(crop, (round(center[0]), round(center[1])), max(4, round(radius * 0.018)), (255, 0, 255), -1)
        tile = np.full((tile_height, tile_width, 3), 250, dtype=np.uint8)
        tile[:image_height] = fit_tile(crop, tile_width, image_height)
        predicted_text = format_value(prediction.get("value"), prediction.get("unit")) if prediction["status"] == "ok" else prediction["status"]
        truth_text = format_value(scored["truth_value"], scored["truth_unit"])
        verdict = "PASS" if scored["correct"] else "FAIL"
        color = (35, 150, 35) if scored["correct"] else (30, 30, 210)
        cv2.putText(tile, f"{sample_id}  {Path(row['relative_path']).name}", (10, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(tile, f"PRED: {predicted_text}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        cv2.putText(tile, f"GT: {truth_text}  [{verdict}]", (10, 451), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        tiles.append(tile)
        markdown_rows.append(
            f"| {sample_id} | `{Path(row['relative_path']).name}` | {predicted_text} | {truth_text} | {verdict} |"
        )

    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 235, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y * tile_height : (y + 1) * tile_height, x * tile_width : (x + 1) * tile_width] = tile
    contact_path = args.output_dir / "reading_predictions_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    primary = evaluation["protocols"]["audited_unique_20"]
    strict = evaluation["protocols"]["strict_18"]
    report = [
        "# Pointer gauge reading evaluation",
        "",
        f"- Audited unique protocol: **{primary['correct']}/{primary['total']} = {primary['accuracy']:.1%}**, coverage {primary['coverage']:.1%}.",
        f"- Strict as-written protocol: **{strict['correct']}/{strict['total']} = {strict['accuracy']:.1%}**, coverage {strict['coverage']:.1%}.",
        "- YOLO detector and third-party pointer segmenter were inference-only; neither was trained or modified.",
        "- Ground truth is used only by this evaluator/report, never by the reading pipeline.",
        "",
        "| ID | Image | Prediction | Ground truth | Result |",
        "|---|---|---:|---:|---|",
        *markdown_rows,
        "",
        "Magenta line in the contact sheet is the final pointer direction used for reading.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str((args.output_dir / 'report.md').resolve()), "contact_sheet": str(contact_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

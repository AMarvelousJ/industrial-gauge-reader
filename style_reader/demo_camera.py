"""RK3576 摄像头实时仪表读数 Demo.

统一入口:复用 style_reader.run_manifest.process_image —— 不复制任何识别逻辑。
流程:
    camera frame -> process_image(entry) [YOLO检测->几何->指针->OCR->mapping->unit]
                    -> overlay(框/指针/值/单位/置信/FPS) -> 实时显示

用法:
    python demo_camera.py                        # 默认摄像头 /dev/video0 (V4L2)
    python demo_camera.py --camera 1             # 指定摄像头索引
    python demo_camera.py --image-dir assets/demo_images   # 无摄像头图片回放
    python demo_camera.py --width 640 --height 480 --max-fps 10

退出:q / ESC
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from style_reader.frozen_detector import FrozenGaugeDetector
from style_reader.ocr_mapping import OCRScaleReader
from style_reader.run_manifest import process_image, segmented_pointer_angle
from pointer_keypoints.inference import PointerKeypointEstimator


def _default_weights() -> tuple[Path, Path | None, Path | None]:
    base = Path(__file__).resolve().parent.parent
    detector = base / "models" / "meter_detector.pt"
    segmenter = base / "models" / "scale_segment.pt"
    keypoints = base / "models" / "pointer_keypoints.pt"
    # Development-machine layout fallback (packaging-independent paths).
    if not detector.is_file():
        detector = base / "runs" / "detect" / "meter_yolov8n_final" / "weights" / "best.pt"
    if not segmenter.is_file():
        segmenter = base / "third_party" / "Gauge-Pointer-Reading" / "scale_segment.pt"
    if not keypoints.is_file():
        keypoints = base / "runs" / "pose" / "runs" / "pointer_keypoints" / "industrial_single_pointer_v1" / "weights" / "best.pt"
    return detector, segmenter, (keypoints if keypoints.is_file() else None)


def build_context(args: argparse.Namespace) -> dict:
    detector, segmenter, keypoints = _default_weights()
    if not detector.is_file():
        raise FileNotFoundError(f"YOLO detection model not found: {detector}")
    if not segmenter.is_file():
        raise FileNotFoundError(f"Pointer segmenter not found: {segmenter}")
    segmenter_model = __import__("ultralytics").YOLO(str(segmenter))
    threshold = 0.75
    threshold_file = Path(__file__).resolve().parent.parent / "models" / "keypoint_threshold.json"
    if threshold_file.is_file():
        threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["confidence_threshold"])
    keypoint_estimator = PointerKeypointEstimator(keypoints, confidence_threshold=threshold) if keypoints else None
    return {
        "detector": FrozenGaugeDetector(detector, confidence=0.25, image_size=640),
        "segmenter": segmenter_model,
        "keypoint_estimator": keypoint_estimator,
        "ocr_reader": OCRScaleReader(),
        "threshold": threshold,
    }


def overlay_result(frame: np.ndarray, row: dict, elapsed_ms: float) -> np.ndarray:
    canvas = frame.copy()
    geometry = (row.get("geometry") or {})
    detection = (row.get("detector") or {})
    # meter box (from YOLO detection: box in source coords; canvas == source crop frame)
    box = detection.get("box_xyxy")
    if box:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(canvas, "meter", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    # pointer direction
    angle = geometry.get("angle_degrees_clockwise_from_top")
    circ = geometry.get("circle") or {}
    scale = float(geometry.get("analysis_scale", 1.0) or 1.0)
    if angle is not None and circ.get("center_x") is not None:
        cx, cy = float(circ["center_x"]) / scale, float(circ["center_y"]) / scale
        radius = float(circ.get("radius", min(canvas.shape[:2]) * 0.35)) / scale
        a = math.radians(float(angle))
        tip = (int(cx + radius * 0.7 * math.sin(a)), int(cy - radius * 0.7 * math.cos(a)))
        cv2.arrowedLine(canvas, (int(cx), int(cy)), tip, (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(canvas, (int(cx), int(cy)), 4, (0, 255, 255), -1)
    # reading + confidence + fps
    value = geometry.get("reading")
    unit = geometry.get("unit")
    method = geometry.get("mapping_method", geometry.get("pointer_method", ""))
    conf = geometry.get("mapping_confidence", geometry.get("pointer_confidence", 0.0))
    text = (f"READ: {value:.3f} {unit}" if value is not None else "READ: (none)") + f"  [{method} conf={conf:.2f}]"
    cv2.rectangle(canvas, (0, 0), (min(720, canvas.shape[1]), 54), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"FPS: {1000.0 / max(elapsed_ms, 1e-3):.1f}  ({elapsed_ms:.0f} ms/frame)",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
    return canvas


def run_camera(args: argparse.Namespace, ctx: dict) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {args.camera}; use --image-dir for replay")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    window = f"Gauge Demo (q/ESC) - analyze-every {args.analyze_every} (display stays live)"
    print("camera ready. press q / ESC to exit")
    frame_index = 0
    last_row = None
    last_analytic_ms = 0.0
    last_update_frames = 0
    show_start = time.time()
    show_frames = 0
    live_fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue
        frame_index += 1
        show_frames += 1
        now = time.time()
        if now - show_start >= 1.0:
            live_fps = show_frames / (now - show_start)
            show_start = now
            show_frames = 0
        # periodic full analysis (slow OCR) - other frames keep LIVE display
        analyze_this = ((frame_index - 1) % max(1, args.analyze_every)) == 0
        if analyze_this:
            t0 = time.perf_counter()
            entry = {
                "absolute_path": frame,
                "relative_path": f"camera_{frame_index:06d}.jpg",
                "sample_id": f"camera-{frame_index:06d}",
            }
            last_row = process_image(
                entry,
                detector=ctx["detector"],
                segmenter=ctx["segmenter"],
                keypoint_estimator=ctx["keypoint_estimator"],
                ocr_reader=ctx["ocr_reader"],
                args=args,
                output_index=frame_index,
                visual_dir=None,
            )
            last_analytic_ms = (time.perf_counter() - t0) * 1000.0
            last_update_frames = 0
        else:
            last_update_frames += 1
        # LIVE canvas: current frame + stable result overlay + live fps
        canvas = frame.copy()
        if last_row is not None:
            # overlay final reading text/pointer from last analysis (coords may lag)
            g = (last_row.get("geometry") or {})
            value = g.get("reading")
            unit = g.get("unit")
            family = (g.get("family") or {}).get("label", "?")
            method = g.get("mapping_method", g.get("pointer_method", ""))
            conf = g.get("mapping_confidence", g.get("pointer_confidence", 0.0))
            text = (f"READ: {value:.3f} {unit}" if value is not None else "READ: (none)") + f" [{method} conf={conf:.2f} family={family}]"
            cv2.rectangle(canvas, (0, 0), (min(760, canvas.shape[1]), 56), (0, 0, 0), -1)
            cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            cv2.putText(canvas, f"live FPS: {live_fps:.1f} | update: {last_update_frames} frames ago | analyze: {last_analytic_ms/1000.0:.1f}s",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)
            # last detection box (lagging) + pointer line from last analysis
            box = (last_row.get("detector") or {}).get("box_xyxy")
            if box:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
            angle = g.get("angle_degrees_clockwise_from_top")
            circ = g.get("circle") or {}
            sc = float(g.get("analysis_scale", 1.0) or 1.0)
            if angle is not None and circ.get("center_x") is not None:
                import math as _m
                cx, cy = float(circ["center_x"]) / sc, float(circ["center_y"]) / sc
                radius = float(circ.get("radius", min(canvas.shape[:2]) * 0.35)) / sc
                a = _m.radians(float(angle))
                tip = (int(cx + radius * 0.7 * _m.sin(a)), int(cy - radius * 0.7 * _m.cos(a)))
                cv2.arrowedLine(canvas, (int(cx), int(cy)), tip, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.22)
        else:
            cv2.rectangle(canvas, (0, 0), (320, 44), (0, 0, 0), -1)
            cv2.putText(canvas, "initializing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, canvas)
        if args.max_fps > 0:
            time.sleep(max(0.0, 1.0 / args.max_fps - 0.005))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_image_dir(args: argparse.Namespace, ctx: dict) -> None:
    image_dir = Path(args.image_dir)
    paths = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise FileNotFoundError(f"no images in {image_dir}")
    print(f"replay mode: {len(paths)} images from {image_dir}")
    window = "Gauge Demo (replay; n/SPACE next, q/ESC quit)"
    index = 0
    while index < len(paths):
        path = paths[index]
        t0 = time.perf_counter()
        row = process_image(
            {"absolute_path": path, "relative_path": path.name, "sample_id": f"replay-{index:04d}"},
            detector=ctx["detector"],
            segmenter=ctx["segmenter"],
            keypoint_estimator=ctx["keypoint_estimator"],
            ocr_reader=ctx["ocr_reader"],
            args=args,
            output_index=index + 1,
            visual_dir=None,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        frame = cv2.imread(str(path))
        if frame is None:
            index += 1
            continue
        canvas = overlay_result(frame, row, elapsed)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        index += 1
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3576 gauge reading camera demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-fps", type=float, default=0.0)
    parser.add_argument("--analyze-every", type=int, default=4, help="run full analysis every N frames; display stays live in between")
    parser.add_argument("--visualize", action="store_true")
    # fields consumed by the shared process_image()
    parser.add_argument("--frame-line-override", action="store_true")
    parser.add_argument("--keypoint-agreement-degrees", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    if args.image_dir is not None:
        run_image_dir(args, ctx)
    else:
        run_camera(args, ctx)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class FrozenGaugeDetector:
    """Read-only wrapper around the user-supplied one-class YOLO detector."""

    def __init__(self, weights: Path, confidence: float = 0.25, image_size: int = 640, padding: float = 0.04):
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.weights = weights.resolve()
        self.confidence = confidence
        self.image_size = image_size
        self.padding = padding
        self.model = YOLO(str(self.weights))

    def crop(self, image_path: Path) -> tuple[np.ndarray | None, dict]:
        results = self.model.predict(
            source=str(image_path), conf=self.confidence, imgsz=self.image_size, max_det=1, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, {"status": "detector_miss", "confidence": None, "box_xyxy": None}
        result = results[0]
        box = result.boxes[0]
        left, top, right, bottom = [float(value) for value in box.xyxy[0].detach().cpu().tolist()]
        width, height = right - left, bottom - top
        image = result.orig_img
        left = max(0, round(left - width * self.padding))
        top = max(0, round(top - height * self.padding))
        right = min(image.shape[1], round(right + width * self.padding))
        bottom = min(image.shape[0], round(bottom + height * self.padding))
        if right <= left or bottom <= top:
            return None, {"status": "invalid_detector_box", "confidence": float(box.conf[0]), "box_xyxy": None}
        crop = np.ascontiguousarray(image[top:bottom, left:right])
        return crop, {
            "status": "ok",
            "confidence": round(float(box.conf[0].detach().cpu().item()), 6),
            "box_xyxy": [left, top, right, bottom],
        }


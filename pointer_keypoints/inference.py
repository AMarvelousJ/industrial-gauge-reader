from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .contract import KeypointEstimate, clockwise_angle_from_top


class PointerKeypointEstimator:
    def __init__(
        self,
        weights: Path,
        *,
        confidence_threshold: float = 0.5,
        image_size: int = 384,
        minimum_length_ratio: float = 0.12,
        maximum_length_ratio: float = 0.75,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be within (0, 1)")
        self.weights = weights.resolve()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.minimum_length_ratio = minimum_length_ratio
        self.maximum_length_ratio = maximum_length_ratio
        self.disambiguate_tip = True
        self.model = YOLO(str(self.weights))
        if getattr(self.model, "task", None) != "pose":
            raise ValueError("pointer keypoint weights must be an Ultralytics pose model")

    @staticmethod
    def _thin_needle_extent(gray: np.ndarray, pivot: tuple[float, float], angle_degrees: float, max_radius: float) -> float:
        """Contiguous thin-dark-line extent from the pivot along a direction.

        Analog needles are thin lines; counterweights / broad arrow tails are
        short and thick. Walking outward, a thin dark line (small perpendicular
        dark count) extends the measure; a thick object stops it. This is the
        "long thin end = reading tip" disambiguation from reading_method_design.
        """
        dark = gray < 118
        height, width = gray.shape
        angle = math.radians(angle_degrees)
        extent = 0.0
        gap = 0
        radius = 1.0
        while radius < max_radius:
            x = pivot[0] + radius * math.sin(angle)
            y = pivot[1] - radius * math.cos(angle)
            xi, yi = int(round(x)), int(round(y))
            if not (3 <= xi < width - 3 and 3 <= yi < height - 3):
                break
            patch = dark[yi - 3 : yi + 4, xi - 3 : xi + 4]
            if patch.any():
                if int(patch.sum()) <= 6:
                    extent = radius
                    gap = 0
                else:
                    gap += 1
                    if gap > 4:
                        break
            else:
                gap += 1
                if gap > 3:
                    break
            radius += 0.5
        return extent

    def _disambiguate_tip_direction(
        self, crop: np.ndarray, pivot: tuple[float, float], tip: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Flip the predicted tip to the longer thin end when the model fell for a
        broad counterweight arrow / shadow pointing the opposite way."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        px, py = pivot
        tx, ty = tip
        if math.dist((px, py), (tx, ty)) <= 1e-9:
            return None
        angle = math.degrees(math.atan2(tx - px, -(ty - py))) % 360.0
        diameter = float(min(crop.shape[:2]))
        max_radius = diameter * 0.5
        forward = self._thin_needle_extent(gray, pivot, angle, max_radius)
        backward = self._thin_needle_extent(gray, pivot, (angle + 180.0) % 360.0, max_radius)
        if backward > forward * 1.5 and backward >= 0.08 * diameter:
            # keep the pivot and the tip length, reverse the direction
            return (2.0 * px - tx, 2.0 * py - ty)
        return None

    def predict(self, crop: np.ndarray) -> KeypointEstimate:
        if crop is None or crop.size == 0:
            raise ValueError("crop must be a non-empty image")
        result = self.model.predict(crop, imgsz=self.image_size, conf=0.05, max_det=1, verbose=False)[0]
        keypoints = result.keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return KeypointEstimate("no_output", None, None, None, None, None, None, self.confidence_threshold, rejection_reason="no_pose_detection")
        xy = keypoints.xy[0].detach().cpu().numpy()
        confidence = None if keypoints.conf is None else keypoints.conf[0].detach().cpu().numpy()
        if xy.shape[0] != 2 or confidence is None or confidence.shape[0] != 2:
            return KeypointEstimate("no_output", None, None, None, None, None, None, self.confidence_threshold, rejection_reason="expected_two_keypoints")
        pivot = (float(xy[0, 0]), float(xy[0, 1]))
        tip = (float(xy[1, 0]), float(xy[1, 1]))
        pivot_confidence, tip_confidence = float(confidence[0]), float(confidence[1])
        if self.disambiguate_tip:
            corrected = self._disambiguate_tip_direction(crop, pivot, tip)
            if corrected is not None:
                tip = corrected
        diameter = float(min(crop.shape[:2]))
        length_ratio = math.dist(pivot, tip) / max(diameter, 1e-9)
        reason = None
        if min(pivot_confidence, tip_confidence) < self.confidence_threshold:
            reason = "confidence_below_threshold"
        elif not self.minimum_length_ratio <= length_ratio <= self.maximum_length_ratio:
            reason = "pointer_length_out_of_range"
        elif not (0 <= pivot[0] < crop.shape[1] and 0 <= pivot[1] < crop.shape[0] and 0 <= tip[0] < crop.shape[1] and 0 <= tip[1] < crop.shape[0]):
            reason = "keypoint_outside_crop"
        status = "accepted" if reason is None else "rejected"
        angle = clockwise_angle_from_top(pivot, tip)
        return KeypointEstimate(
            status,
            pivot,
            tip,
            pivot_confidence,
            tip_confidence,
            angle,
            length_ratio,
            self.confidence_threshold,
            rejection_reason=reason,
        )

from __future__ import annotations

import math
from pathlib import Path

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
        self.model = YOLO(str(self.weights))
        if getattr(self.model, "task", None) != "pose":
            raise ValueError("pointer keypoint weights must be an Ultralytics pose model")

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

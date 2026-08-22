from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence


KEYPOINT_NAMES = ("pivot", "pointer_tip")
TRAINING_TRACKS = ("company_priority", "generalization_guardrail")
COMPLETED_STATUSES = ("accepted", "corrected")


def clockwise_angle_from_top(pivot: Sequence[float], tip: Sequence[float]) -> float:
    """Return the reader's angle convention: top=0, right=90, bottom=180."""
    dx = float(tip[0]) - float(pivot[0])
    dy = float(tip[1]) - float(pivot[1])
    if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) <= 1e-9:
        raise ValueError("pivot and pointer tip must form a finite non-zero vector")
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def circular_distance_degrees(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class KeypointEstimate:
    status: str
    pivot: tuple[float, float] | None
    pointer_tip: tuple[float, float] | None
    pivot_confidence: float | None
    pointer_tip_confidence: float | None
    angle_degrees_clockwise_from_top: float | None
    length_ratio: float | None
    confidence_threshold: float
    coordinate_system: str = "detector_crop"
    rejection_reason: str | None = None

    @property
    def confidence(self) -> float | None:
        values = (self.pivot_confidence, self.pointer_tip_confidence)
        if any(value is None for value in values):
            return None
        return min(float(values[0]), float(values[1]))  # type: ignore[arg-type]

    def with_points(
        self,
        pivot: Sequence[float],
        pointer_tip: Sequence[float],
        *,
        coordinate_system: str,
        dial_diameter: float,
    ) -> "KeypointEstimate":
        pivot_tuple = (float(pivot[0]), float(pivot[1]))
        tip_tuple = (float(pointer_tip[0]), float(pointer_tip[1]))
        length = math.dist(pivot_tuple, tip_tuple)
        return replace(
            self,
            pivot=pivot_tuple,
            pointer_tip=tip_tuple,
            angle_degrees_clockwise_from_top=clockwise_angle_from_top(pivot_tuple, tip_tuple),
            length_ratio=length / max(float(dial_diameter), 1e-9),
            coordinate_system=coordinate_system,
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["confidence"] = self.confidence
        return payload

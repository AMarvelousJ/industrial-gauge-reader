from __future__ import annotations

import pytest

from pointer_keypoints.contract import (
    KeypointEstimate,
    circular_distance_degrees,
    clockwise_angle_from_top,
)


@pytest.mark.parametrize(
    ("tip", "angle"),
    [((0.0, -1.0), 0.0), ((1.0, 0.0), 90.0), ((0.0, 1.0), 180.0), ((-1.0, 0.0), 270.0)],
)
def test_clockwise_angle_from_top_cardinal_directions(tip, angle) -> None:
    assert clockwise_angle_from_top((0.0, 0.0), tip) == pytest.approx(angle)


def test_circular_distance_handles_zero_wrap() -> None:
    assert circular_distance_degrees(359.0, 1.0) == pytest.approx(2.0)


def test_estimate_can_be_transformed_to_canonical_space() -> None:
    estimate = KeypointEstimate("accepted", (10, 10), (10, 0), 0.9, 0.8, 0.0, 0.1, 0.5)
    transformed = estimate.with_points((100, 100), (120, 100), coordinate_system="canonical", dial_diameter=200)
    assert transformed.coordinate_system == "canonical"
    assert transformed.angle_degrees_clockwise_from_top == pytest.approx(90.0)
    assert transformed.length_ratio == pytest.approx(0.1)
    assert transformed.confidence == pytest.approx(0.8)

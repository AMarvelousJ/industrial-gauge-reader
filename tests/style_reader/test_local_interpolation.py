from __future__ import annotations

import pytest

from style_reader.ocr_mapping import interpolate_reading_at_anchors


def _points(raw: list[tuple[float, float]]) -> list[dict]:
    # raw entries are (value, angle) pairs in dial order (angles in [0,360))
    return [{"value": value, "angle": angle, "text": str(round(value)), "score": 0.99} for value, angle in raw]


def test_interpolates_inside_bracket_rg018_case() -> None:
    # RG-018 anchors (as %360): 0@318.2, 10@341.3, 20@1.4, 30@19.8, 40@34.7, 50@48.0
    points = _points([(0.0, 318.23), (10.0, 341.33), (20.0, 1.43), (30.0, 19.76), (40.0, 34.65), (50.0, 48.04)])
    result = interpolate_reading_at_anchors(points, 325.2333)
    assert result is not None
    assert result["status"] == "ok"
    assert result["method"] == "local_anchor_interpolation"
    assert 2.5 <= result["reading"] <= 3.6  # ~3.0 Pa, interpolation only


def test_wrap_around_unwraps_monotonically() -> None:
    # values 0@294.6, 10@314.3, 20@337.0, 60@426.5(66.5+360); pointer at 320
    points = _points([(0.0, 294.58), (10.0, 314.33), (20.0, 337.02), (60.0, 66.47)])
    result = interpolate_reading_at_anchors(points, 320.0)
    assert result is not None
    assert result["anchor_values"] == [10.0, 20.0]
    assert 12.0 < result["reading"] < 14.0


def test_outside_bracket_returns_none() -> None:
    points = _points([(0.0, 294.58), (10.0, 314.33), (20.0, 337.02)])
    assert interpolate_reading_at_anchors(points, 280.0) is None  # before the 0 anchor


def test_single_anchor_returns_none() -> None:
    assert interpolate_reading_at_anchors(_points([(0.0, 300.0)]), 310.0) is None


def test_none_pointer_returns_none() -> None:
    assert interpolate_reading_at_anchors(_points([(0.0, 300.0), (10.0, 330.0)]), None) is None


def test_duplicate_values_dual_ring_return_none() -> None:
    # two rings interleaved: value 0 appears twice -> ambiguous, reject
    points = _points([(0.0, 300.0), (0.0, 340.0), (10.0, 320.0)])
    assert interpolate_reading_at_anchors(points, 310.0) is None


def test_pointer_below_first_anchor_returns_none() -> None:
    points = _points([(0.0, 294.58), (10.0, 314.33), (20.0, 337.02), (30.0, 361.45), (40.0, 385.15), (50.0, 408.3), (60.0, 426.47)])
    assert interpolate_reading_at_anchors(points, 260.0) is None

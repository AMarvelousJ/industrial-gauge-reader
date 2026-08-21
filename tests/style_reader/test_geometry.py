import cv2
import numpy as np

from style_reader.geometry import analyze_pointer, clockwise_angle_degrees


def test_clockwise_angle_convention() -> None:
    center = (100.0, 100.0)
    assert clockwise_angle_degrees(center, (100.0, 10.0)) == 0.0
    assert clockwise_angle_degrees(center, (190.0, 100.0)) == 90.0
    assert clockwise_angle_degrees(center, (100.0, 190.0)) == 180.0
    assert clockwise_angle_degrees(center, (10.0, 100.0)) == 270.0


def test_synthetic_pointer_produces_auditable_angle() -> None:
    image = np.full((500, 500, 3), 245, dtype=np.uint8)
    center = (250, 250)
    cv2.circle(image, center, 215, (30, 30, 30), 5)
    for angle in range(0, 360, 30):
        radians = np.deg2rad(angle)
        inner = (round(center[0] + 185 * np.sin(radians)), round(center[1] - 185 * np.cos(radians)))
        outer = (round(center[0] + 205 * np.sin(radians)), round(center[1] - 205 * np.cos(radians)))
        cv2.line(image, inner, outer, (20, 20, 20), 4)
    target_angle = 60.0
    radians = np.deg2rad(target_angle)
    tip = (round(center[0] + 165 * np.sin(radians)), round(center[1] - 165 * np.cos(radians)))
    cv2.line(image, center, tip, (0, 0, 0), 9)
    cv2.circle(image, center, 13, (20, 20, 20), -1)

    result, visualization = analyze_pointer(image)

    assert result["status"] == "angle_estimated"
    measured = result["angle_degrees_clockwise_from_top"]
    circular_error = abs((measured - target_angle + 180) % 360 - 180)
    assert circular_error <= 8.0
    assert result["reading"] is None
    assert result["reading_status"] == "calibration_required"
    assert visualization.shape[:2] == image.shape[:2]


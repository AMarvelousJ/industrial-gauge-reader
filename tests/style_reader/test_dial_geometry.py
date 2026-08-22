import json

import cv2
import numpy as np

from style_reader.dial_geometry import (
    adapt_circle_estimate,
    estimate_dial_geometry,
    geometry_from_ellipse,
    geometry_from_quadrilateral,
)
from style_reader.geometry import CircleEstimate


def _round_trip_error(geometry, points: np.ndarray) -> float:
    canonical = geometry.source_to_canonical(points)
    restored = geometry.canonical_to_source(canonical)
    return float(np.max(np.linalg.norm(restored - points, axis=1)))


def test_circle_estimate_adapter_and_serializable_diagnostics() -> None:
    circle = CircleEstimate(220.0, 180.0, 140.0, "hough_circle", 0.88)

    geometry = adapt_circle_estimate(circle, (400, 500, 3), canonical_size=512)

    assert geometry.geometry_type == "front_circle"
    assert geometry.source_point_to_canonical((220.0, 180.0)) == (256.0, 256.0)
    assert _round_trip_error(geometry, np.array([[80.0, 180.0], [220.0, 40.0], [350.0, 250.0]])) < 1e-6
    encoded = json.dumps(geometry.as_dict())
    assert '"forward_transform"' in encoded
    assert '"canonical_scale_curve"' in encoded


def test_legacy_roi_circle_adapter_remains_an_explicit_fallback() -> None:
    circle = CircleEstimate(200.0, 150.0, 138.0, "roi_fallback", 0.25)

    geometry = adapt_circle_estimate(circle, (300, 400, 3))

    assert geometry.geometry_type == "roi_fallback"
    assert geometry.confidence <= 0.10
    assert geometry.fallback_reason


def test_ellipse_rectification_maps_boundary_to_canonical_circle() -> None:
    geometry = geometry_from_ellipse(
        center=(310.0, 220.0),
        axes=(360.0, 220.0),
        rotation_degrees=27.0,
        confidence=0.9,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 25)
    rotation = np.array(
        [
            [np.cos(np.deg2rad(27.0)), -np.sin(np.deg2rad(27.0))],
            [np.sin(np.deg2rad(27.0)), np.cos(np.deg2rad(27.0))],
        ]
    )
    source = np.array([310.0, 220.0]) + np.column_stack((180.0 * np.cos(angles), 110.0 * np.sin(angles))) @ rotation.T

    canonical = geometry.source_to_canonical(source)
    radii = np.linalg.norm(canonical - np.array(geometry.canonical_pivot), axis=1)

    assert geometry.geometry_type == "perspective_ellipse"
    assert np.max(np.abs(radii - 512 * 0.45)) < 1e-5
    assert _round_trip_error(geometry, source) < 1e-6


def test_quadrilateral_homography_round_trip_and_corner_mapping() -> None:
    corners = np.array([[90.0, 70.0], [470.0, 105.0], [430.0, 365.0], [55.0, 330.0]])

    geometry = geometry_from_quadrilateral(corners, confidence=0.84, canonical_size=(600, 400))
    canonical_corners = geometry.source_to_canonical(corners)

    assert geometry.geometry_type == "rectangular_sector"
    assert np.allclose(canonical_corners[0], (24.0, 24.0), atol=1e-3)
    assert np.allclose(canonical_corners[2], (575.0, 375.0), atol=1e-3)
    assert geometry.canonical_pivot == (300.0, 328.0)
    assert _round_trip_error(geometry, corners) < 1e-5


def test_estimator_classifies_synthetic_front_circle() -> None:
    image = np.full((520, 520, 3), 245, np.uint8)
    cv2.circle(image, (260, 260), 205, (20, 20, 20), 7)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "front_circle"
    assert geometry.confidence >= 0.70
    assert geometry.reprojection_error < 8.0


def test_estimator_classifies_synthetic_perspective_ellipse() -> None:
    image = np.full((480, 640, 3), 245, np.uint8)
    cv2.ellipse(image, (320, 240), (245, 145), 18, 0, 360, (20, 20, 20), 7)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "perspective_ellipse"
    assert geometry.confidence >= 0.70
    assert geometry.reprojection_error < 8.0
    assert _round_trip_error(geometry, np.array([[320.0, 240.0], [200.0, 190.0], [480.0, 300.0]])) < 1e-5


def test_estimator_classifies_synthetic_quadrilateral() -> None:
    image = np.full((500, 700, 3), 245, np.uint8)
    corners = np.array([[115, 80], [610, 110], [570, 425], [80, 390]], np.int32)
    cv2.polylines(image, [corners], True, (20, 20, 20), 8)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "rectangular_sector"
    assert geometry.confidence >= 0.70
    assert geometry.reprojection_error < 8.0


def test_estimator_recovers_rectangular_sector_from_crop_clipped_frame() -> None:
    """Regression for a detector crop that cuts the bezel into open contours."""
    image = np.full((520, 600, 3), 238, np.uint8)
    # Three long, slightly perspective-distorted bezel sides remain visible;
    # the lower edge is outside the crop/occluded by the instrument housing.
    cv2.line(image, (4, 35), (596, 54), (18, 18, 18), 11)
    cv2.line(image, (23, 28), (49, 519), (18, 18, 18), 11)
    cv2.line(image, (580, 48), (552, 519), (18, 18, 18), 11)
    cv2.rectangle(image, (245, 360), (599, 519), (45, 45, 45), -1)
    for index in range(9):
        x = 105 + index * 45
        cv2.line(image, (x, 180), (x + 12, 215), (30, 30, 30), 4)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "rectangular_sector"
    assert geometry.method == "open_frame_hough_envelope"
    assert geometry.confidence >= 0.70


def test_open_frame_fallback_does_not_override_obvious_round_dial() -> None:
    image = np.full((520, 520, 3), 245, np.uint8)
    cv2.circle(image, (260, 260), 215, (20, 20, 20), 8)
    cv2.line(image, (260, 260), (380, 150), (20, 20, 20), 8)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "front_circle"
    assert geometry.method == "ellipse_affine_rectification"


def test_blank_image_returns_explicit_low_confidence_fallback() -> None:
    image = np.full((300, 500, 3), 255, np.uint8)

    geometry = estimate_dial_geometry(image)

    assert geometry.geometry_type == "roi_fallback"
    assert geometry.confidence <= 0.10
    assert geometry.fallback_reason
    assert _round_trip_error(geometry, np.array([[0.0, 0.0], [250.0, 150.0], [499.0, 299.0]])) < 1e-5

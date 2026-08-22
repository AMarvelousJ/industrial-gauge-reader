"""Dial-shape normalization primitives for multi-form analog gauges.

The existing reader estimates a circle directly in the detector crop.  That is
an effective fast path for a front-facing round dial, but it makes scale angles
systematically wrong after perspective distortion and on rectangular/sector
dials.  This module keeps shape estimation separate from pointer/scale logic:
it maps a supported source boundary into a canonical coordinate system and
always exposes the inverse mapping for auditable overlays.

No learned weights are used here.  Low-evidence images deliberately produce a
low-confidence ``roi_fallback`` instead of being presented as a detected dial.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np

from .geometry import CircleEstimate


Point = tuple[float, float]
Matrix3x3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _matrix_tuple(matrix: np.ndarray) -> Matrix3x3:
    values = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in values)  # type: ignore[return-value]


def _point_tuple(points: np.ndarray | Iterable[Sequence[float]]) -> tuple[Point, ...]:
    array = np.asarray(tuple(points) if not isinstance(points, np.ndarray) else points, dtype=np.float64)
    array = array.reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in array)


def transform_points(points: Sequence[Sequence[float]] | np.ndarray, transform: Matrix3x3 | np.ndarray) -> np.ndarray:
    """Apply a homography to one point or an ``N x 2`` point collection."""
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 1
    array = array.reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(array, np.asarray(transform, dtype=np.float64)).reshape(-1, 2)
    return mapped[0] if single else mapped


def _circle_curve(center: Point, radius: float, samples: int = 73) -> tuple[Point, ...]:
    # Start at dial top and proceed clockwise, matching clockwise_angle_degrees.
    angles = np.linspace(0.0, 2.0 * math.pi, samples)
    return tuple(
        (
            float(center[0] + radius * math.sin(angle)),
            float(center[1] - radius * math.cos(angle)),
        )
        for angle in angles
    )


def _rectangular_arc_curve(center: Point, radius: float, samples: int = 57) -> tuple[Point, ...]:
    angles = np.deg2rad(np.linspace(-70.0, 70.0, samples))
    return tuple(
        (
            float(center[0] + radius * math.sin(angle)),
            float(center[1] - radius * math.cos(angle)),
        )
        for angle in angles
    )


def _order_quad(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = raw.sum(axis=1)
    differences = np.diff(raw, axis=1).reshape(-1)
    ordered[0] = raw[int(np.argmin(sums))]  # top-left
    ordered[2] = raw[int(np.argmax(sums))]  # bottom-right
    ordered[1] = raw[int(np.argmin(differences))]  # top-right
    ordered[3] = raw[int(np.argmax(differences))]  # bottom-left
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("quadrilateral must contain four distinct corners")
    return ordered


@dataclass(frozen=True)
class SourceBoundary:
    """Serializable source-image boundary model used to derive normalization."""

    kind: str
    points: tuple[Point, ...]
    center: Point | None = None
    axes: Point | None = None
    rotation_degrees: float | None = None


@dataclass(frozen=True)
class DialGeometry:
    """Bidirectional mapping between source crop and normalized dial space."""

    geometry_type: str
    source_boundary: SourceBoundary
    forward_transform: Matrix3x3
    inverse_transform: Matrix3x3
    canonical_size: tuple[int, int]
    canonical_pivot: Point
    canonical_scale_curve: tuple[Point, ...]
    reprojection_error: float
    confidence: float
    method: str
    fallback_reason: str | None = None

    def source_to_canonical(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return transform_points(points, self.forward_transform)

    def canonical_to_source(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return transform_points(points, self.inverse_transform)

    def source_point_to_canonical(self, point: Sequence[float]) -> Point:
        mapped = self.source_to_canonical(np.asarray(point, dtype=np.float64))
        return float(mapped[0]), float(mapped[1])

    def canonical_point_to_source(self, point: Sequence[float]) -> Point:
        mapped = self.canonical_to_source(np.asarray(point, dtype=np.float64))
        return float(mapped[0]), float(mapped[1])

    def warp_to_canonical(self, image_bgr: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr must be a non-empty image")
        return cv2.warpPerspective(
            image_bgr,
            np.asarray(self.forward_transform, dtype=np.float64),
            self.canonical_size,
            flags=interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def as_dict(self) -> dict:
        """Return JSON-safe diagnostics; numpy values never escape this API."""
        return asdict(self)


def adapt_circle_estimate(
    circle: CircleEstimate,
    image_shape: Sequence[int],
    canonical_size: int | tuple[int, int] = 512,
) -> DialGeometry:
    """Adapt the legacy ``CircleEstimate`` into the common geometry contract."""
    if circle.radius <= 0:
        raise ValueError("circle.radius must be positive")
    width, height = _canonical_dimensions(canonical_size)
    canonical_center = (width / 2.0, height / 2.0)
    canonical_radius = min(width, height) * 0.45
    scale = canonical_radius / float(circle.radius)
    forward = np.array(
        [
            [scale, 0.0, canonical_center[0] - scale * circle.center_x],
            [0.0, scale, canonical_center[1] - scale * circle.center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(forward)
    source_curve = _circle_curve((circle.center_x, circle.center_y), circle.radius)
    is_fallback = circle.method == "roi_fallback"
    return DialGeometry(
        geometry_type="roi_fallback" if is_fallback else "front_circle",
        source_boundary=SourceBoundary(
            kind="circle" if not is_fallback else "roi",
            points=source_curve,
            center=(float(circle.center_x), float(circle.center_y)),
            axes=(float(2.0 * circle.radius), float(2.0 * circle.radius)),
            rotation_degrees=0.0,
        ),
        forward_transform=_matrix_tuple(forward),
        inverse_transform=_matrix_tuple(inverse),
        canonical_size=(width, height),
        canonical_pivot=canonical_center,
        canonical_scale_curve=_circle_curve(canonical_center, canonical_radius * 0.9),
        reprojection_error=0.0,
        confidence=(
            min(0.10, max(0.0, float(circle.confidence)))
            if is_fallback
            else float(np.clip(circle.confidence, 0.0, 1.0))
        ),
        method=f"circle_adapter:{circle.method}",
        fallback_reason="legacy circle estimator had no circle evidence" if is_fallback else None,
    )


def geometry_from_ellipse(
    center: Sequence[float],
    axes: Sequence[float],
    rotation_degrees: float,
    *,
    confidence: float,
    reprojection_error: float = 0.0,
    canonical_size: int | tuple[int, int] = 512,
    source_boundary_points: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> DialGeometry:
    """Create an affine rectification from an OpenCV-style fitted ellipse.

    ``axes`` are full diameters, matching ``cv2.fitEllipse``.
    """
    source_center = np.asarray(center, dtype=np.float64).reshape(2)
    diameters = np.asarray(axes, dtype=np.float64).reshape(2)
    if np.any(diameters <= 0):
        raise ValueError("ellipse axes must be positive")
    semi_axes = diameters / 2.0
    width, height = _canonical_dimensions(canonical_size)
    canonical_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    canonical_radius = min(width, height) * 0.45
    angle = math.radians(float(rotation_degrees))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    linear = canonical_radius * np.diag(1.0 / semi_axes) @ rotation.T
    translation = canonical_center - linear @ source_center
    forward = np.vstack((np.column_stack((linear, translation)), np.array([0.0, 0.0, 1.0])))
    inverse = np.linalg.inv(forward)

    if source_boundary_points is None:
        angles = np.linspace(0.0, 2.0 * math.pi, 73)
        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        boundary = source_center + (unit * semi_axes) @ rotation.T
    else:
        boundary = np.asarray(source_boundary_points, dtype=np.float64).reshape(-1, 2)
    axis_ratio = float(max(semi_axes) / min(semi_axes))
    geometry_type = "front_circle" if axis_ratio <= 1.12 else "perspective_ellipse"
    return DialGeometry(
        geometry_type=geometry_type,
        source_boundary=SourceBoundary(
            kind="ellipse",
            points=_point_tuple(boundary),
            center=(float(source_center[0]), float(source_center[1])),
            axes=(float(diameters[0]), float(diameters[1])),
            rotation_degrees=float(rotation_degrees),
        ),
        forward_transform=_matrix_tuple(forward),
        inverse_transform=_matrix_tuple(inverse),
        canonical_size=(width, height),
        canonical_pivot=(float(canonical_center[0]), float(canonical_center[1])),
        canonical_scale_curve=_circle_curve(
            (float(canonical_center[0]), float(canonical_center[1])), canonical_radius * 0.9
        ),
        reprojection_error=max(0.0, float(reprojection_error)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method="ellipse_affine_rectification",
    )


def geometry_from_quadrilateral(
    corners: Sequence[Sequence[float]] | np.ndarray,
    *,
    confidence: float,
    reprojection_error: float = 0.0,
    canonical_size: int | tuple[int, int] = 512,
    canonical_pivot: Sequence[float] | None = None,
) -> DialGeometry:
    """Create a perspective rectification for rectangular/sector dial faces."""
    source = _order_quad(corners)
    width, height = _canonical_dimensions(canonical_size)
    margin = 0.06 * min(width, height)
    target = np.asarray(
        [
            [margin, margin],
            [width - 1.0 - margin, margin],
            [width - 1.0 - margin, height - 1.0 - margin],
            [margin, height - 1.0 - margin],
        ],
        dtype=np.float32,
    )
    forward = cv2.getPerspectiveTransform(source, target).astype(np.float64)
    inverse = np.linalg.inv(forward)
    pivot = (
        (float(width) * 0.5, float(height) * 0.82)
        if canonical_pivot is None
        else (float(canonical_pivot[0]), float(canonical_pivot[1]))
    )
    scale_radius = min(float(width) * 0.42, float(height) * 0.68)
    return DialGeometry(
        geometry_type="rectangular_sector",
        source_boundary=SourceBoundary(kind="quadrilateral", points=_point_tuple(source)),
        forward_transform=_matrix_tuple(forward),
        inverse_transform=_matrix_tuple(inverse),
        canonical_size=(width, height),
        canonical_pivot=pivot,
        canonical_scale_curve=_rectangular_arc_curve(pivot, scale_radius),
        reprojection_error=max(0.0, float(reprojection_error)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method="quadrilateral_homography",
    )


def estimate_dial_geometry(
    image_bgr: np.ndarray,
    *,
    canonical_size: int | tuple[int, int] = 512,
) -> DialGeometry:
    """Estimate a supported dial boundary using conservative OpenCV evidence."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty BGR or grayscale image")
    if image_bgr.ndim == 2:
        gray = image_bgr
    elif image_bgr.ndim == 3 and image_bgr.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2GRAY if image_bgr.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image_bgr, conversion)
    else:
        raise ValueError("image_bgr must have shape HxW, HxWx3, or HxWx4")

    height, width = gray.shape
    if min(height, width) < 24:
        return _roi_fallback(gray.shape, canonical_size, "image is too small for boundary estimation")

    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(normalized, (5, 5), 0), 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ellipse_candidates: list[tuple[float, tuple, np.ndarray, float]] = []
    quad_candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    image_area = float(height * width)
    image_diagonal = math.hypot(width, height)
    image_center = np.array([width / 2.0, height / 2.0])

    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        if area < image_area * 0.10 or area > image_area * 0.94:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 1.0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        touches_frame = x <= 1 and y <= 1 and x + box_width >= width - 1 and y + box_height >= height - 1
        if touches_frame:
            continue

        approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            corners = _order_quad(approximation.reshape(4, 2))
            rectangularity = area / max(float(cv2.contourArea(corners)), 1.0)
            center_distance = float(np.linalg.norm(corners.mean(axis=0) - image_center) / image_diagonal)
            fit_error = _quad_boundary_error(contour, corners)
            confidence = (
                0.42
                + 0.28 * float(np.clip(rectangularity, 0.0, 1.0))
                + 0.18 * max(0.0, 1.0 - center_distance / 0.35)
                + 0.12 * max(0.0, 1.0 - fit_error / 5.0)
            )
            quad_candidates.append((confidence, corners, contour, fit_error))

        if len(contour) >= 20:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (axis_a, axis_b), angle = ellipse
            if min(axis_a, axis_b) < 0.22 * min(height, width):
                continue
            if max(axis_a, axis_b) > 1.35 * max(height, width):
                continue
            center_distance = float(np.linalg.norm(np.array([center_x, center_y]) - image_center) / image_diagonal)
            if center_distance > 0.30:
                continue
            radial_error = _ellipse_radial_error(contour, ellipse)
            if radial_error > 0.18:
                continue
            coverage = math.pi * axis_a * axis_b / 4.0 / image_area
            confidence = (
                0.38
                + 0.34 * max(0.0, 1.0 - radial_error / 0.18)
                + 0.18 * max(0.0, 1.0 - center_distance / 0.30)
                + 0.10 * min(1.0, coverage / 0.55)
            )
            canonical_radius = min(_canonical_dimensions(canonical_size)) * 0.45
            ellipse_candidates.append((confidence, ellipse, contour, radial_error * canonical_radius))

    best_ellipse = max(ellipse_candidates, key=lambda candidate: candidate[0], default=None)
    best_quad = max(quad_candidates, key=lambda candidate: candidate[0], default=None)
    hough_frame = _hough_rectangular_envelope(edges)

    # A good ellipse is preferred unless a quadrilateral has materially stronger
    # evidence.  This prevents a coarse polygonal approximation of a circle from
    # routing a round dial into the rectangular branch.
    if best_ellipse is not None and (best_quad is None or best_ellipse[0] >= best_quad[0] - 0.03):
        confidence, ellipse, contour, reprojection_error = best_ellipse
        center, axes, angle = ellipse
        return geometry_from_ellipse(
            center,
            axes,
            angle,
            confidence=confidence,
            reprojection_error=reprojection_error,
            canonical_size=canonical_size,
            source_boundary_points=contour.reshape(-1, 2),
        )
    if best_quad is not None:
        confidence, corners, _contour, source_fit_error = best_quad
        # Report boundary error in canonical pixels so results are comparable
        # across detector-crop resolutions.
        canonical_error = source_fit_error * min(_canonical_dimensions(canonical_size)) / min(height, width)
        return geometry_from_quadrilateral(
            corners,
            confidence=confidence,
            reprojection_error=canonical_error,
            canonical_size=canonical_size,
        )
    if hough_frame is not None:
        corners, confidence, source_fit_error = hough_frame
        canonical_error = source_fit_error * min(_canonical_dimensions(canonical_size)) / min(height, width)
        geometry = geometry_from_quadrilateral(
            corners,
            confidence=confidence,
            reprojection_error=canonical_error,
            canonical_size=canonical_size,
        )
        return replace(geometry, method="open_frame_hough_envelope")
    return _roi_fallback(gray.shape, canonical_size, "no supported dial boundary passed conservative thresholds")


def _canonical_dimensions(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        width = height = size
    else:
        width, height = int(size[0]), int(size[1])
    if width < 32 or height < 32:
        raise ValueError("canonical_size dimensions must be at least 32 pixels")
    return width, height


def _ellipse_radial_error(contour: np.ndarray, ellipse: tuple) -> float:
    (center_x, center_y), (axis_a, axis_b), rotation_degrees = ellipse
    points = contour.reshape(-1, 2).astype(np.float64) - np.array([center_x, center_y])
    angle = math.radians(float(rotation_degrees))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    aligned = points @ rotation
    normalized_radius = np.hypot(aligned[:, 0] / max(axis_a / 2.0, 1e-6), aligned[:, 1] / max(axis_b / 2.0, 1e-6))
    return float(np.median(np.abs(normalized_radius - 1.0)))


def _quad_boundary_error(contour: np.ndarray, corners: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    distances: list[np.ndarray] = []
    for index in range(4):
        start = corners[index].astype(np.float64)
        end = corners[(index + 1) % 4].astype(np.float64)
        delta = end - start
        denominator = max(float(np.dot(delta, delta)), 1e-9)
        projection = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
        nearest = start + projection[:, None] * delta
        distances.append(np.linalg.norm(points - nearest, axis=1))
    return float(np.median(np.min(np.column_stack(distances), axis=1)))


def _hough_rectangular_envelope(edges: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Recover a rectangular dial when the detector crop cuts its outer frame.

    A crop-aligned instrument often has no closed contour: its bottom or side
    border exits the detector ROI.  In that case we accept only a strong
    three-sided frame consisting of one long top edge and two separated long
    side edges.  The strict span and position gates keep circular rims and
    pointer/tick lines out of this fallback.
    """
    height, width = edges.shape
    minimum = min(height, width)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(60, round(minimum * 0.055)),
        minLineLength=max(40, round(minimum * 0.38)),
        maxLineGap=max(12, round(minimum * 0.06)),
    )
    if raw_lines is None:
        return None

    horizontal: list[tuple[float, np.ndarray]] = []
    vertical: list[tuple[float, np.ndarray]] = []
    for values in np.asarray(raw_lines).reshape(-1, 4):
        line = values.astype(np.float64)
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < minimum * 0.38:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        axis_error = min(angle, abs(angle - 90.0), abs(angle - 180.0))
        if axis_error > 8.0:
            continue
        if min(angle, abs(angle - 180.0)) <= 8.0 and length >= width * 0.55:
            horizontal.append((length, line))
        elif abs(angle - 90.0) <= 8.0 and length >= height * 0.48:
            vertical.append((length, line))

    top_lines = [item for item in horizontal if float((item[1][1] + item[1][3]) / 2.0) <= height * 0.28]
    left_lines = [item for item in vertical if float((item[1][0] + item[1][2]) / 2.0) <= width * 0.28]
    right_lines = [item for item in vertical if float((item[1][0] + item[1][2]) / 2.0) >= width * 0.72]
    if not top_lines or not left_lines or not right_lines:
        return None

    # Prefer the longest representative in each extreme-edge cluster.  Length
    # is more stable than taking the outermost single Canny response from a
    # thick bezel.
    top = max(top_lines, key=lambda item: item[0])[1]
    left = max(left_lines, key=lambda item: item[0])[1]
    right = max(right_lines, key=lambda item: item[0])[1]
    left_mid = float((left[0] + left[2]) / 2.0)
    right_mid = float((right[0] + right[2]) / 2.0)
    if right_mid - left_mid < width * 0.62:
        return None

    top_left = _line_intersection(top, left)
    top_right = _line_intersection(top, right)
    if top_left is None or top_right is None:
        return None
    bottom_y = min(
        float(height - 1),
        max(float(max(left[1], left[3])), float(max(right[1], right[3]))),
    )
    if bottom_y < height * 0.72:
        return None
    bottom_line = np.array([0.0, bottom_y, float(width - 1), bottom_y])
    bottom_left = _line_intersection(bottom_line, left)
    bottom_right = _line_intersection(bottom_line, right)
    if bottom_left is None or bottom_right is None:
        return None

    corners = np.asarray([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    x_values, y_values = corners[:, 0], corners[:, 1]
    tolerance = 0.12
    if (
        x_values.min() < -width * tolerance
        or x_values.max() > width * (1.0 + tolerance)
        or y_values.min() < -height * tolerance
        or y_values.max() > height * (1.0 + tolerance)
    ):
        return None
    area_ratio = abs(float(cv2.contourArea(corners))) / float(width * height)
    if area_ratio < 0.48:
        return None

    normalized_support = min(1.0, (len(top_lines) + len(left_lines) + len(right_lines)) / 9.0)
    span_support = min(1.0, (right_mid - left_mid) / (width * 0.85))
    confidence = float(np.clip(0.62 + 0.10 * normalized_support + 0.12 * span_support, 0.0, 0.86))
    # Thick bezels generate several nearly parallel Canny/Hough responses.  Their
    # median offset is a useful, resolution-aware boundary uncertainty.
    offsets: list[float] = []
    top_y = float((top[1] + top[3]) / 2.0)
    offsets.extend(abs(float((line[1] + line[3]) / 2.0) - top_y) for _, line in top_lines)
    offsets.extend(abs(float((line[0] + line[2]) / 2.0) - left_mid) for _, line in left_lines)
    offsets.extend(abs(float((line[0] + line[2]) / 2.0) - right_mid) for _, line in right_lines)
    fit_error = float(np.median(offsets)) if offsets else 0.0
    return _order_quad(corners), confidence, fit_error


def _line_intersection(first: np.ndarray, second: np.ndarray) -> Point | None:
    x1, y1, x2, y2 = (float(value) for value in first)
    x3, y3, x4, y4 = (float(value) for value in second)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-8:
        return None
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    x = (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator
    y = (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator
    return float(x), float(y)


def _roi_fallback(
    image_shape: Sequence[int],
    canonical_size: int | tuple[int, int],
    reason: str,
) -> DialGeometry:
    height, width = int(image_shape[0]), int(image_shape[1])
    canonical_width, canonical_height = _canonical_dimensions(canonical_size)
    source = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [0.0, 0.0],
            [canonical_width - 1.0, 0.0],
            [canonical_width - 1.0, canonical_height - 1.0],
            [0.0, canonical_height - 1.0],
        ],
        dtype=np.float32,
    )
    forward = cv2.getPerspectiveTransform(source, target).astype(np.float64)
    inverse = np.linalg.inv(forward)
    pivot = (canonical_width / 2.0, canonical_height / 2.0)
    return DialGeometry(
        geometry_type="roi_fallback",
        source_boundary=SourceBoundary(kind="roi", points=_point_tuple(source)),
        forward_transform=_matrix_tuple(forward),
        inverse_transform=_matrix_tuple(inverse),
        canonical_size=(canonical_width, canonical_height),
        canonical_pivot=pivot,
        canonical_scale_curve=_circle_curve(pivot, min(canonical_width, canonical_height) * 0.405),
        reprojection_error=0.0,
        confidence=0.10,
        method="roi_resize_fallback",
        fallback_reason=reason,
    )


__all__ = [
    "DialGeometry",
    "SourceBoundary",
    "adapt_circle_estimate",
    "estimate_dial_geometry",
    "geometry_from_ellipse",
    "geometry_from_quadrilateral",
    "transform_points",
]

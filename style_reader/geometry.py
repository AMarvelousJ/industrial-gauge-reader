from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleEstimate:
    center_x: float
    center_y: float
    radius: float
    method: str
    confidence: float


@dataclass(frozen=True)
class LineCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    tip_x: int
    tip_y: int
    angle_degrees: float
    score: float
    center_distance_ratio: float
    length_ratio: float


def clockwise_angle_degrees(center: tuple[float, float], tip: tuple[float, float]) -> float:
    """Angle with dial top as 0 degrees, increasing clockwise."""
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _resize_for_analysis(image: np.ndarray, maximum_side: int = 900) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return image.copy(), scale
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def estimate_circle(gray: np.ndarray) -> CircleEstimate:
    height, width = gray.shape
    minimum = min(height, width)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.8)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, minimum // 3),
        param1=120,
        param2=38,
        minRadius=max(15, round(minimum * 0.24)),
        maxRadius=max(20, round(minimum * 0.53)),
    )
    image_center = np.array([width / 2.0, height / 2.0])
    if circles is not None and len(circles[0]):
        candidates = circles[0]
        normalized_distance = np.linalg.norm(candidates[:, :2] - image_center, axis=1) / minimum
        size_reward = candidates[:, 2] / minimum
        best_index = int(np.argmin(normalized_distance - 0.25 * size_reward))
        x, y, radius = (float(value) for value in candidates[best_index])
        confidence = max(0.35, min(0.95, 0.9 - float(normalized_distance[best_index])))
        return CircleEstimate(x, y, radius, "hough_circle", confidence)
    return CircleEstimate(width / 2.0, height / 2.0, minimum * 0.46, "roi_fallback", 0.25)


def _distance_point_to_line(
    point: tuple[float, float], first: tuple[float, float], second: tuple[float, float]
) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    delta = b - a
    denominator = float(np.linalg.norm(delta))
    if denominator < 1e-6:
        return float("inf")
    offset = p - a
    cross_2d = delta[0] * offset[1] - delta[1] * offset[0]
    return float(abs(cross_2d) / denominator)


def line_candidates(edges: np.ndarray, circle: CircleEstimate) -> list[LineCandidate]:
    radius = circle.radius
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(18, round(radius * 0.07)),
        minLineLength=max(15, round(radius * 0.18)),
        maxLineGap=max(5, round(radius * 0.06)),
    )
    if lines is None:
        return []
    center = (circle.center_x, circle.center_y)
    candidates: list[LineCandidate] = []
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw)
        length = math.hypot(x2 - x1, y2 - y1)
        center_distance = _distance_point_to_line(center, (x1, y1), (x2, y2))
        if center_distance > radius * 0.19:
            continue
        distances = [math.hypot(x1 - center[0], y1 - center[1]), math.hypot(x2 - center[0], y2 - center[1])]
        tip = (x1, y1) if distances[0] >= distances[1] else (x2, y2)
        near_distance, far_distance = min(distances), max(distances)
        if far_distance < radius * 0.33 or far_distance > radius * 1.18:
            continue
        if near_distance > radius * 0.62:
            continue
        line_unit = np.array([x2 - x1, y2 - y1], dtype=np.float64) / max(length, 1e-6)
        radial = np.array([tip[0] - center[0], tip[1] - center[1]], dtype=np.float64)
        radial /= max(float(np.linalg.norm(radial)), 1e-6)
        alignment = abs(float(np.dot(line_unit, radial)))
        length_ratio = min(1.2, length / radius)
        center_ratio = center_distance / radius
        score = (
            0.34 * min(1.0, length_ratio)
            + 0.30 * alignment
            + 0.22 * max(0.0, 1.0 - center_ratio / 0.19)
            + 0.14 * min(1.0, far_distance / (radius * 0.75))
        )
        candidates.append(
            LineCandidate(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                tip_x=tip[0],
                tip_y=tip[1],
                angle_degrees=clockwise_angle_degrees(center, tip),
                score=round(score, 6),
                center_distance_ratio=round(center_ratio, 6),
                length_ratio=round(length_ratio, 6),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def radial_darkness_scan(gray: np.ndarray, circle: CircleEstimate, step_degrees: float = 0.5) -> dict:
    """Score continuous dark radial strokes; useful when Hough returns fragmented edges."""
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    angles = np.arange(0.0, 360.0, step_degrees)
    radii = np.linspace(circle.radius * 0.12, circle.radius * 0.82, 100)
    scores: list[float] = []
    height, width = gray.shape
    for angle in angles:
        radians = math.radians(float(angle))
        xs = np.rint(circle.center_x + radii * math.sin(radians)).astype(np.int32)
        ys = np.rint(circle.center_y - radii * math.cos(radians)).astype(np.int32)
        valid = (xs >= 1) & (xs < width - 1) & (ys >= 1) & (ys < height - 1)
        if valid.sum() < 30:
            scores.append(0.0)
            continue
        xs, ys = xs[valid], ys[valid]
        # Minimum over a 3x3 patch tolerates a thin pointer not falling exactly
        # on the sampled ray. Continuous darkness receives the strongest score.
        samples = np.minimum.reduce(
            [equalized[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        )
        darkness = 1.0 - samples.astype(np.float32) / 255.0
        dark_fraction = float(np.mean(darkness > 0.55))
        continuity = float(np.max(np.convolve((darkness > 0.48).astype(np.float32), np.ones(9), mode="valid")) / 9.0)
        scores.append(0.68 * float(np.mean(darkness)) + 0.20 * dark_fraction + 0.12 * continuity)
    score_array = np.asarray(scores)
    best_index = int(score_array.argmax())
    baseline = float(np.median(score_array))
    spread = float(np.std(score_array))
    confidence = max(0.0, min(1.0, (float(score_array[best_index]) - baseline) / max(0.08, 2.5 * spread)))
    return {
        "angle_degrees": float(angles[best_index]),
        "score": float(score_array[best_index]),
        "baseline": baseline,
        "confidence": confidence,
    }


def tick_angle_candidates(gray: np.ndarray, circle: CircleEstimate) -> list[float]:
    """Return coarse outer-ring darkness peaks for later scale/OCR mapping."""
    angles = np.arange(0.0, 360.0, 1.0)
    radii = np.linspace(circle.radius * 0.74, circle.radius * 0.93, 20)
    height, width = gray.shape
    values: list[float] = []
    for angle in angles:
        radians = math.radians(float(angle))
        xs = np.rint(circle.center_x + radii * math.sin(radians)).astype(np.int32)
        ys = np.rint(circle.center_y - radii * math.cos(radians)).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        values.append(float(np.mean(255 - gray[ys[valid], xs[valid]])) if valid.any() else 0.0)
    signal = np.asarray(values)
    threshold = float(np.mean(signal) + 0.65 * np.std(signal))
    active = signal >= threshold
    groups: list[list[int]] = []
    current: list[int] = []
    for index, enabled in enumerate(active):
        if enabled:
            current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    # Merge wrap-around peak.
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == 359:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    peaks = [max(group, key=lambda index: signal[index]) for group in groups if len(group) <= 14]
    return [float(angle % 360) for angle in peaks[:80]]


def _colored_pointer_candidate(image_bgr: np.ndarray, circle: CircleEstimate) -> dict | None:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 162)) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    best_component = None
    for label in range(1, component_count):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 25:
            continue
        distances = np.hypot(xs - circle.center_x, ys - circle.center_y)
        points = np.column_stack((xs, ys)).astype(np.float64)
        centered = points - points.mean(axis=0)
        _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
        elongation = float(singular_values[0] / max(singular_values[1], 1e-6))
        principal = axes[0]
        center_offset = np.asarray([circle.center_x, circle.center_y]) - points.mean(axis=0)
        line_distance = abs(float(principal[0] * center_offset[1] - principal[1] * center_offset[0]))
        attached = float(distances.min()) <= circle.radius * 0.28
        detached_radial_marker = (
            float(distances.min()) <= circle.radius * 0.78
            and elongation >= 3.0
            and line_distance <= circle.radius * 0.18
        )
        if (not attached and not detached_radial_marker) or float(distances.max()) < circle.radius * 0.35:
            continue
        score = (
            float(distances.max())
            + 0.02 * len(xs)
            + 8.0 * min(elongation, 10.0)
            + (1000.0 if detached_radial_marker and not attached else 0.0)
        )
        if best_component is None or score > best_component[0]:
            best_component = (score, xs, ys, distances, detached_radial_marker and not attached, elongation)
    if best_component is None:
        return None
    _, xs, ys, distances, detached_marker, elongation = best_component
    far = int(distances.argmax())
    if detached_marker:
        tip = (int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))))
    else:
        tip = (int(xs[far]), int(ys[far]))
    coverage = min(1.0, float(distances.max()) / (circle.radius * 0.75))
    return {
        "tip": tip,
        "angle_degrees": clockwise_angle_degrees((circle.center_x, circle.center_y), tip),
        "confidence": min(0.95, 0.45 + 0.4 * coverage),
        "pixel_count": int(len(xs)),
        "detached_scale_marker": bool(detached_marker),
        "elongation": round(float(elongation), 4),
    }


def analyze_pointer(
    image_bgr: np.ndarray,
    ocr_boxes: list | None = None,
    circle_override: CircleEstimate | None = None,
) -> tuple[dict, np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty BGR image")
    analysis, scale = _resize_for_analysis(image_bgr)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    circle = circle_override or estimate_circle(clahe)
    pointer_gray = gray.copy()
    masked_boxes = 0
    for raw_box in ocr_boxes or []:
        points = np.asarray(raw_box, dtype=np.float32) * scale
        left, top = np.floor(points.min(axis=0) - 4).astype(int)
        right, bottom = np.ceil(points.max(axis=0) + 4).astype(int)
        cv2.rectangle(pointer_gray, (max(0, left), max(0, top)), (min(pointer_gray.shape[1] - 1, right), min(pointer_gray.shape[0] - 1, bottom)), 255, -1)
        masked_boxes += 1
    pointer_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pointer_gray)
    edges = cv2.Canny(cv2.GaussianBlur(pointer_clahe, (5, 5), 0), 45, 135)
    candidates = line_candidates(edges, circle)
    radial = radial_darkness_scan(pointer_gray, circle)
    colored = _colored_pointer_candidate(analysis, circle)

    if colored is not None:
        angle = colored["angle_degrees"]
        method = "hsv_red_pointer"
        pointer_confidence = colored["confidence"] * circle.confidence
        tip = colored["tip"]
        status = "angle_estimated"
    elif candidates and candidates[0].score >= 0.52:
        best = candidates[0]
        angle = best.angle_degrees
        method = "hough_line"
        pointer_confidence = min(1.0, best.score) * circle.confidence
        tip = (best.tip_x, best.tip_y)
        status = "angle_estimated"
    elif radial["confidence"] >= 0.12:
        angle = radial["angle_degrees"]
        method = "radial_darkness"
        pointer_confidence = radial["confidence"] * circle.confidence
        radians = math.radians(angle)
        tip = (
            round(circle.center_x + circle.radius * 0.78 * math.sin(radians)),
            round(circle.center_y - circle.radius * 0.78 * math.cos(radians)),
        )
        status = "angle_estimated"
    else:
        angle = None
        method = "none"
        pointer_confidence = 0.0
        tip = None
        status = "pointer_not_found"

    visualization = analysis.copy()
    cv2.circle(
        visualization,
        (round(circle.center_x), round(circle.center_y)),
        round(circle.radius),
        (0, 220, 255),
        2,
    )
    for candidate in candidates[1:9]:
        cv2.line(visualization, (candidate.x1, candidate.y1), (candidate.x2, candidate.y2), (80, 80, 200), 1)
    center_point = (round(circle.center_x), round(circle.center_y))
    cv2.circle(visualization, center_point, 6, (255, 80, 0), -1)
    if tip is not None:
        cv2.line(visualization, center_point, tip, (0, 255, 0), 3)
        cv2.circle(visualization, tip, 6, (0, 0, 255), -1)
        cv2.putText(
            visualization,
            f"angle={angle:.1f} deg ({method})",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    result = {
        "status": status,
        "reading_status": "calibration_required" if status == "angle_estimated" else "unavailable",
        "reading": None,
        "angle_degrees_clockwise_from_top": None if angle is None else round(float(angle), 4),
        "pointer_method": method,
        "pointer_confidence": round(float(pointer_confidence), 6),
        "analysis_scale": scale,
        "circle": asdict(circle),
        "pointer_tip": None if tip is None else {"x": tip[0], "y": tip[1]},
        "radial_scan": {key: round(float(value), 6) for key, value in radial.items()},
        "tick_angle_candidates": tick_angle_candidates(gray, circle),
        "line_candidates": [asdict(candidate) for candidate in candidates[:20]],
        "ocr_boxes_masked_before_pointer_detection": masked_boxes,
        "colored_pointer_candidate": colored,
    }
    return result, visualization

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = "1.0.0"
SHAPE_STRATA = (
    "circular_front",
    "circular_perspective",
    "rectangular_sector",
    "irregular_occluded_offset",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageFingerprint:
    path: Path
    sha256: str
    phash: int
    histogram: tuple[float, ...]
    aspect_ratio: float


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def discover_images(source_root: Path) -> list[Path]:
    """Discover image files only; Markdown and label files are never opened."""
    source_root = Path(source_root)
    paths = [
        path
        for path in source_root.glob("M*/images/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda value: value.as_posix().lower())


def _load_image(path: Path) -> np.ndarray:
    # imdecode handles non-ASCII Windows paths more reliably than cv2.imread.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8]
    values = coefficients.flatten()
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    result = 0
    for index, bit in enumerate(bits):
        if bool(bit):
            result |= 1 << index
    return result


def _color_histogram(image: np.ndarray) -> tuple[float, ...]:
    hsv = cv2.cvtColor(cv2.resize(image, (96, 96)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2).flatten()
    return tuple(float(value) for value in hist)


def fingerprint_image(path: Path) -> ImageFingerprint:
    image = _load_image(path)
    height, width = image.shape[:2]
    return ImageFingerprint(
        path=path,
        sha256=_sha256(path),
        phash=_phash(image),
        histogram=_color_histogram(image),
        aspect_ratio=width / max(height, 1),
    )


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _histogram_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


def cluster_near_duplicates(
    fingerprints: Sequence[ImageFingerprint],
    *,
    strict_hamming: int = 6,
    relaxed_hamming: int = 10,
    histogram_threshold: float = 0.995,
) -> list[str]:
    """Cluster exact and perceptually near-identical images deterministically."""
    union_find = UnionFind(len(fingerprints))
    exact_seen: dict[str, int] = {}
    for index, item in enumerate(fingerprints):
        previous = exact_seen.setdefault(item.sha256, index)
        union_find.union(index, previous)

    for left in range(len(fingerprints)):
        left_item = fingerprints[left]
        for right in range(left + 1, len(fingerprints)):
            right_item = fingerprints[right]
            if abs(math.log(max(left_item.aspect_ratio, 1e-6) / max(right_item.aspect_ratio, 1e-6))) > 0.08:
                continue
            distance = _hamming(left_item.phash, right_item.phash)
            if distance <= strict_hamming or (
                distance <= relaxed_hamming
                and _histogram_similarity(left_item.histogram, right_item.histogram)
                >= histogram_threshold
            ):
                union_find.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(fingerprints)):
        members[union_find.find(index)].append(index)
    root_to_id = {
        root: "dup-" + min(fingerprints[index].sha256 for index in indices)[:16]
        for root, indices in members.items()
    }
    return [root_to_id[union_find.find(index)] for index in range(len(fingerprints))]


def _read_detector_box(image_path: Path, width: int, height: int) -> tuple[tuple[int, int, int, int], str]:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                center_x, center_y, box_width, box_height = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = max(0, int(round((center_x - box_width / 2) * width)))
            y1 = max(0, int(round((center_y - box_height / 2) * height)))
            x2 = min(width, int(round((center_x + box_width / 2) * width)))
            y2 = min(height, int(round((center_y + box_height / 2) * height)))
            if x2 > x1 and y2 > y1:
                candidates.append(((x2 - x1) * (y2 - y1), (x1, y1, x2, y2)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1], "existing_yolo_label"
    return (0, 0, width, height), "full_image_fallback"


def _normalized_point(x: float, y: float, width: int, height: int) -> dict[str, float]:
    return {
        "x": round(float(np.clip(x / max(width, 1), 0.0, 1.0)), 6),
        "y": round(float(np.clip(y / max(height, 1), 0.0, 1.0)), 6),
    }


def _detect_geometry(roi: np.ndarray) -> dict[str, Any]:
    roi_height, roi_width = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(equalized, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi_width * roi_height)
    center = np.array([roi_width / 2, roi_height / 2], dtype=np.float32)

    best_ellipse: tuple[float, tuple[Any, ...]] | None = None
    best_quad: tuple[float, np.ndarray] | None = None
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < 0.08 * roi_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_center = np.asarray(ellipse[0], dtype=np.float32)
            axes = sorted(ellipse[1], reverse=True)
            ellipse_area = math.pi * axes[0] * axes[1] / 4
            if ellipse_area > 0:
                area_fit = min(area, ellipse_area) / max(area, ellipse_area)
                center_score = max(
                    0.0,
                    1.0 - float(np.linalg.norm(ellipse_center - center)) / max(0.45 * min(roi_width, roi_height), 1),
                )
                coverage = min(ellipse_area / roi_area, 1.0)
                score = area_fit * center_score * min(1.0, coverage / 0.35)
                if best_ellipse is None or score > best_ellipse[0]:
                    best_ellipse = (score, ellipse)
        approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            coverage = min(area / roi_area, 1.0)
            quad_center = approximation.reshape(-1, 2).mean(axis=0)
            center_score = max(
                0.0,
                1.0 - float(np.linalg.norm(quad_center - center)) / max(0.45 * min(roi_width, roi_height), 1),
            )
            score = center_score * min(1.0, coverage / 0.45)
            if best_quad is None or score > best_quad[0]:
                best_quad = (score, approximation.reshape(-1, 2))

    if best_ellipse:
        ellipse_score, ellipse = best_ellipse
        (ellipse_x, ellipse_y), (axis_a, axis_b), ellipse_angle = ellipse
        major_axis, minor_axis = max(axis_a, axis_b), min(axis_a, axis_b)
        axis_ratio = minor_axis / max(major_axis, 1e-6)
    else:
        ellipse_score = 0.0
        ellipse_x, ellipse_y = roi_width / 2, roi_height / 2
        major_axis = minor_axis = 0.82 * min(roi_width, roi_height)
        ellipse_angle = 0.0
        axis_ratio = 1.0
    quad_score = best_quad[0] if best_quad else 0.0
    roi_aspect = max(roi_width, roi_height) / max(min(roi_width, roi_height), 1)
    front_score = float(np.clip(0.65 * ellipse_score + 0.35 * math.exp(-((1 - axis_ratio) / 0.13) ** 2), 0, 1))
    perspective_score = float(
        np.clip(0.7 * ellipse_score + 0.3 * min(1.0, abs(1 - axis_ratio) / 0.25), 0, 1)
        * math.exp(-((axis_ratio - 0.72) / 0.28) ** 2)
    )
    rectangular_score = float(np.clip(0.65 * quad_score + 0.35 * min(1.0, max(0.0, roi_aspect - 1.08) / 0.45), 0, 1))
    irregular_score = float(
        np.clip(0.35 + 0.55 * (1 - max(ellipse_score, quad_score)) + 0.1 * min(1.0, abs(1 - roi_aspect)), 0, 1)
    )
    scores = {
        "circular_front": round(front_score, 6),
        "circular_perspective": round(perspective_score, 6),
        "rectangular_sector": round(rectangular_score, 6),
        "irregular_occluded_offset": round(irregular_score, 6),
    }
    predicted_shape = max(scores, key=scores.get)
    if best_quad:
        polygon = best_quad[1].tolist()
    else:
        polygon = [[0, 0], [roi_width - 1, 0], [roi_width - 1, roi_height - 1], [0, roi_height - 1]]
    return {
        "edges": edges,
        "pivot_local": (float(ellipse_x), float(ellipse_y)),
        "ellipse": {
            "center": (float(ellipse_x), float(ellipse_y)),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "angle_deg": float(ellipse_angle),
            "axis_ratio": float(axis_ratio),
            "confidence": float(np.clip(ellipse_score, 0, 1)),
        },
        "polygon": polygon,
        "quad_confidence": float(np.clip(quad_score, 0, 1)),
        "scores": scores,
        "predicted_shape": predicted_shape,
    }


def _red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    high = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return cv2.bitwise_or(low, high)


def _pointer_candidates(roi: np.ndarray, edges: np.ndarray, pivot: tuple[float, float]) -> list[dict[str, Any]]:
    height, width = roi.shape[:2]
    diagonal = math.hypot(width, height)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(0.06 * min(width, height))),
        minLineLength=max(12, int(0.14 * min(width, height))),
        maxLineGap=max(5, int(0.04 * min(width, height))),
    )
    red = _red_mask(roi)
    candidates: list[dict[str, Any]] = []
    if lines is None:
        return candidates
    pivot_array = np.asarray(pivot, dtype=np.float32)
    # OpenCV has returned both (N, 1, 4) and (N, 4) across releases.
    for raw in np.asarray(lines).reshape(-1, 4):
        start = raw[:2].astype(np.float32)
        end = raw[2:].astype(np.float32)
        length = float(np.linalg.norm(end - start))
        if length <= 0:
            continue
        distances = (float(np.linalg.norm(start - pivot_array)), float(np.linalg.norm(end - pivot_array)))
        near, far = (start, end) if distances[0] <= distances[1] else (end, start)
        near_distance, far_distance = min(distances), max(distances)
        if far_distance < 0.16 * min(width, height):
            continue
        proximity_score = math.exp(-near_distance / max(0.12 * min(width, height), 1))
        radial_score = max(0.0, min(1.0, (far_distance - near_distance) / max(length, 1)))
        base_score = min(1.0, length / max(0.42 * diagonal, 1)) * proximity_score * (0.5 + 0.5 * radial_score)
        line_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(line_mask, tuple(start.astype(int)), tuple(end.astype(int)), 255, thickness=max(2, int(diagonal / 220)))
        pixels = line_mask > 0
        red_ratio = float(np.count_nonzero(red[pixels])) / max(int(np.count_nonzero(pixels)), 1)
        role = "red_marker_candidate" if red_ratio >= 0.18 else "measurement_candidate"
        candidates.append(
            {
                "start_local": near.tolist(),
                "end_local": far.tolist(),
                "tip_local": far.tolist(),
                "role": role,
                "red_ratio": red_ratio,
                "confidence": float(np.clip(base_score * (0.8 if role == "red_marker_candidate" else 1.0), 0, 1)),
            }
        )
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        direction = np.asarray(candidate["tip_local"]) - pivot_array
        angle = math.atan2(float(direction[1]), float(direction[0]))
        if any(abs(math.atan2(math.sin(angle - other["_angle"]), math.cos(angle - other["_angle"]))) < math.radians(4) for other in accepted):
            continue
        candidate["_angle"] = angle
        accepted.append(candidate)
        if len(accepted) == 5:
            break
    for candidate in accepted:
        candidate.pop("_angle", None)
    return accepted


def _to_global_point(local: Sequence[float], box: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    return _normalized_point(float(local[0]) + box[0], float(local[1]) + box[1], width, height)


def _record_for_image(
    image_path: Path,
    source_root: Path,
    fingerprint: ImageFingerprint,
    duplicate_cluster_id: str,
) -> dict[str, Any]:
    image = _load_image(image_path)
    height, width = image.shape[:2]
    box, box_source = _read_detector_box(image_path, width, height)
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    geometry = _detect_geometry(roi)
    pivot_local = geometry["pivot_local"]
    candidates = _pointer_candidates(roi, geometry["edges"], pivot_local)
    selected = next((item for item in candidates if item["role"] == "measurement_candidate"), None)

    ellipse = geometry["ellipse"]
    ellipse_center = _to_global_point(ellipse["center"], box, width, height)
    polygon = [
        _to_global_point(point, box, width, height) for point in geometry["polygon"]
    ]
    pointer_records = []
    for index, candidate in enumerate(candidates):
        pointer_records.append(
            {
                "candidate_id": f"pointer-{index + 1}",
                "line": {
                    "start": _to_global_point(candidate["start_local"], box, width, height),
                    "end": _to_global_point(candidate["end_local"], box, width, height),
                },
                "tip": _to_global_point(candidate["tip_local"], box, width, height),
                "auto_role": candidate["role"],
                "red_pixel_ratio": round(candidate["red_ratio"], 6),
                "confidence": round(candidate["confidence"], 6),
                "review": {"status": "pending", "role": None, "accept": None, "comment": None},
            }
        )
    selected_id = None
    if selected is not None:
        selected_index = candidates.index(selected)
        selected_id = f"pointer-{selected_index + 1}"

    relative_path = image_path.relative_to(source_root).as_posix()
    record_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    major_axis_norm = ellipse["major_axis"] / max(width, height)
    minor_axis_norm = ellipse["minor_axis"] / max(width, height)
    scale_shape = "ellipse" if geometry["predicted_shape"].startswith("circular") else "sector_curve"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "image": {
            "path": relative_path,
            "width": width,
            "height": height,
            "sha256": fingerprint.sha256,
            "phash64": f"{fingerprint.phash:016x}",
        },
        "duplicate_cluster_id": duplicate_cluster_id,
        "sampling": {"selected": False, "stratum": None, "split": "pool"},
        "auto_annotation": {
            "dial_boundary": {
                "detector_box": {
                    "x_min": round(x1 / width, 6),
                    "y_min": round(y1 / height, 6),
                    "x_max": round(x2 / width, 6),
                    "y_max": round(y2 / height, 6),
                },
                "source": box_source,
                "polygon": polygon,
                "ellipse": {
                    "center": ellipse_center,
                    "major_axis": round(major_axis_norm, 6),
                    "minor_axis": round(minor_axis_norm, 6),
                    "angle_deg": round(ellipse["angle_deg"], 4),
                    "confidence": round(ellipse["confidence"], 6),
                },
                "review": {"status": "pending", "accept": None, "comment": None},
            },
            "shape": {
                "predicted": geometry["predicted_shape"],
                "scores": geometry["scores"],
                "review": {"status": "pending", "label": None, "comment": None},
            },
            "pivot": {
                "point": _to_global_point(pivot_local, box, width, height),
                "source": "ellipse_center" if ellipse["confidence"] >= 0.2 else "box_center_fallback",
                "confidence": round(max(0.1, ellipse["confidence"]), 6),
                "review": {"status": "pending", "point": None, "accept": None, "comment": None},
            },
            "pointer_candidates": pointer_records,
            "selected_pointer_candidate_id": selected_id,
            "scale_arc": {
                "curve_type": scale_shape,
                "center": _to_global_point(pivot_local, box, width, height),
                "radius_major": round(0.44 * (x2 - x1) / max(width, height), 6),
                "radius_minor": round(0.44 * (y2 - y1) / max(width, height), 6),
                "start_angle_deg": 0.0 if scale_shape == "ellipse" else 180.0,
                "end_angle_deg": 360.0,
                "direction": "unknown",
                "confidence": round(0.35 * max(ellipse["confidence"], geometry["quad_confidence"]), 6),
                "review": {"status": "pending", "accept": None, "comment": None},
            },
        },
        # These values are intentionally never inferred from the Markdown ground truth.
        "human_review": {
            "status": "pending",
            "shape": None,
            "pivot": None,
            "pointer_tip": None,
            "pointer_candidate_id": None,
            "pointer_role": None,
            "scale_arc": None,
            "reading": None,
            "unit": None,
            "range_min": None,
            "range_max": None,
            "minor_division": None,
            "scope_status": None,
            "meter_family": None,
            "physical_meter_id": None,
            "condition": None,
            "training_track": None,
            "source_group": None,
            "brand": None,
            "model": None,
            "comment": None,
        },
    }


def balanced_sample(
    records: list[dict[str, Any]],
    *,
    per_shape: int,
    validation_per_shape: int,
    seed: int,
) -> list[dict[str, Any]]:
    if validation_per_shape < 0 or validation_per_shape > per_shape:
        raise ValueError("validation_per_shape must be between zero and per_shape")
    used_clusters: set[str] = set()
    selected: list[dict[str, Any]] = []
    for stratum in SHAPE_STRATA:
        ranked = sorted(
            records,
            key=lambda record: (
                -record["auto_annotation"]["shape"]["scores"][stratum],
                hashlib.sha256(
                    f"{seed}:{stratum}:{record['record_id']}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        stratum_records: list[dict[str, Any]] = []
        for record in ranked:
            cluster_id = record["duplicate_cluster_id"]
            if cluster_id in used_clusters:
                continue
            used_clusters.add(cluster_id)
            stratum_records.append(record)
            if len(stratum_records) == per_shape:
                break
        if len(stratum_records) != per_shape:
            raise RuntimeError(
                f"only {len(stratum_records)} distinct clusters available for {stratum}; requested {per_shape}"
            )
        split_order = sorted(
            stratum_records,
            key=lambda record: hashlib.sha256(
                f"{seed}:split:{stratum}:{record['record_id']}".encode("utf-8")
            ).hexdigest(),
        )
        frozen_ids = {
            record["record_id"] for record in split_order[:validation_per_shape]
        }
        for record in stratum_records:
            record["sampling"] = {
                "selected": True,
                "stratum": stratum,
                "split": "frozen_validation" if record["record_id"] in frozen_ids else "dev",
            }
            selected.append(record)
    return selected


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _draw_overlay(image: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    output = image.copy()
    boundary = record["auto_annotation"]["dial_boundary"]
    box = boundary["detector_box"]
    x1, y1 = int(box["x_min"] * width), int(box["y_min"] * height)
    x2, y2 = int(box["x_max"] * width), int(box["y_max"] * height)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), max(1, min(width, height) // 350))
    pivot = record["auto_annotation"]["pivot"]["point"]
    pivot_point = (int(pivot["x"] * width), int(pivot["y"] * height))
    cv2.circle(output, pivot_point, max(3, min(width, height) // 80), (255, 80, 0), -1)
    selected_id = record["auto_annotation"]["selected_pointer_candidate_id"]
    for candidate in record["auto_annotation"]["pointer_candidates"]:
        start = candidate["line"]["start"]
        end = candidate["line"]["end"]
        color = (0, 255, 0) if candidate["candidate_id"] == selected_id else (128, 128, 255)
        cv2.line(
            output,
            (int(start["x"] * width), int(start["y"] * height)),
            (int(end["x"] * width), int(end["y"] * height)),
            color,
            max(1, min(width, height) // 250),
        )
    label = f"{record['record_id']} {record['sampling']['stratum']} {record['sampling']['split']}"
    cv2.rectangle(output, (0, 0), (min(width, 12 * len(label)), 28), (0, 0, 0), -1)
    cv2.putText(output, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _save_jpeg(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"unable to encode image: {path}")
    encoded.tofile(str(path))


def _make_contact_sheet(thumbnails: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    tile_width, tile_height, columns = 240, 210, 6
    rows = math.ceil(len(thumbnails) / columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, (label, image) in enumerate(thumbnails):
        scale = min((tile_width - 8) / image.shape[1], (tile_height - 28) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * tile_width + (tile_width - resized.shape[1]) // 2
        y = row * tile_height + 22 + (tile_height - 24 - resized.shape[0]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(sheet, label, (column * tile_width + 5, row * tile_height + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    _save_jpeg(output_path, sheet, quality=90)


def _write_review_csv(csv_path: Path, records: list[dict[str, Any]], *, private: bool) -> None:
    fields = [
        "record_id", "split", "sampling_stratum", "image_path", "duplicate_cluster_id",
        "auto_shape", "review_status", "review_shape", "pivot_x", "pivot_y",
        "pointer_tip_x", "pointer_tip_y", "pointer_candidate_id", "pointer_role",
        "pointer_angle_deg", "reading", "unit", "range_min", "range_max", "minor_division",
        "scope_status", "meter_family", "physical_meter_id", "condition", "training_track",
        "source_group", "brand", "model",
        "comment", "thumbnail",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item["sampling"]["stratum"], item["record_id"])):
            split = record["sampling"]["split"]
            stratum = record["sampling"]["stratum"]
            thumbnail = (
                f"thumbnails/{stratum}/{record['record_id']}.jpg"
                if private
                else f"thumbnails/dev/{stratum}/{record['record_id']}.jpg"
            )
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "split": split,
                    "sampling_stratum": stratum,
                    "image_path": record["image"]["path"],
                    "duplicate_cluster_id": record["duplicate_cluster_id"],
                    "auto_shape": record["auto_annotation"]["shape"]["predicted"],
                    "review_status": "pending",
                    "thumbnail": thumbnail,
                }
            )


def _write_review_files(output_dir: Path, selected: list[dict[str, Any]]) -> None:
    dev_records = [record for record in selected if record["sampling"]["split"] == "dev"]
    frozen_records = [record for record in selected if record["sampling"]["split"] == "frozen_validation"]
    private_dir = output_dir / "frozen_private"
    _write_review_csv(output_dir / "review.csv", dev_records, private=False)
    _write_review_csv(private_dir / "review_frozen.csv", frozen_records, private=True)
    counts = Counter((record["sampling"]["split"], record["sampling"]["stratum"]) for record in selected)
    lines = [
        "# Gauge pre-annotation review package",
        "",
        "This public review file contains the 80 development rows only.",
        "Fill the empty review columns in `review.csv`; frozen labels live only below `frozen_private/`.",
        "",
        "## Split counts",
        "",
        "| split | shape stratum | count |",
        "|---|---|---:|",
    ]
    for stratum in SHAPE_STRATA:
        lines.append(f"| dev | {stratum} | {counts[('dev', stratum)]} |")
    lines.extend(
        [
            "",
            "## Review rules",
            "",
            "- Confirm or correct the dial boundary, shape, pivot, main measurement pointer, and scale arc from the thumbnail and source image.",
            "- Mark red set-point/maximum markers as `red_marker`; do not select them as the measurement pointer.",
            "- Enter reading, unit, range, and minimum division only by independent human review.",
            "- Keep `review_status=pending` and explain the issue in `comment` when the main pointer or scale cannot be determined; resolve it before acceptance.",
            "",
            "## Visual index",
            "",
            "![Development samples](selected_contact_sheet.jpg)",
            "",
        ]
    )
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    private_lines = [
        "# Frozen validation review package (private)",
        "",
        "This directory contains 40 frozen-validation rows. Do not copy its completed labels into the development package.",
        "The existing YOLO boxes in the manifest are annotation aids only and are forbidden as ROI input for frozen end-to-end predictions.",
        "",
        "| shape stratum | count |",
        "|---|---:|",
    ]
    for stratum in SHAPE_STRATA:
        private_lines.append(f"| {stratum} | {counts[('frozen_validation', stratum)]} |")
    private_lines.extend(
        [
            "",
            "Fill `review_frozen.csv` independently and expose it only to the acceptance evaluator.",
            "",
            "![Frozen samples](selected_contact_sheet_frozen.jpg)",
            "",
        ]
    )
    (private_dir / "review_frozen.md").write_text("\n".join(private_lines), encoding="utf-8")


def run_pipeline(
    *,
    source_root: Path,
    output_dir: Path,
    per_shape: int = 30,
    validation_per_shape: int = 10,
    seed: int = 20260822,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    image_paths = discover_images(source_root)
    if not image_paths:
        raise RuntimeError(f"no images found below {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprints = [fingerprint_image(path) for path in image_paths]
    cluster_ids = cluster_near_duplicates(fingerprints)
    records = [
        _record_for_image(path, source_root, fingerprint, cluster_id)
        for path, fingerprint, cluster_id in zip(image_paths, fingerprints, cluster_ids)
    ]
    selected = balanced_sample(
        records,
        per_shape=per_shape,
        validation_per_shape=validation_per_shape,
        seed=seed,
    )
    selected_ids = {record["record_id"] for record in selected}
    cluster_sizes = Counter(record["duplicate_cluster_id"] for record in records)

    private_dir = output_dir / "frozen_private"
    private_dir.mkdir(parents=True, exist_ok=True)
    with (private_dir / "preannotations_all.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest_header = {
            "schema_version": SCHEMA_VERSION,
            "source_root": source_root.as_posix(),
            "selection": {
                "seed": seed,
                "per_shape": per_shape,
                "validation_per_shape": validation_per_shape,
                "method": "shape-score ranking with one representative per exact/perceptual duplicate cluster",
            },
            "roi_provenance": "existing_yolo_label_for_annotation_only",
            "frozen_e2e_roi_policy": "Existing dataset YOLO labels are forbidden; predictions must locate the ROI from image pixels/model inference.",
    }
    dev_selected = [record for record in selected if record["sampling"]["split"] == "dev"]
    frozen_selected = [record for record in selected if record["sampling"]["split"] == "frozen_validation"]
    _json_dump(
        output_dir / "review_manifest.json",
        {**manifest_header, "partition": "dev", "records": sorted(dev_selected, key=lambda item: item["record_id"])},
    )
    _json_dump(
        private_dir / "review_frozen_manifest.json",
        {**manifest_header, "partition": "frozen_validation_private", "records": sorted(frozen_selected, key=lambda item: item["record_id"])},
    )
    _json_dump(
        private_dir / "combined_machine_manifest.json",
        {**manifest_header, "partition": "combined_machine_only", "records": sorted(selected, key=lambda item: item["record_id"])},
    )
    schema_source = Path(__file__).parent / "schemas" / "preannotation.schema.json"
    (output_dir / "preannotation.schema.json").write_text(
        schema_source.read_text(encoding="utf-8"), encoding="utf-8"
    )

    dev_thumbnails: list[tuple[str, np.ndarray]] = []
    frozen_thumbnails: list[tuple[str, np.ndarray]] = []
    for record in selected:
        image = _load_image(source_root / record["image"]["path"])
        overlay = _draw_overlay(image, record)
        split, stratum = record["sampling"]["split"], record["sampling"]["stratum"]
        if split == "dev":
            thumbnail_path = output_dir / "thumbnails" / "dev" / stratum / f"{record['record_id']}.jpg"
        else:
            thumbnail_path = private_dir / "thumbnails" / stratum / f"{record['record_id']}.jpg"
        max_dimension = max(overlay.shape[:2])
        if max_dimension > 900:
            ratio = 900 / max_dimension
            overlay = cv2.resize(overlay, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        _save_jpeg(thumbnail_path, overlay)
        entry = (f"{split[:3]} {stratum[:8]} {record['record_id'][:6]}", overlay)
        if split == "dev":
            dev_thumbnails.append(entry)
        else:
            frozen_thumbnails.append(entry)
    _make_contact_sheet(dev_thumbnails, output_dir / "selected_contact_sheet.jpg")
    _make_contact_sheet(frozen_thumbnails, private_dir / "selected_contact_sheet_frozen.jpg")
    _write_review_files(output_dir, selected)

    split_clusters: dict[str, set[str]] = defaultdict(set)
    for record in selected:
        split_clusters[record["sampling"]["split"]].add(record["duplicate_cluster_id"])
    leakage = sorted(split_clusters["dev"] & split_clusters["frozen_validation"])
    selected_counts = Counter(
        (record["sampling"]["split"], record["sampling"]["stratum"])
        for record in selected
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "image_count": len(records),
        "selected_count": len(selected_ids),
        "duplicate_cluster_count": len(cluster_sizes),
        "multi_image_cluster_count": sum(size > 1 for size in cluster_sizes.values()),
        "largest_cluster_size": max(cluster_sizes.values()),
        "exact_duplicate_hash_group_count": sum(
            count > 1 for count in Counter(record["image"]["sha256"] for record in records).values()
        ),
        "perceptual_cluster_with_multiple_unique_hashes_count": sum(
            len({record["image"]["sha256"] for record in records if record["duplicate_cluster_id"] == cluster_id}) > 1
            for cluster_id in cluster_sizes
        ),
        "cluster_method": "union-find transitive clustering over SHA-256 equality and pHash/color-histogram pair matches",
        "cluster_thresholds": {
            "strict_phash_hamming_max": 6,
            "relaxed_phash_hamming_max": 10,
            "relaxed_color_histogram_cosine_min": 0.995,
            "aspect_log_ratio_max": 0.08
        },
        "cluster_overmerge_limitation": "Union-find uses single-link transitivity; a chain of pairwise-similar images can over-merge a cluster. Human audit is still required.",
        "cross_split_duplicate_cluster_count": len(leakage),
        "cross_split_duplicate_clusters": leakage,
        "full_image_fallback_count": sum(
            record["auto_annotation"]["dial_boundary"]["source"] == "full_image_fallback"
            for record in records
        ),
        "selected_by_split_and_shape": {
            split: {stratum: selected_counts[(split, stratum)] for stratum in SHAPE_STRATA}
            for split in ("dev", "frozen_validation")
        },
        "ground_truth_policy": "No Markdown ground-truth file is read; review values remain null.",
        "roi_provenance": "existing_yolo_label_for_annotation_only",
        "frozen_e2e_roi_policy": "Existing dataset YOLO labels must not be used by frozen end-to-end prediction.",
        "review_partitioning": {
            "dev": "review.csv and review_manifest.json",
            "frozen_private": "frozen_private/review_frozen.csv and frozen_private/review_frozen_manifest.json",
            "combined_machine_only": "frozen_private/combined_machine_manifest.json",
            "all_machine_preannotations": "frozen_private/preannotations_all.jsonl"
        }
    }
    _json_dump(output_dir / "audit.json", summary)
    return summary

"""Map gauge values through physical tick marks rather than OCR text centres.

OCR labels are normally printed offset from the tick that they name.  A caller
supplies label coordinates and detected *primary* tick coordinates; this module
first associates each value with its nearest compatible tick, then interpolates
only between those tick anchors.  It is image-library independent and supports
both circular dial angles and arbitrary monotonic curve coordinates.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


@dataclass(frozen=True)
class CoordinateSystem:
    """Distance and unwrapping rules for tick coordinates.

    ``kind='circular'`` expects a positive period (360 degrees by default).
    ``kind='curve'`` treats coordinates as an ordinary ordered parameter, for
    example arc length along a non-circular scale.
    """

    kind: str = "circular"
    period: float = 360.0

    def __post_init__(self) -> None:
        if self.kind not in {"circular", "curve"}:
            raise ValueError("kind must be 'circular' or 'curve'")
        if self.kind == "circular" and (not math.isfinite(self.period) or self.period <= 0.0):
            raise ValueError("circular coordinates require a positive finite period")

    def normalize(self, coordinate: float) -> float:
        if not math.isfinite(float(coordinate)):
            raise ValueError("coordinate must be finite")
        return float(coordinate) % self.period if self.kind == "circular" else float(coordinate)

    def distance(self, first: float, second: float) -> float:
        first, second = self.normalize(first), self.normalize(second)
        if self.kind == "curve":
            return abs(first - second)
        raw = abs(first - second) % self.period
        return min(raw, self.period - raw)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "period": self.period if self.kind == "circular" else None}


@dataclass(frozen=True)
class PrimaryTick:
    """A physically detected major tick (not an OCR box centre)."""

    tick_id: str
    coordinate: float
    ring_id: str | None = None
    unit: str | None = None
    confidence: float = 1.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tick_id": self.tick_id,
            "coordinate": self.coordinate,
            "ring_id": self.ring_id,
            "unit": self.unit,
            "confidence": self.confidence,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class OCRNumericLabel:
    """One parsed numeric OCR label at its text-layout coordinate."""

    label_id: str
    value: float
    text_coordinate: float
    confidence: float
    ring_id: str | None = None
    unit: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "value": self.value,
            "text_coordinate": self.text_coordinate,
            "confidence": self.confidence,
            "ring_id": self.ring_id,
            "unit": self.unit,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class TickAssociation:
    status: str
    label: OCRNumericLabel
    tick: PrimaryTick | None
    distance: float | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": self.label.as_dict(),
            "tick": None if self.tick is None else self.tick.as_dict(),
            "distance": self.distance,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class TickPointerMapping:
    status: str
    value: float | None
    ring_id: str | None
    unit: str | None
    pointer_coordinate: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "ring_id": self.ring_id,
            "unit": self.unit,
            "pointer_coordinate": self.pointer_coordinate,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class PiecewiseTickScale:
    """A monotonic, piecewise-linear scale indexed by primary tick rays."""

    ring_id: str | None
    unit: str | None
    coordinate_system: CoordinateSystem
    origin: float
    tick_ids: tuple[str, ...]
    tick_coordinates: tuple[float, ...]
    unwrapped_coordinates: tuple[float, ...]
    values: tuple[float, ...]
    direction: str
    association_count: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def _unwrap_pointer(self, coordinate: float) -> float:
        normalized = self.coordinate_system.normalize(coordinate)
        if self.coordinate_system.kind == "curve":
            return normalized
        return (normalized - self.origin) % self.coordinate_system.period

    def map_pointer(self, pointer_coordinate: float) -> TickPointerMapping:
        if not _finite(pointer_coordinate):
            return TickPointerMapping(
                "no_output", None, self.ring_id, self.unit, pointer_coordinate,
                {"reason": "invalid_pointer_coordinate"},
            )
        coordinate = self._unwrap_pointer(float(pointer_coordinate))
        lower, upper = self.unwrapped_coordinates[0], self.unwrapped_coordinates[-1]
        if coordinate < lower - 1e-9 or coordinate > upper + 1e-9:
            return TickPointerMapping(
                "no_output", None, self.ring_id, self.unit, float(pointer_coordinate),
                {"reason": "pointer_outside_calibrated_range", "range": [lower, upper], "unwrapped_coordinate": coordinate},
            )
        index = bisect.bisect_right(self.unwrapped_coordinates, coordinate)
        if index == 0:
            value = self.values[0]
        elif index >= len(self.unwrapped_coordinates):
            value = self.values[-1]
        else:
            x0, x1 = self.unwrapped_coordinates[index - 1], self.unwrapped_coordinates[index]
            y0, y1 = self.values[index - 1], self.values[index]
            fraction = (coordinate - x0) / (x1 - x0)
            value = y0 + fraction * (y1 - y0)
        return TickPointerMapping(
            "ok", round(float(value), 9), self.ring_id, self.unit, float(pointer_coordinate),
            {"unwrapped_coordinate": round(coordinate, 9), "direction": self.direction},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ring_id": self.ring_id,
            "unit": self.unit,
            "coordinate_system": self.coordinate_system.as_dict(),
            "origin": self.origin,
            "tick_ids": list(self.tick_ids),
            "tick_coordinates": list(self.tick_coordinates),
            "unwrapped_coordinates": list(self.unwrapped_coordinates),
            "values": list(self.values),
            "direction": self.direction,
            "association_count": self.association_count,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class TickMappingResult:
    status: str
    coordinate_system: CoordinateSystem
    scales: tuple[PiecewiseTickScale, ...]
    associations: tuple[TickAssociation, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def map_pointer(
        self,
        pointer_coordinate: float,
        *,
        ring_id: str | None = None,
        unit: str | None = None,
    ) -> TickPointerMapping:
        candidates = list(self.scales)
        if ring_id is not None:
            candidates = [scale for scale in candidates if scale.ring_id == ring_id]
        if unit is not None:
            candidates = [scale for scale in candidates if scale.unit == unit]
        if not candidates:
            return TickPointerMapping(
                "no_output", None, ring_id, unit, pointer_coordinate,
                {"reason": "no_matching_tick_scale", "available_groups": self.available_groups()},
            )
        if len(candidates) != 1:
            return TickPointerMapping(
                "ambiguous", None, ring_id, unit, pointer_coordinate,
                {"reason": "multiple_tick_scales", "available_groups": self.available_groups()},
            )
        return candidates[0].map_pointer(pointer_coordinate)

    def available_groups(self) -> list[dict[str, str | None]]:
        return [{"ring_id": scale.ring_id, "unit": scale.unit} for scale in self.scales]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "coordinate_system": self.coordinate_system.as_dict(),
            "scales": [scale.as_dict() for scale in self.scales],
            "associations": [association.as_dict() for association in self.associations],
            "diagnostics": dict(self.diagnostics),
        }


def associate_ocr_to_ticks(
    labels: Sequence[OCRNumericLabel],
    ticks: Sequence[PrimaryTick],
    *,
    coordinate_system: CoordinateSystem = CoordinateSystem(),
    max_tick_distance: float = 15.0,
    ambiguity_distance: float = 0.25,
) -> list[TickAssociation]:
    """Associate every valid OCR numeric label to its nearest compatible tick."""
    if max_tick_distance < 0.0 or ambiguity_distance < 0.0:
        raise ValueError("distance tolerances must be non-negative")
    associations: list[TickAssociation] = []
    for label in labels:
        if not all(_finite(value) for value in (label.value, label.text_coordinate, label.confidence)):
            associations.append(TickAssociation("no_tick", label, None, None, {"reason": "invalid_label"}))
            continue
        if not 0.0 <= float(label.confidence) <= 1.0:
            associations.append(TickAssociation("no_tick", label, None, None, {"reason": "invalid_label_confidence"}))
            continue
        compatible = [tick for tick in ticks if _finite(tick.coordinate) and _finite(tick.confidence) and 0.0 <= float(tick.confidence) <= 1.0]
        if label.ring_id is not None:
            compatible = [tick for tick in compatible if tick.ring_id == label.ring_id]
        if label.unit is not None and any(tick.unit is not None for tick in compatible):
            compatible = [tick for tick in compatible if tick.unit == label.unit]
        if not compatible:
            associations.append(TickAssociation("no_tick", label, None, None, {"reason": "no_compatible_tick"}))
            continue
        ranked = sorted(
            ((coordinate_system.distance(label.text_coordinate, tick.coordinate), tick) for tick in compatible),
            key=lambda item: (item[0], -float(item[1].confidence), item[1].tick_id),
        )
        distance, nearest = ranked[0]
        tied = [tick.tick_id for tied_distance, tick in ranked if abs(tied_distance - distance) <= ambiguity_distance]
        if len(tied) > 1:
            associations.append(TickAssociation("ambiguous", label, None, distance, {"reason": "equidistant_ticks", "tick_ids": tied}))
        elif distance > max_tick_distance:
            associations.append(TickAssociation("no_tick", label, None, distance, {"reason": "nearest_tick_too_far", "max_tick_distance": max_tick_distance, "nearest_tick_id": nearest.tick_id}))
        else:
            associations.append(TickAssociation("associated", label, nearest, distance, {}))
    return associations


def _group_key(association: TickAssociation) -> tuple[str | None, str | None]:
    assert association.tick is not None
    return (
        association.tick.ring_id if association.tick.ring_id is not None else association.label.ring_id,
        association.tick.unit if association.tick.unit is not None else association.label.unit,
    )


def _ordered_anchors(
    anchors: Sequence[tuple[PrimaryTick, OCRNumericLabel]],
    coordinate_system: CoordinateSystem,
) -> tuple[float, list[tuple[float, PrimaryTick, OCRNumericLabel]], str] | None:
    """Pick a circular cut (or ordinary order) that makes values monotonic."""
    if coordinate_system.kind == "curve":
        ordered = sorted((float(tick.coordinate), tick, label) for tick, label in anchors)
        return _validate_monotonic(0.0, ordered)

    options: list[tuple[float, list[tuple[float, PrimaryTick, OCRNumericLabel]], str]] = []
    for origin in sorted({coordinate_system.normalize(tick.coordinate) for tick, _ in anchors}):
        ordered = sorted(
            (
                (coordinate_system.normalize(tick.coordinate) - origin) % coordinate_system.period,
                tick,
                label,
            )
            for tick, label in anchors
        )
        validated = _validate_monotonic(origin, ordered)
        if validated is not None:
            options.append(validated)
    if not options:
        return None
    # Stable choice: the smallest valid origin, then the broadest calibrated arc.
    return max(options, key=lambda item: (item[1][-1][0] - item[1][0][0], -item[0]))


def _validate_monotonic(
    origin: float,
    ordered: list[tuple[float, PrimaryTick, OCRNumericLabel]],
) -> tuple[float, list[tuple[float, PrimaryTick, OCRNumericLabel]], str] | None:
    coordinates = [item[0] for item in ordered]
    values = [float(item[2].value) for item in ordered]
    if any(second - first <= 1e-9 for first, second in zip(coordinates, coordinates[1:])):
        return None
    deltas = [second - first for first, second in zip(values, values[1:])]
    if all(delta > 1e-9 for delta in deltas):
        return origin, ordered, "increasing"
    if all(delta < -1e-9 for delta in deltas):
        return origin, ordered, "decreasing"
    return None


def fit_tick_mapping(
    labels: Sequence[OCRNumericLabel],
    ticks: Sequence[PrimaryTick],
    *,
    coordinate_system: CoordinateSystem = CoordinateSystem(),
    max_tick_distance: float = 15.0,
    ambiguity_distance: float = 0.25,
    min_anchors: int = 2,
) -> TickMappingResult:
    """Build one monotonic piecewise scale per tick ring/unit group."""
    if min_anchors < 2:
        raise ValueError("min_anchors must be at least 2")
    associations = associate_ocr_to_ticks(
        labels, ticks, coordinate_system=coordinate_system,
        max_tick_distance=max_tick_distance, ambiguity_distance=ambiguity_distance,
    )
    groups: dict[tuple[str | None, str | None], list[tuple[PrimaryTick, OCRNumericLabel]]] = {}
    rejected: list[dict[str, Any]] = []
    for association in associations:
        if association.status != "associated" or association.tick is None:
            rejected.append({"label_id": association.label.label_id, "reason": association.status, **dict(association.diagnostics)})
            continue
        groups.setdefault(_group_key(association), []).append((association.tick, association.label))

    scales: list[PiecewiseTickScale] = []
    for (ring_id, unit), anchors in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        per_tick: dict[str, tuple[PrimaryTick, OCRNumericLabel]] = {}
        conflict = False
        for tick, label in anchors:
            existing = per_tick.get(tick.tick_id)
            if existing is None:
                per_tick[tick.tick_id] = (tick, label)
            elif not math.isclose(existing[1].value, label.value, rel_tol=1e-9, abs_tol=1e-9):
                conflict = True
                rejected.append({"label_id": label.label_id, "reason": "conflicting_values_for_tick", "tick_id": tick.tick_id})
            elif label.confidence > existing[1].confidence:
                per_tick[tick.tick_id] = (tick, label)
        selected = list(per_tick.values())
        if conflict or len(selected) < min_anchors:
            continue
        ordered = _ordered_anchors(selected, coordinate_system)
        if ordered is None:
            rejected.extend({"label_id": label.label_id, "reason": "non_monotonic_tick_values", "ring_id": ring_id, "unit": unit} for _, label in selected)
            continue
        origin, sequence, direction = ordered
        scales.append(
            PiecewiseTickScale(
                ring_id=ring_id,
                unit=unit,
                coordinate_system=coordinate_system,
                origin=origin,
                tick_ids=tuple(tick.tick_id for _, tick, _ in sequence),
                tick_coordinates=tuple(round(float(tick.coordinate), 9) for _, tick, _ in sequence),
                unwrapped_coordinates=tuple(round(float(coordinate), 9) for coordinate, _, _ in sequence),
                values=tuple(round(float(label.value), 9) for _, _, label in sequence),
                direction=direction,
                association_count=len(selected),
            )
        )
    status = "ok" if scales else "no_output"
    return TickMappingResult(
        status, coordinate_system, tuple(scales), tuple(associations),
        {
            "label_count": len(labels),
            "tick_count": len(ticks),
            "associated_count": sum(item.status == "associated" for item in associations),
            "rejected": rejected,
            "available_groups": [{"ring_id": scale.ring_id, "unit": scale.unit} for scale in scales],
        },
    )


__all__ = [
    "CoordinateSystem",
    "OCRNumericLabel",
    "PiecewiseTickScale",
    "PrimaryTick",
    "TickAssociation",
    "TickMappingResult",
    "TickPointerMapping",
    "associate_ocr_to_ticks",
    "fit_tick_mapping",
]

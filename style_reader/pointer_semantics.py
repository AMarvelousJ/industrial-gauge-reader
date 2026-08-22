"""Semantic selection of an analog gauge's measurement pointer.

Geometry detectors can find a true needle together with red set-point flags,
peak markers, and unrelated radial strokes.  This module deliberately keeps
that distinction out of low-level image processing: its input is a list of
auditable candidates and its output never lets a detached marker replace the
main measurement pointer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


MARKER_ROLES = frozenset({"detached_marker", "setpoint", "peak_marker", "reference_marker"})


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _circular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class PointerCandidate:
    """One hypothesized radial feature, in clockwise degrees from dial top.

    ``pivot_connected`` is intentionally an explicit input.  Callers that
    have only a distance-to-pivot estimate may leave it as ``None`` and use
    ``pivot_distance_ratio``; a candidate without either kind of attachment
    evidence is diagnostic-only rather than a measurement result.
    """

    candidate_id: str
    angle_degrees: float | None
    confidence: float
    source: str
    pivot_connected: bool | None = None
    pivot_distance_ratio: float | None = None
    extent_ratio: float | None = None
    semantic_role: str = "auto"
    detached: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def resolved_role(self, *, max_pivot_distance_ratio: float) -> str:
        role = self.semantic_role.strip().lower()
        if self.detached or role in MARKER_ROLES:
            return "detached_marker"
        if self.pivot_connected is True:
            return "measurement"
        if self.pivot_connected is False:
            return "unattached"
        if _finite(self.pivot_distance_ratio):
            return (
                "measurement"
                if float(self.pivot_distance_ratio) <= max_pivot_distance_ratio
                else "unattached"
            )
        return "unverified_attachment"

    def as_dict(self, *, max_pivot_distance_ratio: float = 0.22) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "angle_degrees": None if self.angle_degrees is None else round(float(self.angle_degrees) % 360.0, 6),
            "confidence": round(float(self.confidence), 6),
            "source": self.source,
            "pivot_connected": self.pivot_connected,
            "pivot_distance_ratio": self.pivot_distance_ratio,
            "extent_ratio": self.extent_ratio,
            "semantic_role": self.semantic_role,
            "resolved_role": self.resolved_role(max_pivot_distance_ratio=max_pivot_distance_ratio),
            "detached": self.detached,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class PointerSelection:
    """Selection result with a safe, explicit no-output state."""

    status: str
    primary: PointerCandidate | None
    diagnostics: Mapping[str, Any]

    @property
    def angle_degrees(self) -> float | None:
        if self.status != "selected" or self.primary is None:
            return None
        if not _finite(self.primary.angle_degrees):
            return None
        return round(float(self.primary.angle_degrees) % 360.0, 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "angle_degrees": self.angle_degrees,
            "primary": None if self.primary is None else self.primary.as_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _measurement_score(candidate: PointerCandidate) -> float:
    """Conservative deterministic rank among already-attached candidates."""
    confidence = min(1.0, max(0.0, float(candidate.confidence)))
    extent_bonus = 0.0
    if _finite(candidate.extent_ratio):
        extent_bonus = 0.12 * min(1.0, max(0.0, float(candidate.extent_ratio)))
    pivot_bonus = 0.0
    if _finite(candidate.pivot_distance_ratio):
        pivot_bonus = 0.06 * max(0.0, 1.0 - float(candidate.pivot_distance_ratio) / 0.22)
    # A connected segmentation mask observes the full shaft, while Hough often
    # scores a long numeral or frame edge highly.  Keep the preference explicit
    # and auditable instead of letting incomparable raw confidences decide it.
    source_bonus = 0.22 if candidate.source == "mit_scale_segment_pointer_mask" else 0.0
    return confidence + extent_bonus + pivot_bonus + source_bonus


def select_primary_pointer(
    candidates: Sequence[PointerCandidate],
    *,
    max_pivot_distance_ratio: float = 0.22,
    ambiguity_margin: float = 0.06,
    same_ray_degrees: float = 5.0,
) -> PointerSelection:
    """Choose the pivot-attached measurement needle or return no result.

    A detached red/setpoint/peak marker is serialized in diagnostics with its
    rejection reason, even when it has a stronger raw detector confidence.
    Two close-ranked, materially different attached rays result in
    ``ambiguous`` rather than an arbitrary angle.
    """
    if not 0.0 < max_pivot_distance_ratio <= 1.0:
        raise ValueError("max_pivot_distance_ratio must be in (0, 1]")
    if ambiguity_margin < 0.0 or same_ray_degrees < 0.0:
        raise ValueError("ambiguity_margin and same_ray_degrees must be non-negative")

    rejected: list[dict[str, Any]] = []
    eligible: list[tuple[float, PointerCandidate]] = []
    for candidate in candidates:
        role = candidate.resolved_role(max_pivot_distance_ratio=max_pivot_distance_ratio)
        if not _finite(candidate.angle_degrees):
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "invalid_angle", "role": role})
            continue
        if not _finite(candidate.confidence) or not 0.0 <= float(candidate.confidence) <= 1.0:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "invalid_confidence", "role": role})
            continue
        if role != "measurement":
            rejected.append({"candidate_id": candidate.candidate_id, "reason": role, "role": role})
            continue
        eligible.append((_measurement_score(candidate), candidate))

    base_diagnostics = {
        "candidate_count": len(candidates),
        "eligible_measurement_count": len(eligible),
        "rejected_candidates": rejected,
        "marker_candidates": [
            candidate.as_dict(max_pivot_distance_ratio=max_pivot_distance_ratio)
            for candidate in candidates
            if candidate.resolved_role(max_pivot_distance_ratio=max_pivot_distance_ratio) == "detached_marker"
        ],
    }
    if not eligible:
        return PointerSelection("no_output", None, base_diagnostics)

    eligible.sort(key=lambda item: (-item[0], item[1].candidate_id))
    best_score, best = eligible[0]
    competitors = [
        {
            "candidate_id": candidate.candidate_id,
            "score": round(score, 6),
            "angle_distance_degrees": round(_circular_distance(float(best.angle_degrees), float(candidate.angle_degrees)), 6),
        }
        for score, candidate in eligible[1:]
        if best_score - score <= ambiguity_margin
        and _circular_distance(float(best.angle_degrees), float(candidate.angle_degrees)) > same_ray_degrees
    ]
    diagnostics = {
        **base_diagnostics,
        "selected_score": round(best_score, 6),
        "ranked_measurement_candidates": [
            {"candidate_id": candidate.candidate_id, "score": round(score, 6)}
            for score, candidate in eligible
        ],
    }
    if competitors:
        diagnostics["ambiguous_competitors"] = competitors
        return PointerSelection("ambiguous", None, diagnostics)
    return PointerSelection("selected", best, diagnostics)


def candidates_from_geometry(geometry: Mapping[str, Any]) -> list[PointerCandidate]:
    """Adapt existing geometry dictionaries without mutating them.

    Line candidates receive attachment evidence from their existing
    ``center_distance_ratio``.  A radial darkness peak has no observed shaft
    to the pivot and is deliberately diagnostic-only.  The colored candidate
    consumes the detector's ``detached_scale_marker`` flag.
    """
    candidates: list[PointerCandidate] = []
    colored = geometry.get("colored_pointer_candidate") or {}
    if colored:
        candidates.append(
            PointerCandidate(
                candidate_id="colored",
                angle_degrees=colored.get("angle_degrees"),
                confidence=float(colored.get("confidence", 0.0)),
                source="colored",
                pivot_connected=not bool(colored.get("detached_scale_marker", False)),
                extent_ratio=None,
                semantic_role="setpoint" if bool(colored.get("detached_scale_marker", False)) else "auto",
                detached=bool(colored.get("detached_scale_marker", False)),
                diagnostics={key: value for key, value in colored.items() if key not in {"angle_degrees", "confidence"}},
            )
        )
    for index, raw in enumerate(geometry.get("line_candidates") or []):
        candidates.append(
            PointerCandidate(
                candidate_id=f"line:{index}",
                angle_degrees=raw.get("angle_degrees"),
                confidence=float(raw.get("score", 0.0)),
                source="hough_line",
                pivot_distance_ratio=raw.get("center_distance_ratio"),
                extent_ratio=raw.get("length_ratio"),
                diagnostics={"line_index": index},
            )
        )
    radial = geometry.get("radial_scan") or {}
    if radial.get("angle_degrees") is not None:
        candidates.append(
            PointerCandidate(
                candidate_id="radial_scan",
                angle_degrees=radial.get("angle_degrees"),
                confidence=float(radial.get("confidence", 0.0)),
                source="radial_scan",
                pivot_connected=False,
                diagnostics=dict(radial),
            )
        )
    return candidates


__all__ = [
    "MARKER_ROLES",
    "PointerCandidate",
    "PointerSelection",
    "candidates_from_geometry",
    "select_primary_pointer",
]

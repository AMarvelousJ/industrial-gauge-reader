"""Auditable unit inference (T-D2-c).

Resolves the unit for a gauge whose scale fit succeeded but whose unit glyph
was not read by OCR (scale_points are present, reading_status == ok, unit is
None).  The module is deliberately pure and declarative:

    infer_unit(scale_points, ocr_text, meter_family_hint, existing_unit)
      -> {"unit", "confidence", "source", "candidates"}

sources:
  - "ocr"          strong nameplate / fragment evidence
  - "scale_range"  the scale's max value uniquely identifies one unit table
  - "family_prior" family hint + range (differential -> Pa, A/V meters...)
  - "mixed"        multiple unit tables match; industrial prior picks one with
                   a lower confidence and the full candidates list is kept

It NEVER overrides an existing unit: the caller invokes it only when
``unit is None``.  Every produced unit must be recorded as unit_inferred so the
result stays auditable (wrong-but-labeled inference is recoverable, while a
silently wrong unit is not).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Industrial nominal ranges per unit (tolerance 2%).  A value that matches
# exactly one table is a strong signal; several tables => "mixed".
# Non-standard but real scales (0.8, 8, 65, 70, 800, 2000, 7000) are included
# as candidates on every unit that plausibly uses them; the inference never
# hard-maps a scale to one unit - candidates are always reported.
RANGE_TABLES: dict[str, tuple[float, ...]] = {
    "bar": (0.8, 2.5, 4, 6, 8, 10, 16, 25, 40, 60, 100, 160, 250, 400, 600, 800, 1000, 1600, 2500),
    "MPa": (0.1, 0.16, 0.25, 0.4, 0.6, 0.8, 1, 1.6, 2.5, 4, 6, 8, 10, 16, 25, 40),
    "psi": (30, 60, 100, 160, 300, 600, 800, 1000, 2000, 3000, 5000, 7000, 10000),
    "degC": (50, 60, 65, 70, 80, 100, 120, 150, 160, 200, 250, 300, 400, 500, 600, 800),
    "kgf/cm2": (1, 2.5, 4, 6, 8, 10, 16, 25, 40, 60, 100, 250, 400, 600),
    "kPa": (8, 10, 16, 25, 40, 60, 65, 70, 100, 160, 250, 400, 600, 800, 1000, 2000, 7000),
    "Pa": (25, 50, 60, 100, 250, 500, 1000, 2000),
}

# Industrial prior weight when several tables match (bar dominates on the
# observed M01 corpus; degC next; MPa/psi common; kgf/kPa rarer).
PRIOR_WEIGHTS = {
    "bar": 1.00,
    "degC": 0.80,
    "MPa": 0.70,
    "psi": 0.55,
    "kPa": 0.45,
    "kgf/cm2": 0.40,
    "Pa": 0.30,
}

# Family prior: metre family hint -> unit candidates with weights.
FAMILY_PRIOR: dict[str, dict[str, float]] = {
    "single_pointer_differential": {"Pa": 1.0, "kPa": 0.5},
    "square_ammeter": {"A": 1.0},
    "square_voltmeter": {"V": 1.0},
    "round_temperature": {"degC": 1.0},
    "colored_zone": {"bar": 0.7, "MPa": 0.7, "kgf/cm2": 0.4},
    "round_pressure": {"bar": 1.0, "MPa": 0.8, "psi": 0.5, "kgf/cm2": 0.4},
}

# OCR nameplate / fragment hints (strongest evidence; used first).
OCR_HINTS: tuple[tuple[str, str, float], ...] = (
    ("rmometer", "degC", 0.85),
    ("thermometer", "degC", 0.85),
    ("temperature", "degC", 0.85),
    ("pascols", "Pa", 0.80),
    ("passts", "Pa", 0.80),
    ("pascals", "Pa", 0.80),
    ("pascal", "Pa", 0.80),
    ("magnehelic", "Pa", 0.80),
    ("macrohelic", "Pa", 0.80),
    ("differential pressure", "Pa", 0.80),
    ("pressure gauge", "Pa", 0.65),
    ("pressuregauge", "Pa", 0.65),
    ("max pressure", "Pa", 0.65),
    ("kg/cm", "kgf/cm2", 0.75),
    ("en 837", "bar", 0.60),
    ("en837", "bar", 0.60),
)


def _near(value: float, nominal: float) -> bool:
    return abs(value - nominal) / max(nominal, 1e-6) <= 0.02


def _range_matches(max_value: float | None) -> list[str]:
    if max_value is None or not math.isfinite(max_value):
        return []
    return [unit for unit, table in RANGE_TABLES.items() if any(_near(max_value, m) for m in table)]


def _infer_unit_core(
    scale_points: Sequence[dict[str, Any]] | None,
    ocr_text: str,
    meter_family_hint: str | None = None,
    existing_unit: str | None = None,
) -> dict[str, Any]:
    """Return an auditable unit verdict; never used when existing_unit is set.

    Returns {"unit": str|None, "confidence": float, "source": str,
    "candidates": [{unit, confidence}]}.
    """
    if existing_unit:
        return {"unit": None, "confidence": 0.0, "source": "already_set", "candidates": []}

    values = [float(p["value"]) for p in (scale_points or []) if p.get("value") is not None]
    max_value = max(values) if values else None
    lowered = (ocr_text or "").lower()

    # 1) OCR nameplate / fragment evidence (strongest).
    for fragment, unit, confidence in OCR_HINTS:
        if fragment in lowered:
            return {
                "unit": unit,
                "confidence": confidence,
                "source": "ocr",
                "candidates": [{"unit": unit, "confidence": confidence}],
            }
    # A single "MP" token (MPa, common OCR truncation).
    tokens = [t for t in lowered.replace(".", " ").split() if t]
    if "mp" in tokens:
        return {"unit": "MPa", "confidence": 0.70, "source": "ocr", "candidates": [{"unit": "MPa", "confidence": 0.70}]}

    # 2) Family prior + scale range.
    if meter_family_hint and meter_family_hint in FAMILY_PRIOR:
        family_candidates = FAMILY_PRIOR[meter_family_hint]
        family_list = [{"unit": u, "confidence": round(w, 3)} for u, w in family_candidates.items()]
        if meter_family_hint == "single_pointer_differential":
            return {"unit": "Pa", "confidence": 0.72, "source": "family_prior", "candidates": family_list}
        if meter_family_hint in ("square_ammeter", "square_voltmeter", "round_temperature"):
            unit = next(iter(family_candidates))
            return {"unit": unit, "confidence": 0.70, "source": "family_prior", "candidates": family_list}
        # pressure-like family: rank by (prior family weight * range match)
        matches = _range_matches(max_value) or list(family_candidates)
        scored = []
        for unit in matches:
            family_weight = float(family_candidates.get(unit, 0.2))
            scored.append((unit, round(family_weight * PRIOR_WEIGHTS.get(unit, 0.5), 3)))
        scored.sort(key=lambda item: -item[1])
        if scored and scored[0][1] >= 0.40:
            return {
                "unit": scored[0][0],
                "confidence": round(0.45 + 0.05 * scored[0][1], 2),
                "source": "family_prior",
                "candidates": scored,
            }

    # 3) Scale range only.
    matches = _range_matches(max_value)
    if len(matches) == 1:
        unit = matches[0]
        return {"unit": unit, "confidence": 0.62, "source": "scale_range", "candidates": [{"unit": unit, "confidence": 0.62}]}
    if len(matches) > 1:
        scored = sorted(
            [{"unit": u, "confidence": round(PRIOR_WEIGHTS.get(u, 0.4), 3)} for u in matches],
            key=lambda item: -item["confidence"],
        )
        if scored and scored[0]["confidence"] >= PRIOR_WEIGHTS[scored[0]["unit"]]:  # top prior is absolute
            return {
                "unit": scored[0]["unit"],
                "confidence": 0.40,
                "source": "mixed",
                "candidates": scored,
            }

    return {"unit": None, "confidence": 0.0, "source": "none", "candidates": []}


def infer_unit(
    scale_points: Sequence[dict[str, Any]] | None,
    ocr_text: str,
    meter_family_hint: str | None = None,
    existing_unit: str | None = None,
) -> dict[str, Any]:
    """Wrapper: attaches the low_confidence marker; unknown (<0.4) -> no unit."""
    verdict = _infer_unit_core(scale_points, ocr_text, meter_family_hint, existing_unit)
    confidence = float(verdict.get("confidence") or 0.0)
    if verdict.get("unit") is not None and confidence < 0.4:
        return {
            "unit": None,
            "confidence": verdict["confidence"],
            "source": "unknown_low_confidence",
            "candidates": verdict.get("candidates", []),
            "low_confidence": True,
        }
    return {
        **verdict,
        "low_confidence": confidence < 0.6,
    }

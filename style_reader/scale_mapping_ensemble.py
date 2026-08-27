"""Scale-mapping ensemble (T-D3).

Keeps the existing RANSAC/robust-fit mapping (global) and adds the local
anchor interpolation as a second candidate.  The two are computed in parallel
and reconciled:

  - global missing, local valid      -> use LOCAL (unblocks linear-fit failures)
  - both valid, |diff| <= tolerance  -> ENSEMBLE (weighted mean); the weighted
                                        value is recorded for audit, while the
                                        emitted value stays the global one so
                                        existing correct reads are untouched
  - both valid, |diff| >  tolerance  -> keep global, mapping_uncertain=True
  - local invalid                    -> keep global

Every verdict carries an auditable mapping_method (ransac / local / ensemble)
plus mapping_confidence; a disagreement that cannot be trusted is never used to
replace an existing reading.
"""

from __future__ import annotations

from typing import Any


def ensemble_mapping(
    global_value: float | None,
    local_value: float | None,
    *,
    value_span: float | None = None,
    tolerance_ratio: float = 0.15,
    global_confidence: float = 0.65,
    local_confidence: float = 0.70,
    global_extrapolation_degrees: float | None = None,
) -> dict[str, Any]:
    """Reconcile the RANSAC mapping (global) with the anchor interpolation
    (local) and return an auditable verdict.

    ``value_span`` = max(anchor values) - min(anchor values); the agreement
    tolerance defaults to 15% of it (one coarse division scale).
    """
    if global_value is None and local_value is None:
        return {"value": None, "method": "none", "confidence": 0.0, "mapping_uncertain": True, "weighted_value": None}

    if global_value is None or (global_extrapolation_degrees is not None and global_extrapolation_degrees > 10.0):
        # RANSAC failed OR extrapolated far beyond the calibrated arc: prefer the
        # local anchor interpolation when a bracket exists (no extrapolation).
        if local_value is None:
            return {
                "value": None if global_value is None else float(global_value),
                "method": "ransac",
                "confidence": global_confidence,
                "mapping_uncertain": True,
                "weighted_value": None,
            }
        return {
            "value": float(local_value),
            "method": "local",
            "confidence": local_confidence,
            "mapping_uncertain": False,
            "weighted_value": float(local_value),
        }

    if local_value is None:
        return {
            "value": float(global_value),
            "method": "ransac",
            "confidence": global_confidence,
            "mapping_uncertain": False,
            "weighted_value": float(global_value),
        }

    difference = abs(float(global_value) - float(local_value))
    tolerance = tolerance_ratio * abs(value_span) if value_span else None
    agreed = tolerance is not None and difference <= tolerance
    if agreed:
        total = global_confidence + local_confidence
        weighted = (
            global_confidence * float(global_value) + local_confidence * float(local_value)
        ) / total
        return {
            "value": float(global_value),  # keep the tested global value bit-identical
            "method": "ensemble",
            "confidence": round((global_confidence + local_confidence) / 2.0, 3),
            "mapping_uncertain": False,
            "weighted_value": round(float(weighted), 6),
            "agreement": {"difference": round(difference, 6), "tolerance": round(tolerance, 6)},
        }

    return {
        "value": float(global_value),
        "method": "ransac",
        "confidence": global_confidence,
        "mapping_uncertain": True,
        "weighted_value": float(local_value),
        "agreement": {
            "difference": round(difference, 6),
            "tolerance": round(tolerance, 6) if tolerance is not None else None,
        },
    }

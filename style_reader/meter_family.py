"""Topology-aware gauge family routing (meter family).

Instead of a learned visual classifier, this module routes a dial into a small
set of TOPOLOGY families using the geometry signals the reading pipeline already
produces: the dial-shape estimate, the crop aspect, the pointer-mask pivot offset
(off-centre hub = rectangular / box gauge), and - as a WEAK auxiliary signal only -
the dial colour-band statistics.  OCR is deliberately NOT a primary input.

Version 1 exposes exactly four labels:

    circle           round front/perspective dial, centred needle
    rectangular_box  rectangular housing / box gauge (Magnehelic, square meter):
                     off-centre pivot, hub far from the fitted circle
    colored_zone     dial with a green/red colour band (zone-style gauge)
    unknown          insufficient evidence

Purpose: the technical report needs an auditable "meter topology awareness"
field per reading; it is display/metadata only and never influences the reading
pipeline (no reading logic is touched, predictions.json numeric fields are
unchanged - only a new diagnostics field is added).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

FAMILY_SCHEMA_VERSION = "1.0"
FAMILY_METHOD = "topology_rules_v2_fallback"
BOX_ASPECT_THRESHOLD = 1.35
MASK_OFF_CENTER_RATIO_THRESHOLD = 0.25
COLOR_ZONE_RATIO_THRESHOLD = 0.08
# Last-resort weak evidence: box-gauge nameplate vocabulary (Magnehelic).  OCR is
# deliberately a WEAK auxiliary only - it never overrides a strong signal and can
# only rescue a dial that would otherwise default to circle.  Circular gauges do
# not carry these tokens (e.g. WIKAI / GASSYSTEM dials), so no circular dial is
# re-labelled by this fallback.
BOX_OCR_VOCABULARY = (
    "magnehelic",
    "macrohelic",  # common OCR misspelling of the Magnehelic nameplate
    "pascals",
    "max.pressure",
    "max pressure",
    "differential pressure",
)


def color_zone_stats(crop_bgr: np.ndarray) -> dict[str, float]:
    """Weak auxiliary signal: fraction of saturated green/red dial-zone pixels.

    A green/red coloured band (colour-zone gauge) occupies a noticeable share of
    the dial; a plain dial does not.  This is a weak signal only - the topology
    rules treat it as supporting evidence, never as the deciding one.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"green_ratio": 0.0, "red_ratio": 0.0, "zone_ratio": 0.0}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green = ((hue >= 35) & (hue <= 90) & (sat >= 60) & (val >= 40)).astype(np.float32)
    red = (((hue <= 15) | (hue >= 165)) & (sat >= 60) & (val >= 40)).astype(np.float32)
    total = max(1.0, float(crop_bgr.shape[0] * crop_bgr.shape[1]))
    green_ratio = float(green.sum()) / total
    red_ratio = float(red.sum()) / total
    return {
        "green_ratio": round(green_ratio, 6),
        "red_ratio": round(red_ratio, 6),
        "zone_ratio": round(green_ratio + red_ratio, 6),
    }


def _dial_type(geometry: Mapping[str, Any], dial_geometry: Any) -> tuple[str, float]:
    """Extract (geometry_type, confidence) from either a DialGeometry object or a dict."""
    if isinstance(dial_geometry, Mapping):
        gtype = str(dial_geometry.get("geometry_type") or "")
        conf = float(dial_geometry.get("confidence") or 0.0)
        return gtype, conf
    gtype = str(getattr(dial_geometry, "geometry_type", "") or "")
    conf = float(getattr(dial_geometry, "confidence", 0.0) or 0.0)
    return gtype, conf


def classify_family(
    geometry: Mapping[str, Any],
    segmented: Mapping[str, Any] | None,
    dial_geometry: Any | None,
    image_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Route a dial into a topology family.  Pure rule function - no reading logic.

    Args:
        geometry: the analyze_pointer geometry dict (never None).
        segmented: the pointer-mask dict from segmented_pointer_angle (may be None).
        dial_geometry: DialGeometry (or its as_dict) - may be None on missed frames.
        image_meta: optional {"crop_width", "crop_height", "color_zone_ratio"}.

    Returns:
        {"schema_version", "label", "confidence", "reasons", "signals", "method"}
    """
    reasons: list[str] = []
    signals: dict[str, Any] = {}

    geometry_type, dial_confidence = _dial_type(geometry, dial_geometry)
    signals["geometry_type"] = geometry_type
    signals["dial_confidence"] = round(dial_confidence, 4)

    # --- Unknown: no evidence or failed geometry -------------------------------
    if dial_geometry is None or not geometry_type:
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "label": "unknown",
            "confidence": 0.3,
            "reasons": ["no_dial_geometry_evidence"],
            "signals": signals,
            "method": FAMILY_METHOD,
        }
    if geometry_type in ("", "roi_fallback") and dial_confidence < 0.5:
        reasons.append(f"low_evidence_geometry_type={geometry_type or 'none'}_confidence={dial_confidence:.2f}")
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "label": "unknown",
            "confidence": 0.3,
            "reasons": reasons,
            "signals": signals,
            "method": FAMILY_METHOD,
        }

    # --- Rectangular / box: three independent topology signals -----------------
    box_confidence = 0.0
    box_why: list[str] = []
    if geometry_type == "rectangular_sector":
        box_confidence = max(box_confidence, 0.92)
        box_why.append("geometry_type=rectangular_sector")
    aspect: float | None = None
    if image_meta:
        width = float(image_meta.get("crop_width") or 0.0)
        height = float(image_meta.get("crop_height") or 0.0)
        if width > 0 and height > 0:
            aspect = round(max(width, height) / max(min(width, height), 1e-6), 4)
            if aspect >= BOX_ASPECT_THRESHOLD:
                box_confidence = max(box_confidence, 0.8)
                box_why.append(f"crop_aspect={aspect}>={BOX_ASPECT_THRESHOLD}")
    signals["crop_aspect"] = aspect
    ratio: float | None = None
    if segmented is not None:
        ratio = segmented.get("minimum_center_distance_ratio")
        if ratio is None:
            # The raw mask dict carries pixel distance only; the normalised ratio
            # is stored by the pipeline on geometry["segmented_pointer"].
            ratio = ((geometry or {}).get("segmented_pointer") or {}).get("minimum_center_distance_ratio")
        if ratio is not None and float(ratio) > MASK_OFF_CENTER_RATIO_THRESHOLD:
            box_confidence = max(box_confidence, 0.78)
            box_why.append(f"mask_pivot_offset_ratio={float(ratio):.2f}>{MASK_OFF_CENTER_RATIO_THRESHOLD}")
    signals["mask_pivot_offset_ratio"] = ratio
    if box_confidence > 0.0:
        reasons.extend(box_why)
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "label": "rectangular_box",
            "confidence": box_confidence,
            "reasons": reasons,
            "signals": signals,
            "method": FAMILY_METHOD,
        }

    # --- Coloured zone (weak colour-band evidence, circular dials) -------------
    zone_ratio = None
    if image_meta:
        zone_ratio = image_meta.get("color_zone_ratio")
    signals["color_zone_ratio"] = zone_ratio
    if zone_ratio is not None and float(zone_ratio) >= COLOR_ZONE_RATIO_THRESHOLD:
        reasons.append(f"color_zone_ratio={float(zone_ratio):.3f}>={COLOR_ZONE_RATIO_THRESHOLD}")
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "label": "colored_zone",
            "confidence": 0.8,
            "reasons": reasons,
            "signals": signals,
            "method": FAMILY_METHOD,
        }

    # --- Weak fallback: box-gauge OCR vocabulary (never overrides strong signs) -
    # When segmentation misses the pointer (mask absent) and the dial still looks
    # like a circle, a box-gauge nameplate token is the only remaining evidence
    # that this is a rectangular-housing gauge.  Recorded as a fallback with a
    # lower confidence so the label stays auditable.
    ocr_text = str((image_meta or {}).get("ocr_text") or "").lower()
    ocr_hits = [keyword for keyword in BOX_OCR_VOCABULARY if keyword in ocr_text]
    signals["ocr_box_keyword_hits"] = ocr_hits
    signals["family_fallback_used"] = bool(ocr_hits)
    # Informational fingerprint only (never changes the label): an elongated
    # detached red needle hints at an off-centre hub, but it also occurs on
    # ordinary circular gauges (red-tip needles), so it is recorded, not used.
    colored_raw = (geometry or {}).get("colored_pointer_candidate") or {}
    red_off_center = (
        float(colored_raw.get("elongation") or 0.0) >= 5.0
        and bool(colored_raw.get("detached_scale_marker"))
        and float(colored_raw.get("extent_ratio") or 0.0) >= 0.45
    )
    signals["red_needle_off_center_fingerprint"] = red_off_center
    if ocr_hits:
        reasons.append(f"fallback_ocr_box_vocabulary({ocr_hits[0]})")
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "label": "rectangular_box",
            "confidence": 0.62,
            "reasons": reasons,
            "signals": signals,
            "method": FAMILY_METHOD,
        }

    # --- Default circle topology -----------------------------------------------
    reasons.append(f"default_circular_topology({geometry_type}, confidence={dial_confidence:.2f})")
    return {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "label": "circle",
        "confidence": 0.7,
        "reasons": reasons,
        "signals": signals,
        "method": FAMILY_METHOD,
    }


__all__ = [
    "FAMILY_LABELS",
    "classify_family",
    "color_zone_stats",
]

FAMILY_LABELS = ("circle", "rectangular_box", "colored_zone", "unknown")

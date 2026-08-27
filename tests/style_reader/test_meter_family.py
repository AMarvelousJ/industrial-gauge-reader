from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from style_reader.meter_family import classify_family, color_zone_stats


class _Dial:
    def __init__(self, geometry_type: str, confidence: float = 0.9) -> None:
        self.geometry_type = geometry_type
        self.confidence = confidence


def _geometry(dial_type: str = "front_circle", confidence: float = 0.9) -> dict:
    return {"status": "angle_estimated", "circle": {"center_x": 100.0, "center_y": 100.0, "radius": 50.0}}


def test_circle_default() -> None:
    result = classify_family(_geometry(), None, _Dial("front_circle"), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "circle"
    assert result["schema_version"] == "1.0"
    assert result["method"] == "topology_rules_v2_fallback"
    assert result["confidence"] >= 0.5
    assert any("default_circular_topology" in reason for reason in result["reasons"])


def test_rectangular_sector_routes_to_box() -> None:
    result = classify_family(_geometry(), None, _Dial("rectangular_sector"), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "rectangular_box"
    assert "geometry_type=rectangular_sector" in result["reasons"]


def test_elongated_crop_routes_to_box() -> None:
    result = classify_family(_geometry(), None, _Dial("front_circle"), {"crop_width": 640, "crop_height": 400})
    assert result["label"] == "rectangular_box"
    assert any("crop_aspect" in reason for reason in result["reasons"])


def test_off_centre_mask_routes_to_box() -> None:
    segmented = {"minimum_center_distance_ratio": 0.38, "confidence": 0.9, "pca_pivot": [10, 10]}
    result = classify_family(_geometry(), segmented, _Dial("front_circle"), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "rectangular_box"
    assert any("mask_pivot_offset_ratio" in reason for reason in result["reasons"])


def test_centred_mask_keeps_circle() -> None:
    segmented = {"minimum_center_distance_ratio": 0.10, "confidence": 0.9}
    result = classify_family(_geometry(), segmented, _Dial("front_circle"), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "circle"


def test_ratio_from_geometry_segmented_pointer_routes_box() -> None:
    # Pipeline stores the normalised mask offset on geometry, not on the raw mask dict.
    geometry = {"segmented_pointer": {"minimum_center_distance_ratio": 0.31}}
    result = classify_family(geometry, {"minimum_center_distance": 61.0}, _Dial("front_circle"), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "rectangular_box"


def test_color_zone_ratio_routes_to_colored_zone() -> None:
    result = classify_family(
        _geometry(), None, _Dial("front_circle"), {"crop_width": 400, "crop_height": 400, "color_zone_ratio": 0.15}
    )
    assert result["label"] == "colored_zone"
    assert any("color_zone_ratio" in reason for reason in result["reasons"])


def test_box_has_priority_over_color_zone() -> None:
    segmented = {"minimum_center_distance_ratio": 0.40, "confidence": 0.9}
    result = classify_family(
        _geometry(), segmented, _Dial("rectangular_sector"), {"crop_width": 400, "crop_height": 400, "color_zone_ratio": 0.9}
    )
    assert result["label"] == "rectangular_box"


def test_unknown_without_dial_geometry() -> None:
    result = classify_family(_geometry(), None, None, {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "unknown"
    assert result["confidence"] == 0.3


def test_unknown_on_low_evidence_fallback() -> None:
    result = classify_family(_geometry(), None, _Dial("roi_fallback", confidence=0.25), {"crop_width": 400, "crop_height": 400})
    assert result["label"] == "unknown"


def test_color_zone_stats_on_blank_image_is_zero() -> None:
    image = np.full((200, 200, 3), 245, dtype=np.uint8)
    stats = color_zone_stats(image)
    assert stats["zone_ratio"] <= 1e-6


def test_color_zone_stats_detects_green_band() -> None:
    image = np.full((200, 200, 3), 245, dtype=np.uint8)
    image[80:120, 40:200] = (80, 200, 90)  # BGR: a green band
    stats = color_zone_stats(image)
    assert stats["green_ratio"] > 0.05
    assert stats["zone_ratio"] >= 0.05


def test_ocr_box_vocabulary_rescues_missed_mask_gauge() -> None:
    # RG-019 scenario: no mask, circular dial geometry, elongated detached red
    # needle - only the Magnehelic nameplate tokens identify it as a box gauge.
    geometry = _geometry()
    geometry["colored_pointer_candidate"] = {
        "tip": [318, 190], "elongation": 9.5, "detached_scale_marker": True,
        "extent_ratio": 0.51, "base_distance_ratio": 0.37, "confidence": 0.72,
    }
    result = classify_family(
        geometry, None, _Dial("front_circle", confidence=0.997),
        {"crop_width": 500, "crop_height": 515, "color_zone_ratio": 0.01,
         "ocr_text": "pascals 20 30 40 10 50 0 60 MACROHELIC CALIBRATED FOR MAX.PRESSUREOOkPa 2000-60pa CAUTION"},
    )
    assert result["label"] == "rectangular_box"
    assert result["confidence"] == 0.62
    assert any("fallback_ocr_box_vocabulary" in r for r in result["reasons"])
    assert result["signals"]["family_fallback_used"] is True
    assert result["signals"]["red_needle_off_center_fingerprint"] is True


def test_circular_dial_never_matches_ocr_vocabulary() -> None:
    # RG-003 scenario: circular dial with its own nameplate tokens (WIKAI etc.)
    geometry = _geometry()
    geometry["colored_pointer_candidate"] = {
        "tip": [374, 190], "elongation": 5.3, "detached_scale_marker": True,
        "extent_ratio": 0.54, "base_distance_ratio": 0.44, "confidence": 0.8,
    }
    result = classify_family(
        geometry, None, _Dial("front_circle", confidence=0.96),
        {"crop_width": 640, "crop_height": 640, "color_zone_ratio": 0.01,
         "ocr_text": "30 20 40 EN13190 10 50- 0 60 GASSYSTEM WIKAI 108J3S8"},
    )
    assert result["label"] == "circle"
    assert result["signals"]["ocr_box_keyword_hits"] == []


def test_strong_box_signal_beats_ocr_vocabulary() -> None:
    # mask offset remains the stronger evidence; fallback must not override it
    geometry = {"segmented_pointer": {"minimum_center_distance_ratio": 0.30}}
    result = classify_family(
        geometry, {"minimum_center_distance": 61.0}, _Dial("front_circle"),
        {"crop_width": 447, "crop_height": 442, "ocr_text": "WIKAI GASSYSTEM"},
    )
    assert result["label"] == "rectangular_box"
    assert result["confidence"] == 0.78  # strong path, not the 0.62 fallback

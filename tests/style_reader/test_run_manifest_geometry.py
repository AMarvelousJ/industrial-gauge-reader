from __future__ import annotations

import json

import pytest

from pointer_keypoints.contract import KeypointEstimate
from style_reader.dial_geometry import geometry_from_ellipse, geometry_from_quadrilateral
from style_reader.run_manifest import (
    build_stage_diagnostics,
    load_image_list,
    keypoint_in_working_space,
    merge_tick_reading,
    normalization_policy,
    pointer_semantics,
    source_pointer_overlay,
    transform_ocr_to_canonical,
)


def test_normalization_policy_only_applies_to_strong_shape_evidence() -> None:
    strong = geometry_from_ellipse((300, 200), (420, 210), 0, confidence=0.92, reprojection_error=2.0)
    mild = geometry_from_ellipse((300, 200), (330, 300), 0, confidence=0.95, reprojection_error=2.0)
    rectangle = geometry_from_quadrilateral(
        [(20, 20), (380, 30), (370, 380), (30, 370)],
        confidence=0.8,
        reprojection_error=3.0,
    )

    assert normalization_policy(strong) == (True, "strong_perspective_ellipse")
    assert normalization_policy(mild) == (False, "legacy_round_reader")
    assert normalization_policy(rectangle) == (True, "rectangular_sector")


def test_detached_marker_is_rejected_during_manifest_selection() -> None:
    geometry = {
        "status": "angle_estimated",
        "angle_degrees_clockwise_from_top": 220.0,
        "pointer_method": "hsv_red_pointer",
        "pointer_confidence": 0.95,
        "colored_pointer_candidate": {
            "angle_degrees": 220.0,
            "confidence": 0.95,
            "detached_scale_marker": True,
        },
        "line_candidates": [
            {
                "angle_degrees": 42.0,
                "score": 0.70,
                "center_distance_ratio": 0.04,
                "length_ratio": 0.75,
            }
        ],
        "radial_scan": {"angle_degrees": 220.0, "confidence": 0.8},
    }

    pointer_semantics(geometry, segmented=None, radius=200.0)

    assert geometry["angle_degrees_clockwise_from_top"] == 42.0
    assert geometry["selected_pointer_role"] == "measurement"
    assert geometry["pointer_method"] == "semantic_main:hough_line"
    assert geometry["pointer_selection"]["diagnostics"]["marker_candidates"]


def test_validated_keypoints_take_priority_over_legacy_pointer_candidates() -> None:
    geometry = {
        "status": "angle_estimated",
        "line_candidates": [
            {"angle_degrees": 210.0, "score": 0.99, "center_distance_ratio": 0.01, "length_ratio": 0.8}
        ],
    }
    keypoint = KeypointEstimate(
        "accepted", (100.0, 100.0), (140.0, 100.0), 0.82, 0.78, 90.0, 0.2, 0.5
    )

    pointer_semantics(geometry, segmented=None, radius=200.0, keypoint=keypoint)

    assert geometry["angle_degrees_clockwise_from_top"] == pytest.approx(90.0)
    assert geometry["pointer_method"] == "semantic_main:pivot_tip_pose_model"
    assert geometry["pointer_selection"]["primary"]["candidate_id"] == "learned_keypoints"


def test_keypoints_are_transformed_with_rectified_dial() -> None:
    dial = geometry_from_ellipse((300, 180), (400, 200), 0, confidence=0.9)
    estimate = KeypointEstimate(
        "accepted", (300.0, 180.0), (300.0, 80.0), 0.9, 0.9, 0.0, 0.25, 0.5
    )

    transformed = keypoint_in_working_space(
        estimate,
        dial,
        apply_normalization=True,
        working_shape=(512, 512, 3),
    )

    assert transformed.coordinate_system == "canonical"
    assert transformed.pivot == pytest.approx(dial.canonical_pivot)
    assert transformed.angle_degrees_clockwise_from_top == pytest.approx(0.0)


def test_canonical_pointer_overlay_is_mapped_back_to_source_crop() -> None:
    dial = geometry_from_ellipse((300, 180), (400, 200), 0, confidence=0.9)
    overlay = source_pointer_overlay(dial, angle=0.0, radius=230.0)

    assert overlay is not None
    assert abs(overlay["center"][0] - 300.0) < 1e-3
    assert overlay["tip"][1] < overlay["center"][1]


def test_source_ocr_text_is_preserved_while_coordinates_are_rectified() -> None:
    dial = geometry_from_ellipse((300, 180), (400, 200), 0, confidence=0.9)
    source = {
        "items": [
            {
                "text": "200",
                "score": 0.98,
                "box": [[280, 70], [320, 70], [320, 90], [280, 90]],
                "center": [300, 80],
            }
        ],
        "elapsed_seconds": 0.1,
    }

    transformed = transform_ocr_to_canonical(source, dial)

    assert transformed["items"][0]["text"] == "200"
    assert transformed["items"][0]["source_box"] == source["items"][0]["box"]
    assert transformed["items"][0]["center"] != source["items"][0]["center"]


def test_stage_diagnostics_explain_rectangular_pivot_and_tick_fallback() -> None:
    diagnostics = build_stage_diagnostics(
        {
            "dial_geometry": {
                "geometry_type": "rectangular_sector",
                "confidence": 0.84,
                "reprojection_error": 5.4,
            },
            "pointer_selection": {"status": "selected", "diagnostics": {}},
            "pointer_confidence": 0.9,
            "tick_mapping": {"status": "no_output"},
            "segmented_pointer": {"minimum_center_distance_ratio": 0.44},
        }
    )

    assert diagnostics["issues"] == [
        "physical_tick_mapping_unavailable_legacy_scale_fallback_used",
        "rectangular_virtual_pivot_uncertain",
    ]


def test_truth_free_image_list_rejects_poison_reading_field(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "gauge.jpg").write_bytes(b"not decoded by list loader")
    image_list = tmp_path / "images.json"
    image_list.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "images": [
                    {"sample_id": "T-1", "path": "gauge.jpg", "reading": "POISON-12345"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only sample_id and path"):
        load_image_list(image_list, dataset)


def test_truth_free_image_list_resolves_only_scoped_images(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = dataset / "gauge.jpg"
    image.write_bytes(b"not decoded by list loader")
    image_list = tmp_path / "images.json"
    image_list.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "images": [{"sample_id": "T-1", "path": "gauge.jpg"}],
            }
        ),
        encoding="utf-8",
    )

    entries = load_image_list(image_list, dataset)

    assert entries == [
        {"sample_id": "T-1", "relative_path": "gauge.jpg", "absolute_path": image.resolve()}
    ]


def test_valid_tick_mapping_replaces_legacy_when_coordinate_frame_is_trusted() -> None:
    legacy = {"status": "ok", "reading": 14.2, "method": "ocr_text_center"}
    tick = {
        "status": "ok",
        "reading": 15.0,
        "method": "tick_anchored_piecewise",
        "tick_mapping": {"status": "ok"},
        "pointer_mapping": {"status": "ok"},
    }

    merged = merge_tick_reading(legacy, tick, trusted=True)

    assert merged["reading"] == 15.0
    assert merged["method"] == "tick_anchored_piecewise"
    assert merged["legacy_mapping"] == legacy


def test_unverified_tick_peaks_remain_shadow_diagnostics() -> None:
    legacy = {"status": "ok", "reading": 14.2, "method": "ocr_text_center"}
    tick = {"status": "ok", "reading": 15.0, "method": "tick_anchored_piecewise"}

    merged = merge_tick_reading(legacy, tick, trusted=False)

    assert merged["reading"] == 14.2
    assert merged["tick_fallback_reason"] == "shadow_only_unverified_major_tick_or_ring"

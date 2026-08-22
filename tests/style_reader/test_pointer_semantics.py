from __future__ import annotations

import json

from style_reader.pointer_semantics import (
    PointerCandidate,
    candidates_from_geometry,
    select_primary_pointer,
)


def test_detached_red_setpoint_is_diagnostic_and_cannot_override_main_needle() -> None:
    selection = select_primary_pointer(
        [
            PointerCandidate(
                "red-setpoint", 205, 0.99, "colored", pivot_connected=False,
                semantic_role="setpoint", detached=True,
            ),
            PointerCandidate(
                "needle", 42, 0.66, "hough_line", pivot_connected=True,
                extent_ratio=0.72,
            ),
        ]
    )

    assert selection.status == "selected"
    assert selection.angle_degrees == 42.0
    assert selection.primary is not None and selection.primary.candidate_id == "needle"
    assert selection.diagnostics["marker_candidates"][0]["candidate_id"] == "red-setpoint"
    assert json.loads(json.dumps(selection.as_dict()))["primary"]["candidate_id"] == "needle"


def test_competing_attached_rays_produce_ambiguous_not_arbitrary_output() -> None:
    selection = select_primary_pointer(
        [
            PointerCandidate("first", 20, 0.82, "hough_line", pivot_connected=True),
            PointerCandidate("second", 155, 0.79, "hough_line", pivot_connected=True),
        ],
        ambiguity_margin=0.06,
    )

    assert selection.status == "ambiguous"
    assert selection.angle_degrees is None
    assert selection.diagnostics["ambiguous_competitors"][0]["candidate_id"] == "second"


def test_no_pivot_connected_candidate_is_explicit_no_output() -> None:
    selection = select_primary_pointer(
        [
            PointerCandidate("red", 40, 0.95, "colored", pivot_connected=False, detached=True),
            PointerCandidate("scan", 50, 0.80, "radial_scan", pivot_connected=False),
        ]
    )

    assert selection.status == "no_output"
    assert selection.primary is None
    assert {item["reason"] for item in selection.diagnostics["rejected_candidates"]} == {"detached_marker", "unattached"}


def test_geometry_adapter_preserves_detached_marker_and_only_accepts_attached_line() -> None:
    candidates = candidates_from_geometry(
        {
            "colored_pointer_candidate": {"angle_degrees": 220, "confidence": 0.95, "detached_scale_marker": True},
            "line_candidates": [
                {"angle_degrees": 65, "score": 0.70, "center_distance_ratio": 0.08, "length_ratio": 0.65},
            ],
            "radial_scan": {"angle_degrees": 190, "confidence": 0.90},
        }
    )

    selection = select_primary_pointer(candidates)
    assert selection.status == "selected"
    assert selection.primary is not None and selection.primary.candidate_id == "line:0"


def test_high_confidence_segmented_needle_outranks_nearby_hough_text_line() -> None:
    selection = select_primary_pointer(
        [
            PointerCandidate(
                "hough", 15, 0.97, "hough_line", pivot_distance_ratio=0.03,
                extent_ratio=1.2,
            ),
            PointerCandidate(
                "segmented", 22, 0.94, "mit_scale_segment_pointer_mask",
                pivot_distance_ratio=0.08, extent_ratio=0.85,
            ),
        ],
        ambiguity_margin=0.035,
    )

    assert selection.status == "selected"
    assert selection.primary is not None and selection.primary.candidate_id == "segmented"

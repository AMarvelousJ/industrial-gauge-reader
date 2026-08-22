from __future__ import annotations

import json

from style_reader.tick_mapping import (
    CoordinateSystem,
    OCRNumericLabel,
    PrimaryTick,
    fit_tick_mapping,
)


def test_circular_mapping_uses_nearest_tick_not_text_center_angle() -> None:
    ticks = [
        PrimaryTick("zero", 330),
        PrimaryTick("ten", 0),
        PrimaryTick("twenty", 30),
    ]
    labels = [
        OCRNumericLabel("zero-label", 0, 332, 0.99),
        OCRNumericLabel("ten-label", 10, 2, 0.99),
        OCRNumericLabel("twenty-label", 20, 32, 0.99),
    ]

    result = fit_tick_mapping(labels, ticks, max_tick_distance=5)
    mapped = result.map_pointer(15)

    assert result.status == "ok"
    assert [item.tick.coordinate for item in result.associations if item.tick] == [330, 0, 30]
    assert mapped.status == "ok"
    assert mapped.value == 15.0
    assert result.scales[0].tick_coordinates == (330.0, 0.0, 30.0)
    assert json.loads(json.dumps(result.as_dict()))["scales"][0]["values"] == [0.0, 10.0, 20.0]


def test_curve_coordinate_uses_monotonic_piecewise_linear_segments() -> None:
    ticks = [PrimaryTick("a", 0), PrimaryTick("b", 20), PrimaryTick("c", 50)]
    labels = [
        OCRNumericLabel("a-label", 0, 1, 0.95),
        OCRNumericLabel("b-label", 10, 19, 0.95),
        OCRNumericLabel("c-label", 30, 51, 0.95),
    ]

    result = fit_tick_mapping(
        labels, ticks, coordinate_system=CoordinateSystem("curve"), max_tick_distance=2,
    )

    assert result.map_pointer(10).value == 5.0
    assert result.map_pointer(35).value == 20.0
    assert result.map_pointer(55).status == "no_output"


def test_dual_ring_and_unit_groups_require_an_explicit_selection() -> None:
    ticks = [
        PrimaryTick("inner-0", 300, "inner", "MPa"),
        PrimaryTick("inner-1", 0, "inner", "MPa"),
        PrimaryTick("inner-2", 60, "inner", "MPa"),
        PrimaryTick("outer-0", 300, "outer", "psi"),
        PrimaryTick("outer-50", 0, "outer", "psi"),
        PrimaryTick("outer-100", 60, "outer", "psi"),
    ]
    labels = [
        OCRNumericLabel("i0", 0, 301, 0.98, "inner", "MPa"),
        OCRNumericLabel("i1", 1, 1, 0.98, "inner", "MPa"),
        OCRNumericLabel("i2", 2, 61, 0.98, "inner", "MPa"),
        OCRNumericLabel("o0", 0, 299, 0.98, "outer", "psi"),
        OCRNumericLabel("o50", 50, 1, 0.98, "outer", "psi"),
        OCRNumericLabel("o100", 100, 61, 0.98, "outer", "psi"),
    ]

    result = fit_tick_mapping(labels, ticks, max_tick_distance=3)

    assert result.status == "ok"
    assert result.map_pointer(30).status == "ambiguous"
    assert result.map_pointer(30, ring_id="inner").value == 1.5
    assert result.map_pointer(30, unit="psi").value == 75.0


def test_ambiguous_or_non_monotonic_tick_evidence_has_no_output() -> None:
    ambiguous = fit_tick_mapping(
        [OCRNumericLabel("label", 10, 10, 0.95)],
        [PrimaryTick("left", 5), PrimaryTick("right", 15)],
        max_tick_distance=10,
        ambiguity_distance=0.01,
    )
    non_monotonic = fit_tick_mapping(
        [
            OCRNumericLabel("a", 0, 0, 0.95),
            OCRNumericLabel("b", 10, 10, 0.95),
            OCRNumericLabel("c", 5, 20, 0.95),
        ],
        [PrimaryTick("a", 0), PrimaryTick("b", 10), PrimaryTick("c", 20)],
        coordinate_system=CoordinateSystem("curve"),
        max_tick_distance=0,
    )

    assert ambiguous.status == "no_output"
    assert ambiguous.associations[0].status == "ambiguous"
    assert non_monotonic.status == "no_output"
    assert non_monotonic.map_pointer(10).status == "no_output"

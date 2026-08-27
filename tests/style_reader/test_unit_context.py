from __future__ import annotations

from style_reader.run_manifest import infer_unit_from_context


def _mapping(texts: list[str], scale_values: list[float]) -> dict:
    return {
        "ocr": {"items": [{"text": t} for t in texts]},
        "scale_points": [{"value": v} for v in scale_values],
    }


def test_ipa_fragment_low_scale_is_mpa() -> None:
    unit, hint = infer_unit_from_context(_mapping(["0.02", "IPa", "0.1", "0.06"], [0.02, 0.06, 0.1]), {})
    assert unit == "MPa"
    assert hint == "ocr_fragment_ipa_low_scale"


def test_differential_nameplate_is_pa() -> None:
    unit, hint = infer_unit_from_context(_mapping(["APC", "DIFFERENTIAL PRESSUREGAUGE"], [10, 20, 30]), {})
    assert unit == "Pa"
    assert hint == "nameplate_differential"


def test_en837_bar_scale() -> None:
    unit, hint = infer_unit_from_context(_mapping(["EN837", "3"], [1.0, 2.0, 3.0]), {})
    assert unit == "bar"
    assert hint == "nameplate_en837_bar_scale"


def test_ambiguous_round_scale_is_none() -> None:
    # 0-0.8 MPa-vs-bar ambiguity without nameplate clues: no inference (safer).
    unit, hint = infer_unit_from_context(_mapping(["0.4", "0.6", "0.8", "0.2"], [0.2, 0.4, 0.6, 0.8]), {})
    assert unit is None
    assert hint is None


def test_square_meter_without_clue_is_none() -> None:
    unit, hint = infer_unit_from_context(_mapping(["40", "60", "SD.670"], [40.0, 60.0]), {})
    assert unit is None


def test_outlier_value_does_not_break_ipa_low_scale() -> None:
    # N-14 case: OCR reads a spurious 900; realistic scale is 0.02/0.06/0.1
    unit, hint = infer_unit_from_context(
        _mapping(["0.02", "IPa", "900", "0.1", "0.06"], [0.02, 0.02, 0.06, 0.1, 900.0]), {}
    )
    assert unit == "MPa"
    assert hint == "ocr_fragment_ipa_low_scale"

from __future__ import annotations

from style_reader.run_manifest import infer_unit_from_fragments


def _mapping(texts: list[str]) -> dict:
    return {"ocr": {"items": [{"text": text} for text in texts]}}


def test_thermometer_fragment_recovers_degc() -> None:
    unit, hint = infer_unit_from_fragments(_mapping(["TNRMOMETER", "DAEWON", "C5/"]))
    assert unit == "degC"
    assert hint == "rmometer"


def test_pascals_fragment_recovers_pa() -> None:
    unit, hint = infer_unit_from_fragments(_mapping(["Dwyer", "pascols", "MAGNEHELIC"]))
    assert unit == "Pa"


def test_differential_pressure_fragment_recovers_pa() -> None:
    unit, _ = infer_unit_from_fragments(_mapping(["APC", "DIFFERENTIAL PRESSUREGAUGE"]))
    assert unit == "Pa"


def test_mp_token_recovers_mpa() -> None:
    unit, hint = infer_unit_from_fragments(_mapping(["D", "CKD", "MP", "G59D"]))
    assert unit == "MPa"
    assert hint == "token:mp"


def test_no_fragment_returns_none() -> None:
    unit, hint = infer_unit_from_fragments(_mapping(["EN837", "1234"]))
    assert unit is None
    assert hint is None


def test_mp_token_not_fooled_by_inside_word() -> None:
    unit, _ = infer_unit_from_fragments(_mapping(["COMPRESSOR", "VACUUM"]))
    assert unit is None

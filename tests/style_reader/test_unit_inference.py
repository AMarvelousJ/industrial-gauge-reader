from __future__ import annotations

from style_reader.unit_inference import infer_unit


def _pts(*values: float) -> list[dict]:
    return [{"value": v} for v in values]


def test_existing_unit_never_overridden() -> None:
    result = infer_unit(_pts(10.0, 20.0), "", None, existing_unit="bar")
    assert result["unit"] is None
    assert result["source"] == "already_set"


def test_ocr_fragment_wins() -> None:
    result = infer_unit(_pts(50.0, 100.0), "pascols MAGNEHELIC", None)
    assert result["unit"] == "Pa"
    assert result["source"] == "ocr"
    assert result["confidence"] >= 0.7


def test_scale_range_unique_degc() -> None:
    result = infer_unit(_pts(40.0, 80.0, 120.0), "", None)
    assert result["unit"] == "degC"
    assert result["source"] == "scale_range"
    assert result["confidence"] == 0.62


def test_mixed_range_returns_prior_unit_with_candidates() -> None:
    result = infer_unit(_pts(0.0, 5.0, 10.0), "", None)
    assert result["unit"] in ("bar", "MPa", "kgf/cm2", "kPa")
    assert result["source"] == "mixed"
    assert result["confidence"] == 0.40
    assert len(result["candidates"]) > 1


def test_family_prior_differential_is_pa() -> None:
    result = infer_unit(_pts(10.0, 20.0, 30.0), "", "single_pointer_differential")
    assert result["unit"] == "Pa"
    assert result["source"] == "family_prior"


def test_family_prior_square_ammeter_is_a() -> None:
    result = infer_unit(_pts(10.0, 50.0, 100.0), "", "square_ammeter")
    assert result["unit"] == "A"
    assert result["source"] == "family_prior"


def test_no_signal_returns_none() -> None:
    result = infer_unit(_pts(0.8, 1.6, 2.4), "", None)
    assert result["unit"] is None
    assert result["source"] == "none"

from __future__ import annotations

from style_reader.scale_mapping_ensemble import ensemble_mapping


def test_global_missing_local_fills() -> None:
    result = ensemble_mapping(None, 3.03, value_span=60.0)
    assert result["value"] == 3.03
    assert result["method"] == "local"
    assert result["mapping_uncertain"] is False


def test_both_missing_none() -> None:
    result = ensemble_mapping(None, None, value_span=60.0)
    assert result["value"] is None
    assert result["method"] == "none"


def test_agreement_keeps_global_and_records_weighted() -> None:
    result = ensemble_mapping(10.0, 10.4, value_span=10.0)
    assert result["value"] == 10.0          # bit-identical global
    assert result["method"] == "ensemble"
    assert result["weighted_value"] is not None
    assert abs(result["value"] - result["weighted_value"]) < 0.5


def test_disagreement_keeps_global_with_uncertainty() -> None:
    result = ensemble_mapping(10.0, 25.0, value_span=10.0)
    assert result["value"] == 10.0
    assert result["method"] == "ransac"
    assert result["mapping_uncertain"] is True


def test_local_missing_keeps_global() -> None:
    result = ensemble_mapping(186.98, None, value_span=200.0)
    assert result["value"] == 186.98
    assert result["method"] == "ransac"

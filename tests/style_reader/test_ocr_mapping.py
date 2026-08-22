import math

from style_reader.ocr_mapping import ScalePoint, fit_linear_scale, infer_tick_anchored_reading


def test_ransac_scale_fit_rejects_model_number() -> None:
    points = [
        ScalePoint(value=value, text=str(value), score=0.99, x=0, y=0, angle=30 + value * 4)
        for value in (0, 10, 20, 30, 40, 50, 60)
    ]
    points.append(ScalePoint(value=111, text="111", score=0.7, x=0, y=0, angle=190))

    result = fit_linear_scale(points, pointer_angle=130)

    assert result["status"] == "ok"
    assert result["inlier_count"] == 7
    assert abs(result["reading"] - 25.0) < 0.01


def test_tick_anchored_adapter_maps_from_physical_ticks() -> None:
    center = (200.0, 200.0)
    radius = 160.0

    def item(text: str, angle: float) -> dict:
        radians = math.radians(angle)
        x = center[0] + radius * 0.78 * math.sin(radians)
        y = center[1] - radius * 0.78 * math.cos(radians)
        return {"text": text, "score": 0.98, "center": [x, y], "box": []}

    ocr = {"items": [item("0", 333), item("10", 3), item("20", 33)]}
    result = infer_tick_anchored_reading(
        ocr,
        center,
        radius,
        pointer_angle=15.0,
        tick_angles=[330.0, 0.0, 30.0],
    )

    assert result["status"] == "ok"
    assert result["reading"] == 15.0
    assert result["associated_anchor_count"] == 3
    assert result["method"] == "tick_anchored_piecewise"

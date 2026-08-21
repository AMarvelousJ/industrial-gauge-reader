from style_reader.ocr_mapping import ScalePoint, fit_linear_scale


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


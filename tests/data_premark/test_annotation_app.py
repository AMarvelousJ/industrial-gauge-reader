from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from data_premark.annotation_app import (
    AnnotationStore,
    AnnotationValidationError,
    EDITABLE_FIELDS,
    create_server,
)


FIELDS = (
    "record_id", "split", "sampling_stratum", "image_path", "duplicate_cluster_id",
    "auto_shape", *EDITABLE_FIELDS, "thumbnail",
)


def _record(record_id: str, image_path: str, *, split: str = "dev") -> dict:
    return {
        "record_id": record_id,
        "image": {"path": image_path, "width": 200, "height": 100},
        "sampling": {"selected": True, "stratum": "circular_front", "split": split},
        "auto_annotation": {
            "dial_boundary": {
                "detector_box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
            },
            "shape": {"predicted": "circular_front"},
            "pivot": {"point": {"x": 0.5, "y": 0.5}},
            "selected_pointer_candidate_id": "pointer-1",
            "pointer_candidates": [
                {
                    "candidate_id": "pointer-1",
                    "line": {"start": {"x": 0.5, "y": 0.5}, "end": {"x": 0.8, "y": 0.5}},
                    "tip": {"x": 0.8, "y": 0.5},
                    "confidence": 0.9,
                }
            ],
        },
    }


def _fixture(tmp_path: Path, *, split: str = "dev", escaping_path: bool = False) -> tuple[Path, Path, Path]:
    source = tmp_path / "all_set"
    source.mkdir(parents=True)
    (source / "one.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (source / "two.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    records = [
        _record("record-1", "../outside.jpg" if escaping_path else "one.jpg", split=split),
        _record("record-2", "two.jpg", split=split),
    ]
    manifest = tmp_path / "review_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "partition": "dev" if split == "dev" else "frozen_validation_private",
                "source_root": str(source),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        # Deliberately reverse manifest order: joining by row number would corrupt labels.
        for record in reversed(records):
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "split": split,
                    "sampling_stratum": "circular_front",
                    "image_path": record["image"]["path"],
                    "duplicate_cluster_id": f"dup-{record['record_id']}",
                    "auto_shape": "circular_front",
                    "review_status": "pending",
                    "thumbnail": f"{record['record_id']}.jpg",
                }
            )
    return source, manifest, review


def _accepted_payload() -> dict[str, str]:
    return {
        "review_status": "accepted",
        "review_shape": "circular_front",
        "pivot_x": "0.5",
        "pivot_y": "0.5",
        "pointer_candidate_id": "pointer-1",
        "pointer_role": "measurement_pointer",
        "pointer_angle_deg": "359.5",
        "reading": "42.5",
        "unit": "bar",
        "range_min": "0",
        "range_max": "100",
        "minor_division": "1",
        "comment": "双刻度，主针清晰",
    }


def test_store_joins_by_record_id_and_round_trips_unicode_csv(tmp_path: Path) -> None:
    source, manifest, review = _fixture(tmp_path)
    store = AnnotationStore(review_csv=review, manifest_path=manifest, source_root=source)

    state = store.state()
    assert [item["record_id"] for item in state["items"]] == ["record-2", "record-1"]
    assert state["items"][0]["image"]["path"] == "two.jpg"

    result = store.save("record-1", _accepted_payload())

    assert result["completed"] == 1
    assert review.with_suffix(".csv.bak").is_file()
    with review.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["record_id"] for row in rows] == ["record-2", "record-1"]
    saved = next(row for row in rows if row["record_id"] == "record-1")
    assert saved["comment"] == "双刻度，主针清晰"
    assert saved["image_path"] == "one.jpg"
    assert saved["pointer_angle_deg"] == "359.5"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pivot_x", "1.2", "between 0 and 1"),
        ("pointer_angle_deg", "360", "must be in"),
        ("range_max", "0", "greater than"),
        ("minor_division", "-1", "greater than zero"),
        ("reading", "NaN", "finite"),
    ],
)
def test_invalid_completed_update_never_changes_csv(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source, manifest, review = _fixture(tmp_path)
    store = AnnotationStore(review_csv=review, manifest_path=manifest, source_root=source)
    before = review.read_bytes()
    payload = _accepted_payload()
    payload[field] = value

    with pytest.raises(AnnotationValidationError, match=message):
        store.save("record-1", payload)

    assert review.read_bytes() == before
    assert store.rows["record-1"]["review_status"] == "pending"


def test_external_csv_change_is_not_overwritten(tmp_path: Path) -> None:
    source, manifest, review = _fixture(tmp_path)
    store = AnnotationStore(review_csv=review, manifest_path=manifest, source_root=source)
    externally_changed = review.read_bytes() + b"\r\n"
    review.write_bytes(externally_changed)

    with pytest.raises(AnnotationValidationError, match="changed outside"):
        store.save("record-1", _accepted_payload())

    assert review.read_bytes() == externally_changed


def test_dev_mode_refuses_frozen_manifest_and_path_escape(tmp_path: Path) -> None:
    source, frozen_manifest, review = _fixture(tmp_path / "frozen", split="frozen_validation")
    with pytest.raises(AnnotationValidationError, match="refuses a non-dev manifest"):
        AnnotationStore(review_csv=review, manifest_path=frozen_manifest, source_root=source)

    source, manifest, review = _fixture(tmp_path / "escape", escaping_path=True)
    store = AnnotationStore(review_csv=review, manifest_path=manifest, source_root=source)
    with pytest.raises(AnnotationValidationError, match="escapes source root"):
        store.image("record-1")


def test_http_state_image_and_save_endpoints(tmp_path: Path) -> None:
    source, manifest, review = _fixture(tmp_path)
    store = AnnotationStore(review_csv=review, manifest_path=manifest, source_root=source)
    server = create_server(store=store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/api/state", timeout=3) as response:
            state = json.load(response)
        assert state["partition"] == "dev"
        assert state["total"] == 2

        with urlopen(f"{base}/api/image/record-1", timeout=3) as response:
            assert response.read() == b"\xff\xd8\xff\xd9"

        request = Request(
            f"{base}/api/records/record-1",
            data=json.dumps(_accepted_payload(), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            saved = json.load(response)
        assert saved["completed"] == 1

        with pytest.raises(HTTPError) as exc:
            urlopen(f"{base}/api/image/not-found", timeout=3)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

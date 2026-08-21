from pathlib import Path

from style_classifier.data import build_deduplicated_records, file_sha256


def _write(root: Path, style: str, name: str, payload: bytes) -> Path:
    path = root / style / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_hash_dedup_excludes_test_and_cross_style_conflicts(tmp_path: Path) -> None:
    test_image = _write(tmp_path, "M01", "test.jpg", b"test")
    _write(tmp_path, "M01", "test-copy.jpg", b"test")
    _write(tmp_path, "M01", "duplicate-a.jpg", b"same-class")
    _write(tmp_path, "M01", "duplicate-b.jpg", b"same-class")
    _write(tmp_path, "M01", "conflict-a.jpg", b"conflict")
    _write(tmp_path, "M02", "conflict-b.jpg", b"conflict")
    _write(tmp_path, "M02", "usable.jpg", b"usable")

    records, audit = build_deduplicated_records(
        tmp_path, ["M01", "M02"], excluded_hashes={file_sha256(test_image)}
    )

    assert {(record.style, record.path.name) for record in records} == {
        ("M01", "duplicate-a.jpg"),
        ("M02", "usable.jpg"),
    }
    assert audit["test_hash_files_excluded"] == 2
    assert audit["same_class_duplicate_files_removed"] == 1
    assert audit["cross_style_conflict_hash_count"] == 1
    assert audit["cross_style_conflict_file_count"] == 2

from pathlib import Path

from style_classifier.manifest import parse_markdown_manifest, valid_unique_entries


def test_manifest_reports_missing_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "all_set"
    image_dir = root / "M01" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"not decoded by parser")
    markdown = root / "labels.md"
    markdown.write_text(
        "| 图片名称 | 度数 |\n"
        "| --- | --- |\n"
        "| M01/images/one.jpg | 1 bar |\n"
        "| M01/images/one.jpg | 2 bar |\n"
        "| M01/images/ones.jpg | 3 bar |\n",
        encoding="utf-8",
    )

    audit = parse_markdown_manifest(markdown, root)

    assert audit.row_count == 3
    assert audit.unique_path_count == 2
    assert audit.available_row_count == 2
    assert audit.available_unique_count == 1
    assert audit.resolved_row_count == 3
    assert audit.resolved_unique_count == 1
    assert audit.missing_row_count == 1
    assert audit.duplicate_row_count == 1
    assert audit.entries[2].suggestion == "M01/images/one.jpg"
    assert [entry.relative_path for entry in valid_unique_entries(audit)] == ["M01/images/one.jpg"]


def test_manifest_accepts_windows_separators(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "M03" / "images").mkdir(parents=True)
    (root / "M03" / "images" / "sample.png").write_bytes(b"x")
    markdown = tmp_path / "labels.md"
    markdown.write_text("| M03\\images\\sample.png | 0 MPa |\n", encoding="utf-8")

    audit = parse_markdown_manifest(markdown, root)

    assert audit.entries[0].relative_path == "M03/images/sample.png"
    assert audit.entries[0].expected_style == "M03"
    assert audit.entries[0].exists


def test_manifest_repairs_one_extra_leading_zero_without_editing_source(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image_dir = root / "M03" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "M03_P04_D_0058.jpg").write_bytes(b"x")
    markdown = tmp_path / "labels.md"
    original = "| M03/images/M03_P04_D_00058.jpg | 1 MPa |\n"
    markdown.write_text(original, encoding="utf-8")

    audit = parse_markdown_manifest(markdown, root)
    entry = audit.entries[0]

    assert not entry.exists
    assert entry.suggestion == "M03/images/M03_P04_D_0058.jpg"
    assert entry.suggestion_exists
    assert entry.resolution == "repair_candidate"
    assert Path(entry.effective_absolute_path).name == "M03_P04_D_0058.jpg"
    assert markdown.read_text(encoding="utf-8") == original

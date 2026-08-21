from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


ROW_RE = re.compile(r"^\s*\|\s*(M\d{2}[\\/]images[\\/][^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")
STYLE_RE = re.compile(r"^M\d{2}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestEntry:
    line_number: int
    relative_path: str
    expected_style: str
    reading: str
    absolute_path: str
    exists: bool
    is_duplicate: bool
    first_line_number: int | None = None
    suggestion: str | None = None
    suggestion_exists: bool = False
    effective_absolute_path: str | None = None
    resolution: str = "missing"


@dataclass(frozen=True)
class ManifestAudit:
    markdown_path: str
    dataset_root: str
    row_count: int
    unique_path_count: int
    available_row_count: int
    available_unique_count: int
    resolved_row_count: int
    resolved_unique_count: int
    missing_row_count: int
    duplicate_row_count: int
    style_counts_by_row: dict[str, int]
    entries: list[ManifestEntry]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["entries"] = [asdict(entry) for entry in self.entries]
        return payload

    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _suggest_missing(dataset_root: Path, relative_path: str) -> str | None:
    rel = PurePosixPath(relative_path)
    image_dir = dataset_root.joinpath(*rel.parent.parts)
    if not image_dir.is_dir():
        return None
    # The supplied manifest has several final numeric fields with one extra
    # leading zero.  Prefer that precise, auditable repair over fuzzy matching.
    stem_match = re.match(r"^(.*_)(\d+)$", rel.stem)
    if stem_match:
        prefix, digits = stem_match.groups()
        if digits.startswith("0") and len(digits) > 1:
            repaired = f"{prefix}{digits[1:]}{rel.suffix}"
            if (image_dir / repaired).is_file():
                return str(rel.parent / repaired)
    choices = [p.name for p in image_dir.iterdir() if p.is_file()]
    close = difflib.get_close_matches(rel.name, choices, n=1, cutoff=0.72)
    return str(rel.parent / close[0]) if close else None


def parse_markdown_manifest(markdown_path: Path, dataset_root: Path) -> ManifestAudit:
    markdown_path = markdown_path.resolve()
    dataset_root = dataset_root.resolve()
    text = markdown_path.read_text(encoding="utf-8-sig")

    raw_rows: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = ROW_RE.match(line)
        if not match:
            continue
        relative_path = match.group(1).strip().replace("\\", "/")
        reading = match.group(2).strip()
        rel = PurePosixPath(relative_path)
        style = rel.parts[0] if rel.parts else ""
        if not STYLE_RE.fullmatch(style):
            continue
        if rel.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        raw_rows.append((line_number, relative_path, reading))

    first_lines: dict[str, int] = {}
    entries: list[ManifestEntry] = []
    for line_number, relative_path, reading in raw_rows:
        rel = PurePosixPath(relative_path)
        absolute_path = dataset_root.joinpath(*rel.parts)
        duplicate = relative_path in first_lines
        exists = absolute_path.is_file()
        suggestion = None if exists else _suggest_missing(dataset_root, relative_path)
        suggestion_path = (
            dataset_root.joinpath(*PurePosixPath(suggestion).parts) if suggestion else None
        )
        suggestion_exists = bool(suggestion_path and suggestion_path.is_file())
        effective_path = absolute_path if exists else suggestion_path if suggestion_exists else None
        entries.append(
            ManifestEntry(
                line_number=line_number,
                relative_path=relative_path,
                expected_style=rel.parts[0],
                reading=reading,
                absolute_path=str(absolute_path),
                exists=exists,
                is_duplicate=duplicate,
                first_line_number=first_lines.get(relative_path) if duplicate else None,
                suggestion=suggestion,
                suggestion_exists=suggestion_exists,
                effective_absolute_path=str(effective_path) if effective_path else None,
                resolution="exact" if exists else "repair_candidate" if suggestion_exists else "missing",
            )
        )
        first_lines.setdefault(relative_path, line_number)

    unique_available = {entry.relative_path for entry in entries if entry.exists}
    resolved_paths = {
        entry.effective_absolute_path for entry in entries if entry.effective_absolute_path is not None
    }
    return ManifestAudit(
        markdown_path=str(markdown_path),
        dataset_root=str(dataset_root),
        row_count=len(entries),
        unique_path_count=len(first_lines),
        available_row_count=sum(entry.exists for entry in entries),
        available_unique_count=len(unique_available),
        resolved_row_count=sum(entry.effective_absolute_path is not None for entry in entries),
        resolved_unique_count=len(resolved_paths),
        missing_row_count=sum(not entry.exists for entry in entries),
        duplicate_row_count=sum(entry.is_duplicate for entry in entries),
        style_counts_by_row=dict(Counter(entry.expected_style for entry in entries)),
        entries=entries,
    )


def valid_unique_entries(audit: ManifestAudit) -> list[ManifestEntry]:
    seen: set[str] = set()
    result: list[ManifestEntry] = []
    for entry in audit.entries:
        unique_key = entry.effective_absolute_path
        if unique_key is None or unique_key in seen:
            continue
        seen.add(unique_key)
        result.append(entry)
    return result

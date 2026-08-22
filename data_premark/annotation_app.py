"""Local, development-only visual annotation UI for gauge review rows.

The server binds to loopback by default, serves only manifest-listed images,
and atomically updates the selected review CSV. Source images are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import shutil
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .pipeline import SHAPE_STRATA


ALLOWED_REVIEW_STATUSES = {"pending", "accepted", "corrected"}
ALLOWED_POINTER_ROLES = {"", "measurement_pointer", "main_pointer", "measurement"}
SCOPE_STATUSES = (
    "",
    "in_scope",
    "deferred_dual_pointer",
    "deferred_dual_scale",
    "deferred_nested_dial",
    "deferred_automotive",
    "deferred_dial_indicator",
    "deferred_linear",
    "unreadable",
    "other",
)
TRAINING_TRACKS = ("", "company_priority", "generalization_guardrail")
EDITABLE_FIELDS = (
    "review_status",
    "review_shape",
    "pivot_x",
    "pivot_y",
    "pointer_tip_x",
    "pointer_tip_y",
    "pointer_candidate_id",
    "pointer_role",
    "pointer_angle_deg",
    "reading",
    "unit",
    "range_min",
    "range_max",
    "minor_division",
    "scope_status",
    "meter_family",
    "physical_meter_id",
    "condition",
    "training_track",
    "source_group",
    "brand",
    "model",
    "comment",
)
LEGACY_REQUIRED_EDITABLE_FIELDS = tuple(
    field
    for field in EDITABLE_FIELDS
    if field
    not in {
        "pointer_tip_x",
        "pointer_tip_y",
        "scope_status",
        "meter_family",
        "physical_meter_id",
        "condition",
        "training_track",
        "source_group",
        "brand",
        "model",
    }
)
COMPLETED_STATUSES = {"accepted", "corrected"}


class AnnotationValidationError(ValueError):
    """Raised when a browser update violates the annotation contract."""


def _finite_number(value: Any, field: str, *, allow_empty: bool = True) -> float | None:
    if value is None or str(value).strip() == "":
        if allow_empty:
            return None
        raise AnnotationValidationError(f"{field} is required")
    if isinstance(value, bool):
        raise AnnotationValidationError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise AnnotationValidationError(f"{field} must be finite")
    return number


def _format_number(value: float | None) -> str:
    return "" if value is None else format(value, ".10g")


class AnnotationStore:
    """Manifest-backed review state with allowlisted images and atomic saves."""

    def __init__(
        self,
        *,
        review_csv: Path,
        manifest_path: Path,
        source_root: Path | None = None,
        expected_partition: str = "dev",
    ) -> None:
        self.review_csv = Path(review_csv).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.expected_partition = expected_partition
        self._lock = threading.RLock()

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
            raise AnnotationValidationError("manifest must contain a records array")
        manifest_partition = str(manifest.get("partition", ""))
        if expected_partition == "dev" and manifest_partition != "dev":
            raise AnnotationValidationError("development mode refuses a non-dev manifest")
        configured_root = source_root or Path(str(manifest.get("source_root", "")))
        self.source_root = Path(configured_root).resolve()
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source root not found: {self.source_root}")

        selected = [record for record in manifest["records"] if record.get("sampling", {}).get("selected")]
        self.records: dict[str, dict[str, Any]] = {}
        for record in selected:
            record_id = str(record.get("record_id", "")).strip()
            split = str(record.get("sampling", {}).get("split", ""))
            if not record_id or record_id in self.records:
                raise AnnotationValidationError("manifest record IDs must be unique and non-empty")
            if split != expected_partition:
                raise AnnotationValidationError(
                    f"manifest record {record_id} belongs to {split!r}, expected {expected_partition!r}"
                )
            self.records[record_id] = record

        with self.review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        missing_fields = set(LEGACY_REQUIRED_EDITABLE_FIELDS) - set(self.fieldnames)
        if "record_id" not in self.fieldnames or missing_fields:
            raise AnnotationValidationError(
                f"review CSV missing fields: {sorted(missing_fields | ({'record_id'} - set(self.fieldnames)))}"
            )
        # New V1 keypoint metadata is optional for legacy rows.  It is exposed
        # immediately and appended atomically the next time the user saves.
        self.fieldnames.extend(field for field in EDITABLE_FIELDS if field not in self.fieldnames)
        self.rows: dict[str, dict[str, str]] = {}
        self.order: list[str] = []
        for row in rows:
            record_id = str(row.get("record_id", "")).strip()
            if not record_id or record_id in self.rows:
                raise AnnotationValidationError("review CSV record IDs must be unique and non-empty")
            self.rows[record_id] = {field: str(row.get(field, "")) for field in self.fieldnames}
            self.order.append(record_id)
        if set(self.rows) != set(self.records):
            raise AnnotationValidationError("review CSV IDs must exactly match manifest IDs")
        self._csv_digest = hashlib.sha256(self.review_csv.read_bytes()).hexdigest()

    def _image_path(self, record_id: str) -> Path:
        try:
            relative = Path(str(self.records[record_id]["image"]["path"]))
        except KeyError as exc:
            raise KeyError(record_id) from exc
        resolved = (self.source_root / relative).resolve()
        try:
            resolved.relative_to(self.source_root)
        except ValueError as exc:
            raise AnnotationValidationError("manifest image path escapes source root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest image not found: {resolved}")
        return resolved

    def image(self, record_id: str) -> tuple[bytes, str]:
        path = self._image_path(record_id)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), media_type

    def state(self) -> dict[str, Any]:
        with self._lock:
            items = []
            completed = 0
            for record_id in self.order:
                record = self.records[record_id]
                row = self.rows[record_id]
                status = row.get("review_status", "pending").strip().lower() or "pending"
                if status in COMPLETED_STATUSES:
                    completed += 1
                auto = record.get("auto_annotation", {})
                items.append(
                    {
                        "record_id": record_id,
                        "image": record.get("image", {}),
                        "sampling": record.get("sampling", {}),
                        "auto_annotation": {
                            "dial_boundary": auto.get("dial_boundary", {}),
                            "shape": auto.get("shape", {}),
                            "pivot": auto.get("pivot", {}),
                            "pointer_candidates": auto.get("pointer_candidates", []),
                            "selected_pointer_candidate_id": auto.get("selected_pointer_candidate_id"),
                        },
                        "review": {field: row.get(field, "") for field in EDITABLE_FIELDS},
                    }
                )
            return {
                "schema_version": "1.0",
                "partition": self.expected_partition,
                "review_csv": str(self.review_csv),
                "total": len(items),
                "completed": completed,
                "pending": len(items) - completed,
                "shape_labels": list(SHAPE_STRATA),
                "scope_statuses": list(SCOPE_STATUSES),
                "training_tracks": list(TRAINING_TRACKS),
                "items": items,
            }

    def _validated_update(self, payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise AnnotationValidationError("request body must be a JSON object")
        extra = set(payload) - set(EDITABLE_FIELDS)
        if extra:
            raise AnnotationValidationError(f"fields are not editable: {sorted(extra)}")
        cleaned = {field: str(payload.get(field, "")).strip() for field in EDITABLE_FIELDS}
        if any(len(value) > (2000 if field == "comment" else 128) for field, value in cleaned.items()):
            raise AnnotationValidationError("annotation field is too long")

        status = cleaned["review_status"].lower() or "pending"
        if status not in ALLOWED_REVIEW_STATUSES:
            raise AnnotationValidationError("review_status must be pending, accepted, or corrected")
        cleaned["review_status"] = status
        shape = cleaned["review_shape"]
        if shape and shape not in SHAPE_STRATA:
            raise AnnotationValidationError("review_shape is not a supported shape label")
        if cleaned["pointer_role"].lower() not in ALLOWED_POINTER_ROLES:
            raise AnnotationValidationError("pointer_role must identify the main measurement pointer")
        if cleaned["scope_status"] not in SCOPE_STATUSES:
            raise AnnotationValidationError("scope_status is not supported")
        if cleaned["training_track"] not in TRAINING_TRACKS:
            raise AnnotationValidationError("training_track is not supported")

        numeric: dict[str, float | None] = {}
        for field in (
            "pivot_x",
            "pivot_y",
            "pointer_tip_x",
            "pointer_tip_y",
            "pointer_angle_deg",
            "reading",
            "range_min",
            "range_max",
            "minor_division",
        ):
            numeric[field] = _finite_number(cleaned[field], field)
        for field in ("pivot_x", "pivot_y", "pointer_tip_x", "pointer_tip_y"):
            if numeric[field] is not None and not 0.0 <= float(numeric[field]) <= 1.0:
                raise AnnotationValidationError(f"{field} must be between 0 and 1")
        if (numeric["pointer_tip_x"] is None) != (numeric["pointer_tip_y"] is None):
            raise AnnotationValidationError("pointer tip coordinates must be provided together")
        angle = numeric["pointer_angle_deg"]
        if angle is not None and not 0.0 <= angle < 360.0:
            raise AnnotationValidationError("pointer_angle_deg must be in [0, 360)")
        minimum, maximum = numeric["range_min"], numeric["range_max"]
        if minimum is not None and maximum is not None and maximum <= minimum:
            raise AnnotationValidationError("range_max must be greater than range_min")
        minor = numeric["minor_division"]
        if minor is not None and minor <= 0:
            raise AnnotationValidationError("minor_division must be greater than zero")

        if status in COMPLETED_STATUSES:
            required_text = ("review_shape", "pointer_role", "unit")
            required_numeric = ("pivot_x", "pivot_y", "pointer_angle_deg", "reading", "range_min", "range_max")
            missing = [field for field in required_text if not cleaned[field]]
            missing.extend(field for field in required_numeric if numeric[field] is None)
            if missing:
                raise AnnotationValidationError(f"completed annotation is missing: {sorted(missing)}")

        for field, value in numeric.items():
            cleaned[field] = _format_number(value)
        return cleaned

    def save(self, record_id: str, payload: Any) -> dict[str, Any]:
        if record_id not in self.rows:
            raise KeyError(record_id)
        update = self._validated_update(payload)
        with self._lock:
            current_digest = hashlib.sha256(self.review_csv.read_bytes()).hexdigest()
            if current_digest != self._csv_digest:
                raise AnnotationValidationError(
                    "review CSV changed outside this annotation session; reload the app before saving"
                )
            original_row = dict(self.rows[record_id])
            self.rows[record_id].update(update)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8-sig",
                    newline="",
                    dir=self.review_csv.parent,
                    prefix=f".{self.review_csv.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=self.fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for ordered_id in self.order:
                        writer.writerow(self.rows[ordered_id])
                    stream.flush()
                    os.fsync(stream.fileno())
                backup = self.review_csv.with_suffix(self.review_csv.suffix + ".bak")
                shutil.copy2(self.review_csv, backup)
                os.replace(temporary_path, self.review_csv)
                temporary_path = None
                self._csv_digest = hashlib.sha256(self.review_csv.read_bytes()).hexdigest()
            except Exception:
                self.rows[record_id] = original_row
                raise
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
        state = self.state()
        return {
            "record_id": record_id,
            "review": {field: self.rows[record_id].get(field, "") for field in EDITABLE_FIELDS},
            "completed": state["completed"],
            "total": state["total"],
        }


class AnnotationServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: AnnotationStore, static_root: Path):
        super().__init__(address, AnnotationRequestHandler)
        self.store = store
        self.static_root = static_root


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[annotation-ui] {self.address_string()} {format % args}")

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'",
        )

    def _send_bytes(self, payload: bytes, media_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        rendered = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(rendered, "application/json; charset=utf-8", status)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json({"status": "ok", "partition": self.server.store.expected_partition})
            return
        if path == "/api/state":
            self._send_json(self.server.store.state())
            return
        if path.startswith("/api/image/"):
            record_id = unquote(path.removeprefix("/api/image/"))
            if not record_id or "/" in record_id or "\\" in record_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid record ID")
                return
            try:
                payload, media_type = self.server.store.image(record_id)
            except KeyError:
                self._send_error(HTTPStatus.NOT_FOUND, "record not found")
                return
            except (FileNotFoundError, AnnotationValidationError) as exc:
                self._send_error(HTTPStatus.CONFLICT, str(exc))
                return
            self._send_bytes(payload, media_type)
            return
        static_files = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/styles.css": "styles.css",
        }
        filename = static_files.get(path)
        if filename is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        static_path = self.server.static_root / filename
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self._send_bytes(static_path.read_bytes(), f"{media_type}; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not path.startswith("/api/records/"):
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        record_id = unquote(path.removeprefix("/api/records/"))
        if not record_id or "/" in record_id or "\\" in record_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid record ID")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid content length")
            return
        if content_length <= 0 or content_length > 65_536:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body must be 1-65536 bytes")
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = self.server.store.save(record_id, payload)
        except UnicodeDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "request must be UTF-8")
            return
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "request must contain valid JSON")
            return
        except AnnotationValidationError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except KeyError:
            self._send_error(HTTPStatus.NOT_FOUND, "record not found")
            return
        self._send_json(result)


def create_server(
    *,
    store: AnnotationStore,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> AnnotationServer:
    static_root = Path(__file__).with_name("annotation_ui")
    return AnnotationServer((host, port), store, static_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local gauge development annotation UI.")
    parser.add_argument("--review-csv", type=Path, default=Path("outputs/data_premark_v1/review.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/data_premark_v1/review_manifest.json"))
    parser.add_argument("--source-root", type=Path, default=Path("all_set"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("annotation UI may only bind to a loopback address")
    store = AnnotationStore(
        review_csv=args.review_csv,
        manifest_path=args.manifest,
        source_root=args.source_root,
        expected_partition="dev",
    )
    server = create_server(store=store, host=args.host, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Gauge annotation UI: {url}")
    print(f"Development rows: {len(store.order)}; saving to {store.review_csv}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping annotation UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

MODEL_HASHES = {
    "meter_detector.pt": "A0447F659564955C0FFBCD7BD68394745C9C3CA5686117EFBB41A631CE79E1A1",
    "scale_segment.pt": "ED02E20A11E4B4D86A40220A75E0146726E2E67F132F7FEB768EE0948C234007",
    "pointer_keypoints.pt": "80C079BCFF67B920B2F7A711245070F246651B4CBCDFC86DCB8B416DD7C03540",
}

IMPORTS = (
    "cv2",
    "numpy",
    "onnxruntime",
    "rapidocr",
    "torch",
    "torchvision",
    "ultralytics",
    "yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    problems: list[str] = []
    print(f"Python: {sys.version.split()[0]}")

    for module_name in IMPORTS:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] import {module_name}: {version}")
        except Exception as exc:  # pragma: no cover - environment-specific reporting
            problems.append(f"cannot import {module_name}: {exc}")
            print(f"[FAIL] import {module_name}: {exc}")

    for name, expected in MODEL_HASHES.items():
        path = ROOT / "models" / name
        if not path.is_file():
            problems.append(f"missing model: {path}")
            print(f"[FAIL] model {name}: missing (did you run git lfs pull?)")
            continue
        actual = sha256(path)
        if actual != expected:
            problems.append(f"hash mismatch for {name}: {actual}")
            print(f"[FAIL] model {name}: SHA256 mismatch")
        else:
            print(f"[OK] model {name}: {path.stat().st_size / (1024 * 1024):.2f} MiB")

    if problems:
        print("\nSetup is not ready:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("\nSetup is ready. Run pytest or the camera/image demo next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

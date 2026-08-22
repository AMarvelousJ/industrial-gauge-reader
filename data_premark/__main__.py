from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a duplicate-safe, balanced gauge pre-annotation review package."
    )
    parser.add_argument("--source", type=Path, default=Path("all_set"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/data_premark_v1")
    )
    parser.add_argument("--per-shape", type=int, default=30)
    parser.add_argument("--validation-per-shape", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    summary = run_pipeline(
        source_root=args.source,
        output_dir=args.output,
        per_shape=args.per_shape,
        validation_per_shape=args.validation_per_shape,
        seed=args.seed,
    )
    print(
        "generated "
        f"{summary['image_count']} image records; "
        f"{summary['selected_count']} review samples at {args.output}"
    )


if __name__ == "__main__":
    main()

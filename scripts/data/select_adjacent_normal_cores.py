#!/usr/bin/env python3
"""Create the protected, balanced ten-core adjacent-normal routing manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.adjacent_normal_selection import (  # noqa: E402
    create_adjacent_normal_selection,
)
from spatial_benchmark.paths import current_paths  # noqa: E402


def parser() -> argparse.ArgumentParser:
    paths = current_paths()
    clinical = paths.data_root / "clinical"
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--legacy-workbook",
        type=Path,
        default=clinical / "Gastric Study_Old.xlsx",
    )
    command.add_argument(
        "--pathology-review-workbook",
        type=Path,
        default=clinical / "Gastric Study.xlsx",
    )
    command.add_argument(
        "--core-map",
        type=Path,
        default=clinical / "fov_core_map.csv",
    )
    command.add_argument(
        "--raw-dir",
        type=Path,
        default=paths.data_root / "raw",
    )
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--chunksize", type=int, default=100_000)
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing generated manifest.",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    receipt = create_adjacent_normal_selection(
        legacy_workbook=arguments.legacy_workbook,
        pathology_review_workbook=arguments.pathology_review_workbook,
        core_map_csv=arguments.core_map,
        raw_dir=arguments.raw_dir,
        output_path=arguments.output,
        chunksize=arguments.chunksize,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            receipt.to_public_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

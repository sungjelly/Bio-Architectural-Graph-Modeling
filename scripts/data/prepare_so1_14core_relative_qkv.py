#!/usr/bin/env python3
"""Prepare the immutable fit-only SO_1 core-1--14 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.data import DEFAULT_PIXEL_SIZE_UM  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.so1_pooled_full_core import (  # noqa: E402
    prepare_so1_14core_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    parser = argparse.ArgumentParser(
        description=(
            "Resolve neutral SO_1 cores 1 through 14 from the authoritative "
            "slide-qualified map, require complete raw-FOV coverage, and fit "
            "shared 14-core preprocessing."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=paths.data_root / "raw")
    parser.add_argument(
        "--core-map",
        type=Path,
        default=paths.data_root / "clinical/fov_core_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.data_root / "processed/so1_14core_relative_qkv_v1",
    )
    parser.add_argument("--chunksize", type=int, default=8192)
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_so1_14core_cohort(
        raw_dir=args.raw_dir,
        core_map_csv=args.core_map,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        pixel_size_um=args.pixel_size_um,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "aliases": manifest["cohort"]["aliases"],
                "total_cells": manifest["cohort"]["total_cells"],
                "all_raw_fovs_mapped": manifest["routing_audit"][
                    "all_raw_fovs_mapped"
                ],
                "unmapped_fov_count": manifest["routing_audit"][
                    "unmapped_fov_count"
                ],
                "manifest_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

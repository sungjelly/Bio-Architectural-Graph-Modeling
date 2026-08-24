#!/usr/bin/env python3
"""Prepare the immutable fit-only six-Cancer-core cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.cancer_pooled_full_core import (  # noqa: E402
    prepare_cancer_6core_cohort,
)
from spatial_benchmark.data import DEFAULT_PIXEL_SIZE_UM  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the user-attested six Cancer cores through the authoritative "
            "slide-qualified FOV map and fit shared preprocessing."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=_PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--core-map",
        type=Path,
        default=_PROJECT_ROOT / "data/clinical/fov_core_map.csv",
    )
    parser.add_argument(
        "--reconciliation",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "experiments/campaigns"
            / "cmp_20260824_cancer_6core_relative_qkv_multiseed"
            / "clinical_reconciliation_policy.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/cancer_6core_relative_qkv_v1",
    )
    parser.add_argument("--chunksize", type=int, default=8192)
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_cancer_6core_cohort(
        raw_dir=args.raw_dir,
        core_map_csv=args.core_map,
        reconciliation_yaml=args.reconciliation,
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

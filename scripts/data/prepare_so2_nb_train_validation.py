#!/usr/bin/env python3
"""Prepare the reference-only SO2 NB train/validation data overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.so2_nb_data import (  # noqa: E402
    prepare_so2_nb_train_validation_overlay,
)


DEFAULT_CONTRACT = (
    _PROJECT_ROOT
    / "experiments"
    / "campaigns"
    / "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0"
    / "frozen_task_contract.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--protected-grouping-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_so2_nb_train_validation_overlay(
        cohort_dir=args.cohort_dir,
        graph_dir=args.graph_dir,
        protected_grouping_source=args.protected_grouping_source,
        frozen_contract_path=args.frozen_contract,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "artifact_kind": manifest["artifact_kind"],
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "output_dir": str(args.output_dir.resolve()),
                "preprocessing_fingerprint": manifest["preprocessing"][
                    "preprocessing_fingerprint"
                ],
                "split_fingerprint": manifest["split"]["split_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

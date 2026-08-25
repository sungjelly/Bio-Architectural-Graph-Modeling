#!/usr/bin/env python3
"""Run the registered six-core post-training attention-routing analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from spatial_benchmark.attention_niche_pipeline import (
    run_attention_niche_pipeline,
)
from spatial_benchmark.paths import current_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract final-layer attention from completed relative-QKV models "
            "and construct six model-defined spatial attention-routing niches."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        default=os.environ.get("BAGM_RUN_ID"),
    )
    parser.add_argument(
        "--run-scratch",
        type=Path,
        default=(
            Path(os.environ["BAGM_RUN_SCRATCH"])
            if os.environ.get("BAGM_RUN_SCRATCH")
            else None
        ),
    )
    parser.add_argument("--database", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if not arguments.run_id:
        raise SystemExit("--run-id or BAGM_RUN_ID is required")
    if arguments.run_scratch is None:
        raise SystemExit("--run-scratch or BAGM_RUN_SCRATCH is required")
    paths = current_paths()
    database = arguments.database or paths.state_root / "tracking/bagm.sqlite3"
    summary = run_attention_niche_pipeline(
        config_path=arguments.config,
        run_id=str(arguments.run_id),
        run_scratch=arguments.run_scratch,
        database=database,
        paths=paths,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

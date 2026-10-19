#!/usr/bin/env python3
"""Queue-owned entry point for joint-slide Relative-QKV paper-core split training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spatial_benchmark.configuration import load_yaml_mapping, validate_experiment_config
from spatial_benchmark.paths import current_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-scratch", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml_mapping(args.config)
    validate_experiment_config(config)
    if config["evaluation"].get("protocol") != "so12_pearson_relative_qkv_paper_core_split_v1":
        raise ValueError("This entry point requires the SO12 paper-core split protocol.")
    if not args.preflight:
        run_id = os.environ.get("BAGM_RUN_ID")
        worker_scratch = os.environ.get("BAGM_RUN_SCRATCH")
        if (
            not run_id
            or not worker_scratch
            or args.run_scratch.resolve() != Path(worker_scratch).resolve()
            or args.run_scratch.name != run_id
        ):
            raise ValueError("Production requires its queue-owned run ID and scratch path.")
    from spatial_benchmark.so12_paper_split_training import run_training

    run_training(config, args.run_scratch, current_paths(), preflight=args.preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

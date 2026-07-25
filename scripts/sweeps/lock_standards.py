#!/usr/bin/env python3
"""Lock validation-selected standards and emit execution matrices only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.standards_lock import (  # noqa: E402
    create_standards_lock,
    load_standards_lock,
)


def _seeds(value: str) -> list[int]:
    try:
        result = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume sealed validation-only recommendations and atomically "
            "write a standards lock plus concrete confirmation/final matrices. "
            "This utility never launches jobs or opens test outcomes."
        )
    )
    parser.add_argument(
        "--recommendation",
        action="append",
        required=True,
        type=Path,
        help=(
            "Validation-only recommendation JSON; repeat for graph, mask, and "
            "representation stages."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New immutable standards-lock directory.",
    )
    parser.add_argument(
        "--final-seeds",
        type=_seeds,
        default=[0, 1, 2, 3, 4],
        help="Exactly five final seeds (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=_seeds,
        default=[0, 1, 2],
        help="At least three validation confirmation seeds.",
    )
    parser.add_argument(
        "--minimum-confirmation-seeds",
        type=int,
        default=3,
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--rewire-seed", type=int, default=271828)
    parser.add_argument(
        "--enable-g3",
        action="store_true",
        help=(
            "Emit G3 only with a positive validation eligibility "
            "recommendation and matched checkpoint template."
        ),
    )
    parser.add_argument(
        "--g3-checkpoint-template",
        help=(
            "Concrete matched B0 path template containing {seed}; required "
            "only with --enable-g3."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        output = create_standards_lock(
            arguments.recommendation,
            arguments.output,
            final_seeds=arguments.final_seeds,
            confirmation_seeds=arguments.confirmation_seeds,
            minimum_confirmation_seeds=(
                arguments.minimum_confirmation_seeds
            ),
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            rewire_seed=arguments.rewire_seed,
            enable_g3=arguments.enable_g3,
            g3_checkpoint_template=arguments.g3_checkpoint_template,
        )
        manifest, lock, matrices = load_standards_lock(output)
    except (ValueError, FileNotFoundError) as exc:
        print(f"standards lock failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "lock_id": lock["lock_id"],
                "selection_scope": lock["selection_scope"],
                "test_metrics_used_for_selection": False,
                "final_seed_count": len(
                    lock["final_execution"]["seeds"]
                ),
                "matrix_files": sorted(matrices),
                "execution_started": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

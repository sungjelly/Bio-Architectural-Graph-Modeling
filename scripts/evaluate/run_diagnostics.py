#!/usr/bin/env python3
"""Run immutable non-learned controls on fixed benchmark masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.diagnostic_artifacts import (  # noqa: E402
    load_diagnostic_artifact,
    run_diagnostic_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train-only global-mean and split-local nearest-visible "
            "controls on every fixed validation mask. The sealed test split "
            "is evaluated only with --open-test."
        )
    )
    parser.add_argument(
        "--prepared",
        required=True,
        type=Path,
        help="Immutable prepared artifact directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New immutable diagnostic artifact directory.",
    )
    parser.add_argument(
        "--min-distance-um",
        type=float,
        default=0.0,
        help="Minimum nearest-copy source distance in micrometres.",
    )
    parser.add_argument(
        "--distance-block-size",
        type=int,
        default=None,
        help="Optional number of target rows per distance-computation block.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save split-local masks, predictions, and nearest-source arrays.",
    )
    parser.add_argument(
        "--open-test",
        action="store_true",
        help="Explicitly evaluate every fixed test mask.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve())]
    if argv is None:
        command.extend(sys.argv[1:])
    else:
        command.extend(argv)
    output = run_diagnostic_artifact(
        arguments.prepared,
        arguments.output,
        min_distance_um=arguments.min_distance_um,
        open_test=arguments.open_test,
        save_predictions=arguments.save_predictions,
        distance_block_size=arguments.distance_block_size,
        command=command,
    )
    manifest, metrics, _ = load_diagnostic_artifact(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "diagnostic_id": manifest["diagnostic_id"],
                "validation_records": len(metrics["validation"]),
                "test_records": len(metrics["test"]),
                "test_targets_evaluated": metrics[
                    "test_targets_evaluated"
                ],
                "predictions_saved": (
                    manifest["artifacts"]["predictions"] is not None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

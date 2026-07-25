#!/usr/bin/env python3
"""Create a sealed validation-only graph-by-mask interaction diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.interaction_summary import (  # noqa: E402
    create_interaction_summary,
    load_interaction_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        required=True,
        type=Path,
        help="Directory containing the complete sealed 2 x 2 run artifacts.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New immutable diagnostic artifact directory.",
    )
    args = parser.parse_args(argv)
    output = create_interaction_summary(
        args.runs,
        args.output,
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    manifest, diagnostic = load_interaction_summary(output)
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "output": str(output),
                "diagnostic_only": diagnostic["diagnostic_only"],
                "selection_performed": diagnostic["selection_performed"],
                "test_metrics_used": diagnostic["test_metrics_used"],
                "difference_in_differences": diagnostic[
                    "difference_in_differences"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

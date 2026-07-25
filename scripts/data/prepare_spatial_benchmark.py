#!/usr/bin/env python3
"""Prepare one immutable normal-core benchmark artifact from repository root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.artifacts import prepare_artifact  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select, split, preprocess, graph, and mask the approved true-Normal "
            "tissue unit without evaluating the sealed test split."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Versioned YAML preparation configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New immutable artifact directory; an existing path is refused.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(sys.argv[1:] if argv is None else argv)
    output = prepare_artifact(
        arguments.config,
        arguments.output,
        command=command,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "output": str(output),
                "n_cells": manifest["selection"]["n_cells"],
                "n_biological_probes": manifest["features"][
                    "n_biological_probes"
                ],
                "split_id": manifest["split"]["split_id"],
                "test_targets_evaluated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

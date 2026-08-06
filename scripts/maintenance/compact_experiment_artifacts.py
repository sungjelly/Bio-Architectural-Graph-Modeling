#!/usr/bin/env python3
"""Plan or apply an explicit registry-aware experiment-payload cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_benchmark.artifact_retention import (  # noqa: E402
    DEFAULT_CHECKPOINT_MIN_BYTES,
    DEFAULT_PREDICTION_MIN_BYTES,
    compact_registered_artifacts,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a checksum-bound deletion plan for large modern "
            "diagnostic/exploratory payloads; add --apply to execute it."
        )
    )
    parser.add_argument(
        "--database",
        default="state/tracking/bagm.sqlite3",
        help="authoritative local registry path",
    )
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-min-mib",
        type=int,
        default=DEFAULT_CHECKPOINT_MIN_BYTES // (1024 * 1024),
    )
    parser.add_argument(
        "--prediction-min-mib",
        type=int,
        default=DEFAULT_PREDICTION_MIN_BYTES // (1024 * 1024),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="verify hashes, back up SQLite, delete exact planned files, and tombstone rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = current_paths(anchor=PROJECT_ROOT)
    database = Path(arguments.database)
    if not database.is_absolute():
        database = paths.project_root / database
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    result = compact_registered_artifacts(
        Registry(database),
        decision_id=arguments.decision_id,
        output_dir=output_dir,
        paths=paths,
        checkpoint_min_bytes=arguments.checkpoint_min_mib * 1024 * 1024,
        prediction_min_bytes=arguments.prediction_min_mib * 1024 * 1024,
        apply=bool(arguments.apply),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

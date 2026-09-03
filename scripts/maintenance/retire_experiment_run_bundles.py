#!/usr/bin/env python3
"""Plan or apply exact, checksum-bound retirement of complete run bundles."""

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
    retire_registered_run_bundles,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retire every explicitly listed run bundle in one campaign while "
            "preserving registry identities and external checksum receipts."
        )
    )
    parser.add_argument(
        "--database",
        default="state/tracking/bagm.sqlite3",
        help="authoritative local registry path",
    )
    parser.add_argument("--expected-campaign-id", required=True)
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help=(
            "exact run ID to retire; repeat for every run in the campaign "
            "because partial-campaign retirement is rejected"
        ),
    )
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prior-retention-decision-id",
        action="append",
        default=[],
        help="earlier applied decision whose tombstones must retain lineage",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "verify all hashes, snapshot SQLite, tombstone rows, delete only "
            "the planned bundle files, and register external receipts"
        ),
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
    result = retire_registered_run_bundles(
        Registry(database),
        run_ids=arguments.run_id,
        expected_campaign_id=arguments.expected_campaign_id,
        decision_id=arguments.decision_id,
        output_dir=output_dir,
        prior_retention_decision_ids=arguments.prior_retention_decision_id,
        paths=paths,
        apply=bool(arguments.apply),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

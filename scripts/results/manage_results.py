#!/usr/bin/env python3
"""Create, validate, and catalog curated BAGM conclusion records."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.result_catalog import (  # noqa: E402
    EXPERIMENT_TYPES,
    ResultCatalogError,
    create_result_scaffold,
    validate_result_tree,
    write_or_check_catalog,
)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode == 0 and len(commit) == 40:
        return commit
    return "TODO"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Override BAGM_RESULT_ROOT for this invocation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a draft result scaffold.")
    create.add_argument("--result-id", required=True)
    create.add_argument("--title", required=True)
    create.add_argument(
        "--experiment-type", required=True, choices=EXPERIMENT_TYPES
    )
    create.add_argument("--method-family", required=True)
    create.add_argument("--lifecycle-stage", required=True)
    create.add_argument("--study-axis", required=True)
    create.add_argument("--campaign-id", action="append", default=[])

    validate = subparsers.add_parser("validate", help="Validate every result record.")
    validate.add_argument("--verify-payloads", action="store_true")
    validate.add_argument("--verify-sources", action="store_true")

    catalog = subparsers.add_parser(
        "catalog", help="Regenerate or check deterministic result catalogs."
    )
    catalog.add_argument("--check", action="store_true")
    catalog.add_argument("--verify-payloads", action="store_true")
    catalog.add_argument("--verify-sources", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    paths = current_paths()
    if arguments.result_root is not None:
        result_root = arguments.result_root.expanduser()
        if not result_root.is_absolute():
            result_root = paths.project_root / result_root
        paths = replace(paths, result_root=result_root.resolve(strict=False))
    try:
        if arguments.command == "create":
            directory = create_result_scaffold(
                result_id=arguments.result_id,
                title=arguments.title,
                experiment_type=arguments.experiment_type,
                method_family=arguments.method_family,
                lifecycle_stage=arguments.lifecycle_stage,
                study_axis=arguments.study_axis,
                campaign_ids=arguments.campaign_id,
                paths=paths,
                git_commit=_git_commit(),
            )
            print(directory)
            return 0
        if arguments.command == "validate":
            records = validate_result_tree(
                paths=paths,
                verify_payloads=arguments.verify_payloads,
                verify_sources=arguments.verify_sources,
            )
            print(json.dumps({"valid": True, "result_count": len(records)}))
            return 0
        if arguments.command == "catalog":
            catalog = write_or_check_catalog(
                paths=paths,
                check=arguments.check,
                verify_payloads=arguments.verify_payloads,
                verify_sources=arguments.verify_sources,
            )
            print(
                json.dumps(
                    {
                        "valid": True,
                        "checked": arguments.check,
                        "result_count": catalog["result_count"],
                        "catalog_content_sha256": catalog[
                            "catalog_content_sha256"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
    except ResultCatalogError as error:
        print(f"result catalog error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())

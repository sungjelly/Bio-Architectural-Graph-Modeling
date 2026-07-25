#!/usr/bin/env python3
"""Calculate read-only BAGM dataset, directory, and split fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.fingerprints import (  # noqa: E402
    build_path_fingerprint,
    fingerprint_dataset,
    fingerprint_split,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    path = commands.add_parser("path", help="Fingerprint one file or directory.")
    path.add_argument("source", type=Path)
    path.add_argument("--max-file-bytes", type=int)

    dataset = commands.add_parser("dataset", help="Fingerprint one or more dataset roots.")
    dataset.add_argument("source", nargs="+", type=Path)
    dataset.add_argument("--dataset-id")
    dataset.add_argument("--source-version")
    dataset.add_argument("--max-file-bytes", type=int)

    split = commands.add_parser("split", help="Fingerprint logical split assignments.")
    split.add_argument("source", type=Path)
    split.add_argument("--split-id")
    split.add_argument("--method")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "path":
        result = build_path_fingerprint(
            arguments.source,
            max_file_bytes=arguments.max_file_bytes,
        ).summary()
    elif arguments.command == "dataset":
        result = {
            "algorithm": "sha256",
            "fingerprint": fingerprint_dataset(
                arguments.source,
                dataset_id=arguments.dataset_id,
                source_version=arguments.source_version,
                max_file_bytes=arguments.max_file_bytes,
            ),
            "input_count": len(arguments.source),
        }
    else:
        result = {
            "algorithm": "sha256",
            "fingerprint": fingerprint_split(
                arguments.source,
                split_id=arguments.split_id,
                method=arguments.method,
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

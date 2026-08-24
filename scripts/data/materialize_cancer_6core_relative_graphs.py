#!/usr/bin/env python3
"""Materialize immutable radial graphs and shared relative-geometry caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.cancer_relative_graphs import (  # noqa: E402
    DEFAULT_MAX_EDGES_PER_CHUNK,
    DEFAULT_RECEIVER_CHUNK_SIZE,
    prepare_cancer_6core_relative_graphs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact six-core radial-stratified graph collection and "
            "one checksum-bound relative-geometry mmap cache per core."
        )
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/processed/cancer_6core_relative_qkv_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "data/processed/cancer_6core_relative_qkv_graphs_v1"
        ),
    )
    parser.add_argument(
        "--receiver-chunk-size",
        type=int,
        default=DEFAULT_RECEIVER_CHUNK_SIZE,
    )
    parser.add_argument(
        "--max-edges-per-chunk",
        type=int,
        default=DEFAULT_MAX_EDGES_PER_CHUNK,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_cancer_6core_relative_graphs(
        cohort_dir=args.cohort_dir,
        output_dir=args.output_dir,
        receiver_chunk_size=args.receiver_chunk_size,
        max_edges_per_chunk=args.max_edges_per_chunk,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "aliases": manifest["aliases"],
                "manifest_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

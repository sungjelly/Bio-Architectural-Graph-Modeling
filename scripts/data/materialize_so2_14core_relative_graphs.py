#!/usr/bin/env python3
"""Materialize SO_2 radial graphs and shared relative-geometry caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.so2_relative_graphs import (  # noqa: E402
    DEFAULT_MAX_EDGES_PER_CHUNK,
    DEFAULT_RECEIVER_CHUNK_SIZE,
    prepare_so2_14core_relative_graphs,
)


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    parser = argparse.ArgumentParser(
        description=(
            "Build 14 exact radial-stratified graph caches. Matching CAN-15, "
            "CAN-21, and CAN-23 caches are hard-linked only after checksum, "
            "coordinate-order, and parameter verification."
        )
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=paths.data_root / "processed/so2_14core_relative_qkv_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.data_root / "processed/so2_14core_relative_qkv_graphs_v1",
    )
    parser.add_argument(
        "--reuse-cancer-cohort-dir",
        type=Path,
        default=paths.data_root / "processed/cancer_6core_relative_qkv_v1",
    )
    parser.add_argument(
        "--reuse-cancer-graph-dir",
        type=Path,
        default=paths.data_root / "processed/cancer_6core_relative_qkv_graphs_v1",
    )
    parser.add_argument(
        "--no-reuse-cancer-graphs",
        action="store_true",
        help="Regenerate all graphs even when checksum-compatible caches exist.",
    )
    parser.add_argument(
        "--receiver-chunk-size", type=int, default=DEFAULT_RECEIVER_CHUNK_SIZE
    )
    parser.add_argument(
        "--max-edges-per-chunk", type=int, default=DEFAULT_MAX_EDGES_PER_CHUNK
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reuse_cohort = None if args.no_reuse_cancer_graphs else args.reuse_cancer_cohort_dir
    reuse_graph = None if args.no_reuse_cancer_graphs else args.reuse_cancer_graph_dir
    manifest = prepare_so2_14core_relative_graphs(
        cohort_dir=args.cohort_dir,
        output_dir=args.output_dir,
        receiver_chunk_size=args.receiver_chunk_size,
        max_edges_per_chunk=args.max_edges_per_chunk,
        reuse_cancer_cohort_dir=reuse_cohort,
        reuse_cancer_graph_dir=reuse_graph,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "aliases": manifest["aliases"],
                "reuse_audit": manifest["reuse_audit"],
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

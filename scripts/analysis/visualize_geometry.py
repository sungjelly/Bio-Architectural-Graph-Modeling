#!/usr/bin/env python3
"""Render a geometry-only audit figure for one prepared graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.geometry_viz import (  # noqa: E402
    render_geometry_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality split, macroblock, degree, local-edge, "
            "and graph-QC panels without reading expression outcomes or "
            "emitting biological/sample identifiers."
        )
    )
    parser.add_argument(
        "--prepared-data",
        required=True,
        type=Path,
        help="Path to an immutable prepared_data.npz artifact.",
    )
    parser.add_argument(
        "--graph",
        required=True,
        type=Path,
        help="Path to one geometry-only graph-grid NPZ.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New atomic output directory for PNG, PDF, and manifest.",
    )
    parser.add_argument(
        "--window-size-um",
        type=float,
        default=200.0,
        help="Width of the reproducibly selected dense window (default: 200).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; must be at least 150 (default: 300).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        output = render_geometry_report(
            arguments.prepared_data,
            arguments.graph,
            arguments.output,
            window_size_um=arguments.window_size_um,
            dpi=arguments.dpi,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"geometry visualization failed: {exc}", file=sys.stderr)
        return 2
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "geometry_only": manifest["geometry_only"],
                "graph_id": manifest["provenance"]["graph_id"],
                "output": str(output),
                "files": sorted(manifest["files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

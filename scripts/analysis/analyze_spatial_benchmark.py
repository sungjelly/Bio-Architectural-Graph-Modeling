#!/usr/bin/env python3
"""Aggregate locked-seed prediction artifacts for the within-core benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.paths import REPORT_ROOT  # noqa: E402

from spatial_benchmark.analysis import analyze_run_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble isolated model seeds, aggregate fixed mask replicates "
            "within spatial blocks, apply the prespecified B0/G1/rewired gate, "
            "and create descriptive within-core tables and figures."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing isolated <run_id>/manifest.json runs.",
    )
    parser.add_argument(
        "--standards-lock",
        type=Path,
        required=True,
        help=(
            "Immutable standards-lock directory authorizing the exact final "
            "matrix represented by the run manifests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_ROOT / "analyses" / "spatial_benchmark_summary",
        help="Destination for summary JSON/CSV and PNG figures.",
    )
    parser.add_argument(
        "--locked-graph-id",
        help=(
            "Prespecified true-G1 graph ID. Required when more than one graph "
            "candidate is present; never inferred from the best observed loss."
        ),
    )
    parser.add_argument(
        "--graph-qc-dir",
        type=Path,
        help="Optional additional directory/file of graph-QC JSON artifacts.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Sealed evaluation split used for the acceptance gate (default: test).",
    )
    parser.add_argument(
        "--graph-candidate-split",
        default="validation",
        help=(
            "Split used to display graph candidates (default: validation; "
            "falls back to the analysis split when unavailable)."
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        help=(
            "Optional assertion on the locked seed count; identities always "
            "come from the verified standards lock."
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Spatial-block bootstrap confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Deterministic spatial-block bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Bootstrap RNG seed, independent of model/mask seeds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = analyze_run_directory(
            arguments.runs_dir,
            arguments.output_dir,
            standards_lock_dir=arguments.standards_lock,
            locked_graph_id=arguments.locked_graph_id,
            graph_qc_dir=arguments.graph_qc_dir,
            split=arguments.split,
            graph_candidate_split=arguments.graph_candidate_split,
            expected_seeds=arguments.expected_seeds,
            confidence_level=arguments.confidence_level,
            n_bootstrap=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "outcome": summary["acceptance_gate"]["outcome"],
                "supported": summary["acceptance_gate"]["supported"],
                "scope": summary["scope"],
                "summary": str(arguments.output_dir / "summary.json"),
            },
            sort_keys=True,
        )
    )
    # A valid negative result is scientific output, not a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

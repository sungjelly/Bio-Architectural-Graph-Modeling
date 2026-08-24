#!/usr/bin/env python3
"""Independently verify a final seed 0/1/2/3 relative-QKV checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.relative_qkv_checkpoint_verification import (  # noqa: E402
    DEFAULT_ATTENTION_ATOL,
    DEFAULT_ATTENTION_RTOL,
    DEFAULT_METRIC_ATOL,
    DEFAULT_METRIC_RTOL,
    DEFAULT_PREDICTION_ATOL,
    DEFAULT_PREDICTION_RTOL,
    MAX_ATTENTION_RECEIVERS_PER_CORE,
    verify_relative_qkv_checkpoint,
    write_verification_receipt,
)
from spatial_benchmark.relative_qkv_post_training import (  # noqa: E402
    load_prepared_relative_qkv_batches,
)


def _existing(path: Path) -> Path | None:
    return path.resolve() if path.is_file() else None


def _bundle_cross_check_paths(checkpoint: Path) -> dict[str, Path | None]:
    """Discover runner receipts beside ``checkpoints/last.ckpt`` when present."""

    if checkpoint.parent.name != "checkpoints":
        return {
            "expected_core_metrics_path": None,
            "expected_final_metrics_path": None,
            "training_provenance_path": None,
        }
    run_root = checkpoint.parent.parent
    return {
        "expected_core_metrics_path": _existing(
            run_root / "diagnostics" / "held_in_fit_metrics_by_core.json"
        ),
        "expected_final_metrics_path": _existing(run_root / "metrics" / "final.json"),
        "training_provenance_path": _existing(
            run_root / "provenance" / "relative_qkv_training.json"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    campaign = (
        paths.project_root
        / "experiments"
        / "campaigns"
        / "cmp_20260824_cancer_6core_relative_qkv_multiseed"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Verify a standalone seed 0/1/2/3 last.ckpt, plateau decision, all six "
            "fixed held-in fit replays, and bounded selected attention twice."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=paths.data_root / "processed" / "cancer_6core_relative_qkv_v1",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=(
            paths.data_root / "processed" / "cancer_6core_relative_qkv_graphs_v1"
        ),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=campaign,
        help=(
            "Campaign amendment file or directory. For a directory, exactly one "
            "amendment checksum must match the checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--attention-receivers-per-core",
        type=int,
        default=4,
        choices=range(1, MAX_ATTENTION_RECEIVERS_PER_CORE + 1),
        metavar=f"1..{MAX_ATTENTION_RECEIVERS_PER_CORE}",
    )
    parser.add_argument("--prediction-atol", type=float, default=DEFAULT_PREDICTION_ATOL)
    parser.add_argument("--prediction-rtol", type=float, default=DEFAULT_PREDICTION_RTOL)
    parser.add_argument("--attention-atol", type=float, default=DEFAULT_ATTENTION_ATOL)
    parser.add_argument("--attention-rtol", type=float, default=DEFAULT_ATTENTION_RTOL)
    parser.add_argument("--metric-atol", type=float, default=DEFAULT_METRIC_ATOL)
    parser.add_argument("--metric-rtol", type=float, default=DEFAULT_METRIC_RTOL)
    parser.add_argument(
        "--no-bundle-cross-checks",
        action="store_true",
        help=(
            "Do not auto-compare sibling runner metrics/provenance. The checkpoint, "
            "input, plateau, and double-replay gates still run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    cohort_dir = args.cohort_dir.expanduser().resolve(strict=True)
    graph_dir = args.graph_dir.expanduser().resolve(strict=True)
    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    cross_checks = (
        {
            "expected_core_metrics_path": None,
            "expected_final_metrics_path": None,
            "training_provenance_path": None,
        }
        if args.no_bundle_cross_checks
        else _bundle_cross_check_paths(checkpoint)
    )
    receipt = verify_relative_qkv_checkpoint(
        checkpoint,
        batches,
        cohort_manifest_path=cohort_dir / "manifest.json",
        graph_manifest_path=graph_dir / "manifest.json",
        amendment_path=args.amendment,
        device=args.device,
        attention_receivers_per_core=args.attention_receivers_per_core,
        prediction_atol=args.prediction_atol,
        prediction_rtol=args.prediction_rtol,
        attention_atol=args.attention_atol,
        attention_rtol=args.attention_rtol,
        metric_atol=args.metric_atol,
        metric_rtol=args.metric_rtol,
        **cross_checks,
    )
    destination = write_verification_receipt(receipt, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checkpoint_sha256": receipt["checkpoint"]["file_sha256"],
                "completed_global_epochs": receipt["checkpoint"][
                    "completed_global_epochs"
                ],
                "receipt": str(destination),
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

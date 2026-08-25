#!/usr/bin/env python3
"""Run the checksum-gated compact stability report for model seeds 0-3.

Bind this process to one physical GPU with ``CUDA_VISIBLE_DEVICES`` and pass the
corresponding explicit logical device (normally ``cuda:0``).  The command first
validates every immutable run bundle, final checkpoint, strict verification
receipt, prepared-data checksum, protocol checksum, and regenerated locked
gradient request on CPU.  It then loads only one model and stages only one core
on CUDA at a time.  The destination is published atomically and never replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.relative_qkv_four_seed_stability import (  # noqa: E402
    parse_seed_path_specs,
    run_four_seed_stability_pipeline,
)


CAMPAIGN_DIRECTORY = (
    "experiments/campaigns/"
    "cmp_20260824_cancer_6core_relative_qkv_multiseed"
)


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    campaign = paths.project_root / CAMPAIGN_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=ARCHIVED_RUN_DIRECTORY",
        help="Repeat exactly four times for seeds 0, 1, 2, and 3.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="SEED=LAST_CHECKPOINT",
        help="Repeat exactly four times; each path must be the run's last.ckpt.",
    )
    parser.add_argument(
        "--receipt",
        action="append",
        required=True,
        metavar="SEED=STRICT_VERIFICATION_RECEIPT",
        help="Repeat exactly four times for the passed strict checkpoint receipts.",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=paths.data_root / "processed/cancer_6core_relative_qkv_v1",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=paths.data_root
        / "processed/cancer_6core_relative_qkv_graphs_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=campaign / "analysis_protocol_selected_gradient_stability_v1.yaml",
    )
    parser.add_argument(
        "--protocol-sha256",
        type=Path,
        default=campaign / "analysis_protocol_selected_gradient_stability_v1.sha256",
    )
    parser.add_argument(
        "--gradient-requests",
        type=Path,
        default=campaign / "selected_gradient_requests_v1.csv",
    )
    parser.add_argument(
        "--gradient-requests-sha256",
        type=Path,
        default=campaign / "selected_gradient_requests_v1.sha256",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New report-bundle directory; any existing path is rejected.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit logical CUDA device, for example cuda:0.",
    )
    parser.add_argument(
        "--receiver-chunk-size",
        type=int,
        choices=(512,),
        default=512,
        help="Locked exact receiver chunk size (only 512 is accepted).",
    )
    parser.add_argument(
        "--max-edges-per-chunk",
        type=int,
        choices=(200_000,),
        default=200_000,
        help="Locked exact maximum edges per receiver chunk (only 200000 is accepted).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_four_seed_stability_pipeline(
        run_roots=parse_seed_path_specs(args.run, label="run"),
        checkpoint_paths=parse_seed_path_specs(
            args.checkpoint,
            label="checkpoint",
        ),
        receipt_paths=parse_seed_path_specs(args.receipt, label="receipt"),
        cohort_dir=args.cohort_dir,
        graph_dir=args.graph_dir,
        protocol_path=args.protocol,
        protocol_sha256_path=args.protocol_sha256,
        request_csv_path=args.gradient_requests,
        request_sha256_path=args.gradient_requests_sha256,
        destination=args.output,
        device=args.device,
        receiver_chunk_size=args.receiver_chunk_size,
        max_edges_per_chunk=args.max_edges_per_chunk,
        amp=True,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate or verify the locked 24-request relative-QKV gradient table.

This command reads checksum-verified prepared cohort/graph artifacts and the
shared deterministic held-in mask.  It never loads a checkpoint or touches a
GPU.  ``generate`` refuses to overwrite either output; ``verify`` requires the
protocol, protocol checksum, request checksum, and exact canonical regeneration
to agree before any downstream derivative job should start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.relative_qkv_gradient_requests import (  # noqa: E402
    freeze_locked_gradient_request_csv,
    generate_locked_gradient_requests,
    load_and_verify_locked_gradient_requests,
    load_prepared_gradient_request_inputs,
    verify_locked_gradient_protocol,
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
        "--requests-csv",
        type=Path,
        default=campaign / "selected_gradient_requests_v1.csv",
    )
    parser.add_argument(
        "--requests-sha256",
        type=Path,
        default=campaign / "selected_gradient_requests_v1.sha256",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "generate",
        help="Create a new CSV and checksum; refuse every overwrite.",
    )
    verify = subparsers.add_parser(
        "verify",
        help="Verify protocol and regenerate the exact table from prepared inputs.",
    )
    verify.add_argument(
        "--protocol",
        type=Path,
        default=campaign / "analysis_protocol_selected_gradient_stability_v1.yaml",
    )
    verify.add_argument(
        "--protocol-sha256",
        type=Path,
        default=campaign / "analysis_protocol_selected_gradient_stability_v1.sha256",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cores = load_prepared_gradient_request_inputs(
        cohort_dir=args.cohort_dir,
        graph_dir=args.graph_dir,
    )
    if args.command == "generate":
        requests = generate_locked_gradient_requests(cores)
        checksum = freeze_locked_gradient_request_csv(
            requests,
            request_csv_path=args.requests_csv,
            request_sha256_path=args.requests_sha256,
        )
        result = {
            "status": "generated",
            "request_count": len(requests),
            "request_table_sha256": checksum,
        }
    else:
        protocol = verify_locked_gradient_protocol(
            args.protocol,
            args.protocol_sha256,
        )
        requests = load_and_verify_locked_gradient_requests(
            args.requests_csv,
            args.requests_sha256,
            cores=cores,
            protocol=protocol,
        )
        result = {
            "status": "verified",
            "request_count": len(requests),
            "request_table_sha256": protocol["selected_gradient_requests"][
                "request_table_sha256"
            ],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the final MyJJu GeneMAE versus current BAGM ten-core comparison.

The default provider audits and replays all seven GeneMAE, seven current pooled
GAT, and seven current matched-self final checkpoints.  It regenerates exact
fixed masks and canonical prediction ensembles, then writes only aggregate,
opaque-alias-safe report artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.myjju_genemae_comparison import (  # noqa: E402
    CAMPAIGN_ID,
    DEFAULT_OUTPUT_RELATIVE,
    ComparisonProvider,
    run_comparison,
)
from spatial_benchmark.myjju_genemae_provider import (  # noqa: E402
    RegisteredCheckpointComparisonProvider,
)
from spatial_benchmark.paths import current_paths  # noqa: E402


def _provider_factory(reference: str) -> Callable[..., ComparisonProvider]:
    try:
        module_name, attribute = reference.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            "--provider-factory must be importable as module:callable"
        ) from exc
    if not callable(factory):
        raise TypeError("--provider-factory target must be callable")
    return factory


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.report_root / DEFAULT_OUTPUT_RELATIVE,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Evaluation device. Current BAGM campaign GPU exclusions apply.",
    )
    parser.add_argument(
        "--provider-factory",
        default=None,
        help=(
            "Optional injectable module:callable for controlled tests. The "
            "built-in registered-checkpoint provider is the production default."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = current_paths()
    keyword = {
        "paths": paths,
        "database_path": args.database.resolve(strict=False),
        "device_name": str(args.device),
    }
    provider: ComparisonProvider
    if args.provider_factory:
        provider = _provider_factory(str(args.provider_factory))(**keyword)
    else:
        provider = RegisteredCheckpointComparisonProvider(**keyword)
    result = run_comparison(
        provider=provider,
        output_dir=args.output.resolve(strict=False),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "status": result["analysis"]["status"],
                "outcome": result["analysis"]["outcome"],
                "output": result["manifest"]["output_dir"],
                "manifest_sha256": result["manifest"]["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

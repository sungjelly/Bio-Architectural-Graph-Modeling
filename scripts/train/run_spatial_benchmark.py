#!/usr/bin/env python3
"""Train one isolated architecture/seed/graph benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.experiment import run_experiment  # noqa: E402
from spatial_benchmark.paths import PROJECT_ROOT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "b0",
            "b0-matched",
            "broad-field",
            "b1",
            "g1",
            "g2",
            "g3",
        ),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int)
    parser.add_argument("--radius-um", type=float)
    parser.add_argument("--symmetry", choices=("union", "mutual"))
    parser.add_argument("--min-distance-um", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--graph-layers", type=int)
    parser.add_argument("--edge-embedding-dim", type=int)
    parser.add_argument("--curriculum", choices=("P-only", "P+N", "P+N+B"))
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--edge-dropout", type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rewired", action="store_true")
    parser.add_argument("--rewire-seed", type=int, default=271828)
    parser.add_argument("--swaps-per-edge", type=float, default=1.0)
    parser.add_argument(
        "--edge-control",
        choices=("none", "zero", "distance_only", "permuted"),
        default="none",
    )
    parser.add_argument("--pretrained-b0-checkpoint", type=Path)
    parser.add_argument("--g3-frozen-epochs", type=int, default=10)
    parser.add_argument("--g3-joint-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--open-test",
        action="store_true",
        help="Evaluate the sealed test masks. Use only after standards are locked.",
    )
    parser.add_argument(
        "--standards-lock",
        type=Path,
        help=(
            "Verified standards-lock artifact authorizing this exact final "
            "matrix job; required with --open-test."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _defined(values: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph = _defined(
        {
            "k": args.k,
            "radius_um": args.radius_um,
            "symmetry": args.symmetry,
            "min_distance_um": args.min_distance_um,
        }
    )
    model = _defined(
        {
            "hidden_dim": args.hidden_dim,
            "graph_layers": args.graph_layers,
            "edge_embedding_dim": args.edge_embedding_dim,
        }
    )
    training = _defined(
        {
            "device": args.device,
            "curriculum": args.curriculum,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "edge_dropout": args.edge_dropout,
            "amp": args.amp,
        }
    )
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(sys.argv[1:] if argv is None else argv)
    output = run_experiment(
        args.prepared,
        args.output,
        project_root=PROJECT_ROOT,
        model_name=args.model,
        model_seed=args.seed,
        graph_overrides=graph,
        model_overrides=model,
        training_overrides=training,
        rewired=args.rewired,
        rewire_seed=args.rewire_seed,
        swaps_per_edge=args.swaps_per_edge,
        edge_control=args.edge_control,
        pretrained_b0_checkpoint=args.pretrained_b0_checkpoint,
        g3_frozen_epochs=args.g3_frozen_epochs,
        g3_joint_learning_rate=args.g3_joint_learning_rate,
        evaluate_test=args.open_test,
        save_predictions=args.save_predictions,
        standards_lock_path=args.standards_lock,
        command=command,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "output": str(output),
                "model": manifest["model_name"],
                "seed": manifest["model_seed"],
                "graph_id": manifest["graph"]["graph_id"],
                "best_epoch": manifest["training"]["best_epoch"],
                "best_validation_loss": manifest["training"][
                    "best_validation_loss"
                ],
                "sealed_test_opened": manifest["sealed_test_opened"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

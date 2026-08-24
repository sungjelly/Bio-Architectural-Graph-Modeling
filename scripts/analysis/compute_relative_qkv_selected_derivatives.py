#!/usr/bin/env python3
"""Compute selected relative-QKV attention and prediction input derivatives.

The request file is CSV or JSON.  Each row supplies ``core_alias``,
``source_node``, ``source_feature``, ``receiver_node``, and ``target_feature``.
Features may be zero-based indices or exact gene names.  Optional fields are
``request_id``, ``attention_head`` (an integer or ``mean``), and ``layer``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.relative_qkv_post_training import (  # noqa: E402
    RelativeQKVPostTrainingError,
    SelectedDerivativeRequest,
    file_sha256,
    fixed_inference_mask,
    load_core_coordinates_and_genes,
    load_prepared_relative_qkv_batches,
    load_relative_qkv_checkpoint,
    selected_autograd_derivatives,
)


DERIVATIVE_SCHEMA = "cancer_6core_relative_qkv_selected_derivatives_v1"
DEFAULT_MAX_SELECTIONS = 128


def _raw_request_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = [dict(row) for row in csv.DictReader(stream)]
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                value = value.get("selections")
            if not isinstance(value, list):
                raise RelativeQKVPostTrainingError(
                    "JSON requests must be a list or a mapping with selections."
                )
            rows = []
            for row in value:
                if not isinstance(row, Mapping):
                    raise RelativeQKVPostTrainingError(
                        "Every derivative selection must be a mapping."
                    )
                rows.append(dict(row))
        else:
            raise RelativeQKVPostTrainingError(
                "Derivative requests must use a .csv or .json suffix."
            )
    except (OSError, ValueError, csv.Error) as exc:
        raise RelativeQKVPostTrainingError(
            f"Cannot parse derivative request file: {path}."
        ) from exc
    if not rows:
        raise RelativeQKVPostTrainingError("Derivative request file is empty.")
    return rows


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RelativeQKVPostTrainingError(f"{field} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RelativeQKVPostTrainingError(f"{field} must be an integer.") from exc
    if str(value).strip() not in {str(parsed), f"+{parsed}"}:
        raise RelativeQKVPostTrainingError(f"{field} must be an integer.")
    return parsed


def _feature_index(value: Any, genes: Sequence[str], field: str) -> int:
    text = str(value).strip()
    by_name = {name: index for index, name in enumerate(genes)}
    if text in by_name:
        return by_name[text]
    index = _integer(value, field)
    if not 0 <= index < len(genes):
        raise RelativeQKVPostTrainingError(f"{field} is out of range.")
    return index


def load_requests(
    path: Path,
    *,
    genes_by_core: Mapping[str, Sequence[str]],
    default_layer: int,
    max_selections: int,
) -> list[SelectedDerivativeRequest]:
    """Parse and resolve a bounded derivative request table."""

    rows = _raw_request_rows(path)
    if max_selections <= 0 or len(rows) > max_selections:
        raise RelativeQKVPostTrainingError(
            f"Request count must be between 1 and {max_selections}."
        )
    requests: list[SelectedDerivativeRequest] = []
    for index, row in enumerate(rows):
        missing = [
            field
            for field in (
                "core_alias",
                "source_node",
                "source_feature",
                "receiver_node",
                "target_feature",
            )
            if field not in row or str(row[field]).strip() == ""
        ]
        if missing:
            raise RelativeQKVPostTrainingError(
                "Derivative request is missing: " + ", ".join(missing)
            )
        alias = str(row["core_alias"]).strip().upper()
        if alias not in genes_by_core:
            raise RelativeQKVPostTrainingError(
                f"Derivative request contains unknown core {alias!r}."
            )
        head_value = row.get("attention_head", "mean")
        head_text = str(head_value).strip().lower()
        attention_head = (
            None
            if head_text in {"", "mean", "none"}
            else _integer(head_value, "attention_head")
        )
        layer_value = row.get("layer", default_layer)
        if str(layer_value).strip() == "":
            layer_value = default_layer
        genes = genes_by_core[alias]
        requests.append(
            SelectedDerivativeRequest(
                request_id=str(row.get("request_id") or f"request-{index:05d}"),
                core_alias=alias,
                source_node=_integer(row["source_node"], "source_node"),
                source_feature=_feature_index(
                    row["source_feature"], genes, "source_feature"
                ),
                receiver_node=_integer(row["receiver_node"], "receiver_node"),
                target_feature=_feature_index(
                    row["target_feature"], genes, "target_feature"
                ),
                attention_head=attention_head,
                layer=_integer(layer_value, "layer"),
            )
        )
    identifiers = [request.request_id for request in requests]
    if len(set(identifiers)) != len(identifiers):
        raise RelativeQKVPostTrainingError("Derivative request IDs must be unique.")
    return requests


def compute_derivatives(
    *,
    checkpoint_path: Path,
    cohort_dir: Path,
    graph_dir: Path,
    requests_path: Path,
    output_dir: Path,
    device: str,
    default_layer: int,
    max_selections: int,
) -> dict[str, Any]:
    """Compute selected local derivatives and write a new Parquet artifact."""

    destination = output_dir.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    by_alias = {batch.alias: batch for batch in batches}
    genes_by_core = {
        alias: load_core_coordinates_and_genes(cohort_dir, alias=alias)[1]
        for alias in CANCER_ALIASES
    }
    requests = load_requests(
        requests_path,
        genes_by_core=genes_by_core,
        default_layer=default_layer,
        max_selections=max_selections,
    )
    first = batches[0]
    loaded = load_relative_qkv_checkpoint(
        checkpoint_path,
        num_genes=first.n_genes,
        node_covariate_dim=int(first.node_covariates.shape[1]),
        device=device,
    )

    result_rows: list[dict[str, Any]] = []
    mask_receipts: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, int], list[SelectedDerivativeRequest]] = {}
    for request in requests:
        groups.setdefault((request.core_alias, request.layer), []).append(request)
    for (alias, _), group in groups.items():
        batch = by_alias[alias]
        fixed_mask = fixed_inference_mask(batch)
        mask_receipts[alias] = {
            "effective_seed": fixed_mask.seed,
            "checksum_sha256": fixed_mask.checksum_sha256,
            "masked_entry_count": fixed_mask.n_masked_entries,
        }
        rows = selected_autograd_derivatives(
            loaded.model,
            batch,
            fixed_mask.mask,
            group,
        )
        genes = genes_by_core[alias]
        for row in rows:
            row["source_feature_name"] = genes[row["source_feature_index"]]
            row["target_feature_name"] = genes[row["target_feature_index"]]
        result_rows.extend(rows)

    order = {request.request_id: index for index, request in enumerate(requests)}
    result_rows.sort(key=lambda row: order[row["request_id"]])
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        table_path = staging / "derivatives.parquet"
        pq.write_table(
            pa.Table.from_pylist(result_rows),
            table_path,
            compression="zstd",
            use_dictionary=["core_alias", "attention_head"],
        )
        manifest: dict[str, Any] = {
            "schema": DERIVATIVE_SCHEMA,
            "selection_count": len(result_rows),
            "request_sha256": file_sha256(requests_path),
            "checkpoint": {
                "path": loaded.checkpoint_path.as_posix(),
                "sha256": loaded.checkpoint_sha256,
                "model_state_checksum": loaded.payload["model_state_checksum"],
                "completed_global_epochs": loaded.payload.get(
                    "completed_global_epochs"
                ),
            },
            "prepared_inputs": {
                "cohort_manifest_sha256": file_sha256(
                    cohort_dir / "manifest.json"
                ),
                "graph_manifest_sha256": file_sha256(graph_dir / "manifest.json"),
            },
            "fixed_inference_masks": mask_receipts,
            "autograd_scope": (
                "selected source-node/source-feature injections only; no "
                "exhaustive edge-by-gene or node-by-gene Jacobian"
            ),
            "input_scale": "standardized log1p expression",
            "attention_derivative": (
                "selected directed-edge head attention, or head mean, with "
                "respect to the selected source input"
            ),
            "prediction_derivative": (
                "selected receiver/target-feature prediction with respect to "
                "the selected source input; may include multihop paths"
            ),
            "claim_scope": (
                "local scale-dependent model sensitivity on fixed held-in fit "
                "masks; not correlation, biological mechanism, or causality"
            ),
            "output": {
                "path": "derivatives.parquet",
                "sha256": file_sha256(table_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=paths.data_root / "processed/cancer_6core_relative_qkv_v1",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=paths.data_root / "processed/cancer_6core_relative_qkv_graphs_v1",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--max-selections", type=int, default=DEFAULT_MAX_SELECTIONS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = compute_derivatives(
        checkpoint_path=args.checkpoint,
        cohort_dir=args.cohort_dir,
        graph_dir=args.graph_dir,
        requests_path=args.requests,
        output_dir=args.output_dir,
        device=args.device,
        default_layer=args.layer,
        max_selections=args.max_selections,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

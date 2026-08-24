#!/usr/bin/env python3
"""Export receiver-sharded relative-QKV edge routing for one prepared core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.relative_qkv_post_training import (  # noqa: E402
    RelativeQKVPostTrainingError,
    file_sha256,
    fixed_inference_mask,
    load_core_coordinates_and_genes,
    load_prepared_relative_qkv_batches,
    load_relative_qkv_checkpoint,
    reciprocal_edge_ids,
    stream_receiver_attention,
)


EXPORT_SCHEMA = "cancer_6core_relative_qkv_edge_attention_v1"


def _edge_table(
    *,
    alias: str,
    layer_number: int,
    edge_ids: np.ndarray,
    edge_index: np.ndarray,
    coordinates: np.ndarray,
    indegree: np.ndarray,
    reciprocal_ids: np.ndarray,
    pair_keys: np.ndarray,
    attention: np.ndarray,
    content: np.ndarray,
    bias: np.ndarray,
    combined: np.ndarray,
) -> tuple[pa.Table, np.ndarray]:
    """Build one intermediate receiver shard and its directional routing."""

    ids = np.asarray(edge_ids, dtype=np.int64)
    source = edge_index[0, ids].astype(np.int64, copy=False)
    receiver = edge_index[1, ids].astype(np.int64, copy=False)
    arrays = [
        np.asarray(values, dtype=np.float32)
        for values in (attention, content, bias, combined)
    ]
    if any(values.ndim != 2 or values.shape[0] != len(ids) for values in arrays):
        raise RelativeQKVPostTrainingError(
            "Streamed attention channels do not align to edge IDs."
        )
    attention_values, content_values, bias_values, combined_values = arrays
    if not all(np.isfinite(values).all() for values in arrays):
        raise RelativeQKVPostTrainingError(
            "Streamed attention channels contain non-finite values."
        )
    attention_mean = attention_values.mean(axis=1)
    directional = indegree[receiver].astype(np.float32) * attention_mean
    delta = coordinates[source] - coordinates[receiver]
    columns: dict[str, Any] = {
        "edge_id": ids,
        "core_alias": [alias] * len(ids),
        "layer_number": np.full(len(ids), layer_number, dtype=np.int16),
        "source_node": source,
        "receiver_node": receiver,
        "source_x_um": coordinates[source, 0],
        "source_y_um": coordinates[source, 1],
        "receiver_x_um": coordinates[receiver, 0],
        "receiver_y_um": coordinates[receiver, 1],
        "distance_um": np.linalg.norm(delta, axis=1),
        "receiver_in_degree": indegree[receiver].astype(np.int32),
        "attention_mean": attention_mean,
        "content_logit_mean": content_values.mean(axis=1),
        "positional_bias_mean": bias_values.mean(axis=1),
        "combined_logit_mean": combined_values.mean(axis=1),
        "degree_adjusted_attention": directional,
        "reciprocal_edge_id": reciprocal_ids[ids],
        "reciprocal_key": pair_keys[ids],
    }
    for head in range(attention_values.shape[1]):
        columns[f"attention_head_{head:02d}"] = attention_values[:, head]
    return pa.table(columns), directional


def export_attention(
    *,
    checkpoint_path: Path,
    cohort_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    core_alias: str,
    device: str,
    layer: int,
    receiver_shard_size: int,
    max_edges_per_shard: int,
    amp: bool,
) -> dict[str, Any]:
    """Run one bounded receiver-sharded export into a new directory."""

    destination = output_dir.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if receiver_shard_size <= 0 or max_edges_per_shard <= 0:
        raise RelativeQKVPostTrainingError(
            "Receiver and edge shard limits must be positive."
        )
    alias = core_alias.strip().upper()
    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    batch = next(value for value in batches if value.alias == alias)
    coordinates, _ = load_core_coordinates_and_genes(cohort_dir, alias=alias)
    if len(coordinates) != batch.n_nodes:
        raise RelativeQKVPostTrainingError(
            "Plotting coordinates do not align to the prepared core batch."
        )
    fixed_mask = fixed_inference_mask(batch)
    loaded = load_relative_qkv_checkpoint(
        checkpoint_path,
        num_genes=batch.n_genes,
        node_covariate_dim=int(batch.node_covariates.shape[1]),
        device=device,
        receiver_chunk_size=receiver_shard_size,
        max_edges_per_chunk=max_edges_per_shard,
    )
    edge_index = batch.edge_index.detach().cpu().numpy()
    reciprocal_ids, pair_keys = reciprocal_edge_ids(
        edge_index, n_nodes=batch.n_nodes
    )
    indegree = np.bincount(edge_index[1], minlength=batch.n_nodes).astype(np.int64)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    intermediate = staging / "intermediate"
    intermediate.mkdir()
    routing_path = staging / "directional_routing.float32.dat"
    routing = np.memmap(
        routing_path,
        mode="w+",
        dtype=np.float32,
        shape=(batch.n_edges,),
    )
    routing[:] = np.nan
    shard_receipts: list[dict[str, Any]] = []

    try:
        def consume(
            receiver_start: int,
            receiver_stop: int,
            edge_ids: np.ndarray,
            attention: np.ndarray,
            content: np.ndarray,
            bias: np.ndarray,
            combined: np.ndarray,
        ) -> None:
            shard_index = len(shard_receipts)
            table, directional = _edge_table(
                alias=alias,
                layer_number=resolved_layer,
                edge_ids=edge_ids,
                edge_index=edge_index,
                coordinates=coordinates,
                indegree=indegree,
                reciprocal_ids=reciprocal_ids,
                pair_keys=pair_keys,
                attention=attention,
                content=content,
                bias=bias,
                combined=combined,
            )
            routing[edge_ids] = directional
            name = f"part_{shard_index:05d}.parquet"
            pq.write_table(
                table,
                intermediate / name,
                compression="zstd",
                use_dictionary=["core_alias"],
            )
            shard_receipts.append(
                {
                    "receiver_start": int(receiver_start),
                    "receiver_stop": int(receiver_stop),
                    "edge_count": int(len(edge_ids)),
                    "intermediate_name": name,
                }
            )

        # The callback needs the normalized layer value. Resolve it before the
        # streaming call while preserving negative-layer CLI convenience.
        resolved_layer = layer if layer >= 0 else loaded.model.graph_layers + layer
        resolved_layer = stream_receiver_attention(
            loaded.model,
            batch,
            fixed_mask.mask,
            layer=layer,
            amp=amp,
            consumer=consume,
        )
        routing.flush()
        if not np.isfinite(routing).all():
            raise RelativeQKVPostTrainingError(
                "Receiver shards did not cover every directed edge exactly once."
            )

        output_shards: list[dict[str, Any]] = []
        for index, receipt in enumerate(shard_receipts):
            table = pq.read_table(intermediate / receipt["intermediate_name"])
            edge_ids = table.column("edge_id").to_numpy(zero_copy_only=False)
            mutual = np.minimum(routing[edge_ids], routing[reciprocal_ids[edge_ids]])
            table = table.append_column("mutual_routing_score", pa.array(mutual))
            final_name = (
                f"receiver_{receipt['receiver_start']:06d}_"
                f"{receipt['receiver_stop']:06d}_{index:05d}.parquet"
            )
            final_path = staging / final_name
            pq.write_table(
                table,
                final_path,
                compression="zstd",
                use_dictionary=["core_alias"],
            )
            output_shards.append(
                {
                    "path": final_name,
                    "sha256": file_sha256(final_path),
                    "receiver_start": receipt["receiver_start"],
                    "receiver_stop": receipt["receiver_stop"],
                    "edge_count": receipt["edge_count"],
                }
            )

        del routing
        routing_path.unlink()
        shutil.rmtree(intermediate)
        manifest: dict[str, Any] = {
            "schema": EXPORT_SCHEMA,
            "core_alias": alias,
            "layer_number": int(resolved_layer),
            "node_count": batch.n_nodes,
            "directed_edge_count": batch.n_edges,
            "attention_head_count": loaded.model.blocks[
                resolved_layer
            ].attention_heads,
            "receiver_shard_size": int(receiver_shard_size),
            "max_edges_per_shard": int(max_edges_per_shard),
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
            "fixed_inference_mask": {
                "base_seed": 2026082491,
                "effective_seed": fixed_mask.seed,
                "checksum_sha256": fixed_mask.checksum_sha256,
                "masked_entry_count": fixed_mask.n_masked_entries,
            },
            "plotting_coordinates_are_model_inputs": False,
            "numeric_storage": {
                "coordinates_and_distance": "float64",
                "routing_and_logit_channels": "float32",
            },
            "degree_adjustment": "receiver_in_degree * mean_head_attention",
            "reciprocal_key": "min_node * node_count + max_node",
            "mutual_routing_score": (
                "minimum degree-adjusted mean attention across reciprocal edges"
            ),
            "claim_scope": (
                "descriptive computational routing on one fixed held-in fit mask; "
                "not biological importance or causal influence"
            ),
            "amp": bool(amp),
            "shards": output_shards,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        try:
            del routing
        except UnboundLocalError:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths(anchor=__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core", choices=CANCER_ALIASES, required=True)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--receiver-shard-size", type=int, default=512)
    parser.add_argument("--max-edges-per-shard", type=int, default=200_000)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA float16 autocast for the frozen inference replay.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_attention(
        checkpoint_path=args.checkpoint,
        cohort_dir=args.cohort_dir,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        core_alias=args.core,
        device=args.device,
        layer=args.layer,
        receiver_shard_size=args.receiver_shard_size,
        max_edges_per_shard=args.max_edges_per_shard,
        amp=args.amp,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

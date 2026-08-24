#!/usr/bin/env python3
"""Largest-core numerical and memory preflight for the Cancer relative QKV run.

This is deliberately a diagnostic, not a training run.  It uses the real
largest prepared Cancer core to (1) compare the transparent full-edge and
receiver-chunked operators on a complete receiver subset, (2) compare AMP and
FP32 inference on the same subset, (3) execute one complete-core AMP
forward/backward pass without dropping any edge, and (4) verify an independent
state-dict save/load round trip.  The checksum-bound receipt is consumed by the
production runner before seed 0 can start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train.run_pooled_relative_qkv import (  # noqa: E402
    _model_from_config,
    _validate_active_contract,
)
from spatial_benchmark.configuration import compose_config  # noqa: E402
from spatial_benchmark.cancer_relative_graphs import (  # noqa: E402
    load_cancer_relative_qkv_batches,
)
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    make_exact_uniform_training_mask,
    masked_huber_reconstruction_loss,
)
from spatial_benchmark.relative_qkv_graph_transformer import (  # noqa: E402
    RelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.training import (  # noqa: E402
    _autocast_context,
    _make_grad_scaler,
    set_deterministic_seed,
)


RECEIPT_SCHEMA = "cancer_6core_relative_qkv_hardware_preflight_v1"
DEFAULT_CONFIG = Path("configs/experiment/cancer_6core_relative_qkv_seed0.yaml")
DEFAULT_OUTPUT = Path("state/preflight/cancer_6core_relative_qkv_seed0.json")
COMPARISON_RECEIVERS = 128
FULL_CHUNK_MAX_ABS_TOLERANCE = 2e-5
FULL_CHUNK_MEAN_ABS_TOLERANCE = 2e-6
AMP_FP32_MAX_ABS_TOLERANCE = 7.5e-2
AMP_FP32_MEAN_ABS_TOLERANCE = 7.5e-3


class RelativeQKVPreflightError(RuntimeError):
    """Raised when a production gate cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _difference(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    if reference.shape != observed.shape:
        raise RelativeQKVPreflightError("Numerical comparison shapes differ.")
    delta = (reference.float() - observed.float()).abs()
    return {
        "maximum_absolute_difference": float(delta.max().cpu()) if delta.numel() else 0.0,
        "mean_absolute_difference": float(delta.mean().cpu()) if delta.numel() else 0.0,
    }


def _comparison_edge_ids(edge_index: torch.Tensor, receiver_count: int) -> torch.Tensor:
    """Return every incoming edge for the first complete receiver range."""

    if edge_index.device.type != "cpu" or edge_index.shape[0] != 2:
        raise RelativeQKVPreflightError("Preflight expects a CPU edge_index [2, E].")
    receiver_count = min(int(receiver_count), int(edge_index.max().item()) + 1)
    if receiver_count <= 0:
        raise RelativeQKVPreflightError("Comparison receiver range is empty.")
    edge_ids = torch.nonzero(edge_index[1] < receiver_count, as_tuple=False).flatten()
    if not edge_ids.numel():
        raise RelativeQKVPreflightError("Comparison range contains no incoming edges.")
    return edge_ids


def _full_reference_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> RelativeGeometryQKVGraphTransformer:
    model = config["model"]
    assert isinstance(model, Mapping)
    return RelativeGeometryQKVGraphTransformer(
        num_genes=num_genes,
        node_covariate_dim=node_covariate_dim,
        hidden_dim=int(model["hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        attention_head_dim=int(model["attention_head_dim"]),
        graph_layers=int(model["graph_layers"]),
        ffn_dim=int(model["ffn_dim"]),
        decoder_dim=int(model["decoder_dim"]),
        positional_bias_hidden_dim=int(model["positional_bias_hidden_dim"]),
        dropout=float(model["dropout"]),
        attention_dropout=float(model["attention_dropout"]),
        relative_geometry_dim=int(model["relative_geometry_dim"]),
    )


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def run_preflight(
    *,
    config_path: Path,
    output_path: Path,
    comparison_receivers: int = COMPARISON_RECEIVERS,
) -> dict[str, Any]:
    """Run and persist all seed-0 production hardware gates."""

    if not torch.cuda.is_available():
        raise RelativeQKVPreflightError("A CUDA GPU is required for production preflight.")
    resolved = compose_config(config_path, config_root=PROJECT_ROOT / "configs")
    _validate_active_contract(resolved)
    dataset = resolved["dataset"]
    trainer = resolved["trainer"]
    model_section = resolved["model"]
    assert isinstance(dataset, Mapping)
    assert isinstance(trainer, Mapping)
    assert isinstance(model_section, Mapping)
    cohort_dir = (PROJECT_ROOT / str(dataset["prepared_artifact"])).resolve()
    graph_dir = (PROJECT_ROOT / str(dataset["prepared_graph_artifact"])).resolve()
    batches = load_cancer_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    largest = max(batches, key=lambda batch: (batch.n_edges, batch.n_nodes))
    device = torch.device("cuda")
    set_deterministic_seed(0, deterministic=True, warn_only=False)
    torch.cuda.reset_peak_memory_stats(device)

    chunked = _model_from_config(
        resolved,
        num_genes=largest.n_genes,
        node_covariate_dim=int(largest.node_covariates.shape[1]),
    ).to(device)
    reference = _full_reference_from_config(
        resolved,
        num_genes=largest.n_genes,
        node_covariate_dim=int(largest.node_covariates.shape[1]),
    ).to(device)
    reference.load_state_dict(chunked.state_dict(), strict=True)
    parameter_count = sum(parameter.numel() for parameter in chunked.parameters())

    target = largest.target_expression.to(device=device, dtype=torch.float32)
    covariates = largest.node_covariates.to(device=device, dtype=torch.float32)
    mask_realization = make_exact_uniform_training_mask(
        largest,
        0,
        view_index=0,
    )
    gene_mask = torch.from_numpy(np.array(mask_realization.mask, copy=True)).to(
        device=device,
        dtype=torch.bool,
    )
    input_expression = target.masked_fill(gene_mask, 0.0)

    comparison_ids = _comparison_edge_ids(
        largest.edge_index,
        comparison_receivers,
    )
    comparison_edges = largest.edge_index.index_select(1, comparison_ids)
    comparison_geometry = largest.relative_geometry.index_select(0, comparison_ids)
    comparison_targets = torch.arange(
        min(comparison_receivers, largest.n_nodes),
        device=device,
        dtype=torch.long,
    )

    chunked.eval()
    reference.eval()
    with torch.no_grad():
        full_output = reference(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=comparison_targets,
            return_explanations=True,
            attention_receivers=comparison_targets,
        )
        chunked_output = chunked(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=comparison_targets,
            return_explanations=True,
            attention_receivers=comparison_targets.cpu(),
        )
    output_difference = _difference(full_output.prediction, chunked_output.prediction)
    attention_difference = _difference(
        full_output.attention_weights,
        chunked_output.attention_weights,
    )
    full_chunk_passed = bool(
        output_difference["maximum_absolute_difference"]
        <= FULL_CHUNK_MAX_ABS_TOLERANCE
        and output_difference["mean_absolute_difference"]
        <= FULL_CHUNK_MEAN_ABS_TOLERANCE
        and attention_difference["maximum_absolute_difference"]
        <= FULL_CHUNK_MAX_ABS_TOLERANCE
        and attention_difference["mean_absolute_difference"]
        <= FULL_CHUNK_MEAN_ABS_TOLERANCE
        and torch.equal(full_output.edge_index.cpu(), chunked_output.edge_index.cpu())
    )

    amp_dtype = str(trainer["amp_dtype"])
    with torch.no_grad(), _autocast_context(
        enabled=True,
        device=device,
        dtype_name=amp_dtype,
    ):
        amp_output = chunked(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=comparison_targets,
        )
    amp_difference = _difference(chunked_output.prediction, amp_output.prediction)
    amp_equivalence_passed = bool(
        amp_difference["maximum_absolute_difference"] <= AMP_FP32_MAX_ABS_TOLERANCE
        and amp_difference["mean_absolute_difference"] <= AMP_FP32_MEAN_ABS_TOLERANCE
        and bool(torch.isfinite(amp_output.prediction).all())
    )

    del reference, full_output, chunked_output, amp_output
    torch.cuda.empty_cache()
    chunked.train()
    optimizer = torch.optim.AdamW(
        chunked.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scaler = _make_grad_scaler(device, enabled=True)
    optimizer.zero_grad(set_to_none=True)
    production_edges = largest.edge_index.to(device=device, dtype=torch.long)
    geometry_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(trainer["staged_relative_geometry_dtype"])]
    production_geometry = largest.relative_geometry.to(
        device=device,
        dtype=geometry_dtype,
    )
    torch.cuda.synchronize(device)
    production_started = time.monotonic()
    with _autocast_context(enabled=True, device=device, dtype_name=amp_dtype):
        production_output = chunked(
            input_expression,
            gene_mask,
            production_edges,
            production_geometry,
            covariates,
            target_nodes=torch.arange(largest.n_nodes, device=device),
        )
        production_loss = masked_huber_reconstruction_loss(
            production_output.prediction,
            target,
            gene_mask,
            delta=1.0,
        )
    scaler.scale(production_loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        chunked.parameters(),
        float(trainer["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    finite_gradient_gate = bool(
        math.isfinite(float(production_loss.detach().cpu()))
        and math.isfinite(float(gradient_norm.detach().cpu()))
        and _finite_gradients(chunked)
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    production_seconds = time.monotonic() - production_started
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device))

    del production_output
    torch.cuda.empty_cache()

    chunked.eval()
    with tempfile.TemporaryDirectory(prefix="relative-qkv-preflight-") as temporary:
        checkpoint_path = Path(temporary) / "roundtrip.ckpt"
        torch.save(
            {
                "model_state_dict": chunked.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            checkpoint_path,
        )
        loaded = _model_from_config(
            resolved,
            num_genes=largest.n_genes,
            node_covariate_dim=int(largest.node_covariates.shape[1]),
        ).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        loaded.load_state_dict(payload["model_state_dict"], strict=True)
        loaded.eval()
        with torch.no_grad():
            expected = chunked(
                input_expression,
                gene_mask,
                comparison_edges,
                comparison_geometry,
                covariates,
                target_nodes=comparison_targets,
            ).prediction
            observed = loaded(
                input_expression,
                gene_mask,
                comparison_edges,
                comparison_geometry,
                covariates,
                target_nodes=comparison_targets,
            ).prediction
        roundtrip_difference = _difference(expected, observed)
        roundtrip_passed = bool(roundtrip_difference["maximum_absolute_difference"] == 0.0)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "diagnostic_only": True,
        "completed_experiment": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "resolved_config_sha256": _canonical_sha256(resolved),
        "cohort_manifest_sha256": _sha256_file(cohort_dir / "manifest.json"),
        "graph_manifest_sha256": _sha256_file(graph_dir / "manifest.json"),
        "largest_core": {
            "alias": largest.alias,
            "nodes": largest.n_nodes,
            "edges": largest.n_edges,
            "comparison_receivers": int(len(comparison_targets)),
            "comparison_edges": int(comparison_ids.numel()),
            "complete_core_edges_retained_for_memory_gate": largest.n_edges,
        },
        "model": {
            "parameter_count": int(parameter_count),
            "receiver_chunk_size": int(model_section["receiver_chunk_size"]),
            "max_edges_per_chunk": int(model_section["max_edges_per_chunk"]),
            "activation_checkpointing": bool(model_section["activation_checkpointing"]),
            "stage_complete_core_graph_on_device": bool(
                trainer["stage_complete_core_graph_on_device"]
            ),
            "staged_relative_geometry_dtype": str(
                trainer["staged_relative_geometry_dtype"]
            ),
        },
        "gates": {
            "full_chunk_exactness": {
                "passed": full_chunk_passed,
                "prediction": output_difference,
                "attention": attention_difference,
                "maximum_absolute_tolerance": FULL_CHUNK_MAX_ABS_TOLERANCE,
                "mean_absolute_tolerance": FULL_CHUNK_MEAN_ABS_TOLERANCE,
            },
            "amp_fp32_equivalence": {
                "passed": amp_equivalence_passed,
                **amp_difference,
                "maximum_absolute_tolerance": AMP_FP32_MAX_ABS_TOLERANCE,
                "mean_absolute_tolerance": AMP_FP32_MEAN_ABS_TOLERANCE,
            },
            "complete_core_finite_loss_and_gradients": {
                "passed": finite_gradient_gate,
                "masked_huber": float(production_loss.detach().cpu()),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            },
            "save_load_roundtrip": {
                "passed": roundtrip_passed,
                **roundtrip_difference,
            },
            "peak_vram": {
                "passed": peak_vram_bytes < torch.cuda.get_device_properties(device).total_memory,
                "allocated_bytes": peak_vram_bytes,
                "allocated_gib": peak_vram_bytes / (1024**3),
                "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            },
        },
        "timing": {
            "one_complete_largest_core_amp_forward_backward_seconds": production_seconds,
            "ten_view_largest_core_step_projected_seconds_lower_bound": 10.0
            * production_seconds,
            "projection_is_resource_estimate_not_scientific_result": True,
        },
        "mask": {
            "view_index": 0,
            "effective_seed": mask_realization.effective_seed,
            "checksum_sha256": mask_realization.checksum_sha256,
            "masked_entries": mask_realization.n_masked_entries,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_count_visible": torch.cuda.device_count(),
        },
    }
    receipt["all_required_gates_passed"] = all(
        bool(gate["passed"]) for gate in receipt["gates"].values()
    )
    if not receipt["all_required_gates_passed"]:
        receipt["status"] = "failed"
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not receipt["all_required_gates_passed"]:
        raise RelativeQKVPreflightError(
            f"One or more production gates failed; see {output_path}."
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--comparison-receivers",
        type=int,
        default=COMPARISON_RECEIVERS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_preflight(
        config_path=(PROJECT_ROOT / args.config).resolve()
        if not args.config.is_absolute()
        else args.config,
        output_path=(PROJECT_ROOT / args.output).resolve()
        if not args.output.is_absolute()
        else args.output,
        comparison_receivers=args.comparison_receivers,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

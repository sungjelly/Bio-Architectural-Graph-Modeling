#!/usr/bin/env python3
"""Bounded four-rank hardware preflight for SO2 Relative-QKV training."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train.run_so2_14core_relative_qkv import (  # noqa: E402
    CAMPAIGN_ID,
    MODEL_SEED,
    VISIBLE_DEVICES,
    WORLD_SIZE,
    _canonical_sha256,
    _checkpoint_payload,
    _distributed_identity,
    _model_construction,
    _model_from_config,
    _preflight_bound_config,
    _runtime_path,
    _section,
    _validate_contract,
)
from spatial_benchmark.configuration import compose_config  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    CohortRelativeQKVTrainingConfig,
    fit_cohort_relative_qkv_segment,
)
from spatial_benchmark.so2_relative_graphs import (  # noqa: E402
    load_so2_relative_qkv_batches,
)
from spatial_benchmark.so2_training_observability import (  # noqa: E402
    AtomicLatestCheckpointStore,
)


PREFLIGHT_SCHEMA = "so2_14core_relative_qkv_ddp4_preflight_v1"
PREFLIGHT_ALIASES = ("SO2-C22", "SO2-C23")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prior_c23_equivalence(
    paths: Any,
    graph_dir: Path,
    *,
    prior_receipt: Path,
) -> dict[str, Any]:
    prior_receipt = prior_receipt.expanduser().resolve(strict=True)
    try:
        prior = json.loads(prior_receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "The verified CAN-23 full/chunked and AMP/FP32 preflight is required."
        ) from exc
    gates = prior.get("gates") if isinstance(prior, dict) else None
    if (
        not isinstance(gates, dict)
        or prior.get("status") != "passed"
        or prior.get("all_required_gates_passed") is not True
        or not all(
            isinstance(gates.get(name), dict)
            and gates[name].get("passed") is True
            for name in ("amp_fp32_equivalence", "full_chunk_exactness")
        )
        or not isinstance(prior.get("largest_core"), dict)
        or prior["largest_core"].get("alias") != "CAN-23"
    ):
        raise RuntimeError("Prior CAN-23 preflight does not contain passing gates.")
    old_root = paths.data_root / (
        "processed/cancer_6core_relative_qkv_graphs_v1/cores/CAN-23"
    )
    new_root = graph_dir / "cores/SO2-C23"
    file_checks = {}
    for filename in ("edge_index.npy", "relative_geometry.npy"):
        old_sha = sha256_file(old_root / filename)
        new_sha = sha256_file(new_root / filename)
        file_checks[filename] = {
            "prior_sha256": old_sha,
            "so2_sha256": new_sha,
            "byte_identical": old_sha == new_sha,
        }
    if not all(record["byte_identical"] for record in file_checks.values()):
        raise RuntimeError("SO2-C23 graph/geometry is not byte-identical to CAN-23.")
    return {
        "prior_receipt": str(prior_receipt),
        "prior_receipt_sha256": sha256_file(prior_receipt),
        "prior_full_chunk_exactness": gates["full_chunk_exactness"],
        "prior_amp_fp32_equivalence": gates["amp_fp32_equivalence"],
        "files": file_checks,
        "verified": True,
    }


def _preflight_config(
    config: dict[str, Any],
    *,
    rank: int,
    local_rank: int,
) -> CohortRelativeQKVTrainingConfig:
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    return CohortRelativeQKVTrainingConfig(
        model_seed=MODEL_SEED,
        cohort_aliases=PREFLIGHT_ALIASES,
        segment_start_global_epoch=0,
        segment_end_global_epoch=1,
        cores_per_optimizer_update=2,
        mask_views_per_core=10,
        learning_rate=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
        gradient_clip_norm=float(trainer["gradient_clip_norm"]),
        huber_delta=float(trainer["huber_delta"]),
        mask_base_seed=int(masking["mask_base_seed"]),
        core_order_seed=int(trainer["core_order_seed"]),
        amp=bool(trainer["amp"]),
        amp_dtype=str(trainer["amp_dtype"]),
        deterministic=bool(trainer["deterministic"]),
        deterministic_warn_only=bool(trainer["deterministic_warn_only"]),
        device=f"cuda:{local_rank}",
        stage_complete_core_graph_on_device=True,
        staged_relative_geometry_dtype=str(
            trainer["staged_relative_geometry_dtype"]
        ),
        checkpoint_interval_global_epochs=1,
        distributed_world_size=WORLD_SIZE,
        distributed_rank=rank,
    )


def run_preflight(
    *,
    config: dict[str, Any],
    output: Path,
    prior_c23_receipt: Path,
    rank: int,
    local_rank: int,
) -> dict[str, Any] | None:
    paths = current_paths()
    _validate_contract(config)
    dataset = _section(config, "dataset")
    cohort_dir = _runtime_path(dataset["prepared_artifact"], paths)
    graph_dir = _runtime_path(dataset["prepared_graph_artifact"], paths)
    all_batches = load_so2_relative_qkv_batches(
        cohort_dir=cohort_dir, graph_dir=graph_dir
    )
    selected = tuple(batch for batch in all_batches if batch.alias in PREFLIGHT_ALIASES)
    if {batch.alias for batch in selected} != set(PREFLIGHT_ALIASES):
        raise RuntimeError("Preflight could not load its exact paired cores.")
    model = _model_from_config(
        config,
        num_genes=selected[0].n_genes,
        node_covariate_dim=int(selected[0].node_covariates.shape[1]),
    )
    construction = _model_construction(
        model,
        config,
        num_genes=selected[0].n_genes,
        node_covariate_dim=int(selected[0].node_covariates.shape[1]),
    )
    observed_peak: list[float] = []

    def observe_epoch(
        epoch: Any,
        cores: Any,
        updates: Any,
        duration: float,
        peak: float,
    ) -> None:
        del epoch, cores, updates, duration
        observed_peak.append(float(peak))

    result = fit_cohort_relative_qkv_segment(
        model,
        selected,
        _preflight_config(config, rank=rank, local_rank=local_rank),
        epoch_callback=observe_epoch if rank == 0 else None,
    )
    losses = np.asarray(
        [record.masked_huber_loss for record in result.core_history],
        dtype=np.float64,
    )
    gradients = np.asarray(
        [record.gradient_norm for record in result.optimizer_update_history],
        dtype=np.float64,
    )
    finite = bool(
        losses.size == 2
        and gradients.size == 1
        and np.isfinite(losses).all()
        and np.isfinite(gradients).all()
    )
    if not finite:
        raise RuntimeError("Four-rank preflight produced non-finite loss/gradient.")

    if rank == 0:
        if len(observed_peak) != 1 or observed_peak[0] <= 0:
            raise RuntimeError("Four-rank preflight did not record peak VRAM.")
        paths.state_root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".so2-ddp4-checkpoint-", dir=paths.state_root)
        )
        try:
            store = AtomicLatestCheckpointStore(temporary_root)
            receipt = store.save(
                _checkpoint_payload(
                    run_id="r_20260825T000000Z_00000000_s000_f00_a00_00000000",
                    config=config,
                    model_construction=construction,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    resume=result.resume,
                ),
                completed_global_epochs=1,
            )
            loaded = store.load_latest()
            clone = _model_from_config(
                config,
                num_genes=selected[0].n_genes,
                node_covariate_dim=int(selected[0].node_covariates.shape[1]),
            )
            clone.load_state_dict(loaded["model_state_dict"], strict=True)
            reload_verified = all(
                torch.equal(clone.state_dict()[name], tensor.detach().cpu())
                for name, tensor in model.state_dict().items()
            )
            if not reload_verified:
                raise RuntimeError("Four-rank preflight checkpoint reload drifted.")
            checkpoint_sha = receipt.sha256
        finally:
            shutil.rmtree(temporary_root)
        c23_equivalence = _prior_c23_equivalence(
            paths,
            graph_dir,
            prior_receipt=prior_c23_receipt,
        )
        content: dict[str, Any] = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed",
            "all_required_gates_passed": True,
            "completed_experiment": False,
            "diagnostic_only": True,
            "campaign_id": CAMPAIGN_ID,
            "created_at": _utc_now(),
            "resolved_config_sha256": _canonical_sha256(
                _preflight_bound_config(config)
            ),
            "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
            "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
            "selected_core_aliases": list(PREFLIGHT_ALIASES),
            "distributed_world_size": WORLD_SIZE,
            "distributed_backend": "nccl",
            "visible_devices": VISIBLE_DEVICES,
            "elastic_max_restarts": 0,
            "rank_assignments": [
                "core_a_views_0_4",
                "core_a_views_5_9",
                "core_b_views_0_4",
                "core_b_views_5_9",
            ],
            "optimizer_updates": result.optimizer_updates_completed,
            "complete_graph_mask_views": sum(
                record.n_mask_views for record in result.core_history
            ),
            "finite_loss_and_gradients": finite,
            "core_masked_huber": {
                record.alias: record.masked_huber_loss
                for record in result.core_history
            },
            "gradient_norm_before_clip": float(gradients[0]),
            "peak_vram_gib_all_ranks": observed_peak[0],
            "checkpoint_reload_verified": reload_verified,
            "temporary_checkpoint_sha256": checkpoint_sha,
            "so2_c23_prior_equivalence_verified": c23_equivalence["verified"],
            "so2_c23_prior_equivalence": c23_equivalence,
            "model": construction,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        content["receipt_content_sha256"] = _canonical_sha256(content)
        _atomic_json(output, content)
    else:
        content = None
    torch.distributed.barrier()
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prior-c23-receipt",
        type=Path,
        help=(
            "Verified prior CAN-23 full/chunked and AMP/FP32 preflight. "
            "Defaults to state/preflight/cancer_6core_relative_qkv_seed0.json."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, local_rank, _ = _distributed_identity()
    paths = current_paths()
    config = compose_config(args.config, config_root=paths.config_root)
    output = (
        paths.state_root / "preflight/so2_14core_relative_qkv_ddp4.json"
        if args.output is None
        else args.output.resolve()
    )
    prior_c23_receipt = (
        paths.state_root / "preflight/cancer_6core_relative_qkv_seed0.json"
        if args.prior_c23_receipt is None
        else args.prior_c23_receipt.resolve()
    )
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    try:
        receipt = run_preflight(
            config=config,
            output=output,
            prior_c23_receipt=prior_c23_receipt,
            rank=rank,
            local_rank=local_rank,
        )
        if rank == 0:
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

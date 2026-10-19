#!/usr/bin/env python3
"""Train fresh SO2 BCE with one-GPU, core-local target-cell optimizer batches.

Contract: experiments/campaigns/cmp_20261001_so2_binary_expression_target_minibatch/README.md.
The original BCE model, masks, validation, atomic checkpoint transaction,
resumption, stopping controls and final replay are reused. Only the training
epoch and separately identified execution contract change. Balanced contiguous
windows are shuffled within each shuffled core; these are not IID cross-core
minibatches. Clean inputs are staged once per core, while learned embeddings
are recomputed by every forward. No hidden target or patient metadata is added.

Invoke through the registered launcher, which supplies BAGM_RUN_ID,
BAGM_RUN_SCRATCH, one visible CUDA device, and torchrun --nproc-per-node=1.
CLI: --config <scratch/config.resolved.yaml> --run-scratch <scratch>
     [--resume-checkpoint <scratch/checkpoints/latest.ckpt>].

Verification from the source root::

    PYTHONPATH=src:. /venv/main/bin/python scripts/train/run_so2_target_minibatch_binary_expression_bce.py --help
    PYTHONPATH=src:. /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_target_minibatch.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

import numpy as np
import torch

from scripts.train import run_so2_target_isolated_nb as base
# The historical BCE module installs its contract at import time. Restore the
# shared base immediately so importing this new module is observational only;
# explicit ``install_contract`` is the sole entry into the minibatch protocol.
_base_before_import = dict(vars(base))
from scripts.train import run_so2_target_isolated_binary_expression_bce as bce
for _name in set(vars(base)) - set(_base_before_import):
    delattr(base, _name)
for _name, _value in _base_before_import.items():
    setattr(base, _name, _value)
del _base_before_import, _name, _value
from scripts.diagnostics.so2_hurdle_gpu_trial_worker import (
    GIB, ResourceSampler, append_json, host_snapshot, memory_gate, sha256_file,
)
from spatial_benchmark.target_minibatch import (
    expected_updates_per_epoch, target_window_plan, validate_target_windows,
    window_plan_receipt,
)
from spatial_benchmark.so2_binary_target_minibatch import validate_configuration

CAMPAIGN_ID = "cmp_20261001_so2_binary_expression_target_minibatch"
PROTOCOL = "donor_grouped_so2_target_minibatch_binary_expression_bce_earlystop_v1"
CHECKPOINT_SCHEMA = "so2_target_minibatch_binary_expression_bce_checkpoint_v1"
PREFLIGHT_SCHEMA = "so2_binary_target_minibatch_preflight_v1"
EPOCH_SCHEMA = "so2_target_minibatch_binary_expression_bce_epoch_metrics_v1"
PROVENANCE_PATH = "provenance/so2_target_minibatch_binary_expression_training.json"
EXTRA_EPOCH_COLUMNS = (
    "optimizer_updates", "cumulative_optimizer_updates", "target_minibatch_nominal_size",
    "minimum_target_batch_size", "maximum_target_batch_size", "peak_reserved_vram_gib",
    "minimum_sampled_physical_free_gib",
)
_CONFIG: dict[str, Any] = {}
_SCRATCH: Path | None = None
_SAMPLER: ResourceSampler | None = None
_DEVICE_TOTAL = 0
_MIN_CUDA_FREE: int | None = None
_MAX_CGROUP_MEMORY = 0
_EPOCH_RESOURCES: dict[int, dict[str, Any]] = {}
_CURRENT_EPOCH = 0
_INITIALIZATION_RECEIPT: dict[str, Any] = {}
_ORIGINAL_EVALUATE = bce._evaluate_validation


def require(value: bool, message: str) -> None:
    if not value:
        raise base.SO2NBTrainingError(message)


@dataclass(frozen=True)
class TargetWindowGradientContext(base.GradientDirectionUpdateContext):
    """One-core identity accepted by the unchanged scalar gradient trackers."""

    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        require(self.global_epoch >= 0, "Gradient epoch is negative")
        require(self.completed_global_epoch == self.global_epoch + 1, "Gradient epoch mismatch")
        require(self.optimizer_update_in_epoch >= 0 and self.cumulative_optimizer_update > 0,
                "Gradient update index is invalid")
        require(len(self.aliases) == 1 and bool(self.aliases[0]), "Minibatch gradients require one core")


def _progress(event: str, **values: Any) -> None:
    require(_SCRATCH is not None, "Runtime scratch is not initialized")
    payload = {"event": event, "time": time.time(), "run_id": os.environ["BAGM_RUN_ID"], **values}
    append_json(_SCRATCH / "metrics/target_minibatch_progress.jsonl", payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    global _MIN_CUDA_FREE, _MAX_CGROUP_MEMORY
    free, total = torch.cuda.mem_get_info(device)
    require(total == _DEVICE_TOTAL, "CUDA device capacity changed")
    _MIN_CUDA_FREE = free if _MIN_CUDA_FREE is None else min(_MIN_CUDA_FREE, free)
    sampled = _MIN_CUDA_FREE
    if _SAMPLER is not None and _SAMPLER.minimum_free_bytes is not None:
        sampled = min(sampled, _SAMPLER.minimum_free_bytes)
    snapshot = memory_gate(allocated=torch.cuda.max_memory_allocated(device),
                           reserved=torch.cuda.max_memory_reserved(device),
                           total=total, sampled_free=sampled)
    require(snapshot["resource_gate_passed"], "GPU peak/headroom resource gate failed")
    host = host_snapshot()
    _MAX_CGROUP_MEMORY = max(_MAX_CGROUP_MEMORY, int(host["cgroup"].get("memory.current", 0)))
    require(_MAX_CGROUP_MEMORY < 160 * GIB, "Host cgroup exceeded 160 GiB")
    require(_SAMPLER is None or _SAMPLER.error is None, f"Resource sampler error: {getattr(_SAMPLER, 'error', None)}")
    return {**snapshot, "host": host, "maximum_sampled_cgroup_memory_bytes": _MAX_CGROUP_MEMORY}


def _train_epoch(
    *, model: torch.nn.Module, training_model: Any, optimizer: Any, scaler: Any,
    bundle: Any, global_epoch: int, rank: int, device: torch.device,
    trainer: Mapping[str, Any], control_group: Any, global_tracker: Any,
    block_tracker: Any, theta_tracker: Any, theta_view: torch.nn.Module,
) -> tuple[Sequence[str], Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    """Exactly-once target windows, with one optimizer step per balanced window."""
    global _CURRENT_EPOCH
    require(rank == 0 and base.WORLD_SIZE == 1, "Target minibatches require one rank")
    completed = global_epoch + 1
    _CURRENT_EPOCH = completed
    batch_size = int(trainer["target_cells_per_optimizer_update"])
    order_seed = int(trainer["target_window_order_seed"])
    expected = expected_updates_per_epoch([b.n_nodes for b in bundle.training_batches], batch_size)
    require(expected == base.OPTIMIZER_UPDATES_PER_EPOCH, "Optimizer-update configuration differs from cohort")
    ordered = base.cohort_relative_qkv_core_order(completed, aliases=base.TRAINING_ALIASES,
                                                core_order_seed=base.CORE_ORDER_SEED)
    require(len(ordered) == 12 and set(ordered) == set(base.TRAINING_ALIASES), "Incomplete core order")
    local: dict[str, Any] = {}
    coverage: list[Mapping[str, Any]] = []
    update = 0
    batch_sizes: list[int] = []
    started = time.monotonic()
    training_model.train()
    _progress("epoch_training_started", epoch=completed, optimizer_updates_expected=expected)
    try:
        for alias in ordered:
            batch = bundle.batches_by_alias[alias]
            windows = target_window_plan(batch.n_nodes, batch_size, seed=order_seed,
                                         core_alias=alias, global_epoch=completed)
            validate_target_windows(windows, batch.n_nodes)
            plan = window_plan_receipt(batch.n_nodes, batch_size, seed=order_seed,
                                       core_alias=alias, global_epoch=completed)
            # This realization is keyed only by epoch/core/canonical target row.
            # Window order and optimizer batch size cannot change its masks.
            partition, realization = base._partition_and_mask(batch, rank=0, global_epoch=completed)
            visits = np.zeros(batch.n_nodes, dtype=np.uint8)
            base._clear_graph_layout_caches(training_model)
            clean_input, raw_target, covariates = base._stage_core(batch, device=device)
            loss_sum = 0.0; entry_count = 0; positive_count = 0
            _progress("core_started", epoch=completed, core_alias=alias, windows=len(windows))
            for start, stop in windows:
                require(not bool(visits[start:stop].any()), "Duplicate target visit")
                optimizer.zero_grad(set_to_none=True)
                target_input, mask = base._make_target_tensors(
                    clean_input=clean_input,
                    realization=SimpleNamespace(mask=realization.mask[start:stop]),
                    target_start=start, target_stop=stop, device=device,
                )
                base._seed_model_step(base._model_step_seed(completed, update, alias, rank), device)
                output = bce._forward_target_shard(
                    training_model=training_model, batch=batch, clean_input=clean_input,
                    covariates=covariates, target_input=target_input, mask=mask,
                    target_start=start, target_stop=stop, trainer=trainer,
                )
                sums = bce.masked_binary_expression_bce_sums(output.nonzero_logits, raw_target[start:stop], mask)
                require(sums.masked_entries > 0, "Empty target-window objective")
                objective = sums.detection_nll_sum / float(sums.masked_entries)
                require(bool(torch.isfinite(objective)), "Nonfinite minibatch BCE")
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                base._assert_finite_gradients(model)
                context = TargetWindowGradientContext(
                    global_epoch=global_epoch, completed_global_epoch=completed,
                    optimizer_update_in_epoch=update,
                    cumulative_optimizer_update=global_epoch * expected + update + 1,
                    aliases=(alias,),
                )
                global_tracker.observe(model, context)
                block_tracker.observe(model, context)
                theta_tracker.observe(theta_view, context)
                # Initial zero detection weights block upstream gradients only
                # at the very first update, as in the source BCE initialization.
                if context.cumulative_optimizer_update > 1:
                    for block in model.blocks:
                        require(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in block.parameters()),
                                "Graph block has zero gradients after the first update")
                norm = torch.nn.utils.clip_grad_norm_(training_model.parameters(), float(trainer["gradient_clip_norm"]),
                                                      error_if_nonfinite=True)
                require(bool(torch.isfinite(norm)) and float(norm) > 0, "Invalid full-model gradient norm")
                scaler.step(optimizer); scaler.update()
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameters")
                loss_sum += float(sums.detection_nll_sum.detach().double().cpu())
                entry_count += int(sums.masked_entries)
                positive_count += int(sums.positive_entries)
                visits[start:stop] += 1
                batch_sizes.append(stop - start)
                update += 1
                del output, sums, objective, target_input, mask
                # Keep only immutable inputs and CPU graph layouts across steps.
                # Model.forward recomputes every learned clean-source embedding.
                torch.cuda.synchronize(device)
                resources = _resource_snapshot(device)
                _progress("target_update_completed", epoch=completed, update=update,
                          cumulative_optimizer_update=global_epoch * expected + update,
                          core_alias=alias, target_start=start, target_stop=stop,
                          target_cells=stop-start, gradient_norm=float(norm),
                          elapsed_training_seconds=time.monotonic()-started,
                          peak_allocated_bytes=resources["peak_allocated_bytes"],
                          peak_reserved_bytes=resources["peak_reserved_bytes"],
                          headroom_bytes=resources["headroom_bytes"])
            require(bool(np.all(visits == 1)), "Core targets were not visited exactly once")
            require(entry_count == realization.n_masked_entries, "Training masks differ from full-core realization")
            local[alias] = (loss_sum, entry_count, positive_count, batch.n_nodes)
            receipt = dict(base._coverage_receipt(batch=batch, role="training", global_epoch=completed,
                gathered=[base._local_receipt_record(rank=0, batch=batch, partition=partition, realization=realization)]))
            receipt.pop("receipt_sha256")
            receipt.update(target_window_plan=plan, actual_optimizer_updates=len(windows),
                           target_visit_count_min=int(visits.min()), target_visit_count_max=int(visits.max()))
            receipt["receipt_sha256"] = base._canonical_sha256(receipt)
            coverage.append(receipt)
            del clean_input, raw_target, covariates, realization
            base._clear_graph_layout_caches(training_model)
            torch.cuda.empty_cache()
            _progress("core_completed", epoch=completed, core_alias=alias, updates_completed=update)
        require(update == expected, "Incomplete optimizer-update coverage")
        _EPOCH_RESOURCES[completed] = {
            **_resource_snapshot(device), "epoch": completed,
            "optimizer_updates": update, "cumulative_optimizer_updates": global_epoch * expected + update,
            "target_minibatch_nominal_size": batch_size,
            "minimum_target_batch_size": min(batch_sizes), "maximum_target_batch_size": max(batch_sizes),
            "training_duration_seconds": time.monotonic()-started, "coverage_verified": True,
        }
        return ordered, local, coverage
    finally:
        base._clear_graph_layout_caches(training_model)


def _evaluate_validation(**kwargs: Any) -> Any:
    started = time.monotonic()
    _progress("validation_started", epoch=_CURRENT_EPOCH)
    result = _ORIGINAL_EVALUATE(**kwargs)
    resources = _resource_snapshot(kwargs["device"])
    if _CURRENT_EPOCH in _EPOCH_RESOURCES:
        _EPOCH_RESOURCES[_CURRENT_EPOCH].update(resources)
        _EPOCH_RESOURCES[_CURRENT_EPOCH]["validation_duration_seconds"] = time.monotonic()-started
    _progress("validation_completed", epoch=_CURRENT_EPOCH, validation_bce=result.pooled_binary_cross_entropy)
    return result


def _make_epoch_row(**kwargs: Any) -> Mapping[str, Any]:
    row = dict(bce._make_epoch_row(**kwargs))
    epoch = int(kwargs["completed_epoch"])
    resources = _EPOCH_RESOURCES[epoch]
    row.update(schema=EPOCH_SCHEMA, **{k: resources[k] for k in EXTRA_EPOCH_COLUMNS[:5]},
               peak_reserved_vram_gib=resources["peak_reserved_bytes"] / GIB,
               minimum_sampled_physical_free_gib=resources["minimum_sampled_physical_free_bytes"] / GIB)
    require(_SCRATCH is not None, "Runtime scratch is not initialized")
    append_json(_SCRATCH / "metrics/target_minibatch_resources.jsonl", resources)
    return row


def _additional_provenance(validation: Any) -> Mapping[str, Any]:
    return {**bce._additional_provenance(validation),
            "target_partitioning": "balanced_contiguous_windows_shuffled_within_core_single_rank",
            "paired_update_sync": "not_applicable_one_target_window_per_optimizer_update",
            "objective_aggregation": "masked_entry_mean_per_target_window_then_entry_weighted_epoch",
            "target_window_order_seed": _CONFIG["trainer"]["target_window_order_seed"],
            "target_cells_per_optimizer_update": _CONFIG["trainer"]["target_cells_per_optimizer_update"],
            "optimizer_updates_per_epoch": base.OPTIMIZER_UPDATES_PER_EPOCH,
            "clean_learned_embeddings_recomputed_every_update": True,
            "independent_cross_core_minibatches": False,
            "source_run_id": "r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863",
            "execution_profile": _CONFIG["metadata"]["execution_profile"]}


def _load_preflight_receipt(config: Mapping[str, Any], paths: Any, bundle: Any, model: Any) -> Mapping[str, Any]:
    global _SCRATCH, _SAMPLER, _DEVICE_TOTAL, _MIN_CUDA_FREE, _INITIALIZATION_RECEIPT
    _SCRATCH = Path(os.environ["BAGM_RUN_SCRATCH"]).resolve(strict=True)
    base._require_disk_headroom(_SCRATCH)
    require(tuple(b.alias for b in bundle.training_batches) == base.TRAINING_ALIASES, "Training cohort changed")
    require(sum(p.numel() for p in model.parameters()) == bce.EXPECTED_PARAMETER_COUNT, "Model size changed")
    _MIN_CUDA_FREE, _DEVICE_TOTAL = torch.cuda.mem_get_info()
    require(_MIN_CUDA_FREE >= 2 * GIB, "Initial GPU headroom is insufficient")
    threads = int(config["launcher"]["cpu_threads"])
    torch.set_num_threads(threads)
    scope = set(bce.PREFLIGHT_CODE_RELATIVE_PATHS) | {
        Path("scripts/train/run_so2_target_minibatch_binary_expression_bce.py"),
        Path("src/spatial_benchmark/target_minibatch.py"),
        Path("src/spatial_benchmark/so2_binary_target_minibatch.py"),
        Path("scripts/diagnostics/so2_hurdle_gpu_trial_worker.py"),
    }
    code_hashes = {str(p): sha256_file(paths.project_root / p) for p in sorted(scope)}
    pilot = config["metadata"].get("pilot_preflight")
    if config["metadata"]["execution_profile"] == "full":
        require(isinstance(pilot, Mapping), "Full training requires a verified selected pilot")
        path = Path(pilot["receipt_path"])
        require(sha256_file(path) == pilot["receipt_sha256"], "Pilot receipt checksum differs")
        receipt = json.loads(path.read_text())
        require(receipt.get("schema") == PREFLIGHT_SCHEMA and receipt.get("passed") is True, "Pilot preflight failed")
        require(receipt.get("target_cells_per_optimizer_update") == config["trainer"]["target_cells_per_optimizer_update"],
                "Pilot batch size differs")
        require(receipt.get("source_configuration_sha256") == base._canonical_sha256(config["source_configuration"]),
                "Pilot scientific source differs")
        require(receipt.get("frozen_task_contract_sha256") == config["metadata"]["frozen_task_contract_sha256"],
                "Pilot frozen task contract differs")
        require(receipt.get("code_hashes") == code_hashes, "Pilot source code differs from full-run implementation")
    initial = {"schema": "so2_binary_target_minibatch_initialization_v1", "protocol": PROTOCOL,
               "configuration_sha256": base._canonical_sha256(config), "code_hashes": code_hashes,
               "frozen_task_contract_sha256": config["metadata"]["frozen_task_contract_sha256"],
               "source_configuration_sha256": base._canonical_sha256(config["source_configuration"]),
               "pilot_preflight": pilot, "world_size": 1, "parameter_count": bce.EXPECTED_PARAMETER_COUNT,
               "original_four_rank_preflight_claimed": False}
    initial["receipt_content_sha256"] = base._canonical_sha256(initial)
    _INITIALIZATION_RECEIPT = initial
    initial_path = _SCRATCH / "diagnostics/target_minibatch_initialization.json"
    if initial_path.exists():
        require(json.loads(initial_path.read_text()) == initial, "Initialization/code/config differs from restart receipt")
    else:
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_text(json.dumps(initial, sort_keys=True, indent=2)+"\n")
    append_json(_SCRATCH / "diagnostics/target_minibatch_environment.jsonl", {
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "visible_physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "world_size": 1,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "initial_cuda_free_bytes": _MIN_CUDA_FREE, "total_device_bytes": _DEVICE_TOTAL,
        "initial_host": host_snapshot(), "time": time.time(),
    })
    _SAMPLER = ResourceSampler(_SCRATCH / "diagnostics/target_minibatch_resource_samples.jsonl",
                               os.environ["CUDA_VISIBLE_DEVICES"])
    _SAMPLER.start()
    return initial


def _write_additional_final_artifacts(archive: Any, validation: Any) -> None:
    bce._write_additional_final_artifacts(archive, validation)
    if _SAMPLER is not None:
        _SAMPLER.stop()
    resources = _resource_snapshot(torch.device("cuda:0"))
    require(_SCRATCH is not None, "Runtime scratch is not initialized")
    # Resume can leave interrupted post-checkpoint telemetry. Select the last
    # committed record for each completed epoch, while preserving the raw log.
    by_epoch = {}
    for line in (_SCRATCH / "metrics/target_minibatch_resources.jsonl").read_text().splitlines():
        row = json.loads(line); by_epoch[int(row["epoch"])] = row
    epochs = [by_epoch[k] for k in sorted(by_epoch)]
    require(epochs and sorted(by_epoch) == list(range(1, max(by_epoch)+1)), "Missing resource epoch history")
    minimum_free = min(resources["minimum_sampled_physical_free_bytes"],
                       *(r["minimum_sampled_physical_free_bytes"] for r in epochs))
    gates = memory_gate(allocated=max(resources["peak_allocated_bytes"], *(r["peak_allocated_bytes"] for r in epochs)),
                        reserved=max(resources["peak_reserved_bytes"], *(r["peak_reserved_bytes"] for r in epochs)), total=_DEVICE_TOTAL,
                        sampled_free=minimum_free)
    samples = [json.loads(line) for line in
               (_SCRATCH / "diagnostics/target_minibatch_resource_samples.jsonl").read_text().splitlines()]
    require(any("physical_memory_free_bytes" in row for row in samples), "No sampled physical GPU memory evidence")
    peak_cgroup = max(_MAX_CGROUP_MEMORY, *(int(row["cgroup"]["memory.current"]) for row in samples),
                      *(r["maximum_sampled_cgroup_memory_bytes"] for r in epochs))
    gates["maximum_sampled_cgroup_memory_bytes"] = peak_cgroup
    require(peak_cgroup < 160 * GIB, "Sampled host cgroup exceeded 160 GiB")
    require(gates["resource_gate_passed"], "Final resource gate failed")
    archive.write_json("diagnostics/target_minibatch_preflight.json", {
        "schema": PREFLIGHT_SCHEMA, "passed": True, "run_id": os.environ["BAGM_RUN_ID"],
        "source_configuration_sha256": base._canonical_sha256(_CONFIG["source_configuration"]),
        "target_cells_per_optimizer_update": _CONFIG["trainer"]["target_cells_per_optimizer_update"],
        "optimizer_updates_per_global_epoch": base.OPTIMIZER_UPDATES_PER_EPOCH,
        "completed_epochs": len(epochs), "coverage_verified": True, "finite_gradients": True,
        "resource_gates": gates, "epoch_resources": epochs,
        "profile": _CONFIG["metadata"]["execution_profile"],
        "frozen_task_contract_sha256": _CONFIG["metadata"]["frozen_task_contract_sha256"],
        "code_hashes": _INITIALIZATION_RECEIPT["code_hashes"],
        "physical_memory_sampling_seconds": 5, "continuous_physical_peak_observed": False,
    })


def install_contract(config: Mapping[str, Any]) -> None:
    global _CONFIG
    validate_configuration(config)
    _CONFIG = dict(config)
    bce._install_binary_expression_contract()
    trainer = config["trainer"]
    for key, value in {
        "CAMPAIGN_ID": CAMPAIGN_ID, "PROTOCOL": PROTOCOL, "EVALUATION_PROTOCOL": PROTOCOL,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA, "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA,
        "WORLD_SIZE": 1, "VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "OPTIMIZER_UPDATES_PER_EPOCH": int(trainer["optimizer_updates_per_global_epoch"]),
        "MINIMUM_EPOCHS": int(trainer["minimum_global_epochs"]), "MAXIMUM_EPOCHS": int(trainer["max_epochs"]),
        "FROZEN_TASK_CONTRACT_SHA256": config["metadata"]["frozen_task_contract_sha256"],
        "PROVENANCE_RELATIVE_PATH": PROVENANCE_PATH,
        "EPOCH_COLUMNS": bce.EPOCH_COLUMNS + EXTRA_EPOCH_COLUMNS,
        "EPOCH_METRICS_SCHEMA": EPOCH_SCHEMA,
        "ADDITIONAL_REQUIRED_FINAL_OUTPUTS": bce.ADDITIONAL_REQUIRED_FINAL_OUTPUTS + ("diagnostics/target_minibatch_preflight.json",),
        "validate_target_isolated_config": validate_configuration,
        "_train_epoch": _train_epoch, "_load_preflight_receipt": _load_preflight_receipt,
        "_evaluate_validation": _evaluate_validation, "_make_epoch_row": _make_epoch_row,
        "_additional_provenance": _additional_provenance,
        "_write_additional_final_artifacts": _write_additional_final_artifacts,
    }.items():
        setattr(base, key, value)


def main(argv: list[str] | None = None) -> int:
    parser = base.build_parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    config = base.load_yaml_mapping(args.config)
    install_contract(config)
    torch.set_num_threads(int(config["launcher"]["cpu_threads"]))
    torch.set_num_interop_threads(1)
    try:
        return base.main(argv)
    finally:
        if _SAMPLER is not None:
            _SAMPLER.stop()


if __name__ == "__main__":
    raise SystemExit(main())

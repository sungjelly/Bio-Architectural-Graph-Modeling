"""Guarded additive configuration for fixed-budget SO2 BCE optimization audits."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping

from .identifiers import canonical_sha256
from .so2_binary_target_minibatch import (
    SOURCE_CONFIG_SHA, SOURCE_RUN, SOURCE_SUCCESS_SHA, TARGET_BATCH_SIZES,
    build_configuration as minibatch_configuration, optimizer_updates,
)

CAMPAIGN = "cmp_20261001_so2_bce_optimization_audit"
EVALUATION_PROTOCOL = "donor_grouped_so2_bce_optimization_audit_v1"
PRIMARY_METRIC = "val/target_only_masked_binary_cross_entropy"
PREFLIGHT_RELATIVE_PATH = "diagnostics/optimization_audit_preflight.json"
PREFLIGHT_SCHEMA = "so2_bce_optimization_audit_preflight_v1"
PROFILE_EPOCHS = {"smoke": 2, "screen": 8, "confirmation": 32}
GRID_LEARNING_RATES = (3e-5, 1e-4, 3e-4)
BOUNDARY_LEARNING_RATES = (1e-5, 1e-3)
DROPOUT_VALUES = (0.0, 0.1, 0.3)
WEIGHT_DECAYS = (1e-5, 0.01)


def build_configuration(source: Mapping[str, Any], source_bundle: Mapping[str, Any], *,
                        profile: str, batch_mode: str, batch_size: int | None, learning_rate: float,
                        dropout: float, weight_decay: float, seed: int, gpu: str,
                        task_contract_sha256: str, smoke_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = minibatch_configuration(source, source_bundle, gpu=gpu, profile="pilot",
        batch_size=batch_size or 1024, task_contract_sha256=task_contract_sha256)
    maximum = PROFILE_EPOCHS[profile]
    variant = f"{batch_mode}_b{batch_size or 'pair'}_lr{learning_rate:g}_drop{dropout:g}_wd{weight_decay:g}"
    if profile == "smoke":
        axis = "runtime_correctness_smoke"
    elif profile == "confirmation":
        axis = "matched_seed_confirmation"
    elif batch_mode == "core_pair":
        axis = "paired_core_reference"
    elif learning_rate in BOUNDARY_LEARNING_RATES:
        axis = "learning_rate_boundary"
    elif dropout != 0.1:
        axis = "dropout_control"
    elif weight_decay != 1e-5:
        axis = "weight_decay_control"
    else:
        axis = "batch_learning_rate_grid"
    config["seed"] = seed
    config["campaign"] = {"campaign_id": CAMPAIGN, "display_name": "SO2 BCE optimization audit"}
    config["classification"].update({
        "lifecycle_stage": "diagnostic" if profile == "smoke" else "exploratory_screen",
        "study_axis": axis, "scientific_variant": variant,
        "retention_class": "retain_optimization_audit_evidence",
    })
    config["experiment"] = {
        "variant_label": variant, "estimand": "target_isolated_partial_gene_binary_expression_reconstruction",
        "permitted_claim": "runtime_correctness_only" if profile == "smoke" else "exploratory_fixed_split_optimization_comparison",
        "historical_comparator_run_id": SOURCE_RUN,
        "fixed_budget_epochs": maximum,
        "screening_endpoint_epochs": [2] if profile == "smoke" else ([7, 8] if profile == "screen" else [29, 30, 31, 32]),
        "screening_endpoint_aggregation": "arithmetic_mean_validation_bce_over_declared_endpoint_epochs",
        "best_checkpoint_metric_is_not_screening_selection_policy": True,
    }
    config["model"]["dropout"] = dropout
    config["evaluation"].update({"protocol": EVALUATION_PROTOCOL, "active_model_seeds": [seed]})
    metadata = config["metadata"]
    metadata.pop("pilot_preflight")
    metadata.update({"execution_profile": profile,
        "source_scientific_configuration_sha256": SOURCE_CONFIG_SHA,
        "smoke_preflight": deepcopy(dict(smoke_preflight)) if smoke_preflight else None,
        "evaluate_fixed_training_at_best": profile == "confirmation",
        "diagnostic_train_prior_reference_required": True,
        "comparison_axis": axis})
    metadata["preflight_acceptance"]["minimum_free_disk_gib"] = 5.0
    config["launcher"].update({"disk_safety_min_free_gb": 5,
        "hardware_preflight_receipt": (smoke_preflight or {}).get("receipt_path")})
    trainer = config["trainer"]
    trainer.update({"optimizer_batch_mode": batch_mode, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "max_epochs": maximum, "minimum_global_epochs": 1,
        "early_stopping_patience": maximum + 1, "fixed_epoch_budget": True})
    if batch_mode == "core_pair":
        for name in ("optimizer_steps_per_minibatch", "optimizer_zero_grad_per_minibatch",
                     "target_window_partition", "target_window_order", "target_window_order_seed"):
            trainer.pop(name, None)
        trainer.update({"batch_size": 2,
            "batch_unit": "two_complete_core_target_sets_per_optimizer_update",
            "target_cells_per_optimizer_update": None, "cores_per_optimizer_update": 2,
            "cores_staged_sequentially_per_optimizer_update": True,
            "optimizer_updates_per_global_epoch": 6,
            "optimizer_steps_per_paired_core_update": 1, "optimizer_zero_grad_per_paired_core_update": 1,
            "distributed_rank_assignment": ["rank0_all_targets_of_each_paired_core"],
            "ddp_sync_policy": "first_core_backward_under_no_sync_second_core_backward_synced",
            "graph_execution": "single_rank_ddp_two_sequential_complete_core_target_sets",
            "objective_denominator": "globally_reduced_masked_target_entry_count_per_update",
            "local_ddp_loss_scale": "world_size_times_local_bce_sum_divided_by_global_masked_entry_count",
            "objective_aggregation": "sum_masked_entry_bce_across_two_cores_divided_by_pooled_masked_entry_count"})
    return config


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_configuration(config: Mapping[str, Any]) -> None:
    source, bundle = config.get("source_configuration"), config.get("source_bundle")
    if not isinstance(source, Mapping) or canonical_sha256(source) != SOURCE_CONFIG_SHA:
        raise ValueError("Optimization audit requires the exact completed BCE source configuration.")
    if not isinstance(bundle, Mapping) or bundle.get("run_id") != SOURCE_RUN or bundle.get("success_marker_sha256") != SOURCE_SUCCESS_SHA:
        raise ValueError("Optimization audit requires its pinned successful source run.")
    for name in ("config_file_sha256", "manifest_sha256", "checksum_manifest_sha256", "data_receipt_sha256", "split_receipt_sha256"):
        if not _sha(bundle.get(name)):
            raise ValueError(f"Optimization audit requires source {name}.")
    metadata, trainer, model, launcher = (config.get(name, {}) for name in ("metadata", "trainer", "model", "launcher"))
    if not all(isinstance(value, Mapping) for value in (metadata, trainer, model, launcher)):
        raise ValueError("Optimization audit configuration groups must be mappings.")
    profile, mode, batch = metadata.get("execution_profile"), trainer.get("optimizer_batch_mode"), trainer.get("target_cells_per_optimizer_update")
    lr, dropout, wd, seed, gpu = trainer.get("learning_rate"), model.get("dropout"), trainer.get("weight_decay"), config.get("seed"), launcher.get("requested_gpu")
    if profile not in PROFILE_EPOCHS or mode not in {"target_window", "core_pair"} or gpu not in {"0", "1", "2", "3"}:
        raise ValueError("Optimization audit requires a declared profile, batch mode and one physical GPU.")
    if (mode == "target_window" and (not isinstance(batch, int) or batch not in TARGET_BATCH_SIZES)) or (mode == "core_pair" and batch is not None):
        raise ValueError("Optimization audit requires B=512/1024/2048 for windows and null B for paired cores.")
    if lr not in GRID_LEARNING_RATES + BOUNDARY_LEARNING_RATES or dropout not in DROPOUT_VALUES or wd not in WEIGHT_DECAYS:
        raise ValueError("Optimization audit hyperparameters are outside its declared discrete axes.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {0, 1} or (profile != "confirmation" and seed != 0):
        raise ValueError("Only confirmation allows model seed1; all other profiles require seed0.")
    if dropout != 0.1 and wd != 1e-5:
        raise ValueError("Regularization controls vary one factor at a time.")
    if lr in BOUNDARY_LEARNING_RATES and (dropout != 0.1 or wd != 1e-5):
        raise ValueError("Boundary learning-rate controls retain incumbent regularization.")
    if mode == "core_pair" and (lr != 1e-4 or dropout != 0.1 or wd != 1e-5 or seed != 0 or profile == "confirmation"):
        raise ValueError("The paired-core reference is restricted to its declared seed0 incumbent settings.")
    if profile == "smoke" and (lr != 1e-4 or dropout != 0.1 or wd != 1e-5 or (mode == "target_window" and batch != 1024)):
        raise ValueError("The two new smoke profiles use incumbent hyperparameters.")
    contract_sha, smoke = metadata.get("frozen_task_contract_sha256"), metadata.get("smoke_preflight")
    if not _sha(contract_sha):
        raise ValueError("Optimization audit requires a frozen task contract checksum.")
    if profile == "smoke":
        if smoke is not None:
            raise ValueError("A new smoke cannot claim an earlier preflight receipt.")
    else:
        if not isinstance(smoke, Mapping) or not str(smoke.get("run_id", "")).startswith("r_"):
            raise ValueError("Screen and confirmation require a verified new-protocol smoke.")
        if (smoke.get("source_configuration_sha256") != SOURCE_CONFIG_SHA
                or smoke.get("optimizer_batch_mode") != mode
                or not str(smoke.get("receipt_path", "")).endswith("/" + PREFLIGHT_RELATIVE_PATH)):
            raise ValueError("Optimization smoke source, mode or receipt is incompatible.")
        if any(not _sha(smoke.get(name)) for name in ("receipt_sha256", "success_marker_sha256", "checkpoint_sha256")):
            raise ValueError("Optimization smoke requires immutable receipt/checkpoint checksums.")
    expected = build_configuration(source, bundle, profile=profile, batch_mode=mode, batch_size=batch,
        learning_rate=lr, dropout=dropout, weight_decay=wd, seed=seed, gpu=gpu,
        task_contract_sha256=contract_sha, smoke_preflight=smoke)
    attempt = config.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Optimization audit requires a positive execution attempt.")
    expected["attempt"] = attempt
    if config != expected:
        raise ValueError("Optimization audit configuration differs from its declared controlled profile.")


def command_for_configuration(config: Mapping[str, Any], project_root: Path) -> list[str]:
    validate_configuration(config)
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=1", "--max-restarts=0",
            str(project_root / "scripts/train/run_so2_bce_optimization_audit.py"),
            "--config", "{run_scratch}/config.resolved.yaml", "--run-scratch", "{run_scratch}"]

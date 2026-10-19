"""Explicit configuration contract for the additive SO2 BCE minibatch study."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .identifiers import canonical_sha256

CAMPAIGN = "cmp_20261001_so2_binary_expression_target_minibatch"
EVALUATION_PROTOCOL = "donor_grouped_so2_target_minibatch_binary_expression_bce_earlystop_v1"
SOURCE_RUN = "r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863"
SOURCE_CONFIG_SHA = "8b18f2323825a4329da96aab817977946e2abd0bbb186b370c16665086afd66d"
SOURCE_SUCCESS_SHA = "e53af18c271cfb65cd2219be5b94671d2194c6bd911d1ffe0b2e5ecb4f52de72"
PRIMARY_METRIC = "val/target_only_masked_binary_cross_entropy"
TARGET_BATCH_SIZES = (512, 1024, 2048)
TRAINING_CORE_CELLS = (38145, 16229, 9041, 11696, 19119, 12155,
                       4897, 11443, 43462, 14506, 13147, 14856)
PREFLIGHT_RELATIVE_PATH = "diagnostics/target_minibatch_preflight.json"


def optimizer_updates(batch_size: int) -> int:
    return sum((count + batch_size - 1) // batch_size for count in TRAINING_CORE_CELLS)


def is_target_minibatch(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    return isinstance(evaluation, Mapping) and evaluation.get("protocol") == EVALUATION_PROTOCOL


def build_configuration(source: Mapping[str, Any], source_bundle: Mapping[str, Any], *,
                        gpu: str, profile: str, batch_size: int,
                        task_contract_sha256: str,
                        pilot_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(dict(source))
    is_pilot = profile == "pilot"
    stage = "diagnostic" if is_pilot else "exploratory_screen"
    variant = f"so2_binary_expression_core_local_target_batch{batch_size}"
    config.update({
        "campaign": {"campaign_id": CAMPAIGN, "display_name": "SO2 BCE target-cell optimizer minibatches"},
        "classification": {"schema_version": 1, "lifecycle_stage": stage,
            "study_axis": "target_cell_optimizer_minibatch", "scientific_variant": variant,
            "retention_class": "retain_diagnostic_minibatch_pilot" if is_pilot else "retain_exploratory_validation_selected_bce_model",
            "classification_confidence": "high"},
        "experiment": {"variant_label": variant,
            "estimand": "target_isolated_partial_gene_binary_expression_reconstruction",
            "permitted_claim": "runtime_and_numerical_feasibility_only" if is_pilot else "exploratory_single_seed_validation_selected_optimization_comparison",
            "historical_comparator_run_id": SOURCE_RUN},
        "source_configuration": deepcopy(dict(source)), "source_bundle": deepcopy(dict(source_bundle)),
        "metadata": {
            "execution_profile": profile, "fresh_initialization": True,
            "frozen_task_contract_sha256": task_contract_sha256,
            "source_scientific_configuration_sha256": SOURCE_CONFIG_SHA,
            "no_test_split_or_artifacts": True,
            "gradient_diagnostics": deepcopy(source["metadata"]["gradient_diagnostics"]),
            "preflight_acceptance": {"minimum_free_disk_gib": 20.0,
                "minimum_vram_headroom_gib_each_rank": 2.0,
                "peak_vram_gib_all_ranks_max": 22.0, "maximum_cgroup_memory_gib": 160.0},
            "pilot_preflight": deepcopy(dict(pilot_preflight)) if pilot_preflight else None,
            "epochs_are_not_matched_optimizer_step_budgets": True,
        },
        "launcher": {"kind": "local_subprocess", "requested_gpu": str(gpu),
            "requested_gpu_count": 1, "process_count": 1, "distributed": True,
            "distributed_backend": "nccl", "require_exact_visible_devices": str(gpu),
            "set_cuda_visible_devices": True, "elastic_max_restarts": 0,
            "disk_safety_min_free_gb": 20, "heartbeat_seconds": 15,
            "stale_after_seconds": 900, "cpu_threads": 4,
            "hardware_preflight_receipt": None if is_pilot else (pilot_preflight or {}).get("receipt_path"),
            "active_output_root": "scratch/active_runs", "finalized_output_root": "artifacts/runs",
            "capture_stdout": True, "capture_stderr": True},
    })
    evaluation = config["evaluation"]
    evaluation.update({"protocol": EVALUATION_PROTOCOL, "artifact_contract": "predictive",
                       "canonical_prediction_split": "validation", "primary_metric": PRIMARY_METRIC,
                       "matched_protocol_comparator_campaign_id": source["campaign"]["campaign_id"]})
    trainer = config["trainer"]
    for name in ("optimizer_steps_per_paired_core_update", "optimizer_zero_grad_per_paired_core_update",
                 "paired_comparator_core_order_and_shard_seeds_reused_exactly", "target_shard_seed"):
        trainer.pop(name, None)
    trainer.update({
        "batch_size": batch_size, "batch_unit": "core_local_contiguous_target_window_per_optimizer_update",
        "target_cells_per_optimizer_update": batch_size,
        "target_window_partition": "balanced_contiguous_ceil_n_over_b",
        "target_window_order_seed": 2026100101,
        "target_window_order": "seeded_permutation_within_each_seeded_shuffled_core",
        "cores_per_optimizer_update": 1, "cores_staged_sequentially_per_optimizer_update": False,
        "optimizer_updates_per_global_epoch": optimizer_updates(batch_size),
        "optimizer_steps_per_minibatch": 1, "optimizer_zero_grad_per_minibatch": 1,
        "target_shards_per_core": 1, "distributed_world_size": 1,
        "distributed_rank_assignment": ["rank0_all_core_local_target_windows"],
        "ddp_sync_policy": "one_target_window_backward_per_optimizer_update",
        "graph_execution": "single_rank_ddp_core_local_target_minibatches",
        "model_step_rng_derivation": "model_seed_global_epoch_update_core_alias_target_shard",
        "objective_denominator": "masked_target_entry_count_per_optimizer_minibatch",
        "local_ddp_loss_scale": "local_bce_sum_divided_by_minibatch_masked_entry_count",
        "objective_aggregation": "masked_entry_mean_per_target_window_then_entry_weighted_epoch",
        "learned_neighbor_embeddings_recomputed_after_each_optimizer_step": True,
        "max_epochs": 2 if is_pilot else 300,
        "minimum_global_epochs": 1 if is_pilot else 50,
        "fixed_epoch_budget": is_pilot,
    })
    return config


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_configuration(config: Mapping[str, Any]) -> None:
    source, bundle = config.get("source_configuration"), config.get("source_bundle")
    if not isinstance(source, Mapping) or canonical_sha256(source) != SOURCE_CONFIG_SHA:
        raise ValueError("Target-minibatch BCE requires the exact frozen source scientific configuration.")
    if not isinstance(bundle, Mapping) or bundle.get("run_id") != SOURCE_RUN or bundle.get("success_marker_sha256") != SOURCE_SUCCESS_SHA:
        raise ValueError("Target-minibatch BCE requires its pinned completed BCE source bundle.")
    for name in ("config_file_sha256", "manifest_sha256", "checksum_manifest_sha256", "data_receipt_sha256", "split_receipt_sha256"):
        if not _sha(bundle.get(name)):
            raise ValueError(f"Target-minibatch BCE requires source {name}.")
    metadata, launcher, trainer = (config.get(name, {}) for name in ("metadata", "launcher", "trainer"))
    if not all(isinstance(value, Mapping) for value in (metadata, launcher, trainer)):
        raise ValueError("Target-minibatch BCE configuration groups must be mappings.")
    profile, gpu, batch_size = metadata.get("execution_profile"), launcher.get("requested_gpu"), trainer.get("target_cells_per_optimizer_update")
    if profile not in {"pilot", "full"} or gpu not in {"0", "1", "2", "3"} or batch_size not in TARGET_BATCH_SIZES:
        raise ValueError("Target-minibatch BCE requires pilot/full, one GPU 0..3 and B=512/1024/2048.")
    if not _sha(metadata.get("frozen_task_contract_sha256")):
        raise ValueError("Target-minibatch BCE requires its frozen task contract checksum.")
    pilot = metadata.get("pilot_preflight")
    if profile == "pilot" and pilot is not None:
        raise ValueError("A fresh pilot cannot claim a prior full-run hardware preflight.")
    if profile == "full":
        if not isinstance(pilot, Mapping) or not str(pilot.get("run_id", "")).startswith("r_"):
            raise ValueError("Full target-minibatch BCE requires a verified pilot reference.")
        if (pilot.get("source_configuration_sha256") != SOURCE_CONFIG_SHA
                or pilot.get("target_cells_per_optimizer_update") != batch_size
                or not str(pilot.get("receipt_path", "")).endswith("/" + PREFLIGHT_RELATIVE_PATH)):
            raise ValueError("Full target-minibatch BCE pilot has incompatible source, batch or receipt.")
        for key in ("receipt_sha256", "success_marker_sha256", "checkpoint_sha256"):
            if not _sha(pilot.get(key)):
                raise ValueError(f"Full target-minibatch BCE pilot requires {key}.")
    expected = build_configuration(source, bundle, gpu=gpu, profile=profile,
        batch_size=batch_size, task_contract_sha256=metadata["frozen_task_contract_sha256"], pilot_preflight=pilot)
    attempt = config.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Target-minibatch BCE requires a positive execution attempt.")
    expected["attempt"] = attempt
    if config != expected:
        raise ValueError("Target-minibatch BCE differs from its explicit inherited scientific/runtime contract.")

#!/usr/bin/env python3
"""Materialize the joint-slide training contract using the existing registry split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.configuration import compose_config, validate_experiment_config
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

CAMPAIGN = "cmp_20261016_so12_pearson_relative_qkv_train_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="seed0_attempt2")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Unsafe configuration name")
    paths = current_paths()
    registry = Registry()
    dataset = registry.get_dataset("so12_pearson_integration", "v1")
    split = registry.get_split("so12_10train_4diagnostic_donors_v1")
    if not dataset or not split or dataset["verification_status"] != "verified" or split["verification_status"] != "verified":
        raise ValueError("The verified integrated dataset and donor split must already exist.")
    old = compose_config(paths.config_root / "experiment/so2_14core_relative_qkv_seed0_batch2.yaml")
    config = {key: old[key] for key in ("model", "masking", "features", "graph", "launcher")}
    train = [*[f"SO1-C{i:02d}" for i in range(1, 11)], *[f"SO2-C{i:02d}" for i in range(15, 25)]]
    val = [*[f"SO1-C{i:02d}" for i in range(11, 15)], *[f"SO2-C{i:02d}" for i in range(25, 29)]]
    config.update({
        "version": 1, "seed": 0, "fold": 0,
        "campaign": {"campaign_id": CAMPAIGN, "display_name": "SO1+SO2 four-layer QKV donor validation"},
        "experiment": {
            "variant_label": "so12_pearson_relative_qkv_l4_h8_seed0",
            "estimand": "masked_integrated_pearson_feature_reconstruction_in_held_out_donors",
            "permitted_claim": "exploratory_donor_validation_after_preprocessing_diagnostics",
        },
        "classification": {
            "schema_version": 1, "lifecycle_stage": "exploratory_screen",
            "study_axis": "so12_donor_held_out_validation",
            "scientific_variant": "relative_qkv_l4_h8_observed_only_pearson",
            "retention_class": "exploratory_screen", "classification_confidence": "high",
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"], "version": dataset["dataset_version"],
            "split_id": split["split_id"], "dataset_fingerprint": dataset["processed_fingerprint"],
            "split_fingerprint": split["fingerprint"],
            "preprocessing_version": dataset["preprocessing_version"],
            "training_adapter_version": "observed_only_mask_adapter_v1",
            "task": "masked_expression_regression", "biological_target_count": 1000,
            "target_scale": "train_standardized_quasipoisson_pearson_phi1p01_clip10",
            "input_scale": "observed_only_quasipoisson_pearson_frozen_train_statistics",
            "prepared_artifact_reference": "data/processed/so12_pearson_integration_v1",
            "training_core_aliases": train, "validation_core_aliases": val,
            "total_cells": 407659, "training_cells": 301198, "validation_cells": 106461,
            "training_donors": 10, "validation_donors": 4,
            "validation_or_test_partition_present": True,
            "independent_test_partition_present": False,
            "upstream_fully_observed_features_usage": "supervised_targets_only",
            "validation_previously_used_for_preprocessing_diagnostics": True,
            "generalization_scope": "exploratory_donor_holdout",
        },
        "trainer": {
            "learning_rate": 1e-4, "weight_decay": 1e-5, "optimizer": "AdamW", "scheduler": "none",
            "batch_size": 2, "batch_unit": "two_complete_cores_ten_views_each",
            "global_epoch_unit": "twenty_training_core_visits_ten_optimizer_updates",
            "core_visits_per_global_epoch": 20, "cores_per_optimizer_update": 2,
            "optimizer_updates_per_global_epoch": 10, "neighbor_sampling": False,
            "gradient_clip_norm": 1., "huber_delta": 1.,
            "max_epochs": 300, "minimum_global_epochs": 50,
            "early_stopping": True, "early_stopping_patience": 25, "early_stopping_min_delta": 1e-4,
            "validation_every": 1, "monitored_metric": "val/masked_huber", "restore_best": True,
            "primary_checkpoint_role": "best", "checkpoint_policy": "best_and_latest",
            "checkpoint_every_global_epochs": 1,
            "mask_views_per_core_step": 10, "mask_views_per_rank_per_optimizer_update": 5,
            "mask_view_gradient_aggregation": "ddp_exact_mean_of_four_rank_local_means_of_five",
            "precision": "bf16", "amp": True, "amp_dtype": "bfloat16",
            "amp_requires_fp32_equivalence_preflight": True,
            "deterministic": True, "deterministic_warn_only": False,
            "core_order_seed": 2026101602, "distributed": True,
            "distributed_backend": "nccl", "distributed_world_size": 4,
            "graph_execution": "four_rank_ddp_one_complete_core_per_gpu_exact",
            "stage_complete_core_graph_on_device": True, "staged_relative_geometry_dtype": "float16",
            "maximum_simultaneously_staged_cores_per_gpu": 1, "rank_zero_only_artifact_writes": True,
        },
        "evaluation": {
            "registry_version": 1, "task_family": "masked_expression_regression",
            "protocol": "so12_pearson_relative_qkv_train_validation_v1",
            "primary_metric": "val/masked_huber", "primary_direction": "minimize",
            "canonical_prediction_split": "validation", "splits": ["validation"],
            "validation_mask_views": 3, "fixed_mask_bundle": True,
            "mask_seed_namespace": "so12_fixed_validation_v1",
            "uncertainty_unit": "donor", "validation_or_test_selection": True,
            "independent_biological_replicates": 4, "expected_seeds": [0], "expected_folds": [0],
            "reference_predictor": "equal_core_training_target_mean",
        },
        "metadata": {
            "upstream_preprocessing_run": "r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410",
            "task_contract": f"experiments/campaigns/{CAMPAIGN}/README.md",
            "selection_policy": "strict_best_validation_huber_with_separate_min_delta_patience",
            "source_dataset_registration_is_target_only": True,
        },
    })
    config["masking"].update({"mask_base_seed": 2026101601, "train_seed_namespace": "so12_training_v1"})
    config["features"]["fit_scope"] = "pooled_training_cells_only"
    config["features"]["node_expression"]["transformed_with"] = config["dataset"]["input_scale"]
    config["features"]["node_metadata"]["fit_scope"] = "pooled_training_cells_only"
    config["graph"]["graph_batches"] = "twenty_train_eight_validation_disconnected_complete_core_graphs"
    config["launcher"].update({
        "disk_safety_min_free_gb": 5,
        "hardware_preflight_receipt": "state/preflight/so12_relative_qkv_ddp4_attempt2/preflight.json",
        "preflight_receipt": "state/preflight/so12_relative_qkv_ddp4_attempt2/preflight.json",
        "gpu_memory_allocated_limit_gb": 21,
    })
    validate_experiment_config(config)
    if registry.get_campaign(CAMPAIGN) is None:
        registry.create_campaign(CAMPAIGN, name=config["campaign"]["display_name"],
            scientific_question="Predict hidden integrated Pearson features in independent validation donors",
            config={"plan": config["metadata"]["task_contract"]})
    destination = paths.project_root / "experiments/campaigns" / CAMPAIGN / f"configs/{args.name}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config, sort_keys=False)
    if destination.exists() and yaml.safe_load(destination.read_text()) != config:
        raise ValueError(f"Refusing to overwrite changed scientific configuration: {destination}")
    if not destination.exists():
        destination.write_text(payload)
    print(json.dumps({"config": str(destination), "campaign_id": CAMPAIGN, "dataset": dataset["dataset_id"], "split": split["split_id"]}))


if __name__ == "__main__":
    main()

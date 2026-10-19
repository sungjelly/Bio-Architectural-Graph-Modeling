#!/usr/bin/env python3
"""Create the approved seed-0 paper-core split configuration after data preparation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.configuration import validate_experiment_config
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

CAMPAIGN = "cmp_20261017_so12_paper_core_split"
PROTOCOL = "so12_pearson_relative_qkv_paper_core_split_v1"
TRAIN = (1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26)
VALIDATION = (2, 5, 13, 27, 28)
TEST = (15, 23)


def aliases(numbers):
    return [f"SO{1 if n <= 14 else 2}-C{n:02d}" for n in numbers]


def build_config(paths, dataset, split):
    previous = paths.project_root / "experiments/campaigns/cmp_20261016_so12_pearson_relative_qkv_train_validation/configs/seed0_attempt2.yaml"
    config = copy.deepcopy(yaml.safe_load(previous.read_text()))
    config["campaign"] = {"campaign_id": CAMPAIGN, "display_name": "SO1+SO2 core holdout with test cores 15 and 23"}
    config["experiment"] = {
        "variant_label": "so12_paper_split_pearson_relative_qkv_l4_h8_seed0",
        "estimand": "masked_pearson_feature_reconstruction_in_held_out_cores",
        "permitted_claim": "exploratory_core_holdout_with_donor_overlap_and_historical_exposure",
    }
    config["classification"].update(
        study_axis="so12_paper_core_holdout",
        scientific_variant="relative_qkv_l4_h8_train21_val5_test2_observed_only_pearson",
    )
    config["dataset"] = {
        "dataset_id": dataset["dataset_id"], "version": dataset["dataset_version"],
        "split_id": split["split_id"], "dataset_fingerprint": dataset["processed_fingerprint"],
        "split_fingerprint": split["fingerprint"], "preprocessing_version": dataset["preprocessing_version"],
        "training_adapter_version": "observed_only_mask_adapter_v1",
        "task": "masked_expression_regression", "biological_target_count": 1000,
        "target_scale": "train_standardized_quasipoisson_pearson_phi1p01_clip10",
        "input_scale": "observed_only_quasipoisson_pearson_frozen_train_statistics",
        "prepared_artifact_reference": str(paths.data_root / "processed/so12_paper_core_split_v1"),
        "training_core_aliases": aliases(TRAIN), "validation_core_aliases": aliases(VALIDATION),
        "test_core_aliases": aliases(TEST), "total_cells": 407659,
        "training_cells": 264491, "validation_cells": 61561, "test_cells": 81607,
        "validation_or_test_partition_present": True, "independent_test_partition_present": True,
        "upstream_fully_observed_features_usage": "supervised_targets_only",
        "donor_overlap_permitted": True, "historical_test_exposure_disclosed": True,
        "generalization_scope": "core_holdout_with_documented_donor_overlap",
        "normalization_fit_core_aliases": aliases(TRAIN),
        "normalization_application": "one_frozen_training_fit_shared_by_all_partitions",
    }
    config["masking"].update(mask_base_seed=2026101701, train_seed_namespace="so12_paper_training_v1")
    config["features"]["node_metadata"]["transformed_with"] = "pooled_training_median_imputation_log1p_standardization"
    config["graph"]["graph_batches"] = "twenty_one_train_five_validation_two_sealed_test_complete_core_graphs"
    config["trainer"].update(
        batch_unit="two_complete_cores_ten_views_each_with_final_singleton",
        global_epoch_unit="twenty_one_training_core_visits_eleven_optimizer_updates",
        core_visits_per_global_epoch=21, optimizer_updates_per_global_epoch=11,
        mask_views_per_rank_per_optimizer_update="5_for_pairs_3_3_2_2_for_singleton",
        mask_view_gradient_aggregation="ddp_exact_mean_over_core_views_including_singleton",
        singleton_policy="visit_last_core_once_distribute_ten_views_across_four_ranks",
        core_order_seed=2026101702,
    )
    config["evaluation"] = {
        "registry_version": 1, "task_family": "masked_expression_regression", "protocol": PROTOCOL,
        "primary_metric": "val/masked_huber", "primary_direction": "minimize",
        "canonical_prediction_split": "validation", "splits": ["validation", "test"],
        "validation_mask_views": 3, "test_mask_views": 3, "fixed_mask_bundle": True,
        "mask_seed_namespace": "so12_paper_fixed_validation_v1",
        "test_mask_seed_namespace": "so12_paper_fixed_test_v1",
        "test_evaluation_after_best_checkpoint_selection": True,
        "uncertainty_unit": "core", "validation_or_test_selection": True,
        "selection_partition": "validation", "expected_seeds": [0], "expected_folds": [0],
        "reference_predictor": "equal_core_training_target_mean",
    }
    receipt = str(paths.scratch_root / "active_runs/pf_so12_paper_split_seed0/preflight.json")
    config["launcher"].update(preflight_receipt=receipt, hardware_preflight_receipt=receipt)
    config["metadata"] = {
        "task_contract": f"experiments/campaigns/{CAMPAIGN}/README.md",
        "selection_policy": "strict_best_validation_huber_with_separate_min_delta_patience",
        "source_dataset_registration_is_target_only": True,
        "initialization": "fresh_seed0_no_historical_checkpoint",
        "prior_recipe_reference": str(previous.relative_to(paths.project_root)),
        "manual_cancer_regions_used_for_selection_or_inputs": False,
    }
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    paths = current_paths()
    registry = Registry()
    dataset = registry.get_dataset("so12_paper_core_split", "v1")
    split = registry.get_split("so12_paper_21train_5val_2test_v1")
    if not dataset or not split or any(r["verification_status"] != "verified" for r in (dataset, split)):
        raise ValueError("Prepare and verify the new training-only dataset/split first")
    config = build_config(paths, dataset, split)
    validate_experiment_config(config)
    if registry.get_campaign(CAMPAIGN) is None:
        registry.create_campaign(CAMPAIGN, name=config["campaign"]["display_name"],
            scientific_question="Predict masked expression in reserved cancer-region cores using a fresh fit",
            config={"plan": config["metadata"]["task_contract"]})
    destination = paths.project_root / f"experiments/campaigns/{CAMPAIGN}/configs/seed0.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and yaml.safe_load(destination.read_text()) != config:
        raise ValueError(f"Refusing to overwrite changed scientific configuration: {destination}")
    if not destination.exists():
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"config": str(destination), "campaign_id": CAMPAIGN,
                      "dataset": dataset["dataset_id"], "split": split["split_id"]}))


if __name__ == "__main__":
    main()

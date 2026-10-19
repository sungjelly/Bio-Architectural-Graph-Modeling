#!/usr/bin/env python3
"""Register SO2 inputs and materialize reviewable masking configurations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.so2_genemae_masking import CAMPAIGN, METHODS, PRIMARY, PROTOCOL, runtime_path, validate_protocol

IMPLEMENTATION = ["src/spatial_benchmark/myjju_genemae.py", "src/spatial_benchmark/so2_genemae_masking.py",
                  "scripts/train/run_so2_genemae_masking.py"]


def make_config(method, seed, fingerprint, split_fingerprint, *, pilot=False, gate=None):
    paths = current_paths()
    model = yaml.safe_load((paths.config_root / "model/myjju_genemae_source_fidelity.yaml").read_text())["model"]
    graph = yaml.safe_load((paths.config_root / "graph/myjju_genemae_k15_tiled_union.yaml").read_text())["graph"]
    config = {
        "schema_version": 1, "seed": seed, "fold": 0,
        "campaign": {"campaign_id": CAMPAIGN},
        "classification": {"schema_version": 1, "lifecycle_stage": "diagnostic" if pilot else "exploratory_screen",
            "study_axis": "so2_masking_comparison", "scientific_variant": method+"_10pct",
            "retention_class": "diagnostic" if pilot else "exploratory_screen", "classification_confidence": "high"},
        "model": model, "graph": graph,
        "masking": {"type": "expression_masking", "method": method, "rate": .1,
                    "train_seed_namespace": "train-v1", "model_seed_in_masks": False},
        "dataset": {"dataset_id": "so2_genemae_log1p_v1", "version": "v1",
            "split_id": "so2_genemae_all_fit_v1", "dataset_fingerprint": fingerprint,
            "split_fingerprint": split_fingerprint, "preprocessing_version": "elementwise_log1p_counts_v1",
            "task": "masked_expression_regression", "target_scale": "log1p_raw_counts",
            "prepared_artifact_reference": "data/processed/so2_14core_relative_qkv_v1",
            "generalization_scope": "fit_only_transductive", "total_fit_cells": 246063},
        "features": {"use_edge_features": True, "edge_features": ["gaussian_distance_kernel"],
                     "node_features": ["visible_expression", "mask_token"]},
        "trainer": {"learning_rate": .0015, "weight_decay": .0001, "batch_size": 1,
            "max_epochs": 2 if pilot else 200, "optimizer": "AdamW", "scheduler": "cosine",
            "precision": "fp32", "gradient_clip_norm": 5., "restore_best": False,
            "primary_checkpoint_role": "last", "checkpoint_policy": "last_only", "early_stopping": False},
        "evaluation": {"protocol": PROTOCOL, "splits": ["fit"], "canonical_prediction_split": "fit",
            "primary_metric": PRIMARY, "primary_direction": "minimize", "methods": list(METHODS),
            "replicates": 3, "mask_seed_namespace": "evaluation-v1", "diagnostic_only": pilot,
            "expected_seeds": [0,1,2], "expected_folds": [0]},
        "metadata": {"execution_role": "resource_pilot" if pilot else "production",
            "implementation_sha256": {name: sha256_file(runtime_path(name)) for name in IMPLEMENTATION},
            "pilot": gate, "fold_meaning": "one_fitted_cohort_not_cv", "selection_policy": "final_epoch_only"},
    }
    validate_protocol(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run", help="Verified immutable pilot bundle")
    parser.add_argument("--pilot-name", default="pilot", help="New config name for a diagnosed pilot revision")
    args = parser.parse_args()
    paths = current_paths()
    root = paths.data_root / "processed/so2_14core_relative_qkv_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256_file(root/name) != digest:
            raise ValueError(f"Input checksum failed: {name}")
    fingerprint = sha256_file(root / "manifest.json")
    split_payload = {"cohort_manifest": fingerprint, "roles": {a: "fit" for a in manifest["cohort"]["aliases"]},
                     "held_out_patients": False, "unit": "core", "evaluation_mask_namespace": "evaluation-v1"}
    split_fingerprint = canonical_sha256(split_payload)
    registry = Registry()
    registry.initialize()
    if registry.get_campaign(CAMPAIGN) is None:
        registry.create_campaign(CAMPAIGN, name="SO2 GeneMAE 10% masking comparison",
            scientific_question="Reconstruction of whole cells versus genes and spatial blocks",
            config={"plan": f"experiments/campaigns/{CAMPAIGN}/README.md"})
    if registry.get_dataset("so2_genemae_log1p_v1", "v1") is None:
        registry.register_dataset("so2_genemae_log1p_v1", "v1", display_name="SO2 fourteen-core log1p counts",
            protected_source_path=root, processed_fingerprint=fingerprint,
            preprocessing_version="elementwise_log1p_counts_v1", aggregate_sample_count=246063,
            graph_count=14, verification_status="verified", metadata={"manifest": "manifest.json"})
    if registry.get_split("so2_genemae_all_fit_v1") is None:
        registry.register_split("so2_genemae_all_fit_v1", dataset_id="so2_genemae_log1p_v1",
            dataset_version="v1", method="all_fit_transductive", unit="core", fold_count=1,
            fingerprint=split_fingerprint, verification_status="verified", metadata=split_payload)
    directory = paths.project_root / "experiments/campaigns" / CAMPAIGN / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    gate = None
    if args.pilot_run:
        pilot_root = runtime_path(args.pilot_run)
        verify_run_bundle(pilot_root, require_success_contract=True)
        resources = json.loads((pilot_root / "diagnostics/resource.json").read_text())
        if not resources["pilot_gate_passed"]:
            raise ValueError("Pilot failed")
        gate = {"path": args.pilot_run, "success_sha256": sha256_file(pilot_root / "_SUCCESS")}
    if not args.pilot_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Unsafe pilot config name")
    candidates = [(args.pilot_name, "whole_cell", 0, True)] if not gate else [
        (f"{method}_s{seed}", method, seed, False) for method in METHODS for seed in range(3)]
    for name, method, seed, pilot in candidates:
        config = make_config(method, seed, fingerprint, split_fingerprint, pilot=pilot, gate=gate)
        path = directory / f"{name}.yaml"
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite differing config {path}")
        path.write_text(content)
    print(json.dumps({"registered": CAMPAIGN, "configs": [c[0] for c in candidates],
                      "dataset_manifest_sha256": fingerprint}))


if __name__ == "__main__":
    main()

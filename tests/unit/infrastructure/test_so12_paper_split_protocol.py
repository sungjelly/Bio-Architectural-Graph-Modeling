"""Lock the approved three-way split and its selected-checkpoint test artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.train.prepare_so12_paper_split import build_config
from spatial_benchmark.configuration import (
    ConfigurationError, compose_config, validate_experiment_config,
)
from spatial_benchmark.identifiers import create_run_id, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import command_for_config
from spatial_benchmark.run_archive import (
    RunArchive, RunValidationError, verify_run_bundle,
)


ROOT = Path(__file__).resolve().parents[3]
OLD_CAMPAIGN = "cmp_20261016_so12_pearson_relative_qkv_train_validation"
PRIMARY = "val/masked_huber"


@pytest.fixture
def config():
    # Exercise the actual launch configuration builder without registering data
    # or accessing expression matrices, checkpoints, or GPUs.
    return build_config(
        ProjectPaths.from_environment({"BAGM_ROOT": str(ROOT)}),
        {"dataset_id": "so12_paper_core_split", "dataset_version": "v1",
         "processed_fingerprint": "a" * 64, "preprocessing_version": "synthetic_v1"},
        {"split_id": "so12_paper_21train_5val_2test_v1", "fingerprint": "b" * 64},
    )


def test_paper_split_builder_dispatches_separate_four_rank_runner(config, tmp_path):
    validate_experiment_config(config)
    command = command_for_config(
        config, paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}),
    )
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc-per-node=4" in command
    assert "--max-restarts=0" in command
    assert str(tmp_path / "scripts/train/run_so12_paper_split.py") in command
    assert command[-4:] == [
        "--config", "{run_scratch}/config.resolved.yaml",
        "--run-scratch", "{run_scratch}",
    ]
    roles = [set(config["dataset"][f"{role}_core_aliases"])
             for role in ("training", "validation", "test")]
    assert list(map(len, roles)) == [21, 5, 2]
    assert len(set.union(*roles)) == 28


def test_original_joint_slide_protocol_keeps_its_validation_only_runner():
    old = compose_config(ROOT / "experiments/campaigns" / OLD_CAMPAIGN / "configs/seed0_attempt2.yaml")
    command = command_for_config(old)
    assert any(part.endswith("run_so12_relative_qkv.py") for part in command)
    assert old["evaluation"]["splits"] == ["validation"]


@pytest.mark.parametrize("section,field,value", [
    ("dataset", "training_core_aliases", ["SO2-C15"]),
    ("dataset", "validation_core_aliases", ["SO2-C23"]),
    ("dataset", "test_core_aliases", ["SO2-C15", "SO2-C22"]),
    ("dataset", "training_cells", 301198),
    ("dataset", "donor_overlap_permitted", False),
    ("dataset", "historical_test_exposure_disclosed", False),
    ("dataset", "generalization_scope", "independent_donor_holdout"),
    ("dataset", "normalization_fit_core_aliases", ["SO2-C15"]),
    ("dataset", "normalization_application", "fit_each_partition_separately"),
    ("dataset", "input_scale", "fully_observed_precomputed_pearson"),
    ("dataset", "target_scale", "signed_log_pearson"),
    ("evaluation", "selection_partition", "test"),
    ("evaluation", "canonical_prediction_split", "test"),
    ("evaluation", "splits", ["validation"]),
    ("evaluation", "test_mask_views", 1),
    ("evaluation", "test_evaluation_after_best_checkpoint_selection", False),
    ("evaluation", "test_mask_seed_namespace", "so12_paper_fixed_validation_v1"),
    ("evaluation", "expected_seeds", [0, 1]),
    ("masking", "train_seed_namespace", "so12_paper_fixed_test_v1"),
    ("masking", "count_max", 999),
    ("masking", "all_graph_cells_masked_by_same_rule", False),
    ("trainer", "core_visits_per_global_epoch", 20),
    ("trainer", "optimizer_updates_per_global_epoch", 10),
    ("trainer", "distributed_world_size", 2),
    ("trainer", "primary_checkpoint_role", "last"),
    ("model", "graph_layers", 1),
    ("launcher", "process_count", 1),
])
def test_paper_split_rejects_role_selection_and_training_drift(config, section, field, value):
    config[section][field] = value
    with pytest.raises(ConfigurationError):
        validate_experiment_config(config)
    with pytest.raises(ConfigurationError):
        command_for_config(config)


@pytest.fixture
def archive(config, tmp_path):
    run_id = create_run_id(seed=0, fold=0, attempt=1, scientific_id_value=scientific_id(config))
    result = RunArchive.create(
        run_id, paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}),
        resolved_config=config,
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
    )
    result.write_summary({
        "status": "success", "protocol": config["evaluation"]["protocol"],
        "primary_metric_name": PRIMARY, "primary_metric_value": 0.2,
        "best_epoch": 33, "final_epoch": 58,
        "checkpoint": "checkpoints/best.ckpt", "checkpoint_reload_verified": True,
        "test_artifacts_present": True,
    })
    for role in ("best", "last"):
        result.write_bytes(f"checkpoints/{role}.ckpt", f"synthetic-{role}".encode())
    result.append_metric_event({"name": PRIMARY, "value": 0.2, "step": 33})
    result.write_json("metrics/final.json", {
        PRIMARY: 0.2, "test/masked_huber": 0.3,
        "test/pooled/masked_huber": 0.31, "test/pooled/masked_mae": 0.5,
        "test/pooled/masked_mse": 0.6, "test/pooled/pearson": 0.7,
        "test/baseline/equal_core_huber": 0.4,
    })
    result.write_table("metrics/history", [
        {"run_id": run_id, "step": 33, "split": "validation", "name": PRIMARY, "value": 0.2},
    ])
    coverage = []
    for split in ("validation", "test"):
        rows = []
        for alias in config["dataset"][f"{split}_core_aliases"]:
            for view in range(3):
                key = hashlib.sha256(f"{split}:{alias}:{view}".encode()).hexdigest()
                n = {"SO2-C15": 38145, "SO2-C23": 43462}.get(alias, 100)
                rows.append({
                    "run_id": run_id, "sample_key": key,
                    "dataset_id": config["dataset"]["dataset_id"], "split": split,
                    "y_true": 0.1, "y_pred": 0.12, "sample_loss": 0.2,
                    "node_count": n, "edge_count": n * 10,
                    "masked_entry_count": n * 500,
                    "aggregation_unit": "core_mask_view_mean",
                })
                if split == "test":
                    coverage.append({"core_alias": alias, "view": view, "sample_key": key})
        result.write_prediction_jsonl_stream(split, rows)
    result.write_json("diagnostics/test_evaluation.json", {
        "checkpoint": "checkpoints/best.ckpt",
        "checkpoint_sha256": hashlib.sha256(b"synthetic-best").hexdigest(),
        "best_epoch": 33, "selection_completed": True,
        "test_loaded_after_selection": True, "test_used_for_selection": False,
        "validation_replay_verified": True, "test_evaluation_count": 1,
        "mask_namespace": "so12_paper_fixed_test_v1",
        "test_core_aliases": ["SO2-C15", "SO2-C23"], "mask_views": 3,
        "evaluated_core_view_count": 6, "test_cells": 81607,
        "core_view_coverage": coverage,
    })
    result.prepare_log_files()
    for name in ("git", "hardware", "data_fingerprints", "split_fingerprint"):
        result.write_json(f"provenance/{name}.json", {"synthetic": True})
    result.write_text("provenance/uncommitted_changes.patch", "")
    result.write_text("provenance/environment.txt", "python=synthetic\n")
    result.write_text("provenance/command.txt", '{"argv":["synthetic"]}\n')
    return result


def _mutate_json(archive, relative, change):
    path = archive.scratch_path / relative
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def test_final_test_bundle_verifies_with_checkpoint_bound_complete_coverage(archive):
    final_path = archive.finalize_success()
    assert verify_run_bundle(final_path)["status"] == "success"
    assert (final_path / "predictions/validation.jsonl").is_file()
    assert (final_path / "predictions/test.jsonl").is_file()


@pytest.mark.parametrize("field,value", [
    ("selection_completed", False),
    ("test_loaded_after_selection", False),
    ("test_used_for_selection", True),
    ("validation_replay_verified", False),
    ("test_evaluation_count", 2),
    ("mask_namespace", "so12_paper_fixed_validation_v1"),
    ("checkpoint_sha256", hashlib.sha256(b"synthetic-last").hexdigest()),
    ("best_epoch", 58),
    ("mask_views", 1),
    ("test_cells", 81606),
])
def test_archive_rejects_test_receipt_selection_or_identity_drift(archive, field, value):
    _mutate_json(archive, "diagnostics/test_evaluation.json", lambda row: row.update({field: value}))
    with pytest.raises(RunValidationError):
        archive.publish_success_pending()
    assert not archive.artifact_path.exists()


@pytest.mark.parametrize("relative", [
    "predictions/test.jsonl", "diagnostics/test_evaluation.json",
])
def test_archive_requires_test_predictions_and_receipt(archive, relative):
    (archive.scratch_path / relative).unlink()
    with pytest.raises(RunValidationError):
        archive.publish_success_pending()


@pytest.mark.parametrize("field,value", [
    ("core_alias", "SO1-C02"), ("view", 9), ("sample_key", "unbound-sample"),
])
def test_archive_requires_exact_test_core_views_bound_to_predictions(archive, field, value):
    _mutate_json(archive, "diagnostics/test_evaluation.json",
                 lambda row: row["core_view_coverage"][0].update({field: value}))
    with pytest.raises(RunValidationError):
        archive.publish_success_pending()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cell_count", "dataset", "aggregation"])
def test_archive_rejects_incomplete_or_misidentified_test_rows(archive, mutation):
    path = archive.scratch_path / "predictions/test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "cell_count":
        rows[0]["node_count"] -= 1
    elif mutation == "dataset":
        rows[0]["dataset_id"] = "previous_dataset"
    else:
        rows[0]["aggregation_unit"] = "cell"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RunValidationError):
        archive.publish_success_pending()


def test_archive_requires_final_test_metrics(archive):
    _mutate_json(archive, "metrics/final.json", lambda row: row.pop("test/masked_huber"))
    with pytest.raises(RunValidationError, match="required validation/test metrics"):
        archive.publish_success_pending()


def test_archive_allows_explicitly_undefined_test_pearson(archive):
    _mutate_json(archive, "metrics/final.json", lambda row: row.update({"test/pooled/pearson": None}))
    assert verify_run_bundle(archive.finalize_success())["status"] == "success"


def test_archive_binds_summary_selection_metric_to_validation(archive):
    _mutate_json(archive, "summary.json", lambda row: row.update({"primary_metric_value": 0.3}))
    with pytest.raises(RunValidationError, match="selection metric"):
        archive.publish_success_pending()

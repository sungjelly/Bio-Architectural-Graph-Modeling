#!/usr/bin/env python3
"""Catalog the independently fitted scVI baseline without changing BAGM artifacts.

The native predictive archive contains a small auxiliary validation reconstruction
diagnostic. It is unmasked, uses observed library sizes, and is not comparable to
BAGM's masked-expression loss. Only the original full validation ELBO selected
the checkpoint. This command never fits a model or loads test expression.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/analysis")]

from spatial_benchmark.checkpoint_catalog import (
    build_checkpoint_catalog, run_semantics_from_configuration, verify_checkpoint_record,
)
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256, create_run_id, repro_id, scientific_id
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle

CAMPAIGN = "cmp_20261018_so12_cancer_comparison"
EVALUATION = "ev_so12_cancer_comparison_s0_20261018"
PARENT = "r_20261017T044837Z_caf76350_s000_f00_a01_12df41be"
METRIC = "val/scvi_negative_elbo"
NOTE = ("Auxiliary unmasked validation NB mean-count reconstruction, aggregated over "
        "cells within each core and gene, using one fixed-seed posterior draw and "
        "observed library sizes. It is not a masked prediction, is not comparable "
        "to BAGM masked losses, and does not affect checkpoint selection. "
        "No clinical annotations or test expression enter this diagnostic.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_receipt(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def configuration(source, plan):
    dataset = source["config"]["dataset"]
    return {
        "schema": "so12_posthoc_scvi_baseline_v1", "seed": 0, "fold": 0, "attempt": 1,
        "campaign": {"campaign_id": CAMPAIGN},
        "experiment": {"variant_label": "train21_scvi50_nb_slide_batch_seed0"},
        "model": {"name": "scvi", "family": "scvi", "embedding_dim": 50,
                  "hidden_dim": 128, "layers": 2, "gene_likelihood": "nb",
                  "dispersion": "gene", "batch_covariate": "slide", "package_version": "1.5.1"},
        "dataset": {"dataset_id": dataset["dataset_id"], "version": dataset["version"],
                    "split_id": dataset["split_id"], "dataset_fingerprint": source["dataset_fingerprint"],
                    "split_fingerprint": source["split_fingerprint"],
                    "preprocessing_version": "raw_counts_biological_1000_scvi_v1",
                    "input_scale": "nonnegative_integer_raw_counts", "biological_target_count": 1000,
                    "training_cells": plan["training_cells"], "validation_cells": plan["validation_cells"],
                    "training_core_numbers": plan["training_core_numbers"],
                    "validation_core_numbers": plan["validation_core_numbers"],
                    "test_core_numbers": [15, 23], "test_cells_in_fit": 0,
                    "donor_overlap_permitted": True, "historical_test_exposure_disclosed": True,
                    "generalization_scope": "exploratory_core_holdout_with_documented_donor_overlap"},
        "graph": {"graph_id": "none", "neighbor_k": 0}, "masking": {"type": "none"},
        "features": {"use_edge_features": False, "morphology": False, "coordinates": False},
        "trainer": {**plan, "restore_best": True, "primary_checkpoint_role": "best"},
        "evaluation": {"artifact_contract": "predictive", "canonical_prediction_split": "validation",
                       "protocol": "so12_posthoc_scvi_validation_reconstruction_v1",
                       "primary_metric": METRIC, "primary_metric_mode": "min",
                       "original_metric_name": "elbo_validation", "metric_sign_unchanged": True,
                       "test_query_adaptation": False, "auxiliary_prediction_description": NOTE},
        "classification": {"lifecycle_stage": "exploratory_screen", "study_axis": "posthoc_scvi_baseline",
                           "retention_class": "analysis_evidence", "classification_confidence": "high"},
        "metadata": {"comparison_evaluation_id": EVALUATION, "source_bagm_run_id": PARENT,
                     "source_bagm_role": "split_and_comparison_reference_only_not_initialization",
                     "historically_untouched_test_claim": False, "sealed_test_claim": False,
                     "test_representation_evaluated_posthoc": True,
                     "checkpoint_format": "scvi_tools_1.5.1_save_model.pt",
                     "checkpoint_restore_instruction": "Copy checkpoints/best.ckpt to model.pt in a model directory, then SCVI.load with compatible AnnData.",
                     "auxiliary_archive_prediction_reason": "Native predictive archive requires genuine validation predictions; diagnostic is separate from model selection."},
    }


def ensure_json(archive, relative, value):
    path = archive.scratch_path / relative
    if path.exists():
        require(read(path) == value, f"Existing archive JSON differs: {relative}")
    else:
        archive.write_json(relative, value)


def ensure_copy(archive, source, relative):
    target = archive.scratch_path / relative
    if target.exists():
        require(sha256_file(target) == sha256_file(source), f"Existing archive copy differs: {relative}")
    else:
        archive.copy_file(source, relative)


def initialize(output, paths, registry):
    require(output.resolve() == (paths.scratch_root / "active_runs" / EVALUATION).resolve(),
            "Wrong comparison active directory.")
    source = read(output / "provenance/source_verified.json")
    plan = read(output / "baselines/scvi_training_plan.json")
    require(source["source_run_id"] == PARENT and source["checkpoint_epoch"] == 194,
            "Wrong BAGM comparison reference.")
    require(plan["training_cells"] == 264491 and plan["validation_cells"] == 61561
            and plan["test_cells_in_fit"] == 0, "Baseline fit population changed.")
    bindings = read(output / "provenance/scvi_source_binding.json")
    for relative, digest in bindings.items():
        require(sha256_file(paths.project_root / relative) == digest, "Frozen baseline source changed.")
    config = configuration(source, plan)
    events = [json.loads(line) for line in (output / "stages/scvi.jsonl").read_text().splitlines() if line]
    started = next(row["time"] for row in events if row["status"] == "running")
    sci = scientific_id(config)
    run_id = create_run_id(seed=0, fold=0, attempt=1, scientific_id_value=sci,
                          timestamp=datetime.fromisoformat(started),
                          unique_suffix=canonical_sha256({"evaluation": EVALUATION, "sources": bindings})[:8])
    link_path = output / "scvi_run_registration.json"
    if link_path.exists():
        previous = read(link_path)
        require(previous["derived_run_id"] == run_id, "Existing derived run identity differs.")
    installed = read(Path("/venv/so12-scvi/dependency_verification.json"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.project_root, text=True).strip()
    rep = repro_id(config, git_commit=commit,
                   dirty_fingerprint=canonical_sha256(bindings),
                   dataset_fingerprint=source["dataset_fingerprint"],
                   split_fingerprint=source["split_fingerprint"],
                   preprocessing_version=config["dataset"]["preprocessing_version"],
                   environment_fingerprint=canonical_sha256(installed))
    registered = registry.get_run(run_id)
    if registered is None:
        registry.register_variant(sci, campaign_id=CAMPAIGN, configuration=config)
        registry.create_run(run_id, campaign_id=CAMPAIGN, scientific_id=sci, repro_id=rep,
                            seed=0, fold=0, attempt=1, configuration=config, status="running",
                            artifact_path=paths.scratch_root / "active_runs" / run_id,
                            start_time=started, host=socket.gethostname(), git_commit=commit,
                            dirty_status=True, gpu_model="NVIDIA GPU 1; see execution provenance")
    else:
        require(registered["config"] == config and registered["repro_id"] == rep,
                "Registered derived configuration changed.")
    registry.register_run_semantics(run_id, **run_semantics_from_configuration(
        primary_run_id=run_id, configuration=config, scientific_id_value=sci))
    published = RunArchive.artifact_path_for(run_id, paths)
    scratch = paths.scratch_root / "active_runs" / run_id
    if published.exists():
        archive = RunArchive.from_published(run_id, paths=paths)
    elif scratch.exists():
        archive = RunArchive.attach_active(run_id, paths=paths)
    else:
        archive = RunArchive.create(run_id, paths=paths, resolved_config=config)
    receipt = {"schema": "so12_scvi_run_registration_v1", "status": registered["status"] if registered else "running",
               "derived_run_id": run_id, "scientific_id": sci, "repro_id": rep,
               "comparison_evaluation_id": EVALUATION, "source_bagm_run_id": PARENT,
               "model_family": "scvi", "artifact_path": str(published),
               "checkpoint_selection": "minimum_original_full_elbo_validation",
               "historically_untouched_test_claim": False, "sealed_test_claim": False,
               "auxiliary_prediction_description": NOTE, "fit_start_time": started}
    if published.exists() and link_path.exists() and read(link_path).get("status") == "complete":
        return archive, config, read(link_path)
    write_receipt(link_path, receipt)
    if not published.exists():
        ensure_json(archive, "provenance/git.json", {"commit": commit, "dirty": True,
                    "source_binding_scope": "original baseline execution; registration helper separately snapshotted",
                    "source_sha256": bindings})
        ensure_json(archive, "provenance/data_fingerprints.json", {key: source[key] for key in
                    ["dataset_fingerprint", "preprocessing_fingerprint", "fitted_parameters_sha256"]})
        ensure_json(archive, "provenance/split_fingerprint.json", {"split_fingerprint": source["split_fingerprint"],
                    "train": plan["training_core_numbers"], "validation": plan["validation_core_numbers"],
                    "test": [15, 23], "test_cells_in_fit": 0})
        ensure_json(archive, "provenance/hardware.json", {"fit_gpu_physical_index": 1,
                    "archive_validation_diagnostic_device": "cpu", "host": socket.gethostname()})
        ensure_json(archive, "provenance/scvi_environment.json", installed)
        ensure_copy(archive, output / "provenance/scvi_source_binding.json", "provenance/scvi_source_binding.json")
        ensure_copy(archive, output / "provenance/task_contract_at_start.md", "provenance/task_contract_at_start.md")
        ensure_copy(archive, output / "baselines/scvi_training_plan.json", "provenance/scvi_training_plan.json")
        # Registration code may be corrected without changing the fitted model.
        # Retain every exact helper version; never overwrite earlier provenance.
        ensure_copy(archive, Path(__file__), "provenance/code/register_so12_scvi_run_"
                    + sha256_file(Path(__file__))[:16] + ".py")
        for relative in bindings:
            ensure_copy(archive, paths.project_root / relative, "provenance/code/" + relative)
        for relative, text in {
            "provenance/environment.txt": json.dumps(installed, sort_keys=True, indent=2) + "\n",
            "provenance/uncommitted_changes.patch": "# Untracked analysis sources are preserved under provenance/code; immutable training sources were not edited.\n",
            "provenance/command.txt": json.dumps({"fit": ["/venv/so12-scvi/bin/python", "scripts/analysis/so12_cancer_baselines.py", "--output", str(output), "--mode", "scvi"],
                                                  "registration": sys.argv, "cwd": str(paths.project_root)}) + "\n",
            "provenance/archive_contract_explanation.txt": NOTE + "\n",
        }.items():
            if not (archive.scratch_path / relative).exists():
                archive.write_text(relative, text)
    return archive, config, receipt


def validation_predictions(output, source, paths, run_id, freeze):
    """One fixed-seed posterior draw, frozen model, validation-only counts."""
    import anndata
    import numpy as np
    import pandas as pd
    import scvi
    import torch
    import so12_cancer_baselines as baseline
    torch.set_num_threads(4)
    scvi.settings.seed = 0
    baseline.base.configure_deterministic_inference()
    dataset = baseline.load_data(source, paths, ("validation",))
    adata = anndata.AnnData(X=np.concatenate([core.raw_counts for core in dataset.cores]).astype(np.float32),
        obs=pd.DataFrame({"slide": pd.Categorical([str(core.slide) for core in dataset.cores
                                                 for _ in range(core.n_cells)], categories=["1", "2"])}),
        var=pd.DataFrame(index=list(dataset.genes)))
    model = scvi.model.SCVI.load(output / "baselines/scvi_model", adata=adata, accelerator="cpu", device="auto")
    model.module.eval()
    require(baseline.state_hash(model.module.state_dict()) == freeze["state_sha256"], "Reloaded scVI state differs.")
    rows, offset = [], 0
    for index, core in enumerate(dataset.cores):
        predicted = model.get_likelihood_parameters(adata, indices=np.arange(offset, offset + core.n_cells),
                                                    n_samples=1, give_mean=True, batch_size=1024)["mean"]
        require(predicted.shape == core.raw_counts.shape and np.isfinite(predicted).all()
                and (predicted >= 0).all(), "Invalid validation NB mean counts.")
        observed_mean = np.asarray(core.raw_counts).mean(axis=0, dtype=np.float64)
        predicted_mean = predicted.mean(axis=0, dtype=np.float64)
        rows.append({"run_id": run_id, "sample_key": "validation_core_aggregate_" + str(index),
                     "dataset_id": "so12_paper_core_split", "split": "validation",
                     "y_true": observed_mean.tolist(), "y_pred": predicted_mean.tolist(),
                     "node_count": core.n_cells, "effective_mask_rate": 0.0,
                     "sample_loss": float(np.mean((observed_mean - predicted_mean) ** 2))})
        offset += core.n_cells
    require(offset == 61561 and len(rows) == 5, "Incomplete validation reconstruction coverage.")
    require(baseline.state_hash(model.module.state_dict()) == freeze["state_sha256"]
            and sha256_file(output / "baselines/scvi_model/model.pt") == freeze["model_sha256"],
            "Frozen weights changed during auxiliary validation diagnostic.")
    return rows, {"description": NOTE, "core_numbers": [2, 5, 13, 27, 28], "n_cells": offset,
                  "genes": list(dataset.genes), "posterior_draws": 1, "seed": 0,
                  "parameter_updates": 0, "test_cells_loaded": 0, "model_state_unchanged": True,
                  "model_sha256": freeze["model_sha256"], "epoch_zero_based": freeze["selected_epoch_zero_based"]}


def finalize(output, paths, registry, archive, config, receipt):
    if (archive.artifact_path / "_SUCCESS").exists():
        return register_finished(output, paths, registry, archive, receipt)
    require(read(output / "stages/scvi.json")["status"] == "complete", "scVI stage has not completed.")
    manifest = read(output / "scvi_manifest.json")
    freeze = read(output / "baselines/scvi_frozen_before_test.json")
    require(manifest["passed"] is True and manifest["status"] == "complete"
            and manifest["representation_name"] == "scvi" and manifest["n_cells"] == 81607
            and manifest["test_used_for_fitting"] is False, "scVI completion receipt failed.")
    require(manifest["model_sha256"] == freeze["model_sha256"]
            == sha256_file(output / "baselines/scvi_model/model.pt"), "scVI checkpoint hash differs.")
    history = [json.loads(line) for line in (output / "baselines/scvi_training.jsonl").read_text().splitlines() if line]
    best = min(history, key=lambda row: row["elbo_validation"])
    require(best["epoch"] == freeze["selected_epoch_zero_based"]
            and best["elbo_validation"] == freeze["validation_elbo"], "Selected checkpoint is not minimum validation ELBO.")
    if archive.artifact_path.exists():
        archive.mark_success()
        return register_finished(output, paths, registry, archive, receipt)
    source = read(output / "provenance/source_verified.json")
    diagnostic_path = archive.scratch_path / "diagnostics/validation_reconstruction.json"
    if not diagnostic_path.exists():
        rows, diagnostic = validation_predictions(output, source, paths, archive.run_id, freeze)
        predictions = archive.scratch_path / "predictions/validation.jsonl"
        if predictions.exists():
            require([json.loads(line) for line in predictions.read_text().splitlines()] == rows,
                    "Partial validation predictions differ.")
        else:
            archive.write_prediction_jsonl_stream("validation", rows)
        archive.write_json("diagnostics/validation_reconstruction.json", diagnostic)
    ensure_copy(archive, output / "baselines/scvi_model/model.pt", "checkpoints/best.ckpt")
    for source_name, target in [
        ("baselines/scvi_frozen_before_test.json", "provenance/scvi_frozen_before_test.json"),
        ("scvi_manifest.json", "provenance/scvi_manifest.json"),
        ("baselines/scvi_training.jsonl", "provenance/scvi_training.jsonl"),
        ("logs/scvi.log", "logs/stdout.log"),
    ]:
        ensure_copy(archive, output / source_name, target)
    if not (archive.scratch_path / "logs/stderr.log").exists():
        archive.write_text("logs/stderr.log", "# stderr is merged into stdout by the managed stage wrapper.\n")
    if not (archive.scratch_path / "metrics/events.jsonl").exists():
        for row in history:
            archive.append_metric_event({"name": METRIC, "value": row["elbo_validation"], "step": row["epoch"]})
    ensure_json(archive, "metrics/final.json", {METRIC: freeze["validation_elbo"]})
    if not (archive.scratch_path / "metrics/history.jsonl").exists():
        archive.write_text("metrics/history.jsonl", "".join(json.dumps(row) + "\n" for row in history))
    summary = {"status": "success", "model_name": "scvi", "primary_metric_name": METRIC,
               "primary_metric_value": freeze["validation_elbo"], "best_epoch": freeze["selected_epoch_zero_based"],
               "epoch_numbering": "zero_based", "selection_metric_original_name": "elbo_validation",
               "selection_metric_sign_unchanged": True, "training_cells": 264491, "validation_cells": 61561,
               "test_used_for_fitting": False, "test_query_adaptation": False,
               "historically_untouched_test_claim": False, "sealed_test_claim": False,
               "source_bagm_run_id": PARENT, "comparison_evaluation_id": EVALUATION,
               "auxiliary_prediction_description": NOTE, "completed_at": manifest["created_at"],
               "duration_seconds": manifest["elapsed_seconds"], "checkpoint_sha256": freeze["model_sha256"]}
    if not (archive.scratch_path / "summary.json").exists():
        archive.write_summary(summary)
    if not (archive.scratch_path / "manifest.yaml").exists():
        archive.write_manifest({"status": "success", "campaign_id": CAMPAIGN,
                                "model_name": "scvi", "format": "native_predictive_v1"})
    registry.transition_run(archive.run_id, "finalizing")
    archive.publish_success_pending()
    archive.mark_success()
    return register_finished(output, paths, registry, archive, receipt)


def register_finished(output, paths, registry, archive, receipt):
    verification = verify_run_bundle(archive.artifact_path)
    require(verification["valid"] and verification["status"] == "success", "Native scVI bundle failed verification.")
    summary = read(archive.artifact_path / "summary.json")
    checkpoint = archive.artifact_path / "checkpoints/best.ckpt"
    digest = sha256_file(checkpoint)
    require(digest == summary["checkpoint_sha256"], "Published checkpoint differs from selected scVI.")
    with registry.connect() as connection:
        existing = connection.execute("SELECT * FROM artifacts WHERE run_id=? AND path=?",
                                      (archive.run_id, str(checkpoint))).fetchall()
    require(len(existing) <= 1, "Duplicate scVI checkpoint artifact registration.")
    artifact_id = existing[0]["artifact_id"] if existing else registry.record_artifact(
        archive.run_id, evaluation_id=EVALUATION, kind="checkpoint_scvi_best_validation",
        path=checkpoint, sha256=digest, size_bytes=checkpoint.stat().st_size)
    registry.register_checkpoint_metadata(artifact_id, run_id=archive.run_id, role="best",
        retention_class="analysis_evidence", verification_status="verified", best_epoch=summary["best_epoch"],
        monitored_metric=METRIC, monitored_mode="min", monitored_value=summary["primary_metric_value"],
        metadata={"selection_role": "best_validation_elbo", "epoch_numbering": "zero_based",
                  "original_metric_name": "elbo_validation", "metric_sign_unchanged": True,
                  "model_family": "scvi", "checkpoint_format": "scvi_tools_1.5.1_model.pt",
                  "comparison_evaluation_id": EVALUATION, "source_bagm_reference_only": PARENT})
    records = build_checkpoint_catalog(registry, paths, run_ids=[archive.run_id])
    require(len(records) == 1 and records[0]["model_name"] == "scvi", "Checkpoint catalog model semantics differ.")
    checked = verify_checkpoint_record(records[0], paths=paths)
    with registry.connect() as connection:
        metric_exists = connection.execute("SELECT 1 FROM metrics WHERE run_id=? AND name=?",
                                          (archive.run_id, METRIC)).fetchone()
    if not metric_exists:
        registry.record_metric(archive.run_id, METRIC, summary["primary_metric_value"],
                               step=summary["best_epoch"], split="validation")
    if registry.get_run(archive.run_id)["status"] == "running":
        registry.transition_run(archive.run_id, "finalizing")
    registry.transition_run(archive.run_id, "completed", artifact_path=archive.artifact_path,
                            end_time=summary["completed_at"], duration_seconds=summary["duration_seconds"],
                            primary_metric_name=METRIC, primary_metric_value=summary["primary_metric_value"])
    final = {**receipt, "status": "complete", "artifact_path": str(archive.artifact_path),
             "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
             "checkpoint_size_bytes": checkpoint.stat().st_size, "checkpoint_artifact_id": artifact_id,
             "checkpoint_role": "best_validation_elbo", "selected_epoch_zero_based": summary["best_epoch"],
             "validation_negative_elbo": summary["primary_metric_value"], "native_bundle_verified": verification,
             "catalog_model_name": checked["model_name"], "catalog_verified": checked["verification_checked"],
             "registered_at": utc_now()}
    write_receipt(output / "scvi_run_registration.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    paths = current_paths()
    lock = paths.state_root / "locks/so12-scvi-run-registration.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
        archive, config, receipt = initialize(args.output, paths, registry)
        if args.finalize:
            receipt = finalize(args.output, paths, registry, archive, config, receipt)
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

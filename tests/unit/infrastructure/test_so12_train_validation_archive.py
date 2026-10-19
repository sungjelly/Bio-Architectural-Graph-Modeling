"""Exercise the integrated trainer's final artifact contract without GPU work."""

from __future__ import annotations

import hashlib
from pathlib import Path

from spatial_benchmark.checkpoint_catalog import (
    index_checkpoint_catalog,
    show_checkpoint,
)
from spatial_benchmark.configuration import compose_config
from spatial_benchmark.identifiers import create_run_id, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = "cmp_20261016_so12_pearson_relative_qkv_train_validation"
PRIMARY = "val/masked_huber"


def test_so12_final_bundle_accepts_validation_tables_and_indexes_best_and_last(
    tmp_path: Path,
) -> None:
    config = compose_config(
        ROOT / "experiments/campaigns" / CAMPAIGN / "configs/seed0.yaml"
    )
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    scientific = scientific_id(config)
    run_id = create_run_id(
        seed=0, fold=0, attempt=1, scientific_id_value=scientific,
    )
    archive = RunArchive.create(
        run_id, paths=paths, resolved_config=config,
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
    )
    checkpoint_results = {
        "best": {
            "path": "checkpoints/best.ckpt", "epoch": 33,
            "monitored_metric": PRIMARY, "monitored_mode": "min",
            "monitored_value": 0.2,
        },
        "last": {
            "path": "checkpoints/last.ckpt", "epoch": 58,
            "monitored_metric": PRIMARY, "monitored_mode": "min",
            "monitored_value": 0.3,
        },
    }
    archive.write_summary({
        "status": "success", "protocol": config["evaluation"]["protocol"],
        "primary_metric_name": PRIMARY, "primary_metric_value": 0.2,
        "best_epoch": 33, "final_epoch": 58,
        "checkpoint": "checkpoints/best.ckpt",
        "checkpoint_reload_verified": True,
        "checkpoints": checkpoint_results,
        "generalization_estimate": False, "test_artifacts_present": False,
    })
    # Archive/catalog verification checks file integrity rather than loading
    # model state. These unique tiny payloads avoid constructing a real model.
    for role in checkpoint_results:
        archive.write_bytes(f"checkpoints/{role}.ckpt", f"synthetic-{role}".encode())
    archive.append_metric_event({"name": PRIMARY, "value": 0.2, "step": 33})
    archive.write_json("metrics/final.json", {
        PRIMARY: 0.2, "val/pooled/masked_huber": 0.22,
        "val/pooled/masked_mae": 0.4, "val/pooled/masked_mse": 0.5,
        "val/pooled/pearson": 0.6, "val/baseline/equal_core_huber": 0.4,
    })
    history = [
        {"run_id": run_id, "step": epoch, "split": "validation",
         "name": PRIMARY, "value": value}
        for epoch, value in [(33, 0.2), (58, 0.3)]
    ]
    archive.write_table("metrics/history", history)
    predictions = [
        {
            "run_id": run_id,
            "sample_key": hashlib.sha256(f"{alias}:{view}".encode()).hexdigest(),
            "dataset_id": config["dataset"]["dataset_id"], "split": "validation",
            "y_true": 0.1, "y_pred": 0.12, "sample_loss": 0.2,
            "node_count": 100, "edge_count": 1000, "masked_entry_count": 50000,
            "aggregation_unit": "core_mask_view_mean",
        }
        for alias in config["dataset"]["validation_core_aliases"]
        for view in range(config["evaluation"]["validation_mask_views"])
    ]
    assert len(predictions) == 24
    prediction_path = archive.write_predictions("validation", predictions)
    archive.prepare_log_files()
    for name, value in {
        "git.json": {"commit": "synthetic", "dirty": False},
        "hardware.json": {"device": "synthetic"},
        "data_fingerprints.json": {"dataset": "synthetic"},
        "split_fingerprint.json": {"split": "synthetic"},
    }.items():
        archive.write_json(f"provenance/{name}", value)
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=synthetic\n")
    archive.write_text("provenance/command.txt", '{"argv":["synthetic"]}\n')

    # Match the worker's publish -> index -> success-marker ordering.
    final_path = archive.publish_success_pending()
    assert not (final_path / "_SUCCESS").exists()
    registry = Registry(paths.state_root / "tracking/test.sqlite3")
    registry.create_campaign(CAMPAIGN, name="SO12 artifact acceptance fixture")
    registry.register_variant(scientific, campaign_id=CAMPAIGN, configuration=config)
    registry.create_run(
        run_id, campaign_id=CAMPAIGN, scientific_id=scientific,
        repro_id="rep_synthetic", seed=0, fold=0, attempt=1,
        configuration=config, status="completed", artifact_path=final_path,
        primary_metric_name=PRIMARY, primary_metric_value=0.2,
    )
    for role, expected in checkpoint_results.items():
        checkpoint = final_path / expected["path"]
        registry.record_artifact(
            run_id, kind="checkpoints", path=checkpoint,
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            size_bytes=checkpoint.stat().st_size,
        )
    indexed = index_checkpoint_catalog(registry, paths, run_reference=run_id)
    assert indexed["indexed_checkpoints"] == 2
    archive.mark_success()
    assert verify_run_bundle(final_path)["status"] == "success"
    assert (final_path / "predictions" / prediction_path.name).is_file()
    assert not list((final_path / "predictions").glob("test.*"))
    for role, expected in checkpoint_results.items():
        record = show_checkpoint(registry, run_id, paths, role=role, verify=True)
        assert record["best_epoch"] == expected["epoch"]
        assert record["monitored_value"] == expected["monitored_value"]
        assert record["monitored_metric"] == PRIMARY
        assert record["monitored_mode"] == "min"

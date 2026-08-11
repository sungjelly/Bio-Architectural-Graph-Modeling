from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import spatial_benchmark.run_archive as run_archive_module

from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import (
    RunArchive,
    RunArchiveError,
    RunImmutableError,
    RunValidationError,
    _canonical_prediction_split,
    deidentify_prediction_rows,
    validate_prediction_rows,
    verify_run_bundle,
)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def _run_id(suffix: str = "abcd1234") -> str:
    return create_run_id(
        seed=3,
        fold=2,
        attempt=1,
        scientific_id_value="sci_7a91c6e212345678",
        timestamp=datetime(2026, 7, 24, 4, 12, 33, tzinfo=timezone.utc),
        unique_suffix=suffix,
    )


def _prediction(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "sample_key": "sk_0123456789abcdef",
        "graph_id": "graph_1",
        "dataset_id": "gastric_cosmx",
        "split": "validation",
        "fold": 2,
        "y_true": [0.5, 1.5],
        "y_pred": [0.4, 1.4],
        "sample_loss": 0.01,
        "node_count": 30,
        "edge_count": 100,
        "effective_neighbor_count": 6.7,
        "effective_mask_rate": 0.2,
    }


def _write_success_support(archive: RunArchive) -> None:
    archive.append_metric_event({"name": "val/masked_huber", "value": 0.2, "step": 1})
    archive.write_json("metrics/final.json", {"val/masked_huber": 0.2})
    archive.write_table(
        "metrics/history",
        [{"epoch": 1, "train/loss": 0.3, "val/masked_huber": 0.2}],
    )
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "test"})
    archive.write_text("provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n')


@pytest.mark.parametrize(
    "protocol",
    (
        "held_in_full_core_fixed_budget",
        "held_in_pooled_10core_fixed_budget",
    ),
)
def test_canonical_fit_prediction_accepts_explicit_held_in_protocols(
    tmp_path: Path,
    protocol: str,
) -> None:
    (tmp_path / "config.resolved.yaml").write_text(
        "evaluation:\n"
        f"  protocol: {protocol}\n"
        "  canonical_prediction_split: fit\n",
        encoding="utf-8",
    )

    assert _canonical_prediction_split(tmp_path) == "fit"


@pytest.mark.parametrize(
    "protocol",
    (
        "held_out_geometry_masked_reconstruction",
        "held_out_slide_masked_reconstruction",
    ),
)
def test_canonical_test_prediction_accepts_explicit_held_out_protocols(
    tmp_path: Path,
    protocol: str,
) -> None:
    (tmp_path / "config.resolved.yaml").write_text(
        "evaluation:\n"
        f"  protocol: {protocol}\n"
        "  canonical_prediction_split: test\n",
        encoding="utf-8",
    )

    assert _canonical_prediction_split(tmp_path) == "test"


def test_canonical_test_prediction_rejects_missing_held_out_protocol(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.resolved.yaml").write_text(
        "evaluation:\n"
        "  protocol: resource_validation\n"
        "  canonical_prediction_split: test\n",
        encoding="utf-8",
    )

    with pytest.raises(RunValidationError, match="held-out"):
        _canonical_prediction_split(tmp_path)


def test_prediction_deidentification_is_copying_and_salted() -> None:
    source = [
        {
            "patient_id": "protected-patient",
            "cell_id": "protected-cell",
            "run_id": "r_example",
            "dataset_id": "dataset",
            "split": "validation",
            "y_true": 1.0,
            "y_pred": 0.9,
        }
    ]
    deidentified = deidentify_prediction_rows(
        source,
        identifier_fields=["patient_id", "cell_id"],
        salt="local-secret-salt-1234",
    )

    assert source[0]["patient_id"] == "protected-patient"
    assert "patient_id" not in deidentified[0]
    assert "cell_id" not in deidentified[0]
    assert deidentified[0]["sample_key"].startswith("sk_")
    validate_prediction_rows(deidentified)

    unsafe = [{**deidentified[0], "patient_id": "still-protected"}]
    with pytest.raises(RunValidationError, match="identifier"):
        validate_prediction_rows(unsafe)


def test_active_archive_attachment_requires_canonical_owned_scratch(
    tmp_path: Path,
) -> None:
    run_id = _run_id("attach01")
    paths = _paths(tmp_path)
    created = RunArchive.create(run_id, paths=paths)

    attached = RunArchive.attach_active(
        run_id,
        paths=paths,
        scratch_path=created.scratch_path,
    )

    assert attached.scratch_path == created.scratch_path
    attached.write_text("diagnostics/attached.txt", "owned\n")
    with pytest.raises(RunArchiveError, match="canonical run path"):
        RunArchive.attach_active(
            run_id,
            paths=paths,
            scratch_path=tmp_path / "somewhere-else",
        )

    owner = created.scratch_path / ".bagm-run-owner.json"
    owner.write_text(
        json.dumps({"run_id": _run_id("foreign1"), "format_version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(RunArchiveError, match="ownership marker"):
        RunArchive.attach_active(run_id, paths=paths)


def test_successful_run_is_published_after_required_contract(
    tmp_path: Path,
) -> None:
    run_id = _run_id()
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={"model": {"name": "g1"}, "seed": 3},
    )
    assert not (archive.scratch_path / "metrics").exists()
    archive.write_summary(
        {"status": "completed", "primary_metric": "val/loss", "primary_value": 0.2}
    )
    archive.write_bytes("checkpoints/best.ckpt", b"checkpoint")
    _write_success_support(archive)
    table_path = next((archive.scratch_path / "metrics").glob("history.*"))
    prediction_path = archive.write_predictions(
        "validation",
        [_prediction(run_id)],
    )

    final_path = archive.finalize_success()

    assert not archive.scratch_path.exists()
    assert final_path == (
        tmp_path / "artifacts" / "runs" / "2026" / "07" / run_id
    )
    assert (final_path / "_SUCCESS").is_file()
    assert table_path.suffix in {".parquet", ".jsonl", ".csv"}
    assert prediction_path.suffix in {".parquet", ".jsonl", ".csv"}
    assert verify_run_bundle(final_path)["status"] == "success"
    with pytest.raises(RunImmutableError):
        archive.write_text("logs/late.log", "not allowed")


def test_success_marker_publish_is_atomic_and_replayable_after_link_failpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _run_id("markfail")
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={"model": {"name": "g1"}, "seed": 3},
    )
    archive.write_summary(
        {"status": "completed", "primary_metric": "val/loss", "primary_value": 0.2}
    )
    archive.write_bytes("checkpoints/best.ckpt", b"checkpoint")
    _write_success_support(archive)
    archive.write_predictions("validation", [_prediction(run_id)])
    final = archive.publish_success_pending()

    real_link = run_archive_module.os.link

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic hard-crash boundary")

    monkeypatch.setattr(run_archive_module.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic"):
        archive.mark_success()
    assert not (final / "_SUCCESS").exists()
    assert not list(final.glob("._SUCCESS.writing-*"))

    monkeypatch.setattr(run_archive_module.os, "link", real_link)
    assert archive.ensure_success() == final
    assert archive.ensure_success() == final
    assert verify_run_bundle(final)["status"] == "success"


def test_retention_tombstones_must_match_immutable_bundle_manifest(
    tmp_path: Path,
) -> None:
    run_id = _run_id("retained1")
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={"model": {"name": "g1"}, "seed": 3},
    )
    archive.write_summary(
        {"status": "completed", "primary_metric": "val/loss", "primary_value": 0.2}
    )
    checkpoint = archive.write_bytes("checkpoints/best.ckpt", b"checkpoint")
    _write_success_support(archive)
    prediction = archive.write_predictions("validation", [_prediction(run_id)])
    checkpoint_relative = checkpoint.relative_to(archive.scratch_path).as_posix()
    prediction_relative = prediction.relative_to(archive.scratch_path).as_posix()
    final_path = archive.finalize_success()
    manifest = json.loads(
        (final_path / "provenance/artifact_checksums.json").read_text(
            encoding="utf-8"
        )
    )["files"]
    tombstones = {
        checkpoint_relative: manifest[checkpoint_relative],
        prediction_relative: manifest[prediction_relative],
    }
    (final_path / checkpoint_relative).unlink()
    (final_path / prediction_relative).unlink()

    with pytest.raises(RunValidationError, match="missing="):
        verify_run_bundle(final_path)

    verified = verify_run_bundle(
        final_path,
        tombstoned_artifacts=tombstones,
    )
    assert verified["valid"] is True
    assert verified["tombstoned_file_count"] == 2
    assert verified["present_file_count"] == verified["file_count"] - 2

    wrong_tombstones = {
        **tombstones,
        checkpoint_relative: {
            **tombstones[checkpoint_relative],
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(RunValidationError, match="does not match"):
        verify_run_bundle(
            final_path,
            tombstoned_artifacts=wrong_tombstones,
        )


def test_held_in_protocol_requires_and_accepts_fit_predictions(
    tmp_path: Path,
) -> None:
    run_id = _run_id("fitrole1")
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "model": {"name": "g2"},
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
                "canonical_prediction_split": "fit",
                "primary_metric": "fit/whole_node/masked_huber",
            },
        },
    )
    archive.write_summary(
        {
            "status": "completed",
            "primary_metric_name": "fit/whole_node/masked_huber",
            "primary_metric_value": 0.2,
        }
    )
    # Fixed-budget held-in training retains the last epoch; calling it "best"
    # would imply validation selection that did not occur.
    config_path = archive.scratch_path / "config.resolved.yaml"
    config_path.unlink()
    archive.write_resolved_config(
        {
            "model": {"name": "g2"},
            "trainer": {
                "primary_checkpoint_role": "last",
                "restore_best": False,
            },
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
                "canonical_prediction_split": "fit",
                "primary_metric": "fit/whole_node/masked_huber",
            },
        }
    )
    archive.write_bytes("checkpoints/last.ckpt", b"final-fixed-budget")
    archive.append_metric_event(
        {"name": "fit/whole_node/masked_huber", "value": 0.2, "step": 1}
    )
    archive.write_json(
        "metrics/final.json",
        {
            "fit/whole_node/masked_huber": 0.2,
            "fit/whole_node/detection_precision": None,
            "resource/projected_200_epoch_runtime_hours": None,
        },
    )
    archive.write_table(
        "metrics/history",
        [{"epoch": 1, "train/loss": 0.3}],
    )
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split": "full_core_fit_no_holdout"},
    )
    archive.write_text(
        "provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n'
    )
    fit_prediction = {**_prediction(run_id), "split": "fit"}
    archive.write_predictions("fit", [fit_prediction])

    final_path = archive.finalize_success()

    assert (final_path / "_SUCCESS").is_file()
    assert verify_run_bundle(final_path)["status"] == "success"
    assert not any((final_path / "predictions").glob("validation.*"))
    final_metrics = json.loads(
        (final_path / "metrics/final.json").read_text(encoding="utf-8")
    )
    assert final_metrics["fit/whole_node/detection_precision"] is None
    assert final_metrics["resource/projected_200_epoch_runtime_hours"] is None


def test_held_out_protocol_requires_and_accepts_canonical_test_predictions(
    tmp_path: Path,
) -> None:
    run_id = _run_id("testrole")
    metric = "test/observed_near_component_equal_mse"
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "trainer": {"primary_checkpoint_role": "last", "restore_best": False},
            "evaluation": {
                "protocol": "held_out_geometry_masked_reconstruction",
                "canonical_prediction_split": "test",
                "statistical_partition": "outer_geometry_test",
                "primary_metric": metric,
            },
        },
    )
    archive.write_summary(
        {
            "status": "completed",
            "primary_metric_name": metric,
            "primary_metric_value": 0.2,
        }
    )
    archive.write_bytes("checkpoints/last.ckpt", b"held-out-test")
    archive.append_metric_event(
        {"name": metric, "value": 0.2, "step": 0, "split": "test"}
    )
    archive.write_json("metrics/final.json", {metric: 0.2})
    archive.write_table("metrics/history", [{"epoch": 1, "train/loss": 0.3}])
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "test"})
    archive.write_text("provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n')
    archive.write_predictions("test", [{**_prediction(run_id), "split": "test"}])

    final_path = archive.finalize_success()

    assert verify_run_bundle(final_path)["status"] == "success"
    assert not any((final_path / "predictions").glob("validation.*"))


def test_success_rejects_null_configured_primary_metric(tmp_path: Path) -> None:
    run_id = _run_id("nullprimary")
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "model": {"name": "g1"},
            "evaluation": {"primary_metric": "val/masked_huber"},
        },
    )
    archive.write_summary({"status": "completed"})
    archive.write_bytes("checkpoints/best.ckpt", b"checkpoint")
    archive.append_metric_event(
        {"name": "val/masked_huber", "value": 0.2, "step": 1}
    )
    archive.write_json(
        "metrics/final.json",
        {"val/masked_huber": None, "val/auxiliary": 0.2},
    )
    archive.write_table("metrics/history", [{"epoch": 1, "train/loss": 0.3}])
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "test"})
    archive.write_text(
        "provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n'
    )
    archive.write_predictions("validation", [_prediction(run_id)])

    with pytest.raises(RunValidationError, match="primary.*finite numeric"):
        archive.finalize_success()


def test_held_in_protocol_rejects_held_out_artifacts(tmp_path: Path) -> None:
    run_id = _run_id("fitmixed")
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "trainer": {
                "primary_checkpoint_role": "last",
                "restore_best": False,
            },
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
                "canonical_prediction_split": "fit",
            },
        },
    )
    archive.write_summary({"status": "completed"})
    archive.write_bytes("checkpoints/last.ckpt", b"last")
    archive.append_metric_event(
        {"name": "fit/whole_node/masked_huber", "value": 0.2, "step": 1}
    )
    archive.write_json(
        "metrics/final.json",
        {
            "fit/whole_node/masked_huber": 0.2,
            "val/masked_huber": 0.1,
        },
    )
    archive.write_table("metrics/history", [{"epoch": 1, "train/loss": 0.3}])
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "fit"})
    archive.write_text(
        "provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n'
    )
    archive.write_predictions("fit", [{**_prediction(run_id), "split": "fit"}])

    with pytest.raises(RunValidationError, match="held-out final metrics"):
        archive.finalize_success()


def test_success_refuses_missing_or_empty_checkpoint(tmp_path: Path) -> None:
    archive = RunArchive.create(
        _run_id("missing1"),
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={"model": {"name": "g1"}},
    )
    archive.write_summary({"status": "completed"})
    _write_success_support(archive)
    with pytest.raises(RunValidationError, match="best.ckpt"):
        archive.finalize_success()

    archive.write_bytes("checkpoints/best.ckpt", b"")
    with pytest.raises(RunValidationError, match="empty"):
        archive.finalize_success()


def test_failed_run_preserves_partial_output_and_traceback(tmp_path: Path) -> None:
    archive = RunArchive.create(
        _run_id("failure1"),
        paths=_paths(tmp_path),
        resolved_config={"model": {"name": "g1"}},
    )
    archive.append_metric_event({"name": "train/loss", "value": 1.2, "step": 1})

    final_path = archive.finalize_failure(
        RuntimeError("synthetic failure"),
        failure_category="nonzero_exit",
    )

    assert (final_path / "_FAILED").is_file()
    assert "synthetic failure" in (
        final_path / "logs" / "exception.txt"
    ).read_text(encoding="utf-8")
    assert (final_path / "metrics" / "events.jsonl").is_file()
    assert verify_run_bundle(final_path)["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "value", "suffix"),
    (
        ("status", "success", "markerstatus"),
        ("run_id", "r_wrong", "markerrunid"),
        ("content_sha256", "0" * 64, "markerdigest"),
    ),
)
def test_completion_marker_is_cryptographically_bound(
    tmp_path: Path, field: str, value: str, suffix: str
) -> None:
    archive = RunArchive.create(_run_id(suffix), paths=_paths(tmp_path))
    final_path = archive.finalize_failure(
        RuntimeError("synthetic failure"),
        failure_category="nonzero_exit",
    )
    marker = final_path / "_FAILED"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload[field] = value
    marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RunValidationError, match="Completion marker"):
        verify_run_bundle(final_path)


def test_run_files_are_never_overwritten(tmp_path: Path) -> None:
    archive = RunArchive.create(_run_id("collision1"), paths=_paths(tmp_path))
    archive.write_summary({"status": "running"})

    with pytest.raises(RunArchiveError, match="overwrite"):
        archive.write_summary({"status": "different"})

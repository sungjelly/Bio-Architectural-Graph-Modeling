from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import (
    ATTENTION_NICHE_VISUALIZATION_PATCH_MODE,
    ATTENTION_NICHE_VISUALIZATION_PATCH_OUTPUTS,
    ATTENTION_NICHE_VISUALIZATION_PATCH_SPEC,
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


def _write_analysis_success_support(
    archive: RunArchive,
    *,
    metric: str = "analysis/attention_niche_qc_pass_fraction",
) -> None:
    archive.write_summary(
        {
            "status": "success",
            "primary_metric_name": metric,
            "primary_metric_value": 1.0,
        }
    )
    archive.append_metric_event({"name": metric, "value": 1.0, "step": 0})
    archive.write_json("metrics/final.json", {metric: 1.0})
    archive.write_table("metrics/history", [{"step": 0, metric: 1.0}])
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
        "held_in_pooled_so1_14core_relative_qkv_plateau_min150",
        "held_in_pooled_6core_relative_qkv_fixed_budget",
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


def test_analysis_only_run_requires_outputs_but_not_checkpoint_or_predictions(
    tmp_path: Path,
) -> None:
    run_id = _run_id("analysis1")
    metric = "analysis/attention_niche_qc_pass_fraction"
    archive = RunArchive.create(
        run_id,
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "evaluation": {
                "protocol": "posthoc_attention_routing_niche_v1",
                "artifact_contract": "analysis_only",
                "canonical_prediction_split": "analysis",
                "primary_metric": metric,
            },
            "metadata": {
                "required_analysis_outputs": ["analysis_manifest.yaml"]
            },
        },
    )
    archive.write_summary(
        {
            "status": "success",
            "primary_metric_name": metric,
            "primary_metric_value": 1.0,
        }
    )
    archive.append_metric_event({"name": metric, "value": 1.0, "step": 0})
    archive.write_json("metrics/final.json", {metric: 1.0})
    archive.write_table("metrics/history", [{"step": 0, metric: 1.0}])
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "test"})
    archive.write_text("provenance/command.txt", '{"argv":["test"],"cwd":"/tmp"}\n')

    with pytest.raises(RunValidationError, match="missing required output"):
        archive._validate_success_ready()
    archive.write_text("analysis_manifest.yaml", "status: complete\n")
    final_path = archive.finalize_success()

    assert not (final_path / "checkpoints/best.ckpt").exists()
    assert not any((final_path / "predictions").glob("analysis.*"))
    assert verify_run_bundle(final_path)["status"] == "success"


def _visualization_patch_archive(
    tmp_path: Path,
    *,
    suffix: str,
    mode: str,
) -> RunArchive:
    archive = RunArchive.create(
        _run_id(suffix),
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "campaign": {
                "campaign_id": (
                    "cmp_20260825_six_core_attention_routing_niches"
                )
            },
            "evaluation": {
                "protocol": "posthoc_attention_routing_niche_v1",
                "artifact_contract": "analysis_only",
                "canonical_prediction_split": "analysis",
                "primary_metric": (
                    "analysis/attention_niche_qc_pass_fraction"
                ),
            },
            "metadata": {
                "required_analysis_outputs": [
                    "scientific_table_that_must_not_be_copied.parquet"
                ]
            },
            "launcher": {
                "visualization_patch": {
                    **ATTENTION_NICHE_VISUALIZATION_PATCH_SPEC,
                    "mode": mode,
                },
            },
        },
    )
    _write_analysis_success_support(archive)
    return archive


def test_visualization_patch_requires_only_fresh_figure_bundle(
    tmp_path: Path,
) -> None:
    archive = _visualization_patch_archive(
        tmp_path,
        suffix="figpatch",
        mode=ATTENTION_NICHE_VISUALIZATION_PATCH_MODE,
    )
    for relative in ATTENTION_NICHE_VISUALIZATION_PATCH_OUTPUTS:
        archive.write_text(relative, "fresh visualization patch output\n")

    archive._validate_success_ready()


def test_visualization_patch_mode_drift_and_missing_figure_fail_closed(
    tmp_path: Path,
) -> None:
    drifted = _visualization_patch_archive(
        tmp_path / "drifted",
        suffix="figdrift",
        mode="verified_completed_run_visualization_only_v2",
    )
    for relative in ATTENTION_NICHE_VISUALIZATION_PATCH_OUTPUTS:
        drifted.write_text(relative, "fresh visualization patch output\n")
    with pytest.raises(RunValidationError, match="Unsupported.*patch mode"):
        drifted._validate_success_ready()

    missing = _visualization_patch_archive(
        tmp_path / "missing",
        suffix="figmiss1",
        mode=ATTENTION_NICHE_VISUALIZATION_PATCH_MODE,
    )
    for relative in ATTENTION_NICHE_VISUALIZATION_PATCH_OUTPUTS[1:]:
        missing.write_text(relative, "fresh visualization patch output\n")
    with pytest.raises(RunValidationError, match="missing required output"):
        missing._validate_success_ready()


def test_normal_analysis_only_required_outputs_remain_unchanged(
    tmp_path: Path,
) -> None:
    archive = RunArchive.create(
        _run_id("normalan"),
        paths=_paths(tmp_path),
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config={
            "evaluation": {
                "protocol": "posthoc_attention_routing_niche_v1",
                "artifact_contract": "analysis_only",
                "canonical_prediction_split": "analysis",
                "primary_metric": (
                    "analysis/attention_niche_qc_pass_fraction"
                ),
            },
            "metadata": {"required_analysis_outputs": ["ordinary.txt"]},
        },
    )
    _write_analysis_success_support(archive)
    with pytest.raises(RunValidationError, match="missing required output"):
        archive._validate_success_ready()
    archive.write_text("ordinary.txt", "ordinary analysis output\n")
    archive._validate_success_ready()


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


def test_failed_run_retention_accepts_derived_payloads_but_protects_evidence(
    tmp_path: Path,
) -> None:
    archive = RunArchive.create(
        _run_id("failedret"),
        paths=_paths(tmp_path),
        resolved_config={"model": {"name": "g1"}},
    )
    derived = archive.write_bytes(
        "diagnostics/intermediate/directed_attention_edges.parquet", b"derived"
    )
    top_level = archive.write_bytes("mutual_attention_edges.parquet", b"edges")
    final_path = archive.finalize_failure(
        RuntimeError("synthetic failure"),
        failure_category="nonzero_exit",
    )
    manifest = json.loads(
        (final_path / "provenance/artifact_checksums.json").read_text(
            encoding="utf-8"
        )
    )["files"]
    relatives = (
        derived.relative_to(archive.scratch_path).as_posix(),
        top_level.relative_to(archive.scratch_path).as_posix(),
    )
    tombstones = {relative: manifest[relative] for relative in relatives}
    for relative in relatives:
        (final_path / relative).unlink()

    verified = verify_run_bundle(final_path, tombstoned_artifacts=tombstones)
    assert verified["status"] == "failed"
    assert verified["tombstoned_file_count"] == 2

    config_relative = "config.resolved.yaml"
    with pytest.raises(RunValidationError, match="compact audit evidence"):
        verify_run_bundle(
            final_path,
            tombstoned_artifacts={
                **tombstones,
                config_relative: manifest[config_relative],
            },
        )

    compact = "diagnostics/pre_failure_summary.json"
    with pytest.raises(RunValidationError, match="compact audit evidence"):
        verify_run_bundle(
            final_path,
            tombstoned_artifacts={
                **tombstones,
                compact: {"type": "file", "size": 1, "sha256": "0" * 64},
            },
        )


def test_successful_run_retention_rejects_diagnostic_tombstone(tmp_path: Path) -> None:
    run_id = _run_id("successdiag")
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
    diagnostic = archive.write_bytes("diagnostics/intermediate.bin", b"derived")
    relative = diagnostic.relative_to(archive.scratch_path).as_posix()
    final_path = archive.finalize_success()
    manifest = json.loads(
        (final_path / "provenance/artifact_checksums.json").read_text(
            encoding="utf-8"
        )
    )["files"]
    (final_path / relative).unlink()

    with pytest.raises(RunValidationError, match="Successful-run"):
        verify_run_bundle(
            final_path,
            tombstoned_artifacts={relative: manifest[relative]},
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

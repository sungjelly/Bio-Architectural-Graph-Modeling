from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import (
    RunArchive,
    RunArchiveError,
    RunImmutableError,
    RunValidationError,
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

"""Deferred test-only analysis cannot start early or escape its owned outputs."""

from contextlib import ExitStack
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "paper_test_leiden_controller", ROOT / "scripts/analysis/run_so12_paper_test_leiden.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def paths(tmp_path):
    return ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})


@pytest.fixture
def registry(paths):
    result = Registry(paths.state_root / "tracking/test.sqlite3")
    campaign = "cmp_synthetic_training_source"
    config = {"seed": 0}
    identity = scientific_id(config)
    result.create_campaign(campaign, name="source fixture")
    result.register_variant(identity, campaign_id=campaign, configuration=config)
    result.create_run(
        MODULE.RUN, campaign_id=campaign, scientific_id=identity, repro_id="rep_synthetic",
        seed=0, fold=0, attempt=1, configuration=config, status="running",
        artifact_path=paths.artifact_root / "runs/2026/10" / MODULE.RUN,
    )
    return result


@pytest.fixture
def pending(registry, paths):
    out = paths.scratch_root / "active_runs" / MODULE.EVALUATION
    config = {"source_run_id": MODULE.RUN, "test_cores": [15, 23], "analysis_source_sha256": {}}
    MODULE.initialize_evaluation(registry, paths, out, config)
    return out, config


@pytest.mark.parametrize("status", ["pending", "running", "finalizing"])
def test_pending_source_never_inspects_live_bundle_or_checkpoint(paths, monkeypatch, status):
    registry = SimpleNamespace(get_run=lambda run_id: {"status": status, "artifact_path": "/live"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Waiting must inspect registry status only")
    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(MODULE, "record", forbidden)
    assert MODULE.completed_source(registry, paths) is None


@pytest.mark.parametrize("status", ["failed", "cancelled", "pruned", "stale"])
def test_failed_source_cannot_start_analysis(paths, status):
    registry = SimpleNamespace(get_run=lambda run_id: {"status": status})
    with pytest.raises(RuntimeError, match="ended without success"):
        MODULE.completed_source(registry, paths)


def test_completed_source_requires_exact_run_archive_and_success_marker(paths):
    expected = paths.artifact_root / "runs/2026/10" / MODULE.RUN
    row = {"status": "completed", "artifact_path": str(expected)}
    registry = SimpleNamespace(get_run=lambda run_id: row)
    with pytest.raises(RuntimeError, match="success marker"):
        MODULE.completed_source(registry, paths)
    expected.mkdir(parents=True)
    (expected / "_SUCCESS").write_text("synthetic marker; full verifier runs later under locks")
    assert MODULE.completed_source(registry, paths) == expected.resolve()
    row["artifact_path"] = str(paths.scratch_root / "active_runs" / MODULE.RUN)
    with pytest.raises(RuntimeError, match="archive path"):
        MODULE.completed_source(registry, paths)


def test_waiting_evaluation_restarts_without_rewriting_scientific_configuration(registry, paths, pending):
    out, config = pending
    before = (out / "config.resolved.json").read_bytes()
    MODULE.initialize_evaluation(registry, paths, out, config)
    row = MODULE.evaluation_row(registry)
    assert row["status"] == "pending"
    assert row["checkpoint_name"] == "best"
    assert row["run_id"] == MODULE.RUN
    assert (out / "config.resolved.json").read_bytes() == before
    assert not (out / "COMPUTATION_STARTED.json").exists()


def test_pending_analysis_rejects_changed_source_configuration(registry, paths, pending):
    out, config = pending
    with pytest.raises(RuntimeError, match="configuration/source changed"):
        MODULE.initialize_evaluation(registry, paths, out, {**config, "test_cores": [15]})


def test_partial_computation_cannot_silently_restart(registry, paths, pending):
    out, config = pending
    (out / "COMPUTATION_STARTED.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Partial computation"):
        MODULE.initialize_evaluation(registry, paths, out, config)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "another_dataset"), ("split_id", "another_split"),
    ("checkpoint_name", "last"), ("artifact_path", "/unrelated/output"),
    ("status", "running"), ("status", "completed"),
])
def test_existing_evaluation_identity_and_state_are_locked(registry, paths, pending, field, value):
    out, config = pending
    # Field names are fixed test parameters, never external input.
    with registry.transaction() as connection:
        connection.execute(f"UPDATE evaluations SET {field}=? WHERE evaluation_id=?", (value, MODULE.EVALUATION))
    with pytest.raises(RuntimeError):
        MODULE.initialize_evaluation(registry, paths, out, config)


def test_unregistered_existing_output_is_preserved(registry, paths):
    out = paths.scratch_root / "active_runs" / MODULE.EVALUATION
    out.mkdir(parents=True)
    proof = out / "preserved.txt"
    proof.write_text("existing output")
    with pytest.raises(RuntimeError, match="Unregistered"):
        MODULE.initialize_evaluation(registry, paths, out, {})
    assert proof.read_text() == "existing output"
    assert MODULE.evaluation_row(registry) is None


@pytest.mark.parametrize("name", ["bagm-worker.lock", "bagm-worker-gpu-0.lock"])
def test_analysis_waits_for_global_exclusive_or_its_gpu_owner(paths, name):
    locks = paths.state_root / "locks"
    locks.mkdir(parents=True)
    with (locks / name).open("a") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as stack, pytest.raises(BlockingIOError):
            MODULE.acquire_gpu(stack, paths)


def test_analysis_does_not_lock_another_independent_gpu(paths):
    locks = paths.state_root / "locks"
    locks.mkdir(parents=True)
    with (locks / "bagm-worker.lock").open("a") as shared, (locks / "bagm-worker-gpu-1.lock").open("a") as other:
        fcntl.flock(shared.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as stack:
            MODULE.acquire_gpu(stack, paths)
            with (locks / "bagm-worker-gpu-0.lock").open("a") as duplicate:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(duplicate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_interruption_drains_real_child_before_resource_lock_scope_can_exit():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
    )
    stop = Event()
    stop.set()
    try:
        with pytest.raises(InterruptedError, match="child exit"):
            MODULE.wait_child(child, stop)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()


def test_nonzero_child_exit_cannot_publish_success():
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(4)"], start_new_session=True)
    with pytest.raises(RuntimeError, match="code 4"):
        MODULE.wait_child(child, Event())


def test_publication_registration_is_atomic_and_idempotent(registry, paths, pending):
    _, config = pending
    final = paths.report_root / "analyses/so12_paper_test_leiden" / MODULE.EVALUATION
    final.mkdir(parents=True)
    manifest = {"summary": {"test_cells": 81607, "test_cores": [15, 23]}}
    MODULE.write(final / "manifest.json", manifest)
    MODULE.register_publication(registry, final, manifest)
    MODULE.register_publication(registry, final, manifest)
    row = MODULE.evaluation_row(registry)
    assert row["status"] == "completed" and row["artifact_path"] == str(final)
    with registry.connect() as connection:
        records = connection.execute("SELECT * FROM artifacts WHERE evaluation_id=?", (MODULE.EVALUATION,)).fetchall()
    assert len(records) == 1
    assert records[0]["sha256"] == MODULE.sha256_file(final / "manifest.json")
    MODULE.write(final / "manifest.json", {"summary": {"test_cells": 1}})
    with pytest.raises(RuntimeError, match="manifest identity changed"):
        MODULE.register_publication(registry, final, manifest)


def test_empty_manifest_cannot_certify_an_analysis(paths):
    root = paths.scratch_root / "active_runs" / MODULE.EVALUATION
    root.mkdir(parents=True)
    config = {"source_run_id": MODULE.RUN}
    MODULE.write(root / "config.resolved.json", config)
    MODULE.write(root / "manifest.json", {
        "evaluation_id": MODULE.EVALUATION, "source_run_id": MODULE.RUN,
        "summary": {}, "files": {},
    })
    MODULE.write(root / "_SUCCESS", {"manifest_sha256": MODULE.sha256_file(root / "manifest.json")})
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)


def _seal_fixture(root, config):
    """Publish only a synthetic inventory; numerical array checks have own tests."""
    summary = {
        "test_cells": 81607, "test_cores": [15, 23], "representation": "hL",
        "extraction": MODULE.read(root / "extraction_manifest.json"),
        "clustering": MODULE.read(root / "clustering_summary.json"),
        "rendering": MODULE.read(root / "rendering_summary.json"),
    }
    MODULE.write(root / "summary.json", summary)
    manifest = {
        "schema": "so12_paper_test_leiden_report_v1", "evaluation_id": MODULE.EVALUATION,
        "source_run_id": MODULE.RUN,
        "summary": summary,
        "files": {str(path.relative_to(root)): MODULE.record(path)
                  for path in root.rglob("*") if path.is_file()
                  and path.name not in {"manifest.json", "_SUCCESS"}},
    }
    MODULE.write(root / "manifest.json", manifest)
    MODULE.write(root / "_SUCCESS", {"manifest_sha256": MODULE.sha256_file(root / "manifest.json")})
    return manifest


@pytest.fixture
def published(paths):
    root = paths.scratch_root / "active_runs" / MODULE.EVALUATION
    root.mkdir(parents=True)
    config = {"source_run_id": MODULE.RUN, "test_cores": [15, 23]}
    for relative in MODULE.REQUIRED_OUTPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic payload")
    MODULE.write(root / "config.resolved.json", config)
    summary = {
        "status": "complete", "passed": True, "n_cells": 81607,
        "test_core_numbers": [15, 23], "cells_per_core": {"15": 38145, "23": 43462},
    }
    extraction = {
        **summary, "verified": True, "test_only": True, "normalization_refit": False,
        "source_run_id": MODULE.RUN, "dimensions": 256, "all_observed": True,
        "source_modified": False, "core_numbers": [15, 23],
        "representation": "fourth_graph_block_output_before_decoder",
        "checkpoint_sha256": "a" * 64, "preprocessing_fingerprint": "b" * 64,
        "fitted_parameters_sha256": "c" * 64,
        "cores": [{"core_number": core, "n_cells": count}
                  for core, count in ((15, 38145), (23, 43462))],
    }
    MODULE.write(root / "extraction_manifest.json", extraction)
    MODULE.write(root / "clustering_summary.json", summary)
    MODULE.write(root / "rendering_summary.json", summary)
    _seal_fixture(root, config)
    return root, config


def test_complete_publication_inventory_verifies(published):
    root, config = published
    assert MODULE.verify_published(root, config)["evaluation_id"] == MODULE.EVALUATION


def test_publication_requires_outputs_even_if_manifest_is_resealed(published):
    root, config = published
    (root / "clustering/hL_labels_seed1.npy").unlink()
    _seal_fixture(root, config)
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)


def test_publication_rejects_extra_unlisted_file(published):
    root, config = published
    (root / "unlisted.txt").write_text("late extra output")
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)


def test_publication_rejects_symlink_even_when_checksum_matches(published, tmp_path):
    root, config = published
    target = root / "embeddings/hL.npy"
    outside = tmp_path / "outside.npy"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)


def test_publication_rejects_modified_saved_output(published):
    root, config = published
    (root / "cells.csv.gz").write_text("modified")
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)


@pytest.mark.parametrize("filename,field,value", [
    ("extraction_manifest.json", "source_run_id", "another_run"),
    ("extraction_manifest.json", "verified", False),
    ("extraction_manifest.json", "dimensions", 32),
    ("clustering_summary.json", "test_core_numbers", [15]),
    ("clustering_summary.json", "passed", False),
    ("rendering_summary.json", "n_cells", 38145),
    ("rendering_summary.json", "cells_per_core", {"15": 38145, "23": 0}),
])
def test_semantic_stage_coverage_cannot_be_replaced_by_file_checksums(published, filename, field, value):
    root, config = published
    summary = MODULE.read(root / filename)
    summary[field] = value
    MODULE.write(root / filename, summary)
    _seal_fixture(root, config)
    with pytest.raises(RuntimeError):
        MODULE.verify_published(root, config)

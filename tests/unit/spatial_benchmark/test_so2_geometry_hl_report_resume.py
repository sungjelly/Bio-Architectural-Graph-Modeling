"""Publication recovery must preserve completed results and source provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def report_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/analysis/create_so2_geometry_hl_map.py"
    )
    spec = importlib.util.spec_from_file_location("_geometry_hl_resume_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = SimpleNamespace(
        project_root=tmp_path / "project",
        scratch_root=tmp_path / "scratch",
        report_root=tmp_path / "reports",
    )
    registry = object()
    monkeypatch.setattr(module, "current_paths", lambda: paths)
    monkeypatch.setattr(module, "Registry", Mock(return_value=registry))
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(
        module, "resolve_inputs", Mock(side_effect=AssertionError("Source reload ran"))
    )
    monkeypatch.setattr(
        module, "run_jobs", Mock(side_effect=AssertionError("Extraction ran"))
    )
    monkeypatch.setattr(module, "register_report", Mock())
    stage = (
        paths.scratch_root / "active_runs" / module.RUN_ID
        / "posthoc_reports" / module.REPORT / "v1"
    )
    final = paths.report_root / "analyses" / module.REPORT / module.RUN_ID / "v1"
    return module, paths, registry, stage, final


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_completed_staging_publishes_without_rerunning_extraction(
    report_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _paths, registry, stage, final = report_script
    monkeypatch.setattr(module, "SO2_CORE_NUMBERS", (15,))
    checkpoint_sha = "a" * 64
    record = {"core_number": 15, "alias": "SO2-C15", "cell_count": 2}
    embedding = stage / "embeddings/core_15_hL.npz"
    hl = np.arange(512, dtype=np.float32).reshape(2, 256)
    module._write_deterministic_npz(
        embedding,
        {
            "cell_index": np.arange(2, dtype=np.int64),
            "core_number": np.asarray(15, dtype=np.int16),
            "coordinates_um": np.asarray([[0.0, 1.0], [2.0, 3.0]]),
            "hL": hl,
        },
    )
    receipt = module._receipt_with_self_hash(
        {
            "run_id": module.RUN_ID,
            "checkpoint_sha256": checkpoint_sha,
            "core_number": 15,
            "source_core": record,
            "embedding_file": embedding.relative_to(stage).as_posix(),
            "file": module._file_record(embedding),
            "hL_array_sha256": module._array_sha256("hL", hl),
        }
    )
    module._atomic_write_json(stage / "embeddings/core_15_receipt.json", receipt)
    module._atomic_write_json(
        stage / "embeddings/extraction_manifest.json", {"cores": [receipt]}
    )
    png = "figures/synthetic.png"
    (stage / "figures").mkdir()
    Image.new("RGB", (2, 2), "white").save(stage / png)
    module._atomic_write_text(stage / "events.jsonl", '{"event":"complete"}\n')
    manifest = module._receipt_with_self_hash(
        {
            "run_id": module.RUN_ID,
            "status": "complete",
            "provenance": {"checkpoint": {"sha256": checkpoint_sha}},
            "cluster_count": 1,
            "png": png,
            "files": module._file_manifest(stage),
        }
    )
    module._atomic_write_json(stage / "manifest.json", manifest)
    module._atomic_write_text(
        stage / "_SUCCESS", module.sha256_file(stage / "manifest.json") + "\n"
    )
    assert module.verify_report(stage) == manifest
    original = _tree_bytes(stage)

    module.main()

    assert not stage.exists()
    assert _tree_bytes(final) == original
    assert module.verify_report(final) == manifest
    module.register_report.assert_called_once_with(registry, final, manifest)
    module.resolve_inputs.assert_not_called()
    module.run_jobs.assert_not_called()


def test_changed_source_fails_before_writing_existing_provenance(
    report_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, paths, _registry, stage, final = report_script
    relative = Path("src/spatial_benchmark/synthetic_model.py")
    source = paths.project_root / relative
    snapshot = stage / "source" / relative
    source.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    source.write_text("MODEL_VERSION = 2\n")
    snapshot.write_text("MODEL_VERSION = 1\n")
    for name in ("provenance.json", "environment.json", "config.resolved.json"):
        (stage / name).write_text('{"original_attempt": true}\n')
    (stage / "events.jsonl").write_text('{"event":"extraction_interrupted"}\n')
    original = _tree_bytes(stage)
    original_mtimes = {
        path.relative_to(stage).as_posix(): path.stat().st_mtime_ns
        for path in stage.rglob("*")
        if path.is_file()
    }
    event = Mock(side_effect=AssertionError("Failure guard wrote an event"))
    monkeypatch.setattr(module, "event", event)

    with pytest.raises(RuntimeError, match="Source changed since this extraction began"):
        module.main()

    assert _tree_bytes(stage) == original
    assert {
        path.relative_to(stage).as_posix(): path.stat().st_mtime_ns
        for path in stage.rglob("*")
        if path.is_file()
    } == original_mtimes
    assert not final.exists()
    event.assert_not_called()
    module.resolve_inputs.assert_not_called()
    module.run_jobs.assert_not_called()
    module.register_report.assert_not_called()

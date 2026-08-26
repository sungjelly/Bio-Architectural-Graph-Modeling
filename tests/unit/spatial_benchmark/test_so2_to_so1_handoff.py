from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "handoff_so2_to_so1_relative_qkv",
    PROJECT_ROOT / "scripts/train/handoff_so2_to_so1_relative_qkv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_HANDOFF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HANDOFF
_SPEC.loader.exec_module(_HANDOFF)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _so2_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "_SUCCESS").touch()
    checkpoint = root / "checkpoints/last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        root / "summary.json",
        {
            "run_id": _HANDOFF.SO2_RUN_ID,
            "status": "success",
            "final_epoch": 300,
            "optimizer_steps": 2100,
            "fixed_epoch_target_completed": True,
            "checkpoint_reload_verified": True,
            "checkpoint": "checkpoints/last.ckpt",
            "world_size": 4,
        },
    )
    _write_json(
        root / "metrics/final.json",
        {"fit/training/final_global_epoch": 300.0},
    )
    _write_json(
        root / "diagnostics/final_checkpoint_reload_verification.json",
        {
            "verified": True,
            "checkpoint_file_sha256": _HANDOFF.sha256_file(checkpoint),
            "payload": {
                "completed_global_epochs": 300,
                "optimizer_updates_completed": 2100,
                "full_resume_payload_validated": True,
                "fixed_completion_payload_validated": True,
            },
            "fixed_prediction_replay": {"verified": True},
        },
    )
    with (root / "results/epoch_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("schema", "global_epoch"))
        writer.writeheader()
        for epoch in range(1, 301):
            writer.writerow(
                {"schema": "so2_14core_epoch_metrics_v1", "global_epoch": epoch}
            )
    return root


def test_so2_final_bundle_requires_exact_epoch300_completion(tmp_path: Path) -> None:
    root = _so2_bundle(tmp_path)
    receipt = _HANDOFF.validate_so2_final_bundle(root)
    assert receipt["epoch_rows"] == 300
    assert receipt["final_epoch"] == 300
    assert receipt["checkpoint"].endswith("checkpoints/last.ckpt")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary["final_epoch"] = 299
    _write_json(root / "summary.json", summary)
    with pytest.raises(_HANDOFF.HandoffError, match="final_epoch"):
        _HANDOFF.validate_so2_final_bundle(root)


def test_first_so1_epoch_requires_14_core_140_view_single_epoch() -> None:
    row = {
        "schema": "so1_14core_epoch_metrics_v1",
        "run_id": "r_test",
        "model_seed": "0",
        "global_epoch": "1",
        "optimizer_updates_this_epoch": "7",
        "cumulative_optimizer_updates": "7",
        "complete_graph_mask_views": "140",
        "equal_core_mean_masked_huber": "0.25",
    }
    for alias in _HANDOFF.SO1_CORE_ALIASES:
        row["loss_" + alias.lower().replace("-", "_")] = "0.25"
    receipt = _HANDOFF.validate_so1_first_epoch_row(row)
    assert receipt["optimizer_updates"] == 7
    assert receipt["complete_graph_mask_views"] == 140

    row["complete_graph_mask_views"] = "139"
    with pytest.raises(_HANDOFF.HandoffError, match="complete_graph_mask_views"):
        _HANDOFF.validate_so1_first_epoch_row(row)


def test_gpu_idle_gate_requires_all_four_empty_cards() -> None:
    idle = {
        "gpus": [
            {"index": index, "memory_used_mib": 1, "utilization_percent": 0}
            for index in range(4)
        ],
        "compute_processes": [],
    }
    assert _HANDOFF._gpu_snapshot_is_idle(idle)
    busy = {**idle, "compute_processes": ["123, GPU-0, 100"]}
    assert not _HANDOFF._gpu_snapshot_is_idle(busy)

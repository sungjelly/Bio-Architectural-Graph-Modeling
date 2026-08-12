from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest


_PATH = Path(__file__).resolve().parents[3] / "scripts/train/launch_matched_graph_context.py"
_SPEC = importlib.util.spec_from_file_location("launch_matched_graph_context", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fake_authorities() -> dict[str, Any]:
    return {
        "contract": {"path": "contract", "sha256": _MODULE.CONTRACT_SHA256},
        "prepared_root": "prepared",
    }


def _selection_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "matched_graph_context_stage_a_selection",
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "contract_sha256": _MODULE.CONTRACT_SHA256,
        "candidate_design_sha256": _MODULE.candidate_design_sha256(),
        "test_metrics_used_for_selection": False,
        "cross_outer_pooling": False,
        "payload_sha256": "synthetic",
        "advanced_by_outer_fold": {
            str(fold): {
                arm: [
                    {**_MODULE.CANDIDATE_BY_ID[candidate_id].as_dict(), "epoch": 12}
                    for candidate_id in ("c00", "c05", "c11")
                ]
                for arm in _MODULE.ARMS
            }
            for fold in _MODULE.FOLDS
        },
    }


def test_candidate_design_is_the_exact_frozen_sixteen_tuple_authority() -> None:
    assert [candidate.candidate_id for candidate in _MODULE.CANDIDATES] == [
        f"c{index:02d}" for index in range(16)
    ]
    assert _MODULE.CANDIDATE_BY_ID["c08"].as_dict() == {
        "candidate_id": "c08",
        "hidden_width": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.1,
    }
    assert {candidate.hidden_width for candidate in _MODULE.CANDIDATES} == {32, 64, 128}


def test_materialize_stage_a_has_complete_isolated_coverage_and_gate_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "_verify_authorities", lambda *_args, **_kwargs: _fake_authorities())

    def gate(_root: Path, path: Path) -> dict[str, Any]:
        payload = {
            "result": {"result_sha256": "a" * 64},
            "payload_sha256": "b" * 64,
            "passed": True,
        }
        _write_json(path, payload)
        return payload

    monkeypatch.setattr(_MODULE, "_materialize_synthetic_gate", gate)
    output = tmp_path / "stage_a.json"
    plan = _MODULE.materialize_stage_a(tmp_path, output)

    assert len(plan["jobs"]) == 4 * 16 * 4
    slots = {
        (job["arm"], job["candidate_id"], job["seed"], job["fold"])
        for job in plan["jobs"]
    }
    assert len(slots) == 256
    assert {job["seed"] for job in plan["jobs"]} == {_MODULE.STAGE_A_SEED}
    assert len({job["stdout_path"] for job in plan["jobs"]}) == 256
    assert len({job["result_path"] for job in plan["jobs"]}) == 256
    assert plan["authorities"]["synthetic_gate"]["result_sha256"] == "a" * 64


def test_stage_a_selects_each_outer_fold_without_cross_outer_pooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    jobs = [
        {
            "job_id": f"a.{arm}.{candidate.candidate_id}.s{_MODULE.STAGE_A_SEED}.f{fold}",
            "arm": arm,
            "candidate_id": candidate.candidate_id,
            "seed": _MODULE.STAGE_A_SEED,
            "fold": fold,
        }
        for arm in _MODULE.ARMS
        for candidate in _MODULE.CANDIDATES
        for fold in _MODULE.FOLDS
    ]
    plan = {"jobs": jobs, "plan_payload_sha256": "p" * 64}
    monkeypatch.setattr(_MODULE, "_load_plan", lambda *_args, **_kwargs: plan)
    curves: dict[tuple[str, str, int, int], dict[int, float]] = {}
    for job in jobs:
        candidate = _MODULE.CANDIDATE_BY_ID[job["candidate_id"]]
        stratum = [
            item for item in _MODULE.CANDIDATES
            if item.hidden_width == candidate.hidden_width
        ]
        winner = stratum[job["fold"] % len(stratum)].candidate_id
        value = 1.0 if job["candidate_id"] == winner else 2.0
        curves[(job["arm"], job["candidate_id"], job["seed"], job["fold"])] = {
            epoch: value + epoch / 1_000_000 for epoch in _MODULE.EPOCHS
        }
    monkeypatch.setattr(_MODULE, "_aggregate", lambda *_args: (curves, []))

    selected = _MODULE.select_stage_a(tmp_path, plan_path, tmp_path / "selected.json")
    assert selected["cross_outer_pooling"] is False
    assert set(selected["advanced_by_outer_fold"]) == {"0", "1", "2", "3"}
    for fold in _MODULE.FOLDS:
        for arm in _MODULE.ARMS:
            rows = selected["advanced_by_outer_fold"][str(fold)][arm]
            assert {row["hidden_width"] for row in rows} == {32, 64, 128}
    assert (
        selected["advanced_by_outer_fold"]["0"]["no_graph"][0]["candidate_id"]
        != selected["advanced_by_outer_fold"]["1"]["no_graph"][0]["candidate_id"]
    )


def test_stage_b_is_exact_nested_96_job_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    selection = _selection_value()
    monkeypatch.setattr(_MODULE, "_verify_authorities", lambda *_args, **_kwargs: _fake_authorities())
    monkeypatch.setattr(_MODULE, "_load_stage_a_selection", lambda *_args: selection)

    plan = _MODULE.materialize_stage_b(
        tmp_path, selection_path, tmp_path / "stage_b.json"
    )
    assert len(plan["jobs"]) == 96
    assert {
        (job["fold"], job["arm"], job["candidate_id"], job["seed"])
        for job in plan["jobs"]
    } == {
        (fold, arm, candidate, seed)
        for fold in _MODULE.FOLDS
        for arm in _MODULE.ARMS
        for candidate in ("c00", "c05", "c11")
        for seed in _MODULE.STAGE_B_SEEDS
    }


def test_tie_rule_prefers_seed_stability_then_larger_decay_and_lower_dropout() -> None:
    rows = [
        {
            **_MODULE.CANDIDATE_BY_ID["c05"].as_dict(),
            "epoch": 24,
            "validation_mean": 1.0,
            "seed_sd": 0.2,
        },
        {
            **_MODULE.CANDIDATE_BY_ID["c06"].as_dict(),
            "epoch": 24,
            "validation_mean": 1.002,
            "seed_sd": 0.1,
        },
    ]
    selected = _MODULE._near_tie_select(rows)
    assert selected["candidate_id"] == "c06"
    assert selected["near_tie_candidate_ids"] == ["c05", "c06"]


def test_candidate_epoch_near_tie_uses_seed_sd_before_epoch() -> None:
    candidate = _MODULE.CANDIDATE_BY_ID["c05"]
    curves: dict[tuple[str, str, int, int], dict[int, float]] = {}
    seeds = (20260812, 20261812, 20262812)
    for index, seed in enumerate(seeds):
        curve = {epoch: 2.0 for epoch in _MODULE.EPOCHS}
        curve[12] = (0.7, 1.0, 1.3)[index]  # mean 1.0, high seed SD
        curve[24] = 1.002  # within 0.25%, zero seed SD
        curves[("no_graph", "c05", seed, 0)] = curve
    rows = [
        {
            **candidate.as_dict(),
            "epoch": row["epoch"],
            "validation_mean": row["validation_mean"],
            "seed_sd": row["seed_sd"],
        }
        for row in _MODULE._candidate_epoch_rows(
            curves, "no_graph", "c05", seeds, (0,)
        )
    ]
    selected = _MODULE._near_tie_select(rows)
    assert selected["epoch"] == 24
    assert selected["validation_mean"] == pytest.approx(1.002)
    assert selected["seed_sd"] == pytest.approx(0.0)


def test_shared_width_regret_uses_raw_minimum_not_near_tie_selection() -> None:
    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    candidates = {32: "c00", 64: "c05", 128: "c11"}
    for arm in _MODULE.ARMS:
        for width, candidate_id in candidates.items():
            candidate = _MODULE.CANDIDATE_BY_ID[candidate_id]
            if width == 32:
                rows[(arm, width)] = [
                    {
                        **candidate.as_dict(), "epoch": 12,
                        "validation_mean": 1.0, "seed_sd": 0.4,
                    },
                    {
                        **candidate.as_dict(), "epoch": 24,
                        "validation_mean": 1.002, "seed_sd": 0.0,
                    },
                ]
            else:
                rows[(arm, width)] = [
                    {
                        **candidate.as_dict(), "epoch": 12,
                        "validation_mean": 1.001 if width == 64 else 1.2,
                        "seed_sd": 0.1,
                    }
                ]
    selected, diagnostics, shared_width = _MODULE._select_shared_width(rows)
    # The near-tie rule retains the more stable but slightly worse h32 epoch.
    assert selected[("no_graph", 32)]["validation_mean"] == pytest.approx(1.002)
    # Minimax regret still uses the contract's raw best_mse=1.0 at h32.
    assert shared_width == 32
    h32 = next(row for row in diagnostics if row["hidden_width"] == 32)
    assert h32["maximum_relative_regret"] == pytest.approx(0.0)


def test_lock_selection_emits_four_independent_parameter_matched_selections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / name for name in ("a.json", "s.json", "b.json")]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    selection = _selection_value()
    stage_a_jobs = [
        {
            "job_id": f"stage_a.{arm}.{candidate.candidate_id}.s{_MODULE.STAGE_A_SEED}.f{fold}",
            "arm": arm, "candidate_id": candidate.candidate_id,
            "seed": _MODULE.STAGE_A_SEED, "fold": fold,
        }
        for fold in _MODULE.FOLDS for arm in _MODULE.ARMS
        for candidate in _MODULE.CANDIDATES
    ]
    stage_b_jobs = [
        {
            "job_id": f"stage_b.{arm}.{candidate}.s{seed}.f{fold}",
            "arm": arm, "candidate_id": candidate, "seed": seed, "fold": fold,
        }
        for fold in _MODULE.FOLDS for arm in _MODULE.ARMS
        for candidate in ("c00", "c05", "c11")
        for seed in _MODULE.STAGE_B_SEEDS
    ]
    stage_a_plan = {"jobs": stage_a_jobs}
    stage_b_plan = {
        "jobs": stage_b_jobs,
        "source_stage_a_selection": {"sha256": _MODULE._sha256_file(paths[1])},
    }

    def load_plan(_root: Path, _path: Path, *, expected_stage: str) -> dict[str, Any]:
        return stage_a_plan if expected_stage == "stage_a" else stage_b_plan

    monkeypatch.setattr(_MODULE, "_load_plan", load_plan)
    monkeypatch.setattr(_MODULE, "_load_stage_a_selection", lambda *_args: selection)

    curves_a: dict[tuple[str, str, int, int], dict[int, float]] = {}
    curves_b: dict[tuple[str, str, int, int], dict[int, float]] = {}
    desired_width = {0: 32, 1: 64, 2: 128, 3: 64}
    for job in (*stage_a_jobs, *stage_b_jobs):
        candidate = _MODULE.CANDIDATE_BY_ID[job["candidate_id"]]
        base = 1.0 + abs(candidate.hidden_width - desired_width[job["fold"]]) / 100.0
        curve = {epoch: base + abs(epoch - 48) / 100_000 for epoch in _MODULE.EPOCHS}
        key = (job["arm"], job["candidate_id"], job["seed"], job["fold"])
        (curves_a if job["seed"] == _MODULE.STAGE_A_SEED else curves_b)[key] = curve

    def aggregate(_root: Path, plan: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
        curves = curves_a if plan is stage_a_plan else curves_b
        sources = [
            {
                "job_id": job["job_id"], "path": f"results/{job['job_id']}.json",
                "sha256": _MODULE._canonical_sha256(job),
                "success_marker": f"results/{job['job_id']}/_SUCCESS",
                "success_marker_sha256": "f" * 64,
            }
            for job in plan["jobs"]
        ]
        return curves, sources

    monkeypatch.setattr(_MODULE, "_aggregate", aggregate)
    receipt = _MODULE.lock_selection(
        tmp_path, paths[0], paths[1], paths[2], tmp_path / "receipt.json"
    )

    assert receipt["cross_outer_pooling"] is False
    assert set(receipt["selected_by_outer_fold"]) == {"0", "1", "2", "3"}
    for fold, expected_width in desired_width.items():
        blocks = receipt["selected_by_outer_fold"][str(fold)]
        assert {block["config"]["hidden_width"] for block in blocks.values()} == {expected_width}
        assert len({block["parameter_count"] for block in blocks.values()}) == 1
        for block in blocks.values():
            assert block["config"]["batch_size"] == 4096
            assert block["config_sha256"] == _MODULE._canonical_sha256(block["config"])
        sources = receipt["source_tuning_results_by_outer_fold"][str(fold)]
        assert len(sources) == 88
        assert all(row["job_id"].endswith(f".f{fold}") for row in sources)


def test_tuning_reader_accepts_only_exact_nested_roles_and_rejects_test_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifacts/runs/2026/08/run"
    result_path = artifact / "results.json"
    marker = artifact / "_SUCCESS"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    job = {
        "run_id": "run", "arm": "no_graph", "candidate_id": "c00",
        "seed": _MODULE.STAGE_A_SEED, "fold": 2,
        "result_path": result_path.relative_to(tmp_path).as_posix(),
        "success_marker": marker.relative_to(tmp_path).as_posix(),
    }
    result = {
        "campaign_id": _MODULE.CAMPAIGN_ID, "contract_sha256": _MODULE.CONTRACT_SHA256,
        "mode": "tune", "run_id": "run", "arm": "no_graph",
        "seed": _MODULE.STAGE_A_SEED, "fold": 2,
        "config": {**_MODULE.CANDIDATE_BY_ID["c00"].as_dict(), "batch_size": 4096},
        "split_roles": {"train_folds": [0, 1], "validation_fold": 3, "excluded_fold": 2},
        "validation_by_epoch": [
            {"epoch": epoch, "validation_component_equal_mse": 1.0}
            for epoch in _MODULE.EPOCHS
        ],
    }
    _write_json(result_path, result)
    monkeypatch.setattr(_MODULE, "_verify_published_bundle", lambda _path: None)
    curve, _authority = _MODULE._read_tuning_job(tmp_path, job)
    assert set(curve) == set(_MODULE.EPOCHS)

    result["outer_test_mse"] = 0.1
    _write_json(result_path, result)
    with pytest.raises(_MODULE.SelectionError, match="prohibited outcome"):
        _MODULE._read_tuning_job(tmp_path, job)


def test_launcher_fails_closed_when_any_configured_gpu_is_externally_occupied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = object.__new__(_MODULE.FourGpuLauncher)
    launcher.root = tmp_path
    launcher.plan = {"minimum_free_disk_gb": 25.0, "jobs": []}
    monkeypatch.setattr(_MODULE.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100 * 1024**3))
    launcher._external_gpu_processes = lambda: {2}
    with pytest.raises(_MODULE.OrchestrationError, match="externally occupied"):
        launcher._preflight()


def test_synthetic_gate_is_checksum_bound_and_replay_checked(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    first = _MODULE._materialize_synthetic_gate(_PATH.parents[2], path)
    assert first["passed"] is True
    assert first["payload_sha256"] == _MODULE._canonical_sha256(
        {key: value for key, value in first.items() if key != "payload_sha256"}
    )
    assert _MODULE._materialize_synthetic_gate(_PATH.parents[2], path)["result"] == first["result"]
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["result"]["passed"] = False
    _write_json(path, tampered)
    with pytest.raises(_MODULE.OrchestrationError, match="corrupt"):
        _MODULE._materialize_synthetic_gate(_PATH.parents[2], path)

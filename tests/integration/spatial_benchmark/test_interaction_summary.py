from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark import interaction_summary


def _manifest(
    *,
    run_id: str,
    seed: int,
    k: int,
    curriculum: str,
) -> dict[str, Any]:
    graph_config = {
        "edge_dropout": 0.1,
        "k": k,
        "min_distance_um": 0.0,
        "radius_um": 50.0,
        "rbf_bins": 8,
        "symmetry": "union",
    }
    return {
        "run_id": run_id,
        "model_name": "g1",
        "model_seed": seed,
        "sealed_test_opened": False,
        "prepared_artifact": {
            "artifact_id": "prepared-1",
            "manifest_sha256": "a" * 64,
            "split_id": "split-1",
            "validation_mask_bundle_id": "validation-masks-1",
        },
        "graph": {
            "graph_id": f"k{k}_r50_union_rbf8",
            "kind": "true",
            "edge_control": "none",
            "config": graph_config,
        },
        "config": {
            "model": {
                "name": "g1",
                "hidden_dim": 256,
                "graph_layers": 1,
            },
            "run": {
                "model_seed": seed,
                "evaluate_test": False,
                "save_predictions": False,
                "rewired": False,
                "edge_control": "none",
            },
            "graph": graph_config,
            "training": {
                "model_seed": seed,
                "curriculum": curriculum,
                "max_epochs": 60,
                "patience": 12,
                "amp": True,
            },
        },
        "metrics_file": "metrics.json",
    }


def _metrics(
    *,
    node: float,
    block: float,
    opened_test: bool = False,
) -> dict[str, Any]:
    validation = []
    for mode, center in (("node", node), ("block", block)):
        for replicate, offset in ((0, -0.002), (1, 0.002)):
            validation.append(
                {
                    "split": "validation",
                    "mask_mode": mode,
                    "mask_replicate": replicate,
                    "metrics": {
                        "huber": center + offset,
                        "n_masked": 1000,
                    },
                }
            )
    return {
        "validation": validation,
        "test": (
            [
                {
                    "split": "test",
                    "mask_mode": "node",
                    "mask_replicate": 0,
                    "metrics": {"huber": 0.1, "n_masked": 1000},
                }
            ]
            if opened_test
            else []
        ),
        "test_targets_evaluated": opened_test,
    }


def _screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds_by_cell: dict[tuple[int, str], tuple[int, ...]] | None = None,
    nuisance_mutation: tuple[int, str, int] | None = None,
    opened_cell: tuple[int, str, int] | None = None,
) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    manifests: dict[str, dict[str, Any]] = {}
    default_seeds = {
        (k, curriculum): (0, 1)
        for k in (8, 12)
        for curriculum in ("P+N", "P+N+B")
    }
    seed_map = seeds_by_cell or default_seeds
    index = 0
    for (k, curriculum), seeds in seed_map.items():
        for seed in seeds:
            run_dir = runs / f"run-{index}"
            run_dir.mkdir()
            run_id = f"run-{index}"
            manifest = _manifest(
                run_id=run_id,
                seed=seed,
                k=k,
                curriculum=curriculum,
            )
            if nuisance_mutation == (k, curriculum, seed):
                manifest["config"]["training"]["max_epochs"] = 61
            opened = opened_cell == (k, curriculum, seed)
            if opened:
                manifest["sealed_test_opened"] = True
            node = {
                (8, "P+N"): 0.30,
                (12, "P+N"): 0.20,
                (8, "P+N+B"): 0.25,
                (12, "P+N+B"): 0.20,
            }[(k, curriculum)] + 0.01 * seed
            block = {
                (8, "P+N"): 0.40,
                (12, "P+N"): 0.30,
                (8, "P+N+B"): 0.28,
                (12, "P+N+B"): 0.30,
            }[(k, curriculum)] + 0.01 * seed
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": (
                            interaction_summary.RUN_ARTIFACT_KIND
                        ),
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    _metrics(
                        node=node,
                        block=block,
                        opened_test=opened,
                    )
                ),
                encoding="utf-8",
            )
            manifests[run_dir.name] = manifest
            index += 1
    monkeypatch.setattr(
        interaction_summary,
        "load_run_manifest",
        lambda run_dir: manifests[Path(run_dir).name],
    )
    return runs


def test_atomic_diagnostic_reports_paired_difference_in_differences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _screen(tmp_path, monkeypatch)
    output = tmp_path / "interaction"
    created = interaction_summary.create_interaction_summary(
        runs,
        output,
        command=["summarize", "--validation-only"],
    )
    assert created == output.resolve()
    assert not list(tmp_path.glob(".interaction.tmp-*"))
    manifest, diagnostic = interaction_summary.load_interaction_summary(output)

    assert manifest["diagnostic_only"] is True
    assert manifest["selection_performed"] is False
    assert manifest["locked_graph_or_mask_overridden"] is False
    assert manifest["test_metrics_used"] is False
    assert diagnostic["selection_recommendation"]["made"] is False
    assert diagnostic["selection_recommendation"]["action"] == "none"
    assert diagnostic["graph_definitions"][diagnostic["graph_a"]]["k"] == 8
    assert diagnostic["graph_definitions"][diagnostic["graph_b"]]["k"] == 12
    assert diagnostic["curriculum_a"] == "P+N"
    assert diagnostic["curriculum_b"] == "P+N+B"
    effects = {
        item["mask_mode"]: item
        for item in diagnostic["difference_in_differences"]
    }
    assert effects["node"]["mean"] == pytest.approx(0.05)
    assert effects["block"]["mean"] == pytest.approx(0.12)
    assert effects["node"]["n_paired_seeds"] == 2
    assert len(effects["node"]["per_seed"]) == 2

    per_seed = pd.read_csv(output / "validation_per_seed.csv")
    assert set(per_seed["n_mask_replicates"]) == {2}
    assert len(per_seed) == 2 * 2 * 2 * 2
    paired = pd.read_csv(
        output / "paired_difference_in_differences.csv"
    )
    assert len(paired) == 4
    assert set(paired["mask_mode"]) == {"node", "block"}
    assert (output / "validation_interaction.png").stat().st_size > 0

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        interaction_summary.create_interaction_summary(runs, output)


def test_rejects_nonidentical_paired_seed_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = {
        (8, "P+N"): (0, 1),
        (12, "P+N"): (0, 1),
        (8, "P+N+B"): (0, 1),
        (12, "P+N+B"): (1, 2),
    }
    runs = _screen(tmp_path, monkeypatch, seeds_by_cell=seeds)
    with pytest.raises(
        interaction_summary.InteractionSummaryError,
        match="exact identical paired model-seed set",
    ):
        interaction_summary.load_interaction_runs(runs)


def test_rejects_nuisance_changes_outside_graph_and_curriculum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _screen(
        tmp_path,
        monkeypatch,
        nuisance_mutation=(12, "P+N+B", 1),
    )
    with pytest.raises(
        interaction_summary.InteractionSummaryError,
        match="differ in nuisance settings",
    ):
        interaction_summary.load_interaction_runs(runs)


def test_rejects_duplicate_immutable_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _screen(tmp_path, monkeypatch)
    original_loader = interaction_summary.load_run_manifest
    duplicate_id = original_loader(runs / "run-0")["run_id"]

    def duplicate_loader(run_dir: Path) -> dict[str, Any]:
        value = deepcopy(original_loader(run_dir))
        if Path(run_dir).name == "run-1":
            value["run_id"] = duplicate_id
        return value

    monkeypatch.setattr(
        interaction_summary,
        "load_run_manifest",
        duplicate_loader,
    )
    with pytest.raises(
        interaction_summary.InteractionSummaryError,
        match="Duplicate immutable run_id",
    ):
        interaction_summary.load_interaction_runs(runs)


def test_rejects_opened_test_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _screen(
        tmp_path,
        monkeypatch,
        opened_cell=(12, "P+N+B", 1),
    )
    with pytest.raises(
        interaction_summary.InteractionSummaryError,
        match="sealed validation runs only",
    ):
        interaction_summary.load_interaction_runs(runs)


def test_checksum_verifier_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _screen(tmp_path, monkeypatch)
    output = interaction_summary.create_interaction_summary(
        runs,
        tmp_path / "interaction",
    )
    summary = output / "validation_summary.csv"
    summary.write_text(
        summary.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        interaction_summary.InteractionSummaryError,
        match="checksum mismatch",
    ):
        interaction_summary.load_interaction_summary(output)

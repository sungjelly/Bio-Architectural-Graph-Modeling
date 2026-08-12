from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYZER = PROJECT_ROOT / "scripts/analysis/analyze_matched_graph_context.py"
_SPEC = importlib.util.spec_from_file_location(
    "analyze_matched_graph_context_for_tests", ANALYZER
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _selection_block(hidden: int = 32, parameter_count: int = 1_092_032) -> dict[str, Any]:
    config = {
        "candidate_id": "c02",
        "hidden_width": hidden,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "epoch": 48,
        "batch_size": 4096,
    }
    return {
        "candidate_id": "c02",
        "config": config,
        "config_sha256": analysis.canonical_sha256(config),
        "parameter_count": parameter_count,
    }


def _write_selection_receipt(tmp_path: Path) -> Path:
    contract = tmp_path / analysis.CONTRACT_RELATIVE
    contract.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / analysis.CONTRACT_RELATIVE, contract)
    stage_a = tmp_path / "state/stage_a.json"
    stage_b = tmp_path / "state/stage_b.json"
    tuning = tmp_path / "state/tuning.json"
    stage_a.parent.mkdir(parents=True)
    for path in (stage_a, stage_b, tuning):
        path.write_text("{}\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "kind": analysis.SELECTION_KIND,
        "campaign_id": analysis.CAMPAIGN_ID,
        "test_metrics_used_for_selection": False,
        "status": "frozen",
        "contract_sha256": analysis.CONTRACT_SHA256,
        "candidate_design_sha256": analysis.CANDIDATE_DESIGN_SHA256,
        "cross_outer_pooling": False,
        "source_stage_a_plan": {
            "path": stage_a.relative_to(tmp_path).as_posix(),
            "sha256": analysis._sha256_file(stage_a),
        },
        "source_stage_a_selection": {
            "path": stage_a.relative_to(tmp_path).as_posix(),
            "sha256": analysis._sha256_file(stage_a),
        },
        "source_stage_b_plan": {
            "path": stage_b.relative_to(tmp_path).as_posix(),
            "sha256": analysis._sha256_file(stage_b),
        },
        "source_tuning_results_by_outer_fold": {},
        "source_tuning_result_sha256s_by_outer_fold": {},
        "selected_by_outer_fold": {
            str(fold): {arm: _selection_block() for arm in analysis.ARMS}
            for fold in analysis.FOLDS
        },
    }
    for fold in analysis.FOLDS:
        sources = []
        for index in range(88):
            result = tmp_path / f"state/tuning-f{fold}-{index}.json"
            result.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "campaign_id": analysis.CAMPAIGN_ID,
                        "contract_sha256": analysis.CONTRACT_SHA256,
                        "mode": "tune",
                        "fold": fold,
                        "status": "success",
                        "finite_metrics": True,
                        "coverage_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            marker = tmp_path / f"state/tuning-f{fold}-{index}._SUCCESS"
            marker.write_text("{}", encoding="utf-8")
            sources.append(
                {
                    "job_id": f"stage.c02.s{index}.f{fold}",
                    "path": result.relative_to(tmp_path).as_posix(),
                    "sha256": analysis._sha256_file(result),
                    "success_marker": marker.relative_to(tmp_path).as_posix(),
                    "success_marker_sha256": analysis._sha256_file(marker),
                }
            )
        receipt["source_tuning_results_by_outer_fold"][str(fold)] = sources
        receipt["source_tuning_result_sha256s_by_outer_fold"][str(fold)] = sorted(
            row["sha256"] for row in sources
        )
    receipt["payload_sha256"] = analysis.canonical_sha256(receipt)
    path = tmp_path / "selection_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_selection_receipt_requires_four_outer_specific_matched_selections(
    tmp_path: Path,
) -> None:
    path = _write_selection_receipt(tmp_path)
    verified = analysis._verify_selection_receipt(path, project_root=tmp_path)
    assert set(verified["selected_by_outer_fold"]) == {"0", "1", "2", "3"}

    value = json.loads(path.read_text(encoding="utf-8"))
    value["selected_by_outer_fold"].pop("3")
    value["payload_sha256"] = analysis.canonical_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        analysis.MatchedGraphContextAnalysisError, match="four independent outer folds"
    ):
        analysis._verify_selection_receipt(path, project_root=tmp_path)


def test_selection_receipt_rejects_test_use_hash_tamper_and_unmatched_width(
    tmp_path: Path,
) -> None:
    path = _write_selection_receipt(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["test_metrics_used_for_selection"] = True
    value["payload_sha256"] = analysis.canonical_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="selection contract"):
        analysis._verify_selection_receipt(path, project_root=tmp_path)

    path = _write_selection_receipt(tmp_path / "width")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selected_by_outer_fold"]["2"]["observed_near"]["config"][
        "hidden_width"
    ] = 64
    value["selected_by_outer_fold"]["2"]["observed_near"]["config_sha256"] = (
        analysis.canonical_sha256(
            value["selected_by_outer_fold"]["2"]["observed_near"]["config"]
        )
    )
    value["payload_sha256"] = analysis.canonical_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        analysis.MatchedGraphContextAnalysisError,
        match="frozen candidate|share hidden",
    ):
        analysis._verify_selection_receipt(path, project_root=tmp_path / "width")

    path = _write_selection_receipt(tmp_path / "hash")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selected_by_outer_fold"]["0"]["no_graph"]["candidate_id"] = "c99"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="self-digest"):
        analysis._verify_selection_receipt(path, project_root=tmp_path / "hash")


def _run(arm: str, seed: int, fold: int) -> Any:
    return analysis.ConfirmationRun(
        root=Path(f"/tmp/{arm}-{seed}-{fold}"),
        result_path=Path("results.json"),
        result_file_sha256="b" * 64,
        result={
            "parameter_count": 2_060_000,
            "config": {"hidden_width": 32},
        },
        arm=arm,
        seed=seed,
        fold=fold,
        component_rows=(),
        gene_rows=(),
        component_gene_rows=(),
        substitution_rows=(),
    )


def test_confirmation_coverage_is_exact_five_by_four_by_four() -> None:
    runs = [
        _run(arm, seed, fold)
        for arm in analysis.ARMS
        for seed in analysis.SEEDS
        for fold in analysis.FOLDS
    ]
    assert len(analysis._validate_confirmation_coverage(runs)) == 80
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="missing"):
        analysis._validate_confirmation_coverage(runs[:-1])
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="duplicate"):
        analysis._validate_confirmation_coverage(runs + [runs[0]])


def _expected_components() -> tuple[set[tuple[str, int]], dict[tuple[str, int], int]]:
    components = {("SO_1", 101), ("SO_2", 201)}
    return components, {key: 10 for key in components}


def test_component_gene_and_substitution_coverage_fail_closed() -> None:
    components, cells = _expected_components()
    base = {
        "run_id": "r_test",
        "arm": "observed_near",
        "seed": analysis.SEEDS[0],
        "fold": 0,
        "n_cells": 10,
        "mse": 0.8,
        "mae": 0.6,
    }
    rows = [
        {**base, "slide": slide, "component": component, "context_variant": "native"}
        for slide, component in sorted(components)
    ]
    normalized = analysis._normalize_component_rows(
        rows,
        arm="observed_near",
        seed=analysis.SEEDS[0],
        fold=0,
        run_id="r_test",
        expected=components,
        expected_cells=cells,
    )
    assert len(normalized) == 2
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="incomplete"):
        analysis._normalize_component_rows(
            rows[:-1],
            arm="observed_near",
            seed=analysis.SEEDS[0],
            fold=0,
            run_id="r_test",
            expected=components,
            expected_cells=cells,
        )

    genes = tuple(f"G{index}" for index in range(analysis.EXPECTED_GENES))
    gene_rows = [
        {
            "run_id": "r_test",
            "arm": "observed_near",
            "seed": analysis.SEEDS[0],
            "fold": 0,
            "gene_index": index,
            "gene": gene,
            "mse": 0.5,
            "mae": 0.4,
            "pearson": 0.2,
            "n_cells": 20,
        }
        for index, gene in enumerate(genes)
    ]
    assert len(
        analysis._normalize_gene_rows(
            gene_rows,
            arm="observed_near",
            seed=analysis.SEEDS[0],
            fold=0,
            run_id="r_test",
            genes=genes,
        )
    ) == analysis.EXPECTED_GENES
    gene_rows[-1]["mse"] = float("nan")
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="finite"):
        analysis._normalize_gene_rows(
            gene_rows,
            arm="observed_near",
            seed=analysis.SEEDS[0],
            fold=0,
            run_id="r_test",
            genes=genes,
        )

    substitutions = [
        {
            **base,
            "slide": slide,
            "component": component,
            "context_variant": context,
        }
        for slide, component in sorted(components)
        for context in analysis.SUBSTITUTIONS
    ]
    assert len(
        analysis._normalize_substitution_rows(
            substitutions,
            seed=analysis.SEEDS[0],
            fold=0,
            run_id="r_test",
            expected=components,
            expected_cells=cells,
        )
    ) == 10
    with pytest.raises(analysis.MatchedGraphContextAnalysisError, match="incomplete"):
        analysis._normalize_substitution_rows(
            substitutions[:-1],
            seed=analysis.SEEDS[0],
            fold=0,
            run_id="r_test",
            expected=components,
            expected_cells=cells,
        )


def test_slide_stratified_component_bootstrap_is_deterministic_and_paired() -> None:
    baseline = np.linspace(0.8, 1.2, analysis.EXPECTED_COMPONENTS)
    candidate = baseline * 0.9
    slides = ["SO_1"] * 13 + ["SO_2"] * 14
    first = analysis._slide_stratified_bootstrap(
        baseline, candidate, slides, return_draws=True
    )
    second = analysis._slide_stratified_bootstrap(
        baseline, candidate, slides, return_draws=True
    )
    assert first["resamples"] == 10_000
    assert first["point"] == pytest.approx(0.1)
    assert np.array_equal(first["draws"], second["draws"])
    assert np.allclose(first["draws"], 0.1)


def _component_rows(gain: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = [("SO_1", 101 + index) for index in range(13)] + [
        ("SO_2", 201 + index) for index in range(14)
    ]
    for arm in analysis.ARMS:
        value = {
            "no_graph": 1.0,
            "observed_near": 1.0 - gain,
            "permuted_near": 0.99,
            "observed_annular": 0.98,
        }[arm]
        for seed in analysis.SEEDS:
            for slide, component in keys:
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold": 0,
                        "slide": slide,
                        "component": component,
                        "mse": value,
                        "mae": value,
                    }
                )
    return rows


def test_verdict_uses_all_prespecified_primary_criteria() -> None:
    rows = _component_rows(0.03)
    graph_no = analysis._contrast_summary(
        rows,
        baseline="no_graph",
        candidate="observed_near",
        metric="mse",
        bootstrap_seed=analysis.BOOTSTRAP_SEED,
    )
    graph_perm = analysis._contrast_summary(
        rows,
        baseline="permuted_near",
        candidate="observed_near",
        metric="mse",
        bootstrap_seed=analysis.BOOTSTRAP_SEED + 1,
    )
    mae = analysis._contrast_summary(
        rows,
        baseline="no_graph",
        candidate="observed_near",
        metric="mae",
        bootstrap_seed=analysis.BOOTSTRAP_SEED + 2,
    )
    assert analysis._decision(graph_no, graph_perm, mae)["verdict"] == (
        "GRAPH CONTEXT SUPPORTED"
    )
    worsened_mae = dict(mae)
    worsened_mae["mean_difference"] = -1e-6
    assert analysis._decision(graph_no, graph_perm, worsened_mae)["verdict"] == (
        "INCONCLUSIVE"
    )


def test_fixed_target_programs_are_exactly_bound_to_contract() -> None:
    contract = analysis.load_yaml_mapping(PROJECT_ROOT / analysis.CONTRACT_RELATIVE)
    assert contract["target_programs"] == {
        key: list(value) for key, value in analysis.TARGET_PROGRAMS.items()
    }
    genes = json.loads(
        (
            PROJECT_ROOT
            / "data/processed/same_gene_robustness_v1/variants/"
            "v0_within_fov_log1p_all/genes.json"
        ).read_text(encoding="utf-8")
    )
    assert all(gene in genes for family in analysis.TARGET_PROGRAMS.values() for gene in family)


def test_load_confirmation_accepts_actual_runner_shaped_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the analyzer/runner boundary with the production field names."""

    bundle = tmp_path / "artifacts/runs/2026/08/r_test"
    metrics = bundle / "metrics"
    checkpoints = bundle / "checkpoints"
    metrics.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    selection_path = tmp_path / "selection_receipt.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    genes = tuple(f"G{index}" for index in range(analysis.EXPECTED_GENES))
    seed = analysis.SEEDS[0]
    config = _selection_block()["config"]
    selected = {
        "candidate_id": config["candidate_id"],
        "config": config,
        "config_sha256": analysis.canonical_sha256(config),
        "parameter_count": 1_092_032,
    }
    receipt = {
        "payload_sha256": "a" * 64,
        "selected_by_outer_fold": {
            str(fold): {arm: selected for arm in analysis.ARMS}
            for fold in analysis.FOLDS
        },
    }

    component_rows = [
        {
            "run_id": bundle.name,
            "arm": "no_graph",
            "seed": seed,
            "fold": 0,
            "slide": "SO_1",
            "component": 101,
            "n_cells": 3,
            "mse": 0.5,
            "mae": 0.4,
            "context_variant": "native",
        }
    ]
    gene_rows = [
        {
            "run_id": bundle.name,
            "arm": "no_graph",
            "seed": seed,
            "fold": 0,
            "gene_index": index,
            "gene": gene,
            "n_cells": 3,
            "mse": 0.5,
            "mae": 0.4,
            "pearson": 0.1,
            "context_variant": "native",
        }
        for index, gene in enumerate(genes)
    ]
    component_gene_rows = [
        {
            "run_id": bundle.name,
            "arm": "no_graph",
            "seed": seed,
            "fold": 0,
            "slide": "SO_1",
            "component": 101,
            "n_cells": 3,
            "gene_index": index,
            "gene": gene,
            "mse": 0.5,
            "mae": 0.4,
        }
        for index, gene in enumerate(genes)
    ]

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
        path = metrics / name
        path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    component_path = write_jsonl("component_metrics.jsonl", component_rows)
    gene_path = write_jsonl("gene_metrics.jsonl", gene_rows)
    component_gene_path = write_jsonl(
        "component_gene_metrics.jsonl", component_gene_rows
    )
    checkpoint_path = checkpoints / "last.ckpt"
    checkpoint_path.write_bytes(b"runner-shaped-checkpoint")

    def output(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(bundle).as_posix(),
            "sha256": analysis._sha256_file(path),
        }

    result = {
        "schema_version": 1,
        "status": "success",
        "finite_metrics": True,
        "coverage_complete": True,
        "campaign_id": analysis.CAMPAIGN_ID,
        "contract_sha256": analysis.CONTRACT_SHA256,
        "mode": "confirm",
        "run_id": bundle.name,
        "arm": "no_graph",
        "fold": 0,
        "seed": seed,
        "candidate_id": config["candidate_id"],
        "config": config,
        "config_sha256": analysis.canonical_sha256(config),
        "resolved_config_sha256": "f" * 64,
        "input": {
            "manifest_sha256": "b" * 64,
            "prepared_manifest_sha256": "b" * 64,
            "integrity_manifest_sha256": "c" * 64,
            "processed_fingerprint": (
                "01c525695883784befc1b9ebbe37a6d96b248d5b18450f46a1255e571bd3819e"
            ),
            "split_fingerprint": (
                "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"
            ),
        },
        "split_roles": {"train_folds": [1, 2, 3], "test_fold": 0},
        "parameter_count": 1_092_032,
        "normalization_sha256": "d" * 64,
        "projection_sha256": "e" * 64,
        "metrics": {
            "test_component_equal_mse": 0.5,
            "test_component_equal_mae": 0.4,
        },
        "outputs": {
            "component_metrics": output(component_path),
            "gene_metrics": output(gene_path),
            "component_gene_metrics": output(component_gene_path),
            "checkpoint": output(checkpoint_path),
        },
        "selection_receipt": {
            "path": selection_path.relative_to(tmp_path).as_posix(),
            "sha256": analysis._sha256_file(selection_path),
            "payload_sha256": receipt["payload_sha256"],
        },
    }
    (bundle / "results.json").write_text(
        json.dumps(result, allow_nan=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        analysis,
        "verify_run_bundle",
        lambda _root, require_success_contract: {"status": "success"},
    )
    loaded = analysis._load_confirmation_run(
        bundle,
        receipt=receipt,
        receipt_path=selection_path,
        receipt_file_sha256=analysis._sha256_file(selection_path),
        receipt_payload_sha256=str(receipt["payload_sha256"]),
        expected_components={0: {("SO_1", 101)}},
        component_cells={("SO_1", 101): 3},
        genes=genes,
        project_root=tmp_path,
    )
    assert loaded.key == ("no_graph", seed, 0)
    assert len(loaded.component_gene_rows) == analysis.EXPECTED_GENES

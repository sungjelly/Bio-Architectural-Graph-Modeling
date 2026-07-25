from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analysis" / "summarize_screen.py"
SPEC = importlib.util.spec_from_file_location(
    "normal_core_summarize_screen",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
summarize_screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarize_screen
SPEC.loader.exec_module(summarize_screen)


def _full_manifest(
    *,
    run_id: str,
    model: str = "g1",
    seed: int = 0,
    curriculum: str = "P+N+B",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_name": model,
        "model_seed": seed,
        "sealed_test_opened": False,
        "prepared_artifact": {
            "artifact_id": "prepared-1",
            "manifest_sha256": "prepared-sha256",
            "split_id": "split-1",
            "validation_mask_bundle_id": "validation-masks-1",
        },
        "graph": {
            "graph_id": "graph-1",
            "kind": "true",
            "edge_control": "none",
            "config": {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "min_distance_um": 0.0,
            },
        },
        "config": {
            "run": {"model_seed": seed},
            "training": {
                "model_seed": seed,
                "curriculum": curriculum,
            },
            "model": {
                "name": model,
                "hidden_dim": 256,
                "graph_layers": 1,
                "edge_embedding_dim": 32,
            },
            "graph": {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "min_distance_um": 0.0,
            },
        },
        "training": {"best_epoch": 7},
        "timing": {"runtime_seconds": 12.5},
        "resources": {"peak_cuda_memory_bytes": 1024},
        "metrics_file": "metrics.json",
    }


def _validation_metrics(huber: float = 0.2) -> dict[str, Any]:
    return {
        "validation": [
            {
                "mask_mode": "node",
                "mask_replicate": 0,
                "metrics": {
                    "huber": huber,
                    "mse": 2.0 * huber,
                    "mae": 1.5 * huber,
                    "n_masked": 100,
                },
            }
        ],
        "test": [],
        "test_targets_evaluated": False,
    }


def _screen_records(
    *,
    model: str,
    candidate_id: str,
    field: str,
    field_value: Any,
    node_losses: tuple[float, ...],
    block_losses: tuple[float, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed, (node_loss, block_loss) in enumerate(
        zip(node_losses, block_losses, strict=True)
    ):
        for mask_mode, huber in (
            ("node", node_loss),
            ("block", block_loss),
        ):
            records.append(
                {
                    "model": model,
                    "graph_kind": "true",
                    "candidate_id": candidate_id,
                    "mask_mode": mask_mode,
                    "model_seed": seed,
                    "mask_replicate": 0,
                    "huber": huber,
                    "mse": 2.0 * huber,
                    "mae": 1.5 * huber,
                    "runtime_seconds": 10.0,
                    "peak_cuda_memory_bytes": 1024,
                    field: field_value,
                }
            )
    return records


def _manual_summary(
    candidate_values: dict[
        str,
        tuple[float, float, float],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, (node_mean, block_mean, node_sd) in (
        candidate_values.items()
    ):
        rows.extend(
            [
                {
                    "model": "g1",
                    "graph_kind": "true",
                    "candidate_id": candidate_id,
                    "mask_mode": "node",
                    "n_seeds": 3,
                    "huber_mean": node_mean,
                    "huber_sd": node_sd,
                },
                {
                    "model": "g1",
                    "graph_kind": "true",
                    "candidate_id": candidate_id,
                    "mask_mode": "block",
                    "n_seeds": 3,
                    "huber_mean": block_mean,
                    "huber_sd": 0.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _graph_qc_row(
    *,
    k: int,
    isolated_nodes: int = 0,
) -> dict[str, Any]:
    return {
        "k": k,
        "radius_um": 50.0,
        "symmetry": "union",
        "selection_n_nodes": 1000,
        "selection_n_isolated_nodes": isolated_nodes,
        "selection_median_degree": 8.0,
        "selection_edge_distance_max_um": 49.0,
        "selection_zero_distance_edges": 0,
        "selection_self_loops": 0,
        "selection_duplicate_directed_edges": 0,
        "selection_cross_group_edges": 0,
        "selection_directed_edge_pairs_are_symmetric": True,
    }


def _write_graph_qc_artifact(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    selection_scope: str = "train_and_validation_geometry_only",
    test_geometry_used: bool = False,
    test_expression_evaluated: bool = False,
) -> Path:
    root.mkdir(parents=True)
    qc_path = root / "graph_qc.csv"
    pd.DataFrame.from_records(rows).to_csv(qc_path, index=False)
    manifest = {
        "artifact_kind": "geometry_only_graph_grid",
        "selection_scope": selection_scope,
        "test_geometry_used_for_selection_qc": test_geometry_used,
        "test_expression_targets_evaluated": test_expression_evaluated,
        "prepared_artifact_id": "prepared-1",
        "split_id": "split-1",
        "files": {
            qc_path.name: summarize_screen.sha256_file(qc_path),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return qc_path


def test_load_rejects_duplicate_candidate_seed_mask_evaluations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests: dict[str, dict[str, Any]] = {}
    for index in range(2):
        run_dir = tmp_path / f"run-{index}"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model_name": "g1",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(_validation_metrics()),
            encoding="utf-8",
        )
        manifests[run_dir.name] = _full_manifest(
            run_id=f"immutable-run-{index}"
        )

    monkeypatch.setattr(
        summarize_screen,
        "load_run_manifest",
        lambda run_dir: manifests[Path(run_dir).name],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate candidate/model/seed/mask evaluations",
    ):
        summarize_screen._load(tmp_path, "mask")


def test_recommend_requires_one_identical_paired_seed_set() -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "model": "g1",
                "graph_kind": "true",
                "candidate_id": candidate,
                "model_seed": seed,
                "curriculum": curriculum,
            }
            for candidate, curriculum, seeds in (
                ("candidate-a", "P+N", (0, 1, 2)),
                ("candidate-b", "P+N+B", (1, 2, 3)),
            )
            for seed in seeds
        ]
    )

    with pytest.raises(
        ValueError,
        match="identical paired seed set",
    ):
        summarize_screen._recommend(
            pd.DataFrame(),
            frame,
            selection="mask",
            required_seeds=3,
        )


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (
            "graph",
            {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "min_distance_um": 2.0,
            },
        ),
        ("mask", {"curriculum": "P+N+B"}),
        ("hidden", {"hidden_dim": 256}),
        ("depth", {"graph_layers": 2}),
        ("edge_embedding", {"edge_embedding_dim": 32}),
        (
            "representation",
            {
                "hidden_dim": 256,
                "graph_layers": 2,
                "edge_embedding_dim": 32,
            },
        ),
    ],
)
def test_candidate_fields_are_specific_to_selection(
    selection: str,
    expected: dict[str, Any],
) -> None:
    manifest = _full_manifest(run_id="field-test")
    manifest["graph"]["config"]["min_distance_um"] = 2.0
    manifest["config"]["model"]["graph_layers"] = 2

    assert summarize_screen._candidate_fields(manifest, selection) == expected


def test_edge_embedding_selection_uses_g2_and_reports_g2_ranking() -> None:
    records: list[dict[str, Any]] = []
    expected_ids: dict[int, str] = {}
    for dimension, node, block in (
        (16, 0.20, 0.10),
        (32, 0.12, 0.40),
        (64, 0.15, 0.05),
    ):
        candidate_id = summarize_screen._candidate_id(
            {"edge_embedding_dim": dimension}
        )
        expected_ids[dimension] = candidate_id
        records.extend(
            _screen_records(
                model="g2",
                candidate_id=candidate_id,
                field="edge_embedding_dim",
                field_value=dimension,
                node_losses=(node, node, node),
                block_losses=(block, block, block),
            )
        )
    records.extend(
        _screen_records(
            model="g1",
            candidate_id="g1-distractor",
            field="edge_embedding_dim",
            field_value=999,
            node_losses=(0.001, 0.001, 0.001),
            block_losses=(0.001, 0.001, 0.001),
        )
    )
    frame = pd.DataFrame.from_records(records)

    recommendation = summarize_screen._recommend(
        summarize_screen._aggregate(frame),
        frame,
        selection="edge_embedding",
        required_seeds=3,
    )

    assert recommendation["locked"] is True
    assert recommendation["standard"] == {"edge_embedding_dim": 32}
    assert recommendation["candidate_id"] == expected_ids[32]
    assert recommendation["paired_seeds"] == [0, 1, 2]
    assert [
        row["candidate_id"]
        for row in recommendation["ranked_candidates"]
    ] == [
        expected_ids[32],
        expected_ids[64],
        expected_ids[16],
    ]


def test_recommendation_ranking_uses_prespecified_lexicographic_ties() -> None:
    candidate_values = {
        "primary-best": (0.09, 0.99, 0.99),
        "block-next": (0.10, 0.20, 0.99),
        "a-stable-tie": (0.10, 0.30, 0.01),
        "b-stable-tie": (0.10, 0.30, 0.01),
        "less-stable": (0.10, 0.30, 0.02),
    }
    frame = pd.DataFrame.from_records(
        [
            {
                "model": "g1",
                "graph_kind": "true",
                "candidate_id": candidate_id,
                "model_seed": seed,
                "hidden_dim": 100 + index,
            }
            for index, candidate_id in enumerate(candidate_values)
            for seed in (0, 1, 2)
        ]
    )

    recommendation = summarize_screen._recommend(
        _manual_summary(candidate_values),
        frame,
        selection="hidden",
        required_seeds=3,
    )

    ranked = recommendation["ranked_candidates"]
    assert [row["candidate_id"] for row in ranked] == [
        "primary-best",
        "block-next",
        "a-stable-tie",
        "b-stable-tie",
        "less-stable",
    ]
    assert ranked[0] == {
        "candidate_id": "primary-best",
        "whole_node_huber": 0.09,
        "spatial_block_huber": 0.99,
        "whole_node_seed_sd": 0.99,
        "paired_seeds": [0, 1, 2],
    }
    assert recommendation["test_metrics_used"] is False
    assert recommendation["decision_rule"].startswith(
        "lexicographically minimize validation whole-node"
    )


def test_graph_qc_attaches_only_verified_selection_scope_and_flags_geometry(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "prepared_artifact_id": "prepared-1",
                "split_id": "split-1",
                "k": 8,
                "radius_um": 50.0,
                "symmetry": "union",
            },
            {
                "prepared_artifact_id": "prepared-1",
                "split_id": "split-1",
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
            },
        ]
    )
    qc_path = _write_graph_qc_artifact(
        tmp_path / "verified-qc",
        [
            _graph_qc_row(k=8),
            _graph_qc_row(k=12, isolated_nodes=11),
        ],
    )

    attached = summarize_screen._attach_graph_qc(frame, qc_path)

    assert attached["qc_eligible"].tolist() == [True, False]
    assert attached["qc_isolated_rate"].tolist() == [0.0, 0.011]


@pytest.mark.parametrize(
    ("selection_scope", "test_geometry_used", "test_expression_evaluated"),
    [
        ("all_splits_geometry", False, False),
        ("train_and_validation_geometry_only", True, False),
        ("train_and_validation_geometry_only", False, True),
    ],
)
def test_graph_qc_rejects_non_validation_only_provenance(
    tmp_path: Path,
    selection_scope: str,
    test_geometry_used: bool,
    test_expression_evaluated: bool,
) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "prepared_artifact_id": "prepared-1",
                "split_id": "split-1",
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
            }
        ]
    )
    qc_path = _write_graph_qc_artifact(
        tmp_path / "invalid-qc",
        [_graph_qc_row(k=12)],
        selection_scope=selection_scope,
        test_geometry_used=test_geometry_used,
        test_expression_evaluated=test_expression_evaluated,
    )

    with pytest.raises(
        ValueError,
        match="Graph-QC artifact is incompatible or unverified",
    ):
        summarize_screen._attach_graph_qc(frame, qc_path)


def test_graph_qc_rejects_tables_without_selection_scope_columns(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "prepared_artifact_id": "prepared-1",
                "split_id": "split-1",
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
            }
        ]
    )
    row = _graph_qc_row(k=12)
    row.pop("selection_cross_group_edges")
    qc_path = _write_graph_qc_artifact(
        tmp_path / "incomplete-qc",
        [row],
    )

    with pytest.raises(
        ValueError,
        match="lacks selection-scope fields",
    ):
        summarize_screen._attach_graph_qc(frame, qc_path)

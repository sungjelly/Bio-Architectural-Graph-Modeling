from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.artifacts import (  # noqa: E402
    load_prepared_artifact,
    prepare_artifact,
)
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS  # noqa: E402
from spatial_benchmark.experiment import (  # noqa: E402
    ExperimentContractError,
    _canonical_hash,
    _edge_control,
    _make_graph,
    _manifest_content_hash,
    _prepared_graph,
    _split_view,
    load_run_manifest,
    run_experiment,
)
from spatial_benchmark.standards_lock import (  # noqa: E402
    canonical_job_hash,
    create_standards_lock,
)


def _write_nested_csv(raw_dir: Path, basename: str, frame: pd.DataFrame) -> None:
    directory = raw_dir / basename
    directory.mkdir(parents=True)
    frame.to_csv(directory / basename, index=False)


def _prepare_synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "synthetic-project"
    raw = project / "data" / "raw"
    clinical = project / "data" / "clinical"
    raw.mkdir(parents=True)
    clinical.mkdir(parents=True)

    selected_tissue_unit = 731
    pd.DataFrame(
        [
            ["header", None, None],
            [selected_tissue_unit, "restricted-synthetic-donor", "정상"],
            [732, "other-restricted-donor", "정상인접"],
        ]
    ).to_excel(
        clinical / "Gastric Study_Old.xlsx",
        header=False,
        index=False,
    )
    pd.DataFrame(
        {
            "슬라이드번호": [selected_tissue_unit, 732],
            "진단명": ["diagnosis", "diagnosis"],
            "결과": ["reviewed", "reviewed"],
            "비고 (수정진단)": [np.nan, np.nan],
        }
    ).to_excel(clinical / "Gastric Study.xlsx", index=False)
    pd.DataFrame(
        {
            "slide": ["SO_2"] * 10,
            "core_label": [selected_tissue_unit] * 9 + [732],
            "fov": list(range(1, 11)),
        }
    ).to_csv(clinical / "fov_core_map.csv", index=False)

    expression_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    offsets = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    )
    for fov in range(1, 11):
        grid_x = (fov - 1) % 3
        grid_y = (fov - 1) // 3
        for cell_id, (offset_x, offset_y) in enumerate(offsets, start=1):
            expression_rows.append(
                {
                    "fov": fov,
                    "cell_ID": cell_id,
                    "GeneA": (fov + cell_id) % 7,
                    "GeneB/C": (2 * fov + cell_id) % 11,
                    "GeneD": (3 * fov + 2 * cell_id) % 13,
                    "Negative1": cell_id,
                    "SystemControl1": fov,
                }
            )
            row: dict[str, object] = {
                "fov": fov,
                "cell_ID": cell_id,
                "slide_ID": 2,
                "CenterX_global_px": grid_x * 20.0 + offset_x,
                "CenterY_global_px": grid_y * 20.0 + offset_y,
                "qcCellsPassed": cell_id % 2 == 0,
            }
            for index, column in enumerate(ALLOWED_METADATA_COLUMNS):
                row[column] = float(index + cell_id + fov / 10)
            metadata_rows.append(row)
    _write_nested_csv(
        raw,
        "26040302SO_2_exprMat_file.csv",
        pd.DataFrame(expression_rows),
    )
    _write_nested_csv(
        raw,
        "26040302SO_2_metadata_file.csv",
        pd.DataFrame(metadata_rows),
    )

    config = {
        "version": 1,
        "project_root": str(project),
        "data": {
            "chunksize": 7,
            "expected_biological_probes": 3,
            "pixel_size_um": 1.0,
            "qc_policy": "all",
        },
        "split": {
            "block_size_um": 10.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 17,
            "fov_aware": True,
        },
        "graph": {
            "k": 3,
            "radius_um": 3.0,
            "symmetry": "union",
            "min_distance_um": 0.0,
            "rbf_bins": 4,
            "edge_dropout": 0.0,
        },
        "masking": {
            "mask_seed": 29,
            "curriculum": "P-only",
            "warmup_epochs": 0,
            "partial_gene_rate": 0.34,
            "whole_node_rate": 0.25,
            "block_node_rate": 0.25,
            "validation_replicates": 1,
            "test_replicates": 1,
        },
        "model": {
            "name": "g1",
            "hidden_dim": 8,
            "attention_heads": 2,
            "graph_layers": 1,
            "ffn_dim": 16,
            "decoder_dim": 16,
            "edge_hidden_dim": 4,
            "edge_embedding_dim": 4,
            "dropout": 0.0,
        },
        "optimization": {
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "amp": False,
        },
    }
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    prepared = prepare_artifact(config_path, tmp_path / "prepared")
    return project, prepared


def _create_synthetic_standards_lock(tmp_path: Path) -> Path:
    recommendations = tmp_path / "lock-recommendations"
    recommendations.mkdir()
    values = (
        (
            "graph",
            {
                "k": 3,
                "radius_um": 3.0,
                "symmetry": "union",
                "min_distance_um": 0.0,
            },
        ),
        ("mask", {"curriculum": "P-only"}),
        ("hidden", {"hidden_dim": 128}),
        ("depth", {"graph_layers": 1}),
        ("edge_embedding", {"edge_embedding_dim": 16}),
    )
    paths: list[Path] = []
    for selection, standard in values:
        path = recommendations / f"{selection}.json"
        path.write_text(
            json.dumps(
                {
                    "locked": True,
                    "selection": selection,
                    "candidate_id": f"{selection}-candidate",
                    "required_seeds": 3,
                    "standard": standard,
                    "test_metrics_used": False,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return create_standards_lock(
        paths,
        tmp_path / "standards-lock",
        max_epochs=1,
        patience=1,
    )


def _manual_arrays(seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(["train", "val", "test"]), 36)
    coordinates = np.concatenate(
        [
            rng.uniform(0.0, 20.0, size=(36, 2)) + offset
            for offset in (0.0, 100.0, 200.0)
        ],
        axis=0,
    )
    return {
        "coordinates_um": coordinates,
        "split_labels": labels,
        "fov": np.repeat(np.arange(3), 36),
        "target_expression": rng.normal(size=(len(labels), 3)).astype(np.float32),
        "node_covariates": rng.normal(size=(len(labels), 4)).astype(np.float32),
        "macroblock_ids": np.asarray(
            [f"block-{index // 6}" for index in range(len(labels))]
        ),
    }


def _graph_config(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "k": 5,
        "radius_um": 15.0,
        "symmetry": "union",
        "min_distance_um": 0.0,
        "rbf_bins": 4,
        "rewire_distance_bins": 4,
        "min_rewire_success_fraction": 0.0,
        "max_rewire_distance_mean_change": 1.0,
    }
    result.update(updates)
    return result


def test_graph_override_local_remap_and_train_only_edge_scaling() -> None:
    arrays = _manual_arrays()
    sparse, sparse_attributes = _make_graph(
        arrays,
        _graph_config(k=2),
        rewired=False,
        rewire_seed=0,
        swaps_per_edge=1.0,
    )
    dense, _ = _make_graph(
        arrays,
        _graph_config(k=7),
        rewired=False,
        rewire_seed=0,
        swaps_per_edge=1.0,
    )
    assert dense.edge_index.shape[1] > sparse.edge_index.shape[1]
    source, target = sparse.edge_index
    labels = arrays["split_labels"]
    assert not np.any(labels[source] != labels[target])
    train_edges = labels[source] == "train"
    np.testing.assert_allclose(
        sparse_attributes[train_edges].mean(axis=0),
        0.0,
        atol=2e-6,
    )

    for name, label in (("train", "train"), ("validation", "val"), ("test", "test")):
        view, node_index = _split_view(name, arrays, sparse, sparse_attributes)
        assert view.num_nodes == int(np.sum(labels == label))
        assert view.num_edges == int(np.sum(labels[source] == label))
        assert np.array_equal(node_index, np.flatnonzero(labels == label))
        if view.edge_index.numel():
            assert int(view.edge_index.min()) >= 0
            assert int(view.edge_index.max()) < view.num_nodes


def test_rewiring_is_degree_preserving_and_split_isolated() -> None:
    arrays = _manual_arrays()
    config = _graph_config()
    base, _ = _make_graph(
        arrays,
        config,
        rewired=False,
        rewire_seed=41,
        swaps_per_edge=0.05,
    )
    rewired, rewired_attributes = _make_graph(
        arrays,
        config,
        rewired=True,
        rewire_seed=41,
        swaps_per_edge=0.05,
        base_graph=base,
    )
    base_degree = np.bincount(base.edge_index[0], minlength=base.n_nodes)
    rewired_degree = np.bincount(
        rewired.edge_index[0], minlength=rewired.n_nodes
    )
    np.testing.assert_array_equal(rewired_degree, base_degree)
    assert rewired.metadata["degree_preserved_exactly"] is True
    assert rewired.metadata["distance_bins_fitted_per_split"] is True
    assert set(rewired.metadata["split_rewires"]) == {
        "train",
        "validation",
        "test",
    }
    assert rewired.metadata["rewire_successful_swaps"] > 0
    labels = arrays["split_labels"]
    source, target = rewired.edge_index
    assert not np.any(labels[source] != labels[target])

    altered = {key: np.array(value, copy=True) for key, value in arrays.items()}
    test_nodes = altered["split_labels"] == "test"
    test_coordinates = altered["coordinates_um"][test_nodes]
    center = test_coordinates.mean(axis=0)
    altered["coordinates_um"][test_nodes] = (
        (test_coordinates - center) * np.asarray([0.55, 0.8])
        + center
        + np.asarray([800.0, -500.0])
    )
    altered_graph, altered_attributes = _make_graph(
        altered,
        config,
        rewired=True,
        rewire_seed=41,
        swaps_per_edge=0.05,
        base_graph=base,
    )
    train_edges = labels[rewired.edge_index[0]] == "train"
    altered_train_edges = (
        altered["split_labels"][altered_graph.edge_index[0]] == "train"
    )
    np.testing.assert_array_equal(
        rewired.edge_index[:, train_edges],
        altered_graph.edge_index[:, altered_train_edges],
    )
    np.testing.assert_allclose(
        rewired_attributes[train_edges],
        altered_attributes[altered_train_edges],
    )

    with pytest.raises(ExperimentContractError, match="swaps_per_edge"):
        _make_graph(
            arrays,
            config,
            rewired=True,
            rewire_seed=0,
            swaps_per_edge=0.0,
        )


def test_edge_controls_are_split_safe_train_fitted_and_deterministic() -> None:
    names = (
        "distance_um",
        "distance_over_radius",
        "dx_over_radius",
        "dy_over_radius",
        "rbf_0",
        "fov_seam",
    )
    split_values = np.repeat(np.asarray(["train", "val", "test"]), 16)
    attributes = np.zeros((len(split_values), len(names)), dtype=np.float32)
    attributes[:, 0] = np.tile(np.repeat(np.arange(4), 4), 3)
    attributes[:, 1] = attributes[:, 0]
    attributes[:, 2] = np.arange(len(attributes))
    attributes[:, 3] = 100 + np.arange(len(attributes))
    attributes[:, 4] = 1.0
    attributes[:, 5] = np.repeat(np.arange(3), 16)
    source = np.arange(len(attributes), dtype=np.int64)
    target = np.concatenate(
        [start + (np.arange(16) + 1) % 16 for start in (0, 16, 32)]
    )
    edge_index = np.stack([source, target])

    zeroed = _edge_control(
        attributes, names, edge_index, split_values, "zero", seed=7
    )
    assert not zeroed.any()
    distance_only = _edge_control(
        attributes, names, edge_index, split_values, "distance_only", seed=7
    )
    np.testing.assert_array_equal(
        distance_only[:, [0, 1, 4]], attributes[:, [0, 1, 4]]
    )
    assert not distance_only[:, [2, 3, 5]].any()

    permuted = _edge_control(
        attributes, names, edge_index, split_values, "permuted", seed=7
    )
    repeated = _edge_control(
        attributes, names, edge_index, split_values, "permuted", seed=7
    )
    np.testing.assert_array_equal(permuted, repeated)
    for split_name in ("train", "val", "test"):
        mask = split_values == split_name
        np.testing.assert_array_equal(
            np.sort(permuted[mask, 2]), np.sort(attributes[mask, 2])
        )

    altered = attributes.copy()
    altered[split_values == "test"] += 10_000
    isolated = _edge_control(
        altered, names, edge_index, split_values, "permuted", seed=7
    )
    np.testing.assert_array_equal(
        permuted[split_values == "train"],
        isolated[split_values == "train"],
    )


def test_run_contract_rejects_inapplicable_and_unknown_controls(
    tmp_path: Path,
) -> None:
    missing_prepared = tmp_path / "not-needed"
    common = {
        "prepared_path": missing_prepared,
        "project_root": tmp_path,
        "model_seed": 1,
    }
    with pytest.raises(ExperimentContractError, match="Unknown graph"):
        run_experiment(
            output_path=tmp_path / "unknown",
            model_name="b1",
            graph_overrides={"typo_k": 2},
            **common,
        )
    with pytest.raises(ExperimentContractError, match="Cell-autonomous"):
        run_experiment(
            output_path=tmp_path / "self-graph",
            model_name="b0",
            graph_overrides={"k": 2},
            **common,
        )
    with pytest.raises(ExperimentContractError, match="cell-autonomous"):
        run_experiment(
            output_path=tmp_path / "self-rewired",
            model_name="b0-matched",
            rewired=True,
            **common,
        )
    with pytest.raises(ExperimentContractError, match="Cell-autonomous"):
        run_experiment(
            output_path=tmp_path / "broad-field-graph",
            model_name="broad-field",
            graph_overrides={"radius_um": 20.0},
            **common,
        )
    with pytest.raises(ExperimentContractError, match="only for G2/G3"):
        run_experiment(
            output_path=tmp_path / "wrong-edge-control",
            model_name="b1",
            edge_control="zero",
            **common,
        )
    with pytest.raises(ExperimentContractError, match="does not consume"):
        run_experiment(
            output_path=tmp_path / "ignored-model-override",
            model_name="b1",
            model_overrides={"message_dim": 8},
            **common,
        )
    with pytest.raises(ExperimentContractError, match="rewired=True"):
        run_experiment(
            output_path=tmp_path / "inactive-rewire-override",
            model_name="g1",
            graph_overrides={"rewire_distance_bins": 3},
            **common,
        )
    with pytest.raises(ExperimentContractError, match="inside the immutable"):
        run_experiment(
            prepared_path=missing_prepared,
            output_path=missing_prepared / "run",
            project_root=tmp_path,
            model_name="b1",
            model_seed=1,
        )
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_open_test_authorization_fails_before_prepared_artifact_access(
    tmp_path: Path,
) -> None:
    missing_prepared = tmp_path / "missing-prepared"
    common = {
        "prepared_path": missing_prepared,
        "project_root": tmp_path,
        "output_path": tmp_path / "never-published",
        "model_name": "b0",
        "model_seed": 0,
        "evaluate_test": True,
    }
    with pytest.raises(
        ExperimentContractError, match="requires a verified standards lock"
    ):
        run_experiment(**common)

    lock_path = _create_synthetic_standards_lock(tmp_path)
    with pytest.raises(
        ExperimentContractError, match="not exactly authorized"
    ):
        run_experiment(
            **common,
            standards_lock_path=lock_path,
        )
    assert not (tmp_path / "never-published").exists()


def test_open_test_run_records_exact_lock_and_job_authorization(
    tmp_path: Path,
) -> None:
    project, prepared = _prepare_synthetic_project(tmp_path)
    lock_path = _create_synthetic_standards_lock(tmp_path)
    output = tmp_path / "authorized-open-test-run"
    run_experiment(
        prepared,
        output,
        project_root=project,
        model_name="b0",
        model_seed=0,
        model_overrides={"hidden_dim": 128},
        training_overrides={
            "device": "cpu",
            "curriculum": "P-only",
            "max_epochs": 1,
            "patience": 1,
            "amp": True,
        },
        evaluate_test=True,
        save_predictions=True,
        standards_lock_path=lock_path,
    )
    manifest = load_run_manifest(output)
    authorization = manifest["standards_lock"]
    assert manifest["sealed_test_opened"] is True
    assert authorization["authorization_version"] == 1
    assert authorization["matrix_role"] == "final_execution"
    assert authorization["final_matrix_file"] == (
        "matrix_final_locked_ladder.yaml"
    )
    assert authorization["condition"] == "b0"
    assert authorization["canonical_job"] == {
        "model": "b0",
        "seed": 0,
        "hidden_dim": 128,
        "curriculum": "P-only",
        "max_epochs": 1,
        "patience": 1,
        "amp": True,
    }
    assert authorization["canonical_job_hash"] == canonical_job_hash(
        authorization["canonical_job"]
    )
    assert len(authorization["lock_id"]) == 20
    assert len(authorization["artifact_id"]) == 16
    assert len(authorization["final_matrix_sha256"]) == 64
    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["test_targets_evaluated"] is True
    assert metrics["test"]


def test_cpu_run_is_sealed_atomic_and_manifest_verified(tmp_path: Path) -> None:
    project, prepared = _prepare_synthetic_project(tmp_path)
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train" / "run_spatial_benchmark.py"),
            "--prepared",
            str(prepared),
            "--output",
            str(output),
            "--model",
            "b1",
            "--seed",
            "101",
            "--device",
            "cpu",
            "--k",
            "2",
            "--radius-um",
            "2.25",
            "--hidden-dim",
            "8",
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--edge-dropout",
            "0",
            "--no-amp",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["output"] == str(output.resolve())
    assert summary["model"] == "b1"
    published = output.resolve()
    manifest = load_run_manifest(published)
    assert manifest["graph"]["config"]["k"] == 2
    assert manifest["graph"]["config"]["radius_um"] == 2.25
    assert manifest["graph"]["qc"]["cross_group_edges"] == 0
    assert manifest["graph"]["edge_control_contract"]["name"] == "none"
    assert manifest["config"]["model"]["name"] == "b1"
    assert "graph_layers" not in manifest["config"]["model"]
    assert "edge_embedding_dim" not in manifest["config"]["model"]
    assert manifest["sealed_test_opened"] is False
    assert manifest["split_node_counts"]["test"] is None
    assert "status_short" not in manifest["provenance"]["git"]
    assert "restricted-synthetic-donor" not in json.dumps(manifest)
    assert len(manifest["run_id"]) == 20
    metrics = json.loads((published / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation"]
    assert metrics["test"] == []
    assert metrics["test_targets_evaluated"] is False
    with np.load(published / "predictions.npz", allow_pickle=False) as archive:
        assert any(name.startswith("validation__") for name in archive.files)
        assert not any(name.startswith("test__") for name in archive.files)
        np.testing.assert_array_equal(
            archive["validation__cell_ids"],
            np.arange(len(archive["validation__cell_ids"])),
        )

    with pytest.raises(FileExistsError):
        run_experiment(
            prepared,
            output,
            project_root=project,
            model_name="b1",
            model_seed=101,
        )
    assert not list(tmp_path.glob(".run.tmp-*"))

    broad_output = tmp_path / "broad-field-run"
    run_experiment(
        prepared,
        broad_output,
        project_root=project,
        model_name="broad-field",
        model_seed=102,
        training_overrides={
            "device": "cpu",
            "max_epochs": 1,
            "patience": 1,
            "edge_dropout": 0.0,
            "amp": False,
        },
        save_predictions=False,
    )
    broad_manifest = load_run_manifest(broad_output)
    assert broad_manifest["graph"]["kind"] == "broad_field"
    assert broad_manifest["graph"]["graph_id"] == "broad-spatial-field"
    assert broad_manifest["graph"]["base_graph_id"] is None
    spatial_control = broad_manifest["spatial_control"]
    assert spatial_control["basis"] == (
        "global_standardized_polynomial_degree_2"
    )
    assert spatial_control["fit_scope"] == "training split coordinates only"
    assert spatial_control["contains_cell_or_region_ids"] is False
    assert spatial_control["contains_graph_or_neighbor_features"] is False
    _, prepared_arrays, _ = load_prepared_artifact(prepared)
    assert prepared_arrays is not None
    training_coordinates = prepared_arrays["coordinates_um"][
        prepared_arrays["split_labels"] == "train"
    ]
    np.testing.assert_allclose(
        spatial_control["center_um"],
        training_coordinates.mean(axis=0),
    )
    np.testing.assert_allclose(
        spatial_control["scale_um"],
        training_coordinates.std(axis=0, ddof=0),
    )
    checkpoint = torch.load(
        broad_output / "model_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["spatial_control"] == spatial_control

    invalid_spatial = tmp_path / "invalid-spatial-provenance"
    shutil.copytree(broad_output, invalid_spatial)
    invalid_manifest_path = invalid_spatial / "manifest.json"
    invalid_manifest = json.loads(
        invalid_manifest_path.read_text(encoding="utf-8")
    )
    invalid_control = dict(invalid_manifest["spatial_control"])
    invalid_control["contains_knots_or_spatial_lookup_embeddings"] = True
    control_core = dict(invalid_control)
    control_core.pop("basis_fit_checksum")
    invalid_control["basis_fit_checksum"] = _canonical_hash(control_core)
    invalid_manifest["spatial_control"] = invalid_control
    invalid_manifest["training"]["spatial_control"] = invalid_control
    invalid_manifest["manifest_content_sha256"] = _manifest_content_hash(
        invalid_manifest
    )
    invalid_manifest_path.write_text(
        json.dumps(invalid_manifest),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentContractError, match="spatial-field provenance"
    ):
        load_run_manifest(invalid_spatial)

    tampered = tmp_path / "tampered-run"
    shutil.copytree(published, tampered)
    with (tampered / "model_state.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ExperimentContractError, match="checksum mismatch"):
        load_run_manifest(tampered)

    tampered_manifest = tmp_path / "tampered-manifest"
    shutil.copytree(published, tampered_manifest)
    manifest_path = tampered_manifest / "manifest.json"
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["model_seed"] += 1
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        ExperimentContractError, match="manifest content checksum"
    ):
        load_run_manifest(tampered_manifest)

    extra_entry = tmp_path / "run-with-extra-entry"
    shutil.copytree(published, extra_entry)
    (extra_entry / "undeclared").mkdir()
    with pytest.raises(
        ExperimentContractError, match="undeclared directory"
    ):
        load_run_manifest(extra_entry)


def test_prepared_fixture_stays_split_safe(tmp_path: Path) -> None:
    _, prepared = _prepare_synthetic_project(tmp_path)
    manifest, arrays, _ = load_prepared_artifact(prepared)
    assert arrays is not None
    restored_graph, restored_attributes = _prepared_graph(manifest, arrays)
    np.testing.assert_array_equal(restored_graph.edge_index, arrays["edge_index"])
    np.testing.assert_array_equal(
        restored_attributes, arrays["edge_attributes"]
    )
    assert manifest["fixed_masks"]["test_targets_evaluated"] is False
    source, target = arrays["edge_index"]
    assert not np.any(
        arrays["split_labels"][source] != arrays["split_labels"][target]
    )

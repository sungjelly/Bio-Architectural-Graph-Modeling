from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.artifacts import (  # noqa: E402
    ArtifactContractError,
    load_prepare_config,
    load_prepared_artifact,
    prepare_artifact,
)
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS  # noqa: E402


def _write_nested_csv(raw_dir: Path, basename: str, frame: pd.DataFrame) -> None:
    directory = raw_dir / basename
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / basename, index=False)


def _make_project(tmp_path: Path) -> tuple[Path, Path, str]:
    project = tmp_path / "synthetic-project"
    raw = project / "data" / "raw"
    clinical = project / "data" / "clinical"
    raw.mkdir(parents=True)
    clinical.mkdir(parents=True)

    restricted_value = "synthetic-restricted-donor-secret"
    private_tissue_unit = 731
    pd.DataFrame(
        [
            ["header", None, None],
            [private_tissue_unit, restricted_value, "정상"],
            [732, "synthetic-other-private-value", "정상인접"],
        ]
    ).to_excel(
        clinical / "Gastric Study_Old.xlsx",
        header=False,
        index=False,
    )
    pd.DataFrame(
        {
            "슬라이드번호": [private_tissue_unit, 732],
            "진단명": ["diagnosis", "diagnosis"],
            "결과": ["reviewed", "reviewed"],
            "비고 (수정진단)": [np.nan, np.nan],
        }
    ).to_excel(clinical / "Gastric Study.xlsx", index=False)
    fov_values = np.arange(1, 10, dtype=int)
    pd.DataFrame(
        {
            "slide": ["SO_2"] * 10,
            "core_label": [private_tissue_unit] * 9 + [732],
            "fov": [*fov_values.tolist(), 10],
        }
    ).to_csv(clinical / "fov_core_map.csv", index=False)

    expression_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    cell_offsets = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0))
    for fov in range(1, 11):
        grid_x = (fov - 1) % 3
        grid_y = (fov - 1) // 3
        for cell_id, (offset_x, offset_y) in enumerate(cell_offsets, start=1):
            expression_rows.append(
                {
                    "fov": fov,
                    "cell_ID": cell_id,
                    "GeneA": (fov + cell_id) % 7,
                    "Negative1": cell_id,
                    "GeneB/C": (2 * fov + cell_id) % 11,
                    "SystemControl1": fov,
                }
            )
            row: dict[str, object] = {
                "fov": fov,
                "cell_ID": cell_id,
                "slide_ID": 2,
                "CenterX_global_px": grid_x * 100.0 + offset_x,
                "CenterY_global_px": grid_y * 100.0 + offset_y,
                "qcCellsPassed": (cell_id % 2) == 1,
            }
            for index, column in enumerate(ALLOWED_METADATA_COLUMNS):
                row[column] = float(index + 1 + cell_id + fov / 10)
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
            "chunksize": 5,
            "expected_biological_probes": 2,
            "pixel_size_um": 1.0,
            "qc_policy": "all",
        },
        "split": {
            "block_size_um": 50.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 9,
            "fov_aware": True,
        },
        "graph": {
            "k": 3,
            "radius_um": 10.0,
            "symmetry": "union",
            "min_distance_um": 0.0,
            "rbf_bins": 4,
        },
        "masking": {
            "mask_seed": 123,
            "curriculum": "P+N+B",
            "warmup_epochs": 2,
            "partial_gene_rate": 0.5,
            "whole_node_rate": 0.25,
            "block_node_rate": 0.25,
            "validation_replicates": 2,
            "test_replicates": 3,
        },
    }
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return project, config_path, restricted_value


def _all_mapping_keys(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key))
            result.extend(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_all_mapping_keys(child))
    return result


def test_prepare_artifact_is_train_fitted_split_safe_and_verifiable(
    tmp_path: Path,
) -> None:
    _, config_path, restricted_value = _make_project(tmp_path)
    output = tmp_path / "prepared"
    prepared = prepare_artifact(
        config_path,
        output,
        command=["synthetic-prepare"],
    )
    assert prepared == output.resolve()
    assert (prepared / "prepared_data.npz").is_file()
    assert (prepared / "manifest.json").is_file()
    assert (prepared / "checksums.sha256").is_file()
    assert (prepared / "fixed_masks" / "validation" / "masks.npz").is_file()
    assert (prepared / "fixed_masks" / "test" / "masks.npz").is_file()

    manifest, arrays, masks = load_prepared_artifact(prepared)
    assert arrays is not None
    manifest_text = json.dumps(manifest, sort_keys=True)
    assert restricted_value not in manifest_text
    keys = _all_mapping_keys(manifest)
    assert "donor_id" not in keys
    assert "core_label" not in keys
    assert manifest["selection"]["restricted_identifiers_emitted"] is False
    assert manifest["features"]["n_biological_probes"] == 2
    assert manifest["features"]["measured_metadata_names"] == list(
        ALLOWED_METADATA_COLUMNS
    )
    assert manifest["features"]["coordinates_are_model_covariates"] is False
    assert manifest["features"]["qc_indicator_is_model_covariate"] is False
    assert manifest["split"]["fov_disjoint"] is True
    assert manifest["split"]["assignment_precedes_fitted_preprocessing"] is True
    assert manifest["split"]["test_usage_during_preparation"] == (
        "fixed mask generation only"
    )
    assert manifest["fixed_masks"]["test_targets_evaluated"] is False
    assert set(masks) == {"validation", "test"}
    assert set(masks["validation"].manifest["splits"]) == {"validation"}
    assert set(masks["test"].manifest["splits"]) == {"test"}
    assert masks["validation"].manifest["replicates"] == 2
    assert masks["test"].manifest["replicates"] == 3

    labels = arrays["split_labels"]
    fov = arrays["fov"]
    for value in np.unique(fov):
        assert np.unique(labels[fov == value]).size == 1
    source, target = arrays["edge_index"]
    assert not np.any(labels[source] != labels[target])
    assert manifest["graph"]["cross_split_edges"] == 0
    assert manifest["graph"]["split_graphs_constructed_independently"] is True
    for split_name in ("train", "validation", "test"):
        local_edges = arrays[f"{split_name}_edge_index"]
        n_nodes = len(arrays[f"{split_name}_node_index"])
        if local_edges.size:
            assert local_edges.min() >= 0
            assert local_edges.max() < n_nodes
        assert local_edges.shape[1] == arrays[
            f"{split_name}_edge_attributes"
        ].shape[0]

    train_nodes = arrays["train_node_index"]
    np.testing.assert_allclose(
        arrays["target_expression"][train_nodes].mean(axis=0),
        0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        arrays["node_covariates"][train_nodes].mean(axis=0),
        0.0,
        atol=2e-6,
    )
    train_edges = arrays["train_edge_mask"]
    np.testing.assert_allclose(
        arrays["edge_attributes"][train_edges].mean(axis=0),
        0.0,
        atol=2e-5,
    )
    assert masks["validation"].manifest["splits"]["validation"]["n_nodes"] == len(
        arrays["validation_node_index"]
    )
    assert masks["test"].manifest["splits"]["test"]["n_nodes"] == len(
        arrays["test_node_index"]
    )
    with pytest.raises(FileExistsError):
        prepare_artifact(config_path, prepared)


def test_config_rejects_an_identifier_override_and_failure_is_atomic(
    tmp_path: Path,
) -> None:
    _, config_path, _ = _make_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["donor_id"] = "must-not-be-configurable"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="Unknown"):
        load_prepare_config(invalid_path)

    # Ambiguous exact labels fail before publication; no partial destination or
    # sibling temporary directory remains.
    project = Path(config["project_root"])
    legacy = project / "data" / "clinical" / "Gastric Study_Old.xlsx"
    pd.DataFrame([[731, "private-a", "정상"], [732, "private-b", "정상"]]).to_excel(
        legacy,
        header=False,
        index=False,
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(Exception):
        prepare_artifact(config_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-exist.tmp-*"))


def test_canonical_full_workflow_configs_are_valid_prepare_inputs() -> None:
    for name in (
        "normal_core_spatial_benchmark_legacy.yaml",
        "normal_core_spatial_benchmark_smoke_legacy.yaml",
    ):
        config = load_prepare_config(PROJECT_ROOT / "configs" / "experiment" / name)
        assert config["data"]["expected_biological_probes"] == 1000
        assert config["split"]["fov_aware"] is True
        assert config["masking"]["validation_replicates"] > 0
        assert config["masking"]["test_replicates"] > 0
        assert config["model"]["name"] == "g1"


def test_prepare_cli_and_tamper_detection(tmp_path: Path) -> None:
    _, config_path, restricted_value = _make_project(tmp_path)
    output = tmp_path / "cli-prepared"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data" / "prepare_spatial_benchmark.py"),
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["output"] == str(output.resolve())
    assert summary["n_cells"] == 36
    assert summary["n_biological_probes"] == 2
    assert summary["test_targets_evaluated"] is False
    assert restricted_value not in completed.stdout

    data_path = output / "prepared_data.npz"
    with data_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactContractError, match="checksum mismatch"):
        load_prepared_artifact(output)

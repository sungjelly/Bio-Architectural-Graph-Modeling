from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatial_benchmark.artifacts import (  # noqa: E402
    load_prepared_artifact,
    prepare_artifact,
)
from spatial_benchmark.diagnostic_artifacts import (  # noqa: E402
    DIAGNOSTIC_ARTIFACT_KIND,
    DiagnosticArtifactError,
    load_diagnostic_artifact,
    run_diagnostic_artifact,
)
from test_prepare import _make_project  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_prepared(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("diagnostic-artifact")
    _, config_path, restricted_value = _make_project(root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["masking"]["validation_replicates"] = 1
    config["masking"]["test_replicates"] = 1
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    prepared = prepare_artifact(
        config_path,
        root / "prepared",
        command=["synthetic-diagnostic-prepare"],
    )
    return prepared, restricted_value


def _records_by_key(
    records: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(record["mask_entry_id"]), str(record["control"])): record
        for record in records
    }


def test_diagnostic_cli_is_validation_only_by_default_and_tamper_evident(
    synthetic_prepared: tuple[Path, str],
    tmp_path: Path,
) -> None:
    prepared, restricted_value = synthetic_prepared
    output = tmp_path / "diagnostic-validation"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate" / "run_diagnostics.py"),
            "--prepared",
            str(prepared),
            "--output",
            str(output),
            "--min-distance-um",
            "1.0",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "diagnostic_id": summary["diagnostic_id"],
        "output": str(output.resolve()),
        "predictions_saved": False,
        "test_records": 0,
        "test_targets_evaluated": False,
        "validation_records": 6,
    }
    assert len(summary["diagnostic_id"]) == 20
    assert restricted_value not in completed.stdout

    manifest, metrics, predictions = load_diagnostic_artifact(output)
    prepared_manifest, _, bundles = load_prepared_artifact(
        prepared, load_arrays=False
    )
    assert predictions is None
    assert manifest["artifact_kind"] == DIAGNOSTIC_ARTIFACT_KIND
    assert manifest["evaluated_splits"] == ["validation"]
    assert manifest["sealed_test_opened"] is False
    assert manifest["split_node_counts"]["test"] is None
    assert set(manifest["evaluated_mask_entries"]) == {"validation"}
    assert set(
        manifest["prepared_artifact"]["evaluated_mask_bundles"]
    ) == {"validation"}
    assert set(
        manifest["run_contract"]["evaluated_mask_bundles"]
    ) == {"validation"}
    assert metrics["test"] == []
    assert metrics["test_targets_evaluated"] is False
    assert len(metrics["validation"]) == 6
    assert len(manifest["evaluations"]) == 6
    assert not (output / "predictions.npz").exists()
    assert (output / "checksums.sha256").is_file()
    assert bundles["test"].bundle_id not in json.dumps(
        manifest, sort_keys=True
    )
    assert prepared_manifest["fixed_masks"]["test_targets_evaluated"] is False

    records = _records_by_key(metrics["validation"])
    assert len(records) == 6
    for record in records.values():
        assert len(record["mask_checksum"]) == 64
        assert record["fit_scope"] == "train nodes only"
        assert record["candidate_scope"] == "validation nodes only"
        assert record["metrics"]["n_masked"] == record[
            "n_evaluated_entries"
        ]
        assert record["metrics"]["blocks"]
        assert sum(
            block["n_masked"] for block in record["metrics"]["blocks"]
        ) == record["n_evaluated_entries"]
        if record["control"] == "train_global_gene_mean":
            assert record["source_copy_rate"] == 0.0
            assert record["min_distance_um"] is None
        else:
            assert record["min_distance_um"] == 1.0
            assert 0.0 <= record["source_copy_rate"] <= 1.0

    serialized = (
        (output / "manifest.json").read_text(encoding="utf-8")
        + (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert restricted_value not in serialized
    with pytest.raises(FileExistsError):
        run_diagnostic_artifact(prepared, output)

    with (output / "metrics.json").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(DiagnosticArtifactError, match="checksum mismatch"):
        load_diagnostic_artifact(output)


def test_open_test_predictions_are_train_fitted_visible_and_split_local(
    synthetic_prepared: tuple[Path, str],
    tmp_path: Path,
) -> None:
    prepared, restricted_value = synthetic_prepared
    output = run_diagnostic_artifact(
        prepared,
        tmp_path / "diagnostic-open-test",
        min_distance_um=1.0,
        open_test=True,
        save_predictions=True,
        distance_block_size=2,
        command=["synthetic-open-test"],
    )
    manifest, metrics, prediction_arrays = load_diagnostic_artifact(
        output, load_predictions=True
    )
    assert prediction_arrays is not None
    assert manifest["evaluated_splits"] == ["validation", "test"]
    assert manifest["sealed_test_opened"] is True
    assert manifest["split_node_counts"]["test"] > 0
    assert manifest["run_contract"]["distance_block_size"] == 2
    assert metrics["test_targets_evaluated"] is True
    assert len(metrics["validation"]) == 6
    assert len(metrics["test"]) == 6
    assert any(key.startswith("test__") for key in prediction_arrays)
    assert restricted_value not in (
        (output / "manifest.json").read_text(encoding="utf-8")
        + (output / "metrics.json").read_text(encoding="utf-8")
    )

    _, arrays, _ = load_prepared_artifact(prepared)
    assert arrays is not None
    train_mean = np.asarray(
        arrays["target_expression"][arrays["train_node_index"]],
        dtype=np.float64,
    ).mean(axis=0)
    for declaration in manifest["evaluations"]:
        split = declaration["split"]
        split_index = arrays[f"{split}_node_index"]
        split_expression = np.asarray(
            arrays["target_expression"][split_index], dtype=np.float64
        )
        keys = declaration["prediction_arrays"]
        mask = prediction_arrays[keys["mask"]]
        prediction = prediction_arrays[keys["prediction"]]
        masked_rows, masked_genes = np.nonzero(mask)
        if declaration["control"] == "train_global_gene_mean":
            np.testing.assert_allclose(
                prediction[masked_rows, masked_genes],
                train_mean[masked_genes],
                rtol=1e-6,
                atol=1e-6,
            )
            continue

        source = prediction_arrays[keys["source_node_index"]]
        distance = prediction_arrays[keys["source_distance_um"]]
        copied = mask & (source >= 0)
        copied_rows, copied_genes = np.nonzero(copied)
        copied_sources = source[copied_rows, copied_genes]
        assert np.all(copied_sources < len(split_index))
        assert np.all(copied_sources != copied_rows)
        assert not np.any(mask[copied_sources, copied_genes])
        assert np.all(distance[copied] >= 1.0)
        np.testing.assert_allclose(
            prediction[copied_rows, copied_genes],
            split_expression[copied_sources, copied_genes],
            rtol=1e-6,
            atol=1e-6,
        )
        fallback = mask & (source < 0)
        fallback_rows, fallback_genes = np.nonzero(fallback)
        np.testing.assert_allclose(
            prediction[fallback_rows, fallback_genes],
            train_mean[fallback_genes],
            rtol=1e-6,
            atol=1e-6,
        )

    invalid = tmp_path / "invalid-distance"
    with pytest.raises(DiagnosticArtifactError, match="min_distance_um"):
        run_diagnostic_artifact(
            prepared,
            invalid,
            min_distance_um=-1.0,
        )
    assert not invalid.exists()
    assert not list(tmp_path.glob(".invalid-distance.tmp-*"))
    with pytest.raises(
        DiagnosticArtifactError, match="inside its prepared artifact"
    ):
        run_diagnostic_artifact(
            prepared,
            prepared / "forbidden-diagnostic",
        )
    assert not (prepared / "forbidden-diagnostic").exists()

    sealed_output = run_diagnostic_artifact(
        prepared,
        tmp_path / "diagnostic-sealed-predictions",
        min_distance_um=1.0,
        save_predictions=True,
        distance_block_size=1,
        command=["synthetic-sealed-predictions"],
    )
    sealed_manifest, sealed_metrics, sealed_predictions = (
        load_diagnostic_artifact(
            sealed_output,
            load_predictions=True,
        )
    )
    assert sealed_predictions is not None
    assert sealed_manifest["evaluated_splits"] == ["validation"]
    assert sealed_metrics["test"] == []
    assert sealed_metrics["validation"] == metrics["validation"]
    assert not any(
        key.startswith("test__") for key in sealed_predictions
    )

    dangling = tmp_path / "dangling-diagnostic"
    dangling.symlink_to(tmp_path / "dangling-target", target_is_directory=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_diagnostic_artifact(prepared, dangling)
    assert dangling.is_symlink()
    assert not (tmp_path / "dangling-target").exists()

    with (output / "predictions.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(DiagnosticArtifactError, match="checksum mismatch"):
        load_diagnostic_artifact(output)


def test_zero_rate_fixed_masks_remain_valid_diagnostic_records(
    tmp_path: Path,
) -> None:
    _, config_path, _ = _make_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["masking"].update(
        {
            "partial_gene_rate": 0.0,
            "whole_node_rate": 0.0,
            "block_node_rate": 0.0,
            "validation_replicates": 1,
            "test_replicates": 1,
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    prepared = prepare_artifact(config_path, tmp_path / "zero-prepared")
    output = run_diagnostic_artifact(
        prepared,
        tmp_path / "zero-diagnostics",
    )
    _, metrics, _ = load_diagnostic_artifact(output)
    assert len(metrics["validation"]) == 6
    assert metrics["test"] == []
    for record in metrics["validation"]:
        assert record["n_evaluated_entries"] == 0
        assert record["n_copied_entries"] == 0
        assert record["n_fallback_mean_entries"] == 0
        assert record["source_copy_rate"] == 0.0
        assert record["metrics"]["n_masked"] == 0
        assert record["metrics"]["huber"] is None
        assert sum(
            block["n_masked"] for block in record["metrics"]["blocks"]
        ) == 0

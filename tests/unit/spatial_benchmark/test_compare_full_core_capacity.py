from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest

from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "analysis" / "compare_full_core_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "compare_full_core_capacity_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CapacityComparisonError = _MODULE.CapacityComparisonError
compare_full_core_capacity = _MODULE.compare_full_core_capacity
write_comparison = _MODULE.write_comparison


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def _run_id(role: str) -> str:
    return create_run_id(
        seed=0,
        fold=0,
        attempt=1,
        scientific_id_value=f"sci_{role}_1234567890abcdef",
        timestamp=datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
        unique_suffix=f"{role}01",
    )


def _checksum(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_exact_table(
    archive: RunArchive,
    stem: str,
    rows: Sequence[Mapping[str, Any]],
    table_format: str,
) -> None:
    if table_format == "jsonl":
        archive.write_text(
            f"{stem}.jsonl",
            "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        )
        return
    assert table_format == "csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    archive.write_text(f"{stem}.csv", buffer.getvalue())


def _history_rows(
    *,
    gradient_spike: bool = False,
    same_mode_deterioration: bool = False,
) -> list[dict[str, Any]]:
    modes = ("partial_gene", "whole_node", "spatial_block")
    rows = []
    for epoch in range(200):
        train_loss = 1.0 - 0.001 * epoch
        if same_mode_deterioration and epoch == 199:
            train_loss = 2.0
        rows.append(
            {
                "run_id": "bound-by-table-context",
                "split": "fit",
                "training_protocol": "held_in_full_core_fixed_budget",
                "epoch": epoch,
                "mask_mode": modes[epoch % len(modes)],
                "mask_seed": 1000 + epoch,
                "mask_checksum": _checksum(f"epoch-mask-{epoch}"),
                "edge_dropout_seed": 2000 + epoch,
                "edge_checksum": _checksum("graph"),
                "n_masked_entries": 100 + epoch,
                "n_target_nodes": 10 + epoch,
                "n_edges_used": 5000,
                "train_loss": train_loss,
                "diagnostic_loss": train_loss,
                "gradient_norm": 11.0 if gradient_spike and epoch == 199 else 1.0,
                "duration_seconds": 0.1,
                "peak_cuda_memory_bytes": 1024,
            }
        )
    return rows


def _evaluation_rows(
    whole_node_huber: Sequence[float],
    *,
    mismatched_whole_replicate: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = {
        "partial_gene": (0.20, 0.21, 0.22),
        "whole_node": tuple(whole_node_huber),
        "spatial_block": (0.50, 0.51, 0.52),
    }
    for mode, losses in values.items():
        for replicate, loss in enumerate(losses):
            checksum_label = f"evaluation-{mode}-{replicate}"
            if mode == "whole_node" and replicate == mismatched_whole_replicate:
                checksum_label += "-mismatch"
            rows.append(
                {
                    "split": "fit",
                    "mask_mode": mode,
                    "mask_replicate": replicate,
                    "mask_entry_id": f"fit-{mode}-{replicate}",
                    "mask_seed": 3000 + replicate,
                    "mask_checksum": _checksum(checksum_label),
                    "n_masked": 1000 + replicate,
                    "masked_huber": loss,
                    "masked_mse": loss * 2.0,
                    "masked_mae": loss / 2.0,
                }
            )
    return rows


def _config(model_name: str) -> dict[str, Any]:
    node_contract = {
        "fit_scope": "all_nodes_transductive",
        "node_expression": {
            "biological_targets": 1000,
            "transformed_with": "full_core_fit",
            "explicit_mask_channel": True,
        },
        "node_metadata": {
            "transformed_with": "full_core_fit",
            "fields": ["Area", "Width"],
        },
        "prohibited_node_inputs": [
            "direct_identifiers",
            "absolute_or_local_coordinates",
        ],
    }
    is_g2 = model_name == "g2"
    return {
        "model": {"name": model_name},
        "dataset": {
            "dataset_id": "synthetic_full_core",
            "version": "v1",
            "dataset_fingerprint": _checksum("dataset"),
            "preprocessing_version": "full_core_fit_v1",
            "split_id": _checksum("split")[:16],
            "split_fingerprint": _checksum("split"),
            "experimental_unit": "single_spatial_core",
            "validation_or_test_partition_present": False,
        },
        "features": {
            **node_contract,
            "use_edge_features": is_g2,
            "edge_features": (
                {"fields": ["distance_um"]} if is_g2 else []
            ),
        },
        "graph": {
            "kind": "exact_spatial_knn_radius_guard",
            "k": 1000,
            "neighbor_k": 1000,
            "radius_um": 650.0,
            "symmetry": "mutual",
            "full_core_graph": True,
            "edge_dropout": 0.0,
        },
        "masking": {
            "fit_replicates": 3,
            "validation_replicates": 0,
            "test_replicates": 0,
            "mask_seed": 314159,
        },
        "trainer": {
            "max_epochs": 200,
            "fixed_epoch_budget": True,
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "neighbor_sampling": False,
            "graph_execution": "full_core_exact_no_neighbor_sampling",
        },
        "evaluation": {
            "protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": "fit/whole_node/masked_huber",
            "splits": ["fit"],
            "mask_replicates_per_mode": 3,
            "fixed_mask_bundle": True,
            "generalization_estimate": False,
            "validation_or_test_selection": False,
        },
        "campaign": {
            "campaign_id": "cmp_20260725_full_core_high_k_capacity"
        },
        "seed": 0,
        "fold": 0,
    }


def _make_bundle(
    root: Path,
    *,
    role: str,
    whole_node_huber: Sequence[float],
    table_format: str = "jsonl",
    parameter_count: int = 123_456,
    primary_override: float | None = None,
    gradient_spike: bool = False,
    same_mode_deterioration: bool = False,
    mismatched_whole_replicate: int | None = None,
) -> Path:
    model_name = "g2" if role == "g2" else "b0_g2_matched"
    run_id = _run_id(role)
    config = _config(model_name)
    archive = RunArchive.create(
        run_id,
        paths=_paths(root),
        manifest={"status": "success"},
        resolved_config=config,
    )
    history = _history_rows(
        gradient_spike=gradient_spike,
        same_mode_deterioration=same_mode_deterioration,
    )
    evaluations = _evaluation_rows(
        whole_node_huber,
        mismatched_whole_replicate=mismatched_whole_replicate,
    )
    primary = (
        sum(whole_node_huber) / len(whole_node_huber)
        if primary_override is None
        else primary_override
    )
    final_metrics = {
        "fit/whole_node/masked_huber": primary,
        "resource/parameter_count": parameter_count,
    }
    archive.append_metric_event(
        {
            "name": "fit/whole_node/masked_huber",
            "value": primary,
            "step": 199,
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    _write_exact_table(archive, "metrics/history", history, table_format)
    _write_exact_table(
        archive,
        "metrics/evaluation_replicates",
        evaluations,
        table_format,
    )
    archive.write_predictions(
        "fit",
        [
            {
                "run_id": run_id,
                "sample_key": "sk_synthetic_cell",
                "dataset_id": "synthetic_full_core",
                "split": "fit",
                "y_true": [0.0, 1.0],
                "y_pred": [0.1, 0.9],
            }
        ],
    )
    archive.write_bytes("checkpoints/last.ckpt", b"synthetic-checkpoint")
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_id": "synthetic_full_core",
            "dataset_fingerprint": _checksum("dataset"),
            "preprocessing_version": "full_core_fit_v1",
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {
            "split_id": _checksum("split")[:16],
            "split_fingerprint": _checksum("split"),
        },
    )
    archive.write_text(
        "provenance/command.txt",
        '{"argv":["synthetic"],"cwd":"/tmp"}\n',
    )
    archive.write_json(
        "provenance/full_core_training.json",
        {
            "training_protocol": "held_in_full_core_fixed_budget",
            "graph_execution": "full_core_exact_no_neighbor_sampling",
            "checkpoint_policy": "final_epoch_no_validation_selection",
            "fixed_epoch_budget": 200,
            "final_epoch": 199,
            "model_seed": 0,
            "parameter_count": parameter_count,
        },
    )
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "preprocessing_checksums": {
                "preprocessing_sha256": _checksum("dataset")
            },
            "materialized_identity_verification": {
                "dataset_fingerprint": _checksum("dataset"),
                "split_fingerprint": _checksum("split"),
            },
            "graph_checksums": {"graph_sha256": _checksum("graph")},
            "graph_config": config["graph"],
        },
    )
    mask_manifest = {
        "bundle_checksum": _checksum("evaluation-mask-bundle"),
        "entries": [
            {
                "entry_id": row["mask_entry_id"],
                "seed": row["mask_seed"],
                "mask_checksum": row["mask_checksum"],
                "replicate": row["mask_replicate"],
                "split": "fit",
                "spec": {"label": row["mask_mode"]},
                "summary": {"n_masked_entries": row["n_masked"]},
            }
            for row in _evaluation_rows(whole_node_huber)
        ],
    }
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": mask_manifest,
            "used_for_gradient_updates": False,
            "used_for_checkpoint_selection": False,
        },
    )
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "final_epoch": 199,
            "all_epochs_completed": True,
            "all_losses_and_gradients_finite": True,
        },
    )
    archive.write_summary(
        {
            "status": "success",
            "training_exit_status": "success",
            "evaluation_protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "model_name": model_name,
            "model_seed": 0,
            "final_epoch": 199,
            "fixed_epoch_budget": 200,
            "checkpoint_role": "last",
            "primary_metric_name": "fit/whole_node/masked_huber",
            "primary_metric_value": primary,
            "metrics": final_metrics,
            "parameter_count": parameter_count,
            "graph_sha256": _checksum("graph"),
            "graph_directed_edges": 5000,
            "evaluation_mask_bundle_sha256": _checksum(
                "evaluation-mask-bundle"
            ),
            "evaluation_mask_replicates_per_mode": 3,
            "evaluation_metrics_include_all_configured_replicates_per_mode": True,
            "canonical_prediction_selection": {
                "split": "fit",
                "mask_mode": "whole_node",
                "mask_replicate": 0,
            },
            "diagnostic_resource_pilot": False,
            "conclusion_eligible": True,
            "generalization_estimate": False,
        }
    )
    return archive.finalize_success()


@pytest.mark.parametrize("table_format", ("jsonl", "csv"))
def test_comparison_passes_locked_gate_and_writes_exclusive_outputs(
    tmp_path: Path,
    table_format: str,
) -> None:
    g2 = _make_bundle(
        tmp_path,
        role="g2",
        whole_node_huber=(0.97, 0.96, 0.95),
        table_format=table_format,
    )
    matched_self = _make_bundle(
        tmp_path,
        role="self",
        whole_node_huber=(1.00, 0.99, 0.98),
        table_format=table_format,
    )

    result = compare_full_core_capacity(g2, matched_self)

    assert result["locked_gate"]["passes"] is True
    assert result["facts"]["equal_parameter_count"] is True
    assert result["facts"]["parameter_count"] == 123_456
    assert result["facts"]["relative_mean_g2_gain"] == pytest.approx(
        (0.99 - 0.96) / 0.99
    )
    assert [
        row["self_minus_g2_masked_huber"]
        for row in result["facts"]["paired_whole_node_replicates"]
    ] == pytest.approx([0.03, 0.03, 0.03])
    output = write_comparison(result, tmp_path / "comparison")
    assert json.loads((output / "comparison.json").read_text())["status"] == "complete"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "not an estimate of generalization" in report
    assert "does not verify the earlier held-out predictive hypothesis" in report
    with pytest.raises(CapacityComparisonError, match="output already exists"):
        write_comparison(result, output)


def test_comparison_rejects_replicate_mean_drift(tmp_path: Path) -> None:
    g2 = _make_bundle(
        tmp_path,
        role="g2",
        whole_node_huber=(0.97, 0.96, 0.95),
        primary_override=0.95,
    )
    matched_self = _make_bundle(
        tmp_path,
        role="self",
        whole_node_huber=(1.00, 0.99, 0.98),
    )

    with pytest.raises(CapacityComparisonError, match="replicate mean"):
        compare_full_core_capacity(g2, matched_self)


def test_comparison_rejects_unequal_parameter_counts(tmp_path: Path) -> None:
    g2 = _make_bundle(
        tmp_path,
        role="g2",
        whole_node_huber=(0.97, 0.96, 0.95),
        parameter_count=123_456,
    )
    matched_self = _make_bundle(
        tmp_path,
        role="self",
        whole_node_huber=(1.00, 0.99, 0.98),
        parameter_count=123_457,
    )

    with pytest.raises(CapacityComparisonError, match="parameter counts differ"):
        compare_full_core_capacity(g2, matched_self)


@pytest.mark.parametrize(
    ("audit_argument", "reason"),
    (
        (
            {"gradient_spike": True},
            "last-20 gradient maximum exceeds 10x",
        ),
        (
            {"same_mode_deterioration": True},
            "last loss exceeds 1.25x",
        ),
    ),
)
def test_comparison_records_prespecified_divergence_as_negative_gate(
    tmp_path: Path,
    audit_argument: Mapping[str, bool],
    reason: str,
) -> None:
    g2 = _make_bundle(
        tmp_path,
        role="g2",
        whole_node_huber=(0.97, 0.96, 0.95),
        **audit_argument,
    )
    matched_self = _make_bundle(
        tmp_path,
        role="self",
        whole_node_huber=(1.00, 0.99, 0.98),
    )

    result = compare_full_core_capacity(g2, matched_self)

    assert result["locked_gate"]["passes"] is False
    assert (
        result["locked_gate"]["criteria"][
            "neither_run_has_unresolved_divergence"
        ]
        is False
    )
    assert any(
        reason in item
        for item in result["training_audit"]["g2"]["divergence_reasons"]
    )


def test_comparison_rejects_evaluation_rows_stale_against_mask_manifest(
    tmp_path: Path,
) -> None:
    g2 = _make_bundle(
        tmp_path,
        role="g2",
        whole_node_huber=(0.97, 0.96, 0.95),
    )
    matched_self = _make_bundle(
        tmp_path,
        role="self",
        whole_node_huber=(1.00, 0.99, 0.98),
        mismatched_whole_replicate=1,
    )

    with pytest.raises(CapacityComparisonError, match="fixed-mask manifest"):
        compare_full_core_capacity(g2, matched_self)

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import pytest

from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "analysis"
    / "compare_full_core_qkv_large_k.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_full_core_qkv_large_k_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

QKVLargeKComparisonError = _MODULE.QKVLargeKComparisonError
compare_full_core_qkv_large_k = (
    _MODULE.compare_full_core_qkv_large_k
)
write_comparison = _MODULE.write_comparison

_CAMPAIGN = "cmp_20260726_full_core_qkv_large_k"
_DATASET_SHA256 = (
    "a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5"
)
_SPLIT_SHA256 = (
    "2c8c59fb659401cc202126cb14154f064d2e3380819aff647f06a30db354d84c"
)
_GRAPH = {
    "k1000": (
        1000,
        "23d9b09af45fca21e4f30ef04a6921765b19399eff3e3ef78e371f6b019004e6",
        21_029_944,
    ),
    "k5000": (
        5000,
        "50f293972c443011a80abfbd81bf7cc7f44a35ccb39794e900b67dc4d2ca8d85",
        101_237_016,
    ),
    "matched_self": (
        5000,
        "50f293972c443011a80abfbd81bf7cc7f44a35ccb39794e900b67dc4d2ca8d85",
        101_237_016,
    ),
}
_PARAMETERS = 111_177_064
_EDGE_FIELDS = [
    "distance_um",
    "distance_over_radius",
    "log1p_distance_um",
    "delta_x_over_radius",
    "delta_y_over_radius",
    "cos_theta",
    "sin_theta",
    "cos_2theta",
    "sin_2theta",
    "distance_rbf_0",
    "distance_rbf_1",
    "distance_rbf_2",
    "distance_rbf_3",
    "distance_rbf_4",
    "distance_rbf_5",
    "distance_rbf_6",
    "distance_rbf_7",
]


def _checksum(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_GIT_COMMIT = "6081a49496f3848594649bc3705573e7818df545"
_DIRTY_FINGERPRINT = _checksum("shared-dirty-tree")
_SOFTWARE_ENVIRONMENT_FINGERPRINT = _checksum(
    "shared-software-environment"
)
_REPRODUCTION_ENVIRONMENT_FINGERPRINT = _checksum(
    "shared-software-and-hardware-environment"
)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def _run_id(role: str) -> str:
    token = {
        "k1000": "k1",
        "k5000": "k5",
        "matched_self": "ms",
    }[role]
    return create_run_id(
        seed=0,
        fold=0,
        attempt=1,
        scientific_id_value=f"sci_{token}1234567890abcdef",
        timestamp=datetime(2026, 7, 26, 15, 0, tzinfo=timezone.utc),
        unique_suffix=f"{token}qkv01",
    )


def _write_jsonl(
    archive: RunArchive,
    relative_path: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    archive.write_text(
        relative_path,
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n" for row in rows
        ),
    )


def _model(role: str, *, hidden_dim: int = 1024) -> dict[str, Any]:
    is_graph = role != "matched_self"
    model = {
        "name": "qkv-gat" if is_graph else "qkv-gat-matched-self",
        "family": (
            "edge_aware_qkv_graph_transformer"
            if is_graph
            else "qkv_parameter_matched_self_control"
        ),
        "embedding_dim": hidden_dim,
        "hidden_dim": hidden_dim,
        "graph_layers": 8,
        "attention_heads": 16,
        "attention_head_dim": 64,
        "ffn_dim": 4096,
        "decoder_dim": 4096,
        "edge_hidden_dim": 256,
        "edge_embedding_dim": 128,
        "edge_conditioning_mode": "bias_gate",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "receiver_chunk_size": 64,
        "max_edges_per_chunk": 200_000,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "uses_graph_inputs": is_graph,
        "uses_edge_inputs": is_graph,
    }
    if not is_graph:
        model.update(
            {
                "edge_attribute_dim": 17,
                "parameter_match_reference": (
                    "edge_aware_qkv_graph_transformer"
                ),
                "parameter_matching_method": (
                    "repurpose_qkv_and_edge_parameters_as_within_cell_routing"
                ),
                "comparison_target": (
                    "qkv_gat_same_data_masks_width_depth_seed_optimizer_and_budget"
                ),
            }
        )
    return model


def _config(
    role: str,
    *,
    graph_sha256: str,
    graph_k: int,
    graph_edges: int,
    epochs: int,
    hidden_dim: int,
) -> dict[str, Any]:
    is_graph = role != "matched_self"
    node_features = {
        "fit_scope": "all_nodes_transductive",
        "node_expression": {
            "biological_targets": 1000,
            "transformed_with": (
                "full_core_fitted_gene_wise_standardized_log1p"
            ),
            "masked_values_filled_with": 0.0,
            "explicit_mask_channel": True,
        },
        "node_metadata": {
            "transformed_with": (
                "full_core_fitted_median_imputation_log1p_standardization"
            ),
            "fields": ["Area", "Width"],
        },
        "prohibited_node_inputs": [
            "direct_identifiers",
            "absolute_or_local_coordinates",
            "expression_derived_library_size",
            "vendor_cell_type_cluster_neighborhood_or_niche",
        ],
    }
    return {
        "version": 1,
        "seed": 0,
        "fold": 0,
        "attempt": 1,
        "campaign": {"campaign_id": _CAMPAIGN},
        "experiment": {
            "variant_label": f"synthetic_{role}",
            "estimand": (
                "held_in_full_core_whole_node_masked_reconstruction"
            ),
            "permitted_claim": (
                "one_core_transductive_representation_capacity"
            ),
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": "exploratory_screen",
            "study_axis": "full_core_qkv_large_k_capacity",
            "retention_class": "retain_exploratory_evidence",
            "classification_confidence": "high",
        },
        "model": _model(role, hidden_dim=hidden_dim),
        "dataset": {
            "dataset_id": "cosmx_normal_core_full_core_fit_v1",
            "version": "full_core_fit_v1",
            "dataset_fingerprint": _DATASET_SHA256,
            "preprocessing_version": "full_core_fit_v1",
            "split_id": _SPLIT_SHA256[:16],
            "split_fingerprint": _SPLIT_SHA256,
            "experimental_unit": "single_spatial_core",
            "preprocessing_fit_scope": "all_nodes_transductive",
            "validation_or_test_partition_present": False,
        },
        "features": {
            **node_features,
            "use_edge_features": is_graph,
            "edge_features": (
                {
                    "fit_scope": (
                        "all_retained_directed_edges_transductive"
                    ),
                    "standardization": "full_core_edge_wise",
                    "fields": _EDGE_FIELDS,
                }
                if is_graph
                else []
            ),
        },
        "graph": {
            "kind": "exact_spatial_knn_radius_guard",
            "neighbor_k": graph_k,
            "k": graph_k,
            "radius_um": 1200.0,
            "radius_guard_um": 1200.0,
            "radius_role": "common_nontruncating_post_knn_guard",
            "candidate_selection": (
                "exact_knn_before_radius_guard_validation"
            ),
            "symmetry": "mutual",
            "min_distance_um": 0.0,
            "rbf_bins": 8,
            "edge_dropout": 0.0,
            "self_loops": False,
            "two_directed_edges_per_relation": True,
            "cross_split_edges": False,
            "split_graphs_constructed_independently": False,
            "full_core_graph": True,
            "coordinates_are_node_covariates": False,
            "exact_search_epsilon": 0.0,
            "expected_materialized_graph_sha256": graph_sha256,
            "expected_directed_edges": graph_edges,
        },
        "masking": {
            "type": "mixed_expression_masking",
            "curriculum": "P+N+B",
            "rates": {
                "partial_gene": 0.2,
                "whole_node": 0.1,
                "spatial_block": 0.1,
            },
            "warmup_epochs": 10,
            "mask_seed": 314159,
            "fit_replicates": 3,
            "validation_replicates": 0,
            "test_replicates": 0,
        },
        "trainer": {
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "gradient_clip_norm": 1.0,
            "max_epochs": epochs,
            "fixed_epoch_budget": True,
            "early_stopping": False,
            "validation_every": None,
            "restore_best": False,
            "monitored_metric": None,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
            "neighbor_sampling": False,
            "graph_execution": (
                "full_core_exact_no_neighbor_sampling"
            ),
        },
        "evaluation": {
            "protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": "fit/whole_node/masked_huber",
            "splits": ["fit"],
            "mask_modes": [
                "partial_gene",
                "whole_node",
                "spatial_block",
            ],
            "mask_replicates_per_mode": 3,
            "fixed_mask_bundle": True,
            "generalization_estimate": False,
            "validation_or_test_selection": False,
        },
    }


def _mask_material(
    *,
    mask_drift: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode_mapping = (
        ("partial_gene", "partial"),
        ("whole_node", "node"),
        ("spatial_block", "block"),
    )
    rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    drift_label = "-drift" if mask_drift else ""
    for mode_index, (public_mode, internal_mode) in enumerate(mode_mapping):
        for replicate in range(3):
            entry_id = f"fit__{mode_index:02d}-{internal_mode}__r{replicate:03d}"
            checksum = _checksum(
                f"fixed-mask-{public_mode}-{replicate}{drift_label}"
            )
            seed = 31_000 + 100 * mode_index + replicate
            n_masked = 100_000 + 100 * mode_index + replicate
            rows.append(
                {
                    "split": "fit",
                    "mask_mode": public_mode,
                    "mask_replicate": replicate,
                    "mask_entry_id": entry_id,
                    "mask_seed": seed,
                    "mask_checksum": checksum,
                    "n_masked": n_masked,
                }
            )
            entries.append(
                {
                    "entry_id": entry_id,
                    "split": "fit",
                    "spec": {
                        "mode": internal_mode,
                        "label": public_mode,
                    },
                    "replicate": replicate,
                    "seed": seed,
                    "mask_checksum": checksum,
                    "summary": {"n_masked_entries": n_masked},
                }
            )
    manifest = {
        "format_version": 1,
        "bundle_checksum": _checksum(
            f"fixed-mask-bundle{drift_label}"
        ),
        "entries": entries,
    }
    return rows, manifest


def _evaluation_rows(
    whole_huber: Sequence[float],
    whole_pve: Sequence[float],
    *,
    mask_drift: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities, manifest = _mask_material(mask_drift=mask_drift)
    rows: list[dict[str, Any]] = []
    for identity in identities:
        mode = str(identity["mask_mode"])
        replicate = int(identity["mask_replicate"])
        if mode == "whole_node":
            huber = float(whole_huber[replicate])
            pve = float(whole_pve[replicate])
        elif mode == "partial_gene":
            huber = 0.2 + 0.01 * replicate
            pve = 20.0 + replicate
        else:
            huber = 0.5 + 0.01 * replicate
            pve = 2.0 + replicate
        rows.append(
            {
                **identity,
                "masked_huber": huber,
                "masked_mse": 2.0 * huber,
                "masked_mae": 0.5 * huber,
                "masked_r2": pve / 100.0,
                "masked_percent_variance_explained": pve,
            }
        )
    return rows, manifest


def _history_rows(
    *,
    role: str,
    epochs: int,
    graph_edges: int,
    nonfinite_epoch: int | None,
) -> list[dict[str, Any]]:
    modes = ("partial", "node", "block")
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_loss = 1.0 - 0.001 * min(epoch, 299)
        if epoch == nonfinite_epoch:
            train_loss = float("nan")
        rows.append(
            {
                "run_id": "bound-by-bundle",
                "split": "fit",
                "training_protocol": (
                    "held_in_full_core_fixed_budget"
                ),
                "epoch": epoch,
                "mask_mode": modes[epoch % 3],
                "mask_seed": 1000 + epoch,
                "mask_checksum": _checksum(f"epoch-mask-{epoch}"),
                "edge_dropout_seed": 2000 + epoch,
                "edge_checksum": _checksum(f"edge-{role}"),
                "n_masked_entries": 10_000 + epoch,
                "n_target_nodes": 100 + epoch,
                "n_edges_used": (
                    graph_edges if role != "matched_self" else 0
                ),
                "train_loss": train_loss,
                "diagnostic_loss": None,
                "gradient_norm": 1.0,
                "duration_seconds": 0.1,
                "peak_cuda_memory_bytes": 1024,
            }
        )
    return rows


def _constructor(role: str, hidden_dim: int) -> dict[str, Any]:
    arguments = {
        "num_genes": 1000,
        "node_covariate_dim": 22,
        "edge_attribute_dim": 17,
        "hidden_dim": hidden_dim,
        "attention_heads": 16,
        "attention_head_dim": 64,
        "graph_layers": 8,
        "ffn_dim": 4096,
        "decoder_dim": 4096,
        "edge_hidden_dim": 256,
        "edge_embedding_dim": 128,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "edge_conditioning_mode": "bias_gate",
    }
    if role != "matched_self":
        arguments.update(
            {
                "receiver_chunk_size": 64,
                "max_edges_per_chunk": 200_000,
                "activation_checkpointing": True,
            }
        )
    return {
        "canonical_model_key": (
            "qkvgat" if role != "matched_self" else "qkvgatmatchedself"
        ),
        "implementation_class": (
            "spatial_benchmark.qkv_graph_transformer."
            "ReceiverChunkedEdgeAwareQKVGraphTransformer"
            if role != "matched_self"
            else (
                "spatial_benchmark.qkv_self_control."
                "QKVParameterMatchedSelfControl"
            )
        ),
        "constructor_arguments": arguments,
    }


def _make_bundle(
    root: Path,
    *,
    role: str,
    whole_huber: Sequence[float],
    whole_pve: Sequence[float],
    parameter_count: int = _PARAMETERS,
    graph_sha256_override: str | None = None,
    graph_k_override: int | None = None,
    graph_edges_override: int | None = None,
    epochs: int = 300,
    hidden_dim: int = 1024,
    mask_drift: bool = False,
    nonfinite_epoch: int | None = None,
    include_val_prediction: bool = False,
    git_commit: str = _GIT_COMMIT,
    dirty_fingerprint: str | None = _DIRTY_FINGERPRINT,
    software_environment_fingerprint: str = (
        _SOFTWARE_ENVIRONMENT_FINGERPRINT
    ),
    reproduction_environment_fingerprint: str = (
        _REPRODUCTION_ENVIRONMENT_FINGERPRINT
    ),
) -> Path:
    expected_k, expected_graph_sha256, expected_edges = _GRAPH[role]
    graph_sha256 = (
        expected_graph_sha256
        if graph_sha256_override is None
        else graph_sha256_override
    )
    graph_k = (
        expected_k if graph_k_override is None else graph_k_override
    )
    graph_edges = (
        expected_edges
        if graph_edges_override is None
        else graph_edges_override
    )
    run_id = _run_id(role)
    config = _config(
        role,
        graph_sha256=graph_sha256,
        graph_k=graph_k,
        graph_edges=graph_edges,
        epochs=epochs,
        hidden_dim=hidden_dim,
    )
    archive = RunArchive.create(
        run_id,
        paths=_paths(root),
        manifest={"status": "success"},
        resolved_config=config,
    )
    evaluations, mask_manifest = _evaluation_rows(
        whole_huber,
        whole_pve,
        mask_drift=mask_drift,
    )
    mean_huber = statistics.fmean(whole_huber)
    mean_r2 = statistics.fmean(value / 100.0 for value in whole_pve)
    mean_pve = statistics.fmean(whole_pve)
    final_metrics = {
        "fit/whole_node/masked_huber": mean_huber,
        "fit/whole_node/masked_r2": mean_r2,
        "fit/whole_node/masked_percent_variance_explained": mean_pve,
        "resource/parameter_count": parameter_count,
    }
    archive.append_metric_event(
        {
            "name": "fit/whole_node/masked_huber",
            "value": mean_huber,
            "step": epochs - 1,
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    _write_jsonl(
        archive,
        "metrics/history.jsonl",
        _history_rows(
            role=role,
            epochs=epochs,
            graph_edges=graph_edges,
            nonfinite_epoch=nonfinite_epoch,
        ),
    )
    _write_jsonl(
        archive,
        "metrics/evaluation_replicates.jsonl",
        evaluations,
    )
    prediction = {
        "run_id": run_id,
        "sample_key": "sk_synthetic_cell",
        "dataset_id": "cosmx_normal_core_full_core_fit_v1",
        "split": "fit",
        "y_true": [0.0, 1.0],
        "y_pred": [0.1, 0.9],
    }
    archive.write_predictions("fit", [prediction])
    if include_val_prediction:
        archive.write_predictions(
            "val", [{**prediction, "split": "val"}]
        )
    archive.write_bytes("checkpoints/last.ckpt", b"synthetic-checkpoint")
    archive.prepare_log_files()
    archive.write_json(
        "provenance/git.json",
        {
            "commit": git_commit,
            "dirty": dirty_fingerprint is not None,
            "dirty_fingerprint": dirty_fingerprint,
        },
    )
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text(
        "provenance/environment.txt",
        "\n".join(
            (
                "python=3.12.13",
                "implementation=CPython",
                (
                    "environment_fingerprint="
                    f"{software_environment_fingerprint}"
                ),
                "",
                "[packages]",
                "torch==2.11.0",
                "",
                "[hardware_runtime]",
                '{"gpu_model":"NVIDIA GeForce RTX 3090"}',
                "[bagm_repro]",
                (
                    "environment_fingerprint="
                    f"{reproduction_environment_fingerprint}"
                ),
                "",
            )
        ),
    )
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_id": "cosmx_normal_core_full_core_fit_v1",
            "dataset_fingerprint": _DATASET_SHA256,
            "preprocessing_version": "full_core_fit_v1",
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {
            "split_id": _SPLIT_SHA256[:16],
            "split_fingerprint": _SPLIT_SHA256,
        },
    )
    archive.write_text(
        "provenance/command.txt",
        '{"argv":["synthetic"],"cwd":"project-root"}\n',
    )
    archive.write_json(
        "provenance/full_core_training.json",
        {
            "training_protocol": "held_in_full_core_fixed_budget",
            "graph_execution": "full_core_exact_no_neighbor_sampling",
            "checkpoint_policy": (
                "final_epoch_no_validation_selection"
            ),
            "fixed_epoch_budget": epochs,
            "final_epoch": epochs - 1,
            "model_seed": 0,
            "epoch_mask_seed": 314159,
            "parameter_count": parameter_count,
            "model_construction": _constructor(role, hidden_dim),
        },
    )
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "fit_scope": "all_nodes_transductive",
            "generalization_estimate": False,
            "preprocessing_checksums": {
                "preprocessing_sha256": _DATASET_SHA256
            },
            "materialized_identity_verification": {
                "dataset_fingerprint": _DATASET_SHA256,
                "split_fingerprint": _SPLIT_SHA256,
            },
            "graph_checksums": {"graph_sha256": graph_sha256},
            "graph_config": config["graph"],
        },
    )
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": mask_manifest,
            "seed_namespace": (
                "held-in-full-core-fixed-evaluation; "
                "disjoint from epoch masks"
            ),
            "used_for_gradient_updates": False,
            "used_for_checkpoint_selection": False,
        },
    )
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "final_epoch": epochs - 1,
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
            "model_name": config["model"]["name"],
            "model_seed": 0,
            "final_epoch": epochs - 1,
            "fixed_epoch_budget": epochs,
            "checkpoint_role": "last",
            "primary_metric_name": "fit/whole_node/masked_huber",
            "primary_metric_value": mean_huber,
            "metrics": final_metrics,
            "parameter_count": parameter_count,
            "graph_sha256": graph_sha256,
            "graph_directed_edges": graph_edges,
            "evaluation_mask_bundle_sha256": mask_manifest[
                "bundle_checksum"
            ],
            "evaluation_mask_replicates_per_mode": 3,
            (
                "evaluation_metrics_include_all_configured_replicates_per_mode"
            ): True,
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


def _three_runs(
    root: Path,
    *,
    k1000_options: Mapping[str, Any] | None = None,
    k5000_options: Mapping[str, Any] | None = None,
    self_options: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    options = {
        "k1000": {
            "whole_huber": (0.97, 0.96, 0.95),
            "whole_pve": (8.0, 9.0, 10.0),
            **dict(k1000_options or {}),
        },
        "k5000": {
            "whole_huber": (0.90, 0.89, 0.88),
            "whole_pve": (12.0, 13.0, 14.0),
            **dict(k5000_options or {}),
        },
        "matched_self": {
            "whole_huber": (1.00, 0.99, 0.98),
            "whole_pve": (5.0, 6.0, 7.0),
            **dict(self_options or {}),
        },
    }
    k1000 = _make_bundle(root, role="k1000", **options["k1000"])
    k5000 = _make_bundle(root, role="k5000", **options["k5000"])
    matched_self = _make_bundle(
        root, role="matched_self", **options["matched_self"]
    )
    return k1000, k5000, matched_self


def test_comparison_passes_gates_and_writes_compact_outputs(
    tmp_path: Path,
) -> None:
    runs = _three_runs(tmp_path)

    result = compare_full_core_qkv_large_k(*runs)

    assert result["representation_gate"]["passes"] is True
    assert result["evaluated_graph_run_ids"] == [
        result["runs"]["k1000"]["run_id"],
        result["runs"]["k5000"]["run_id"],
    ]
    assert result["representation_gate"]["eligible_graph_run_ids"] == [
        result["runs"]["k1000"]["run_id"],
        result["runs"]["k5000"]["run_id"],
    ]
    assert result["k_gate"]["passes"] is True
    compatibility = result["compatibility"]
    assert compatibility[
        "identical_source_and_environment_provenance"
    ] is True
    assert compatibility["git_commit"] == _GIT_COMMIT
    assert compatibility["git_dirty"] is True
    assert (
        compatibility["dirty_tree_fingerprint"] == _DIRTY_FINGERPRINT
    )
    assert (
        compatibility["software_environment_fingerprint"]
        == _SOFTWARE_ENVIRONMENT_FINGERPRINT
    )
    assert (
        compatibility["reproduction_environment_fingerprint"]
        == _REPRODUCTION_ENVIRONMENT_FINGERPRINT
    )
    k_comparison = result["aggregate_comparisons"]["k5000_vs_k1000"]
    assert k_comparison["relative_mean_huber_gain_percent"] == pytest.approx(
        100.0 * (0.96 - 0.89) / 0.96
    )
    assert k_comparison[
        "mean_pve_difference_candidate_minus_reference_points"
    ] == pytest.approx(4.0)
    assert k_comparison[
        "all_three_huber_replicates_favor_candidate"
    ] is True
    assert len(result["paired_whole_node_replicates"]) == 9

    output = write_comparison(result, tmp_path / "comparison")
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.json",
        "paired_whole_node.csv",
        "report.md",
    ]
    persisted = json.loads(
        (output / "comparison.json").read_text(encoding="utf-8")
    )
    assert persisted["representation_gate"]["passes"] is True
    assert (
        sum(
            1
            for line in (output / "paired_whole_node.csv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
        == 10
    )
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Evaluated graph run IDs" in report
    assert "Source/environment provenance: identical" in report
    assert "not classification accuracy" in report
    assert "not an estimate of generalization" in report
    assert "technical repeats" in report
    with pytest.raises(
        QKVLargeKComparisonError, match="already exists"
    ):
        write_comparison(result, output)


def test_representation_gate_exposes_no_eligible_run_when_locked_rules_fail(
    tmp_path: Path,
) -> None:
    k1000, k5000, matched_self = _three_runs(
        tmp_path,
        k1000_options={
            "whole_huber": (0.99, 0.98, 0.99),
            "whole_pve": (-1.0, -1.0, -1.0),
        },
        k5000_options={
            "whole_huber": (0.98, 1.00, 0.97),
            "whole_pve": (1.0, 1.0, 1.0),
        },
    )

    result = compare_full_core_qkv_large_k(
        k1000, k5000, matched_self
    )

    assert result["representation_gate"]["passes"] is False
    assert result["representation_gate"]["eligible_graph_run_ids"] == []
    assert result["representation_gate"]["candidate_gates"]["k1000"][
        "criteria"
    ]["graph_mean_masked_r2_is_positive"] is False
    assert result["representation_gate"]["candidate_gates"]["k5000"][
        "criteria"
    ]["all_three_masks_favor_graph_on_huber"] is False


@pytest.mark.parametrize(
    ("role", "options", "message"),
    (
        (
            "k1000",
            {"graph_sha256_override": _checksum("wrong-graph")},
            "locked k1000 graph contract",
        ),
        (
            "k5000",
            {"hidden_dim": 2048},
            "exact same QKV model config",
        ),
        (
            "matched_self",
            {"parameter_count": _PARAMETERS + 1},
            "parameter counts differ",
        ),
        (
            "matched_self",
            {"mask_drift": True},
            "fixed evaluation mask bundle identity",
        ),
        (
            "k1000",
            {"epochs": 299},
            "fixed 300-epoch budget",
        ),
        (
            "k5000",
            {"nonfinite_epoch": 17},
            "must be finite",
        ),
        (
            "matched_self",
            {"include_val_prediction": True},
            "held-out prediction artifacts",
        ),
        (
            "k1000",
            {"git_commit": "f" * 40},
            "git commit provenance",
        ),
        (
            "matched_self",
            {"git_commit": "unknown"},
            "lowercase Git object ID",
        ),
        (
            "k5000",
            {"dirty_fingerprint": _checksum("drifted-dirty-tree")},
            "dirty-tree fingerprint provenance",
        ),
        (
            "matched_self",
            {
                "software_environment_fingerprint": _checksum(
                    "drifted-software-environment"
                )
            },
            "software environment fingerprint provenance",
        ),
        (
            "k1000",
            {
                "reproduction_environment_fingerprint": _checksum(
                    "drifted-reproduction-environment"
                )
            },
            "reproduction environment fingerprint provenance",
        ),
        (
            "k5000",
            {"software_environment_fingerprint": "unverified"},
            "lowercase SHA-256",
        ),
    ),
)
def test_comparison_rejects_locked_contract_drift(
    tmp_path: Path,
    role: str,
    options: Mapping[str, Any],
    message: str,
) -> None:
    role_options = {
        "k1000_options": options if role == "k1000" else None,
        "k5000_options": options if role == "k5000" else None,
        "self_options": options if role == "matched_self" else None,
    }
    runs = _three_runs(tmp_path, **role_options)

    with pytest.raises(QKVLargeKComparisonError, match=message):
        compare_full_core_qkv_large_k(*runs)

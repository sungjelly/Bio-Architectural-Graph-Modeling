"""Focused synthetic tests for the worker-owned full-core capacity runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.full_core import (
    FullCoreData,
    FullCorePreprocessingChecksums,
    FullCorePreprocessingQC,
    build_exact_mutual_knn_graph,
)
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/run_full_core_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_full_core_capacity_runner_module",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_core() -> FullCoreData:
    generator = np.random.default_rng(20260725)
    n_nodes = 12
    n_genes = 5
    n_metadata = 3
    counts = generator.poisson(3.0, size=(n_nodes, n_genes)).astype(
        np.int32
    )
    transformed = np.log1p(counts.astype(np.float64))
    expression_mean = transformed.mean(axis=0)
    expression_scale = transformed.std(axis=0)
    target = (
        (transformed - expression_mean) / expression_scale
    ).astype(np.float32)
    raw_metadata = generator.uniform(
        0.5, 4.0, size=(n_nodes, n_metadata)
    )
    logged_metadata = np.log1p(raw_metadata)
    metadata_mean = logged_metadata.mean(axis=0)
    metadata_scale = logged_metadata.std(axis=0)
    covariates = (
        (logged_metadata - metadata_mean) / metadata_scale
    ).astype(np.float32)
    coordinates = np.asarray(
        [(x, y) for y in range(3) for x in range(4)],
        dtype=np.float64,
    )
    preprocessing_sha256 = _sha("synthetic-preprocessing")
    return FullCoreData(
        expression_counts=counts,
        target_expression=target,
        node_covariates=covariates,
        coordinates_um=coordinates,
        macroblock_ids=np.repeat(np.arange(3), 4),
        gene_names=tuple(f"gene_{index}" for index in range(n_genes)),
        metadata_names=tuple(
            f"metadata_{index}" for index in range(n_metadata)
        ),
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        metadata_median=np.median(raw_metadata, axis=0),
        metadata_mean=metadata_mean,
        metadata_scale=metadata_scale,
        metadata_missing_indicator_indices=np.empty(0, dtype=np.int64),
        preprocessing_qc=FullCorePreprocessingQC(
            n_nodes=n_nodes,
            n_genes=n_genes,
            n_measured_metadata_features=n_metadata,
            n_model_covariates=n_metadata,
            n_metadata_missing_values=0,
            n_constant_expression_features=0,
            n_constant_metadata_features=0,
            expression_max_abs_fitted_mean=float(
                np.abs(target.mean(axis=0)).max()
            ),
            metadata_max_abs_fitted_mean=float(
                np.abs(covariates.mean(axis=0)).max()
            ),
            source_metadata_roundtrip_max_abs_error=0.0,
            fit_scope="all_nodes_transductive",
            metadata_reconstruction="synthetic_exact",
            metadata_source_precision="float64",
            protected_identifier_arrays_returned=False,
            all_outputs_finite=True,
        ),
        checksums=FullCorePreprocessingChecksums(
            source_artifact_id="synthetic-artifact",
            source_prepared_data_sha256=_sha("synthetic-source"),
            expression_counts_sha256=_sha("synthetic-counts"),
            target_expression_sha256=_sha("synthetic-targets"),
            node_covariates_sha256=_sha("synthetic-covariates"),
            coordinates_um_sha256=_sha("synthetic-coordinates"),
            macroblock_ids_sha256=_sha("synthetic-blocks"),
            preprocessing_sha256=preprocessing_sha256,
        ),
    )


def _config(
    model_name: str,
    core: FullCoreData,
    graph: Any,
) -> dict[str, Any]:
    split_basis = {
        "schema": "full_core_no_holdout_roles_v1",
        "dataset_fingerprint": core.checksums.preprocessing_sha256,
        "n_nodes": core.n_nodes,
        "role_assignment": (
            "every row in verified prepared artifact order assigned fit"
        ),
        "role_counts": {
            "fit": core.n_nodes,
            "validation": 0,
            "test": 0,
        },
        "experimental_unit": "single_spatial_core",
    }
    uses_edges = model_name in {"g2", "g2-tokenized", "qkv-gat"}
    family = {
        "g2": "edge_conditioned_gatv2",
        "g2-tokenized": "tokenized_edge_conditioned_gatv2",
        "qkv-gat": "edge_aware_qkv_graph_transformer",
        "qkv-gat-matched-self": "qkv_parameter_matched_self_control",
    }.get(model_name, "edge_parameter_matched_self_control")
    model = {
        "name": model_name,
        "family": family,
        "embedding_dim": 8,
        "hidden_dim": 8,
        "attention_heads": 2,
        "graph_layers": 1,
        "ffn_dim": 12,
        "decoder_dim": 10,
        "edge_hidden_dim": 6,
        "edge_embedding_dim": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "receiver_chunk_size": 4,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": uses_edges,
    }
    if model_name in {"qkv-gat", "qkv-gat-matched-self"}:
        model.update(
            {
                "edge_conditioning_mode": "vector",
                "max_edges_per_chunk": 5,
                # Retained for graph-configuration parity.  The matched
                # control runner must ignore graph-execution-only settings.
                "exact_receiver_partitioning": True,
            }
        )
    token_task = model_name == "g2-tokenized"
    if token_task:
        model.update(
            {
                "num_expression_tokens": 4,
                "tokenizer_schema": (
                    "raw_count_tokens_0_1_2_3plus_v1"
                ),
            }
        )
    return {
        "version": 1,
        "seed": 7,
        "fold": 0,
        "attempt": 1,
        "campaign": {"campaign_id": "synthetic-full-core-capacity"},
        "model": model,
        "masking": {
            "type": "mixed_expression_masking",
            "curriculum": "P+N+B",
            "rates": {
                "partial_gene": 0.4,
                "whole_node": 0.25,
                "spatial_block": 0.25,
            },
            "warmup_epochs": 0,
            "block_shape": "disk",
            "block_width_um": None,
            "mask_seed": 314159,
        },
        "dataset": {
            "dataset_id": "synthetic_full_core",
            "version": "fit_v1",
            "split_id": canonical_sha256(split_basis)[:16],
            "dataset_fingerprint": core.checksums.preprocessing_sha256,
            "dataset_fingerprint_basis": {
                "source_artifact_id": (
                    core.checksums.source_artifact_id
                ),
                "source_prepared_data_sha256": (
                    core.checksums.source_prepared_data_sha256
                ),
                "materialized_preprocessing_sha256": (
                    core.checksums.preprocessing_sha256
                ),
            },
            "split_fingerprint": canonical_sha256(split_basis),
            "split_fingerprint_basis": split_basis,
            "prepared_artifact_reference": "protected/synthetic",
            "task": (
                "masked_expression_token_classification"
                if token_task
                else "masked_expression_regression"
            ),
            **(
                {
                    "target_scale": "raw_count_token_0_1_2_3plus",
                    "tokenization": {
                    "schema": "raw_count_tokens_0_1_2_3plus_v1",
                    "source_scale": "raw_biological_probe_counts",
                    "num_output_tokens": 4,
                    "mask_token_id": 4,
                    "mask_token_is_output": False,
                    "fixed_vocabulary": True,
                    "fit_required": False,
                    "count_mapping": {
                        "0": 0,
                        "1": 1,
                        "2": 2,
                        "3+": 3,
                    },
                    },
                }
                if token_task
                else {}
            ),
            "preprocessing_fit_scope": "all_nodes_transductive",
            "experimental_unit": "single_spatial_core",
            "validation_or_test_partition_present": False,
        },
        "features": {
            "use_edge_features": uses_edges,
            "edge_features": ["distance_um"] if uses_edges else [],
        },
        "graph": {
            "kind": "exact_spatial_knn_radius_guard",
            "neighbor_k": 2,
            "k": 2,
            "radius_um": 5.0,
            "radius_guard_um": 5.0,
            "symmetry": "mutual",
            "full_core_graph": True,
            # Deliberately omit graph.neighbor_sampling. The locked graph
            # config does too; the trainer is the sampling authority.
            "edge_dropout": 0.0,
            "expected_materialized_graph_sha256": (
                graph.checksums.graph_sha256
            ),
            "expected_directed_edges": graph.qc.n_directed_edges,
        },
        "trainer": {
            "learning_rate": 1e-3,
            "batch_size": 1,
            "neighbor_sampling": False,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "huber_delta": 1.0,
            "max_epochs": 1,
            "fixed_epoch_budget": True,
            "amp": False,
            "deterministic": True,
            "deterministic_warn_only": False,
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
            "device": "cpu",
            **(
                {
                    "objective": (
                        "unweighted_masked_categorical_cross_entropy"
                    )
                }
                if token_task
                else {}
            ),
        },
        "evaluation": {
            "task_family": (
                "masked_expression_token_classification"
                if token_task
                else "masked_expression_regression"
            ),
            "protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": (
                "fit/whole_node/masked_token_accuracy_percent"
                if token_task
                else "fit/whole_node/masked_huber"
            ),
            "primary_direction": "maximize" if token_task else "minimize",
            "splits": ["fit"],
            "mask_modes": [
                "partial_gene",
                "whole_node",
                "spatial_block",
            ],
            "mask_replicates_per_mode": 3,
        },
    }


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
    )


def _set_path_environment(
    monkeypatch: pytest.MonkeyPatch,
    paths: ProjectPaths,
) -> None:
    monkeypatch.setenv("BAGM_ROOT", str(paths.project_root))
    for variable, path in (
        ("BAGM_CONFIG_ROOT", paths.config_root),
        ("BAGM_DATA_ROOT", paths.data_root),
        ("BAGM_ARTIFACT_ROOT", paths.artifact_root),
        ("BAGM_STATE_ROOT", paths.state_root),
        ("BAGM_SCRATCH_ROOT", paths.scratch_root),
        ("BAGM_CACHE_ROOT", paths.cache_root),
        ("BAGM_EXPORT_ROOT", paths.export_root),
        ("BAGM_REPORT_ROOT", paths.report_root),
    ):
        monkeypatch.setenv(variable, str(path))


_WORKER_FILES = (
    "config.resolved.yaml",
    "manifest.yaml",
    "logs/stdout.log",
    "logs/stderr.log",
    "provenance/git.json",
    "provenance/uncommitted_changes.patch",
    "provenance/environment.txt",
    "provenance/hardware.json",
    "provenance/data_fingerprints.json",
    "provenance/split_fingerprint.json",
    "provenance/command.txt",
)


def _worker_archive(
    run_id: str,
    config: dict[str, Any],
    paths: ProjectPaths,
) -> tuple[RunArchive, dict[str, bytes]]:
    archive = RunArchive.create(
        run_id,
        paths=paths,
        resolved_config=config,
    )
    archive.write_manifest({"status": "running"})
    archive.prepare_log_files()
    for relative in _WORKER_FILES[4:]:
        archive.write_text(relative, f"worker-owned:{relative}\n")
    before = {
        relative: (archive.scratch_path / relative).read_bytes()
        for relative in _WORKER_FILES
    }
    return archive, before


def _attach(
    monkeypatch: pytest.MonkeyPatch,
    archive: RunArchive,
) -> RunArchive:
    monkeypatch.setenv("BAGM_RUN_ID", archive.run_id)
    monkeypatch.setenv("BAGM_RUN_SCRATCH", str(archive.scratch_path))
    monkeypatch.setenv(
        "BAGM_CONFIG_PATH",
        str(archive.scratch_path / "config.resolved.yaml"),
    )
    attached, _ = _RUNNER._worker_archive_and_config(
        argparse.Namespace(
            config=archive.scratch_path / "config.resolved.yaml",
            run_scratch=archive.scratch_path,
        )
    )
    assert attached.scratch_path == archive.scratch_path
    return attached


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_contract_allows_only_the_explicit_one_replicate_pilot() -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    ordinary = _config("g2", core, graph)
    assert _RUNNER._validate_capacity_contract(ordinary) == "g2"

    invalid_one = deepcopy(ordinary)
    invalid_one["evaluation"]["mask_replicates_per_mode"] = 1
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="requires exactly three",
    ):
        _RUNNER._validate_capacity_contract(invalid_one)

    pilot = deepcopy(invalid_one)
    pilot["trainer"]["diagnostic_resource_pilot"] = True
    pilot["evaluation"]["diagnostic_only"] = True
    pilot["evaluation"]["conclusion_bearing"] = False
    assert _RUNNER._validate_capacity_contract(pilot) == "g2"
    masks = _RUNNER._evaluation_masks(
        pilot,
        core,
        _RUNNER._training_config(pilot),
    )
    assert len(masks.manifest["entries"]) == 3

    pilot["evaluation"]["conclusion_bearing"] = True
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="conclusion_bearing=false",
    ):
        _RUNNER._validate_capacity_contract(pilot)


def test_qkv_contract_and_constructor_are_explicit_and_reconstructable() -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    config = _config("qkv-gat", core, graph)
    assert _RUNNER._validate_capacity_contract(config) == "qkvgat"

    construction = _RUNNER._model_construction_record(
        model_key="qkvgat",
        core=core,
        graph=graph,
        model_config=config["model"],
    )
    assert construction["implementation_class"].endswith(
        ".ReceiverChunkedEdgeAwareQKVGraphTransformer"
    )
    arguments = construction["constructor_arguments"]
    assert arguments["edge_conditioning_mode"] == "vector"
    assert arguments["receiver_chunk_size"] == 4
    assert arguments["max_edges_per_chunk"] == 5
    assert arguments["activation_checkpointing"] is True
    model = _RUNNER._make_model(
        model_key="qkvgat",
        core=core,
        graph=graph,
        model_config=config["model"],
    )
    assert isinstance(
        model,
        _RUNNER.ReceiverChunkedEdgeAwareQKVGraphTransformer,
    )
    assert model.max_edges_per_chunk == 5

    without_edges = deepcopy(config)
    without_edges["features"]["use_edge_features"] = False
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="requires measured edge features",
    ):
        _RUNNER._validate_capacity_contract(without_edges)

    approximate = deepcopy(config)
    approximate["model"]["exact_receiver_partitioning"] = False
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="exact receiver partitioning",
    ):
        _RUNNER._validate_capacity_contract(approximate)

    invalid_edge_bound = deepcopy(config)
    invalid_edge_bound["model"]["max_edges_per_chunk"] = 0
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="max_edges_per_chunk",
    ):
        _RUNNER._validate_capacity_contract(invalid_edge_bound)

    missing_edge_bound = deepcopy(config)
    missing_edge_bound["model"]["max_edges_per_chunk"] = None
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="max_edges_per_chunk",
    ):
        _RUNNER._validate_capacity_contract(missing_edge_bound)

    without_checkpointing = deepcopy(config)
    without_checkpointing["model"]["activation_checkpointing"] = False
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="activation_checkpointing=true",
    ):
        _RUNNER._validate_capacity_contract(without_checkpointing)


def test_tokenized_g2_contract_and_constructor_are_explicit() -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    config = _config("g2-tokenized", core, graph)
    assert _RUNNER._validate_capacity_contract(config) == "g2tokenized"

    construction = _RUNNER._model_construction_record(
        model_key="g2tokenized",
        core=core,
        graph=graph,
        model_config=config["model"],
    )
    assert construction["implementation_class"].endswith(
        ".TokenizedReceiverChunkedEdgeConditionedGATv2"
    )
    assert construction["constructor_arguments"]["num_expression_tokens"] == 4
    model = _RUNNER._make_model(
        model_key="g2tokenized",
        core=core,
        graph=graph,
        model_config=config["model"],
    )
    assert isinstance(
        model,
        _RUNNER.TokenizedReceiverChunkedEdgeConditionedGATv2,
    )

    wrong_task = deepcopy(config)
    wrong_task["dataset"]["task"] = "masked_expression_regression"
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="matching dataset.task",
    ):
        _RUNNER._validate_capacity_contract(wrong_task)

    wrong_vocabulary = deepcopy(config)
    wrong_vocabulary["model"]["num_expression_tokens"] = 5
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="four output tokens",
    ):
        _RUNNER._validate_capacity_contract(wrong_vocabulary)


def test_qkv_matched_self_has_exact_capacity_without_graph_execution() -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    graph_config = _config("qkv-gat", core, graph)
    self_config = _config("qkv-gat-matched-self", core, graph)
    assert _RUNNER._validate_capacity_contract(self_config) == (
        "qkvgatmatchedself"
    )

    graph_model = _RUNNER._make_model(
        model_key="qkvgat",
        core=core,
        graph=graph,
        model_config=graph_config["model"],
    )
    self_model = _RUNNER._make_model(
        model_key="qkvgatmatchedself",
        core=core,
        graph=graph,
        model_config=self_config["model"],
    )
    graph_parameters = {
        name: tuple(parameter.shape)
        for name, parameter in graph_model.named_parameters()
    }
    self_parameters = {
        name: tuple(parameter.shape)
        for name, parameter in self_model.named_parameters()
    }
    assert self_parameters == graph_parameters
    assert sum(
        parameter.numel() for parameter in self_model.parameters()
    ) == sum(parameter.numel() for parameter in graph_model.parameters())

    construction = _RUNNER._model_construction_record(
        model_key="qkvgatmatchedself",
        core=core,
        graph=graph,
        model_config=self_config["model"],
    )
    assert construction["implementation_class"].endswith(
        ".QKVParameterMatchedSelfControl"
    )
    arguments = construction["constructor_arguments"]
    assert arguments["edge_conditioning_mode"] == "vector"
    assert "receiver_chunk_size" not in arguments
    assert "max_edges_per_chunk" not in arguments
    assert "activation_checkpointing" not in arguments
    assert self_model.receiver_chunk_size == 256
    assert self_model.max_edges_per_chunk is None
    assert self_model.activation_checkpointing is True

    graph_features = deepcopy(self_config)
    graph_features["features"]["use_edge_features"] = True
    with pytest.raises(
        _RUNNER.FullCoreRunnerError,
        match="must not consume edge features",
    ):
        _RUNNER._validate_capacity_contract(graph_features)


def test_final_metrics_average_replicate_r2_without_clipping() -> None:
    replicate_rows = [
        {
            "mask_mode": mode,
            "masked_r2": value,
        }
        for mode in ("partial_gene", "whole_node", "spatial_block")
        for value in (-2.0, -1.0, 0.0)
    ]
    metrics = _RUNNER._final_metrics(
        replicate_rows=replicate_rows,
        training=SimpleNamespace(
            history=[SimpleNamespace(peak_cuda_memory_bytes=0)]
        ),
        training_duration=1.0,
        evaluation_duration=2.0,
        total_duration=3.0,
        parameter_count=10,
        checkpoint_size=20,
        replicates_per_mode=3,
        graph_construction_duration=4.0,
    )
    for mode in ("partial_gene", "whole_node", "spatial_block"):
        assert metrics[f"fit/{mode}/masked_r2"] == pytest.approx(-1.0)
        assert metrics[
            f"fit/{mode}/masked_percent_variance_explained"
        ] == pytest.approx(-100.0)


def test_synthetic_paired_runs_write_only_scientific_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    paths = _paths(tmp_path)
    _set_path_environment(monkeypatch, paths)
    salt = "synthetic-test-salt-at-least-16-bytes"
    runs: dict[str, tuple[RunArchive, Any, list[dict[str, Any]]]] = {}
    for index, model_name in enumerate(
        (
            "g2",
            "qkv-gat",
            "qkv-gat-matched-self",
            "b0-g2-matched",
        ),
        start=1,
    ):
        config = _config(model_name, core, graph)
        run_id = (
            f"r_20260725T12000{index}Z_deadbeef_s7_f0_a1_"
            f"{model_name}"
        )
        archive, worker_before = _worker_archive(
            run_id,
            config,
            paths,
        )
        attached = _attach(monkeypatch, archive)
        result = _RUNNER.run_full_core_capacity(
            config,
            attached,
            sample_key_salt=salt,
            full_core_data=core,
            receiver_graph=graph,
        )

        for relative, content in worker_before.items():
            assert (archive.scratch_path / relative).read_bytes() == content
        assert not (archive.scratch_path / "_SUCCESS").exists()
        assert not (archive.scratch_path / "predictions/validation.jsonl").exists()
        assert not (archive.scratch_path / "predictions/test.jsonl").exists()
        for relative in (
            "checkpoints/last.ckpt",
            "predictions/fit.jsonl",
            "metrics/events.jsonl",
            "metrics/final.json",
            "summary.json",
            "diagnostics/full_core_preprocessing.json",
            "diagnostics/graph_statistics.json",
            "diagnostics/mask_statistics.json",
            "diagnostics/training_convergence.json",
            "provenance/full_core_inputs.json",
            "provenance/fixed_evaluation_masks.json",
            "provenance/full_core_training.json",
        ):
            assert (archive.scratch_path / relative).is_file()
        assert list((archive.scratch_path / "metrics").glob("history.*"))
        assert list(
            (archive.scratch_path / "metrics").glob(
                "evaluation_replicates.*"
            )
        )

        rows = _read_jsonl(archive.scratch_path / "predictions/fit.jsonl")
        assert rows
        assert {row["split"] for row in rows} == {"fit"}
        assert {row["masking_type"] for row in rows} == {"whole_node"}
        assert {row["mask_replicate"] for row in rows} == {0}
        assert len({row["sample_key"] for row in rows}) == len(rows)
        assert all(row["sample_key"].startswith("sk_") for row in rows)
        assert all("_protected_local_index" not in row for row in rows)
        assert all(
            not any(
                token in key.lower()
                for token in ("patient", "donor", "cell_id", "barcode")
            )
            for row in rows
            for key in row
        )

        fixed_masks = json.loads(
            (
                archive.scratch_path
                / "provenance/fixed_evaluation_masks.json"
            ).read_text(encoding="utf-8")
        )
        assert len(fixed_masks["bundle_manifest"]["entries"]) == 9
        final_metrics = json.loads(
            (archive.scratch_path / "metrics/final.json").read_text(
                encoding="utf-8"
            )
        )
        assert np.isfinite(
            final_metrics["fit/whole_node/masked_huber"]
        )
        assert np.isfinite(final_metrics["fit/whole_node/masked_r2"])
        assert final_metrics[
            "fit/whole_node/masked_percent_variance_explained"
        ] == pytest.approx(
            100.0 * final_metrics["fit/whole_node/masked_r2"]
        )
        assert "resource/training_duration_seconds" in final_metrics
        assert "resource/inference_duration_seconds" in final_metrics
        assert "graph/construction_duration_seconds" in final_metrics
        assert not any(
            name.startswith("fit/resource/") for name in final_metrics
        )

        checkpoint = torch.load(
            archive.scratch_path / "checkpoints/last.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["checkpoint_role"] == "last"
        assert checkpoint["epoch"] == 0
        assert checkpoint["graph_sha256"] == graph.checksums.graph_sha256
        construction = checkpoint["model_construction"]
        assert construction["canonical_model_key"] == (
            "qkvgat"
            if model_name == "qkv-gat"
            else _RUNNER._model_key(model_name)
        )
        reconstructed_class = {
            "g2": _RUNNER.ReceiverChunkedEdgeConditionedGATv2,
            "qkv-gat": (
                _RUNNER.ReceiverChunkedEdgeAwareQKVGraphTransformer
            ),
            "qkv-gat-matched-self": (
                _RUNNER.QKVParameterMatchedSelfControl
            ),
            "b0-g2-matched": _RUNNER.EdgeParameterMatchedSelfControl,
        }[model_name]
        reconstructed = reconstructed_class(
            **construction["constructor_arguments"]
        )
        reconstructed.load_state_dict(checkpoint["model_state_dict"])
        training_provenance = json.loads(
            (
                archive.scratch_path
                / "provenance/full_core_training.json"
            ).read_text(encoding="utf-8")
        )
        assert training_provenance["model_construction"] == construction
        expected_edge_count = (
            graph.qc.n_directed_edges
            if model_name in {"g2", "qkv-gat"}
            else 0
        )
        assert {row["edge_count"] for row in rows} == {
            expected_edge_count
        }
        assert (
            result.summary["canonical_prediction_selection"]["mask_mode"]
            == "whole_node"
        )
        assert (
            result.summary["canonical_prediction_selection"][
                "mask_replicate"
            ]
            == 0
        )
        runs[model_name] = (archive, result, rows)

    g2_archive, g2_result, g2_rows = runs["g2"]
    self_archive, self_result, self_rows = runs["b0-g2-matched"]
    assert g2_result.summary["parameter_count"] == (
        self_result.summary["parameter_count"]
    )
    assert g2_result.summary["evaluation_mask_bundle_sha256"] == (
        self_result.summary["evaluation_mask_bundle_sha256"]
    )
    g2_identity = {
        (
            row["sample_key"],
            row["masking_type"],
            row["mask_replicate"],
            tuple(row["target_indices"]),
        )
        for row in g2_rows
    }
    self_identity = {
        (
            row["sample_key"],
            row["masking_type"],
            row["mask_replicate"],
            tuple(row["target_indices"]),
        )
        for row in self_rows
    }
    assert g2_identity == self_identity
    assert g2_archive.scratch_path != self_archive.scratch_path

    qkv_archive, qkv_result, qkv_rows = runs["qkv-gat"]
    qkv_self_archive, qkv_self_result, qkv_self_rows = runs[
        "qkv-gat-matched-self"
    ]
    assert qkv_result.summary["parameter_count"] == (
        qkv_self_result.summary["parameter_count"]
    )
    assert qkv_result.summary["evaluation_mask_bundle_sha256"] == (
        qkv_self_result.summary["evaluation_mask_bundle_sha256"]
    )
    assert {row["edge_count"] for row in qkv_rows} == {
        graph.qc.n_directed_edges
    }
    assert {row["edge_count"] for row in qkv_self_rows} == {0}
    assert qkv_archive.scratch_path != qkv_self_archive.scratch_path


def test_synthetic_token_run_writes_percent_metrics_and_integer_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _synthetic_core()
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=2,
        radius_guard_um=5.0,
        query_chunk_size=4,
        receiver_chunk_size=4,
        mutual_search_chunk_size=16,
        workers=1,
    )
    config = _config("g2-tokenized", core, graph)
    paths = _paths(tmp_path)
    _set_path_environment(monkeypatch, paths)
    archive, _ = _worker_archive(
        "r_20260726T130000Z_categorical_s7_f0_a1_token",
        config,
        paths,
    )
    attached = _attach(monkeypatch, archive)
    result = _RUNNER.run_full_core_capacity(
        config,
        attached,
        sample_key_salt="synthetic-test-salt-at-least-16-bytes",
        full_core_data=core,
        receiver_graph=graph,
    )

    token_audit = json.loads(
        (
            archive.scratch_path
            / "diagnostics/expression_tokenization.json"
        ).read_text(encoding="utf-8")
    )
    assert token_audit["spec"]["name"] == (
        "raw_count_tokens_0_1_2_3plus_v1"
    )
    assert token_audit["input_mask_token_is_output_class"] is False
    assert token_audit["per_gene_modal_baseline_scope"] == (
        "all_fit_transductive"
    )
    assert token_audit["per_gene_modal_baseline_entrywise_holdout"] is False

    fixed_masks = json.loads(
        (
            archive.scratch_path
            / "provenance/fixed_evaluation_masks.json"
        ).read_text(encoding="utf-8")
    )
    assert fixed_masks["seed_namespace"] == (
        "held-in-full-core-fixed-evaluation"
    )
    assert fixed_masks["seed_derivation_relationship"] == (
        "separate_from_epoch_mask_seed_derivation"
    )
    assert fixed_masks["entrywise_holdout"] is False
    assert fixed_masks["entries_may_overlap_training_masks"] is True
    assert fixed_masks["used_for_gradient_updates"] is False
    assert fixed_masks["used_for_gradient_updates_scope"] == (
        "exact_fixed_evaluation_mask_realizations_only"
    )
    assert (
        fixed_masks[
            "evaluation_mask_realizations_used_directly_for_gradient_updates"
        ]
        is False
    )

    final_metrics = json.loads(
        (archive.scratch_path / "metrics/final.json").read_text(
            encoding="utf-8"
        )
    )
    for name in (
        "fit/whole_node/masked_token_accuracy_percent",
        "fit/whole_node/masked_token_balanced_accuracy_percent",
        "fit/whole_node/masked_nonzero_token_accuracy_percent",
        "fit/whole_node/baseline_per_gene_modal_accuracy_percent",
    ):
        assert 0.0 <= final_metrics[name] <= 100.0
    assert np.isfinite(
        final_metrics["fit/whole_node/masked_token_cross_entropy"]
    )
    assert result.primary_metric_name == (
        "fit/whole_node/masked_token_accuracy_percent"
    )
    assert result.summary["generalization_estimate"] is False
    assert result.summary["task_family"] == (
        "masked_expression_token_classification"
    )

    rows = _read_jsonl(archive.scratch_path / "predictions/fit.jsonl")
    assert rows
    assert all(
        isinstance(value, int)
        for row in rows
        for value in (*row["y_true"], *row["y_pred"])
    )
    assert all(
        0 <= value <= 3
        for row in rows
        for value in (*row["y_true"], *row["y_pred"])
    )

    checkpoint = torch.load(
        archive.scratch_path / "checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["task_family"] == (
        "masked_expression_token_classification"
    )
    assert checkpoint["tokenization"]["spec"]["mask_input_token"]["id"] == 4
    assert checkpoint["tokenization"]["per_gene_modal_baseline_scope"] == (
        "all_fit_transductive"
    )
    assert checkpoint["tokenization"][
        "per_gene_modal_baseline_entrywise_holdout"
    ] is False

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train import materialize_myjju_genemae_campaign as materialize
from scripts.train import run_myjju_genemae_pooled as runner
from spatial_benchmark.masking import create_fixed_mask_bundle
from spatial_benchmark.myjju_genemae import (
    GeneMAE,
    build_symmetric_knn_graph,
    count_parameters,
    make_source_model,
)
from spatial_benchmark.pooled_full_core import ANC_ALIASES
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import command_for_config
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle


def _small_tile(
    *,
    alias: str = "ANC-01",
    node_count: int = 6,
    tile_index: int = 0,
) -> runner.SourceTile:
    coordinates = np.column_stack(
        [
            np.arange(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
        ]
    )
    edge_index, edge_attr = build_symmetric_knn_graph(coordinates, k=2)
    null_seed = runner._stable_seed(
        runner.GRAPH_NULL_NAMESPACE, alias, tile_index
    )
    permuted = np.asarray(
        runner.permute_graph_node_labels(
            edge_index, num_nodes=node_count, seed=null_seed
        ),
        dtype=np.int64,
    )
    indices = np.arange(node_count, dtype=np.int64)
    indices_sha = runner._array_sha256("node_indices", indices)
    edge_sha = runner._array_sha256("edge_index", edge_index)
    attr_sha = runner._array_sha256("edge_attr", edge_attr)
    permuted_sha = runner._array_sha256(
        "permuted_edge_index", permuted
    )
    graph_sha = runner.canonical_sha256(
        {
            "alias": alias,
            "tile_index": tile_index,
            "node_indices_sha256": indices_sha,
            "edge_index_sha256": edge_sha,
            "edge_attr_sha256": attr_sha,
            "constructed_graph_self_loops": False,
            "convolution_add_self_loops": True,
        }
    )
    return runner.SourceTile(
        alias=alias,
        tile_index=tile_index,
        node_indices=indices,
        edge_index=edge_index,
        edge_attr=edge_attr,
        permuted_edge_index=permuted,
        node_indices_sha256=indices_sha,
        edge_index_sha256=edge_sha,
        edge_attr_sha256=attr_sha,
        permuted_edge_index_sha256=permuted_sha,
        graph_sha256=graph_sha,
        graph_null_seed=null_seed,
    )


def test_materialized_production_config_preserves_registered_input_identity() -> None:
    receipt_path = (
        materialize.PROJECT_ROOT
        / materialize.COMPARATOR_RECEIPT_RELATIVE
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    registered = receipt["cohort"]
    sections: dict[str, dict] = {}
    for section, reference in materialize.COMPONENTS.items():
        sections[section], _ = materialize._component(reference, section)
    sections["trainer_production"], _ = materialize._component(
        materialize.TRAINERS["production"], "trainer"
    )
    sections["evaluation_production"], _ = materialize._component(
        materialize.EVALUATIONS["production"], "evaluation"
    )
    dataset, _ = materialize._component(
        materialize.DATASET_COMPONENT, "dataset"
    )
    identity = {
        "split_id": registered["split_id"],
        "split_fingerprint": registered["split_fingerprint"],
        "split_fingerprint_basis": registered[
            "split_fingerprint_basis"
        ],
        "dataset_fingerprint": registered["dataset_fingerprint"],
        "ordered_gene_schema_sha256": (
            registered["cohort_checksums"]["ordered_gene_schema_sha256"]
        ),
        "cores": registered["cores"],
    }

    config = materialize._build_config(
        role="production",
        seed=3,
        requested_gpu=3,
        output_reference=materialize.LOCKED_RELATIVE.as_posix(),
        components=sections,
        dataset=dataset,
        cohort_identity=identity,
        graph_identity={"schema": "test", "graph_bundle_sha256": "a" * 64},
        mask_identity={alias: {"test": True} for alias in ANC_ALIASES},
        source_identity={"commit": runner.SOURCE_COMMIT},
    )

    assert config["dataset"]["split_id"] == "1331b7ed9f3b2d73"
    assert config["dataset"]["split_fingerprint"] == (
        "1331b7ed9f3b2d7370b9ae3ada94b3af11364fddccbf58cb8979b7adf71e936a"
    )
    assert config["dataset"]["preprocessing_version"] == (
        "pooled_10core_equal_core_expression_v1"
    )
    assert config["dataset"]["count_representation"]["transform"] == (
        "full_cell_log1p_cp10k_before_masking"
    )
    assert any(
        part.endswith("run_myjju_genemae_pooled.py")
        for part in command_for_config(config)
    )


def test_materialized_common_mask_identity_round_trips_into_runner_schema() -> None:
    coordinates = np.column_stack(
        [
            np.arange(24, dtype=np.float64),
            np.arange(24, dtype=np.float64) % 3,
        ]
    )
    core = SimpleNamespace(
        alias="ANC-01",
        coordinates_um=coordinates,
        n_genes=7,
    )
    source_bundle = create_fixed_mask_bundle(
        {"fit": coordinates},
        7,
        runner._common_specs(),
        replicates=3,
        base_seed=12345,
    )
    comparator_source = {
        "base_seed": 12345,
        "bundle_checksum": source_bundle.checksum,
        "entries": [
            {
                "entry_id": entry["entry_id"],
                "mode": entry["spec"]["label"],
                "replicate": entry["replicate"],
                "seed": entry["seed"],
                "mask_checksum": entry["mask_checksum"],
            }
            for entry in source_bundle.manifest["entries"]
        ],
    }

    first = runner.regenerate_evaluation_masks(
        core, comparator_source=comparator_source
    )
    identity = first.identity()
    second = runner.regenerate_evaluation_masks(
        core,
        comparator_source=identity["common"],
        native_base_seed=identity["native"]["base_seed"],
    )

    assert second.identity() == identity
    for left, right in zip(
        first.all_masks, second.all_masks, strict=True
    ):
        np.testing.assert_array_equal(left.mask, right.mask)


def test_aggregate_uses_scale_specific_primary_without_ambiguous_alias() -> None:
    rows = []
    for alias in ANC_ALIASES:
        for label in ("common_20", "native_50"):
            for condition in ("observed", "node_label_permuted"):
                for replicate in range(3):
                    rows.append(
                        {
                            "biological_unit_alias": alias,
                            "mask_label": label,
                            "graph_condition": condition,
                            "mask_replicate": replicate,
                            **{
                                field: 1.0
                                for field in runner.REQUIRED_EVALUATION_METRIC_FIELDS
                            },
                        }
                    )

    result = runner._aggregate_metrics(rows)

    assert result[runner.PRIMARY_METRIC] == pytest.approx(1.0)
    assert "fit/partial_gene/masked_huber" not in result
    assert (
        "fit/partial_gene/anc_01/log1p_cp10k_masked_huber"
        in result
    )
    normalized = runner.evaluation_metric_audit_rows(rows)
    round_tripped = json.loads(json.dumps(normalized))
    assert runner.canonical_sha256(
        runner.evaluation_metric_audit_rows(round_tripped)
    ) == runner.canonical_sha256(normalized)
    assert isinstance(normalized[0]["n_masked"], int)
    assert isinstance(normalized[0]["masked_huber"], float)


def test_source_tiling_records_internal_convolution_self_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores = []
    for index, alias in enumerate(ANC_ALIASES):
        coordinates = np.asarray(
            [[0.0, float(index)], [1.0, float(index)]],
            dtype=np.float64,
        )
        cores.append(
            SimpleNamespace(
                alias=alias,
                coordinates_um=coordinates,
                n_nodes=2,
            )
        )
    cohort = SimpleNamespace(aliases=ANC_ALIASES, cores=tuple(cores))
    monkeypatch.setattr(runner, "EXPECTED_TOTAL_NODES", 20)

    tiled = runner.prepare_source_tiles(cohort, k=1, max_nodes=2)
    identity = tiled.identity()

    assert len(tiled.tiles) == 10
    assert identity["constructed_graph_self_loops"] is False
    assert identity["convolution_add_self_loops"] is True
    assert identity["cross_core_edges"] is False
    assert identity["cross_tile_edges"] is False
    assert all(
        not np.any(tile.edge_index[0] == tile.edge_index[1])
        for tile in tiled.tiles
    )


def test_source_model_initialization_is_seed_controlled() -> None:
    def digest(seed: int) -> str:
        runner.seed_all(seed)
        model = make_source_model()
        value = runner.state_dict_sha256(model.state_dict())
        del model
        gc.collect()
        return value

    first = digest(41)
    repeated = digest(41)
    other = digest(42)

    assert first == repeated
    assert first != other


def test_determinism_provenance_does_not_overclaim_cuda_reproducibility() -> None:
    facts = runner._determinism_facts(
        model_seed=5, device=torch.device("cuda")
    )

    assert facts["python_random_seed"] == 5
    assert facts["tile_order"] == "deterministic_seed_specific"
    assert facts["training_masks"].startswith("deterministic")
    assert facts["exact_bitwise_reproducibility_promised"] is False
    assert facts["known_nondeterministic_operations"]
    assert "GATv2 CUDA scatter/reduction" in facts[
        "known_nondeterministic_operations"
    ][0]


def test_runtime_device_resolves_queue_visible_cuda_to_index_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert runner._resolve_runtime_device(None) == torch.device("cuda:0")
    assert runner._resolve_runtime_device("cuda") == torch.device("cuda:0")
    assert runner._resolve_runtime_device("cuda:1") == torch.device("cuda:1")
    assert runner._resolve_runtime_device("cpu") == torch.device("cpu")


def test_locked_mask_sources_use_the_runner_schema() -> None:
    config = runner.load_yaml_mapping(
        runner.PROJECT_ROOT
        / runner.LOCKED_RELATIVE
        / "pilot_configs/seed-00_myjju_genemae_resource_pilot.yaml"
    )
    mask_sources = config["evaluation"]["prior_mask_sources"]

    for alias in ANC_ALIASES:
        configured, common, native_base_seed = (
            runner._configured_mask_source(mask_sources, alias)
        )
        assert configured["common"] is common
        assert common["source"] == (
            "exact_regeneration_of_current_bagm_partial_gene_masks"
        )
        assert isinstance(native_base_seed, int)


def test_small_training_visits_every_tile_and_uses_zero_based_final_epoch() -> None:
    tile = _small_tile(node_count=6)
    tiled = runner.TiledCohort(
        aliases=("ANC-01",),
        tiles=(tile,),
        k=2,
        max_nodes=6,
    )
    counts = np.arange(18, dtype=np.float32).reshape(6, 3)
    core = SimpleNamespace(alias="ANC-01")
    cohort = SimpleNamespace(cores=(core,))
    runner.seed_all(7)
    model = GeneMAE(
        in_dim=3,
        hidden=8,
        embed=4,
        heads=2,
        layers=2,
        dropout=0.0,
        drop_edge_p=0.0,
        gnn_decoder=False,
        self_branch=True,
        self_hidden=7,
    )

    result = runner.train_source_model(
        model=model,
        cohort=cohort,
        tiled=tiled,
        normalized_expression={"ANC-01": counts},
        trainer={
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 2,
            "train_mask_rate": 0.5,
            "gradient_clip_norm": 5.0,
        },
        model_seed=7,
        device="cpu",
    )

    assert result.completed_epochs == 2
    assert runner.final_epoch_index(result.completed_epochs) == 1
    assert result.optimizer_steps == 2
    assert [row["epoch_number"] for row in result.epoch_history] == [1, 2]
    assert all(row["every_tile_once"] for row in result.epoch_history)
    assert result.all_gradients_finite
    assert result.all_parameters_finite
    assert runner.final_epoch_index(200) == 199


def test_raw_prediction_provider_returns_full_core_ordered_matrix() -> None:
    tile = _small_tile(node_count=6)
    counts = np.asarray(
        [
            [1, 2, 0],
            [0, 1, 4],
            [3, 2, 1],
            [5, 0, 2],
            [1, 1, 1],
            [2, 3, 4],
        ],
        dtype=np.int64,
    )
    core = SimpleNamespace(
        alias="ANC-01",
        expression_counts=counts,
        n_nodes=6,
        n_genes=3,
    )
    runner.seed_all(8)
    model = GeneMAE(
        in_dim=3,
        hidden=8,
        embed=4,
        heads=2,
        layers=2,
        dropout=0.0,
        drop_edge_p=0.0,
        gnn_decoder=False,
        self_branch=True,
        self_hidden=7,
    )
    mask = np.zeros((6, 3), dtype=bool)
    mask[:, 0] = True

    observed = runner.predict_core_mask(
        model,
        core,
        [tile],
        mask,
        device="cpu",
        graph_condition="observed",
    )
    permuted = runner.predict_core_mask(
        model,
        core,
        [tile],
        mask,
        device="cpu",
        graph_condition="node_label_permuted",
    )

    assert observed.shape == (6, 3)
    assert permuted.shape == (6, 3)
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    assert np.isfinite(permuted).all()


def test_checkpoint_digest_and_strict_replay_share_one_algorithm() -> None:
    runner.seed_all(13)
    model = make_source_model()
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    payload = {
        "model_name": "myjju-genemae",
        "parameter_count": count_parameters(model),
        "external_source_commit": runner.SOURCE_COMMIT,
        "implementation_file_sha256": {
            "model_module": {
                "reference": runner.IMPLEMENTATION_FILES["model_module"],
                "sha256": runner.PINNED_MODEL_MODULE_SHA256,
            }
        },
        "model_state_dict": state,
        "state_dict_sha256": runner.state_dict_sha256(state),
    }

    replay = runner.reconstruct_model_from_checkpoint(payload, device="cpu")

    assert runner.state_dict_sha256(replay.state_dict()) == payload[
        "state_dict_sha256"
    ]
    assert count_parameters(replay) == runner.EXPECTED_PARAMETER_COUNT
    tile = _small_tile(node_count=6)
    fixed_replay = runner.verify_checkpoint_replay(
        original_model=model,
        reloaded_model=replay,
        core=SimpleNamespace(alias="ANC-01"),
        tile=tile,
        normalized_expression=np.zeros((6, 1000), dtype=np.float32),
        device="cpu",
    )
    assert fixed_replay["state_dict_exact"] is True
    assert fixed_replay["maximum_absolute_difference"] == 0.0
    assert fixed_replay["deterministic_cpu_replay"] is True
    assert fixed_replay["replay_passed"] is True


def test_success_status_and_zero_based_epoch_publish_as_verified_bundle(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(
        project_root=tmp_path,
        config_root=tmp_path / "configs",
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        state_root=tmp_path / "state",
        scratch_root=tmp_path / "scratch",
        cache_root=tmp_path / "cache",
        export_root=tmp_path / "exports",
        report_root=tmp_path / "reports",
    )
    paths.ensure_runtime_directories()
    run_id = (
        "r_20260730T120000Z_deadbeef_s000_f00_a01_myjju-genemae"
    )
    config = {
        "trainer": {
            "primary_checkpoint_role": "last",
            "restore_best": False,
        },
        "evaluation": {
            "protocol": "held_in_pooled_10core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": runner.PRIMARY_METRIC,
        },
    }
    archive = RunArchive.create(
        run_id, paths=paths, resolved_config=config
    )
    archive.write_manifest({"status": "success"})
    archive.prepare_log_files()
    for relative in (
        "provenance/git.json",
        "provenance/uncommitted_changes.patch",
        "provenance/environment.txt",
        "provenance/hardware.json",
        "provenance/data_fingerprints.json",
        "provenance/split_fingerprint.json",
        "provenance/command.txt",
    ):
        archive.write_text(relative, "{}\n")
    archive.write_table(
        "metrics/history",
        [{"epoch_number": 1, "masked_huber": 0.5}],
        fallback="jsonl",
    )
    archive.append_metric_event(
        {"name": runner.PRIMARY_METRIC, "value": 0.5}
    )
    archive.write_json(
        "metrics/final.json", {runner.PRIMARY_METRIC: 0.5}
    )
    archive.write_bytes("checkpoints/last.ckpt", b"checkpoint")
    archive.write_predictions(
        "fit",
        [
            {
                "run_id": run_id,
                "sample_key": "opaque-core-key",
                "dataset_id": "synthetic",
                "split": "fit",
                "y_true": [0.0],
                "y_pred": [0.1],
            }
        ],
        fallback="jsonl",
    )
    archive.write_summary(
        {
            "status": "success",
            "primary_metric_name": runner.PRIMARY_METRIC,
            "primary_metric_value": 0.5,
            "final_epoch": 1,
            "completed_global_epochs": 2,
        }
    )

    published = archive.finalize_success()
    verification = verify_run_bundle(
        published, require_success_contract=True
    )

    assert verification["status"] == "success"
    summary = json.loads(
        (published / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "success"
    assert summary["final_epoch"] == 1
    assert summary["completed_global_epochs"] == 2

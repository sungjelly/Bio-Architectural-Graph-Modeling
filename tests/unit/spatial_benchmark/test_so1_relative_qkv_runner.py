from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.so1_pooled_full_core import SO1_ALIASES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
GEOMETRY_EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiment/"
    "so1_14core_geometry_modulated_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_so1_14core_relative_qkv",
    PROJECT_ROOT / "scripts/train/run_so1_14core_relative_qkv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _resolved_config() -> dict[str, object]:
    return compose_config(EXPERIMENT_CONFIG, config_root=PROJECT_ROOT / "configs")


def _geometry_config() -> dict[str, object]:
    return compose_config(
        GEOMETRY_EXPERIMENT_CONFIG,
        config_root=PROJECT_ROOT / "configs",
    )


def test_locked_fresh_runner_contract_and_parser() -> None:
    config = _resolved_config()
    _RUNNER._validate_contract(config)
    parsed = _RUNNER.build_parser().parse_args(
        ["--config", "resolved.yaml", "--run-scratch", "scratch/run"]
    )
    assert parsed.config == Path("resolved.yaml")
    assert parsed.run_scratch == Path("scratch/run")
    assert parsed.resume_checkpoint is None
    assert _RUNNER.MODEL_SEED == 0
    assert _RUNNER.WORLD_SIZE == 4
    assert _RUNNER.VISIBLE_DEVICES == "0,1,2,3"
    assert _RUNNER.PLATEAU_FIRST_AUDIT_EPOCH == 150
    assert _RUNNER.PLATEAU_AUDIT_INTERVAL == 25
    assert _RUNNER.PLATEAU_WINDOW_EPOCHS == 50


def test_training_config_keeps_four_rank_batch2_and_ten_views() -> None:
    training = _RUNNER._training_config(
        _resolved_config(),
        start_epoch=150,
        end_epoch=175,
        rank=3,
        local_rank=3,
    )
    assert training.model_seed == 0
    assert training.cohort_aliases == SO1_ALIASES
    assert training.segment_start_global_epoch == 150
    assert training.segment_end_global_epoch == 175
    assert training.cores_per_optimizer_update == 2
    assert training.optimizer_updates_per_global_epoch == 7
    assert training.mask_views_per_core == 10
    assert training.losses_per_optimizer_update == 20
    assert training.distributed_world_size == 4
    assert training.distributed_rank == 3
    assert training.device == "cuda:3"
    assert training.stage_complete_core_graph_on_device is True


def test_geometry_protocol_constructs_exact_untied_four_block_model() -> None:
    config = _geometry_config()

    _RUNNER._validate_contract(config)
    assert _RUNNER._is_geometry_modulated(config)
    assert _RUNNER._block_gradient_count(config) == 4
    model = _RUNNER._model_from_config(
        config,
        num_genes=1000,
        node_covariate_dim=22,
    )

    assert type(model).__name__ == (
        "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_134_088
    assert _RUNNER._geometry_modulated4_block_topology(model) == {
        "verified": True,
        "graph_block_count": 4,
        "unique_graph_block_objects": 4,
        "unique_graph_block_parameter_sets": 4,
        "state_dict_block_indices": [0, 1, 2, 3],
        "graph_block_weight_tying": "none",
    }


def test_geometry_checkpoint_rejects_cross_campaign_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save(
        {
            "checkpoint_schema": _RUNNER.CHECKPOINT_SCHEMA,
            "campaign_id": _RUNNER.CAMPAIGN_ID,
            "model_seed": 0,
        },
        checkpoint,
    )

    with pytest.raises(
        _RUNNER.SO114CoreRunnerError,
        match="resume.campaign_id",
    ):
        _RUNNER._load_resume_checkpoint(
            checkpoint,
            config=_geometry_config(),
            model_construction={},
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("trainer", "cores_per_optimizer_update", 1, "cores_per_optimizer_update"),
        ("trainer", "mask_views_per_core_step", 9, "mask_views_per_core_step"),
        ("trainer", "minimum_global_epochs", 149, "minimum_global_epochs"),
        (
            "trainer",
            "plateau_absolute_relative_half_window_change_max",
            0.001,
            "plateau_absolute_relative_half_window_change_max",
        ),
        (
            "trainer",
            "plateau_normalized_absolute_slope_per_epoch_max",
            0.0001,
            "plateau_normalized_absolute_slope_per_epoch_max",
        ),
        ("launcher", "process_count", 3, "process_count"),
    ],
)
def test_contract_fails_closed_on_execution_or_plateau_drift(
    section: str,
    field: str,
    value: object,
    error: str,
) -> None:
    config = deepcopy(_resolved_config())
    config[section][field] = value  # type: ignore[index]
    with pytest.raises(Exception, match=error):
        _RUNNER._validate_contract(config)


def test_strict_plateau_can_stop_no_earlier_than_epoch_175() -> None:
    trainer = _resolved_config()["trainer"]
    first = _RUNNER._strict_plateau_payload([1.0] * 150, trainer=trainer)
    assert first["completed_global_epochs"] == 150
    assert first["plateau_consecutive_passing_audits"] == 1
    assert first["should_stop"] is False
    assert first["training_stop_applied"] is False

    second = _RUNNER._strict_plateau_payload([1.0] * 175, trainer=trainer)
    assert second["completed_global_epochs"] == 175
    assert second["plateau_consecutive_passing_audits"] == 2
    assert second["should_stop"] is True
    assert second["training_stop_applied"] is True
    assert second["final_epoch"] == 175
    assert second["absolute_relative_half_window_change_max"] == pytest.approx(
        0.0005
    )
    assert second["normalized_absolute_slope_per_epoch_max"] == pytest.approx(
        0.000025
    )


def test_resume_targets_the_next_prespecified_audit_boundary() -> None:
    assert _RUNNER._next_fixed_plateau_audit_epoch(
        0, first_audit_epoch=150, audit_interval=25
    ) == 150
    assert _RUNNER._next_fixed_plateau_audit_epoch(
        150, first_audit_epoch=150, audit_interval=25
    ) == 175
    assert _RUNNER._next_fixed_plateau_audit_epoch(
        163, first_audit_epoch=150, audit_interval=25
    ) == 175
    assert _RUNNER._is_fixed_plateau_audit_boundary(
        175, first_audit_epoch=150, audit_interval=25
    )
    assert not _RUNNER._is_fixed_plateau_audit_boundary(
        163, first_audit_epoch=150, audit_interval=25
    )


def test_resume_compatibility_binds_mask_and_plateau_schedule() -> None:
    source = _resolved_config()
    changed_mask = deepcopy(source)
    changed_mask["masking"]["mask_base_seed"] += 1  # type: ignore[index,operator]
    assert _RUNNER._resume_compatible_config(source) != (
        _RUNNER._resume_compatible_config(changed_mask)
    )

    changed_plateau = deepcopy(source)
    changed_plateau["trainer"][  # type: ignore[index]
        "plateau_absolute_relative_half_window_change_max"
    ] = 0.001
    assert _RUNNER._resume_compatible_config(source) != (
        _RUNNER._resume_compatible_config(changed_plateau)
    )


def test_hardware_preflight_receipt_is_checksum_and_manifest_bound(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_environment(
        {
            "BAGM_ROOT": str(tmp_path),
            "BAGM_DATA_ROOT": str(tmp_path / "data"),
            "BAGM_STATE_ROOT": str(tmp_path / "state"),
        }
    )
    cohort_dir = paths.data_root / "cohort"
    graph_dir = paths.data_root / "graphs"
    cohort_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    (cohort_dir / "manifest.json").write_text("cohort\n", encoding="utf-8")
    (graph_dir / "manifest.json").write_text("graph\n", encoding="utf-8")
    config = _resolved_config()
    content = {
        "schema": "so1_14core_relative_qkv_ddp4_preflight_v1",
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "distributed_world_size": 4,
        "distributed_backend": "nccl",
        "visible_devices": "0,1,2,3",
        "elastic_max_restarts": 0,
        "optimizer_updates": 1,
        "complete_graph_mask_views": 20,
        "checkpoint_reload_verified": True,
        "finite_loss_and_gradients": True,
        "prior_relative_qkv_equivalence_verified": True,
        "cohort_manifest_sha256": _RUNNER.sha256_file(
            cohort_dir / "manifest.json"
        ),
        "graph_manifest_sha256": _RUNNER.sha256_file(graph_dir / "manifest.json"),
        "resolved_config_sha256": _RUNNER._canonical_sha256(
            _RUNNER._preflight_bound_config(config)
        ),
        "peak_vram_gib_all_ranks": 12.5,
    }
    receipt = dict(content)
    receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(content)
    receipt_path = (
        paths.state_root
        / "preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    validated = _RUNNER._validate_hardware_preflight(
        config,
        paths=paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    assert validated["peak_vram_gib_all_ranks"] == pytest.approx(12.5)

    (graph_dir / "manifest.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(_RUNNER.SO114CoreRunnerError, match="manifest has changed"):
        _RUNNER._validate_hardware_preflight(
            config,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )


def test_geometry_preflight_is_distinct_and_requires_all_numerical_gates(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_environment(
        {
            "BAGM_ROOT": str(tmp_path),
            "BAGM_DATA_ROOT": str(tmp_path / "data"),
            "BAGM_STATE_ROOT": str(tmp_path / "state"),
        }
    )
    cohort_dir = paths.data_root / "cohort"
    graph_dir = paths.data_root / "graphs"
    cohort_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    (cohort_dir / "manifest.json").write_text("cohort\n", encoding="utf-8")
    (graph_dir / "manifest.json").write_text("graph\n", encoding="utf-8")
    config = _geometry_config()
    gradient_summary = {
        "schema": _RUNNER.GRADIENT_DIRECTION_METRICS_SCHEMA,
        "global_epoch": 1,
        "trainable_parameter_count": 5_134_088,
        "optimizer_updates_observed": 1,
        "gradient_norm_mean_before_clip": 0.5,
        "consecutive_optimizer_step_cosine_valid_pairs": 0,
        "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
        "resume_boundary_unavailable": False,
    }
    content = {
        "schema": _RUNNER.GEOMETRY_MODULATED_PREFLIGHT_SCHEMA,
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "campaign_id": _RUNNER.GEOMETRY_MODULATED_CAMPAIGN_ID,
        "distributed_world_size": 4,
        "distributed_backend": "nccl",
        "visible_devices": "0,1,2,3",
        "elastic_max_restarts": 0,
        "optimizer_updates": 1,
        "complete_graph_mask_views": 20,
        "checkpoint_reload_verified": True,
        "finite_loss_and_gradients": True,
        "prior_relative_qkv_equivalence_verified": True,
        "parameter_count": 5_134_088,
        "gradient_direction_observer_verified": True,
        "all_unique_graph_blocks_receive_gradients": True,
        "block_gradient_direction_observer_verified": True,
        "peak_vram_gib_all_ranks_max": 22.0,
        "minimum_vram_headroom_gib_each_rank": 2.0,
        "vram_acceptance_passed": True,
        "geometry_modulated_numerical_gates": {
            "passed": True,
            "full_chunk_exactness": {"passed": True},
            "amp_fp32_equivalence": {"passed": True},
            "learned_geometry_heads": {"passed": True},
        },
        "cohort_manifest_sha256": _RUNNER.sha256_file(
            cohort_dir / "manifest.json"
        ),
        "graph_manifest_sha256": _RUNNER.sha256_file(
            graph_dir / "manifest.json"
        ),
        "resolved_config_sha256": _RUNNER._canonical_sha256(
            _RUNNER._preflight_bound_config(config)
        ),
        "model": {
            "class": (
                "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
            ),
            "num_genes": 1000,
            "node_covariate_dim": 22,
            **config["model"],  # type: ignore[dict-item]
        },
        "gradient_direction_preflight_summary": gradient_summary,
        "gradient_norm_before_clip": 0.5,
        "geometry_modulated_graph_block_topology": {
            "verified": True,
            "graph_block_count": 4,
            "unique_graph_block_objects": 4,
            "unique_graph_block_parameter_sets": 4,
            "state_dict_block_indices": [0, 1, 2, 3],
            "graph_block_weight_tying": "none",
        },
        "graph_block_gradient_diagnostics": [
            {
                "block_index": index,
                "block_name": f"blocks.{index}",
                "trainable_parameter_count": 831_880,
                "parameters_missing_gradient": 0,
                "gradient_norm_before_clip": 0.1 + index,
            }
            for index in range(4)
        ],
        "block_gradient_direction_preflight_summary": [
            {
                "schema": _RUNNER.BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
                "global_epoch": 1,
                "block_index": index,
                "block_name": f"blocks.{index}",
                "trainable_parameter_count": 831_880,
                "optimizer_updates_observed": 1,
                "gradient_norm_mean_before_clip": 0.1 + index,
                "consecutive_optimizer_step_cosine_valid_pairs": 0,
                "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
                "resume_boundary_unavailable": False,
            }
            for index in range(4)
        ],
        "peak_vram_gib_all_ranks": 21.5,
        "measured_vram_headroom_gib_each_rank": 2.5,
    }
    receipt = dict(content)
    receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(content)
    receipt_path = (
        paths.state_root
        / "preflight/"
        "so1_14core_geometry_modulated_relative_qkv_ddp4_plateau_min150.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _RUNNER._validate_hardware_preflight(
        config,
        paths=paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )["campaign_id"] == _RUNNER.GEOMETRY_MODULATED_CAMPAIGN_ID

    content["geometry_modulated_numerical_gates"][  # type: ignore[index]
        "learned_geometry_heads"
    ]["passed"] = False
    receipt = dict(content)
    receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(content)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(_RUNNER.SO114CoreRunnerError, match="learned_geometry_heads"):
        _RUNNER._validate_hardware_preflight(
            config,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )


def test_preparation_artifacts_are_bound_to_config_hashes(tmp_path: Path) -> None:
    cohort_dir = tmp_path / "cohort"
    graph_dir = tmp_path / "graphs"
    cohort_dir.mkdir()
    graph_dir.mkdir()

    cohort = {"artifact_kind": "cohort"}
    cohort["manifest_content_sha256"] = _RUNNER._canonical_sha256(cohort)
    cohort_path = cohort_dir / "manifest.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")

    completed = {"artifact_kind": "completed"}
    completed["manifest_content_sha256"] = _RUNNER._canonical_sha256(completed)
    completed_path = graph_dir / "cohort_manifest_with_graphs.json"
    completed_path.write_text(json.dumps(completed), encoding="utf-8")

    graph = {
        "artifact_kind": "graphs",
        "cohort_manifest_sha256": _RUNNER.sha256_file(cohort_path),
        "completed_cohort_manifest_sha256": _RUNNER.sha256_file(completed_path),
    }
    graph["manifest_content_sha256"] = _RUNNER._canonical_sha256(graph)
    graph_path = graph_dir / "manifest.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    dataset = {
        "immutable_manifest_status": "verified_materialized_and_hash_bound",
        "dataset_fingerprint": cohort["manifest_content_sha256"],
        "cohort_manifest_file_sha256": _RUNNER.sha256_file(cohort_path),
        "graph_manifest_file_sha256": _RUNNER.sha256_file(graph_path),
        "graph_manifest_content_sha256": graph["manifest_content_sha256"],
        "completed_cohort_manifest_sha256": _RUNNER.sha256_file(completed_path),
    }
    observed = _RUNNER._validate_bound_preparation(
        dataset,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    assert observed["graph_manifest_file_sha256"] == dataset[
        "graph_manifest_file_sha256"
    ]

    drifted = dict(dataset)
    drifted["graph_manifest_content_sha256"] = "0" * 64
    with pytest.raises(_RUNNER.SO114CoreRunnerError, match="graph_manifest_content"):
        _RUNNER._validate_bound_preparation(
            drifted,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )


def test_epoch_event_append_is_idempotent_after_checkpoint(tmp_path: Path) -> None:
    class Archive:
        scratch_path = tmp_path

        def append_metric_event(self, event: dict[str, object]) -> Path:
            target = self.scratch_path / "metrics/events.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            return target

    archive = Archive()
    row = {
        "global_epoch": 1,
        "equal_core_mean_masked_huber": 0.5,
        "run_id": "r_test",
    }
    assert _RUNNER._append_epoch_metric_event_once(archive, row) is True
    assert _RUNNER._append_epoch_metric_event_once(archive, row) is False
    lines = (tmp_path / "metrics/events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1

    drifted = {**row, "equal_core_mean_masked_huber": 0.6}
    with pytest.raises(_RUNNER.SO114CoreRunnerError, match="drifted"):
        _RUNNER._append_epoch_metric_event_once(archive, drifted)

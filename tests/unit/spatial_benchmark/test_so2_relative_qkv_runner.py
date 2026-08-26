from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import pytest
import torch

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.pooled_relative_qkv_training_v2 import (
    SO2_14CORE_ALIASES,
    CohortRelativeQKVCoreBatch,
    CohortRelativeQKVTrainingConfig,
    fit_cohort_relative_qkv_segment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "run_so2_14core_relative_qkv",
    PROJECT_ROOT / "scripts/train/run_so2_14core_relative_qkv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


class _ReplayModel(torch.nn.Module):
    def __init__(self, n_genes: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.linspace(0.8, 1.2, n_genes))

    def forward(
        self,
        *,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        edge_index: torch.Tensor,
        relative_geometry: torch.Tensor,
        node_covariates: torch.Tensor,
        target_nodes: list[int],
    ) -> SimpleNamespace:
        del gene_mask, edge_index, relative_geometry
        selected = torch.as_tensor(target_nodes, device=input_expression.device)
        prediction = input_expression.index_select(0, selected) * self.scale
        prediction = prediction + node_covariates.index_select(0, selected)[:, :1]
        return SimpleNamespace(prediction=prediction)


class _TrainerModel(torch.nn.Module):
    def __init__(self, n_genes: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.1, 0.1, n_genes))

    def forward(
        self,
        *,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        edge_index: torch.Tensor,
        relative_geometry: torch.Tensor,
        node_covariates: torch.Tensor,
        target_nodes: torch.Tensor,
    ) -> torch.Tensor:
        del gene_mask, edge_index, relative_geometry, node_covariates, target_nodes
        return input_expression + self.bias


def _tiny_cohort_batches() -> tuple[CohortRelativeQKVCoreBatch, ...]:
    return tuple(
        CohortRelativeQKVCoreBatch(
            alias=alias,
            target_expression=torch.tensor([[0.2, 0.7]], dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            relative_geometry=torch.empty((0, 70), dtype=torch.float32),
            node_covariates=torch.tensor([[0.1]], dtype=torch.float32),
        )
        for alias in SO2_14CORE_ALIASES
    )


def test_runner_imports_and_parser_requires_worker_paths() -> None:
    parser = _RUNNER.build_parser()
    parsed = parser.parse_args(
        ["--config", "resolved.yaml", "--run-scratch", "scratch/run"]
    )
    assert parsed.config == Path("resolved.yaml")
    assert parsed.run_scratch == Path("scratch/run")
    assert _RUNNER.WORLD_SIZE == 4
    assert _RUNNER.VISIBLE_DEVICES == "0,1,2,3"


def test_resume_at_epoch_163_targets_next_fixed_audit_epoch_175() -> None:
    assert _RUNNER._resume_plateau_decision(
        [1.0] * 163,
        completed_global_epochs=163,
        first_audit_epoch=150,
        audit_interval=25,
    ) is None
    assert _RUNNER._next_fixed_plateau_audit_epoch(
        163,
        first_audit_epoch=150,
        audit_interval=25,
    ) == 175


def test_resume_after_passing_audit_rechecks_before_training() -> None:
    decision = _RUNNER._resume_plateau_decision(
        [1.0] * 175,
        completed_global_epochs=175,
        first_audit_epoch=150,
        audit_interval=25,
    )
    assert decision is not None
    assert decision.completed_global_epochs == 175
    assert decision.consecutive_passing_audits == 2
    assert decision.should_stop is True
    assert decision.final_epoch == 175


def test_fixed_continuation_resume_compatibility_is_schedule_only() -> None:
    source = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    continuation = compose_config(
        PROJECT_ROOT
        / "configs/experiment/"
        "so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    assert _RUNNER._resume_compatible_config(source) == (
        _RUNNER._resume_compatible_config(continuation)
    )

    drifted = dict(continuation)
    drifted["trainer"] = dict(continuation["trainer"])
    drifted["trainer"]["learning_rate"] = 2e-4
    assert _RUNNER._resume_compatible_config(source) != (
        _RUNNER._resume_compatible_config(drifted)
    )

    drifted_mask = dict(continuation)
    drifted_mask["masking"] = dict(continuation["masking"])
    drifted_mask["masking"]["mask_base_seed"] += 1
    assert _RUNNER._resume_compatible_config(source) != (
        _RUNNER._resume_compatible_config(drifted_mask)
    )


def test_fixed_continuation_plan_disables_callbacks_and_stops_exactly_at_300() -> None:
    continuation = compose_config(
        PROJECT_ROOT
        / "configs/experiment/"
        "so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    resume = SimpleNamespace(
        completed_global_epochs=175,
        optimizer_updates_per_global_epoch=7,
    )
    plan = _RUNNER._fixed_continuation_plan(continuation, resume)
    assert plan == {
        "segment_start_global_epoch": 175,
        "segment_end_global_epoch": 300,
        "checkpoint_callback_enabled": False,
        "plateau_stopping_enabled": False,
        "epochs_this_segment": 125,
        "optimizer_updates_this_segment": 875,
    }


def test_fixed_completion_requires_epoch_300_and_2100_updates() -> None:
    completed = _RUNNER._fixed_completion_payload(
        SimpleNamespace(
            completed_global_epochs=300,
            optimizer_updates_completed=2100,
            optimizer_updates_per_global_epoch=7,
        ),
        source_checkpoint_sha256=(
            _RUNNER.FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
    )
    assert completed["fixed_budget_completed"] is True
    assert completed["plateau_stopping_enabled"] is False
    assert completed["early_stop_applied"] is False

    early = _RUNNER._fixed_completion_payload(
        SimpleNamespace(
            completed_global_epochs=299,
            optimizer_updates_completed=2093,
            optimizer_updates_per_global_epoch=7,
        ),
        source_checkpoint_sha256=(
            _RUNNER.FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
    )
    assert early["fixed_budget_completed"] is False


@pytest.mark.parametrize("diagnostic_confirmed", [False, True])
def test_fixed_final_validation_accepts_strict_diagnostic_pass_or_fail(
    diagnostic_confirmed: bool,
) -> None:
    resume = SimpleNamespace(
        completed_global_epochs=300,
        optimizer_updates_completed=2100,
        optimizer_updates_per_global_epoch=7,
    )
    completion = _RUNNER._fixed_completion_payload(
        resume,
        source_checkpoint_sha256=(
            _RUNNER.FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
    )
    strict = {
        "schema": "so2_14core_strict_plateau_diagnostic_v1",
        "completed_global_epochs": 300,
        "role": "diagnostic_only_never_stopping",
        "training_stop_applied": False,
        "plateau_should_stop": False,
        "diagnostic_confirmed": diagnostic_confirmed,
    }
    _RUNNER._validate_fixed_final_metadata(
        completion,
        strict,
        expected_resume=resume,
    )


def test_terminal_strict_audit_verifies_callback_write_without_overwrite(
    tmp_path: Path,
) -> None:
    class _Archive:
        def __init__(self, root: Path) -> None:
            self.scratch_path = root
            self.write_count = 0

        def write_json(self, relative: Path, payload: object) -> None:
            self.write_count += 1
            target = self.scratch_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    archive = _Archive(tmp_path)
    payload = {
        "schema": "so2_14core_strict_plateau_diagnostic_v1",
        "completed_global_epochs": 300,
        "diagnostic_confirmed": False,
    }
    first = _RUNNER._write_or_verify_strict_plateau_audit(archive, payload)
    second = _RUNNER._write_or_verify_strict_plateau_audit(archive, payload)
    assert first == second
    assert archive.write_count == 1

    drifted = dict(payload)
    drifted["diagnostic_confirmed"] = True
    with pytest.raises(_RUNNER.SO214CoreRunnerError, match="drifted"):
        _RUNNER._write_or_verify_strict_plateau_audit(archive, drifted)


def test_fixed_checkpoint_replay_compares_fresh_predictions_and_state() -> None:
    batch = SimpleNamespace(
        alias="SO2-C15",
        n_nodes=4,
        n_genes=3,
        target_expression=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        node_covariates=torch.full((4, 1), 0.25),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        relative_geometry=torch.empty((0, 70), dtype=torch.float32),
    )
    in_memory = _ReplayModel(3)
    reloaded = _ReplayModel(3)
    reloaded.load_state_dict(in_memory.state_dict(), strict=True)
    expected_checksum = _RUNNER._tree_sha256(in_memory.state_dict())
    receipt = _RUNNER._fixed_checkpoint_prediction_replay(
        in_memory_model=in_memory,
        reloaded_model=reloaded,
        batch=batch,
        expected_model_state_checksum=expected_checksum,
        device=torch.device("cpu"),
    )
    assert receipt["verified"] is True
    assert receipt["maximum_absolute_difference"] == 0.0
    assert receipt["in_memory_prediction_checksum"] == (
        receipt["reloaded_prediction_checksum"]
    )

    with torch.no_grad():
        reloaded.scale.add_(0.01)
    with pytest.raises(_RUNNER.SO214CoreRunnerError, match="state checksum"):
        _RUNNER._fixed_checkpoint_prediction_replay(
            in_memory_model=in_memory,
            reloaded_model=reloaded,
            batch=batch,
            expected_model_state_checksum=expected_checksum,
            device=torch.device("cpu"),
        )


def test_final_checkpoint_payload_revalidates_all_resume_checksums(
    tmp_path: Path,
) -> None:
    model = _TrainerModel(2)
    result = fit_cohort_relative_qkv_segment(
        model,
        _tiny_cohort_batches(),
        CohortRelativeQKVTrainingConfig(
            model_seed=0,
            segment_end_global_epoch=1,
            learning_rate=0.01,
            device="cpu",
        ),
    )
    config = {"attempt": 1, "launcher": {}}
    construction = {"class": "_TrainerModel", "num_genes": 2}
    plateau = {"should_stop": True, "final_epoch": 1}
    payload = _RUNNER._checkpoint_payload(
        run_id="r_test",
        config=config,
        model_construction=construction,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        resume=result.resume,
        plateau=plateau,
    )
    checkpoint = tmp_path / "last.ckpt"
    torch.save(payload, checkpoint)

    loaded, receipt = _RUNNER._validate_final_checkpoint_payload(
        checkpoint,
        config=config,
        model_construction=construction,
        expected_resume=result.resume,
        expected_plateau=plateau,
    )
    assert receipt["full_resume_payload_validated"] is True
    assert receipt["plateau_payload_validated"] is True
    assert loaded.resume_checksum == result.resume.resume_checksum

    corrupted = dict(payload)
    corrupted["optimizer_state_checksum"] = "0" * 64
    torch.save(corrupted, checkpoint)
    with pytest.raises(Exception, match="optimizer checksum mismatch"):
        _RUNNER._validate_final_checkpoint_payload(
            checkpoint,
            config=config,
            model_construction=construction,
            expected_resume=result.resume,
            expected_plateau=plateau,
        )


def test_runtime_path_respects_independent_runtime_root_overrides(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_environment(
        {
            "BAGM_ROOT": str(tmp_path / "source"),
            "BAGM_DATA_ROOT": str(tmp_path / "runtime-data"),
            "BAGM_STATE_ROOT": str(tmp_path / "runtime-state"),
            "BAGM_ARTIFACT_ROOT": str(tmp_path / "runtime-artifacts"),
        }
    )
    assert _RUNNER._runtime_path("data/processed/cohort", paths) == (
        tmp_path / "runtime-data/processed/cohort"
    ).resolve()
    assert _RUNNER._runtime_path("state/preflight/receipt.json", paths) == (
        tmp_path / "runtime-state/preflight/receipt.json"
    ).resolve()
    absolute = (tmp_path / "absolute/checkpoint.ckpt").resolve()
    assert _RUNNER._runtime_path(absolute, paths) == absolute


def test_distributed_identity_rejects_visible_device_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "0,1,2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_RUNNER.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(_RUNNER.SO214CoreRunnerError, match="exactly 0,1,2,3"):
        _RUNNER._distributed_identity()


def test_distributed_identity_accepts_exact_single_node_four_rank_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "RANK": "2",
        "LOCAL_RANK": "2",
        "WORLD_SIZE": "4",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(_RUNNER.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_RUNNER.torch.cuda, "device_count", lambda: 4)
    assert _RUNNER._distributed_identity() == (2, 2, 4)


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
    config = {
        "launcher": {
            "hardware_preflight_receipt": (
                "state/preflight/so2_14core_relative_qkv_ddp4.json"
            )
        }
    }
    content = {
        "schema": "so2_14core_relative_qkv_ddp4_preflight_v1",
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
        "so2_c23_prior_equivalence_verified": True,
        "cohort_manifest_sha256": _RUNNER.sha256_file(
            cohort_dir / "manifest.json"
        ),
        "graph_manifest_sha256": _RUNNER.sha256_file(
            graph_dir / "manifest.json"
        ),
        "resolved_config_sha256": _RUNNER._canonical_sha256(
            _RUNNER._preflight_bound_config(config)
        ),
        "peak_vram_gib_all_ranks": 12.5,
    }
    receipt = dict(content)
    receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(content)
    receipt_path = paths.state_root / "preflight/so2_14core_relative_qkv_ddp4.json"
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
    with pytest.raises(_RUNNER.SO214CoreRunnerError, match="manifest has changed"):
        _RUNNER._validate_hardware_preflight(
            config,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )

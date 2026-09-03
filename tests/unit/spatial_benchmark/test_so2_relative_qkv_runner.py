from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json
from copy import deepcopy
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


def test_recurrent_contract_is_additive_and_uses_literal_plateau_protocol() -> None:
    recurrent = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    _RUNNER._validate_contract(recurrent)

    assert _RUNNER._campaign_id(recurrent) == _RUNNER.RECURRENT_CAMPAIGN_ID
    assert _RUNNER._preflight_path(recurrent) == _RUNNER.RECURRENT_PREFLIGHT
    assert _RUNNER._preflight_schema(recurrent) == (
        _RUNNER.RECURRENT_PREFLIGHT_SCHEMA
    )
    model = recurrent["model"]
    assert model["graph_layers"] == 1
    assert model["unique_graph_blocks"] == 1
    assert model["recurrent_unroll_steps"] == 4
    assert model["effective_graph_depth"] == 4
    assert model["graph_block_weight_tying"] == "all_steps"

    baseline = compose_config(
        PROJECT_ROOT / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    assert recurrent["trainer"] == baseline["trainer"]
    _RUNNER._validate_contract(baseline)


def test_recurrent_model_builder_registers_one_block_unrolled_four_times() -> None:
    recurrent = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    model = _RUNNER._model_from_config(
        recurrent,
        num_genes=1000,
        node_covariate_dim=22,
    )
    construction = _RUNNER._model_construction(
        model,
        recurrent,
        num_genes=1000,
        node_covariate_dim=22,
    )

    assert type(model).__name__ == (
        "ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer"
    )
    assert len(model.blocks) == 1
    assert model.graph_layers == 1
    assert model.unique_graph_blocks == 1
    assert model.recurrent_unroll_steps == 4
    assert model.effective_graph_depth == 4
    assert model.graph_block_weight_tying == "all_steps"
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        _RUNNER.EXPECTED_RECURRENT_PARAMETER_COUNT
    )
    assert construction["class"] == type(model).__name__
    assert construction["graph_block_weight_tying"] == "all_steps"
    block_keys = [key for key in model.state_dict() if key.startswith("blocks.")]
    assert block_keys
    assert all(key.startswith("blocks.0.") for key in block_keys)


def test_recurrent_contract_rejects_architecture_drift() -> None:
    recurrent = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    drifted = deepcopy(recurrent)
    drifted["model"]["recurrent_unroll_steps"] = 3
    with pytest.raises(Exception, match="recurrent_unroll_steps"):
        _RUNNER._validate_contract(drifted)


def test_untied8_contract_and_model_are_exactly_eight_unique_blocks() -> None:
    untied = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    baseline = compose_config(
        PROJECT_ROOT / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    _RUNNER._validate_contract(untied)

    assert _RUNNER._campaign_id(untied) == _RUNNER.UNTIED8_CAMPAIGN_ID
    assert _RUNNER._preflight_path(untied) == _RUNNER.UNTIED8_PREFLIGHT
    assert _RUNNER._preflight_schema(untied) == _RUNNER.UNTIED8_PREFLIGHT_SCHEMA
    assert untied["trainer"] == baseline["trainer"]
    assert untied["model"]["graph_layers"] == 8
    assert untied["model"]["unique_graph_blocks"] == 8
    assert untied["model"]["effective_graph_depth"] == 8
    assert untied["model"]["graph_block_weight_tying"] == "none"
    assert "recurrent_unroll_steps" not in untied["model"]

    model = _RUNNER._model_from_config(
        untied,
        num_genes=1000,
        node_covariate_dim=22,
    )
    topology = _RUNNER._untied8_block_topology(model)
    assert type(model).__name__ == (
        "ReceiverChunkedRelativeGeometryQKVGraphTransformer"
    )
    assert len(model.blocks) == 8
    assert len({id(block) for block in model.blocks}) == 8
    assert topology["verified"] is True
    assert topology["state_dict_block_indices"] == list(range(8))
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        _RUNNER.EXPECTED_UNTIED8_PARAMETER_COUNT
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("graph_layers", 7), ("graph_block_weight_tying", "all_steps")],
)
def test_untied8_contract_rejects_architecture_drift(
    field: str,
    value: object,
) -> None:
    untied = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    drifted = deepcopy(untied)
    drifted["model"][field] = value
    with pytest.raises(Exception, match=field):
        _RUNNER._validate_contract(drifted)


def test_untied8_epoch300_unconfirmed_plateau_is_typed_non_success() -> None:
    untied = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    decision = SimpleNamespace(completed_global_epochs=300, should_stop=False)
    assert _RUNNER._untied8_operational_review_required(untied, decision) is True
    assert issubclass(
        _RUNNER.SO214CoreOperationalReviewRequired,
        _RUNNER.SO214CoreRunnerError,
    )

    payload = _RUNNER._untied8_operational_review_payload(
        run_id="r_review",
        parameter_count=_RUNNER.EXPECTED_UNTIED8_PARAMETER_COUNT,
        plateau={"completed_global_epochs": 300, "should_stop": False},
        checkpoint_sha256="a" * 64,
        epoch_metrics_rows=300,
        gradient_direction_rows=300,
        block_gradient_rows=2400,
    )
    assert payload["status"] == "operational_review_required"
    assert payload["plateau_confirmed"] is False
    assert payload["scientific_convergence_claim"] is False
    assert payload["queue_terminal_representation"] == "failed_inconclusive"
    assert payload["checkpoint"] == "checkpoints/latest.ckpt"
    assert payload["resume_requires_explicit_contract_amendment_and_approval"] is True

    assert not _RUNNER._untied8_operational_review_required(
        untied,
        SimpleNamespace(completed_global_epochs=300, should_stop=True),
    )
    assert not _RUNNER._untied8_operational_review_required(
        untied,
        SimpleNamespace(completed_global_epochs=275, should_stop=False),
    )


def test_untied8_finalization_requires_global_and_block_scalar_csvs(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    global_csv = results / "gradient_direction_metrics.csv"
    block_csv = results / "gradient_direction_by_block.csv"
    global_csv.write_text("global\n", encoding="utf-8")
    block_csv.write_text("block\n", encoding="utf-8")

    observed = _RUNNER._validate_gradient_direction_scalar_files(
        tmp_path,
        gradient_csv=global_csv,
        block_gradient_csv=block_csv,
    )
    assert observed == (block_csv, global_csv)

    unexpected = results / "gradient_direction_vectors.pt"
    unexpected.write_bytes(b"must not be persisted")
    with pytest.raises(
        _RUNNER.SO214CoreRunnerError,
        match="scalar_only_files",
    ):
        _RUNNER._validate_gradient_direction_scalar_files(
            tmp_path,
            gradient_csv=global_csv,
            block_gradient_csv=block_csv,
        )


def test_recurrent_resume_rejects_untied_model_construction(tmp_path: Path) -> None:
    recurrent = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    baseline = compose_config(
        PROJECT_ROOT / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    trainer_model = _TrainerModel(2)
    result = fit_cohort_relative_qkv_segment(
        trainer_model,
        _tiny_cohort_batches(),
        CohortRelativeQKVTrainingConfig(
            model_seed=0,
            segment_end_global_epoch=1,
            learning_rate=0.01,
            device="cpu",
        ),
    )
    tied_construction = {
        "class": "ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer",
        "num_genes": 1000,
        "node_covariate_dim": 22,
        **recurrent["model"],
    }
    untied_construction = {
        "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
        "num_genes": 1000,
        "node_covariate_dim": 22,
        **baseline["model"],
    }
    checkpoint = tmp_path / "latest.ckpt"
    torch.save(
        _RUNNER._checkpoint_payload(
            run_id="r_untied",
            config=recurrent,
            model_construction=untied_construction,
            parameter_count=5_003_016,
            resume=result.resume,
        ),
        checkpoint,
    )

    with pytest.raises(
        _RUNNER.SO214CoreRunnerError,
        match="resume.model_construction",
    ):
        _RUNNER._load_resume_checkpoint(
            checkpoint,
            config=recurrent,
            model_construction=tied_construction,
        )


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
    _RUNNER._validate_contract(source)
    _RUNNER._validate_contract(continuation)
    assert _RUNNER._campaign_id(continuation) == _RUNNER.CAMPAIGN_ID
    assert _RUNNER._preflight_path(continuation) == (
        _RUNNER.FIXED_CONTINUATION_PREFLIGHT
    )
    assert _RUNNER._preflight_schema(continuation) == (
        _RUNNER.BASELINE_PREFLIGHT_SCHEMA
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
    config = {
        "attempt": 1,
        "campaign": {"campaign_id": _RUNNER.CAMPAIGN_ID},
        "evaluation": {"protocol": _RUNNER.PLATEAU_PROTOCOL},
        "launcher": {},
    }
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
    assert not any(
        str(field).startswith("gradient_direction") for field in payload
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


def test_gradient_direction_resume_rebinds_source_run_id_and_reconciles(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_run"
    destination_root = tmp_path / "destination_run"
    (source_root / "checkpoints").mkdir(parents=True)
    destination_root.mkdir()
    source_checkpoint = source_root / "checkpoints/latest.ckpt"
    source_checkpoint.write_bytes(b"checkpoint identity only")
    source_writer = _RUNNER.DurableGradientDirectionCSV(
        source_root,
        run_id="r_source",
        model_seed=0,
    )

    def scalar_row(epoch: int) -> dict[str, object]:
        return {
            "schema": _RUNNER.GRADIENT_DIRECTION_METRICS_SCHEMA,
            "global_epoch": epoch,
            "trainable_parameter_count": 3,
            "optimizer_updates_observed": 7,
            "gradient_norm_mean_before_clip": 1.0,
            "gradient_norm_min_before_clip": 0.5,
            "gradient_norm_max_before_clip": 1.5,
            "consecutive_optimizer_step_cosine_mean": 0.25,
            "consecutive_optimizer_step_cosine_median": 0.25,
            "consecutive_optimizer_step_cosine_min": 0.0,
            "consecutive_optimizer_step_cosine_max": 0.5,
            "consecutive_optimizer_step_cosine_valid_pairs": (
                6 if epoch == 1 else 7
            ),
            "epoch_aggregate_gradient_cosine_to_previous_epoch": (
                None if epoch == 1 else 0.75
            ),
            "resume_boundary_unavailable": False,
        }

    for epoch in range(1, 4):
        source_writer.append(scalar_row(epoch))

    lineage = _RUNNER._copy_resume_gradient_direction_csv(
        run_id="r_destination",
        source_checkpoint=source_checkpoint,
        archive=SimpleNamespace(scratch_path=destination_root),
        completed_epoch=2,
    )
    destination_writer = _RUNNER.DurableGradientDirectionCSV(
        destination_root,
        run_id="r_destination",
        model_seed=0,
    )
    imported = destination_writer.read_rows()
    source_rows = source_writer.read_rows()

    assert len(imported) == 2
    assert len(source_rows) == 3
    assert {row["run_id"] for row in imported} == {"r_destination"}
    assert {row["run_id"] for row in source_rows} == {"r_source"}
    assert imported[1]["consecutive_optimizer_step_cosine_valid_pairs"] == "7"
    assert lineage["source_run_id"] == "r_source"
    assert lineage["destination_run_id"] == "r_destination"
    assert lineage["run_id_rebound"] is True
    assert lineage["source_rows"] == 3
    assert lineage["imported_rows"] == 2


def test_block_gradient_resume_rebinds_complete_epoch_groups(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_run"
    destination_root = tmp_path / "destination_run"
    (source_root / "checkpoints").mkdir(parents=True)
    destination_root.mkdir()
    source_checkpoint = source_root / "checkpoints/latest.ckpt"
    source_checkpoint.write_bytes(b"checkpoint identity only")
    source_writer = _RUNNER.DurableBlockGradientDirectionCSV(
        source_root,
        run_id="r_source",
        model_seed=0,
        expected_blocks=8,
    )

    def epoch_group(epoch: int) -> list[dict[str, object]]:
        return [
            {
                "schema": _RUNNER.BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
                "global_epoch": epoch,
                "block_index": index,
                "block_name": f"blocks.{index}",
                "trainable_parameter_count": 799_112,
                "optimizer_updates_observed": 7,
                "gradient_norm_mean_before_clip": 1.0 + index,
                "gradient_norm_min_before_clip": 0.5 + index,
                "gradient_norm_max_before_clip": 1.5 + index,
                "consecutive_optimizer_step_cosine_mean": 0.25,
                "consecutive_optimizer_step_cosine_median": 0.25,
                "consecutive_optimizer_step_cosine_min": 0.0,
                "consecutive_optimizer_step_cosine_max": 0.5,
                "consecutive_optimizer_step_cosine_valid_pairs": (
                    6 if epoch == 1 else 7
                ),
                "epoch_aggregate_gradient_cosine_to_previous_epoch": (
                    None if epoch == 1 else 0.75
                ),
                "resume_boundary_unavailable": False,
            }
            for index in range(8)
        ]

    source_writer.append(epoch_group(1))
    source_writer.append(epoch_group(2))
    lineage = _RUNNER._copy_resume_block_gradient_direction_csv(
        run_id="r_destination",
        source_checkpoint=source_checkpoint,
        archive=SimpleNamespace(scratch_path=destination_root),
        completed_epoch=1,
    )
    destination_writer = _RUNNER.DurableBlockGradientDirectionCSV(
        destination_root,
        run_id="r_destination",
        model_seed=0,
        expected_blocks=8,
    )
    imported = destination_writer.read_rows()

    assert len(imported) == 8
    assert destination_writer.completed_epochs == 1
    assert {row["run_id"] for row in imported} == {"r_destination"}
    assert [int(row["block_index"]) for row in imported] == list(range(8))
    assert lineage["source_run_id"] == "r_source"
    assert lineage["destination_run_id"] == "r_destination"
    assert lineage["run_id_rebound"] is True
    assert lineage["source_epoch_groups"] == 2
    assert lineage["imported_epoch_groups"] == 1
    assert lineage["imported_rows"] == 8


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
        "campaign": {"campaign_id": _RUNNER.CAMPAIGN_ID},
        "evaluation": {"protocol": _RUNNER.PLATEAU_PROTOCOL},
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


def test_recurrent_preflight_is_campaign_architecture_and_parameter_bound(
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
    resolved = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    content = {
        "schema": _RUNNER.RECURRENT_PREFLIGHT_SCHEMA,
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "campaign_id": _RUNNER.RECURRENT_CAMPAIGN_ID,
        "distributed_world_size": 4,
        "distributed_backend": "nccl",
        "visible_devices": "0,1,2,3",
        "elastic_max_restarts": 0,
        "optimizer_updates": 1,
        "complete_graph_mask_views": 20,
        "checkpoint_reload_verified": True,
        "finite_loss_and_gradients": True,
        "so2_c23_prior_equivalence_verified": True,
        "gradient_direction_observer_verified": True,
        "gradient_norm_before_clip": 1.25,
        "parameter_count": _RUNNER.EXPECTED_RECURRENT_PARAMETER_COUNT,
        "cohort_manifest_sha256": _RUNNER.sha256_file(
            cohort_dir / "manifest.json"
        ),
        "graph_manifest_sha256": _RUNNER.sha256_file(
            graph_dir / "manifest.json"
        ),
        "resolved_config_sha256": _RUNNER._canonical_sha256(
            _RUNNER._preflight_bound_config(resolved)
        ),
        "peak_vram_gib_all_ranks": 12.5,
        "model": {
            "class": (
                "ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer"
            ),
            "num_genes": 1000,
            "node_covariate_dim": 22,
            **resolved["model"],
        },
        "gradient_direction_preflight_summary": {
            "schema": _RUNNER.GRADIENT_DIRECTION_METRICS_SCHEMA,
            "global_epoch": 1,
            "trainable_parameter_count": 2_605_680,
            "optimizer_updates_observed": 1,
            "gradient_norm_mean_before_clip": 1.25,
            "consecutive_optimizer_step_cosine_valid_pairs": 0,
            "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
            "resume_boundary_unavailable": False,
        },
    }
    receipt_path = (
        paths.state_root
        / "preflight/so2_14core_recurrent_relative_qkv_ddp4.json"
    )
    receipt_path.parent.mkdir(parents=True)

    def write_receipt(value: dict[str, object]) -> None:
        receipt = dict(value)
        receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(value)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    write_receipt(content)
    validated = _RUNNER._validate_hardware_preflight(
        resolved,
        paths=paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    assert validated["parameter_count"] == 2_605_680

    norm_drifted = deepcopy(content)
    norm_drifted["gradient_direction_preflight_summary"][
        "gradient_norm_mean_before_clip"
    ] = 1.5
    write_receipt(norm_drifted)
    with pytest.raises(
        _RUNNER.SO214CoreRunnerError,
        match="observer norm did not match",
    ):
        _RUNNER._validate_hardware_preflight(
            resolved,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )


def test_untied8_preflight_binds_all_blocks_and_vram_gate(tmp_path: Path) -> None:
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
    resolved = compose_config(
        PROJECT_ROOT
        / "configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    block_diagnostics = [
        {
            "block_index": index,
            "block_name": f"blocks.{index}",
            "trainable_parameter_count": 799_112,
            "parameters_with_gradient": 1,
            "parameters_missing_gradient": 0,
            "gradient_norm_before_clip": 0.5 + index,
        }
        for index in range(8)
    ]
    block_summaries = [
        {
            "schema": _RUNNER.BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
            "global_epoch": 1,
            "block_index": index,
            "block_name": f"blocks.{index}",
            "trainable_parameter_count": 799_112,
            "optimizer_updates_observed": 1,
            "gradient_norm_mean_before_clip": 0.5 + index,
            "gradient_norm_min_before_clip": 0.5 + index,
            "gradient_norm_max_before_clip": 0.5 + index,
            "consecutive_optimizer_step_cosine_mean": None,
            "consecutive_optimizer_step_cosine_median": None,
            "consecutive_optimizer_step_cosine_min": None,
            "consecutive_optimizer_step_cosine_max": None,
            "consecutive_optimizer_step_cosine_valid_pairs": 0,
            "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
            "resume_boundary_unavailable": False,
        }
        for index in range(8)
    ]
    content = {
        "schema": _RUNNER.UNTIED8_PREFLIGHT_SCHEMA,
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "campaign_id": _RUNNER.UNTIED8_CAMPAIGN_ID,
        "distributed_world_size": 4,
        "distributed_backend": "nccl",
        "visible_devices": "0,1,2,3",
        "elastic_max_restarts": 0,
        "optimizer_updates": 1,
        "complete_graph_mask_views": 20,
        "checkpoint_reload_verified": True,
        "finite_loss_and_gradients": True,
        "so2_c23_prior_equivalence_verified": True,
        "gradient_direction_observer_verified": True,
        "gradient_norm_before_clip": 1.25,
        "parameter_count": _RUNNER.EXPECTED_UNTIED8_PARAMETER_COUNT,
        "cohort_manifest_sha256": _RUNNER.sha256_file(
            cohort_dir / "manifest.json"
        ),
        "graph_manifest_sha256": _RUNNER.sha256_file(graph_dir / "manifest.json"),
        "resolved_config_sha256": _RUNNER._canonical_sha256(
            _RUNNER._preflight_bound_config(resolved)
        ),
        "peak_vram_gib_all_ranks": 21.5,
        "peak_vram_gib_all_ranks_max": 22.0,
        "minimum_vram_headroom_gib_each_rank": 2.0,
        "measured_vram_headroom_gib_each_rank": 2.5,
        "vram_acceptance_passed": True,
        "model": {
            "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
            "num_genes": 1000,
            "node_covariate_dim": 22,
            **resolved["model"],
        },
        "gradient_direction_preflight_summary": {
            "schema": _RUNNER.GRADIENT_DIRECTION_METRICS_SCHEMA,
            "global_epoch": 1,
            "trainable_parameter_count": _RUNNER.EXPECTED_UNTIED8_PARAMETER_COUNT,
            "optimizer_updates_observed": 1,
            "gradient_norm_mean_before_clip": 1.25,
            "consecutive_optimizer_step_cosine_valid_pairs": 0,
            "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
            "resume_boundary_unavailable": False,
        },
        "untied_graph_block_topology": {
            "verified": True,
            "graph_block_count": 8,
            "unique_graph_block_objects": 8,
            "unique_graph_block_parameter_sets": 8,
            "state_dict_block_indices": list(range(8)),
            "graph_block_weight_tying": "none",
        },
        "all_unique_graph_blocks_receive_gradients": True,
        "graph_block_gradient_diagnostics": block_diagnostics,
        "block_gradient_direction_observer_verified": True,
        "block_gradient_direction_preflight_summary": block_summaries,
    }
    receipt_path = paths.state_root / _RUNNER.UNTIED8_PREFLIGHT.removeprefix("state/")
    receipt_path.parent.mkdir(parents=True)

    def write_receipt(value: dict[str, object]) -> None:
        receipt = dict(value)
        receipt["receipt_content_sha256"] = _RUNNER._canonical_sha256(value)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    write_receipt(content)
    validated = _RUNNER._validate_hardware_preflight(
        resolved,
        paths=paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    assert validated["parameter_count"] == 8_199_464
    assert validated["all_unique_graph_blocks_receive_gradients"] is True
    assert validated["vram_acceptance_passed"] is True

    vram_drifted = deepcopy(content)
    vram_drifted["peak_vram_gib_all_ranks"] = 22.1
    vram_drifted["measured_vram_headroom_gib_each_rank"] = 1.9
    write_receipt(vram_drifted)
    with pytest.raises(_RUNNER.SO214CoreRunnerError, match="22.0 GiB"):
        _RUNNER._validate_hardware_preflight(
            resolved,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )

    gradient_drifted = deepcopy(content)
    gradient_drifted["graph_block_gradient_diagnostics"][7][
        "parameters_missing_gradient"
    ] = 1
    write_receipt(gradient_drifted)
    with pytest.raises(
        _RUNNER.SO214CoreRunnerError,
        match=r"block_gradient\[7\].parameters_missing_gradient",
    ):
        _RUNNER._validate_hardware_preflight(
            resolved,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )

    drifted = deepcopy(content)
    drifted["model"]["graph_block_weight_tying"] = "all_steps"
    write_receipt(drifted)
    with pytest.raises(
        _RUNNER.SO214CoreRunnerError,
        match="preflight.model.graph_block_weight_tying",
    ):
        _RUNNER._validate_hardware_preflight(
            resolved,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )

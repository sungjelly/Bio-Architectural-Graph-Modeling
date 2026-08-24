from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spatial_benchmark.configuration import compose_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "run_pooled_relative_qkv",
    PROJECT_ROOT / "scripts/train/run_pooled_relative_qkv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)
RelativeQKVRunnerError = _RUNNER.RelativeQKVRunnerError
_checkpoint_payload = _RUNNER._checkpoint_payload
_configured_model_seed = _RUNNER._configured_model_seed
_load_resume_checkpoint = _RUNNER._load_resume_checkpoint
_model_from_config = _RUNNER._model_from_config
_preflight_bound_config = _RUNNER._preflight_bound_config
_seeded_model_from_config = _RUNNER._seeded_model_from_config
_seed_plateau_decision = _RUNNER._seed_plateau_decision
_training_config = _RUNNER._training_config
_validate_active_contract = _RUNNER._validate_active_contract
_validate_hardware_preflight = _RUNNER._validate_hardware_preflight


def _resolved() -> dict[str, object]:
    return compose_config(
        PROJECT_ROOT / "configs/experiment/cancer_6core_relative_qkv_seed0.yaml",
        config_root=PROJECT_ROOT / "configs",
    )


def test_active_runner_accepts_configured_seeds_zero_through_three() -> None:
    config = _resolved()
    for seed in (0, 1, 2, 3):
        candidate = deepcopy(config)
        candidate["seed"] = seed
        _validate_active_contract(candidate)
        assert _configured_model_seed(candidate) == seed
    invalid = deepcopy(config)
    invalid["seed"] = 4
    with pytest.raises(RelativeQKVRunnerError, match="one of.*0, 1, 2, 3"):
        _validate_active_contract(invalid)
    invalid["seed"] = True
    with pytest.raises(RelativeQKVRunnerError, match="integer model seed"):
        _validate_active_contract(invalid)
    invalid = deepcopy(config)
    invalid["trainer"]["mask_views_per_core_step"] = 9
    with pytest.raises(RelativeQKVRunnerError, match="mask_views_per_core_step"):
        _validate_active_contract(invalid)


@pytest.mark.parametrize("model_seed", (0, 1, 2, 3))
def test_each_seed_plateau_stops_only_after_two_passing_audits(
    model_seed: int,
) -> None:
    first = _seed_plateau_decision(np.ones(150), model_seed=model_seed)
    assert first["model_seed"] == model_seed
    assert first["current_audit"]["qualifying_passed"]
    assert not first["previous_audit"]["qualifying_passed"]
    assert first["should_stop"] is False
    second = _seed_plateau_decision(np.ones(175), model_seed=model_seed)
    assert second["consecutive_passing_audits"] == 2
    assert second["should_stop"] is True
    assert second["final_epoch"] == 175


def test_production_model_construction_uses_locked_dimensions() -> None:
    config = _resolved()
    model = _model_from_config(config, num_genes=1000, node_covariate_dim=23)
    assert len(model.blocks) == 4
    assert model.blocks[0].hidden_dim == 256
    assert model.blocks[0].attention_heads == 8
    assert model.blocks[0].attention_head_dim == 32
    assert model.receiver_chunk_size == 512
    assert model.max_edges_per_chunk == 200_000
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


@pytest.mark.parametrize("model_seed", (0, 1, 2, 3))
def test_production_model_initialization_is_bound_to_config_seed(
    model_seed: int,
) -> None:
    config = _resolved()
    config["seed"] = model_seed
    torch.manual_seed(91)
    first = _seeded_model_from_config(
        config, num_genes=17, node_covariate_dim=3
    )
    first_state = {
        name: tensor.detach().clone() for name, tensor in first.state_dict().items()
    }
    torch.manual_seed(8128)
    second = _seeded_model_from_config(
        config, num_genes=17, node_covariate_dim=3
    )
    assert first_state.keys() == second.state_dict().keys()
    assert all(
        torch.equal(first_state[name], tensor)
        for name, tensor in second.state_dict().items()
    )


def test_different_config_seeds_initialize_different_models() -> None:
    states = []
    for model_seed in (0, 1, 2, 3):
        config = _resolved()
        config["seed"] = model_seed
        model = _seeded_model_from_config(
            config, num_genes=17, node_covariate_dim=3
        )
        states.append(
            model.encoder.expression_projection.weight.detach().clone()
        )
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            assert not torch.equal(states[left], states[right])


def test_training_config_uses_model_seed_but_keeps_paired_schedule_seeds() -> None:
    training_configs = []
    for model_seed in (0, 1, 2, 3):
        config = _resolved()
        config["seed"] = model_seed
        training_configs.append(_training_config(config, start_epoch=0, end_epoch=150))
    assert [value.model_seed for value in training_configs] == [0, 1, 2, 3]
    assert len({value.mask_base_seed for value in training_configs}) == 1
    assert len({value.core_order_seed for value in training_configs}) == 1


def test_checkpoint_payload_records_and_requires_actual_model_seed() -> None:
    config = _resolved()
    config["seed"] = 2
    resume = SimpleNamespace(
        model_seed=2,
        completed_global_epochs=25,
        optimizer_steps_completed=150,
        mask_base_seed=2026082401,
        core_order_seed=2026082402,
        mask_views_per_core_step=10,
        model_state_dict={},
        model_state_checksum="a" * 64,
        optimizer_state_dict={},
        optimizer_state_checksum="b" * 64,
        scaler_state_dict={},
        scaler_state_checksum="c" * 64,
        core_history=(),
        global_history=(),
        history_checksum="d" * 64,
        resume_checksum="e" * 64,
    )
    payload = _checkpoint_payload(
        archive=SimpleNamespace(run_id="run-seed-2"),
        config=config,
        model_construction={"class": "Synthetic"},
        parameter_count=1,
        resume=resume,
    )
    assert payload["model_seed"] == 2
    resume.model_seed = 1
    with pytest.raises(RelativeQKVRunnerError, match="resume.model_seed"):
        _checkpoint_payload(
            archive=SimpleNamespace(run_id="run-seed-2"),
            config=config,
            model_construction={"class": "Synthetic"},
            parameter_count=1,
            resume=resume,
        )


def test_resume_checkpoint_rejects_a_different_actual_model_seed(
    tmp_path: Path,
) -> None:
    config = _resolved()
    config["seed"] = 2
    checkpoint = tmp_path / "wrong-seed.ckpt"
    torch.save(
        {
            "checkpoint_schema": "cancer_6core_relative_qkv_resume_v1",
            "campaign_id": _RUNNER.CAMPAIGN_ID,
            "model_seed": 1,
        },
        checkpoint,
    )
    with pytest.raises(RelativeQKVRunnerError, match="resume.model_seed"):
        _load_resume_checkpoint(
            checkpoint,
            config=config,
            model_construction={"class": "Synthetic"},
        )


def test_production_runner_requires_checksum_bound_hardware_preflight(
    tmp_path: Path,
) -> None:
    config = _resolved()
    cohort = tmp_path / "cohort"
    graph = tmp_path / "graph"
    cohort.mkdir()
    graph.mkdir()
    (cohort / "manifest.json").write_text('{"cohort": true}\n')
    (graph / "manifest.json").write_text('{"graph": true}\n')
    config["dataset"]["prepared_artifact"] = str(cohort)
    config["dataset"]["prepared_graph_artifact"] = str(graph)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt = {
        "schema": "cancer_6core_relative_qkv_hardware_preflight_v1",
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "cohort_manifest_sha256": sha256(cohort / "manifest.json"),
        "graph_manifest_sha256": sha256(graph / "manifest.json"),
        "model": {
            "receiver_chunk_size": config["model"]["receiver_chunk_size"],
            "max_edges_per_chunk": config["model"]["max_edges_per_chunk"],
            "activation_checkpointing": config["model"]["activation_checkpointing"],
            "stage_complete_core_graph_on_device": config["trainer"][
                "stage_complete_core_graph_on_device"
            ],
            "staged_relative_geometry_dtype": config["trainer"][
                "staged_relative_geometry_dtype"
            ],
        },
    }
    receipt["receipt_content_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(receipt))
    assert _validate_hardware_preflight(config, receipt_path=path) == receipt
    for model_seed in (1, 2, 3):
        paired_config = compose_config(
            PROJECT_ROOT
            / f"configs/experiment/cancer_6core_relative_qkv_seed{model_seed}.yaml",
            config_root=PROJECT_ROOT / "configs",
        )
        paired_config["dataset"]["prepared_artifact"] = str(cohort)
        paired_config["dataset"]["prepared_graph_artifact"] = str(graph)
        assert _preflight_bound_config(paired_config) == _preflight_bound_config(
            config
        )
        assert (
            _validate_hardware_preflight(paired_config, receipt_path=path) == receipt
        )
    retry_config = deepcopy(config)
    retry_config["attempt"] = 2
    assert _validate_hardware_preflight(retry_config, receipt_path=path) == receipt

    scientific_drift = deepcopy(config)
    scientific_drift["seed"] = 1
    scientific_drift["model"]["dropout"] = 0.2
    assert _preflight_bound_config(scientific_drift) != _preflight_bound_config(config)
    with pytest.raises(RelativeQKVRunnerError, match="configuration changed"):
        _validate_hardware_preflight(scientific_drift, receipt_path=path)

    receipt["model"]["receiver_chunk_size"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(RelativeQKVRunnerError, match="checksum mismatch"):
        _validate_hardware_preflight(config, receipt_path=path)

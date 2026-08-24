from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

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
_model_from_config = _RUNNER._model_from_config
_seed_plateau_decision = _RUNNER._seed_plateau_decision
_validate_active_contract = _RUNNER._validate_active_contract
_validate_hardware_preflight = _RUNNER._validate_hardware_preflight


def _resolved() -> dict[str, object]:
    return compose_config(
        PROJECT_ROOT / "configs/experiment/cancer_6core_relative_qkv_seed0.yaml",
        config_root=PROJECT_ROOT / "configs",
    )


def test_active_runner_accepts_only_seed0_amended_contract() -> None:
    config = _resolved()
    _validate_active_contract(config)
    invalid = deepcopy(config)
    invalid["seed"] = 1
    with pytest.raises(RelativeQKVRunnerError, match="seed must be 0"):
        _validate_active_contract(invalid)
    invalid = deepcopy(config)
    invalid["trainer"]["mask_views_per_core_step"] = 9
    with pytest.raises(RelativeQKVRunnerError, match="mask_views_per_core_step"):
        _validate_active_contract(invalid)


def test_seed0_plateau_stops_only_after_two_passing_audits() -> None:
    first = _seed_plateau_decision(np.ones(150))
    assert first["current_audit"]["qualifying_passed"]
    assert not first["previous_audit"]["qualifying_passed"]
    assert first["should_stop"] is False
    second = _seed_plateau_decision(np.ones(175))
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

    receipt["model"]["receiver_chunk_size"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(RelativeQKVRunnerError, match="checksum mismatch"):
        _validate_hardware_preflight(config, receipt_path=path)

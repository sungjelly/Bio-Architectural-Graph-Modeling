from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.negative_binomial import (
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
)
from spatial_benchmark.paths import current_paths
from spatial_benchmark.so2_nb_data import SO2NBCoreBatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "preflight_so2_geometry_modulated_nb_ddp",
    PROJECT_ROOT
    / "scripts/diagnostics/preflight_so2_geometry_modulated_nb_ddp.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREFLIGHT
_SPEC.loader.exec_module(_PREFLIGHT)


@pytest.fixture(scope="module")
def frozen_model() -> ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer:
    torch.manual_seed(0)
    return ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=1_000,
        node_covariate_dim=22,
        hidden_dim=256,
        attention_heads=8,
        attention_head_dim=32,
        graph_layers=4,
        ffn_dim=1_024,
        decoder_dim=1_024,
        geometry_hidden_dim=128,
        dropout=0.1,
        attention_dropout=0.0,
        receiver_chunk_size=128,
        max_edges_per_chunk=50_000,
        activation_checkpointing=True,
    )


def _compact_model() -> ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer:
    torch.manual_seed(0)
    return ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=7,
        node_covariate_dim=22,
        hidden_dim=16,
        attention_heads=2,
        attention_head_dim=8,
        graph_layers=4,
        ffn_dim=32,
        decoder_dim=16,
        geometry_hidden_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=4,
        max_edges_per_chunk=64,
        activation_checkpointing=False,
    )


def _compact_batch() -> SO2NBCoreBatch:
    torch.manual_seed(5)
    nodes, genes = 6, 7
    source = torch.arange(nodes, dtype=torch.long).repeat(2)
    receiver = torch.cat(
        (torch.arange(nodes).roll(1), torch.arange(nodes).roll(2))
    )
    edges = torch.stack((source, receiver), dim=0)
    return SO2NBCoreBatch(
        alias="SO2-C15",
        role="train",
        input_expression=torch.randn(nodes, genes, dtype=torch.float32),
        raw_count_target=torch.randint(
            0, 730, (nodes, genes), dtype=torch.int32
        ),
        node_covariates=torch.randn(nodes, 22, dtype=torch.float32),
        edge_index=edges,
        relative_geometry=torch.randn(edges.shape[1], 70, dtype=torch.float32),
    )


def _valid_receipt() -> dict[str, object]:
    device_bytes = 25_296_044_032
    device_gib = device_bytes / 1024**3
    identities = [
        {
            "rank": rank,
            "local_rank": rank,
            "name": "NVIDIA GeForce RTX 3090",
            "total_memory_bytes": device_bytes,
            "total_memory_gib": device_gib,
            "compute_capability": [8, 6],
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        for rank in range(4)
    ]
    vram = _PREFLIGHT._vram_receipt_from_records(
        [
            {
                "rank": rank,
                "local_rank": rank,
                "total_memory_bytes": device_bytes,
                "peak_allocated_vram_gib": 20.25 - 0.1 * rank,
                "peak_reserved_vram_gib": 20.5 - 0.1 * rank,
            }
            for rank in range(4)
        ]
    )
    gates = {
        name: {"passed": True} for name in _PREFLIGHT.REQUIRED_GATE_NAMES
    }
    gates["peak_vram"] = vram
    gates["gpu_inventory"] = {"passed": True, "devices": identities}
    receipt: dict[str, object] = {
        "schema": _PREFLIGHT.PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "campaign_id": _PREFLIGHT.CAMPAIGN_ID,
        "protocol": _PREFLIGHT.PROTOCOL,
        "world_size": 4,
        "control_plane_backend": "gloo",
        "parameter_count": 5_135_088,
        "receiver_chunk_size": 128,
        "train_update_completed": True,
        "validation_completed": True,
        "checkpoint_reload_verified": True,
        "test_artifacts_present": False,
        "configuration_sha256": "1" * 64,
        "resolved_config_sha256": "1" * 64,
        "configuration_file_sha256": _PREFLIGHT.sha256_file(
            PROJECT_ROOT / _PREFLIGHT.SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH
        ),
        "frozen_task_contract_sha256": _PREFLIGHT.FROZEN_TASK_CONTRACT_SHA256,
        "overlay_manifest_sha256": "3" * 64,
        "overlay_manifest_content_sha256": "4" * 64,
        "preprocessing_fingerprint": "5" * 64,
        "split_fingerprint": "6" * 64,
        "configuration_source_file_sha256": _PREFLIGHT._configuration_file_hashes(
            PROJECT_ROOT / _PREFLIGHT.SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH,
            PROJECT_ROOT / "configs",
        ),
        "code_file_sha256": _PREFLIGHT._code_file_hashes(),
        "code_hash_scope": _PREFLIGHT.CODE_HASH_SCOPE,
        "gpu_identities": identities,
        "peak_vram_gib_all_ranks": vram["peak_vram_gib_all_ranks"],
        "peak_reserved_vram_gib_all_ranks": vram[
            "peak_reserved_vram_gib_all_ranks"
        ],
        "minimum_vram_headroom_gib_each_rank": vram[
            "minimum_vram_headroom_gib_each_rank"
        ],
        "minimum_reserved_vram_headroom_gib_each_rank": vram[
            "minimum_reserved_vram_headroom_gib_each_rank"
        ],
        "gates": gates,
    }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    return receipt


def test_cli_requires_exact_three_paths_and_locks_identity() -> None:
    parser = _PREFLIGHT.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "experiment.yaml", "--overlay", "overlay"])
    parsed = parser.parse_args(
        [
            "--config",
            "experiment.yaml",
            "--overlay",
            "overlay",
            "--output",
            "receipt.json",
        ]
    )
    assert parsed == argparse.Namespace(
        config=Path("experiment.yaml"),
        overlay=Path("overlay"),
        output=Path("receipt.json"),
    )
    assert _PREFLIGHT.PREFLIGHT_SCHEMA == (
        "so2_geometry_modulated_nb_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.PREFLIGHT_TRAINING_PAIR == ("SO2-C23", "SO2-C24")
    assert _PREFLIGHT.WORLD_SIZE == 4
    assert _PREFLIGHT.VISIBLE_DEVICES == "0,1,2,3"
    assert _PREFLIGHT.CODE_RELATIVE_PATHS == tuple(
        _PREFLIGHT.RUNNER_PREFLIGHT_CODE_FILES
    )
    assert _PREFLIGHT.REQUIRED_GATE_NAMES == frozenset(
        _PREFLIGHT.RUNNER_PREFLIGHT_REQUIRED_GATES
    )
    assert _PREFLIGHT.CODE_HASH_SCOPE == _PREFLIGHT.RUNNER_CODE_HASH_SCOPE
    assert "amp_fp32_equivalence" in _PREFLIGHT.REQUIRED_GATE_NAMES


def test_resolves_composed_config_and_exact_overlay_output_paths() -> None:
    paths = current_paths(anchor=PROJECT_ROOT)
    args = argparse.Namespace(
        config=Path("configs/experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"),
        overlay=Path("data/processed/so2_nb_train12_val2_v1"),
        output=Path(
            "state/preflight/so2_geometry_modulated_nb_train12_val2_ddp4.json"
        ),
    )
    config_path, overlay, output, config = _PREFLIGHT._resolve_required_paths(
        args, paths=paths
    )
    assert config_path.is_file()
    assert overlay == paths.data_root / "processed/so2_nb_train12_val2_v1"
    assert output == paths.state_root / (
        "preflight/so2_geometry_modulated_nb_train12_val2_ddp4.json"
    )
    assert config["model"]["graph_layers"] == 4
    assert config["model"]["expected_trainable_parameter_count"] == 5_135_088
    assert config["dataset"]["test_core_aliases"] == []
    assert canonical_sha256(config) == canonical_sha256(config)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("overlay", Path("data/processed/not-the-frozen-overlay")),
        ("output", Path("state/preflight/not-the-frozen-receipt.json")),
    ],
)
def test_rejects_mutated_runtime_paths(field: str, replacement: Path) -> None:
    paths = current_paths(anchor=PROJECT_ROOT)
    values = {
        "config": Path(
            "configs/experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"
        ),
        "overlay": Path("data/processed/so2_nb_train12_val2_v1"),
        "output": Path(
            "state/preflight/so2_geometry_modulated_nb_train12_val2_ddp4.json"
        ),
    }
    values[field] = replacement
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._resolve_required_paths(argparse.Namespace(**values), paths=paths)


def test_frozen_model_topology_and_neutral_geometry(
    frozen_model: ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
) -> None:
    topology = _PREFLIGHT._model_topology_receipt(frozen_model)
    assert topology == {
        "passed": True,
        "model_class": (
            "ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer"
        ),
        "parameter_count": 5_135_088,
        "graph_block_count": 4,
        "distinct_graph_block_objects": 4,
        "disjoint_graph_block_parameter_sets": True,
        "per_block_trainable_parameter_count": [831_880] * 4,
        "raw_theta_parameter_count": 1_000,
        "raw_theta_registered_at_model_root": True,
    }
    neutral = _PREFLIGHT._neutral_geometry_receipt(
        frozen_model, torch.randn(11, 70)
    )
    assert neutral["passed"] is True
    assert len(neutral["blocks"]) == 4
    assert all(row["modulation_max_abs_from_one"] == 0 for row in neutral["blocks"])
    assert all(row["bias_max_abs"] == 0 for row in neutral["blocks"])


def test_topology_rejects_parameter_or_block_mutation(
    frozen_model: ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
) -> None:
    original = frozen_model.blocks
    frozen_model.blocks = torch.nn.ModuleList(list(original[:3]))
    try:
        with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
            _PREFLIGHT._model_topology_receipt(frozen_model)
    finally:
        frozen_model.blocks = original


def test_raw_target_perturbation_cannot_change_forward_mu() -> None:
    result = _PREFLIGHT._raw_target_isolation_receipt(
        _compact_model(), _compact_batch()
    )
    assert result["passed"] is True
    assert result["forward_mu_bit_identical"] is True
    assert result["perturbed_raw_target_sha256"][0] != result[
        "perturbed_raw_target_sha256"
    ][1]
    assert result["forward_mu_sha256"][0] == result["forward_mu_sha256"][1]
    assert "raw_count_target" not in result["forward_parameter_names"]


def test_amp_fp32_equivalence_uses_real_edges_and_fp32_nb_likelihood() -> None:
    model = _compact_model()
    assert model.training is True
    result = _PREFLIGHT._amp_fp32_equivalence_receipt(
        model,
        _compact_batch(),
        device=torch.device("cpu"),
        amp_dtype="auto",
        staged_geometry_dtype="float16",
    )
    assert result["passed"] is True
    assert result["comparison_edges"] == 12
    assert result["comparison_nodes"] == 6
    assert result["likelihood_dtype"] == "float32_outside_autocast"
    assert result["mean"]["maximum_absolute_difference"] <= result["mean"][
        "maximum_absolute_tolerance"
    ]
    loss = result["masked_full_constant_nb2_nll"]
    assert loss["absolute_difference"] <= loss["combined_absolute_tolerance"]
    assert loss["tolerance_rule"] == "abs_delta_le_atol_plus_rtol_times_abs_fp32"
    assert result["gradient_comparison_performed"] is False
    assert result["gradient_gate_separate"] == "preclip_gradient_flow"
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_amp_and_observed_memory_config_fields_are_frozen() -> None:
    paths = current_paths(anchor=PROJECT_ROOT)
    args = argparse.Namespace(
        config=_PREFLIGHT.SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH,
        overlay=Path("data/processed/so2_nb_train12_val2_v1"),
        output=Path(
            "state/preflight/so2_geometry_modulated_nb_train12_val2_ddp4.json"
        ),
    )
    _, _, _, config = _PREFLIGHT._resolve_required_paths(args, paths=paths)
    _PREFLIGHT._validate_amp_configuration(config)
    assert config["metadata"]["preflight_acceptance"] == {
        "peak_vram_gib_all_ranks_max": 22.0,
        "minimum_vram_headroom_gib_each_rank": 2.0,
        "minimum_observed_cuda_memory_gib_each_rank": 23.0,
    }

    mutations = (
        ("amp", False),
        ("likelihood_compute_dtype", "float16"),
        ("likelihood_outside_autocast", False),
        ("amp_requires_fp32_equivalence_preflight", False),
    )
    for field, value in mutations:
        mutated = deepcopy(config)
        mutated["trainer"][field] = value
        with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
            _PREFLIGHT._validate_amp_configuration(mutated)

    obsolete_memory = deepcopy(config)
    acceptance = obsolete_memory["metadata"]["preflight_acceptance"]
    acceptance["gpu_memory_gib_each_rank"] = 24.0
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_amp_configuration(obsolete_memory)


def test_full_constant_nb2_synthetic_and_real_like_losses_decrease() -> None:
    synthetic = _PREFLIGHT._synthetic_loss_decrease_receipt()
    assert synthetic["passed"] is True
    assert synthetic["minimum_count"] == 0
    assert synthetic["maximum_count"] == 729
    assert synthetic["likelihood_dtype"] == "float32"
    assert synthetic["final_full_constant_masked_nb2_nll"] < synthetic[
        "initial_full_constant_masked_nb2_nll"
    ]

    real_like = _PREFLIGHT._loss_decrease_check(
        torch.tensor(
            [[0, 0, 1, 7], [2, 4, 0, 18], [0, 3, 9, 31]],
            dtype=torch.int32,
        ),
        steps=12,
    )
    assert real_like["passed"] is True
    assert real_like["finite_nonzero_gradients"] is True


def test_gradient_snapshot_is_finite_nonzero_and_rejects_zero() -> None:
    layer = torch.nn.Linear(3, 2)
    layer(torch.ones(4, 3)).square().sum().backward()
    result = _PREFLIGHT._gradient_snapshot(layer, label="test")
    assert result["finite_nonzero"] is True
    assert result["gradient_norm_before_clip"] > 0
    assert result["read_timing"] == (
        "post_amp_unscale_pre_gradient_clip_pre_optimizer_step"
    )
    layer.zero_grad(set_to_none=False)
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._gradient_snapshot(layer, label="zero")


def test_state_hash_is_order_stable_and_detects_mutation() -> None:
    first = {
        "b": torch.tensor([3], dtype=torch.int64),
        "a": torch.tensor(2.0),
    }
    reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert _PREFLIGHT._state_dict_sha256(first) == _PREFLIGHT._state_dict_sha256(
        reordered
    )
    reordered["a"].add_(1)
    assert _PREFLIGHT._state_dict_sha256(first) != _PREFLIGHT._state_dict_sha256(
        reordered
    )


def test_config_contract_and_code_hash_inventory() -> None:
    config_path = PROJECT_ROOT / (
        "configs/experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"
    )
    config_hashes = _PREFLIGHT._configuration_file_hashes(
        config_path, PROJECT_ROOT / "configs"
    )
    assert len(config_hashes) == 9
    assert str(config_path.relative_to(PROJECT_ROOT)) in config_hashes
    assert all(len(value) == 64 for value in config_hashes.values())
    code_hashes = _PREFLIGHT._code_file_hashes()
    assert set(code_hashes) == {str(path) for path in _PREFLIGHT.CODE_RELATIVE_PATHS}
    assert {
        "src/spatial_benchmark/gradient_direction_observability.py",
        "src/spatial_benchmark/pooled_relative_qkv_training_v2.py",
        "src/spatial_benchmark/pooled_relative_qkv_training.py",
        "src/spatial_benchmark/adjacency_ablation.py",
        "src/spatial_benchmark/masking.py",
        "src/spatial_benchmark/training.py",
        "src/spatial_benchmark/relative_qkv_graph_transformer.py",
        "src/spatial_benchmark/models.py",
        "src/spatial_benchmark/paths.py",
        "src/spatial_benchmark/run_archive.py",
        "src/spatial_benchmark/identifiers.py",
        "src/spatial_benchmark/fingerprints.py",
        "src/spatial_benchmark/so2_pooled_full_core.py",
        "src/spatial_benchmark/so2_relative_graphs.py",
        "src/spatial_benchmark/queueing.py",
    } < set(code_hashes)
    assert len(code_hashes) == 22
    assert _PREFLIGHT.CODE_HASH_SCOPE == (
        "campaign_numerical_and_data_integrity_resolution_plus_runner_archive_"
        "and_queue_launch_finalization"
    )
    assert all(len(value) == 64 for value in code_hashes.values())
    assert _PREFLIGHT.sha256_file(
        PROJECT_ROOT / _PREFLIGHT.FROZEN_CONTRACT_RELATIVE_PATH
    ) == _PREFLIGHT.FROZEN_TASK_CONTRACT_SHA256


def test_gpu_inventory_accepts_observed_3090_capacity_and_exact_capability() -> None:
    device_bytes = 25_296_044_032
    device_gib = device_bytes / 1024**3
    identities = [
        {
            "rank": rank,
            "local_rank": rank,
            "name": "NVIDIA GeForce RTX 3090",
            "total_memory_bytes": device_bytes,
            "total_memory_gib": device_gib,
            "compute_capability": [8, 6],
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        for rank in range(4)
    ]
    receipt = _PREFLIGHT._gpu_inventory_from_identities(identities)
    assert receipt["passed"] is True
    assert receipt["devices"][0]["total_memory_gib"] == pytest.approx(
        23.55877685546875
    )

    wrong_capability = json.loads(json.dumps(identities))
    wrong_capability[3]["compute_capability"] = [8, 0]
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._gpu_inventory_from_identities(wrong_capability)


def test_vram_headroom_uses_each_observed_device_total_and_reserved_peak() -> None:
    device_bytes = 25_296_044_032
    records = [
        {
            "rank": rank,
            "local_rank": rank,
            "total_memory_bytes": device_bytes,
            "peak_allocated_vram_gib": 20.0 + rank * 0.1,
            "peak_reserved_vram_gib": 20.2 + rank * 0.1,
        }
        for rank in range(4)
    ]
    receipt = _PREFLIGHT._vram_receipt_from_records(records)
    observed_total = device_bytes / 1024**3
    assert receipt["peak_vram_gib_all_ranks"] == pytest.approx(20.3)
    assert receipt["peak_reserved_vram_gib_all_ranks"] == pytest.approx(20.5)
    assert receipt["minimum_vram_headroom_gib_each_rank"] == pytest.approx(
        observed_total - 20.3
    )
    assert receipt["minimum_reserved_vram_headroom_gib_each_rank"] == pytest.approx(
        observed_total - 20.5
    )
    assert receipt["minimum_vram_headroom_gib_each_rank"] != pytest.approx(
        24.0 - 20.3
    )

    unsafe = json.loads(json.dumps(records))
    unsafe[0]["peak_reserved_vram_gib"] = observed_total - 1.99
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._vram_receipt_from_records(unsafe)


def test_atomic_json_replaces_without_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "nested/receipt.json"
    _PREFLIGHT._atomic_json(output, {"b": 2, "a": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert [path.name for path in output.parent.iterdir()] == ["receipt.json"]


def test_receipt_validation_accepts_exact_contract_and_rejects_mutations() -> None:
    receipt = _valid_receipt()
    _PREFLIGHT._validate_receipt(receipt)

    failed_gate = json.loads(json.dumps(receipt))
    failed_gate["gates"]["peak_vram"]["passed"] = False
    failed_gate["receipt_content_sha256"] = canonical_sha256(
        {key: value for key, value in failed_gate.items() if key != "receipt_content_sha256"}
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(failed_gate)

    missing_amp_gate = json.loads(json.dumps(receipt))
    del missing_amp_gate["gates"]["amp_fp32_equivalence"]
    missing_amp_gate["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in missing_amp_gate.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(missing_amp_gate)

    drifted_code = json.loads(json.dumps(receipt))
    drifted_code["code_file_sha256"][str(_PREFLIGHT.CODE_RELATIVE_PATHS[-1])] = (
        "9" * 64
    )
    drifted_code["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in drifted_code.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(drifted_code)

    wrong_headroom = json.loads(json.dumps(receipt))
    wrong_headroom["minimum_vram_headroom_gib_each_rank"] = 3.74
    wrong_headroom["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in wrong_headroom.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(wrong_headroom)

    wrong_identity = json.loads(json.dumps(receipt))
    wrong_identity["gpu_identities"][0]["name"] = "stale-device-name"
    wrong_identity["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in wrong_identity.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(wrong_identity)

    stale_capacity = json.loads(json.dumps(receipt))
    stale_total_bytes = 25_100_000_000
    stale_total_gib = stale_total_bytes / 1024**3
    for identity_list in (
        stale_capacity["gpu_identities"],
        stale_capacity["gates"]["gpu_inventory"]["devices"],
    ):
        identity_list[3]["total_memory_bytes"] = stale_total_bytes
        identity_list[3]["total_memory_gib"] = stale_total_gib
    stale_capacity["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in stale_capacity.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(stale_capacity)

    stale_runtime = json.loads(json.dumps(receipt))
    for identity_list in (
        stale_runtime["gpu_identities"],
        stale_runtime["gates"]["gpu_inventory"]["devices"],
    ):
        identity_list[1]["torch_version"] = "stale-torch-version"
    stale_runtime["receipt_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in stale_runtime.items()
            if key != "receipt_content_sha256"
        }
    )
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(stale_runtime)

    bad_self_hash = dict(receipt)
    bad_self_hash["receipt_content_sha256"] = "0" * 64
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(bad_self_hash)

    missing_self_hash = dict(receipt)
    del missing_self_hash["receipt_content_sha256"]
    with pytest.raises(_PREFLIGHT.SO2NBPreflightError):
        _PREFLIGHT._validate_receipt(missing_self_hash)

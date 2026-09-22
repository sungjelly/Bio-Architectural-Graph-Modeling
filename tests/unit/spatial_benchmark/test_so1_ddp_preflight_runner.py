from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (
    CohortRelativeQKVCoreBatch,
)
from spatial_benchmark.so1_pooled_full_core import EXPECTED_CELL_COUNTS_BY_CORE


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
    "preflight_so1_14core_relative_qkv_ddp",
    PROJECT_ROOT / "scripts/diagnostics/preflight_so1_14core_relative_qkv_ddp.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREFLIGHT
_SPEC.loader.exec_module(_PREFLIGHT)


def _resolved_config() -> dict[str, object]:
    return compose_config(EXPERIMENT_CONFIG, config_root=PROJECT_ROOT / "configs")


def _resolved_geometry_config() -> dict[str, object]:
    return compose_config(
        GEOMETRY_EXPERIMENT_CONFIG,
        config_root=PROJECT_ROOT / "configs",
    )


def _prior_receipt(path: Path, *, full_chunk_passed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "cancer_6core_relative_qkv_preflight_v1",
                "status": "passed",
                "all_required_gates_passed": True,
                "largest_core": {"alias": "CAN-23"},
                "gates": {
                    "full_chunk_exactness": {
                        "passed": full_chunk_passed,
                        "maximum_absolute_difference": 0.0,
                    },
                    "amp_fp32_equivalence": {
                        "passed": True,
                        "maximum_absolute_difference": 1e-4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _prior_geometry_operator_receipt(path: Path) -> Path:
    value = {
        "schema": _PREFLIGHT.PRIOR_GEOMETRY_OPERATOR_SCHEMA,
        "status": "passed",
        "all_required_gates_passed": True,
        "campaign_id": _PREFLIGHT.PRIOR_GEOMETRY_OPERATOR_CAMPAIGN_ID,
        "parameter_count": 5_134_088,
        "selected_core_aliases": ["SO2-C22", "SO2-C23"],
        "cohort_manifest_sha256": "cohort-sha",
        "graph_manifest_sha256": "graph-sha",
        "geometry_modulated_numerical_gates": {
            "passed": True,
            "full_chunk_exactness": {"passed": True},
            "amp_fp32_equivalence": {"passed": True},
            "learned_geometry_heads": {"passed": True},
        },
        "model": {
            "class": (
                "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
            ),
            "attention_score_mechanism": "geometry_modulated_cosine_qkv_v1",
            "graph_layers": 4,
            "unique_graph_blocks": 4,
            "graph_block_weight_tying": "none",
            "relative_geometry_value_injection": False,
            "relative_geometry_role": (
                "attention_logit_modulation_and_bias_only"
            ),
        },
        "geometry_modulated_graph_block_topology": {
            "verified": True,
            "graph_block_count": 4,
            "unique_graph_block_objects": 4,
            "unique_graph_block_parameter_sets": 4,
            "graph_block_weight_tying": "none",
        },
    }
    value["receipt_content_sha256"] = _PREFLIGHT._canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_preflight_import_defaults_and_locked_largest_pair() -> None:
    assert _PREFLIGHT.PREFLIGHT_SCHEMA == (
        "so1_14core_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.PREFLIGHT_ALIASES == ("SO1-C06", "SO1-C11")
    assert EXPECTED_CELL_COUNTS_BY_CORE[6] == 18_212
    assert EXPECTED_CELL_COUNTS_BY_CORE[11] == 18_145
    assert sorted(EXPECTED_CELL_COUNTS_BY_CORE.values(), reverse=True)[:2] == [
        18_212,
        18_145,
    ]
    assert _PREFLIGHT.DEFAULT_OUTPUT == Path(
        "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
    )
    parsed = _PREFLIGHT.build_parser().parse_args(["--config", "experiment.yaml"])
    assert parsed.config == Path("experiment.yaml")
    assert parsed.output is None
    assert parsed.prior_equivalence_receipt is None


def test_preflight_training_config_is_one_ddp4_paired_update() -> None:
    config = _PREFLIGHT._preflight_config(
        _resolved_config(), rank=2, local_rank=2
    )
    assert config.model_seed == 0
    assert config.cohort_aliases == ("SO1-C06", "SO1-C11")
    assert config.segment_start_global_epoch == 0
    assert config.segment_end_global_epoch == 1
    assert config.cores_per_optimizer_update == 2
    assert config.optimizer_updates_per_global_epoch == 1
    assert config.mask_views_per_core == 10
    assert config.losses_per_optimizer_update == 20
    assert config.distributed_world_size == 4
    assert config.distributed_rank == 2
    assert config.device == "cuda:2"
    assert config.checkpoint_interval_global_epochs == 1


def test_prior_relative_qkv_equivalence_is_scoped_and_checksum_bound(
    tmp_path: Path,
) -> None:
    prior = _prior_receipt(tmp_path / "prior.json")
    verified = _PREFLIGHT._prior_relative_qkv_equivalence(
        prior_receipt=prior
    )
    assert verified["verified"] is True
    assert verified["scope"] == "shared_relative_qkv_implementation_only"
    assert verified["so1_graph_identity_claimed"] is False
    assert verified["prior_largest_core_alias"] == "CAN-23"
    assert verified["prior_receipt_sha256"] == sha256_file(prior)
    assert verified["prior_full_chunk_exactness"]["passed"] is True
    assert verified["prior_amp_fp32_equivalence"]["passed"] is True


def test_prior_relative_qkv_equivalence_fails_closed_on_gate_drift(
    tmp_path: Path,
) -> None:
    prior = _prior_receipt(tmp_path / "prior.json", full_chunk_passed=False)
    with pytest.raises(RuntimeError, match="lacks the passing"):
        _PREFLIGHT._prior_relative_qkv_equivalence(prior_receipt=prior)

    prior.write_text(
        json.dumps(
            {
                "status": "passed",
                "all_required_gates_passed": True,
                "largest_core": {"alias": "CAN-22"},
                "gates": {
                    "full_chunk_exactness": {"passed": True},
                    "amp_fp32_equivalence": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks the passing"):
        _PREFLIGHT._prior_relative_qkv_equivalence(prior_receipt=prior)


def test_atomic_receipt_write_replaces_without_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    first = {"schema": _PREFLIGHT.PREFLIGHT_SCHEMA, "status": "passed"}
    _PREFLIGHT._atomic_json(output, first)
    assert json.loads(output.read_text(encoding="utf-8")) == first

    second = {**first, "prior_relative_qkv_equivalence_verified": True}
    _PREFLIGHT._atomic_json(output, second)
    assert json.loads(output.read_text(encoding="utf-8")) == second
    assert not list(tmp_path.glob(".receipt.json.*.writing"))


def test_geometry_preflight_config_builds_exact_neutral_four_block_model() -> None:
    config = _resolved_geometry_config()
    _PREFLIGHT._validate_contract(config)
    model = _PREFLIGHT._model_from_config(
        config,
        num_genes=1000,
        node_covariate_dim=22,
    )

    assert isinstance(
        model,
        ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_134_088
    assert _PREFLIGHT._preflight_schema(config) == (
        "so1_14core_geometry_modulated_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT._geometry_modulated4_block_topology(model) == {
        "verified": True,
        "graph_block_count": 4,
        "unique_graph_block_objects": 4,
        "unique_graph_block_parameter_sets": 4,
        "state_dict_block_indices": [0, 1, 2, 3],
        "graph_block_weight_tying": "none",
    }
    initialization = _PREFLIGHT._geometry_modulated_initialization_checks(model)
    assert initialization["passed"] is True
    assert len(initialization["blocks"]) == 4

    parsed = _PREFLIGHT.build_parser().parse_args(["--config", "geometry.yaml"])
    assert parsed.prior_geometry_operator_receipt is None
    assert config["launcher"]["hardware_preflight_receipt"] == (
        "state/preflight/"
        "so1_14core_geometry_modulated_relative_qkv_ddp4_plateau_min150.json"
    )


def test_prior_geometry_operator_evidence_is_hash_bound_and_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _prior_geometry_operator_receipt(tmp_path / "geometry.json")
    monkeypatch.setattr(
        _PREFLIGHT,
        "EXPECTED_PRIOR_GEOMETRY_OPERATOR_RECEIPT_SHA256",
        sha256_file(prior),
    )
    verified = _PREFLIGHT._prior_geometry_operator_evidence(
        prior_receipt=prior
    )

    assert verified["verified"] is True
    assert verified["scope"] == (
        "shared_geometry_modulated_attention_implementation_only"
    )
    assert verified["so1_graph_identity_claimed"] is False
    assert verified["prior_core_aliases"] == ["SO2-C22", "SO2-C23"]
    assert verified["prior_receipt_sha256"] == sha256_file(prior)
    assert verified["prior_full_chunk_exactness"]["passed"] is True
    assert verified["prior_amp_fp32_equivalence"]["passed"] is True
    assert verified["prior_learned_geometry_heads"]["passed"] is True


def test_prior_geometry_operator_evidence_fails_closed_on_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _prior_geometry_operator_receipt(tmp_path / "geometry.json")
    value = json.loads(prior.read_text(encoding="utf-8"))
    value["status"] = "failed"
    prior.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        _PREFLIGHT,
        "EXPECTED_PRIOR_GEOMETRY_OPERATOR_RECEIPT_SHA256",
        sha256_file(prior),
    )
    with pytest.raises(RuntimeError, match="content checksum mismatch"):
        _PREFLIGHT._prior_geometry_operator_evidence(prior_receipt=prior)


def _tiny_geometry_config() -> dict[str, object]:
    return {
        "model": {
            "hidden_dim": 12,
            "attention_heads": 3,
            "attention_head_dim": 4,
            "graph_layers": 4,
            "ffn_dim": 24,
            "decoder_dim": 16,
            "geometry_hidden_dim": 10,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "relative_geometry_dim": 70,
            "qk_normalization_epsilon": 1e-6,
            "logit_scale_initial": 1.8856180831641267,
            "logit_scale_minimum": 0.1,
            "logit_scale_maximum": 20.0,
            "modulation_amplitude": 0.5,
            "geometry_bias_bound": 1.0,
        },
        "trainer": {"amp_dtype": "auto"},
        "masking": {"mask_base_seed": 2026082591},
    }


def _tiny_core(alias: str, *, seed: int) -> CohortRelativeQKVCoreBatch:
    generator = torch.Generator().manual_seed(seed)
    edges = torch.tensor(
        [
            [4, 1, 5, 2, 0, 3, 2, 4, 0, 1, 3],
            [2, 0, 4, 1, 3, 1, 0, 3, 2, 4, 0],
        ],
        dtype=torch.long,
    )
    return CohortRelativeQKVCoreBatch(
        alias=alias,
        target_expression=torch.randn(6, 5, generator=generator),
        edge_index=edges,
        relative_geometry=torch.randn(
            edges.shape[1], 70, generator=generator
        ),
        node_covariates=torch.randn(6, 2, generator=generator),
    )


def test_geometry_numerical_gate_uses_post_update_state_on_both_so1_cores() -> None:
    config = _tiny_geometry_config()
    torch.manual_seed(941)
    model = ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer(
        num_genes=5,
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=24,
        decoder_dim=16,
        geometry_hidden_dim=10,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        max_edges_per_chunk=3,
        activation_checkpointing=False,
    )
    with torch.no_grad():
        for block in model.blocks:
            block.geometry_encoder.modulation_projection.weight.normal_(
                mean=0.0, std=1e-3
            )
            block.geometry_encoder.bias_projection.weight.normal_(
                mean=0.0, std=1e-3
            )
            block.raw_attention_logit_scale.add_(0.01)

    result = _PREFLIGHT._geometry_modulated_numerical_checks(
        model,
        (
            _tiny_core("SO1-C06", seed=947),
            _tiny_core("SO1-C11", seed=953),
        ),
        config,
        device=torch.device("cpu"),
    )

    assert result["passed"] is True
    assert result["comparison_core_aliases"] == ["SO1-C06", "SO1-C11"]
    assert result["full_chunk_exactness"]["passed"] is True
    assert result["amp_fp32_equivalence"]["passed"] is True
    assert result["learned_geometry_heads"]["passed"] is True
    assert [record["alias"] for record in result["core_checks"]] == [
        "SO1-C06",
        "SO1-C11",
    ]
    for core in result["core_checks"]:
        assert len(core["learned_geometry_heads"]["blocks"]) == 4
        assert all(
            block["logit_scale_max_abs_change_from_initial"] > 0.0
            for block in core["learned_geometry_heads"]["blocks"]
        )

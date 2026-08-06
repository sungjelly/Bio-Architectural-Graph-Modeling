"""Focused CPU contracts for the frozen hybrid-count archive runner."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.full_core import EDGE_ATTRIBUTE_NAMES
from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/run_hybrid_count_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_hybrid_count_capacity_runner_module", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _component(path: str, section: str) -> dict[str, Any]:
    payload = yaml.safe_load((_ROOT / path).read_text(encoding="utf-8"))
    return dict(payload[section])


def _config(
    model_name: str = "hybrid-count-gat", *, pilot: bool = True
) -> dict[str, Any]:
    uses_graph = model_name == "hybrid-count-gat"
    alias = "ANC-01" if pilot else "ANC-02"
    model_path = (
        "configs/model/hybrid_count_gat.yaml"
        if uses_graph
        else "configs/model/hybrid_count_matched_self.yaml"
    )
    trainer_path = (
        "configs/trainer/full_core_hybrid_resource_pilot_2.yaml"
        if pilot
        else "configs/trainer/full_core_hybrid_fixed_200.yaml"
    )
    evaluation_path = (
        "configs/evaluation/held_in_full_core_hybrid_count_pilot_v1.yaml"
        if pilot
        else "configs/evaluation/held_in_full_core_hybrid_count_v1.yaml"
    )
    model = _component(model_path, "model")
    trainer = _component(trainer_path, "trainer")
    # The explicit optimizer is part of the frozen provenance contract.  Keep
    # this assignment so the test remains diagnostic if a stale component is
    # encountered during concurrent materialization work.
    trainer.setdefault("optimizer", "AdamW")
    trainer["amp_authorization"] = {
        "mode": (
            "same_batch_fp32_amp_equivalence_diagnostic"
            if pilot
            else "require_external_pilot_gate_receipt"
        ),
        "receipt_schema": "hybrid_count_pilot_gate_v1",
        "receipt_reference": (
            "scratch/locked_campaigns/"
            "cmp_20260729_adjacent_normal_10core_hybrid_count_gat/"
            "pilot_gate_receipt.json"
        ),
        "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
    }
    evaluation = _component(evaluation_path, "evaluation")
    metadata_fields = list(_RUNNER._EXPECTED_METADATA_FIELDS)
    graph = {
        "kind": "exact_spatial_knn_radius_guard",
        "neighbor_k": 1000,
        "k": 1000,
        "radius_um": 2000.0,
        "radius_guard_um": 2000.0,
        "symmetry": "mutual",
        "edge_dropout": 0.0,
        "self_loops": False,
        "full_core_graph": True,
        "coordinates_are_node_covariates": False,
        "expected_materialized_graph_sha256": "a" * 64,
        "expected_directed_edges": 1000,
    }
    return {
        "campaign": {"campaign_id": _RUNNER._CAMPAIGN_ID},
        "experiment": {
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
        },
        "model": model,
        "dataset": {
            "dataset_id": f"cosmx_{alias.lower().replace('-', '')}_fit_v1",
            "version": "fit_v1",
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "prepared_artifact_reference": (
                "data/processed/adjacent_normal_10core_qkv_large_k_v1/"
                f"{alias.lower()}/prepared_v1"
            ),
            "task": "masked_expression_hybrid_count",
            "target_scale": (
                "raw_biological_probe_counts_with_per_gene_all_fit_"
                "standardized_log1p"
            ),
            "biological_target_count": 1000,
            "preprocessing_fit_scope": "all_nodes_transductive",
            "validation_or_test_partition_present": False,
            "patient_generalization_supported": False,
            "count_representation": {
                "schema": _RUNNER._REPRESENTATION_SCHEMA,
                "source_scale": "raw_biological_probe_counts",
                "num_output_states": 8,
                "mask_token_id": 8,
                "mask_token_is_output": False,
                "fixed_boundaries": True,
                "fit_required": False,
                "count_mapping": {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "4-7": 4,
                    "8-15": 5,
                    "16-31": 6,
                    "32+": 7,
                },
                "continuous_channel": {
                    "transform": "per_gene_all_fit_standardized_log1p",
                    "masked_value": 0.0,
                },
            },
        },
        "features": {
            "fit_scope": "all_nodes_transductive",
            "use_edge_features": uses_graph,
            "node_expression": {
                "biological_targets": 1000,
                "source_scale": "raw_biological_probe_counts",
                "discrete_transform": "fixed_hybrid_count_states",
                "continuous_transform": (
                    "per_gene_all_fit_standardized_log1p"
                ),
                "masked_discrete_value": "input_only_mask_token_8",
                "masked_continuous_value": 0.0,
                "explicit_mask_authoritative_inside_model": True,
            },
            "node_metadata": {
                "transformed_with": (
                    "full_core_fitted_median_imputation_log1p_"
                    "standardization"
                ),
                "fields": metadata_fields,
            },
            "edge_features": (
                {
                    "fit_scope": "all_retained_directed_edges_transductive",
                    "standardization": "full_core_edge_wise",
                    "fields": list(EDGE_ATTRIBUTE_NAMES),
                }
                if uses_graph
                else []
            ),
            "prohibited_node_inputs": [
                "direct_identifiers",
                "absolute_or_local_coordinates",
                "expression_derived_library_size",
                "rna_derived_qc",
                "vendor_cell_type_cluster_neighborhood_or_niche",
                "hidden_target_values",
            ],
        },
        "graph": graph,
        "masking": {
            "type": "mixed_expression_masking",
            "curriculum": "P+N+B",
            "rate": {
                "partial_gene": 0.2,
                "whole_node": 0.1,
                "spatial_block": 0.1,
            },
            "rates": {
                "partial_gene": 0.2,
                "whole_node": 0.1,
                "spatial_block": 0.1,
            },
            "post_warmup_probabilities": {
                "partial_gene": 0.6,
                "whole_node": 0.3,
                "spatial_block": 0.1,
            },
            "warmup_epochs": 10,
            "block_shape": "disk",
            "block_width_um": None,
            "mask_seed": 314159,
            "mask_expression_only": True,
            "explicit_gene_mask_channel": True,
        },
        "trainer": trainer,
        "evaluation": evaluation,
        "metadata": {
            "locked_config_materialization_receipt": (
                "scratch/locked_campaigns/"
                "cmp_20260729_adjacent_normal_10core_hybrid_count_gat/"
                "locked_config_materialization.json"
            )
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


@pytest.mark.parametrize(
    "model_name",
    ["hybrid-count-gat", "hybrid-count-matched-self"],
)
def test_strict_contract_accepts_only_frozen_alias_safe_pilot(
    model_name: str,
) -> None:
    config = _config(model_name)
    contract = _RUNNER._validate_hybrid_contract(config)
    assert contract.biological_unit_alias == "ANC-01"
    assert contract.diagnostic_resource_pilot is True
    assert contract.uses_graph is (model_name == "hybrid-count-gat")

    wrong_seed = deepcopy(config)
    wrong_seed["seed"] = 1
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="seed=0"):
        _RUNNER._validate_hybrid_contract(wrong_seed)

    wrong_graph = deepcopy(config)
    wrong_graph["graph"]["k"] = 999
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="graph.k"):
        _RUNNER._validate_hybrid_contract(wrong_graph)

    direct_identifier = deepcopy(config)
    direct_identifier["dataset"]["patient_id"] = "prohibited"
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="alias-only"):
        _RUNNER._validate_hybrid_contract(direct_identifier)


def test_contract_rejects_radius_rate_leakage_and_token_drift() -> None:
    config = _config()

    wrong_radius = deepcopy(config)
    wrong_radius["graph"]["radius_guard_um"] = 1999.0
    with pytest.raises(
        _RUNNER.HybridCountRunnerError, match="graph.radius_guard_um"
    ):
        _RUNNER._validate_hybrid_contract(wrong_radius)

    inconsistent_rate_alias = deepcopy(config)
    inconsistent_rate_alias["masking"]["rate"]["whole_node"] = 0.2
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="masking.rate"):
        _RUNNER._validate_hybrid_contract(inconsistent_rate_alias)

    missing_leakage_guard = deepcopy(config)
    missing_leakage_guard["features"]["prohibited_node_inputs"].remove(
        "hidden_target_values"
    )
    with pytest.raises(
        _RUNNER.HybridCountRunnerError, match="leakage prohibition"
    ):
        _RUNNER._validate_hybrid_contract(missing_leakage_guard)

    reverse_mapping = deepcopy(config)
    reverse_mapping["dataset"]["count_representation"]["count_mapping"] = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4-7",
        "5": "8-15",
        "6": "16-31",
        "7": "32+",
    }
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="count_mapping"):
        _RUNNER._validate_hybrid_contract(reverse_mapping)

    wrong_optimizer = deepcopy(config)
    wrong_optimizer["trainer"]["optimizer"] = "SGD"
    with pytest.raises(_RUNNER.HybridCountRunnerError, match="optimizer"):
        _RUNNER._validate_hybrid_contract(wrong_optimizer)


def test_contract_accepts_retry_attempt_two_and_rejects_attempt_three() -> None:
    retry = _config()
    retry["attempt"] = 2
    contract = _RUNNER._validate_hybrid_contract(retry)
    assert contract.biological_unit_alias == "ANC-01"

    exhausted = deepcopy(retry)
    exhausted["attempt"] = 3
    with pytest.raises(
        _RUNNER.HybridCountRunnerError,
        match=r"attempt in \{1,2\}",
    ):
        _RUNNER._validate_hybrid_contract(exhausted)


def _production_receipt_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], Any]:
    locked = tmp_path / _RUNNER._PILOT_RECEIPT_RELATIVE.parent
    locked.mkdir(parents=True)
    materialization = {
        "schema_version": 1,
        "receipt_kind": "hybrid_count_locked_config_materialization_v1",
        "campaign_id": _RUNNER._CAMPAIGN_ID,
    }
    materialization["checksum"] = canonical_sha256(materialization)
    (locked / "locked_config_materialization.json").write_text(
        json.dumps(materialization), encoding="utf-8"
    )
    gate = {
        "schema_version": 1,
        "receipt_kind": "hybrid_count_pilot_gate_v1",
        "campaign_id": _RUNNER._CAMPAIGN_ID,
        "materialization_checksum": materialization["checksum"],
        "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
        "thresholds": dict(_RUNNER._PILOT_GATE_THRESHOLDS),
        "same_frozen_precision_batch": True,
        "same_evaluation_masks": True,
        "same_verified_graph": True,
        "failure_reasons": [],
        "gate_passed": True,
        "production_authorized": True,
        "jobs": [
            {
                "alias": "ANC-01",
                "arm": arm,
                "verified_bundle": True,
                "finite_losses_and_gradients": True,
                "parameter_match": True,
                "parameter_count": _RUNNER._EXPECTED_PARAMETER_COUNT,
                "precision_equivalence_passed": True,
                "peak_vram_passed": True,
                "projected_runtime_passed": True,
                "runner_pilot_gate_passed": True,
            }
            for arm in ("hybrid-gat-k1000", "hybrid-matched-self")
        ],
    }
    gate["checksum"] = canonical_sha256(gate)
    (locked / "pilot_gate_receipt.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    config = _config(pilot=False)
    contract = _RUNNER._validate_hybrid_contract(config)
    return locked, materialization, gate, (contract, config)


def _write_resigned_gate(path: Path, gate: dict[str, Any]) -> None:
    payload = deepcopy(gate)
    payload.pop("checksum", None)
    payload["checksum"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_production_receipt_matches_enqueuer_schema(tmp_path: Path) -> None:
    _, materialization, _, context = _production_receipt_fixture(tmp_path)
    contract, config = context
    record = _RUNNER._validate_production_pilot_receipt(
        tmp_path, contract, config
    )
    assert record is not None
    assert record["passed"] is True
    assert record["materialization_checksum"] == materialization["checksum"]


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("receipt", "gate_passed", False),
        ("receipt", "production_authorized", False),
        (
            "receipt",
            "thresholds",
            {
                **_RUNNER._PILOT_GATE_THRESHOLDS,
                "peak_allocated_vram_gib_maximum": 20.6,
            },
        ),
        (
            "receipt",
            "thresholds",
            {
                **_RUNNER._PILOT_GATE_THRESHOLDS,
                "unfrozen_threshold": 1,
            },
        ),
        ("receipt", "same_frozen_precision_batch", False),
        ("receipt", "same_evaluation_masks", False),
        ("receipt", "same_verified_graph", False),
        (
            "receipt",
            "failure_reasons",
            ["hybrid-gat-k1000:synthetic_failure"],
        ),
        ("job", "verified_bundle", False),
        ("job", "finite_losses_and_gradients", False),
        ("job", "parameter_match", False),
        ("job", "parameter_count", 11_674_881),
        ("job", "precision_equivalence_passed", False),
        ("job", "peak_vram_passed", False),
        ("job", "projected_runtime_passed", False),
        ("job", "runner_pilot_gate_passed", False),
    ],
)
def test_production_receipt_rejects_resigned_evidence_tampering(
    tmp_path: Path,
    location: str,
    field: str,
    value: Any,
) -> None:
    locked, _, gate, context = _production_receipt_fixture(tmp_path)
    contract, config = context
    if location == "receipt":
        gate[field] = value
    else:
        gate["jobs"][0][field] = value
    _write_resigned_gate(locked / "pilot_gate_receipt.json", gate)

    with pytest.raises(_RUNNER.HybridCountRunnerError, match="pilot"):
        _RUNNER._validate_production_pilot_receipt(tmp_path, contract, config)


@pytest.mark.parametrize("mutation", ["missing_job", "duplicate_arm"])
def test_production_receipt_requires_exact_two_pilot_jobs(
    tmp_path: Path,
    mutation: str,
) -> None:
    locked, _, gate, context = _production_receipt_fixture(tmp_path)
    contract, config = context
    if mutation == "missing_job":
        gate["jobs"].pop()
    else:
        gate["jobs"][1]["arm"] = gate["jobs"][0]["arm"]
    _write_resigned_gate(locked / "pilot_gate_receipt.json", gate)

    with pytest.raises(_RUNNER.HybridCountRunnerError, match="pilot"):
        _RUNNER._validate_production_pilot_receipt(tmp_path, contract, config)


def test_paired_model_audit_uses_exact_frozen_budget_and_json_safe_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: dict[str, torch.nn.Module] = {}
    graph_class = _RUNNER.HybridReceiverChunkedEdgeConditionedGATv2
    self_class = _RUNNER.HybridEdgeParameterMatchedSelfControl

    def graph_factory(**kwargs: Any) -> torch.nn.Module:
        constructed["graph"] = graph_class(**kwargs)
        return constructed["graph"]

    def self_factory(**kwargs: Any) -> torch.nn.Module:
        constructed["self"] = self_class(**kwargs)
        return constructed["self"]

    monkeypatch.setattr(
        _RUNNER,
        "HybridReceiverChunkedEdgeConditionedGATv2",
        graph_factory,
    )
    monkeypatch.setattr(
        _RUNNER,
        "HybridEdgeParameterMatchedSelfControl",
        self_factory,
    )
    core = SimpleNamespace(
        n_genes=1000,
        expression_mean=np.zeros(1000, dtype=np.float32),
        expression_scale=np.ones(1000, dtype=np.float32),
        node_covariates=np.zeros((2, 22), dtype=np.float32),
    )
    graph = SimpleNamespace(edge_attribute_names=EDGE_ATTRIBUTE_NAMES)
    model_config = _config()["model"]
    model, construction, audit = _RUNNER._paired_models(
        core=core,
        graph=graph,
        model_config=model_config,
        selected_model_name="hybrid-count-gat",
        seed=0,
    )
    assert audit["trainable_parameter_count_graph"] == 11_674_880
    assert audit["trainable_parameter_count_self"] == 11_674_880
    assert audit["encoder_parameter_shapes_identical"] is True
    assert audit["decoder_parameter_shapes_identical"] is True
    assert audit["encoder_initial_state_bit_identical"] is True
    assert audit["decoder_initial_state_bit_identical"] is True
    for component in ("encoder", "decoder"):
        graph_state = getattr(constructed["graph"], component).state_dict()
        self_state = getattr(constructed["self"], component).state_dict()
        assert list(graph_state) == list(self_state)
        for name in graph_state:
            torch.testing.assert_close(
                graph_state[name],
                self_state[name],
                rtol=0,
                atol=0,
            )
    assert audit["graph_layer_parameter_budgets"] == audit[
        "self_layer_parameter_budgets"
    ]
    assert len(model.state_dict()) == 52
    json.dumps(construction, allow_nan=False)


def test_matched_self_output_is_invariant_to_topology_and_edge_attributes() -> None:
    model = _RUNNER.HybridEdgeParameterMatchedSelfControl(
        num_genes=3,
        edge_attribute_dim=17,
        expression_mean=np.zeros(3, dtype=np.float32),
        expression_scale=np.ones(3, dtype=np.float32),
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        graph_layers=2,
        ffn_dim=15,
        decoder_dim=11,
        edge_hidden_dim=7,
        edge_embedding_dim=6,
        dropout=0.0,
        attention_dropout=0.0,
    ).eval()
    counts = torch.tensor(
        [[0.0, 1.0, 4.0], [2.0, 3.0, 8.0], [1.0, 7.0, 16.0]]
    )
    mask = torch.tensor(
        [[True, False, False], [False, True, False], [False, False, True]]
    )
    covariates = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-0.5, 0.5]])
    first_edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    second_edges = torch.tensor([[0, 2], [2, 1]])
    first_attributes = torch.randn(3, 17)
    second_attributes = torch.randn(2, 17) * 100.0
    with torch.no_grad():
        first = model(
            counts,
            mask,
            edge_index=first_edges,
            edge_attributes=first_attributes,
            node_covariates=covariates,
        ).prediction
        second = model(
            counts,
            mask,
            edge_index=second_edges,
            edge_attributes=second_attributes,
            node_covariates=covariates,
        ).prediction
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_metric_rows_preserve_null_precision_and_state_vectors() -> None:
    metrics = {
        "n_masked": 8,
        "hybrid_loss": 1.2,
        "detection_precision": None,
        "state8_support": [1] * 8,
        "state8_recall": [1.0, None, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5],
        "collapsed4_support": [2] * 4,
        "collapsed4_recall": [1.0, None, 0.0, 0.5],
    }
    entry = {
        "spec": {"mode": "node"},
        "replicate": 0,
        "entry_id": "fit-whole-node-r0",
        "seed": 7,
        "mask_checksum": "b" * 64,
    }
    row = _RUNNER._replicate_metric_row(entry=entry, metrics=metrics)
    assert row["detection_precision"] is None
    assert row["state8_recall"][1] is None
    rows = [
        {**row, "mask_mode": mode, "mask_replicate": replicate}
        for mode in _RUNNER._REQUIRED_PUBLIC_MASKS
        for replicate in range(3)
    ]
    final = _RUNNER._final_metrics(
        replicate_rows=rows,
        training=SimpleNamespace(final_epoch=199, final_train_loss=0.7),
        training_duration=1.0,
        evaluation_duration=2.0,
        total_duration=3.0,
        parameter_count=11_674_880,
        checkpoint_size=10,
        replicates_per_mode=3,
        graph_duration=4.0,
        data_duration=5.0,
        peak_vram_bytes=6,
        projected_runtime_hours=None,
    )
    assert final["fit/whole_node/detection_precision"] is None
    assert final["fit/whole_node/hybrid_loss"] == pytest.approx(1.2)

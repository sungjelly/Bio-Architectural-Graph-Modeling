from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.qkv_graph_transformer import (
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "analysis"
    / "analyze_full_core_qkv_interpretability.py"
)


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "analyze_full_core_qkv_interpretability",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_gate(
    path: Path,
    *,
    passes: bool,
    run_ids: list[str],
    evaluated_run_ids: list[str] | None = None,
) -> None:
    evaluated = (
        list(run_ids)
        if evaluated_run_ids is None
        else list(evaluated_run_ids)
    )
    path.write_text(
        json.dumps(
            {
                "campaign_id": (
                    "cmp_20260726_full_core_qkv_large_k"
                ),
                "evaluated_graph_run_ids": evaluated,
                "representation_gate": {
                    "passes": passes,
                    "eligible_graph_run_ids": run_ids,
                }
            }
        ),
        encoding="utf-8",
    )


def test_capacity_gate_is_bound_and_negative_override_is_diagnostic(
    tmp_path: Path,
) -> None:
    module = _load_script()
    path = tmp_path / "comparison.json"
    _write_gate(path, passes=True, run_ids=["run-a", "run-b"])

    positive = module._capacity_gate(
        path,
        run_id="run-a",
        allow_negative=False,
    )
    assert positive["passes"] is True
    assert positive["diagnostic_only"] is False
    assert len(positive["report_sha256"]) == 64

    _write_gate(
        path,
        passes=True,
        run_ids=["run-a"],
        evaluated_run_ids=["run-a", "run-b"],
    )
    with pytest.raises(
        module.QKVInterpretabilityError,
        match="individual graph-vs-self",
    ):
        module._capacity_gate(
            path,
            run_id="run-b",
            allow_negative=False,
        )

    _write_gate(
        path,
        passes=False,
        run_ids=[],
        evaluated_run_ids=["run-a", "run-b"],
    )
    with pytest.raises(
        module.QKVInterpretabilityError,
        match="representation gate failed",
    ):
        module._capacity_gate(
            path,
            run_id="run-a",
            allow_negative=False,
        )
    diagnostic = module._capacity_gate(
        path,
        run_id="run-a",
        allow_negative=True,
    )
    assert diagnostic["passes"] is False
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["negative_gate_override_used"] is True

    with pytest.raises(
        module.QKVInterpretabilityError,
        match="not bound",
    ):
        module._capacity_gate(
            path,
            run_id="another-run",
            allow_negative=True,
        )


def test_locked_run_contract_accepts_repository_qkv_full_config() -> None:
    module = _load_script()
    config = compose_config(
        PROJECT_ROOT / "configs/experiment/full_core_qkv_k1000.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    root = Path("/tmp/r_qkv_contract")
    manifest = {"campaign_id": module.CAMPAIGN_ID}
    summary = {
        "run_id": root.name,
        "status": "success",
        "training_exit_status": "success",
        "diagnostic_resource_pilot": False,
        "conclusion_eligible": True,
        "generalization_estimate": False,
        "model_name": "qkv-gat",
        "fixed_epoch_budget": 300,
        "final_epoch": 299,
        "checkpoint_role": "last",
        "evaluation_protocol": module.PROTOCOL,
        "canonical_prediction_selection": {
            "split": "fit",
            "mask_mode": "whole_node",
            "mask_replicate": 0,
        },
    }
    training = {
        "training_protocol": module.PROTOCOL,
        "final_epoch": 299,
        "fixed_epoch_budget": 300,
        "model_construction": {
            "canonical_model_key": "qkvgat",
            "implementation_class": module.IMPLEMENTATION_CLASS,
        },
    }

    module._validate_run_contract(
        root,
        config,
        manifest,
        summary,
        training,
    )


def test_final_layer_recomputation_matches_production_qkv_forward() -> None:
    module = _load_script()
    torch.manual_seed(9)
    model = ReceiverChunkedEdgeAwareQKVGraphTransformer(
        num_genes=5,
        edge_attribute_dim=3,
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        attention_head_dim=4,
        graph_layers=2,
        ffn_dim=20,
        decoder_dim=9,
        edge_hidden_dim=7,
        edge_embedding_dim=6,
        edge_conditioning_mode="bias_gate",
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        max_edges_per_chunk=6,
        activation_checkpointing=False,
    ).eval()
    expression = torch.randn(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[1] = True
    covariates = torch.randn(4, 2)
    # Receiver-sorted complete directed graph without self loops.
    pairs = [
        (source, receiver)
        for receiver in range(4)
        for source in range(4)
        if source != receiver
    ]
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    edge_attributes = torch.randn(len(pairs), 3)
    selected = torch.arange(4, dtype=torch.long)

    with torch.no_grad():
        penultimate = module._compute_penultimate(
            model,
            expression,
            mask,
            covariates,
            edge_index,
            edge_attributes,
            amp=False,
            amp_dtype="auto",
        )
        components = module._final_layer_components(
            model,
            penultimate,
            edge_index,
            edge_attributes,
            selected,
        )
        reconstructed, recomputed_attention, _ = (
            module._readout_with_deleted_edges(
                model,
                components,
            )
        )
        production = model(
            expression,
            mask,
            edge_index,
            edge_attributes,
            covariates,
            return_explanations=True,
            target_nodes=selected,
            attention_receivers=selected,
        )

    assert torch.allclose(
        reconstructed,
        production.prediction,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        components["attention"],
        production.attention_weights,
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(
        recomputed_attention,
        components["attention"],
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.all(components["value_gate"] > 0)
    assert torch.all(components["value_gate"] < 2)
    local_receiver = components["local_receiver"]
    sums = torch.zeros(4, 3)
    sums.index_add_(0, local_receiver, components["attention"])
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_effective_routing_is_attention_times_gate_then_renormalized() -> None:
    module = _load_script()
    attention = np.asarray(
        [
            [0.8, 0.6],
            [0.2, 0.4],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    gate = np.asarray(
        [
            [0.25, 1.0],
            [2.0, 1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ],
        dtype=np.float64,
    )
    receiver = np.asarray([0, 0, 1, 1], dtype=np.int64)
    effective, mass = module._normalize_effective(
        attention * gate,
        receiver,
        2,
    )

    assert np.allclose(mass, [[0.6, 1.0], [1.0, 1.0]])
    assert effective[:2, 0] == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    sums = np.zeros((2, 2), dtype=np.float64)
    np.add.at(sums, receiver, effective)
    assert np.allclose(sums, np.ones((2, 2)))

    summary = module._routing_summary(
        receiver,
        effective,
        np.asarray([10.0, 30.0, 20.0, 40.0]),
        receiver_count=2,
    )
    assert summary["receiver_count"] == 2
    assert summary["head_count"] == 2
    assert summary["maximum_normalization_error"] < 1e-12
    assert summary["incoming_degree"]["mean"] == 2.0

    zero_distribution, zero_mass = module._normalize_effective(
        np.zeros((4, 2), dtype=np.float64),
        receiver,
        2,
    )
    assert np.array_equal(zero_distribution, np.zeros((4, 2)))
    assert np.array_equal(zero_mass, np.zeros((2, 2)))
    zero_summary = module._routing_summary(
        receiver,
        zero_distribution,
        np.asarray([10.0, 30.0, 20.0, 40.0]),
        receiver_count=2,
    )
    assert zero_summary["zero_mass_receiver_heads"] == 4
    assert zero_summary["effective_neighbor_count_per_receiver_head"] is None


def test_deletion_plan_ranks_effective_routing_and_matches_distance_bins() -> None:
    module = _load_script()
    receiver = np.repeat(np.arange(2, dtype=np.int64), 20)
    distance = np.tile(np.arange(1, 21, dtype=np.float64), 2)
    routing = np.zeros(40, dtype=np.float64)
    routing[[3, 7, 25, 29]] = [0.9, 0.8, 0.95, 0.85]

    first = module._distance_matched_deletion_plan(
        receiver,
        routing,
        distance,
        receiver_count=2,
        fraction=0.10,
        null_replicates=4,
        namespace="unit-test",
    )
    second = module._distance_matched_deletion_plan(
        receiver,
        routing,
        distance,
        receiver_count=2,
        fraction=0.10,
        null_replicates=4,
        namespace="unit-test",
    )

    assert np.array_equal(first["top_positions"], [3, 7, 25, 29])
    assert np.array_equal(
        first["top_positions"],
        second["top_positions"],
    )
    assert len(first["null_positions"]) == 4
    for left, right in zip(
        first["null_positions"],
        second["null_positions"],
        strict=True,
    ):
        assert np.array_equal(left, right)
        assert len(left) == len(first["top_positions"])
        assert not np.intersect1d(left, first["top_positions"]).size
        assert np.bincount(receiver[left], minlength=2).tolist() == [2, 2]


def test_prediction_change_reports_faithfulness_not_accuracy() -> None:
    module = _load_script()
    target = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    baseline = target + 0.1
    perturbed = target + 0.3

    result = module._prediction_change(
        baseline,
        perturbed,
        target,
        huber_delta=1.0,
    )

    assert result["mean_absolute_prediction_change"] == pytest.approx(0.2)
    assert result["huber_change"] > 0
    assert result["mae_change"] > 0
    assert "accuracy" not in result


def test_tls_annotation_is_optional_and_never_used_for_selection() -> None:
    module = _load_script()
    unavailable = module._tls_annotation(
        ("GENE_A", "GENE_B"),
        np.zeros((3, 2), dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([[0.4], [0.6]], dtype=np.float64),
    )
    assert unavailable["status"] == "unavailable_due_panel_coverage"
    assert unavailable["exploratory_annotation_only"] is True
    assert unavailable["selection_used_tls_expression"] is False

    genes = ("CCL19", "CCR7", "OTHER")
    expression = np.asarray(
        [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    partial = module._tls_annotation(
        genes,
        expression,
        np.asarray([0], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
    )
    assert partial["status"] == "computed_partial_panel"
    assert partial["selection_used_tls_expression"] is False
    assert partial[
        "effective_routing_weighted_sender_organizer_score"
    ]["mean"] == pytest.approx(2.5)


def test_aggregate_output_guard_rejects_row_level_identifiers() -> None:
    module = _load_script()
    module._assert_aggregate_only(
        {
            "provenance": {"run_id": "run"},
            "summary": {"values": [1.0, 2.0]},
        }
    )
    with pytest.raises(
        module.QKVInterpretabilityError,
        match="forbidden key",
    ):
        module._assert_aggregate_only(
            {"routing": {"receiver_indices": [1, 2]}}
        )

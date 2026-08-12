from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.matched_graph_context import (
    ARMS,
    BASE_FEATURE_COUNT,
    CANDIDATES,
    CONTRACT_SHA256,
    EVALUATION_EPOCHS,
    FAITHFULNESS_CONTEXTS,
    MatchedGraphContextError,
    MetricAccumulator,
    PreprocessingState,
    ScaleState,
    StreamingMoments,
    build_matched_model,
    deterministic_synthetic_gate,
    frozen_rank27_projection,
    no_graph_context,
    run_synthetic_recovery_gates,
    selection_payload_sha256,
    short_edge_removed_context,
    split_roles,
    trainable_parameter_count,
    validate_selection_receipt,
)


def test_all_arms_have_identical_literal_architecture_and_parameter_count() -> None:
    state_keys = None
    counts = set()
    for arm in ARMS:
        model = build_matched_model(
            arm, hidden_width=64, dropout=0.1, seed=20260812
        )
        counts.add(trainable_parameter_count(model))
        keys = tuple(model.state_dict())
        state_keys = keys if state_keys is None else state_keys
        assert keys == state_keys
    assert len(counts) == 1
    # base: 27*1000+1000; linear context: 1000*1000;
    # hidden: 1000*64+64; output: 64*1000.
    assert counts == {1_156_064}


def test_frozen_no_graph_projection_is_deterministic_rank_27() -> None:
    first = frozen_rank27_projection()
    second = frozen_rank27_projection()
    assert first.shape == (BASE_FEATURE_COUNT, 1000)
    assert np.array_equal(first, second)
    assert np.linalg.matrix_rank(first.astype(np.float64)) == BASE_FEATURE_COUNT
    base = np.eye(BASE_FEATURE_COUNT, dtype=np.float32)
    assert np.array_equal(no_graph_context(base, first), first)


def test_split_roles_exclude_outer_outcomes_from_tuning() -> None:
    for outer in range(4):
        roles = split_roles("tune", outer)
        assert roles["validation_fold"] == (outer + 1) % 4
        assert outer not in roles["train_folds"]
        assert roles["validation_fold"] not in roles["train_folds"]
        assert "test_fold" not in roles
        confirm = split_roles("confirm", outer)
        assert set(confirm["train_folds"]) == set(range(4)) - {outer}
        assert confirm["test_fold"] == outer


def test_streaming_normalization_uses_only_supplied_training_values() -> None:
    fit = StreamingMoments(2)
    fit.update(np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32))
    state = fit.finish()
    # A wildly shifted held-out population is transformed, never fit.
    held_out = np.array([[1001.0, 2002.0]], dtype=np.float32)
    transformed = state.transform(held_out)
    assert state.count == 2
    assert np.array_equal(state.mean, np.array([2.0, 4.0], dtype=np.float32))
    assert np.array_equal(state.scale, np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(transformed, np.array([[999.0, 999.0]], dtype=np.float32))


def test_short_edge_context_filters_only_frozen_csr_and_zero_fills_empty() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [25.0, 0.0], [30.0, 0.0]]
    )
    # Receiver 0 has frozen endpoints at 5, 10, and 25 um; receiver 4 only a
    # 5-um endpoint.  Cell 4 is never added to receiver 0 despite being 30 um.
    indptr = np.array([0, 3, 3, 3, 3, 4], dtype=np.int64)
    indices = np.array([1, 2, 3, 3], dtype=np.int32)
    expression = np.arange(10, dtype=np.float32).reshape(5, 2)
    context, degree = short_edge_removed_context(
        receiver_indices=np.array([0, 4]),
        coordinates=coordinates,
        indptr=indptr,
        indices=indices,
        source_expression=expression,
    )
    assert degree.tolist() == [2, 0]
    assert np.array_equal(context[0], expression[[2, 3]].mean(axis=0))
    assert np.array_equal(context[1], np.zeros(2, dtype=np.float32))


def test_metric_accumulator_reports_all_genes_and_component_equal_metrics() -> None:
    accumulator = MetricAccumulator(genes=2)
    accumulator.update(
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
        slide="S1",
        components=np.array([1, 2]),
    )
    rows = accumulator.component_rows()
    assert [row["mse"] for row in rows] == [1.0, 9.0]
    assert accumulator.component_equal() == (5.0, 2.0)
    genes = accumulator.gene_rows(("G0", "G1"))
    assert len(genes) == 2
    assert {row["gene_index"] for row in genes} == {0, 1}
    assert all(np.isfinite(row["pearson"]) for row in genes)
    component_genes = accumulator.component_gene_rows(("G0", "G1"))
    assert len(component_genes) == 4
    assert {
        (row["slide"], row["component"], row["gene_index"])
        for row in component_genes
    } == {("S1", component, gene) for component in (1, 2) for gene in (0, 1)}
    assert [row["mse"] for row in component_genes] == [1.0, 1.0, 9.0, 9.0]


def test_synthetic_positive_and_null_gates_are_deterministic_and_pass() -> None:
    first = run_synthetic_recovery_gates()
    second = run_synthetic_recovery_gates()
    assert first == second
    assert first["passed"]
    assert first["positive"]["relative_gain"] >= 0.05
    assert abs(first["null"]["relative_gain"]) <= 0.01
    assert first["result_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "result_sha256"}
    )


def test_synthetic_null_exact_regression() -> None:
    result = deterministic_synthetic_gate(
        seed=20260812, planted_context=False, samples=3072
    )
    assert result["passed"] is True
    assert result["relative_gain"] == pytest.approx(
        -0.0037968980762839847, rel=0.0, abs=1.0e-12
    )


def _selection_payload(*, shared: bool = True) -> dict[str, object]:
    candidates = ("c00", "c00", "c00", "c00") if shared else ("c00", "c05", "c00", "c00")
    selected_by_fold = {}
    for fold in range(4):
        selected = {}
        for arm, candidate_id in zip(ARMS, candidates, strict=True):
            value = dict(CANDIDATES[candidate_id])
            value["epoch"] = EVALUATION_EPOCHS[-1]
            value["batch_size"] = 4096
            parameter_count = trainable_parameter_count(
                build_matched_model(
                    arm,
                    hidden_width=value["hidden_width"],
                    dropout=value["dropout"],
                    seed=0,
                )
            )
            selected[arm] = {
                "candidate_id": candidate_id,
                "config": value,
                "config_sha256": canonical_sha256(value),
                "parameter_count": parameter_count,
            }
        selected_by_fold[str(fold)] = selected
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "matched_graph_context_selection_receipt",
        "campaign_id": "cmp_20260812_matched_graph_context_nested_cv",
        "contract_sha256": CONTRACT_SHA256,
        "status": "frozen",
        "test_metrics_used_for_selection": False,
        "cross_outer_pooling": False,
        "source_tuning_result_sha256s_by_outer_fold": {
            str(fold): [
                canonical_sha256({"fold": fold, "job": job}) for job in range(88)
            ]
            for fold in range(4)
        },
        "selected_by_outer_fold": selected_by_fold,
    }
    payload["payload_sha256"] = selection_payload_sha256(payload)
    return payload


def test_selection_receipt_enforces_shared_hidden_parameter_match(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(_selection_payload()), encoding="utf-8")
    payload, config = validate_selection_receipt(
        path, arm="observed_near", outer_fold=2
    )
    assert payload["payload_sha256"]
    assert config["hidden_width"] == 32

    path.write_text(json.dumps(_selection_payload(shared=False)), encoding="utf-8")
    with pytest.raises(MatchedGraphContextError, match="shared-hidden"):
        validate_selection_receipt(path, arm="observed_near", outer_fold=2)


def test_selection_receipt_rejects_missing_outer_fold_authority(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    payload = _selection_payload()
    del payload["selected_by_outer_fold"]["1"]  # type: ignore[index]
    payload["payload_sha256"] = selection_payload_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatchedGraphContextError, match="four independent"):
        validate_selection_receipt(path, arm="observed_near", outer_fold=1)


def test_faithfulness_context_contract_is_complete() -> None:
    assert FAITHFULNESS_CONTEXTS == (
        "native",
        "zero",
        "permuted_near",
        "observed_annular",
        "observed_near_10_25",
    )


def test_checkpoint_serialization_replays_forward_predictions() -> None:
    runner_path = (
        Path(__file__).resolve().parents[3]
        / "scripts/train/run_matched_graph_context.py"
    )
    spec = importlib.util.spec_from_file_location("matched_runner_for_test", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    model = build_matched_model(
        "observed_near", hidden_width=32, dropout=0.1, seed=20260812
    )
    two = ScaleState(
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        count=3,
    )
    base = ScaleState(
        mean=np.zeros(27, dtype=np.float32),
        scale=np.ones(27, dtype=np.float32),
        count=3,
    )
    genes = ScaleState(
        mean=np.zeros(1000, dtype=np.float32),
        scale=np.ones(1000, dtype=np.float32),
        count=3,
    )
    preprocessing = PreprocessingState(
        coordinate={"SO_1": two, "SO_2": two},
        base=base,
        target=genes,
        context=genes,
        projection_sha256="a" * 64,
        training_folds=(0, 1, 2),
    )
    content, state_hash, replay = runner._checkpoint_bytes(
        model,
        config={
            "candidate_id": "c00",
            "hidden_width": 32,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "epoch": 12,
            "batch_size": 4096,
        },
        preprocessing=preprocessing,
        selection_receipt={"payload_sha256": "b" * 64},
    )
    assert content
    assert len(state_hash) == 64
    assert replay["status"] == "passed"
    assert replay["finite"] is True
    assert replay["bitwise_equal"] is True

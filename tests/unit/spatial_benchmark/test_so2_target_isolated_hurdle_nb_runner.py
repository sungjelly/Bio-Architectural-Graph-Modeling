"""Pure contracts for the dedicated SO2 target-isolated hurdle-NB runner."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest
import torch

from scripts.train import run_so2_target_isolated_nb as base_runner
from spatial_benchmark.so2_nb_training import EarlyStoppingState


_PATCHED_BASE_NAMES = (
    "CAMPAIGN_ID",
    "MODEL_NAME",
    "MODEL_FAMILY",
    "EVALUATION_PROTOCOL",
    "PROTOCOL",
    "PREFLIGHT_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "EPOCH_METRICS_SCHEMA",
    "PER_CORE_METRICS_SCHEMA",
    "EXPECTED_PARAMETER_COUNT",
    "FROZEN_TASK_CONTRACT_SHA256",
    "SOURCE_EXPERIMENT_CONFIG",
    "FROZEN_TASK_CONTRACT",
    "PREFLIGHT_CODE_HASH_SCOPE",
    "PREFLIGHT_CODE_RELATIVE_PATHS",
    "PREFLIGHT_REQUIRED_GATES",
    "PRIMARY_METRIC",
    "TRAIN_PRIMARY_METRIC",
    "PROVENANCE_RELATIVE_PATH",
    "OBJECTIVE_AGGREGATION",
    "EPOCH_COLUMNS",
    "PER_CORE_COLUMNS",
    "validate_target_isolated_config",
    "_build_model",
    "_forward_target_shard",
    "_paired_update",
    "ValidationResult",
    "_validation_sum_tensor",
    "_metrics_from_sums",
    "_evaluate_validation",
    "_aggregate_training_epoch",
    "_make_epoch_row",
    "_per_core_rows",
    "_archive_history_rows",
    "_validation_prediction_rows",
    "_final_metrics",
    "_additional_provenance",
)
_MISSING = object()


@pytest.fixture(scope="module")
def runner() -> Iterator[ModuleType]:
    """Import the patching runner without contaminating later base-runner tests."""

    originals = {
        name: getattr(base_runner, name, _MISSING) for name in _PATCHED_BASE_NAMES
    }
    module = importlib.import_module(
        "scripts.train.run_so2_target_isolated_hurdle_nb"
    )
    module._install_hurdle_contract()
    try:
        yield module
    finally:
        for name, value in originals.items():
            if value is _MISSING:
                delattr(base_runner, name)
            else:
                setattr(base_runner, name, value)


def _validation_core_metrics(scale: float = 1.0) -> dict[str, float | int]:
    return {
        "target_cells": int(2 * scale),
        "masked_entries": int(10 * scale),
        "positive_entries": int(4 * scale),
        "pooled_masked_hurdle_negative_binomial_nll": 2.0 * scale,
        "masked_detection_bce": 0.4 * scale,
        "positive_zero_truncated_negative_binomial_nll": 1.5 * scale,
        "masked_raw_count_mae": 1.1 * scale,
        "masked_raw_count_rmse": 1.2 * scale,
        "masked_log1p_mae": 0.5 * scale,
        "masked_log1p_rmse": 0.6 * scale,
        "masked_poisson_deviance": 1.3 * scale,
        "positive_raw_count_mae": 1.4 * scale,
        "positive_raw_count_rmse": 1.5 * scale,
        "positive_log1p_mae": 0.7 * scale,
        "positive_log1p_rmse": 0.8 * scale,
        "observed_positive_rate": 0.4,
        "predicted_positive_probability_mean": 0.35,
        "detection_brier_score": 0.2,
        "detection_true_positive": int(3 * scale),
        "detection_true_negative": int(5 * scale),
        "detection_false_positive": int(1 * scale),
        "detection_false_negative": int(1 * scale),
        "detection_recall": 0.75,
        "detection_specificity": 5.0 / 6.0,
        "detection_precision": 0.75,
        "detection_f1": 0.75,
        "observed_count_mean": 2.2 * scale,
        "predicted_count_mean": 2.0 * scale,
    }


def _validation_result(runner: ModuleType) -> Any:
    per_core = {
        alias: _validation_core_metrics(float(index + 1))
        for index, alias in enumerate(base_runner.VALIDATION_ALIASES)
    }
    receipts = {
        alias: {"receipt_sha256": str(index + 1) * 64}
        for index, alias in enumerate(base_runner.VALIDATION_ALIASES)
    }
    return runner.ValidationResult(
        pooled_hurdle_negative_binomial_nll=2.5,
        detection_bce=0.45,
        positive_count_nll=1.6,
        raw_count_mae=1.1,
        raw_count_rmse=1.2,
        log1p_mae=0.5,
        log1p_rmse=0.6,
        poisson_deviance=1.3,
        positive_raw_count_mae=1.4,
        positive_raw_count_rmse=1.5,
        positive_log1p_mae=0.7,
        positive_log1p_rmse=0.8,
        observed_positive_rate=0.4,
        predicted_positive_probability_mean=0.35,
        detection_brier_score=0.2,
        detection_true_positive=9,
        detection_true_negative=15,
        detection_false_positive=3,
        detection_false_negative=3,
        detection_recall=0.75,
        detection_specificity=5.0 / 6.0,
        detection_precision=0.75,
        detection_f1=0.75,
        masked_entries=30,
        positive_entries=12,
        target_cells=6,
        observed_count_mean=2.2,
        predicted_count_mean=2.0,
        per_core=per_core,
        coverage_receipts=receipts,
    )


def _training_rows() -> dict[str, Mapping[str, float | int]]:
    return {
        alias: {
            "target_cells": 1,
            "masked_entries": 10,
            "positive_entries": 2,
            "joint_nll_sum": 20.0 + index,
            "detection_nll_sum": 4.0 + index,
            "positive_count_nll_sum": 6.0 + index,
            "pooled_masked_hurdle_negative_binomial_nll": 2.0 + index / 10.0,
            "masked_detection_bce": 0.4 + index / 10.0,
            "positive_zero_truncated_negative_binomial_nll": 3.0 + index / 2.0,
            "observed_positive_rate": 0.2,
        }
        for index, alias in enumerate(base_runner.TRAINING_ALIASES)
    }


def test_import_installs_the_complete_hurdle_contract(runner: ModuleType) -> None:
    constant_names = _PATCHED_BASE_NAMES[
        : _PATCHED_BASE_NAMES.index("validate_target_isolated_config")
    ]
    for name in constant_names:
        assert getattr(base_runner, name) == getattr(runner, name)

    expected_functions = {
        "validate_target_isolated_config": runner._require_config_values,
        "_build_model": runner._build_model,
        "_forward_target_shard": runner._forward_target_shard,
        "_paired_update": runner._paired_update,
        "ValidationResult": runner.ValidationResult,
        "_validation_sum_tensor": runner._validation_sum_tensor,
        "_metrics_from_sums": runner._metrics_from_sums,
        "_evaluate_validation": runner._evaluate_validation,
        "_aggregate_training_epoch": runner._aggregate_training_epoch,
        "_make_epoch_row": runner._make_epoch_row,
        "_per_core_rows": runner._per_core_rows,
        "_archive_history_rows": runner._archive_history_rows,
        "_validation_prediction_rows": runner._validation_prediction_rows,
        "_final_metrics": runner._final_metrics,
        "_additional_provenance": runner._additional_provenance,
    }
    for name, expected in expected_functions.items():
        assert getattr(base_runner, name) is expected


def test_validation_tensor_and_reduction_preserve_confusion_metrics(
    runner: ModuleType,
) -> None:
    probability = torch.tensor([[0.4, 0.5], [0.9, 0.8]])
    conditional_mean = torch.tensor([[1.2, 1.5], [2.5, 1.1]])
    output = SimpleNamespace(
        nonzero_logits=torch.logit(probability),
        nonzero_probability=probability,
        nb_mean=torch.tensor([[0.2, 0.8], [1.7, 0.3]]),
        theta=torch.tensor([1.0, 2.0]),
        positive_conditional_mean=conditional_mean,
        marginal_mean=probability * conditional_mean,
    )
    raw_target = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    mask = torch.ones_like(raw_target, dtype=torch.bool)

    values = runner._validation_sum_tensor(
        output=output,
        raw_target=raw_target,
        mask=mask,
        target_cells=2,
    )
    by_name = dict(zip(runner._VALIDATION_SUM_FIELDS, values.tolist(), strict=True))
    metrics = runner._metrics_from_sums(values)

    assert by_name["masked_entries"] == 4
    assert by_name["positive_entries"] == 2
    assert by_name["detection_true_positive"] == 2
    assert by_name["detection_true_negative"] == 1
    assert by_name["detection_false_positive"] == 1
    assert by_name["detection_false_negative"] == 0
    assert metrics["observed_positive_rate"] == pytest.approx(0.5)
    assert metrics["predicted_positive_probability_mean"] == pytest.approx(0.65)
    assert metrics["detection_brier_score"] == pytest.approx(0.265)
    assert metrics["detection_recall"] == pytest.approx(1.0)
    assert metrics["detection_specificity"] == pytest.approx(0.5)
    assert metrics["detection_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["detection_f1"] == pytest.approx(0.8)


def test_validation_sufficient_statistics_use_exact_support_denominators(
    runner: ModuleType,
) -> None:
    raw = {
        "masked_entries": 20,
        "positive_entries": 8,
        "joint_nll_sum": 40,
        "detection_nll_sum": 10,
        "positive_count_nll_sum": 24,
        "raw_absolute_error_sum": 30,
        "raw_squared_error_sum": 80,
        "log1p_absolute_error_sum": 10,
        "log1p_squared_error_sum": 20,
        "poisson_deviance_sum": 50,
        "positive_raw_absolute_error_sum": 16,
        "positive_raw_squared_error_sum": 32,
        "positive_log1p_absolute_error_sum": 4,
        "positive_log1p_squared_error_sum": 8,
        "observed_positive_sum": 8,
        "predicted_positive_probability_sum": 7,
        "detection_brier_sum": 2,
        "detection_true_positive": 6,
        "detection_true_negative": 10,
        "detection_false_positive": 2,
        "detection_false_negative": 2,
        "observed_count_sum": 60,
        "predicted_count_sum": 50,
        "target_cells": 4,
    }
    values = torch.tensor(
        [raw[field] for field in runner._VALIDATION_SUM_FIELDS],
        dtype=torch.float64,
    )

    metrics = runner._metrics_from_sums(values)

    assert metrics["pooled_masked_hurdle_negative_binomial_nll"] == 2.0
    assert metrics["masked_detection_bce"] == 0.5
    assert metrics["positive_zero_truncated_negative_binomial_nll"] == 3.0
    assert metrics["masked_raw_count_mae"] == 1.5
    assert metrics["masked_raw_count_rmse"] == 2.0
    assert metrics["positive_raw_count_mae"] == 2.0
    assert metrics["positive_raw_count_rmse"] == 2.0
    assert metrics["detection_recall"] == 0.75
    assert metrics["detection_specificity"] == pytest.approx(10.0 / 12.0)
    assert metrics["detection_precision"] == 0.75
    assert metrics["detection_f1"] == 0.75
    assert metrics["observed_count_mean"] == 3.0
    assert metrics["predicted_count_mean"] == 2.5


def test_training_epoch_pooling_reduces_all_ranks_before_dividing(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = tuple(base_runner.TRAINING_ALIASES)
    global_rows: dict[str, tuple[float, float, float, int, int, int]] = {}
    local_rows: dict[str, tuple[float, float, float, int, int, int]] = {}
    for index, alias in enumerate(aliases):
        targets = 17_000 if index < len(aliases) - 1 else 21_696
        masked = targets * (index + 2)
        positives = masked * (index + 1) // (index + 3)
        detection_sum = masked * (0.1 + 0.01 * index)
        positive_sum = positives * (1.0 + 0.2 * index)
        joint_sum = detection_sum + positive_sum
        global_rows[alias] = (
            joint_sum,
            detection_sum,
            positive_sum,
            masked,
            positives,
            targets,
        )
        local_rows[alias] = (
            joint_sum * 0.1,
            detection_sum * 0.1,
            positive_sum * 0.1,
            masked // 4,
            positives // 4,
            targets // 4,
        )

    global_transport = torch.tensor(
        [global_rows[alias] for alias in aliases], dtype=torch.float64
    )
    local_transport = torch.tensor(
        [local_rows[alias] for alias in aliases], dtype=torch.float64
    )
    remote_transport = global_transport - local_transport
    calls = 0

    def fake_all_reduce(
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(tensor, local_transport, rtol=0, atol=0)
        tensor.add_(remote_transport)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    pooled, masked_entries, target_cells, per_core = (
        runner._aggregate_training_epoch(local_rows, device=torch.device("cpu"))
    )

    expected_entries = sum(row[3] for row in global_rows.values())
    expected_joint = sum(row[0] for row in global_rows.values())
    assert calls == 1
    assert target_cells == 208_696
    assert masked_entries == expected_entries
    assert pooled == pytest.approx(expected_joint / expected_entries)
    for alias in aliases:
        joint, detection, positive, count, positives, targets = global_rows[alias]
        row = per_core[alias]
        assert row["target_cells"] == targets
        assert row["masked_entries"] == count
        assert row["positive_entries"] == positives
        assert row["pooled_masked_hurdle_negative_binomial_nll"] == pytest.approx(
            joint / count
        )
        assert row["masked_detection_bce"] == pytest.approx(detection / count)
        assert row[
            "positive_zero_truncated_negative_binomial_nll"
        ] == pytest.approx(positive / positives)


def test_epoch_and_per_core_rows_have_exact_declared_columns(
    runner: ModuleType,
) -> None:
    training = _training_rows()
    validation = _validation_result(runner)
    training_receipts = {
        alias: {"receipt_sha256": "a" * 64}
        for alias in base_runner.TRAINING_ALIASES
    }
    selection = EarlyStoppingState(
        best_value=2.5,
        best_epoch=2,
        completed_epoch=3,
        improved=False,
    )
    patience = EarlyStoppingState(
        best_value=2.4,
        best_epoch=1,
        bad_validations=2,
        completed_epoch=3,
        improved=False,
    )

    epoch = runner._make_epoch_row(
        completed_epoch=3,
        train_nll=2.1,
        train_masked_entries=120,
        train_target_cells=12,
        training=training,
        validation=validation,
        theta=(0.3, 1.0, 4.0),
        learning_rate=1e-4,
        duration=6.0,
        peak_vram=7.5,
        checkpoint_selection=selection,
        patience_control=patience,
        core_order=base_runner.TRAINING_ALIASES,
        training_receipts=training_receipts,
        global_gradient={"cosine": 0.1},
        block_gradients=({"block": 0}, {"block": 1}),
        theta_gradient={"cosine": 0.2},
    )
    rows = runner._per_core_rows(
        completed_epoch=3,
        training=training,
        training_receipts=training_receipts,
        validation=validation,
    )

    assert len(runner.EPOCH_COLUMNS) == len(set(runner.EPOCH_COLUMNS))
    assert tuple(epoch) == runner.EPOCH_COLUMNS
    assert set(epoch) == set(runner.EPOCH_COLUMNS)
    assert epoch["train_masked_detection_bce"] == pytest.approx(114.0 / 120.0)
    assert epoch[
        "train_positive_zero_truncated_negative_binomial_nll"
    ] == pytest.approx(138.0 / 24.0)
    assert epoch["training_targets_per_second"] == 2.0

    assert len(runner.PER_CORE_COLUMNS) == len(set(runner.PER_CORE_COLUMNS))
    assert len(rows) == len(base_runner.TRAINING_ALIASES) + len(
        base_runner.VALIDATION_ALIASES
    )
    for row in rows:
        assert set(row) == set(runner.PER_CORE_COLUMNS)
    training_row = next(row for row in rows if row["split"] == "training")
    validation_row = next(row for row in rows if row["split"] == "validation")
    assert training_row["masked_raw_count_mae"] == ""
    assert training_row["detection_true_positive"] == ""
    assert validation_row["masked_raw_count_mae"] != ""
    assert validation_row["detection_true_positive"] != ""

"""Pure contracts for the SO2 target-isolated binary-expression runner."""

from __future__ import annotations

import importlib
import json
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
    "PREFLIGHT_TRACKED_HEAD_COUNT_FIELD",
    "PREFLIGHT_TRACKED_HEAD_PARAMETER_COUNT",
    "TRACKED_HEAD_GRADIENT_RELATIVE_PATH",
    "ADDITIONAL_REQUIRED_FINAL_OUTPUTS",
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
    "_write_additional_final_artifacts",
    "_theta_summary",
    "_make_tracked_head_view",
)
_MISSING = object()


@pytest.fixture(scope="module")
def runner() -> Iterator[ModuleType]:
    originals = {
        name: getattr(base_runner, name, _MISSING) for name in _PATCHED_BASE_NAMES
    }
    module = importlib.import_module(
        "scripts.train.run_so2_target_isolated_binary_expression_bce"
    )
    module._install_binary_expression_contract()
    try:
        yield module
    finally:
        for name, value in originals.items():
            if value is _MISSING:
                delattr(base_runner, name)
            else:
                setattr(base_runner, name, value)


def _per_core_metrics(scale: int = 1) -> dict[str, float | int]:
    tp, tn, fp, fn = 3 * scale, 5 * scale, 1 * scale, 1 * scale
    return {
        "target_cells": 2 * scale,
        "masked_entries": 10 * scale,
        "positive_entries": 4 * scale,
        "pooled_masked_binary_cross_entropy": 0.25,
        "observed_positive_rate": 0.4,
        "predicted_positive_probability_mean": 0.35,
        "predicted_positive_class_rate": 0.4,
        "detection_brier_score": 0.2,
        "detection_true_positive": tp,
        "detection_true_negative": tn,
        "detection_false_positive": fp,
        "detection_false_negative": fn,
        "detection_accuracy": 0.8,
        "detection_balanced_accuracy": (0.75 + 5.0 / 6.0) / 2.0,
        "detection_recall": 0.75,
        "detection_specificity": 5.0 / 6.0,
        "detection_precision": 0.75,
        "detection_f1": 0.75,
        "detection_mcc": 7.0 / 12.0,
        "detection_auroc_histogram_4096": 0.9,
        "detection_auprc_histogram_4096": 0.8,
        "detection_expected_calibration_error_histogram_4096": 0.1,
    }


def _validation_result(runner: ModuleType) -> Any:
    per_core = {
        alias: _per_core_metrics(index + 1)
        for index, alias in enumerate(base_runner.VALIDATION_ALIASES)
    }
    receipts = {
        alias: {"receipt_sha256": str(index + 1) * 64}
        for index, alias in enumerate(base_runner.VALIDATION_ALIASES)
    }
    zero_histogram = (0,) * runner.HISTOGRAM_BIN_COUNT
    zero_probability = (0.0,) * runner.HISTOGRAM_BIN_COUNT
    return runner.ValidationResult(
        pooled_binary_cross_entropy=0.25,
        observed_positive_rate=0.4,
        predicted_positive_probability_mean=0.35,
        predicted_positive_class_rate=0.4,
        detection_brier_score=0.2,
        detection_true_positive=9,
        detection_true_negative=15,
        detection_false_positive=3,
        detection_false_negative=3,
        detection_accuracy=0.8,
        detection_balanced_accuracy=(0.75 + 5.0 / 6.0) / 2.0,
        detection_recall=0.75,
        detection_specificity=5.0 / 6.0,
        detection_precision=0.75,
        detection_f1=0.75,
        detection_mcc=7.0 / 12.0,
        detection_auroc_histogram_4096=0.9,
        detection_auprc_histogram_4096=0.8,
        detection_expected_calibration_error_histogram_4096=0.1,
        operating_threshold_metrics={
            "0.50": {"threshold": 0.5, "detection_f1": 0.75}
        },
        masked_entries=30,
        positive_entries=12,
        target_cells=6,
        positive_histogram_4096=zero_histogram,
        negative_histogram_4096=zero_histogram,
        probability_sum_histogram_4096=zero_probability,
        per_core=per_core,
        coverage_receipts=receipts,
    )


def _training_rows() -> dict[str, Mapping[str, float | int]]:
    return {
        alias: {
            "target_cells": 1,
            "masked_entries": 10,
            "positive_entries": 2,
            "detection_nll_sum": 4.0 + index,
            "pooled_masked_binary_cross_entropy": 0.4 + index / 10.0,
            "observed_positive_rate": 0.2,
        }
        for index, alias in enumerate(base_runner.TRAINING_ALIASES)
    }


def test_import_installs_binary_contract_and_head_gradient_hooks(
    runner: ModuleType,
) -> None:
    assert base_runner.CAMPAIGN_ID == runner.CAMPAIGN_ID
    assert base_runner.MODEL_NAME == runner.MODEL_NAME
    assert base_runner.PRIMARY_METRIC == runner.PRIMARY_METRIC
    assert base_runner.EXPECTED_PARAMETER_COUNT == 5_134_088
    assert base_runner.PREFLIGHT_TRACKED_HEAD_COUNT_FIELD == (
        "detection_head_parameter_count"
    )
    assert base_runner.TRACKED_HEAD_GRADIENT_RELATIVE_PATH.endswith(
        "gradient_direction_detection_head.csv"
    )
    assert base_runner._make_tracked_head_view is runner._make_tracked_head_view
    assert base_runner._write_additional_final_artifacts is (
        runner._write_additional_final_artifacts
    )


def test_validation_statistics_cover_all_requested_binary_metrics(
    runner: ModuleType,
) -> None:
    probability = torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float32)
    raw_target = torch.tensor([[0, 0], [1, 3]], dtype=torch.int64)
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    output = SimpleNamespace(
        nonzero_logits=torch.logit(probability),
        nonzero_probability=probability,
    )

    values = runner._validation_sum_tensor(
        output=output,
        raw_target=raw_target,
        mask=mask,
        target_cells=2,
    )
    metrics = runner._metrics_from_sums(values)

    assert values.shape == (runner._VALIDATION_VECTOR_LENGTH,)
    assert metrics["masked_entries"] == 4
    assert metrics["positive_entries"] == 2
    assert metrics["observed_positive_rate"] == 0.5
    assert metrics["predicted_positive_probability_mean"] == pytest.approx(0.5)
    assert metrics["detection_brier_score"] == pytest.approx(0.025)
    assert metrics["detection_true_positive"] == 2
    assert metrics["detection_true_negative"] == 2
    assert metrics["detection_false_positive"] == 0
    assert metrics["detection_false_negative"] == 0
    assert metrics["detection_accuracy"] == 1.0
    assert metrics["detection_balanced_accuracy"] == 1.0
    assert metrics["detection_precision"] == 1.0
    assert metrics["detection_recall"] == 1.0
    assert metrics["detection_specificity"] == 1.0
    assert metrics["detection_f1"] == 1.0
    assert metrics["detection_mcc"] == 1.0
    assert metrics["detection_auroc_histogram_4096"] == 1.0
    assert metrics["detection_auprc_histogram_4096"] == 1.0
    assert metrics[
        "detection_expected_calibration_error_histogram_4096"
    ] == pytest.approx(0.15)
    assert set(metrics["operating_threshold_metrics"]) == {
        f"{threshold:.2f}" for threshold in runner.OPERATING_THRESHOLDS
    }


def test_histogram_auc_handles_ties_without_prediction_dump(
    runner: ModuleType,
) -> None:
    positive = torch.zeros(runner.HISTOGRAM_BIN_COUNT, dtype=torch.float64)
    negative = torch.zeros_like(positive)
    probability_sum = torch.zeros_like(positive)
    tied_bin = runner.HISTOGRAM_BIN_COUNT // 2
    positive[tied_bin] = 2
    negative[tied_bin] = 2
    probability_sum[tied_bin] = 2.0

    auroc, auprc, ece = runner._histogram_auc_metrics(
        positive, negative, probability_sum
    )

    assert auroc == pytest.approx(0.5)
    assert auprc == pytest.approx(0.5)
    assert ece == pytest.approx(0.0)


def test_training_epoch_pooling_reduces_before_dividing(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = tuple(base_runner.TRAINING_ALIASES)
    target_counts = [17_000] * 11 + [21_696]
    global_rows = {
        alias: (
            float((index + 1) * target_counts[index]),
            2 * target_counts[index],
            target_counts[index] // 4,
            target_counts[index],
        )
        for index, alias in enumerate(aliases)
    }
    local_rows = {
        alias: (
            row[0] * 0.25,
            row[1] // 4,
            row[2] // 4,
            row[3] // 4,
        )
        for alias, row in global_rows.items()
    }
    global_transport = torch.tensor(
        [global_rows[alias] for alias in aliases], dtype=torch.float64
    )
    local_transport = torch.tensor(
        [local_rows[alias] for alias in aliases], dtype=torch.float64
    )
    remote = global_transport - local_transport

    def fake_all_reduce(
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp | None = None,
    ) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        torch.testing.assert_close(tensor, local_transport, rtol=0, atol=0)
        tensor.add_(remote)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    pooled, entries, targets, per_core = runner._aggregate_training_epoch(
        local_rows, device=torch.device("cpu")
    )

    expected_entries = sum(row[1] for row in global_rows.values())
    expected_loss = sum(row[0] for row in global_rows.values())
    assert targets == 208_696
    assert entries == expected_entries
    assert pooled == pytest.approx(expected_loss / expected_entries)
    assert set(per_core) == set(aliases)


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
        best_value=0.25,
        best_epoch=2,
        completed_epoch=3,
        improved=False,
    )
    patience = EarlyStoppingState(
        best_value=0.24,
        best_epoch=1,
        bad_validations=2,
        completed_epoch=3,
        improved=False,
    )

    epoch = runner._make_epoch_row(
        completed_epoch=3,
        train_nll=0.4,
        train_masked_entries=120,
        train_target_cells=12,
        training=training,
        validation=validation,
        theta=(-4.0, -2.0, 1.0),
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
    assert epoch["validation_detection_accuracy"] == 0.8
    assert epoch["validation_detection_balanced_accuracy"] == pytest.approx(
        (0.75 + 5.0 / 6.0) / 2.0
    )
    assert epoch["validation_detection_mcc"] == pytest.approx(7.0 / 12.0)
    assert epoch[
        "validation_detection_expected_calibration_error_histogram_4096"
    ] == 0.1
    assert json.loads(epoch["validation_operating_threshold_metrics_json"])[
        "0.50"
    ]["threshold"] == 0.5
    assert epoch["training_targets_per_second"] == 2.0

    assert len(runner.PER_CORE_COLUMNS) == len(set(runner.PER_CORE_COLUMNS))
    assert len(rows) == len(base_runner.TRAINING_ALIASES) + len(
        base_runner.VALIDATION_ALIASES
    )
    for row in rows:
        assert set(row) == set(runner.PER_CORE_COLUMNS)
    training_row = next(row for row in rows if row["split"] == "training")
    validation_row = next(row for row in rows if row["split"] == "validation")
    assert training_row["detection_true_positive"] == ""
    assert training_row["detection_auroc_histogram_4096"] == ""
    assert validation_row["detection_true_positive"] != ""
    assert validation_row["detection_auroc_histogram_4096"] != ""


def test_final_metrics_and_provenance_are_detection_only(
    runner: ModuleType,
) -> None:
    validation = _validation_result(runner)

    metrics = runner._final_metrics(validation)
    provenance = runner._additional_provenance(validation)

    assert metrics[runner.PRIMARY_METRIC] == 0.25
    assert metrics["val/unseen_donor/detection_accuracy"] == 0.8
    assert metrics["val/unseen_donor/detection_precision"] == 0.75
    assert metrics["val/unseen_donor/detection_recall"] == 0.75
    assert metrics[
        "val/unseen_donor/detection_matthews_correlation_coefficient"
    ] == pytest.approx(7.0 / 12.0)
    assert metrics[
        "val/unseen_donor/detection_expected_calibration_error_histogram_4096"
    ] == 0.1
    assert not any("count_mae" in key or "negative_binomial" in key for key in metrics)
    assert provenance["count_mean_or_dispersion_learned"] is False
    assert provenance["histogram_curve_approximation"][
        "prediction_vectors_persisted"
    ] is False


def test_compact_curve_artifact_contains_histograms_not_predictions(
    runner: ModuleType,
) -> None:
    validation = _validation_result(runner)

    class FakeArchive:
        def __init__(self) -> None:
            self.path: str | None = None
            self.payload: Mapping[str, Any] | None = None

        def write_json(self, path: str, payload: Mapping[str, Any]) -> None:
            self.path = path
            self.payload = payload

    archive = FakeArchive()
    runner._write_additional_final_artifacts(archive, validation)

    assert archive.path == runner.DETECTION_CURVE_RELATIVE_PATH
    assert archive.payload is not None
    assert archive.payload["bin_count"] == 4096
    assert len(
        archive.payload["positive_counts_by_ascending_probability_bin"]
    ) == 4096
    assert len(
        archive.payload["probability_sums_by_ascending_probability_bin"]
    ) == 4096
    assert archive.payload["per_entry_predictions_persisted"] is False


def test_partial_finalization_cleanup_removes_compact_curve(
    runner: ModuleType,
    tmp_path: Any,
) -> None:
    curve = tmp_path / runner.DETECTION_CURVE_RELATIVE_PATH
    curve.parent.mkdir(parents=True)
    curve.write_text("partial", encoding="utf-8")

    base_runner._remove_partial_finalization_outputs(tmp_path)

    assert not curve.exists()


def test_detection_head_tracking_uses_weight_and_bias(runner: ModuleType) -> None:
    model = SimpleNamespace(detection_head=torch.nn.Linear(4, 3))

    view = runner._make_tracked_head_view(model)
    minimum, median, maximum = runner._detection_head_summary(model)

    assert view is model.detection_head
    assert sum(parameter.numel() for parameter in view.parameters()) == 15
    assert minimum <= median <= maximum

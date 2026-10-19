"""Scientific report guards must fail before publishing invalid comparisons."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[3] / "scripts/analysis/report_so2_binary_expression_bce.py"
    spec = importlib.util.spec_from_file_location("_so2_binary_bce_report_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(report, *, binary: bool) -> dict:
    """Small aggregate fixture: 30 TP, 50 TN, 10 FP, 10 FN; no individual rows."""
    detection = {
        "detection_true_positive": 30,
        "detection_true_negative": 50,
        "detection_false_positive": 10,
        "detection_false_negative": 10,
        "detection_bce": .4,
        "detection_brier_score": .15,
        "detection_precision": .75,
        "detection_recall": .75,
        "detection_f1": .75,
        "detection_specificity": 5 / 6,
        "detection_threshold": .5,
        "observed_positive_rate": .4,
        "predicted_positive_probability_mean": .35,
    }
    metrics = {report.PREFIX + key: value for key, value in detection.items()}
    metrics.update({"val/masked_entries": 100, "val/positive_entries": 40, "val/target_cells": 10})
    if binary:
        metrics[report.PRIMARY] = .4
        del metrics[report.PREFIX + "detection_bce"]
    return {
        "metrics": metrics,
        "config": {"graph": {"construction": "geometry_only", "radius": 5},
                   "dataset": {"dataset_fingerprint": "d" * 64}},
        "provenance": {
            "preprocessing_fingerprint": "p" * 64,
            "split_fingerprint": "s" * 64,
            "overlay_manifest_sha256": "o" * 64,
            "validation_coverage_receipts_sha256": "v" * 64,
            "training_cells": 20, "validation_cells": 10,
            "training_aliases": ["SO2-C15"], "validation_aliases": ["SO2-C27"],
            "detection_threshold": .5,
            "detection_initialization": {"fit_scope": "training_only", "fingerprint": "i" * 64},
        },
        "epochs": [{"global_epoch": "1", "core_order_json": '["SO2-C15"]',
                    "training_coverage_receipts_json": '{"mask_sha":"train_mask_1"}',
                    "validation_coverage_receipts_json": '{"mask_sha":"validation_mask"}'}],
    }


@pytest.mark.parametrize("binary", [False, True])
def test_confusion_metrics_match_independent_hand_calculation(report, binary: bool) -> None:
    metrics = report.detection_metrics(_bundle(report, binary=binary), binary=binary)
    assert metrics["bce"] == .4
    assert metrics["detection_accuracy"] == pytest.approx(4 / 5)
    assert metrics["detection_balanced_accuracy"] == pytest.approx(19 / 24)
    assert metrics["detection_matthews_correlation_coefficient"] == pytest.approx(7 / 12)
    assert metrics["predicted_positive_class_rate"] == pytest.approx(2 / 5)


def test_all_negative_predictions_have_defined_zero_mcc(report) -> None:
    bundle = _bundle(report, binary=False)
    for suffix, value in {"true_positive": 0, "true_negative": 60, "false_positive": 0, "false_negative": 40}.items():
        bundle["metrics"][report.PREFIX + "detection_" + suffix] = value
    for key, value in {"detection_precision": 0, "detection_recall": 0, "detection_f1": 0, "detection_specificity": 1}.items():
        bundle["metrics"][report.PREFIX + key] = value
    metrics = report.detection_metrics(bundle, binary=False)
    assert metrics["detection_matthews_correlation_coefficient"] == 0
    assert metrics["detection_accuracy"] == .6
    assert metrics["detection_balanced_accuracy"] == .5


def test_recorded_rate_cannot_contradict_saved_confusion_counts(report) -> None:
    bundle = _bundle(report, binary=False)
    bundle["metrics"][report.PREFIX + "detection_recall"] = .95
    with pytest.raises(ValueError, match="Metric disagreement: Derived detection_recall"):
        report.detection_metrics(bundle, binary=False)


@pytest.mark.parametrize("field,value", [("val/masked_entries", 101), ("val/positive_entries", 41)])
def test_confusion_counts_cannot_disagree_with_evaluation_support(report, field: str, value: int) -> None:
    bundle = _bundle(report, binary=False)
    bundle["metrics"][field] = value
    with pytest.raises(ValueError, match="Confusion matrix support differs"):
        report.detection_metrics(bundle, binary=False)


@pytest.mark.parametrize("tp,fn", [(30.5, 9.5), (-1, 41)])
def test_fractional_or_negative_confusion_counts_rejected_even_when_sums_match(report, tp: float, fn: float) -> None:
    bundle = _bundle(report, binary=False)
    bundle["metrics"][report.PREFIX + "detection_true_positive"] = tp
    bundle["metrics"][report.PREFIX + "detection_false_negative"] = fn
    with pytest.raises(ValueError, match="nonnegative integers"):
        report.detection_metrics(bundle, binary=False)


def test_replayed_integer_counts_require_exact_equality_at_large_support(report) -> None:
    # A relative tolerance would incorrectly accept a one-entry change here.
    best = {"validation_detection_true_negative": "17000000"}
    final = {report.PREFIX + "detection_true_negative": 17000001}
    with pytest.raises(ValueError, match="confusion count differs"):
        report.audit_replayed_detection(best, final)
    final[report.PREFIX + "detection_true_negative"] = 17000000
    report.audit_replayed_detection(best, final)


@pytest.mark.parametrize("field,value", [("detection_recall", float("nan")), ("detection_brier_score", float("inf"))])
def test_nonfinite_metric_rejected(report, field: str, value: float) -> None:
    bundle = _bundle(report, binary=False)
    bundle["metrics"][report.PREFIX + field] = value
    with pytest.raises(ValueError, match="Nonfinite metric"):
        report.detection_metrics(bundle, binary=False)


@pytest.mark.parametrize("location,marker,error", [
    ("scratch/active_runs", None, "Expected one finalized bundle"),
    ("artifacts/runs/2099/01", None, "Successful completion marker absent"),
    ("artifacts/runs/2099/01", "_FAILED", "Successful completion marker absent"),
])
def test_active_or_unsuccessful_run_cannot_publish_report(
    report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str, marker: str | None, error: str,
) -> None:
    run_id = "r_synthetic_unfinished"
    run = tmp_path / location / run_id
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"status":"running"}\n')
    if marker:
        (run / marker).write_text('{}\n')
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setenv("BAGM_ROOT", str(tmp_path))
    monkeypatch.setenv("BAGM_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BAGM_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(sys, "argv", ["report", "--run-id", run_id])
    verification = Mock(side_effect=AssertionError("Unfinished source reached checksum verification"))
    monkeypatch.setattr(report, "verify_run_bundle", verification)
    with pytest.raises(ValueError, match=error):
        report.main()
    verification.assert_not_called()
    assert not (tmp_path / "reports").exists()
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_identical_comparator_protocol_passes_with_zero_metric_differences(report) -> None:
    current, comparator = _bundle(report, binary=True), _bundle(report, binary=False)
    comparison = report.compare(current, comparator)
    assert comparison
    assert all(row["binary_minus_hurdle"] == 0 for row in comparison)
    assert current["comparison_checks"]["exact_mask_receipts_and_core_order_match_through_epoch"] == 1


@pytest.mark.parametrize("kind,error", [
    ("graph", "Resolved graph configuration differs"),
    ("dataset", "Dataset fingerprint differs"),
    ("split", "Comparator mismatch: split_fingerprint"),
    ("validation_receipt_hash", "Comparator mismatch: validation_coverage_receipts_sha256"),
    ("training_mask", "Matched epoch protocol differs: training_coverage_receipts_json"),
    ("validation_mask", "Matched epoch protocol differs: validation_coverage_receipts_json"),
    ("core_order", "Matched epoch protocol differs: core_order_json"),
    ("threshold", "Final detection threshold is not 0.5"),
])
def test_comparator_mismatch_cannot_produce_a_matched_comparison(report, kind: str, error: str) -> None:
    current, comparator = _bundle(report, binary=True), _bundle(report, binary=False)
    original = deepcopy(current)
    if kind == "graph":
        comparator["config"]["graph"]["radius"] = 6
    elif kind == "dataset":
        comparator["config"]["dataset"]["dataset_fingerprint"] = "x" * 64
    elif kind == "split":
        comparator["provenance"]["split_fingerprint"] = "x" * 64
    elif kind == "validation_receipt_hash":
        comparator["provenance"]["validation_coverage_receipts_sha256"] = "x" * 64
    elif kind == "threshold":
        comparator["metrics"][report.PREFIX + "detection_threshold"] = .4
    else:
        field = {"training_mask": "training_coverage_receipts_json", "validation_mask": "validation_coverage_receipts_json",
                 "core_order": "core_order_json"}[kind]
        comparator["epochs"][0][field] = '"changed"'
    with pytest.raises(ValueError, match=error):
        report.compare(current, comparator)
    assert current == original


def test_existing_report_version_is_preserved_before_source_access(
    report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "r_synthetic_completed"
    output = tmp_path / "reports/analyses/so2_binary_expression_bce" / run_id / "v1"
    output.mkdir(parents=True)
    (output / "report.html").write_text("original report")
    monkeypatch.setenv("BAGM_ROOT", str(tmp_path))
    monkeypatch.setenv("BAGM_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(sys, "argv", ["report", "--run-id", run_id])
    loader = Mock(side_effect=AssertionError("Existing version read new sources"))
    monkeypatch.setattr(report, "load_bundle", loader)
    with pytest.raises(ValueError, match="Report version already exists"):
        report.main()
    loader.assert_not_called()
    assert (output / "report.html").read_text() == "original report"

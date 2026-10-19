"""Independent runtime controls for the fixed-budget optimization audit.

The controller tests deliberately execute every validation transition. Merely
checking configuration fields missed the earlier minimum-equals-maximum bug.
All observations here are synthetic; no source data or CUDA work is required.
"""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.train import run_so2_target_isolated_nb as base
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.so2_nb_training import EarlyStoppingState, update_early_stopping
from spatial_benchmark.target_minibatch import target_window_plan


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def audit():
    original = dict(vars(base))
    module = importlib.import_module("scripts.train.run_so2_bce_optimization_audit")
    nested_originals = [(value, dict(vars(value))) for value in (module, module.minibatch, module.bce)]
    try:
        yield module
    finally:
        for value, snapshot in (*nested_originals, (base, original)):
            for name in tuple(vars(value)):
                if name not in snapshot:
                    delattr(value, name)
            vars(value).update(snapshot)


@pytest.fixture
def configuration(monkeypatch):
    from spatial_benchmark import so2_bce_optimization_audit as contract

    # Real architecture/trainer fields, synthetic provenance. No dataset read.
    source = {"seed": 0, "fold": 0, "attempt": 1,
              "campaign": {"campaign_id": "synthetic_completed_bce"},
              "metadata": {"gradient_diagnostics": {"enabled": True}},
              "evaluation": {"primary_metric": contract.PRIMARY_METRIC},
              "masking": {"frozen_fixture_mask_seed": 731},
              "dataset": {"dataset_id": "synthetic_no_data_read"}}
    for group, filename in (
        ("model", "target_isolated_geometry_modulated_relative_qkv_gat_binary_expression_bce_256_l4_h8"),
        ("trainer", "so2_target_isolated_geometry_modulated_binary_expression_bce_train12_val2_earlystop"),
    ):
        source[group] = yaml.safe_load((ROOT / "configs" / group / (filename + ".yaml")).read_text())[group]
    monkeypatch.setattr(contract, "SOURCE_CONFIG_SHA", canonical_sha256(source))
    bundle = {"run_id": contract.SOURCE_RUN, "success_marker_sha256": contract.SOURCE_SUCCESS_SHA,
              **{key: "a" * 64 for key in ("config_file_sha256", "manifest_sha256",
                  "checksum_manifest_sha256", "data_receipt_sha256", "split_receipt_sha256")}}

    def build(*, profile="screen", mode="target_window", batch_size=1024,
              learning_rate=1e-4, dropout=0.1, weight_decay=1e-5, seed=0):
        smoke = None if profile == "smoke" else {
            "run_id": "r_synthetic_new_protocol_smoke",
            "source_configuration_sha256": contract.SOURCE_CONFIG_SHA,
            "optimizer_batch_mode": mode,
            "receipt_path": "/synthetic/" + contract.PREFLIGHT_RELATIVE_PATH,
            "receipt_sha256": "b" * 64, "success_marker_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        }
        result = contract.build_configuration(
            source, bundle, profile=profile, batch_mode=mode,
            batch_size=None if mode == "core_pair" else batch_size,
            learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay,
            seed=seed, gpu="0", task_contract_sha256="e" * 64, smoke_preflight=smoke,
        )
        contract.validate_configuration(result)
        return result

    return build


@pytest.mark.parametrize("profile,mode,epochs", [
    ("smoke", "target_window", 2), ("smoke", "core_pair", 2),
    ("screen", "target_window", 8), ("screen", "core_pair", 8),
    ("confirmation", "target_window", 32),
])
@pytest.mark.parametrize("curve", ["flat", "worsening", "improving", "sub_min_delta"])
def test_installed_controller_completes_exact_fixed_budget(
    audit, configuration, profile, mode, epochs, curve,
):
    config = configuration(profile=profile, mode=mode)
    audit.install_contract(config)
    assert base.MINIMUM_EPOCHS == 1
    assert base.MAXIMUM_EPOCHS == epochs
    assert base.EARLY_STOPPING_PATIENCE > epochs
    state, selection = EarlyStoppingState(), EarlyStoppingState()
    values = []
    for epoch in range(1, epochs + 1):
        assert not state.should_stop, "A fixed-budget run stopped before its endpoint"
        value = {"flat": 0.3, "worsening": 0.3 + epoch * 0.001,
                 "improving": 0.3 - epoch * 0.001,
                 "sub_min_delta": 0.3 - epoch * 0.000001}[curve]
        values.append(value)
        state = update_early_stopping(
            state, value, completed_epoch=epoch,
            minimum_epochs=base.MINIMUM_EPOCHS, maximum_epochs=base.MAXIMUM_EPOCHS,
            patience=base.EARLY_STOPPING_PATIENCE, min_delta=base.EARLY_STOPPING_MIN_DELTA,
        )
        selection = base._update_exact_checkpoint_selection(selection, value, completed_epoch=epoch)
        assert state.completed_epoch == epoch
        assert state.should_stop == (epoch == epochs)
        assert not selection.should_stop
    assert state.stop_reason == "maximum_epochs"
    assert selection.best_value == min(values)
    assert selection.best_epoch == 1 + values.index(min(values))
    if curve == "sub_min_delta":
        assert state.best_epoch == 1
        assert selection.best_epoch == epochs


def test_model_seed_changes_initialization_and_dropout_rng_but_preserves_masks_and_order(
    audit, configuration, monkeypatch,
):
    first = SimpleNamespace(n_genes=1000, node_covariates=torch.zeros(2, 22))
    bundle = SimpleNamespace(training_batches=(first,))
    core = SimpleNamespace(alias="SO2-C15", n_nodes=17, n_genes=11)
    results = []
    for seed in (0, 1, 0):
        config = configuration(profile="confirmation", seed=seed)
        audit.install_contract(config)
        assert base.MODEL_SEED == seed
        # Prevalence fitting has its own data contract. Supply the same synthetic
        # training-only probabilities to inspect actual production initialization.
        monkeypatch.setitem(base._build_model.__globals__, "_training_nonzero_prevalence",
                            lambda _: torch.full((1000,), 0.2))
        model = base._build_model(config, bundle)
        _, training = base._partition_and_mask(core, rank=0, global_epoch=3)
        _, validation = base._partition_and_mask(core, rank=0, global_epoch=None)
        order = base.cohort_relative_qkv_core_order(
            3, aliases=base.TRAINING_ALIASES, core_order_seed=base.CORE_ORDER_SEED,
        )
        windows = target_window_plan(
            1031, 512, seed=config["trainer"]["target_window_order_seed"],
            core_alias=core.alias, global_epoch=3,
        )
        results.append({
            "weight": model.encoder.expression_projection.weight.detach().clone(),
            "bias": model.detection_head.bias.detach().clone(),
            "step_seed": base._model_step_seed(3, 7, core.alias, 0),
            "train_mask": training.mask.copy(), "validation_mask": validation.mask.copy(),
            "order": order, "windows": windows,
        })
        assert torch.count_nonzero(model.detection_head.weight) == 0
    assert not torch.equal(results[0]["weight"], results[1]["weight"])
    assert torch.equal(results[0]["weight"], results[2]["weight"])
    assert results[0]["step_seed"] != results[1]["step_seed"]
    assert results[0]["step_seed"] == results[2]["step_seed"]
    for actual in results[1:]:
        assert torch.equal(results[0]["bias"], actual["bias"])
        for key in ("train_mask", "validation_mask"):
            assert np.array_equal(results[0][key], actual[key])
        for key in ("order", "windows"):
            assert results[0][key] == actual[key]


@pytest.mark.parametrize("learning_rate,dropout,weight_decay", [
    (3e-5, 0.0, 1e-5), (3e-4, 0.3, 1e-5), (1e-4, 0.1, 0.01),
])
def test_declared_hyperparameters_reach_real_model_and_adamw(
    audit, configuration, monkeypatch, learning_rate, dropout, weight_decay,
):
    config = configuration(learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay)
    original_config = copy.deepcopy(config)
    audit.install_contract(config)
    monkeypatch.setitem(base._build_model.__globals__, "_training_nonzero_prevalence",
                        lambda _: torch.full((1000,), 0.2))
    bundle = SimpleNamespace(training_batches=(SimpleNamespace(
        n_genes=1000, node_covariates=torch.zeros(2, 22)),))
    model = base._build_model(config, bundle)
    probabilities = [module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert probabilities and all(value == dropout for value in probabilities)
    assert all(block.attention_dropout_probability == 0.0 for block in model.blocks)
    monkeypatch.setattr(base, "_wrap_target_model_ddp", lambda value: value)
    monkeypatch.setattr(base, "_make_grad_scaler", lambda *_: object())
    training_model, optimizer, scheduler, _ = base._initialize_training(model, config, torch.device("cpu"))
    assert training_model is model
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["lr"] == learning_rate for group in optimizer.param_groups)
    assert all(group["weight_decay"] == weight_decay for group in optimizer.param_groups)
    assert scheduler.optimizer is optimizer
    assert config == original_config, "Installing a run must not rewrite its resolved configuration"


def test_importing_audit_does_not_install_or_mutate_an_existing_runner_contract():
    code = """
from scripts.train import run_so2_target_isolated_nb as base
before = dict(base.__dict__)
from scripts.train import run_so2_bce_optimization_audit
changed = [name for name in before if base.__dict__.get(name) is not before[name]]
added = set(base.__dict__) - set(before)
assert not changed, changed
assert not added, added
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_prevalence_reference_matches_expanded_masked_entries_including_score_ties(audit):
    # The second and third genes have identical scores and unequal support.
    # The last gene has no masked observations and must carry zero weight.
    p = np.array([0.8, 0.5, 0.5, 0.1, 0.99])
    positives = np.array([2, 1, 2, 0, 0], dtype=np.int64)
    counts = np.array([3, 4, 2, 1, 0], dtype=np.int64)
    labels = np.concatenate([np.r_[np.ones(k), np.zeros(n-k)] for k, n in zip(positives, counts)])
    scores = np.repeat(p, counts)
    metrics = audit.prevalence_reference_metrics(p, positives, counts)
    assert metrics["masked_entries"] == 10
    assert metrics["positive_entries"] == 5
    expected_bce = np.mean(-labels * np.log(scores) - (1-labels) * np.log1p(-scores))
    assert metrics["pooled_masked_binary_cross_entropy"] == pytest.approx(expected_bce)
    assert metrics["binary_cross_entropy_sum"] == pytest.approx(expected_bce * len(labels))
    assert metrics["detection_brier_score"] == pytest.approx(np.mean((scores-labels)**2))
    assert metrics["predicted_positive_probability_mean"] == pytest.approx(scores.mean())
    assert metrics["observed_positive_rate"] == pytest.approx(labels.mean())
    pairwise_auc = np.mean([
        float(a > b) + 0.5 * float(a == b)
        for a in scores[labels == 1] for b in scores[labels == 0]
    ])
    # Each positive receives the precision of its entire tied threshold group.
    exact_ap = np.mean([labels[scores >= score].mean() for score in scores[labels == 1]])
    assert metrics["detection_auroc_exact"] == pytest.approx(pairwise_auc)
    assert metrics["detection_average_precision_exact"] == pytest.approx(exact_ap)
    assert metrics["detection_true_positive"] == 5
    assert metrics["detection_false_positive"] == 4
    assert metrics["detection_true_negative"] == 1
    assert metrics["detection_false_negative"] == 0
    assert metrics["detection_f1"] == pytest.approx(10/14)
    assert metrics["detection_precision"] == pytest.approx(5/9)
    assert metrics["detection_recall"] == 1.0
    # A gene permutation must not break ties according to incidental order.
    permutation = np.array([2, 4, 0, 3, 1])
    reordered = audit.prevalence_reference_metrics(p[permutation], positives[permutation], counts[permutation])
    for key, value in metrics.items():
        assert reordered[key] == pytest.approx(value), key


@pytest.mark.parametrize("failure", ["fractional", "too_many_positive", "all_positive", "nan", "zero_probability"])
def test_prevalence_reference_rejects_invalid_support_and_probabilities(audit, failure):
    p, positives, counts = np.array([0.2, 0.7]), np.array([1, 2]), np.array([4, 3])
    if failure == "fractional":
        positives = positives.astype(float) + 0.1
    elif failure == "too_many_positive":
        positives[0] = 5
    elif failure == "all_positive":
        positives = counts.copy()
    elif failure == "nan":
        p[0] = np.nan
    else:
        p[0] = 0.0
    with pytest.raises(base.SO2NBTrainingError):
        audit.prevalence_reference_metrics(p, positives, counts)


@pytest.mark.parametrize("profile,expected_epochs", [
    ("smoke", [2]), ("screen", [7, 8]), ("confirmation", [29, 30, 31, 32]),
])
def test_endpoint_uses_declared_terminal_epochs_even_when_an_earlier_checkpoint_is_better(
    audit, configuration, profile, expected_epochs,
):
    config = configuration(profile=profile)
    rows = [{"global_epoch": str(epoch),
             "validation_pooled_masked_binary_cross_entropy": str(0.2 + epoch * 0.001)}
            for epoch in range(1, config["trainer"]["max_epochs"] + 1)]
    rows[0]["validation_pooled_masked_binary_cross_entropy"] = "0.01"
    result = audit.terminal_validation_endpoint(rows, config["experiment"]["screening_endpoint_epochs"])
    expected_values = [0.2 + epoch * 0.001 for epoch in expected_epochs]
    assert result["endpoint_epochs"] == expected_epochs
    assert result["validation_bce_values"] == pytest.approx(expected_values)
    assert result["mean_validation_bce"] == pytest.approx(np.mean(expected_values))
    assert result["mean_validation_bce"] > min(float(row["validation_pooled_masked_binary_cross_entropy"]) for row in rows)
    assert result["best_checkpoint_metric_used_for_endpoint_selection"] is False


@pytest.mark.parametrize("failure", ["missing", "duplicate_row", "duplicate_endpoint", "nan", "empty_endpoint"])
def test_endpoint_rejects_incomplete_or_ambiguous_terminal_evidence(audit, failure):
    rows = [{"global_epoch": 7, "validation_pooled_masked_binary_cross_entropy": 0.25},
            {"global_epoch": 8, "validation_pooled_masked_binary_cross_entropy": 0.26}]
    endpoint = [7, 8]
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate_row":
        rows.append(dict(rows[-1]))
    elif failure == "duplicate_endpoint":
        endpoint = [7, 7, 8]
    elif failure == "nan":
        rows[-1]["validation_pooled_masked_binary_cross_entropy"] = float("nan")
    else:
        endpoint = []
    with pytest.raises(base.SO2NBTrainingError):
        audit.terminal_validation_endpoint(rows, endpoint)

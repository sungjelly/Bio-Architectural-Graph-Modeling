from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.analysis import report_local_gene_early_stopping as report
from spatial_benchmark.local_gene_stopping import LocalGeneEarlyStopping
from spatial_benchmark.paths import current_paths
from spatial_benchmark.result_catalog import validate_result_manifest


def stopping_trace(values):
    policy = LocalGeneEarlyStopping()
    rows = []
    for epoch, value in enumerate(values, 1):
        decision = policy.observe(epoch=epoch, count_nll=value)
        rows.append({"epoch": epoch, "val/count_nll": value,
            "stopping_decision": asdict(decision), "stopping_state": policy.state_dict()})
    state = policy.state_dict()
    summary = {"stop_reason": policy.stop_reason, "early_stopping_triggered": policy.stop_reason == "patience",
        "safety_cap_reached": len(rows) >= 500, "best_epoch": state["state"]["best_epoch"],
        "primary_metric_value": state["state"]["best_count_nll"], "stopping_state": state}
    receipt = {key: summary[key] for key in ("stop_reason", "early_stopping_triggered", "safety_cap_reached")}
    receipt["policy_state"] = state
    return rows, summary, receipt


def test_independent_replay_keeps_strict_best_and_anchor_distinct():
    history, summary, receipt = stopping_trace([.3] + [.3001] * 18 + [.29995] + [.3001] * 30)
    actual = report.verify_stopping_history(history, summary, receipt)
    assert actual["stop_reason"] == "patience"
    assert actual["best_epoch"] == 20 and actual["anchor_epoch"] == 1
    assert not actual["global_convergence_established"]


def test_cap500_is_not_early_stopping_or_convergence():
    history, summary, receipt = stopping_trace([1. - epoch * .0002 for epoch in range(500)])
    actual = report.verify_stopping_history(history, summary, receipt)
    assert actual["stop_reason"] == "maximum_epochs"
    assert actual["safety_cap_reached"] and not actual["early_stopping_triggered"]
    assert "convergence is unestablished" in actual["interpretation"]
    receipt["early_stopping_triggered"] = True
    with pytest.raises(ValueError, match="receipt or summary"):
        report.verify_stopping_history(history, summary, receipt)


def test_rejects_fake_stop_or_corrupted_policy_state():
    history, summary, receipt = stopping_trace([.3] * 50)
    history[-1]["stopping_decision"]["stop_reason"] = "maximum_epochs"
    with pytest.raises(ValueError, match="independent replay"):
        report.verify_stopping_history(history, summary, receipt)
    history, summary, receipt = stopping_trace([.3] * 50)
    history[20]["stopping_state"] = {"fake": True}
    with pytest.raises(ValueError, match="state history"):
        report.verify_stopping_history(history, summary, receipt)


def rows():
    values = []
    for variant in ("local", "self"):
        for seed in (0, 1):
            epochs = 50 + seed + (10 if variant == "self" else 0)
            metric = {"count_nll": .32 + .001 * seed + (.003 if variant == "self" else 0),
                "raw_count_mae": .21, "raw_count_rmse": 1.15}
            values.append({"run_id": f"{variant}_{seed}", "variant": variant, "seed": seed,
                "final_epoch": epochs, "best_epoch": 10, "metrics": metric,
                "source_hashes": {"file": "a" * 64}, "data_provenance": {"split": "same"},
                "validation_masks": {"mask": "same"}, "training_coverage": [{"epoch": e, "receipt": "same"} for e in range(1, epochs + 1)],
                "optimization_steps": epochs * 209, "baseline": {"pooled": metric.copy()},
                "per_core": {alias: metric.copy() for alias in report.VALIDATION_ALIASES},
                "config": {**{k: {"same": True} for k in ("dataset", "masking", "trainer", "evaluation", "metadata")},
                    "model": {"family": "local_gene_residual", "hidden_dim": 512, "hidden_layers": 2, "dropout": .1, "output_distribution": "hurdle"},
                    "graph": {"kind": "fixed_local_summary" if variant == "local" else "none", "neighbor_k": 16 if variant == "local" else 0, "radius_um": 75., "self_loops": False}},
                "stopping": {"stop_reason": "patience", "early_stopping_triggered": True}})
    for row in values:
        row["baseline"] = deepcopy(values[0]["baseline"])
    return values


HISTORICAL = [{"label": "BAGM hurdle", "count_nll": .3282417527371811, "raw_count_rmse": 1.1684848056663228}]


def test_common_prefix_allows_different_stop_epochs_but_rejects_mask_drift():
    values = rows()
    report.verify_comparison(values)
    values[1]["training_coverage"][10]["receipt"] = "different"
    with pytest.raises(ValueError, match="common prefix"):
        report.verify_comparison(values)


def test_scientific_coverage_requires_all_seeds_and_exact_update_counts():
    with pytest.raises(ValueError, match="Exactly four"):
        report.verify_comparison(rows()[:3])
    values = rows(); values[0]["optimization_steps"] -= 1
    with pytest.raises(ValueError, match="Optimizer count"):
        report.verify_comparison(values)


def test_material_gain_and_local_gain_remain_separate():
    values = rows()
    assert report.assess(values, HISTORICAL)["historical_endpoint_criterion_passed"]
    values[3]["metrics"]["count_nll"] = .310
    result = report.assess(values, HISTORICAL)
    assert result["historical_endpoint_criterion_passed"]
    assert not result["both_seeds_local_beats_self_nll"]
    values[1]["metrics"]["raw_count_rmse"] = 1.19
    assert not report.assess(values, HISTORICAL)["historical_endpoint_criterion_passed"]


def test_incomplete_runs_cannot_create_any_report(tmp_path, monkeypatch):
    paths = SimpleNamespace(report_root=tmp_path / "reports", state_root=tmp_path / "state")
    monkeypatch.setattr(report, "current_paths", lambda: paths)
    monkeypatch.setattr(report, "Registry", lambda *args, **kwargs: None)
    def incomplete(*args):
        raise ValueError("Run is not completed")
    monkeypatch.setattr(report, "load_run", incomplete)
    with pytest.raises(ValueError, match="not completed"):
        report.main(["--run-ids", "a", "b", "c", "d", "--publish-result"])
    assert not paths.report_root.exists()


def synthetic_bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"; root.mkdir()
    def save(name, value):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    history, stop_summary, receipt = stopping_trace([.3 + .001 * abs(e - 20) for e in range(1, 51)])
    metric = {"count_nll": .3, "raw_count_mae": .21, "raw_count_rmse": 1.15, "masked_entries": report.MASKED_ENTRIES}
    lr, best, bad = .001, float("inf"), 0
    for row in history:
        row.update(learning_rate=lr, steps=209, cumulative_steps=row["epoch"]*209)
        if row["val/count_nll"] < best - .0001:
            best, bad = row["val/count_nll"], 0
        else:
            bad += 1
        if bad > 7:
            lr, bad = max(lr*.5, .00001), 0
        row["learning_rate_after_scheduler"] = lr
        row.update(target_visits=208_696, seconds=1.)
        row.update({"val/"+key: value for key, value in metric.items() if key != "count_nll"})
    terminal = {**metric, "count_nll": history[-1]["val/count_nll"]}
    config = {"campaign": {"campaign_id": report.CAMPAIGN}, "seed": 0,
        "metadata": {"profile": "full", "frozen_contract_sha256": "contract"},
        "model": {"variant": "local", "family": "local_gene_residual", "use_local": True, "hidden_dim": 512, "hidden_layers": 2, "dropout": .1, "output_distribution": "bernoulli_hurdle_zero_truncated_nb2"},
        "evaluation": {"primary_metric": "val/count_nll"},
        "trainer": {"epochs": 500, "minimum_epochs": 50, "maximum_epochs": 500, "batch_size": 1024, "learning_rate": .001,
            "minimum_learning_rate": .00001, "weight_decay": .00001, "gradient_clip": 1., "ordering_seed": 2026100401,
            "scheduler": "reduce_on_plateau", "scheduler_factor": .5, "scheduler_patience": 7, "scheduler_threshold": .0001,
            "scheduler_threshold_mode": "abs", "precision": "bfloat16_autocast_fp32_likelihood",
            "early_stopping": {"patience": 30, "min_delta": .0001, "min_epochs": 50, "max_epochs": 500}},
        "masking": {"kind": "target_isolated_uniform_1_1000", "training_base_seed": 2026092601, "validation_namespace": "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"},
        "dataset": {"dataset_fingerprint": "data", "split_fingerprint": "split", "preprocessing_fingerprint": "preprocessing"},
        "launcher": {"requested_gpu": "0"}}
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    summary = {**stop_summary, "run_id": "synthetic_run", "status": "completed", "variant": "local", "seed": 0,
        "primary_metric_name": "val/count_nll", "final_epoch": 50, "checkpoint_reload_verified": True,
        "terminal_checkpoint_reload_verified": True, "terminal_mean_count_nll": sum(r["val/count_nll"] for r in history[-4:]) / 4,
        "parameter_count": 2_329_432, "duration_seconds": 55., "optimization_steps": 50 * 209, "peak_vram_gb": 1.,
        "selected_metrics": metric, "terminal_metrics": terminal, "recent_epoch_ids": [47, 48, 49, 50], "last_learning_rate": lr}
    save("summary.json", summary)
    save("metrics/final.json", {"val/"+k:v for k,v in metric.items()})
    save("metrics/terminal.json", {"val/"+k:v for k,v in terminal.items()})
    save("metrics/recent_epochs.json", {"epochs": [47, 48, 49, 50], "aggregation": "unweighted_epoch_mean_last_up_to_four_complete_epochs",
        "metrics": {k: sum(row["val/"+k] for row in history[-4:])/4 for k in metric}})
    (root / "metrics/epoch_history.jsonl").write_text("\n".join(json.dumps(r) for r in history))
    save("diagnostics/stopping.json", receipt)
    save("provenance/data_fingerprints.json", {"overlay_manifest_content_sha256": "data", "split_fingerprint": "split",
        "preprocessing_fingerprint": "preprocessing", "source_core_checksums_verified": True, "local_self_exclusion": True,
        "training_cells": 208_696, "validation_cells": 37_367})
    supports = dict(zip(sorted(report.VALIDATION_ALIASES), [report.MASKED_ENTRIES//2, report.MASKED_ENTRIES-report.MASKED_ENTRIES//2]))
    save("diagnostics/validation_masks.json", {a: {"masked_entry_count": n, "covers_all_cells": True, "fixed_across_epochs": True} for a,n in supports.items()})
    visits = dict.fromkeys(sorted(report.TRAIN_ALIASES), 208_696//12); visits["SO2-C26"] += 208_696%12
    coverage = [{"epoch": e, "cores": {a: {"target_visits": n, "masked_entries": n*500, "mask_sha256": "a"*64, "order_sha256": "b"*64} for a,n in visits.items()}} for e in range(1,51)]
    save("diagnostics/coverage.json", coverage)
    (root / "diagnostics/coverage.jsonl").write_text("\n".join(json.dumps(r) for r in coverage))
    for name, epoch_key, epoch in (("checkpoint_replay", "best_epoch", 20), ("terminal_checkpoint_replay", "final_epoch", 50)):
        save(f"diagnostics/{name}.json", {"passed": True, "maximum_absolute_delta": 0., epoch_key: epoch, "checkpoint_sha256": "hash"})
    save("diagnostics/preflight.json", {"passed": True, "source_hashes": {"model": "source"}, "frozen_contract_sha256": "contract"})
    save("provenance/source_hashes.json", {"model": "source"})
    for role in ("initial", "selected"):
        save(f"diagnostics/precision_{role}.json", {"passed": True, "absolute_delta": .00001, "tolerance": .001})
    save("results/per_core.json", {a: {**metric, "masked_entries": n} for a,n in supports.items()})
    names = ["1_250", "251_500", "501_750", "751_1000"]
    strata = {name: {**metric, "masked_entries": report.MASKED_ENTRIES//4} for name in names}; strata[names[-1]]["masked_entries"] += report.MASKED_ENTRIES%4
    save("results/mask_strata.json", strata)
    save("diagnostics/resources.json", [{"rss_peak_bytes": 4096}])
    save("diagnostics/train_fitted_constant_baseline.json", {"pooled": metric})
    monkeypatch.setattr(report, "_verified_root", lambda *args: (root, {"campaign_id": report.CAMPAIGN}))
    monkeypatch.setattr(report, "sha256_file", lambda *args: "hash")
    registry = SimpleNamespace(list_checkpoint_catalog=lambda **kwargs: [{"path": "checkpoints/"+name, "sha256": "hash"} for name in ("best.ckpt", "last.ckpt")])
    return root, save, registry


def test_loader_verifies_real_patience_and_both_checkpoint_receipts(tmp_path, monkeypatch):
    root, save, registry = synthetic_bundle(tmp_path, monkeypatch)
    loaded = report.load_run("synthetic_run", object(), registry)
    assert loaded["best_epoch"] == 20 and loaded["final_epoch"] == 50
    assert loaded["stopping"]["early_stopping_triggered"]
    save("diagnostics/terminal_checkpoint_replay.json", {"passed": False, "maximum_absolute_delta": 1.})
    with pytest.raises(ValueError, match="Terminal-checkpoint replay"):
        report.load_run("synthetic_run", object(), registry)


def test_loader_rejects_missing_catalog_and_recent_metric_drift(tmp_path, monkeypatch):
    root, save, registry = synthetic_bundle(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="catalog"):
        report.load_run("synthetic_run", object(), SimpleNamespace(list_checkpoint_catalog=lambda **kwargs: []))
    recent = json.loads((root/"metrics/recent_epochs.json").read_text()); recent["metrics"]["count_nll"] += .01
    save("metrics/recent_epochs.json", recent)
    with pytest.raises(ValueError, match="Recent-epoch metric"):
        report.load_run("synthetic_run", object(), registry)


def test_curated_record_validates_and_preserves_cap_uncertainty(tmp_path):
    values = rows(); values[0]["metrics"]["count_nll"] = .34
    values[0]["stopping"] = {"stop_reason": "maximum_epochs", "early_stopping_triggered": False}
    analysis = {"assessment": report.assess(values, HISTORICAL), "runs": values, "failed_attempts": [], "git_commit": "a"*40}
    source = tmp_path / "source.json"; source.write_text("{}")
    sources = [{"kind": "report", "root": "report", "path": "source.json", "sha256": report.sha256_file(source), "verification_status": "verified"}]
    record = report.result_record(analysis, sources)
    assert record["outcome"] == "inconclusive"
    paths = replace(current_paths(), report_root=tmp_path, result_root=tmp_path/"results")
    directory = paths.result_root / "predictive_benchmark/local_gene_residual" / report.RESULT_ID
    directory.mkdir(parents=True)
    (directory/"README.md").write_text("This is the audited synthetic result.")
    (directory/"result.yaml").write_text(yaml.safe_dump(record))
    validate_result_manifest(directory/"result.yaml", paths=paths, verify_sources=True)
    assert "no sealed test" in record["conclusion"]["remaining_uncertainty"]


def test_loader_rejects_lr_or_update_history_drift(tmp_path, monkeypatch):
    root, save, registry = synthetic_bundle(tmp_path, monkeypatch)
    path = root/"metrics/epoch_history.jsonl"
    history = [json.loads(line) for line in path.read_text().splitlines()]
    history[30]["learning_rate_after_scheduler"] = .01
    path.write_text("\n".join(json.dumps(row) for row in history))
    with pytest.raises(ValueError, match="Learning-rate history"):
        report.load_run("synthetic_run", object(), registry)

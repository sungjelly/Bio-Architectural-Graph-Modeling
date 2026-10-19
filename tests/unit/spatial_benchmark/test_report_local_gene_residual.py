from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from scripts.analysis import report_local_gene_residual as report


def rows():
    output = []
    for variant in ("local", "self"):
        for seed in (0, 1):
            nll = .320 + .001 * seed + (.003 if variant == "self" else 0)
            metric = {"count_nll": nll, "raw_count_mae": .21, "raw_count_rmse": 1.15}
            output.append({
                "run_id": f"{variant}_{seed}", "variant": variant, "seed": seed,
                "metrics": metric, "per_core": {alias: deepcopy(metric) for alias in report.VALIDATION_ALIASES},
                "source_hashes": {"model.py": "a" * 64}, "data_provenance": {"split": "b" * 64},
                "validation_masks": {"SO2-C27": "c" * 64}, "training_coverage_sha256": "d" * 64,
                "optimization_steps": 32 * 209, "baseline": {"pooled": {"count_nll": .39, "raw_count_mae": .3, "raw_count_rmse": 1.5}},
                "config": {**{key: {"same": True} for key in ("dataset", "graph", "masking", "trainer", "evaluation", "metadata")},
                    "graph": {"kind": "fixed_local_summary" if variant == "local" else "none",
                              "neighbor_k": 16 if variant == "local" else 0, "radius_um": 75.0, "self_loops": False},
                    "model": {"family": "local_gene_residual", "hidden_dim": 512, "hidden_layers": 2,
                              "dropout": .1, "output_distribution": "hurdle"}},
            })
    return output


HISTORICAL = [{"label": "BAGM hurdle", "count_nll": .3282417527371811, "raw_count_rmse": 1.1684848056663228}]


def test_assessment_keeps_endpoint_and_local_gain_separate():
    observed = report.assess(rows(), HISTORICAL)
    assert observed["historical_endpoint_criterion_passed"]
    assert observed["both_seeds_local_beats_self_nll"]
    assert observed["local_mean_selected_count_nll"] == pytest.approx(.3205)
    changed = rows()
    changed[3]["metrics"]["count_nll"] = .310
    observed = report.assess(changed, HISTORICAL)
    assert observed["historical_endpoint_criterion_passed"]
    assert not observed["both_seeds_local_beats_self_nll"]


def test_good_mean_cannot_hide_one_failing_seed():
    changed = rows()
    changed[0]["metrics"]["count_nll"] = .300
    changed[1]["metrics"]["count_nll"] = .340
    observed = report.assess(changed, HISTORICAL)
    assert observed["local_mean_nll_margin_passed"]
    assert not observed["both_local_seeds_improve_historical_nll"]
    assert not observed["historical_endpoint_criterion_passed"]


def test_rmse_guard_is_per_seed_and_thresholds_are_prespecified():
    changed = rows()
    changed[1]["metrics"]["raw_count_rmse"] = 1.19
    observed = report.assess(changed, HISTORICAL)
    assert observed["nll_margin_threshold"] == .3262417527
    assert observed["rmse_upper_threshold"] == 1.1801696537
    assert not observed["both_local_seeds_rmse_guard_passed"]
    assert not observed["historical_endpoint_criterion_passed"]
    changed[1]["metrics"]["raw_count_rmse"] = report.RMSE_THRESHOLD
    assert report.assess(changed, HISTORICAL)["historical_endpoint_criterion_passed"]


def test_comparison_accepts_complete_paired_experiment():
    report.verify_comparison(rows())


def test_comparison_requires_variant_specific_local_or_no_graph():
    changed = rows()
    changed[2]["config"]["graph"]["neighbor_k"] = 16
    with pytest.raises(ValueError, match="Variant-specific local graph"):
        report.verify_comparison(changed)
    changed = rows()
    changed[0]["config"]["graph"]["radius_um"] = 100.0
    with pytest.raises(ValueError, match="Variant-specific local graph"):
        report.verify_comparison(changed)


@pytest.mark.parametrize("field", ["source_hashes", "data_provenance", "validation_masks", "training_coverage_sha256"])
def test_comparison_rejects_unpaired_sources_and_masks(field):
    changed = rows()
    changed[1][field] = "changed"
    with pytest.raises(ValueError, match="paired comparison field"):
        report.verify_comparison(changed)


def test_comparison_rejects_missing_or_duplicated_seeds_and_different_schedule():
    with pytest.raises(ValueError, match="Exactly four"):
        report.verify_comparison(rows()[:3])
    changed = rows()
    changed[1]["seed"] = 0
    with pytest.raises(ValueError, match="crossed with seeds"):
        report.verify_comparison(changed)
    changed = rows()
    changed[3]["config"]["trainer"]["epochs"] = 64
    with pytest.raises(ValueError, match="scientific configuration"):
        report.verify_comparison(changed)


def test_report_refuses_incomplete_runs_before_creating_output(tmp_path, monkeypatch):
    paths = SimpleNamespace(report_root=tmp_path / "reports", state_root=tmp_path / "state")
    monkeypatch.setattr(report, "current_paths", lambda: paths)
    monkeypatch.setattr(report, "Registry", lambda *args, **kwargs: object())
    def incomplete(*args, **kwargs):
        raise ValueError("Run is not completed")
    monkeypatch.setattr(report, "load_run", incomplete)
    with pytest.raises(ValueError, match="not completed"):
        report.main(["--run-ids", "a", "b", "c", "d", "--version", "v1"])
    assert not paths.report_root.exists()


def test_render_preserves_exploratory_and_metric_semantics():
    values = rows()
    for row in values:
        row.update(best_epoch=20, final_epoch=32, terminal_mean_count_nll=.321,
            parameter_count=2_327_288, duration_seconds=1800., mean_epoch_seconds=50.,
            peak_vram_gib=1.2, peak_rss_gib=6.)
        row["metrics"].update(log1p_count_mae=.15, positive_marginal_raw_count_mae=1.3,
            positive_conditional_raw_count_mae=.9, detection_bce=.23, detection_average_precision=.32)
    historical = [{**HISTORICAL[0], "run_id": "historical_run", "best_epoch": 164,
        "final_epoch": 171, "raw_count_mae": .226, "parameter_count": 6_160_088,
        "duration_seconds": 10158.0}]
    text = report.render_report({"runs": values, "historical": historical,
        "assessment": report.assess(values, historical), "failed_attempts": [],
        "zero_predictor_mae": .151250518})
    assert "not an independent or confirmatory study" in text
    assert "dispersion fixed at θ=1" in text
    assert "maximum" in text and "numpy.array_split" in text
    assert "Observed-positive marginal errors" in text
    assert all(f"`{row['run_id']}`" in text for row in values)


def test_load_run_checks_complete_history_and_strict_checkpoint_selection(tmp_path, monkeypatch):
    """Exercise the scientific report contract with synthetic aggregate files."""
    root = tmp_path / "synthetic_run"
    root.mkdir()
    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    config = {
        "campaign": {"campaign_id": report.CAMPAIGN}, "seed": 0,
        "metadata": {"profile": "full", "frozen_contract_sha256": "contract"},
        "model": {"variant": "local", "family": "local_gene_residual", "use_local": True,
            "hidden_dim": 512, "hidden_layers": 2, "dropout": .1,
            "output_distribution": "bernoulli_hurdle_zero_truncated_nb2"},
        "evaluation": {"primary_metric": "val/count_nll"},
        "trainer": {"epochs": 32, "batch_size": 1024, "learning_rate": .001,
            "minimum_learning_rate": .0001, "weight_decay": .00001, "gradient_clip": 1.,
            "ordering_seed": 2026100401, "scheduler": "fixed_cosine",
            "precision": "bfloat16_autocast_fp32_likelihood"},
        "masking": {"kind": "target_isolated_uniform_1_1000", "training_base_seed": 2026092601,
            "validation_namespace": "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"},
        "dataset": {"dataset_fingerprint": "data", "split_fingerprint": "split", "preprocessing_fingerprint": "preprocessing"},
        "launcher": {"requested_gpu": "0"},
    }
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    metric = {"count_nll": .3, "raw_count_mae": .21, "raw_count_rmse": 1.15,
        "masked_entries": report.MASKED_ENTRIES}
    history = [{"epoch": epoch, "target_visits": 208_696, "seconds": 1.,
        **{"val/" + key: value for key, value in metric.items()},
        "val/count_nll": .3 + .001 * abs(epoch - 20)} for epoch in range(1, 33)]
    summary = {"run_id": "synthetic_run", "status": "completed", "variant": "local", "seed": 0,
        "primary_metric_name": "val/count_nll", "primary_metric_value": .3,
        "final_epoch": 32, "best_epoch": 20, "checkpoint_reload_verified": True,
        "terminal_mean_count_nll": .3105, "parameter_count": 2000, "duration_seconds": 35.,
        "optimization_steps": 6688, "peak_vram_gb": 1.}
    save("summary.json", summary)
    save("metrics/final.json", {"val/" + key: value for key, value in metric.items()})
    (root / "metrics/epoch_history.jsonl").write_text("\n".join(json.dumps(row) for row in history))
    save("provenance/data_fingerprints.json", {"overlay_manifest_content_sha256": "data",
        "split_fingerprint": "split", "preprocessing_fingerprint": "preprocessing",
        "source_core_checksums_verified": True, "local_self_exclusion": True,
        "training_cells": 208_696, "validation_cells": 37_367})
    supports = {"SO2-C27": report.MASKED_ENTRIES // 2,
                "SO2-C28": report.MASKED_ENTRIES - report.MASKED_ENTRIES // 2}
    save("diagnostics/validation_masks.json", {alias: {"masked_entry_count": count,
        "covers_all_cells": True, "fixed_across_epochs": True} for alias, count in supports.items()})
    visits = dict.fromkeys(sorted(report.TRAIN_ALIASES), 208_696 // 12)
    visits["SO2-C26"] += 208_696 % 12
    save("diagnostics/coverage.json", [{"epoch": epoch, "cores": {alias: {
        "target_visits": count, "masked_entries": count * 500, "mask_sha256": "a" * 64}
        for alias, count in visits.items()}} for epoch in range(1, 33)])
    save("diagnostics/checkpoint_replay.json", {"passed": True, "maximum_absolute_delta": 0.,
        "best_epoch": 20, "checkpoint_sha256": "synthetic_checksum"})
    save("diagnostics/preflight.json", {"passed": True, "source_hashes": {"model": "source"},
        "frozen_contract_sha256": "contract"})
    save("provenance/source_hashes.json", {"model": "source"})
    for role in ("initial", "selected"):
        save(f"diagnostics/precision_{role}.json", {"passed": True, "absolute_delta": .00001, "tolerance": .001})
    save("results/per_core.json", {alias: {**metric, "masked_entries": count} for alias, count in supports.items()})
    names = ["1_250", "251_500", "501_750", "751_1000"]
    strata = {name: {**metric, "masked_entries": report.MASKED_ENTRIES // 4} for name in names}
    strata[names[-1]]["masked_entries"] += report.MASKED_ENTRIES % 4
    save("results/mask_strata.json", strata)
    save("diagnostics/resources.json", [{"rss_peak_bytes": 4096}])
    save("diagnostics/train_fitted_constant_baseline.json", {"pooled": metric})
    monkeypatch.setattr(report, "_verified_root", lambda *args: (root, {"campaign_id": report.CAMPAIGN}))
    monkeypatch.setattr(report, "sha256_file", lambda *args: "synthetic_checksum")
    loaded = report.load_run("synthetic_run", object(), object())
    assert loaded["best_epoch"] == 20
    assert loaded["terminal_mean_count_nll"] == pytest.approx(.3105)
    summary["best_epoch"] = 19
    save("summary.json", summary)
    with pytest.raises(ValueError, match="strict earliest validation NLL minimum"):
        report.load_run("synthetic_run", object(), object())

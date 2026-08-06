from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import statistics
import sys
from typing import Any, Mapping, Sequence

import pytest
import yaml

from spatial_benchmark.configuration import compose_config


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "analysis" / "compare_g2_token_multiseed.py"
_SPEC = importlib.util.spec_from_file_location(
    "compare_g2_token_multiseed_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

G2TokenComparisonError = _MODULE.G2TokenComparisonError
compare_g2_token_multiseed = _MODULE.compare_g2_token_multiseed
write_comparison = _MODULE.write_comparison

_BASELINE = "g2_tokenized_width512_exact_k1000_full_core"
_WIDER = "g2_tokenized_width1024_exact_k1000_full_core"
_PILOT = "g2_tokenized_width1024_exact_k1000_resource_pilot"
_CONFIGS = {
    _BASELINE: "configs/experiment/full_core_g2_tokenized_multiseed_baseline.yaml",
    _WIDER: "configs/experiment/full_core_g2_tokenized_multiseed_width1024.yaml",
    _PILOT: (
        "configs/experiment/"
        "full_core_g2_tokenized_multiseed_width1024_resource_pilot.yaml"
    ),
}
_TOKENS = {
    _BASELINE: "6dda5587",
    _WIDER: "56a25fc4",
    _PILOT: "d86b9849",
}
_PARAMETERS = {
    _BASELINE: 7_062_880,
    _WIDER: 18_834_784,
    _PILOT: 18_834_784,
}
_SUPPORTS = (9_000, 400, 350, 250)
_OUTPUTS = {
    ".g2-token-comparison-owner.json",
    "comparison.json",
    "per_run.csv",
    "paired.csv",
    "report.md",
}


def _checksum(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _configuration(
    variant: str,
    *,
    seed: int,
    attempt: int = 1,
) -> dict[str, Any]:
    config = deepcopy(compose_config(_CONFIGS[variant]))
    config["seed"] = seed
    config["fold"] = 0
    config["attempt"] = attempt
    return config


def _run_id(
    variant: str,
    seed: int,
    *,
    attempt: int = 1,
    suffix: str = "synthetic",
) -> str:
    return (
        f"r_20260726T120000Z_{_TOKENS[variant]}_s{seed:03d}_f00_"
        f"a{attempt:02d}_{suffix}"
    )


def _token_audit() -> dict[str, Any]:
    modes = [0] * 1_000
    n_entries = 24_245_000
    counts = [22_254_214, 1_099_754, 649_566, 241_466]
    prevalence = [value / n_entries for value in counts]
    return {
        "spec": deepcopy(_MODULE._TOKEN_SPEC),
        "shape": [24_245, 1_000],
        "n_cells": 24_245,
        "n_genes": 1_000,
        "n_entries": n_entries,
        "token_counts": counts,
        "token_prevalence": prevalence,
        "token_prevalence_percent": [100.0 * value for value in prevalence],
        "all_output_tokens_present": True,
        "gene_class_coverage": {
            "classes_present_per_gene": [4] * 1_000,
            "genes_with_all_output_tokens": 1_000,
            "genes_with_all_output_tokens_percent": 100.0,
            "all_genes_have_all_output_tokens": True,
            "genes_containing_each_output_token": [1_000] * 4,
            "gene_coverage_percent_by_output_token": [100.0] * 4,
        },
        "token_checksum_sha256": _MODULE._EXPECTED_TOKEN_MATRIX_SHA256,
        "per_gene_modal_tokens": modes,
        "per_gene_modal_tokens_checksum_sha256": (
            _MODULE._int64_vector_sha256(modes)
        ),
        "thresholds_fitted": False,
        "fit_scope": "all_nodes_transductive",
        "input_mask_token_is_output_class": False,
    }


def _token_row(
    *,
    mode: str,
    replicate: int,
    diagonal: Sequence[int],
    mask_label: str,
) -> dict[str, Any]:
    confusion = [[0] * 4 for _ in range(4)]
    for token, support in enumerate(_SUPPORTS):
        confusion[token][token] = diagonal[token]
        confusion[token][(token + 1) % 4] = support - diagonal[token]
    recalls = [
        100.0 * diagonal[token] / _SUPPORTS[token] for token in range(4)
    ]
    exact = 100.0 * sum(diagonal) / sum(_SUPPORTS)
    balanced = statistics.fmean(recalls)
    nonzero = 100.0 * sum(diagonal[1:]) / sum(_SUPPORTS[1:])
    empirical = 100.0 * sum(
        (support / sum(_SUPPORTS)) ** 2 for support in _SUPPORTS
    )
    row: dict[str, Any] = {
        "split": "fit",
        "mask_mode": mode,
        "mask_replicate": replicate,
        "mask_entry_id": f"fit-{mode}-{replicate}",
        "mask_seed": 30_000 + replicate,
        "mask_checksum": _checksum(f"{mask_label}-{mode}-{replicate}"),
        "n_masked": sum(_SUPPORTS),
        "masked_token_cross_entropy": 1.0 - exact / 200.0,
        "masked_token_accuracy_percent": exact,
        "masked_token_balanced_accuracy_percent": balanced,
        "masked_nonzero_token_accuracy_percent": nonzero,
        "baseline_uniform_accuracy_percent": 25.0,
        "baseline_empirical_frequency_accuracy_percent": empirical,
        "baseline_always_zero_accuracy_percent": 90.0,
        "baseline_always_zero_balanced_accuracy_percent": 25.0,
        "baseline_per_gene_modal_accuracy_percent": 91.8,
        "baseline_per_gene_modal_balanced_accuracy_percent": 28.3,
        "baseline_per_gene_modal_nonzero_accuracy_percent": 1.5,
    }
    for token in range(4):
        row[f"token_{token}_support"] = _SUPPORTS[token]
        row[f"token_{token}_recall_percent"] = recalls[token]
        for predicted in range(4):
            row[f"confusion_{token}_{predicted}"] = confusion[token][predicted]
    return row


def _evaluation_rows(
    *,
    replicates: int,
    diagonal: Sequence[int],
    mask_label: str,
) -> list[dict[str, Any]]:
    return [
        _token_row(
            mode=mode,
            replicate=replicate,
            diagonal=diagonal,
            mask_label=mask_label,
        )
        for mode in ("partial_gene", "whole_node", "spatial_block")
        for replicate in range(replicates)
    ]


def _history_rows(
    *,
    run_id: str,
    seed: int,
    epochs: int,
    schedule_drift: bool,
    peak_vram_gb: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        label = f"epoch-{seed}-{epoch}"
        if schedule_drift and epoch == epochs - 1:
            label += "-drift"
        rows.append(
            {
                "run_id": run_id,
                "split": "fit",
                "training_protocol": (
                    "held_in_full_core_fixed_budget_token_classification"
                ),
                "epoch": epoch,
                "mask_mode": ("partial", "node", "block")[epoch % 3],
                "mask_seed": 10_000 + seed * 1_000 + epoch,
                "mask_checksum": _checksum(label),
                "edge_dropout_seed": 20_000 + seed * 1_000 + epoch,
                "edge_checksum": _checksum(f"edge-{epoch}"),
                "n_masked_entries": 10_000 + epoch,
                "n_target_nodes": 100 + epoch,
                "n_edges_used": 21_029_944,
                "train_loss": 1.2 - 0.001 * epoch,
                "diagnostic_loss": None,
                "gradient_norm": 0.5,
                "duration_seconds": 0.1,
                "peak_cuda_memory_bytes": round(peak_vram_gb * 1024**3),
            }
        )
    return rows


def _slope(values: Sequence[float]) -> float:
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / sum((index - x_mean) ** 2 for index in range(len(values)))


def _make_bundle(
    root: Path,
    *,
    variant: str,
    seed: int,
    diagonal: Sequence[int],
    suffix: str = "synthetic",
    schedule_drift: bool = False,
    peak_vram_gb: float | None = None,
    training_duration_seconds: float | None = None,
    token_spec_drift: bool = False,
) -> Path:
    pilot = variant == _PILOT
    epochs = 2 if pilot else 200
    replicates = 1 if pilot else 3
    run_id = _run_id(variant, seed, suffix=suffix)
    directory = root / "artifacts" / run_id
    directory.mkdir(parents=True)
    (directory / "_SUCCESS").write_text("", encoding="utf-8")
    config = _configuration(variant, seed=seed)
    (directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        directory / "provenance/queue.json",
        {
            "job_id": f"q_{_TOKENS[variant]}_{suffix}_{seed}",
            "attempt": 1,
            "retry_of": None,
        },
    )
    audit = _token_audit()
    if token_spec_drift:
        audit["spec"]["output_tokens"][3]["label"] = "drifted"
    _write_json(directory / "diagnostics/expression_tokenization.json", audit)
    history = _history_rows(
        run_id=run_id,
        seed=seed,
        epochs=epochs,
        schedule_drift=schedule_drift,
        peak_vram_gb=(
            peak_vram_gb
            if peak_vram_gb is not None
            else (8.7 if pilot else (8.0 if variant == _BASELINE else 9.0))
        ),
    )
    evaluations = _evaluation_rows(
        replicates=replicates,
        diagonal=diagonal,
        mask_label="shared-mask",
    )
    internal_modes = {
        "partial_gene": "partial",
        "whole_node": "node",
        "spatial_block": "block",
    }
    mask_entries = []
    for spec_index, row in enumerate(evaluations):
        mode = str(row["mask_mode"])
        internal_mode = internal_modes[mode]
        mask_entries.append(
            {
                "entry_id": row["mask_entry_id"],
                "split": "fit",
                "spec_index": spec_index // replicates,
                "spec_id": mode.replace("_", "-"),
                "spec": {
                    "mode": internal_mode,
                    "label": mode,
                    "partial_gene_rate": 0.2,
                    "node_rate": 0.1,
                    "block_node_rate": 0.1,
                    "block_width_um": None,
                    "block_shape": "disk",
                },
                "replicate": row["mask_replicate"],
                "seed": row["mask_seed"],
                "shape": [24_245, 1_000],
                "mask_checksum": row["mask_checksum"],
                "summary": {
                    "mode": internal_mode,
                    "seed": row["mask_seed"],
                    "shape": [24_245, 1_000],
                    "n_masked_entries": row["n_masked"],
                    "n_eligible_nodes": 24_245,
                    "n_selected_nodes": 2_425,
                },
            }
        )
    mask_manifest: dict[str, Any] = {
        "format_version": 1,
        "base_seed": 12345,
        "n_genes": 1_000,
        "replicates": replicates,
        "splits": {
            "fit": {
                "coordinates_checksum": _checksum("coordinates"),
                "eligible_nodes_checksum": _checksum("eligible"),
                "n_eligible_nodes": 24_245,
                "n_nodes": 24_245,
            }
        },
        "entries": mask_entries,
        "seed_contract": (
            "Mask seeds derive only from base_seed, split, specification, "
            "and replicate; model seeds are excluded."
        ),
    }
    bundle_checksum = _MODULE._digest(mask_manifest)
    mask_manifest["bundle_checksum"] = bundle_checksum
    mask_manifest["bundle_id"] = bundle_checksum[:16]
    _write_json(
        directory / "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": mask_manifest,
            "seed_namespace": "synthetic separate seed derivation",
            "used_for_checkpoint_selection": False,
            "used_for_gradient_updates": False,
        },
    )
    _write_jsonl(directory / "metrics/history.jsonl", history)
    _write_jsonl(directory / "metrics/evaluation_replicates.jsonl", evaluations)
    whole = [row for row in evaluations if row["mask_mode"] == "whole_node"]
    mean_fields = (
        "masked_token_cross_entropy",
        "masked_token_accuracy_percent",
        "masked_token_balanced_accuracy_percent",
        "masked_nonzero_token_accuracy_percent",
        *_MODULE._BASELINE_FIELDS,
        *(f"token_{token}_recall_percent" for token in range(4)),
    )
    final: dict[str, Any] = {
        f"fit/whole_node/{field}": statistics.fmean(
            float(row[field]) for row in whole
        )
        for field in mean_fields
    }
    summed_duration = sum(float(row["duration_seconds"]) for row in history)
    training_duration = (
        training_duration_seconds
        if training_duration_seconds is not None
        else summed_duration + 1.0
    )
    total_duration = training_duration + 5.0
    peak = max(int(row["peak_cuda_memory_bytes"]) for row in history) / 1024**3
    final.update(
        {
            "resource/parameter_count": _PARAMETERS[variant],
            "resource/training_duration_seconds": training_duration,
            "resource/total_duration_seconds": total_duration,
            "resource/peak_vram_gb": peak,
        }
    )
    _write_json(directory / "metrics/final.json", final)
    losses = [float(row["train_loss"]) for row in history]
    _write_json(
        directory / "diagnostics/training_convergence.json",
        {
            "final_epoch": epochs - 1,
            "final_train_loss": losses[-1],
            "minimum_observed_train_loss": min(losses),
            "last_20_epoch_loss_slope": _slope(losses[-min(20, epochs) :]),
            "all_epochs_completed": True,
            "all_losses_and_gradients_finite": True,
            "objective": "masked_token_cross_entropy",
        },
    )
    exact = float(final["fit/whole_node/masked_token_accuracy_percent"])
    summary = {
        "run_id": run_id,
        "status": "success",
        "training_exit_status": "success",
        "evaluation_protocol": "held_in_full_core_fixed_budget",
        "task_family": "masked_expression_token_classification",
        "model_name": "g2-tokenized",
        "model_seed": seed,
        "generalization_estimate": False,
        "diagnostic_resource_pilot": pilot,
        "conclusion_eligible": not pilot,
        "evaluation_mask_replicates_per_mode": replicates,
        "evaluation_metrics_include_all_configured_replicates_per_mode": True,
        "fixed_epoch_budget": epochs,
        "final_epoch": epochs - 1,
        "primary_metric_name": (
            "fit/whole_node/masked_token_accuracy_percent"
        ),
        "primary_metric_value": exact,
        "metrics": final,
        "parameter_count": _PARAMETERS[variant],
        "duration_seconds": total_duration,
        "peak_vram_gb": peak,
        "graph_sha256": _MODULE._EXPECTED_GRAPH_SHA256,
        "graph_directed_edges": 21_029_944,
        "evaluation_mask_bundle_sha256": bundle_checksum,
    }
    _write_json(directory / "summary.json", summary)
    return directory


def _six_runs(
    root: Path,
    *,
    baseline_diagonal: Sequence[int] = (8_700, 80, 70, 50),
    wider_diagonal: Sequence[int] = (8_800, 100, 100, 75),
    baseline_by_seed: Mapping[int, Sequence[int]] | None = None,
    wider_by_seed: Mapping[int, Sequence[int]] | None = None,
    drift_variant: str | None = None,
    drift_seed: int = 2,
    schedule_drift: bool = False,
    token_spec_drift: bool = False,
) -> list[Path]:
    grouped: dict[str, dict[int, Path]] = {_BASELINE: {}, _WIDER: {}}
    for variant, diagonal in (
        (_BASELINE, baseline_diagonal),
        (_WIDER, wider_diagonal),
    ):
        for seed in (0, 1, 2):
            targeted = variant == drift_variant and seed == drift_seed
            selected_diagonal = (
                (baseline_by_seed or {}).get(seed, diagonal)
                if variant == _BASELINE
                else (wider_by_seed or {}).get(seed, diagonal)
            )
            grouped[variant][seed] = _make_bundle(
                root,
                variant=variant,
                seed=seed,
                diagonal=selected_diagonal,
                suffix=f"{variant[-8:]}-{seed}",
                schedule_drift=schedule_drift and targeted,
                token_spec_drift=token_spec_drift and targeted,
            )
    return [
        grouped[_WIDER][2],
        grouped[_BASELINE][0],
        grouped[_WIDER][0],
        grouped[_BASELINE][2],
        grouped[_BASELINE][1],
        grouped[_WIDER][1],
    ]


def _registry(
    root: Path,
    bundles: Sequence[Path],
    *,
    name: str = "registry.sqlite3",
    drop_run_id: str | None = None,
    add_extra: bool = False,
    failed_run_id: str | None = None,
) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE queue_jobs (
            job_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            run_id TEXT,
            status TEXT NOT NULL,
            failure_category TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    records: list[tuple[Any, ...]] = []
    for index, bundle in enumerate(bundles):
        run_id = str(json.loads((bundle / "summary.json").read_text())["run_id"])
        if run_id == drop_run_id:
            continue
        job_id = str(
            json.loads((bundle / "provenance/queue.json").read_text())["job_id"]
        )
        failed = run_id == failed_run_id
        records.append(
            (
                job_id,
                _MODULE._CAMPAIGN_ID,
                run_id,
                "failed" if failed else "completed",
                "synthetic_failure" if failed else None,
                "synthetic last error" if failed else None,
                f"2026-07-26T12:00:{index:02d}Z",
            )
        )
    if add_extra:
        records.append(
            (
                "q_unexpected_extra",
                _MODULE._CAMPAIGN_ID,
                "r_20260726T120100Z_deadbeef_s000_f00_a01_extra",
                "completed",
                None,
                None,
                "2026-07-26T12:01:00Z",
            )
        )
    connection.executemany(
        """
        INSERT INTO queue_jobs(
            job_id, campaign_id, run_id, status, failure_category,
            last_error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture(autouse=True)
def _accept_synthetic_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "verify_run_bundle",
        lambda *_args, **_kwargs: {"status": "success"},
    )


def test_full_comparison_exports_percentages_baselines_and_csv(
    tmp_path: Path,
) -> None:
    runs = _six_runs(tmp_path)
    pilot = _make_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )

    result = compare_g2_token_multiseed(
        runs,
        resource_pilot=pilot,
        registry=_registry(tmp_path, [*runs, pilot]),
    )

    assert result["status"] == "complete"
    assert result["h1_width_gate"]["passes"] is True
    assert result["relaxed_jacobian_gate"]["status"] == "skipped"
    assert result["resource_pilot"]["status"] == "passed"
    assert result["registry_reconciliation"]["status"] == "passed"
    assert result["registry_reconciliation"]["observed_job_count"] == 7
    assert result["scope"]["evaluation_entries_may_overlap_training_masks"] is True
    assert (
        result["scope"]["per_gene_modal_baseline_scope"]
        == "all_fit_transductive_reference"
    )
    baseline = result["variant_aggregates"][_BASELINE]
    assert baseline["baselines"]["baseline_uniform_accuracy_percent"]["mean"] == 25.0
    assert set(baseline["token_recalls_percent"]) == {"0", "1", "2", "3"}
    assert len(result["runs"]) == 6
    assert len(result["whole_node_masks"]) == 18
    assert len(result["paired_seed_deltas"]["per_seed"]) == 3

    output = write_comparison(result, tmp_path / "comparison")
    assert {path.name for path in output.iterdir()} == _OUTPUTS
    assert len((output / "per_run.csv").read_text().splitlines()) == 7
    assert len((output / "paired.csv").read_text().splitlines()) == 4
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "held-in categorical reconstruction" in report
    assert "all-zero exact baseline" in report
    assert "Per-mask whole-node results" in report
    assert "Resources and convergence" in report
    with pytest.raises(
        G2TokenComparisonError,
        match="must be under reports",
    ):
        write_comparison(result, _ROOT / "artifacts" / "runs" / "forbidden")


def test_all_wider_seeds_above_95_require_conditional_analysis(
    tmp_path: Path,
) -> None:
    runs = _six_runs(
        tmp_path,
        baseline_diagonal=(8_900, 200, 180, 150),
        wider_diagonal=(8_990, 350, 300, 230),
    )
    pilot = _make_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )

    result = compare_g2_token_multiseed(
        runs,
        resource_pilot=pilot,
        registry=_registry(tmp_path, [*runs, pilot]),
    )

    assert result["status"] == "conditional_analysis_required"
    assert result["relaxed_jacobian_gate"]["status"] == "required"
    assert result["relaxed_jacobian_gate"]["jacobians_computed"] is False
    assert "Conditional analysis required" in result["relaxed_jacobian_gate"]["reason"]


def test_exact_95_and_one_lower_seed_keep_jacobian_skipped_and_mixed_h1_fails(
    tmp_path: Path,
) -> None:
    exactly_95 = (8_950, 250, 180, 120)
    runs = _six_runs(
        tmp_path / "boundary",
        baseline_diagonal=(8_700, 80, 70, 50),
        wider_diagonal=exactly_95,
    )
    pilot = _make_bundle(
        tmp_path / "boundary",
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    boundary = compare_g2_token_multiseed(
        runs,
        resource_pilot=pilot,
        registry=_registry(
            tmp_path / "boundary",
            [*runs, pilot],
        ),
    )
    assert boundary["relaxed_jacobian_gate"]["status"] == "skipped"
    assert set(
        boundary["relaxed_jacobian_gate"]["wider_seed_values_percent"].values()
    ) == {95.0}

    mixed_runs = _six_runs(
        tmp_path / "mixed",
        baseline_diagonal=(8_700, 80, 70, 50),
        wider_diagonal=(8_850, 150, 130, 90),
        wider_by_seed={2: (8_600, 70, 60, 40)},
    )
    mixed_pilot = _make_bundle(
        tmp_path / "mixed",
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    mixed = compare_g2_token_multiseed(
        mixed_runs,
        resource_pilot=mixed_pilot,
        registry=_registry(
            tmp_path / "mixed",
            [*mixed_runs, mixed_pilot],
        ),
    )
    assert mixed["h1_width_gate"]["criteria"][
        "no_seed_decreases_on_exact_accuracy"
    ] is False
    assert mixed["h1_width_gate"]["criteria"][
        "no_seed_decreases_on_balanced_accuracy"
    ] is False
    assert mixed["h1_width_gate"]["passes"] is False
    assert mixed["relaxed_jacobian_gate"]["status"] == "skipped"


def test_duplicate_seed_and_alignment_drift_are_rejected(tmp_path: Path) -> None:
    runs = _six_runs(tmp_path)
    pilot = _make_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    duplicate = _make_bundle(
        tmp_path,
        variant=_WIDER,
        seed=2,
        diagonal=(8_800, 100, 100, 75),
        suffix="duplicate-seed",
    )
    replaced = [duplicate if path == runs[-1] else path for path in runs]
    with pytest.raises(G2TokenComparisonError, match="duplicates seed"):
        compare_g2_token_multiseed(
            replaced,
            resource_pilot=pilot,
            registry=_registry(
                tmp_path,
                [*runs, pilot],
                name="duplicate-registry.sqlite3",
            ),
        )

    drifted = _six_runs(
        tmp_path / "schedule",
        drift_variant=_WIDER,
        schedule_drift=True,
    )
    drift_pilot = _make_bundle(
        tmp_path / "schedule",
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    with pytest.raises(G2TokenComparisonError, match="schedules are not paired"):
        compare_g2_token_multiseed(
            drifted,
            resource_pilot=drift_pilot,
            registry=_registry(
                tmp_path / "schedule",
                [*drifted, drift_pilot],
            ),
        )


def test_vocabulary_drift_and_failed_pilot_gate_are_visible(
    tmp_path: Path,
) -> None:
    drifted = _six_runs(
        tmp_path / "vocabulary",
        drift_variant=_WIDER,
        token_spec_drift=True,
    )
    pilot = _make_bundle(
        tmp_path / "vocabulary",
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    with pytest.raises(G2TokenComparisonError, match="token vocabulary drifted"):
        compare_g2_token_multiseed(
            drifted,
            resource_pilot=pilot,
            registry=_registry(
                tmp_path / "vocabulary",
                [*drifted, pilot],
            ),
        )

    runs = _six_runs(tmp_path / "pilot-gate")
    failing_pilot = _make_bundle(
        tmp_path / "pilot-gate",
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
        peak_vram_gb=21.0,
    )
    result = compare_g2_token_multiseed(
        runs,
        resource_pilot=failing_pilot,
        registry=_registry(
            tmp_path / "pilot-gate",
            [*runs, failing_pilot],
        ),
    )
    assert result["status"] == "protocol_failed"
    assert result["resource_pilot"]["status"] == "failed"
    assert result["resource_pilot"]["criteria"][
        "peak_vram_at_most_20_5_gib"
    ] is False


def test_negative_confusion_count_is_rejected(tmp_path: Path) -> None:
    runs = _six_runs(tmp_path)
    pilot = _make_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    target = runs[0] / "metrics/evaluation_replicates.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    row = next(value for value in rows if value["mask_mode"] == "whole_node")
    row["confusion_0_1"] = -1
    row["confusion_0_2"] += 9_000 - row["confusion_0_0"] + 1
    row["confusion_0_3"] = 0
    _write_jsonl(target, rows)

    with pytest.raises(G2TokenComparisonError, match="cannot be negative"):
        compare_g2_token_multiseed(
            runs,
            resource_pilot=pilot,
            registry=_registry(tmp_path, [*runs, pilot]),
        )


def test_registry_rejects_failed_extra_and_missing_campaign_jobs(
    tmp_path: Path,
) -> None:
    runs = _six_runs(tmp_path)
    pilot = _make_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        diagonal=(8_000, 80, 70, 50),
        suffix="pilot",
    )
    bundles = [*runs, pilot]
    target_run_id = str(
        json.loads((runs[0] / "summary.json").read_text())["run_id"]
    )
    failed = _registry(
        tmp_path,
        bundles,
        name="failed-registry.sqlite3",
        failed_run_id=target_run_id,
    )
    with pytest.raises(G2TokenComparisonError, match="not a clean completed job"):
        compare_g2_token_multiseed(
            runs,
            resource_pilot=pilot,
            registry=failed,
        )

    extra = _registry(
        tmp_path,
        bundles,
        name="extra-registry.sqlite3",
        add_extra=True,
    )
    with pytest.raises(G2TokenComparisonError, match="exactly 7 jobs; found 8"):
        compare_g2_token_multiseed(
            runs,
            resource_pilot=pilot,
            registry=extra,
        )

    missing = _registry(
        tmp_path,
        bundles,
        name="missing-registry.sqlite3",
        drop_run_id=target_run_id,
    )
    with pytest.raises(G2TokenComparisonError, match="exactly 7 jobs; found 6"):
        compare_g2_token_multiseed(
            runs,
            resource_pilot=pilot,
            registry=missing,
        )

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import pytest

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT / "scripts" / "analysis" / "compare_g2_width_multiseed.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_g2_width_multiseed_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

G2WidthComparisonError = _MODULE.G2WidthComparisonError
compare_g2_width_multiseed = _MODULE.compare_g2_width_multiseed
write_comparison = _MODULE.write_comparison

_BASELINE = "g2_width512_exact_k1000_full_core"
_WIDER = "g2_width1024_exact_k1000_full_core"
_PILOT = "g2_width1024_exact_k1000_resource_pilot"
_PARAMETERS = {
    _BASELINE: 3_987_880,
    _WIDER: 12_687_784,
    _PILOT: 12_687_784,
}
_CONFIG_PATHS = {
    _BASELINE: "configs/experiment/full_core_g2_multiseed_baseline.yaml",
    _WIDER: "configs/experiment/full_core_g2_multiseed_width1024.yaml",
    _PILOT: (
        "configs/experiment/"
        "full_core_g2_multiseed_width1024_resource_pilot.yaml"
    ),
}
_SCIENTIFIC_IDS = {
    _BASELINE: (
        "e0ce78224f85f6654d49f86cf5328de41a16e905b12d984e8105392431c1d999"
    ),
    _WIDER: (
        "04a51acf353f8073b288187d8a64bc959b63424ad61fb84ac6df275aaf1da790"
    ),
    _PILOT: (
        "1334b662fbdf517205612bf671a5d808cd04d5c3ddf966a052449fbdb192b7f5"
    ),
}
_GRAPH_SHA256 = (
    "2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060"
)
_GRAPH_EDGES = 21_029_944
_OUTPUT_NAMES = {
    ".g2-width-comparison-owner.json",
    "comparison.json",
    "report.md",
}


def _checksum(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def _job_id(label: str) -> str:
    return "q_" + _checksum(label)[:20]


def _run_id(
    variant: str,
    seed: int,
    attempt: int,
    suffix_tag: str = "",
) -> str:
    token = {
        _BASELINE: "b",
        _WIDER: "w",
        _PILOT: "p",
    }[variant]
    return create_run_id(
        seed=seed,
        fold=0,
        attempt=attempt,
        scientific_id_value=f"sci_{_SCIENTIFIC_IDS[variant]}",
        timestamp=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        unique_suffix=f"{token}{seed}a{attempt}{suffix_tag}run",
    )


def _config(
    variant: str,
    *,
    seed: int,
    attempt: int,
) -> dict[str, Any]:
    value = deepcopy(compose_config(_CONFIG_PATHS[variant]))
    value["seed"] = seed
    value["fold"] = 0
    value["attempt"] = attempt
    return value


def _write_jsonl(
    archive: RunArchive,
    relative_path: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    return archive.write_text(
        relative_path,
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n" for row in rows
        ),
    )


def _write_table(
    archive: RunArchive,
    stem: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    table_format: str,
) -> Path:
    if table_format == "jsonl":
        return _write_jsonl(archive, f"{stem}.jsonl", rows)
    assert table_format == "parquet"
    path = archive.write_table(stem, rows, fallback="jsonl")
    assert path.suffix == ".parquet"
    return path


def _slope(values: Sequence[float]) -> float:
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / sum((index - x_mean) ** 2 for index in range(len(values)))


def _history_rows(
    *,
    run_id: str,
    seed: int,
    epochs: int,
    loss_shift: float,
    epoch_duration: float,
    peak_vram_gb: float,
    row_count: int | None = None,
    schedule_drift: bool = False,
    nonfinite_history: bool = False,
) -> list[dict[str, Any]]:
    count = epochs if row_count is None else row_count
    rows: list[dict[str, Any]] = []
    modes = ("partial", "node", "block")
    peak_bytes = round(peak_vram_gb * (1024**3))
    for epoch in range(count):
        mask_label = f"epoch-mask-s{seed}-{epoch}"
        if schedule_drift and epoch == count - 1:
            mask_label += "-drift"
        train_loss = 1.0 + loss_shift - 0.001 * epoch
        if nonfinite_history and epoch == count - 1:
            train_loss = float("nan")
        rows.append(
            {
                "run_id": run_id,
                "split": "fit",
                "training_protocol": "held_in_full_core_fixed_budget",
                "epoch": epoch,
                "mask_mode": modes[epoch % len(modes)],
                "mask_seed": 10_000 + seed * 1_000 + epoch,
                "mask_checksum": _checksum(mask_label),
                "edge_dropout_seed": 20_000 + seed * 1_000 + epoch,
                "edge_checksum": _checksum(f"undropped-graph-{epoch}"),
                "n_masked_entries": 100_000 + epoch,
                "n_target_nodes": 1_000 + epoch,
                "n_edges_used": _GRAPH_EDGES,
                "train_loss": train_loss,
                "diagnostic_loss": None,
                "gradient_norm": 0.5 + 0.001 * epoch,
                "duration_seconds": epoch_duration,
                "peak_cuda_memory_bytes": peak_bytes,
            }
        )
    return rows


def _evaluation_rows(
    *,
    replicates: int,
    mean_huber: float,
    mean_pve: float,
    mask_label: str,
    invalid_pve: bool,
    undefined_r2: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, huber_base, pve_base in (
        ("partial_gene", mean_huber * 0.8, mean_pve + 1.0),
        ("whole_node", mean_huber, mean_pve),
        ("spatial_block", mean_huber * 1.2, mean_pve - 1.0),
    ):
        center = (replicates - 1) / 2.0
        for replicate in range(replicates):
            offset = (replicate - center) * 0.01
            pve = pve_base + offset
            if invalid_pve and mode == "whole_node" and replicate == 0:
                pve += 1.0
            r2: float | None = (pve_base + offset) / 100.0
            if undefined_r2 and mode == "whole_node" and replicate == 0:
                r2 = None
            rows.append(
                {
                    "split": "fit",
                    "mask_mode": mode,
                    "mask_replicate": replicate,
                    "mask_entry_id": f"fit-{mode}-{replicate}",
                    "mask_seed": 31_000 + replicate,
                    "mask_checksum": _checksum(
                        f"{mask_label}-{mode}-{replicate}"
                    ),
                    "n_masked": 100_000 + replicate,
                    "masked_huber": huber_base + offset,
                    "masked_mse": 2.0 * (huber_base + offset),
                    "masked_mae": 0.5 * (huber_base + offset),
                    "masked_r2": r2,
                    "masked_percent_variance_explained": pve,
                }
            )
    return rows


def _write_required_provenance(
    archive: RunArchive,
    *,
    config: Mapping[str, Any],
    attempt: int,
    job_id: str,
    retry_of: str | None,
) -> None:
    dataset = config["dataset"]
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "preprocessing_version": dataset["preprocessing_version"],
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split_fingerprint": dataset["split_fingerprint"]},
    )
    archive.write_text(
        "provenance/command.txt",
        '{"argv":["synthetic"],"cwd":"project-root"}\n',
    )
    archive.write_json(
        "provenance/queue.json",
        {
            "job_id": job_id,
            "attempt": attempt,
            "retry_of": retry_of,
            "config_sha256": _checksum("synthetic-config"),
        },
    )


def _make_success_bundle(
    root: Path,
    *,
    variant: str,
    seed: int,
    mean_huber: float,
    mean_pve: float,
    attempt: int = 1,
    retry_of: str | None = None,
    mask_label: str = "shared-evaluation-mask",
    loss_shift: float = 0.0,
    epoch_duration: float = 0.05,
    peak_vram_gb: float | None = None,
    parameter_count: int | None = None,
    history_row_count: int | None = None,
    schedule_drift: bool = False,
    nonfinite_history: bool = False,
    invalid_pve: bool = False,
    undefined_r2: bool = False,
    final_metric_drift: bool = False,
    contract_drift: str | None = None,
    table_format: str = "jsonl",
    ambiguous_history: bool = False,
    suffix_tag: str = "",
    queue_job_id: str | None = None,
) -> Path:
    pilot = variant == _PILOT
    epochs = 2 if pilot else 200
    replicates = 1 if pilot else 3
    config = _config(variant, seed=seed, attempt=attempt)
    if contract_drift == "graph":
        config["graph"]["query_chunk_size"] += 1
    elif contract_drift == "dataset":
        config["dataset"]["target_scale"] = "drifted_scale"
    elif contract_drift == "model":
        config["model"]["receiver_chunk_size"] += 1
    elif contract_drift == "fold":
        config["fold"] = 1
    run_id = _run_id(variant, seed, attempt, suffix_tag)
    queue_job_id = queue_job_id or _job_id(
        f"{variant}-{seed}-{attempt}-{suffix_tag}"
    )
    archive = RunArchive.create(
        run_id,
        paths=_paths(root),
        manifest={"status": "success"},
        resolved_config=config,
    )
    history = _history_rows(
        run_id=run_id,
        seed=seed,
        epochs=epochs,
        loss_shift=loss_shift,
        epoch_duration=epoch_duration,
        peak_vram_gb=(
            peak_vram_gb
            if peak_vram_gb is not None
            else (20.0 if pilot else (10.0 if variant == _BASELINE else 18.0))
        ),
        row_count=history_row_count,
        schedule_drift=schedule_drift,
        nonfinite_history=nonfinite_history,
    )
    evaluations = _evaluation_rows(
        replicates=replicates,
        mean_huber=mean_huber,
        mean_pve=mean_pve,
        mask_label=mask_label,
        invalid_pve=invalid_pve,
        undefined_r2=undefined_r2,
    )
    _write_table(
        archive,
        "metrics/history",
        history,
        table_format=table_format,
    )
    _write_table(
        archive,
        "metrics/evaluation_replicates",
        evaluations,
        table_format=table_format,
    )
    if ambiguous_history:
        archive.write_bytes("metrics/history.parquet", b"ambiguous")

    valid_losses = [
        float(row["train_loss"])
        for row in history
        if math.isfinite(float(row["train_loss"]))
    ]
    expected_losses = [
        1.0 + loss_shift - 0.001 * epoch for epoch in range(epochs)
    ]
    summed_epoch_duration = sum(
        float(row["duration_seconds"]) for row in history
    )
    recorded_training_duration = summed_epoch_duration + 1.0
    total_duration = recorded_training_duration + 5.0
    effective_peak_vram = max(
        int(row["peak_cuda_memory_bytes"]) for row in history
    ) / (1024**3)
    parameter_count = (
        _PARAMETERS[variant]
        if parameter_count is None
        else parameter_count
    )
    whole_rows = [
        row for row in evaluations if row["mask_mode"] == "whole_node"
    ]
    aggregate_huber = statistics.fmean(
        float(row["masked_huber"]) for row in whole_rows
    )
    aggregate_r2 = statistics.fmean(
        (mean_pve + (index - (replicates - 1) / 2.0) * 0.01) / 100.0
        for index in range(replicates)
    )
    aggregate_pve = 100.0 * aggregate_r2
    summary_metrics = {
        "fit/whole_node/masked_huber": aggregate_huber,
        "fit/whole_node/masked_r2": aggregate_r2,
        "fit/whole_node/masked_percent_variance_explained": aggregate_pve,
        "resource/training_duration_seconds": recorded_training_duration,
        "resource/total_duration_seconds": total_duration,
        "resource/parameter_count": parameter_count,
        "resource/peak_vram_gb": effective_peak_vram,
    }
    final_metrics = dict(summary_metrics)
    if final_metric_drift:
        final_metrics[
            "fit/whole_node/masked_percent_variance_explained"
        ] += 0.1
    archive.append_metric_event(
        {
            "name": "fit/whole_node/masked_huber",
            "value": aggregate_huber,
            "step": epochs - 1,
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    archive.write_predictions(
        "fit",
        [
            {
                "run_id": run_id,
                "sample_key": "sk_synthetic_cell",
                "dataset_id": config["dataset"]["dataset_id"],
                "split": "fit",
                "y_true": [0.0, 1.0],
                "y_pred": [0.1, 0.9],
            }
        ],
    )
    archive.write_bytes("checkpoints/last.ckpt", b"synthetic-checkpoint")
    archive.prepare_log_files()
    _write_required_provenance(
        archive,
        config=config,
        attempt=attempt,
        job_id=queue_job_id,
        retry_of=retry_of,
    )
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "final_epoch": epochs - 1,
            "final_train_loss": expected_losses[-1],
            "minimum_observed_train_loss": min(expected_losses),
            "last_20_epoch_loss_slope": _slope(
                expected_losses[-min(20, len(expected_losses)) :]
            ),
            "all_epochs_completed": len(history) == epochs,
            "all_losses_and_gradients_finite": (
                len(valid_losses) == len(history)
            ),
        },
    )
    archive.write_summary(
        {
            "status": "success",
            "training_exit_status": "success",
            "evaluation_protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "model_name": "g2",
            "model_seed": seed,
            "final_epoch": epochs - 1,
            "fixed_epoch_budget": epochs,
            "checkpoint_role": "last",
            "primary_metric_name": "fit/whole_node/masked_huber",
            "primary_metric_value": aggregate_huber,
            "metrics": summary_metrics,
            "parameter_count": parameter_count,
            "duration_seconds": total_duration,
            "peak_vram_gb": effective_peak_vram,
            "graph_sha256": _GRAPH_SHA256,
            "graph_directed_edges": _GRAPH_EDGES,
            "evaluation_mask_bundle_sha256": _checksum(
                f"{mask_label}-bundle"
            ),
            "evaluation_mask_replicates_per_mode": replicates,
            "evaluation_metrics_include_all_configured_replicates_per_mode": True,
            "diagnostic_resource_pilot": pilot,
            "conclusion_eligible": not pilot,
            "generalization_estimate": False,
        }
    )
    return archive.finalize_success()


def _make_failed_bundle(
    root: Path,
    *,
    variant: str,
    seed: int,
    attempt: int,
    retry_of: str | None = None,
    failure_category: str = "cuda_oom",
    suffix_tag: str = "failed",
    job_id: str | None = None,
) -> Path:
    config = _config(variant, seed=seed, attempt=attempt)
    run_id = _run_id(variant, seed, attempt, suffix_tag)
    archive = RunArchive.create(
        run_id,
        paths=_paths(root),
        resolved_config=config,
    )
    _write_required_provenance(
        archive,
        config=config,
        attempt=attempt,
        job_id=job_id
        or _job_id(f"failed-{variant}-{seed}-{attempt}-{suffix_tag}"),
        retry_of=retry_of,
    )
    return archive.finalize_failure(
        "synthetic failure at /protected/patient/source",
        failure_category=failure_category,
        traceback_text="synthetic traceback with /protected/patient/source",
    )


def _six_runs(
    root: Path,
    *,
    baseline_huber: Sequence[float],
    baseline_pve: Sequence[float],
    wider_huber: Sequence[float],
    wider_pve: Sequence[float],
    targeted_options: Mapping[str, Any] | None = None,
    target_variant: str = _WIDER,
    target_seed: int = 2,
    success_attempts: Mapping[tuple[str, int], int] | None = None,
) -> list[Path]:
    grouped: dict[str, dict[int, Path]] = {_BASELINE: {}, _WIDER: {}}
    for variant, hubers, pves in (
        (_BASELINE, baseline_huber, baseline_pve),
        (_WIDER, wider_huber, wider_pve),
    ):
        for seed in (0, 1, 2):
            attempt = (success_attempts or {}).get((variant, seed), 1)
            options: dict[str, Any] = {
                "loss_shift": 0.01 * seed + (0.02 if variant == _WIDER else 0.0),
                "epoch_duration": (
                    0.05 + 0.005 * seed + (0.02 if variant == _WIDER else 0.0)
                ),
            }
            if attempt > 1:
                options["retry_of"] = _job_id(
                    f"prior-{variant}-{seed}-{attempt - 1}"
                )
            if variant == target_variant and seed == target_seed:
                options.update(dict(targeted_options or {}))
            grouped[variant][seed] = _make_success_bundle(
                root,
                variant=variant,
                seed=seed,
                attempt=attempt,
                mean_huber=hubers[seed],
                mean_pve=pves[seed],
                **options,
            )
    return [
        grouped[_WIDER][2],
        grouped[_BASELINE][0],
        grouped[_WIDER][0],
        grouped[_BASELINE][2],
        grouped[_BASELINE][1],
        grouped[_WIDER][1],
    ]


def _passing_runs(root: Path, **kwargs: Any) -> list[Path]:
    return _six_runs(
        root,
        baseline_huber=(1.0, 1.1, 0.9),
        baseline_pve=(93.0, 94.0, 95.0),
        wider_huber=(0.95, 1.04, 0.85),
        wider_pve=(96.0, 97.0, 98.0),
        **kwargs,
    )


def test_full_audit_exports_resources_failures_and_required_jacobian_status(
    tmp_path: Path,
) -> None:
    prior_job = _job_id(f"prior-{_BASELINE}-0-1")
    runs = _passing_runs(
        tmp_path,
        success_attempts={(_BASELINE, 0): 2},
        targeted_options=None,
    )
    failed = _make_failed_bundle(
        tmp_path,
        variant=_BASELINE,
        seed=0,
        attempt=1,
        retry_of=None,
        job_id=prior_job,
    )
    pilot = _make_success_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        mean_huber=1.2,
        mean_pve=80.0,
        epoch_duration=50.0,
        peak_vram_gb=20.0,
        suffix_tag="pilot",
    )

    result = compare_g2_width_multiseed(
        runs,
        failed_run_directories=[failed],
        resource_pilot=pilot,
        expected_failed_run_count=1,
    )

    assert prior_job != result["runs"][0]["job_id"]
    assert result["runs"][0]["attempt"] == 2
    assert result["runs"][0]["retry_of"] is not None
    assert result["status"] == "jacobian_required"
    assert result["h1_width_gate"]["passes"] is True
    assert result["jacobian_gate"]["status"] == "eligible"
    assert result["jacobian_gate"]["jacobians_computed"] is False
    assert "ratio of aggregate means" in result["h1_width_gate"][
        "statistic_definition"
    ]
    baseline = result["variant_aggregates"][_BASELINE]
    assert baseline["resources"]["total_duration_seconds"]["n"] == 3
    assert baseline["resources"]["summed_epoch_duration_seconds"]["mean"] > 0
    assert baseline["convergence"]["final_train_loss"]["n"] == 3
    assert result["verified_identity"][
        "paired_training_mask_and_edge_dropout_schedules"
    ] is True
    assert result["resource_pilot"]["status"] == "passed"
    assert result["resource_pilot"][
        "projected_200_epoch_training_seconds"
    ] <= 21_600.0
    assert result["failed_runs"]["count"] == 1
    failure = result["failed_runs"]["runs"][0]
    assert failure["failure_category"] == "cuda_oom"
    assert failure["attempt"] == 1
    serialized = json.dumps(result, sort_keys=True)
    assert "/protected/" not in serialized
    assert str(tmp_path) not in serialized

    output = tmp_path / "comparison"
    write_comparison(result, output)
    assert {path.name for path in output.iterdir()} == _OUTPUT_NAMES
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "ratio of aggregate means" in report
    assert "Failed-attempt ledger" in report
    assert "resource gate **PASSED**" in report
    assert "Comparison artifact status: `jacobian_required`" in report

    replacement = json.loads(json.dumps(result))
    replacement["replacement_test_marker"] = True
    write_comparison(replacement, output, overwrite=True)
    assert json.loads(
        (output / "comparison.json").read_text(encoding="utf-8")
    )["replacement_test_marker"] is True


def test_skipped_jacobian_gate_is_complete_and_h1_rejects_mixed_seed(
    tmp_path: Path,
) -> None:
    runs = _six_runs(
        tmp_path,
        baseline_huber=(1.0, 1.0, 1.0),
        baseline_pve=(90.0, 90.0, 90.0),
        wider_huber=(0.94, 0.94, 1.01),
        wider_pve=(96.0, 97.0, 95.0),
    )

    pilot = _make_success_bundle(
        tmp_path,
        variant=_PILOT,
        seed=0,
        mean_huber=1.2,
        mean_pve=80.0,
        epoch_duration=50.0,
        peak_vram_gb=20.0,
        suffix_tag="pilot",
    )
    result = compare_g2_width_multiseed(
        runs,
        resource_pilot=pilot,
        expected_failed_run_count=0,
    )

    assert result["h1_width_gate"]["observed"][
        "relative_mean_huber_reduction"
    ] > 0.02
    assert result["h1_width_gate"]["criteria"][
        "all_seeds_favor_wider_on_huber"
    ] is False
    assert result["h1_width_gate"]["passes"] is False
    assert result["jacobian_gate"]["status"] == "skipped"
    assert result["status"] == "complete"
    assert result["resource_pilot"]["status"] == "passed"


def test_human_readable_safe_retry_job_id_is_supported(tmp_path: Path) -> None:
    runs = _passing_runs(
        tmp_path,
        success_attempts={(_WIDER, 0): 2},
        target_variant=_WIDER,
        target_seed=0,
        targeted_options={
            "queue_job_id": "q_thermalretry7_0cc55ff7",
            "retry_of": "q_0cc55ff72768a7abd1b8",
        },
    )

    failed = _make_failed_bundle(
        tmp_path,
        variant=_WIDER,
        seed=0,
        attempt=1,
        job_id="q_0cc55ff72768a7abd1b8",
    )
    result = compare_g2_width_multiseed(
        runs,
        failed_run_directories=[failed],
        expected_failed_run_count=1,
    )
    retry = next(
        row
        for row in result["runs"]
        if row["variant_label"] == _WIDER and row["seed"] == 0
    )

    assert retry["attempt"] == 2
    assert retry["job_id"] == "q_thermalretry7_0cc55ff7"
    assert retry["retry_of"] == "q_0cc55ff72768a7abd1b8"


def test_completion_requires_pilot_and_reconciled_failure_count(
    tmp_path: Path,
) -> None:
    result = compare_g2_width_multiseed(_passing_runs(tmp_path))

    assert result["status"] == "protocol_incomplete"
    assert result["protocol_completion"]["complete"] is False
    assert set(result["protocol_completion"]["issues"]) == {
        "required_resource_pilot_not_supplied",
        "failed_run_ledger_count_not_reconciled",
    }


def test_retry_requires_matching_supplied_failed_predecessor(
    tmp_path: Path,
) -> None:
    runs = _passing_runs(
        tmp_path,
        success_attempts={(_WIDER, 0): 2},
        target_variant=_WIDER,
        target_seed=0,
        targeted_options={
            "queue_job_id": "q_thermalretry7_0cc55ff7",
            "retry_of": "q_0cc55ff72768a7abd1b8",
        },
    )

    with pytest.raises(G2WidthComparisonError, match="predecessor is absent"):
        compare_g2_width_multiseed(
            runs,
            expected_failed_run_count=0,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("graph", r"config\.graph"),
        ("dataset", r"config\.dataset"),
        ("model", "model shape or receiver chunk"),
        ("fold", "fold must be 0"),
    ),
)
def test_strict_locked_campaign_contract_rejects_any_section_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    runs = _passing_runs(
        tmp_path,
        targeted_options={"contract_drift": drift},
    )
    with pytest.raises(G2WidthComparisonError, match=message):
        compare_g2_width_multiseed(runs)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"history_row_count": 199}, "exactly 200 complete history rows"),
        ({"nonfinite_history": True}, "finite number"),
        ({"schedule_drift": True}, "schedules are not paired"),
    ),
)
def test_history_completeness_finiteness_and_paired_schedule_are_mandatory(
    tmp_path: Path,
    options: Mapping[str, Any],
    message: str,
) -> None:
    runs = _passing_runs(tmp_path, targeted_options=options)
    with pytest.raises(G2WidthComparisonError, match=message):
        compare_g2_width_multiseed(runs)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"invalid_pve": True}, r"PVE is not 100 \* R2"),
        ({"undefined_r2": True}, "must be a finite number"),
        ({"final_metric_drift": True}, "metrics/final.json disagrees"),
        (
            {"parameter_count": _PARAMETERS[_WIDER] + 1},
            "parameter count",
        ),
    ),
)
def test_metric_reconciliation_and_parameter_counts_are_strict(
    tmp_path: Path,
    options: Mapping[str, Any],
    message: str,
) -> None:
    runs = _passing_runs(tmp_path, targeted_options=options)
    with pytest.raises(G2WidthComparisonError, match=message):
        compare_g2_width_multiseed(runs)


def test_r2_and_pve_cannot_exceed_the_mathematical_upper_bound(
    tmp_path: Path,
) -> None:
    runs = _six_runs(
        tmp_path,
        baseline_huber=(1.0, 1.0, 1.0),
        baseline_pve=(90.0, 90.0, 90.0),
        wider_huber=(0.9, 0.9, 0.9),
        wider_pve=(96.0, 97.0, 101.0),
    )

    with pytest.raises(G2WidthComparisonError, match="R2 cannot exceed 1"):
        compare_g2_width_multiseed(runs)


def test_jsonl_parquet_table_ambiguity_is_rejected(tmp_path: Path) -> None:
    runs = _passing_runs(
        tmp_path,
        targeted_options={"ambiguous_history": True},
    )
    with pytest.raises(G2WidthComparisonError, match="exactly one metrics/history"):
        compare_g2_width_multiseed(runs)


def test_parquet_runner_representation_is_supported_when_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    runs = _passing_runs(
        tmp_path,
        targeted_options={"table_format": "parquet"},
    )
    result = compare_g2_width_multiseed(runs)
    wide_seed_two = next(
        row
        for row in result["runs"]
        if row["variant_label"] == _WIDER and row["seed"] == 2
    )
    assert wide_seed_two["history_table_format"] == "parquet"
    assert wide_seed_two["evaluation_table_format"] == "parquet"


def test_overwrite_never_removes_unowned_or_augmented_directories(
    tmp_path: Path,
) -> None:
    result = compare_g2_width_multiseed(_passing_runs(tmp_path))
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    sentinel = arbitrary / "sentinel.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(G2WidthComparisonError, match="owned directory"):
        write_comparison(result, arbitrary, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "preserve me"

    owned = tmp_path / "owned"
    write_comparison(result, owned)
    extra = owned / "sentinel.txt"
    extra.write_text("also preserve", encoding="utf-8")
    original = (owned / "comparison.json").read_bytes()
    with pytest.raises(G2WidthComparisonError, match="exactly"):
        write_comparison(result, owned, overwrite=True)
    assert extra.read_text(encoding="utf-8") == "also preserve"
    assert (owned / "comparison.json").read_bytes() == original

    with pytest.raises(G2WidthComparisonError, match="protected"):
        write_comparison(result, _ROOT, overwrite=True)


def test_exactly_six_success_markers_are_required(tmp_path: Path) -> None:
    runs = _passing_runs(tmp_path)
    with pytest.raises(G2WidthComparisonError, match="exactly six"):
        compare_g2_width_multiseed(runs[:5])

    (runs[0] / "_SUCCESS").unlink()
    with pytest.raises(G2WidthComparisonError, match="_SUCCESS"):
        compare_g2_width_multiseed(runs)

"""Focused contract tests for multiscale hurdle gate verification."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import yaml

from spatial_benchmark.identifiers import (
    canonical_sha256,
    create_run_id,
    scientific_id,
)
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import (
    RunArchive,
    RunValidationError,
)


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT / "scripts" / "train" / "verify_multiscale_hurdle_gates.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_multiscale_hurdle_gates_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_ENQUEUE_SCRIPT = (
    _ROOT / "scripts" / "train" / "enqueue_multiscale_hurdle_campaign.py"
)
_ENQUEUE_SPEC = importlib.util.spec_from_file_location(
    "enqueue_multiscale_hurdle_for_gate_tests",
    _ENQUEUE_SCRIPT,
)
assert _ENQUEUE_SPEC is not None and _ENQUEUE_SPEC.loader is not None
_ENQUEUE = importlib.util.module_from_spec(_ENQUEUE_SPEC)
sys.modules[_ENQUEUE_SPEC.name] = _ENQUEUE
_ENQUEUE_SPEC.loader.exec_module(_ENQUEUE)

_RUNNER_SCRIPT = (
    _ROOT / "scripts" / "train" / "run_multiscale_hurdle_capacity.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_multiscale_hurdle_for_gate_tests",
    _RUNNER_SCRIPT,
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)

GateError = _MODULE.MultiscaleHurdleGateError
verify_resource_gate = _MODULE.verify_resource_gate
verify_representation_gate = _MODULE.verify_representation_gate


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _paths(project_root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(project_root)})


def _config(*, alias: str, resource: bool) -> dict[str, Any]:
    return {
        "campaign": {"campaign_id": _MODULE.CAMPAIGN_ID},
        "dataset": {
            "dataset_id": f"synthetic_{alias.lower()}",
            "version": "v1",
            "split_id": f"split_{alias.lower()}",
        },
        "experiment": {
            "biological_unit_alias": alias,
            "arm": "self",
            "resource_pilot": resource,
        },
        "model": {
            "name": "multiscale-hurdle-count",
            "family": "additive_multiscale_hurdle_count",
        },
        "trainer": {
            "primary_checkpoint_role": "last",
            "restore_best": False,
            "max_epochs": 2 if resource else 200,
        },
        "evaluation": {
            "protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": "fit/whole_node/hurdle_loss",
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


def _job_record(
    *,
    alias: str,
    arm: str,
    config_sha256: str,
    index: int,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "arm": arm,
        "config": f"locked/{alias.lower()}-{arm}.yaml",
        "config_sha256": config_sha256,
        "file_sha256": _sha(f"{alias}-{arm}-file-{index}"),
        "requested_gpu": sorted(_ENQUEUE.SAFE_GPU_IDS)[
            index % len(_ENQUEUE.SAFE_GPU_IDS)
        ],
        "n_nodes": 7_000 + index,
        "graph_bundle_sha256": _sha(
            f"{alias}-{arm}-graph-bundle-{index}"
        ),
        "local_source_permutation_sha256": _sha(
            f"{alias}-local-source-permutation"
        ),
    }


def _materialization(
    configs: Mapping[tuple[str, bool], Mapping[str, Any]],
) -> dict[str, Any]:
    pilots = [
        _job_record(
            alias=alias,
            arm="self",
            config_sha256=canonical_sha256(configs[(alias, True)]),
            index=index,
        )
        for index, alias in enumerate(_MODULE.STAGE1_ALIASES)
    ]
    science: list[dict[str, Any]] = []
    index = 2
    for alias in _ENQUEUE.STAGE2_ALIASES:
        for arm in _ENQUEUE.ARMS:
            config_sha = (
                canonical_sha256(configs[(alias, False)])
                if arm == "self" and (alias, False) in configs
                else _sha(f"{alias}-{arm}-science-config")
            )
            science.append(
                _job_record(
                    alias=alias,
                    arm=arm,
                    config_sha256=config_sha,
                    index=index,
                )
            )
            index += 1
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract": {
                "reference": (
                    "experiments/campaigns/"
                    f"{_MODULE.CAMPAIGN_ID}/frozen_task_contract.yaml"
                ),
                "sha256": _MODULE.FROZEN_CONTRACT_SHA256,
            },
            "pilot_jobs": pilots,
            "science_jobs": science,
        }
    )


def _final_metrics(
    *,
    detection: float = 0.65,
    prevalence: float = 0.51,
    huber_improvement: float = 0.03,
    state_improvement: float = 0.04,
) -> dict[str, float]:
    return {
        "fit/whole_node/hurdle_loss": 0.7,
        "fit/whole_node/detection_balanced_accuracy": detection,
        (
            "fit/whole_node/"
            "reference_per_gene_detection_balanced_accuracy"
        ): prevalence,
        (
            "fit/whole_node/"
            "positive_continuous_huber_relative_improvement_"
            "over_per_gene_reference"
        ): huber_improvement,
        (
            "fit/whole_node/"
            "positive_count_state_mae_relative_improvement_"
            "over_per_gene_reference"
        ): state_improvement,
    }


def _resource_summary(
    *,
    peak_vram: float = 10.0,
    peak_passed: bool = True,
) -> dict[str, Any]:
    return {
        "status": "success",
        "primary_metric_name": "fit/whole_node/hurdle_loss",
        "primary_metric_value": 0.7,
        "finite_losses_and_gradients": True,
        "parameter_count": 7_559_184,
        "parameter_match": True,
        "fp32_amp_absolute_loss_discrepancy": 5e-4,
        "precision_equivalence_passed": True,
        "peak_allocated_vram_gib": peak_vram,
        "peak_vram_passed": peak_passed,
        "projected_gpu_hours_per_200_epochs": 0.4,
        "projected_runtime_passed": True,
        "filesystem_used_decimal_gb": 42.0,
        "disk_safety_passed": True,
        "runner_pilot_gate_passed": peak_passed,
    }


def _runner_resource_summary() -> dict[str, Any]:
    """Return the summary keys emitted by the real runner."""

    return {
        "status": "success",
        "primary_metric_name": "fit/whole_node/hurdle_loss",
        "primary_metric_value": 0.7,
        "parameter_count": 7_559_184,
        "exact_parameter_shape_match": True,
        "fp32_amp_absolute_loss_discrepancy": 5e-4,
        "peak_allocated_vram_gib": 10.0,
        "projected_gpu_hours_per_200_epochs": 0.4,
        "filesystem_used_decimal_gb": 42.0,
        "runner_pilot_gate_passed": True,
    }


def _prediction(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_key": f"sk_{_sha(run_id)[:16]}",
        "dataset_id": "synthetic_gate_dataset",
        "split": "fit",
        "y_true": [0.0, 1.0],
        "y_pred": [0.1, 0.9],
        "sample_loss": 0.1,
    }


def _resource_diagnostic(
    *,
    alias: str,
    resource_pilot: bool,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    peak_value = summary.get("peak_allocated_vram_gib", 10.0)
    peak = float(peak_value)
    discrepancy = float(
        summary.get("fp32_amp_absolute_loss_discrepancy", 5e-4)
    )
    projected = float(
        summary.get("projected_gpu_hours_per_200_epochs", 0.4)
    )
    disk = float(summary.get("filesystem_used_decimal_gb", 42.0))
    precision_passed = bool(
        summary.get(
            "precision_equivalence_passed",
            discrepancy
            <= _MODULE.RESOURCE_THRESHOLDS[
                "fp32_amp_absolute_loss_discrepancy_maximum"
            ],
        )
    )
    peak_passed = bool(
        summary.get(
            "peak_vram_passed",
            peak
            <= _MODULE.RESOURCE_THRESHOLDS[
                "stage1_peak_allocated_vram_gib_maximum"
            ],
        )
    )
    projected_passed = bool(
        summary.get(
            "projected_runtime_passed",
            projected
            <= _MODULE.RESOURCE_THRESHOLDS[
                "stage1_projected_gpu_hours_per_200_epoch_core_maximum"
            ],
        )
    )
    disk_passed = bool(
        summary.get(
            "disk_safety_passed",
            disk
            < _MODULE.RESOURCE_THRESHOLDS[
                "filesystem_used_decimal_gb_hard_stop"
            ],
        )
    )
    return {
        "schema": "multiscale_hurdle_resource_diagnostic_v1",
        "resource_pilot": resource_pilot,
        "arm": "self",
        "biological_unit_alias": alias,
        "thresholds": dict(_MODULE.RESOURCE_THRESHOLDS),
        "finite_losses_and_gradients": bool(
            summary.get("finite_losses_and_gradients", True)
        ),
        "parameter_count": summary.get("parameter_count", 7_559_184),
        "parameter_match": bool(
            summary.get(
                "parameter_match",
                summary.get("exact_parameter_shape_match", True),
            )
        ),
        "fp32_amp_absolute_loss_discrepancy": discrepancy,
        "peak_allocated_vram_gib": peak_value,
        "projected_gpu_hours_per_200_epochs": projected,
        "filesystem_used_decimal_gb": disk,
        "checks": {
            "precision_equivalence_passed": precision_passed,
            "peak_vram_passed": peak_passed,
            "projected_runtime_passed": projected_passed,
            "disk_safety_passed": disk_passed,
            "parameter_match_passed": True,
        },
        "runner_pilot_gate_passed": bool(
            summary.get(
                "runner_pilot_gate_passed",
                precision_passed
                and peak_passed
                and projected_passed
                and disk_passed,
            )
        ),
    }


def _write_archive_support(
    archive: RunArchive,
    *,
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, float],
    resource_diagnostic: Mapping[str, Any],
) -> None:
    archive.write_summary(summary)
    archive.write_bytes("checkpoints/last.ckpt", b"locked-last-checkpoint")
    archive.append_metric_event(
        {
            "name": "fit/whole_node/hurdle_loss",
            "value": final_metrics["fit/whole_node/hurdle_loss"],
            "step": 1,
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    archive.write_json(
        "diagnostics/resource_usage.json",
        resource_diagnostic,
    )
    archive.write_table(
        "metrics/history",
        [{"epoch": 1, "train/hurdle_loss": 0.8}],
    )
    archive.write_predictions("fit", [_prediction(archive.run_id)])
    archive.prepare_log_files()
    archive.write_json(
        "provenance/git.json", {"commit": "test", "dirty": False}
    )
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json(
        "provenance/data_fingerprints.json",
        {"dataset": "synthetic"},
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split": "all-fit"},
    )
    archive.write_text(
        "provenance/command.txt",
        '{"argv":["synthetic-gate-run"],"cwd":"project"}\n',
    )


def _complete_registered_run(
    *,
    project_root: Path,
    registry: Registry,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, float],
    index: int,
) -> tuple[str, Path, str]:
    variant_id = scientific_id(config)
    registry.register_variant(
        variant_id,
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
    )
    job = registry.enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
        command=["synthetic-gate-run"],
        maximum_attempts=2,
        requested_gpu=None,
        job_id=f"job_gate_{index:02d}",
    )
    worker_id = f"gate-worker-{index}"
    claimed = registry.claim_next(worker_id, requested_gpu=None)
    assert claimed is not None and claimed["job_id"] == job["job_id"]
    run_id = create_run_id(
        seed=0,
        fold=0,
        attempt=1,
        scientific_id_value=variant_id,
        timestamp=datetime(
            2026,
            7,
            29,
            14,
            0,
            index,
            tzinfo=timezone.utc,
        ),
        unique_suffix=f"gate{index:04d}",
    )
    archive = RunArchive.create(
        run_id,
        paths=_paths(project_root),
        manifest={
            "status": "completed",
            "lifecycle_status_source": (
                "registry_and_completion_marker"
            ),
        },
        resolved_config=config,
    )
    bound_summary = {**summary, "run_id": run_id}
    _write_archive_support(
        archive,
        summary=bound_summary,
        final_metrics=final_metrics,
        resource_diagnostic=_resource_diagnostic(
            alias=str(
                config["experiment"]["biological_unit_alias"]
            ),
            resource_pilot=bool(
                config["experiment"]["resource_pilot"]
            ),
            summary=summary,
        ),
    )
    registry.register_run_for_job(
        job["job_id"],
        run_id=run_id,
        scientific_id=variant_id,
        repro_id=f"rep_{_sha(f'repro-{index}')[:24]}",
        artifact_path=archive.artifact_path,
        resolved_configuration=config,
        status="running",
    )
    published = archive.publish_success_pending()
    registry.begin_run_and_job_finalization(
        job_id=job["job_id"],
        run_id=run_id,
        worker_id=worker_id,
        artifact_path=published,
        end_time=utc_now(),
        duration_seconds=1.0,
        primary_metric_name="fit/whole_node/hurdle_loss",
        primary_metric_value=final_metrics[
            "fit/whole_node/hurdle_loss"
        ],
        parameter_count=7_559_184,
    )
    archive.mark_success()
    registry.complete_run_and_job(
        job_id=job["job_id"],
        run_id=run_id,
        worker_id=worker_id,
    )
    return run_id, published, str(job["job_id"])


def _campaign_fixture(
    tmp_path: Path,
    *,
    resource_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    science_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _ENQUEUE._PROJECT_ROOT = project_root
    registry = Registry(project_root / "state" / "tracking.sqlite3")
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="Synthetic multiscale hurdle gate campaign",
    )
    configs = {
        (alias, resource): _config(alias=alias, resource=resource)
        for alias in _MODULE.STAGE1_ALIASES
        for resource in (True, False)
    }
    materialization = _materialization(configs)
    materialization_path = project_root / _MODULE.MATERIALIZATION_RELATIVE
    _write_json(materialization_path, materialization)

    run_records: dict[tuple[str, bool], tuple[str, Path, str]] = {}
    index = 1
    for alias in _MODULE.STAGE1_ALIASES:
        summary = (
            resource_summaries.get(alias, _resource_summary())
            if resource_summaries is not None
            else _resource_summary()
        )
        run_records[(alias, True)] = _complete_registered_run(
            project_root=project_root,
            registry=registry,
            config=configs[(alias, True)],
            summary=summary,
            final_metrics=_final_metrics(),
            index=index,
        )
        index += 1
    for alias in _MODULE.STAGE1_ALIASES:
        metrics = (
            science_metrics.get(alias, _final_metrics())
            if science_metrics is not None
            else _final_metrics()
        )
        run_records[(alias, False)] = _complete_registered_run(
            project_root=project_root,
            registry=registry,
            config=configs[(alias, False)],
            summary={
                "status": "success",
                "primary_metric_name": "fit/whole_node/hurdle_loss",
                "primary_metric_value": metrics[
                    "fit/whole_node/hurdle_loss"
                ],
            },
            final_metrics=metrics,
            index=index,
        )
        index += 1
    return {
        "project_root": project_root,
        "registry": registry,
        "configs": configs,
        "materialization": materialization,
        "materialization_path": materialization_path,
        "runs": run_records,
        "resource_path": project_root / _MODULE.RESOURCE_GATE_RELATIVE,
        "representation_path": (
            project_root / _MODULE.REPRESENTATION_GATE_RELATIVE
        ),
    }


def _assert_receipt_checksum(
    path: Path,
    receipt: Mapping[str, Any],
) -> None:
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == receipt
    checksum = persisted.pop("checksum")
    assert checksum == canonical_sha256(persisted)
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_passing_resource_receipt_is_enqueue_and_runner_compatible(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)

    receipt = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    assert receipt["gate_passed"] is True
    assert receipt["production_authorized"] is True
    assert receipt["failure_reasons"] == []
    assert {
        (job["alias"], job["arm"]) for job in receipt["jobs"]
    } == {("ANC-03", "self"), ("ANC-05", "self")}
    _assert_receipt_checksum(fixture["resource_path"], receipt)
    loaded = _ENQUEUE._load_resource_gate(
        fixture["resource_path"],
        materialization=fixture["materialization"],
    )
    assert loaded == receipt
    assert _RUNNER._RESOURCE_GATE_KIND == receipt["receipt_kind"]
    assert _RUNNER._RESOURCE_GATE_THRESHOLDS == receipt["thresholds"]
    for job in receipt["jobs"]:
        _RUNNER._validate_bound_success_job(
            job,
            project_root=fixture["project_root"],
            expected_alias=job["alias"],
            expected_config_sha256=job["config_sha256"],
        )


def test_runner_shaped_resource_summary_passes(
    tmp_path: Path,
) -> None:
    summaries = {
        alias: _runner_resource_summary()
        for alias in _MODULE.STAGE1_ALIASES
    }
    fixture = _campaign_fixture(
        tmp_path,
        resource_summaries=summaries,
    )

    receipt = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    assert receipt["gate_passed"] is True
    assert receipt["production_authorized"] is True


def test_passing_resource_receipt_never_has_runner_invalid_numbers(
    tmp_path: Path,
) -> None:
    invalid = _resource_summary()
    invalid["peak_allocated_vram_gib"] = "10.0"
    fixture = _campaign_fixture(
        tmp_path,
        resource_summaries={
            "ANC-03": invalid,
            "ANC-05": _resource_summary(),
        },
    )

    try:
        receipt = verify_resource_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["resource_path"],
        )
    except GateError:
        return
    if receipt["gate_passed"] is False:
        return
    for job in receipt["jobs"]:
        _RUNNER._validate_bound_success_job(
            job,
            project_root=fixture["project_root"],
            expected_alias=job["alias"],
            expected_config_sha256=job["config_sha256"],
        )


def test_resource_threshold_failure_writes_bound_negative_receipt(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(
        tmp_path,
        resource_summaries={
            "ANC-03": _resource_summary(),
            "ANC-05": _resource_summary(
                peak_vram=12.1,
                peak_passed=False,
            ),
        },
    )

    receipt = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    assert receipt["gate_passed"] is False
    assert receipt["production_authorized"] is False
    assert receipt["failure_reasons"] == [
        "ANC-05 failed one or more resource gates"
    ]
    _assert_receipt_checksum(fixture["resource_path"], receipt)
    with pytest.raises(
        _ENQUEUE.MultiscaleHurdleEnqueueError,
        match="passing resource gate",
    ):
        _ENQUEUE._load_resource_gate(
            fixture["resource_path"],
            materialization=fixture["materialization"],
        )


def test_negative_resource_measurement_cannot_authorize_next_stage(
    tmp_path: Path,
) -> None:
    invalid = _resource_summary()
    invalid["peak_allocated_vram_gib"] = -1.0
    invalid["peak_vram_passed"] = True
    fixture = _campaign_fixture(
        tmp_path,
        resource_summaries={
            "ANC-03": invalid,
            "ANC-05": _resource_summary(),
        },
    )

    receipt = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    assert receipt["gate_passed"] is False
    assert receipt["production_authorized"] is False


def test_resigned_negative_resource_evidence_cannot_authorize_h1(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(
        tmp_path,
        resource_summaries={
            "ANC-03": _resource_summary(),
            "ANC-05": _resource_summary(
                peak_vram=12.1,
                peak_passed=False,
            ),
        },
    )
    receipt = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )
    forged = deepcopy(receipt)
    forged.pop("checksum")
    forged["gate_passed"] = True
    forged["production_authorized"] = True
    forged["failure_reasons"] = []
    _write_json(fixture["resource_path"], _signed(forged))

    with pytest.raises(GateError, match="resource"):
        verify_representation_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["representation_path"],
        )


def test_passing_representation_receipt_is_enqueue_and_runner_compatible(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    resource = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    receipt = verify_representation_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["representation_path"],
    )

    assert receipt["gate_passed"] is True
    assert receipt["both_h1_gates_passed"] is True
    assert receipt["stage2_authorized"] is True
    assert receipt["resource_gate_checksum"] == resource["checksum"]
    _assert_receipt_checksum(fixture["representation_path"], receipt)
    loaded = _ENQUEUE._load_representation_gate(
        fixture["representation_path"],
        materialization=fixture["materialization"],
        resource_gate=resource,
    )
    assert loaded == receipt
    assert _RUNNER._REPRESENTATION_GATE_KIND == receipt["receipt_kind"]
    assert _RUNNER._REPRESENTATION_GATE_THRESHOLDS == (
        receipt["thresholds"]
    )
    for job in receipt["jobs"]:
        _RUNNER._validate_bound_success_job(
            job,
            project_root=fixture["project_root"],
            expected_alias=job["alias"],
            expected_config_sha256=job["config_sha256"],
            require_h1=True,
        )


@pytest.mark.parametrize(
    ("failed_alias", "metric_updates"),
    [
        (
            "ANC-03",
            {
                "fit/whole_node/detection_balanced_accuracy": 0.50,
                (
                    "fit/whole_node/"
                    "reference_per_gene_detection_balanced_accuracy"
                ): 0.51,
            },
        ),
        (
            "ANC-05",
            {
                (
                    "fit/whole_node/"
                    "positive_continuous_huber_relative_improvement_"
                    "over_per_gene_reference"
                ): 0.019,
            },
        ),
        (
            "ANC-05",
            {
                (
                    "fit/whole_node/"
                    "positive_count_state_mae_relative_improvement_"
                    "over_per_gene_reference"
                ): 0.019,
            },
        ),
    ],
)
def test_representation_h1_mapping_writes_negative_receipt(
    tmp_path: Path,
    failed_alias: str,
    metric_updates: Mapping[str, float],
) -> None:
    science_metrics = {
        alias: _final_metrics() for alias in _MODULE.STAGE1_ALIASES
    }
    science_metrics[failed_alias] = {
        **science_metrics[failed_alias],
        **metric_updates,
    }
    fixture = _campaign_fixture(
        tmp_path,
        science_metrics=science_metrics,
    )
    verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    receipt = verify_representation_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["representation_path"],
    )

    assert receipt["gate_passed"] is False
    assert receipt["both_h1_gates_passed"] is False
    assert receipt["stage2_authorized"] is False
    assert receipt["failure_reasons"] == [
        f"{failed_alias} failed one or more H1 criteria"
    ]
    by_alias = {job["alias"]: job for job in receipt["jobs"]}
    assert by_alias[failed_alias]["h1_gate_passed"] is False
    other = (set(_MODULE.STAGE1_ALIASES) - {failed_alias}).pop()
    assert by_alias[other]["h1_gate_passed"] is True
    _assert_receipt_checksum(fixture["representation_path"], receipt)


def test_out_of_range_detection_accuracy_cannot_pass_h1(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(
        tmp_path,
        science_metrics={
            "ANC-03": _final_metrics(
                detection=1.2,
                prevalence=1.1,
            ),
            "ANC-05": _final_metrics(),
        },
    )
    verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )

    with pytest.raises(GateError, match=r"outside \[0, 1\]"):
        verify_representation_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["representation_path"],
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_completed_run_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    registry = fixture["registry"]
    _, _, job_id = fixture["runs"][("ANC-03", True)]
    if mutation == "missing":
        with registry.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE queue_jobs SET status = 'failed' WHERE job_id = ?",
                (job_id,),
            )
        message = "no completed registered run"
    else:
        _complete_registered_run(
            project_root=fixture["project_root"],
            registry=registry,
            config=fixture["configs"][("ANC-03", True)],
            summary=_resource_summary(),
            final_metrics=_final_metrics(),
            index=10,
        )
        message = "multiple completed runs"

    with pytest.raises(GateError, match=message):
        verify_resource_gate(
            project_root=fixture["project_root"],
            registry=registry,
            output_path=fixture["resource_path"],
        )

    assert not fixture["resource_path"].exists()


def test_materialization_checksum_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    materialization = deepcopy(fixture["materialization"])
    materialization["campaign_id"] = "changed"
    _write_json(fixture["materialization_path"], materialization)

    with pytest.raises(GateError, match="checksum does not verify"):
        verify_resource_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["resource_path"],
        )


def test_resigned_materialization_schema_drift_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    materialization = deepcopy(fixture["materialization"])
    materialization.pop("checksum")
    materialization["schema_version"] = 2
    _write_json(
        fixture["materialization_path"],
        _signed(materialization),
    )

    with pytest.raises(GateError, match="identity"):
        verify_resource_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["resource_path"],
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    payload = {"value": 1}
    checksum = canonical_sha256(payload)
    path = tmp_path / f"duplicate-gate-{checksum[:12]}.json"
    path.write_text(
        (
            '{"value":1,"value":1,'
            f'"checksum":"{checksum}"'
            "}\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="duplicate (?:JSON )?key"):
        _MODULE._verified_payload(path, label="duplicate fixture")


def test_bundle_config_binding_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    changed = _config(alias="ANC-03", resource=True)
    changed["trainer"]["max_epochs"] = 3
    changed_sha = canonical_sha256(changed)
    materialization = deepcopy(fixture["materialization"])
    materialization.pop("checksum")
    target = next(
        job
        for job in materialization["pilot_jobs"]
        if job["alias"] == "ANC-03"
    )
    target["config_sha256"] = changed_sha
    _write_json(
        fixture["materialization_path"],
        _signed(materialization),
    )
    _, _, job_id = fixture["runs"][("ANC-03", True)]
    with fixture["registry"].transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE queue_jobs SET canonical_config_json = ?
            WHERE job_id = ?
            """,
            (
                json.dumps(changed, sort_keys=True, separators=(",", ":")),
                job_id,
            ),
        )

    with pytest.raises(
        GateError,
        match="resolved prerequisite config checksum changed",
    ):
        verify_resource_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["resource_path"],
        )


@pytest.mark.parametrize("target", ["config", "success"])
def test_real_bundle_checksum_tampering_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    _, bundle, _ = fixture["runs"][("ANC-03", True)]
    path = (
        bundle / "config.resolved.yaml"
        if target == "config"
        else bundle / "_SUCCESS"
    )
    path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RunValidationError, match="checksum|marker"):
        verify_resource_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["resource_path"],
        )


def test_representation_refuses_resigned_resource_identity_tampering(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    resource = verify_resource_gate(
        project_root=fixture["project_root"],
        registry=fixture["registry"],
        output_path=fixture["resource_path"],
    )
    resource.pop("checksum")
    resource["frozen_contract_sha256"] = "0" * 64
    _write_json(fixture["resource_path"], _signed(resource))

    with pytest.raises(
        GateError,
        match="passing resource gate|passing resource schema",
    ):
        verify_representation_gate(
            project_root=fixture["project_root"],
            registry=fixture["registry"],
            output_path=fixture["representation_path"],
        )

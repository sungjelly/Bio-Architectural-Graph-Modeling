from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.pooled_hybrid_count_training import (
    PooledCoreEpochRecord,
    PooledGlobalEpochRecord,
)


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT / "scripts" / "train" / "verify_pooled_hybrid_pilot_gate.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_pooled_hybrid_pilot_gate_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

GateError = _MODULE.PooledHybridPilotGateError


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _materialization() -> dict[str, Any]:
    jobs = [
        {
            "arm": arm,
            "seed": 0,
            "config": f"configs/{arm}.yaml",
            "config_sha256": canonical_sha256({"arm": arm, "seed": 0}),
            "file_sha256": "f" * 64,
            "requested_gpu": index,
        }
        for index, arm in enumerate(_MODULE.ARMS)
    ]
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
            "frozen_contract": {
                "sha256": _MODULE.EXPECTED_CONTRACT_SHA256,
            },
            "counts": {
                "aliases": 10,
                "pilot_configs": 2,
                "production_configs": 14,
                "production_seeds": 7,
            },
            "cohort": {"aliases": list(_MODULE.ALIASES)},
            "pilot_jobs": jobs,
        }
    )


def _enqueue(materialization: Mapping[str, Any]) -> dict[str, Any]:
    jobs = [
        {
            **job,
            "job_id": f"job-{index}",
            "maximum_attempts": 2,
        }
        for index, job in enumerate(materialization["pilot_jobs"])
    ]
    for job in jobs:
        job.pop("config")
        job.pop("file_sha256")
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_ENQUEUE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "pilot",
            "materialization_checksum": materialization["checksum"],
            "pilot_gate_checksum": None,
            "complete": True,
            "jobs": jobs,
        }
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}\n',
        '{"value":NaN}\n',
        '{"value":-Infinity}\n',
    ],
)
def test_materialization_strict_json_fails_before_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    materialization = tmp_path / "materialization.json"
    materialization.write_text(raw, encoding="utf-8")
    enqueue = tmp_path / "enqueue.json"
    enqueue.write_text("{}\n", encoding="utf-8")
    constructed = False

    def forbidden(_path: Path) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid receipt reached registry")

    monkeypatch.setattr(_MODULE, "Registry", forbidden)
    with pytest.raises(GateError):
        _MODULE.verify_pilot_gate(
            materialization_path=materialization,
            enqueue_receipt_path=enqueue,
            output_path=tmp_path / "gate.json",
            database_path=tmp_path / "tracking.sqlite3",
        )
    assert constructed is False
    assert not (tmp_path / "gate.json").exists()


def test_materialization_and_enqueue_are_checksum_bound(
    tmp_path: Path,
) -> None:
    materialization = _materialization()
    tampered = deepcopy(materialization)
    tampered["parameter_count"] += 1
    materialization_path = tmp_path / "materialization.json"
    _write_json(materialization_path, tampered)
    with pytest.raises(GateError, match="checksum does not verify"):
        _MODULE._load_materialization(materialization_path)

    enqueue = _enqueue(materialization)
    enqueue.pop("checksum")
    enqueue["materialization_checksum"] = "0" * 64
    enqueue_path = tmp_path / "enqueue.json"
    _write_json(enqueue_path, _signed(enqueue))
    with pytest.raises(GateError, match="incomplete or mismatched"):
        _MODULE._load_enqueue_receipt(
            enqueue_path, materialization=materialization
        )


def _core_step_rows() -> list[dict[str, Any]]:
    rows = []
    step = 0
    for epoch in range(2):
        for alias in _MODULE.ALIASES:
            rows.append(
                {
                    "global_epoch": epoch,
                    "optimizer_step": step,
                    "alias": alias,
                    "hybrid_loss": 1.0,
                    "detection_bce": 1.0,
                    "ordinal_bce": 1.0,
                    "positive_continuous_huber": 1.0,
                    "gradient_norm": 1.0,
                }
            )
            step += 1
    return rows


def test_core_step_history_requires_every_alias_once_per_epoch() -> None:
    rows = _core_step_rows()
    _MODULE._verify_core_step_history(rows)
    rows[9]["alias"] = rows[0]["alias"]
    with pytest.raises(GateError, match="exactly once"):
        _MODULE._verify_core_step_history(rows)


@pytest.mark.parametrize(
    "mutation",
    ["missing_alias", "nan", "bad_difference", "bad_pass", "bad_maximum"],
)
def test_precision_evidence_fails_closed(
    mutation: str,
) -> None:
    per_core = {
        alias: {
            "fp32_total_loss": 1.0,
            "amp_total_loss": 1.0005,
            "absolute_total_loss_discrepancy": 0.0005,
            "mask_checksum": canonical_sha256({"alias": alias}),
            "passed": True,
        }
        for alias in _MODULE.ALIASES
    }
    payload: dict[str, Any] = {
        "per_core": per_core,
        "maximum_observed_discrepancy": 0.0005,
        "all_cores_passed": True,
    }
    if mutation == "missing_alias":
        per_core.pop(_MODULE.ALIASES[-1])
    elif mutation == "nan":
        per_core[_MODULE.ALIASES[0]]["amp_total_loss"] = float("nan")
    elif mutation == "bad_difference":
        per_core[_MODULE.ALIASES[0]][
            "absolute_total_loss_discrepancy"
        ] = 0.0004
    elif mutation == "bad_pass":
        per_core[_MODULE.ALIASES[0]]["passed"] = False
    else:
        payload["maximum_observed_discrepancy"] = 0.0004
    with pytest.raises(GateError):
        _MODULE._verify_precision(payload)


def _verified_job(
    arm: str,
    *,
    mask_suffix: str = "",
    disk_passed: bool = True,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": 0,
        "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
        "verified_bundle": True,
        "finite_losses_and_gradients": True,
        "all_20_optimizer_steps_completed": True,
        "every_core_once_each_epoch": True,
        "parameter_match": True,
        "paired_initialization_match": True,
        "precision_equivalence_passed": True,
        "peak_vram_passed": True,
        "peak_host_memory_passed": True,
        "projected_runtime_passed": True,
        "projected_disk_passed": disk_passed,
        "runner_pilot_gate_passed": disk_passed,
        "evaluation_mask_bundle_sha256": "e" * 64,
        "graph_bundle_sha256": "g" * 64,
        "encoder_initial_state_sha256": "a" * 64,
        "decoder_initial_state_sha256": "d" * 64,
        "precision_mask_checksums": {
            alias: canonical_sha256({"alias": alias, "suffix": mask_suffix})
            for alias in _MODULE.ALIASES
        },
    }


class _NoopRegistry:
    pass


def _patch_aggregate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jobs: list[dict[str, Any]],
) -> None:
    monkeypatch.setattr(_MODULE, "Registry", lambda _path: _NoopRegistry())
    monkeypatch.setattr(
        _MODULE,
        "_resolve_pilot_attempt_lineages",
        lambda _registry, *, enqueue, planned: {
            slot: {"slot": slot} for slot in _MODULE.EXPECTED_JOBS
        },
    )
    by_arm = {job["arm"]: job for job in jobs}
    monkeypatch.setattr(
        _MODULE,
        "_verify_one_pilot",
        lambda _registry, receipt_job, **_kwargs: deepcopy(
            by_arm[receipt_job["arm"]]
        ),
    )


def test_threshold_failure_writes_non_authorizing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization = _materialization()
    enqueue = _enqueue(materialization)
    materialization_path = tmp_path / "materialization.json"
    enqueue_path = tmp_path / "enqueue.json"
    output = tmp_path / "gate.json"
    _write_json(materialization_path, materialization)
    _write_json(enqueue_path, enqueue)
    jobs = [
        _verified_job(_MODULE.ARMS[0], disk_passed=False),
        _verified_job(_MODULE.ARMS[1]),
    ]
    _patch_aggregate(monkeypatch, jobs=jobs)

    gate = _MODULE.verify_pilot_gate(
        materialization_path=materialization_path,
        enqueue_receipt_path=enqueue_path,
        output_path=output,
        database_path=tmp_path / "tracking.sqlite3",
    )
    assert gate["gate_passed"] is False
    assert gate["production_authorized"] is False
    assert any("projected_disk_passed" in item for item in gate["failure_reasons"])
    assert output.is_file()
    assert _MODULE._verified_checksum(
        json.loads(output.read_text(encoding="utf-8")),
        label="gate",
    ) == gate["checksum"]


def test_cross_arm_frozen_batch_mismatch_writes_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization = _materialization()
    enqueue = _enqueue(materialization)
    materialization_path = tmp_path / "materialization.json"
    enqueue_path = tmp_path / "enqueue.json"
    output = tmp_path / "gate.json"
    _write_json(materialization_path, materialization)
    _write_json(enqueue_path, enqueue)
    jobs = [
        _verified_job(_MODULE.ARMS[0]),
        _verified_job(_MODULE.ARMS[1], mask_suffix="tampered"),
    ]
    _patch_aggregate(monkeypatch, jobs=jobs)
    with pytest.raises(GateError, match="different frozen precision mask"):
        _MODULE.verify_pilot_gate(
            materialization_path=materialization_path,
            enqueue_receipt_path=enqueue_path,
            output_path=output,
            database_path=tmp_path / "tracking.sqlite3",
        )
    assert not output.exists()


def test_passing_gate_binds_exact_thresholds_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization = _materialization()
    enqueue = _enqueue(materialization)
    materialization_path = tmp_path / "materialization.json"
    enqueue_path = tmp_path / "enqueue.json"
    output = tmp_path / "gate.json"
    _write_json(materialization_path, materialization)
    _write_json(enqueue_path, enqueue)
    _patch_aggregate(
        monkeypatch,
        jobs=[_verified_job(arm) for arm in _MODULE.ARMS],
    )
    gate = _MODULE.verify_pilot_gate(
        materialization_path=materialization_path,
        enqueue_receipt_path=enqueue_path,
        output_path=output,
        database_path=tmp_path / "tracking.sqlite3",
    )
    assert gate["gate_passed"] is True
    assert gate["production_authorized"] is True
    assert gate["thresholds"] == _MODULE.PILOT_GATE_THRESHOLDS
    assert gate["materialization_checksum"] == materialization["checksum"]
    assert gate["pilot_enqueue_receipt_checksum"] == enqueue["checksum"]


def test_runner_diagnostics_round_trip_through_gate_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_path = (
        _ROOT / "scripts" / "train" / "run_pooled_hybrid_count_capacity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pooled_runner_for_gate_contract_test", runner_path
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    precision_results = {
        alias: runner.PrecisionEquivalenceResult(
            mask_checksum=canonical_sha256({"mask": alias}),
            fp32_total_loss=1.0,
            amp_total_loss=1.0005,
            absolute_total_loss_discrepancy=0.0005,
            fp32_components={"detection": 1.0},
            amp_components={"detection": 1.0005},
            amp_dtype="float16",
            peak_cuda_memory_bytes=1024,
            passed=True,
        )
        for alias in _MODULE.ALIASES
    }
    precision = runner._precision_payload(
        precision_results,
        materialization_checksum="a" * 64,
        config_sha256="b" * 64,
    )
    checksums, maximum, passed, _values = _MODULE._verify_precision(precision)
    assert set(checksums) == set(_MODULE.ALIASES)
    assert maximum == pytest.approx(0.0005)
    assert passed is True

    core_records = []
    optimizer_step = 0
    for epoch in range(2):
        for step_in_epoch, alias in enumerate(_MODULE.ALIASES):
            core_records.append(
                PooledCoreEpochRecord(
                    global_epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    optimizer_step=optimizer_step,
                    alias=alias,
                    n_nodes=10,
                    n_edges=20,
                    mask_mode="whole_node",
                    mask_seed=100 + optimizer_step,
                    mask_checksum=canonical_sha256(
                        {"epoch": epoch, "alias": alias}
                    ),
                    n_masked_entries=10,
                    n_zero_targets=5,
                    n_positive_targets=5,
                    n_target_nodes=1,
                    train_hybrid_loss=1.0,
                    train_detection_bce=1.0,
                    train_ordinal_bce=1.0,
                    train_positive_continuous_huber=1.0,
                    gradient_norm=1.0,
                    duration_seconds=0.1,
                    peak_cuda_memory_bytes=1024,
                )
            )
            optimizer_step += 1
    _MODULE._verify_core_step_history(
        [vars(record) for record in core_records]
    )
    global_records = tuple(
        PooledGlobalEpochRecord(
            epoch=epoch,
            ordered_aliases=_MODULE.ALIASES,
            cores_visited=10,
            optimizer_steps=10,
            mean_hybrid_loss=1.0,
            mean_detection_bce=1.0,
            mean_ordinal_bce=1.0,
            mean_positive_continuous_huber=1.0,
            duration_seconds=10.0,
            peak_cuda_memory_bytes=1024,
        )
        for epoch in range(2)
    )
    training = SimpleNamespace(
        global_history=global_records,
        core_history=tuple(core_records),
        fixed_epoch_budget=2,
        optimizer_steps_completed=20,
        completed_global_epochs=2,
        device="cpu",
    )
    monkeypatch.setattr(runner, "_peak_host_memory_bytes", lambda: 1024**3)
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            100 * 1024**3, 20 * 1024**3, 80 * 1024**3
        ),
    )
    resource = runner._resource_diagnostic(
        archive=SimpleNamespace(
            paths=SimpleNamespace(artifact_root=tmp_path)
        ),
        contract=SimpleNamespace(
            diagnostic_resource_pilot=True,
            uses_graph=True,
            public_variant=_MODULE.ARMS[0],
        ),
        configured_training=SimpleNamespace(amp=True),
        effective_training=SimpleNamespace(amp=True),
        precision_payload=precision,
        training=training,
        total_duration=25.0,
        training_duration=20.0,
        evaluation_duration=5.0,
        peak_vram_bytes=2 * 1024**3,
        parameter_audit={
            "exact_trainable_parameter_match": True,
            "trainable_parameter_count_graph": (
                _MODULE.EXPECTED_PARAMETER_COUNT
            ),
            "trainable_parameter_count_self": (
                _MODULE.EXPECTED_PARAMETER_COUNT
            ),
            "encoder_initial_state_bit_identical": True,
            "decoder_initial_state_bit_identical": True,
        },
        materialization_checksum="a" * 64,
        config_sha256="b" * 64,
    )
    expected_check_keys = {
        "finite_losses_and_gradients",
        "parameter_match_passed",
        "paired_initialization_passed",
        "precision_equivalence_passed",
        "peak_vram_passed",
        "peak_host_memory_passed",
        "projected_runtime_passed",
        "projected_disk_passed",
        "every_core_once_each_epoch",
        "all_20_optimizer_steps_completed",
    }
    assert set(resource["pilot_checks"]) == expected_check_keys
    assert all(value is True for value in resource["pilot_checks"].values())
    assert resource["pilot_gate_passed"] is True

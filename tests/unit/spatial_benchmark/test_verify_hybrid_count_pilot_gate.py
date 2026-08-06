from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "verify_hybrid_count_pilot_gate.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_hybrid_count_pilot_gate_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

GateError = _MODULE.HybridCountPilotGateError
verify_pilot_gate = _MODULE.verify_pilot_gate


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialization() -> dict[str, Any]:
    jobs = [
        {
            "alias": "ANC-01",
            "arm": arm,
            "config_sha256": canonical_sha256({"arm": arm}),
            "requested_gpu": index,
        }
        for index, arm in enumerate(sorted(_MODULE._MODEL_TO_ARM.values()))
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
            "pilot_jobs": jobs,
        }
    )


def _enqueue_receipt(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    planned = {
        (job["alias"], job["arm"]): job
        for job in materialization["pilot_jobs"]
    }
    jobs = []
    for index, key in enumerate(sorted(planned)):
        source = planned[key]
        jobs.append(
            {
                "alias": key[0],
                "arm": key[1],
                "config_sha256": source["config_sha256"],
                "requested_gpu": source["requested_gpu"],
                "maximum_attempts": 2,
                "job_id": f"job-{index}",
            }
        )
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
        '{"value":Infinity}\n',
    ],
)
def test_malformed_materialization_fails_before_registry_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    materialization = tmp_path / "materialization.json"
    materialization.write_text(raw, encoding="utf-8")
    enqueue = tmp_path / "enqueue.json"
    enqueue.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "gate.json"
    registry_constructed = False

    def forbidden_registry(_path: Path) -> None:
        nonlocal registry_constructed
        registry_constructed = True
        raise AssertionError("invalid preflight reached Registry")

    monkeypatch.setattr(_MODULE, "Registry", forbidden_registry)

    with pytest.raises(GateError):
        verify_pilot_gate(
            materialization_path=materialization,
            enqueue_receipt_path=enqueue,
            output_path=output,
            database_path=tmp_path / "tracking.sqlite3",
        )

    assert registry_constructed is False
    assert not output.exists()


def test_materialization_and_enqueue_checksums_are_bound(
    tmp_path: Path,
) -> None:
    materialization = _materialization()
    materialization_path = tmp_path / "materialization.json"
    materialization["parameter_count"] += 1
    _write_json(materialization_path, materialization)

    with pytest.raises(GateError, match="checksum does not verify"):
        _MODULE._load_materialization(materialization_path)

    materialization = _materialization()
    enqueue = _enqueue_receipt(materialization)
    enqueue["materialization_checksum"] = "f" * 64
    enqueue = _signed(enqueue)
    enqueue_path = tmp_path / "enqueue.json"
    _write_json(enqueue_path, enqueue)

    with pytest.raises(GateError, match="incomplete or mismatched"):
        _MODULE._load_enqueue_receipt(
            enqueue_path,
            materialization=materialization,
        )


def _minimal_config(
    *,
    arm: str,
    alias: str = "ANC-01",
    attempt: int = 1,
) -> dict[str, Any]:
    model_name = {
        "hybrid-gat-k1000": "hybrid-count-gat",
        "hybrid-matched-self": "hybrid-count-matched-self",
    }[arm]
    return {
        "campaign": {"campaign_id": _MODULE.CAMPAIGN_ID},
        "dataset": {"biological_unit_alias": alias},
        "experiment": {
            "biological_unit_alias": alias,
            "arm": arm,
            "resource_pilot": True,
        },
        "model": {"name": model_name},
        "trainer": {
            "optimizer": "AdamW",
            "max_epochs": 2,
            "diagnostic_resource_pilot": True,
        },
        "evaluation": {
            "mask_replicates_per_mode": 1,
            "diagnostic_only": True,
        },
        "seed": 0,
        "fold": 0,
        "attempt": attempt,
    }


class _PilotRegistry:
    def __init__(self, root: Path, *, completed_attempt: int) -> None:
        self.root = root
        self.completed_attempt = completed_attempt

    def get_job(self, job_id: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "completed",
            "run_id": "pilot-run",
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        if run_id == "failed-run":
            return {
                "run_id": run_id,
                "campaign_id": _MODULE.CAMPAIGN_ID,
                "status": "failed",
                "attempt": 1,
                "retry_of": None,
            }
        return {
            "run_id": run_id,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "status": "completed",
            "artifact_path": str(self.root),
            "attempt": self.completed_attempt,
            "retry_of": (
                "failed-run" if self.completed_attempt == 2 else None
            ),
        }


def _write_pilot_bundle(
    root: Path,
    *,
    arm: str,
    discrepancy: float,
    peak_vram: float,
    projected_hours: float,
    runner_gate_passed: bool,
    precision_passed: bool,
    attempt: int = 1,
    encoder_identical: bool = True,
    decoder_identical: bool = True,
) -> dict[str, Any]:
    config = _minimal_config(arm=arm, attempt=attempt)
    (root / "diagnostics").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "metrics").mkdir()
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        root / "summary.json",
        {
            "run_id": "pilot-run",
            "status": "success",
            "training_exit_status": "success",
            "diagnostic_resource_pilot": True,
            "conclusion_eligible": False,
            "final_epoch": 1,
            "fixed_epoch_budget": 2,
            "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
            "exact_parameter_match": True,
            "pilot_gate_passed": runner_gate_passed,
            "evaluation_mask_bundle_sha256": "e" * 64,
            "graph_sha256": "g" * 64,
        },
    )
    _write_json(
        root / "diagnostics" / "resource_usage.json",
        {
            "diagnostic_resource_pilot": True,
            "public_variant": arm,
            "biological_unit_alias": "ANC-01",
            "epochs_completed": 2,
            "finite_losses_and_gradients": True,
            "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
            "peak_allocated_vram_gib": peak_vram,
            "projected_200_epoch_runtime_hours": projected_hours,
            "pilot_gate_passed": runner_gate_passed,
        },
    )
    _write_json(
        root / "diagnostics" / "fp32_amp_equivalence.json",
        {
            "absolute_total_loss_discrepancy": discrepancy,
            "passed": precision_passed,
            "mask_checksum": "m" * 64,
            "fp32_total_loss": 1.0,
            "amp_total_loss": 1.0 + discrepancy,
        },
    )
    _write_json(
        root / "diagnostics" / "parameter_structure_audit.json",
        {
            "exact_trainable_parameter_match": True,
            "trainable_parameter_count_graph": (
                _MODULE.EXPECTED_PARAMETER_COUNT
            ),
            "trainable_parameter_count_self": (
                _MODULE.EXPECTED_PARAMETER_COUNT
            ),
            "encoder_initial_state_bit_identical": encoder_identical,
            "decoder_initial_state_bit_identical": decoder_identical,
        },
    )
    _write_json(
        root / "diagnostics" / "training_convergence.json",
        {
            "all_epochs_completed": True,
            "all_losses_and_gradients_finite": True,
        },
    )
    _write_json(
        root / "provenance" / "full_core_training.json",
        {"final_epoch": 1, "fixed_epoch_budget": 2},
    )
    history_rows = [
        {
            "epoch": epoch,
            "train_hybrid_loss": 1.0,
            "train_detection_bce": 1.0,
            "train_ordinal_bce": 1.0,
            "train_positive_continuous_huber": 1.0,
            "gradient_norm": 1.0,
            "duration_seconds": 1.0,
        }
        for epoch in (0, 1)
    ]
    (root / "metrics" / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history_rows),
        encoding="utf-8",
    )
    _write_json(root / "_SUCCESS", {"content_sha256": "s" * 64})
    return config


def _verify_bundle_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str,
    discrepancy: float,
    peak_vram: float,
    projected_hours: float,
    runner_gate_passed: bool,
    precision_passed: bool,
    attempt: int = 1,
    encoder_identical: bool = True,
    decoder_identical: bool = True,
) -> dict[str, Any]:
    root = tmp_path / "bundle"
    config = _write_pilot_bundle(
        root,
        arm=arm,
        discrepancy=discrepancy,
        peak_vram=peak_vram,
        projected_hours=projected_hours,
        runner_gate_passed=runner_gate_passed,
        precision_passed=precision_passed,
        attempt=attempt,
        encoder_identical=encoder_identical,
        decoder_identical=decoder_identical,
    )
    monkeypatch.setattr(
        _MODULE,
        "validate_experiment_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        _MODULE,
        "verify_run_bundle",
        lambda _root: {"valid": True, "status": "success"},
    )
    monkeypatch.setattr(
        _MODULE,
        "_verify_checkpoint",
        lambda _registry, *, run_id: {
            "checkpoint_id": f"checkpoint-{run_id}",
            "checkpoint_sha256": "c" * 64,
            "checkpoint_epoch": 1,
            "checkpoint_role": "last",
            "checkpoint_verified": True,
        },
    )
    planned_config = dict(config)
    planned_config["attempt"] = 1
    planned = {
        "alias": "ANC-01",
        "arm": arm,
        "config_sha256": canonical_sha256(planned_config),
        "requested_gpu": 0,
    }
    receipt = {
        **planned,
        "job_id": "job-0",
        "maximum_attempts": 2,
    }
    if attempt == 1:
        attempts = [
            {
                "job_id": "job-0",
                "attempt": 1,
                "maximum_attempts": 2,
                "status": "completed",
                "run_id": "pilot-run",
                "failure_category": None,
                "retry_of": None,
                "is_original_enqueue_job": True,
                "selected_completed_attempt": True,
            }
        ]
        completed_job = {
            "job_id": "job-0",
            "attempt_count": 1,
            "run_id": "pilot-run",
        }
    else:
        attempts = [
            {
                "job_id": "job-0",
                "attempt": 1,
                "maximum_attempts": 2,
                "status": "failed",
                "run_id": "failed-run",
                "failure_category": "resource_failure",
                "retry_of": None,
                "is_original_enqueue_job": True,
                "selected_completed_attempt": False,
            },
            {
                "job_id": "job-1",
                "attempt": 2,
                "maximum_attempts": 2,
                "status": "completed",
                "run_id": "pilot-run",
                "failure_category": None,
                "retry_of": "job-0",
                "is_original_enqueue_job": False,
                "selected_completed_attempt": True,
            },
        ]
        completed_job = {
            "job_id": "job-1",
            "attempt_count": 2,
            "run_id": "pilot-run",
        }
    lineage = {
        "original_job_id": "job-0",
        "completed_job": completed_job,
        "attempts": attempts,
        "failed_attempts": attempts[:-1],
    }
    return _MODULE._verify_one_pilot(
        _PilotRegistry(root, completed_attempt=attempt),
        receipt,
        planned,
        lineage,
    )


@pytest.mark.parametrize(
    ("arm", "discrepancy", "peak", "hours"),
    [
        ("hybrid-gat-k1000", 1e-3, 20.5, 6.0),
        ("hybrid-matched-self", 1e-3, 20.5, 99.0),
    ],
)
def test_one_pilot_accepts_exact_thresholds_and_self_runtime_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    discrepancy: float,
    peak: float,
    hours: float,
) -> None:
    result = _verify_bundle_case(
        tmp_path,
        monkeypatch,
        arm=arm,
        discrepancy=discrepancy,
        peak_vram=peak,
        projected_hours=hours,
        runner_gate_passed=True,
        precision_passed=True,
    )

    assert result["precision_equivalence_passed"] is True
    assert result["peak_vram_passed"] is True
    assert result["projected_runtime_passed"] is True
    assert result["runner_pilot_gate_passed"] is True


@pytest.mark.parametrize(
    ("discrepancy", "peak", "hours", "failed_field"),
    [
        (0.001001, 20.5, 6.0, "precision_equivalence_passed"),
        (0.001, 20.5001, 6.0, "peak_vram_passed"),
        (0.001, 20.5, 6.0001, "projected_runtime_passed"),
    ],
)
def test_one_pilot_recomputes_each_frozen_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    discrepancy: float,
    peak: float,
    hours: float,
    failed_field: str,
) -> None:
    result = _verify_bundle_case(
        tmp_path,
        monkeypatch,
        arm="hybrid-gat-k1000",
        discrepancy=discrepancy,
        peak_vram=peak,
        projected_hours=hours,
        runner_gate_passed=False,
        precision_passed=discrepancy <= _MODULE.MAX_AMP_DISCREPANCY,
    )

    assert result[failed_field] is False
    assert result["runner_pilot_gate_passed"] is False


def test_one_pilot_rejects_runner_verifier_threshold_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(GateError, match="decisions differ"):
        _verify_bundle_case(
            tmp_path,
            monkeypatch,
            arm="hybrid-gat-k1000",
            discrepancy=0.001001,
            peak_vram=20.5,
            projected_hours=6.0,
            runner_gate_passed=True,
            precision_passed=False,
        )


@pytest.mark.parametrize(
    ("encoder_identical", "decoder_identical"),
    [(False, True), (True, False)],
)
def test_one_pilot_requires_bit_identical_common_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_identical: bool,
    decoder_identical: bool,
) -> None:
    with pytest.raises(GateError, match="structural diagnostics"):
        _verify_bundle_case(
            tmp_path,
            monkeypatch,
            arm="hybrid-gat-k1000",
            discrepancy=0.001,
            peak_vram=20.5,
            projected_hours=6.0,
            runner_gate_passed=True,
            precision_passed=True,
            encoder_identical=encoder_identical,
            decoder_identical=decoder_identical,
        )


def test_one_pilot_accepts_retry_bundle_and_preserves_root_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _verify_bundle_case(
        tmp_path,
        monkeypatch,
        arm="hybrid-gat-k1000",
        discrepancy=0.001,
        peak_vram=20.5,
        projected_hours=6.0,
        runner_gate_passed=True,
        precision_passed=True,
        attempt=2,
    )

    assert result["job_id"] == "job-0"
    assert result["original_enqueue_job_id"] == "job-0"
    assert result["completed_job_id"] == "job-1"
    assert result["completed_attempt"] == 2
    assert result["attempt_count"] == 2
    assert result["failed_attempts"] == [result["attempts"][0]]
    assert result["failed_attempts"][0]["failure_category"] == (
        "resource_failure"
    )
    assert result["encoder_initial_state_bit_identical"] is True
    assert result["decoder_initial_state_bit_identical"] is True


def _lineage_fixture() -> tuple[
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    materialization = _materialization()
    planned = {
        (str(job["alias"]), str(job["arm"])): dict(job)
        for job in materialization["pilot_jobs"]
    }
    enqueue = _enqueue_receipt(materialization)
    inventory: list[dict[str, Any]] = []
    for receipt_job in enqueue["jobs"]:
        slot = (str(receipt_job["alias"]), str(receipt_job["arm"]))
        plan = planned[slot]
        inventory.append(
            {
                "job_id": str(receipt_job["job_id"]),
                "campaign_id": _MODULE.CAMPAIGN_ID,
                "experiment_config_reference": None,
                "canonical_config": {"arm": slot[1]},
                "command": ["run-pilot"],
                "priority": 50,
                "status": "completed",
                "attempt_count": 1,
                "maximum_attempts": 2,
                "requested_gpu": str(plan["requested_gpu"]),
                "retry_of": None,
                "run_id": f"run-{slot[1]}",
                "failure_category": None,
            }
        )
    return enqueue, planned, inventory


def _add_successful_retry(
    enqueue: Mapping[str, Any],
    inventory: list[dict[str, Any]],
    *,
    arm: str = "hybrid-gat-k1000",
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = next(job for job in enqueue["jobs"] if job["arm"] == arm)
    root = next(job for job in inventory if job["job_id"] == receipt["job_id"])
    root["status"] = "failed"
    root["failure_category"] = "resource_failure"
    root["run_id"] = f"failed-{arm}"
    child = {
        **deepcopy(root),
        "job_id": f"retry-{arm}",
        "status": "completed",
        "attempt_count": 2,
        "retry_of": root["job_id"],
        "run_id": f"completed-{arm}",
        "failure_category": None,
    }
    inventory.append(child)
    return root, child


def test_lineage_resolver_accepts_one_linear_completed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue, planned, inventory = _lineage_fixture()
    root, child = _add_successful_retry(enqueue, inventory)
    monkeypatch.setattr(
        _MODULE,
        "_campaign_queue_jobs",
        lambda _registry: deepcopy(inventory),
    )

    resolved = _MODULE._resolve_pilot_attempt_lineages(
        object(),
        enqueue=enqueue,
        planned=planned,
    )

    lineage = resolved[("ANC-01", "hybrid-gat-k1000")]
    assert lineage["original_job_id"] == root["job_id"]
    assert lineage["completed_job"]["job_id"] == child["job_id"]
    assert [row["status"] for row in lineage["attempts"]] == [
        "failed",
        "completed",
    ]
    assert lineage["failed_attempts"][0]["run_id"] == root["run_id"]


def test_campaign_queue_inventory_decodes_registry_rows(
    tmp_path: Path,
) -> None:
    registry = _MODULE.Registry(tmp_path / "tracking.sqlite3")
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="Synthetic hybrid retry inventory",
    )
    config = {"arm": "hybrid-gat-k1000", "attempt": 1}
    root = registry.enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
        command=["run-pilot", "--locked"],
        experiment_config_reference="configs/pilot.yaml",
        maximum_attempts=2,
        requested_gpu="0",
    )

    rows = _MODULE._campaign_queue_jobs(registry)

    assert len(rows) == 1
    assert rows[0]["job_id"] == root["job_id"]
    assert rows[0]["canonical_config"] == config
    assert rows[0]["command"] == ["run-pilot", "--locked"]
    assert rows[0]["attempt_count"] == 1
    assert rows[0]["retry_of"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("branch", "branches"),
        ("duplicate_completion", "failed terminal attempts"),
        ("unrelated_root", "unrelated or duplicate"),
        ("changed_retry", "retry semantics changed"),
        ("no_completed_terminal", "no completed terminal"),
        ("exceeds_maximum", "exceeds maximum_attempts"),
    ],
)
def test_lineage_resolver_fails_closed_on_ambiguous_or_unrelated_jobs(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    enqueue, planned, inventory = _lineage_fixture()
    root, child = _add_successful_retry(enqueue, inventory)
    if mutation == "branch":
        branch = deepcopy(child)
        branch["job_id"] = "retry-branch"
        branch["run_id"] = "completed-branch"
        inventory.append(branch)
    elif mutation == "duplicate_completion":
        root["status"] = "completed"
        root["failure_category"] = None
    elif mutation == "unrelated_root":
        unrelated = deepcopy(child)
        unrelated.update(
            {
                "job_id": "unrelated-root",
                "attempt_count": 1,
                "retry_of": None,
                "run_id": "unrelated-run",
            }
        )
        inventory.append(unrelated)
    elif mutation == "changed_retry":
        child["requested_gpu"] = "7"
    elif mutation == "no_completed_terminal":
        child["status"] = "failed"
        child["failure_category"] = "resource_failure"
    else:
        child["status"] = "failed"
        child["failure_category"] = "resource_failure"
        third = {
            **deepcopy(child),
            "job_id": "attempt-three",
            "status": "completed",
            "attempt_count": 3,
            "retry_of": child["job_id"],
            "run_id": "completed-attempt-three",
            "failure_category": None,
        }
        inventory.append(third)
    monkeypatch.setattr(
        _MODULE,
        "_campaign_queue_jobs",
        lambda _registry: deepcopy(inventory),
    )

    with pytest.raises(GateError, match=message):
        _MODULE._resolve_pilot_attempt_lineages(
            object(),
            enqueue=enqueue,
            planned=planned,
        )


def _verified_job(
    arm: str,
    *,
    passed: bool = True,
    precision_mask: str = "m" * 64,
) -> dict[str, Any]:
    return {
        "alias": "ANC-01",
        "arm": arm,
        "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
        "precision_mask_checksum": precision_mask,
        "evaluation_mask_bundle_sha256": "e" * 64,
        "graph_sha256": "g" * 64,
        "verified_bundle": True,
        "finite_losses_and_gradients": True,
        "parameter_match": True,
        "precision_equivalence_passed": passed,
        "peak_vram_passed": True,
        "projected_runtime_passed": True,
        "runner_pilot_gate_passed": passed,
    }


def _run_gate_with_mocked_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Path]:
    materialization = _materialization()
    enqueue = _enqueue_receipt(materialization)
    materialization_path = tmp_path / "materialization.json"
    enqueue_path = tmp_path / "enqueue.json"
    output = tmp_path / "gate.json"
    _write_json(materialization_path, materialization)
    _write_json(enqueue_path, enqueue)
    monkeypatch.setattr(_MODULE, "Registry", lambda _path: object())
    monkeypatch.setattr(
        _MODULE,
        "_resolve_pilot_attempt_lineages",
        lambda _registry, *, enqueue, planned: {
            key: {"slot": key} for key in planned
        },
    )
    monkeypatch.setattr(
        _MODULE,
        "_verify_one_pilot",
        lambda _registry, receipt_job, planned_job, lineage: deepcopy(
            dict(jobs[str(receipt_job["arm"])])
        ),
    )
    gate = verify_pilot_gate(
        materialization_path=materialization_path,
        enqueue_receipt_path=enqueue_path,
        output_path=output,
        database_path=tmp_path / "tracking.sqlite3",
    )
    return gate, output


def test_gate_writes_checksum_bound_negative_result_for_threshold_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = {
        "hybrid-gat-k1000": _verified_job(
            "hybrid-gat-k1000",
            passed=False,
        ),
        "hybrid-matched-self": _verified_job(
            "hybrid-matched-self",
        ),
    }

    gate, output = _run_gate_with_mocked_jobs(
        tmp_path,
        monkeypatch,
        jobs,
    )

    assert gate["gate_passed"] is False
    assert gate["production_authorized"] is False
    assert gate["failure_reasons"] == [
        "hybrid-gat-k1000:precision_equivalence_passed",
        "hybrid-gat-k1000:runner_pilot_gate_passed",
    ]
    written = json.loads(output.read_text(encoding="utf-8"))
    checksum = written.pop("checksum")
    assert checksum == canonical_sha256(written)


def test_gate_rejects_cross_arm_identity_mismatch_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = {
        "hybrid-gat-k1000": _verified_job(
            "hybrid-gat-k1000",
            precision_mask="a" * 64,
        ),
        "hybrid-matched-self": _verified_job(
            "hybrid-matched-self",
            precision_mask="b" * 64,
        ),
    }

    with pytest.raises(GateError, match="same frozen batch"):
        _run_gate_with_mocked_jobs(tmp_path, monkeypatch, jobs)

    assert not (tmp_path / "gate.json").exists()

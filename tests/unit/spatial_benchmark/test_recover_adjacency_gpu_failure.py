from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/recover_adjacency_gpu_failure.py"
_SPEC = importlib.util.spec_from_file_location(
    "recover_adjacency_gpu_failure_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _config(fold: int, seed: int, condition: str) -> dict[str, object]:
    return {
        "experiment": {"stage": "primary"},
        "graph": {"adjacency_condition": condition},
        "fold": fold,
        "seed": seed,
        "attempt": 1,
    }


def _command(fold: int, seed: int, condition: str) -> list[str]:
    return ["python", "runner.py", str(fold), str(seed), condition]


def _plan() -> dict[str, object]:
    primary: list[dict[str, object]] = []
    index = 0
    for fold in _MODULE.FOLDS:
        for seed in _MODULE.SEEDS:
            for condition in _MODULE.CONDITIONS:
                root_id = f"q_root_{index:02d}"
                run_id = f"r_root_{index:02d}"
                if index < 7:
                    status = "completed"
                    role = "completed_before_fault"
                    evidence = None
                elif index == 7:
                    status = "failed"
                    role = "initial_cuda_launch_failure"
                    evidence = "e" * 64
                    root_id = _MODULE.FIRST_FAILED_JOB_ID
                    run_id = _MODULE.FIRST_FAILED_RUN_ID
                else:
                    status = "failed"
                    role = "post_fault_cuda_initialization_failure"
                    evidence = "e" * 64
                config = _config(fold, seed, condition)
                command = _command(fold, seed, condition)
                marker = "_SUCCESS" if status == "completed" else "_FAILED"
                primary.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "attempt": 2,
                        "root_job_id": root_id,
                        "root_run_id": run_id,
                        "root_status": status,
                        "root_requested_gpu": _MODULE._slot(fold, seed),
                        "worker_slot": _MODULE._slot(fold, seed),
                        "retry_job_id": _MODULE._retry_job_id(root_id),
                        "canonical_config_sha256": canonical_sha256(config),
                        "command_sha256": canonical_sha256(command),
                        "config_reference": (
                            f"scratch/configs/{fold}-{seed}-{condition}.yaml"
                        ),
                        "bundle": {
                            "reference": f"artifacts/runs/{run_id}",
                            "status": (
                                "success" if status == "completed" else "failed"
                            ),
                            "marker": marker,
                            "marker_sha256": "a" * 64,
                            "checksum_manifest_sha256": "b" * 64,
                            "verified_file_count": 12,
                        },
                        "failure_role": role,
                        "evidence_sha256": evidence,
                    }
                )
                index += 1
    null = []
    for fold in _MODULE.FOLDS:
        for seed in _MODULE.SEEDS:
            null.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "condition": _MODULE.NULL_CONDITION,
                    "attempt": 1,
                    "worker_slot": _MODULE._slot(fold, seed),
                    "canonical_config_sha256": "c" * 64,
                    "command_sha256": "d" * 64,
                    "config_reference": f"scratch/null/{fold}-{seed}.yaml",
                }
            )
    return _MODULE._signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PLAN_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "recovery_contract": {
                "reference": _MODULE.RECOVERY_CONTRACT_REFERENCE.as_posix(),
                "sha256": _MODULE.RECOVERY_CONTRACT_SHA256,
            },
            "scientific_contract_sha256": _MODULE.SCIENTIFIC_CONTRACT_SHA256,
            "materialization": {
                "reference": _MODULE.MATERIALIZATION_REFERENCE.as_posix(),
                "checksum": _MODULE.MATERIALIZATION_CHECKSUM,
            },
            "execution": {
                "device": "cpu",
                "primary_attempt": 2,
                "conditional_null_attempt": 1,
                "maximum_attempts": 2,
                "worker_slots": list(_MODULE.WORKER_SLOTS),
                "torch_intraop_threads_per_process": 4,
                "torch_interop_threads_per_process": 1,
                "concurrent_workers": 8,
                "normalized_resolved_config_delta": {"attempt": [1, 2]},
            },
            "inventory": {
                "primary_roots": 50,
                "primary_completed_attempt_1": 7,
                "primary_failed_attempt_1": 43,
                "primary_retry_slots": 50,
                "conditional_null_slots": 25,
            },
            "failure_classification": {
                "kind": _MODULE.FAILURE_KIND,
                "first_failed_job_id": _MODULE.FIRST_FAILED_JOB_ID,
                "first_failed_run_id": _MODULE.FIRST_FAILED_RUN_ID,
                "failed_pci_function": "0000:0f:00.0",
                "failed_requested_gpu": 7,
                "launch_failure_count": 1,
                "initialization_failure_count": 42,
                "completed_before_fault_count": 7,
            },
            "primary_jobs": primary,
            "conditional_null_jobs": null,
        }
    )


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_plan_loader_is_signed_complete_and_failure_role_strict(
    tmp_path: Path,
) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    _write(path, plan)
    assert _MODULE.load_recovery_plan(path) == plan

    tampered = deepcopy(plan)
    tampered["primary_jobs"][8]["failure_role"] = (
        "initial_cuda_launch_failure"
    )
    tampered.pop("checksum")
    tampered = _MODULE._signed(tampered)
    _write(path, tampered)
    with pytest.raises(_MODULE.AdjacencyRecoveryError, match="slot inventory"):
        _MODULE.load_recovery_plan(path)


def test_plan_rejects_a_tampered_root_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = []
    roots = []
    for fold in _MODULE.FOLDS:
        for seed in _MODULE.SEEDS:
            for condition in _MODULE.CONDITIONS:
                config = _config(fold, seed, condition)
                materialized.append(
                    {
                        "stage": "primary",
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "config_payload": config,
                        "config_sha256": canonical_sha256(config),
                        "config_reference": f"primary/{fold}-{seed}-{condition}",
                    }
                )
                roots.append(
                    {
                        "job_id": f"root-{fold}-{seed}-{condition}",
                        "run_id": f"run-{fold}-{seed}-{condition}",
                        "canonical_config": config,
                        "command": ["tampered"] if not roots else ["expected"],
                        "experiment_config_reference": (
                            f"primary/{fold}-{seed}-{condition}"
                        ),
                        "attempt_count": 1,
                        "maximum_attempts": 1,
                        "retry_of": None,
                        "status": "completed",
                        "requested_gpu": str(_MODULE._slot(fold, seed)),
                    }
                )
    for fold in _MODULE.FOLDS:
        for seed in _MODULE.SEEDS:
            config = {
                "experiment": {"stage": "null"},
                "graph": {"adjacency_condition": _MODULE.NULL_CONDITION},
                "fold": fold,
                "seed": seed,
                "attempt": 1,
            }
            materialized.append(
                {
                    "stage": "null",
                    "fold": fold,
                    "seed": seed,
                    "condition": _MODULE.NULL_CONDITION,
                    "config_payload": config,
                    "config_sha256": canonical_sha256(config),
                    "config_reference": f"null/{fold}-{seed}",
                }
            )
    monkeypatch.setattr(_MODULE, "_validate_recovery_contract", lambda _: {})
    monkeypatch.setattr(
        _MODULE, "_load_materialization", lambda _: ({"jobs": materialized}, {})
    )
    monkeypatch.setattr(_MODULE, "_primary_root_jobs", lambda _: roots)
    monkeypatch.setattr(_MODULE, "command_for_config", lambda _: ["expected"])
    registry = type(
        "FakeRegistry", (), {"get_run": lambda self, _: {"status": "completed"}}
    )()
    with pytest.raises(_MODULE.AdjacencyRecoveryError, match="queue configuration"):
        _MODULE._assemble_plan(
            registry=registry,
            materialization_path=Path("unused"),
            recovery_contract_path=Path("unused"),
        )


def test_bundle_failure_evidence_classifies_only_the_observed_cuda_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    monkeypatch.setattr(_MODULE, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        _MODULE,
        "verify_run_bundle",
        lambda *_args, **_kwargs: {"status": "failed", "file_count": 4},
    )

    def failed_bundle(run_id: str, stderr: str) -> Path:
        bundle = project / "artifacts/runs/2026/08" / run_id
        (bundle / "logs").mkdir(parents=True)
        (bundle / "provenance").mkdir()
        (bundle / "_FAILED").write_text("{}", encoding="utf-8")
        (bundle / "provenance/artifact_checksums.json").write_text(
            "{}", encoding="utf-8"
        )
        (bundle / "logs/stderr.log").write_text(stderr, encoding="utf-8")
        return bundle

    launch_bundle = failed_bundle(
        _MODULE.FIRST_FAILED_RUN_ID,
        "CUDA error: unspecified launch failure\n",
    )
    _, role, evidence = _MODULE._bundle_record(
        root_job={
            "job_id": _MODULE.FIRST_FAILED_JOB_ID,
            "run_id": _MODULE.FIRST_FAILED_RUN_ID,
            "status": "failed",
        },
        run={"artifact_path": str(launch_bundle)},
    )
    assert role == "initial_cuda_launch_failure"
    assert evidence == _MODULE._sha256_file(launch_bundle / "logs/stderr.log")

    init_run_id = "r_initialization_cascade"
    init_bundle = failed_bundle(
        init_run_id,
        "CUDA initialization: CUDA unknown error\n"
        "queue-owned runs require a visible CUDA GPU\n",
    )
    _, role, _ = _MODULE._bundle_record(
        root_job={
            "job_id": "q_later_failure",
            "run_id": init_run_id,
            "status": "failed",
        },
        run={"artifact_path": str(init_bundle)},
    )
    assert role == "post_fault_cuda_initialization_failure"

    unrelated_run_id = "r_unrelated_failure"
    unrelated_bundle = failed_bundle(unrelated_run_id, "unrelated error\n")
    with pytest.raises(_MODULE.AdjacencyRecoveryError, match="outside"):
        _MODULE._bundle_record(
            root_job={
                "job_id": "q_unrelated_failure",
                "run_id": unrelated_run_id,
                "status": "failed",
            },
            run={"artifact_path": str(unrelated_bundle)},
        )


def test_enqueue_inserts_exactly_one_attempt_two_retry_per_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan_path = tmp_path / _MODULE.PLAN_FILENAME
    receipt_path = tmp_path / _MODULE.ENQUEUE_FILENAME
    _write(plan_path, plan)
    database = tmp_path / "registry.sqlite3"
    registry = Registry(database)
    registry.initialize()
    registry.create_campaign(_MODULE.CAMPAIGN_ID, name="fixture")
    for item in plan["primary_jobs"]:
        fold = int(item["fold"])
        seed = int(item["seed"])
        condition = str(item["condition"])
        registry.enqueue(
            campaign_id=_MODULE.CAMPAIGN_ID,
            configuration=_config(fold, seed, condition),
            command=_command(fold, seed, condition),
            experiment_config_reference=str(item["config_reference"]),
            priority=50,
            maximum_attempts=1,
            requested_gpu=str(item["worker_slot"]),
            job_id=str(item["root_job_id"]),
        )
    monkeypatch.setattr(_MODULE, "_verify_plan_against_state", lambda **_: None)
    kwargs = {
        "database_path": database,
        "materialization_path": tmp_path / "unused-materialization",
        "recovery_contract_path": tmp_path / "unused-contract",
        "plan_path": plan_path,
        "receipt_path": receipt_path,
    }
    first = _MODULE.enqueue_recovery(**kwargs)
    second = _MODULE.enqueue_recovery(**kwargs)
    assert first == second
    assert len(first["primary_retries"]) == 50
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM queue_jobs WHERE retry_of IS NOT NULL"
        ).fetchall()
    assert len(rows) == 50
    assert {int(row["attempt_count"]) for row in rows} == {2}
    assert {int(row["maximum_attempts"]) for row in rows} == {2}
    assert {str(row["status"]) for row in rows} == {"queued"}
    assert {str(row["requested_gpu"]) for row in rows} == {
        str(slot) for slot in _MODULE.WORKER_SLOTS
    }

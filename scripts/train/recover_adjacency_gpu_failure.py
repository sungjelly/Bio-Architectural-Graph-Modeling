#!/usr/bin/env python3
"""Plan and enqueue the frozen CPU recovery after the host CUDA failure."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260802_adjacent_normal_grouped_adjacency_ablation"
SCIENTIFIC_CONTRACT_SHA256 = (
    "08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff"
)
RECOVERY_CONTRACT_SHA256 = (
    "84230db441cd03d158a28bc79dfecee495fee1b0864dae2311a0a9041f7b6bb0"
)
MATERIALIZATION_CHECKSUM = (
    "0cc606d73a4979808581a032138cc18ef629952ed1a4884903ecd229bdfc4303"
)
FIRST_FAILED_JOB_ID = "q_e51d0dc30ba40bfecd1f"
FIRST_FAILED_RUN_ID = "r_20260802T162033Z_002d9ede_s002_f01_a01_9e38bd17"
RECOVERY_CONTRACT_REFERENCE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "hardware_recovery_contract.yaml"
)
MATERIALIZATION_REFERENCE = (
    Path("scratch/locked_campaigns")
    / CAMPAIGN_ID
    / "materialization_receipt.json"
)
PLAN_FILENAME = "primary_cpu_recovery_plan_receipt.json"
ENQUEUE_FILENAME = "primary_cpu_recovery_enqueue_receipt.json"
PLAN_KIND = "adjacency_ablation_cpu_recovery_plan_v1"
ENQUEUE_KIND = "adjacency_ablation_cpu_recovery_enqueue_v1"
FAILURE_KIND = "cuda_launch_then_global_initialization_cascade"
CONDITIONS = ("spatial", "isolated")
NULL_CONDITION = "position_permuted_null"
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
WORKER_SLOTS = tuple(range(8))
_HEX = frozenset("0123456789abcdef")

_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "campaign_id",
        "recovery_contract",
        "scientific_contract_sha256",
        "materialization",
        "execution",
        "inventory",
        "failure_classification",
        "primary_jobs",
        "conditional_null_jobs",
        "checksum",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "device",
        "primary_attempt",
        "conditional_null_attempt",
        "maximum_attempts",
        "worker_slots",
        "torch_intraop_threads_per_process",
        "torch_interop_threads_per_process",
        "concurrent_workers",
        "normalized_resolved_config_delta",
    }
)
_INVENTORY_KEYS = frozenset(
    {
        "primary_roots",
        "primary_completed_attempt_1",
        "primary_failed_attempt_1",
        "primary_retry_slots",
        "conditional_null_slots",
    }
)
_FAILURE_KEYS = frozenset(
    {
        "kind",
        "first_failed_job_id",
        "first_failed_run_id",
        "failed_pci_function",
        "failed_requested_gpu",
        "launch_failure_count",
        "initialization_failure_count",
        "completed_before_fault_count",
    }
)
_PRIMARY_JOB_KEYS = frozenset(
    {
        "fold",
        "seed",
        "condition",
        "attempt",
        "root_job_id",
        "root_run_id",
        "root_status",
        "root_requested_gpu",
        "worker_slot",
        "retry_job_id",
        "canonical_config_sha256",
        "command_sha256",
        "config_reference",
        "bundle",
        "failure_role",
        "evidence_sha256",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "reference",
        "status",
        "marker",
        "marker_sha256",
        "checksum_manifest_sha256",
        "verified_file_count",
    }
)
_NULL_JOB_KEYS = frozenset(
    {
        "fold",
        "seed",
        "condition",
        "attempt",
        "worker_slot",
        "canonical_config_sha256",
        "command_sha256",
        "config_reference",
    }
)
_ENQUEUE_KEYS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "campaign_id",
        "recovery_contract_sha256",
        "plan_reference",
        "plan_checksum",
        "complete",
        "inventory",
        "primary_retries",
        "checksum",
    }
)
_ENQUEUE_INVENTORY_KEYS = frozenset(
    {"primary_retries", "attempt", "maximum_attempts"}
)
_RETRY_KEYS = frozenset(
    {
        "fold",
        "seed",
        "condition",
        "root_job_id",
        "retry_job_id",
        "attempt_count",
        "maximum_attempts",
        "retry_of",
        "worker_slot",
        "canonical_config_sha256",
        "command_sha256",
        "config_reference",
        "queue_status_at_receipt",
    }
)


class AdjacencyRecoveryError(RuntimeError):
    """Raised when recovery evidence or a retry identity is not exact."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjacencyRecoveryError(f"{label} must be a mapping")
    return value


def _require_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise AdjacencyRecoveryError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise AdjacencyRecoveryError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AdjacencyRecoveryError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdjacencyRecoveryError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except AdjacencyRecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdjacencyRecoveryError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _verify_checksum(payload: Mapping[str, Any], label: str) -> str:
    checksum = _hex_digest(payload.get("checksum"), f"{label} checksum")
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise AdjacencyRecoveryError(f"{label} checksum does not verify")
    return checksum


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["checksum"] = canonical_sha256(result)
    return result


def _slot(fold: int, seed: int) -> int:
    return (fold * len(SEEDS) + seed) % len(WORKER_SLOTS)


def _retry_job_id(root_job_id: str) -> str:
    digest = hashlib.sha256(
        (
            RECOVERY_CONTRACT_SHA256
            + "\0"
            + root_job_id
            + "\0primary-attempt-2"
        ).encode("utf-8")
    ).hexdigest()
    return "q_" + digest[:20]


def _project_reference(path: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise AdjacencyRecoveryError(
            f"{label} must remain under the BAGM project root"
        ) from exc


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != content:
            raise AdjacencyRecoveryError(
                f"refusing to overwrite immutable recovery receipt {path}"
            )
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _campaign_lock(locked_root: Path) -> Iterator[None]:
    lock_path = locked_root / ".primary-cpu-recovery.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_recovery_contract(path: Path) -> dict[str, Any]:
    if not path.is_file() or _sha256_file(path) != RECOVERY_CONTRACT_SHA256:
        raise AdjacencyRecoveryError(
            "hardware recovery contract is missing or its checksum changed"
        )
    contract = load_yaml_mapping(path)
    observed = _mapping(contract.get("observed_failure"), "observed failure")
    design = _mapping(contract.get("recovery_design"), "recovery design")
    conditional = _mapping(contract.get("conditional_null"), "conditional null")
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("recovery_kind")
        != "uniform_cpu_reexecution_after_host_cuda_failure"
        or contract.get("scientific_contract_sha256")
        != SCIENTIFIC_CONTRACT_SHA256
        or contract.get("materialization_checksum")
        != MATERIALIZATION_CHECKSUM
        or observed.get("first_failed_job_id") != FIRST_FAILED_JOB_ID
        or observed.get("first_failed_run_id") != FIRST_FAILED_RUN_ID
        or observed.get("completed_primary_attempt_1_runs") != 7
        or observed.get("failed_primary_attempt_1_runs") != 43
        or design.get("primary_scope") != "all_50_fold_seed_condition_slots"
        or design.get("primary_attempt") != 2
        or design.get("execution_device") != "cpu"
        or design.get("requested_worker_slots") != list(WORKER_SLOTS)
        or design.get("torch_intraop_threads_per_process") != 4
        or design.get("torch_interop_threads_per_process") != 1
        or design.get("concurrent_workers") != 8
        or design.get("permitted_resolved_config_difference")
        != "attempt_1_to_attempt_2_only"
        or conditional.get("execution_device") != "cpu"
        or conditional.get("attempt") != 1
        or conditional.get("run_count_if_triggered") != 25
    ):
        raise AdjacencyRecoveryError(
            "hardware recovery contract differs from the frozen recovery design"
        )
    return contract


def _validate_primary_job_record(job: Mapping[str, Any]) -> tuple[int, int, str]:
    _require_keys(job, _PRIMARY_JOB_KEYS, "primary recovery job")
    bundle = _mapping(job.get("bundle"), "primary recovery bundle")
    _require_keys(bundle, _BUNDLE_KEYS, "primary recovery bundle")
    fold = int(job.get("fold", -1))
    seed = int(job.get("seed", -1))
    condition = str(job.get("condition"))
    if (
        fold not in FOLDS
        or seed not in SEEDS
        or condition not in CONDITIONS
        or job.get("attempt") != 2
        or job.get("worker_slot") != _slot(fold, seed)
        or job.get("root_requested_gpu") != _slot(fold, seed)
        or job.get("retry_job_id") != _retry_job_id(str(job.get("root_job_id")))
        or job.get("root_status") not in {"completed", "failed"}
        or bundle.get("status")
        != ("success" if job.get("root_status") == "completed" else "failed")
        or bundle.get("marker")
        != ("_SUCCESS" if job.get("root_status") == "completed" else "_FAILED")
    ):
        raise AdjacencyRecoveryError("primary recovery job identity is invalid")
    for key in (
        "canonical_config_sha256",
        "command_sha256",
        "marker_sha256",
        "checksum_manifest_sha256",
    ):
        source = job if key in job else bundle
        _hex_digest(source.get(key), f"primary job {key}")
    if job.get("root_status") == "completed":
        if (
            job.get("failure_role") != "completed_before_fault"
            or job.get("evidence_sha256") is not None
        ):
            raise AdjacencyRecoveryError(
                "completed root has an invalid failure classification"
            )
    else:
        if job.get("failure_role") not in {
            "initial_cuda_launch_failure",
            "post_fault_cuda_initialization_failure",
        }:
            raise AdjacencyRecoveryError(
                "failed root has an invalid failure classification"
            )
        _hex_digest(job.get("evidence_sha256"), "failure evidence SHA-256")
    return fold, seed, condition


def _validate_null_job_record(job: Mapping[str, Any]) -> tuple[int, int, str]:
    _require_keys(job, _NULL_JOB_KEYS, "conditional null recovery job")
    fold = int(job.get("fold", -1))
    seed = int(job.get("seed", -1))
    condition = str(job.get("condition"))
    if (
        fold not in FOLDS
        or seed not in SEEDS
        or condition != NULL_CONDITION
        or job.get("attempt") != 1
        or job.get("worker_slot") != _slot(fold, seed)
    ):
        raise AdjacencyRecoveryError("conditional null authorization is invalid")
    _hex_digest(job.get("canonical_config_sha256"), "null config SHA-256")
    _hex_digest(job.get("command_sha256"), "null command SHA-256")
    return fold, seed, condition


def _validate_recovery_plan_payload(
    raw_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(raw_payload)
    _require_keys(payload, _PLAN_KEYS, "CPU recovery plan")
    _verify_checksum(payload, "CPU recovery plan")
    recovery = _mapping(payload.get("recovery_contract"), "recovery contract")
    materialization = _mapping(payload.get("materialization"), "materialization")
    execution = _mapping(payload.get("execution"), "recovery execution")
    inventory = _mapping(payload.get("inventory"), "recovery inventory")
    failure = _mapping(
        payload.get("failure_classification"), "failure classification"
    )
    _require_keys(execution, _EXECUTION_KEYS, "recovery execution")
    _require_keys(inventory, _INVENTORY_KEYS, "recovery inventory")
    _require_keys(failure, _FAILURE_KEYS, "failure classification")
    if (
        payload.get("schema_version") != 1
        or payload.get("receipt_kind") != PLAN_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or recovery
        != {
            "reference": RECOVERY_CONTRACT_REFERENCE.as_posix(),
            "sha256": RECOVERY_CONTRACT_SHA256,
        }
        or payload.get("scientific_contract_sha256")
        != SCIENTIFIC_CONTRACT_SHA256
        or materialization
        != {
            "reference": MATERIALIZATION_REFERENCE.as_posix(),
            "checksum": MATERIALIZATION_CHECKSUM,
        }
        or execution
        != {
            "device": "cpu",
            "primary_attempt": 2,
            "conditional_null_attempt": 1,
            "maximum_attempts": 2,
            "worker_slots": list(WORKER_SLOTS),
            "torch_intraop_threads_per_process": 4,
            "torch_interop_threads_per_process": 1,
            "concurrent_workers": 8,
            "normalized_resolved_config_delta": {"attempt": [1, 2]},
        }
        or inventory
        != {
            "primary_roots": 50,
            "primary_completed_attempt_1": 7,
            "primary_failed_attempt_1": 43,
            "primary_retry_slots": 50,
            "conditional_null_slots": 25,
        }
        or failure
        != {
            "kind": FAILURE_KIND,
            "first_failed_job_id": FIRST_FAILED_JOB_ID,
            "first_failed_run_id": FIRST_FAILED_RUN_ID,
            "failed_pci_function": "0000:0f:00.0",
            "failed_requested_gpu": 7,
            "launch_failure_count": 1,
            "initialization_failure_count": 42,
            "completed_before_fault_count": 7,
        }
    ):
        raise AdjacencyRecoveryError(
            "CPU recovery plan differs from the frozen authorization"
        )
    primary = payload.get("primary_jobs")
    conditional_null = payload.get("conditional_null_jobs")
    if not isinstance(primary, list) or not isinstance(conditional_null, list):
        raise AdjacencyRecoveryError("recovery job inventories must be lists")
    primary_slots = [
        _validate_primary_job_record(_mapping(job, "primary job"))
        for job in primary
    ]
    null_slots = [
        _validate_null_job_record(_mapping(job, "null job"))
        for job in conditional_null
    ]
    expected_primary = {
        (fold, seed, condition)
        for fold in FOLDS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    expected_null = {
        (fold, seed, NULL_CONDITION) for fold in FOLDS for seed in SEEDS
    }
    if (
        len(primary_slots) != 50
        or set(primary_slots) != expected_primary
        or len(set(primary_slots)) != 50
        or len(null_slots) != 25
        or set(null_slots) != expected_null
        or len(set(null_slots)) != 25
        or len({job["root_job_id"] for job in primary}) != 50
        or len({job["root_run_id"] for job in primary}) != 50
        or len({job["retry_job_id"] for job in primary}) != 50
        or sum(job["root_status"] == "completed" for job in primary) != 7
        or sum(job["root_status"] == "failed" for job in primary) != 43
        or sum(
            job["failure_role"] == "completed_before_fault" for job in primary
        )
        != 7
        or sum(
            job["failure_role"] == "initial_cuda_launch_failure"
            for job in primary
        )
        != 1
        or sum(
            job["failure_role"]
            == "post_fault_cuda_initialization_failure"
            for job in primary
        )
        != 42
    ):
        raise AdjacencyRecoveryError("recovery plan slot inventory is incomplete")
    launch_record = next(
        job
        for job in primary
        if job["failure_role"] == "initial_cuda_launch_failure"
    )
    if (
        launch_record["root_job_id"] != FIRST_FAILED_JOB_ID
        or launch_record["root_run_id"] != FIRST_FAILED_RUN_ID
    ):
        raise AdjacencyRecoveryError(
            "initial CUDA launch-failure identity differs from the contract"
        )
    return payload


def load_recovery_plan(path: str | Path) -> dict[str, Any]:
    """Load and strictly verify the signed CPU recovery authorization plan."""

    return _validate_recovery_plan_payload(
        _strict_json(Path(path), "CPU recovery plan")
    )


def _validate_recovery_enqueue_payload(
    raw_payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validated_plan = _validate_recovery_plan_payload(plan)
    plan_checksum = str(validated_plan["checksum"])
    payload = dict(raw_payload)
    _require_keys(payload, _ENQUEUE_KEYS, "CPU recovery enqueue receipt")
    _verify_checksum(payload, "CPU recovery enqueue receipt")
    inventory = _mapping(payload.get("inventory"), "enqueue inventory")
    _require_keys(inventory, _ENQUEUE_INVENTORY_KEYS, "enqueue inventory")
    retries = payload.get("primary_retries")
    if (
        payload.get("schema_version") != 1
        or payload.get("receipt_kind") != ENQUEUE_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("recovery_contract_sha256")
        != RECOVERY_CONTRACT_SHA256
        or payload.get("plan_reference")
        != (
            Path("scratch/locked_campaigns") / CAMPAIGN_ID / PLAN_FILENAME
        ).as_posix()
        or payload.get("plan_checksum") != plan_checksum
        or payload.get("complete") is not True
        or inventory
        != {"primary_retries": 50, "attempt": 2, "maximum_attempts": 2}
        or not isinstance(retries, list)
        or len(retries) != 50
    ):
        raise AdjacencyRecoveryError(
            "CPU recovery enqueue receipt differs from its plan"
        )
    expected = {
        str(job["retry_job_id"]): job for job in plan["primary_jobs"]
    }
    seen: set[str] = set()
    for raw in retries:
        retry = _mapping(raw, "primary retry receipt job")
        _require_keys(retry, _RETRY_KEYS, "primary retry receipt job")
        retry_id = str(retry.get("retry_job_id"))
        planned = expected.get(retry_id)
        if (
            planned is None
            or retry_id in seen
            or retry.get("fold") != planned.get("fold")
            or retry.get("seed") != planned.get("seed")
            or retry.get("condition") != planned.get("condition")
            or retry.get("root_job_id") != planned.get("root_job_id")
            or retry.get("attempt_count") != 2
            or retry.get("maximum_attempts") != 2
            or retry.get("retry_of") != planned.get("root_job_id")
            or retry.get("worker_slot") != planned.get("worker_slot")
            or retry.get("canonical_config_sha256")
            != planned.get("canonical_config_sha256")
            or retry.get("command_sha256") != planned.get("command_sha256")
            or retry.get("config_reference") != planned.get("config_reference")
            or retry.get("queue_status_at_receipt") != "queued"
        ):
            raise AdjacencyRecoveryError(
                "primary retry receipt identity differs from the plan"
            )
        seen.add(retry_id)
    if set(seen) != set(expected):
        raise AdjacencyRecoveryError("primary retry receipt is incomplete")
    return payload


def load_recovery_enqueue(
    path: str | Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Load and strictly verify the signed primary retry enqueue receipt."""

    return _validate_recovery_enqueue_payload(
        _strict_json(Path(path), "CPU recovery enqueue receipt"), plan=plan
    )


def _load_materialization(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from enqueue_adjacency_ablation import _load_materialization as loader

        receipt, manifest = loader(path)
    except Exception as exc:
        raise AdjacencyRecoveryError(
            "primary materialization verification failed"
        ) from exc
    if receipt.get("checksum") != MATERIALIZATION_CHECKSUM:
        raise AdjacencyRecoveryError(
            "materialization checksum differs from the recovery contract"
        )
    return receipt, manifest


def _primary_root_jobs(registry: Registry) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        identifiers = [
            str(row["job_id"])
            for row in connection.execute(
                """
                SELECT job_id FROM queue_jobs
                WHERE campaign_id = ? AND retry_of IS NULL
                ORDER BY created_at, job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
    result: list[dict[str, Any]] = []
    for identifier in identifiers:
        job = registry.get_job(identifier)
        if job is None:
            raise AdjacencyRecoveryError("root queue job disappeared during audit")
        config = _mapping(job.get("canonical_config"), "root canonical config")
        experiment = _mapping(config.get("experiment"), "root experiment")
        if experiment.get("stage") == "primary":
            result.append(job)
    if len(result) != 50:
        raise AdjacencyRecoveryError(
            f"expected exactly 50 primary root jobs, found {len(result)}"
        )
    return result


def _bundle_record(
    *, root_job: Mapping[str, Any], run: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str | None]:
    run_id = str(root_job.get("run_id"))
    artifact_path = Path(str(run.get("artifact_path", "")))
    if artifact_path.name != run_id:
        raise AdjacencyRecoveryError("root run artifact path has wrong identity")
    status = str(root_job.get("status"))
    verification = verify_run_bundle(
        artifact_path, require_success_contract=status == "completed"
    )
    expected_bundle_status = "success" if status == "completed" else "failed"
    if verification.get("status") != expected_bundle_status:
        raise AdjacencyRecoveryError(
            "root registry status differs from its verified immutable bundle"
        )
    marker = "_SUCCESS" if status == "completed" else "_FAILED"
    marker_path = artifact_path / marker
    checksum_path = artifact_path / "provenance/artifact_checksums.json"
    record = {
        "reference": _project_reference(artifact_path, "root run bundle"),
        "status": expected_bundle_status,
        "marker": marker,
        "marker_sha256": _sha256_file(marker_path),
        "checksum_manifest_sha256": _sha256_file(checksum_path),
        "verified_file_count": int(verification["file_count"]),
    }
    if status == "completed":
        return record, "completed_before_fault", None
    stderr_path = artifact_path / "logs/stderr.log"
    stderr = stderr_path.read_text(encoding="utf-8", errors="strict")
    if str(root_job.get("job_id")) == FIRST_FAILED_JOB_ID:
        if (
            run_id != FIRST_FAILED_RUN_ID
            or "CUDA error: unspecified launch failure" not in stderr
        ):
            raise AdjacencyRecoveryError(
                "first CUDA launch failure evidence does not match the contract"
            )
        role = "initial_cuda_launch_failure"
    elif (
        "CUDA initialization: CUDA unknown error" in stderr
        and "queue-owned runs require a visible CUDA GPU" in stderr
    ):
        role = "post_fault_cuda_initialization_failure"
    else:
        raise AdjacencyRecoveryError(
            f"failed run {run_id} is outside the observed CUDA failure cascade"
        )
    return record, role, _sha256_file(stderr_path)


def _assemble_plan(
    *,
    registry: Registry,
    materialization_path: Path,
    recovery_contract_path: Path,
) -> dict[str, Any]:
    _validate_recovery_contract(recovery_contract_path)
    materialization, _ = _load_materialization(materialization_path)
    materialized_primary = {
        (int(item["fold"]), int(item["seed"]), str(item["condition"])): item
        for item in materialization["jobs"]
        if item["stage"] == "primary"
    }
    materialized_null = {
        (int(item["fold"]), int(item["seed"]), str(item["condition"])): item
        for item in materialization["jobs"]
        if item["stage"] == "null"
    }
    expected_primary = {
        (fold, seed, condition)
        for fold in FOLDS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    expected_null = {
        (fold, seed, NULL_CONDITION) for fold in FOLDS for seed in SEEDS
    }
    if (
        set(materialized_primary) != expected_primary
        or set(materialized_null) != expected_null
    ):
        raise AdjacencyRecoveryError(
            "materialization lacks the exact primary or conditional-null slots"
        )

    observed_slots: set[tuple[int, int, str]] = set()
    primary_records: list[dict[str, Any]] = []
    launch_failures = 0
    initialization_failures = 0
    completed = 0
    for root_job in _primary_root_jobs(registry):
        config = _mapping(root_job.get("canonical_config"), "root config")
        fold = int(config.get("fold", -1))
        seed = int(config.get("seed", -1))
        graph = _mapping(config.get("graph"), "root graph")
        condition = str(graph.get("adjacency_condition"))
        slot = (fold, seed, condition)
        materialized = materialized_primary.get(slot)
        if slot in observed_slots or materialized is None:
            raise AdjacencyRecoveryError("root primary slot is duplicate or unknown")
        observed_slots.add(slot)
        run_id = str(root_job.get("run_id", ""))
        run = registry.get_run(run_id)
        status = str(root_job.get("status"))
        worker_slot = _slot(fold, seed)
        expected_command = command_for_config(
            _mapping(materialized.get("config_payload"), "materialized config")
        )
        if (
            root_job.get("attempt_count") != 1
            or root_job.get("maximum_attempts") != 1
            or root_job.get("retry_of") is not None
            or status not in {"completed", "failed"}
            or str(root_job.get("requested_gpu")) != str(worker_slot)
            or config.get("attempt") != 1
            or config != materialized.get("config_payload")
            or canonical_sha256(config) != materialized.get("config_sha256")
            or root_job.get("command") != expected_command
        ):
            raise AdjacencyRecoveryError(
                "root primary queue configuration or attempt identity changed"
            )
        expected_command_sha = canonical_sha256(expected_command)
        if (
            str(root_job.get("experiment_config_reference"))
            != str(materialized.get("config_reference"))
            or run is None
            or run.get("status") != status
            or run.get("config") != config
            or run.get("attempt") != 1
            or run.get("seed") != seed
            or run.get("fold") != fold
            or run.get("failure_category")
            != (None if status == "completed" else "nonzero_exit")
        ):
            raise AdjacencyRecoveryError(
                "root queue, run, and materialization identities disagree"
            )
        bundle, failure_role, evidence_sha = _bundle_record(
            root_job=root_job, run=run
        )
        if failure_role == "completed_before_fault":
            completed += 1
        elif failure_role == "initial_cuda_launch_failure":
            launch_failures += 1
        else:
            initialization_failures += 1
        primary_records.append(
            {
                "fold": fold,
                "seed": seed,
                "condition": condition,
                "attempt": 2,
                "root_job_id": str(root_job["job_id"]),
                "root_run_id": run_id,
                "root_status": status,
                "root_requested_gpu": worker_slot,
                "worker_slot": worker_slot,
                "retry_job_id": _retry_job_id(str(root_job["job_id"])),
                "canonical_config_sha256": canonical_sha256(config),
                "command_sha256": expected_command_sha,
                "config_reference": str(root_job["experiment_config_reference"]),
                "bundle": bundle,
                "failure_role": failure_role,
                "evidence_sha256": evidence_sha,
            }
        )
    if (
        observed_slots != expected_primary
        or completed != 7
        or launch_failures != 1
        or initialization_failures != 42
    ):
        raise AdjacencyRecoveryError(
            "root outcome or CUDA cascade counts differ from the recovery contract"
        )

    null_records = [
        {
            "fold": fold,
            "seed": seed,
            "condition": NULL_CONDITION,
            "attempt": 1,
            "worker_slot": _slot(fold, seed),
            "canonical_config_sha256": str(
                materialized_null[(fold, seed, NULL_CONDITION)]["config_sha256"]
            ),
            "command_sha256": canonical_sha256(
                command_for_config(
                    materialized_null[(fold, seed, NULL_CONDITION)]["config_payload"]
                )
            ),
            "config_reference": str(
                materialized_null[(fold, seed, NULL_CONDITION)]["config_reference"]
            ),
        }
        for fold in FOLDS
        for seed in SEEDS
    ]
    primary_records.sort(
        key=lambda item: (item["fold"], item["seed"], item["condition"])
    )
    payload = _signed(
        {
            "schema_version": 1,
            "receipt_kind": PLAN_KIND,
            "campaign_id": CAMPAIGN_ID,
            "recovery_contract": {
                "reference": RECOVERY_CONTRACT_REFERENCE.as_posix(),
                "sha256": RECOVERY_CONTRACT_SHA256,
            },
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT_SHA256,
            "materialization": {
                "reference": MATERIALIZATION_REFERENCE.as_posix(),
                "checksum": MATERIALIZATION_CHECKSUM,
            },
            "execution": {
                "device": "cpu",
                "primary_attempt": 2,
                "conditional_null_attempt": 1,
                "maximum_attempts": 2,
                "worker_slots": list(WORKER_SLOTS),
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
                "kind": FAILURE_KIND,
                "first_failed_job_id": FIRST_FAILED_JOB_ID,
                "first_failed_run_id": FIRST_FAILED_RUN_ID,
                "failed_pci_function": "0000:0f:00.0",
                "failed_requested_gpu": 7,
                "launch_failure_count": 1,
                "initialization_failure_count": 42,
                "completed_before_fault_count": 7,
            },
            "primary_jobs": primary_records,
            "conditional_null_jobs": null_records,
        }
    )
    return payload


def create_plan(
    *,
    database_path: Path,
    materialization_path: Path,
    recovery_contract_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    with _campaign_lock(plan_path.parent):
        plan = _assemble_plan(
            registry=Registry(database_path),
            materialization_path=materialization_path,
            recovery_contract_path=recovery_contract_path,
        )
        load_recovery_plan_from_mapping(plan)
        _write_immutable(plan_path, plan)
        return plan


def load_recovery_plan_from_mapping(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a plan mapping without filesystem I/O."""

    return _validate_recovery_plan_payload(plan)


def _root_by_id(registry: Registry, root_job_id: str) -> dict[str, Any]:
    root = registry.get_job(root_job_id)
    if root is None or root.get("retry_of") is not None:
        raise AdjacencyRecoveryError(f"missing primary root job {root_job_id}")
    return root


def _validate_retry_row(
    row: Mapping[str, Any], *, planned: Mapping[str, Any], root: Mapping[str, Any]
) -> None:
    if (
        row.get("job_id") != planned.get("retry_job_id")
        or row.get("campaign_id") != CAMPAIGN_ID
        or row.get("canonical_config") != root.get("canonical_config")
        or row.get("command") != root.get("command")
        or row.get("experiment_config_reference")
        != root.get("experiment_config_reference")
        or row.get("attempt_count") != 2
        or row.get("maximum_attempts") != 2
        or row.get("retry_of") != root.get("job_id")
        or str(row.get("requested_gpu")) != str(planned.get("worker_slot"))
        or int(row.get("priority", -1)) != int(root.get("priority", -2))
    ):
        raise AdjacencyRecoveryError(
            "existing retry differs from its root and signed recovery plan"
        )


def _existing_retries(registry: Registry) -> dict[str, dict[str, Any]]:
    with registry.connect() as connection:
        identifiers = [
            str(row["job_id"])
            for row in connection.execute(
                """
                SELECT job_id FROM queue_jobs
                WHERE campaign_id = ? AND retry_of IS NOT NULL
                ORDER BY created_at, job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
    result: dict[str, dict[str, Any]] = {}
    for identifier in identifiers:
        row = registry.get_job(identifier)
        if row is None:
            raise AdjacencyRecoveryError("retry queue row disappeared")
        root_id = str(row.get("retry_of"))
        if root_id in result:
            raise AdjacencyRecoveryError("one root has multiple retry rows")
        result[root_id] = row
    return result


def _verify_plan_against_state(
    *,
    plan: Mapping[str, Any],
    registry: Registry,
    materialization_path: Path,
    recovery_contract_path: Path,
) -> None:
    current = _assemble_plan(
        registry=registry,
        materialization_path=materialization_path,
        recovery_contract_path=recovery_contract_path,
    )
    if current != plan:
        raise AdjacencyRecoveryError(
            "registry, bundles, or materialization changed after plan signing"
        )


def enqueue_recovery(
    *,
    database_path: Path,
    materialization_path: Path,
    recovery_contract_path: Path,
    plan_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    with _campaign_lock(plan_path.parent):
        plan = load_recovery_plan(plan_path)
        registry = Registry(database_path)
        _verify_plan_against_state(
            plan=plan,
            registry=registry,
            materialization_path=materialization_path,
            recovery_contract_path=recovery_contract_path,
        )
        expected_by_root = {
            str(job["root_job_id"]): job for job in plan["primary_jobs"]
        }
        existing = _existing_retries(registry)
        if set(existing).difference(expected_by_root):
            raise AdjacencyRecoveryError(
                "campaign has an unauthorized retry outside the recovery plan"
            )
        roots: dict[str, dict[str, Any]] = {}
        for root_id, planned in expected_by_root.items():
            root = _root_by_id(registry, root_id)
            roots[root_id] = root
            if root_id in existing:
                _validate_retry_row(existing[root_id], planned=planned, root=root)

        if receipt_path.exists():
            receipt = load_recovery_enqueue(receipt_path, plan=plan)
            if set(existing) != set(expected_by_root):
                raise AdjacencyRecoveryError(
                    "enqueue receipt exists but retry queue inventory is incomplete"
                )
            return receipt

        for root_id, planned in expected_by_root.items():
            if root_id in existing:
                if existing[root_id].get("status") != "queued":
                    raise AdjacencyRecoveryError(
                        "partial pre-receipt retry is no longer queued"
                    )
                continue
            root = roots[root_id]
            row = registry.enqueue(
                campaign_id=CAMPAIGN_ID,
                configuration=_mapping(root["canonical_config"], "root config"),
                command=list(root["command"]),
                experiment_config_reference=str(
                    root["experiment_config_reference"]
                ),
                priority=int(root["priority"]),
                maximum_attempts=2,
                requested_gpu=str(planned["worker_slot"]),
                job_id=str(planned["retry_job_id"]),
                attempt_count=2,
                retry_of=root_id,
            )
            _validate_retry_row(row, planned=planned, root=root)
            existing[root_id] = row

        retry_records = []
        for planned in plan["primary_jobs"]:
            root_id = str(planned["root_job_id"])
            row = existing[root_id]
            if row.get("status") != "queued":
                raise AdjacencyRecoveryError(
                    "new recovery retry was claimed before receipt finalization"
                )
            retry_records.append(
                {
                    "fold": planned["fold"],
                    "seed": planned["seed"],
                    "condition": planned["condition"],
                    "root_job_id": root_id,
                    "retry_job_id": planned["retry_job_id"],
                    "attempt_count": 2,
                    "maximum_attempts": 2,
                    "retry_of": root_id,
                    "worker_slot": planned["worker_slot"],
                    "canonical_config_sha256": planned[
                        "canonical_config_sha256"
                    ],
                    "command_sha256": planned["command_sha256"],
                    "config_reference": planned["config_reference"],
                    "queue_status_at_receipt": "queued",
                }
            )
        receipt = _signed(
            {
                "schema_version": 1,
                "receipt_kind": ENQUEUE_KIND,
                "campaign_id": CAMPAIGN_ID,
                "recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
                "plan_reference": (
                    Path("scratch/locked_campaigns")
                    / CAMPAIGN_ID
                    / PLAN_FILENAME
                ).as_posix(),
                "plan_checksum": plan["checksum"],
                "complete": True,
                "inventory": {
                    "primary_retries": 50,
                    "attempt": 2,
                    "maximum_attempts": 2,
                },
                "primary_retries": retry_records,
            }
        )
        load_recovery_enqueue_from_mapping(receipt, plan=plan)
        _write_immutable(receipt_path, receipt)
        return receipt


def load_recovery_enqueue_from_mapping(
    receipt: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Strictly validate an enqueue mapping without filesystem I/O."""

    return _validate_recovery_enqueue_payload(receipt, plan=plan)


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "enqueue"))
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=paths.project_root / MATERIALIZATION_REFERENCE,
    )
    parser.add_argument(
        "--recovery-contract",
        type=Path,
        default=paths.project_root / RECOVERY_CONTRACT_REFERENCE,
    )
    parser.add_argument("--plan", type=Path, default=locked / PLAN_FILENAME)
    parser.add_argument(
        "--receipt", type=Path, default=locked / ENQUEUE_FILENAME
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "plan":
        result = create_plan(
            database_path=args.database.resolve(),
            materialization_path=args.materialization.resolve(),
            recovery_contract_path=args.recovery_contract.resolve(),
            plan_path=args.plan.resolve(),
        )
        receipt_path = args.plan.resolve()
        count = len(result["primary_jobs"])
    else:
        result = enqueue_recovery(
            database_path=args.database.resolve(),
            materialization_path=args.materialization.resolve(),
            recovery_contract_path=args.recovery_contract.resolve(),
            plan_path=args.plan.resolve(),
            receipt_path=args.receipt.resolve(),
        )
        receipt_path = args.receipt.resolve()
        count = len(result["primary_retries"])
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "stage": args.stage,
                "count": count,
                "checksum": result["checksum"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

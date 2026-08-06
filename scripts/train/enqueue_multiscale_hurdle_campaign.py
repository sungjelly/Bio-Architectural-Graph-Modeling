#!/usr/bin/env python3
"""Safely enqueue one locked stage of the multiscale hurdle campaign.

The materialization receipt is authoritative.  Every configuration and all
existing campaign jobs are preflighted before the first registry mutation.
Science stages require checksum-bound gate receipts whose run bundles contain
the recorded successful completion markers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.identifiers import (  # noqa: E402
    canonical_sha256,
    scientific_id,
)
from spatial_benchmark.local_source_permutation import (  # noqa: E402
    LocalSourcePermutationError,
    verify_local_source_permutation_receipt,
)
from spatial_benchmark.multiscale_hurdle_contract import (  # noqa: E402
    ACTIVE_CONTRACT_AMENDMENT_RELATIVE,
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    ARMS,
    CAMPAIGN_ID,
    FROZEN_CONTRACT_SHA256,
    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM,
    REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
    ROUTING,
    SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
    SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


MATERIALIZATION_KIND = (
    "multiscale_hurdle_locked_config_materialization_v1"
)
RESOURCE_GATE_KIND = "multiscale_hurdle_resource_gate_v1"
REPRESENTATION_GATE_KIND = (
    "multiscale_hurdle_representation_gate_v1"
)
CONTRACT_AMENDMENT_RELATIVE = ACTIVE_CONTRACT_AMENDMENT_RELATIVE
CONTRACT_AMENDMENT_SHA256 = ACTIVE_CONTRACT_AMENDMENT_SHA256
STAGE1_ALIASES = ("ANC-03", "ANC-05")
STAGE2_ALIASES = ("ANC-02", "ANC-03", "ANC-05", "ANC-06", "ANC-09")
SAFE_GPU_IDS = frozenset({0, 1, 2, 3, 5, 6, 7})
MAXIMUM_ATTEMPTS = 2
DISK_USED_DECIMAL_GB_HARD_STOP = 55.0
EXPECTED_PARAMETER_COUNT = 7_559_184
EXPECTED_RESOURCE_LIMITS = {
    "preferred_aggregate_gpu_hours": 12.0,
    "absolute_aggregate_gpu_hours": 24.0,
    "stage1_peak_allocated_vram_gib": 12.0,
    "per_device_peak_allocated_vram_gib": 20.5,
    "aggregate_observed_process_vram_gib": 50.0,
    "filesystem_used_decimal_gb_hard_stop": 55.0,
}
RESOURCE_THRESHOLDS = {
    "stage1_peak_allocated_vram_gib_maximum": 12.0,
    "fp32_amp_absolute_loss_discrepancy_maximum": 1e-3,
    "stage1_projected_gpu_hours_per_200_epoch_core_maximum": 0.5,
    "filesystem_used_decimal_gb_hard_stop": 55.0,
}
REPRESENTATION_THRESHOLDS = {
    "minimum_relative_improvement": 0.02,
    "required_core_count": 2,
    "required_core_total": 2,
}
_ROUTING = ROUTING
_STAGE_PRIORITY = {"resource": 60, "stage1": 50, "stage2": 40}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(
    r"^r_[0-9]{8}T[0-9]{6}Z_[a-z0-9]{8}_s[0-9]{3}_"
    r"f[0-9]{2}_a[0-9]{2}_[a-z0-9_-]{4,32}$"
)
_PROHIBITED_IDENTIFIER_KEYS = frozenset(
    {
        "cell_id",
        "core_id",
        "core_label",
        "donor_id",
        "fov",
        "patient_id",
        "slide",
    }
)


class MultiscaleHurdleEnqueueError(RuntimeError):
    """Raised when a staged enqueue invariant is incomplete or changed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiscaleHurdleEnqueueError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise MultiscaleHurdleEnqueueError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MultiscaleHurdleEnqueueError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except MultiscaleHurdleEnqueueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiscaleHurdleEnqueueError(
            f"{label} is not strict JSON"
        ) from exc
    return dict(_mapping(value, label))


def _verify_checksum(payload: Mapping[str, Any], *, label: str) -> str:
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or _HEX_64.fullmatch(checksum) is None:
        raise MultiscaleHurdleEnqueueError(f"{label} checksum is malformed")
    canonical = dict(payload)
    canonical.pop("checksum", None)
    if canonical_sha256(canonical) != checksum:
        raise MultiscaleHurdleEnqueueError(
            f"{label} checksum does not verify"
        )
    return checksum


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_direct_identifiers(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _PROHIBITED_IDENTIFIER_KEYS:
                raise MultiscaleHurdleEnqueueError(
                    f"{label} contains prohibited identifier field {key!r}"
                )
            _assert_no_direct_identifiers(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_no_direct_identifiers(child, label=label)


def _resolve_project_reference(
    value: Any,
    *,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value.strip():
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be a nonempty project-relative path"
        )
    reference = Path(value)
    if reference.is_absolute():
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be project-relative"
        )
    unresolved = _PROJECT_ROOT / reference
    if unresolved.is_symlink():
        raise MultiscaleHurdleEnqueueError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise MultiscaleHurdleEnqueueError(
            f"{label} escapes the project root"
        ) from exc
    if require_file and not resolved.is_file():
        raise MultiscaleHurdleEnqueueError(
            f"{label} is not an available file"
        )
    if require_directory and not resolved.is_dir():
        raise MultiscaleHurdleEnqueueError(
            f"{label} is not an available directory"
        )
    return reference, resolved


def _hex_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _gpu(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be a safe integer GPU"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be a safe integer GPU"
        ) from exc
    if result not in SAFE_GPU_IDS:
        raise MultiscaleHurdleEnqueueError(
            f"{label} selects unsafe GPU {result}; GPU 4 is excluded"
        )
    return result


def _materialized_jobs(
    materialization: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    result: list[list[Mapping[str, Any]]] = []
    for field, expected in (
        (
            "pilot_jobs",
            {(alias, "self") for alias in STAGE1_ALIASES},
        ),
        (
            "science_jobs",
            {(alias, arm) for alias in STAGE2_ALIASES for arm in ARMS},
        ),
    ):
        raw_jobs = materialization.get(field)
        if not isinstance(raw_jobs, list):
            raise MultiscaleHurdleEnqueueError(
                f"materialization {field} must be a list"
            )
        jobs = [
            _mapping(item, f"materialization {field} item")
            for item in raw_jobs
        ]
        observed = [
            (str(item.get("alias")), str(item.get("arm")))
            for item in jobs
        ]
        if len(jobs) != len(expected) or set(observed) != expected:
            raise MultiscaleHurdleEnqueueError(
                f"materialization {field} does not contain the exact job matrix"
            )
        for item in jobs:
            _gpu(item.get("requested_gpu"), label=f"{field} requested GPU")
            _hex_digest(
                item.get("config_sha256"),
                label=f"{field} canonical config checksum",
            )
            _hex_digest(
                item.get("file_sha256"),
                label=f"{field} config file checksum",
            )
            _hex_digest(
                item.get("graph_bundle_sha256"),
                label=f"{field} graph bundle checksum",
            )
            _hex_digest(
                item.get("local_source_permutation_sha256"),
                label=f"{field} source permutation checksum",
            )
        result.append(jobs)
    return result[0], result[1]


def _load_materialization(path: Path) -> dict[str, Any]:
    payload = _strict_json(path, label="locked materialization")
    _verify_checksum(payload, label="locked materialization")
    _assert_no_direct_identifiers(payload, label="locked materialization")
    counts = _mapping(payload.get("counts"), "materialization counts")
    parameter_audit = _mapping(
        payload.get("parameter_audit"), "materialization parameter audit"
    )
    limits = _mapping(
        payload.get("resource_limits"), "materialization resource limits"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("receipt_kind") != MATERIALIZATION_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or dict(counts)
        != {"cores": 5, "pilot_configs": 2, "science_configs": 20}
        or payload.get("registry_mutation_performed") is not False
        or payload.get("queue_mutation_performed") is not False
        or payload.get("training_performed") is not False
        or parameter_audit.get("trainable_parameter_count")
        != EXPECTED_PARAMETER_COUNT
        or parameter_audit.get("matched_arms") != list(ARMS)
        or dict(limits) != EXPECTED_RESOURCE_LIMITS
    ):
        raise MultiscaleHurdleEnqueueError(
            "locked materialization does not describe the frozen campaign"
        )
    _hex_digest(
        parameter_audit.get("named_parameter_shapes_sha256"),
        label="parameter-shape checksum",
    )
    allowed = payload.get("allowed_gpu_ids")
    if (
        not isinstance(allowed, list)
        or any(isinstance(value, bool) for value in allowed)
        or set(allowed) != SAFE_GPU_IDS
        or len(allowed) != len(SAFE_GPU_IDS)
    ):
        raise MultiscaleHurdleEnqueueError(
            "materialization safe GPU set changed or includes GPU 4"
        )
    frozen = _mapping(
        payload.get("frozen_contract"), "materialization frozen contract"
    )
    _, contract_path = _resolve_project_reference(
        frozen.get("reference"),
        label="frozen task contract",
        require_file=True,
    )
    if (
        frozen.get("sha256") != FROZEN_CONTRACT_SHA256
        or _sha256_file(contract_path) != FROZEN_CONTRACT_SHA256
    ):
        raise MultiscaleHurdleEnqueueError(
            "frozen task contract checksum changed"
        )
    amendment = _mapping(
        payload.get("contract_amendment"),
        "materialization contract amendment",
    )
    _, amendment_path = _resolve_project_reference(
        amendment.get("reference"),
        label="contract amendment",
        require_file=True,
    )
    if (
        amendment.get("reference")
        != CONTRACT_AMENDMENT_RELATIVE.as_posix()
        or amendment.get("sha256") != CONTRACT_AMENDMENT_SHA256
        or _sha256_file(amendment_path) != CONTRACT_AMENDMENT_SHA256
    ):
        raise MultiscaleHurdleEnqueueError(
            "sender-state permutation amendment checksum changed"
        )
    superseded = _mapping(
        amendment.get("supersedes"),
        "superseded contract amendment",
    )
    _, superseded_path = _resolve_project_reference(
        superseded.get("reference"),
        label="superseded contract amendment",
        require_file=True,
    )
    if (
        superseded.get("reference")
        != SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
        or superseded.get("sha256")
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
        or superseded.get("retained_as_negative_design_record") is not True
        or _sha256_file(superseded_path)
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
    ):
        raise MultiscaleHurdleEnqueueError(
            "superseded rewiring amendment is not retained exactly"
        )
    supplement = _mapping(
        amendment.get("required_supplement"),
        "required contract supplement",
    )
    _, supplement_path = _resolve_project_reference(
        supplement.get("reference"),
        label="required contract supplement",
        require_file=True,
    )
    if (
        supplement.get("reference")
        != REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE.as_posix()
        or supplement.get("sha256")
        != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        or supplement.get(
            "mask_noninterference_gate_required_before_gpu_training"
        )
        is not True
        or _sha256_file(supplement_path)
        != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
    ):
        raise MultiscaleHurdleEnqueueError(
            "required amendment003 mask-safety supplement changed"
        )
    graph_contract = _mapping(
        payload.get("fixed_graph_contract"),
        "fixed graph contract",
    )
    permutation_contract = _mapping(
        graph_contract.get("local_source_permutation"),
        "fixed local source permutation contract",
    )
    if (
        graph_contract.get("original_rewired_arm_authorized") is not False
        or permutation_contract.get(
            "minimum_node_mapping_changed_fraction"
        )
        != LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
        or permutation_contract.get("displacement_threshold_um")
        != LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
        or permutation_contract.get(
            "minimum_node_displacement_above_threshold_fraction"
        )
        != LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
        or permutation_contract.get(
            "minimum_local_edge_slot_sender_identity_changed_fraction"
        )
        != LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
        or permutation_contract.get("uses_random_seed") is not False
    ):
        raise MultiscaleHurdleEnqueueError(
            "materialization lacks the frozen sender-permutation gates"
        )
    cores = payload.get("cores")
    if not isinstance(cores, list) or len(cores) != len(STAGE2_ALIASES):
        raise MultiscaleHurdleEnqueueError(
            "materialization must contain exactly five core records"
        )
    aliases: list[str] = []
    for raw_core in cores:
        core = _mapping(raw_core, "materialization core")
        alias = str(core.get("alias"))
        aliases.append(alias)
        graph_receipt = _mapping(
            core.get("graph_receipt"), f"{alias} graph receipt"
        )
        if core.get("graph_receipt_sha256") != canonical_sha256(
            graph_receipt
        ):
            raise MultiscaleHurdleEnqueueError(
                f"{alias} graph receipt checksum changed"
            )
        permutation = _mapping(
            graph_receipt.get("local_source_permutation"),
            f"{alias} local source permutation",
        )
        try:
            verify_local_source_permutation_receipt(permutation)
        except LocalSourcePermutationError as exc:
            raise MultiscaleHurdleEnqueueError(
                f"{alias} sender-state permutation receipt is invalid"
            ) from exc
    if len(aliases) != len(set(aliases)) or set(aliases) != set(
        STAGE2_ALIASES
    ):
        raise MultiscaleHurdleEnqueueError(
            "materialization core aliases changed or are duplicated"
        )
    _materialized_jobs(payload)
    for field in (
        "resource_gate_receipt_reference",
        "representation_gate_receipt_reference",
    ):
        _resolve_project_reference(payload.get(field), label=field)
    return payload


def _job_maps(
    materialization: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    pilot, science = _materialized_jobs(materialization)
    return (
        {(str(job["alias"]), str(job["arm"])): job for job in pilot},
        {(str(job["alias"]), str(job["arm"])): job for job in science},
    )


def _stage_keys(stage: str) -> tuple[tuple[str, str], ...]:
    if stage == "resource":
        return tuple((alias, "self") for alias in STAGE1_ALIASES)
    if stage == "stage1":
        return tuple((alias, "self") for alias in STAGE1_ALIASES)
    if stage == "stage2":
        return tuple(
            (alias, arm)
            for alias in STAGE2_ALIASES
            for arm in ARMS
            if not (arm == "self" and alias in STAGE1_ALIASES)
        )
    raise MultiscaleHurdleEnqueueError(
        "stage must be resource, stage1, or stage2"
    )


def _jobs_for_stage(
    materialization: Mapping[str, Any],
    stage: str,
) -> list[Mapping[str, Any]]:
    pilot, science = _job_maps(materialization)
    source = pilot if stage == "resource" else science
    return [source[key] for key in _stage_keys(stage)]


def _stage_for_materialized_job(*, pilot: bool, alias: str, arm: str) -> str:
    if pilot:
        return "resource"
    if arm == "self" and alias in STAGE1_ALIASES:
        return "stage1"
    return "stage2"


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise MultiscaleHurdleEnqueueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MultiscaleHurdleEnqueueError(
            f"{label} must be finite"
        ) from exc
    if not math.isfinite(result):
        raise MultiscaleHurdleEnqueueError(f"{label} must be finite")
    return result


def _validate_gate_bundle(
    job: Mapping[str, Any],
    *,
    label: str,
) -> None:
    run_id = job.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise MultiscaleHurdleEnqueueError(
            f"{label} has an invalid immutable run ID"
        )
    _, bundle = _resolve_project_reference(
        job.get("bundle_reference"),
        label=f"{label} run bundle",
        require_directory=True,
    )
    if bundle.name != run_id:
        raise MultiscaleHurdleEnqueueError(
            f"{label} run ID differs from its immutable bundle"
        )
    marker = bundle / "_SUCCESS"
    if marker.is_symlink() or not marker.is_file():
        raise MultiscaleHurdleEnqueueError(
            f"{label} lacks an immutable _SUCCESS marker"
        )
    success = _strict_json(marker, label=f"{label} _SUCCESS marker")
    expected = _hex_digest(
        job.get("success_marker_content_sha256"),
        label=f"{label} completion-marker checksum",
    )
    if (
        success.get("run_id") != run_id
        or success.get("status") != "success"
        or success.get("content_sha256") != expected
    ):
        raise MultiscaleHurdleEnqueueError(
            f"{label} completion-marker identity or checksum changed"
        )
    if job.get("verified_bundle") is not True:
        raise MultiscaleHurdleEnqueueError(
            f"{label} run bundle is not verified"
        )


def _validate_gate_job_matrix(
    jobs: Any,
    *,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    label: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(jobs, list):
        raise MultiscaleHurdleEnqueueError(f"{label} jobs must be a list")
    mapped = [_mapping(job, f"{label} job") for job in jobs]
    observed = [
        (str(job.get("alias")), str(job.get("arm"))) for job in mapped
    ]
    if len(mapped) != len(expected) or set(observed) != set(expected):
        raise MultiscaleHurdleEnqueueError(
            f"{label} does not bind the exact prerequisite job matrix"
        )
    run_ids: set[str] = set()
    for job in mapped:
        key = (str(job.get("alias")), str(job.get("arm")))
        if job.get("config_sha256") != expected[key].get("config_sha256"):
            raise MultiscaleHurdleEnqueueError(
                f"{label} config checksum differs from materialization"
            )
        _validate_gate_bundle(job, label=f"{label} {key[0]} {key[1]}")
        run_id = str(job["run_id"])
        if run_id in run_ids:
            raise MultiscaleHurdleEnqueueError(
                f"{label} repeats an immutable run ID"
            )
        run_ids.add(run_id)
    return mapped


def _load_resource_gate(
    path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _strict_json(path, label="resource gate receipt")
    _verify_checksum(gate, label="resource gate receipt")
    _assert_no_direct_identifiers(gate, label="resource gate receipt")
    frozen = _mapping(
        materialization.get("frozen_contract"), "frozen contract"
    )
    pilot, _ = _job_maps(materialization)
    expected = {
        key: pilot[key] for key in (("ANC-03", "self"), ("ANC-05", "self"))
    }
    jobs = _validate_gate_job_matrix(
        gate.get("jobs"),
        expected=expected,
        label="resource gate",
    )
    if (
        gate.get("schema_version") != 1
        or gate.get("receipt_kind") != RESOURCE_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("stage") != "resource"
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("frozen_contract_sha256") != frozen.get("sha256")
        or gate.get("thresholds") != RESOURCE_THRESHOLDS
        or gate.get("complete") is not True
        or gate.get("gate_passed") is not True
        or gate.get("production_authorized") is not True
        or gate.get("failure_reasons") != []
    ):
        raise MultiscaleHurdleEnqueueError(
            "stage1 requires the exact passing resource gate"
        )
    for job in jobs:
        discrepancy = _finite_number(
            job.get("fp32_amp_absolute_loss_discrepancy"),
            label="resource precision discrepancy",
        )
        peak = _finite_number(
            job.get("peak_allocated_vram_gib"),
            label="resource peak VRAM",
        )
        projected = _finite_number(
            job.get("projected_gpu_hours_per_200_epochs"),
            label="resource projected runtime",
        )
        disk_used = _finite_number(
            job.get("filesystem_used_decimal_gb"),
            label="resource filesystem use",
        )
        if (
            job.get("finite_losses_and_gradients") is not True
            or job.get("parameter_match") is not True
            or job.get("precision_equivalence_passed") is not True
            or job.get("peak_vram_passed") is not True
            or job.get("projected_runtime_passed") is not True
            or job.get("disk_safety_passed") is not True
            or job.get("runner_pilot_gate_passed") is not True
            or job.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or discrepancy < 0.0
            or discrepancy
            > RESOURCE_THRESHOLDS[
                "fp32_amp_absolute_loss_discrepancy_maximum"
            ]
            or peak
            > RESOURCE_THRESHOLDS[
                "stage1_peak_allocated_vram_gib_maximum"
            ]
            or peak < 0.0
            or projected
            > RESOURCE_THRESHOLDS[
                "stage1_projected_gpu_hours_per_200_epoch_core_maximum"
            ]
            or projected < 0.0
            or disk_used < 0.0
            or disk_used
            >= RESOURCE_THRESHOLDS[
                "filesystem_used_decimal_gb_hard_stop"
            ]
        ):
            raise MultiscaleHurdleEnqueueError(
                "resource gate contains failing or inconsistent evidence"
            )
    return gate


def _load_representation_gate(
    path: Path,
    *,
    materialization: Mapping[str, Any],
    resource_gate: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _strict_json(path, label="representation gate receipt")
    _verify_checksum(gate, label="representation gate receipt")
    _assert_no_direct_identifiers(gate, label="representation gate receipt")
    frozen = _mapping(
        materialization.get("frozen_contract"), "frozen contract"
    )
    _, science = _job_maps(materialization)
    expected = {
        key: science[key]
        for key in (("ANC-03", "self"), ("ANC-05", "self"))
    }
    jobs = _validate_gate_job_matrix(
        gate.get("jobs"),
        expected=expected,
        label="representation gate",
    )
    if (
        gate.get("schema_version") != 1
        or gate.get("receipt_kind") != REPRESENTATION_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("stage") != "stage1"
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("frozen_contract_sha256") != frozen.get("sha256")
        or gate.get("resource_gate_checksum")
        != resource_gate.get("checksum")
        or gate.get("thresholds") != REPRESENTATION_THRESHOLDS
        or gate.get("complete") is not True
        or gate.get("gate_passed") is not True
        or gate.get("both_h1_gates_passed") is not True
        or gate.get("stage2_authorized") is not True
        or gate.get("failure_reasons") != []
    ):
        raise MultiscaleHurdleEnqueueError(
            "stage2 requires the exact passing representation gate"
        )
    for job in jobs:
        detection = _finite_number(
            job.get("detection_balanced_accuracy"),
            label="representation detection balanced accuracy",
        )
        prevalence = _finite_number(
            job.get("prevalence_reference_balanced_accuracy"),
            label="representation prevalence-reference balanced accuracy",
        )
        huber = _finite_number(
            job.get("positive_continuous_huber_relative_improvement"),
            label="representation Huber improvement",
        )
        state = _finite_number(
            job.get("positive_count_state_mae_relative_improvement"),
            label="representation count-state improvement",
        )
        if (
            job.get("h1_gate_passed") is not True
            or job.get(
                "detection_balanced_accuracy_above_prevalence_reference"
            )
            is not True
            or not 0.0 <= detection <= 1.0
            or not 0.0 <= prevalence <= 1.0
            or detection <= prevalence
            or huber
            < REPRESENTATION_THRESHOLDS["minimum_relative_improvement"]
            or state
            < REPRESENTATION_THRESHOLDS["minimum_relative_improvement"]
        ):
            raise MultiscaleHurdleEnqueueError(
                "representation gate requires both cores to pass every H1 rule"
            )
    return gate


def _materialization_gate_path(
    materialization: Mapping[str, Any],
    *,
    field: str,
    supplied: Path | None,
) -> Path:
    _, expected = _resolve_project_reference(
        materialization.get(field),
        label=field,
        require_file=True,
    )
    if supplied is not None and supplied.resolve() != expected:
        raise MultiscaleHurdleEnqueueError(
            f"{field} differs from the frozen materialization reference"
        )
    return expected


def _core_map(
    materialization: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(_mapping(core, "core").get("alias")): _mapping(core, "core")
        for core in materialization["cores"]
    }


def _validate_registered_dataset(
    config: Mapping[str, Any],
    *,
    registry: Registry,
) -> None:
    dataset = _mapping(config.get("dataset"), "config dataset")
    registered = registry.get_dataset(
        str(dataset.get("dataset_id")), str(dataset.get("version"))
    )
    split = registry.get_split(str(dataset.get("split_id")))
    if registered is None or split is None:
        raise MultiscaleHurdleEnqueueError(
            "config dataset or split is not registered"
        )
    if (
        registered.get("processed_fingerprint")
        != dataset.get("dataset_fingerprint")
        or registered.get("preprocessing_version")
        != dataset.get("preprocessing_version")
        or split.get("dataset_id") != dataset.get("dataset_id")
        or split.get("dataset_version") != dataset.get("version")
        or split.get("fingerprint") != dataset.get("split_fingerprint")
    ):
        raise MultiscaleHurdleEnqueueError(
            "registered dataset or split identity differs from config"
        )
    _, prepared = _resolve_project_reference(
        dataset.get("prepared_artifact_reference"),
        label="prepared artifact",
        require_directory=True,
    )
    protected = registered.get("protected_source_path")
    if protected is not None:
        protected_path = Path(str(protected))
        registered_path = (
            protected_path.resolve()
            if protected_path.is_absolute()
            else (_PROJECT_ROOT / protected_path).resolve()
        )
        if registered_path != prepared:
            raise MultiscaleHurdleEnqueueError(
                "registered protected path differs from prepared artifact"
            )


def _validate_job(
    raw: Mapping[str, Any],
    *,
    pilot: bool,
    materialization: Mapping[str, Any],
    materialization_path: Path,
    registry: Registry,
) -> tuple[dict[str, Any], Path, str, int, str, str, str]:
    alias = str(raw.get("alias"))
    arm = str(raw.get("arm"))
    gpu = _gpu(raw.get("requested_gpu"), label="planned requested GPU")
    reference, config_path = _resolve_project_reference(
        raw.get("config"), label="locked config", require_file=True
    )
    if _sha256_file(config_path) != raw.get("file_sha256"):
        raise MultiscaleHurdleEnqueueError(
            "locked config file checksum changed"
        )
    config = dict(load_yaml_mapping(config_path))
    validate_experiment_config(config)
    _assert_no_direct_identifiers(config, label="locked config")
    digest = canonical_sha256(config)
    if digest != raw.get("config_sha256"):
        raise MultiscaleHurdleEnqueueError(
            "locked canonical config checksum changed"
        )

    campaign = _mapping(config.get("campaign"), "config campaign")
    experiment = _mapping(config.get("experiment"), "config experiment")
    metadata = _mapping(config.get("metadata"), "config metadata")
    model = _mapping(config.get("model"), "config model")
    graph = _mapping(config.get("graph"), "config graph")
    trainer = _mapping(config.get("trainer"), "config trainer")
    evaluation = _mapping(config.get("evaluation"), "config evaluation")
    launcher = _mapping(config.get("launcher"), "config launcher")
    dataset = _mapping(config.get("dataset"), "config dataset")
    frozen = _mapping(
        materialization.get("frozen_contract"), "frozen contract"
    )
    core = _core_map(materialization).get(alias)
    if core is None:
        raise MultiscaleHurdleEnqueueError("config alias is not materialized")
    expected_regional, expected_local = _ROUTING.get(arm, ("", ""))
    graph_receipt = _mapping(
        core.get("graph_receipt"), f"{alias} graph receipt"
    )
    bundle_checksums = _mapping(
        graph_receipt.get("bundle_checksums"),
        f"{alias} graph bundle checksums",
    )
    receipt_permutation = _mapping(
        graph_receipt.get("local_source_permutation"),
        f"{alias} source permutation receipt",
    )
    config_permutation = _mapping(
        graph.get("local_source_permutation"),
        f"{alias} config source permutation",
    )
    try:
        verify_local_source_permutation_receipt(receipt_permutation)
        verify_local_source_permutation_receipt(config_permutation)
    except LocalSourcePermutationError as exc:
        raise MultiscaleHurdleEnqueueError(
            f"{alias} source permutation receipt is invalid"
        ) from exc
    expected_replicates = 1 if pilot else 3
    expected_epochs = 2 if pilot else 200
    role = "resource_pilot" if pilot else "science"
    _, referenced_materialization = _resolve_project_reference(
        metadata.get("locked_config_materialization_receipt"),
        label="config materialization receipt",
        require_file=True,
    )
    _, referenced_contract = _resolve_project_reference(
        campaign.get("frozen_contract"),
        label="config frozen contract",
        require_file=True,
    )
    _, materialized_contract = _resolve_project_reference(
        frozen.get("reference"),
        label="materialized frozen contract",
        require_file=True,
    )
    materialized_amendment = _mapping(
        materialization.get("contract_amendment"),
        "materialized contract amendment",
    )
    materialized_supplement = _mapping(
        materialized_amendment.get("required_supplement"),
        "materialized required contract supplement",
    )
    _, referenced_amendment = _resolve_project_reference(
        campaign.get("contract_amendment"),
        label="config contract amendment",
        require_file=True,
    )
    _, expected_amendment = _resolve_project_reference(
        materialized_amendment.get("reference"),
        label="materialized contract amendment",
        require_file=True,
    )
    _, referenced_supplement = _resolve_project_reference(
        campaign.get("contract_supplement"),
        label="config contract supplement",
        require_file=True,
    )
    _, expected_supplement = _resolve_project_reference(
        materialized_supplement.get("reference"),
        label="materialized required contract supplement",
        require_file=True,
    )
    if (
        referenced_materialization != materialization_path.resolve()
        or referenced_contract != materialized_contract
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("exploratory") is not True
        or campaign.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
        or referenced_amendment != expected_amendment
        or campaign.get("contract_amendment_sha256")
        != CONTRACT_AMENDMENT_SHA256
        or materialized_amendment.get("sha256")
        != CONTRACT_AMENDMENT_SHA256
        or referenced_supplement != expected_supplement
        or campaign.get("contract_supplement_sha256")
        != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        or materialized_supplement.get("sha256")
        != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        or materialized_supplement.get(
            "mask_noninterference_gate_required_before_gpu_training"
        )
        is not True
        or campaign.get("superseded_contract_amendment")
        != SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
        or campaign.get("superseded_contract_amendment_sha256")
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
        or campaign.get(
            "superseded_amendment_retained_as_negative_record"
        )
        is not True
        or experiment.get("arm") != arm
        or experiment.get("biological_unit_alias") != alias
        or bool(experiment.get("resource_pilot")) is not pilot
        or experiment.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or metadata.get("execution_role") != role
        or metadata.get("sender_state_permutation_amendment_enforced")
        is not True
        or metadata.get("mask_noninterference_supplement_enforced")
        is not True
        or metadata.get("original_rewired_arm_authorized") is not False
        or model.get("name") != "multiscale-hurdle-count"
        or model.get("family") != "additive_multiscale_hurdle_count"
        or model.get("parameter_match_group") != "multiscale_hurdle_v1"
        or model.get("count_representation_schema")
        != "hurdle_detection_plus_positive_standardized_log1p_v1"
        or int(model.get("output_channels_per_gene", -1)) != 2
        or model.get("regional_routing") != expected_regional
        or model.get("local_routing") != expected_local
        or model.get("exact_additive_decomposition") is not True
        or model.get("regional_output_zero_initialized") is not True
        or model.get("local_output_zero_initialized") is not True
        or graph.get("expected_materialized_graph_sha256")
        != bundle_checksums.get("bundle_sha256")
        or graph.get("expected_graph_receipt_sha256")
        != core.get("graph_receipt_sha256")
        or dict(config_permutation) != dict(receipt_permutation)
        or raw.get("local_source_permutation_sha256")
        != receipt_permutation.get("checksum")
        or "rewired_local" in graph
        or raw.get("graph_bundle_sha256")
        != bundle_checksums.get("bundle_sha256")
        or int(raw.get("n_nodes", -1)) != int(core.get("n_nodes", -2))
        or int(graph.get("neighbor_k", -1)) != 64
        or int(graph.get("k", -1)) != 64
        or trainer.get("optimizer") != "AdamW"
        or int(trainer.get("max_epochs", -1)) != expected_epochs
        or trainer.get("fixed_epoch_budget") is not True
        or trainer.get("restore_best") is not False
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("checkpoint_policy") != "last_only"
        or trainer.get("neighbor_sampling") is not False
        or evaluation.get("task_family")
        != "masked_expression_hurdle_count"
        or evaluation.get("protocol") != "held_in_full_core_fixed_budget"
        or evaluation.get("primary_metric") != "fit/whole_node/hurdle_loss"
        or int(evaluation.get("mask_replicates_per_mode", -1))
        != expected_replicates
        or launcher.get("requested_gpu") != str(gpu)
        or launcher.get("requested_gpu_count") != 1
        or launcher.get("disk_safety_max_used_decimal_gb")
        != DISK_USED_DECIMAL_GB_HARD_STOP
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or dataset.get("task") != "masked_expression_hurdle_count"
        or dataset.get("frozen_task_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or int(config.get("seed", -1)) != 0
        or int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) != 1
    ):
        raise MultiscaleHurdleEnqueueError(
            "locked config differs from the frozen campaign contract"
        )
    if not pilot:
        authorization = _mapping(
            trainer.get("amp_authorization"), "science AMP authorization"
        )
        if (
            authorization.get("mode")
            != "require_external_resource_gate_receipt"
            or authorization.get("receipt_schema") != RESOURCE_GATE_KIND
            or authorization.get("receipt_reference")
            != materialization.get("resource_gate_receipt_reference")
            or authorization.get("frozen_contract_sha256")
            != FROZEN_CONTRACT_SHA256
        ):
            raise MultiscaleHurdleEnqueueError(
                "science config does not require the frozen resource gate"
            )
    _validate_registered_dataset(config, registry=registry)
    stage = _stage_for_materialized_job(
        pilot=pilot, alias=alias, arm=arm
    )
    return config, reference, digest, gpu, arm, alias, stage


def _runner_command() -> list[str]:
    return [
        sys.executable,
        str(
            _PROJECT_ROOT
            / "scripts"
            / "train"
            / "run_multiscale_hurdle_capacity.py"
        ),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def _existing_jobs_by_digest(
    registry: Registry,
) -> dict[str, Mapping[str, Any]]:
    try:
        with registry.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM queue_jobs
                WHERE campaign_id = ? AND retry_of IS NULL
                ORDER BY created_at, job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
    except Exception as exc:
        raise MultiscaleHurdleEnqueueError(
            "cannot inspect all root jobs for the campaign"
        ) from exc
    result: dict[str, Mapping[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        try:
            config = _mapping(
                json.loads(str(row["canonical_config_json"])),
                "existing queue config",
            )
            command = json.loads(str(row["command_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MultiscaleHurdleEnqueueError(
                "existing campaign queue job contains invalid JSON"
            ) from exc
        row["canonical_config"] = config
        row["command"] = command
        digest = canonical_sha256(config)
        if digest in result:
            raise MultiscaleHurdleEnqueueError(
                "campaign contains duplicate root canonical configurations"
            )
        result[digest] = row
    return result


def _validate_existing(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    reference: Path,
    gpu: int,
    stage: str,
) -> None:
    if (
        int(row.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(row.get("attempt_count", -1)) != 1
        or row.get("retry_of") is not None
        or str(row.get("requested_gpu")) != str(gpu)
        or int(row.get("priority", -1)) != _STAGE_PRIORITY[stage]
        or str(row.get("experiment_config_reference")) != str(reference)
        or row.get("command") != _runner_command()
        or canonical_sha256(
            _mapping(row.get("canonical_config"), "existing config")
        )
        != canonical_sha256(config)
    ):
        raise MultiscaleHurdleEnqueueError(
            "existing root job has wrong command, config, GPU, stage "
            "priority, or maximum-attempt contract"
        )


def _disk_used_decimal_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.used) / 1_000_000_000.0


def _assert_disk_safe() -> float:
    used = _disk_used_decimal_gb(_PROJECT_ROOT)
    if not math.isfinite(used) or used >= DISK_USED_DECIMAL_GB_HARD_STOP:
        raise MultiscaleHurdleEnqueueError(
            "filesystem used space must remain below 55.0 decimal GB"
        )
    return used


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_no_direct_identifiers(payload, label="enqueue receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


@contextmanager
def _campaign_lock(database_path: Path) -> Iterator[None]:
    lock_path = (
        database_path.parent
        / f".{database_path.name}.{CAMPAIGN_ID}.enqueue.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _receipt_payload(
    *,
    stage: str,
    materialization: Mapping[str, Any],
    resource_gate: Mapping[str, Any] | None,
    representation_gate: Mapping[str, Any] | None,
    disk_used_decimal_gb: float,
    jobs: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": f"multiscale_hurdle_{stage}_enqueue_v1",
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "materialization_checksum": materialization["checksum"],
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "resource_gate_checksum": (
            None if resource_gate is None else resource_gate["checksum"]
        ),
        "representation_gate_checksum": (
            None
            if representation_gate is None
            else representation_gate["checksum"]
        ),
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "disk_used_decimal_gb_at_enqueue": disk_used_decimal_gb,
        "disk_used_decimal_gb_hard_stop": (
            DISK_USED_DECIMAL_GB_HARD_STOP
        ),
        "complete": len(jobs) == expected_count,
        "jobs": jobs,
    }
    payload["checksum"] = canonical_sha256(payload)
    return payload


def enqueue_stage(
    *,
    stage: str,
    materialization_path: Path,
    receipt_path: Path,
    database_path: Path,
    resource_gate_path: Path | None = None,
    representation_gate_path: Path | None = None,
) -> dict[str, Any]:
    """Preflight and idempotently enqueue exactly one frozen campaign stage."""

    _stage_keys(stage)
    with _campaign_lock(database_path):
        materialization_path = materialization_path.resolve()
        materialization = _load_materialization(materialization_path)
        resource_gate: Mapping[str, Any] | None = None
        representation_gate: Mapping[str, Any] | None = None
        if stage in {"stage1", "stage2"}:
            resource_path = _materialization_gate_path(
                materialization,
                field="resource_gate_receipt_reference",
                supplied=resource_gate_path,
            )
            resource_gate = _load_resource_gate(
                resource_path, materialization=materialization
            )
        if stage == "stage2":
            assert resource_gate is not None
            representation_path = _materialization_gate_path(
                materialization,
                field="representation_gate_receipt_reference",
                supplied=representation_gate_path,
            )
            representation_gate = _load_representation_gate(
                representation_path,
                materialization=materialization,
                resource_gate=resource_gate,
            )
        disk_used = _assert_disk_safe()
        registry = Registry(database_path)
        campaign = registry.get_campaign(CAMPAIGN_ID)
        if campaign is None or campaign.get("campaign_id") != CAMPAIGN_ID:
            raise MultiscaleHurdleEnqueueError(
                "frozen campaign is not registered"
            )
        existing = _existing_jobs_by_digest(registry)

        pilot_jobs, science_jobs = _materialized_jobs(materialization)
        all_planned_digests: set[str] = set()
        validated: list[
            tuple[dict[str, Any], Path, str, int, str, str, str]
        ] = []
        for pilot, raw_jobs in (
            (True, pilot_jobs),
            (False, science_jobs),
        ):
            for raw in raw_jobs:
                item = _validate_job(
                    raw,
                    pilot=pilot,
                    materialization=materialization,
                    materialization_path=materialization_path,
                    registry=registry,
                )
                config, reference, digest, gpu, _arm, _alias, job_stage = item
                if digest in all_planned_digests:
                    raise MultiscaleHurdleEnqueueError(
                        "two locked jobs have the same canonical config"
                    )
                all_planned_digests.add(digest)
                matched = existing.get(digest)
                if matched is not None:
                    _validate_existing(
                        matched,
                        config=config,
                        reference=reference,
                        gpu=gpu,
                        stage=job_stage,
                    )
                if job_stage == stage:
                    validated.append(item)
        if set(existing).difference(all_planned_digests):
            raise MultiscaleHurdleEnqueueError(
                "campaign contains an unexpected root queue job"
            )
        expected_keys = _stage_keys(stage)
        observed_keys = {(item[5], item[4]) for item in validated}
        if len(validated) != len(expected_keys) or observed_keys != set(
            expected_keys
        ):
            raise MultiscaleHurdleEnqueueError(
                "validated stage does not match the exact frozen job matrix"
            )
        order = {key: index for index, key in enumerate(expected_keys)}
        validated.sort(key=lambda item: order[(item[5], item[4])])

        receipt_rows: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for config, reference, digest, gpu, arm, alias, job_stage in validated:
            registry.register_variant(
                scientific_id(config),
                campaign_id=CAMPAIGN_ID,
                configuration=config,
            )
            row = existing.get(digest)
            if row is None:
                row = registry.enqueue(
                    campaign_id=CAMPAIGN_ID,
                    configuration=config,
                    command=_runner_command(),
                    experiment_config_reference=reference,
                    priority=_STAGE_PRIORITY[job_stage],
                    maximum_attempts=MAXIMUM_ATTEMPTS,
                    requested_gpu=str(gpu),
                )
                existing[digest] = row
            receipt_rows.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "config_sha256": digest,
                    "job_id": str(row["job_id"]),
                    "requested_gpu": gpu,
                    "maximum_attempts": MAXIMUM_ATTEMPTS,
                }
            )
            result = _receipt_payload(
                stage=stage,
                materialization=materialization,
                resource_gate=resource_gate,
                representation_gate=representation_gate,
                disk_used_decimal_gb=disk_used,
                jobs=receipt_rows,
                expected_count=len(validated),
            )
            _atomic_receipt(receipt_path, result)
        assert result is not None
        return result


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("resource", "stage1", "stage2"),
        required=True,
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument("--resource-gate", type=Path, default=None)
    parser.add_argument("--representation-gate", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    locked = current_paths().scratch_root / "locked_campaigns" / CAMPAIGN_ID
    receipt_path = (
        args.receipt or locked / f"{args.stage}_enqueue_receipt.json"
    )
    result = enqueue_stage(
        stage=args.stage,
        materialization_path=args.materialization.resolve(),
        resource_gate_path=(
            None
            if args.resource_gate is None
            else args.resource_gate.resolve()
        ),
        representation_gate_path=(
            None
            if args.representation_gate is None
            else args.representation_gate.resolve()
        ),
        receipt_path=receipt_path.resolve(),
        database_path=args.database.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "stage": args.stage,
                "complete": result["complete"],
                "job_count": len(result["jobs"]),
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

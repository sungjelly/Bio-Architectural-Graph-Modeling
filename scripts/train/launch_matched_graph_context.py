#!/usr/bin/env python3
"""Materialize, select, and safely launch the matched graph-context campaign.

This coordinator is the outcome firewall between validation-only tuning and
outer-fold confirmation.  It freezes concrete job matrices, accepts only
strict checksum-bound tuning ``results.json`` bundles, and publishes the
selection receipt before a confirmation matrix can be created.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "cmp_20260812_matched_graph_context_nested_cv"
CONTRACT_RELATIVE = Path(
    "experiments/campaigns/cmp_20260812_matched_graph_context_nested_cv/"
    "frozen_task_contract.yaml"
)
CONTRACT_SHA256 = "4e39e1ef623a5ce4e1d67f4733bdea790000745d523a8845bcd11a0ddea77b73"
PREPARED_RELATIVE = Path(
    "data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all"
)
PREPARED_MANIFEST_SHA256 = (
    "e59546beeed2e8e2523248e89ac6738bf90aa65f28248a42a7091c54255d0f32"
)
PREPARED_INTEGRITY_SHA256 = (
    "65a6dde8e9aa7f733bb1f06075c3a0dedd2084a89c55eaa575c91276a45aff18"
)
PROCESSED_FINGERPRINT = (
    "01c525695883784befc1b9ebbe37a6d96b248d5b18450f46a1255e571bd3819e"
)
SPLIT_FINGERPRINT = (
    "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"
)
ARMS = ("no_graph", "observed_near", "permuted_near", "observed_annular")
FOLDS = (0, 1, 2, 3)
STAGE_A_SEED = 20260812
STAGE_B_SEEDS = (20261812, 20262812)
CONFIRMATION_SEEDS = (20260812, 20261812, 20262812, 20263812, 20264812)
EPOCHS = (12, 24, 48, 96, 192)
GPU_IDS = (0, 1, 2, 3)
MINIMUM_FREE_DISK_GB = 25.0
NEAR_TIE_RELATIVE_TOLERANCE = 0.0025
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN_ID = re.compile(
    r"^r_(?P<year>\d{4})(?P<month>\d{2})\d{2}T\d{6}Z_"
    r"[a-z0-9]+_s\d+_f\d+_a\d+_[a-z0-9_-]+$"
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    hidden_width: int
    learning_rate: float
    weight_decay: float
    dropout: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hidden_width": self.hidden_width,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
        }


# Exact pre-outcome, contract-frozen space-filling design.  Never synthesize a
# Cartesian product here: candidate identity is a scientific authority.
CANDIDATES = (
    Candidate("c00", 32, 3e-4, 0.0, 0.0),
    Candidate("c01", 32, 3e-4, 1e-3, 0.2),
    Candidate("c02", 32, 1e-3, 1e-4, 0.1),
    Candidate("c03", 32, 3e-3, 0.0, 0.2),
    Candidate("c04", 32, 3e-3, 1e-3, 0.0),
    Candidate("c05", 64, 3e-4, 0.0, 0.1),
    Candidate("c06", 64, 3e-4, 1e-3, 0.0),
    Candidate("c07", 64, 1e-3, 0.0, 0.2),
    Candidate("c08", 64, 1e-3, 1e-4, 0.1),
    Candidate("c09", 64, 3e-3, 1e-4, 0.0),
    Candidate("c10", 64, 3e-3, 1e-3, 0.2),
    Candidate("c11", 128, 3e-4, 1e-4, 0.2),
    Candidate("c12", 128, 3e-4, 1e-3, 0.0),
    Candidate("c13", 128, 1e-3, 0.0, 0.0),
    Candidate("c14", 128, 1e-3, 1e-3, 0.1),
    Candidate("c15", 128, 3e-3, 1e-4, 0.1),
)
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}


class OrchestrationError(RuntimeError):
    """A frozen workflow invariant or execution safety check failed."""


class SelectionError(OrchestrationError):
    """Validation-only evidence was malformed, incomplete, or inconsistent."""


def _verify_published_bundle(result_path: Path) -> None:
    """Verify the checksum-bound immutable bundle containing ``result_path``."""

    if result_path.name != "results.json":
        raise OrchestrationError("result authority must be a bundle results.json")
    source_root = str(PROJECT_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from spatial_benchmark.run_archive import verify_run_bundle

        verification = verify_run_bundle(
            result_path.parent, require_success_contract=False
        )
    except Exception as error:
        raise OrchestrationError(
            f"published bundle checksum verification failed: {result_path.parent}: {error}"
        ) from error
    if verification.get("valid") is not True or verification.get("status") != "success":
        raise OrchestrationError("published bundle is not verified success")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise OrchestrationError(f"path escapes project root: {path}") from error


def _resolve_under(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise OrchestrationError(f"path escapes project root: {value}") from error
    if resolved.is_symlink():
        raise OrchestrationError(f"authority path may not be a symlink: {resolved}")
    if must_exist and not resolved.is_file():
        raise OrchestrationError(f"required file is missing: {resolved}")
    return resolved


def _strict_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SelectionError(f"{label} contains non-finite value {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except SelectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"{label} is not strict readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OrchestrationError(f"refusing symlink output: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise OrchestrationError(f"immutable output already exists: {path}") from error
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_authorities(root: Path, prepared_root: Path | None = None) -> dict[str, Any]:
    contract = _resolve_under(root, CONTRACT_RELATIVE, must_exist=True)
    if _sha256_file(contract) != CONTRACT_SHA256:
        raise OrchestrationError("frozen task contract checksum changed")
    prepared = (prepared_root or root / PREPARED_RELATIVE).resolve(strict=True)
    manifest = prepared / "manifest.json"
    integrity = prepared / "integrity_manifest.json"
    if _sha256_file(manifest) != PREPARED_MANIFEST_SHA256:
        raise OrchestrationError("prepared manifest checksum changed")
    if _sha256_file(integrity) != PREPARED_INTEGRITY_SHA256:
        raise OrchestrationError("prepared integrity-manifest checksum changed")
    manifest_value = _strict_json(manifest, label="prepared manifest")
    if manifest_value.get("processed_fingerprint") != PROCESSED_FINGERPRINT:
        raise OrchestrationError("processed fingerprint changed")
    if manifest_value.get("split_fingerprint") != SPLIT_FINGERPRINT:
        raise OrchestrationError("split fingerprint changed")
    # Reuse the prior campaign's audited strict V0 verifier.  This verifies all
    # 60 integrity entries (safe relative target, storage type, size, SHA,
    # ndarray shape/dtype) and recomputes within-FOV, no-self-edge,
    # split-isolation, degree, and receiver-collision-free permutation
    # invariants.  A manifest checksum alone is not sufficient preflight.
    verifier_path = root / "scripts/train/run_same_gene_robustness.py"
    module_name = "_bagm_strict_same_gene_variant_verifier"
    module = sys.modules.get(module_name)
    if module is None:
        specification = importlib.util.spec_from_file_location(module_name, verifier_path)
        if specification is None or specification.loader is None:
            raise OrchestrationError("could not load strict V0 verifier")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
    try:
        verified = module._verify_variant_root(
            prepared, project_root=root, binding=None
        )
    except Exception as error:
        raise OrchestrationError(f"strict V0 content/graph verification failed: {error}") from error
    if (
        verified.manifest_sha256 != PREPARED_MANIFEST_SHA256
        or verified.integrity_manifest_sha256 != PREPARED_INTEGRITY_SHA256
        or verified.processed_fingerprint != PROCESSED_FINGERPRINT
        or verified.split_fingerprint != SPLIT_FINGERPRINT
    ):
        raise OrchestrationError("strict V0 verifier returned changed authorities")
    source_paths = (
        root / "scripts/train/launch_matched_graph_context.py",
        root / "scripts/train/run_matched_graph_context.py",
        root / "src/spatial_benchmark/matched_graph_context.py",
    )
    for source in source_paths:
        if source.is_symlink() or not source.is_file():
            raise OrchestrationError(f"execution source is unavailable: {source}")
    return {
        "contract": {"path": _relative(contract, root), "sha256": CONTRACT_SHA256},
        "prepared_root": _relative(prepared, root),
        "prepared_manifest": {
            "path": _relative(manifest, root),
            "sha256": PREPARED_MANIFEST_SHA256,
        },
        "prepared_integrity_manifest": {
            "path": _relative(integrity, root),
            "sha256": PREPARED_INTEGRITY_SHA256,
        },
        "processed_fingerprint": PROCESSED_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "execution_sources": [
            {
                "path": _relative(source, root),
                "sha256": _sha256_file(source),
            }
            for source in source_paths
        ],
    }


def _materialize_synthetic_gate(root: Path, path: Path) -> dict[str, Any]:
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from spatial_benchmark.matched_graph_context import run_synthetic_recovery_gates

    result = run_synthetic_recovery_gates(STAGE_A_SEED)
    if result.get("passed") is not True:
        raise OrchestrationError(f"synthetic recovery gate failed: {result}")
    for control in ("positive", "null"):
        row = result.get(control)
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise OrchestrationError(f"synthetic {control} control failed")
        for metric in ("observed_mse", "no_graph_mse", "relative_gain"):
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise OrchestrationError(f"synthetic {control} metric is nonfinite")
    result_sha = result.get("result_sha256")
    payload_without_result = dict(result)
    payload_without_result.pop("result_sha256", None)
    if result_sha != _canonical_sha256(payload_without_result):
        raise OrchestrationError("synthetic gate result self-checksum failed")
    payload = {
        "schema_version": 1,
        "kind": "matched_graph_context_synthetic_gate",
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": CONTRACT_SHA256,
        "created_at": _utc_now(),
        "result": result,
        "passed": True,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    if path.exists():
        existing = _strict_json(path, label="synthetic gate receipt")
        digest = existing.pop("payload_sha256", None)
        if digest != _canonical_sha256(existing):
            raise OrchestrationError("existing synthetic gate receipt is corrupt")
        # Creation time is operational only; the scientific result must replay.
        if existing.get("result") != result or existing.get("contract_sha256") != CONTRACT_SHA256:
            raise OrchestrationError("existing synthetic gate receipt differs from replay")
        existing["payload_sha256"] = digest
        return existing
    _atomic_json(path, payload, exclusive=True)
    return payload


def candidate_design_sha256() -> str:
    return _canonical_sha256([candidate.as_dict() for candidate in CANDIDATES])


def _run_id(stage: str, arm: str, seed: int, fold: int, candidate_id: str) -> str:
    # Run IDs are materialized identities, not generated by workers.  The date
    # partition is the contract-free physical current date; the suffix binds
    # the full scientific slot and prevents concurrent attempts sharing output.
    now = datetime.now(timezone.utc)
    slot = {
        "campaign": CAMPAIGN_ID,
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "candidate_id": candidate_id,
        "contract": CONTRACT_SHA256,
    }
    token = _canonical_sha256(slot)[:8]
    scientific = _canonical_sha256({key: value for key, value in slot.items() if key not in {"seed", "fold"}})[:8]
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"r_{stamp}_{scientific}_s{seed}_f{fold:02d}_a01_{token}"


def _artifact_relative(run_id: str) -> Path:
    match = SAFE_RUN_ID.fullmatch(run_id)
    if match is None:
        raise OrchestrationError(f"invalid canonical run id: {run_id}")
    return Path("artifacts/runs") / match.group("year") / match.group("month") / run_id


def _job(
    *, root: Path, stage: str, arm: str, seed: int, fold: int, candidate_id: str,
    selection_receipt: Path | None = None,
) -> dict[str, Any]:
    run_id = _run_id(stage, arm, seed, fold, candidate_id)
    artifact = _artifact_relative(run_id)
    argv = [
        "/venv/main/bin/python", "scripts/train/run_matched_graph_context.py",
        "--mode", "confirm" if stage == "confirmation" else "tune",
        "--run-id", run_id, "--arm", arm, "--fold", str(fold),
        "--seed", str(seed), "--device", "cuda:0",
    ]
    if stage == "confirmation":
        if selection_receipt is None:
            raise OrchestrationError("confirmation requires a selection receipt")
        argv += ["--selection-receipt", _relative(selection_receipt, root)]
    else:
        argv += ["--candidate-id", candidate_id]
    job_id = f"{stage}.{arm}.{candidate_id}.s{seed}.f{fold}"
    log_root = Path("state/logs") / CAMPAIGN_ID / stage
    return {
        "job_id": job_id,
        "stage": stage,
        "arm": arm,
        "candidate_id": candidate_id,
        "seed": seed,
        "fold": fold,
        "run_id": run_id,
        "gpu": "auto",
        "argv": argv,
        "stdout_path": (log_root / f"{job_id}.stdout.log").as_posix(),
        "stderr_path": (log_root / f"{job_id}.stderr.log").as_posix(),
        "result_path": (artifact / "results.json").as_posix(),
        "success_marker": (artifact / "_SUCCESS").as_posix(),
    }


def _write_plan(root: Path, output: Path, *, stage: str, jobs: Sequence[Mapping[str, Any]],
                authorities: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not jobs:
        raise OrchestrationError("cannot materialize an empty plan")
    job_ids = [str(job["job_id"]) for job in jobs]
    run_ids = [str(job["run_id"]) for job in jobs]
    write_paths = [
        str(job[key]) for job in jobs
        for key in ("stdout_path", "stderr_path", "result_path", "success_marker")
    ]
    if len(job_ids) != len(set(job_ids)) or len(run_ids) != len(set(run_ids)):
        raise OrchestrationError("plan job and run identities must be unique")
    if len(write_paths) != len(set(write_paths)):
        raise OrchestrationError("plan output and log paths must be isolated")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "matched_graph_context_job_plan",
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "created_at": _utc_now(),
        "contract_sha256": CONTRACT_SHA256,
        "candidate_design_sha256": candidate_design_sha256(),
        "authorities": dict(authorities),
        "minimum_free_disk_gb": MINIMUM_FREE_DISK_GB,
        "gpu_ids": list(GPU_IDS),
        "jobs": [dict(job) for job in jobs],
    }
    if extra:
        payload.update(extra)
    payload["plan_payload_sha256"] = _canonical_sha256(payload)
    _atomic_json(output, payload, exclusive=True)
    return payload


def materialize_stage_a(root: Path, output: Path, *, prepared_root: Path | None = None) -> dict[str, Any]:
    authorities = _verify_authorities(root, prepared_root)
    gate_path = _default_state(root) / "synthetic_gate.json"
    gate = _materialize_synthetic_gate(root, gate_path)
    authorities["synthetic_gate"] = {
        "path": _relative(gate_path, root),
        "sha256": _sha256_file(gate_path),
        "result_sha256": gate["result"]["result_sha256"],
        "payload_sha256": gate["payload_sha256"],
    }
    jobs = [
        _job(root=root, stage="stage_a", arm=arm, seed=STAGE_A_SEED,
             fold=fold, candidate_id=candidate.candidate_id)
        for arm in ARMS for candidate in CANDIDATES for fold in FOLDS
    ]
    if len(jobs) != 4 * 16 * 4:
        raise AssertionError("stage A coverage construction changed")
    return _write_plan(root, output, stage="stage_a", jobs=jobs, authorities=authorities)


def _load_plan(root: Path, path: Path, *, expected_stage: str) -> dict[str, Any]:
    plan = _strict_json(path, label=f"{expected_stage} plan")
    digest = plan.pop("plan_payload_sha256", None)
    if not isinstance(digest, str) or digest != _canonical_sha256(plan):
        raise OrchestrationError(f"{expected_stage} plan self-checksum failed")
    plan["plan_payload_sha256"] = digest
    if plan.get("kind") != "matched_graph_context_job_plan":
        raise OrchestrationError("unrecognized plan kind")
    if plan.get("campaign_id") != CAMPAIGN_ID or plan.get("stage") != expected_stage:
        raise OrchestrationError("plan campaign or stage mismatch")
    if plan.get("contract_sha256") != CONTRACT_SHA256:
        raise OrchestrationError("plan contract binding mismatch")
    if plan.get("candidate_design_sha256") != candidate_design_sha256():
        raise OrchestrationError("plan candidate design changed")
    authorities = _verify_authorities(root)
    observed_authorities = plan.get("authorities")
    if not isinstance(observed_authorities, Mapping):
        raise OrchestrationError("plan input authorities are malformed")
    expected_authorities = dict(authorities)
    if expected_stage == "stage_a":
        gate_authority = observed_authorities.get("synthetic_gate")
        if not isinstance(gate_authority, Mapping):
            raise OrchestrationError("Stage A plan lacks synthetic gate authority")
        gate_path = _resolve_under(root, str(gate_authority.get("path")), must_exist=True)
        gate = _strict_json(gate_path, label="synthetic gate receipt")
        digest = gate.pop("payload_sha256", None)
        if digest != _canonical_sha256(gate) or gate.get("passed") is not True:
            raise OrchestrationError("synthetic gate receipt no longer verifies")
        gate["payload_sha256"] = digest
        expected_authorities["synthetic_gate"] = {
            "path": _relative(gate_path, root), "sha256": _sha256_file(gate_path),
            "result_sha256": gate["result"]["result_sha256"],
            "payload_sha256": digest,
        }
    if dict(observed_authorities) != expected_authorities:
        raise OrchestrationError("plan input authorities changed")
    if expected_stage == "stage_b":
        source = plan.get("source_stage_a_selection")
        if not isinstance(source, Mapping):
            raise OrchestrationError("Stage B plan lacks Stage A selection authority")
        selection_path = _resolve_under(
            root, str(source.get("path")), must_exist=True
        )
        if _sha256_file(selection_path) != source.get("sha256"):
            raise OrchestrationError("Stage A selection changed after Stage B materialization")
        selection = _load_stage_a_selection(root, selection_path)
        if selection.get("payload_sha256") != source.get("payload_sha256"):
            raise OrchestrationError("Stage A selection payload binding changed")
    if expected_stage == "confirmation":
        receipt_authority = plan.get("selection_receipt")
        if not isinstance(receipt_authority, Mapping):
            raise OrchestrationError("confirmation plan lacks selection receipt authority")
        receipt_path = _resolve_under(
            root, str(receipt_authority.get("path")), must_exist=True
        )
        if _sha256_file(receipt_path) != receipt_authority.get("sha256"):
            raise OrchestrationError("selection receipt changed after confirmation materialization")
        receipt = _load_receipt(root, receipt_path)
        if receipt.get("payload_sha256") != receipt_authority.get("payload_sha256"):
            raise OrchestrationError("selection receipt payload binding changed")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise OrchestrationError("plan jobs are missing")
    return plan


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SelectionError(f"{label} must be finite and nonnegative")
    return result


def _validation_curve(result: Mapping[str, Any]) -> dict[int, float]:
    raw = result.get("validation_by_epoch")
    if not isinstance(raw, list):
        raise SelectionError("tune result is missing validation_by_epoch")
    curve: dict[int, float] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise SelectionError(f"validation_by_epoch[{index}] must be an object")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in EPOCHS:
            raise SelectionError("validation curve contains an unapproved epoch")
        if epoch in curve:
            raise SelectionError("validation curve contains duplicate epochs")
        metric = row.get("validation_component_equal_mse")
        if metric is None and isinstance(row.get("metrics"), Mapping):
            metric = row["metrics"].get("validation_component_equal_mse")
        curve[epoch] = _finite_float(metric, label=f"epoch {epoch} validation MSE")
    if set(curve) != set(EPOCHS):
        raise SelectionError("validation curve does not cover all frozen epochs")
    return curve


def _forbidden_tuning_key(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}" if path else str(key)
            if "test" in normalized or normalized in {"outer", "outer_fold_metric"}:
                raise SelectionError(f"tuning bundle contains prohibited outcome key {child_path}")
            _forbidden_tuning_key(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_tuning_key(child, f"{path}[{index}]")


def _read_tuning_job(root: Path, job: Mapping[str, Any]) -> tuple[dict[int, float], dict[str, Any]]:
    result_path = _resolve_under(root, str(job.get("result_path")), must_exist=True)
    marker = _resolve_under(root, str(job.get("success_marker")), must_exist=True)
    if marker.name != "_SUCCESS":
        raise SelectionError("unexpected success-marker identity")
    _verify_published_bundle(result_path)
    result = _strict_json(result_path, label="tuning result")
    _forbidden_tuning_key(result)
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "mode": "tune",
        "run_id": job.get("run_id"),
        "arm": job.get("arm"),
        "fold": job.get("fold"),
        "seed": job.get("seed"),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise SelectionError(f"tuning result {key} binding mismatch")
    if result.get("contract_sha256") != CONTRACT_SHA256:
        raise SelectionError("tuning result contract checksum mismatch")
    candidate_id = str(job.get("candidate_id"))
    config = result.get("config")
    if not isinstance(config, Mapping) or config.get("candidate_id") != candidate_id:
        raise SelectionError("tuning result candidate binding mismatch")
    frozen = CANDIDATE_BY_ID.get(candidate_id)
    if frozen is None:
        raise SelectionError("tuning result uses an unknown candidate")
    for key, expected_value in frozen.as_dict().items():
        if config.get(key) != expected_value:
            raise SelectionError(f"tuning candidate field {key} changed")
    split_roles = result.get("split_roles")
    outer_fold = int(job["fold"])
    expected_roles = {
        "train_folds": [
            fold
            for fold in FOLDS
            if fold not in {outer_fold, (outer_fold + 1) % 4}
        ],
        "validation_fold": (outer_fold + 1) % 4,
        "excluded_fold": outer_fold,
    }
    if not isinstance(split_roles, Mapping):
        raise SelectionError("tuning result lacks split role evidence")
    normalized_roles = dict(split_roles)
    if isinstance(normalized_roles.get("train_folds"), tuple):
        normalized_roles["train_folds"] = list(normalized_roles["train_folds"])
    if normalized_roles != expected_roles:
        raise SelectionError("tuning result nested split roles changed")
    return _validation_curve(result), {
        "path": _relative(result_path, root),
        "sha256": _sha256_file(result_path),
        "success_marker": _relative(marker, root),
        "success_marker_sha256": _sha256_file(marker),
    }


def _aggregate(root: Path, plan: Mapping[str, Any]) -> tuple[dict[tuple[str, str, int, int], dict[int, float]], list[dict[str, Any]]]:
    curves: dict[tuple[str, str, int, int], dict[int, float]] = {}
    sources: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        key = (str(job["arm"]), str(job["candidate_id"]), int(job["seed"]), int(job["fold"]))
        if key in curves:
            raise SelectionError(f"duplicate tuning slot {key}")
        curve, authority = _read_tuning_job(root, job)
        curves[key] = curve
        sources.append({**authority, "job_id": job["job_id"]})
    return curves, sources


def _candidate_epoch_score(curves: Mapping[tuple[str, str, int, int], Mapping[int, float]],
                           arm: str, candidate_id: str, seeds: Sequence[int],
                           folds: Sequence[int]) -> tuple[int, float, float, dict[int, float]]:
    rows = _candidate_epoch_rows(
        curves, arm, candidate_id, seeds, folds
    )
    # One epoch is selected from the aggregate, never independently by fold.
    selected = min(rows, key=lambda row: (row["validation_mean"], row["epoch"]))
    return (
        int(selected["epoch"]),
        float(selected["validation_mean"]),
        float(selected["seed_sd"]),
        dict(selected["seed_means_raw"]),
    )


def _candidate_epoch_rows(
    curves: Mapping[tuple[str, str, int, int], Mapping[int, float]],
    arm: str,
    candidate_id: str,
    seeds: Sequence[int],
    folds: Sequence[int],
) -> list[dict[str, Any]]:
    """Score every prespecified epoch after aggregating the allowed folds."""

    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        seed_means: dict[int, float] = {}
        for seed in seeds:
            values = [
                curves[(arm, candidate_id, seed, fold)][epoch]
                for fold in folds
            ]
            seed_means[seed] = sum(values) / len(values)
        mean = sum(seed_means.values()) / len(seed_means)
        variance = (
            sum((value - mean) ** 2 for value in seed_means.values())
            / len(seed_means)
        )
        rows.append(
            {
                "epoch": epoch,
                "validation_mean": mean,
                "seed_sd": math.sqrt(variance),
                "seed_means_raw": seed_means,
            }
        )
    return rows


def select_stage_a(root: Path, plan_path: Path, output: Path) -> dict[str, Any]:
    plan = _load_plan(root, plan_path, expected_stage="stage_a")
    curves, sources = _aggregate(root, plan)
    expected = {(arm, candidate.candidate_id, STAGE_A_SEED, fold)
                for arm in ARMS for candidate in CANDIDATES for fold in FOLDS}
    if set(curves) != expected:
        raise SelectionError("stage A coverage is incomplete or unexpected")
    advanced: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for outer_fold in FOLDS:
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            selected: list[dict[str, Any]] = []
            for hidden in (32, 64, 128):
                ranked: list[tuple[float, str, int, float]] = []
                for candidate in CANDIDATES:
                    if candidate.hidden_width != hidden:
                        continue
                    epoch, mean, seed_sd, _ = _candidate_epoch_score(
                        curves, arm, candidate.candidate_id, (STAGE_A_SEED,),
                        (outer_fold,),
                    )
                    ranked.append((mean, candidate.candidate_id, epoch, seed_sd))
                mean, candidate_id, epoch, seed_sd = min(ranked)
                selected.append({
                    **CANDIDATE_BY_ID[candidate_id].as_dict(), "epoch": epoch,
                    "stage_a_validation_mean": mean, "stage_a_seed_sd": seed_sd,
                })
            by_arm[arm] = selected
        advanced[str(outer_fold)] = by_arm
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "matched_graph_context_stage_a_selection",
        "campaign_id": CAMPAIGN_ID,
        "created_at": _utc_now(),
        "contract_sha256": CONTRACT_SHA256,
        "candidate_design_sha256": candidate_design_sha256(),
        "source_plan": {"path": _relative(plan_path, root), "sha256": _sha256_file(plan_path),
                        "payload_sha256": plan["plan_payload_sha256"]},
        "test_metrics_used_for_selection": False,
        "advance_rule": "best_candidate_per_outer_fold_per_arm_per_hidden_width",
        "cross_outer_pooling": False,
        "advanced_by_outer_fold": advanced,
        "source_tuning_results": sources,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    _atomic_json(output, payload, exclusive=True)
    return payload


def _load_stage_a_selection(root: Path, path: Path) -> dict[str, Any]:
    value = _strict_json(path, label="stage A selection")
    digest = value.pop("payload_sha256", None)
    if digest != _canonical_sha256(value):
        raise SelectionError("stage A selection self-checksum failed")
    value["payload_sha256"] = digest
    if value.get("kind") != "matched_graph_context_stage_a_selection":
        raise SelectionError("unrecognized stage A selection kind")
    if value.get("campaign_id") != CAMPAIGN_ID or value.get("contract_sha256") != CONTRACT_SHA256:
        raise SelectionError("stage A selection authority mismatch")
    if value.get("test_metrics_used_for_selection") is not False:
        raise SelectionError("stage A selection is not validation-only")
    advanced = value.get("advanced_by_outer_fold")
    if not isinstance(advanced, Mapping) or set(advanced) != {str(fold) for fold in FOLDS}:
        raise SelectionError("stage A selection must cover four outer folds")
    if value.get("cross_outer_pooling") is not False:
        raise SelectionError("stage A selection does not prohibit cross-outer pooling")
    for outer_fold in FOLDS:
        by_arm = advanced[str(outer_fold)]
        if not isinstance(by_arm, Mapping) or set(by_arm) != set(ARMS):
            raise SelectionError("stage A selection must cover all arms per outer fold")
        for arm in ARMS:
            rows = by_arm[arm]
            if not isinstance(rows, list) or len(rows) != 3:
                raise SelectionError("stage A must advance exactly three candidates per outer/arm")
            if {row.get("hidden_width") for row in rows if isinstance(row, Mapping)} != {32, 64, 128}:
                raise SelectionError("stage A must preserve all hidden-width strata")
    return value


def materialize_stage_b(root: Path, stage_a_selection: Path, output: Path,
                        *, prepared_root: Path | None = None) -> dict[str, Any]:
    authorities = _verify_authorities(root, prepared_root)
    selection = _load_stage_a_selection(root, stage_a_selection)
    jobs = [
        _job(root=root, stage="stage_b", arm=arm, seed=seed, fold=outer_fold,
             candidate_id=str(row["candidate_id"]))
        for outer_fold in FOLDS
        for arm in ARMS
        for row in selection["advanced_by_outer_fold"][str(outer_fold)][arm]
        for seed in STAGE_B_SEEDS
    ]
    if len(jobs) != 4 * 4 * 3 * 2:
        raise AssertionError("stage B nested coverage construction changed")
    return _write_plan(
        root, output, stage="stage_b", jobs=jobs, authorities=authorities,
        extra={
            "source_stage_a_selection": {
                "path": _relative(stage_a_selection, root), "sha256": _sha256_file(stage_a_selection),
                "payload_sha256": selection["payload_sha256"],
            },
            "candidates_by_outer_fold": {
                str(fold): {
                    arm: [str(row["candidate_id"]) for row in selection["advanced_by_outer_fold"][str(fold)][arm]]
                    for arm in ARMS
                }
                for fold in FOLDS
            },
            "stage_b_additional_seeds": list(STAGE_B_SEEDS),
        },
    )


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["seed_sd"]),
        -float(row["weight_decay"]),
        float(row["dropout"]),
        float(row["learning_rate"]),
        str(row["candidate_id"]),
        int(row["epoch"]),
    )


def _near_tie_select(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SelectionError("cannot select from an empty candidate set")
    minimum = min(float(row["validation_mean"]) for row in rows)
    eligible = [row for row in rows if float(row["validation_mean"]) <= minimum * (1.0 + NEAR_TIE_RELATIVE_TOLERANCE)]
    selected = dict(min(eligible, key=_ranking_key))
    selected["near_tie_minimum"] = minimum
    selected["near_tie_relative_tolerance"] = NEAR_TIE_RELATIVE_TOLERANCE
    selected["near_tie_candidate_ids"] = sorted(str(row["candidate_id"]) for row in eligible)
    return selected


def _select_shared_width(
    rows_by_arm_width: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]], int
]:
    """Choose a shared width from raw minima, retaining near-tie arm configs.

    Regret is defined by the contract using ``best_mse``.  It must therefore
    use each stratum's raw minimum rather than the possibly slightly worse
    configuration retained by the variance-first 0.25% near-tie rule.
    """

    best_by_arm_width = {
        (arm, width): _near_tie_select(rows_by_arm_width[(arm, width)])
        for arm in ARMS for width in (32, 64, 128)
    }
    minimum_by_arm_width = {
        (arm, width): min(
            float(row["validation_mean"])
            for row in rows_by_arm_width[(arm, width)]
        )
        for arm in ARMS for width in (32, 64, 128)
    }
    arm_minima = {
        arm: min(
            minimum_by_arm_width[(arm, width)]
            for width in (32, 64, 128)
        )
        for arm in ARMS
    }
    width_rows: list[dict[str, Any]] = []
    for width in (32, 64, 128):
        regrets = {
            arm: minimum_by_arm_width[(arm, width)] / arm_minima[arm] - 1.0
            for arm in ARMS
        }
        width_rows.append({
            "hidden_width": width,
            "relative_regret_by_arm": regrets,
            "maximum_relative_regret": max(regrets.values()),
            "mean_relative_regret": sum(regrets.values()) / len(regrets),
        })
    shared = min(
        width_rows,
        key=lambda row: (
            row["maximum_relative_regret"],
            row["mean_relative_regret"],
            row["hidden_width"],
        ),
    )
    return best_by_arm_width, width_rows, int(shared["hidden_width"])


def lock_selection(root: Path, stage_a_plan_path: Path, stage_a_selection_path: Path,
                   stage_b_plan_path: Path, output: Path) -> dict[str, Any]:
    stage_a_plan = _load_plan(root, stage_a_plan_path, expected_stage="stage_a")
    stage_a_selection = _load_stage_a_selection(root, stage_a_selection_path)
    stage_b_plan = _load_plan(root, stage_b_plan_path, expected_stage="stage_b")
    source = stage_b_plan.get("source_stage_a_selection")
    if not isinstance(source, Mapping) or source.get("sha256") != _sha256_file(stage_a_selection_path):
        raise SelectionError("stage B plan does not bind the supplied Stage A selection")
    curves_a, sources_a = _aggregate(root, stage_a_plan)
    curves_b, sources_b = _aggregate(root, stage_b_plan)
    curves = dict(curves_a)
    overlap = set(curves).intersection(curves_b)
    if overlap:
        raise SelectionError("Stage A and B repeat the same tuning slots")
    curves.update(curves_b)
    source_by_job = {
        str(row["job_id"]): row for row in (*sources_a, *sources_b)
    }
    selected_by_outer: dict[str, dict[str, Any]] = {}
    diagnostics_by_outer: dict[str, dict[str, Any]] = {}
    sources_by_outer: dict[str, list[dict[str, Any]]] = {}
    for outer_fold in FOLDS:
        advanced = stage_a_selection["advanced_by_outer_fold"][str(outer_fold)]
        rows_by_arm_width: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        selected_candidate_ids: dict[str, set[str]] = {}
        for arm in ARMS:
            selected_candidate_ids[arm] = {
                str(row["candidate_id"]) for row in advanced[arm]
            }
            for candidate_id in sorted(selected_candidate_ids[arm]):
                expected_slots = {
                    (arm, candidate_id, seed, outer_fold)
                    for seed in (STAGE_A_SEED, *STAGE_B_SEEDS)
                }
                if not expected_slots.issubset(curves):
                    raise SelectionError(
                        f"nested Stage B coverage incomplete for outer {outer_fold} "
                        f"{arm}/{candidate_id}"
                    )
                epoch_rows = _candidate_epoch_rows(
                    curves, arm, candidate_id,
                    (STAGE_A_SEED, *STAGE_B_SEEDS), (outer_fold,),
                )
                candidate = CANDIDATE_BY_ID[candidate_id]
                for epoch_row in epoch_rows:
                    rows_by_arm_width[(arm, candidate.hidden_width)].append({
                        **candidate.as_dict(),
                        "epoch": epoch_row["epoch"],
                        "validation_mean": epoch_row["validation_mean"],
                        "seed_sd": epoch_row["seed_sd"],
                        "seed_means": {
                            str(key): value
                            for key, value in sorted(
                                epoch_row["seed_means_raw"].items()
                            )
                        },
                    })
        best_by_arm_width, width_rows, shared_width = _select_shared_width(
            rows_by_arm_width
        )
        parameter_count = 1_028_000 + 2_001 * shared_width
        selected_arms: dict[str, Any] = {}
        for arm in ARMS:
            selected = best_by_arm_width[(arm, shared_width)]
            config = {
                key: selected[key]
                for key in (
                    "candidate_id", "hidden_width", "learning_rate",
                    "weight_decay", "dropout", "epoch",
                )
            }
            config["batch_size"] = 4096
            selected_arms[arm] = {
                "candidate_id": selected["candidate_id"],
                "config": config,
                "config_sha256": _canonical_sha256(config),
                "parameter_count": parameter_count,
                "validation_mean": selected["validation_mean"],
                "seed_sd": selected["seed_sd"],
                "seed_means": selected["seed_means"],
                "near_tie_minimum": selected["near_tie_minimum"],
                "near_tie_relative_tolerance": selected[
                    "near_tie_relative_tolerance"
                ],
                "near_tie_candidate_ids": selected["near_tie_candidate_ids"],
            }
        selected_by_outer[str(outer_fold)] = selected_arms
        diagnostics_by_outer[str(outer_fold)] = {
            "shared_hidden_width": shared_width,
            "shared_parameter_count": parameter_count,
            "shared_hidden_diagnostics": width_rows,
            "validation_fold": (outer_fold + 1) % 4,
            "cross_outer_pooling": False,
        }
        relevant_ids = {
            str(job["job_id"])
            for plan in (stage_a_plan, stage_b_plan)
            for job in plan["jobs"]
            if int(job["fold"]) == outer_fold
            and (
                plan is stage_a_plan
                or str(job["candidate_id"]) in selected_candidate_ids[str(job["arm"])]
            )
        }
        sources_by_outer[str(outer_fold)] = sorted(
            (source_by_job[job_id] for job_id in relevant_ids),
            key=lambda row: str(row["job_id"]),
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "matched_graph_context_selection_receipt",
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen",
        "created_at": _utc_now(),
        "contract_sha256": CONTRACT_SHA256,
        "candidate_design_sha256": candidate_design_sha256(),
        "test_metrics_used_for_selection": False,
        "selection_metric": "validation_component_equal_mse",
        "selection_rule": {
            "shared_hidden": "minimax_relative_arm_regret_then_mean_regret_then_smaller_width",
            "arm_specific_near_tie_relative_tolerance": NEAR_TIE_RELATIVE_TOLERANCE,
            "arm_specific_tie_break": ["lower_seed_sd", "larger_weight_decay", "lower_dropout", "lower_learning_rate", "candidate_id"],
        },
        "cross_outer_pooling": False,
        "selected_by_outer_fold": selected_by_outer,
        "selection_diagnostics_by_outer_fold": diagnostics_by_outer,
        "source_stage_a_plan": {"path": _relative(stage_a_plan_path, root), "sha256": _sha256_file(stage_a_plan_path)},
        "source_stage_a_selection": {"path": _relative(stage_a_selection_path, root), "sha256": _sha256_file(stage_a_selection_path)},
        "source_stage_b_plan": {"path": _relative(stage_b_plan_path, root), "sha256": _sha256_file(stage_b_plan_path)},
        "source_tuning_results_by_outer_fold": sources_by_outer,
        "source_tuning_result_sha256s_by_outer_fold": {
            fold: sorted(row["sha256"] for row in rows)
            for fold, rows in sources_by_outer.items()
        },
    }
    receipt["payload_sha256"] = _canonical_sha256(receipt)
    # Exclusive atomic publication is the irreversible test-outcome firewall.
    _atomic_json(output, receipt, exclusive=True)
    return receipt


def _load_receipt(root: Path, path: Path) -> dict[str, Any]:
    value = _strict_json(path, label="selection receipt")
    digest = value.pop("payload_sha256", None)
    if digest != _canonical_sha256(value):
        raise SelectionError("selection receipt self-checksum failed")
    value["payload_sha256"] = digest
    if value.get("kind") != "matched_graph_context_selection_receipt":
        raise SelectionError("unrecognized selection receipt kind")
    if value.get("campaign_id") != CAMPAIGN_ID or value.get("contract_sha256") != CONTRACT_SHA256:
        raise SelectionError("selection receipt campaign/contract mismatch")
    if value.get("status") != "frozen" or value.get("test_metrics_used_for_selection") is not False:
        raise SelectionError("selection receipt is not frozen validation-only authority")
    if value.get("cross_outer_pooling") is not False:
        raise SelectionError("selection receipt permits cross-outer pooling")
    selected = value.get("selected_by_outer_fold")
    expected_folds = {str(fold) for fold in FOLDS}
    if not isinstance(selected, Mapping) or set(selected) != expected_folds:
        raise SelectionError("selection receipt must contain four nested selections")
    source_groups = value.get("source_tuning_results_by_outer_fold")
    if not isinstance(source_groups, Mapping) or set(source_groups) != expected_folds:
        raise SelectionError("selection receipt lacks per-outer source authorities")
    for outer_fold in FOLDS:
        configs = selected[str(outer_fold)]
        if not isinstance(configs, Mapping) or set(configs) != set(ARMS):
            raise SelectionError("each outer fold must select all four arms")
        widths: set[Any] = set()
        counts: set[Any] = set()
        for arm, block in configs.items():
            if not isinstance(block, Mapping):
                raise SelectionError("selected arm block must be an object")
            config = block.get("config")
            if not isinstance(config, Mapping):
                raise SelectionError("selected arm lacks exact config")
            if block.get("candidate_id") != config.get("candidate_id"):
                raise SelectionError("selected candidate identity mismatch")
            if block.get("config_sha256") != _canonical_sha256(config):
                raise SelectionError("selected config self-checksum failed")
            widths.add(config.get("hidden_width"))
            counts.add(block.get("parameter_count"))
        if len(widths) != 1 or len(counts) != 1:
            raise SelectionError("selected arms are not exactly parameter matched")
        for authority in source_groups[str(outer_fold)]:
            if not isinstance(authority, Mapping) or int(str(authority["job_id"]).rsplit(".f", 1)[1]) != outer_fold:
                raise SelectionError("per-outer source group contains another outer fold")
    for authority_name in ("source_stage_a_plan", "source_stage_a_selection", "source_stage_b_plan"):
        authority = value.get(authority_name)
        if not isinstance(authority, Mapping) or set(authority) < {"path", "sha256"}:
            raise SelectionError(f"receipt lacks {authority_name} authority")
        source_path = _resolve_under(root, str(authority["path"]), must_exist=True)
        if _sha256_file(source_path) != authority["sha256"]:
            raise SelectionError(f"receipt source authority drifted: {authority_name}")
    return value


def materialize_confirmation(root: Path, receipt_path: Path, output: Path,
                             *, prepared_root: Path | None = None) -> dict[str, Any]:
    authorities = _verify_authorities(root, prepared_root)
    receipt = _load_receipt(root, receipt_path)
    jobs = [
        _job(root=root, stage="confirmation", arm=arm, seed=seed, fold=fold,
             candidate_id=str(receipt["selected_by_outer_fold"][str(fold)][arm]["candidate_id"]),
             selection_receipt=receipt_path)
        for arm in ARMS for seed in CONFIRMATION_SEEDS for fold in FOLDS
    ]
    if len(jobs) != 80:
        raise AssertionError("confirmation coverage construction changed")
    return _write_plan(
        root, output, stage="confirmation", jobs=jobs, authorities=authorities,
        extra={
            "selection_receipt": {
                "path": _relative(receipt_path, root),
                "sha256": _sha256_file(receipt_path),
                "payload_sha256": receipt["payload_sha256"],
            },
            "selected_by_outer_fold": receipt["selected_by_outer_fold"],
        },
    )


@dataclass(slots=True)
class Running:
    job: dict[str, Any]
    gpu: int
    process: subprocess.Popen[bytes]
    stdout: Any
    stderr: Any


class FourGpuLauncher:
    """Run at most one isolated child on each physical GPU slot."""

    def __init__(self, root: Path, plan_path: Path, stage: str, *, poll_seconds: float = 0.2) -> None:
        self.root = root.resolve()
        self.plan_path = plan_path.resolve(strict=True)
        self.plan = _load_plan(self.root, self.plan_path, expected_stage=stage)
        self.stage = stage
        self.poll_seconds = poll_seconds
        state = self.root / "state" / "matched_graph_context" / CAMPAIGN_ID
        self.lock_path = state / f"{stage}.launcher.lock"
        self.ledger_path = state / f"{stage}.ledger.json"
        self.stop_signal: int | None = None

    def request_stop(self, signum: int, *_: object) -> None:
        self.stop_signal = int(signum)

    def _preflight(self) -> None:
        self._check_disk()
        occupied = self._external_gpu_processes()
        if occupied.intersection(GPU_IDS):
            raise OrchestrationError(
                f"configured physical GPU slots are externally occupied: {sorted(occupied)}"
            )
        for job in self.plan["jobs"]:
            for key in ("stdout_path", "stderr_path", "result_path", "success_marker"):
                _resolve_under(self.root, str(job[key]))
            if not isinstance(job.get("argv"), list) or not job["argv"]:
                raise OrchestrationError("job argv must be a nonempty list")

    def _check_disk(self) -> None:
        free = shutil.disk_usage(self.root).free / (1024 ** 3)
        if free < float(self.plan["minimum_free_disk_gb"]):
            raise OrchestrationError(
                f"free disk {free:.2f} GiB is below contract floor"
            )

    def _external_gpu_processes(self) -> set[int]:
        """Return physical slots with compute processes not owned by this launcher.

        ``nvidia-smi`` exposes UUIDs for compute processes, so first bind every
        physical index to its UUID.  A query failure is unsafe and fails closed.
        """

        try:
            inventory = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30,
            ).stdout
            uuid_to_gpu: dict[str, int] = {}
            for line in inventory.splitlines():
                index, uuid = (item.strip() for item in line.split(",", 1))
                uuid_to_gpu[uuid] = int(index)
            processes = subprocess.run(
                [
                    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise OrchestrationError("could not audit physical GPU occupancy") from error
        own_pids = {
            child.process.pid
            for child in getattr(self, "_running_snapshot", {}).values()
        }
        occupied: set[int] = set()
        for line in processes.splitlines():
            if not line.strip():
                continue
            uuid, pid_text = (item.strip() for item in line.split(",", 1))
            try:
                pid = int(pid_text)
                gpu = uuid_to_gpu[uuid]
            except (KeyError, ValueError) as error:
                raise OrchestrationError("malformed GPU process inventory") from error
            if pid not in own_pids:
                occupied.add(gpu)
        return occupied

    def _complete(self, job: Mapping[str, Any]) -> bool:
        marker = _resolve_under(self.root, str(job["success_marker"]))
        result_path = _resolve_under(self.root, str(job["result_path"]))
        if not marker.is_file() or not result_path.is_file():
            return False
        _verify_published_bundle(result_path)
        result = _strict_json(result_path, label="completed job result")
        identity_matches = (
            result.get("campaign_id") == CAMPAIGN_ID
            and result.get("run_id") == job.get("run_id")
            and result.get("arm") == job.get("arm")
            and result.get("fold") == job.get("fold")
            and result.get("seed") == job.get("seed")
            and result.get("contract_sha256") == CONTRACT_SHA256
            and result.get("mode") == ("confirm" if self.stage == "confirmation" else "tune")
        )
        peak = result.get("peak_vram_gb")
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or float(peak) > 20.5
        ):
            raise OrchestrationError("job peak VRAM is absent, nonfinite, or above 20.5 GiB")
        return identity_matches

    def run(self) -> int:
        self._preflight()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise OrchestrationError("launcher lock may not be a symlink")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OrchestrationError("another launcher owns this stage") from error
            return self._run_locked()

    def _run_locked(self) -> int:
        pending = deque(dict(job) for job in self.plan["jobs"] if not self._complete(job))
        completed = {str(job["job_id"]): {"status": "skipped"}
                     for job in self.plan["jobs"] if self._complete(job)}
        running: dict[int, Running] = {}
        self._running_snapshot = running
        prior_handlers: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                prior_handlers[signum] = signal.signal(signum, self.request_stop)
            except ValueError:
                pass
        try:
            while pending or running:
                if self.stop_signal is not None:
                    for child in running.values():
                        child.process.send_signal(self.stop_signal)
                    pending.clear()
                free_gpus = [gpu for gpu in GPU_IDS if gpu not in running]
                while pending and free_gpus and self.stop_signal is None:
                    self._check_disk()
                    externally_occupied = self._external_gpu_processes()
                    collision = externally_occupied.intersection(free_gpus)
                    if collision:
                        raise OrchestrationError(
                            "external process occupied a pending GPU slot: "
                            f"{sorted(collision)}"
                        )
                    gpu = free_gpus.pop(0)
                    job = pending.popleft()
                    stdout_path = _resolve_under(self.root, job["stdout_path"])
                    stderr_path = _resolve_under(self.root, job["stderr_path"])
                    stdout_path.parent.mkdir(parents=True, exist_ok=True)
                    stderr_path.parent.mkdir(parents=True, exist_ok=True)
                    stdout = stdout_path.open("ab", buffering=0)
                    stderr = stderr_path.open("ab", buffering=0)
                    env = os.environ.copy()
                    env.update({
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONPATH": str(self.root / "src"),
                        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    })
                    try:
                        process = subprocess.Popen(
                            list(job["argv"]), cwd=self.root, env=env,
                            stdout=stdout, stderr=stderr, shell=False,
                            start_new_session=True,
                        )
                    except BaseException:
                        stdout.close(); stderr.close()
                        raise
                    running[gpu] = Running(job, gpu, process, stdout, stderr)
                    completed[job["job_id"]] = {"status": "running", "gpu": gpu, "pid": process.pid}
                    _atomic_json(self.ledger_path, {
                        "schema_version": 1, "stage": self.stage,
                        "plan_sha256": _sha256_file(self.plan_path), "updated_at": _utc_now(),
                        "jobs": completed,
                    })
                progressed = False
                for gpu, child in list(running.items()):
                    code = child.process.poll()
                    if code is None:
                        continue
                    progressed = True
                    child.stdout.close(); child.stderr.close()
                    del running[gpu]
                    status = "completed" if code == 0 and self._complete(child.job) else "failed"
                    completed[child.job["job_id"]] = {"status": status, "gpu": gpu, "exit_code": code}
                    if status != "completed":
                        self.stop_signal = self.stop_signal or signal.SIGTERM
                        for other in running.values():
                            other.process.terminate()
                    self._check_disk()
                _atomic_json(self.ledger_path, {
                    "schema_version": 1, "stage": self.stage,
                    "plan_sha256": _sha256_file(self.plan_path), "updated_at": _utc_now(),
                    "jobs": completed,
                })
                if not progressed and running:
                    time.sleep(self.poll_seconds)
            if self.stop_signal is not None:
                return 128 + int(self.stop_signal)
            return 0 if all(row["status"] in {"completed", "skipped"} for row in completed.values()) else 1
        finally:
            for child in running.values():
                if child.process.poll() is None:
                    child.process.terminate()
            deadline = time.monotonic() + 30.0
            for child in running.values():
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    child.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.process.kill()
                    child.process.wait()
                if not child.stdout.closed:
                    child.stdout.close()
                if not child.stderr.closed:
                    child.stderr.close()
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)


def _default_state(root: Path) -> Path:
    return root / "state" / "matched_graph_context" / CAMPAIGN_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize", help="materialize the complete Stage A matrix")
    materialize.add_argument("--output", type=Path)
    for name, stage in (("launch-stage-a", "stage_a"), ("launch-stage-b", "stage_b"), ("launch-confirmation", "confirmation")):
        child = sub.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
    select_a = sub.add_parser("select-stage-a")
    select_a.add_argument("--plan", type=Path, required=True)
    select_a.add_argument("--output", type=Path)
    materialize_b = sub.add_parser("materialize-stage-b")
    materialize_b.add_argument("--stage-a-selection", type=Path, required=True)
    materialize_b.add_argument("--output", type=Path)
    lock = sub.add_parser("lock-selection")
    lock.add_argument("--stage-a-plan", type=Path, required=True)
    lock.add_argument("--stage-a-selection", type=Path, required=True)
    lock.add_argument("--stage-b-plan", type=Path, required=True)
    lock.add_argument("--output", type=Path)
    confirmation = sub.add_parser("materialize-confirmation")
    confirmation.add_argument("--selection-receipt", type=Path, required=True)
    confirmation.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.project_root.resolve(strict=True)
    state = _default_state(root)
    try:
        if arguments.command == "materialize":
            output = arguments.output or state / "stage_a_plan.json"
            result = materialize_stage_a(root, _resolve_under(root, output))
        elif arguments.command.startswith("launch-"):
            stage = {"launch-stage-a": "stage_a", "launch-stage-b": "stage_b", "launch-confirmation": "confirmation"}[arguments.command]
            return FourGpuLauncher(root, _resolve_under(root, arguments.plan, must_exist=True), stage).run()
        elif arguments.command == "select-stage-a":
            output = arguments.output or state / "stage_a_selection.json"
            result = select_stage_a(root, _resolve_under(root, arguments.plan, must_exist=True), _resolve_under(root, output))
        elif arguments.command == "materialize-stage-b":
            output = arguments.output or state / "stage_b_plan.json"
            result = materialize_stage_b(root, _resolve_under(root, arguments.stage_a_selection, must_exist=True), _resolve_under(root, output))
        elif arguments.command == "lock-selection":
            output = arguments.output or state / "selection_receipt.json"
            result = lock_selection(
                root, _resolve_under(root, arguments.stage_a_plan, must_exist=True),
                _resolve_under(root, arguments.stage_a_selection, must_exist=True),
                _resolve_under(root, arguments.stage_b_plan, must_exist=True),
                _resolve_under(root, output),
            )
        elif arguments.command == "materialize-confirmation":
            output = arguments.output or state / "confirmation_plan.json"
            result = materialize_confirmation(root, _resolve_under(root, arguments.selection_receipt, must_exist=True), _resolve_under(root, output))
        else:  # pragma: no cover
            raise AssertionError(arguments.command)
    except (OSError, ValueError, OrchestrationError) as error:
        print(f"matched graph-context orchestration failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "campaign_id": CAMPAIGN_ID, "command": arguments.command,
        "output_sha256": result.get("payload_sha256", result.get("plan_payload_sha256")),
        "test_metrics_used_for_selection": result.get("test_metrics_used_for_selection"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

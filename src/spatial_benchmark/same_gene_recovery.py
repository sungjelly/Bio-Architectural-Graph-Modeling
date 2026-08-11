"""Immutable technical-recovery authority for the same-gene campaign.

The scientific contract is deliberately *not* amended after launch.  This
module validates a narrowly scoped execution amendment which joins an already
materialized parent launch to one corrected child launch.  The child launch
hashes the amendment while the amendment hashes the child launch payload with
its own source row removed.  That normalized, two-way binding avoids an
impossible self-referential hash.

Only source/configuration and failed-attempt lifecycle metadata are handled
here.  Scientific result payloads are never opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from .identifiers import canonical_sha256


CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
CONTRACT_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml"
)
ENVIRONMENT_LOCK_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/environment_lock.json"
)
TECHNICAL_AMENDMENT_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "pilot_attempt2_source_recovery_v1.json"
)
TECHNICAL_AMENDMENT_SCHEMA_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "technical_amendment.schema.json"
)
AUTHORIZED_CORRECTIONS = (
    "honor_the_frozen_932_gene_mask_without_fold_local_reselection",
    "encode_the_identity_oracle_without_nonfinite_json_numbers",
    "reject_any_nonfinite_value_in_the_complete_result_payload_before_publish",
    (
        "serialize_checkpoint_normalization_statistics_and_validate_"
        "recorded_live_and_replay_mse"
    ),
    "define_residual_gene_eligibility_from_pretransform_prepared_expression",
    (
        "restore_pilot_final_train_to_the_uncapped_eligible_non_test_split_"
        "while_refit_remains_tuning_train_capped"
    ),
    (
        "evaluate_the_budget_fold_support_gate_against_the_prespecified_"
        "selected_near_vs_anchor_near_comparison"
    ),
    "require_exact_recomputed_csv_schema_and_content_coherence",
    (
        "make_analysis_verify_only_recompute_and_bind_json_npz_csv_provenance_"
        "report_and_run_lineage_to_current_authorities"
    ),
    "support_hash_bound_cross_launch_failed_attempt_lineage",
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VARIANT = re.compile(r"^V[0-6]$")
_SCIENTIFIC_ID = re.compile(r"^sci_[0-9a-f]{16}$")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
TECHNICAL_AMENDMENT_ID = (
    "ta_cmp_20260810_same_gene_robustness_pilot_a2_source_recovery_v1"
)

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "amendment_id",
        "campaign_id",
        "status",
        "frozen_at",
        "scope",
        "scientific_effects_accessed",
        "contract",
        "parent_authority",
        "child_authority",
        "authorized_corrections",
        "scientific_invariants",
        "failed_attempts",
    }
)
_CONTRACT_KEYS = frozenset({"path", "sha256"})
_PARENT_KEYS = frozenset(
    {
        "launch_path",
        "launch_sha256",
        "source_manifest_sha256",
        "pilot_plan_path",
        "pilot_plan_sha256",
        "pilot_ledger_path",
        "pilot_ledger_sha256",
        "git_commit",
    }
)
_CHILD_KEYS = frozenset(
    {
        "launch_path",
        "launch_core_sha256",
        "source_list_path",
        "source_list_sha256",
        "profile",
        "first_recovery_attempt",
        "required_variants",
    }
)
_INVARIANT_KEYS = frozenset(
    {
        "campaign_id_unchanged",
        "contract_sha256_unchanged",
        "prepared_data_fingerprints_unchanged",
        "environment_lock_sha256_unchanged",
        "scientific_payload_must_match_parent",
        "seeds_folds_hyperparameters_gates_unchanged",
        "all_latest_pilot_attempts_share_child_launch",
        "production_blocked_until_all_seven_receipts_verify",
    }
)
_FAILED_KEYS = frozenset(
    {
        "variant",
        "attempt",
        "model_seed",
        "fold",
        "scientific_id",
        "run_id",
        "materialized_config_sha256",
        "artifact_path",
        "failed_marker_sha256",
        "exception_sha256",
        "registry_status",
        "artifact_status",
        "failure_category",
    }
)
_LAUNCH_KEYS = frozenset({"campaign_id", "contract", "sources"})
_SOURCE_KEYS = frozenset({"path", "size", "sha"})


class TechnicalAmendmentError(RuntimeError):
    """Raised when a recovery authority is missing, mutable, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = set(value)
    if observed != set(expected):
        raise TechnicalAmendmentError(
            f"{label} keys differ; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise TechnicalAmendmentError(f"{label} must be a lowercase SHA-256")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TechnicalAmendmentError(f"{label} must be a nonempty string")
    return value


def _project_path(
    project_root: Path,
    value: str | Path,
    *,
    label: str,
    file: bool = True,
    allow_absolute: bool = False,
) -> Path:
    root = project_root.resolve(strict=True)
    raw = Path(value)
    if raw.is_absolute():
        if not allow_absolute:
            raise TechnicalAmendmentError(f"{label} must be project-relative")
        candidate = raw
    else:
        if raw.as_posix() != str(value) or ".." in raw.parts or "\\" in str(value):
            raise TechnicalAmendmentError(f"{label} is not a normalized project path")
        candidate = root / raw
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise TechnicalAmendmentError(f"{label} escapes project root") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise TechnicalAmendmentError(f"{label} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=file)
    except OSError as error:
        raise TechnicalAmendmentError(f"{label} is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise TechnicalAmendmentError(f"{label} resolves outside project root") from error
    if file and not resolved.is_file():
        raise TechnicalAmendmentError(f"{label} is not a regular file")
    return resolved


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            project_root.resolve(strict=True)
        ).as_posix()
    except ValueError as error:
        raise TechnicalAmendmentError("authority path escapes project root") from error


def strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TechnicalAmendmentError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            object_pairs_hook=unique_object,
        )
    except TechnicalAmendmentError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise TechnicalAmendmentError(f"cannot read strict {label}") from error
    if not isinstance(value, dict):
        raise TechnicalAmendmentError(f"{label} must contain an object")
    return value


def normalized_source_rows(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            raise TechnicalAmendmentError(f"launch source {index} is malformed")
        _exact_keys(raw, _SOURCE_KEYS, label=f"launch source {index}")
        path = _string(raw["path"], label=f"launch source {index}.path")
        if Path(path).is_absolute() or Path(path).as_posix() != path or ".." in Path(path).parts:
            raise TechnicalAmendmentError("launch source path is not normalized")
        if path in seen:
            raise TechnicalAmendmentError("launch source path is duplicated")
        seen.add(path)
        size = raw["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TechnicalAmendmentError("launch source size is invalid")
        rows.append({"path": path, "size": size, "sha": _sha(raw["sha"], label="source SHA")})
    return tuple(sorted(rows, key=lambda row: row["path"]))


def source_manifest_sha256(sources: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(list(normalized_source_rows(sources)))


def launch_core_sha256(
    launch_payload: Mapping[str, Any],
    *,
    amendment_path: str = TECHNICAL_AMENDMENT_RELATIVE_PATH,
) -> str:
    """Hash a child launch after removing only its amendment source row."""

    _exact_keys(launch_payload, _LAUNCH_KEYS, label="launch manifest")
    sources = normalized_source_rows(launch_payload["sources"])
    remaining = [row for row in sources if row["path"] != amendment_path]
    if len(remaining) != len(sources) - 1:
        raise TechnicalAmendmentError(
            "child launch must contain exactly one technical-amendment source"
        )
    core = {
        "campaign_id": launch_payload["campaign_id"],
        "contract": launch_payload["contract"],
        "sources": remaining,
    }
    return canonical_sha256(core)


@dataclass(frozen=True, slots=True)
class TechnicalAmendment:
    path: Path
    sha256: str
    payload: dict[str, Any]
    contract_sha256: str
    parent_launch_path: Path
    parent_launch_sha256: str
    parent_source_manifest_sha256: str
    parent_plan_path: Path
    parent_plan_sha256: str
    parent_ledger_path: Path
    parent_ledger_sha256: str
    parent_git_commit: str
    child_launch_path: Path
    child_launch_core_sha256: str
    child_source_list_path: Path
    child_source_list_sha256: str
    failed_attempts: dict[str, dict[str, Any]]


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise TechnicalAmendmentError("amendment frozen_at is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise TechnicalAmendmentError("amendment frozen_at is invalid") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise TechnicalAmendmentError("amendment frozen_at is not canonical UTC")
    return value


def load_technical_amendment(
    amendment_path: str | Path,
    *,
    project_root: str | Path,
) -> TechnicalAmendment:
    root = Path(project_root).resolve(strict=True)
    path = _project_path(
        root,
        amendment_path,
        label="technical amendment",
        allow_absolute=True,
    )
    if _relative(path, root) != TECHNICAL_AMENDMENT_RELATIVE_PATH:
        raise TechnicalAmendmentError("technical amendment path is not canonical")
    payload = strict_json(path, label="technical amendment")
    _exact_keys(payload, _TOP_KEYS, label="technical amendment")
    if (
        payload["schema_version"] != 1
        or payload["campaign_id"] != CAMPAIGN_ID
        or payload["status"] != "frozen_technical_recovery"
        or payload["scope"] != "execution_only_no_scientific_change"
        or payload["scientific_effects_accessed"] is not False
    ):
        raise TechnicalAmendmentError("technical amendment scope/status is invalid")
    if payload["amendment_id"] != TECHNICAL_AMENDMENT_ID:
        raise TechnicalAmendmentError("technical amendment_id changed")
    _validate_timestamp(payload["frozen_at"])

    contract = payload["contract"]
    parent = payload["parent_authority"]
    child = payload["child_authority"]
    invariants = payload["scientific_invariants"]
    if not all(isinstance(value, Mapping) for value in (contract, parent, child, invariants)):
        raise TechnicalAmendmentError("amendment authority sections must be objects")
    _exact_keys(contract, _CONTRACT_KEYS, label="amendment contract")
    _exact_keys(parent, _PARENT_KEYS, label="parent authority")
    _exact_keys(child, _CHILD_KEYS, label="child authority")
    _exact_keys(invariants, _INVARIANT_KEYS, label="scientific invariants")
    if any(value is not True for value in invariants.values()):
        raise TechnicalAmendmentError("every scientific invariant must be true")
    if contract["path"] != CONTRACT_RELATIVE_PATH:
        raise TechnicalAmendmentError("amendment contract path changed")
    contract_sha = _sha(contract["sha256"], label="contract SHA")
    contract_path = _project_path(root, contract["path"], label="amendment contract")
    if sha256_file(contract_path) != contract_sha:
        raise TechnicalAmendmentError("amendment contract SHA changed")

    parent_launch = _project_path(root, parent["launch_path"], label="parent launch")
    parent_plan = _project_path(root, parent["pilot_plan_path"], label="parent pilot plan")
    parent_ledger = _project_path(root, parent["pilot_ledger_path"], label="parent pilot ledger")
    parent_launch_sha = _sha(parent["launch_sha256"], label="parent launch SHA")
    parent_plan_sha = _sha(parent["pilot_plan_sha256"], label="parent plan SHA")
    parent_ledger_sha = _sha(parent["pilot_ledger_sha256"], label="parent ledger SHA")
    if (
        sha256_file(parent_launch) != parent_launch_sha
        or sha256_file(parent_plan) != parent_plan_sha
        or sha256_file(parent_ledger) != parent_ledger_sha
    ):
        raise TechnicalAmendmentError("a parent launch/plan/ledger authority changed")
    commit = parent["git_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise TechnicalAmendmentError("parent git_commit must be a full commit SHA")

    child_launch = _project_path(
        root, child["launch_path"], label="child launch", file=False
    )
    child_source_list = _project_path(
        root, child["source_list_path"], label="child source list"
    )
    if (
        child["profile"] != "pilot"
        or child["first_recovery_attempt"] != 2
        or child["required_variants"] != [f"V{index}" for index in range(7)]
    ):
        raise TechnicalAmendmentError("child recovery inventory changed")
    source_list_sha = _sha(child["source_list_sha256"], label="child source-list SHA")
    if sha256_file(child_source_list) != source_list_sha:
        raise TechnicalAmendmentError("child source list changed")

    corrections = payload["authorized_corrections"]
    if corrections != list(AUTHORIZED_CORRECTIONS):
        raise TechnicalAmendmentError("authorized technical corrections changed")

    failed = payload["failed_attempts"]
    if not isinstance(failed, list) or len(failed) != 7:
        raise TechnicalAmendmentError("amendment must bind seven failed attempts")
    failed_by_variant: dict[str, dict[str, Any]] = {}
    if [raw.get("variant") for raw in failed if isinstance(raw, Mapping)] != [
        f"V{index}" for index in range(7)
    ]:
        raise TechnicalAmendmentError("failed-attempt order changed")
    for index, raw in enumerate(failed):
        if not isinstance(raw, Mapping):
            raise TechnicalAmendmentError("failed-attempt entry is malformed")
        _exact_keys(raw, _FAILED_KEYS, label=f"failed attempt {index}")
        variant = raw["variant"]
        if not isinstance(variant, str) or _VARIANT.fullmatch(variant) is None:
            raise TechnicalAmendmentError("failed-attempt variant is invalid")
        if variant in failed_by_variant:
            raise TechnicalAmendmentError("failed-attempt variant is duplicated")
        if (
            raw["attempt"] != 1
            or raw["model_seed"] != 20260810
            or raw["fold"] != 0
            or raw["registry_status"] != "failed"
            or raw["artifact_status"] != "failed"
            or raw["failure_category"] != "same_gene_nonlinear_run_failure"
        ):
            raise TechnicalAmendmentError("failed-attempt lifecycle changed")
        for key in (
            "materialized_config_sha256",
            "failed_marker_sha256",
            "exception_sha256",
        ):
            _sha(raw[key], label=f"failed attempt {key}")
        if (
            not isinstance(raw["scientific_id"], str)
            or _SCIENTIFIC_ID.fullmatch(raw["scientific_id"]) is None
        ):
            raise TechnicalAmendmentError("failed scientific_id is invalid")
        _string(raw["run_id"], label="failed run_id")
        artifact_path = _string(raw["artifact_path"], label="failed artifact_path")
        artifact = Path(artifact_path)
        if (
            artifact.is_absolute()
            or artifact.as_posix() != artifact_path
            or ".." in artifact.parts
            or "\\" in artifact_path
        ):
            raise TechnicalAmendmentError("failed artifact_path is not normalized")
        failed_by_variant[variant] = dict(raw)
    if set(failed_by_variant) != {f"V{index}" for index in range(7)}:
        raise TechnicalAmendmentError("failed-attempt inventory is incomplete")

    return TechnicalAmendment(
        path=path,
        sha256=sha256_file(path),
        payload=payload,
        contract_sha256=contract_sha,
        parent_launch_path=parent_launch,
        parent_launch_sha256=parent_launch_sha,
        parent_source_manifest_sha256=_sha(
            parent["source_manifest_sha256"], label="parent source-manifest SHA"
        ),
        parent_plan_path=parent_plan,
        parent_plan_sha256=parent_plan_sha,
        parent_ledger_path=parent_ledger,
        parent_ledger_sha256=parent_ledger_sha,
        parent_git_commit=commit,
        child_launch_path=child_launch,
        child_launch_core_sha256=_sha(
            child["launch_core_sha256"], label="child launch-core SHA"
        ),
        child_source_list_path=child_source_list,
        child_source_list_sha256=source_list_sha,
        failed_attempts=failed_by_variant,
    )


def verify_parent_git_sources(
    amendment: TechnicalAmendment,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verify every historical launch source against the bound Git commit."""

    root = Path(project_root).resolve(strict=True)
    launch = strict_json(amendment.parent_launch_path, label="parent launch")
    _exact_keys(launch, _LAUNCH_KEYS, label="parent launch")
    if launch["campaign_id"] != CAMPAIGN_ID:
        raise TechnicalAmendmentError("parent launch campaign changed")
    contract = launch["contract"]
    if not isinstance(contract, Mapping) or set(contract) != {"path", "sha"}:
        raise TechnicalAmendmentError("parent launch contract is malformed")
    if (
        contract["path"] != CONTRACT_RELATIVE_PATH
        or contract["sha"] != amendment.contract_sha256
    ):
        raise TechnicalAmendmentError("parent launch contract changed")
    rows = normalized_source_rows(launch["sources"])
    if source_manifest_sha256(rows) != amendment.parent_source_manifest_sha256:
        raise TechnicalAmendmentError("parent source-manifest SHA changed")
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", f"{amendment.parent_git_commit}^{{commit}}"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
    except OSError as error:
        raise TechnicalAmendmentError("cannot invoke git for parent authority") from error
    if (
        resolved.returncode != 0
        or resolved.stdout.decode("ascii", errors="strict").strip()
        != amendment.parent_git_commit
    ):
        raise TechnicalAmendmentError("parent Git commit is unavailable")
    for row in rows:
        try:
            shown = subprocess.run(
                ["git", "show", f"{amendment.parent_git_commit}:{row['path']}"],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
            )
        except OSError as error:
            raise TechnicalAmendmentError("cannot read a parent Git source") from error
        if (
            shown.returncode != 0
            or len(shown.stdout) != row["size"]
            or hashlib.sha256(shown.stdout).hexdigest() != row["sha"]
        ):
            raise TechnicalAmendmentError(
                f"parent Git source identity changed: {row['path']}"
            )
    return launch


def verify_child_launch_binding(
    amendment: TechnicalAmendment,
    launch_payload: Mapping[str, Any],
    *,
    launch_path: str | Path,
    project_root: str | Path,
) -> None:
    """Verify the non-circular amendment↔child-launch binding."""

    root = Path(project_root).resolve(strict=True)
    child_path = _project_path(
        root,
        launch_path,
        label="child launch",
        allow_absolute=True,
    )
    if child_path != amendment.child_launch_path:
        raise TechnicalAmendmentError("amendment points at a different child launch")
    _exact_keys(launch_payload, _LAUNCH_KEYS, label="child launch")
    if launch_payload["campaign_id"] != CAMPAIGN_ID:
        raise TechnicalAmendmentError("child launch campaign changed")
    contract = launch_payload["contract"]
    if not isinstance(contract, Mapping) or set(contract) != {"path", "sha"}:
        raise TechnicalAmendmentError("child launch contract is malformed")
    if (
        contract["path"] != CONTRACT_RELATIVE_PATH
        or contract["sha"] != amendment.contract_sha256
    ):
        raise TechnicalAmendmentError("child launch contract changed")
    rows = normalized_source_rows(launch_payload["sources"])
    amendment_rows = [
        row for row in rows if row["path"] == TECHNICAL_AMENDMENT_RELATIVE_PATH
    ]
    if amendment_rows != [
        {
            "path": TECHNICAL_AMENDMENT_RELATIVE_PATH,
            "size": amendment.path.stat().st_size,
            "sha": amendment.sha256,
        }
    ]:
        raise TechnicalAmendmentError("child launch does not hash the amendment")
    source_list_relative = _relative(amendment.child_source_list_path, root)
    source_list_rows = [row for row in rows if row["path"] == source_list_relative]
    if source_list_rows != [
        {
            "path": source_list_relative,
            "size": amendment.child_source_list_path.stat().st_size,
            "sha": amendment.child_source_list_sha256,
        }
    ]:
        raise TechnicalAmendmentError("child launch does not hash its source list")
    parent = strict_json(amendment.parent_launch_path, label="parent launch")
    parent_rows = {row["path"]: row for row in normalized_source_rows(parent["sources"])}
    child_rows = {row["path"]: row for row in rows}
    parent_environment = parent_rows.get(ENVIRONMENT_LOCK_RELATIVE_PATH)
    child_environment = child_rows.get(ENVIRONMENT_LOCK_RELATIVE_PATH)
    if parent_environment is None or child_environment != parent_environment:
        raise TechnicalAmendmentError("environment-lock source identity changed")
    if launch_core_sha256(launch_payload) != amendment.child_launch_core_sha256:
        raise TechnicalAmendmentError("child launch core SHA changed")


__all__ = [
    "TECHNICAL_AMENDMENT_ID",
    "AUTHORIZED_CORRECTIONS",
    "CAMPAIGN_ID",
    "CONTRACT_RELATIVE_PATH",
    "ENVIRONMENT_LOCK_RELATIVE_PATH",
    "TECHNICAL_AMENDMENT_RELATIVE_PATH",
    "TECHNICAL_AMENDMENT_SCHEMA_RELATIVE_PATH",
    "TechnicalAmendment",
    "TechnicalAmendmentError",
    "launch_core_sha256",
    "load_technical_amendment",
    "normalized_source_rows",
    "sha256_file",
    "source_manifest_sha256",
    "strict_json",
    "verify_child_launch_binding",
    "verify_parent_git_sources",
]

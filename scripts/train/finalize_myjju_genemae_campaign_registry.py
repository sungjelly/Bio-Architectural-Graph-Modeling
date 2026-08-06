#!/usr/bin/env python3
"""Finalize the MyJJu GeneMAE ten-core campaign in the local registry.

This versioned, campaign-specific command accepts the exact published
comparison manifest and an explicitly approved SHA-256.  It verifies the
complete report inventory, frozen gate outcome, seven production slots,
registered final checkpoints, and terminal campaign state.  It then takes an
online SQLite backup and conditionally changes only the campaign lifecycle
status from ``planned`` to ``complete``.

Failed pilot and production attempts are retained.  Run, queue, artifact, and
checkpoint rows are never updated by this command.  Re-running against an
already-complete campaign is a verified no-op with a new append-only receipt
and backup.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry, utc_now  # noqa: E402


CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
COMPARISON_KIND = "myjju_genemae_10core_comparison"
RECEIPT_KIND = "myjju_genemae_campaign_registry_finalization_v1"
GENEMAE_MODEL_KEY = "myjju-genemae"
PRODUCTION_SEEDS = tuple(range(7))
EXPECTED_PARAMETER_COUNT = 6_888_016
EXPECTED_COMPLETED_EPOCHS = 200
EXPECTED_FINAL_EPOCH = 199
PRIMARY_METRIC = "fit/partial_gene/log1p_cp10k_masked_huber"
FROZEN_CONTRACT_SHA256 = (
    "6f171b5bece63df943dd8fe679121fbdd41baf290339d64576dc6bfec847eada"
)
TERMINAL_QUEUE_STATUSES = frozenset(
    {"completed", "failed", "pruned", "cancelled", "stale"}
)
TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "pruned", "cancelled"}
)
GATE_NAMES = ("primary_comparison", "baseline", "graph_use")
REQUIRED_REPORT_FILES = frozenset(
    {
        "comparison.json",
        "provenance.json",
        "run_audit.json",
        "attempt_inventory.json",
        "registered_failures.json",
        "pilot_inventory.json",
        "report.md",
        "report.html",
    }
)
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")


class GeneMAECampaignFinalizationError(RuntimeError):
    """Raised when finalization evidence or registry state is invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneMAECampaignFinalizationError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GeneMAECampaignFinalizationError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeneMAECampaignFinalizationError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GeneMAECampaignFinalizationError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneMAECampaignFinalizationError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise GeneMAECampaignFinalizationError(f"{label} must be finite")
    return number


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise GeneMAECampaignFinalizationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GeneMAECampaignFinalizationError(
            f"cannot hash required file: {path}"
        ) from exc
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise GeneMAECampaignFinalizationError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GeneMAECampaignFinalizationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except GeneMAECampaignFinalizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneMAECampaignFinalizationError(
            f"{label} is not strict JSON"
        ) from exc


def _expected_manifest_path(paths: ProjectPaths) -> Path:
    return (
        _comparison_report_root(paths)
        / "comparison"
        / "manifest.json"
    ).resolve(strict=False)


def _comparison_report_root(paths: ProjectPaths) -> Path:
    return (
        paths.report_root
        / "analyses"
        / "myjju_genemae_10core_comparison"
    ).resolve(strict=False)


def _is_versioned_comparison_directory(name: str) -> bool:
    if name == "comparison":
        return True
    prefix = "comparison_v"
    suffix = name.removeprefix(prefix)
    return (
        name.startswith(prefix)
        and suffix.isdigit()
        and not suffix.startswith("0")
        and int(suffix) >= 1
    )


def _reconciliation_directory(paths: ProjectPaths) -> Path:
    return (
        _comparison_report_root(paths)
        / "registry_reconciliation"
    ).resolve(strict=False)


def _require_child(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise GeneMAECampaignFinalizationError(
            f"{label} must remain under {root.resolve(strict=False)}"
        ) from exc
    return resolved


def _safe_report_file(directory: Path, reference: str) -> Path:
    pure = PurePosixPath(reference)
    if (
        not reference
        or pure.is_absolute()
        or reference != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise GeneMAECampaignFinalizationError(
            f"unsafe comparison file reference: {reference!r}"
        )
    return _require_child(
        directory.joinpath(*pure.parts),
        directory,
        label=f"comparison file {reference}",
    )


def _validate_gates(comparison: Mapping[str, Any]) -> tuple[str, list[str]]:
    raw_gates = _list(comparison.get("frozen_gates"), "comparison frozen gates")
    gates: dict[str, bool] = {}
    for index, raw in enumerate(raw_gates):
        gate = _mapping(raw, f"comparison gate {index}")
        name = gate.get("gate")
        passed = gate.get("passed")
        if (
            not isinstance(name, str)
            or name in gates
            or not isinstance(passed, bool)
        ):
            raise GeneMAECampaignFinalizationError(
                "comparison contains an invalid or duplicate gate"
            )
        gates[name] = passed
    if set(gates) != set(GATE_NAMES):
        raise GeneMAECampaignFinalizationError(
            "comparison does not contain the exact frozen gate family"
        )
    failed = [name for name in GATE_NAMES if not gates[name]]
    reported = _list(comparison.get("failed_gates"), "comparison failed gates")
    if reported != failed:
        raise GeneMAECampaignFinalizationError(
            "comparison failed-gate inventory is inconsistent"
        )
    outcome = "supported" if gates["primary_comparison"] else "negative"
    if comparison.get("outcome") != outcome:
        raise GeneMAECampaignFinalizationError(
            "comparison outcome is inconsistent with the primary frozen gate"
        )
    return outcome, failed


def _validate_run_audit(
    raw_rows: Any,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    rows = _list(raw_rows, "comparison run audit")
    selected = [
        _mapping(row, f"run audit row {index}")
        for index, row in enumerate(rows)
        if isinstance(row, Mapping) and row.get("model_key") == GENEMAE_MODEL_KEY
    ]
    if len(selected) != len(PRODUCTION_SEEDS):
        raise GeneMAECampaignFinalizationError(
            "run audit does not contain exactly seven GeneMAE members"
        )
    by_seed: dict[int, dict[str, Any]] = {}
    run_ids: set[str] = set()
    for raw in selected:
        seed = _integer(raw.get("seed"), "run audit seed")
        run_id = raw.get("run_id")
        if (
            seed not in PRODUCTION_SEEDS
            or seed in by_seed
            or not isinstance(run_id, str)
            or not run_id.startswith("r_")
            or run_id in run_ids
        ):
            raise GeneMAECampaignFinalizationError(
                "run audit has invalid GeneMAE seed or run identity"
            )
        if (
            raw.get("bundle_verified") is not True
            or raw.get("registry_artifacts_verified") is not True
            or raw.get("checkpoint_catalog_verified") is not True
            or raw.get("checkpoint_role") != "last"
            or _integer(raw.get("parameter_count"), "run audit parameter count")
            != EXPECTED_PARAMETER_COUNT
            or _integer(raw.get("completed_epochs"), "run audit completed epochs")
            != EXPECTED_COMPLETED_EPOCHS
            or _integer(raw.get("final_epoch"), "run audit final epoch")
            != EXPECTED_FINAL_EPOCH
        ):
            raise GeneMAECampaignFinalizationError(
                f"GeneMAE seed {seed} lacks complete production verification"
            )
        checkpoint_sha = _digest(
            raw.get("checkpoint_file_sha256"),
            f"GeneMAE seed {seed} checkpoint checksum",
        )
        state_sha = _digest(
            raw.get("state_dict_sha256"),
            f"GeneMAE seed {seed} state checksum",
        )
        config_sha = _digest(
            raw.get("config_sha256"),
            f"GeneMAE seed {seed} config checksum",
        )
        by_seed[seed] = {
            "seed": seed,
            "run_id": run_id,
            "attempt": _integer(raw.get("attempt"), "run audit attempt"),
            "checkpoint_sha256": checkpoint_sha,
            "state_dict_sha256": state_sha,
            "config_sha256": config_sha,
        }
        run_ids.add(run_id)
    if set(by_seed) != set(PRODUCTION_SEEDS):
        raise GeneMAECampaignFinalizationError(
            "run audit lacks exact GeneMAE seeds 0 through 6"
        )
    return by_seed, [by_seed[seed]["run_id"] for seed in PRODUCTION_SEEDS]


def _reported_genemae_pilot_statuses(
    raw_rows: Any,
    *,
    label: str,
    failures_only: bool,
) -> dict[str, str]:
    rows = _list(raw_rows, label)
    result: dict[str, str] = {}
    for index, raw in enumerate(rows):
        if (
            not isinstance(raw, Mapping)
            or raw.get("model_key") != GENEMAE_MODEL_KEY
            or raw.get("stage") != "pilot"
        ):
            continue
        row = _mapping(raw, f"{label} row {index}")
        run_id = row.get("run_id")
        status = row.get("status")
        if (
            not isinstance(run_id, str)
            or not run_id.startswith("r_")
            or run_id in result
            or not isinstance(status, str)
            or status not in TERMINAL_RUN_STATUSES
            or _integer(row.get("seed"), f"{label} seed") != 0
            or _integer(row.get("attempt"), f"{label} attempt") < 1
        ):
            raise GeneMAECampaignFinalizationError(
                f"{label} contains an invalid GeneMAE pilot row"
            )
        if failures_only and status not in {"failed", "pruned", "cancelled"}:
            raise GeneMAECampaignFinalizationError(
                f"{label} contains a non-failed GeneMAE pilot row"
            )
        result[run_id] = status
    return dict(sorted(result.items()))


def _validate_report(
    *,
    paths: ProjectPaths,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_digest = _digest(
        expected_manifest_sha256, "expected comparison manifest checksum"
    )
    path = manifest_path.resolve(strict=False)
    report_root = _comparison_report_root(paths)
    if (
        path.name != "manifest.json"
        or path.parent.parent != report_root
        or not _is_versioned_comparison_directory(path.parent.name)
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison manifest is not a canonical versioned MyJJu campaign "
            "report"
        )
    if not path.is_file() or path.is_symlink():
        raise GeneMAECampaignFinalizationError(
            f"comparison manifest is missing or is a symlink: {path}"
        )
    manifest_sha = _sha256_file(path)
    if not hmac.compare_digest(manifest_sha, expected_digest):
        raise GeneMAECampaignFinalizationError(
            "comparison manifest does not match the explicitly approved checksum"
        )
    manifest = _mapping(
        _strict_json(path, label="comparison manifest"),
        "comparison manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != COMPARISON_KIND
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("portable_single_file_html") is not True
        or manifest.get("protected_identifiers_emitted") is not False
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison manifest identity or privacy contract is invalid"
        )

    directory = path.parent
    raw_files = _mapping(manifest.get("files"), "comparison manifest files")
    if (
        "manifest.json" in raw_files
        or not REQUIRED_REPORT_FILES.issubset(raw_files)
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison manifest lacks required report files"
        )
    verified_files: dict[str, dict[str, Any]] = {}
    for reference, raw_entry in sorted(raw_files.items()):
        if not isinstance(reference, str):
            raise GeneMAECampaignFinalizationError(
                "comparison manifest contains a non-string file reference"
            )
        candidate = _safe_report_file(directory, reference)
        if not candidate.is_file() or candidate.is_symlink():
            raise GeneMAECampaignFinalizationError(
                f"comparison file is missing or is a symlink: {reference}"
            )
        entry = _mapping(raw_entry, f"comparison manifest entry {reference}")
        expected_size = _integer(
            entry.get("size_bytes"), f"comparison file size {reference}"
        )
        if expected_size < 0:
            raise GeneMAECampaignFinalizationError(
                f"comparison file size is negative: {reference}"
            )
        expected_file_sha = _digest(
            entry.get("sha256"), f"comparison file checksum {reference}"
        )
        actual_size = candidate.stat().st_size
        actual_sha = _sha256_file(candidate)
        if actual_size != expected_size or not hmac.compare_digest(
            actual_sha, expected_file_sha
        ):
            raise GeneMAECampaignFinalizationError(
                f"comparison file size or checksum mismatch: {reference}"
            )
        verified_files[reference] = {
            "sha256": actual_sha,
            "size_bytes": actual_size,
        }
    actual_files: set[str] = set()
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            raise GeneMAECampaignFinalizationError(
                f"comparison report contains a symlink: {candidate}"
            )
        if candidate.is_file() and candidate != path:
            actual_files.add(candidate.relative_to(directory).as_posix())
    if actual_files != set(raw_files):
        raise GeneMAECampaignFinalizationError(
            "comparison directory and manifest file inventories differ"
        )

    for key, reference in (
        ("comparison_sha256", "comparison.json"),
        ("provenance_sha256", "provenance.json"),
    ):
        if not hmac.compare_digest(
            _digest(manifest.get(key), f"manifest {key}"),
            verified_files[reference]["sha256"],
        ):
            raise GeneMAECampaignFinalizationError(
                f"manifest {key} disagrees with its file entry"
            )

    comparison = _mapping(
        _strict_json(directory / "comparison.json", label="comparison analysis"),
        "comparison analysis",
    )
    if (
        comparison.get("schema_version") != 1
        or comparison.get("artifact_kind") != COMPARISON_KIND
        or comparison.get("campaign_id") != CAMPAIGN_ID
        or comparison.get("status") != "complete"
        or comparison.get("exploratory") is not True
        or comparison.get("estimand") != "held_in_partial_gene_reconstruction"
        or comparison.get("biological_unit") != "tissue_core"
        or comparison.get("core_count") != 10
        or comparison.get("model_seeds") != list(PRODUCTION_SEEDS)
        or comparison.get("seeds_are_biological_replicates") is not False
        or comparison.get("protected_identifiers_emitted") is not False
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison analysis identity or estimand is invalid"
        )
    outcome, failed_gates = _validate_gates(comparison)
    coverage = _mapping(comparison.get("coverage"), "comparison coverage")
    if (
        _integer(coverage.get("expected_batches"), "expected batches") != 180
        or _integer(coverage.get("observed_batches"), "observed batches") != 180
        or _integer(coverage.get("genemae_members"), "GeneMAE members") != 7
        or _integer(coverage.get("bagm_gat_members"), "BAGM GAT members") != 7
        or _integer(coverage.get("bagm_self_members"), "BAGM self members") != 7
        or _integer(
            coverage.get("registered_failures_before_completion"),
            "registered failures",
        )
        < 0
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison does not document exact evaluation coverage"
        )

    run_audit, ordered_run_ids = _validate_run_audit(
        _strict_json(directory / "run_audit.json", label="run audit")
    )
    provenance = _mapping(
        _strict_json(directory / "provenance.json", label="comparison provenance"),
        "comparison provenance",
    )
    run_ids = _mapping(provenance.get("run_ids"), "provenance run IDs")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("campaign_id") != CAMPAIGN_ID
        or provenance.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
        or provenance.get("read_only_registry_audit") is not True
        or provenance.get("protected_identifiers_emitted") is not False
        or run_ids.get(GENEMAE_MODEL_KEY) != ordered_run_ids
    ):
        raise GeneMAECampaignFinalizationError(
            "comparison provenance does not bind the selected GeneMAE runs"
        )
    checkpoint_sha = _mapping(
        provenance.get("checkpoint_sha256"), "provenance checkpoint checksums"
    )
    for seed in PRODUCTION_SEEDS:
        key = f"{GENEMAE_MODEL_KEY}:seed-{seed}"
        if checkpoint_sha.get(key) != run_audit[seed]["checkpoint_sha256"]:
            raise GeneMAECampaignFinalizationError(
                f"provenance checkpoint checksum changed for GeneMAE seed {seed}"
            )

    attempt_pilots = _reported_genemae_pilot_statuses(
        _strict_json(
            directory / "attempt_inventory.json",
            label="attempt_inventory.json",
        ),
        label="attempt_inventory.json",
        failures_only=False,
    )
    reported_failures = _reported_genemae_pilot_statuses(
        _strict_json(
            directory / "registered_failures.json",
            label="registered_failures.json",
        ),
        label="registered_failures.json",
        failures_only=True,
    )
    pilot_inventory = _reported_genemae_pilot_statuses(
        _strict_json(
            directory / "pilot_inventory.json",
            label="pilot_inventory.json",
        ),
        label="pilot_inventory.json",
        failures_only=False,
    )
    if attempt_pilots != pilot_inventory:
        raise GeneMAECampaignFinalizationError(
            "GeneMAE pilot rows differ between attempt and pilot inventories"
        )
    if any(
        pilot_inventory.get(run_id) != status
        for run_id, status in reported_failures.items()
    ):
        raise GeneMAECampaignFinalizationError(
            "GeneMAE pilot failure inventory is not a subset of pilot attempts"
        )

    return {
        "directory": directory.as_posix(),
        "manifest_path": path.as_posix(),
        "manifest_sha256": manifest_sha,
        "manifest_size_bytes": path.stat().st_size,
        "comparison_sha256": verified_files["comparison.json"]["sha256"],
        "provenance_sha256": verified_files["provenance.json"]["sha256"],
        "verified_file_count": len(verified_files),
        "verified_files_checksum": canonical_sha256(verified_files),
        "status": "complete",
        "outcome": outcome,
        "failed_gates": failed_gates,
        "production_by_seed": run_audit,
        "reported_genemae_pilots": pilot_inventory,
        "reported_genemae_pilot_failures": reported_failures,
    }


def _decode_config(value: Any, *, label: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GeneMAECampaignFinalizationError(
            f"{label} is invalid JSON"
        ) from exc
    return _mapping(decoded, label)


def _config_identity(
    config: Mapping[str, Any], *, label: str
) -> tuple[str, int]:
    campaign = _mapping(config.get("campaign"), f"{label} campaign")
    metadata = _mapping(config.get("metadata"), f"{label} metadata")
    model = _mapping(config.get("model"), f"{label} model")
    trainer = _mapping(config.get("trainer"), f"{label} trainer")
    experiment = _mapping(config.get("experiment"), f"{label} experiment")
    classification = _mapping(
        config.get("classification"), f"{label} classification"
    )
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or model.get("name") != GENEMAE_MODEL_KEY
        or model.get("family") != "myjju_dual_path_genemae"
        or model.get("expected_trainable_parameters") != EXPECTED_PARAMETER_COUNT
    ):
        raise GeneMAECampaignFinalizationError(
            f"{label} has an invalid campaign or model identity"
        )
    seed = _integer(config.get("seed"), f"{label} seed")
    role = metadata.get("execution_role")
    if role == "production":
        if (
            seed not in PRODUCTION_SEEDS
            or classification.get("lifecycle_stage") != "exploratory_screen"
            or experiment.get("resource_pilot") is not False
            or experiment.get("conclusion_eligible") is not True
            or experiment.get("variant_label")
            != "myjju_genemae_ensemble_member"
            or trainer.get("diagnostic_resource_pilot") is not False
            or trainer.get("fixed_epoch_budget") is not True
            or trainer.get("max_epochs") != EXPECTED_COMPLETED_EPOCHS
            or trainer.get("checkpoint_policy") != "last_only"
            or trainer.get("primary_checkpoint_role") != "last"
        ):
            raise GeneMAECampaignFinalizationError(
                f"{label} violates the production configuration"
            )
        return "production", seed
    if role == "resource_pilot":
        if (
            seed != 0
            or classification.get("lifecycle_stage") != "diagnostic"
            or experiment.get("resource_pilot") is not True
            or experiment.get("conclusion_eligible") is not False
            or experiment.get("variant_label")
            != "myjju_genemae_resource_pilot"
            or trainer.get("diagnostic_resource_pilot") is not True
            or trainer.get("max_epochs") != 2
        ):
            raise GeneMAECampaignFinalizationError(
                f"{label} violates the resource-pilot configuration"
            )
        return "resource_pilot", seed
    raise GeneMAECampaignFinalizationError(
        f"{label} has unsupported execution role {role!r}"
    )


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        result[status] = result.get(status, 0) + 1
    return dict(sorted(result.items()))


def _row_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(row) for row in rows])


def _verify_checkpoint(
    connection: sqlite3.Connection,
    *,
    paths: ProjectPaths,
    run: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(run["run_id"])
    rows = connection.execute(
        """
        SELECT c.artifact_id, c.role, c.best_epoch, c.monitored_metric,
               c.monitored_mode, c.monitored_value, c.retention_class,
               c.verification_status, a.kind, a.path, a.sha256, a.size_bytes,
               a.status AS artifact_status
        FROM checkpoint_catalog c
        JOIN artifacts a ON a.artifact_id = c.artifact_id
        WHERE c.run_id = ? AND c.role = 'last'
        ORDER BY c.artifact_id
        """,
        (run_id,),
    ).fetchall()
    if len(rows) != 1:
        raise GeneMAECampaignFinalizationError(
            f"{run_id} must have exactly one cataloged last checkpoint"
        )
    row = dict(rows[0])
    checkpoint = Path(str(row["path"]))
    if not checkpoint.is_absolute():
        checkpoint = paths.project_root / checkpoint
    checkpoint = checkpoint.resolve(strict=False)
    artifact_root = Path(str(run["artifact_path"]))
    if not artifact_root.is_absolute():
        artifact_root = paths.project_root / artifact_root
    artifact_root = artifact_root.resolve(strict=False)
    expected_path = artifact_root / "checkpoints" / "last.ckpt"
    config_path = artifact_root / "config.resolved.yaml"
    success_marker = artifact_root / "_SUCCESS"
    registered_sha = _digest(row.get("sha256"), f"{run_id} checkpoint SHA-256")
    if (
        checkpoint != expected_path
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not config_path.is_file()
        or config_path.is_symlink()
        or _sha256_file(config_path) != expected["config_sha256"]
        or not success_marker.is_file()
        or success_marker.is_symlink()
        or "checkpoint" not in str(row["kind"]).lower()
        or row.get("artifact_status") != "present"
        or row.get("verification_status") != "verified"
        or row.get("role") != "last"
        or _integer(row.get("best_epoch"), f"{run_id} final epoch")
        != EXPECTED_FINAL_EPOCH
        or row.get("monitored_metric") != PRIMARY_METRIC
        or row.get("monitored_mode") != "min"
        or _finite(row.get("monitored_value"), f"{run_id} monitored value") < 0
        or row.get("retention_class") != "retain_exploratory_evidence"
        or registered_sha != expected["checkpoint_sha256"]
        or checkpoint.stat().st_size
        != _integer(row.get("size_bytes"), f"{run_id} checkpoint size")
        or _sha256_file(checkpoint) != registered_sha
    ):
        raise GeneMAECampaignFinalizationError(
            f"{run_id} checkpoint catalog or file verification failed"
        )
    return {
        "run_id": run_id,
        "artifact_id": _integer(row["artifact_id"], f"{run_id} artifact ID"),
        "path": checkpoint.as_posix(),
        "sha256": registered_sha,
        "size_bytes": int(row["size_bytes"]),
        "role": "last",
        "final_epoch": EXPECTED_FINAL_EPOCH,
        "verification_status": "verified",
    }


def _registry_snapshot(
    connection: sqlite3.Connection,
    *,
    paths: ProjectPaths,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_row = connection.execute(
        """
        SELECT campaign_id, status, config_json, created_at, updated_at
        FROM campaigns WHERE campaign_id = ?
        """,
        (CAMPAIGN_ID,),
    ).fetchone()
    if campaign_row is None:
        raise GeneMAECampaignFinalizationError(
            "MyJJu GeneMAE campaign is absent from the authoritative registry"
        )
    campaign = dict(campaign_row)
    if campaign["status"] not in {"planned", "complete"}:
        raise GeneMAECampaignFinalizationError(
            f"campaign status cannot be finalized safely: {campaign['status']!r}"
        )

    queue_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT *
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    ]
    run_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT *
            FROM runs
            WHERE campaign_id = ?
            ORDER BY created_at, run_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    ]
    if not queue_rows or not run_rows:
        raise GeneMAECampaignFinalizationError(
            "campaign registry has no queue/run evidence"
        )
    artifact_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT a.*
            FROM artifacts a
            JOIN runs r ON r.run_id = a.run_id
            WHERE r.campaign_id = ?
            ORDER BY a.artifact_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    ]
    checkpoint_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT c.*
            FROM checkpoint_catalog c
            JOIN runs r ON r.run_id = c.run_id
            WHERE r.campaign_id = ?
            ORDER BY c.artifact_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    ]
    failure_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT f.*
            FROM failures f
            LEFT JOIN runs r ON r.run_id = f.run_id
            LEFT JOIN queue_jobs q ON q.job_id = f.job_id
            WHERE r.campaign_id = ? OR q.campaign_id = ?
            ORDER BY f.failure_id
            """,
            (CAMPAIGN_ID, CAMPAIGN_ID),
        ).fetchall()
    ]
    nonterminal_queue = {
        str(row["status"])
        for row in queue_rows
        if str(row["status"]) not in TERMINAL_QUEUE_STATUSES
    }
    nonterminal_runs = {
        str(row["status"])
        for row in run_rows
        if str(row["status"]) not in TERMINAL_RUN_STATUSES
    }
    if nonterminal_queue or nonterminal_runs:
        raise GeneMAECampaignFinalizationError(
            "campaign has active or unknown queue/run states: "
            f"queue={sorted(nonterminal_queue)}, runs={sorted(nonterminal_runs)}"
        )

    run_by_id = {str(row["run_id"]): row for row in run_rows}
    if len(run_by_id) != len(run_rows):
        raise GeneMAECampaignFinalizationError(
            "campaign registry contains duplicate run identities"
        )
    production: dict[int, list[dict[str, Any]]] = {}
    pilots: list[dict[str, Any]] = []
    linked_run_ids: set[str] = set()
    compatible_status = {
        "completed": "completed",
        "failed": "failed",
        "pruned": "pruned",
        "cancelled": "cancelled",
        "stale": "failed",
    }
    for queue in queue_rows:
        run_id = queue.get("run_id")
        if not isinstance(run_id, str) or run_id not in run_by_id:
            raise GeneMAECampaignFinalizationError(
                "terminal campaign queue row lacks its registered run"
            )
        if run_id in linked_run_ids:
            raise GeneMAECampaignFinalizationError(
                f"campaign run {run_id} is linked from multiple queue rows"
            )
        linked_run_ids.add(run_id)
        run = run_by_id[run_id]
        queue_config = _decode_config(
            queue["canonical_config_json"], label=f"queue {queue['job_id']} config"
        )
        run_config = _decode_config(
            run["config_json"], label=f"run {run_id} config"
        )
        queue_role, queue_seed = _config_identity(
            queue_config, label=f"queue {queue['job_id']}"
        )
        run_role, run_seed = _config_identity(
            run_config, label=f"run {run_id}"
        )
        if (
            (queue_role, queue_seed) != (run_role, run_seed)
            or int(run["seed"]) != run_seed
            or int(run["attempt"]) != int(queue["attempt_count"])
            or compatible_status.get(str(queue["status"])) != str(run["status"])
        ):
            raise GeneMAECampaignFinalizationError(
                f"queue/run identity differs for {run_id}"
            )
        record = {
            "job_id": str(queue["job_id"]),
            "run_id": run_id,
            "seed": run_seed,
            "attempt": int(run["attempt"]),
            "status": str(run["status"]),
            "retry_of": run.get("retry_of"),
            "failure_category": run.get("failure_category"),
            "run": run,
        }
        if run_role == "production":
            production.setdefault(run_seed, []).append(record)
        else:
            pilots.append(record)
    if linked_run_ids != set(run_by_id):
        raise GeneMAECampaignFinalizationError(
            "campaign contains a run without its queue provenance"
        )
    if set(production) != set(PRODUCTION_SEEDS):
        raise GeneMAECampaignFinalizationError(
            "registry lacks exact GeneMAE production seeds 0 through 6"
        )
    if not pilots or not any(row["status"] == "completed" for row in pilots):
        raise GeneMAECampaignFinalizationError(
            "registry lacks a completed resource pilot"
        )

    report_slots = _mapping(
        comparison.get("production_by_seed"), "comparison production slots"
    )
    selected_slots: dict[str, dict[str, Any]] = {}
    failed_production: list[str] = []
    for seed in PRODUCTION_SEEDS:
        lineage = sorted(
            production[seed], key=lambda row: (row["attempt"], row["run_id"])
        )
        attempts = [row["attempt"] for row in lineage]
        if attempts != list(range(1, len(lineage) + 1)) or len(lineage) > 2:
            raise GeneMAECampaignFinalizationError(
                f"GeneMAE seed {seed} has an invalid retry lineage"
            )
        completed = [row for row in lineage if row["status"] == "completed"]
        if len(completed) != 1 or completed[0] is not lineage[-1]:
            raise GeneMAECampaignFinalizationError(
                f"GeneMAE seed {seed} lacks one terminal completed attempt"
            )
        previous_run_id: str | None = None
        for row in lineage:
            if row["retry_of"] != previous_run_id:
                raise GeneMAECampaignFinalizationError(
                    f"GeneMAE seed {seed} retry linkage changed"
                )
            previous_run_id = row["run_id"]
        for row in lineage[:-1]:
            if row["status"] not in {"failed", "pruned", "cancelled"}:
                raise GeneMAECampaignFinalizationError(
                    f"GeneMAE seed {seed} has a nonterminal prior attempt"
                )
            failed_production.append(row["run_id"])
        selected = completed[0]
        expected = _mapping(report_slots.get(seed), f"report seed {seed}")
        run = selected["run"]
        if (
            selected["run_id"] != expected.get("run_id")
            or selected["attempt"] != expected.get("attempt")
            or run.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or run.get("primary_metric_name") != PRIMARY_METRIC
            or _finite(
                run.get("primary_metric_value"),
                f"GeneMAE seed {seed} primary metric",
            )
            < 0
            or not run.get("artifact_path")
        ):
            raise GeneMAECampaignFinalizationError(
                f"registry and comparison differ for GeneMAE seed {seed}"
            )
        checkpoint = _verify_checkpoint(
            connection,
            paths=paths,
            run=run,
            expected=expected,
        )
        selected_slots[str(seed)] = {
            "seed": seed,
            "run_id": selected["run_id"],
            "attempt": selected["attempt"],
            "checkpoint": checkpoint,
        }

    pilot_failures = sorted(
        row["run_id"]
        for row in pilots
        if row["status"] in {"failed", "pruned", "cancelled"}
    )
    pilot_completed = sorted(
        row["run_id"] for row in pilots if row["status"] == "completed"
    )
    registry_pilot_statuses = dict(
        sorted((row["run_id"], row["status"]) for row in pilots)
    )
    registry_pilot_failure_statuses = {
        run_id: status
        for run_id, status in registry_pilot_statuses.items()
        if status in {"failed", "pruned", "cancelled"}
    }
    if comparison.get("reported_genemae_pilots") != registry_pilot_statuses:
        raise GeneMAECampaignFinalizationError(
            "report and registry GeneMAE pilot inventories differ"
        )
    if (
        comparison.get("reported_genemae_pilot_failures")
        != registry_pilot_failure_statuses
    ):
        raise GeneMAECampaignFinalizationError(
            "report and registry GeneMAE pilot failure inventories differ"
        )
    return {
        "campaign": campaign,
        "queue_status_counts": _status_counts(queue_rows),
        "run_status_counts": _status_counts(run_rows),
        "queue_rows_checksum": _row_checksum(queue_rows),
        "run_rows_checksum": _row_checksum(run_rows),
        "artifact_rows_checksum": _row_checksum(artifact_rows),
        "checkpoint_rows_checksum": _row_checksum(checkpoint_rows),
        "failure_rows_checksum": _row_checksum(failure_rows),
        "queue_row_count": len(queue_rows),
        "run_row_count": len(run_rows),
        "artifact_row_count": len(artifact_rows),
        "checkpoint_row_count": len(checkpoint_rows),
        "failure_row_count": len(failure_rows),
        "production_slots": selected_slots,
        "failed_production_run_ids": sorted(failed_production),
        "resource_pilot": {
            "completed_run_ids": pilot_completed,
            "failed_run_ids": pilot_failures,
            "failure_count": len(pilot_failures),
            "row_count": len(pilots),
        },
        "active_queue_or_run_states": False,
    }


def _read_registry_snapshot(
    *,
    database_path: Path,
    paths: ProjectPaths,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    if not database_path.is_file():
        raise GeneMAECampaignFinalizationError(
            f"registry database is missing: {database_path}"
        )
    uri = database_path.resolve(strict=False).as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            integrity = [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check").fetchall()
            ]
            if integrity != ["ok"]:
                raise GeneMAECampaignFinalizationError(
                    f"registry integrity check failed: {integrity}"
                )
            return _registry_snapshot(
                connection, paths=paths, comparison=comparison
            )
    except GeneMAECampaignFinalizationError:
        raise
    except sqlite3.Error as exc:
        raise GeneMAECampaignFinalizationError(
            "cannot inspect the authoritative registry"
        ) from exc


def _online_backup(*, database_path: Path, backup_path: Path) -> dict[str, Any]:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists() or backup_path.is_symlink():
        raise FileExistsError(f"database backup already exists: {backup_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_uri = database_path.resolve(strict=False).as_uri() + "?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source:
            with sqlite3.connect(temporary) as destination:
                source.backup(destination)
        with sqlite3.connect(temporary) as check:
            integrity = [
                str(row[0])
                for row in check.execute("PRAGMA integrity_check").fetchall()
            ]
        if integrity != ["ok"]:
            raise GeneMAECampaignFinalizationError(
                f"database backup integrity check failed: {integrity}"
            )
        os.link(temporary, backup_path)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": backup_path.as_posix(),
        "sha256": _sha256_file(backup_path),
        "size_bytes": backup_path.stat().st_size,
        "integrity_check": "ok",
    }


@contextmanager
def _application_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"append-only receipt already exists: {path}")
    serialized = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _timestamp_slug(timestamp: str) -> str:
    return timestamp.replace("-", "").replace(":", "").replace(".", "")


def _immutable_registry_evidence(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: snapshot[key]
        for key in (
            "queue_status_counts",
            "run_status_counts",
            "queue_rows_checksum",
            "run_rows_checksum",
            "artifact_rows_checksum",
            "checkpoint_rows_checksum",
            "failure_rows_checksum",
            "queue_row_count",
            "run_row_count",
            "artifact_row_count",
            "checkpoint_row_count",
            "failure_row_count",
            "production_slots",
            "failed_production_run_ids",
            "resource_pilot",
            "active_queue_or_run_states",
        )
    }


def finalize_campaign(
    *,
    paths: ProjectPaths,
    database_path: Path,
    comparison_manifest_path: Path,
    expected_manifest_sha256: str,
    receipt_output_path: Path,
    backup_path: Path | None = None,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    """Finalize the exact MyJJu campaign and return its signed receipt."""

    database = database_path.resolve(strict=False)
    canonical_database = (
        paths.state_root / "tracking" / "bagm.sqlite3"
    ).resolve(strict=False)
    if database != canonical_database:
        raise GeneMAECampaignFinalizationError(
            "only the authoritative BAGM registry may be finalized"
        )
    receipt_path = _require_child(
        receipt_output_path,
        _reconciliation_directory(paths),
        label="finalization receipt",
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(
            f"append-only receipt already exists: {receipt_path}"
        )
    timestamp = finalized_at or utc_now()
    lock_path = (
        paths.state_root / "tracking" / f".{CAMPAIGN_ID}.finalization.lock"
    )
    with _application_lock(lock_path):
        comparison = _validate_report(
            paths=paths,
            manifest_path=comparison_manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        preflight = _read_registry_snapshot(
            database_path=database,
            paths=paths,
            comparison=comparison,
        )
        if backup_path is None:
            backup_path = (
                paths.state_root
                / "tracking"
                / "backups"
                / (
                    "bagm.before_myjju_genemae_campaign_finalization."
                    f"{_timestamp_slug(timestamp)}."
                    f"{comparison['manifest_sha256'][:12]}.sqlite3"
                )
            )
        backup = _require_child(
            backup_path,
            paths.state_root / "tracking" / "backups",
            label="registry backup",
        )
        backup_evidence = _online_backup(
            database_path=database, backup_path=backup
        )
        backup_snapshot = _read_registry_snapshot(
            database_path=backup,
            paths=paths,
            comparison=comparison,
        )
        if backup_snapshot != preflight:
            raise GeneMAECampaignFinalizationError(
                "campaign registry state changed while the online backup was made"
            )

        registry = Registry(database, initialize=False)
        changed_count = 0
        with registry.transaction(immediate=True) as connection:
            current = _registry_snapshot(
                connection, paths=paths, comparison=comparison
            )
            if current != backup_snapshot:
                raise GeneMAECampaignFinalizationError(
                    "campaign registry state changed before BEGIN IMMEDIATE"
                )
            campaign = current["campaign"]
            config_json = str(campaign["config_json"])
            if campaign["status"] == "planned":
                cursor = connection.execute(
                    """
                    UPDATE campaigns
                    SET status = 'complete', updated_at = ?
                    WHERE campaign_id = ?
                      AND status = 'planned'
                      AND config_json = ?
                      AND updated_at = ?
                    """,
                    (
                        timestamp,
                        CAMPAIGN_ID,
                        config_json,
                        campaign["updated_at"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise GeneMAECampaignFinalizationError(
                        "campaign planned-to-complete CAS changed an invalid "
                        f"number of rows: {cursor.rowcount}"
                    )
                changed_count = 1
            post = _registry_snapshot(
                connection, paths=paths, comparison=comparison
            )
            if (
                post["campaign"]["status"] != "complete"
                or post["campaign"]["config_json"] != config_json
                or post["campaign"]["created_at"] != campaign["created_at"]
                or _immutable_registry_evidence(post)
                != _immutable_registry_evidence(current)
            ):
                raise GeneMAECampaignFinalizationError(
                    "campaign finalization changed prohibited registry state"
                )
            if changed_count == 0 and post != current:
                raise GeneMAECampaignFinalizationError(
                    "already-complete idempotent verification changed registry state"
                )

        config_sha = hashlib.sha256(
            str(backup_snapshot["campaign"]["config_json"]).encode("utf-8")
        ).hexdigest()
        receipt_payload = {
            "schema_version": 1,
            "receipt_kind": RECEIPT_KIND,
            "campaign_id": CAMPAIGN_ID,
            "finalized_at": timestamp,
            "comparison": {
                key: value
                for key, value in comparison.items()
                if key != "production_by_seed"
            },
            "database": {
                "path": database.as_posix(),
                "backup": backup_evidence,
                "integrity_check_before": "ok",
            },
            "registry_evidence": _immutable_registry_evidence(backup_snapshot),
            "transaction": {
                "mode": "BEGIN IMMEDIATE",
                "compare_and_swap": "planned->complete",
                "pre_status": backup_snapshot["campaign"]["status"],
                "post_status": "complete",
                "changed_count": changed_count,
                "idempotent_already_complete": changed_count == 0,
                "config_json_sha256_before": config_sha,
                "config_json_sha256_after": config_sha,
                "config_json_unchanged": True,
                "active_queue_or_run_states": False,
                "run_or_queue_rows_modified": False,
                "failed_pilot_rows_preserved": True,
                "failed_production_rows_preserved": True,
            },
            "checksum_algorithm": "canonical-json-sha256",
        }
        receipt = {
            **receipt_payload,
            "checksum": canonical_sha256(receipt_payload),
        }
        _write_new_json(receipt_path, receipt)
        return receipt


def _parser(paths: ProjectPaths) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    parser.add_argument(
        "--comparison-manifest",
        required=True,
        type=Path,
        help="Exact canonical comparison/manifest.json to approve.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="Exact explicitly approved SHA-256 of the comparison manifest.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="Append-only finalization receipt path.",
    )
    parser.add_argument(
        "--backup-output",
        type=Path,
        help="Append-only online SQLite backup path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    paths = current_paths()
    arguments = _parser(paths).parse_args(argv)
    timestamp = utc_now()
    receipt_output = arguments.receipt_output or (
        _reconciliation_directory(paths)
        / f"campaign_finalization_{_timestamp_slug(timestamp)}.json"
    )
    receipt = finalize_campaign(
        paths=paths,
        database_path=arguments.database,
        comparison_manifest_path=arguments.comparison_manifest,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        receipt_output_path=receipt_output,
        backup_path=arguments.backup_output,
        finalized_at=timestamp,
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "status": receipt["transaction"]["post_status"],
                "outcome": receipt["comparison"]["outcome"],
                "changed_count": receipt["transaction"]["changed_count"],
                "receipt": receipt_output.resolve(strict=False).as_posix(),
                "checksum": receipt["checksum"],
                "backup": receipt["database"]["backup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

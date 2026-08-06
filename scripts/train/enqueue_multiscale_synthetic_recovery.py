#!/usr/bin/env python3
"""Validate and idempotently enqueue the single locked Stage-0 recovery run.

This command registers one queue job; it does not start a worker or train a
model. Use ``--check-only`` for a mutation-free preflight.
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
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.diagnostics.run_multiscale_synthetic_recovery import (  # noqa: E402
    validate_synthetic_runner_config,
)
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.identifiers import canonical_sha256, scientific_id  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_multiscale_hurdle_count_pilot"
CONFIG_REFERENCE = (
    "experiments/campaigns/"
    f"{CAMPAIGN_ID}/stage0_synthetic_recovery_config.yaml"
)
CONFIG_FILE_SHA256 = (
    "61a2fcf80f9f7e0ae4bae58448682fe6c286c12689f1fda526bbd9268eba7e00"
)
CONFIG_CANONICAL_SHA256 = (
    "3088e25b5aef1462de7723dc84059933619c5ad73a23eb28c5c483ed8191a519"
)
PRERUN_REFERENCE = (
    "experiments/campaigns/"
    f"{CAMPAIGN_ID}/stage0_prerun_negative_diagnostic_v1.json"
)
PRERUN_FILE_SHA256 = (
    "87d21ba39de9972d787ebec185d4de433743bfc623349f57287275060bc22c72"
)
PRERUN_PAYLOAD_CHECKSUM = (
    "4b06b1d3d08afe560ed727cb1e0da53254b9f358626fda067b2de6cc224dade3"
)
MAXIMUM_ATTEMPTS = 1
PRIORITY = 100
SAFE_GPU_IDS = frozenset({"0", "1", "2", "3", "5", "6", "7"})
DISK_USED_DECIMAL_GB_HARD_STOP = 55.0
RECEIPT_KIND = "multiscale_stage0_enqueue_v1"


class Stage0EnqueueError(RuntimeError):
    """Raised when the locked Stage-0 registration contract is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage0EnqueueError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage0EnqueueError(
            f"pre-run diagnostic contains non-finite constant {value!r}"
        )

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage0EnqueueError(
                    f"pre-run diagnostic repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage0EnqueueError(
            "pre-run diagnostic is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise Stage0EnqueueError("pre-run diagnostic root must be a mapping")
    return value


def _resolve_locked_reference(reference: str, *, file_sha256: str) -> Path:
    path = (_PROJECT_ROOT / reference).resolve()
    try:
        path.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Stage0EnqueueError("locked reference escapes the project") from exc
    if not path.is_file() or _sha256_file(path) != file_sha256:
        raise Stage0EnqueueError(
            f"locked file is absent or changed: {reference}"
        )
    return path


def _validate_prerun_receipt(config: Mapping[str, Any]) -> None:
    path = _resolve_locked_reference(
        PRERUN_REFERENCE,
        file_sha256=PRERUN_FILE_SHA256,
    )
    receipt = _strict_json(path)
    checksum = receipt.pop("checksum", None)
    if (
        checksum != PRERUN_PAYLOAD_CHECKSUM
        or canonical_sha256(receipt) != PRERUN_PAYLOAD_CHECKSUM
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("receipt_kind")
        != "multiscale_stage0_prerun_negative_diagnostic_v1"
        or _mapping(receipt.get("gate_observations"), "gate observations").get(
            "overall_outcome"
        )
        != "negative"
        or _mapping(receipt.get("decision"), "pre-run decision").get(
            "no_post_outcome_tuning"
        )
        is not True
    ):
        raise Stage0EnqueueError(
            "pre-run negative diagnostic checksum or decision changed"
        )
    metadata = _mapping(config.get("metadata"), "config metadata")
    binding = _mapping(
        metadata.get("pre_run_negative_diagnostic"),
        "config pre-run diagnostic binding",
    )
    if dict(binding) != {
        "reference": PRERUN_REFERENCE,
        "file_sha256": PRERUN_FILE_SHA256,
        "payload_checksum": PRERUN_PAYLOAD_CHECKSUM,
        "registered_stage0_result_inferred": False,
    }:
        raise Stage0EnqueueError(
            "config does not bind the exact negative pre-run diagnostic"
        )


def _expected_runner_command(config: Mapping[str, Any]) -> list[str]:
    expected = [
        sys.executable,
        str(
            _PROJECT_ROOT
            / "scripts/diagnostics/run_multiscale_synthetic_recovery.py"
        ),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]
    observed = command_for_config(config)
    if observed != expected:
        raise Stage0EnqueueError(
            "derived queue command is not the locked Stage-0 runner"
        )
    return expected


def load_locked_config(path: Path | None = None) -> dict[str, Any]:
    """Load and verify the immutable Stage-0 resolved configuration."""

    expected_path = _resolve_locked_reference(
        CONFIG_REFERENCE,
        file_sha256=CONFIG_FILE_SHA256,
    )
    selected = expected_path if path is None else path.resolve()
    if selected != expected_path:
        raise Stage0EnqueueError(
            "Stage-0 config must be the authoritative campaign-local file"
        )
    config = dict(load_yaml_mapping(selected))
    validate_experiment_config(config)
    validate_synthetic_runner_config(config)
    if canonical_sha256(config) != CONFIG_CANONICAL_SHA256:
        raise Stage0EnqueueError("canonical Stage-0 configuration changed")
    metadata = _mapping(config.get("metadata"), "config metadata")
    launcher = _mapping(config.get("launcher"), "config launcher")
    experiment = _mapping(config.get("experiment"), "config experiment")
    if (
        metadata.get("execution_role") != "stage0_synthetic_recovery"
        or metadata.get("no_post_outcome_tuning") is not True
        or experiment.get("stage") != 0
        or experiment.get("conclusion_eligible") is not False
        or str(launcher.get("requested_gpu")) not in SAFE_GPU_IDS
        or int(launcher.get("requested_gpu_count", -1)) != 1
        or launcher.get("disk_safety_max_used_decimal_gb")
        != DISK_USED_DECIMAL_GB_HARD_STOP
    ):
        raise Stage0EnqueueError("Stage-0 execution or resource contract changed")
    _validate_prerun_receipt(config)
    _expected_runner_command(config)
    return config


def _assert_registered_inputs(
    registry: Registry,
    config: Mapping[str, Any],
) -> None:
    campaign = registry.get_campaign(CAMPAIGN_ID)
    if campaign is None:
        raise Stage0EnqueueError("campaign is not registered")
    dataset = _mapping(config.get("dataset"), "config dataset")
    record = registry.get_dataset(
        str(dataset["dataset_id"]),
        str(dataset["version"]),
    )
    if record is None:
        raise Stage0EnqueueError("configured dataset is not registered")
    allowed_fingerprints = {
        str(record.get("raw_fingerprint")),
        str(record.get("processed_fingerprint")),
    }
    if str(dataset.get("dataset_fingerprint")) not in allowed_fingerprints:
        raise Stage0EnqueueError(
            "configured dataset fingerprint differs from the registry"
        )
    split = registry.get_split(str(dataset["split_id"]))
    if (
        split is None
        or split.get("dataset_id") != dataset.get("dataset_id")
        or split.get("dataset_version") != dataset.get("version")
        or split.get("fingerprint") != dataset.get("split_fingerprint")
    ):
        raise Stage0EnqueueError(
            "configured split identity differs from the registry"
        )
    prepared = (_PROJECT_ROOT / str(dataset["prepared_artifact_reference"])).resolve()
    if not prepared.is_dir():
        raise Stage0EnqueueError("prepared geometry artifact is unavailable")


def _disk_used_decimal_gb() -> float:
    value = float(shutil.disk_usage(_PROJECT_ROOT).used) / 1_000_000_000.0
    if (
        not math.isfinite(value)
        or value >= DISK_USED_DECIMAL_GB_HARD_STOP
    ):
        raise Stage0EnqueueError(
            "filesystem used space reached the 55.0 GB hard stop"
        )
    return value


def _existing_root_jobs(
    registry: Registry,
) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM queue_jobs
            WHERE campaign_id = ? AND retry_of IS NULL
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    return [dict(row) for row in rows]


def _validate_existing_job(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    command: Sequence[str],
    requested_gpu: str,
) -> None:
    try:
        existing_config = json.loads(str(row["canonical_config_json"]))
        existing_command = json.loads(str(row["command_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Stage0EnqueueError("existing queue job contains invalid JSON") from exc
    if (
        canonical_sha256(existing_config) != canonical_sha256(config)
        or existing_command != list(command)
        or int(row.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(row.get("attempt_count", -1)) != 1
        or str(row.get("requested_gpu")) != requested_gpu
        or int(row.get("priority", -1)) != PRIORITY
        or str(row.get("experiment_config_reference")) != CONFIG_REFERENCE
    ):
        raise Stage0EnqueueError(
            "existing Stage-0 job differs from the locked registration"
        )


@contextmanager
def _campaign_lock(database_path: Path) -> Iterator[None]:
    lock_path = (
        database_path.parent
        / f".{database_path.name}.{CAMPAIGN_ID}.stage0.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def enqueue_stage0(
    *,
    database_path: Path,
    receipt_path: Path,
    config_path: Path | None = None,
    check_only: bool = False,
) -> dict[str, Any]:
    """Preflight and optionally enqueue exactly one immutable Stage-0 job."""

    with _campaign_lock(database_path):
        config = load_locked_config(config_path)
        command = _expected_runner_command(config)
        requested_gpu = str(
            _mapping(config.get("launcher"), "config launcher")["requested_gpu"]
        )
        disk_used = _disk_used_decimal_gb()
        registry = Registry(database_path)
        _assert_registered_inputs(registry, config)
        matching: dict[str, Any] | None = None
        for row in _existing_root_jobs(registry):
            try:
                candidate = json.loads(str(row["canonical_config_json"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise Stage0EnqueueError(
                    "existing campaign queue job contains invalid config JSON"
                ) from exc
            metadata = (
                candidate.get("metadata", {})
                if isinstance(candidate, Mapping)
                else {}
            )
            if (
                isinstance(metadata, Mapping)
                and metadata.get("execution_role")
                == "stage0_synthetic_recovery"
            ):
                if matching is not None:
                    raise Stage0EnqueueError(
                        "campaign contains more than one Stage-0 root job"
                    )
                _validate_existing_job(
                    row,
                    config=config,
                    command=command,
                    requested_gpu=requested_gpu,
                )
                matching = row
        mutation_performed = False
        if not check_only and matching is None:
            registry.register_variant(
                scientific_id(config),
                campaign_id=CAMPAIGN_ID,
                configuration=config,
            )
            matching = registry.enqueue(
                campaign_id=CAMPAIGN_ID,
                configuration=config,
                command=command,
                experiment_config_reference=CONFIG_REFERENCE,
                priority=PRIORITY,
                maximum_attempts=MAXIMUM_ATTEMPTS,
                requested_gpu=requested_gpu,
            )
            mutation_performed = True
        payload: dict[str, Any] = {
            "schema_version": 1,
            "receipt_kind": RECEIPT_KIND,
            "campaign_id": CAMPAIGN_ID,
            "check_only": bool(check_only),
            "queue_mutation_performed": mutation_performed,
            "complete": matching is not None or check_only,
            "job_id": None if matching is None else str(matching["job_id"]),
            "config_reference": CONFIG_REFERENCE,
            "config_file_sha256": CONFIG_FILE_SHA256,
            "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
            "pre_run_negative_diagnostic_reference": PRERUN_REFERENCE,
            "pre_run_negative_diagnostic_file_sha256": PRERUN_FILE_SHA256,
            "pre_run_negative_diagnostic_payload_checksum": (
                PRERUN_PAYLOAD_CHECKSUM
            ),
            "no_post_outcome_tuning": True,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "requested_gpu": requested_gpu,
            "priority": PRIORITY,
            "disk_used_decimal_gb_at_preflight": disk_used,
            "disk_used_decimal_gb_hard_stop": (
                DISK_USED_DECIMAL_GB_HARD_STOP
            ),
            "registered_stage0_result_inferred_from_prerun": False,
            "training_started": False,
        }
        payload["checksum"] = canonical_sha256(payload)
        if not check_only:
            _atomic_json(receipt_path, payload)
        return payload


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / CONFIG_REFERENCE,
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=(
            paths.scratch_root
            / "locked_campaigns"
            / CAMPAIGN_ID
            / "stage0_enqueue_receipt.json"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without registering a variant or queue job.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = enqueue_stage0(
        database_path=args.database.resolve(),
        receipt_path=args.receipt.resolve(),
        config_path=args.config.resolve(),
        check_only=bool(args.check_only),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "check_only": receipt["check_only"],
                "complete": receipt["complete"],
                "job_id": receipt["job_id"],
                "queue_mutation_performed": receipt[
                    "queue_mutation_performed"
                ],
                "receipt": None if args.check_only else str(args.receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

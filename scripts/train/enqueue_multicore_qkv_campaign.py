#!/usr/bin/env python3
"""Idempotently enqueue the locked 30-job multi-core QKV campaign.

The materialization receipt is authoritative.  A canonical configuration may
map to at most one queue job in this campaign.  The script writes an atomic
receipt after every enqueue, so an interrupted invocation resumes without
creating duplicate fits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping


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
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("k1000", "k5000", "matched_self")
MAXIMUM_ATTEMPTS = 2
_PRIORITY = {"k5000": 30, "k1000": 20, "matched_self": 10}
_EXPECTED_K = {"k1000": 1000, "k5000": 5000, "matched_self": 5000}


class LockedEnqueueError(RuntimeError):
    """Raised when an enqueue plan is incomplete, changed, or duplicated."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedEnqueueError(f"{label} must be a mapping.")
    return value


def _load_materialization(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    materialization = dict(_mapping(value, "materialization"))
    checksum = str(materialization.get("checksum", ""))
    payload = {
        key: entry
        for key, entry in materialization.items()
        if key != "checksum"
    }
    if checksum != canonical_sha256(payload):
        raise LockedEnqueueError(
            "Campaign materialization checksum does not verify."
        )
    if (
        materialization.get("schema_version") != 1
        or
        materialization.get("campaign_id") != CAMPAIGN_ID
        or materialization.get("job_count") != 30
    ):
        raise LockedEnqueueError(
            "Materialization is not the locked 30-job campaign."
        )
    jobs = materialization.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 30:
        raise LockedEnqueueError(
            "Materialization must contain exactly 30 jobs."
        )
    observed = {
        (str(job.get("alias")), str(job.get("arm")))
        for job in jobs
        if isinstance(job, Mapping)
    }
    expected = {(alias, arm) for alias in ALIASES for arm in ARMS}
    if observed != expected:
        raise LockedEnqueueError(
            "Materialization does not contain one job per alias and arm."
        )
    return materialization


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
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
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


@contextmanager
def _campaign_enqueue_lock(database_path: Path) -> Iterator[None]:
    """Serialize check-and-enqueue for this campaign and SQLite registry."""

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


def _existing_jobs_by_digest(
    registry: Registry,
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    try:
        with registry.connect() as connection:
            raw_rows = connection.execute(
                """
                SELECT * FROM queue_jobs
                WHERE campaign_id = ? AND retry_of IS NULL
                ORDER BY created_at, job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
    except Exception as exc:
        raise LockedEnqueueError(
            "Cannot inspect all root queue jobs for the locked campaign."
        ) from exc
    for raw_row in raw_rows:
        row = dict(raw_row)
        try:
            row["canonical_config"] = json.loads(
                str(row.pop("canonical_config_json"))
            )
            row["command"] = json.loads(str(row.pop("command_json")))
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedEnqueueError(
                f"Queue job {row.get('job_id')} has invalid JSON."
            ) from exc
        config = _mapping(
            row.get("canonical_config"),
            f"queue job {row.get('job_id')} config",
        )
        digest = canonical_sha256(config)
        result.setdefault(digest, []).append(row)
    duplicates = {
        digest: rows for digest, rows in result.items() if len(rows) > 1
    }
    if duplicates:
        raise LockedEnqueueError(
            "Campaign already contains duplicate canonical configurations."
        )
    return result


def _validated_job(
    raw_job: Mapping[str, Any],
    *,
    registry: Registry,
) -> tuple[dict[str, Any], Path, str, int]:
    alias = str(raw_job.get("alias"))
    arm = str(raw_job.get("arm"))
    try:
        gpu = int(raw_job.get("requested_gpu"))
    except (TypeError, ValueError) as exc:
        raise LockedEnqueueError(
            "A planned GPU must be an integer from 0 through 7."
        ) from exc
    if alias not in ALIASES or arm not in ARMS or not 0 <= gpu < 8:
        raise LockedEnqueueError("A planned alias, arm, or GPU is invalid.")
    reference = Path(str(raw_job.get("config")))
    if reference.is_absolute():
        raise LockedEnqueueError(
            "Production config references must be project-relative."
        )
    config_path = (_PROJECT_ROOT / reference).resolve()
    try:
        config_path.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise LockedEnqueueError(
            "Production config escapes the project root."
        ) from exc
    config = load_yaml_mapping(config_path)
    validate_experiment_config(config)
    campaign = _mapping(config.get("campaign"), "config campaign")
    dataset = _mapping(config.get("dataset"), "config dataset")
    launcher = _mapping(config.get("launcher"), "config launcher")
    experiment = _mapping(config.get("experiment"), "config experiment")
    graph = _mapping(config.get("graph"), "config graph")
    model = _mapping(config.get("model"), "config model")
    trainer = _mapping(config.get("trainer"), "config trainer")
    evaluation = _mapping(config.get("evaluation"), "config evaluation")
    is_graph = arm != "matched_self"
    expected_family = (
        "edge_aware_qkv_graph_transformer"
        if is_graph
        else "qkv_parameter_matched_self_control"
    )
    expected_variant = (
        f"{alias.lower().replace('-', '')}_qkv_{arm}_full_core"
    )
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or launcher.get("requested_gpu") != str(gpu)
        or experiment.get("variant_label") != expected_variant
        or int(graph.get("k", -1)) != _EXPECTED_K[arm]
        or int(graph.get("neighbor_k", -1)) != _EXPECTED_K[arm]
        or model.get("family") != expected_family
        or model.get("uses_graph_inputs") is not is_graph
        or model.get("uses_edge_inputs") is not is_graph
        or int(config.get("seed", -1)) != 0
        or int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) != 1
        or int(trainer.get("max_epochs", -1)) != 300
        or evaluation.get("protocol") != "held_in_full_core_fixed_budget"
        or evaluation.get("canonical_prediction_split") != "fit"
        or evaluation.get("generalization_estimate") is not False
    ):
        raise LockedEnqueueError(
            "Config campaign, alias, arm, model, graph, budget, or GPU "
            "differs from its locked plan."
        )
    digest = canonical_sha256(config)
    if str(raw_job.get("config_sha256", "")) != digest:
        raise LockedEnqueueError(
            "Production config differs from its materialization digest."
        )
    registered = registry.get_dataset(
        str(dataset.get("dataset_id")),
        str(dataset.get("version")),
    )
    split = registry.get_split(str(dataset.get("split_id")))
    if registered is None or split is None:
        raise LockedEnqueueError(
            "A production dataset or split is not registered."
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
        raise LockedEnqueueError(
            "Registered dataset or split identity differs from the config."
        )
    prepared_reference = Path(
        str(dataset.get("prepared_artifact_reference"))
    )
    if prepared_reference.is_absolute():
        raise LockedEnqueueError(
            "Prepared artifact references must be project-relative."
        )
    prepared = (_PROJECT_ROOT / prepared_reference).resolve()
    try:
        prepared.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise LockedEnqueueError(
            "Prepared artifact reference escapes the project root."
        ) from exc
    if not prepared.is_dir():
        raise LockedEnqueueError(
            "A production prepared artifact is unavailable."
        )
    registered_path = registered.get("protected_source_path")
    if registered_path is not None:
        registered_reference = Path(str(registered_path))
        registered_prepared = (
            registered_reference.resolve()
            if registered_reference.is_absolute()
            else (_PROJECT_ROOT / registered_reference).resolve()
        )
        if registered_prepared != prepared:
            raise LockedEnqueueError(
                "Registered protected path differs from the prepared artifact."
            )
    return config, reference, digest, gpu


def _validate_existing_job(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    reference: Path,
    arm: str,
    gpu: int,
) -> None:
    expected_command = command_for_config(config)
    if (
        int(row.get("maximum_attempts")) != MAXIMUM_ATTEMPTS
        or int(row.get("attempt_count")) != 1
        or row.get("retry_of") is not None
        or str(row.get("requested_gpu")) != str(gpu)
        or int(row.get("priority")) != _PRIORITY[arm]
        or str(row.get("experiment_config_reference")) != str(reference)
        or row.get("command") != expected_command
    ):
        raise LockedEnqueueError(
            "Existing root queue job has the wrong command, config "
            "reference, priority, GPU, or attempt budget."
        )


def enqueue_locked_campaign(
    *,
    materialization_path: Path,
    receipt_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    with _campaign_enqueue_lock(database_path):
        materialization = _load_materialization(materialization_path)
        registry = Registry(database_path)
        if registry.get_campaign(CAMPAIGN_ID) is None:
            raise LockedEnqueueError("The locked campaign is not registered.")
        existing = _existing_jobs_by_digest(registry)

        # Complete every read-only validation before the first registry write.
        validated: list[
            tuple[Mapping[str, Any], dict[str, Any], Path, str, int]
        ] = []
        planned_digests: set[str] = set()
        for raw in materialization["jobs"]:
            raw_job = _mapping(raw, "planned job")
            config, reference, digest, gpu = _validated_job(
                raw_job,
                registry=registry,
            )
            if digest in planned_digests:
                raise LockedEnqueueError(
                    "Two planned jobs have the same canonical configuration."
                )
            planned_digests.add(digest)
            matched = existing.get(digest, [])
            if matched:
                _validate_existing_job(
                    matched[0],
                    config=config,
                    reference=reference,
                    arm=str(raw_job["arm"]),
                    gpu=gpu,
                )
            validated.append(
                (raw_job, config, reference, digest, gpu)
            )
        unexpected_existing = set(existing).difference(planned_digests)
        if unexpected_existing:
            raise LockedEnqueueError(
                "Campaign contains an unexpected root queue job outside "
                "the locked 30-config materialization."
            )

        receipt_rows: list[dict[str, Any]] = []
        partial: dict[str, Any] = {}
        for raw_job, config, reference, digest, gpu in validated:
            registry.register_variant(
                scientific_id(config),
                campaign_id=CAMPAIGN_ID,
                configuration=config,
            )
            matched = existing.get(digest, [])
            if matched:
                row = matched[0]
                disposition = "reused"
            else:
                row = registry.enqueue(
                    campaign_id=CAMPAIGN_ID,
                    configuration=config,
                    command=command_for_config(config),
                    experiment_config_reference=reference,
                    priority=_PRIORITY[str(raw_job["arm"])],
                    maximum_attempts=MAXIMUM_ATTEMPTS,
                    requested_gpu=str(gpu),
                )
                existing[digest] = [row]
                disposition = "enqueued"

            receipt_rows.append(
                {
                    "alias": str(raw_job["alias"]),
                    "arm": str(raw_job["arm"]),
                    "config_sha256": digest,
                    "job_id": str(row["job_id"]),
                    "requested_gpu": gpu,
                    "maximum_attempts": MAXIMUM_ATTEMPTS,
                    "disposition": disposition,
                }
            )
            partial = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "materialization_checksum": materialization["checksum"],
                "complete": len(receipt_rows) == 30,
                "jobs": receipt_rows,
            }
            partial["checksum"] = canonical_sha256(partial)
            _atomic_receipt(receipt_path, partial)

        return partial


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    state_dir = (
        paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=state_dir / "campaign_materialization.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=state_dir / "enqueue_receipt.json",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = enqueue_locked_campaign(
        materialization_path=arguments.materialization.resolve(),
        receipt_path=arguments.receipt.resolve(),
        database_path=arguments.database.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "complete": result["complete"],
                "job_count": len(result["jobs"]),
                "receipt": str(arguments.receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

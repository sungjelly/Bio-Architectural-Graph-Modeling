#!/usr/bin/env python3
"""Idempotently enqueue the exact self-hurdle resource or science stage."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train.run_self_hurdle_capacity import (  # noqa: E402
    CAMPAIGN_ID,
    DISK_USED_DECIMAL_GB_MAX,
    RECEIPT_KIND,
    RESOURCE_GATE_KIND,
    _validate_config,
    _verify_science_gate,
)
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import (  # noqa: E402
    canonical_sha256,
    scientific_id,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


SAFE_GPUS = frozenset({0, 1, 2, 3, 5, 6, 7})
ALIASES = frozenset({"ANC-03", "ANC-05"})
MAXIMUM_ATTEMPTS = 2
PRIORITY = {"resource": 60, "science": 50}


class SelfHurdleEnqueueError(RuntimeError):
    pass


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfHurdleEnqueueError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfHurdleEnqueueError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelfHurdleEnqueueError(
                    f"{label} contains non-finite constant {token}"
                )
            ),
            object_pairs_hook=unique,
        )
    except SelfHurdleEnqueueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelfHurdleEnqueueError(f"{label} is invalid") from exc
    return dict(_mapping(value, label))


def _materialization(path: Path) -> dict[str, Any]:
    value = _strict_json(path, "locked materialization")
    checksum = value.get("checksum")
    core = dict(value)
    core.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or canonical_sha256(core) != checksum
        or value.get("receipt_kind") != RECEIPT_KIND
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("training_performed") is not False
        or value.get("registry_mutation_performed") is not False
        or value.get("queue_mutation_performed") is not False
        or _mapping(value.get("graph_contract"), "graph contract").get(
            "construction_performed"
        )
        is not False
        or _mapping(value.get("counts"), "counts")
        != {"cores": 2, "resource_configs": 2, "science_configs": 2}
    ):
        raise SelfHurdleEnqueueError(
            "locked materialization identity is invalid"
        )
    return value


def _jobs(
    materialization: Mapping[str, Any],
    stage: str,
) -> list[Mapping[str, Any]]:
    values = materialization.get(f"{stage}_jobs")
    if not isinstance(values, list):
        raise SelfHurdleEnqueueError(f"{stage} job list is missing")
    jobs = [_mapping(item, f"{stage} job") for item in values]
    if (
        len(jobs) != 2
        or {str(item.get("alias")) for item in jobs} != ALIASES
        or any(item.get("stage") != stage for item in jobs)
    ):
        raise SelfHurdleEnqueueError(
            f"{stage} job matrix is not exactly the two frozen cores"
        )
    return jobs


def _config(
    item: Mapping[str, Any],
    *,
    stage: str,
    materialization: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str, int]:
    reference = item.get("config")
    if not isinstance(reference, str) or not reference.strip():
        raise SelfHurdleEnqueueError("config reference is missing")
    path = (PROJECT_ROOT / reference).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SelfHurdleEnqueueError(
            "config reference escapes project root"
        ) from exc
    if not path.is_file() or sha256_file(path) != item.get("file_sha256"):
        raise SelfHurdleEnqueueError("locked config file checksum drifted")
    config = dict(load_yaml_mapping(path))
    validate_experiment_config(config)
    digest = canonical_sha256(config)
    if digest != item.get("config_sha256"):
        raise SelfHurdleEnqueueError("canonical config checksum drifted")
    contract = _validate_config(config)
    alias = str(item.get("alias"))
    if (
        contract.alias != alias
        or contract.resource_pilot != (stage == "resource")
    ):
        raise SelfHurdleEnqueueError("config stage or alias changed")
    metadata = _mapping(config.get("metadata"), "metadata")
    receipt_path = (
        PROJECT_ROOT
        / str(metadata.get("locked_config_materialization_receipt"))
    ).resolve()
    if receipt_path.is_file():
        referenced = _materialization(receipt_path)
        if referenced.get("checksum") != materialization.get("checksum"):
            raise SelfHurdleEnqueueError(
                "config references another materialization"
            )
    else:
        raise SelfHurdleEnqueueError(
            "config materialization reference is missing"
        )
    gpu = item.get("requested_gpu")
    if (
        isinstance(gpu, bool)
        or not isinstance(gpu, int)
        or gpu not in SAFE_GPUS
        or str(_mapping(config.get("launcher"), "launcher").get(
            "requested_gpu"
        ))
        != str(gpu)
    ):
        raise SelfHurdleEnqueueError("config selects an unsafe GPU")
    command = command_for_config(config)
    if not any(
        part.endswith("run_self_hurdle_capacity.py") for part in command
    ):
        raise SelfHurdleEnqueueError("config routes to the wrong runner")
    return config, Path(reference), digest, gpu


def _registered_inputs(config: Mapping[str, Any], registry: Registry) -> None:
    dataset = _mapping(config.get("dataset"), "dataset")
    record = registry.get_dataset(
        str(dataset.get("dataset_id")),
        str(dataset.get("version")),
    )
    split = registry.get_split(str(dataset.get("split_id")))
    if (
        record is None
        or split is None
        or str(record.get("processed_fingerprint"))
        != str(dataset.get("dataset_fingerprint"))
        or str(split.get("fingerprint"))
        != str(dataset.get("split_fingerprint"))
    ):
        raise SelfHurdleEnqueueError(
            "dataset or split is not registered with the locked checksum"
        )


def _existing(registry: Registry) -> dict[str, Mapping[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM queue_jobs
            WHERE campaign_id = ? AND retry_of IS NULL
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        try:
            config = json.loads(str(row["canonical_config_json"]))
            row["command"] = json.loads(str(row["command_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SelfHurdleEnqueueError(
                "existing campaign job has invalid JSON"
            ) from exc
        digest = canonical_sha256(_mapping(config, "existing config"))
        if digest in result:
            raise SelfHurdleEnqueueError(
                "duplicate existing root configuration"
            )
        result[digest] = row
    return result


def _disk_preflight() -> None:
    usage = shutil.disk_usage(PROJECT_ROOT)
    used = (usage.total - usage.free) / 1e9
    if used >= DISK_USED_DECIMAL_GB_MAX:
        raise SelfHurdleEnqueueError(
            f"filesystem uses {used:.3f} GB; hard stop is 55 GB"
        )


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


@contextmanager
def _lock(database: Path) -> Iterator[None]:
    path = database.parent / f".{database.name}.{CAMPAIGN_ID}.enqueue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def enqueue_stage(
    *,
    stage: str,
    materialization_path: Path,
    receipt_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    if stage not in {"resource", "science"}:
        raise SelfHurdleEnqueueError("stage must be resource or science")
    with _lock(database_path):
        _disk_preflight()
        materialization = _materialization(materialization_path)
        if stage == "science":
            contract = _validate_config(
                _config(
                    _jobs(materialization, "science")[0],
                    stage="science",
                    materialization=materialization,
                )[0]
            )
            _verify_science_gate(
                PROJECT_ROOT, materialization, contract
            )
        registry = Registry(database_path)
        if registry.get_campaign(CAMPAIGN_ID) is None:
            raise SelfHurdleEnqueueError("campaign is not registered")
        existing = _existing(registry)
        planned: dict[str, tuple[dict[str, Any], Path, int, str]] = {}
        stage_rows: list[tuple[dict[str, Any], Path, str, int]] = []
        for planned_stage in ("resource", "science"):
            for item in _jobs(materialization, planned_stage):
                config, reference, digest, gpu = _config(
                    item,
                    stage=planned_stage,
                    materialization=materialization,
                )
                _registered_inputs(config, registry)
                if digest in planned:
                    raise SelfHurdleEnqueueError(
                        "two planned jobs share a canonical config"
                    )
                planned[digest] = (
                    config, reference, gpu, planned_stage
                )
                current = existing.get(digest)
                if current is not None and (
                    int(current.get("maximum_attempts", -1))
                    != MAXIMUM_ATTEMPTS
                    or str(current.get("requested_gpu")) != str(gpu)
                    or int(current.get("priority", -1))
                    != PRIORITY[planned_stage]
                    or current.get("command")
                    != command_for_config(config)
                ):
                    raise SelfHurdleEnqueueError(
                        "existing job differs from frozen scheduling"
                    )
                if planned_stage == stage:
                    stage_rows.append((config, reference, digest, gpu))
        if set(existing).difference(planned):
            raise SelfHurdleEnqueueError(
                "campaign contains an unexpected root job"
            )

        result_rows: list[dict[str, Any]] = []
        for config, reference, digest, gpu in stage_rows:
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
                    command=command_for_config(config),
                    experiment_config_reference=reference,
                    priority=PRIORITY[stage],
                    maximum_attempts=MAXIMUM_ATTEMPTS,
                    requested_gpu=str(gpu),
                )
                existing[digest] = row
            result_rows.append(
                {
                    "alias": _mapping(
                        config.get("experiment"), "experiment"
                    )["biological_unit_alias"],
                    "config_sha256": digest,
                    "job_id": str(row["job_id"]),
                    "requested_gpu": gpu,
                    "maximum_attempts": MAXIMUM_ATTEMPTS,
                }
            )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_kind": f"self_hurdle_{stage}_enqueue_v1",
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "materialization_checksum": materialization["checksum"],
            "complete": len(result_rows) == 2,
            "jobs": result_rows,
        }
        receipt["checksum"] = canonical_sha256(receipt)
        _atomic(receipt_path, receipt)
        return receipt


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("resource", "science"), required=True
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    locked = current_paths().scratch_root / "locked_campaigns" / CAMPAIGN_ID
    receipt_path = (
        args.receipt
        if args.receipt is not None
        else locked / f"{args.stage}_enqueue_receipt.json"
    )
    result = enqueue_stage(
        stage=args.stage,
        materialization_path=args.materialization.resolve(),
        receipt_path=receipt_path.resolve(),
        database_path=args.database.resolve(),
    )
    print(
        json.dumps(
            {
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

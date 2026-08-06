#!/usr/bin/env python3
"""Idempotently enqueue the locked hybrid-count pilot or production stage.

Every configuration is validated against the checksum-bound materialization
before the first registry write.  Production additionally requires a passing,
checksum-bound two-arm pilot-gate receipt.  The campaign lock serializes the
check-and-enqueue transaction and the enqueue receipt is rewritten atomically
after each job so an interrupted invocation can resume without duplicate fits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
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
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
MATERIALIZATION_KIND = "hybrid_count_locked_config_materialization_v1"
PILOT_GATE_KIND = "hybrid_count_pilot_gate_v1"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("hybrid-gat-k1000", "hybrid-matched-self")
SAFE_GPU_IDS = frozenset({0, 1, 2, 3, 5, 6, 7})
MAXIMUM_ATTEMPTS = 2
EXPECTED_PARAMETER_COUNT = 11_674_880
PILOT_GATE_THRESHOLDS = {
    "peak_allocated_vram_gib_maximum": 20.5,
    "fp32_amp_absolute_total_loss_discrepancy_maximum": 1e-3,
    "projected_gat_runtime_hours_per_core_maximum": 6.0,
}
_PILOT_GATE_REQUIRED_TRUE_FIELDS = (
    "same_frozen_precision_batch",
    "same_evaluation_masks",
    "same_verified_graph",
)
_PILOT_JOB_REQUIRED_TRUE_FIELDS = (
    "verified_bundle",
    "finite_losses_and_gradients",
    "parameter_match",
    "precision_equivalence_passed",
    "peak_vram_passed",
    "projected_runtime_passed",
    "runner_pilot_gate_passed",
)
_EXPECTED_MODEL = {
    "hybrid-gat-k1000": (
        "hybrid-count-gat",
        "hybrid_count_edge_conditioned_gatv2",
        True,
    ),
    "hybrid-matched-self": (
        "hybrid-count-matched-self",
        "hybrid_count_parameter_matched_self_control",
        False,
    ),
}
_PRIORITY = {
    "pilot": {"hybrid-gat-k1000": 50, "hybrid-matched-self": 40},
    "production": {"hybrid-gat-k1000": 30, "hybrid-matched-self": 20},
}


class HybridCountEnqueueError(RuntimeError):
    """Raised when a locked enqueue plan is changed or incomplete."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridCountEnqueueError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise HybridCountEnqueueError(
            f"{label} contains a non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HybridCountEnqueueError(
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
    except HybridCountEnqueueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HybridCountEnqueueError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _verify_checksum(payload: Mapping[str, Any], *, label: str) -> str:
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise HybridCountEnqueueError(f"{label} checksum is malformed")
    canonical = dict(payload)
    canonical.pop("checksum", None)
    if canonical_sha256(canonical) != checksum:
        raise HybridCountEnqueueError(f"{label} checksum does not verify")
    return checksum


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_reference(
    value: Any,
    *,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value.strip():
        raise HybridCountEnqueueError(
            f"{label} must be a nonempty project-relative path"
        )
    reference = Path(value)
    if reference.is_absolute():
        raise HybridCountEnqueueError(f"{label} must be project-relative")
    unresolved = _PROJECT_ROOT / reference
    if unresolved.is_symlink():
        raise HybridCountEnqueueError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HybridCountEnqueueError(f"{label} escapes the project root") from exc
    if require_file and not resolved.is_file():
        raise HybridCountEnqueueError(f"{label} is not an available file")
    if require_directory and not resolved.is_dir():
        raise HybridCountEnqueueError(f"{label} is not an available directory")
    return reference, resolved


def _load_materialization(path: Path) -> dict[str, Any]:
    payload = _strict_json(path, label="locked materialization")
    _verify_checksum(payload, label="locked materialization")
    counts = _mapping(payload.get("counts"), "materialization counts")
    if (
        payload.get("schema_version") != 1
        or payload.get("receipt_kind") != MATERIALIZATION_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or counts.get("aliases") != 10
        or counts.get("pilot_configs") != 2
        or counts.get("production_configs") != 20
        or payload.get("registry_mutation_performed") is not False
        or payload.get("queue_mutation_performed") is not False
        or payload.get("training_performed") is not False
        or payload.get("parameter_count") != EXPECTED_PARAMETER_COUNT
    ):
        raise HybridCountEnqueueError(
            "locked materialization does not describe the frozen campaign"
        )
    allowed = payload.get("allowed_gpu_ids")
    if not isinstance(allowed, list) or set(allowed) != SAFE_GPU_IDS:
        raise HybridCountEnqueueError("materialization safe GPU set changed")
    frozen = _mapping(
        payload.get("frozen_contract"), "materialization frozen contract"
    )
    _, contract_path = _resolve_project_reference(
        frozen.get("reference"),
        label="frozen task contract",
        require_file=True,
    )
    if _sha256_file(contract_path) != frozen.get("sha256"):
        raise HybridCountEnqueueError("frozen task contract checksum changed")
    cores = payload.get("cores")
    if not isinstance(cores, list) or {
        str(core.get("alias"))
        for core in cores
        if isinstance(core, Mapping)
    } != set(ALIASES):
        raise HybridCountEnqueueError(
            "materialization must contain the exact ten alias-safe cores"
        )
    return payload


def _load_pilot_gate(
    path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _strict_json(path, label="pilot gate receipt")
    _verify_checksum(gate, label="pilot gate receipt")
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    thresholds = gate.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or dict(thresholds) != PILOT_GATE_THRESHOLDS
    ):
        raise HybridCountEnqueueError(
            "pilot gate does not declare the exact frozen thresholds"
        )
    if any(
        gate.get(field) is not True
        for field in _PILOT_GATE_REQUIRED_TRUE_FIELDS
    ):
        raise HybridCountEnqueueError(
            "pilot gate does not verify the frozen batch, masks, and graph"
        )
    if gate.get("failure_reasons") != []:
        raise HybridCountEnqueueError(
            "pilot gate must contain no failure reasons"
        )
    jobs = gate.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
    ):
        raise HybridCountEnqueueError(
            "pilot gate must contain exactly two job mappings"
        )
    observed = {
        (str(job.get("alias")), str(job.get("arm")))
        for job in jobs
    }
    if (
        gate.get("schema_version") != 1
        or gate.get("receipt_kind") != PILOT_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("frozen_contract_sha256") != frozen.get("sha256")
        or gate.get("gate_passed") is not True
        or gate.get("production_authorized") is not True
        or observed
        != {
            ("ANC-01", "hybrid-gat-k1000"),
            ("ANC-01", "hybrid-matched-self"),
        }
    ):
        raise HybridCountEnqueueError(
            "production requires the exact passing two-arm pilot gate"
        )
    for job in jobs:
        arm = str(job.get("arm"))
        if job.get("parameter_count") != EXPECTED_PARAMETER_COUNT or any(
            job.get(field) is not True
            for field in _PILOT_JOB_REQUIRED_TRUE_FIELDS
        ):
            raise HybridCountEnqueueError(
                f"pilot gate arm {arm} is unverified or invalid"
            )
    return gate


def _core_map(materialization: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(core["alias"]): core
        for core in materialization["cores"]
        if isinstance(core, Mapping)
    }


def _jobs_for_stage(
    materialization: Mapping[str, Any], stage: str
) -> list[Mapping[str, Any]]:
    field = "pilot_jobs" if stage == "pilot" else "production_jobs"
    raw = materialization.get(field)
    if not isinstance(raw, list):
        raise HybridCountEnqueueError(f"materialization {field} must be a list")
    jobs = [_mapping(item, f"{stage} planned job") for item in raw]
    observed = {(str(item.get("alias")), str(item.get("arm"))) for item in jobs}
    expected = (
        {("ANC-01", arm) for arm in ARMS}
        if stage == "pilot"
        else {(alias, arm) for alias in ALIASES for arm in ARMS}
    )
    if len(jobs) != len(expected) or observed != expected:
        raise HybridCountEnqueueError(
            f"{stage} plan does not contain exact alias-by-arm coverage"
        )
    return jobs


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
        raise HybridCountEnqueueError("config dataset or split is not registered")
    if (
        registered.get("processed_fingerprint")
        != dataset.get("dataset_fingerprint")
        or registered.get("preprocessing_version")
        != dataset.get("preprocessing_version")
        or split.get("dataset_id") != dataset.get("dataset_id")
        or split.get("dataset_version") != dataset.get("version")
        or split.get("fingerprint") != dataset.get("split_fingerprint")
    ):
        raise HybridCountEnqueueError(
            "registered dataset or split identity differs from config"
        )
    _, prepared = _resolve_project_reference(
        dataset.get("prepared_artifact_reference"),
        label="prepared artifact reference",
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
            raise HybridCountEnqueueError(
                "registered protected path differs from prepared artifact"
            )


def _validate_job(
    raw: Mapping[str, Any],
    *,
    stage: str,
    materialization: Mapping[str, Any],
    registry: Registry,
) -> tuple[dict[str, Any], Path, str, int, str]:
    alias = str(raw.get("alias"))
    arm = str(raw.get("arm"))
    try:
        gpu = int(raw.get("requested_gpu"))
    except (TypeError, ValueError) as exc:
        raise HybridCountEnqueueError("planned GPU must be an integer") from exc
    if alias not in ALIASES or arm not in ARMS or gpu not in SAFE_GPU_IDS:
        raise HybridCountEnqueueError("planned alias, arm, or GPU is invalid")
    reference, config_path = _resolve_project_reference(
        raw.get("config"), label="locked config", require_file=True
    )
    if _sha256_file(config_path) != raw.get("file_sha256"):
        raise HybridCountEnqueueError("locked config file checksum changed")
    config = load_yaml_mapping(config_path)
    validate_experiment_config(config)
    digest = canonical_sha256(config)
    if digest != raw.get("config_sha256"):
        raise HybridCountEnqueueError("locked canonical config digest changed")

    campaign = _mapping(config.get("campaign"), "config campaign")
    experiment = _mapping(config.get("experiment"), "config experiment")
    metadata = _mapping(config.get("metadata"), "config metadata")
    model = _mapping(config.get("model"), "config model")
    features = _mapping(config.get("features"), "config features")
    graph = _mapping(config.get("graph"), "config graph")
    trainer = _mapping(config.get("trainer"), "config trainer")
    evaluation = _mapping(config.get("evaluation"), "config evaluation")
    launcher = _mapping(config.get("launcher"), "config launcher")
    dataset = _mapping(config.get("dataset"), "config dataset")
    expected_name, expected_family, uses_graph = _EXPECTED_MODEL[arm]
    expected_epochs = 2 if stage == "pilot" else 200
    expected_replicates = 1 if stage == "pilot" else 3
    core = _core_map(materialization)[alias]
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    materialization_reference = metadata.get(
        "locked_config_materialization_receipt"
    )
    _, materialization_path = _resolve_project_reference(
        materialization_reference,
        label="config materialization receipt",
        require_file=True,
    )
    referenced_materialization = _load_materialization(materialization_path)
    if referenced_materialization.get("checksum") != materialization.get(
        "checksum"
    ):
        raise HybridCountEnqueueError(
            "config references a different locked materialization"
        )
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("exploratory") is not True
        or campaign.get("frozen_contract_sha256") != frozen.get("sha256")
        or experiment.get("arm") != arm
        or experiment.get("biological_unit_alias") != alias
        or bool(experiment.get("resource_pilot")) != (stage == "pilot")
        or metadata.get("execution_role")
        != ("resource_pilot" if stage == "pilot" else "production")
        or model.get("name") != expected_name
        or model.get("family") != expected_family
        or model.get("uses_graph_inputs") is not uses_graph
        or model.get("uses_edge_inputs") is not uses_graph
        or features.get("use_edge_features") is not uses_graph
        or int(graph.get("k", -1)) != 1000
        or int(graph.get("neighbor_k", -1)) != 1000
        or graph.get("symmetry") != "mutual"
        or float(graph.get("radius_guard_um", -1)) != 2000.0
        or graph.get("expected_materialized_graph_sha256")
        != core.get("k1000_graph_sha256")
        or int(graph.get("expected_directed_edges", -1))
        != int(core.get("k1000_directed_edges", -2))
        or int(trainer.get("max_epochs", -1)) != expected_epochs
        or trainer.get("fixed_epoch_budget") is not True
        or trainer.get("restore_best") is not False
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("optimizer") != "AdamW"
        or float(trainer.get("learning_rate", -1)) != 3e-4
        or float(trainer.get("weight_decay", -1)) != 1e-4
        or float(trainer.get("gradient_clip_norm", -1)) != 1.0
        or float(graph.get("edge_dropout", -1)) != 0.0
        or trainer.get("neighbor_sampling") is not False
        or evaluation.get("protocol") != "held_in_full_core_fixed_budget"
        or evaluation.get("task_family") != "masked_expression_hybrid_count"
        or evaluation.get("primary_metric") != "fit/whole_node/hybrid_loss"
        or int(evaluation.get("mask_replicates_per_mode", -1))
        != expected_replicates
        or launcher.get("requested_gpu") != str(gpu)
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or dataset.get("task") != "masked_expression_hybrid_count"
        or int(config.get("seed", -1)) != 0
        or int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) != 1
    ):
        raise HybridCountEnqueueError(
            "locked config differs from its campaign, alias, model, graph, "
            "optimizer, budget, evaluation, or GPU contract"
        )
    if stage == "production":
        authorization = _mapping(
            trainer.get("amp_authorization"), "production AMP authorization"
        )
        if (
            authorization.get("mode")
            != "require_external_pilot_gate_receipt"
            or authorization.get("receipt_schema") != PILOT_GATE_KIND
            or authorization.get("receipt_reference")
            != materialization.get("pilot_gate_receipt_reference")
        ):
            raise HybridCountEnqueueError(
                "production config does not require the frozen pilot gate"
            )
    _validate_registered_dataset(config, registry=registry)
    command = command_for_config(config)
    if not any(part.endswith("run_hybrid_count_capacity.py") for part in command):
        raise HybridCountEnqueueError("hybrid config routes to the wrong runner")
    return config, reference, digest, gpu, arm


def _existing_jobs_by_digest(
    registry: Registry,
) -> dict[str, Mapping[str, Any]]:
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
    for raw_row in rows:
        row = dict(raw_row)
        try:
            config = _mapping(
                json.loads(str(row["canonical_config_json"])),
                "existing queue config",
            )
            row["canonical_config"] = config
            row["command"] = json.loads(str(row["command_json"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridCountEnqueueError(
                "existing campaign queue job contains invalid JSON"
            ) from exc
        digest = canonical_sha256(config)
        if digest in result:
            raise HybridCountEnqueueError(
                "campaign already has duplicate root canonical configurations"
            )
        result[digest] = row
    return result


def _validate_existing(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    reference: Path,
    stage: str,
    arm: str,
    gpu: int,
) -> None:
    if (
        int(row.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(row.get("attempt_count", -1)) != 1
        or row.get("retry_of") is not None
        or str(row.get("requested_gpu")) != str(gpu)
        or int(row.get("priority", -1)) != _PRIORITY[stage][arm]
        or str(row.get("experiment_config_reference")) != str(reference)
        or row.get("command") != command_for_config(config)
    ):
        raise HybridCountEnqueueError(
            "existing root job has wrong command, config, priority, GPU, or "
            "attempt budget"
        )


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True,
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
            handle.write(content)
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


def enqueue_stage(
    *,
    stage: str,
    materialization_path: Path,
    gate_path: Path,
    receipt_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    if stage not in {"pilot", "production"}:
        raise HybridCountEnqueueError("stage must be pilot or production")
    with _campaign_lock(database_path):
        materialization = _load_materialization(materialization_path)
        gate = (
            _load_pilot_gate(gate_path, materialization=materialization)
            if stage == "production"
            else None
        )
        registry = Registry(database_path)
        if registry.get_campaign(CAMPAIGN_ID) is None:
            raise HybridCountEnqueueError("campaign is not registered")
        existing = _existing_jobs_by_digest(registry)

        # Validate every materialized config, including the other stage, so
        # an unexpected pre-existing campaign job cannot be hidden by a
        # stage-specific invocation.
        all_planned_digests: set[str] = set()
        validated_stage: list[
            tuple[dict[str, Any], Path, str, int, str, str]
        ] = []
        for planned_stage in ("pilot", "production"):
            for raw in _jobs_for_stage(materialization, planned_stage):
                config, reference, digest, gpu, arm = _validate_job(
                    raw,
                    stage=planned_stage,
                    materialization=materialization,
                    registry=registry,
                )
                if digest in all_planned_digests:
                    raise HybridCountEnqueueError(
                        "two locked jobs have the same canonical config"
                    )
                all_planned_digests.add(digest)
                matched = existing.get(digest)
                if matched is not None:
                    _validate_existing(
                        matched,
                        config=config,
                        reference=reference,
                        stage=planned_stage,
                        arm=arm,
                        gpu=gpu,
                    )
                if planned_stage == stage:
                    alias = str(
                        _mapping(config.get("experiment"), "experiment").get(
                            "biological_unit_alias"
                        )
                    )
                    validated_stage.append(
                        (config, reference, digest, gpu, arm, alias)
                    )
        unexpected = set(existing).difference(all_planned_digests)
        if unexpected:
            raise HybridCountEnqueueError(
                "campaign contains an unexpected root queue job"
            )

        receipt_rows: list[dict[str, Any]] = []
        for config, reference, digest, gpu, arm, alias in validated_stage:
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
                    priority=_PRIORITY[stage][arm],
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
            partial: dict[str, Any] = {
                "schema_version": 1,
                "receipt_kind": f"hybrid_count_{stage}_enqueue_v1",
                "campaign_id": CAMPAIGN_ID,
                "stage": stage,
                "materialization_checksum": materialization["checksum"],
                "pilot_gate_checksum": (
                    None if gate is None else gate["checksum"]
                ),
                "complete": len(receipt_rows) == len(validated_stage),
                "jobs": receipt_rows,
            }
            partial["checksum"] = canonical_sha256(partial)
            _atomic_receipt(receipt_path, partial)
        return partial


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "production"), required=True)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument(
        "--pilot-gate",
        type=Path,
        default=locked / "pilot_gate_receipt.json",
    )
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
    receipt = args.receipt or locked / f"{args.stage}_enqueue_receipt.json"
    result = enqueue_stage(
        stage=args.stage,
        materialization_path=args.materialization.resolve(),
        gate_path=args.pilot_gate.resolve(),
        receipt_path=receipt.resolve(),
        database_path=args.database.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "stage": args.stage,
                "complete": result["complete"],
                "job_count": len(result["jobs"]),
                "receipt": str(receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

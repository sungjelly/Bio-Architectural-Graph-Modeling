#!/usr/bin/env python3
"""Idempotently enqueue the locked pooled-hybrid pilot or production stage.

The materialization receipt and every locked configuration are verified before
the first registry write.  A production enqueue additionally requires the
checksum-bound passing pilot receipt produced by
``verify_pooled_hybrid_pilot_gate.py``.
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
from spatial_benchmark.identifiers import canonical_sha256, scientific_id  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
MATERIALIZATION_KIND = "pooled_hybrid_count_locked_config_materialization_v1"
PILOT_GATE_KIND = "pooled_hybrid_count_pilot_gate_v1"
CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("pooled-hybrid-gat-k1000", "pooled-hybrid-matched-self")
SEEDS = tuple(range(7))
SAFE_GPU_IDS = frozenset({0, 1, 2, 3, 5, 6, 7})
PILOT_GPU_BY_ARM = {
    "pooled-hybrid-gat-k1000": 0,
    "pooled-hybrid-matched-self": 1,
}
MAXIMUM_ATTEMPTS = 2
EXPECTED_PARAMETER_COUNT = 11_674_880
PILOT_GATE_THRESHOLDS = {
    "fp32_amp_absolute_total_loss_discrepancy_each_core_maximum": 1e-3,
    "peak_allocated_vram_gib_maximum": 20.5,
    "peak_host_memory_gib_per_process_maximum": 40.0,
    "projected_200_epoch_gat_runtime_hours_maximum": 6.0,
    "projected_final_free_disk_gib_minimum": 27.5,
}
_EXPECTED_MODEL = {
    "pooled-hybrid-gat-k1000": (
        "hybrid-count-gat",
        "hybrid_count_edge_conditioned_gatv2",
        True,
    ),
    "pooled-hybrid-matched-self": (
        "hybrid-count-matched-self",
        "hybrid_count_parameter_matched_self_control",
        False,
    ),
}
_PRIORITY = {
    "pilot": {
        "pooled-hybrid-gat-k1000": 50,
        "pooled-hybrid-matched-self": 40,
    },
    "production": {
        "pooled-hybrid-gat-k1000": 30,
        "pooled-hybrid-matched-self": 20,
    },
}
_PILOT_JOB_TRUE_FIELDS = (
    "verified_bundle",
    "checkpoint_verified",
    "finite_losses_and_gradients",
    "all_20_optimizer_steps_completed",
    "every_core_once_each_epoch",
    "parameter_match",
    "paired_initialization_match",
    "precision_equivalence_passed",
    "peak_vram_passed",
    "peak_host_memory_passed",
    "projected_runtime_passed",
    "projected_disk_passed",
    "runner_pilot_gate_passed",
)
_PILOT_GATE_TRUE_FIELDS = (
    "same_frozen_precision_batches_all_cores",
    "same_evaluation_masks",
    "same_verified_graph_bundle",
    "paired_initialization_digests_match",
)


class PooledHybridEnqueueError(RuntimeError):
    """Raised when a locked pooled enqueue plan is invalid or changed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PooledHybridEnqueueError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PooledHybridEnqueueError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise PooledHybridEnqueueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PooledHybridEnqueueError(f"{label} must be an integer")
    return converted


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PooledHybridEnqueueError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PooledHybridEnqueueError(
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
    except PooledHybridEnqueueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PooledHybridEnqueueError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _verify_checksum(payload: Mapping[str, Any], *, label: str) -> str:
    checksum = payload.get("checksum")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise PooledHybridEnqueueError(f"{label} checksum is malformed")
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise PooledHybridEnqueueError(f"{label} checksum does not verify")
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
        raise PooledHybridEnqueueError(
            f"{label} must be a nonempty project-relative path"
        )
    reference = Path(value)
    if reference.is_absolute():
        raise PooledHybridEnqueueError(f"{label} must be project-relative")
    unresolved = _PROJECT_ROOT / reference
    if unresolved.is_symlink():
        raise PooledHybridEnqueueError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PooledHybridEnqueueError(f"{label} escapes the project root") from exc
    if require_file and not resolved.is_file():
        raise PooledHybridEnqueueError(f"{label} is not an available file")
    if require_directory and not resolved.is_dir():
        raise PooledHybridEnqueueError(f"{label} is not an available directory")
    return reference, resolved


def _slot(job: Mapping[str, Any]) -> tuple[str, int]:
    return str(job.get("arm")), _integer(job.get("seed"), "planned seed")


def _expected_slots(stage: str) -> set[tuple[str, int]]:
    if stage == "pilot":
        return {(arm, 0) for arm in ARMS}
    return {(arm, seed) for arm in ARMS for seed in SEEDS}


def _jobs_for_stage(
    materialization: Mapping[str, Any], stage: str
) -> list[Mapping[str, Any]]:
    field = "pilot_jobs" if stage == "pilot" else "production_jobs"
    raw = materialization.get(field)
    if not isinstance(raw, list) or not all(
        isinstance(item, Mapping) for item in raw
    ):
        raise PooledHybridEnqueueError(f"materialization {field} must be mappings")
    jobs = [dict(item) for item in raw]
    observed = {_slot(job) for job in jobs}
    expected = _expected_slots(stage)
    if len(jobs) != len(expected) or observed != expected:
        raise PooledHybridEnqueueError(
            f"{stage} plan must contain exactly {len(expected)} pooled "
            "arm-by-seed members"
        )
    return jobs


def _load_materialization(path: Path) -> dict[str, Any]:
    payload = _strict_json(path, label="locked materialization")
    _verify_checksum(payload, label="locked materialization")
    counts = _mapping(payload.get("counts"), "materialization counts")
    frozen = _mapping(
        payload.get("frozen_contract"), "materialization frozen contract"
    )
    cohort = _mapping(payload.get("cohort"), "materialization cohort")
    if (
        payload.get("schema_version") != 1
        or payload.get("receipt_kind") != MATERIALIZATION_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or counts.get("aliases") != 10
        or counts.get("pilot_configs") != 2
        or counts.get("production_configs") != 14
        or counts.get("production_seeds") != 7
        or payload.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or payload.get("registry_mutation_performed") is not False
        or payload.get("queue_mutation_performed") is not False
        or payload.get("training_performed") is not False
        or frozen.get("sha256") != CONTRACT_SHA256
    ):
        raise PooledHybridEnqueueError(
            "locked materialization does not describe the frozen pooled campaign"
        )
    allowed = payload.get("allowed_gpu_ids")
    if not isinstance(allowed, list) or set(allowed) != SAFE_GPU_IDS:
        raise PooledHybridEnqueueError("materialization safe GPU set changed")
    _, contract_path = _resolve_project_reference(
        frozen.get("reference"), label="frozen task contract", require_file=True
    )
    if _sha256_file(contract_path) != CONTRACT_SHA256:
        raise PooledHybridEnqueueError("frozen task contract checksum changed")
    aliases = cohort.get("aliases")
    graph_sources = payload.get("graph_sources")
    mask_sources = payload.get("evaluation_mask_sources")
    if (
        not isinstance(aliases, list)
        or tuple(aliases) != ALIASES
        or not isinstance(graph_sources, Mapping)
        or set(graph_sources) != set(ALIASES)
        or not isinstance(mask_sources, Mapping)
        or set(mask_sources) != set(ALIASES)
    ):
        raise PooledHybridEnqueueError(
            "materialization must contain exact alias-safe graph and mask sources"
        )
    pilot_jobs = _jobs_for_stage(payload, "pilot")
    production_jobs = _jobs_for_stage(payload, "production")
    if {
        str(job.get("arm")): _integer(job.get("requested_gpu"), "pilot GPU")
        for job in pilot_jobs
    } != PILOT_GPU_BY_ARM:
        raise PooledHybridEnqueueError("pilot GPUs must be fixed to 0 and 1")
    for job in [*pilot_jobs, *production_jobs]:
        if _integer(job.get("requested_gpu"), "planned GPU") not in SAFE_GPU_IDS:
            raise PooledHybridEnqueueError("planned job uses an unsafe GPU")
    return payload


def _load_pilot_gate(
    path: Path, *, materialization: Mapping[str, Any]
) -> dict[str, Any]:
    gate = _strict_json(path, label="pilot gate receipt")
    _verify_checksum(gate, label="pilot gate receipt")
    thresholds = gate.get("thresholds")
    jobs = gate.get("jobs")
    if (
        gate.get("schema_version") != 1
        or gate.get("receipt_kind") != PILOT_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("materialization_checksum") != materialization.get("checksum")
        or gate.get("frozen_contract_sha256") != CONTRACT_SHA256
        or not isinstance(thresholds, Mapping)
        or dict(thresholds) != PILOT_GATE_THRESHOLDS
        or gate.get("failure_reasons") != []
        or gate.get("gate_passed") is not True
        or gate.get("production_authorized") is not True
        or any(gate.get(field) is not True for field in _PILOT_GATE_TRUE_FIELDS)
        or not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
    ):
        raise PooledHybridEnqueueError(
            "production requires the exact checksum-bound passing pooled pilot gate"
        )
    if {_slot(job) for job in jobs} != _expected_slots("pilot"):
        raise PooledHybridEnqueueError(
            "production requires exactly the two pooled pilot arms"
        )
    for job in jobs:
        if (
            job.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or job.get("materialization_checksum")
            != materialization.get("checksum")
            or any(job.get(field) is not True for field in _PILOT_JOB_TRUE_FIELDS)
            or job.get("checkpoint_role") != "last"
            or job.get("checkpoint_epoch") != 1
            or not isinstance(job.get("checkpoint_id"), str)
            or not str(job.get("checkpoint_id")).strip()
            or not isinstance(job.get("checkpoint_sha256"), str)
            or len(str(job.get("checkpoint_sha256"))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(job.get("checkpoint_sha256"))
            )
        ):
            raise PooledHybridEnqueueError(
                f"pilot gate arm {job.get('arm')} is unverified or invalid"
            )
    return gate


def _load_existing_enqueue_receipt(
    path: Path,
    *,
    stage: str,
    materialization: Mapping[str, Any],
    gate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise PooledHybridEnqueueError(
            "existing enqueue receipt is not a safe regular file"
        )
    receipt = _strict_json(path, label="existing enqueue receipt")
    _verify_checksum(receipt, label="existing enqueue receipt")
    jobs = receipt.get("jobs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_kind")
        != f"pooled_hybrid_count_{stage}_enqueue_v1"
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("stage") != stage
        or receipt.get("materialization_checksum")
        != materialization.get("checksum")
        or receipt.get("pilot_gate_checksum")
        != (None if gate is None else gate.get("checksum"))
        or not isinstance(receipt.get("complete"), bool)
        or not isinstance(jobs, list)
        or not all(isinstance(job, Mapping) for job in jobs)
    ):
        raise PooledHybridEnqueueError(
            "existing enqueue receipt conflicts with the locked stage"
        )
    observed = [_slot(job) for job in jobs]
    planned = [_slot(job) for job in _jobs_for_stage(materialization, stage)]
    if (
        len(observed) != len(set(observed))
        or observed != planned[: len(observed)]
        or (receipt["complete"] is True and len(observed) != len(planned))
        or (receipt["complete"] is False and len(observed) >= len(planned))
    ):
        raise PooledHybridEnqueueError(
            "existing enqueue receipt is not an exact stage prefix"
        )
    return receipt


def _validate_registered_dataset(
    config: Mapping[str, Any], *, registry: Registry
) -> None:
    dataset = _mapping(config.get("dataset"), "config dataset")
    registered = registry.get_dataset(
        str(dataset.get("dataset_id")), str(dataset.get("version"))
    )
    split = registry.get_split(str(dataset.get("split_id")))
    if registered is None or split is None:
        raise PooledHybridEnqueueError(
            "pooled dataset or split is not registered"
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
        raise PooledHybridEnqueueError(
            "registered pooled dataset or split identity differs from config"
        )
    aliases = dataset.get("core_aliases")
    if not isinstance(aliases, list) or tuple(aliases) != ALIASES:
        raise PooledHybridEnqueueError("config pooled alias order changed")
    prepared = _mapping(
        dataset.get("prepared_artifacts"), "prepared artifact map"
    )
    if set(prepared) != set(ALIASES):
        raise PooledHybridEnqueueError(
            "config must reference exactly ten prepared artifacts"
        )
    for alias in ALIASES:
        _resolve_project_reference(
            prepared[alias],
            label=f"{alias} prepared artifact",
            require_directory=True,
        )


def _validate_job(
    raw: Mapping[str, Any],
    *,
    stage: str,
    materialization: Mapping[str, Any],
    registry: Registry,
) -> tuple[dict[str, Any], Path, str, int, str, int]:
    arm, seed = _slot(raw)
    gpu = _integer(raw.get("requested_gpu"), "planned GPU")
    if (arm, seed) not in _expected_slots(stage) or gpu not in SAFE_GPU_IDS:
        raise PooledHybridEnqueueError("planned arm, seed, or GPU is invalid")
    reference, config_path = _resolve_project_reference(
        raw.get("config"), label="locked config", require_file=True
    )
    if _sha256_file(config_path) != raw.get("file_sha256"):
        raise PooledHybridEnqueueError("locked config file checksum changed")
    config = load_yaml_mapping(config_path)
    validate_experiment_config(config)
    digest = canonical_sha256(config)
    if digest != raw.get("config_sha256"):
        raise PooledHybridEnqueueError("locked canonical config digest changed")

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
    expected_steps = 20 if stage == "pilot" else 2000
    cohort = _mapping(materialization.get("cohort"), "materialization cohort")
    graph_sources = _mapping(
        materialization.get("graph_sources"), "materialization graph sources"
    )
    mask_sources = _mapping(
        materialization.get("evaluation_mask_sources"),
        "materialization evaluation mask sources",
    )
    expected_core_graphs = {
        alias: {
            "n_nodes": graph_sources[alias]["n_nodes"],
            "graph_sha256": graph_sources[alias]["graph_sha256"],
            "n_directed_edges": graph_sources[alias]["n_directed_edges"],
        }
        for alias in ALIASES
    }

    receipt_reference = metadata.get("locked_config_materialization_receipt")
    _, receipt_path = _resolve_project_reference(
        receipt_reference,
        label="config materialization receipt",
        require_file=True,
    )
    referenced = _load_materialization(receipt_path)
    if referenced.get("checksum") != materialization.get("checksum"):
        raise PooledHybridEnqueueError(
            "config references a different locked materialization"
        )
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("exploratory") is not True
        or campaign.get("frozen_contract_sha256") != CONTRACT_SHA256
        or experiment.get("arm") != arm
        or bool(experiment.get("resource_pilot")) != (stage == "pilot")
        or metadata.get("execution_role")
        != ("resource_pilot" if stage == "pilot" else "production")
        or model.get("name") != expected_name
        or model.get("family") != expected_family
        or model.get("uses_graph_inputs") is not uses_graph
        or model.get("uses_edge_inputs") is not uses_graph
        or features.get("use_edge_features") is not uses_graph
        or graph.get("cross_core_edges") is not False
        or int(graph.get("k", -1)) != 1000
        or graph.get("symmetry") != "mutual"
        or float(graph.get("radius_guard_um", -1)) != 2000.0
        or float(graph.get("edge_dropout", -1)) != 0.0
        or int(trainer.get("max_epochs", -1)) != expected_epochs
        or int(trainer.get("optimizer_steps_per_epoch", -1)) != 10
        or int(trainer.get("total_optimizer_steps", -1)) != expected_steps
        or trainer.get("core_sampling") != "exactly_once_per_epoch"
        or int(trainer.get("core_order_seed", -1)) != 271828
        or trainer.get("fixed_epoch_budget") is not True
        or trainer.get("restore_best") is not False
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("optimizer") != "AdamW"
        or float(trainer.get("learning_rate", -1)) != 3e-4
        or float(trainer.get("weight_decay", -1)) != 1e-4
        or float(trainer.get("gradient_clip_norm", -1)) != 1.0
        or trainer.get("neighbor_sampling") is not False
        or evaluation.get("protocol")
        != "held_in_pooled_10core_fixed_budget"
        or evaluation.get("task_family") != "masked_expression_hybrid_count"
        or evaluation.get("primary_metric") != "fit/whole_node/hybrid_loss"
        or int(evaluation.get("mask_replicates_per_mode", -1)) != 3
        or launcher.get("requested_gpu") != str(gpu)
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or dataset.get("fit_scope")
        != "all_117386_cells_across_ten_cores_transductive"
        or int(dataset.get("total_fit_cells", -1)) != 117386
        or dataset.get("dataset_fingerprint")
        != cohort.get("dataset_fingerprint")
        or dataset.get("split_id") != cohort.get("split_id")
        or dataset.get("split_fingerprint")
        != cohort.get("split_fingerprint")
        or dataset.get("prepared_artifacts")
        != cohort.get("prepared_artifacts")
        or graph.get("expected_core_graphs") != expected_core_graphs
        or evaluation.get("prior_mask_sources") != mask_sources
        or int(config.get("seed", -1)) != seed
        or int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) != 1
    ):
        raise PooledHybridEnqueueError(
            "locked pooled config differs from its frozen arm, seed, graph, "
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
            raise PooledHybridEnqueueError(
                "production config does not require the pooled pilot gate"
            )
    _validate_registered_dataset(config, registry=registry)
    command = command_for_config(config)
    if not any(
        part.endswith("run_pooled_hybrid_count_capacity.py") for part in command
    ):
        raise PooledHybridEnqueueError("pooled config routes to the wrong runner")
    return config, reference, digest, gpu, arm, seed


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
                "existing canonical config",
            )
            row["canonical_config"] = config
            row["command"] = json.loads(str(row["command_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PooledHybridEnqueueError(
                "existing campaign queue job contains invalid JSON"
            ) from exc
        digest = canonical_sha256(config)
        if digest in result:
            raise PooledHybridEnqueueError(
                "campaign has duplicate root canonical configurations"
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
        raise PooledHybridEnqueueError(
            "existing root job has wrong command, config, priority, GPU, or "
            "attempt budget"
        )


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
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
        if temporary_name and Path(temporary_name).exists():
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
        raise PooledHybridEnqueueError("stage must be pilot or production")
    with _campaign_lock(database_path):
        materialization = _load_materialization(materialization_path)
        gate = (
            _load_pilot_gate(gate_path, materialization=materialization)
            if stage == "production"
            else None
        )
        prior_receipt = _load_existing_enqueue_receipt(
            receipt_path,
            stage=stage,
            materialization=materialization,
            gate=gate,
        )
        registry = Registry(database_path)
        if registry.get_campaign(CAMPAIGN_ID) is None:
            raise PooledHybridEnqueueError("campaign is not registered")
        existing = _existing_jobs_by_digest(registry)

        all_digests: set[str] = set()
        validated: list[
            tuple[dict[str, Any], Path, str, int, str, int]
        ] = []
        for planned_stage in ("pilot", "production"):
            for raw in _jobs_for_stage(materialization, planned_stage):
                item = _validate_job(
                    raw,
                    stage=planned_stage,
                    materialization=materialization,
                    registry=registry,
                )
                config, reference, digest, gpu, arm, _seed = item
                if digest in all_digests:
                    raise PooledHybridEnqueueError(
                        "two locked jobs have the same canonical config"
                    )
                all_digests.add(digest)
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
                    validated.append(item)
        if set(existing).difference(all_digests):
            raise PooledHybridEnqueueError(
                "campaign contains an unexpected root queue job"
            )
        if prior_receipt is not None:
            planned_by_slot = {
                (arm, seed): (digest, gpu)
                for _config, _reference, digest, gpu, arm, seed in validated
            }
            for receipt_job in prior_receipt["jobs"]:
                slot = _slot(receipt_job)
                digest, gpu = planned_by_slot[slot]
                existing_job = existing.get(digest)
                if (
                    existing_job is None
                    or receipt_job.get("config_sha256") != digest
                    or receipt_job.get("requested_gpu") != gpu
                    or receipt_job.get("maximum_attempts")
                    != MAXIMUM_ATTEMPTS
                    or str(receipt_job.get("job_id"))
                    != str(existing_job.get("job_id"))
                ):
                    raise PooledHybridEnqueueError(
                        "existing enqueue receipt differs from registry lineage"
                    )

        receipt_rows: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for config, reference, digest, gpu, arm, seed in validated:
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
                    "arm": arm,
                    "seed": seed,
                    "config_sha256": digest,
                    "job_id": str(row["job_id"]),
                    "requested_gpu": gpu,
                    "maximum_attempts": MAXIMUM_ATTEMPTS,
                }
            )
            result = {
                "schema_version": 1,
                "receipt_kind": f"pooled_hybrid_count_{stage}_enqueue_v1",
                "campaign_id": CAMPAIGN_ID,
                "stage": stage,
                "materialization_checksum": materialization["checksum"],
                "pilot_gate_checksum": None if gate is None else gate["checksum"],
                "complete": len(receipt_rows) == len(validated),
                "jobs": receipt_rows,
            }
            result["checksum"] = canonical_sha256(result)
            _atomic_receipt(receipt_path, result)
        if result is None:
            raise PooledHybridEnqueueError("locked stage contains no jobs")
        return result


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
        "--pilot-gate", type=Path, default=locked / "pilot_gate_receipt.json"
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
                "checksum": result["checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

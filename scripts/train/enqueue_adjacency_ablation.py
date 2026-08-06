#!/usr/bin/env python3
"""Materialize and enqueue the frozen grouped adjacency ablation.

This entry point is intentionally campaign-specific.  It verifies the prepared
artifact, writes immutable resolved configurations, and idempotently enqueues
one explicitly requested stage.  Primary and topology-null work fail closed
unless their checksum-bound gate receipts verify.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "train"))

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


CAMPAIGN_ID = "cmp_20260802_adjacent_normal_grouped_adjacency_ablation"
CONTRACT_SHA256 = (
    "08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff"
)
DATASET_ID = "cosmx_adjacent_normal_grouped_adjacency_v1"
DATASET_VERSION = "adjacent_normal_grouped_adjacency_v1"
SPLIT_ID = "adjacent_normal_10donor_slide_balanced_5fold_v1"
MODEL_NAME = "mean-adjacency-sage"
MODEL_FAMILY = "explicit_self_mean_adjacency_graphsage"
EXPECTED_PARAMETER_COUNT = 645_736
CONDITIONS = ("spatial", "isolated")
NULL_CONDITION = "position_permuted_null"
STAGES = ("smoke", "pilot", "primary", "null")
EPOCHS = {"smoke": 1, "pilot": 5, "primary": 80, "null": 80}
EXPECTED_COUNTS = {"smoke": 2, "pilot": 2, "primary": 50, "null": 25}
STAGE_PRIORITY = {"smoke": 80, "pilot": 70, "primary": 50, "null": 40}
MAXIMUM_ATTEMPTS = 1
MATERIALIZATION_KIND = "adjacency_ablation_config_materialization_v1"
PILOT_GATE_KIND = "adjacency_ablation_paired_pilot_gate_v1"
NULL_TRIGGER_KIND = "adjacency_ablation_null_trigger_v1"
RECOVERY_PLAN_KIND = "adjacency_ablation_cpu_recovery_plan_v1"
RECOVERY_ENQUEUE_KIND = "adjacency_ablation_cpu_recovery_enqueue_v1"
RECOVERY_CONTRACT_SHA256 = (
    "84230db441cd03d158a28bc79dfecee495fee1b0864dae2311a0a9041f7b6bb0"
)
RECOVERY_CONTRACT_REFERENCE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "hardware_recovery_contract.yaml"
)
RECOVERY_PLAN_FILENAME = "primary_cpu_recovery_plan_receipt.json"
RECOVERY_ENQUEUE_FILENAME = "primary_cpu_recovery_enqueue_receipt.json"
PREPARED_ARTIFACT_KIND = (
    "adjacent_normal_grouped_adjacency_ablation_prepared_v1"
)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
GPU_IDS = tuple(range(8))
_HEX = frozenset("0123456789abcdef")


class AdjacencyAblationEnqueueError(RuntimeError):
    """Raised when a frozen materialization, gate, or queue plan drifts."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjacencyAblationEnqueueError(f"{label} must be a mapping")
    return value


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
        raise AdjacencyAblationEnqueueError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AdjacencyAblationEnqueueError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdjacencyAblationEnqueueError(
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
    except AdjacencyAblationEnqueueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdjacencyAblationEnqueueError(
            f"{label} is not strict JSON: {path}"
        ) from exc
    return dict(_mapping(value, label))


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["checksum"] = canonical_sha256(result)
    return result


def _verify_signed(payload: Mapping[str, Any], label: str) -> str:
    checksum = _hex_digest(payload.get("checksum"), f"{label} checksum")
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise AdjacencyAblationEnqueueError(f"{label} checksum does not verify")
    return checksum


def _project_reference(path: Path, label: str) -> tuple[Path, Path]:
    resolved = path.resolve()
    try:
        reference = resolved.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise AdjacencyAblationEnqueueError(
            f"{label} must remain under the BAGM project root"
        ) from exc
    if resolved.is_symlink():
        raise AdjacencyAblationEnqueueError(f"{label} cannot be a symlink")
    return reference, resolved


def _contract() -> tuple[Path, str]:
    path = (
        _PROJECT_ROOT
        / "experiments/campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if not path.is_file() or _sha256_file(path) != CONTRACT_SHA256:
        raise AdjacencyAblationEnqueueError(
            "frozen task contract is missing or its checksum changed"
        )
    contract = load_yaml_mapping(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign_id") != CAMPAIGN_ID
        or _mapping(contract.get("model"), "contract model").get("name")
        != MODEL_NAME
        or _mapping(contract.get("model"), "contract model").get("family")
        != MODEL_FAMILY
    ):
        raise AdjacencyAblationEnqueueError(
            "frozen task contract identity or model changed"
        )
    return path, CONTRACT_SHA256


def _prepared_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise AdjacencyAblationEnqueueError(
            f"prepared manifest is unavailable: {manifest_path}"
        )
    try:
        from prepare_adjacency_ablation import verify_prepared_artifact

        manifest = verify_prepared_artifact(manifest_path)
    except AdjacencyAblationEnqueueError:
        raise
    except Exception as exc:
        raise AdjacencyAblationEnqueueError(
            "prepared adjacency artifact verification failed"
        ) from exc
    manifest = dict(_mapping(manifest, "prepared manifest"))
    dataset = _mapping(manifest.get("dataset"), "prepared dataset")
    split = _mapping(manifest.get("split"), "prepared split")
    graph = _mapping(manifest.get("graph"), "prepared graph")
    masking = _mapping(manifest.get("masking"), "prepared masking")
    preprocessing = _mapping(
        manifest.get("preprocessing"), "prepared preprocessing"
    )
    registry_contract = _mapping(
        manifest.get("registry_contract"), "prepared registry contract"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != PREPARED_ARTIFACT_KIND
        or dataset.get("dataset_id") != DATASET_ID
        or dataset.get("dataset_version") != DATASET_VERSION
        or dataset.get("core_count") != 10
        or dataset.get("cell_count") != 117_386
        or dataset.get("fov_group_count") != 139
        or dataset.get("n_genes") != 1000
        or split.get("split_id") != SPLIT_ID
        or split.get("fold_count") != 5
        or split.get("train_groups") != 7
        or split.get("validation_groups") != 1
        or split.get("test_groups") != 2
        or graph.get("k") != 12
        or float(graph.get("radius_um", -1)) != 50.0
        or graph.get("symmetry") != "union"
        or graph.get("grouping") != "raw_fov_within_core"
        or graph.get("edge_features") is not False
        or set(graph.get("arms", []))
        != {"spatial", "isolated", "position_permuted_null"}
        or graph.get("null_seed") != 2_026_080_202
        or masking.get("distribution")
        != "exact_uniform_integer_0_through_G_per_cell"
        or masking.get("positions") != "without_replacement"
        or masking.get("repetitions") != 3
        or masking.get("base_seed") != 20_260_802
        or preprocessing.get("scale_floor") != 1.0e-6
        or registry_contract.get("explicit_register_flag_required") is not True
        or registry_contract.get("dataset_id") != DATASET_ID
        or registry_contract.get("dataset_version") != DATASET_VERSION
        or registry_contract.get("split_id") != SPLIT_ID
    ):
        raise AdjacencyAblationEnqueueError(
            "prepared manifest differs from the frozen cohort, split, graph, "
            "mask, preprocessing, or registry contract"
        )
    content_sha = _hex_digest(
        manifest.get("content_sha256"), "prepared content_sha256"
    )
    if manifest.get("artifact_id") != content_sha[:16]:
        raise AdjacencyAblationEnqueueError(
            "prepared artifact_id is not derived from content_sha256"
        )
    if set(_mapping(manifest.get("cores"), "prepared cores")) != set(ALIASES):
        raise AdjacencyAblationEnqueueError(
            "prepared manifest does not contain the exact ten opaque aliases"
        )
    folds = _mapping(manifest.get("folds"), "prepared folds")
    test_counts = {alias: 0 for alias in ALIASES}
    for fold in FOLDS:
        item = _mapping(folds.get(str(fold)), f"prepared fold {fold}")
        train = tuple(item.get("train_aliases", []))
        validation = tuple(item.get("validation_aliases", []))
        test = tuple(item.get("test_aliases", []))
        if (
            len(train) != 7
            or len(validation) != 1
            or len(test) != 2
            or set(train) | set(validation) | set(test) != set(ALIASES)
            or set(train) & set(validation)
            or set(train) & set(test)
            or set(validation) & set(test)
        ):
            raise AdjacencyAblationEnqueueError(
                f"prepared fold {fold} is not a disjoint 7/1/2 split"
            )
        for alias in test:
            test_counts[str(alias)] += 1
    if set(folds) != {str(fold) for fold in FOLDS} or set(test_counts.values()) != {1}:
        raise AdjacencyAblationEnqueueError(
            "prepared folds do not test every core exactly once"
        )
    return manifest, manifest_path.resolve()


def _gpu_for(fold: int, seed: int) -> int:
    return GPU_IDS[(fold * len(SEEDS) + seed) % len(GPU_IDS)]


def _stage_slots(stage: str) -> list[tuple[int, int, str]]:
    if stage in {"smoke", "pilot"}:
        return [(0, 0, condition) for condition in CONDITIONS]
    if stage == "primary":
        return [
            (fold, seed, condition)
            for fold in FOLDS
            for seed in SEEDS
            for condition in CONDITIONS
        ]
    if stage == "null":
        return [
            (fold, seed, NULL_CONDITION)
            for fold in FOLDS
            for seed in SEEDS
        ]
    raise AdjacencyAblationEnqueueError(f"unknown stage {stage!r}")


def _build_config(
    *,
    stage: str,
    fold: int,
    seed: int,
    condition: str,
    manifest: Mapping[str, Any],
    manifest_reference: Path,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    dataset = _mapping(manifest.get("dataset"), "prepared dataset")
    split = _mapping(manifest.get("split"), "prepared split")
    graph = _mapping(manifest.get("graph"), "prepared graph")
    masking = _mapping(manifest.get("masking"), "prepared masking")
    preprocessing = _mapping(
        manifest.get("preprocessing"), "prepared preprocessing"
    )
    epochs = EPOCHS[stage]
    gpu = _gpu_for(fold, seed)
    config: dict[str, Any] = {
        "schema_version": 1,
        "name": f"adjacency-ablation-{stage}",
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "exploratory": True,
            "frozen_contract_sha256": CONTRACT_SHA256,
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": (
                "diagnostic" if stage in {"smoke", "pilot"} else "exploratory_screen"
            ),
            "study_axis": "adjacent_normal_grouped_adjacency_ablation",
            "scientific_variant": f"{stage}_{condition}",
            "retention_class": (
                "discardable_diagnostic"
                if stage == "smoke"
                else "retain_exploratory_evidence"
            ),
            "classification_confidence": "high",
        },
        "experiment": {"stage": stage},
        "dataset": {
            "dataset_id": DATASET_ID,
            "version": DATASET_VERSION,
            "split_id": SPLIT_ID,
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "split_fingerprint": split["assignment_fingerprint"],
            "preprocessing_version": preprocessing["preprocessing_fingerprint"],
            "prepared_manifest": manifest_reference.as_posix(),
            "prepared_manifest_sha256": manifest_file_sha256,
            "prepared_content_sha256": manifest["content_sha256"],
            "task": "masked_expression_regression",
            "target_scale": "standardized_gene_wise_log1p_raw_count",
            "biological_probe_count": 1000,
        },
        "features": {
            "use_edge_features": False,
            "node_features": ["masked_expression", "binary_mask_indicator"],
            "coordinates_as_model_input": False,
        },
        "graph": {
            "adjacency_condition": condition,
            "neighbor_k": 12,
            "k": 12,
            "radius_um": 50.0,
            "symmetry": "union",
            "grouping": "raw_fov_within_core",
            "self_loops": "one_per_cell_in_all_arms",
            "null_kind": graph["null_kind"],
            "null_seed": 2_026_080_202,
            "graph_fingerprint": graph["graph_fingerprint"],
        },
        "masking": {
            "type": "exact_uniform_count_expression_masking",
            "rate": {"minimum": 0.0, "maximum": 1.0},
            "distribution": "exact_uniform_integer_0_through_G_per_cell",
            "positions": "without_replacement",
            "mask_indicator": True,
            "dynamic_training": True,
            "training_base_seed": 2_026_080_201,
            "evaluation_replicates": 3,
            "evaluation_base_seed": 20_260_802,
            "mask_fingerprint": masking["mask_fingerprint"],
        },
        "preprocessing": {
            "expression": "gene_wise_standardized_log1p_raw_count",
            "fit_scope": "seven_training_cores_only_equal_core_moments",
            "scale_floor": 1.0e-6,
            "feature_selection": "none",
            "library_size_normalization": "none",
            "preprocessing_fingerprint": preprocessing[
                "preprocessing_fingerprint"
            ],
        },
        "model": {
            "name": MODEL_NAME,
            "family": MODEL_FAMILY,
            "embedding_dim": 128,
            "hidden_dim": 128,
            "ffn_dim": 256,
            "decoder_dim": 256,
            "graph_layers": 1,
            "dropout": 0.1,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "trainer": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 1.0,
            "precision": "fp32",
            "batch_size": 1,
            "max_epochs": epochs,
            "expected_updates_per_epoch": 7,
            "expected_total_optimizer_updates": 7 * epochs,
            "validation_interval_epochs": 5,
            "early_stopping": False,
            "restore_best": True,
            "primary_checkpoint_role": "best",
            "checkpoint_policy": "minimum_fixed_validation_huber",
            "paired_core_order": True,
        },
        "evaluation": {
            "protocol": "grouped_core_adjacency_ablation_v1",
            "task_family": "masked_expression_regression",
            "primary_metric": "val/masked_huber",
            "primary_direction": "minimize",
            "canonical_prediction_split": "validation",
            "splits": ["validation", "test"],
            "fixed_mask_replicates": 3,
            "core_equal_aggregation": True,
        },
        "launcher": {"requested_gpu": str(gpu)},
        "seed": seed,
        "fold": fold,
        "attempt": 1,
    }
    validate_experiment_config(config)
    expected_command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts/train/run_adjacency_ablation.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]
    if command_for_config(config) != expected_command:
        raise AdjacencyAblationEnqueueError(
            "resolved adjacency config routes to an unexpected command"
        )
    return config


def _paired_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    _mapping(result["classification"], "classification").pop(
        "scientific_variant", None
    )
    _mapping(result["graph"], "graph").pop("adjacency_condition", None)
    return result


def _assert_paired_configs(jobs: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for job in jobs:
        stage = str(job["stage"])
        if stage == "null":
            continue
        slot = (stage, int(job["fold"]), int(job["seed"]))
        grouped.setdefault(slot, {})[str(job["condition"])] = _mapping(
            job["config_payload"], "materialized config"
        )
    for slot, paired in grouped.items():
        if set(paired) != set(CONDITIONS) or _paired_payload(
            paired["spatial"]
        ) != _paired_payload(paired["isolated"]):
            raise AdjacencyAblationEnqueueError(
                f"paired configs for {slot} differ beyond adjacency and variant"
            )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(payload), sort_keys=True, allow_unicode=False
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != content:
            raise AdjacencyAblationEnqueueError(
                f"refusing to overwrite immutable campaign file {path}"
            )
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _campaign_lock(locked_root: Path) -> Iterator[None]:
    lock_path = locked_root / ".materialize-enqueue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _registry_contract(registry: Registry, manifest: Mapping[str, Any]) -> None:
    dataset = _mapping(manifest.get("dataset"), "prepared dataset")
    split = _mapping(manifest.get("split"), "prepared split")
    registered_dataset = registry.get_dataset(DATASET_ID, DATASET_VERSION)
    registered_split = registry.get_split(SPLIT_ID)
    if registered_dataset is None or registered_split is None:
        raise AdjacencyAblationEnqueueError(
            "prepared dataset/split are not registered; rerun preparation with "
            "the explicit --register flag"
        )
    if (
        registered_dataset.get("processed_fingerprint")
        != dataset.get("dataset_fingerprint")
        or registered_split.get("dataset_id") != DATASET_ID
        or registered_split.get("dataset_version") != DATASET_VERSION
        or registered_split.get("fingerprint")
        != split.get("assignment_fingerprint")
        or registered_split.get("fold_count") != 5
        or registered_split.get("unit") != "donor_core_one_to_one"
    ):
        raise AdjacencyAblationEnqueueError(
            "registered dataset or split differs from the verified artifact"
        )
    registry.create_campaign(
        CAMPAIGN_ID,
        name="Grouped adjacent-normal adjacency ablation",
        scientific_question=(
            "Does fixed spatial adjacency improve donor/core-held-out masked "
            "transcript reconstruction over identity adjacency?"
        ),
        config={
            "exploratory": True,
            "frozen_contract_sha256": CONTRACT_SHA256,
        },
        status="planned",
    )


def materialize(
    *,
    manifest_path: Path,
    locked_root: Path,
    database_path: Path,
) -> dict[str, Any]:
    with _campaign_lock(locked_root):
        contract_path, contract_sha = _contract()
        manifest, resolved_manifest = _prepared_manifest(manifest_path)
        manifest_reference, _ = _project_reference(
            resolved_manifest, "prepared manifest"
        )
        contract_reference, _ = _project_reference(
            contract_path, "frozen task contract"
        )
        manifest_file_sha = _sha256_file(resolved_manifest)
        planned: list[dict[str, Any]] = []
        for stage in STAGES:
            for fold, seed, condition in _stage_slots(stage):
                config = _build_config(
                    stage=stage,
                    fold=fold,
                    seed=seed,
                    condition=condition,
                    manifest=manifest,
                    manifest_reference=manifest_reference,
                    manifest_file_sha256=manifest_file_sha,
                )
                config_path = (
                    locked_root
                    / "configs"
                    / stage
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / f"{condition}.yaml"
                )
                config_reference, _ = _project_reference(
                    config_path, "locked config"
                )
                content = _yaml_bytes(config)
                planned.append(
                    {
                        "stage": stage,
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "requested_gpu": _gpu_for(fold, seed),
                        "config_reference": config_reference.as_posix(),
                        "config_sha256": canonical_sha256(config),
                        "config_file_sha256": hashlib.sha256(content).hexdigest(),
                        "scientific_id": scientific_id(config),
                        "config_payload": config,
                        "content": content,
                        "path": config_path,
                    }
                )
        if {
            stage: sum(job["stage"] == stage for job in planned)
            for stage in STAGES
        } != EXPECTED_COUNTS:
            raise AdjacencyAblationEnqueueError(
                "materialized stage run counts differ from the frozen design"
            )
        _assert_paired_configs(planned)
        for job in planned:
            _write_immutable(Path(job["path"]), bytes(job["content"]))

        registry = Registry(database_path)
        registry.initialize()
        _registry_contract(registry, manifest)
        for job in planned:
            registry.register_variant(
                str(job["scientific_id"]),
                campaign_id=CAMPAIGN_ID,
                configuration=_mapping(job["config_payload"], "config"),
            )

        receipt_jobs = [
            {
                key: job[key]
                for key in (
                    "stage",
                    "fold",
                    "seed",
                    "condition",
                    "requested_gpu",
                    "config_reference",
                    "config_sha256",
                    "config_file_sha256",
                    "scientific_id",
                )
            }
            for job in planned
        ]
        receipt = _signed(
            {
                "schema_version": 1,
                "receipt_kind": MATERIALIZATION_KIND,
                "campaign_id": CAMPAIGN_ID,
                "frozen_contract": {
                    "reference": contract_reference.as_posix(),
                    "sha256": contract_sha,
                },
                "prepared_manifest": {
                    "reference": manifest_reference.as_posix(),
                    "file_sha256": manifest_file_sha,
                    "content_sha256": manifest["content_sha256"],
                    "artifact_id": manifest["artifact_id"],
                },
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "dataset_fingerprint": _mapping(
                    manifest["dataset"], "dataset"
                )["dataset_fingerprint"],
                "split_id": SPLIT_ID,
                "split_fingerprint": _mapping(manifest["split"], "split")[
                    "assignment_fingerprint"
                ],
                "counts": dict(EXPECTED_COUNTS),
                "jobs": receipt_jobs,
            }
        )
        _write_immutable(
            locked_root / "materialization_receipt.json", _json_bytes(receipt)
        )
        return receipt


def _load_materialization(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _strict_json(path, "materialization receipt")
    _verify_signed(receipt, "materialization receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != MATERIALIZATION_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("counts") != EXPECTED_COUNTS
        or receipt.get("dataset_id") != DATASET_ID
        or receipt.get("dataset_version") != DATASET_VERSION
        or receipt.get("split_id") != SPLIT_ID
    ):
        raise AdjacencyAblationEnqueueError(
            "materialization receipt differs from the frozen campaign"
        )
    frozen = _mapping(receipt.get("frozen_contract"), "receipt contract")
    contract_reference = _PROJECT_ROOT / str(frozen.get("reference", ""))
    if (
        frozen.get("sha256") != CONTRACT_SHA256
        or not contract_reference.is_file()
        or _sha256_file(contract_reference) != CONTRACT_SHA256
    ):
        raise AdjacencyAblationEnqueueError(
            "materialization frozen contract binding changed"
        )
    prepared = _mapping(
        receipt.get("prepared_manifest"), "receipt prepared manifest"
    )
    manifest_path = _PROJECT_ROOT / str(prepared.get("reference", ""))
    manifest, verified_path = _prepared_manifest(manifest_path)
    if (
        verified_path != manifest_path.resolve()
        or _sha256_file(verified_path) != prepared.get("file_sha256")
        or manifest.get("content_sha256") != prepared.get("content_sha256")
        or manifest.get("artifact_id") != prepared.get("artifact_id")
        or _mapping(manifest.get("dataset"), "dataset").get(
            "dataset_fingerprint"
        )
        != receipt.get("dataset_fingerprint")
        or _mapping(manifest.get("split"), "split").get(
            "assignment_fingerprint"
        )
        != receipt.get("split_fingerprint")
    ):
        raise AdjacencyAblationEnqueueError(
            "prepared artifact no longer matches materialization"
        )
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != sum(EXPECTED_COUNTS.values()):
        raise AdjacencyAblationEnqueueError(
            "materialization job inventory is incomplete"
        )
    observed_slots: set[tuple[str, int, int, str]] = set()
    validated: list[dict[str, Any]] = []
    for raw in jobs:
        job = dict(_mapping(raw, "materialized job"))
        stage = str(job.get("stage"))
        fold = int(job.get("fold", -1))
        seed = int(job.get("seed", -1))
        condition = str(job.get("condition"))
        slot = (stage, fold, seed, condition)
        if slot not in {
            (candidate_stage, candidate_fold, candidate_seed, candidate_condition)
            for candidate_stage in STAGES
            for candidate_fold, candidate_seed, candidate_condition in _stage_slots(
                candidate_stage
            )
        } or slot in observed_slots:
            raise AdjacencyAblationEnqueueError(
                "materialization has an unknown or duplicate job slot"
            )
        observed_slots.add(slot)
        config_path = _PROJECT_ROOT / str(job.get("config_reference", ""))
        if (
            not config_path.is_file()
            or _sha256_file(config_path) != job.get("config_file_sha256")
        ):
            raise AdjacencyAblationEnqueueError(
                f"locked config file changed for {slot}"
            )
        config = load_yaml_mapping(config_path)
        expected = _build_config(
            stage=stage,
            fold=fold,
            seed=seed,
            condition=condition,
            manifest=manifest,
            manifest_reference=Path(str(prepared["reference"])),
            manifest_file_sha256=str(prepared["file_sha256"]),
        )
        if (
            config != expected
            or canonical_sha256(config) != job.get("config_sha256")
            or scientific_id(config) != job.get("scientific_id")
            or job.get("requested_gpu") != _gpu_for(fold, seed)
        ):
            raise AdjacencyAblationEnqueueError(
                f"locked config payload or identity changed for {slot}"
            )
        job["config_payload"] = config
        validated.append(job)
    _assert_paired_configs(validated)
    receipt["jobs"] = validated
    return receipt, manifest


def _attempt_config(
    config: Mapping[str, Any], *, attempt: int
) -> tuple[dict[str, Any], str]:
    resolved = deepcopy(dict(config))
    resolved["attempt"] = int(attempt)
    return resolved, canonical_sha256(resolved)


def _load_recovery_authorization(
    plan_path: Path,
    enqueue_path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from recover_adjacency_gpu_failure import (
            load_recovery_enqueue,
            load_recovery_plan,
        )

        plan = dict(load_recovery_plan(plan_path))
        enqueue = dict(load_recovery_enqueue(enqueue_path, plan=plan))
    except Exception as exc:
        raise AdjacencyAblationEnqueueError(
            "CPU recovery plan/enqueue receipt verification failed"
        ) from exc
    contract_path = _PROJECT_ROOT / RECOVERY_CONTRACT_REFERENCE
    if (
        plan.get("receipt_kind") != RECOVERY_PLAN_KIND
        or enqueue.get("receipt_kind") != RECOVERY_ENQUEUE_KIND
        or plan.get("campaign_id") != CAMPAIGN_ID
        or enqueue.get("campaign_id") != CAMPAIGN_ID
        or _mapping(plan.get("materialization"), "recovery materialization").get(
            "checksum"
        )
        != materialization.get("checksum")
        or enqueue.get("plan_checksum") != plan.get("checksum")
        or not contract_path.is_file()
        or _sha256_file(contract_path) != RECOVERY_CONTRACT_SHA256
    ):
        raise AdjacencyAblationEnqueueError(
            "CPU recovery receipts differ from the frozen campaign"
        )
    materialized = {
        (str(item["stage"]), int(item["fold"]), int(item["seed"]), str(item["condition"])): item
        for item in materialization["jobs"]
    }
    retries = enqueue.get("primary_retries")
    if not isinstance(retries, list) or len(retries) != 50:
        raise AdjacencyAblationEnqueueError("recovery enqueue grid is incomplete")
    retry_by_slot = {
        (int(item["fold"]), int(item["seed"]), str(item["condition"])): item
        for item in retries
    }
    if len(retry_by_slot) != 50:
        raise AdjacencyAblationEnqueueError("recovery enqueue repeats a slot")
    primary: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in plan["primary_jobs"]:
        item = _mapping(raw, "recovery primary job")
        slot = (int(item["fold"]), int(item["seed"]), str(item["condition"]))
        base = materialized.get(("primary", *slot))
        retry = retry_by_slot.get(slot)
        if base is None or retry is None or slot in primary:
            raise AdjacencyAblationEnqueueError(
                "recovery primary slot is unauthorized"
            )
        resolved_config, resolved_sha = _attempt_config(
            _mapping(base.get("config_payload"), "materialized primary config"),
            attempt=2,
        )
        if (
            item.get("attempt") != 2
            or item.get("canonical_config_sha256") != base.get("config_sha256")
            or item.get("config_reference") != base.get("config_reference")
            or item.get("command_sha256")
            != canonical_sha256(command_for_config(resolved_config))
            or retry.get("retry_job_id") != item.get("retry_job_id")
            or retry.get("root_job_id") != item.get("root_job_id")
            or retry.get("retry_of") != item.get("root_job_id")
            or retry.get("attempt_count") != 2
            or retry.get("maximum_attempts") != 2
            or retry.get("canonical_config_sha256") != base.get("config_sha256")
            or retry.get("command_sha256") != item.get("command_sha256")
        ):
            raise AdjacencyAblationEnqueueError(
                f"recovery primary identity changed for {slot}"
            )
        primary[slot] = {
            "plan": dict(item),
            "enqueue": dict(retry),
            "materialized_job": base,
            "resolved_config": resolved_config,
            "resolved_config_sha256": resolved_sha,
        }
    expected_primary = {
        (fold, seed, condition)
        for fold in FOLDS for seed in SEEDS for condition in CONDITIONS
    }
    if set(primary) != expected_primary:
        raise AdjacencyAblationEnqueueError("recovery primary grid changed")
    conditional_null: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in plan["conditional_null_jobs"]:
        item = _mapping(raw, "recovery conditional null job")
        slot = (int(item["fold"]), int(item["seed"]), str(item["condition"]))
        base = materialized.get(("null", *slot))
        if base is None or slot in conditional_null:
            raise AdjacencyAblationEnqueueError(
                "conditional recovery null slot is unauthorized"
            )
        config = _mapping(base.get("config_payload"), "materialized null config")
        if (
            item.get("attempt") != 1
            or item.get("condition") != NULL_CONDITION
            or item.get("canonical_config_sha256") != base.get("config_sha256")
            or item.get("config_reference") != base.get("config_reference")
            or item.get("command_sha256")
            != canonical_sha256(command_for_config(config))
            or canonical_sha256(config) != base.get("config_sha256")
        ):
            raise AdjacencyAblationEnqueueError(
                f"conditional recovery null identity changed for {slot}"
            )
        conditional_null[slot] = {"plan": dict(item), "materialized_job": base}
    expected_null = {
        (fold, seed, NULL_CONDITION) for fold in FOLDS for seed in SEEDS
    }
    if set(conditional_null) != expected_null:
        raise AdjacencyAblationEnqueueError("conditional null recovery grid changed")
    return {
        "plan": plan,
        "enqueue": enqueue,
        "primary": primary,
        "null": conditional_null,
    }


def _load_pilot_gate(
    path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _strict_json(path, "paired pilot gate")
    _verify_signed(gate, "paired pilot gate")
    jobs = gate.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise AdjacencyAblationEnqueueError(
            "paired pilot gate must contain exactly two jobs"
        )
    expected = {
        str(item["condition"]): str(item["config_sha256"])
        for item in materialization["jobs"]
        if item["stage"] == "pilot"
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for raw in jobs:
        job = _mapping(raw, "paired pilot gate job")
        condition = str(job.get("condition"))
        if condition in observed:
            raise AdjacencyAblationEnqueueError(
                "paired pilot gate repeats a condition"
            )
        observed[condition] = job
        if (
            job.get("config_sha256") != expected.get(condition)
            or not str(job.get("run_id", "")).strip()
            or any(
                job.get(field) is not True
                for field in (
                    "finite_metrics",
                    "coverage_complete",
                    "update_count_verified",
                    "resource_limits_passed",
                )
            )
        ):
            raise AdjacencyAblationEnqueueError(
                f"paired pilot gate job {condition!r} failed verification"
            )
        _hex_digest(
            job.get("initial_state_sha256"),
            f"paired pilot {condition} initial state",
        )
        _hex_digest(
            job.get("training_mask_schedule_sha256"),
            f"paired pilot {condition} mask schedule",
        )
    if (
        gate.get("schema_version") != 1
        or gate.get("receipt_kind") != PILOT_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("frozen_contract_sha256") != CONTRACT_SHA256
        or gate.get("passed") is not True
        or gate.get("fold") != 0
        or gate.get("seed") != 0
        or set(observed) != set(CONDITIONS)
        or observed["spatial"].get("initial_state_sha256")
        != observed["isolated"].get("initial_state_sha256")
        or observed["spatial"].get("training_mask_schedule_sha256")
        != observed["isolated"].get("training_mask_schedule_sha256")
    ):
        raise AdjacencyAblationEnqueueError(
            "primary requires the exact passing paired pilot gate"
        )
    return gate


def _load_null_trigger(
    path: Path,
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    trigger = _strict_json(path, "null trigger receipt")
    _verify_signed(trigger, "null trigger receipt")
    jobs = trigger.get("primary_jobs")
    expected = _mapping(recovery.get("primary"), "recovery primary")
    observed: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    if not isinstance(jobs, list):
        raise AdjacencyAblationEnqueueError(
            "null trigger primary_jobs must be a list"
        )
    for raw in jobs:
        job = _mapping(raw, "null trigger primary job")
        slot = (
            int(job.get("fold", -1)),
            int(job.get("seed", -1)),
            str(job.get("condition")),
        )
        if slot in observed:
            raise AdjacencyAblationEnqueueError(
                "null trigger repeats a primary slot"
            )
        observed[slot] = job
        authorization = expected.get(slot)
        if (
            authorization is None
            or job.get("attempt") != 2
            or job.get("job_id")
            != _mapping(authorization, "recovery primary slot").get(
                "enqueue"
            )["retry_job_id"]
            or job.get("config_sha256")
            != _mapping(authorization, "recovery primary slot").get(
                "resolved_config_sha256"
            )
            or job.get("materialized_config_sha256")
            != _mapping(
                _mapping(authorization, "recovery primary slot").get(
                    "materialized_job"
                ),
                "materialized primary job",
            ).get("config_sha256")
            or not str(job.get("run_id", "")).strip()
        ):
            raise AdjacencyAblationEnqueueError(
                f"null trigger primary identity is invalid for {slot}"
            )
    difference = trigger.get("graph_minus_isolated_huber")
    if (
        trigger.get("schema_version") != 1
        or trigger.get("receipt_kind") != NULL_TRIGGER_KIND
        or trigger.get("campaign_id") != CAMPAIGN_ID
        or trigger.get("materialization_checksum")
        != materialization.get("checksum")
        or trigger.get("recovery_plan_checksum")
        != _mapping(recovery.get("plan"), "recovery plan").get("checksum")
        or trigger.get("recovery_enqueue_checksum")
        != _mapping(recovery.get("enqueue"), "recovery enqueue").get(
            "checksum"
        )
        or trigger.get("frozen_contract_sha256") != CONTRACT_SHA256
        or trigger.get("triggered") is not True
        or trigger.get("graph_huber_lower_than_isolated") is not True
        or trigger.get("scientific_audit_passed") is not True
        or trigger.get("scientific_audit_errors") != []
        or isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or float(difference) >= 0.0
        or set(observed) != set(expected)
    ):
        raise AdjacencyAblationEnqueueError(
            "null requires a favorable aggregate and complete 50-run identity"
        )
    return trigger


def _existing_jobs(registry: Registry) -> dict[str, Mapping[str, Any]]:
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
    for row in rows:
        item = dict(row)
        try:
            config = _mapping(
                json.loads(str(item["canonical_config_json"])),
                "existing queue config",
            )
            item["configuration"] = config
            item["command"] = json.loads(str(item["command_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AdjacencyAblationEnqueueError(
                "existing campaign queue row contains invalid JSON"
            ) from exc
        digest = canonical_sha256(config)
        if digest in result:
            raise AdjacencyAblationEnqueueError(
                "campaign contains duplicate root queue configurations"
            )
        result[digest] = item
    return result


def _verify_completed_gate_runs(
    *,
    registry: Registry,
    existing: Mapping[str, Mapping[str, Any]],
    gate_jobs: Sequence[Mapping[str, Any]],
) -> None:
    for gate_job in gate_jobs:
        digest = str(gate_job["config_sha256"])
        row = existing.get(digest)
        run_id = str(gate_job["run_id"])
        run = registry.get_run(run_id)
        if (
            row is None
            or row.get("status") != "completed"
            or row.get("run_id") != run_id
            or run is None
            or run.get("status") != "completed"
        ):
            raise AdjacencyAblationEnqueueError(
                f"gate run {run_id!r} is not a completed registered run"
            )


def _verify_completed_recovery_gate_runs(
    *,
    registry: Registry,
    recovery: Mapping[str, Any],
    gate_jobs: Sequence[Mapping[str, Any]],
) -> None:
    authorized = _mapping(recovery.get("primary"), "recovery primary")
    with registry.connect() as connection:
        retry_ids = {
            str(row["job_id"])
            for row in connection.execute(
                """
                SELECT job_id FROM queue_jobs
                WHERE campaign_id = ? AND retry_of IS NOT NULL
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        }
    expected_retry_ids = {
        str(_mapping(item, "recovery slot")["enqueue"]["retry_job_id"])
        for item in authorized.values()
    }
    if retry_ids != expected_retry_ids:
        raise AdjacencyAblationEnqueueError(
            "campaign contains missing or unauthorized primary retry rows"
        )
    for gate_job in gate_jobs:
        slot = (
            int(gate_job["fold"]),
            int(gate_job["seed"]),
            str(gate_job["condition"]),
        )
        authorization = _mapping(authorized.get(slot), f"recovery slot {slot}")
        planned = _mapping(authorization.get("plan"), "recovery plan job")
        queued = _mapping(authorization.get("enqueue"), "recovery enqueue job")
        retry_job_id = str(queued["retry_job_id"])
        row = registry.get_job(retry_job_id)
        run_id = str(gate_job["run_id"])
        run = registry.get_run(run_id)
        root = registry.get_job(str(planned["root_job_id"]))
        if (
            row is None
            or root is None
            or row.get("status") != "completed"
            or row.get("run_id") != run_id
            or row.get("retry_of") != planned.get("root_job_id")
            or int(row.get("attempt_count", -1)) != 2
            or int(row.get("maximum_attempts", -1)) != 2
            or str(row.get("requested_gpu")) != str(planned.get("worker_slot"))
            or row.get("canonical_config") != root.get("canonical_config")
            or run is None
            or run.get("status") != "completed"
            or int(run.get("attempt", -1)) != 2
            or run.get("retry_of") != root.get("run_id")
            or canonical_sha256(_mapping(run.get("config"), "retry run config"))
            != authorization.get("resolved_config_sha256")
            or gate_job.get("job_id") != retry_job_id
            or gate_job.get("config_sha256")
            != authorization.get("resolved_config_sha256")
        ):
            raise AdjacencyAblationEnqueueError(
                f"recovery gate run {run_id!r} is not the authorized completed retry"
            )


def enqueue_stage(
    *,
    stage: str,
    materialization_path: Path,
    database_path: Path,
    receipt_path: Path,
    pilot_gate_path: Path,
    null_trigger_path: Path,
    recovery_plan_path: Path | None = None,
    recovery_enqueue_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise AdjacencyAblationEnqueueError(
            "enqueue stage must be smoke, pilot, primary, or null"
        )
    locked_root = materialization_path.parent
    _project_reference(materialization_path, "materialization receipt")
    if receipt_path.resolve().parent != locked_root.resolve():
        raise AdjacencyAblationEnqueueError(
            "enqueue receipts must remain in the locked campaign directory"
        )
    with _campaign_lock(locked_root):
        materialization, manifest = _load_materialization(materialization_path)
        recovery = None
        if stage == "null":
            if recovery_plan_path is None or recovery_enqueue_path is None:
                raise AdjacencyAblationEnqueueError(
                    "null enqueue requires explicit CPU recovery receipts"
                )
            recovery = _load_recovery_authorization(
                recovery_plan_path,
                recovery_enqueue_path,
                materialization=materialization,
            )
        pilot_gate = (
            _load_pilot_gate(
                pilot_gate_path, materialization=materialization
            )
            if stage == "primary"
            else None
        )
        null_trigger = (
            _load_null_trigger(
                null_trigger_path,
                materialization=materialization,
                recovery=_mapping(recovery, "recovery authorization"),
            )
            if stage == "null"
            else None
        )
        registry = Registry(database_path)
        registry.initialize()
        _registry_contract(registry, manifest)
        existing = _existing_jobs(registry)
        if pilot_gate is not None:
            _verify_completed_gate_runs(
                registry=registry,
                existing=existing,
                gate_jobs=pilot_gate["jobs"],
            )
        if null_trigger is not None:
            _verify_completed_recovery_gate_runs(
                registry=registry,
                recovery=_mapping(recovery, "recovery authorization"),
                gate_jobs=null_trigger["primary_jobs"],
            )

        all_jobs = [
            dict(_mapping(item, "materialized job"))
            for item in materialization["jobs"]
        ]
        planned_digests = {str(item["config_sha256"]) for item in all_jobs}
        if set(existing).difference(planned_digests):
            raise AdjacencyAblationEnqueueError(
                "campaign contains an unexpected root queue configuration"
            )
        selected = [item for item in all_jobs if item["stage"] == stage]
        if len(selected) != EXPECTED_COUNTS[stage]:
            raise AdjacencyAblationEnqueueError(
                f"{stage} materialization run count changed"
            )
        receipt_jobs: list[dict[str, Any]] = []
        for item in selected:
            config = _mapping(item["config_payload"], "locked config")
            digest = str(item["config_sha256"])
            gpu = int(item["requested_gpu"])
            if stage == "null":
                authorization = _mapping(
                    _mapping(recovery, "recovery authorization")["null"].get(
                        (int(item["fold"]), int(item["seed"]), str(item["condition"]))
                    ),
                    "conditional recovery null slot",
                )
                gpu = int(_mapping(authorization["plan"], "null plan job")["worker_slot"])
            reference = Path(str(item["config_reference"]))
            command = command_for_config(config)
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
                    command=command,
                    experiment_config_reference=reference,
                    priority=STAGE_PRIORITY[stage],
                    maximum_attempts=MAXIMUM_ATTEMPTS,
                    requested_gpu=str(gpu),
                )
                existing[digest] = row
            elif (
                int(row.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
                or int(row.get("attempt_count", -1)) != 1
                or row.get("retry_of") is not None
                or str(row.get("requested_gpu")) != str(gpu)
                or int(row.get("priority", -1)) != STAGE_PRIORITY[stage]
                or str(row.get("experiment_config_reference")) != str(reference)
                or row.get("command") != command
            ):
                raise AdjacencyAblationEnqueueError(
                    "existing job has a changed command, GPU, priority, "
                    "reference, or attempt budget"
                )
            receipt_jobs.append(
                {
                    "stage": stage,
                    "fold": item["fold"],
                    "seed": item["seed"],
                    "condition": item["condition"],
                    "config_sha256": digest,
                    "scientific_id": item["scientific_id"],
                    "job_id": str(row["job_id"]),
                    "requested_gpu": gpu,
                    "maximum_attempts": MAXIMUM_ATTEMPTS,
                }
            )
        receipt_payload: dict[str, Any] = {
            "schema_version": 1,
            "receipt_kind": f"adjacency_ablation_{stage}_enqueue_v1",
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "materialization_checksum": materialization["checksum"],
            "pilot_gate_checksum": (
                None if pilot_gate is None else pilot_gate["checksum"]
            ),
            "null_trigger_checksum": (
                None if null_trigger is None else null_trigger["checksum"]
            ),
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "complete": True,
            "jobs": receipt_jobs,
        }
        # Preserve the already-signed smoke, pilot, and primary receipt schema.
        # Only the conditionally enqueued null stage is part of CPU recovery.
        if recovery is not None:
            receipt_payload.update(
                {
                    "recovery_plan_checksum": _mapping(
                        recovery["plan"], "recovery plan"
                    )["checksum"],
                    "recovery_enqueue_checksum": _mapping(
                        recovery["enqueue"], "recovery enqueue"
                    )["checksum"],
                    "execution_device": "cpu",
                }
            )
        receipt = _signed(receipt_payload)
        _write_immutable(receipt_path, _json_bytes(receipt))
        return receipt


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("materialize", *STAGES))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            paths.data_root
            / "processed/adjacent_normal_grouped_adjacency_ablation_v1/manifest.json"
        ),
    )
    parser.add_argument("--locked-root", type=Path, default=locked)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "materialization_receipt.json",
    )
    parser.add_argument(
        "--pilot-gate",
        type=Path,
        default=locked / "pilot_gate_receipt.json",
    )
    parser.add_argument(
        "--null-trigger",
        type=Path,
        default=(
            paths.project_root
            / "reports/analyses/adjacent_normal_grouped_adjacency_ablation"
            / "null_trigger_receipt.json"
        ),
    )
    parser.add_argument(
        "--recovery-plan",
        type=Path,
        default=locked / RECOVERY_PLAN_FILENAME,
    )
    parser.add_argument(
        "--recovery-enqueue",
        type=Path,
        default=locked / RECOVERY_ENQUEUE_FILENAME,
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
    if args.stage == "materialize":
        result = materialize(
            manifest_path=args.manifest.resolve(),
            locked_root=args.locked_root.resolve(),
            database_path=args.database.resolve(),
        )
        receipt_path = args.locked_root.resolve() / "materialization_receipt.json"
    else:
        receipt_path = (
            args.receipt.resolve()
            if args.receipt is not None
            else args.locked_root.resolve()
            / f"{args.stage}_enqueue_receipt.json"
        )
        result = enqueue_stage(
            stage=args.stage,
            materialization_path=args.materialization.resolve(),
            database_path=args.database.resolve(),
            receipt_path=receipt_path,
            pilot_gate_path=args.pilot_gate.resolve(),
            null_trigger_path=args.null_trigger.resolve(),
            recovery_plan_path=args.recovery_plan.resolve(),
            recovery_enqueue_path=args.recovery_enqueue.resolve(),
        )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "stage": args.stage,
                "complete": True,
                "job_count": len(result["jobs"]),
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

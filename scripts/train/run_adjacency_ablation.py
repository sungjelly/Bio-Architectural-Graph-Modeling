#!/usr/bin/env python3
"""Run one queue-owned member of the grouped adjacency ablation.

Only the explicit adjacency tensor varies among conditions.  The model,
initialization, masking schedule, preprocessing, optimizer, update count, and
evaluation targets are otherwise shared literally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.adjacency_ablation import (  # noqa: E402
    ADJACENCY_ARMS,
    CORE_ALIASES,
    ExplicitSelfMeanGraphSAGE,
    NEIGHBOR_OBSERVED_BIN_ORDER,
    TARGET_MASK_BIN_ORDER,
    build_seeded_explicit_self_model,
    derive_evaluation_mask_seed,
    derive_training_mask_seed,
    evaluate_fixed_mask_strata,
    mask_realization_sha256,
    masked_regression_summary,
    ndarray_sha256,
    sample_uniform_mask_torch,
    state_dict_sha256,
    trainable_parameter_count,
    validate_explicit_self_adjacency,
)
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.metrics import masked_huber_loss  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)


CAMPAIGN_ID = "cmp_20260802_adjacent_normal_grouped_adjacency_ablation"
CONTRACT_SHA256 = (
    "08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff"
)
CONTRACT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "frozen_task_contract.yaml"
)
RECOVERY_CONTRACT_SHA256 = (
    "84230db441cd03d158a28bc79dfecee495fee1b0864dae2311a0a9041f7b6bb0"
)
RECOVERY_CONTRACT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "hardware_recovery_contract.yaml"
)
MATERIALIZATION_CHECKSUM = (
    "0cc606d73a4979808581a032138cc18ef629952ed1a4884903ecd229bdfc4303"
)
LOCKED_CAMPAIGN_RELATIVE = Path("scratch/locked_campaigns") / CAMPAIGN_ID
MATERIALIZATION_RELATIVE = LOCKED_CAMPAIGN_RELATIVE / "materialization_receipt.json"
CPU_RECOVERY_PLAN_RELATIVE = (
    LOCKED_CAMPAIGN_RELATIVE / "primary_cpu_recovery_plan_receipt.json"
)
CPU_RECOVERY_PLAN_KIND = "adjacency_ablation_cpu_recovery_plan_v1"
CPU_RECOVERY_ENQUEUE_CHECKSUM = (
    "ed849d49e0b5d62e7bbfce1b50ecff5c86fe65f59927dc9adda00523dc1873ca"
)
NULL_ENQUEUE_RELATIVE = LOCKED_CAMPAIGN_RELATIVE / "null_enqueue_receipt.json"
NULL_ENQUEUE_KIND = "adjacency_ablation_null_enqueue_v1"
CPU_RECOVERY_INTRAOP_THREADS = 4
CPU_RECOVERY_INTEROP_THREADS = 1
EXPECTED_GENES = 1000
EXPECTED_PARAMETER_COUNT = 645_736
EXPECTED_DATASET_ID = "cosmx_adjacent_normal_grouped_adjacency_v1"
EXPECTED_DATASET_VERSION = "adjacent_normal_grouped_adjacency_v1"
EXPECTED_SPLIT_ID = "adjacent_normal_10donor_slide_balanced_5fold_v1"
EXPECTED_PREPARED_MANIFEST = (
    "data/processed/adjacent_normal_grouped_adjacency_ablation_v1/manifest.json"
)
EXPECTED_PREPARED_MANIFEST_SHA256 = (
    "bc469fb6a5d7ec398428cb458c42bf87e4de35b507edfd550b45b1bc27c4910c"
)
EXPECTED_PREPARED_CONTENT_SHA256 = (
    "a4793720a6e148e829c906577bb51b0d4dba6a3e5668f18be2ce3acf51946392"
)
EXPECTED_DATASET_FINGERPRINT = (
    "d125df1626646079e4d28e3339851c4d45a937b30f9052759c49dbe2aff100f3"
)
EXPECTED_SPLIT_FINGERPRINT = (
    "e0b80b9bbb715f27ccb30e6496b65e9c2ef74b1b901f66b1c2a81ff08d154580"
)
EXPECTED_PREPROCESSING_FINGERPRINT = (
    "4aef8ec895e851abd5131281329259e2db49df12d7ae44b50cf3d0a27808daed"
)
PRIMARY_METRIC = "val/masked_huber"
EXPECTED_ALIASES = tuple(CORE_ALIASES)
CONDITION_TO_ARRAY = {
    "spatial": "spatial_edge_index",
    "isolated": "isolated_edge_index",
    "position_permuted_null": "position_permuted_null_edge_index",
}
EXPECTED_EPOCHS = {"smoke": 1, "pilot": 5, "primary": 80, "null": 80}
PILOT_MAX_VRAM_GIB = 20.5
PILOT_MAX_HOST_GIB = 40.0
PILOT_MAX_PROJECTED_FULL_SECONDS = 45 * 60


class AdjacencyRunnerError(RuntimeError):
    """Raised before a run can depart from the frozen comparison."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjacencyRunnerError(f"{label} must be a mapping")
    return value


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(config.get(name), f"config.{name}")


def _project_path(value: object, *, label: str) -> Path:
    reference = Path(str(value))
    if reference.is_absolute():
        raise AdjacencyRunnerError(f"{label} must be project-relative")
    root = current_paths().project_root.resolve()
    resolved = (root / reference).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AdjacencyRunnerError(f"{label} escapes the project root") from error
    return resolved


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AdjacencyRunnerError(
            f"{label}={actual!r}, expected frozen value {expected!r}"
        )


def _validate_config(config: Mapping[str, Any]) -> tuple[str, str, int, int]:
    validate_experiment_config(config)
    campaign = _section(config, "campaign")
    _require_equal(campaign.get("campaign_id"), CAMPAIGN_ID, "campaign_id")
    _require_equal(
        campaign.get("frozen_contract_sha256"),
        CONTRACT_SHA256,
        "campaign.frozen_contract_sha256",
    )
    experiment = _section(config, "experiment")
    stage = str(experiment.get("stage", ""))
    if stage not in EXPECTED_EPOCHS:
        raise AdjacencyRunnerError(f"unknown experiment stage: {stage!r}")
    if "condition" in experiment:
        raise AdjacencyRunnerError(
            "experiment.condition is prohibited; adjacency is the sole arm field"
        )
    graph = _section(config, "graph")
    condition = str(graph.get("adjacency_condition", ""))
    if condition not in ADJACENCY_ARMS:
        raise AdjacencyRunnerError(f"unknown adjacency condition: {condition!r}")
    if stage != "null" and condition == "position_permuted_null":
        raise AdjacencyRunnerError("the null adjacency is restricted to null stage")
    if stage == "null" and condition != "position_permuted_null":
        raise AdjacencyRunnerError("null stage requires position_permuted_null")

    fold = int(config.get("fold", -1))
    seed = int(config.get("seed", -1))
    if fold not in range(5) or seed not in range(5):
        raise AdjacencyRunnerError("fold and seed must both be in 0 through 4")
    if stage in {"smoke", "pilot"} and (fold, seed) != (0, 0):
        raise AdjacencyRunnerError("diagnostic stages are fixed to fold 0 seed 0")
    attempt = config.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise AdjacencyRunnerError("attempt must be an integer")
    allowed_attempts = {1, 2} if stage == "primary" else {1}
    if attempt not in allowed_attempts:
        raise AdjacencyRunnerError(
            f"attempt={attempt!r} is not allowed for stage {stage!r}"
        )

    model = _section(config, "model")
    expected_model = {
        "name": "mean-adjacency-sage",
        "family": "explicit_self_mean_adjacency_graphsage",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "ffn_dim": 256,
        "decoder_dim": 256,
        "graph_layers": 1,
        "dropout": 0.1,
    }
    for key, expected in expected_model.items():
        _require_equal(model.get(key), expected, f"model.{key}")
    features = _section(config, "features")
    _require_equal(features.get("use_edge_features"), False, "edge features")

    dataset = _section(config, "dataset")
    for key, expected in {
        "dataset_id": EXPECTED_DATASET_ID,
        "version": EXPECTED_DATASET_VERSION,
        "split_id": EXPECTED_SPLIT_ID,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "preprocessing_version": EXPECTED_PREPROCESSING_FINGERPRINT,
        "prepared_manifest": EXPECTED_PREPARED_MANIFEST,
        "prepared_manifest_sha256": EXPECTED_PREPARED_MANIFEST_SHA256,
        "prepared_content_sha256": EXPECTED_PREPARED_CONTENT_SHA256,
        "task": "masked_expression_regression",
        "target_scale": "standardized_gene_wise_log1p_raw_count",
        "biological_probe_count": 1000,
    }.items():
        _require_equal(dataset.get(key), expected, f"dataset.{key}")

    preprocessing = _section(config, "preprocessing")
    for key, expected in {
        "expression": "gene_wise_standardized_log1p_raw_count",
        "fit_scope": "seven_training_cores_only_equal_core_moments",
        "scale_floor": 1.0e-6,
        "feature_selection": "none",
        "library_size_normalization": "none",
        "preprocessing_fingerprint": EXPECTED_PREPROCESSING_FINGERPRINT,
    }.items():
        _require_equal(preprocessing.get(key), expected, f"preprocessing.{key}")

    for key, expected in {
        "neighbor_k": 12,
        "radius_um": 50.0,
        "symmetry": "union",
        "grouping": "raw_fov_within_core",
        "adjacency_condition": condition,
    }.items():
        _require_equal(graph.get(key), expected, f"graph.{key}")

    masking = _section(config, "masking")
    for key, expected in {
        "type": "exact_uniform_count_expression_masking",
        "training_base_seed": 2026080201,
        "evaluation_base_seed": 20260802,
        "evaluation_replicates": 3,
        "mask_indicator": True,
    }.items():
        _require_equal(masking.get(key), expected, f"masking.{key}")

    trainer = _section(config, "trainer")
    expected_trainer = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
        "batch_size": 1,
        "max_epochs": EXPECTED_EPOCHS[stage],
        "validation_interval_epochs": 5,
        "early_stopping": False,
        "restore_best": True,
        "primary_checkpoint_role": "best",
        "checkpoint_policy": "minimum_fixed_validation_huber",
    }
    for key, expected in expected_trainer.items():
        _require_equal(trainer.get(key), expected, f"trainer.{key}")

    evaluation = _section(config, "evaluation")
    for key, expected in {
        "protocol": "grouped_core_adjacency_ablation_v1",
        "task_family": "masked_expression_regression",
        "primary_metric": PRIMARY_METRIC,
        "primary_direction": "minimize",
        "canonical_prediction_split": "validation",
        "splits": ["validation", "test"],
        "fixed_mask_replicates": 3,
    }.items():
        _require_equal(evaluation.get(key), expected, f"evaluation.{key}")
    return stage, condition, fold, seed


def _seed_all(seed: int, *, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed() also calls every accelerator backend.  Recovery runs
    # are deliberately CPU-only even if the host later exposes CUDA again, so
    # seed the CPU generator directly and touch CUDA only for a CUDA execution.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_runner_model(*, seed: int, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda":
        return build_seeded_explicit_self_model(
            EXPECTED_GENES,
            seed=seed,
            hidden_dim=128,
            ffn_dim=256,
            decoder_dim=256,
            dropout=0.1,
        )
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(int(seed))
        return ExplicitSelfMeanGraphSAGE(
            EXPECTED_GENES,
            hidden_dim=128,
            ffn_dim=256,
            decoder_dim=256,
            dropout=0.1,
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _epoch_core_order(aliases: Sequence[str], *, fold: int, seed: int, epoch: int) -> tuple[str, ...]:
    rng = np.random.default_rng(
        _stable_seed("adjacency-ablation-core-order-v1", fold, seed, epoch)
    )
    order = np.asarray(list(aliases), dtype="U8")
    rng.shuffle(order)
    return tuple(str(value) for value in order.tolist())


@dataclass
class CoreData:
    alias: str
    standardized_expression: np.ndarray
    edge_index: np.ndarray
    true_spatial_edge_index: np.ndarray
    fov_group: np.ndarray
    qc_passed: np.ndarray
    evaluation_mask_reference: Path
    data_sha256: str
    adjacency_sha256: str
    selected_adjacency_sha256: str
    masks_sha256: str

    @property
    def n_cells(self) -> int:
        return int(self.standardized_expression.shape[0])


@dataclass
class PreparedFold:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    fold: int
    train_aliases: tuple[str, ...]
    validation_aliases: tuple[str, ...]
    test_aliases: tuple[str, ...]
    gene_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    preprocessing_sha256: str
    cores: dict[str, CoreData]


def _load_prepared_fold(
    config: Mapping[str, Any], *, condition: str, fold: int
) -> PreparedFold:
    dataset = _section(config, "dataset")
    manifest_ref = dataset.get("prepared_manifest")
    if manifest_ref is None:
        manifest_ref = dataset.get("prepared_artifact_reference")
        if manifest_ref is not None:
            manifest_ref = str(Path(str(manifest_ref)) / "manifest.json")
    if manifest_ref is None:
        raise AdjacencyRunnerError("dataset prepared manifest is missing")
    manifest_path = _project_path(manifest_ref, label="dataset.prepared_manifest")
    manifest_file_sha = sha256_file(manifest_path)
    _require_equal(
        manifest_file_sha,
        EXPECTED_PREPARED_MANIFEST_SHA256,
        "dataset.prepared_manifest_sha256",
    )

    # The preparation script owns the complete content-hash and NPZ schema
    # verifier.  Importing it here avoids a second, subtly divergent contract.
    from scripts.train.prepare_adjacency_ablation import verify_prepared_artifact

    manifest = dict(verify_prepared_artifact(manifest_path))
    _require_equal(manifest.get("schema_version"), 1, "prepared schema")
    content_sha = str(manifest.get("content_sha256", ""))
    _require_equal(
        content_sha, EXPECTED_PREPARED_CONTENT_SHA256, "prepared content hash"
    )
    manifest_dataset = _mapping(manifest.get("dataset"), "prepared.dataset")
    _require_equal(
        manifest_dataset.get("dataset_id"),
        "cosmx_adjacent_normal_grouped_adjacency_v1",
        "prepared dataset_id",
    )
    _require_equal(manifest_dataset.get("n_genes"), EXPECTED_GENES, "gene count")
    _require_equal(
        manifest_dataset.get("dataset_fingerprint"),
        EXPECTED_DATASET_FINGERPRINT,
        "prepared dataset fingerprint",
    )
    prepared_split = _mapping(manifest.get("split"), "prepared.split")
    _require_equal(prepared_split.get("split_id"), EXPECTED_SPLIT_ID, "prepared split_id")
    _require_equal(
        prepared_split.get("assignment_fingerprint"),
        EXPECTED_SPLIT_FINGERPRINT,
        "prepared split fingerprint",
    )
    prepared_preprocessing = _mapping(
        manifest.get("preprocessing"), "prepared.preprocessing"
    )
    _require_equal(
        prepared_preprocessing.get("preprocessing_fingerprint"),
        EXPECTED_PREPROCESSING_FINGERPRINT,
        "prepared preprocessing fingerprint",
    )
    features = _mapping(manifest.get("features"), "prepared.features")
    gene_names = tuple(str(value) for value in features.get("gene_names", ()))
    if len(gene_names) != EXPECTED_GENES or len(set(gene_names)) != EXPECTED_GENES:
        raise AdjacencyRunnerError("prepared ordered gene schema is invalid")
    if any(
        name.lower().startswith(("negative", "systemcontrol"))
        for name in gene_names
    ):
        raise AdjacencyRunnerError("technical probes remain in biological targets")

    root = manifest_path.parent
    files = _mapping(manifest.get("files"), "prepared.files")
    fold_record = _mapping(
        _mapping(manifest.get("folds"), "prepared.folds").get(str(fold)),
        f"prepared fold {fold}",
    )
    train_aliases = tuple(str(x) for x in fold_record["train_aliases"])
    validation_aliases = tuple(str(x) for x in fold_record["validation_aliases"])
    test_aliases = tuple(str(x) for x in fold_record["test_aliases"])
    if (
        len(train_aliases),
        len(validation_aliases),
        len(test_aliases),
    ) != (7, 1, 2):
        raise AdjacencyRunnerError("prepared fold is not 7/1/2")
    if set(train_aliases + validation_aliases + test_aliases) != set(
        EXPECTED_ALIASES
    ):
        raise AdjacencyRunnerError("prepared fold does not partition ten aliases")
    preprocess_ref = Path(str(fold_record["preprocessing_reference"]))
    preprocess_path = (root / preprocess_ref).resolve()
    try:
        preprocess_path.relative_to(root.resolve())
    except ValueError as error:
        raise AdjacencyRunnerError("preprocessing reference escapes artifact") from error
    if sha256_file(preprocess_path) != str(files[preprocess_ref.as_posix()]):
        raise AdjacencyRunnerError("preprocessing file hash mismatch")
    with np.load(preprocess_path, allow_pickle=False) as payload:
        if set(payload.files) != {"log1p_mean", "log1p_scale"}:
            raise AdjacencyRunnerError("unexpected preprocessing arrays")
        mean = np.asarray(payload["log1p_mean"], dtype=np.float64)
        scale = np.asarray(payload["log1p_scale"], dtype=np.float64)
    if mean.shape != (EXPECTED_GENES,) or scale.shape != mean.shape:
        raise AdjacencyRunnerError("preprocessing vector shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < 1e-6):
        raise AdjacencyRunnerError("preprocessing vectors are invalid")

    cores_record = _mapping(manifest.get("cores"), "prepared.cores")
    cores: dict[str, CoreData] = {}
    for alias in EXPECTED_ALIASES:
        record = _mapping(cores_record.get(alias), f"prepared core {alias}")
        data_ref = Path(str(record["data_reference"]))
        adjacency_ref = Path(str(record["adjacency_reference"]))
        mask_ref = Path(str(record["evaluation_mask_reference"]))
        for reference in (data_ref, adjacency_ref, mask_ref):
            if reference.as_posix() not in files:
                raise AdjacencyRunnerError(f"unregistered prepared file: {reference}")
        data_path = (root / data_ref).resolve()
        adjacency_path = (root / adjacency_ref).resolve()
        mask_path = (root / mask_ref).resolve()
        for path in (data_path, adjacency_path, mask_path):
            try:
                path.relative_to(root.resolve())
            except ValueError as error:
                raise AdjacencyRunnerError("prepared reference escapes root") from error
        data_sha = sha256_file(data_path)
        adjacency_sha = sha256_file(adjacency_path)
        masks_sha = sha256_file(mask_path)
        if data_sha != str(files[data_ref.as_posix()]):
            raise AdjacencyRunnerError(f"{alias} data hash mismatch")
        if adjacency_sha != str(files[adjacency_ref.as_posix()]):
            raise AdjacencyRunnerError(f"{alias} adjacency hash mismatch")
        if masks_sha != str(files[mask_ref.as_posix()]):
            raise AdjacencyRunnerError(f"{alias} mask hash mismatch")
        with np.load(data_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "expression_counts",
                "coordinates_um",
                "fov_group",
                "qc_passed",
            }:
                raise AdjacencyRunnerError(f"unexpected {alias} data arrays")
            counts = np.asarray(payload["expression_counts"])
            coordinates = np.asarray(payload["coordinates_um"], dtype=np.float64)
            fov = np.asarray(payload["fov_group"])
            qc = np.asarray(payload["qc_passed"], dtype=np.bool_)
        if counts.ndim != 2 or counts.shape[1] != EXPECTED_GENES:
            raise AdjacencyRunnerError(f"{alias} count shape mismatch")
        if coordinates.shape != (counts.shape[0], 2) or fov.shape != (
            counts.shape[0],
        ) or qc.shape != (counts.shape[0],):
            raise AdjacencyRunnerError(f"{alias} routing arrays are misaligned")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise AdjacencyRunnerError(f"{alias} counts are not nonnegative integers")
        standardized = (
            np.log1p(counts.astype(np.float64, copy=False)) - mean
        ) / scale
        standardized = np.asarray(standardized, dtype=np.float32)
        if not np.isfinite(standardized).all():
            raise AdjacencyRunnerError(f"{alias} standardized expression is nonfinite")
        with np.load(adjacency_path, allow_pickle=False) as payload:
            expected_arrays = set(CONDITION_TO_ARRAY.values())
            if set(payload.files) != expected_arrays:
                raise AdjacencyRunnerError(f"unexpected {alias} adjacency arrays")
            edge_index = np.asarray(payload[CONDITION_TO_ARRAY[condition]], dtype=np.int64)
            spatial = np.asarray(payload["spatial_edge_index"], dtype=np.int64)
        edge_tensor = torch.from_numpy(np.array(edge_index, copy=True))
        validate_explicit_self_adjacency(edge_tensor, num_nodes=counts.shape[0])
        if np.any(fov[edge_index[0]] != fov[edge_index[1]]):
            raise AdjacencyRunnerError(f"{alias} contains a cross-FOV model edge")
        true_spatial = spatial[:, spatial[0] != spatial[1]]
        if np.any(fov[true_spatial[0]] != fov[true_spatial[1]]):
            raise AdjacencyRunnerError(f"{alias} true graph contains a cross-FOV edge")
        cores[alias] = CoreData(
            alias=alias,
            standardized_expression=standardized,
            edge_index=edge_index,
            true_spatial_edge_index=true_spatial,
            fov_group=fov,
            qc_passed=qc,
            evaluation_mask_reference=mask_path,
            data_sha256=data_sha,
            adjacency_sha256=adjacency_sha,
            selected_adjacency_sha256=ndarray_sha256(edge_index),
            masks_sha256=masks_sha,
        )

    return PreparedFold(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_file_sha,
        content_sha256=content_sha,
        fold=fold,
        train_aliases=train_aliases,
        validation_aliases=validation_aliases,
        test_aliases=test_aliases,
        gene_names=gene_names,
        mean=mean,
        scale=scale,
        preprocessing_sha256=sha256_file(preprocess_path),
        cores=cores,
    )


def _fixed_masks(core: CoreData) -> list[tuple[np.ndarray, np.ndarray, int, str]]:
    with np.load(core.evaluation_mask_reference, allow_pickle=False) as payload:
        if set(payload.files) != {"packed_masks", "masked_counts", "mask_seeds"}:
            raise AdjacencyRunnerError(f"unexpected mask arrays for {core.alias}")
        packed = np.asarray(payload["packed_masks"], dtype=np.uint8)
        counts = np.asarray(payload["masked_counts"], dtype=np.int64)
        seeds = np.asarray(payload["mask_seeds"], dtype=np.uint64)
    expected_shape = (3, core.n_cells, EXPECTED_GENES // 8)
    if packed.shape != expected_shape or counts.shape != (3, core.n_cells) or seeds.shape != (3,):
        raise AdjacencyRunnerError(f"fixed mask bundle shape mismatch for {core.alias}")
    masks = np.unpackbits(packed, axis=-1, count=EXPECTED_GENES, bitorder="little").astype(
        np.bool_, copy=False
    )
    output: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    for replicate in range(3):
        expected_seed = derive_evaluation_mask_seed(
            core_alias=core.alias, replicate_index=replicate
        )
        seed = int(seeds[replicate])
        if seed != expected_seed:
            raise AdjacencyRunnerError(f"fixed mask seed mismatch for {core.alias}")
        if not np.array_equal(
            masks[replicate].sum(axis=1, dtype=np.int64), counts[replicate]
        ):
            raise AdjacencyRunnerError(f"fixed mask counts mismatch for {core.alias}")
        checksum = mask_realization_sha256(
            masks[replicate], counts[replicate], seed=seed
        )
        output.append((masks[replicate], counts[replicate], seed, checksum))
    return output


def _worker_archive_and_config(
    args: argparse.Namespace,
) -> tuple[RunArchive, dict[str, Any]]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise AdjacencyRunnerError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required; use the queue worker"
        )
    supplied = args.run_scratch.resolve(strict=False)
    if supplied != Path(environment_scratch).resolve(strict=False):
        raise AdjacencyRunnerError("--run-scratch does not match BAGM_RUN_SCRATCH")
    expected_config = supplied / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise AdjacencyRunnerError("--config must be the worker-owned config")
    archive = RunArchive.attach_active(
        run_id, paths=current_paths(), scratch_path=supplied
    )
    config = dict(load_yaml_mapping(expected_config))
    _validate_config(config)
    return archive, config


@dataclass
class DeviceCore:
    expression: torch.Tensor
    edge_index: torch.Tensor


def _device_cache(data: PreparedFold, device: torch.device) -> dict[str, DeviceCore]:
    cache: dict[str, DeviceCore] = {}
    for alias, core in data.cores.items():
        cache[alias] = DeviceCore(
            expression=torch.from_numpy(
                np.array(core.standardized_expression, copy=False)
            ).to(device=device, dtype=torch.float32),
            edge_index=torch.from_numpy(np.array(core.edge_index, copy=True)).to(
                device=device, dtype=torch.long
            ),
        )
    return cache


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _standardized_huber_stats(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[float, int]:
    selected = np.asarray(mask, dtype=np.bool_)
    count = int(selected.sum())
    if count <= 0:
        raise AdjacencyRunnerError("evaluation mask selects no entries")
    difference = np.asarray(prediction, dtype=np.float64)[selected] - np.asarray(
        target, dtype=np.float64
    )[selected]
    absolute = np.abs(difference)
    huber = np.where(
        absolute <= 1.0,
        0.5 * np.square(difference),
        absolute - 0.5,
    )
    return float(huber.mean()), count


def _validation_huber(
    model: torch.nn.Module,
    data: PreparedFold,
    cache: Mapping[str, DeviceCore],
    *,
    device: torch.device,
) -> tuple[float, list[str]]:
    model.eval()
    replicate_losses: list[float] = []
    checksums: list[str] = []
    with torch.no_grad():
        for alias in data.validation_aliases:
            core = data.cores[alias]
            resident = cache[alias]
            for mask_np, _counts, _seed, checksum in _fixed_masks(core):
                mask = torch.from_numpy(np.array(mask_np, copy=True)).to(
                    device=device, dtype=torch.bool
                )
                prediction = model(
                    resident.expression, mask, resident.edge_index
                ).prediction
                loss_sum = masked_huber_loss(
                    resident.expression,
                    prediction,
                    mask,
                    reduction="sum",
                )
                count = int(mask.sum().item())
                replicate_losses.append(float(loss_sum.item()) / count)
                checksums.append(checksum)
                del mask, prediction, loss_sum
    if not replicate_losses or not all(
        math.isfinite(value) for value in replicate_losses
    ):
        raise FloatingPointError("validation Huber is nonfinite or unsupported")
    return float(np.mean(replicate_losses)), checksums


@dataclass
class TrainingResult:
    model: torch.nn.Module
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_huber: float
    optimizer_steps: int
    initial_state_sha256: str
    best_state_sha256: str
    training_mask_schedule_sha256: str
    evaluation_mask_schedule_sha256: str
    all_gradients_finite: bool
    elapsed_seconds: float


def _train(
    model: torch.nn.Module,
    data: PreparedFold,
    cache: Mapping[str, DeviceCore],
    config: Mapping[str, Any],
    *,
    fold: int,
    model_seed: int,
    device: torch.device,
) -> TrainingResult:
    trainer = _section(config, "trainer")
    max_epochs = int(trainer["max_epochs"])
    validation_interval = int(trainer["validation_interval_epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    initial_sha = state_dict_sha256(model)
    history: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    validation_mask_checksums: list[str] = []
    best_value = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0
    all_gradients_finite = True
    started = time.monotonic()
    for epoch in range(max_epochs):
        epoch_started = time.monotonic()
        model.train()
        order = _epoch_core_order(
            data.train_aliases, fold=fold, seed=model_seed, epoch=epoch
        )
        core_losses: list[float] = []
        masked_entries = 0
        for alias in order:
            core = data.cores[alias]
            resident = cache[alias]
            mask_seed = derive_training_mask_seed(
                fold_index=fold,
                model_seed=model_seed,
                epoch=epoch,
                core_alias=alias,
            )
            mask, masked_counts, mask_checksum = sample_uniform_mask_torch(
                core.n_cells,
                EXPECTED_GENES,
                seed=mask_seed,
                device=device,
            )
            exact_count = int(mask.sum().item())
            if exact_count != int(masked_counts.sum().item()):
                raise AdjacencyRunnerError("dynamic mask exact-count audit failed")
            if exact_count <= 0:
                raise AdjacencyRunnerError("a full core unexpectedly masks no genes")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                resident.expression, mask, resident.edge_index
            ).prediction
            loss = masked_huber_loss(
                resident.expression, prediction, mask, reduction="mean"
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("training loss is nonfinite")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all().item()
                ):
                    all_gradients_finite = False
                    raise FloatingPointError("training gradient is nonfinite")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(trainer["gradient_clip_norm"])
            )
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("model parameter is nonfinite")
            optimizer_steps += 1
            masked_entries += exact_count
            core_losses.append(float(loss.item()))
            schedule_rows.append(
                {
                    "epoch": epoch,
                    "core_alias": alias,
                    "mask_seed": int(mask_seed),
                    "mask_checksum": mask_checksum,
                    "masked_entries": exact_count,
                }
            )
            del mask, masked_counts, prediction, loss

        validation_value: float | None = None
        if (epoch + 1) % validation_interval == 0 or epoch + 1 == max_epochs:
            validation_value, fixed_checksums = _validation_huber(
                model, data, cache, device=device
            )
            validation_mask_checksums.extend(fixed_checksums)
            if validation_value < best_value:
                best_value = validation_value
                best_epoch = epoch
                best_state = _cpu_state(model)
        history.append(
            {
                "epoch": epoch,
                "epoch_number": epoch + 1,
                "training_core_huber_mean": float(np.mean(core_losses)),
                "training_core_huber_min": float(np.min(core_losses)),
                "training_core_huber_max": float(np.max(core_losses)),
                "masked_entries": masked_entries,
                "optimizer_steps_cumulative": optimizer_steps,
                "core_order": list(order),
                "validation_huber": validation_value,
                "duration_seconds": time.monotonic() - epoch_started,
            }
        )
    if best_state is None or best_epoch < 0 or not math.isfinite(best_value):
        raise AdjacencyRunnerError("no finite validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    best_sha = state_dict_sha256(model)
    if best_sha != state_dict_sha256(best_state):
        raise AdjacencyRunnerError("restored best state checksum mismatch")
    expected_steps = max_epochs * len(data.train_aliases)
    if optimizer_steps != expected_steps:
        raise AdjacencyRunnerError(
            f"optimizer step mismatch: {optimizer_steps} != {expected_steps}"
        )
    schedule_sha = canonical_sha256(
        {
            "schema": "adjacency_ablation_training_masks_v1",
            "fold": fold,
            "model_seed": model_seed,
            "rows": schedule_rows,
        }
    )
    evaluation_sha = canonical_sha256(
        {
            "schema": "adjacency_ablation_validation_masks_v1",
            "fold": fold,
            "checksums_in_validation_order": validation_mask_checksums,
        }
    )
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_huber=best_value,
        optimizer_steps=optimizer_steps,
        initial_state_sha256=initial_sha,
        best_state_sha256=best_sha,
        training_mask_schedule_sha256=schedule_sha,
        evaluation_mask_schedule_sha256=evaluation_sha,
        all_gradients_finite=all_gradients_finite,
        elapsed_seconds=time.monotonic() - started,
    )


@dataclass
class EvaluationResult:
    per_core_rows: list[dict[str, Any]]
    target_bin_rows: list[dict[str, Any]]
    neighbor_bin_rows: list[dict[str, Any]]
    canonical_prediction: dict[str, Any] | None
    mask_identity_sha256: str


def _metric_row(
    summary: Mapping[str, Any],
    *,
    standardized_huber: float,
    core: CoreData,
    split: str,
    replicate: int,
    mask_seed: int,
    mask_checksum: str,
) -> dict[str, Any]:
    return {
        "core_alias": core.alias,
        "split": split,
        "mask_replicate": replicate,
        "mask_seed": int(mask_seed),
        "mask_checksum": mask_checksum,
        "n_cells": core.n_cells,
        "n_masked": int(summary["n_masked"]),
        "standardized_huber": float(standardized_huber),
        "log1p_huber": float(summary["huber"]),
        "log1p_mae": float(summary["mae"]),
        "log1p_mse": float(summary["mse"]),
        "log1p_rmse": float(summary["rmse"]),
        "gene_pearson": float(summary["gene_pearson_mean"]),
        "gene_spearman": float(summary["gene_spearman_mean"]),
        "cell_pearson": float(summary["cell_pearson_mean"]),
        "cell_spearman": float(summary["cell_spearman_mean"]),
        "n_valid_gene_pearson": int(summary["n_valid_gene_pearson"]),
        "n_valid_gene_spearman": int(summary["n_valid_gene_spearman"]),
        "n_valid_cell_pearson": int(summary["n_valid_cell_pearson"]),
        "n_valid_cell_spearman": int(summary["n_valid_cell_spearman"]),
        "qc_passed_cells": int(core.qc_passed.sum()),
        "edge_count": int(core.edge_index.shape[1]),
    }


def _stratum_row(
    summary: Mapping[str, Any],
    *,
    core: CoreData,
    split: str,
    replicate: int,
    mask_seed: int,
    mask_checksum: str,
    label_field: str,
    label: str,
) -> dict[str, Any]:
    return {
        "core_alias": core.alias,
        "split": split,
        "mask_replicate": replicate,
        "mask_seed": int(mask_seed),
        "mask_checksum": mask_checksum,
        label_field: label,
        "n_cells": int(summary["n_cells"]),
        "n_masked": int(summary["n_masked"]),
        "log1p_huber": (
            float(summary["huber"])
            if math.isfinite(float(summary["huber"]))
            else None
        ),
        "log1p_mae": (
            float(summary["mae"])
            if math.isfinite(float(summary["mae"]))
            else None
        ),
        "log1p_mse": (
            float(summary["mse"])
            if math.isfinite(float(summary["mse"]))
            else None
        ),
        "log1p_rmse": (
            float(summary["rmse"])
            if math.isfinite(float(summary["rmse"]))
            else None
        ),
    }


def _evaluate_split(
    model: torch.nn.Module,
    data: PreparedFold,
    cache: Mapping[str, DeviceCore],
    *,
    aliases: Sequence[str],
    split: str,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    per_core: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    mask_identities: list[dict[str, Any]] = []
    canonical: dict[str, Any] | None = None
    with torch.no_grad():
        for alias in aliases:
            core = data.cores[alias]
            resident = cache[alias]
            fixed_masks = _fixed_masks(core)
            for replicate, (
                mask_np,
                counts,
                mask_seed,
                mask_checksum,
            ) in enumerate(fixed_masks):
                mask = torch.from_numpy(np.array(mask_np, copy=True)).to(
                    device=device, dtype=torch.bool
                )
                prediction_std_tensor = model(
                    resident.expression, mask, resident.edge_index
                ).prediction
                prediction_std = prediction_std_tensor.detach().cpu().numpy()
                target_std = core.standardized_expression
                standardized_huber, _ = _standardized_huber_stats(
                    target_std, prediction_std, mask_np
                )
                target_log1p = (
                    target_std.astype(np.float64, copy=False) * data.scale
                    + data.mean
                )
                prediction_log1p = (
                    prediction_std.astype(np.float64, copy=False) * data.scale
                    + data.mean
                )
                strata = evaluate_fixed_mask_strata(
                    target_log1p,
                    prediction_log1p,
                    mask_np,
                    core.true_spatial_edge_index,
                )
                overall = _mapping(strata.get("overall"), "overall metrics")
                metric_row = _metric_row(
                    overall,
                    standardized_huber=standardized_huber,
                    core=core,
                    split=split,
                    replicate=replicate,
                    mask_seed=mask_seed,
                    mask_checksum=mask_checksum,
                )
                metric_row["zero_mask_cells"] = int(np.sum(counts == 0))
                metric_row["fully_masked_cells"] = int(
                    np.sum(counts == EXPECTED_GENES)
                )
                per_core.append(metric_row)
                for label in TARGET_MASK_BIN_ORDER:
                    target_rows.append(
                        _stratum_row(
                            _mapping(
                                _mapping(strata, "strata")["target_mask_bins"][
                                    label
                                ],
                                f"target bin {label}",
                            ),
                            core=core,
                            split=split,
                            replicate=replicate,
                            mask_seed=mask_seed,
                            mask_checksum=mask_checksum,
                            label_field="target_mask_bin",
                            label=label,
                        )
                    )
                for label in NEIGHBOR_OBSERVED_BIN_ORDER:
                    neighbor_rows.append(
                        _stratum_row(
                            _mapping(
                                _mapping(strata, "strata")[
                                    "neighbor_observed_bins"
                                ][label],
                                f"neighbor bin {label}",
                            ),
                            core=core,
                            split=split,
                            replicate=replicate,
                            mask_seed=mask_seed,
                            mask_checksum=mask_checksum,
                            label_field="neighbor_observed_bin",
                            label=label,
                        )
                    )
                mask_identities.append(
                    {
                        "core_alias": alias,
                        "replicate": replicate,
                        "seed": mask_seed,
                        "checksum": mask_checksum,
                        "masked_counts_sha256": ndarray_sha256(counts),
                    }
                )
                if canonical is None and split == "validation" and replicate == 0:
                    gene_counts = mask_np.sum(axis=0, dtype=np.int64)
                    if np.any(gene_counts == 0):
                        raise AdjacencyRunnerError(
                            "canonical validation mask leaves a gene unsupported"
                        )
                    canonical = {
                        "core_alias": alias,
                        "y_true": (
                            np.where(mask_np, target_log1p, 0.0).sum(
                                axis=0, dtype=np.float64
                            )
                            / gene_counts
                        ).astype(float).tolist(),
                        "y_pred": (
                            np.where(mask_np, prediction_log1p, 0.0).sum(
                                axis=0, dtype=np.float64
                            )
                            / gene_counts
                        ).astype(float).tolist(),
                        "effective_mask_rate": float(mask_np.mean()),
                    }
                del mask, prediction_std_tensor, prediction_std
                del target_log1p, prediction_log1p, overall, strata
                gc.collect()
    return EvaluationResult(
        per_core_rows=per_core,
        target_bin_rows=target_rows,
        neighbor_bin_rows=neighbor_rows,
        canonical_prediction=canonical,
        mask_identity_sha256=canonical_sha256(mask_identities),
    )


def _equal_core_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    by_core: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            raise FloatingPointError(f"nonfinite {field} in per-core metrics")
        by_core.setdefault(str(row["core_alias"]), []).append(float(value))
    if not by_core:
        raise AdjacencyRunnerError(f"no rows available for {field}")
    core_means = [float(np.mean(values)) for values in by_core.values()]
    return float(np.mean(core_means))


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    training: TrainingResult,
    data: PreparedFold,
    condition: str,
    parameter_count: int,
) -> bytes:
    state = _cpu_state(training.model)
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "campaign_id": CAMPAIGN_ID,
        "model_name": "mean-adjacency-sage",
        "model_family": "explicit_self_mean_adjacency_graphsage",
        "checkpoint_role": "best",
        "checkpoint_policy": "minimum_fixed_validation_huber",
        "selection_policy": "minimum_prespecified_fixed_validation_huber",
        "monitored_metric": PRIMARY_METRIC,
        "monitored_value": training.best_validation_huber,
        "best_epoch": training.best_epoch,
        "epoch": training.best_epoch,
        "seed": int(config["seed"]),
        "fold": int(config["fold"]),
        "condition": condition,
        "parameter_count": parameter_count,
        "model_state_dict": state,
        "state_dict_sha256": training.best_state_sha256,
        "initial_state_sha256": training.initial_state_sha256,
        "training_mask_schedule_sha256": training.training_mask_schedule_sha256,
        "prepared_content_sha256": data.content_sha256,
        "preprocessing_sha256": data.preprocessing_sha256,
        "runtime_config_sha256": canonical_sha256(config),
        "retention_class": _section(config, "classification").get(
            "retention_class"
        ),
        "target_scale": "gene_wise_standardized_log1p_raw_count",
        "scientific_evaluation_scale": "unclipped_log1p_raw_count",
        "edge_features_used": False,
        "coordinates_used_as_model_inputs": False,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _peak_host_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The repository execution image
    # is Linux, but keep the conversion explicit for local test portability.
    return peak if sys.platform == "darwin" else peak * 1024


def _verify_contract_file() -> None:
    path = current_paths().project_root / CONTRACT_RELATIVE
    if sha256_file(path) != CONTRACT_SHA256:
        raise AdjacencyRunnerError("frozen task contract file checksum changed")


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Resolved device plus auditable recovery authorization, if applicable."""

    device: torch.device
    mode: str
    queue_job_id: str | None
    torch_intraop_threads: int
    torch_interop_threads: int
    recovery_contract_reference: str | None = None
    recovery_contract_sha256: str | None = None
    recovery_plan_reference: str | None = None
    recovery_plan_checksum: str | None = None
    recovery_plan_file_sha256: str | None = None
    recovery_enqueue_checksum: str | None = None
    null_enqueue_reference: str | None = None
    null_enqueue_checksum: str | None = None
    null_enqueue_file_sha256: str | None = None
    null_trigger_checksum: str | None = None
    recovery_worker_slot: int | None = None
    recovery_root_job_id: str | None = None

    def audit_fields(self) -> dict[str, Any]:
        return {
            "execution_device": str(self.device),
            "execution_mode": self.mode,
            "queue_job_id": self.queue_job_id,
            "torch_intraop_threads": self.torch_intraop_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "hardware_recovery_used": self.mode == "cpu_hardware_recovery",
            "recovery_contract_reference": self.recovery_contract_reference,
            "recovery_contract_sha256": self.recovery_contract_sha256,
            "recovery_plan_reference": self.recovery_plan_reference,
            "recovery_plan_checksum": self.recovery_plan_checksum,
            "recovery_plan_file_sha256": self.recovery_plan_file_sha256,
            "recovery_enqueue_checksum": self.recovery_enqueue_checksum,
            "null_enqueue_reference": self.null_enqueue_reference,
            "null_enqueue_checksum": self.null_enqueue_checksum,
            "null_enqueue_file_sha256": self.null_enqueue_file_sha256,
            "null_trigger_checksum": self.null_trigger_checksum,
            "recovery_worker_slot": self.recovery_worker_slot,
            "recovery_root_job_id": self.recovery_root_job_id,
        }


def _strict_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdjacencyRunnerError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except AdjacencyRunnerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdjacencyRunnerError(f"cannot read {label}: {path}") from error
    if not isinstance(raw, dict):
        raise AdjacencyRunnerError(f"{label} must be a JSON object")
    return raw


def _verify_recovery_contract_file() -> None:
    path = current_paths().project_root / RECOVERY_CONTRACT_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise AdjacencyRunnerError(
            "frozen hardware recovery contract is missing or is not a regular file"
        )
    if sha256_file(path) != RECOVERY_CONTRACT_SHA256:
        raise AdjacencyRunnerError(
            "frozen hardware recovery contract checksum changed"
        )


def _materialized_recovery_slots() -> dict[tuple[str, int, int, str], dict[str, Any]]:
    path = current_paths().project_root / MATERIALIZATION_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise AdjacencyRunnerError(
            "immutable adjacency materialization receipt is missing"
        )
    receipt = _strict_json_mapping(path, label="materialization receipt")
    signed = dict(receipt)
    checksum = signed.pop("checksum", None)
    if (
        checksum != MATERIALIZATION_CHECKSUM
        or canonical_sha256(signed) != MATERIALIZATION_CHECKSUM
        or receipt.get("schema_version") != 1
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("receipt_kind")
        != "adjacency_ablation_config_materialization_v1"
    ):
        raise AdjacencyRunnerError(
            "immutable adjacency materialization receipt failed verification"
        )
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list):
        raise AdjacencyRunnerError("materialization jobs must be a list")
    result: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for raw in jobs:
        job = _mapping(raw, "materialization job")
        stage = str(job.get("stage"))
        if stage not in {"primary", "null"}:
            continue
        slot = (
            stage,
            int(job.get("fold", -1)),
            int(job.get("seed", -1)),
            str(job.get("condition")),
        )
        if slot in result:
            raise AdjacencyRunnerError(
                f"materialization repeats recovery slot {slot!r}"
            )
        result[slot] = dict(job)
    expected_primary = {
        ("primary", fold, seed, condition)
        for fold in range(5)
        for seed in range(5)
        for condition in ("spatial", "isolated")
    }
    expected_null = {
        ("null", fold, seed, "position_permuted_null")
        for fold in range(5)
        for seed in range(5)
    }
    if set(result) != expected_primary | expected_null:
        raise AdjacencyRunnerError(
            "materialization recovery slot inventory is incomplete"
        )
    return result


def _load_verified_cpu_recovery_plan() -> tuple[dict[str, Any], str]:
    _verify_recovery_contract_file()
    path = current_paths().project_root / CPU_RECOVERY_PLAN_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise AdjacencyRunnerError(
            "signed CPU recovery plan is missing or is not a regular file"
        )
    try:
        from scripts.train.recover_adjacency_gpu_failure import (
            load_recovery_plan,
        )

        plan = load_recovery_plan(path)
    except Exception as error:
        raise AdjacencyRunnerError(
            "signed CPU recovery plan failed strict verification"
        ) from error
    if plan.get("receipt_kind") != CPU_RECOVERY_PLAN_KIND:
        raise AdjacencyRunnerError("CPU recovery plan kind changed")
    return plan, sha256_file(path)


def _configure_cpu_recovery_threads() -> tuple[int, int]:
    try:
        if torch.get_num_interop_threads() != CPU_RECOVERY_INTEROP_THREADS:
            torch.set_num_interop_threads(CPU_RECOVERY_INTEROP_THREADS)
        torch.set_num_threads(CPU_RECOVERY_INTRAOP_THREADS)
    except RuntimeError as error:
        raise AdjacencyRunnerError(
            "CPU recovery thread settings could not be frozen before execution"
        ) from error
    intraop = int(torch.get_num_threads())
    interop = int(torch.get_num_interop_threads())
    if (
        intraop != CPU_RECOVERY_INTRAOP_THREADS
        or interop != CPU_RECOVERY_INTEROP_THREADS
    ):
        raise AdjacencyRunnerError("CPU recovery thread settings differ from plan")
    return intraop, interop


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_null_enqueue_authorization(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    materialized: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    planned: Mapping[str, Any],
    fold: int,
    seed: int,
    condition: str,
    queue_job_id: str,
) -> dict[str, str]:
    path = current_paths().project_root / NULL_ENQUEUE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise AdjacencyRunnerError(
            "immutable null enqueue receipt is missing or is not a regular file"
        )
    receipt = _strict_json_mapping(path, label="null enqueue receipt")
    expected_top_keys = {
        "schema_version",
        "receipt_kind",
        "campaign_id",
        "stage",
        "materialization_checksum",
        "pilot_gate_checksum",
        "null_trigger_checksum",
        "recovery_plan_checksum",
        "recovery_enqueue_checksum",
        "execution_device",
        "maximum_attempts",
        "complete",
        "jobs",
        "checksum",
    }
    if set(receipt) != expected_top_keys:
        raise AdjacencyRunnerError("null enqueue receipt schema changed")
    signed = dict(receipt)
    receipt_checksum = signed.pop("checksum", None)
    try:
        checksum_matches = (
            _is_sha256(receipt_checksum)
            and canonical_sha256(signed) == receipt_checksum
        )
    except (TypeError, ValueError):
        checksum_matches = False
    if (
        not checksum_matches
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != NULL_ENQUEUE_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("stage") != "null"
        or receipt.get("materialization_checksum") != MATERIALIZATION_CHECKSUM
        or receipt.get("pilot_gate_checksum") is not None
        or not _is_sha256(receipt.get("null_trigger_checksum"))
        or receipt.get("recovery_plan_checksum") != plan.get("checksum")
        or receipt.get("recovery_enqueue_checksum")
        != CPU_RECOVERY_ENQUEUE_CHECKSUM
        or receipt.get("execution_device") != "cpu"
        or receipt.get("maximum_attempts") != 1
        or receipt.get("complete") is not True
    ):
        raise AdjacencyRunnerError(
            "null enqueue receipt failed signed recovery verification"
        )

    raw_jobs = receipt.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 25:
        raise AdjacencyRunnerError("null enqueue receipt grid is incomplete")
    expected_job_keys = {
        "stage",
        "fold",
        "seed",
        "condition",
        "config_sha256",
        "scientific_id",
        "job_id",
        "requested_gpu",
        "maximum_attempts",
    }
    plan_jobs = plan.get("conditional_null_jobs")
    if not isinstance(plan_jobs, list):
        raise AdjacencyRunnerError("conditional null plan inventory is invalid")
    plan_by_slot = {
        (int(item["fold"]), int(item["seed"]), str(item["condition"])): item
        for item in plan_jobs
    }
    observed: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    job_ids: set[str] = set()
    for raw in raw_jobs:
        job = _mapping(raw, "null enqueue job")
        if set(job) != expected_job_keys:
            raise AdjacencyRunnerError("null enqueue job schema changed")
        slot = (
            int(job.get("fold", -1)),
            int(job.get("seed", -1)),
            str(job.get("condition")),
        )
        if slot in observed:
            raise AdjacencyRunnerError("null enqueue receipt repeats a slot")
        base = materialized.get(("null", *slot))
        authorization = plan_by_slot.get(slot)
        job_id = str(job.get("job_id", ""))
        if (
            base is None
            or authorization is None
            or job.get("stage") != "null"
            or job.get("config_sha256") != base.get("config_sha256")
            or job.get("config_sha256")
            != authorization.get("canonical_config_sha256")
            or job.get("scientific_id") != base.get("scientific_id")
            or job.get("requested_gpu") != authorization.get("worker_slot")
            or job.get("maximum_attempts") != 1
            or not job_id
            or job_id in job_ids
            or authorization.get("config_reference")
            != base.get("config_reference")
        ):
            raise AdjacencyRunnerError(
                f"null enqueue identity is invalid for slot {slot!r}"
            )
        observed[slot] = job
        job_ids.add(job_id)
    expected_slots = {
        (candidate_fold, candidate_seed, "position_permuted_null")
        for candidate_fold in range(5)
        for candidate_seed in range(5)
    }
    if set(observed) != expected_slots:
        raise AdjacencyRunnerError("null enqueue receipt grid is incomplete")

    target = observed.get((fold, seed, condition))
    if target is None or target.get("job_id") != queue_job_id:
        raise AdjacencyRunnerError(
            "queue job is not the null job authorized by the immutable enqueue receipt"
        )
    config_path = _project_path(
        planned.get("config_reference"), label="null recovery config reference"
    )
    materialized_target = materialized[("null", fold, seed, condition)]
    if (
        not config_path.is_file()
        or sha256_file(config_path)
        != materialized_target.get("config_file_sha256")
    ):
        raise AdjacencyRunnerError("immutable null configuration file changed")
    locked_config = load_yaml_mapping(config_path)
    try:
        from spatial_benchmark.queueing import command_for_config

        command_sha = canonical_sha256(command_for_config(locked_config))
    except Exception as error:
        raise AdjacencyRunnerError(
            "null recovery command authorization could not be reconstructed"
        ) from error
    if (
        locked_config != config
        or canonical_sha256(locked_config) != planned.get("canonical_config_sha256")
        or command_sha != planned.get("command_sha256")
    ):
        raise AdjacencyRunnerError(
            "null queue config/reference/command differs from signed authorization"
        )
    return {
        "recovery_enqueue_checksum": str(receipt["recovery_enqueue_checksum"]),
        "null_enqueue_reference": NULL_ENQUEUE_RELATIVE.as_posix(),
        "null_enqueue_checksum": str(receipt_checksum),
        "null_enqueue_file_sha256": sha256_file(path),
        "null_trigger_checksum": str(receipt["null_trigger_checksum"]),
    }


def _authorize_cpu_recovery(
    config: Mapping[str, Any],
    *,
    stage: str,
    condition: str,
    fold: int,
    seed: int,
) -> ExecutionAuthorization:
    attempt = int(config["attempt"])
    if not (
        (stage == "primary" and attempt == 2)
        or (stage == "null" and attempt == 1)
    ):
        raise AdjacencyRunnerError(
            "CUDA is unavailable and this stage/attempt has no signed CPU recovery authorization"
        )
    plan, plan_file_sha = _load_verified_cpu_recovery_plan()
    materialized = _materialized_recovery_slots()
    inventory_key = (
        "primary_jobs" if stage == "primary" else "conditional_null_jobs"
    )
    raw_inventory = plan.get(inventory_key)
    if not isinstance(raw_inventory, list):
        raise AdjacencyRunnerError("CPU recovery plan inventory is invalid")
    matches = [
        _mapping(raw, "CPU recovery plan job")
        for raw in raw_inventory
        if raw.get("fold") == fold
        and raw.get("seed") == seed
        and raw.get("condition") == condition
        and raw.get("attempt") == attempt
    ]
    if len(matches) != 1:
        raise AdjacencyRunnerError(
            "CPU recovery plan does not authorize this exact execution slot"
        )
    planned = matches[0]
    normalized_config = dict(config)
    if stage == "primary":
        normalized_config["attempt"] = 1
    config_sha = canonical_sha256(normalized_config)
    materialized_job = materialized[(stage, fold, seed, condition)]
    if (
        planned.get("canonical_config_sha256") != config_sha
        or materialized_job.get("config_sha256") != config_sha
        or planned.get("config_reference")
        != materialized_job.get("config_reference")
    ):
        raise AdjacencyRunnerError(
            "CPU recovery slot does not match the immutable scientific configuration"
        )

    queue_job_id = os.environ.get("BAGM_JOB_ID", "").strip()
    if not queue_job_id:
        raise AdjacencyRunnerError(
            "CPU recovery requires the queue-owned BAGM_JOB_ID identity"
        )
    root_job_id: str | None = None
    null_audit: dict[str, str] = {}
    if stage == "primary":
        if planned.get("retry_job_id") != queue_job_id:
            raise AdjacencyRunnerError(
                "queue job is not the retry authorized for this CPU recovery slot"
            )
        root_job_id = str(planned["root_job_id"])
    else:
        null_audit = _verify_null_enqueue_authorization(
            config,
            plan=plan,
            materialized=materialized,
            planned=planned,
            fold=fold,
            seed=seed,
            condition=condition,
            queue_job_id=queue_job_id,
        )

    intraop, interop = _configure_cpu_recovery_threads()
    return ExecutionAuthorization(
        device=torch.device("cpu"),
        mode="cpu_hardware_recovery",
        queue_job_id=queue_job_id,
        torch_intraop_threads=intraop,
        torch_interop_threads=interop,
        recovery_contract_reference=RECOVERY_CONTRACT_RELATIVE.as_posix(),
        recovery_contract_sha256=RECOVERY_CONTRACT_SHA256,
        recovery_plan_reference=CPU_RECOVERY_PLAN_RELATIVE.as_posix(),
        recovery_plan_checksum=str(plan["checksum"]),
        recovery_plan_file_sha256=plan_file_sha,
        recovery_enqueue_checksum=null_audit.get("recovery_enqueue_checksum"),
        null_enqueue_reference=null_audit.get("null_enqueue_reference"),
        null_enqueue_checksum=null_audit.get("null_enqueue_checksum"),
        null_enqueue_file_sha256=null_audit.get("null_enqueue_file_sha256"),
        null_trigger_checksum=null_audit.get("null_trigger_checksum"),
        recovery_worker_slot=int(planned["worker_slot"]),
        recovery_root_job_id=root_job_id,
    )


def _resolve_execution(
    config: Mapping[str, Any],
    *,
    stage: str,
    condition: str,
    fold: int,
    seed: int,
    requested_device: str | torch.device | None,
) -> ExecutionAuthorization:
    requested = None if requested_device is None else torch.device(requested_device)
    if requested is not None and requested.type not in {"cpu", "cuda"}:
        raise AdjacencyRunnerError(
            f"unsupported execution device type {requested.type!r}"
        )
    recovery_slot = (
        stage == "primary" and int(config["attempt"]) == 2
    ) or (stage == "null" and int(config["attempt"]) == 1)
    if recovery_slot:
        if requested is not None and requested.type == "cuda":
            raise AdjacencyRunnerError(
                "signed hardware recovery slots are uniformly CPU-only"
            )
        return _authorize_cpu_recovery(
            config,
            stage=stage,
            condition=condition,
            fold=fold,
            seed=seed,
        )
    if torch.cuda.is_available():
        if requested is not None and requested.type == "cpu":
            raise AdjacencyRunnerError(
                "CPU execution is prohibited while CUDA is available"
            )
        target = requested or torch.device("cuda:0")
        if target.index is None:
            target = torch.device("cuda:0")
        return ExecutionAuthorization(
            device=target,
            mode="cuda",
            queue_job_id=os.environ.get("BAGM_JOB_ID"),
            torch_intraop_threads=int(torch.get_num_threads()),
            torch_interop_threads=int(torch.get_num_interop_threads()),
        )
    if requested is not None and requested.type == "cuda":
        raise AdjacencyRunnerError("the requested CUDA device is unavailable")
    raise AdjacencyRunnerError(
        "CUDA is unavailable and this stage/attempt has no signed CPU recovery authorization"
    )


def run_adjacency_ablation(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Execute one paired-condition member in a worker-owned archive."""

    stage, condition, fold, model_seed = _validate_config(config)
    _verify_contract_file()
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 UTF-8 bytes"
        )
    execution = _resolve_execution(
        config,
        stage=stage,
        condition=condition,
        fold=fold,
        seed=model_seed,
        requested_device=device,
    )
    target_device = execution.device
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)

    _configure_determinism()
    execution_audit = execution.audit_fields()

    total_started = time.monotonic()
    data_started = time.monotonic()
    data = _load_prepared_fold(config, condition=condition, fold=fold)
    cache = _device_cache(data, target_device)
    data_seconds = time.monotonic() - data_started

    _seed_all(model_seed, device=target_device)
    model = _build_runner_model(
        seed=model_seed, device=target_device
    ).to(target_device)
    parameter_count = trainable_parameter_count(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise AdjacencyRunnerError(
            f"parameter count changed: {parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )
    # Reset stochastic training streams after constructor verification.  The
    # separate mask generator is seed-addressed and cannot consume this stream.
    _seed_all(model_seed, device=target_device)
    training = _train(
        model,
        data,
        cache,
        config,
        fold=fold,
        model_seed=model_seed,
        device=target_device,
    )
    archive.write_table(
        "metrics/history",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "model_seed": model_seed,
                **row,
            }
            for row in training.history
        ],
        fallback="jsonl",
    )
    for row in training.history:
        archive.append_metric_event(
            {
                "name": "fit/training/masked_huber",
                "value": row["training_core_huber_mean"],
                "step": row["epoch_number"],
            }
        )
        if row["validation_huber"] is not None:
            archive.append_metric_event(
                {
                    "name": PRIMARY_METRIC,
                    "value": row["validation_huber"],
                    "step": row["epoch_number"],
                }
            )

    evaluation_started = time.monotonic()
    validation = _evaluate_split(
        training.model,
        data,
        cache,
        aliases=data.validation_aliases,
        split="validation",
        device=target_device,
    )
    testing = _evaluate_split(
        training.model,
        data,
        cache,
        aliases=data.test_aliases,
        split="test",
        device=target_device,
    )
    evaluation_seconds = time.monotonic() - evaluation_started
    if (
        len(validation.per_core_rows) != 3
        or len(testing.per_core_rows) != 6
        or len(testing.target_bin_rows) != 30
        or len(testing.neighbor_bin_rows) != 24
    ):
        raise AdjacencyRunnerError("evaluation coverage is incomplete")
    archive.write_table(
        "metrics/per_validation_replicate",
        [
            {
                "run_id": archive.run_id,
                "model_seed": model_seed,
                "condition": condition,
                **row,
            }
            for row in validation.per_core_rows
        ],
        fallback="jsonl",
    )
    archive.write_table(
        "metrics/per_core_replicate",
        [
            {
                "run_id": archive.run_id,
                "model_seed": model_seed,
                "condition": condition,
                **row,
            }
            for row in testing.per_core_rows
        ],
        fallback="jsonl",
    )
    archive.write_table(
        "metrics/per_target_mask_bin",
        [
            {
                "run_id": archive.run_id,
                "model_seed": model_seed,
                "condition": condition,
                **row,
            }
            for row in testing.target_bin_rows
        ],
        fallback="jsonl",
    )
    archive.write_table(
        "metrics/per_neighbor_observation_bin",
        [
            {
                "run_id": archive.run_id,
                "model_seed": model_seed,
                "condition": condition,
                **row,
            }
            for row in testing.neighbor_bin_rows
        ],
        fallback="jsonl",
    )

    val_huber = _equal_core_metric(
        validation.per_core_rows, "standardized_huber"
    )
    if not math.isclose(
        val_huber,
        training.best_validation_huber,
        rel_tol=5e-6,
        abs_tol=5e-7,
    ):
        raise AdjacencyRunnerError(
            "restored best validation metric differs from checkpoint selection"
        )
    final_metrics = {
        PRIMARY_METRIC: val_huber,
        "val/log1p_masked_huber": _equal_core_metric(
            validation.per_core_rows, "log1p_huber"
        ),
        "val/log1p_masked_mae": _equal_core_metric(
            validation.per_core_rows, "log1p_mae"
        ),
        "val/log1p_masked_rmse": _equal_core_metric(
            validation.per_core_rows, "log1p_rmse"
        ),
        "test/masked_huber": _equal_core_metric(
            testing.per_core_rows, "standardized_huber"
        ),
        "test/log1p_masked_huber": _equal_core_metric(
            testing.per_core_rows, "log1p_huber"
        ),
        "test/log1p_masked_mae": _equal_core_metric(
            testing.per_core_rows, "log1p_mae"
        ),
        "test/log1p_masked_rmse": _equal_core_metric(
            testing.per_core_rows, "log1p_rmse"
        ),
    }
    if not all(math.isfinite(float(value)) for value in final_metrics.values()):
        raise FloatingPointError("one or more aggregate metrics are nonfinite")
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event({"name": name, "value": value})

    canonical = validation.canonical_prediction
    if canonical is None:
        raise AdjacencyRunnerError("canonical validation prediction was not captured")
    raw_prediction = {
        "_protected_alias": canonical["core_alias"],
        "run_id": archive.run_id,
        "dataset_id": str(_section(config, "dataset")["dataset_id"]),
        "split": "validation",
        "y_true": canonical["y_true"],
        "y_pred": canonical["y_pred"],
        "graph_id": data.cores[
            str(canonical["core_alias"])
        ].selected_adjacency_sha256,
        "fold": fold,
        "node_count": data.cores[str(canonical["core_alias"])].n_cells,
        "edge_count": int(
            data.cores[str(canonical["core_alias"])].edge_index.shape[1]
        ),
        "effective_mask_rate": canonical["effective_mask_rate"],
        "subgroup_core_alias": canonical["core_alias"],
        "prediction_granularity": "per_gene_mean_over_fixed_masked_entries",
    }
    prediction_rows = deidentify_prediction_rows(
        [raw_prediction],
        identifier_fields=["_protected_alias"],
        salt=sample_key_salt,
        namespace=(
            "bagm:adjacency-ablation:validation:"
            f"{_section(config, 'dataset')['dataset_id']}"
        ),
    )
    prediction_path = archive.write_predictions(
        "validation", prediction_rows, fallback="jsonl"
    )

    checkpoint_path = archive.write_bytes(
        "checkpoints/best.ckpt",
        _checkpoint_bytes(
            archive=archive,
            config=config,
            training=training,
            data=data,
            condition=condition,
            parameter_count=parameter_count,
        ),
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    reloaded_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    reloaded = _build_runner_model(seed=model_seed, device=torch.device("cpu"))
    reloaded.load_state_dict(reloaded_payload["model_state_dict"], strict=True)
    replay_sha = state_dict_sha256(reloaded)
    if replay_sha != training.best_state_sha256:
        raise AdjacencyRunnerError("checkpoint strict reload checksum mismatch")
    archive.write_json(
        "diagnostics/checkpoint_replay.json",
        {
            "strict_state_dict_load": True,
            "checkpoint_sha256": checkpoint_sha,
            "expected_state_dict_sha256": training.best_state_sha256,
            "reloaded_state_dict_sha256": replay_sha,
            "verified": True,
        },
    )
    del reloaded, reloaded_payload

    graph_audit = {}
    for alias, core in data.cores.items():
        edges = core.edge_index
        loops = edges[0] == edges[1]
        graph_audit[alias] = {
            "n_cells": core.n_cells,
            "n_fov_groups": int(np.unique(core.fov_group).size),
            "n_directed_edges": int(edges.shape[1]),
            "n_self_loops": int(loops.sum()),
            "n_off_diagonal_edges": int((~loops).sum()),
            "cross_fov_edges": int(
                np.sum(core.fov_group[edges[0]] != core.fov_group[edges[1]])
            ),
            "exact_one_self_loop_per_cell": int(loops.sum()) == core.n_cells,
            "adjacency_sha256": ndarray_sha256(edges),
            "true_spatial_off_diagonal_sha256": ndarray_sha256(
                core.true_spatial_edge_index
            ),
        }
    archive.write_json(
        "diagnostics/graph_audit.json",
        {
            "condition": condition,
            "definition": {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "grouping": "raw_fov_within_core",
                "coordinates_as_model_input": False,
                "edge_features": False,
            },
            "cores": graph_audit,
        },
    )
    archive.write_json(
        "diagnostics/pairing_and_leakage_audit.json",
        {
            **execution_audit,
            "fold": fold,
            "model_seed": model_seed,
            "condition": condition,
            "train_aliases": list(data.train_aliases),
            "validation_aliases": list(data.validation_aliases),
            "test_aliases": list(data.test_aliases),
            "split_disjoint": len(
                set(
                    data.train_aliases
                    + data.validation_aliases
                    + data.test_aliases
                )
            )
            == 10,
            "preprocessing_fit_aliases": list(data.train_aliases),
            "preprocessing_sha256": data.preprocessing_sha256,
            "initial_state_sha256": training.initial_state_sha256,
            "training_mask_schedule_sha256": training.training_mask_schedule_sha256,
            "validation_selection_mask_schedule_sha256": (
                training.evaluation_mask_schedule_sha256
            ),
            "validation_evaluation_mask_identity_sha256": (
                validation.mask_identity_sha256
            ),
            "test_evaluation_mask_identity_sha256": testing.mask_identity_sha256,
            "model_inputs": ["masked_standardized_log1p_counts", "binary_mask"],
            "prohibited_inputs_absent": [
                "cell_type",
                "cluster",
                "niche",
                "donor",
                "core",
                "tissue_stage",
                "coordinates",
                "target_derived_library_size",
            ],
            "masked_neighbor_inputs_only": True,
        },
    )

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(target_device))
    else:
        peak_vram_bytes = 0
    peak_host_bytes = _peak_host_bytes()
    epoch_seconds = training.elapsed_seconds / int(
        _section(config, "trainer")["max_epochs"]
    )
    projected_full_seconds = (
        data_seconds + evaluation_seconds + epoch_seconds * 80.0
    )
    resource_limits_passed = (
        peak_vram_bytes / 1024**3 <= PILOT_MAX_VRAM_GIB
        and peak_host_bytes / 1024**3 <= PILOT_MAX_HOST_GIB
        and projected_full_seconds <= PILOT_MAX_PROJECTED_FULL_SECONDS
    )
    expected_updates = int(_section(config, "trainer")["max_epochs"]) * 7
    finite_metrics = all(
        math.isfinite(float(value)) for value in final_metrics.values()
    )
    coverage_complete = (
        len(training.history) == int(_section(config, "trainer")["max_epochs"])
        and len(validation.per_core_rows) == 3
        and len(testing.per_core_rows) == 6
        and set(data.cores) == set(EXPECTED_ALIASES)
    )
    update_count_verified = training.optimizer_steps == expected_updates
    archive.write_json(
        "diagnostics/resources.json",
        {
            **execution_audit,
            "device": str(target_device),
            "gpu_name": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "peak_vram_gib": peak_vram_bytes / 1024**3,
            "peak_host_memory_gib": peak_host_bytes / 1024**3,
            "training_seconds": training.elapsed_seconds,
            "data_seconds": data_seconds,
            "evaluation_seconds": evaluation_seconds,
            "projected_full_seconds": projected_full_seconds,
            "limits": {
                "peak_vram_gib": PILOT_MAX_VRAM_GIB,
                "peak_host_memory_gib": PILOT_MAX_HOST_GIB,
                "projected_full_seconds": PILOT_MAX_PROJECTED_FULL_SECONDS,
            },
            "resource_limits_passed": resource_limits_passed,
        },
    )
    archive.write_json(
        "provenance/scientific_inputs.json",
        {
            **execution_audit,
            "prepared_manifest": data.manifest_path.relative_to(
                current_paths().project_root
            ).as_posix(),
            "prepared_manifest_sha256": data.manifest_sha256,
            "prepared_content_sha256": data.content_sha256,
            "preprocessing_sha256": data.preprocessing_sha256,
            "gene_schema_sha256": canonical_sha256(list(data.gene_names)),
            "frozen_contract_sha256": CONTRACT_SHA256,
            "fold": fold,
            "condition": condition,
        },
    )

    summary = {
        **execution_audit,
        "run_id": archive.run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "condition": condition,
        "fold": fold,
        "model_seed": model_seed,
        "model_name": "mean-adjacency-sage",
        "model_family": "explicit_self_mean_adjacency_graphsage",
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": val_huber,
        "parameter_count": parameter_count,
        "initial_state_sha256": training.initial_state_sha256,
        "best_state_sha256": training.best_state_sha256,
        "training_mask_schedule_sha256": training.training_mask_schedule_sha256,
        "validation_selection_mask_schedule_sha256": (
            training.evaluation_mask_schedule_sha256
        ),
        "validation_mask_identity_sha256": validation.mask_identity_sha256,
        "test_mask_identity_sha256": testing.mask_identity_sha256,
        "best_epoch": training.best_epoch,
        "final_epoch": int(_section(config, "trainer")["max_epochs"]) - 1,
        "completed_global_epochs": int(_section(config, "trainer")["max_epochs"]),
        "optimizer_steps": training.optimizer_steps,
        "checkpoint_role": "best",
        "checkpoint_path": "checkpoints/best.ckpt",
        "checkpoint_sha256": checkpoint_sha,
        "prediction_path": prediction_path.relative_to(
            archive.scratch_path
        ).as_posix(),
        "config_sha256": canonical_sha256(config),
        "prepared_content_sha256": data.content_sha256,
        "prepared_manifest_sha256": data.manifest_sha256,
        "preprocessing_sha256": data.preprocessing_sha256,
        "peak_vram_gib": peak_vram_bytes / 1024**3,
        "peak_vram_gb": peak_vram_bytes / 1024**3,
        "peak_host_memory_bytes": peak_host_bytes,
        "duration_seconds": time.monotonic() - total_started,
        "finite_metrics": finite_metrics,
        "coverage_complete": coverage_complete,
        "update_count_verified": update_count_verified,
        "resource_limits_passed": resource_limits_passed,
        "all_gradients_finite": training.all_gradients_finite,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "conclusion_eligible": stage in {"primary", "null"},
        "generalization_estimate": stage in {"primary", "null"},
        "exploratory": True,
        "maximum_claim": (
            "diagnostic feasibility only"
            if stage in {"smoke", "pilot"}
            else (
                "predictive value of spatial neighboring-cell information for "
                "masked transcript reconstruction in selected Adjacent Normal tissue"
            )
        ),
        "failures": [],
    }
    archive.write_summary(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    summary = run_adjacency_ablation(
        config,
        archive,
        sample_key_salt=os.environ.get("BAGM_SAMPLE_KEY_SALT", ""),
    )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "primary_metric_name": summary["primary_metric_name"],
                "primary_metric_value": summary["primary_metric_value"],
                "checkpoint": summary["checkpoint_path"],
                "predictions": summary["prediction_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

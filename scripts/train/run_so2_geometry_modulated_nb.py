#!/usr/bin/env python3
"""Train the frozen SO2 donor-grouped geometry-modulated NB2 model.

Production execution is one four-rank ``torchrun`` process.  Each rank handles
five mask views for one member of a paired-core optimizer update, making DDP's
ordinary rank mean the exact mean of twenty view losses.  Validation uses the
same partition over two held-out cores and ten immutable views, but never
constructs a backward graph.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
    GRADIENT_DIRECTION_METRICS_COLUMNS,
    BlockGradientDirectionTracker,
    FullGradientDirectionTracker,
    GradientDirectionUpdateContext,
)
from spatial_benchmark.negative_binomial import (  # noqa: E402
    NegativeBinomialModelOutput,
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
    masked_negative_binomial_metrics,
    masked_negative_binomial_nll,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    _clear_graph_layout_caches,
    _clone_tree_to_cpu,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    cohort_relative_qkv_core_order,
    cohort_relative_qkv_distributed_assignment,
    cohort_relative_qkv_model_step_seed,
    make_cohort_exact_uniform_training_mask,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    validate_prediction_rows,
)
from spatial_benchmark.so2_nb_data import (  # noqa: E402
    SO2NBDataBundle,
    SO2NBCoreBatch,
    SO2_NB_TEST_ALIASES,
    SO2_NB_TRAINING_ALIASES,
    SO2_NB_VALIDATION_ALIASES,
    load_so2_nb_data,
)
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    AtomicBestLatestCheckpointStore,
    CHECKPOINT_SCHEMA,
    DurableNBEpochMetricsCSV,
    DurableNBPerCoreMetricsCSV,
    DurableScalarCSV,
    EPOCH_METRICS_SCHEMA,
    EXPECTED_PARAMETER_COUNT,
    EarlyStoppingState,
    PROTOCOL,
    SO2NBTrainingError,
    TRAINING_ALIASES,
    VALIDATION_ALIASES,
    ValidationAggregate,
    ValidationViewStatistics,
    aggregate_training_nll,
    aggregate_validation_statistics,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
    update_early_stopping,
    validation_statistics_from_tensor,
    validation_statistics_tensor,
    write_nb_training_plots,
)
from spatial_benchmark.training import (  # noqa: E402
    _autocast_context,
    _make_grad_scaler,
    set_deterministic_seed,
)


CAMPAIGN_ID = (
    "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0"
)
PREFLIGHT_SCHEMA = "so2_geometry_modulated_nb_ddp4_preflight_v1"
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
MODEL_SEED = 0
MASK_BASE_SEED = 2026082401
CORE_ORDER_SEED = 2026082402
MAXIMUM_EPOCHS = 300
MINIMUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4
VALIDATION_VIEW_COUNT = 10
TRAINING_VIEW_COUNT = 10
TRAINING_VIEWS_PER_EPOCH = 120
PREFLIGHT_MAX_VRAM_GIB = 22.0
PREFLIGHT_MINIMUM_HEADROOM_GIB = 2.0
PREFLIGHT_TRAINING_PAIR = ("SO2-C23", "SO2-C24")
FROZEN_TASK_CONTRACT_SHA256 = (
    "13ff231bf842ecddfe7bfe000660d108d04b829d9706978972e0cbd06b39d74f"
)
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"
)
FROZEN_TASK_CONTRACT = Path(
    "experiments/campaigns/"
    "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0/"
    "frozen_task_contract.yaml"
)
PREFLIGHT_CODE_FILES = (
    Path("scripts/diagnostics/preflight_so2_geometry_modulated_nb_ddp.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/negative_binomial.py"),
    Path("src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/so2_nb_data.py"),
    Path("src/spatial_benchmark/so2_nb_training.py"),
    Path("src/spatial_benchmark/configuration.py"),
    Path("src/spatial_benchmark/gradient_direction_observability.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training_v2.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training.py"),
    Path("src/spatial_benchmark/adjacency_ablation.py"),
    Path("src/spatial_benchmark/masking.py"),
    Path("src/spatial_benchmark/training.py"),
    Path("src/spatial_benchmark/relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/models.py"),
    Path("src/spatial_benchmark/paths.py"),
    Path("src/spatial_benchmark/run_archive.py"),
    Path("src/spatial_benchmark/identifiers.py"),
    Path("src/spatial_benchmark/fingerprints.py"),
    Path("src/spatial_benchmark/so2_pooled_full_core.py"),
    Path("src/spatial_benchmark/so2_relative_graphs.py"),
    Path("src/spatial_benchmark/queueing.py"),
)
CODE_HASH_SCOPE = (
    "campaign_numerical_and_data_integrity_resolution_plus_runner_archive_"
    "and_queue_launch_finalization"
)


def _digest_framed_bytes(digest: Any, value: bytes) -> None:
    """Write one unambiguous length-prefixed byte field to *digest*."""

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _checkpoint_mapping_sort_key(value: Any) -> tuple[str, str, str]:
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _update_checkpoint_tree_digest(digest: Any, value: Any, *, path: str) -> None:
    """Hash checkpoint structure with type-domain separation.

    ``ReduceLROnPlateau`` legitimately persists ``mode_worse=+/-inf``.  Python
    and NumPy scalar infinities therefore have explicit encodings that cannot
    collide with an ordinary tuple or string.  Scalar NaNs remain forbidden;
    tensor and ndarray bytes are hashed verbatim after independent finite-state
    checks in the training and checkpoint validation paths.
    """

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.layout != torch.strided:
            raise SO2NBTrainingError(
                f"Unsupported checkpoint tensor layout at {path}: {tensor.layout}."
            )
        digest.update(b"tensor\0")
        _digest_framed_bytes(digest, str(tensor.dtype).encode("ascii"))
        _digest_framed_bytes(
            digest, np.asarray(tensor.shape, dtype=np.int64).tobytes(order="C")
        )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _digest_framed_bytes(digest, raw)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise SO2NBTrainingError(
                f"Object ndarray cannot be checksummed at {path}."
            )
        digest.update(b"ndarray\0")
        _digest_framed_bytes(digest, array.dtype.str.encode("ascii"))
        _digest_framed_bytes(
            digest, np.asarray(array.shape, dtype=np.int64).tobytes(order="C")
        )
        _digest_framed_bytes(digest, array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        ordered = sorted(value, key=_checkpoint_mapping_sort_key)
        sort_keys = [_checkpoint_mapping_sort_key(key) for key in ordered]
        if len(sort_keys) != len(set(sort_keys)):
            raise SO2NBTrainingError(
                f"Checkpoint mapping keys have ambiguous ordering at {path}."
            )
        for index, key in enumerate(ordered):
            _update_checkpoint_tree_digest(
                digest, key, path=f"{path}.key[{index}]"
            )
            _update_checkpoint_tree_digest(
                digest, value[key], path=f"{path}[{key!r}]"
            )
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        for index, child in enumerate(value):
            _update_checkpoint_tree_digest(
                digest, child, path=f"{path}[{index}]"
            )
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        for index, child in enumerate(value):
            _update_checkpoint_tree_digest(
                digest, child, path=f"{path}[{index}]"
            )
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0\1" if value else b"bool\0\0")
        return
    if isinstance(value, int):
        digest.update(b"int\0")
        _digest_framed_bytes(digest, str(value).encode("ascii"))
        return
    if isinstance(value, np.generic):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and np.isnan(array).item():
            raise SO2NBTrainingError(
                f"NaN NumPy scalar cannot appear in checkpoint state at {path}."
            )
        digest.update(b"numpy_scalar\0")
        _digest_framed_bytes(digest, array.dtype.str.encode("ascii"))
        if np.issubdtype(array.dtype, np.floating) and np.isinf(array).item():
            digest.update(
                b"positive_infinity\0"
                if float(array.item()) > 0
                else b"negative_infinity\0"
            )
        else:
            _digest_framed_bytes(digest, array.tobytes(order="C"))
        return
    if isinstance(value, float):
        digest.update(b"float\0")
        if math.isnan(value):
            raise SO2NBTrainingError(
                f"NaN scalar cannot appear in checkpoint state at {path}."
            )
        if math.isinf(value):
            digest.update(b"positive_infinity\0" if value > 0 else b"negative_infinity\0")
        else:
            _digest_framed_bytes(digest, value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        digest.update(b"string\0")
        _digest_framed_bytes(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        _digest_framed_bytes(digest, value)
        return
    raise SO2NBTrainingError(
        f"Unsupported checkpoint value at {path}: {type(value).__name__}."
    )


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_checkpoint_tree_digest(digest, value, path="root")
    return digest.hexdigest()
PREFLIGHT_REQUIRED_GATES = frozenset(
    {
        "configuration_and_overlay",
        "model_topology",
        "neutral_geometry_initialization",
        "raw_target_forward_isolation",
        "full_constant_fp32_nb2",
        "amp_fp32_equivalence",
        "synthetic_nb2_loss_decrease",
        "real_subset_nb2_loss_decrease",
        "ddp_parameter_sync",
        "complete_pair_training_update",
        "preclip_gradient_flow",
        "fixed_validation_mask_regeneration",
        "fixed_validation_forward",
        "checkpoint_best_latest_reload",
        "gpu_inventory",
        "peak_vram",
        "no_test_split_or_artifacts",
    }
)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise SO2NBTrainingError(f"Resolved configuration requires {name!r} mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise SO2NBTrainingError(
            f"{field} must be {expected!r}; received {actual!r}."
        )


@contextmanager
def _synchronized_rank_zero_phase(
    *,
    rank: int,
    control_group: Any,
    phase: str,
    exception_type: type[Exception] = SO2NBTrainingError,
) -> Any:
    """Broadcast rank-zero success or failure before any rank can continue.

    The body must contain rank-zero-only work.  Peers enter the context with a
    no-op body, then wait on the CPU-backed control group.  A normal Python
    exception on rank zero is serialized into a small envelope so every rank
    raises the same phase failure instead of waiting in a later collective.
    """

    rank_zero_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        if int(rank) != 0:
            raise
        traceback.print_exc()
        rank_zero_error = error

    envelope: list[Any] = [None]
    if int(rank) == 0:
        if rank_zero_error is None:
            envelope[0] = {"ok": True}
        else:
            envelope[0] = {
                "ok": False,
                "error_type": type(rank_zero_error).__name__,
                "error": str(rank_zero_error) or repr(rank_zero_error),
            }
    torch.distributed.broadcast_object_list(
        envelope, src=0, group=control_group
    )
    status = envelope[0]
    if not isinstance(status, Mapping) or type(status.get("ok")) is not bool:
        raise exception_type(
            f"Rank-zero {phase} returned a malformed synchronization envelope."
        )
    if status["ok"] is not True:
        synchronized_error = exception_type(
            f"Rank-zero {phase} failed with {status.get('error_type', 'Exception')}: "
            f"{status.get('error', 'no error message')}"
        )
        if int(rank) == 0 and rank_zero_error is not None:
            raise synchronized_error from rank_zero_error
        raise synchronized_error


def _broadcast_rank_zero_result(
    *,
    rank: int,
    control_group: Any,
    phase: str,
    operation: Callable[[], Any],
    exception_type: type[Exception] = SO2NBTrainingError,
) -> Any:
    """Run a CPU-only rank-zero operation and broadcast its result or failure.

    Every rank must call this helper in the same collective order.  Only rank
    zero invokes ``operation``; that callback must not launch CUDA work or any
    distributed collective.  Python failures are converted to a small,
    pickle-safe envelope before every rank raises the same phase error.
    """

    rank_zero_error: BaseException | None = None
    envelope: list[Any] = [None]
    if int(rank) == 0:
        try:
            envelope[0] = {"ok": True, "result": operation()}
        except BaseException as error:
            traceback.print_exc()
            rank_zero_error = error
            envelope[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error) or repr(error),
            }

    torch.distributed.broadcast_object_list(
        envelope, src=0, group=control_group
    )
    outcome = envelope[0]
    if not isinstance(outcome, Mapping) or type(outcome.get("ok")) is not bool:
        raise exception_type(
            f"Rank-zero {phase} returned a malformed result envelope."
        )
    if outcome["ok"] is not True:
        synchronized_error = exception_type(
            f"Rank-zero {phase} failed with "
            f"{outcome.get('error_type', 'Exception')}: "
            f"{outcome.get('error', 'no error message')}"
        )
        if int(rank) == 0 and rank_zero_error is not None:
            raise synchronized_error from rank_zero_error
        raise synchronized_error
    if "result" not in outcome:
        raise exception_type(
            f"Rank-zero {phase} returned a malformed success envelope."
        )
    return outcome["result"]


def _campaign_id(config: Mapping[str, Any]) -> str:
    return str(_section(config, "campaign").get("campaign_id", "")).strip()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _configuration_source_file_hashes(paths: ProjectPaths) -> Mapping[str, str]:
    """Hash the source experiment and every directly selected default file."""

    config_path = (paths.project_root / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
    root = load_yaml_mapping(config_path)
    defaults = root.get("defaults")
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise SO2NBTrainingError("Source experiment defaults are malformed.")
    files = {config_path}
    for entry in defaults:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise SO2NBTrainingError("Source experiment default entry is malformed.")
        group, name = next(iter(entry.items()))
        files.add((paths.config_root / str(group) / f"{name}.yaml").resolve(strict=True))
    return {
        str(path.relative_to(paths.project_root)): sha256_file(path)
        for path in sorted(files)
    }


def _preflight_code_file_hashes(paths: ProjectPaths) -> Mapping[str, str]:
    return {
        str(relative): sha256_file((paths.project_root / relative).resolve(strict=True))
        for relative in PREFLIGHT_CODE_FILES
    }


def _overlay_source_artifacts(bundle: SO2NBDataBundle) -> Mapping[str, Any]:
    try:
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBTrainingError("Overlay manifest became unreadable.") from exc
    if not isinstance(manifest, Mapping):
        raise SO2NBTrainingError("Overlay manifest must be a mapping.")
    unsigned = dict(manifest)
    stored_content_sha = unsigned.pop("manifest_content_sha256", None)
    _require_equal(
        stored_content_sha,
        _canonical_sha256(unsigned),
        field="overlay.manifest_content_sha256",
    )
    _require_equal(
        stored_content_sha,
        bundle.manifest_content_sha256,
        field="overlay.bundle_manifest_content_sha256",
    )
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise SO2NBTrainingError("Overlay manifest lacks source-artifact evidence.")
    return dict(source_artifacts)


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise SO2NBTrainingError(f"Unsafe configured runtime path: {raw!r}.")
    roots = {
        "data": paths.data_root,
        "artifacts": paths.artifact_root,
        "state": paths.state_root,
        "scratch": paths.scratch_root,
        "reports": paths.report_root,
        "configs": paths.config_root,
    }
    if candidate.parts[0] in roots:
        return roots[candidate.parts[0]].joinpath(*candidate.parts[1:]).resolve(
            strict=False
        )
    return (paths.project_root / candidate).resolve(strict=False)


def validate_nb_config(config: Mapping[str, Any]) -> None:
    """Reassert runner-critical frozen values after general config validation."""

    validate_experiment_config(config)
    _require_equal(config.get("seed"), MODEL_SEED, field="seed")
    _require_equal(_campaign_id(config), CAMPAIGN_ID, field="campaign.campaign_id")
    metadata = _section(config, "metadata")
    preflight_acceptance = metadata.get("preflight_acceptance")
    if not isinstance(preflight_acceptance, Mapping):
        raise SO2NBTrainingError("metadata.preflight_acceptance must be a mapping.")
    _require_equal(
        preflight_acceptance.get("minimum_observed_cuda_memory_gib_each_rank"),
        23.0,
        field=(
            "metadata.preflight_acceptance."
            "minimum_observed_cuda_memory_gib_each_rank"
        ),
    )
    if "gpu_memory_gib_each_rank" in preflight_acceptance:
        raise SO2NBTrainingError(
            "metadata.preflight_acceptance.gpu_memory_gib_each_rank is obsolete."
        )
    evaluation = _section(config, "evaluation")
    _require_equal(evaluation.get("protocol"), PROTOCOL, field="evaluation.protocol")
    _require_equal(evaluation.get("splits"), ["validation"], field="evaluation.splits")
    _require_equal(evaluation.get("test_split_present"), False, field="evaluation.test")

    dataset = _section(config, "dataset")
    _require_equal(
        tuple(dataset.get("training_core_aliases", ())),
        TRAINING_ALIASES,
        field="dataset.training_core_aliases",
    )
    _require_equal(
        tuple(dataset.get("validation_core_aliases", ())),
        VALIDATION_ALIASES,
        field="dataset.validation_core_aliases",
    )
    _require_equal(
        tuple(dataset.get("test_core_aliases", ())),
        (),
        field="dataset.test_core_aliases",
    )
    _require_equal(dataset.get("test_partition_present"), False, field="dataset.test")

    model = _section(config, "model")
    for field, expected in {
        "hidden_dim": 256,
        "graph_layers": 4,
        "unique_graph_blocks": 4,
        "effective_graph_depth": 4,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
        "output_mean_epsilon": 1e-4,
        "inverse_dispersion_epsilon": 1e-4,
        "inverse_dispersion_initial_value": 1.0,
    }.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")

    masking = _section(config, "masking")
    for field, expected in {
        "count_min": 0,
        "count_max": 1000,
        "independent_views_per_core_epoch": TRAINING_VIEW_COUNT,
        "mask_base_seed": MASK_BASE_SEED,
    }.items():
        _require_equal(masking.get(field), expected, field=f"masking.{field}")
    fixed_masks = masking.get("validation_masks")
    if not isinstance(fixed_masks, Mapping):
        raise SO2NBTrainingError("masking.validation_masks must be a mapping.")
    _require_equal(
        fixed_masks.get("independent_views_per_core"),
        VALIDATION_VIEW_COUNT,
        field="masking.validation_masks.independent_views_per_core",
    )
    _require_equal(
        fixed_masks.get("fixed_across_epochs"),
        True,
        field="masking.validation_masks.fixed_across_epochs",
    )

    trainer = _section(config, "trainer")
    for field, expected in {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "gradient_clip_norm": 1.0,
        "max_epochs": MAXIMUM_EPOCHS,
        "minimum_global_epochs": MINIMUM_EPOCHS,
        "optimizer_updates_per_global_epoch": 6,
        "mask_views_per_core_step": TRAINING_VIEW_COUNT,
        "scheduler": "reduce_lr_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "scheduler_threshold_mode": "abs",
        "scheduler_threshold": 1e-4,
        "scheduler_min_learning_rate": 1e-6,
        "early_stopping": True,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "early_stopping_patience_clock_starts_after_minimum_epoch": True,
        "validation_every": 1,
        "restore_best": True,
        "primary_checkpoint_role": "best",
        "distributed_world_size": WORLD_SIZE,
        "amp": True,
        "likelihood_compute_dtype": "float32",
        "likelihood_outside_autocast": True,
        "amp_requires_fp32_equivalence_preflight": True,
    }.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")

    launcher = _section(config, "launcher")
    _require_equal(launcher.get("process_count"), WORLD_SIZE, field="launcher.process_count")
    _require_equal(
        launcher.get("require_exact_visible_devices"),
        VISIBLE_DEVICES,
        field="launcher.require_exact_visible_devices",
    )


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO2NBTrainingError("This runner must be invoked by torchrun.") from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise SO2NBTrainingError("Exactly four distributed ranks are required.")
    if local_rank not in range(WORLD_SIZE) or rank != local_rank:
        raise SO2NBTrainingError("Single-node rank and LOCAL_RANK must match.")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise SO2NBTrainingError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise SO2NBTrainingError("Four logical CUDA devices are required.")
    return rank, local_rank, world_size


def _load_worker_config(
    args: argparse.Namespace,
) -> tuple[str, Path | None, dict[str, Any], ProjectPaths]:
    paths = current_paths()
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch or args.run_scratch is None:
        raise SO2NBTrainingError(
            "BAGM_RUN_ID, BAGM_RUN_SCRATCH, and --run-scratch are required."
        )
    scratch = args.run_scratch.resolve(strict=False)
    if scratch != Path(environment_scratch).resolve(strict=False):
        raise SO2NBTrainingError("--run-scratch does not match BAGM_RUN_SCRATCH.")
    expected_config = scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise SO2NBTrainingError("--config must be the worker-resolved config.")
    expected_scratch = (paths.scratch_root / "active_runs" / run_id).resolve(False)
    if scratch != expected_scratch:
        raise SO2NBTrainingError("Run scratch is not canonical for BAGM_RUN_ID.")
    config = load_yaml_mapping(expected_config)
    validate_nb_config(config)
    return run_id, scratch, config, paths


def _validate_bundle_binding(
    config: Mapping[str, Any], bundle: SO2NBDataBundle
) -> None:
    dataset = _section(config, "dataset")
    expected = {
        "dataset.dataset_fingerprint": (
            dataset.get("dataset_fingerprint"), bundle.manifest_content_sha256
        ),
        "dataset.overlay_manifest_file_sha256": (
            dataset.get("overlay_manifest_file_sha256"), bundle.manifest_sha256
        ),
        "dataset.split_fingerprint": (
            dataset.get("split_fingerprint"), bundle.split_fingerprint
        ),
        "dataset.preprocessing_fingerprint": (
            dataset.get("preprocessing_fingerprint"), bundle.preprocessing_fingerprint
        ),
    }
    for field, (configured, loaded) in expected.items():
        _require_equal(configured, loaded, field=field)
    _require_equal(
        dataset.get("dataset_fingerprint_role"),
        "immutable_overlay_manifest_content_sha256",
        field="dataset.dataset_fingerprint_role",
    )


def _load_data(config: Mapping[str, Any], paths: ProjectPaths) -> SO2NBDataBundle:
    dataset = _section(config, "dataset")
    bundle = load_so2_nb_data(
        _runtime_path(dataset["prepared_artifact_reference"], paths),
        cohort_dir=_runtime_path(dataset["source_cohort_artifact"], paths),
        graph_dir=_runtime_path(dataset["source_graph_artifact"], paths),
    )
    _validate_bundle_binding(config, bundle)
    return bundle


def _build_model(
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    *,
    receiver_chunk_override: int | None = None,
) -> ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer:
    model_config = _section(config, "model")
    trainer = _section(config, "trainer")
    set_deterministic_seed(
        MODEL_SEED,
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
    first = bundle.training_batches[0]
    model = ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=first.n_genes,
        node_covariate_dim=int(first.node_covariates.shape[1]),
        hidden_dim=int(model_config["hidden_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        attention_head_dim=int(model_config["attention_head_dim"]),
        graph_layers=int(model_config["graph_layers"]),
        ffn_dim=int(model_config["ffn_dim"]),
        decoder_dim=int(model_config["decoder_dim"]),
        geometry_hidden_dim=int(model_config["geometry_hidden_dim"]),
        dropout=float(model_config["dropout"]),
        attention_dropout=float(model_config["attention_dropout"]),
        relative_geometry_dim=int(model_config["relative_geometry_dim"]),
        qk_normalization_epsilon=float(model_config["qk_normalization_epsilon"]),
        logit_scale_initial=float(model_config["logit_scale_initial"]),
        logit_scale_minimum=float(model_config["logit_scale_minimum"]),
        logit_scale_maximum=float(model_config["logit_scale_maximum"]),
        modulation_amplitude=float(model_config["modulation_amplitude"]),
        geometry_bias_bound=float(model_config["geometry_bias_bound"]),
        receiver_chunk_size=(
            int(receiver_chunk_override)
            if receiver_chunk_override is not None
            else int(model_config["receiver_chunk_size"])
        ),
        max_edges_per_chunk=int(model_config["max_edges_per_chunk"]),
        activation_checkpointing=bool(model_config["activation_checkpointing"]),
        mean_epsilon=float(model_config["output_mean_epsilon"]),
        inverse_dispersion_epsilon=float(
            model_config["inverse_dispersion_epsilon"]
        ),
        inverse_dispersion_initial_value=float(
            model_config["inverse_dispersion_initial_value"]
        ),
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    _require_equal(count, EXPECTED_PARAMETER_COUNT, field="model.parameter_count")
    blocks = tuple(model.blocks)
    if len(blocks) != 4 or len({id(block) for block in blocks}) != 4:
        raise SO2NBTrainingError("Model must contain four unique graph blocks.")
    parameter_ids = [id(p) for block in blocks for p in block.parameters()]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise SO2NBTrainingError("Graph block parameter sets overlap.")
    _require_equal(model.raw_theta.numel(), 1000, field="model.raw_theta")
    return model


def _current_gpu_identity() -> Mapping[str, Any]:
    try:
        rank = int(os.environ["RANK"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO2NBTrainingError("RANK is required for live GPU receipt binding.") from exc
    local_rank = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "total_memory_gib": int(properties.total_memory) / float(1024**3),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
    }


def _validate_preflight_gpu_identities(
    receipt: Mapping[str, Any],
    gates: Mapping[str, Any],
    per_rank_vram: Sequence[Mapping[str, Any]],
    *,
    current_identity: Mapping[str, Any],
) -> None:
    """Bind preflight hardware evidence to the live rank and VRAM records."""

    raw_identities = receipt.get("gpu_identities")
    if not isinstance(raw_identities, Sequence) or isinstance(
        raw_identities, (str, bytes)
    ) or len(raw_identities) != WORLD_SIZE:
        raise SO2NBTrainingError("Preflight must contain exactly four GPU identities.")
    identities = [dict(item) for item in raw_identities if isinstance(item, Mapping)]
    if len(identities) != WORLD_SIZE:
        raise SO2NBTrainingError("A preflight GPU identity is malformed.")
    identity_fields = {
        "rank",
        "local_rank",
        "name",
        "total_memory_bytes",
        "total_memory_gib",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
    }
    if any(set(identity) != identity_fields for identity in identities):
        raise SO2NBTrainingError("Preflight GPU identity schema drifted.")
    inventory = gates.get("gpu_inventory")
    if not isinstance(inventory, Mapping):
        raise SO2NBTrainingError("Preflight GPU-inventory gate is malformed.")
    _require_equal(
        inventory.get("devices"), raw_identities, field="preflight.gpu_inventory.devices"
    )
    by_rank = {int(identity["rank"]): identity for identity in identities}
    if (
        set(by_rank) != set(range(WORLD_SIZE))
        or {int(identity["local_rank"]) for identity in identities}
        != set(range(WORLD_SIZE))
        or any(
            identity["name"] != "NVIDIA GeForce RTX 3090"
            or identity["compute_capability"] != [8, 6]
            or int(identity["total_memory_bytes"]) <= 0
            or not math.isclose(
                float(identity["total_memory_gib"]),
                int(identity["total_memory_bytes"]) / float(1024**3),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not str(identity["torch_version"])
            or not str(identity["cuda_runtime"])
            for identity in identities
        )
    ):
        raise SO2NBTrainingError("Preflight GPU identities violate the frozen hardware contract.")

    vram_by_rank = {int(record.get("rank", -1)): record for record in per_rank_vram}
    if set(vram_by_rank) != set(by_rank):
        raise SO2NBTrainingError("GPU identity and VRAM rank coverage disagree.")
    for rank, identity in by_rank.items():
        vram = vram_by_rank[rank]
        for field in ("local_rank", "total_memory_bytes"):
            _require_equal(
                int(vram.get(field, -1)),
                int(identity[field]),
                field=f"preflight.gpu_vram_identity.{rank}.{field}",
            )

    live_rank = int(current_identity.get("rank", -1))
    if live_rank not in by_rank:
        raise SO2NBTrainingError("Live DDP rank is absent from the GPU receipt.")
    expected_live = by_rank[live_rank]
    for field in (
        "rank",
        "local_rank",
        "name",
        "total_memory_bytes",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
    ):
        _require_equal(
            current_identity.get(field),
            expected_live.get(field),
            field=f"preflight.live_gpu_identity.{field}",
        )


def _load_preflight_receipt(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    bundle: SO2NBDataBundle,
    model: nn.Module,
) -> Mapping[str, Any]:
    launcher = _section(config, "launcher")
    path = _runtime_path(launcher["hardware_preflight_receipt"], paths)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBTrainingError("Hardware preflight receipt is missing or unreadable.") from exc
    if not isinstance(receipt, Mapping):
        raise SO2NBTrainingError("Hardware preflight receipt must be a mapping.")
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "world_size": WORLD_SIZE,
        "control_plane_backend": "gloo",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "checkpoint_reload_verified": True,
        "train_update_completed": True,
        "validation_completed": True,
        "test_artifacts_present": False,
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "configuration_sha256": _canonical_sha256(config),
        "resolved_config_sha256": _canonical_sha256(config),
        "configuration_file_sha256": sha256_file(
            (paths.project_root / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
        ),
        "configuration_source_file_sha256": _configuration_source_file_hashes(paths),
        "code_hash_scope": CODE_HASH_SCOPE,
        "code_file_sha256": _preflight_code_file_hashes(paths),
        "source_artifacts": _overlay_source_artifacts(bundle),
    }
    for field, value in expected.items():
        _require_equal(receipt.get(field), value, field=f"preflight.{field}")
    _require_equal(
        receipt.get("receiver_chunk_size"),
        int(getattr(model, "receiver_chunk_size")),
        field="preflight.receiver_chunk_size",
    )
    gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != PREFLIGHT_REQUIRED_GATES:
        raise SO2NBTrainingError("Hardware preflight required-gate set is incomplete.")
    if any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        raise SO2NBTrainingError("Hardware preflight contains a failed gate.")
    stored_receipt_sha = receipt.get("receipt_content_sha256")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_content_sha256", None)
    _require_equal(
        stored_receipt_sha,
        _canonical_sha256(unsigned_receipt),
        field="preflight.receipt_content_sha256",
    )
    contract_path = (paths.project_root / FROZEN_TASK_CONTRACT).resolve(strict=True)
    _require_equal(
        sha256_file(contract_path),
        FROZEN_TASK_CONTRACT_SHA256,
        field="frozen_task_contract.file_sha256",
    )
    peak = float(receipt.get("peak_vram_gib_all_ranks", float("nan")))
    peak_reserved = float(
        receipt.get("peak_reserved_vram_gib_all_ranks", float("nan"))
    )
    headroom = float(receipt.get("minimum_vram_headroom_gib_each_rank", float("nan")))
    reserved_headroom = float(
        receipt.get("minimum_reserved_vram_headroom_gib_each_rank", float("nan"))
    )
    peak_gate = gates["peak_vram"]
    per_rank_vram = peak_gate.get("per_rank")
    if not isinstance(per_rank_vram, Sequence) or isinstance(
        per_rank_vram, (str, bytes)
    ) or len(per_rank_vram) != WORLD_SIZE:
        raise SO2NBTrainingError("Hardware preflight lacks four-rank VRAM evidence.")
    if any(not isinstance(record, Mapping) for record in per_rank_vram):
        raise SO2NBTrainingError("Hardware preflight VRAM record is malformed.")
    _validate_preflight_gpu_identities(
        receipt,
        gates,
        per_rank_vram,
        current_identity=_current_gpu_identity(),
    )
    normalized_vram: list[tuple[int, int, float, float, float]] = []
    for record in per_rank_vram:
        if not isinstance(record, Mapping):
            raise SO2NBTrainingError("Hardware preflight VRAM record is malformed.")
        total = int(record.get("total_memory_bytes", 0)) / float(1024**3)
        allocated = float(record.get("peak_allocated_vram_gib", float("nan")))
        reserved = float(record.get("peak_reserved_vram_gib", float("nan")))
        normalized_vram.append(
            (
                int(record.get("rank", -1)),
                int(record.get("local_rank", -1)),
                total,
                allocated,
                reserved,
            )
        )
    observed_peak = max(record[3] for record in normalized_vram)
    observed_reserved = max(record[4] for record in normalized_vram)
    observed_headroom = min(record[2] - record[3] for record in normalized_vram)
    observed_reserved_headroom = min(
        record[2] - record[4] for record in normalized_vram
    )
    if (
        {record[0] for record in normalized_vram} != set(range(WORLD_SIZE))
        or {record[1] for record in normalized_vram} != set(range(WORLD_SIZE))
        or any(
            not all(math.isfinite(value) for value in record[2:])
            or record[2] < 23.0
            or not (0.0 < record[3] <= record[4] <= record[2])
            for record in normalized_vram
        )
        or not math.isfinite(peak)
        or peak <= 0.0
        or peak > PREFLIGHT_MAX_VRAM_GIB
        or not math.isfinite(peak_reserved)
        or peak_reserved < peak
        or not math.isfinite(headroom)
        or headroom < PREFLIGHT_MINIMUM_HEADROOM_GIB
        or not math.isfinite(reserved_headroom)
        or reserved_headroom < PREFLIGHT_MINIMUM_HEADROOM_GIB
        or not math.isclose(peak, observed_peak, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            peak_reserved, observed_reserved, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            headroom, observed_headroom, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            reserved_headroom,
            observed_reserved_headroom,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SO2NBTrainingError("Hardware preflight VRAM gate failed.")
    return receipt


def _training_mask(
    batch: SO2NBCoreBatch,
    *,
    global_epoch: int,
    view_index: int,
) -> Any:
    """Build a stochastic Uniform{0,...,G} mask independent of model seed."""

    if view_index not in range(TRAINING_VIEW_COUNT):
        raise SO2NBTrainingError("Training mask view must be 0 through 9.")
    return make_cohort_exact_uniform_training_mask(
        batch,
        global_epoch,
        view_index=view_index,
        mask_base_seed=MASK_BASE_SEED,
        chunk_cells=2048,
        maximum_zero_total_resamples=1024,
    )


def _model_view_seed(
    global_epoch: int,
    update_index: int,
    alias: str,
    view_index: int,
) -> int:
    return cohort_relative_qkv_model_step_seed(
        MODEL_SEED,
        global_epoch,
        update_index,
        alias,
        view_index,
    )


def _seed_model_view(seed: int, device: torch.device) -> None:
    torch.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def _stage_batch(
    batch: SO2NBCoreBatch,
    *,
    device: torch.device,
    trainer: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    raw_target = batch.raw_count_target.to(device=device, dtype=torch.int32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    if bool(trainer["stage_complete_core_graph_on_device"]):
        edges = batch.edge_index.to(device=device, dtype=torch.long)
        geometry_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[str(trainer["staged_relative_geometry_dtype"]).lower()]
        geometry = batch.relative_geometry.to(device=device, dtype=geometry_dtype)
    else:
        edges = batch.edge_index
        geometry = batch.relative_geometry
    return input_expression, raw_target, covariates, edges, geometry


def _assert_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise SO2NBTrainingError(f"Trainable parameter {name} has no gradient.")
        gradient = parameter.grad
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite gradient in {name}.")


class _RawThetaParameterView(nn.Module):
    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        self.register_parameter("raw_theta", parameter)


def _gather_objects(local: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(gathered, local)
    return gathered


def _paired_update(
    *,
    model: ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
    training_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    pair: tuple[str, str],
    update_index: int,
    global_epoch: int,
    rank: int,
    device: torch.device,
    batches: Mapping[str, SO2NBCoreBatch],
    trainer: Mapping[str, Any],
    global_tracker: FullGradientDirectionTracker,
    block_tracker: BlockGradientDirectionTracker,
    theta_tracker: FullGradientDirectionTracker,
    theta_view: nn.Module,
) -> tuple[list[Mapping[str, Any]], float]:
    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    assignment = cohort_relative_qkv_distributed_assignment(
        pair, rank=rank, world_size=WORLD_SIZE
    )
    batch = batches[assignment.alias]
    _clear_graph_layout_caches(training_model)
    input_expression, raw_target, covariates, edges, geometry = _stage_batch(
        batch, device=device, trainer=trainer
    )
    local_records: list[Mapping[str, Any]] = []
    try:
        target_nodes = torch.arange(batch.n_nodes, device=device, dtype=torch.long)
        for view_index in assignment.view_indices:
            _seed_model_view(
                _model_view_seed(global_epoch, update_index, batch.alias, view_index),
                device,
            )
            realization = _training_mask(
                batch, global_epoch=global_epoch, view_index=view_index
            )
            mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
                device=device, dtype=torch.bool
            )
            masked_input = input_expression.masked_fill(mask, 0.0)
            with _autocast_context(
                enabled=bool(trainer["amp"]),
                device=device,
                dtype_name=str(trainer["amp_dtype"]),
            ):
                output = training_model(
                    input_expression=masked_input,
                    gene_mask=mask,
                    edge_index=edges,
                    relative_geometry=geometry,
                    node_covariates=covariates,
                    target_nodes=target_nodes,
                )
            if not isinstance(output, NegativeBinomialModelOutput):
                raise SO2NBTrainingError("NB model returned the wrong output type.")
            loss = masked_negative_binomial_nll(
                output.mu, output.theta, raw_target, mask
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Training NB NLL is non-finite.")
            scaler.scale(loss / float(assignment.local_loss_divisor)).backward()
            counts = realization.masked_gene_counts
            local_records.append(
                {
                    "alias": batch.alias,
                    "split_role": "training",
                    "view_index": int(view_index),
                    "negative_binomial_nll": float(loss.detach().cpu()),
                    "n_masked_entries": int(realization.n_masked_entries),
                    "initial_mask_seed": int(realization.initial_seed),
                    "effective_mask_seed": int(realization.effective_seed),
                    "zero_total_mask_resamples": int(
                        realization.zero_total_resample_count
                    ),
                    "mask_seed": int(realization.effective_seed),
                    "mask_checksum": str(realization.checksum_sha256),
                    "masked_count_min": int(counts.min()),
                    "masked_count_mean": float(counts.mean()),
                    "masked_count_median": float(np.median(counts)),
                    "masked_count_max": int(counts.max()),
                    "zero_mask_cells": int(np.count_nonzero(counts == 0)),
                    "full_mask_cells": int(np.count_nonzero(counts == batch.n_genes)),
                }
            )
            del output, loss, masked_input, mask
        gathered = _gather_objects(local_records)
        complete_records = [record for rank_records in gathered for record in rank_records]
        if len(complete_records) != 20:
            raise SO2NBTrainingError("A paired update must cover exactly twenty views.")

        scaler.unscale_(optimizer)
        _assert_finite_gradients(model)
        context = GradientDirectionUpdateContext(
            global_epoch=global_epoch,
            completed_global_epoch=global_epoch + 1,
            optimizer_update_in_epoch=update_index,
            cumulative_optimizer_update=global_epoch * 6 + update_index + 1,
            aliases=pair,
        )
        # Keep every rank on the same CUDA workload before the next NCCL
        # collective.  Rank-zero-only GPU observations can leave peer ranks
        # spinning in the following collective while rank zero synchronizes
        # gradient-vector kernels, which can deadlock on a single-host NCCL
        # process group.  Only rank zero persists the resulting scalars.
        global_tracker.observe(model, context)
        block_tracker.observe(model, context)
        theta_tracker.observe(theta_view, context)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            training_model.parameters(),
            float(trainer["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        norm_tensor = gradient_norm.detach().to(device=device, dtype=torch.float64)
        minimum = norm_tensor.clone()
        maximum = norm_tensor.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        if not torch.isclose(minimum, maximum, rtol=1e-5, atol=1e-7):
            raise SO2NBTrainingError("Post-DDP gradient norms disagree across ranks.")
        scaler.step(optimizer)
        scaler.update()
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"Non-finite parameter after update: {name}.")
        return complete_records, float(gradient_norm.detach().cpu())
    finally:
        _clear_graph_layout_caches(training_model)
        del input_expression, raw_target, covariates, edges, geometry
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _validation_assignment(rank: int) -> tuple[str, tuple[int, ...]]:
    alias = VALIDATION_ALIASES[0 if rank < 2 else 1]
    start = (rank % 2) * 5
    return alias, tuple(range(start, start + 5))


def _evaluate_validation(
    *,
    bundle: SO2NBDataBundle,
    training_model: nn.Module,
    rank: int,
    device: torch.device,
    trainer: Mapping[str, Any],
) -> tuple[ValidationAggregate, Mapping[str, Mapping[str, Any]]]:
    training_model.eval()
    alias, view_indices = _validation_assignment(rank)
    batch = bundle.batches_by_alias[alias]
    _clear_graph_layout_caches(training_model)
    input_expression, raw_target, covariates, edges, geometry = _stage_batch(
        batch, device=device, trainer=trainer
    )
    local_statistics: list[ValidationViewStatistics] = []
    local_receipts: dict[str, Mapping[str, Any]] = {}
    try:
        target_nodes = torch.arange(batch.n_nodes, device=device, dtype=torch.long)
        with torch.no_grad():
            for view_index in view_indices:
                realization = bundle.validation_mask(alias, view_index)
                if realization.n_masked_entries <= 0:
                    raise SO2NBTrainingError("Fixed validation mask is empty.")
                mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
                    device=device, dtype=torch.bool
                )
                masked_input = input_expression.masked_fill(mask, 0.0)
                with _autocast_context(
                    enabled=bool(trainer["amp"]),
                    device=device,
                    dtype_name=str(trainer["amp_dtype"]),
                ):
                    output = training_model(
                        input_expression=masked_input,
                        gene_mask=mask,
                        edge_index=edges,
                        relative_geometry=geometry,
                        node_covariates=covariates,
                        target_nodes=target_nodes,
                    )
                if not isinstance(output, NegativeBinomialModelOutput):
                    raise SO2NBTrainingError("Validation model output is not NB2.")
                metrics = masked_negative_binomial_metrics(
                    output.mu, output.theta, raw_target, mask
                )
                _require_equal(
                    int(metrics.n_masked_entries),
                    int(realization.n_masked_entries),
                    field="validation.n_masked_entries",
                )
                observed_count_sum = float(
                    raw_target.masked_select(mask).sum(dtype=torch.float64).cpu()
                )
                predicted_count_sum = float(
                    output.mu.masked_select(mask).sum(dtype=torch.float64).cpu()
                )
                local_statistics.append(
                    ValidationViewStatistics.from_metrics(
                        alias=alias,
                        view_index=view_index,
                        metrics=metrics,
                        observed_count_sum=observed_count_sum,
                        predicted_count_sum=predicted_count_sum,
                    )
                )
                local_receipts[f"{alias}:{view_index}"] = realization.to_receipt()
                del output, metrics, masked_input, mask
        tensor = validation_statistics_tensor(local_statistics, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        aggregate = aggregate_validation_statistics(
            validation_statistics_from_tensor(tensor)
        )
        receipt_parts = _gather_objects(local_receipts)
        receipts: dict[str, Mapping[str, Any]] = {}
        for part in receipt_parts:
            for key, value in part.items():
                if key in receipts:
                    raise SO2NBTrainingError("Duplicate fixed validation mask receipt.")
                receipts[key] = value
        expected_keys = {
            f"{core}:{view}"
            for core in VALIDATION_ALIASES
            for view in range(VALIDATION_VIEW_COUNT)
        }
        if set(receipts) != expected_keys:
            raise SO2NBTrainingError("Fixed validation mask coverage is incomplete.")
        return aggregate, dict(sorted(receipts.items()))
    finally:
        training_model.train()
        _clear_graph_layout_caches(training_model)
        del input_expression, raw_target, covariates, edges, geometry
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _all_rank_rng_states() -> tuple[Mapping[str, Any], ...]:
    return tuple(_gather_objects(capture_rng_state()))


def _checkpoint_state_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    early_state: EarlyStoppingState,
    bundle: SO2NBDataBundle,
    rng_states: Sequence[Mapping[str, Any]],
    validation_mask_receipts: Mapping[str, Mapping[str, Any]],
    configuration_sha256: str,
) -> Mapping[str, Any]:
    """Snapshot current train state to CPU on every rank before rendezvous."""

    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "completed_epoch": int(completed_epoch),
        "optimizer_updates_completed": int(completed_epoch) * 6,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": _clone_tree_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_tree_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _clone_tree_to_cpu(scheduler.state_dict()),
        "amp_scaler_state_dict": _clone_tree_to_cpu(scaler.state_dict()),
        "rng_states": tuple(rng_states),
        "early_stopping_state": asdict(early_state),
        "best_validation_metric": early_state.best_value,
        "best_epoch": early_state.best_epoch,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": configuration_sha256,
        "validation_mask_receipts": dict(validation_mask_receipts),
        "gradient_vectors_persisted": False,
        "test_artifacts_present": False,
    }


def _checkpoint_payload_from_state(
    checkpoint_state: Mapping[str, Any],
    *,
    role: str,
    best_checkpoint_sha256: str | None,
    embedded_best_checkpoint: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    prohibited = {
        "checkpoint_role",
        "best_checkpoint_sha256",
        "embedded_best_checkpoint",
        "embedded_best_checkpoint_tree_sha256",
    }
    if prohibited.intersection(checkpoint_state):
        raise SO2NBTrainingError("Checkpoint state already contains role fields.")
    payload: dict[str, Any] = {
        **dict(checkpoint_state),
        "checkpoint_role": role,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "embedded_best_checkpoint": None,
        "embedded_best_checkpoint_tree_sha256": None,
    }
    if role == "best":
        if best_checkpoint_sha256 is not None or embedded_best_checkpoint is not None:
            raise SO2NBTrainingError("Best checkpoint payload cannot reference itself.")
        return payload
    if role != "latest":
        raise SO2NBTrainingError("Checkpoint role must be latest or best.")
    if best_checkpoint_sha256 is None:
        raise SO2NBTrainingError("Latest checkpoint must bind its best file checksum.")
    if embedded_best_checkpoint is None:
        # Epoch one is simultaneously current and best in the bounded preflight.
        early_state = EarlyStoppingState(
            **dict(payload["early_stopping_state"])
        )
        if (
            early_state.best_epoch != int(payload["completed_epoch"])
            or not early_state.improved
        ):
            raise SO2NBTrainingError(
                "A non-best latest checkpoint requires the persisted best payload."
            )
        embedded_best_checkpoint = {
            **payload,
            "checkpoint_role": "best",
            "best_checkpoint_sha256": None,
        }
    embedded = _clone_tree_to_cpu(dict(embedded_best_checkpoint))
    payload["embedded_best_checkpoint"] = embedded
    payload["embedded_best_checkpoint_tree_sha256"] = _tree_sha256(embedded)
    return payload


def _checkpoint_payload(
    *,
    role: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    early_state: EarlyStoppingState,
    bundle: SO2NBDataBundle,
    rng_states: Sequence[Mapping[str, Any]],
    validation_mask_receipts: Mapping[str, Mapping[str, Any]],
    configuration_sha256: str,
    best_checkpoint_sha256: str | None,
    embedded_best_checkpoint: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Compatibility wrapper for callers that do not need rank alignment."""

    state = _checkpoint_state_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=completed_epoch,
        early_state=early_state,
        bundle=bundle,
        rng_states=rng_states,
        validation_mask_receipts=validation_mask_receipts,
        configuration_sha256=configuration_sha256,
    )
    return _checkpoint_payload_from_state(
        state,
        role=role,
        best_checkpoint_sha256=best_checkpoint_sha256,
        embedded_best_checkpoint=embedded_best_checkpoint,
    )


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    bundle: SO2NBDataBundle,
    rank: int,
    configuration_sha256: str,
) -> tuple[EarlyStoppingState, Mapping[str, Mapping[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SO2NBTrainingError("Resume checkpoint is not a mapping.")
    for field, expected in {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_role": "latest",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": configuration_sha256,
        "gradient_vectors_persisted": False,
        "test_artifacts_present": False,
    }.items():
        _require_equal(payload.get(field), expected, field=f"resume.{field}")
    early = EarlyStoppingState(**dict(payload["early_stopping_state"]))
    _require_equal(
        payload.get("completed_epoch"), early.completed_epoch, field="resume.epoch"
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(dict(payload["amp_scaler_state_dict"]))
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, Sequence) or len(rng_states) != WORLD_SIZE:
        raise SO2NBTrainingError("Resume checkpoint lacks four rank RNG states.")
    restore_rng_state(rng_states[rank])
    receipts = payload.get("validation_mask_receipts")
    if not isinstance(receipts, Mapping) or len(receipts) != 20:
        raise SO2NBTrainingError("Resume checkpoint lacks fixed validation receipts.")
    sibling_best = path.parent / "best.ckpt"
    expected_best_sha = payload.get("best_checkpoint_sha256")
    if not sibling_best.is_file() or sha256_file(sibling_best) != expected_best_sha:
        raise SO2NBTrainingError(
            "Resume checkpoint transaction is incomplete after rank-zero reconciliation."
        )
    embedded = payload.get("embedded_best_checkpoint")
    if not isinstance(embedded, Mapping):
        raise SO2NBTrainingError("Resume checkpoint lacks embedded best recovery state.")
    _require_equal(
        _tree_sha256(embedded),
        payload.get("embedded_best_checkpoint_tree_sha256"),
        field="resume.embedded_best_checkpoint_tree_sha256",
    )
    sibling_payload = torch.load(sibling_best, map_location="cpu", weights_only=False)
    if not isinstance(sibling_payload, Mapping):
        raise SO2NBTrainingError("Resume best checkpoint is not a mapping.")
    _require_equal(
        _tree_sha256(sibling_payload),
        payload.get("embedded_best_checkpoint_tree_sha256"),
        field="resume.best_checkpoint_tree_sha256",
    )
    return early, dict(receipts)


def _reconcile_checkpoint_transaction(
    store: AtomicBestLatestCheckpointStore,
) -> bool:
    """Make ``best.ckpt`` match the last atomically committed latest payload.

    Metrics are written before checkpoints and ``latest.ckpt`` is the commit
    point.  A crash after replacing best but before replacing latest is safely
    rolled back by the complete best payload embedded in the old latest.
    """

    latest = store.load("latest")
    embedded = latest.get("embedded_best_checkpoint")
    if not isinstance(embedded, Mapping):
        raise SO2NBTrainingError("Latest checkpoint lacks embedded best state.")
    embedded_sha = _tree_sha256(embedded)
    _require_equal(
        latest.get("embedded_best_checkpoint_tree_sha256"),
        embedded_sha,
        field="latest.embedded_best_checkpoint_tree_sha256",
    )
    expected_file_sha = latest.get("best_checkpoint_sha256")
    sibling_matches = (
        store.best_path.is_file()
        and sha256_file(store.best_path) == expected_file_sha
    )
    if sibling_matches:
        sibling = store.load("best")
        _require_equal(
            _tree_sha256(sibling), embedded_sha, field="latest.best_checkpoint_tree"
        )
        return False

    restored = store.save_best(embedded)
    repaired_latest = dict(latest)
    repaired_latest["best_checkpoint_sha256"] = restored.sha256
    store.save_latest(repaired_latest)
    return True


def _peak_vram_all_ranks(device: torch.device) -> float:
    local = torch.tensor(
        [float(torch.cuda.max_memory_allocated(device))],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX)
    return float(local.item()) / float(1024**3)


def _theta_summary(model: Any) -> tuple[float, float, float]:
    theta = model.theta.detach().float()
    if not bool(torch.isfinite(theta).all()) or not bool((theta > 0).all()):
        raise FloatingPointError("Inverse dispersion is non-finite or non-positive.")
    return (
        float(theta.min().cpu()),
        float(theta.median().cpu()),
        float(theta.max().cpu()),
    )


def _gradient_rows(summary: Any, *, run_id: str) -> Mapping[str, Any]:
    row = asdict(summary)
    row["run_id"] = str(run_id)
    row["model_seed"] = MODEL_SEED
    return row


def _make_epoch_row(
    *,
    completed_epoch: int,
    train_nll: float,
    train_per_core: Mapping[str, float],
    validation: ValidationAggregate,
    theta: tuple[float, float, float],
    learning_rate: float,
    duration: float,
    peak_vram: float,
    early: EarlyStoppingState,
    global_gradient: Mapping[str, Any],
    block_gradients: Sequence[Mapping[str, Any]],
    theta_gradient: Mapping[str, Any],
    training_mask_receipts: Sequence[Mapping[str, Any]],
    validation_mask_receipts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema": EPOCH_METRICS_SCHEMA,
        "global_epoch": completed_epoch,
        "training_equal_core_masked_nb_nll": train_nll,
        "validation_equal_core_masked_nb_nll": validation.primary_equal_core_nll,
        "validation_pooled_masked_nb_nll": validation.pooled_nll,
        "validation_masked_raw_count_mae": validation.raw_count_mae,
        "validation_masked_raw_count_rmse": validation.raw_count_rmse,
        "validation_masked_log1p_mae": validation.log1p_mae,
        "validation_masked_log1p_rmse": validation.log1p_rmse,
        "validation_masked_poisson_deviance": validation.poisson_deviance,
        "validation_observed_zero_rate": validation.observed_zero_rate,
        "validation_predicted_zero_probability_mean": (
            validation.predicted_zero_probability_mean
        ),
        "validation_zero_brier_score": validation.zero_brier_score,
        "inverse_dispersion_min": theta[0],
        "inverse_dispersion_median": theta[1],
        "inverse_dispersion_max": theta[2],
        "learning_rate": learning_rate,
        "epoch_duration_seconds": duration,
        "training_views_per_second": TRAINING_VIEWS_PER_EPOCH / duration,
        "peak_vram_gib_all_ranks": peak_vram,
        "best_validation_metric": early.best_value,
        "best_epoch": early.best_epoch,
        "bad_validations": early.bad_validations,
        "improved": early.improved,
        "should_stop": early.should_stop,
        "stop_reason": early.stop_reason or "",
        "global_gradient_json": global_gradient,
        "block_gradients_json": list(block_gradients),
        "dispersion_gradient_json": theta_gradient,
        "training_per_core_json": dict(train_per_core),
        "validation_per_core_json": dict(validation.per_core),
        "training_mask_receipts_json": list(training_mask_receipts),
        "validation_mask_checksums_json": dict(validation_mask_receipts),
    }


def _validation_prediction_rows(
    *,
    run_id: str,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    validation: ValidationAggregate,
) -> list[Mapping[str, Any]]:
    """Make 20 storage-small prediction summaries, never full matrices."""

    dataset_id = str(_section(config, "dataset")["dataset_id"])
    rows: list[Mapping[str, Any]] = []
    for key, metrics in validation.per_view.items():
        alias = str(metrics["core_alias"])
        view_index = int(metrics["view_index"])
        batch = bundle.batches_by_alias[alias]
        masked_entries = int(metrics["masked_entries"])
        sample_key = hashlib.sha256(
            f"bagm.so2.nb.validation-summary.v1:{alias}:{view_index}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "run_id": run_id,
                "sample_key": sample_key,
                "dataset_id": dataset_id,
                "split": "validation",
                "y_true": float(metrics["observed_count_mean"]),
                "y_pred": float(metrics["predicted_count_mean"]),
                "sample_loss": float(metrics["masked_negative_binomial_nll"]),
                "effective_mask_rate": masked_entries
                / float(batch.n_nodes * batch.n_genes),
                "node_count": batch.n_nodes,
                "edge_count": batch.n_edges,
                "mask_view_index": view_index,
                "masked_entry_count": masked_entries,
                "aggregation_unit": "core_mask_view_masked_entry_mean",
            }
        )
    if len(rows) != len(VALIDATION_ALIASES) * VALIDATION_VIEW_COUNT:
        raise SO2NBTrainingError("Validation prediction summaries are incomplete.")
    return rows


def _archive_history_rows(
    run_id: str, rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    integers = {"global_epoch", "best_epoch", "bad_validations"}
    booleans = {"improved", "should_stop"}
    strings = {"schema", "stop_reason"}
    result: list[Mapping[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"run_id": run_id, "split": "train_validation"}
        for field, raw in row.items():
            if field.endswith("_json"):
                continue
            if field in integers:
                record[field] = int(raw)
            elif field in booleans:
                record[field] = (
                    raw
                    if isinstance(raw, bool)
                    else str(raw).strip().lower() == "true"
                )
            elif field in strings:
                record[field] = str(raw)
            else:
                record[field] = float(raw)
        result.append(record)
    return result


def _reconcile_metric_events(
    path: Path,
    *,
    checkpoint_epoch: int,
    truncate_uncommitted: bool = True,
) -> None:
    """Atomically truncate live metric events to the latest commit point."""

    epoch = int(checkpoint_epoch)
    if epoch < 0:
        raise SO2NBTrainingError("Metric-event checkpoint epoch cannot be negative.")
    rows: list[Mapping[str, Any]] = []
    if path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SO2NBTrainingError("Metric event log is unreadable.") from exc
        if any(not isinstance(row, Mapping) for row in rows):
            raise SO2NBTrainingError("Metric event log contains a non-mapping row.")
    retained = [row for row in rows if int(row.get("step", -1)) <= epoch]
    expected_names = {
        "train/masked_negative_binomial_nll",
        "val/unseen_donor/masked_negative_binomial_nll",
    }
    for step in range(1, epoch + 1):
        names = [str(row.get("name")) for row in retained if int(row["step"]) == step]
        if len(names) != 2 or set(names) != expected_names:
            raise SO2NBTrainingError(
                f"Metric events do not completely cover committed epoch {step}."
            )
    if any(int(row.get("step", -1)) not in range(1, epoch + 1) for row in retained):
        raise SO2NBTrainingError("Metric event log has an invalid committed step.")
    if len(retained) != len(rows) and not truncate_uncommitted:
        raise SO2NBTrainingError(
            "Metric event log contains rows beyond the committed final epoch."
        )
    if len(retained) != len(rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".events.jsonl.", suffix=".writing", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for row in retained:
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def _remove_partial_finalization_outputs(scratch: Path) -> None:
    """Remove only reproducible outputs written after terminal best replay."""

    candidates = [
        scratch / "summary.json",
        scratch / "metrics/final.json",
        scratch / "diagnostics/final_checkpoint_reload_verification.json",
        scratch / "diagnostics/hardware_preflight.json",
        scratch / "provenance/so2_geometry_modulated_nb_training.json",
    ]
    for stem in (scratch / "metrics/history", scratch / "predictions/validation"):
        candidates.extend(stem.with_suffix(suffix) for suffix in (".parquet", ".jsonl", ".csv"))
        candidates.append(stem.with_name(f".{stem.name}.parquet.writing"))
    for path in candidates:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise SO2NBTrainingError(f"Unsafe partial finalization path: {path}.")
        path.unlink(missing_ok=True)


def _remove_interrupted_atomic_temps(scratch: Path) -> None:
    """Discard only runner-owned temporary files left by an interrupted replace."""

    patterns = {
        scratch / "checkpoints": (".best.ckpt.*.writing", ".latest.ckpt.*.writing"),
        scratch / "results": (
            ".epoch_metrics.csv.*.writing",
            ".per_core_epoch_metrics.csv.*.writing",
            ".gradient_direction_metrics.csv.*.writing",
            ".gradient_direction_by_block.csv.*.writing",
            ".gradient_direction_dispersion.csv.*.writing",
        ),
        scratch / "metrics": (".events.jsonl.*.writing", ".history.parquet.writing"),
        scratch / "predictions": (".validation.parquet.writing",),
    }
    for directory, globs in patterns.items():
        if not directory.is_dir():
            continue
        for pattern in globs:
            for path in directory.glob(pattern):
                if path.is_symlink() or not path.is_file():
                    raise SO2NBTrainingError(
                        f"Unsafe interrupted atomic-write path: {path}."
                    )
                path.unlink()


def _read_runner_table(scratch: Path, stem: str) -> list[Mapping[str, Any]]:
    """Read the single table representation emitted by ``RunArchive``."""

    base = scratch / stem
    candidates = [
        base.with_suffix(suffix)
        for suffix in (".parquet", ".jsonl")
        if base.with_suffix(suffix).is_file()
    ]
    if len(candidates) != 1:
        raise SO2NBTrainingError(
            f"Runner output {stem!r} must have exactly one parquet/jsonl representation."
        )
    path = candidates[0]
    if path.is_symlink() or path.stat().st_size == 0:
        raise SO2NBTrainingError(f"Runner output table is empty or unsafe: {path}.")
    try:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            import pyarrow.parquet as parquet

            rows = parquet.read_table(path).to_pylist()
    except (ImportError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SO2NBTrainingError(f"Runner output table is unreadable: {path}.") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise SO2NBTrainingError(f"Runner output table is empty or malformed: {path}.")
    return [dict(row) for row in rows]


def _load_json_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise SO2NBTrainingError(f"Missing or unsafe {description}: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBTrainingError(f"Unreadable {description}: {path}.") from exc
    if not isinstance(value, Mapping):
        raise SO2NBTrainingError(f"{description} must contain a mapping.")
    return dict(value)


def _validate_runner_outputs(
    scratch: Path,
    *,
    run_id: str,
    checkpoint_store: AtomicBestLatestCheckpointStore,
    expect_latest: bool,
    expected_summary: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate child-owned outputs without requiring the queue-owned manifest.

    The queue writes ``manifest.yaml`` only after this process exits.  Calling
    the archive-wide success validator here would therefore make every normal
    training completion fail.  This narrower gate covers every conclusion-
    bearing output owned by this runner and deliberately leaves the manifest
    and queue-owned provenance to the queue's final success validation.
    """

    required_regular_files = (
        "results/epoch_metrics.csv",
        "results/per_core_epoch_metrics.csv",
        "results/gradient_direction_metrics.csv",
        "results/gradient_direction_by_block.csv",
        "results/gradient_direction_dispersion.csv",
        "metrics/events.jsonl",
        "metrics/final.json",
        "diagnostics/final_checkpoint_reload_verification.json",
        "diagnostics/hardware_preflight.json",
        "provenance/so2_geometry_modulated_nb_training.json",
        "summary.json",
    )
    for relative in required_regular_files:
        path = scratch / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SO2NBTrainingError(
                f"Missing, empty, or unsafe runner-owned output: {relative}."
            )

    summary = _load_json_mapping(
        scratch / "summary.json", description="runner summary"
    )
    _require_equal(summary.get("run_id"), run_id, field="summary.run_id")
    _require_equal(summary.get("status"), "success", field="summary.status")
    if expected_summary is not None:
        _require_equal(
            _canonical_sha256(summary),
            _canonical_sha256(dict(expected_summary)),
            field="summary.content_sha256",
        )
    final_epoch = int(summary.get("final_epoch", 0))
    best_epoch = int(summary.get("best_epoch", 0))
    if final_epoch < 1 or best_epoch not in range(1, final_epoch + 1):
        raise SO2NBTrainingError("Summary epoch fields are invalid.")
    _require_equal(
        int(summary.get("epoch_metrics_rows", -1)),
        final_epoch,
        field="summary.epoch_metrics_rows",
    )

    best = checkpoint_store.load("best")
    _require_equal(
        int(best["completed_epoch"]), best_epoch, field="best.completed_epoch"
    )
    best_sha256 = sha256_file(checkpoint_store.best_path)
    if expect_latest:
        latest = checkpoint_store.load("latest")
        _require_equal(
            int(latest["completed_epoch"]),
            final_epoch,
            field="latest.completed_epoch",
        )
        _require_equal(
            latest.get("best_checkpoint_sha256"),
            best_sha256,
            field="latest.best_checkpoint_sha256",
        )
        expected_checkpoint_files = {"best.ckpt", "latest.ckpt"}
    else:
        expected_checkpoint_files = {"best.ckpt"}
    actual_checkpoint_files = {
        path.name for path in checkpoint_store.directory.iterdir() if path.is_file()
    }
    _require_equal(
        actual_checkpoint_files,
        expected_checkpoint_files,
        field="checkpoint.layout",
    )

    epoch_rows = DurableNBEpochMetricsCSV(scratch).rows()
    _require_equal(len(epoch_rows), final_epoch, field="epoch_metrics.row_count")
    _require_equal(
        int(epoch_rows[-1]["best_epoch"]), best_epoch, field="epoch_metrics.best_epoch"
    )
    per_core_rows = DurableNBPerCoreMetricsCSV(scratch).rows()
    _require_equal(
        len(per_core_rows),
        final_epoch * (len(TRAINING_ALIASES) + len(VALIDATION_ALIASES)),
        field="per_core_epoch_metrics.row_count",
    )
    for epoch in range(1, final_epoch + 1):
        rows_at_epoch = [
            row for row in per_core_rows if int(row.get("global_epoch", -1)) == epoch
        ]
        train_aliases = {
            str(row.get("core_alias"))
            for row in rows_at_epoch
            if row.get("split") == "training"
        }
        validation_aliases = {
            str(row.get("core_alias"))
            for row in rows_at_epoch
            if row.get("split") == "validation"
        }
        _require_equal(
            train_aliases, set(TRAINING_ALIASES), field="per_core.train_aliases"
        )
        _require_equal(
            validation_aliases,
            set(VALIDATION_ALIASES),
            field="per_core.validation_aliases",
        )

    for relative, columns, rows_per_epoch in (
        (
            "results/gradient_direction_metrics.csv",
            GRADIENT_DIRECTION_METRICS_COLUMNS,
            1,
        ),
        (
            "results/gradient_direction_by_block.csv",
            BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
            4,
        ),
        (
            "results/gradient_direction_dispersion.csv",
            GRADIENT_DIRECTION_METRICS_COLUMNS,
            1,
        ),
    ):
        gradient_rows = DurableScalarCSV(
            scratch, relative, columns, rows_per_epoch=rows_per_epoch
        ).rows()
        _require_equal(
            len(gradient_rows),
            final_epoch * rows_per_epoch,
            field=f"{relative}.row_count",
        )
        _require_equal(
            [int(row["global_epoch"]) for row in gradient_rows],
            [epoch for epoch in range(1, final_epoch + 1) for _ in range(rows_per_epoch)],
            field=f"{relative}.global_epochs",
        )

    history = _read_runner_table(scratch, "metrics/history")
    _require_equal(len(history), final_epoch, field="metrics.history.row_count")
    _require_equal(
        [int(row.get("global_epoch", -1)) for row in history],
        list(range(1, final_epoch + 1)),
        field="metrics.history.global_epochs",
    )
    if any(row.get("run_id") != run_id for row in history):
        raise SO2NBTrainingError("Metrics history is not bound to the run_id.")

    predictions = _read_runner_table(scratch, "predictions/validation")
    try:
        validated_predictions = validate_prediction_rows(
            predictions, expected_run_id=run_id, expected_split="validation"
        )
    except Exception as exc:
        raise SO2NBTrainingError("Validation prediction summaries are malformed.") from exc
    _require_equal(
        len(validated_predictions),
        len(VALIDATION_ALIASES) * VALIDATION_VIEW_COUNT,
        field="predictions.validation.row_count",
    )
    if len({str(row["sample_key"]) for row in validated_predictions}) != len(
        validated_predictions
    ):
        raise SO2NBTrainingError("Validation prediction sample keys are not unique.")

    final_metrics = _load_json_mapping(
        scratch / "metrics/final.json", description="final metrics"
    )
    primary_name = str(summary.get("primary_metric_name", ""))
    if primary_name not in final_metrics:
        raise SO2NBTrainingError("Final metrics omit the declared primary metric.")
    primary_value = float(summary.get("primary_metric_value", float("nan")))
    if not math.isfinite(primary_value) or not math.isclose(
        primary_value,
        float(final_metrics[primary_name]),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise SO2NBTrainingError("Summary and final primary metrics disagree.")

    _reconcile_metric_events(
        scratch / "metrics/events.jsonl",
        checkpoint_epoch=final_epoch,
        truncate_uncommitted=False,
    )
    reload_receipt = _load_json_mapping(
        scratch / "diagnostics/final_checkpoint_reload_verification.json",
        description="final checkpoint reload receipt",
    )
    _require_equal(reload_receipt.get("verified"), True, field="reload.verified")
    _require_equal(
        reload_receipt.get("checkpoint_sha256"),
        best_sha256,
        field="reload.checkpoint_sha256",
    )

    provenance = _load_json_mapping(
        scratch / "provenance/so2_geometry_modulated_nb_training.json",
        description="SO2 NB training provenance",
    )
    _require_equal(
        int(provenance.get("validation_prediction_summary_rows", -1)),
        len(validated_predictions),
        field="provenance.validation_prediction_summary_rows",
    )
    plots = provenance.get("plots")
    if not isinstance(plots, Mapping) or not plots:
        raise SO2NBTrainingError("Training provenance does not declare plots.")
    for relative in plots.values():
        candidate = Path(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SO2NBTrainingError("Training provenance has an unsafe plot path.")
        path = scratch / candidate
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SO2NBTrainingError(f"Declared training plot is missing: {candidate}.")

    forbidden_test_outputs = [
        path
        for directory in (scratch / "predictions", scratch / "metrics")
        if directory.is_dir()
        for path in directory.glob("test*")
    ]
    if forbidden_test_outputs:
        raise SO2NBTrainingError("Test artifacts are forbidden for this protocol.")
    return summary


def _remaining_training_epochs(state: EarlyStoppingState) -> range:
    """Return no work for a terminal latest checkpoint awaiting finalization."""

    start = int(state.completed_epoch)
    return range(start, start if state.should_stop else MAXIMUM_EPOCHS)


def _initialize_training(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, Any, Any]:
    trainer = _section(config, "trainer")
    model.to(device)
    training_model = DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        forward_sync_buffers=False,
    )
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(trainer["scheduler_factor"]),
        patience=int(trainer["scheduler_patience"]),
        threshold=float(trainer["scheduler_threshold"]),
        threshold_mode=str(trainer["scheduler_threshold_mode"]),
        min_lr=float(trainer["scheduler_min_learning_rate"]),
    )
    scaler = _make_grad_scaler(device, bool(trainer["amp"]))
    return training_model, optimizer, scheduler, scaler


def run_distributed(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    control_group: Any,
) -> Mapping[str, Any] | None:
    started = time.monotonic()
    run_id, scratch, config, paths = _load_worker_config(args)
    assert scratch is not None
    bundle = _load_data(config, paths)
    if tuple(batch.alias for batch in bundle.training_batches) != SO2_NB_TRAINING_ALIASES:
        raise SO2NBTrainingError("Training data aliases drifted.")
    if tuple(batch.alias for batch in bundle.validation_batches) != SO2_NB_VALIDATION_ALIASES:
        raise SO2NBTrainingError("Validation data aliases drifted.")
    _require_equal(SO2_NB_TEST_ALIASES, (), field="data.test_aliases")

    model = _build_model(config, bundle)
    device = torch.device(f"cuda:{local_rank}")
    training_model, optimizer, scheduler, scaler = _initialize_training(
        model, config, device
    )
    preflight = _load_preflight_receipt(config, paths, bundle, model)
    trainer = _section(config, "trainer")
    config_sha = _canonical_sha256(config)

    global_tracker = FullGradientDirectionTracker(
        expected_optimizer_updates_per_epoch=6,
        resume_boundary_unavailable=args.resume_checkpoint is not None,
    )
    block_tracker = BlockGradientDirectionTracker(
        expected_blocks=4,
        expected_optimizer_updates_per_epoch=6,
        resume_boundary_unavailable=args.resume_checkpoint is not None,
    )
    theta_tracker = FullGradientDirectionTracker(
        expected_optimizer_updates_per_epoch=6,
        resume_boundary_unavailable=args.resume_checkpoint is not None,
    )
    theta_view = _RawThetaParameterView(model.raw_theta)

    archive: RunArchive | None = None
    checkpoint_store: AtomicBestLatestCheckpointStore | None = None
    epoch_writer: DurableNBEpochMetricsCSV | None = None
    core_writer: DurableNBPerCoreMetricsCSV | None = None
    global_writer: DurableScalarCSV | None = None
    block_writer: DurableScalarCSV | None = None
    theta_writer: DurableScalarCSV | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="training archive initialization",
    ):
        if rank == 0:
            archive = RunArchive.attach_active(
                run_id, paths=paths, scratch_path=scratch
            )
            _remove_interrupted_atomic_temps(scratch)
            checkpoint_store = AtomicBestLatestCheckpointStore(scratch)
            epoch_writer = DurableNBEpochMetricsCSV(scratch)
            core_writer = DurableNBPerCoreMetricsCSV(scratch)
            global_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_metrics.csv",
                GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=1,
            )
            block_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_by_block.csv",
                BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=4,
            )
            theta_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_dispersion.csv",
                GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=1,
            )

    early_state = EarlyStoppingState()
    fixed_receipts: Mapping[str, Mapping[str, Any]] | None = None
    start_epoch = 0
    finalized_summary: Mapping[str, Any] | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="training finalized-run reconciliation",
    ):
        if rank == 0:
            assert archive and checkpoint_store
            if (
                checkpoint_store.best_path.is_file()
                and not checkpoint_store.latest_path.is_file()
            ):
                if (scratch / "summary.json").is_file():
                    loaded_summary = _validate_runner_outputs(
                        scratch,
                        run_id=run_id,
                        checkpoint_store=checkpoint_store,
                        expect_latest=False,
                    )
                    finalized_summary = dict(loaded_summary)
                else:
                    checkpoint_store.discard_best_without_latest()
            elif (
                (scratch / "summary.json").is_file()
                and not checkpoint_store.latest_path.is_file()
            ):
                raise SO2NBTrainingError(
                    "Finalized summary exists without a retained best checkpoint."
                )
    finalized_values: list[Any] = [finalized_summary if rank == 0 else None]
    torch.distributed.broadcast_object_list(
        finalized_values, src=0, group=control_group
    )
    if finalized_values[0] is not None:
        return dict(finalized_values[0]) if rank == 0 else None

    resume_path = args.resume_checkpoint
    if resume_path is None and (scratch / "checkpoints/latest.ckpt").is_file():
        resume_path = scratch / "checkpoints/latest.ckpt"
    if resume_path is not None:
        resume_path = resume_path.resolve(strict=True)
        if resume_path.parent != (scratch / "checkpoints").resolve(False):
            if rank == 0:
                raise SO2NBTrainingError(
                    "Resume latest and best must remain in the active run checkpoint directory."
                )
            raise SO2NBTrainingError("Non-canonical resume checkpoint.")
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="resume checkpoint transaction reconciliation",
        ):
            if rank == 0:
                assert checkpoint_store is not None
                _reconcile_checkpoint_transaction(checkpoint_store)
        early_state, fixed_receipts = _load_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            rank=rank,
            configuration_sha256=config_sha,
        )
        start_epoch = early_state.completed_epoch
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="resume metric transaction reconciliation",
        ):
            if rank == 0:
                assert (
                    epoch_writer
                    and core_writer
                    and global_writer
                    and block_writer
                    and theta_writer
                )
                epoch_writer.reconcile(checkpoint_epoch=start_epoch)
                core_writer.reconcile(checkpoint_epoch=start_epoch)
                global_writer.reconcile(checkpoint_epoch=start_epoch)
                block_writer.reconcile(checkpoint_epoch=start_epoch)
                theta_writer.reconcile(checkpoint_epoch=start_epoch)
                _reconcile_metric_events(
                    scratch / "metrics/events.jsonl", checkpoint_epoch=start_epoch
                )
                _remove_partial_finalization_outputs(scratch)
        global_tracker.reset_for_resume_boundary()
        block_tracker.reset_for_resume_boundary()
        theta_tracker.reset_for_resume_boundary()
    else:
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="fresh-run metric transaction reconciliation",
        ):
            if rank == 0:
                assert (
                    epoch_writer
                    and core_writer
                    and global_writer
                    and block_writer
                    and theta_writer
                )
                epoch_writer.reconcile(checkpoint_epoch=0)
                core_writer.reconcile(checkpoint_epoch=0)
                global_writer.reconcile(checkpoint_epoch=0)
                block_writer.reconcile(checkpoint_epoch=0)
                theta_writer.reconcile(checkpoint_epoch=0)
                _reconcile_metric_events(
                    scratch / "metrics/events.jsonl", checkpoint_epoch=0
                )

    for global_epoch in _remaining_training_epochs(early_state):
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        ordered = cohort_relative_qkv_core_order(
            global_epoch,
            aliases=TRAINING_ALIASES,
            core_order_seed=CORE_ORDER_SEED,
        )
        pairs = tuple(
            (ordered[index], ordered[index + 1])
            for index in range(0, len(ordered), 2)
        )
        if len(pairs) != 6:
            raise SO2NBTrainingError("Each epoch must contain six paired updates.")
        train_records: list[Mapping[str, Any]] = []
        gradient_norms: list[float] = []
        for update_index, pair in enumerate(pairs):
            records, gradient_norm = _paired_update(
                model=model,
                training_model=training_model,
                optimizer=optimizer,
                scaler=scaler,
                pair=pair,
                update_index=update_index,
                global_epoch=global_epoch,
                rank=rank,
                device=device,
                batches=bundle.batches_by_alias,
                trainer=trainer,
                global_tracker=global_tracker,
                block_tracker=block_tracker,
                theta_tracker=theta_tracker,
                theta_view=theta_view,
            )
            train_records.extend(records)
            gradient_norms.append(gradient_norm)
        train_nll, train_per_core = aggregate_training_nll(train_records)
        global_gradient_summary = global_tracker.complete_epoch(global_epoch)
        block_gradient_summaries = block_tracker.complete_epoch(global_epoch)
        theta_gradient_summary = theta_tracker.complete_epoch(global_epoch)
        if rank == 0:
            global_gradient = _gradient_rows(
                global_gradient_summary, run_id=run_id
            )
            block_gradients = [
                _gradient_rows(summary, run_id=run_id)
                for summary in block_gradient_summaries
            ]
            theta_gradient = _gradient_rows(
                theta_gradient_summary, run_id=run_id
            )
        else:
            global_gradient = {}
            block_gradients = []
            theta_gradient = {}

        validation, receipts = _evaluate_validation(
            bundle=bundle,
            training_model=training_model,
            rank=rank,
            device=device,
            trainer=trainer,
        )
        if fixed_receipts is None:
            fixed_receipts = receipts
        elif _canonical_sha256(fixed_receipts) != _canonical_sha256(receipts):
            raise SO2NBTrainingError("Fixed validation masks changed between epochs.")
        scheduler.step(validation.primary_equal_core_nll)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        def update_early_state() -> Mapping[str, Any]:
            return asdict(
                update_early_stopping(
                    early_state,
                    validation.primary_equal_core_nll,
                    completed_epoch=global_epoch + 1,
                    minimum_epochs=MINIMUM_EPOCHS,
                    patience=EARLY_STOPPING_PATIENCE,
                    min_delta=EARLY_STOPPING_MIN_DELTA,
                    maximum_epochs=MAXIMUM_EPOCHS,
                )
            )

        early_value = _broadcast_rank_zero_result(
            rank=rank,
            control_group=control_group,
            phase=f"epoch-{global_epoch + 1} early-stopping update",
            operation=update_early_state,
        )
        if not isinstance(early_value, Mapping):
            raise SO2NBTrainingError(
                "Rank-zero early-stopping result is not a mapping."
            )
        early_state = EarlyStoppingState(**dict(early_value))

        peak = _peak_vram_all_ranks(device)
        duration = time.monotonic() - epoch_started
        if not math.isfinite(duration) or duration <= 0.0:
            raise SO2NBTrainingError("Epoch duration is invalid.")
        theta = _theta_summary(model)
        rng_states = _all_rank_rng_states()

        # CUDA-to-CPU checkpoint copies must finish on every rank before peers
        # wait for rank-zero persistence.  The auxiliary Gloo rendezvous keeps
        # that wait off the GPUs; an NCCL wait here can starve rank zero while
        # it synchronizes CUDA tensors into a checkpoint snapshot.
        checkpoint_state = _checkpoint_state_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=global_epoch + 1,
            early_state=early_state,
            bundle=bundle,
            rng_states=rng_states,
            validation_mask_receipts=receipts,
            configuration_sha256=config_sha,
        )
        torch.distributed.barrier(group=control_group)

        def persist_epoch() -> Mapping[str, Any]:
            assert archive and checkpoint_store and epoch_writer and core_writer
            assert global_writer and block_writer and theta_writer
            row = _make_epoch_row(
                completed_epoch=global_epoch + 1,
                train_nll=train_nll,
                train_per_core=train_per_core,
                validation=validation,
                theta=theta,
                learning_rate=learning_rate,
                duration=duration,
                peak_vram=peak,
                early=early_state,
                global_gradient=global_gradient,
                block_gradients=block_gradients,
                theta_gradient=theta_gradient,
                training_mask_receipts=train_records,
                validation_mask_receipts=receipts,
            )
            epoch_writer.append(row)
            core_writer.append_epoch(
                global_epoch=global_epoch + 1,
                training_per_core=train_per_core,
                validation_per_core=validation.per_core,
            )
            global_writer.append_epoch(
                [global_gradient], global_epoch=global_epoch + 1
            )
            block_writer.append_epoch(
                block_gradients, global_epoch=global_epoch + 1
            )
            theta_writer.append_epoch(
                [theta_gradient], global_epoch=global_epoch + 1
            )
            archive.append_metric_event(
                {
                    "name": "train/masked_negative_binomial_nll",
                    "value": train_nll,
                    "step": global_epoch + 1,
                }
            )
            archive.append_metric_event(
                {
                    "name": "val/unseen_donor/masked_negative_binomial_nll",
                    "value": validation.primary_equal_core_nll,
                    "step": global_epoch + 1,
                }
            )

            if early_state.improved:
                best_payload = _checkpoint_payload_from_state(
                    checkpoint_state,
                    role="best",
                    best_checkpoint_sha256=None,
                )
                best_receipt = checkpoint_store.save_best(best_payload)
                best_receipt_sha256 = best_receipt.sha256
            else:
                if not checkpoint_store.best_path.is_file():
                    raise SO2NBTrainingError(
                        "Validation-best checkpoint is missing."
                    )
                best_payload = checkpoint_store.load("best")
                best_receipt_sha256 = sha256_file(checkpoint_store.best_path)
                _require_equal(
                    int(best_payload["completed_epoch"]),
                    early_state.best_epoch,
                    field="best.completed_epoch",
                )
                _require_equal(
                    best_payload.get("best_validation_metric"),
                    early_state.best_value,
                    field="best.validation_metric",
                )
            latest_receipt = checkpoint_store.save_latest(
                _checkpoint_payload_from_state(
                    checkpoint_state,
                    role="latest",
                    best_checkpoint_sha256=best_receipt_sha256,
                    embedded_best_checkpoint=best_payload,
                )
            )
            print(
                json.dumps(
                    {
                        "epoch": global_epoch + 1,
                        "train_nb_nll": train_nll,
                        "validation_nb_nll": validation.primary_equal_core_nll,
                        "best_epoch": early_state.best_epoch,
                        "bad_validations": early_state.bad_validations,
                        "learning_rate": learning_rate,
                        "peak_vram_gib": peak,
                        "should_stop": early_state.should_stop,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            return {
                "completed_epoch": global_epoch + 1,
                "latest_checkpoint_sha256": latest_receipt.sha256,
            }

        persistence_result = _broadcast_rank_zero_result(
            rank=rank,
            control_group=control_group,
            phase=f"epoch-{global_epoch + 1} persistence",
            operation=persist_epoch,
        )
        if not isinstance(persistence_result, Mapping):
            raise SO2NBTrainingError(
                "Rank-zero epoch-persistence result is not a mapping."
            )
        _require_equal(
            int(persistence_result.get("completed_epoch", -1)),
            global_epoch + 1,
            field="persistence.completed_epoch",
        )
        latest_checkpoint_sha256 = persistence_result.get(
            "latest_checkpoint_sha256"
        )
        if (
            not isinstance(latest_checkpoint_sha256, str)
            or len(latest_checkpoint_sha256) != 64
        ):
            raise SO2NBTrainingError(
                "Rank-zero epoch-persistence checksum is invalid."
            )
        del checkpoint_state
        if early_state.should_stop:
            break

    if early_state.best_epoch <= 0 or fixed_receipts is None:
        raise SO2NBTrainingError("Training ended without a validation best.")
    best_path = scratch / "checkpoints/best.ckpt"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    reloaded_validation, reloaded_receipts = _evaluate_validation(
        bundle=bundle,
        training_model=training_model,
        rank=rank,
        device=device,
        trainer=trainer,
    )
    if (
        not math.isclose(
            reloaded_validation.primary_equal_core_nll,
            float(early_state.best_value),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or _canonical_sha256(reloaded_receipts) != _canonical_sha256(fixed_receipts)
    ):
        raise SO2NBTrainingError("Reloaded best checkpoint failed validation replay.")
    torch.distributed.barrier()
    if rank != 0:
        return None
    assert archive and checkpoint_store and epoch_writer
    final_checkpoint_sha256 = sha256_file(checkpoint_store.best_path)
    rows = epoch_writer.rows()
    plots = write_nb_training_plots(scratch, rows)
    archive.write_table(
        "metrics/history", _archive_history_rows(run_id, rows), fallback="jsonl"
    )
    prediction_rows = _validation_prediction_rows(
        run_id=run_id,
        config=config,
        bundle=bundle,
        validation=reloaded_validation,
    )
    archive.write_predictions("validation", prediction_rows, fallback="jsonl")
    final_metrics = {
        "val/unseen_donor/masked_negative_binomial_nll": (
            reloaded_validation.primary_equal_core_nll
        ),
        "val/unseen_donor/pooled_masked_negative_binomial_nll": (
            reloaded_validation.pooled_nll
        ),
        "val/unseen_donor/raw_count_mae": reloaded_validation.raw_count_mae,
        "val/unseen_donor/raw_count_rmse": reloaded_validation.raw_count_rmse,
        "val/unseen_donor/log1p_mae": reloaded_validation.log1p_mae,
        "val/unseen_donor/log1p_rmse": reloaded_validation.log1p_rmse,
        "val/unseen_donor/poisson_deviance": reloaded_validation.poisson_deviance,
        "val/unseen_donor/observed_zero_rate": reloaded_validation.observed_zero_rate,
        "val/unseen_donor/predicted_zero_probability_mean": (
            reloaded_validation.predicted_zero_probability_mean
        ),
        "val/unseen_donor/zero_brier_score": reloaded_validation.zero_brier_score,
    }
    archive.write_json("metrics/final.json", final_metrics)
    archive.write_json(
        "diagnostics/final_checkpoint_reload_verification.json",
        {
            "verified": True,
            "checkpoint": "checkpoints/best.ckpt",
            "checkpoint_sha256": final_checkpoint_sha256,
            "best_epoch": early_state.best_epoch,
            "best_validation_metric": early_state.best_value,
            "replayed_validation_metric": reloaded_validation.primary_equal_core_nll,
            "validation_mask_receipts_sha256": _canonical_sha256(fixed_receipts),
        },
    )
    archive.write_json("diagnostics/hardware_preflight.json", dict(preflight))
    archive.write_json(
        "provenance/so2_geometry_modulated_nb_training.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "protocol": PROTOCOL,
            "training_aliases": list(TRAINING_ALIASES),
            "validation_aliases": list(VALIDATION_ALIASES),
            "test_aliases": [],
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
            "split_fingerprint": bundle.split_fingerprint,
            "overlay_manifest_sha256": bundle.manifest_sha256,
            "validation_mask_receipts_sha256": _canonical_sha256(fixed_receipts),
            "checkpoint_sha256": final_checkpoint_sha256,
            "checkpoint_layout": "final_best_only",
            "gradient_vectors_persisted": False,
            "full_prediction_matrices_persisted": False,
            "validation_prediction_summary_rows": len(prediction_rows),
            "test_artifacts_present": False,
            "plots": plots,
        },
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "model_name": "geometry-modulated-relative-qkv-gat-nb2",
        "model_seed": MODEL_SEED,
        "final_epoch": early_state.completed_epoch,
        "best_epoch": early_state.best_epoch,
        "optimizer_steps": early_state.completed_epoch * 6,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "primary_metric_name": "val/unseen_donor/masked_negative_binomial_nll",
        "primary_metric_value": reloaded_validation.primary_equal_core_nll,
        "checkpoint": "checkpoints/best.ckpt",
        "checkpoint_reload_verified": True,
        "checkpoint_count": 1,
        "epoch_metrics_rows": len(rows),
        "loss_recorded_every_epoch": True,
        "gradient_direction_recorded_every_epoch": True,
        "stopped_early": early_state.stop_reason == "patience",
        "stop_reason": early_state.stop_reason,
        "peak_vram_gib": max(float(row["peak_vram_gib_all_ranks"]) for row in rows),
        "duration_seconds": time.monotonic() - started,
        "world_size": WORLD_SIZE,
        "generalization_estimate": False,
        "unbiased_test_estimate": False,
        "test_artifacts_present": False,
    }
    archive.write_summary(summary)
    _require_equal(
        sha256_file(checkpoint_store.best_path),
        final_checkpoint_sha256,
        field="final.checkpoint_sha256",
    )
    _validate_runner_outputs(
        scratch,
        run_id=run_id,
        checkpoint_store=checkpoint_store,
        expect_latest=True,
        expected_summary=summary,
    )
    # This is intentionally the final mutation in the child.  The queue writes
    # manifest.yaml and performs the archive-wide success validation only after
    # this subprocess returns.
    checkpoint_store.finalize_best_only()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SO2 C15--C26 and validate C27--C28 with NB2 DDP."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, local_rank, _ = _distributed_identity()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(minutes=30)
    )
    control_group = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)),
        backend="gloo",
        timeout=timedelta(minutes=5),
    )
    try:
        summary = run_distributed(
            args,
            rank=rank,
            local_rank=local_rank,
            control_group=control_group,
        )
        if rank == 0 and summary is not None:
            print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Four-rank launch gate for target-isolated SO2 NB2 training.

Run this script with ``torchrun --nproc-per-node=4``.  It composes the frozen
campaign configuration, verifies the reference-only SO2 data overlay, executes
one real full-core training target shard and one fixed-mask validation shard on
every rank, and publishes one checksum-bound JSON receipt.  It does not launch
the experiment and retains no checkpoint or prediction artifact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from typing import Any, ContextManager, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    compose_config,
    load_yaml_mapping,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.negative_binomial import (  # noqa: E402
    NegativeBinomialModelOutput,
    masked_negative_binomial_nll,
    masked_negative_binomial_nll_sum,
    nb2_inverse_dispersion_from_raw,
    nb2_mean_from_logits,
    negative_binomial_nll_elementwise,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.so2_nb_data import (  # noqa: E402
    SO2NBDataBundle,
    SO2NBCoreBatch,
    SO2_NB_TRAINING_ALIASES,
    SO2_NB_VALIDATION_ALIASES,
    load_so2_nb_data,
)
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    AtomicBestLatestCheckpointStore,
    EarlyStoppingState,
    sha256_file,
)
from spatial_benchmark.target_cell_masking import (  # noqa: E402
    make_contiguous_target_partition,
    make_contiguous_target_partitions,
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
    target_partition_coverage_receipt,
)
from spatial_benchmark.target_isolated_negative_binomial import (  # noqa: E402
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)


CAMPAIGN_ID = (
    "cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0"
)
PROTOCOL = (
    "donor_grouped_so2_target_isolated_nb_once_per_cell_earlystop_v1"
)
PREFLIGHT_SCHEMA = "so2_target_isolated_nb_once_per_cell_ddp4_preflight_v1"
CHECKPOINT_SCHEMA = "so2_target_isolated_nb_once_per_cell_checkpoint_v1"
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
EXPECTED_PARAMETER_COUNT = 5_135_088
EXPECTED_BLOCK_COUNT = 4
EXPECTED_BLOCK_PARAMETER_COUNT = 831_880
EXPECTED_GENES = 1_000
EXPECTED_COVARIATES = 22
EXPECTED_TRAINING_CELLS = 208_696
EXPECTED_VALIDATION_CELLS = 37_367
REPRESENTATIVE_TRAINING_ALIAS = "SO2-C23"
REPRESENTATIVE_VALIDATION_ALIAS = "SO2-C27"
REPRESENTATIVE_TRAINING_NODES = 43_462
REPRESENTATIVE_TRAINING_EDGES = 9_972_300
REPRESENTATIVE_VALIDATION_NODES = 24_245
REPRESENTATIVE_VALIDATION_EDGES = 5_558_646
MAXIMUM_PEAK_VRAM_GIB = 22.0
MINIMUM_VRAM_HEADROOM_GIB = 2.0
MINIMUM_FREE_DISK_GIB = 20.0
AMP_MU_MAX_ABS_TOLERANCE = 7.5e-2
AMP_MU_MEAN_ABS_TOLERANCE = 7.5e-3
AMP_NLL_ABS_TOLERANCE = 2.5e-2
AMP_NLL_RELATIVE_TOLERANCE = 2.5e-3
FROZEN_TASK_CONTRACT_SHA256 = (
    "dfaf10f1e828dffc1369629ae41dce133a42d0686aeeedba30a9d97c5b296bbc"
)
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiment/"
    "so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0.yaml"
)
FROZEN_TASK_CONTRACT = Path(
    "experiments/campaigns/"
    "cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0/"
    "frozen_task_contract.yaml"
)
CODE_HASH_SCOPE = (
    "target_isolated_model_masking_data_training_preflight_runner_and_"
    "configuration_launch_contract"
)
CODE_RELATIVE_PATHS = (
    Path("scripts/diagnostics/preflight_so2_target_isolated_nb_ddp.py"),
    Path("scripts/train/run_so2_target_isolated_nb.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/target_isolated_negative_binomial.py"),
    Path("src/spatial_benchmark/target_cell_masking.py"),
    Path("src/spatial_benchmark/target_cell_training.py"),
    Path("src/spatial_benchmark/negative_binomial.py"),
    Path("src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/models.py"),
    Path("src/spatial_benchmark/so2_nb_data.py"),
    Path("src/spatial_benchmark/so2_nb_training.py"),
    Path("src/spatial_benchmark/gradient_direction_observability.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training_v2.py"),
    Path("src/spatial_benchmark/run_archive.py"),
    Path("src/spatial_benchmark/training.py"),
    Path("src/spatial_benchmark/queueing.py"),
    Path("src/spatial_benchmark/configuration.py"),
    Path("src/spatial_benchmark/paths.py"),
    Path("src/spatial_benchmark/identifiers.py"),
)
_CONTROL_GROUP: Any | None = None
REQUIRED_GATE_NAMES = frozenset(
    {
        "configuration_and_contract",
        "model_topology",
        "shared_encoder_two_stream",
        "synthetic_target_coverage",
        "target_mask_contract",
        "neighbor_context_unmasked",
        "incoming_nonself_routing",
        "masked_target_value_invariance",
        "raw_target_forward_isolation",
        "no_return_leakage",
        "unequal_count_ddp_weighting",
        "fixed_validation_mask_replay",
        "fixed_validation_metric_replay",
        "full_constant_fp32_nb2",
        "amp_fp32_equivalence",
        "real_train_shard_forward_backward",
        "preclip_gradient_flow",
        "bounded_nb2_loss_decrease",
        "checkpoint_roundtrip",
        "ddp_parameter_sync",
        "gpu_inventory",
        "peak_vram",
        "disk_space",
        "no_test_split_or_artifacts",
    }
)


class TargetIsolatedNBPreflightError(RuntimeError):
    """Raised when a launch-safety gate fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TargetIsolatedNBPreflightError(
            f"Resolved configuration requires a {name!r} mapping."
        )
    return value


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise TargetIsolatedNBPreflightError(
            f"Unsafe configured runtime path: {raw!r}."
        )
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


def _argument_path(raw: Path | None, *, default: Path) -> Path:
    candidate = default if raw is None else raw.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def _resolve_required_paths(
    args: argparse.Namespace,
) -> tuple[
    ProjectPaths,
    Path,
    dict[str, Any],
    Path,
    Path,
    Path,
    Path,
]:
    paths = current_paths(anchor=PROJECT_ROOT)
    paths.validate()
    if paths.project_root.resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise TargetIsolatedNBPreflightError(
            "BAGM_ROOT must identify the source tree containing this preflight."
        )
    if paths.config_root.resolve(strict=False) != (
        PROJECT_ROOT / "configs"
    ).resolve(strict=False):
        raise TargetIsolatedNBPreflightError(
            "BAGM_CONFIG_ROOT must identify this source tree's configs directory."
        )

    expected_config = (PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG).resolve(
        strict=True
    )
    config_path = _argument_path(args.config, default=expected_config)
    if config_path != expected_config:
        raise TargetIsolatedNBPreflightError(
            "--config must be the frozen target-isolated experiment source."
        )
    config = compose_config(
        config_path,
        config_root=paths.config_root,
        validate=True,
    )
    validate_target_isolated_config(config)
    dataset = _section(config, "dataset")
    launcher = _section(config, "launcher")
    expected_overlay = _runtime_path(
        dataset["prepared_artifact_reference"], paths
    )
    expected_cohort = _runtime_path(dataset["source_cohort_artifact"], paths)
    expected_graph = _runtime_path(dataset["source_graph_artifact"], paths)
    expected_output = _runtime_path(
        launcher["hardware_preflight_receipt"], paths
    )
    resolved = (
        _argument_path(args.overlay_dir, default=expected_overlay),
        _argument_path(args.cohort_dir, default=expected_cohort),
        _argument_path(args.graph_dir, default=expected_graph),
        _argument_path(args.output, default=expected_output),
    )
    expected = (
        expected_overlay,
        expected_cohort,
        expected_graph,
        expected_output,
    )
    if resolved != expected:
        raise TargetIsolatedNBPreflightError(
            "Data/output overrides must resolve to the paths frozen in config."
        )
    overlay, cohort, graph, output = resolved
    for value, label in (
        (overlay, "overlay"),
        (cohort, "cohort"),
        (graph, "graph"),
    ):
        if not value.is_dir():
            raise TargetIsolatedNBPreflightError(
                f"Configured {label} directory is absent: {value}."
            )
    contract = (PROJECT_ROOT / FROZEN_TASK_CONTRACT).resolve(strict=True)
    if sha256_file(contract) != FROZEN_TASK_CONTRACT_SHA256:
        raise TargetIsolatedNBPreflightError(
            "Frozen task contract file checksum drifted."
        )
    if output.is_symlink():
        raise TargetIsolatedNBPreflightError(
            "Preflight receipt path may not be a symbolic link."
        )
    return paths, config_path, config, overlay, cohort, graph, output


def _validate_bundle_binding(
    config: Mapping[str, Any], bundle: SO2NBDataBundle
) -> None:
    dataset = _section(config, "dataset")
    expected = {
        "dataset_fingerprint": bundle.manifest_content_sha256,
        "overlay_manifest_file_sha256": bundle.manifest_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "dataset_fingerprint_role": (
            "immutable_overlay_manifest_content_sha256"
        ),
    }
    for field, value in expected.items():
        if dataset.get(field) != value:
            raise TargetIsolatedNBPreflightError(
                f"dataset.{field} is not bound to the loaded SO2 overlay."
            )


def _tensor_sha256(tensor: Tensor) -> str:
    work = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(work.dtype).encode("ascii"))
    digest.update(json.dumps(list(work.shape), separators=(",", ":")).encode("ascii"))
    digest.update(work.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TargetIsolatedNBPreflightError(
                f"State entry {name!r} is not a tensor."
            )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_to_cpu(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(child) for child in value)
    if isinstance(value, list):
        return [_clone_to_cpu(child) for child in value]
    return value


def _tree_sha256(value: Any) -> str:
    """Hash checkpoint trees with explicit type framing."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            digest.update(b"tensor\0")
            digest.update(_tensor_sha256(item).encode("ascii"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple\0")
            for child in item:
                update(child)
        elif isinstance(item, list):
            digest.update(b"list\0")
            for child in item:
                update(child)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, np.generic):
            update(item.item())
        elif item is None:
            digest.update(b"none\0")
        elif isinstance(item, bool):
            digest.update(b"bool\1" if item else b"bool\0")
        elif isinstance(item, int):
            digest.update(b"int\0" + str(item).encode("ascii") + b"\0")
        elif isinstance(item, float):
            if math.isnan(item):
                raise TargetIsolatedNBPreflightError(
                    "NaN cannot be checksummed in checkpoint state."
                )
            digest.update(b"float\0" + item.hex().encode("ascii") + b"\0")
        elif isinstance(item, str):
            digest.update(b"string\0" + item.encode("utf-8") + b"\0")
        elif isinstance(item, bytes):
            digest.update(b"bytes\0" + item + b"\0")
        else:
            raise TargetIsolatedNBPreflightError(
                f"Unsupported checkpoint checksum type: {type(item).__name__}."
            )

    update(value)
    return digest.hexdigest()


def _all_gather_objects(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(
        gathered,
        value,
        group=_CONTROL_GROUP,
    )
    return gathered


def _broadcast_rank_zero_object(value: Any, *, rank: int, group: Any) -> Any:
    payload = [value if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0, group=group)
    return payload[0]


def _synchronized_rank_zero_phase(
    name: str,
    *,
    rank: int,
    group: Any,
    action: Any,
) -> Any:
    """Run a rank-zero callable and broadcast success or failure to all peers."""

    envelope: Mapping[str, Any] | None = None
    if rank == 0:
        try:
            envelope = {"ok": True, "value": action()}
        except BaseException as exc:
            envelope = {
                "ok": False,
                "phase": name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    envelope = _broadcast_rank_zero_object(
        envelope,
        rank=rank,
        group=group,
    )
    if not isinstance(envelope, Mapping):
        raise TargetIsolatedNBPreflightError(
            f"Rank-zero phase {name!r} returned a malformed envelope."
        )
    if envelope.get("ok") is not True:
        raise TargetIsolatedNBPreflightError(
            f"Rank-zero phase {name!r} failed: "
            f"{envelope.get('error_type')}: {envelope.get('error')}\n"
            f"{envelope.get('traceback', '')}"
        )
    return envelope.get("value")


def _autocast_context(
    *,
    enabled: bool,
    device: torch.device,
    dtype_name: str,
) -> ContextManager[object]:
    if not enabled:
        return nullcontext()
    normalized = str(dtype_name).lower()
    if normalized == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    elif normalized == "float16" and device.type == "cuda":
        dtype = torch.float16
    elif normalized == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise TargetIsolatedNBPreflightError("Unsupported AMP dtype configuration.")
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _campaign_id(config: Mapping[str, Any]) -> str:
    campaign = config.get("campaign")
    if not isinstance(campaign, Mapping):
        raise TargetIsolatedNBPreflightError("campaign must be a mapping.")
    value = str(campaign.get("campaign_id", "")).strip()
    if not value:
        raise TargetIsolatedNBPreflightError("campaign.campaign_id is required.")
    return value


def validate_target_isolated_config(config: Mapping[str, Any]) -> None:
    """Reassert launch-critical values after general config validation."""

    if config.get("seed") != 0 or _campaign_id(config) != CAMPAIGN_ID:
        raise TargetIsolatedNBPreflightError("Campaign identity or model seed drifted.")
    model = _section(config, "model")
    expected_model = {
        "name": "target-isolated-geometry-modulated-relative-qkv-gat-nb2",
        "family": "target_isolated_geometry_modulated_relative_qkv_graph_transformer",
        "hidden_dim": 256,
        "graph_layers": 4,
        "unique_graph_blocks": 4,
        "graph_block_weight_tying": "none",
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "relative_geometry_dim": 70,
        "geometry_hidden_dim": 128,
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "receiver_chunk_size": 128,
        "max_edges_per_chunk": 50_000,
        "activation_checkpointing": True,
        "fp32_attention_scoring": True,
        "fp32_attention_accumulation": True,
        "incoming_nonself_edges_only": True,
        "neighbor_key_value_state_evolves_across_blocks": False,
        "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise TargetIsolatedNBPreflightError(
                f"model.{field} must be {expected!r}; got {model.get(field)!r}."
            )

    masking = _section(config, "masking")
    expected_masking = {
        "type": "target_only_uniform_per_cell_integer_count",
        "count_min": 1,
        "count_max": EXPECTED_GENES,
        "positions_without_replacement": True,
        "training_mask_realizations_per_target_per_global_epoch": 1,
        "target_visit_count_per_global_epoch": 1,
        "target_cell_only": True,
        "neighbor_cells_artificially_masked": False,
        "only_target_query_rows_masked": True,
        "neighbor_context_rows_fully_observed": True,
        "model_seed_in_mask_derivation": False,
    }
    for field, expected in expected_masking.items():
        if masking.get(field) != expected:
            raise TargetIsolatedNBPreflightError(
                f"masking.{field} must be {expected!r}; got {masking.get(field)!r}."
            )
    if not isinstance(masking.get("mask_base_seed"), int) or not str(
        masking.get("training_mask_seed_namespace", "")
    ).strip():
        raise TargetIsolatedNBPreflightError(
            "Training mask base seed and namespace are required."
        )
    validation_masks = masking.get("validation_masks")
    if not isinstance(validation_masks, Mapping) or (
        validation_masks.get("fixed_across_epochs") is not True
        or validation_masks.get("mask_realizations_per_target") != 1
        or not str(validation_masks.get("seed_namespace", "")).strip()
    ):
        raise TargetIsolatedNBPreflightError(
            "Fixed one-mask-per-target validation configuration drifted."
        )

    dataset = _section(config, "dataset")
    expected_dataset = {
        "training_cells": EXPECTED_TRAINING_CELLS,
        "validation_cells": EXPECTED_VALIDATION_CELLS,
        "test_cells": 0,
        "biological_target_count": EXPECTED_GENES,
        "node_covariate_count": EXPECTED_COVARIATES,
        "training_core_aliases": list(SO2_NB_TRAINING_ALIASES),
        "validation_core_aliases": list(SO2_NB_VALIDATION_ALIASES),
        "test_core_aliases": [],
        "test_partition_present": False,
    }
    for field, expected in expected_dataset.items():
        if dataset.get(field) != expected:
            raise TargetIsolatedNBPreflightError(
                f"dataset.{field} must be {expected!r}; got {dataset.get(field)!r}."
            )

    trainer = _section(config, "trainer")
    expected_trainer = {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "cores_per_optimizer_update": 2,
        "target_shards_per_core": WORLD_SIZE,
        "objective": "pooled_masked_entry_full_constant_negative_binomial_nb2_nll",
        "local_ddp_loss_scale": (
            "world_size_times_local_nll_sum_divided_by_global_masked_entry_count"
        ),
        "likelihood_compute_dtype": "float32",
        "likelihood_outside_autocast": True,
        "amp": True,
        "amp_requires_fp32_equivalence_preflight": True,
        "distributed": True,
        "distributed_backend": "nccl",
        "distributed_world_size": WORLD_SIZE,
        "stage_complete_core_graph_on_device": False,
        "staged_relative_geometry_dtype": "float32",
    }
    for field, expected in expected_trainer.items():
        if trainer.get(field) != expected:
            raise TargetIsolatedNBPreflightError(
                f"trainer.{field} must be {expected!r}; got {trainer.get(field)!r}."
            )

    launcher = _section(config, "launcher")
    expected_launcher = {
        "process_count": WORLD_SIZE,
        "require_exact_visible_devices": VISIBLE_DEVICES,
        "preflight_peak_vram_gib_all_ranks_max": MAXIMUM_PEAK_VRAM_GIB,
        "preflight_minimum_vram_headroom_gib_each_rank": (
            MINIMUM_VRAM_HEADROOM_GIB
        ),
        "disk_safety_min_free_gb": 20,
    }
    for field, expected in expected_launcher.items():
        if launcher.get(field) != expected:
            raise TargetIsolatedNBPreflightError(
                f"launcher.{field} must be {expected!r}; got {launcher.get(field)!r}."
            )

    metadata = _section(config, "metadata")
    if metadata.get("frozen_task_contract_sha256") != FROZEN_TASK_CONTRACT_SHA256:
        raise TargetIsolatedNBPreflightError(
            "metadata.frozen_task_contract_sha256 drifted."
        )
    acceptance = metadata.get("preflight_acceptance")
    if not isinstance(acceptance, Mapping) or dict(acceptance) != {
        "peak_vram_gib_all_ranks_max": MAXIMUM_PEAK_VRAM_GIB,
        "minimum_vram_headroom_gib_each_rank": MINIMUM_VRAM_HEADROOM_GIB,
        "minimum_free_disk_gib": MINIMUM_FREE_DISK_GIB,
    }:
        raise TargetIsolatedNBPreflightError("Preflight acceptance bounds drifted.")


def _build_model(
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
) -> TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer:
    model_config = _section(config, "model")
    reference = bundle.training_batches[0]
    torch.manual_seed(int(config["seed"]))
    model = TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=reference.n_genes,
        node_covariate_dim=reference.node_covariates.shape[1],
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
        qk_normalization_epsilon=float(
            model_config["qk_normalization_epsilon"]
        ),
        logit_scale_initial=float(model_config["logit_scale_initial"]),
        logit_scale_minimum=float(model_config["logit_scale_minimum"]),
        logit_scale_maximum=float(model_config["logit_scale_maximum"]),
        modulation_amplitude=float(model_config["modulation_amplitude"]),
        geometry_bias_bound=float(model_config["geometry_bias_bound"]),
        receiver_chunk_size=int(model_config["receiver_chunk_size"]),
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
    return model


def _model_topology_receipt(nn_model: nn.Module) -> dict[str, Any]:
    count = sum(parameter.numel() for parameter in nn_model.parameters())
    blocks = getattr(nn_model, "blocks", None)
    raw_theta = getattr(nn_model, "raw_theta", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TargetIsolatedNBPreflightError("Model blocks are not a ModuleList.")
    block_counts = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in blocks
    ]
    parameter_sets = [
        {id(parameter) for parameter in block.parameters()} for block in blocks
    ]
    disjoint = all(
        left.isdisjoint(right)
        for index, left in enumerate(parameter_sets)
        for right in parameter_sets[index + 1 :]
    )
    state_keys = list(nn_model.state_dict())
    state_key_sha = canonical_sha256(state_keys)
    passed = bool(
        isinstance(
            nn_model,
            TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
        )
        and count == EXPECTED_PARAMETER_COUNT
        and len(blocks) == EXPECTED_BLOCK_COUNT
        and len({id(block) for block in blocks}) == EXPECTED_BLOCK_COUNT
        and disjoint
        and block_counts
        == [EXPECTED_BLOCK_PARAMETER_COUNT for _ in range(EXPECTED_BLOCK_COUNT)]
        and isinstance(raw_theta, nn.Parameter)
        and raw_theta.numel() == EXPECTED_GENES
        and list(name for name, _ in nn_model.named_modules()).count("encoder") == 1
    )
    if not passed:
        raise TargetIsolatedNBPreflightError(
            "Target-isolated model topology or parameter count drifted."
        )
    return {
        "passed": True,
        "class": type(nn_model).__name__,
        "parameter_count": count,
        "graph_block_count": len(blocks),
        "unique_graph_block_count": len({id(block) for block in blocks}),
        "per_block_parameter_count": block_counts,
        "raw_theta_parameter_count": raw_theta.numel(),
        "state_dict_key_sha256": state_key_sha,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
        "disjoint_graph_block_parameter_sets": True,
        "shared_encoder_module_count": 1,
    }


def _disk_space_receipt(output: Path) -> dict[str, Any]:
    probe = output.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gib = usage.free / float(1024**3)
    if free_gib < MINIMUM_FREE_DISK_GIB:
        raise TargetIsolatedNBPreflightError(
            f"Free disk {free_gib:.3f} GiB is below {MINIMUM_FREE_DISK_GIB:.1f} GiB."
        )
    return {
        "passed": True,
        "filesystem_probe": str(probe),
        "minimum_free_disk_gib_required": MINIMUM_FREE_DISK_GIB,
        "observed_free_disk_gib": free_gib,
        "observed_free_disk_bytes": int(usage.free),
    }


def _gpu_identity(device: torch.device, *, rank: int, local_rank: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "total_memory_gib": properties.total_memory / float(1024**3),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
    }


def _gpu_inventory_receipt(
    device: torch.device,
    *,
    rank: int,
    local_rank: int,
) -> dict[str, Any]:
    identities = _all_gather_objects(
        _gpu_identity(device, rank=rank, local_rank=local_rank)
    )
    ranks = {int(value["rank"]) for value in identities}
    local_ranks = {int(value["local_rank"]) for value in identities}
    if ranks != set(range(WORLD_SIZE)) or local_ranks != set(range(WORLD_SIZE)):
        raise TargetIsolatedNBPreflightError("GPU rank mapping is incomplete.")
    if any(int(item["total_memory_bytes"]) <= 0 for item in identities):
        raise TargetIsolatedNBPreflightError("A CUDA device reports no memory.")
    return {
        "passed": True,
        "visible_devices": VISIBLE_DEVICES,
        "device_count": len(identities),
        "rank_mapping_complete": True,
        "devices": sorted(identities, key=lambda item: int(item["rank"])),
    }


def _configuration_file_hashes(
    config_path: Path,
    config_root: Path,
) -> dict[str, str]:
    files = {config_path.resolve(strict=True)}
    root = load_yaml_mapping(config_path)
    defaults = root.get("defaults", ())
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise TargetIsolatedNBPreflightError("Experiment defaults must be a sequence.")
    for entry in defaults:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise TargetIsolatedNBPreflightError(
                "Experiment default entry has invalid shape."
            )
        group, name = next(iter(entry.items()))
        files.add((config_root / str(group) / f"{name}.yaml").resolve(strict=True))
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in sorted(files)
    }


def _code_file_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CODE_RELATIVE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise TargetIsolatedNBPreflightError(
                f"Receipt-bound code file is absent: {relative}."
            )
        hashes[str(relative)] = sha256_file(path)
    return hashes


def _overlay_evidence(bundle: SO2NBDataBundle) -> dict[str, Any]:
    try:
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetIsolatedNBPreflightError(
            "Verified overlay manifest became unreadable."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise TargetIsolatedNBPreflightError("Overlay manifest is not a mapping.")
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_content_sha256", None)
    if stored != canonical_sha256(unsigned):
        raise TargetIsolatedNBPreflightError("Overlay content checksum drifted.")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise TargetIsolatedNBPreflightError("Overlay source hashes are absent.")
    return {
        "manifest_path": str(bundle.manifest_path),
        "manifest_file_sha256": bundle.manifest_sha256,
        "manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "source_artifacts": json.loads(
            json.dumps(source_artifacts, sort_keys=True, allow_nan=False)
        ),
    }


def _dataset_receipt(bundle: SO2NBDataBundle) -> dict[str, Any]:
    training_cells = sum(batch.n_nodes for batch in bundle.training_batches)
    validation_cells = sum(batch.n_nodes for batch in bundle.validation_batches)
    if (
        training_cells != EXPECTED_TRAINING_CELLS
        or validation_cells != EXPECTED_VALIDATION_CELLS
        or any(batch.n_genes != EXPECTED_GENES for batch in bundle.all_batches)
        or any(
            batch.node_covariates.shape[1] != EXPECTED_COVARIATES
            for batch in bundle.all_batches
        )
    ):
        raise TargetIsolatedNBPreflightError("Loaded SO2 roster dimensions drifted.")
    per_core = {
        batch.alias: {
            "role": batch.role,
            "n_nodes": batch.n_nodes,
            "n_edges": batch.n_edges,
            "n_genes": batch.n_genes,
        }
        for batch in bundle.all_batches
    }
    representative_training = per_core[REPRESENTATIVE_TRAINING_ALIAS]
    representative_validation = per_core[REPRESENTATIVE_VALIDATION_ALIAS]
    if representative_training != {
        "role": "train",
        "n_nodes": REPRESENTATIVE_TRAINING_NODES,
        "n_edges": REPRESENTATIVE_TRAINING_EDGES,
        "n_genes": EXPECTED_GENES,
    } or representative_validation != {
        "role": "validation",
        "n_nodes": REPRESENTATIVE_VALIDATION_NODES,
        "n_edges": REPRESENTATIVE_VALIDATION_EDGES,
        "n_genes": EXPECTED_GENES,
    }:
        raise TargetIsolatedNBPreflightError(
            "Representative C23/C27 graph dimensions drifted."
        )
    all_records = list(per_core.values())
    validation_records = [
        per_core[alias] for alias in SO2_NB_VALIDATION_ALIASES
    ]
    training_is_largest = bool(
        representative_training["n_nodes"]
        == max(record["n_nodes"] for record in all_records)
        and representative_training["n_edges"]
        == max(record["n_edges"] for record in all_records)
    )
    validation_is_largest = bool(
        representative_validation["n_nodes"]
        == max(record["n_nodes"] for record in validation_records)
        and representative_validation["n_edges"]
        == max(record["n_edges"] for record in validation_records)
    )
    if not training_is_largest or not validation_is_largest:
        raise TargetIsolatedNBPreflightError(
            "Representative train/validation cores do not upper-bound their roles."
        )
    return {
        "training_aliases": list(SO2_NB_TRAINING_ALIASES),
        "validation_aliases": list(SO2_NB_VALIDATION_ALIASES),
        "test_aliases": [],
        "training_core_count": len(bundle.training_batches),
        "validation_core_count": len(bundle.validation_batches),
        "test_core_count": 0,
        "training_cells": training_cells,
        "validation_cells": validation_cells,
        "test_cells": 0,
        "biological_target_count": EXPECTED_GENES,
        "node_covariate_count": EXPECTED_COVARIATES,
        "representative_training_alias": REPRESENTATIVE_TRAINING_ALIAS,
        "representative_validation_alias": REPRESENTATIVE_VALIDATION_ALIAS,
        "representative_training_is_largest_by_nodes_and_edges": (
            training_is_largest
        ),
        "representative_validation_is_largest_by_nodes_and_edges": (
            validation_is_largest
        ),
        "per_core": per_core,
    }


def _synthetic_target_coverage_receipt(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    masking = _section(config, "masking")
    n_cells = 29
    n_genes = EXPECTED_GENES
    partitions = make_contiguous_target_partitions(n_cells, WORLD_SIZE)
    coverage = target_partition_coverage_receipt(partitions)
    visit_counts = np.zeros(n_cells, dtype=np.int64)
    receipts: list[Mapping[str, Any]] = []
    minimum = n_genes
    maximum = 0
    for partition in partitions:
        realization = make_training_target_cell_masks(
            n_cells,
            n_genes,
            base_seed=int(masking["mask_base_seed"]),
            namespace=str(masking["training_mask_seed_namespace"]),
            core_alias=REPRESENTATIVE_TRAINING_ALIAS,
            global_epoch=1,
            target_indices=partition.target_indices,
        )
        visit_counts[partition.target_indices] += 1
        minimum = min(minimum, int(realization.masked_gene_counts.min()))
        maximum = max(maximum, int(realization.masked_gene_counts.max()))
        receipts.append(realization.to_receipt())
    if (
        not np.array_equal(visit_counts, np.ones(n_cells, dtype=np.int64))
        or minimum < 1
        or maximum > n_genes
    ):
        raise TargetIsolatedNBPreflightError(
            "Synthetic target coverage or nonempty-mask gate failed."
        )
    return {
        "passed": True,
        "synthetic_cells": n_cells,
        "world_size": WORLD_SIZE,
        "visit_count_min": int(visit_counts.min()),
        "visit_count_max": int(visit_counts.max()),
        "unique_targets": int(np.count_nonzero(visit_counts)),
        "coverage_receipt": coverage,
        "mask_receipts_sha256": canonical_sha256(receipts),
        "masked_gene_count_min": minimum,
        "masked_gene_count_max": maximum,
        "training_mask_seed_namespace": str(
            masking["training_mask_seed_namespace"]
        ),
        "global_epoch": 1,
    }


def _full_constant_fp32_nb2_receipt() -> dict[str, Any]:
    absolute_tolerance = 2e-5
    relative_tolerance = 2e-6
    mu = torch.tensor(
        [[0.1, 1.3, 7.0], [3.2, 0.5, 19.0]],
        dtype=torch.float32,
    )
    theta = torch.tensor([0.7, 1.0, 4.0], dtype=torch.float32)
    raw_target = torch.tensor(
        [[0, 1, 729], [2, 0, 17]],
        dtype=torch.int32,
    )
    mask = torch.tensor(
        [[True, False, True], [False, True, True]],
        dtype=torch.bool,
    )
    actual = negative_binomial_nll_elementwise(mu, theta, raw_target)
    distribution = torch.distributions.NegativeBinomial(
        total_count=theta.view(1, -1),
        logits=torch.log(mu) - torch.log(theta.view(1, -1)),
    )
    expected = -distribution.log_prob(raw_target.float())
    actual_sum = masked_negative_binomial_nll_sum(
        mu,
        theta,
        raw_target,
        mask,
    )
    expected_sum = expected.masked_select(mask).sum(dtype=torch.float32)
    absolute_error = (actual - expected).abs()
    allowed_error = absolute_tolerance + relative_tolerance * expected.abs()
    scaled_error_ratio = absolute_error / allowed_error
    maximum_error = float(absolute_error.max())
    maximum_allowed_error = float(allowed_error.max())
    maximum_scaled_error_ratio = float(scaled_error_ratio.max())
    sum_error = abs(float(actual_sum) - float(expected_sum))
    sum_allowed_error = absolute_tolerance + relative_tolerance * abs(
        float(expected_sum)
    )
    if (
        actual.dtype != torch.float32
        or actual_sum.dtype != torch.float32
        or not bool(torch.isfinite(actual).all())
        or not bool((absolute_error <= allowed_error).all())
        or sum_error > sum_allowed_error
    ):
        raise TargetIsolatedNBPreflightError(
            "FP32 full-constant NB2 likelihood gate failed."
        )
    return {
        "passed": True,
        "dtype": "float32",
        "includes_full_combinatorial_constant": True,
        "contains_zero_and_count_729": True,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_error_vs_torch_distribution": maximum_error,
        "maximum_allowed_elementwise_error": maximum_allowed_error,
        "maximum_scaled_error_ratio": maximum_scaled_error_ratio,
        "summed_nll_absolute_error": sum_error,
        "summed_nll_allowed_error": sum_allowed_error,
        "masked_entries": int(mask.sum()),
    }


def _loss_decrease_check(raw_counts: Tensor) -> dict[str, Any]:
    counts = raw_counts.detach().cpu().to(torch.int32)
    logits = nn.Parameter(torch.zeros(counts.shape, dtype=torch.float32))
    raw_theta = nn.Parameter(
        torch.full(
            (counts.shape[1],),
            math.log(math.expm1(1.0 - 1e-4)),
            dtype=torch.float32,
        )
    )
    mask = torch.ones_like(counts, dtype=torch.bool)
    optimizer = torch.optim.Adam((logits, raw_theta), lr=5e-2)

    def objective() -> Tensor:
        return masked_negative_binomial_nll(
            nb2_mean_from_logits(logits),
            nb2_inverse_dispersion_from_raw(raw_theta),
            counts,
            mask,
        )

    initial = float(objective().detach())
    gradients_finite_nonzero = True
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        gradients_finite_nonzero = bool(
            gradients_finite_nonzero
            and all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(torch.linalg.vector_norm(parameter.grad)) > 0.0
                for parameter in (logits, raw_theta)
            )
        )
        optimizer.step()
    final = float(objective().detach())
    if not gradients_finite_nonzero or not final < initial:
        raise TargetIsolatedNBPreflightError("Bounded NB2 loss did not decrease.")
    return {
        "initial_nll": initial,
        "final_nll": final,
        "decrease": initial - final,
        "steps": 16,
        "finite_nonzero_gradients": True,
        "shape": list(counts.shape),
    }


def _bounded_loss_decrease_receipt(batch: SO2NBCoreBatch) -> dict[str, Any]:
    synthetic = torch.tensor(
        [
            [0, 1, 2, 5, 729],
            [7, 0, 19, 3, 128],
            [1, 4, 0, 31, 512],
            [2, 8, 16, 0, 255],
        ],
        dtype=torch.int32,
    )
    real = batch.raw_count_target[:32, :64]
    return {
        "passed": True,
        "synthetic": _loss_decrease_check(synthetic),
        "real_subset": {
            "source_alias": batch.alias,
            **_loss_decrease_check(real),
        },
    }


class _SyntheticNBParameters(nn.Module):
    """Two scalar parameters for the unequal-count DDP reduction proof."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_mu = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.raw_theta = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))

    def forward(self, rows: int) -> tuple[Tensor, Tensor]:
        logits = self.raw_mu.expand(rows, 1)
        raw_theta = self.raw_theta.expand(1)
        return nb2_mean_from_logits(logits), nb2_inverse_dispersion_from_raw(
            raw_theta
        )


def _unequal_count_ddp_weighting_receipt(
    *,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    torch.manual_seed(73)
    module = _SyntheticNBParameters().to(device)
    distributed = DistributedDataParallel(
        module,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
    )
    local_rows = rank + 1
    local_target = (
        torch.arange(local_rows, device=device, dtype=torch.int32).view(-1, 1)
        + rank
    )
    local_mask = torch.ones_like(local_target, dtype=torch.bool)
    local_count = torch.tensor(local_mask.numel(), device=device, dtype=torch.int64)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    mu, theta = distributed(local_rows)
    local_sum = masked_negative_binomial_nll_sum(
        mu,
        theta,
        local_target,
        local_mask,
    )
    scaled = local_sum * (float(WORLD_SIZE) / global_count.to(torch.float32))
    scaled.backward()
    observed = torch.stack(
        (
            module.raw_mu.grad.detach(),
            module.raw_theta.grad.detach(),
        )
    )
    gathered_gradients = _all_gather_objects(observed.cpu().tolist())

    reference_raw_mu = torch.tensor(
        0.25,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    reference_raw_theta = torch.tensor(
        -0.15,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    all_targets = torch.cat(
        [
            (
                torch.arange(other_rank + 1, device=device, dtype=torch.int32)
                + other_rank
            ).view(-1, 1)
            for other_rank in range(WORLD_SIZE)
        ]
    )
    reference_mu = nb2_mean_from_logits(
        reference_raw_mu.expand(all_targets.shape[0], 1)
    )
    reference_theta = nb2_inverse_dispersion_from_raw(
        reference_raw_theta.expand(1)
    )
    reference_sum = masked_negative_binomial_nll_sum(
        reference_mu,
        reference_theta,
        all_targets,
        torch.ones_like(all_targets, dtype=torch.bool),
    )
    reference_loss = reference_sum / float(all_targets.numel())
    reference_loss.backward()
    expected = torch.stack(
        (reference_raw_mu.grad, reference_raw_theta.grad)
    ).detach()
    maximum_error = float((observed - expected).abs().max().cpu())
    rank_gradients_equal = all(
        np.allclose(gathered_gradients[0], item, rtol=0.0, atol=1e-7)
        for item in gathered_gradients[1:]
    )
    if (
        int(global_count) != sum(range(1, WORLD_SIZE + 1))
        or maximum_error > 2e-6
        or not rank_gradients_equal
    ):
        raise TargetIsolatedNBPreflightError(
            "Unequal-count DDP weighting differs from the pooled reference."
        )
    del distributed, module
    return {
        "passed": True,
        "local_masked_entry_counts_by_rank": list(range(1, WORLD_SIZE + 1)),
        "global_masked_entry_count": int(global_count),
        "ddp_loss_scale_rule": "world_size_times_local_sum_over_global_count",
        "ddp_gradients_by_rank": gathered_gradients,
        "single_process_reference_gradient": expected.cpu().tolist(),
        "maximum_absolute_gradient_error": maximum_error,
        "tolerance": 2e-6,
    }


def _synchronized_state_receipt(
    nn_model: nn.Module,
    *,
    phase: str,
) -> dict[str, Any]:
    local_hash = _state_dict_sha256(nn_model.state_dict())
    hashes = [str(value) for value in _all_gather_objects(local_hash)]
    if len(set(hashes)) != 1:
        raise TargetIsolatedNBPreflightError(
            f"DDP parameters diverged during {phase}."
        )
    return {
        "passed": True,
        "phase": phase,
        "per_rank_state_dict_sha256": hashes,
    }


def _gradient_scope_receipt(module: nn.Module, *, label: str) -> dict[str, Any]:
    gradients: list[Tensor] = []
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        gradients.append(gradient.reshape(-1))
    if missing or nonfinite or not gradients:
        raise TargetIsolatedNBPreflightError(
            f"Gradient scope {label} is incomplete or non-finite; "
            f"missing={missing}, nonfinite={nonfinite}."
        )
    vector = torch.cat(gradients)
    norm = float(torch.linalg.vector_norm(vector).cpu())
    nonzero = int(torch.count_nonzero(vector).cpu())
    if not math.isfinite(norm) or norm <= 0.0 or nonzero <= 0:
        raise TargetIsolatedNBPreflightError(
            f"Gradient scope {label} is identically zero."
        )
    return {
        "label": label,
        "parameter_tensor_count": len(gradients),
        "gradient_element_count": int(vector.numel()),
        "nonzero_gradient_element_count": nonzero,
        "l2_norm_before_clip": norm,
        "finite": True,
        "nonzero": True,
    }


def _training_mask_for_rank(
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[Any, Any]:
    partition = make_contiguous_target_partition(
        batch.n_nodes,
        world_size=WORLD_SIZE,
        rank=rank,
    )
    masking = _section(config, "masking")
    realization = make_training_target_cell_masks(
        batch.n_nodes,
        batch.n_genes,
        base_seed=int(masking["mask_base_seed"]),
        namespace=str(masking["training_mask_seed_namespace"]),
        core_alias=batch.alias,
        global_epoch=1,
        target_indices=partition.target_indices,
    )
    return partition, realization


def _validation_mask_for_rank(
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[Any, Any]:
    partition = make_contiguous_target_partition(
        batch.n_nodes,
        world_size=WORLD_SIZE,
        rank=rank,
    )
    masking = _section(config, "masking")
    validation = masking["validation_masks"]
    realization = make_validation_target_cell_masks(
        batch.n_nodes,
        batch.n_genes,
        namespace=str(validation["seed_namespace"]),
        core_alias=batch.alias,
        target_indices=partition.target_indices,
    )
    return partition, realization


def _make_target_inputs(
    clean_expression: Tensor,
    mask: Tensor,
    *,
    start: int,
    stop: int,
) -> Tensor:
    target = clean_expression[start:stop].clone()
    target.masked_fill_(mask, 0.0)
    if not bool((target.masked_select(mask) == 0).all()):
        raise TargetIsolatedNBPreflightError("Target input masking failed.")
    if not torch.equal(
        target.masked_select(~mask),
        clean_expression[start:stop].masked_select(~mask),
    ):
        raise TargetIsolatedNBPreflightError("Visible target input values drifted.")
    return target


def _real_training_shard_update(
    *,
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer = _section(config, "trainer")
    partition, realization = _training_mask_for_rank(batch, config, rank=rank)
    if partition.n_targets <= 0:
        raise TargetIsolatedNBPreflightError("Representative shard is empty.")
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(device=device)
    target_input = _make_target_inputs(
        input_expression,
        mask,
        start=partition.start,
        stop=partition.stop,
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device,
        dtype=torch.int32,
    )
    local_count = torch.tensor(int(mask.sum()), device=device, dtype=torch.int64)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    optimizer.zero_grad(set_to_none=True)
    ddp_model.train()
    with _autocast_context(
        enabled=True,
        device=device,
        dtype_name=str(trainer["amp_dtype"]),
    ):
        output = ddp_model(
            input_expression=input_expression,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=batch.edge_index,
            relative_geometry=batch.relative_geometry,
            node_covariates=covariates,
            target_start=partition.start,
            target_stop=partition.stop,
        )
    if not isinstance(output, NegativeBinomialModelOutput):
        raise TargetIsolatedNBPreflightError("Real training output type drifted.")
    local_sum = masked_negative_binomial_nll_sum(
        output.mu,
        output.theta,
        raw_target,
        mask,
    )
    scaled_loss = local_sum * (
        float(WORLD_SIZE) / global_count.to(dtype=torch.float32)
    )
    if scaled_loss.dtype != torch.float32 or not bool(torch.isfinite(scaled_loss)):
        raise TargetIsolatedNBPreflightError("Real training loss is not finite FP32.")
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer)

    block_gradients = [
        _gradient_scope_receipt(block, label=f"graph_block_{index}")
        for index, block in enumerate(model.blocks)
    ]
    theta_gradient = model.raw_theta.grad
    if theta_gradient is None or not bool(torch.isfinite(theta_gradient).all()):
        raise TargetIsolatedNBPreflightError("raw_theta gradient is absent/non-finite.")
    theta_norm = float(torch.linalg.vector_norm(theta_gradient.float()).cpu())
    if theta_norm <= 0.0:
        raise TargetIsolatedNBPreflightError("raw_theta gradient is zero.")
    all_trainable = _gradient_scope_receipt(model, label="full_model")
    clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(trainer["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    pooled_sum = local_sum.detach().clone()
    torch.distributed.all_reduce(pooled_sum, op=torch.distributed.ReduceOp.SUM)
    pooled_nll = float((pooled_sum / global_count.to(torch.float32)).cpu())
    local_records = _all_gather_objects(
        {
            "rank": rank,
            "target_start": partition.start,
            "target_stop": partition.stop,
            "target_count": partition.n_targets,
            "masked_entries": int(local_count),
            "local_nll_sum": float(local_sum.detach().cpu()),
            "mask_receipt_sha256": realization.to_receipt()["receipt_sha256"],
        }
    )
    global_target_count = sum(
        int(item["target_count"]) for item in local_records
    )
    ordered_ranges = [
        (int(item["target_start"]), int(item["target_stop"]))
        for item in sorted(local_records, key=lambda item: int(item["rank"]))
    ]
    expected_ranges = [
        (partition.start, partition.stop)
        for partition in make_contiguous_target_partitions(
            batch.n_nodes,
            WORLD_SIZE,
        )
    ]
    if global_target_count != batch.n_nodes or ordered_ranges != expected_ranges:
        raise TargetIsolatedNBPreflightError(
            "Real training target shards are not exact contiguous coverage."
        )
    if not math.isfinite(pooled_nll):
        raise TargetIsolatedNBPreflightError("Pooled real training NLL is non-finite.")
    training_receipt = {
        "passed": True,
        "alias": batch.alias,
        "full_core_nodes": batch.n_nodes,
        "full_core_edges": batch.n_edges,
        "rank_shards": sorted(local_records, key=lambda item: int(item["rank"])),
        "global_target_count": global_target_count,
        "global_masked_entry_count": int(global_count),
        "pooled_masked_nb2_nll": pooled_nll,
        "local_loss_scale_rule": (
            "world_size_times_local_nll_sum_divided_by_global_masked_entry_count"
        ),
        "optimizer_steps": 1,
        "clip_norm_before_clip": float(clip_norm.detach().cpu()),
    }
    gradient_receipt = {
        "passed": True,
        "full_model": all_trainable,
        "blocks": block_gradients,
        "raw_theta": {
            "finite": True,
            "nonzero": True,
            "l2_norm_before_clip": theta_norm,
        },
    }
    del output, target_input, mask, raw_target, input_expression, covariates
    model.clear_edge_layout_cache()
    return training_receipt, gradient_receipt


def _real_validation_shard_replay(
    *,
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    ddp_model: DistributedDataParallel,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer = _section(config, "trainer")
    partition, first_realization = _validation_mask_for_rank(
        batch,
        config,
        rank=rank,
    )
    _, replay_realization = _validation_mask_for_rank(
        batch,
        config,
        rank=rank,
    )
    first_receipt = first_realization.to_receipt()
    replay_receipt = replay_realization.to_receipt()
    if first_receipt != replay_receipt or not np.array_equal(
        first_realization.mask,
        replay_realization.mask,
    ):
        raise TargetIsolatedNBPreflightError("Fixed validation masks did not replay.")
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(first_realization.mask, copy=True)).to(
        device=device
    )
    target_input = _make_target_inputs(
        input_expression,
        mask,
        start=partition.start,
        stop=partition.stop,
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device,
        dtype=torch.int32,
    )
    local_count = torch.tensor(int(mask.sum()), device=device, dtype=torch.int64)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    ddp_model.eval()

    def evaluate_once() -> float:
        with torch.no_grad(), _autocast_context(
            enabled=True,
            device=device,
            dtype_name=str(trainer["amp_dtype"]),
        ):
            output = ddp_model(
                input_expression=input_expression,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=batch.edge_index,
                relative_geometry=batch.relative_geometry,
                node_covariates=covariates,
                target_start=partition.start,
                target_stop=partition.stop,
            )
        if not isinstance(output, NegativeBinomialModelOutput):
            raise TargetIsolatedNBPreflightError("Validation output type drifted.")
        local_sum = masked_negative_binomial_nll_sum(
            output.mu,
            output.theta,
            raw_target,
            mask,
        ).detach()
        torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        value = float((local_sum / global_count.to(torch.float32)).cpu())
        if not math.isfinite(value):
            raise TargetIsolatedNBPreflightError("Validation NLL is non-finite.")
        return value

    first_metric = evaluate_once()
    replay_metric = evaluate_once()
    if first_metric != replay_metric:
        raise TargetIsolatedNBPreflightError(
            "Fixed validation pooled metric failed exact replay."
        )
    local_receipts = _all_gather_objects(
        {
            "rank": rank,
            "target_start": partition.start,
            "target_stop": partition.stop,
            "target_count": partition.n_targets,
            "masked_entries": int(local_count),
            "receipt_sha256": first_receipt["receipt_sha256"],
            "mask_sha256": first_receipt["mask_sha256"],
        }
    )
    total_targets = sum(int(item["target_count"]) for item in local_receipts)
    if total_targets != batch.n_nodes:
        raise TargetIsolatedNBPreflightError(
            "Validation target shards are not exhaustive."
        )
    mask_gate = {
        "passed": True,
        "alias": batch.alias,
        "fixed_across_epochs": True,
        "rank_receipts": sorted(
            local_receipts,
            key=lambda item: int(item["rank"]),
        ),
        "rank_receipts_sha256": canonical_sha256(local_receipts),
        "global_target_count": total_targets,
        "global_masked_entry_count": int(global_count),
        "all_masks_nonempty": True,
    }
    metric_gate = {
        "passed": True,
        "alias": batch.alias,
        "full_core_nodes": batch.n_nodes,
        "full_core_edges": batch.n_edges,
        "first_pooled_masked_nb2_nll": first_metric,
        "replayed_pooled_masked_nb2_nll": replay_metric,
        "bit_exact_replay": True,
    }
    del target_input, mask, raw_target, input_expression, covariates
    model.clear_edge_layout_cache()
    return mask_gate, metric_gate


def _peak_vram_receipt(
    device: torch.device,
    *,
    rank: int,
    local_rank: int,
) -> dict[str, Any]:
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    local["allocated_headroom_bytes"] = (
        local["total_memory_bytes"] - local["peak_allocated_bytes"]
    )
    local["reserved_headroom_bytes"] = (
        local["total_memory_bytes"] - local["peak_reserved_bytes"]
    )
    records = sorted(
        _all_gather_objects(local),
        key=lambda item: int(item["rank"]),
    )
    peak_allocated = max(int(item["peak_allocated_bytes"]) for item in records)
    peak_reserved = max(int(item["peak_reserved_bytes"]) for item in records)
    minimum_allocated_headroom = min(
        int(item["allocated_headroom_bytes"]) for item in records
    )
    minimum_reserved_headroom = min(
        int(item["reserved_headroom_bytes"]) for item in records
    )
    gib = float(1024**3)
    receipt = {
        "passed": True,
        "maximum_peak_vram_gib_allowed": MAXIMUM_PEAK_VRAM_GIB,
        "minimum_vram_headroom_gib_required": MINIMUM_VRAM_HEADROOM_GIB,
        "peak_allocated_vram_gib_all_ranks": peak_allocated / gib,
        "peak_reserved_vram_gib_all_ranks": peak_reserved / gib,
        "minimum_allocated_headroom_gib_all_ranks": (
            minimum_allocated_headroom / gib
        ),
        "minimum_reserved_headroom_gib_all_ranks": (
            minimum_reserved_headroom / gib
        ),
        "per_rank": [
            {
                **item,
                "peak_allocated_gib": item["peak_allocated_bytes"] / gib,
                "peak_reserved_gib": item["peak_reserved_bytes"] / gib,
                "allocated_headroom_gib": item["allocated_headroom_bytes"] / gib,
                "reserved_headroom_gib": item["reserved_headroom_bytes"] / gib,
            }
            for item in records
        ],
    }
    if (
        receipt["peak_allocated_vram_gib_all_ranks"] > MAXIMUM_PEAK_VRAM_GIB
        or receipt["peak_reserved_vram_gib_all_ranks"] > MAXIMUM_PEAK_VRAM_GIB
        or receipt["minimum_allocated_headroom_gib_all_ranks"]
        < MINIMUM_VRAM_HEADROOM_GIB
        or receipt["minimum_reserved_headroom_gib_all_ranks"]
        < MINIMUM_VRAM_HEADROOM_GIB
    ):
        raise TargetIsolatedNBPreflightError("Peak VRAM/headroom gate failed.")
    return receipt


def _synthetic_model_inputs(
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(9917)
    clean = torch.randn(4, EXPECTED_GENES, generator=generator).to(device)
    covariates = torch.randn(4, EXPECTED_COVARIATES, generator=generator).to(
        device
    )
    edge_index = torch.tensor(
        [
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70, generator=generator)
    mask = torch.zeros(2, EXPECTED_GENES, dtype=torch.bool, device=device)
    mask[0, [0, 7, 13, 101]] = True
    mask[1, [1, 5, 17, 103]] = True
    target_input = _make_target_inputs(clean, mask, start=0, stop=2)
    raw_target = torch.arange(
        2 * EXPECTED_GENES,
        device=device,
        dtype=torch.int64,
    ).view(2, EXPECTED_GENES)
    raw_target = torch.remainder(raw_target, 37).to(torch.int32)
    return clean, target_input, mask, covariates, edge_index, geometry, raw_target


def _model_invariant_receipts(
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    *,
    device: torch.device,
) -> dict[str, Mapping[str, Any]]:
    (
        clean,
        target_input,
        mask,
        covariates,
        edge_index,
        geometry,
        _,
    ) = _synthetic_model_inputs(device=device)
    previous_training = model.training
    model.eval()
    encoder_mask_counts: list[int] = []
    geometry_edge_counts: list[list[int]] = [[] for _ in model.blocks]

    def encoder_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        gene_mask = kwargs.get("gene_mask")
        if not isinstance(gene_mask, Tensor):
            raise TargetIsolatedNBPreflightError("Encoder mask hook saw no mask.")
        encoder_mask_counts.append(int(gene_mask.sum()))

    handles = [
        model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True)
    ]
    for index, block in enumerate(model.blocks):
        handles.append(
            block.geometry_encoder.register_forward_pre_hook(
                lambda _module, arguments, block_index=index: (
                    geometry_edge_counts[block_index].append(
                        int(arguments[0].shape[0])
                    )
                )
            )
        )
    try:
        with torch.no_grad():
            baseline = model(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=edge_index,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
    finally:
        for handle in handles:
            handle.remove()
    if encoder_mask_counts != [0, int(mask.sum())]:
        raise TargetIsolatedNBPreflightError(
            "Clean source encoder received a nonzero mask or streams did not share it."
        )
    if any(sum(counts) != 6 for counts in geometry_edge_counts):
        raise TargetIsolatedNBPreflightError(
            "A graph block evaluated edges outside the target receiver range."
        )

    changed_hidden = clean.clone()
    changed_hidden[0, 0] += 10_000.0
    with torch.no_grad():
        hidden_output = model(
            input_expression=changed_hidden,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    own_equal = torch.equal(baseline.mu[0], hidden_output.mu[0])
    cross_changed = not torch.equal(baseline.mu[1], hidden_output.mu[1])

    changed_neighbor = clean.clone()
    changed_neighbor[2, 2] += 100.0
    with torch.no_grad():
        neighbor_output = model(
            input_expression=changed_neighbor,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    neighbor_changed = not torch.equal(baseline.mu[0], neighbor_output.mu[0])

    changed_visible = clean.clone()
    changed_visible[0, 2] += 100.0
    changed_target_input = target_input.clone()
    changed_target_input[0, 2] = changed_visible[0, 2]
    with torch.no_grad():
        visible_output = model(
            input_expression=changed_visible,
            target_input_expression=changed_target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    visible_changed = not torch.equal(baseline.mu[0], visible_output.mu[0])

    self_loop_rejected = False
    loop_edges = edge_index.clone()
    loop_edges[0, 0] = loop_edges[1, 0]
    try:
        model(
            input_expression=clean,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=loop_edges,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        self_loop_rejected = "self loop" in str(exc)

    nonzero_withheld_rejected = False
    bad_target = target_input.clone()
    bad_target[0, 0] = 1.0
    try:
        model(
            input_expression=clean,
            target_input_expression=bad_target,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        nonzero_withheld_rejected = "exactly zero" in str(exc)

    visible_drift_rejected = False
    bad_visible = target_input.clone()
    bad_visible[0, 2] += 1.0
    try:
        model(
            input_expression=clean,
            target_input_expression=bad_visible,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        visible_drift_rejected = "visible" in str(exc)

    forward_parameters = set(inspect.signature(model.forward).parameters)
    forbidden_forward = {
        "raw_target",
        "raw_count_target",
        "target_counts",
        "library_total",
        "size_factor",
        "offset",
    }
    raw_target_absent = not bool(forward_parameters & forbidden_forward)
    if not all(
        (
            own_equal,
            cross_changed,
            neighbor_changed,
            visible_changed,
            self_loop_rejected,
            nonzero_withheld_rejected,
            visible_drift_rejected,
            raw_target_absent,
            getattr(model, "source_bank_evolves", None) is False,
        )
    ):
        raise TargetIsolatedNBPreflightError(
            "Target/context routing or leakage invariant failed."
        )
    model.train(previous_training)
    model.clear_edge_layout_cache()
    return {
        "shared_encoder_two_stream": {
            "passed": True,
            "shared_encoder_object_count": 1,
            "encoder_masked_entry_counts_by_call": encoder_mask_counts,
            "clean_source_bank_static": True,
        },
        "target_mask_contract": {
            "passed": True,
            "every_target_mask_nonempty": True,
            "masked_target_input_exact_zero_enforced": nonzero_withheld_rejected,
            "visible_target_input_exact_clean_match_enforced": visible_drift_rejected,
        },
        "neighbor_context_unmasked": {
            "passed": True,
            "clean_source_encoder_masked_entries": encoder_mask_counts[0],
            "simultaneous_target_clean_source_effect_observed": cross_changed,
        },
        "incoming_nonself_routing": {
            "passed": True,
            "synthetic_full_graph_edges": int(edge_index.shape[1]),
            "target_incoming_edges_per_block": [
                sum(counts) for counts in geometry_edge_counts
            ],
            "self_loop_rejected": self_loop_rejected,
        },
        "masked_target_value_invariance": {
            "passed": True,
            "own_output_bit_identical": own_equal,
            "clean_source_changed_other_target": cross_changed,
        },
        "raw_target_forward_isolation": {
            "passed": True,
            "forward_parameter_names": sorted(forward_parameters),
            "forbidden_raw_target_or_total_inputs_absent": raw_target_absent,
        },
        "no_return_leakage": {
            "passed": True,
            "cycle_edges_present": True,
            "source_bank_static_across_all_four_blocks": True,
            "withheld_value_cannot_return_to_own_output": own_equal,
            "visible_target_effect_observed": visible_changed,
            "clean_neighbor_effect_observed": neighbor_changed,
        },
    }


def _amp_fp32_equivalence_receipt(
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    *,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    (
        clean,
        target_input,
        mask,
        covariates,
        edge_index,
        geometry,
        raw_target,
    ) = _synthetic_model_inputs(device=device)
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            fp32 = model(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=edge_index,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
            with _autocast_context(
                enabled=True,
                device=device,
                dtype_name=amp_dtype,
            ):
                amp = model(
                    input_expression=clean,
                    target_input_expression=target_input,
                    target_gene_mask=mask,
                    edge_index=edge_index,
                    relative_geometry=geometry,
                    node_covariates=covariates,
                    target_start=0,
                    target_stop=2,
                )
            fp32_loss = masked_negative_binomial_nll(
                fp32.mu,
                fp32.theta,
                raw_target,
                mask,
            )
            amp_loss = masked_negative_binomial_nll(
                amp.mu,
                amp.theta,
                raw_target,
                mask,
            )
    finally:
        model.train(previous_training)
    difference = (fp32.mu - amp.mu).abs()
    maximum = float(difference.max().cpu())
    mean = float(difference.mean().cpu())
    loss_absolute = abs(float(fp32_loss) - float(amp_loss))
    loss_relative = loss_absolute / max(abs(float(fp32_loss)), 1e-12)
    allowed_loss = AMP_NLL_ABS_TOLERANCE + (
        AMP_NLL_RELATIVE_TOLERANCE * abs(float(fp32_loss))
    )
    if (
        fp32.mu.dtype != torch.float32
        or amp.mu.dtype != torch.float32
        or fp32_loss.dtype != torch.float32
        or amp_loss.dtype != torch.float32
        or maximum > AMP_MU_MAX_ABS_TOLERANCE
        or mean > AMP_MU_MEAN_ABS_TOLERANCE
        or loss_absolute > allowed_loss
    ):
        raise TargetIsolatedNBPreflightError("AMP/FP32 equivalence gate failed.")
    model.clear_edge_layout_cache()
    return {
        "passed": True,
        "synthetic_nodes": int(clean.shape[0]),
        "synthetic_edges": int(edge_index.shape[1]),
        "target_count": int(mask.shape[0]),
        "masked_entries": int(mask.sum()),
        "amp_dtype": amp_dtype,
        "output_and_likelihood_dtype": "float32",
        "mu_maximum_absolute_difference": maximum,
        "mu_mean_absolute_difference": mean,
        "nll_absolute_difference": loss_absolute,
        "nll_relative_difference": loss_relative,
        "nll_combined_absolute_tolerance": allowed_loss,
    }


def _checkpoint_roundtrip_receipt(
    *,
    output_parent: Path,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    validation_metric: float,
) -> dict[str, Any]:
    output_parent.mkdir(parents=True, exist_ok=True)
    early = EarlyStoppingState(
        best_value=float(validation_metric),
        best_epoch=1,
        bad_validations=0,
        completed_epoch=1,
        improved=True,
        should_stop=False,
        stop_reason=None,
    )
    state: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "completed_epoch": 1,
        "optimizer_updates_completed": 1,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": _clone_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_to_cpu(optimizer.state_dict()),
        "amp_scaler_state_dict": _clone_to_cpu(scaler.state_dict()),
        "early_stopping_state": asdict(early),
        "best_validation_metric": float(validation_metric),
        "best_epoch": 1,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": canonical_sha256(config),
        "gradient_vectors_persisted": False,
        "test_artifacts_present": False,
    }
    best_payload: dict[str, Any] = {
        **state,
        "checkpoint_role": "best",
        "best_checkpoint_sha256": None,
        "embedded_best_checkpoint": None,
        "embedded_best_checkpoint_tree_sha256": None,
    }
    with tempfile.TemporaryDirectory(
        prefix=".target-isolated-nb-preflight-checkpoint-",
        dir=output_parent,
    ) as temporary:
        store = AtomicBestLatestCheckpointStore(
            temporary,
            checkpoint_schema=CHECKPOINT_SCHEMA,
            protocol=PROTOCOL,
        )
        best = store.save_best(best_payload)
        latest_payload: dict[str, Any] = {
            **state,
            "checkpoint_role": "latest",
            "best_checkpoint_sha256": best.sha256,
            "embedded_best_checkpoint": _clone_to_cpu(best_payload),
            "embedded_best_checkpoint_tree_sha256": _tree_sha256(best_payload),
        }
        latest = store.save_latest(latest_payload)
        inventory_before = sorted(path.name for path in store.directory.iterdir())
        loaded_best = store.load("best")
        loaded_latest = store.load("latest")
        clone = _build_model(config, bundle)
        clone.load_state_dict(loaded_best["model_state_dict"], strict=True)
        expected_hash = _state_dict_sha256(state["model_state_dict"])
        best_hash = _state_dict_sha256(clone.state_dict())
        latest_hash = _state_dict_sha256(loaded_latest["model_state_dict"])
        final = store.finalize_best_only()
        final_payload = store.load("best")
        final_hash = _state_dict_sha256(final_payload["model_state_dict"])
        inventory_after = sorted(path.name for path in store.directory.iterdir())
        if not (
            inventory_before == ["best.ckpt", "latest.ckpt"]
            and inventory_after == ["best.ckpt"]
            and expected_hash == best_hash == latest_hash == final_hash
            and final.sha256 == best.sha256
            and latest_payload["embedded_best_checkpoint_tree_sha256"]
            == _tree_sha256(loaded_latest["embedded_best_checkpoint"])
        ):
            raise TargetIsolatedNBPreflightError(
                "Atomic custom-schema checkpoint roundtrip failed."
            )
        receipt = {
            "passed": True,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "protocol": PROTOCOL,
            "best_checkpoint_sha256": best.sha256,
            "latest_checkpoint_sha256": latest.sha256,
            "model_state_sha256": expected_hash,
            "active_inventory_before_finalization": inventory_before,
            "retained_inventory_after_finalization": inventory_after,
            "temporary_directory_removed_on_exit": True,
        }
    return receipt


def validate_preflight_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the exact receipt contract consumed by the production runner."""

    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "diagnostic_only": True,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "world_size": WORLD_SIZE,
        "distributed_backend": "nccl",
        "control_plane_backend": "gloo",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "code_hash_scope": CODE_HASH_SCOPE,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise TargetIsolatedNBPreflightError(
                f"Receipt {field} must be {value!r}; got {receipt.get(field)!r}."
            )
    for field in (
        "configuration_sha256",
        "resolved_config_sha256",
        "configuration_file_sha256",
        "frozen_task_contract_sha256",
        "overlay_manifest_sha256",
        "overlay_manifest_content_sha256",
        "preprocessing_fingerprint",
        "split_fingerprint",
    ):
        if not _is_sha256(receipt.get(field)):
            raise TargetIsolatedNBPreflightError(
                f"Receipt {field} is not a SHA-256 digest."
            )
    if receipt.get("configuration_sha256") != receipt.get(
        "resolved_config_sha256"
    ):
        raise TargetIsolatedNBPreflightError("Resolved config hashes disagree.")
    for field in ("configuration_source_file_sha256", "code_file_sha256"):
        values = receipt.get(field)
        if not isinstance(values, Mapping) or not values or any(
            not _is_sha256(value) for value in values.values()
        ):
            raise TargetIsolatedNBPreflightError(
                f"Receipt {field} is missing checksum entries."
            )
    if set(receipt["code_file_sha256"]) != {
        str(path) for path in CODE_RELATIVE_PATHS
    }:
        raise TargetIsolatedNBPreflightError("Receipt code hash scope is incomplete.")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise TargetIsolatedNBPreflightError("Source artifact hashes are absent.")
    if (
        not _is_sha256(receipt.get("source_artifacts_sha256"))
        or receipt["source_artifacts_sha256"]
        != canonical_sha256(source_artifacts)
    ):
        raise TargetIsolatedNBPreflightError(
            "Source-artifact evidence checksum is invalid."
        )
    gpu_identities = receipt.get("gpu_identities")
    gpu_identity_fields = {
        "rank",
        "local_rank",
        "name",
        "total_memory_bytes",
        "total_memory_gib",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
    }
    if (
        not isinstance(gpu_identities, Sequence)
        or isinstance(gpu_identities, (str, bytes))
        or len(gpu_identities) != WORLD_SIZE
        or any(
            not isinstance(item, Mapping)
            or set(item) != gpu_identity_fields
            for item in gpu_identities
        )
    ):
        raise TargetIsolatedNBPreflightError("GPU identity inventory is incomplete.")
    if {int(item["rank"]) for item in gpu_identities} != set(range(WORLD_SIZE)):
        raise TargetIsolatedNBPreflightError("GPU rank mapping drifted.")
    if (
        not _is_sha256(receipt.get("gpu_identities_sha256"))
        or receipt["gpu_identities_sha256"]
        != canonical_sha256(gpu_identities)
    ):
        raise TargetIsolatedNBPreflightError(
            "GPU identity evidence checksum is invalid."
        )
    dataset = receipt.get("dataset")
    expected_dataset = {
        "training_aliases": list(SO2_NB_TRAINING_ALIASES),
        "validation_aliases": list(SO2_NB_VALIDATION_ALIASES),
        "test_aliases": [],
        "training_core_count": 12,
        "validation_core_count": 2,
        "test_core_count": 0,
        "training_cells": EXPECTED_TRAINING_CELLS,
        "validation_cells": EXPECTED_VALIDATION_CELLS,
        "test_cells": 0,
        "biological_target_count": EXPECTED_GENES,
        "node_covariate_count": EXPECTED_COVARIATES,
        "representative_training_alias": REPRESENTATIVE_TRAINING_ALIAS,
        "representative_validation_alias": REPRESENTATIVE_VALIDATION_ALIAS,
        "representative_training_is_largest_by_nodes_and_edges": True,
        "representative_validation_is_largest_by_nodes_and_edges": True,
        "per_core": (
            dataset.get("per_core") if isinstance(dataset, Mapping) else None
        ),
    }
    if (
        not isinstance(dataset, Mapping)
        or set(dataset) != set(expected_dataset)
        or any(
            dataset.get(field) != expected_value
            for field, expected_value in expected_dataset.items()
        )
    ):
        raise TargetIsolatedNBPreflightError("Receipt dataset roster drifted.")
    per_core = dataset["per_core"]
    expected_aliases = set(SO2_NB_TRAINING_ALIASES) | set(
        SO2_NB_VALIDATION_ALIASES
    )
    if not isinstance(per_core, Mapping) or set(per_core) != expected_aliases:
        raise TargetIsolatedNBPreflightError(
            "Receipt per-core dataset evidence is incomplete."
        )
    for alias, record in per_core.items():
        expected_role = (
            "train" if alias in SO2_NB_TRAINING_ALIASES else "validation"
        )
        if (
            not isinstance(record, Mapping)
            or set(record) != {"role", "n_nodes", "n_edges", "n_genes"}
            or record.get("role") != expected_role
            or not isinstance(record.get("n_nodes"), int)
            or int(record["n_nodes"]) <= 0
            or not isinstance(record.get("n_edges"), int)
            or int(record["n_edges"]) <= 0
            or record.get("n_genes") != EXPECTED_GENES
        ):
            raise TargetIsolatedNBPreflightError(
                f"Receipt per-core record is invalid for {alias}."
            )
    if (
        per_core[REPRESENTATIVE_TRAINING_ALIAS]
        != {
            "role": "train",
            "n_nodes": REPRESENTATIVE_TRAINING_NODES,
            "n_edges": REPRESENTATIVE_TRAINING_EDGES,
            "n_genes": EXPECTED_GENES,
        }
        or per_core[REPRESENTATIVE_VALIDATION_ALIAS]
        != {
            "role": "validation",
            "n_nodes": REPRESENTATIVE_VALIDATION_NODES,
            "n_edges": REPRESENTATIVE_VALIDATION_EDGES,
            "n_genes": EXPECTED_GENES,
        }
        or REPRESENTATIVE_TRAINING_NODES
        != max(int(record["n_nodes"]) for record in per_core.values())
        or REPRESENTATIVE_TRAINING_EDGES
        != max(int(record["n_edges"]) for record in per_core.values())
        or REPRESENTATIVE_VALIDATION_NODES
        != max(
            int(per_core[alias]["n_nodes"])
            for alias in SO2_NB_VALIDATION_ALIASES
        )
        or REPRESENTATIVE_VALIDATION_EDGES
        != max(
            int(per_core[alias]["n_edges"])
            for alias in SO2_NB_VALIDATION_ALIASES
        )
    ):
        raise TargetIsolatedNBPreflightError(
            "Receipt C23/C27 representative upper-bound proof failed."
        )
    if (
        not _is_sha256(receipt.get("dataset_evidence_sha256"))
        or receipt["dataset_evidence_sha256"] != canonical_sha256(dataset)
    ):
        raise TargetIsolatedNBPreflightError(
            "Dataset evidence checksum is invalid."
        )
    model = receipt.get("model")
    expected_model = {
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "graph_block_count": EXPECTED_BLOCK_COUNT,
        "unique_graph_block_count": EXPECTED_BLOCK_COUNT,
        "per_block_parameter_count": [
            EXPECTED_BLOCK_PARAMETER_COUNT for _ in range(EXPECTED_BLOCK_COUNT)
        ],
        "raw_theta_parameter_count": EXPECTED_GENES,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
    }
    expected_model_fields = {
        "class",
        "parameter_count",
        "graph_block_count",
        "unique_graph_block_count",
        "per_block_parameter_count",
        "raw_theta_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    }
    if (
        not isinstance(model, Mapping)
        or set(model) != expected_model_fields
        or any(
            model.get(field) != expected_value
            for field, expected_value in expected_model.items()
        )
        or not isinstance(model.get("class"), str)
        or not _is_sha256(model.get("state_dict_key_sha256"))
    ):
        raise TargetIsolatedNBPreflightError("Receipt model topology drifted.")
    gates = receipt.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != REQUIRED_GATE_NAMES
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in gates.values()
        )
    ):
        raise TargetIsolatedNBPreflightError("Required gate inventory is incomplete.")
    resources = receipt.get("resource_gates")
    expected_resource_fields = {
        "minimum_free_disk_gib_required",
        "observed_free_disk_gib",
        "maximum_peak_vram_gib_allowed",
        "minimum_vram_headroom_gib_required",
        "peak_allocated_vram_gib_all_ranks",
        "peak_reserved_vram_gib_all_ranks",
        "minimum_allocated_headroom_gib_all_ranks",
        "minimum_reserved_headroom_gib_all_ranks",
        "per_rank",
    }
    if (
        not isinstance(resources, Mapping)
        or set(resources) != expected_resource_fields
        or (
            float(resources.get("minimum_free_disk_gib_required", -1))
            != MINIMUM_FREE_DISK_GIB
            or float(resources.get("observed_free_disk_gib", -1))
            < MINIMUM_FREE_DISK_GIB
            or float(resources.get("maximum_peak_vram_gib_allowed", -1))
            != MAXIMUM_PEAK_VRAM_GIB
            or float(resources.get("minimum_vram_headroom_gib_required", -1))
            != MINIMUM_VRAM_HEADROOM_GIB
            or float(
                resources.get("peak_allocated_vram_gib_all_ranks", math.inf)
            )
            > MAXIMUM_PEAK_VRAM_GIB
            or float(
                resources.get("peak_reserved_vram_gib_all_ranks", math.inf)
            )
            > MAXIMUM_PEAK_VRAM_GIB
            or float(
                resources.get("minimum_allocated_headroom_gib_all_ranks", -1)
            )
            < MINIMUM_VRAM_HEADROOM_GIB
            or float(
                resources.get("minimum_reserved_headroom_gib_all_ranks", -1)
            )
            < MINIMUM_VRAM_HEADROOM_GIB
        )
    ):
        raise TargetIsolatedNBPreflightError("Receipt resource bounds failed.")
    stored = receipt.get("receipt_content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_content_sha256", None)
    if not _is_sha256(stored) or stored != canonical_sha256(unsigned):
        raise TargetIsolatedNBPreflightError("Receipt self-checksum is invalid.")


def _configuration_contract_receipt(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    bundle: SO2NBDataBundle,
) -> dict[str, Any]:
    local = {
        "configuration_sha256": canonical_sha256(config),
        "configuration_file_sha256": sha256_file(config_path),
        "frozen_task_contract_sha256": sha256_file(
            (PROJECT_ROOT / FROZEN_TASK_CONTRACT).resolve(strict=True)
        ),
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
    }
    per_rank = _all_gather_objects(local)
    if any(item != local for item in per_rank):
        raise TargetIsolatedNBPreflightError(
            "Ranks resolved different configuration or data identities."
        )
    return {
        "passed": True,
        **local,
        "rank_identity_count": len(per_rank),
        "all_ranks_identical": True,
    }


def _no_test_artifacts_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _section(config, "dataset")
    evaluation = _section(config, "evaluation")
    metadata = _section(config, "metadata")
    passed = bool(
        dataset.get("test_core_aliases") == []
        and dataset.get("test_cells") == 0
        and dataset.get("test_partition_present") is False
        and evaluation.get("splits") == ["validation"]
        and evaluation.get("test_split_present") is False
        and metadata.get("no_test_split_or_artifacts") is True
    )
    if not passed:
        raise TargetIsolatedNBPreflightError(
            "The train/validation-only no-test-artifact contract drifted."
        )
    return {
        "passed": True,
        "test_aliases": [],
        "test_cells": 0,
        "evaluation_splits": ["validation"],
        "test_predictions_written": False,
        "full_prediction_matrices_written": False,
        "diagnostic_checkpoint_retained": False,
        "completed_experiment": False,
    }


def _public_model_evidence(topology: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "class",
        "parameter_count",
        "graph_block_count",
        "unique_graph_block_count",
        "per_block_parameter_count",
        "raw_theta_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    )
    return {field: topology[field] for field in fields}


def _resource_evidence(
    disk: Mapping[str, Any],
    vram: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "minimum_free_disk_gib_required": disk[
            "minimum_free_disk_gib_required"
        ],
        "observed_free_disk_gib": disk["observed_free_disk_gib"],
        "maximum_peak_vram_gib_allowed": vram[
            "maximum_peak_vram_gib_allowed"
        ],
        "minimum_vram_headroom_gib_required": vram[
            "minimum_vram_headroom_gib_required"
        ],
        "peak_allocated_vram_gib_all_ranks": vram[
            "peak_allocated_vram_gib_all_ranks"
        ],
        "peak_reserved_vram_gib_all_ranks": vram[
            "peak_reserved_vram_gib_all_ranks"
        ],
        "minimum_allocated_headroom_gib_all_ranks": vram[
            "minimum_allocated_headroom_gib_all_ranks"
        ],
        "minimum_reserved_headroom_gib_all_ranks": vram[
            "minimum_reserved_headroom_gib_all_ranks"
        ],
        "per_rank": vram["per_rank"],
    }


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetIsolatedNBPreflightError(
            "This preflight must be invoked through torchrun."
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise TargetIsolatedNBPreflightError(
            "Exactly four torchrun ranks are required."
        )
    if local_rank != rank:
        raise TargetIsolatedNBPreflightError(
            "This single-node preflight requires RANK == LOCAL_RANK."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise TargetIsolatedNBPreflightError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise TargetIsolatedNBPreflightError(
            "Exactly four visible CUDA devices are required."
        )
    return rank, local_rank, world_size


def _make_grad_scaler() -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[return-value]


def _remove_stale_receipt(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise TargetIsolatedNBPreflightError(
            "Refusing to replace a symbolic-link preflight receipt."
        )
    if output.exists() and not output.is_file():
        raise TargetIsolatedNBPreflightError(
            "Preflight receipt destination is not a regular file."
        )
    output.unlink(missing_ok=True)
    descriptor = os.open(
        output.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _wrap_target_model_ddp(model: nn.Module) -> DistributedDataParallel:
    """Wrap the target model without migrating its CPU graph arguments."""

    return DistributedDataParallel(
        model,
        # Inputs are already placed explicitly.  Leaving device_ids unset is
        # essential here: DDP otherwise migrates every Tensor kwarg, including
        # the receiver-major edge and geometry tensors that the model streams
        # from CPU in bounded chunks.
        device_ids=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def run_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    """Execute all four-rank gates and atomically publish a passed receipt."""

    global _CONTROL_GROUP
    rank, local_rank, _ = _distributed_identity()
    if args.timeout_seconds < 300:
        raise TargetIsolatedNBPreflightError(
            "--timeout-seconds must be at least 300."
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(args.timeout_seconds)),
    )
    _CONTROL_GROUP = torch.distributed.new_group(
        backend="gloo",
        timeout=timedelta(seconds=int(args.timeout_seconds)),
    )

    (
        paths,
        config_path,
        config,
        overlay_dir,
        cohort_dir,
        graph_dir,
        output,
    ) = _resolve_required_paths(args)
    _synchronized_rank_zero_phase(
        "invalidate_stale_receipt",
        rank=rank,
        group=_CONTROL_GROUP,
        action=lambda: _remove_stale_receipt(output),
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    torch.cuda.manual_seed_all(int(config["seed"]))
    trainer = _section(config, "trainer")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(
        bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )

    bundle = load_so2_nb_data(
        overlay_dir,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    _validate_bundle_binding(config, bundle)
    configuration_gate = _configuration_contract_receipt(
        config=config,
        config_path=config_path,
        bundle=bundle,
    )
    dataset_evidence = _dataset_receipt(bundle)
    overlay_evidence = _overlay_evidence(bundle)
    training_batch = bundle.batches_by_alias[REPRESENTATIVE_TRAINING_ALIAS]
    validation_batch = bundle.batches_by_alias[REPRESENTATIVE_VALIDATION_ALIAS]

    model = _build_model(config, bundle).to(device)
    topology = _model_topology_receipt(model)
    ddp_model = _wrap_target_model_ddp(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scaler = _make_grad_scaler()
    sync_before = _synchronized_state_receipt(
        model,
        phase="after_ddp_initial_broadcast",
    )

    coverage_gate = _synthetic_target_coverage_receipt(config)
    fp32_gate = _full_constant_fp32_nb2_receipt()
    unequal_count_gate = _unequal_count_ddp_weighting_receipt(
        device=device,
        rank=rank,
    )
    no_test_gate = _no_test_artifacts_receipt(config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_gate, gradient_gate = _real_training_shard_update(
        model=model,
        ddp_model=ddp_model,
        optimizer=optimizer,
        scaler=scaler,
        batch=training_batch,
        config=config,
        device=device,
        rank=rank,
    )
    sync_after = _synchronized_state_receipt(
        model,
        phase="after_real_c23_optimizer_update",
    )
    validation_mask_gate, validation_metric_gate = (
        _real_validation_shard_replay(
            model=model,
            ddp_model=ddp_model,
            batch=validation_batch,
            config=config,
            device=device,
            rank=rank,
        )
    )
    invariant_gates = _model_invariant_receipts(model, device=device)
    amp_gate = _amp_fp32_equivalence_receipt(
        model,
        device=device,
        amp_dtype=str(trainer["amp_dtype"]),
    )
    torch.cuda.synchronize(device)
    vram_gate = _peak_vram_receipt(
        device,
        rank=rank,
        local_rank=local_rank,
    )
    loss_decrease_gate = _bounded_loss_decrease_receipt(training_batch)

    def rank_zero_artifact_evidence() -> dict[str, Any]:
        # Disk is gated before writing the bounded temporary checkpoint.
        disk = _disk_space_receipt(output)
        checkpoint = _checkpoint_roundtrip_receipt(
            output_parent=output.parent,
            config=config,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            validation_metric=float(
                validation_metric_gate[
                    "first_pooled_masked_nb2_nll"
                ]
            ),
        )
        return {
            "disk": disk,
            "checkpoint": checkpoint,
            "configuration_sources": _configuration_file_hashes(
                config_path,
                paths.config_root,
            ),
            "code_hashes": _code_file_hashes(),
        }

    rank_zero_evidence = _synchronized_rank_zero_phase(
        "checkpoint_disk_and_hash_evidence",
        rank=rank,
        group=_CONTROL_GROUP,
        action=rank_zero_artifact_evidence,
    )
    if not isinstance(rank_zero_evidence, Mapping):
        raise TargetIsolatedNBPreflightError(
            "Rank-zero artifact evidence is malformed."
        )
    checkpoint_gate = rank_zero_evidence.get("checkpoint")
    disk_gate = rank_zero_evidence.get("disk")
    configuration_sources = rank_zero_evidence.get("configuration_sources")
    code_hashes = rank_zero_evidence.get("code_hashes")
    if not all(
        isinstance(value, Mapping)
        for value in (
            checkpoint_gate,
            disk_gate,
            configuration_sources,
            code_hashes,
        )
    ):
        raise TargetIsolatedNBPreflightError(
            "Rank-zero receipt evidence broadcast failed."
        )

    gpu_gate = _gpu_inventory_receipt(
        device,
        rank=rank,
        local_rank=local_rank,
    )
    ddp_sync_gate = {
        "passed": True,
        "before_optimizer_update": sync_before,
        "after_optimizer_update": sync_after,
    }
    gates: dict[str, Mapping[str, Any]] = {
        "configuration_and_contract": configuration_gate,
        "model_topology": topology,
        "synthetic_target_coverage": coverage_gate,
        "unequal_count_ddp_weighting": unequal_count_gate,
        "fixed_validation_mask_replay": validation_mask_gate,
        "fixed_validation_metric_replay": validation_metric_gate,
        "full_constant_fp32_nb2": fp32_gate,
        "amp_fp32_equivalence": amp_gate,
        "real_train_shard_forward_backward": training_gate,
        "preclip_gradient_flow": gradient_gate,
        "bounded_nb2_loss_decrease": loss_decrease_gate,
        "checkpoint_roundtrip": checkpoint_gate,
        "ddp_parameter_sync": ddp_sync_gate,
        "gpu_inventory": gpu_gate,
        "peak_vram": vram_gate,
        "disk_space": disk_gate,
        "no_test_split_or_artifacts": no_test_gate,
        **invariant_gates,
    }
    if set(gates) != REQUIRED_GATE_NAMES or any(
        gate.get("passed") is not True for gate in gates.values()
    ):
        raise TargetIsolatedNBPreflightError(
            "Internal required-gate assembly is incomplete."
        )

    gpu_identities = list(gpu_gate["devices"])
    resources = _resource_evidence(disk_gate, vram_gate)
    unsigned_receipt: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "diagnostic_only": True,
        "created_at": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "world_size": WORLD_SIZE,
        "distributed_backend": "nccl",
        "control_plane_backend": "gloo",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "configuration_sha256": canonical_sha256(config),
        "resolved_config_sha256": canonical_sha256(config),
        "configuration_file_sha256": sha256_file(config_path),
        "configuration_source_file_sha256": dict(configuration_sources),
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "source_artifacts": overlay_evidence["source_artifacts"],
        "source_artifacts_sha256": canonical_sha256(
            overlay_evidence["source_artifacts"]
        ),
        "code_file_sha256": dict(code_hashes),
        "code_hash_scope": CODE_HASH_SCOPE,
        "gpu_identities": gpu_identities,
        "gpu_identities_sha256": canonical_sha256(gpu_identities),
        "dataset": dataset_evidence,
        "dataset_evidence_sha256": canonical_sha256(dataset_evidence),
        "model": _public_model_evidence(topology),
        "resource_gates": resources,
        "gates": gates,
    }
    unsigned_receipt["receipt_content_sha256"] = canonical_sha256(
        unsigned_receipt
    )
    def publish_receipt() -> Mapping[str, Any]:
        validate_preflight_receipt(unsigned_receipt)
        _atomic_json(output, unsigned_receipt)
        written = json.loads(output.read_text(encoding="utf-8"))
        validate_preflight_receipt(written)
        return written

    receipt = _synchronized_rank_zero_phase(
        "publish_checksum_bound_receipt",
        rank=rank,
        group=_CONTROL_GROUP,
        action=publish_receipt,
    )
    if not isinstance(receipt, Mapping):
        raise TargetIsolatedNBPreflightError(
            "Published receipt broadcast failed."
        )
    validate_preflight_receipt(receipt)
    torch.distributed.barrier(group=_CONTROL_GROUP)
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "receipt": str(output),
                    "receipt_content_sha256": receipt[
                        "receipt_content_sha256"
                    ],
                    "representative_training_alias": (
                        REPRESENTATIVE_TRAINING_ALIAS
                    ),
                    "representative_validation_alias": (
                        REPRESENTATIVE_VALIDATION_ALIAS
                    ),
                    "peak_reserved_vram_gib_all_ranks": resources[
                        "peak_reserved_vram_gib_all_ranks"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG,
        help="Frozen source experiment YAML (overrides are fail-closed).",
    )
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--cohort-dir", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7_200,
        help="Per-collective timeout for the real full-core gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_preflight(args)
        return 0
    except BaseException:
        rank = os.environ.get("RANK", "unknown")
        print(
            f"target-isolated NB2 preflight failed on rank {rank}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        global _CONTROL_GROUP
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            finally:
                _CONTROL_GROUP = None


if __name__ == "__main__":
    raise SystemExit(main())

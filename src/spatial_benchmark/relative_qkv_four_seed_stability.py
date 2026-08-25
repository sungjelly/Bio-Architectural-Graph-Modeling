"""Compact, checksum-bound four-seed stability analysis for relative QKV models.

The production entry point validates four immutable plateau-complete run bundles
and their strict checkpoint receipts before allocating CUDA memory.  It then
replays one model and one complete core at a time on the common held-in mask.
Only fixed receiver probes, full node embeddings, compact mutual-pair candidates,
and 24 prespecified derivatives are retained.

This is a transductive, held-in analysis.  Its outputs describe model routing,
representation agreement, and local model sensitivity.  They do not establish
direct signaling, biological mechanism, or causality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv
import ctypes
import errno
import gc
import hashlib
import hmac
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .cancer_pooled_full_core import CANCER_ALIASES
from .fingerprints import sha256_file
from .pooled_relative_qkv_training import (
    PLATEAU_FIRST_ALLOWED_STOP_EPOCH,
    PooledRelativeQKVCoreBatch,
    epoch_boundary_resume_from_checkpoint,
)
from .relative_qkv_checkpoint_verification import (
    DEFAULT_ATTENTION_ATOL,
    DEFAULT_ATTENTION_RTOL,
    DEFAULT_PREDICTION_ATOL,
    DEFAULT_PREDICTION_RTOL,
    VERIFICATION_SCHEMA,
)
from .relative_qkv_gradient_requests import (
    EXPECTED_REQUEST_COUNT as LOCKED_GRADIENT_REQUEST_COUNT,
    GRADIENT_REQUEST_SCHEMA,
    RADIAL_SHELLS,
    REQUEST_CSV_FIELDS,
    LockedGradientRequest,
    group_selected_derivative_requests,
    load_and_verify_locked_gradient_requests,
    load_prepared_gradient_request_inputs,
    verify_locked_gradient_protocol,
)
from .relative_qkv_post_training import (
    CAMPAIGN_ID,
    FIXED_INFERENCE_MASK_BASE_SEED,
    FIXED_INFERENCE_MASK_NAMESPACE,
    SelectedDerivativeRequest,
    fixed_inference_mask,
    load_prepared_relative_qkv_batches,
    load_relative_qkv_checkpoint,
    selected_autograd_derivatives,
    stream_receiver_attention,
)
from .relative_qkv_stability import (
    FixedSeedArray,
    RelationshipEnsembleSummary,
    attention_head_signature,
    content_position_contribution_agreement,
    embedding_stability,
    match_attention_heads,
    matched_head_attention_stability,
    positional_bias_response_stability,
    selected_gradient_stability,
    summarize_relationship_ensemble,
)
from .run_archive import verify_run_bundle
from .training import set_deterministic_seed


ANALYSIS_SCHEMA = "cancer_6core_relative_qkv_four_seed_stability_v1"
MANIFEST_SCHEMA = "cancer_6core_relative_qkv_four_seed_stability_manifest_v1"
ACTIVE_SEEDS = (0, 1, 2, 3)
DEFERRED_SEEDS = (4,)
EXPECTED_SEED_COUNT = 4
REFERENCE_SEED = 0
RECEIVER_PROBES_PER_CORE = 64
HEAD_TOP_EDGE_FRACTION = 0.05
MUTUAL_TOP_K_PER_CORE_SEED = 100
QUANTILE_LEVELS = (0.05, 0.25, 0.75, 0.95)
ANALYSIS_DETERMINISTIC_SEED = 2026082497
LOCKED_ANALYSIS_PROTOCOL_SHA256 = (
    "ec7c57df87083a54a81a661e50319fdb614993992701f48a620efaa80622696e"
)
LOCKED_GRADIENT_REQUEST_TABLE_SHA256 = (
    "2ebe0d5fbf59f268d8b60d9b634a4ecf5d2621083ee05257eabb568af0bec4d2"
)
FINAL_LAYER = -1
EXPECTED_GRADIENT_REQUEST_COUNT = 24
DEFAULT_RECEIVER_CHUNK_SIZE = 512
DEFAULT_MAX_EDGES_PER_CHUNK = 200_000
LOCKED_PARAMETER_COUNT = 5_003_016
LOCKED_NODE_COUNT = 117_996
LOCKED_GENE_COUNT = 1000
LOCKED_HIDDEN_DIM = 256
LOCKED_HEAD_COUNT = 8
LOCKED_FIXED_METRIC_NAMES = (
    "fit/uniform_per_cell/masked_huber",
    "fit/uniform_per_cell/masked_mae",
    "fit/uniform_per_cell/masked_mse",
    "fit/uniform_per_cell/masked_r2",
)
ANALYSIS_SOURCE_FILES = (
    "src/spatial_benchmark/relative_qkv_four_seed_stability.py",
    "src/spatial_benchmark/relative_qkv_gradient_requests.py",
    "src/spatial_benchmark/relative_qkv_graph_transformer.py",
    "src/spatial_benchmark/relative_qkv_post_training.py",
    "src/spatial_benchmark/relative_qkv_stability.py",
    "scripts/analysis/run_relative_qkv_four_seed_stability.py",
    "scripts/analysis/prepare_relative_qkv_gradient_requests.py",
)
LOCKED_MODEL_CONSTRUCTION: Mapping[str, Any] = {
    "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
    "num_genes": 1000,
    "node_covariate_dim": 22,
    "activation_checkpointing": True,
    "attention_dropout": 0.0,
    "attention_head_dim": 32,
    "attention_heads": 8,
    "decoder_dim": 1024,
    "dropout": 0.1,
    "edge_key_vectors": False,
    "edge_value_gates": False,
    "edge_value_vectors": False,
    "embedding_dim": 256,
    "exact_receiver_partitioning": True,
    "family": "relative_geometry_qkv_graph_transformer",
    "ffn_dim": 1024,
    "fp32_attention_accumulation": True,
    "graph_layers": 4,
    "hidden_dim": 256,
    "implicit_self_loops": False,
    "max_edges_per_chunk": DEFAULT_MAX_EDGES_PER_CHUNK,
    "name": "relative-qkv-gat",
    "positional_bias_final_zero_init": True,
    "positional_bias_hidden_dim": 128,
    "receiver_chunk_size": DEFAULT_RECEIVER_CHUNK_SIZE,
    "relative_geometry_dim": 70,
    "relative_geometry_role": "attention_logit_bias_only",
    "trainable_edge_identifiers": False,
    "trainable_node_identifiers": False,
    "uses_edge_inputs": False,
    "uses_graph_inputs": True,
    "uses_relative_position": True,
}


class FourSeedStabilityError(RuntimeError):
    """Raised when a four-seed analysis input or output contract is violated."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for content checksums."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(b"relative-qkv-four-seed-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FourSeedStabilityError(f"{location} must be a mapping.")
    return value


def _finite_float(value: Any, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FourSeedStabilityError(f"{location} must be numeric.") from exc
    if not math.isfinite(number):
        raise FourSeedStabilityError(f"{location} must be finite.")
    return number


def parse_seed_path_specs(
    values: Sequence[str],
    *,
    label: str,
) -> dict[int, Path]:
    """Parse exactly ``0=path`` through ``3=path`` without silent replacement."""

    parsed: dict[int, Path] = {}
    for raw in values:
        seed_text, separator, path_text = str(raw).partition("=")
        if not separator or not seed_text or not path_text:
            raise FourSeedStabilityError(
                f"Every {label} must use SEED=PATH syntax."
            )
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise FourSeedStabilityError(f"{label} seed must be an integer.") from exc
        if seed in parsed:
            raise FourSeedStabilityError(f"Duplicate {label} seed {seed}.")
        parsed[seed] = Path(path_text).expanduser().resolve(strict=False)
    if tuple(sorted(parsed)) != ACTIVE_SEEDS:
        raise FourSeedStabilityError(
            f"{label} inputs must contain exactly seeds {list(ACTIVE_SEEDS)}."
        )
    return parsed


def evenly_spaced_receivers(n_nodes: int, count: int = RECEIVER_PROBES_PER_CORE) -> np.ndarray:
    """Return deterministic, endpoint-inclusive receiver probes."""

    if isinstance(n_nodes, bool) or int(n_nodes) <= 0:
        raise FourSeedStabilityError("n_nodes must be positive.")
    if isinstance(count, bool) or int(count) <= 0 or int(count) > int(n_nodes):
        raise FourSeedStabilityError("receiver probe count must be in [1, n_nodes].")
    if int(count) == 1:
        receivers = np.zeros(1, dtype=np.int64)
    else:
        # Match the locked protocol literally. ``np.linspace(...,
        # dtype=int)`` rounds during float-to-integer conversion and can
        # disagree with floor(q * (n - 1) / (count - 1)).
        receivers = (
            np.arange(int(count), dtype=np.int64) * (int(n_nodes) - 1)
        ) // (int(count) - 1)
    if len(receivers) != int(count) or len(np.unique(receivers)) != int(count):
        raise FourSeedStabilityError(
            "Evenly spaced receiver selection did not produce the exact count."
        )
    receivers.setflags(write=False)
    return receivers


def receiver_centered_logits(
    values: np.ndarray,
    receiver_group_ids: Sequence[object],
) -> np.ndarray:
    """Remove receiver/head constants that are null under receiver softmax."""

    logits = np.asarray(values, dtype=np.float64)
    groups = np.asarray(tuple(receiver_group_ids))
    if (
        logits.ndim != 2
        or groups.ndim != 1
        or len(groups) != logits.shape[0]
        or logits.shape[0] == 0
        or not np.isfinite(logits).all()
    ):
        raise FourSeedStabilityError(
            "Logits and receiver groups must be finite and edge aligned."
        )
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    sums = np.zeros((group_count, logits.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, logits)
    counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
    centered = logits - sums[inverse] / counts[inverse, None]
    return np.ascontiguousarray(centered)


@dataclass(frozen=True)
class MutualPairScores:
    """Canonical reciprocal pair scores in graph-stable pair-position order."""

    canonical_edge_ids: np.ndarray = field(repr=False)
    pair_keys: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)


def vectorized_mutual_pair_scores(
    edge_index: np.ndarray | torch.Tensor,
    directional_routing: np.ndarray,
    *,
    n_nodes: int,
) -> MutualPairScores:
    """Calculate reciprocal minima once per undirected pair.

    ``directional_routing`` is ``receiver_in_degree * mean_head_attention`` in
    directed edge order.  Prepared edges must be unique, loop-free, reciprocal,
    and receiver-major/source-major sorted.  No Python edge dictionary is built.
    """

    edges = (
        edge_index.detach().cpu().numpy()
        if isinstance(edge_index, torch.Tensor)
        else np.asarray(edge_index)
    )
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or edges.dtype.kind not in "iu"
        or isinstance(n_nodes, bool)
        or int(n_nodes) <= 0
    ):
        raise FourSeedStabilityError(
            "edge_index must be integral [2, edges] with positive n_nodes."
        )
    source = edges[0].astype(np.int64, copy=False)
    receiver = edges[1].astype(np.int64, copy=False)
    routing = np.asarray(directional_routing, dtype=np.float32)
    if (
        routing.ndim != 1
        or len(routing) != len(source)
        or not np.isfinite(routing).all()
        or np.any(routing < 0.0)
    ):
        raise FourSeedStabilityError(
            "directional_routing must be finite, non-negative, and edge aligned."
        )
    if len(source) == 0 or np.any(source == receiver):
        raise FourSeedStabilityError("Mutual routing requires loop-free edges.")
    directed_codes = receiver * int(n_nodes) + source
    if np.any(directed_codes[1:] <= directed_codes[:-1]):
        raise FourSeedStabilityError(
            "Edges must be unique receiver-major/source-major sorted."
        )
    canonical_ids = np.flatnonzero(source < receiver).astype(np.int64, copy=False)
    if len(canonical_ids) * 2 != len(source):
        raise FourSeedStabilityError(
            "Graph does not contain exactly two directions per reciprocal pair."
        )
    reverse_codes = (
        source[canonical_ids] * int(n_nodes) + receiver[canonical_ids]
    )
    reciprocal_ids = np.searchsorted(directed_codes, reverse_codes)
    if np.any(reciprocal_ids >= len(directed_codes)) or not np.array_equal(
        directed_codes[reciprocal_ids], reverse_codes
    ):
        raise FourSeedStabilityError("Graph is missing a reciprocal directed edge.")
    pair_keys = (
        source[canonical_ids] * int(n_nodes) + receiver[canonical_ids]
    ).astype(np.int64, copy=False)
    if len(np.unique(pair_keys)) != len(pair_keys):
        raise FourSeedStabilityError("Canonical reciprocal pair keys are duplicated.")
    scores = np.minimum(
        routing[canonical_ids],
        routing[reciprocal_ids],
    ).astype(np.float32, copy=False)
    return MutualPairScores(
        canonical_edge_ids=np.ascontiguousarray(canonical_ids),
        pair_keys=np.ascontiguousarray(pair_keys),
        scores=np.ascontiguousarray(scores),
    )


def top_mutual_pair_positions(
    scores: np.ndarray,
    pair_keys: np.ndarray,
    *,
    top_k: int = MUTUAL_TOP_K_PER_CORE_SEED,
) -> np.ndarray:
    """Choose deterministic top pair positions, tie-broken by canonical key."""

    values = np.asarray(scores, dtype=np.float64)
    keys = np.asarray(pair_keys, dtype=np.int64)
    if (
        values.ndim != 1
        or keys.ndim != 1
        or values.shape != keys.shape
        or not np.isfinite(values).all()
        or len(np.unique(keys)) != len(keys)
    ):
        raise FourSeedStabilityError("Mutual scores and pair keys must align.")
    if isinstance(top_k, bool) or not 1 <= int(top_k) <= len(values):
        raise FourSeedStabilityError("top_k is outside the available pair count.")
    order = np.lexsort((keys, -values))[: int(top_k)]
    return np.asarray(order, dtype=np.int64)


def scalar_summary(
    values: Sequence[float],
    *,
    quantile_levels: Sequence[float] = QUANTILE_LEVELS,
) -> dict[str, Any]:
    """Return finite four-member descriptive ensemble-spread statistics."""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (EXPECTED_SEED_COUNT,) or not np.isfinite(array).all():
        raise FourSeedStabilityError("Summary requires exactly four finite values.")
    levels = np.asarray(tuple(quantile_levels), dtype=np.float64)
    if (
        levels.ndim != 1
        or len(levels) == 0
        or np.any(levels < 0.0)
        or np.any(levels > 1.0)
        or np.any(levels[1:] <= levels[:-1])
    ):
        raise FourSeedStabilityError("Quantiles must increase strictly within [0,1].")
    quantiles = np.quantile(array, levels, method="linear")
    return {
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
        "empirical_quantiles": {
            f"{level:g}": float(value)
            for level, value in zip(levels.tolist(), quantiles.tolist(), strict=True)
        },
        "ensemble_member_count": EXPECTED_SEED_COUNT,
        "spread_label": "four-seed ensemble spread",
        "calibrated_confidence_interval": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with sorted names and fixed ZIP metadata."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite NPZ: {destination}")
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise FourSeedStabilityError(f"Invalid NPZ array name: {name!r}.")
            array = np.ascontiguousarray(np.asarray(arrays[name]))
            if array.dtype.hasobject:
                raise FourSeedStabilityError("NPZ arrays may not use object dtype.")
            stream = BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


@dataclass(frozen=True)
class ValidatedMember:
    seed: int
    run_id: str
    run_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    receipt_path: Path
    receipt_file_sha256: str
    receipt_content_sha256: str
    completed_global_epochs: int
    loss_curve: np.ndarray = field(repr=False)
    fixed_metrics: Mapping[str, float]
    fixed_masks: Mapping[str, Mapping[str, Any]]
    model_state_sha256: str
    history_sha256: str
    parameter_count: int
    model_construction_sha256: str
    mask_base_seed: int
    core_order_seed: int
    training_schedule_epoch_sha256: tuple[str, ...] = field(repr=False)


def _load_json(path: Path, location: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FourSeedStabilityError(f"Cannot read {location}: {path}.") from exc
    return _mapping(value, location)


def _validate_receipt_content(receipt: Mapping[str, Any], *, seed: int) -> str:
    content = dict(receipt)
    observed = str(content.pop("receipt_content_sha256", ""))
    expected = canonical_sha256(content)
    if not hmac.compare_digest(observed, expected):
        raise FourSeedStabilityError(
            f"Seed {seed} strict receipt content checksum is invalid."
        )
    if receipt.get("schema") != VERIFICATION_SCHEMA or receipt.get("status") != "passed":
        raise FourSeedStabilityError(f"Seed {seed} strict receipt did not pass.")
    if receipt.get("campaign_id") != CAMPAIGN_ID or receipt.get("model_seed") != seed:
        raise FourSeedStabilityError(f"Seed {seed} strict receipt identity drifted.")
    plateau = _mapping(receipt.get("plateau_verification"), "plateau verification")
    decision = _mapping(plateau.get("recomputed_decision"), "plateau decision")
    if (
        plateau.get("status") != "passed"
        or decision.get("should_stop") is not True
        or decision.get("model_seed") != seed
        or decision.get("validation_or_test_metric") is not False
        or decision.get("checkpoint_selection_metric") is not False
    ):
        raise FourSeedStabilityError(f"Seed {seed} final plateau gate is invalid.")
    execution = _mapping(receipt.get("execution"), "strict receipt execution")
    if (
        execution.get("deterministic") is not True
        or execution.get("deterministic_warn_only") is not False
        or execution.get("deterministic_seed") != seed
        or execution.get("deterministic_algorithms_enabled") is not True
        or execution.get("deterministic_algorithms_warn_only_enabled") is not False
        or execution.get("cublas_workspace_config") not in {":4096:8", ":16:8"}
    ):
        raise FourSeedStabilityError(
            f"Seed {seed} strict receipt lacks deterministic execution evidence."
        )
    if receipt.get("generalization_claim_supported") is not False or receipt.get(
        "causal_claim_supported"
    ) is not False:
        raise FourSeedStabilityError(f"Seed {seed} strict receipt claim scope drifted.")
    return observed


def _validate_exact_replay_array(
    record: Mapping[str, Any],
    *,
    location: str,
    atol: float,
    rtol: float,
    expected_shape: tuple[int, ...],
) -> None:
    raw_shape = record.get("shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or not raw_shape
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_shape
        )
    ):
        raise FourSeedStabilityError(f"{location} shape is invalid.")
    shape = tuple(int(value) for value in raw_shape)
    first_sha256 = record.get("first_sha256")
    second_sha256 = record.get("second_sha256")
    valid_sha256 = lambda value: (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if (
        record.get("byte_identical") is not True
        or record.get("within_tolerance") is not True
        or record.get("dtype") != "float32"
        or shape != expected_shape
        or any(dimension <= 0 for dimension in shape)
        or int(record.get("element_count", 0)) != math.prod(shape)
        or int(record.get("failing_element_count", -1)) != 0
        or not valid_sha256(first_sha256)
        or not valid_sha256(second_sha256)
        or not hmac.compare_digest(first_sha256, second_sha256)
        or not math.isclose(
            _finite_float(record.get("maximum_absolute_difference"), location),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_float(record.get("mean_absolute_difference"), location),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_float(record.get("atol"), location),
            float(atol),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_float(record.get("rtol"), location),
            float(rtol),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise FourSeedStabilityError(f"{location} is not an exact strict replay.")


def _validate_core_replay_receipt(
    row: Mapping[str, Any],
    *,
    seed: int,
    alias: str,
) -> None:
    n_nodes = int(row.get("n_nodes", 0))
    if n_nodes <= 0:
        raise FourSeedStabilityError(f"Seed {seed} {alias} node count is invalid.")
    _validate_exact_replay_array(
        _mapping(row.get("prediction_replay"), f"seed {seed} {alias} prediction"),
        location=f"seed {seed} {alias} prediction",
        atol=DEFAULT_PREDICTION_ATOL,
        rtol=DEFAULT_PREDICTION_RTOL,
        expected_shape=(n_nodes, 1000),
    )
    attention = _mapping(
        row.get("selected_attention_replay"),
        f"seed {seed} {alias} selected attention",
    )
    selected_directed_edges = int(attention.get("selected_directed_edges", 0))
    selected_receivers = tuple(attention.get("selected_receivers", ()))
    edge_sha256 = attention.get("edge_index_sha256")
    reload_edge_sha256 = attention.get("edge_index_reload_sha256")
    valid_sha256 = lambda value: (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if (
        not valid_sha256(edge_sha256)
        or not valid_sha256(reload_edge_sha256)
        or not hmac.compare_digest(edge_sha256, reload_edge_sha256)
        or int(attention.get("attention_heads", 0)) != 8
        or selected_directed_edges <= 0
        or len(selected_receivers) != 4
        or any(
            isinstance(receiver, bool)
            or not isinstance(receiver, (int, np.integer))
            for receiver in selected_receivers
        )
        or len(set(int(receiver) for receiver in selected_receivers)) != 4
        or any(
            int(receiver) < 0 or int(receiver) >= n_nodes
            for receiver in selected_receivers
        )
        or _finite_float(
            attention.get("maximum_logit_composition_error"),
            f"seed {seed} {alias} logit composition",
        )
        != 0.0
        or _finite_float(
            attention.get("maximum_receiver_head_normalization_error"),
            f"seed {seed} {alias} attention normalization",
        )
        > 1e-6
    ):
        raise FourSeedStabilityError(
            f"Seed {seed} {alias} selected-attention replay drifted."
        )
    channels = _mapping(
        attention.get("channels"),
        f"seed {seed} {alias} selected-attention channels",
    )
    expected_channels = {"attention", "content", "positional_bias", "combined"}
    if set(channels) != expected_channels:
        raise FourSeedStabilityError(
            f"Seed {seed} {alias} selected-attention channels drifted."
        )
    for name in sorted(expected_channels):
        _validate_exact_replay_array(
            _mapping(channels[name], f"seed {seed} {alias} {name}"),
            location=f"seed {seed} {alias} {name}",
            atol=DEFAULT_ATTENTION_ATOL,
            rtol=DEFAULT_ATTENTION_RTOL,
            expected_shape=(selected_directed_edges, 8),
        )


def _load_loss_curve(run_root: Path, *, run_id: str, completed: int) -> np.ndarray:
    table_path = run_root / "metrics/history.parquet"
    try:
        table = pq.read_table(
            table_path,
            columns=[
                "run_id",
                "global_epoch",
                "completed_global_epochs",
                "equal_core_mean_masked_huber",
            ],
        )
    except (OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise FourSeedStabilityError(
            f"Cannot read immutable training history: {table_path}."
        ) from exc
    rows = table.to_pylist()
    if len(rows) != completed:
        raise FourSeedStabilityError("Training history length does not match checkpoint.")
    losses: list[float] = []
    for index, row in enumerate(rows):
        if (
            row.get("run_id") != run_id
            or int(row.get("global_epoch")) != index
            or int(row.get("completed_global_epochs")) != index + 1
        ):
            raise FourSeedStabilityError("Training history identity/order drifted.")
        losses.append(
            _finite_float(
                row.get("equal_core_mean_masked_huber"),
                f"training history epoch {index}",
            )
        )
    return np.asarray(losses, dtype=np.float64)


def _training_schedule_epoch_hashes(resume: Any) -> tuple[str, ...]:
    """Hash only model-seed-independent order and ten-view mask receipts."""

    core_by_epoch: dict[int, list[Any]] = {}
    for record in resume.core_history:
        core_by_epoch.setdefault(int(record.global_epoch), []).append(record)
    hashes: list[str] = []
    for epoch in resume.global_history:
        epoch_index = int(epoch.global_epoch)
        core_records = sorted(
            core_by_epoch.get(epoch_index, ()),
            key=lambda value: int(value.step_in_epoch),
        )
        if len(core_records) != len(CANCER_ALIASES):
            raise FourSeedStabilityError(
                f"Training schedule epoch {epoch_index} lacks six core steps."
            )
        payload = {
            "global_epoch": epoch_index,
            "ordered_aliases": list(epoch.ordered_aliases),
            "core_steps": [
                {
                    "step_in_epoch": int(core.step_in_epoch),
                    "optimizer_step": int(core.optimizer_step),
                    "alias": str(core.alias),
                    "n_nodes": int(core.n_nodes),
                    "n_edges": int(core.n_edges),
                    "n_mask_views": int(core.n_mask_views),
                    "n_masked_entries_across_views": int(
                        core.n_masked_entries_across_views
                    ),
                    "mask_views": [
                        {
                            "view_index": int(view.view_index),
                            "initial_mask_seed": int(view.initial_mask_seed),
                            "effective_mask_seed": int(view.effective_mask_seed),
                            "zero_total_mask_resamples": int(
                                view.zero_total_mask_resamples
                            ),
                            "mask_checksum_sha256": str(view.mask_checksum_sha256),
                            "n_masked_entries": int(view.n_masked_entries),
                            "masked_count_min": int(view.masked_count_min),
                            "masked_count_mean": float(view.masked_count_mean),
                            "masked_count_median": float(view.masked_count_median),
                            "masked_count_max": int(view.masked_count_max),
                            "zero_mask_cells": int(view.zero_mask_cells),
                            "full_mask_cells": int(view.full_mask_cells),
                        }
                        for view in core.mask_views
                    ],
                }
                for core in core_records
            ],
        }
        hashes.append(canonical_sha256(payload))
    if len(hashes) != int(resume.completed_global_epochs):
        raise FourSeedStabilityError("Training schedule epoch count drifted.")
    return tuple(hashes)


def validate_member(
    *,
    seed: int,
    run_root: Path,
    receipt_path: Path,
    cohort_manifest_sha256: str,
    graph_manifest_sha256: str,
) -> ValidatedMember:
    """Fail closed on one immutable run, checkpoint, and strict receipt."""

    if seed not in ACTIVE_SEEDS:
        raise FourSeedStabilityError(f"Unexpected active model seed {seed}.")
    root = run_root.expanduser().resolve(strict=True)
    if root.name == "scratch" or "active_runs" in root.parts:
        raise FourSeedStabilityError("Stability analysis requires a published run bundle.")
    bundle = verify_run_bundle(root)
    if bundle.get("valid") is not True or bundle.get("status") != "success":
        raise FourSeedStabilityError(f"Seed {seed} immutable run bundle is not valid.")
    run_id = root.name
    receipt_source = receipt_path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_source, f"seed {seed} strict receipt")
    receipt_content_sha256 = _validate_receipt_content(receipt, seed=seed)
    if receipt.get("run_id") != run_id:
        raise FourSeedStabilityError(f"Seed {seed} receipt/run ID mismatch.")
    checkpoint_path = (root / "checkpoints/last.ckpt").resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = _mapping(receipt.get("checkpoint"), "strict receipt checkpoint")
    if (
        Path(str(checkpoint.get("path"))).resolve(strict=True) != checkpoint_path
        or checkpoint.get("file_sha256") != checkpoint_sha256
        or checkpoint.get("independently_reloadable") is not True
        or int(checkpoint.get("independent_reload_count", 0)) < 2
    ):
        raise FourSeedStabilityError(f"Seed {seed} checkpoint receipt drifted.")
    completed = int(checkpoint.get("completed_global_epochs"))
    decision = _mapping(
        _mapping(receipt.get("plateau_verification"), "plateau verification").get(
            "recomputed_decision"
        ),
        "plateau decision",
    )
    if (
        int(decision.get("final_epoch")) != completed
        or completed < PLATEAU_FIRST_ALLOWED_STOP_EPOCH
        or completed % 25 != 0
    ):
        raise FourSeedStabilityError(f"Seed {seed} plateau epoch/checkpoint mismatch.")
    prepared = _mapping(receipt.get("prepared_inputs"), "strict prepared inputs")
    if (
        tuple(prepared.get("core_aliases", ())) != CANCER_ALIASES
        or prepared.get("cohort_manifest_file_sha256") != cohort_manifest_sha256
        or prepared.get("graph_manifest_file_sha256") != graph_manifest_sha256
        or prepared.get("raw_coordinates_are_model_inputs") is not False
    ):
        raise FourSeedStabilityError(f"Seed {seed} fixed prepared inputs drifted.")
    cross_checks = _mapping(receipt.get("artifact_cross_checks"), "artifact checks")
    expected_cross_check_paths = {
        "held_in_fit_metrics_by_core": (
            root / "diagnostics/held_in_fit_metrics_by_core.json"
        ),
        "final_metrics": root / "metrics/final.json",
        "training_provenance": root / "provenance/relative_qkv_training.json",
    }
    for name, expected_path in expected_cross_check_paths.items():
        record = _mapping(cross_checks.get(name), name)
        resolved_expected = expected_path.resolve(strict=True)
        try:
            resolved_record = Path(str(record.get("path"))).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FourSeedStabilityError(
                f"Seed {seed} {name} cross-check path is invalid."
            ) from exc
        if (
            record.get("status") != "passed"
            or resolved_record != resolved_expected
            or record.get("file_sha256") != sha256_file(resolved_expected)
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} {name} cross-check is absent or checksum-drifted."
            )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise FourSeedStabilityError(f"Cannot inspect seed {seed} checkpoint.") from exc
    payload = _mapping(payload, "checkpoint payload")
    if (
        payload.get("run_id") != run_id
        or payload.get("model_seed") != seed
        or int(payload.get("completed_global_epochs")) != completed
        or payload.get("model_state_checksum") != checkpoint.get("model_state_sha256")
        or payload.get("history_checksum") != checkpoint.get("history_sha256")
    ):
        raise FourSeedStabilityError(f"Seed {seed} checkpoint payload drifted.")
    try:
        resume = epoch_boundary_resume_from_checkpoint(payload)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise FourSeedStabilityError(
            f"Seed {seed} checkpoint resume/history checksum validation failed."
        ) from exc
    model_construction = dict(
        _mapping(payload.get("model_construction"), "checkpoint model construction")
    )
    parameter_count = int(payload.get("parameter_count", 0))
    if model_construction != dict(LOCKED_MODEL_CONSTRUCTION) or parameter_count != (
        LOCKED_PARAMETER_COUNT
    ):
        raise FourSeedStabilityError(
            f"Seed {seed} does not use the locked production architecture."
        )
    schedule_epoch_sha256 = _training_schedule_epoch_hashes(resume)
    checkpoint_loss_curve = np.asarray(
        [
            record.equal_core_mean_masked_huber
            for record in resume.global_history
        ],
        dtype=np.float64,
    )
    held_in_replay = _mapping(receipt.get("held_in_fit_replay"), "held-in fit replay")
    if (
        held_in_replay.get("status") != "passed"
        or held_in_replay.get("role")
        != "held_in_fit_diagnostic_not_validation_or_test"
        or held_in_replay.get("fixed_masks_identical_across_reloads") is not True
    ):
        raise FourSeedStabilityError(f"Seed {seed} held-in replay scope drifted.")
    raw_per_core = held_in_replay.get("per_core")
    if not isinstance(raw_per_core, list) or len(raw_per_core) != len(CANCER_ALIASES):
        raise FourSeedStabilityError(f"Seed {seed} held-in core receipt drifted.")
    fixed_masks: dict[str, Mapping[str, Any]] = {}
    for alias, raw_row in zip(CANCER_ALIASES, raw_per_core, strict=True):
        row = _mapping(raw_row, f"seed {seed} held-in core {alias}")
        if row.get("alias") != alias:
            raise FourSeedStabilityError(f"Seed {seed} held-in aliases drifted.")
        _validate_core_replay_receipt(row, seed=seed, alias=alias)
        fixed_masks[alias] = {
            "seed": int(row.get("mask_seed")),
            "checksum_sha256": str(row.get("mask_checksum")),
            "masked_entry_count": int(row.get("n_masked_entries")),
        }
    fixed_metrics = _mapping(
        held_in_replay.get("equal_core_metrics"),
        "equal-core fixed metrics",
    )
    expected_metric_names = LOCKED_FIXED_METRIC_NAMES
    if set(fixed_metrics) != set(expected_metric_names):
        raise FourSeedStabilityError(f"Seed {seed} fixed metric schema drifted.")
    metrics = {
        name: _finite_float(fixed_metrics[name], f"seed {seed} {name}")
        for name in expected_metric_names
    }
    loss_curve = _load_loss_curve(root, run_id=run_id, completed=completed)
    if not np.array_equal(loss_curve, checkpoint_loss_curve):
        raise FourSeedStabilityError(
            f"Seed {seed} archived and checkpoint training curves drifted."
        )
    if not math.isclose(
        float(loss_curve[-1]),
        _finite_float(
            _load_json(root / "metrics/final.json", "final metrics").get(
                "fit/training/final_equal_core_masked_huber"
            ),
            "final training loss",
        ),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise FourSeedStabilityError(f"Seed {seed} final training loss drifted.")
    return ValidatedMember(
        seed=seed,
        run_id=run_id,
        run_root=root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        receipt_path=receipt_source,
        receipt_file_sha256=sha256_file(receipt_source),
        receipt_content_sha256=receipt_content_sha256,
        completed_global_epochs=completed,
        loss_curve=loss_curve,
        fixed_metrics=metrics,
        fixed_masks=fixed_masks,
        model_state_sha256=str(checkpoint["model_state_sha256"]),
        history_sha256=str(checkpoint["history_sha256"]),
        parameter_count=parameter_count,
        model_construction_sha256=canonical_sha256(model_construction),
        mask_base_seed=int(resume.mask_base_seed),
        core_order_seed=int(resume.core_order_seed),
        training_schedule_epoch_sha256=schedule_epoch_sha256,
    )


def validate_four_members(
    run_roots: Mapping[int, Path],
    receipt_paths: Mapping[int, Path],
    *,
    cohort_manifest_path: Path,
    graph_manifest_path: Path,
) -> tuple[ValidatedMember, ...]:
    if tuple(sorted(run_roots)) != ACTIVE_SEEDS or tuple(sorted(receipt_paths)) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Exactly seeds 0,1,2,3 are required.")
    cohort_sha = sha256_file(cohort_manifest_path)
    graph_sha = sha256_file(graph_manifest_path)
    members = tuple(
        validate_member(
            seed=seed,
            run_root=run_roots[seed],
            receipt_path=receipt_paths[seed],
            cohort_manifest_sha256=cohort_sha,
            graph_manifest_sha256=graph_sha,
        )
        for seed in ACTIVE_SEEDS
    )
    run_ids = {member.run_id for member in members}
    checkpoint_hashes = {member.checkpoint_sha256 for member in members}
    state_hashes = {member.model_state_sha256 for member in members}
    if len(run_ids) != EXPECTED_SEED_COUNT:
        raise FourSeedStabilityError("Four distinct run IDs are required.")
    if len(checkpoint_hashes) != EXPECTED_SEED_COUNT or len(state_hashes) != EXPECTED_SEED_COUNT:
        raise FourSeedStabilityError("Each model seed must have distinct final weights.")
    construction_hashes = {member.model_construction_sha256 for member in members}
    if (
        len(construction_hashes) != 1
        or any(member.parameter_count != LOCKED_PARAMETER_COUNT for member in members)
    ):
        raise FourSeedStabilityError("Model construction differs across seeds.")
    if len({member.mask_base_seed for member in members}) != 1 or len(
        {member.core_order_seed for member in members}
    ) != 1:
        raise FourSeedStabilityError("Training mask/core-order base seeds differ.")
    shared_epochs = min(member.completed_global_epochs for member in members)
    reference_schedule = members[0].training_schedule_epoch_sha256[:shared_epochs]
    if any(
        member.training_schedule_epoch_sha256[:shared_epochs] != reference_schedule
        for member in members[1:]
    ):
        raise FourSeedStabilityError(
            "Training core order or ten-view mask schedule differs across seeds."
        )
    reference_masks = members[0].fixed_masks
    if any(member.fixed_masks != reference_masks for member in members[1:]):
        raise FourSeedStabilityError(
            "Strict receipts do not share the exact fixed inference masks."
        )
    return members


def install_deterministic_cuda_contract(device: str | torch.device) -> torch.device:
    """Install the serialized analysis contract before any CUDA allocation."""

    resolved = torch.device(device)
    if resolved != torch.device("cuda:0"):
        raise FourSeedStabilityError(
            "Production stability extraction requires singleton logical cuda:0."
        )
    if torch.cuda.is_initialized():
        raise FourSeedStabilityError(
            "CUDA was initialized before the deterministic analysis contract."
        )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {None, ":4096:8"}:
        raise FourSeedStabilityError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or exactly :4096:8."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_deterministic_seed(
        ANALYSIS_DETERMINISTIC_SEED,
        deterministic=True,
        warn_only=False,
    )
    if (
        torch.are_deterministic_algorithms_enabled() is not True
        or torch.is_deterministic_algorithms_warn_only_enabled() is not False
    ):
        raise FourSeedStabilityError("Deterministic CUDA algorithms were not enabled.")
    return resolved


@dataclass(frozen=True)
class SeedCompactExtraction:
    """One seed's compact fixed-input replay held in host memory/temp files."""

    seed: int
    completed_global_epochs: int
    embedding_ids: tuple[str, ...]
    embeddings: np.ndarray = field(repr=False)
    fixed_edge_ids: tuple[str, ...]
    fixed_edge_core: np.ndarray = field(repr=False)
    fixed_edge_number: np.ndarray = field(repr=False)
    fixed_edge_source: np.ndarray = field(repr=False)
    fixed_edge_receiver: np.ndarray = field(repr=False)
    attention: np.ndarray = field(repr=False)
    content_logits: np.ndarray = field(repr=False)
    positional_bias: np.ndarray = field(repr=False)
    combined_logits: np.ndarray = field(repr=False)
    mutual_score_paths: Mapping[str, Path] = field(repr=False)
    mutual_top_positions: Mapping[str, np.ndarray] = field(repr=False)
    gradient_rows: tuple[Mapping[str, Any], ...]
    fixed_mask_receipts: Mapping[str, Mapping[str, Any]]


def _selected_receiver_edge_ids(
    edge_index: np.ndarray,
    receivers: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    receiver_mask = np.zeros(int(n_nodes), dtype=np.bool_)
    receiver_mask[np.asarray(receivers, dtype=np.int64)] = True
    selected = np.flatnonzero(receiver_mask[edge_index[1]]).astype(
        np.int64, copy=False
    )
    if len(selected) == 0:
        raise FourSeedStabilityError("Fixed receivers have no incoming edges.")
    return selected


def _validate_fixed_attention(
    *,
    edge_index: np.ndarray,
    edge_ids: np.ndarray,
    attention: np.ndarray,
    content: np.ndarray,
    bias: np.ndarray,
    combined: np.ndarray,
    receivers: np.ndarray,
) -> None:
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
        attention,
        content,
        bias,
        combined,
    ))
    if (
        any(value.ndim != 2 for value in arrays)
        or len({value.shape for value in arrays}) != 1
        or arrays[0].shape[0] != len(edge_ids)
        or not all(np.isfinite(value).all() for value in arrays)
    ):
        raise FourSeedStabilityError("Fixed receiver attention channels misalign.")
    attention_values, content_values, bias_values, combined_values = arrays
    composition_error = float(
        np.max(np.abs(combined_values - (content_values + bias_values)))
    )
    if composition_error > 1e-5:
        raise FourSeedStabilityError("Combined logits do not equal content plus bias.")
    selected_receivers = edge_index[1, edge_ids]
    if not np.isin(selected_receivers, receivers).all():
        raise FourSeedStabilityError("Fixed edge set contains an unselected receiver.")
    for receiver in receivers.tolist():
        # The exported probabilities are intentionally FP32, but accumulating a
        # few hundred of them again in host FP32 can cross the 1e-6 audit gate
        # solely because NumPy and the GPU softmax use different reduction
        # orders.  Accumulate the already-exported values in FP64 so this check
        # measures their represented probability mass instead of adding a
        # second FP32 reduction error.
        sums = attention_values[selected_receivers == receiver].sum(
            axis=0,
            dtype=np.float64,
        )
        if not np.allclose(sums, np.ones_like(sums), atol=1e-6, rtol=0.0):
            maximum_error = float(np.max(np.abs(sums - 1.0)))
            raise FourSeedStabilityError(
                "Fixed receiver attention does not normalize to one per head "
                f"(receiver={receiver}, maximum_abs_error={maximum_error:.17g})."
            )


def _extract_core_compact(
    *,
    model: torch.nn.Module,
    batch: PooledRelativeQKVCoreBatch,
    member: ValidatedMember,
    derivative_requests: Sequence[SelectedDerivativeRequest],
    request_metadata: Mapping[str, Mapping[str, Any]],
    temporary_dir: Path,
    amp: bool,
) -> dict[str, Any]:
    alias = batch.alias
    mask = fixed_inference_mask(batch)
    expected_mask = member.fixed_masks[alias]
    if (
        mask.seed != int(expected_mask["seed"])
        or mask.checksum_sha256 != expected_mask["checksum_sha256"]
        or mask.n_masked_entries != int(expected_mask["masked_entry_count"])
    ):
        raise FourSeedStabilityError(
            f"Seed {member.seed} {alias} fixed inference mask drifted."
        )
    receivers = evenly_spaced_receivers(batch.n_nodes)
    edge_index = np.ascontiguousarray(batch.edge_index.detach().cpu().numpy())
    selected_expected = _selected_receiver_edge_ids(
        edge_index,
        receivers,
        n_nodes=batch.n_nodes,
    )
    indegree = np.bincount(
        edge_index[1], minlength=batch.n_nodes
    ).astype(np.float32)
    directional_path = temporary_dir / (
        f"seed_{member.seed}_{alias}_directional.float32.dat"
    )
    directional = np.memmap(
        directional_path,
        mode="w+",
        dtype=np.float32,
        shape=(batch.n_edges,),
    )
    directional[:] = np.nan
    selected_ids_parts: list[np.ndarray] = []
    channel_parts: dict[str, list[np.ndarray]] = {
        "attention": [],
        "content": [],
        "bias": [],
        "combined": [],
    }
    embedding_holder: list[np.ndarray] = []
    receiver_lookup = np.zeros(batch.n_nodes, dtype=np.bool_)
    receiver_lookup[receivers] = True

    def consume(
        _receiver_start: int,
        _receiver_stop: int,
        edge_ids: np.ndarray,
        attention: np.ndarray,
        content: np.ndarray,
        bias: np.ndarray,
        combined: np.ndarray,
    ) -> None:
        ids = np.asarray(edge_ids, dtype=np.int64)
        receiver = edge_index[1, ids]
        attention_values = np.asarray(attention, dtype=np.float32)
        if attention_values.ndim != 2 or attention_values.shape[0] != len(ids):
            raise FourSeedStabilityError("Streamed attention is edge-misaligned.")
        directional[ids] = indegree[receiver] * attention_values.mean(axis=1)
        keep = receiver_lookup[receiver]
        if bool(keep.any()):
            selected_ids_parts.append(np.ascontiguousarray(ids[keep]))
            for name, values in (
                ("attention", attention),
                ("content", content),
                ("bias", bias),
                ("combined", combined),
            ):
                channel_parts[name].append(
                    np.ascontiguousarray(np.asarray(values, dtype=np.float32)[keep])
                )

    resolved_layer = stream_receiver_attention(
        model,
        batch,
        mask.mask,
        layer=FINAL_LAYER,
        amp=amp,
        consumer=consume,
        node_embedding_consumer=embedding_holder.append,
    )
    if resolved_layer != model.graph_layers - 1:
        raise FourSeedStabilityError("Fixed attention replay did not use final layer.")
    directional.flush()
    if not np.isfinite(directional).all() or len(embedding_holder) != 1:
        raise FourSeedStabilityError("Core stream did not cover all routing/embeddings.")
    selected_ids = np.concatenate(selected_ids_parts)
    order = np.argsort(selected_ids, kind="stable")
    selected_ids = selected_ids[order]
    if not np.array_equal(selected_ids, selected_expected):
        raise FourSeedStabilityError("Fixed receiver edge identity/order drifted.")
    channels = {
        name: np.concatenate(parts, axis=0)[order]
        for name, parts in channel_parts.items()
    }
    _validate_fixed_attention(
        edge_index=edge_index,
        edge_ids=selected_ids,
        attention=channels["attention"],
        content=channels["content"],
        bias=channels["bias"],
        combined=channels["combined"],
        receivers=receivers,
    )
    mutual = vectorized_mutual_pair_scores(
        edge_index,
        directional,
        n_nodes=batch.n_nodes,
    )
    mutual_path = temporary_dir / f"seed_{member.seed}_{alias}_mutual.npy"
    if mutual_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary scores: {mutual_path}")
    np.save(mutual_path, mutual.scores, allow_pickle=False)
    top_positions = top_mutual_pair_positions(
        mutual.scores,
        mutual.pair_keys,
    )
    del directional
    directional_path.unlink()

    gradient_rows = selected_autograd_derivatives(
        model,
        batch,
        mask.mask,
        derivative_requests,
    )
    if len(gradient_rows) != len(derivative_requests):
        raise FourSeedStabilityError("Selected derivative count drifted.")
    enriched_gradients: list[dict[str, Any]] = []
    for row in gradient_rows:
        request_id = str(row["request_id"])
        metadata = request_metadata.get(request_id)
        if metadata is None:
            raise FourSeedStabilityError("Derivative request metadata is missing.")
        if (
            row.get("core_alias") != metadata.get("core_alias")
            or int(row.get("source_node")) != int(metadata.get("source_node"))
            or int(row.get("receiver_node")) != int(metadata.get("receiver_node"))
            or int(row.get("source_feature_index"))
            != int(metadata.get("source_feature_index"))
            or int(row.get("target_feature_index"))
            != int(metadata.get("target_feature_index"))
            or row.get("source_feature_observed") is not True
            or row.get("target_feature_masked") is not True
            or row.get("attention_head") != "mean"
            or int(row.get("layer_number")) != model.graph_layers - 1
        ):
            raise FourSeedStabilityError("Derivative execution contract drifted.")
        enriched_gradients.append(
            {
                "seed": member.seed,
                **dict(metadata),
                **dict(row),
            }
        )
    embedding = np.asarray(embedding_holder[0], dtype=np.float32)
    if (
        embedding.shape != (batch.n_nodes, model.hidden_dim)
        or not np.isfinite(embedding).all()
    ):
        raise FourSeedStabilityError("Final node embedding schema drifted.")
    return {
        "alias": alias,
        "receivers": receivers,
        "embedding": np.ascontiguousarray(embedding),
        "edge_ids": np.ascontiguousarray(selected_ids),
        "edge_source": np.ascontiguousarray(edge_index[0, selected_ids]),
        "edge_receiver": np.ascontiguousarray(edge_index[1, selected_ids]),
        "attention": np.ascontiguousarray(channels["attention"]),
        "content": np.ascontiguousarray(channels["content"]),
        "bias": np.ascontiguousarray(channels["bias"]),
        "combined": np.ascontiguousarray(channels["combined"]),
        "mutual_path": mutual_path,
        "mutual_top_positions": np.ascontiguousarray(top_positions),
        "gradient_rows": enriched_gradients,
        "fixed_mask_receipt": {
            "effective_seed": mask.seed,
            "checksum_sha256": mask.checksum_sha256,
            "masked_entry_count": mask.n_masked_entries,
        },
    }


def extract_member_compact(
    member: ValidatedMember,
    batches: Sequence[PooledRelativeQKVCoreBatch],
    *,
    derivative_requests_by_alias: Mapping[str, Sequence[SelectedDerivativeRequest]],
    request_metadata: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    temporary_dir: Path,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
    amp: bool = True,
) -> SeedCompactExtraction:
    """Replay one seed sequentially across six complete-core CPU graphs."""

    materialized = tuple(batches)
    if tuple(batch.alias for batch in materialized) != CANCER_ALIASES:
        raise FourSeedStabilityError("Prepared batch alias/order drifted.")
    if tuple(derivative_requests_by_alias) != CANCER_ALIASES:
        raise FourSeedStabilityError("Gradient request alias/order drifted.")
    loaded = load_relative_qkv_checkpoint(
        member.checkpoint_path,
        num_genes=materialized[0].n_genes,
        node_covariate_dim=int(materialized[0].node_covariates.shape[1]),
        device="cpu",
        receiver_chunk_size=receiver_chunk_size,
        max_edges_per_chunk=max_edges_per_chunk,
    )
    if (
        loaded.checkpoint_sha256 != member.checkpoint_sha256
        or loaded.payload.get("model_seed") != member.seed
        or loaded.payload.get("run_id") != member.run_id
    ):
        raise FourSeedStabilityError("Loaded model identity drifted after validation.")
    loaded.model.to(device)
    loaded.model.eval()
    core_results: list[dict[str, Any]] = []
    try:
        for batch in materialized:
            requests = tuple(derivative_requests_by_alias[batch.alias])
            if len(requests) != 4:
                raise FourSeedStabilityError(
                    f"{batch.alias} must have exactly four derivative requests."
                )
            core_results.append(
                _extract_core_compact(
                    model=loaded.model,
                    batch=batch,
                    member=member,
                    derivative_requests=requests,
                    request_metadata=request_metadata,
                    temporary_dir=temporary_dir,
                    amp=amp,
                )
            )
            loaded.model.clear_edge_layout_cache()
            torch.cuda.empty_cache()
    finally:
        loaded.model.to("cpu")
        loaded.model.clear_edge_layout_cache()
        torch.cuda.empty_cache()

    embedding_ids: list[str] = []
    fixed_edge_ids: list[str] = []
    for result in core_results:
        alias = str(result["alias"])
        embedding_ids.extend(
            f"{alias}:node:{node:08d}"
            for node in range(len(result["embedding"]))
        )
        fixed_edge_ids.extend(
            f"{alias}:layer:final:edge:{int(edge_id):012d}"
            for edge_id in result["edge_ids"]
        )
    extraction = SeedCompactExtraction(
        seed=member.seed,
        completed_global_epochs=member.completed_global_epochs,
        embedding_ids=tuple(embedding_ids),
        embeddings=np.concatenate(
            [result["embedding"] for result in core_results], axis=0
        ),
        fixed_edge_ids=tuple(fixed_edge_ids),
        fixed_edge_core=np.concatenate(
            [
                np.full(len(result["edge_ids"]), result["alias"], dtype="U6")
                for result in core_results
            ]
        ),
        fixed_edge_number=np.concatenate(
            [result["edge_ids"] for result in core_results]
        ),
        fixed_edge_source=np.concatenate(
            [result["edge_source"] for result in core_results]
        ),
        fixed_edge_receiver=np.concatenate(
            [result["edge_receiver"] for result in core_results]
        ),
        attention=np.concatenate(
            [result["attention"] for result in core_results], axis=0
        ),
        content_logits=np.concatenate(
            [result["content"] for result in core_results], axis=0
        ),
        positional_bias=np.concatenate(
            [result["bias"] for result in core_results], axis=0
        ),
        combined_logits=np.concatenate(
            [result["combined"] for result in core_results], axis=0
        ),
        mutual_score_paths={
            str(result["alias"]): Path(result["mutual_path"])
            for result in core_results
        },
        mutual_top_positions={
            str(result["alias"]): np.asarray(
                result["mutual_top_positions"], dtype=np.int64
            )
            for result in core_results
        },
        gradient_rows=tuple(
            row for result in core_results for row in result["gradient_rows"]
        ),
        fixed_mask_receipts={
            str(result["alias"]): dict(result["fixed_mask_receipt"])
            for result in core_results
        },
    )
    if len(extraction.gradient_rows) != EXPECTED_GRADIENT_REQUEST_COUNT:
        raise FourSeedStabilityError("Seed extraction did not return 24 gradients.")
    return extraction


@dataclass(frozen=True)
class FourSeedAnalysisResult:
    report: Mapping[str, Any]
    tables: Mapping[str, pa.Table] = field(repr=False)
    arrays: Mapping[str, Mapping[str, np.ndarray]] = field(repr=False)


def _aligned_extractions(
    extractions: Sequence[SeedCompactExtraction],
) -> tuple[SeedCompactExtraction, ...]:
    ordered = tuple(sorted(extractions, key=lambda value: value.seed))
    if tuple(value.seed for value in ordered) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Compact inputs must be exactly seeds 0,1,2,3.")
    reference = ordered[0]
    for value in ordered[1:]:
        if (
            value.embedding_ids != reference.embedding_ids
            or value.fixed_edge_ids != reference.fixed_edge_ids
            or not np.array_equal(value.fixed_edge_core, reference.fixed_edge_core)
            or not np.array_equal(value.fixed_edge_number, reference.fixed_edge_number)
            or not np.array_equal(value.fixed_edge_source, reference.fixed_edge_source)
            or not np.array_equal(value.fixed_edge_receiver, reference.fixed_edge_receiver)
            or value.fixed_mask_receipts != reference.fixed_mask_receipts
        ):
            raise FourSeedStabilityError(
                "Fixed node/edge/mask identities are not aligned across seeds."
            )
    embedding_shapes = {value.embeddings.shape for value in ordered}
    channel_shapes = {
        (
            value.attention.shape,
            value.content_logits.shape,
            value.positional_bias.shape,
            value.combined_logits.shape,
        )
        for value in ordered
    }
    if len(embedding_shapes) != 1 or len(channel_shapes) != 1:
        raise FourSeedStabilityError("Compact numeric shapes drifted across seeds.")
    for value in ordered:
        if (
            value.attention.ndim != 2
            or value.attention.shape != value.content_logits.shape
            or value.attention.shape != value.positional_bias.shape
            or value.attention.shape != value.combined_logits.shape
            or not all(
                np.isfinite(array).all()
                for array in (
                    value.embeddings,
                    value.attention,
                    value.content_logits,
                    value.positional_bias,
                    value.combined_logits,
                )
            )
        ):
            raise FourSeedStabilityError("Compact numeric arrays are invalid.")
    return ordered


def _relationship_summary_rows(
    summary: RelationshipEnsembleSummary,
    *,
    metadata: Mapping[object, Mapping[str, Any]],
    score_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in summary.to_rows():
        relationship_id = raw.pop("relationship_id")
        row = {
            "relationship_id": str(relationship_id),
            "score_name": score_name,
            **dict(metadata.get(relationship_id, {})),
            **raw,
            "ensemble_member_count": EXPECTED_SEED_COUNT,
            "spread_label": "four-seed ensemble spread",
            "calibrated_confidence_interval": False,
        }
        rows.append(row)
    return rows


def _gradient_stability_outputs(
    ordered: Sequence[SeedCompactExtraction],
    *,
    value_name: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], Mapping[str, Any]]:
    request_ids = tuple(str(row["request_id"]) for row in ordered[0].gradient_rows)
    if len(request_ids) != EXPECTED_GRADIENT_REQUEST_COUNT or len(set(request_ids)) != len(
        request_ids
    ):
        raise FourSeedStabilityError("Gradient request IDs are incomplete or duplicated.")
    metadata: dict[object, Mapping[str, Any]] = {}
    records: list[FixedSeedArray] = []
    for extraction in ordered:
        observed_ids = tuple(str(row["request_id"]) for row in extraction.gradient_rows)
        if observed_ids != request_ids:
            raise FourSeedStabilityError("Gradient request order drifted across seeds.")
        values = np.asarray(
            [
                _finite_float(row.get(value_name), f"{value_name} {request_id}")
                for row, request_id in zip(
                    extraction.gradient_rows, request_ids, strict=True
                )
            ],
            dtype=np.float64,
        )
        records.append(
            FixedSeedArray(
                seed=extraction.seed,
                fixed_input_ids=request_ids,
                values=values,
            )
        )
        if extraction.seed == REFERENCE_SEED:
            for row in extraction.gradient_rows:
                request_id = str(row["request_id"])
                metadata[request_id] = {
                    key: row[key]
                    for key in (
                        "core_alias",
                        "shell",
                        "distance_um",
                        "source_node",
                        "receiver_node",
                        "source_feature_index",
                        "source_feature_name",
                        "target_feature_index",
                        "target_feature_name",
                        "attention_head",
                        "layer_number",
                    )
                    if key in row
                }
    stability = selected_gradient_stability(
        records,
        magnitude_threshold=0.0,
        quantile_levels=QUANTILE_LEVELS,
        expected_seed_count=EXPECTED_SEED_COUNT,
    )
    rows = _relationship_summary_rows(
        stability.summary,
        metadata=metadata,
        score_name=value_name,
    )
    for index, row in enumerate(rows):
        row["ensemble_median_sign"] = float(stability.ensemble_median_sign[index])
        row["consistent_sign_support_count"] = int(
            stability.consistent_sign_support_count[index]
        )
        row["consistent_sign_fraction"] = float(
            stability.consistent_sign_fraction[index]
        )
    arrays = {
        f"{value_name}_pairwise_seed_spearman": stability.pairwise_seed_spearman,
        f"{value_name}_ensemble_median_sign": stability.ensemble_median_sign,
        f"{value_name}_consistent_sign_support_count": (
            stability.consistent_sign_support_count
        ),
        f"{value_name}_consistent_sign_fraction": stability.consistent_sign_fraction,
    }
    report = {
        "analysis_scope": stability.analysis_scope,
        "summary_row_count": len(rows),
        "pairwise_seed_spearman": _json_safe(stability.pairwise_seed_spearman),
    }
    return rows, arrays, report


def _mutual_stability_rows(
    ordered: Sequence[SeedCompactExtraction],
    batches: Sequence[PooledRelativeQKVCoreBatch],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    by_alias = {batch.alias: batch for batch in batches}
    if tuple(by_alias) != CANCER_ALIASES:
        raise FourSeedStabilityError("Mutual-pair batch aliases drifted.")
    all_rows: list[dict[str, Any]] = []
    per_core_report: dict[str, Any] = {}
    for alias in CANCER_ALIASES:
        batch = by_alias[alias]
        edge_index = np.asarray(batch.edge_index.detach().cpu().numpy())
        identity = vectorized_mutual_pair_scores(
            edge_index,
            np.zeros(batch.n_edges, dtype=np.float32),
            n_nodes=batch.n_nodes,
        )
        top_by_seed = {
            extraction.seed: np.asarray(
                extraction.mutual_top_positions[alias], dtype=np.int64
            )
            for extraction in ordered
        }
        if any(len(values) != MUTUAL_TOP_K_PER_CORE_SEED for values in top_by_seed.values()):
            raise FourSeedStabilityError(f"{alias} mutual top-100 selection drifted.")
        union_positions = np.unique(
            np.concatenate([top_by_seed[seed] for seed in ACTIVE_SEEDS])
        )
        if (
            len(union_positions) < MUTUAL_TOP_K_PER_CORE_SEED
            or len(union_positions) > MUTUAL_TOP_K_PER_CORE_SEED * EXPECTED_SEED_COUNT
            or np.any(union_positions < 0)
            or np.any(union_positions >= len(identity.scores))
        ):
            raise FourSeedStabilityError(f"{alias} mutual top-pair union is invalid.")
        relationship_ids = tuple(
            f"{alias}:pair:{int(identity.pair_keys[position]):016d}"
            for position in union_positions
        )
        records: list[FixedSeedArray] = []
        support_records: list[FixedSeedArray] = []
        for extraction in ordered:
            scores = np.load(
                extraction.mutual_score_paths[alias],
                mmap_mode="r",
                allow_pickle=False,
            )
            if scores.shape != identity.scores.shape or not np.isfinite(scores).all():
                raise FourSeedStabilityError(f"{alias} mutual score array drifted.")
            expected_top = top_mutual_pair_positions(
                scores,
                identity.pair_keys,
            )
            if not np.array_equal(
                expected_top,
                top_by_seed[extraction.seed],
            ):
                raise FourSeedStabilityError(
                    f"{alias} seed {extraction.seed} mutual top-100 receipt drifted."
                )
            values = np.asarray(scores[union_positions], dtype=np.float64)
            support = np.isin(
                union_positions,
                top_by_seed[extraction.seed],
                assume_unique=True,
            ).astype(np.float64)
            records.append(
                FixedSeedArray(
                    seed=extraction.seed,
                    fixed_input_ids=relationship_ids,
                    values=values,
                )
            )
            support_records.append(
                FixedSeedArray(
                    seed=extraction.seed,
                    fixed_input_ids=relationship_ids,
                    values=support,
                )
            )
        summary = summarize_relationship_ensemble(
            records,
            support_records=support_records,
            quantile_levels=QUANTILE_LEVELS,
            expected_seed_count=EXPECTED_SEED_COUNT,
        )
        canonical_edges = identity.canonical_edge_ids[union_positions]
        metadata = {
            relationship_id: {
                "core_alias": alias,
                "pair_position": int(position),
                "pair_key": int(identity.pair_keys[position]),
                "node_low": int(edge_index[0, edge_id]),
                "node_high": int(edge_index[1, edge_id]),
            }
            for relationship_id, position, edge_id in zip(
                relationship_ids,
                union_positions.tolist(),
                canonical_edges.tolist(),
                strict=True,
            )
        }
        rows = _relationship_summary_rows(
            summary,
            metadata=metadata,
            score_name="mutual_attention_routing_score",
        )
        records_by_seed = {record.seed: record for record in records}
        support_by_seed = {record.seed: record for record in support_records}
        for row_index, row in enumerate(rows):
            for seed in ACTIVE_SEEDS:
                row[f"seed_{seed}_score"] = float(
                    np.asarray(records_by_seed[seed].values)[row_index]
                )
                row[f"seed_{seed}_top100_support"] = bool(
                    np.asarray(support_by_seed[seed].values)[row_index] > 0.0
                )
        all_rows.extend(rows)
        per_core_report[alias] = {
            "top_k_per_seed": MUTUAL_TOP_K_PER_CORE_SEED,
            "union_pair_count": len(rows),
            "support_count_range": [
                int(summary.support_count.min()),
                int(summary.support_count.max()),
            ],
        }
    return all_rows, per_core_report


def _fixed_edge_table(ordered: Sequence[SeedCompactExtraction]) -> pa.Table:
    tables: list[pa.Table] = []
    for extraction in ordered:
        columns: dict[str, Any] = {
            "seed": np.full(len(extraction.fixed_edge_ids), extraction.seed, dtype=np.int8),
            "fixed_edge_id": list(extraction.fixed_edge_ids),
            "core_alias": extraction.fixed_edge_core,
            "edge_id": extraction.fixed_edge_number,
            "source_node": extraction.fixed_edge_source,
            "receiver_node": extraction.fixed_edge_receiver,
            "attention_mean": np.asarray(
                extraction.attention.mean(axis=1), dtype=np.float32
            ),
            "content_logit_mean": np.asarray(
                extraction.content_logits.mean(axis=1), dtype=np.float32
            ),
            "positional_bias_mean": np.asarray(
                extraction.positional_bias.mean(axis=1), dtype=np.float32
            ),
            "combined_logit_mean": np.asarray(
                extraction.combined_logits.mean(axis=1), dtype=np.float32
            ),
        }
        for prefix, values in (
            ("attention", extraction.attention),
            ("content_logit", extraction.content_logits),
            ("positional_bias", extraction.positional_bias),
            ("combined_logit", extraction.combined_logits),
        ):
            for head in range(values.shape[1]):
                columns[f"{prefix}_head_{head:02d}"] = np.asarray(
                    values[:, head], dtype=np.float32
                )
        tables.append(pa.table(columns))
    return pa.concat_tables(tables)


def _fixed_edge_stability_table(
    ordered: Sequence[SeedCompactExtraction],
    *,
    receiver_group_ids: Sequence[object],
) -> tuple[pa.Table, Mapping[str, Any]]:
    """Summarize permutation-invariant fixed-probe edge scores across seeds."""

    reference = ordered[0]
    relationship_ids = reference.fixed_edge_ids
    edge_count = len(relationship_ids)
    if edge_count == 0:
        raise FourSeedStabilityError("Fixed-probe edge summary cannot be empty.")
    rows = [
        {
            "fixed_edge_id": relationship_id,
            "core_alias": str(reference.fixed_edge_core[index]),
            "edge_id": int(reference.fixed_edge_number[index]),
            "source_node": int(reference.fixed_edge_source[index]),
            "receiver_node": int(reference.fixed_edge_receiver[index]),
        }
        for index, relationship_id in enumerate(relationship_ids)
    ]
    score_arrays: Mapping[str, tuple[np.ndarray, ...]] = {
        "attention_mean_head": tuple(
            np.asarray(value.attention, dtype=np.float64).mean(axis=1)
            for value in ordered
        ),
        "receiver_centered_content_logit_mean_head": tuple(
            receiver_centered_logits(value.content_logits, receiver_group_ids).mean(
                axis=1
            )
            for value in ordered
        ),
        "receiver_centered_positional_bias_mean_head": tuple(
            receiver_centered_logits(value.positional_bias, receiver_group_ids).mean(
                axis=1
            )
            for value in ordered
        ),
        "receiver_centered_combined_logit_mean_head": tuple(
            receiver_centered_logits(value.combined_logits, receiver_group_ids).mean(
                axis=1
            )
            for value in ordered
        ),
    }
    definitions: dict[str, Any] = {}
    for score_name, arrays in score_arrays.items():
        records = tuple(
            FixedSeedArray(
                seed=value.seed,
                fixed_input_ids=relationship_ids,
                values=np.ascontiguousarray(array),
            )
            for value, array in zip(ordered, arrays, strict=True)
        )
        support_records: tuple[FixedSeedArray, ...] | None = None
        if score_name == "attention_mean_head":
            top_k = max(1, int(math.ceil(edge_count * HEAD_TOP_EDGE_FRACTION)))
            supports: list[FixedSeedArray] = []
            for value, array in zip(ordered, arrays, strict=True):
                order = np.lexsort((np.arange(edge_count, dtype=np.int64), -array))
                support = np.zeros(edge_count, dtype=np.float64)
                support[order[:top_k]] = 1.0
                supports.append(
                    FixedSeedArray(
                        seed=value.seed,
                        fixed_input_ids=relationship_ids,
                        values=support,
                    )
                )
            support_records = tuple(supports)
            support_definition = (
                "membership in that seed's top 5% mean-head attention values "
                "over the locked fixed-probe edge set"
            )
            definitions[score_name] = {
                "support_definition": support_definition,
                "top_fraction": HEAD_TOP_EDGE_FRACTION,
                "top_edge_count_per_seed": top_k,
            }
        else:
            support_definition = "absolute score strictly greater than zero"
            definitions[score_name] = {
                "support_definition": support_definition,
            }
        summary = summarize_relationship_ensemble(
            records,
            support_records=support_records,
            support_magnitude_threshold=0.0,
            quantile_levels=QUANTILE_LEVELS,
            expected_seed_count=EXPECTED_SEED_COUNT,
        )
        summary_rows = summary.to_rows()
        if len(summary_rows) != edge_count:
            raise FourSeedStabilityError("Fixed-probe edge summary length drifted.")
        for index, raw in enumerate(summary_rows):
            if raw.pop("relationship_id") != relationship_ids[index]:
                raise FourSeedStabilityError(
                    "Fixed-probe edge summary identity drifted."
                )
            for key, value in raw.items():
                rows[index][f"{score_name}_{key}"] = value
            rows[index][f"{score_name}_support_definition"] = support_definition
            for seed, array in zip(ACTIVE_SEEDS, arrays, strict=True):
                rows[index][f"{score_name}_seed_{seed}"] = float(array[index])
    return pa.Table.from_pylist(rows), {
        "row_count": edge_count,
        "scores": definitions,
        "quantile_method": "numpy linear interpolation",
        "standard_deviation_ddof": 1,
    }


def compute_four_seed_analysis(
    members: Sequence[ValidatedMember],
    extractions: Sequence[SeedCompactExtraction],
    batches: Sequence[PooledRelativeQKVCoreBatch],
    *,
    protocol_provenance: Mapping[str, Any],
) -> FourSeedAnalysisResult:
    """Assemble all required four-seed statistics from compact fixed inputs."""

    ordered = _aligned_extractions(extractions)
    member_order = tuple(sorted(members, key=lambda value: value.seed))
    if tuple(member.seed for member in member_order) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Validated member seeds drifted.")
    if any(
        extraction.completed_global_epochs != member.completed_global_epochs
        for extraction, member in zip(ordered, member_order, strict=True)
    ):
        raise FourSeedStabilityError("Extraction/checkpoint epoch drifted.")

    embedding_records = tuple(
        FixedSeedArray(
            seed=value.seed,
            fixed_input_ids=value.embedding_ids,
            values=value.embeddings,
        )
        for value in ordered
    )
    embedding_report = embedding_stability(
        embedding_records,
        include_orthogonal_procrustes=True,
        expected_seed_count=EXPECTED_SEED_COUNT,
    )
    edge_records = {
        name: tuple(
            FixedSeedArray(
                seed=value.seed,
                fixed_input_ids=value.fixed_edge_ids,
                values=np.asarray(getattr(value, attribute)),
            )
            for value in ordered
        )
        for name, attribute in (
            ("attention", "attention"),
            ("content", "content_logits"),
            ("bias", "positional_bias"),
            ("combined", "combined_logits"),
        )
    }
    receiver_group_ids = tuple(
        f"{core}:receiver:{int(receiver):08d}"
        for core, receiver in zip(
            ordered[0].fixed_edge_core.tolist(),
            ordered[0].fixed_edge_receiver.tolist(),
            strict=True,
        )
    )
    centered_content_records = tuple(
        FixedSeedArray(
            seed=value.seed,
            fixed_input_ids=value.fixed_edge_ids,
            values=receiver_centered_logits(
                value.content_logits,
                receiver_group_ids,
            ),
        )
        for value in ordered
    )
    centered_bias_records = tuple(
        FixedSeedArray(
            seed=value.seed,
            fixed_input_ids=value.fixed_edge_ids,
            values=receiver_centered_logits(
                value.positional_bias,
                receiver_group_ids,
            ),
        )
        for value in ordered
    )
    signature_ids = tuple(
        f"{channel}:{edge_id}"
        for channel in ("attention", "content", "bias")
        for edge_id in ordered[0].fixed_edge_ids
    )
    signature_records = tuple(
        FixedSeedArray(
            seed=value.seed,
            fixed_input_ids=signature_ids,
            values=attention_head_signature(
                value.attention,
                content_logits=centered_content.values,
                positional_bias=centered_bias.values,
            ),
        )
        for value, centered_content, centered_bias in zip(
            ordered,
            centered_content_records,
            centered_bias_records,
            strict=True,
        )
    )
    head_alignment = match_attention_heads(
        signature_records,
        reference_seed=REFERENCE_SEED,
        expected_seed_count=EXPECTED_SEED_COUNT,
    )
    attention_stability = matched_head_attention_stability(
        edge_records["attention"],
        head_alignment,
        top_fraction=HEAD_TOP_EDGE_FRACTION,
    )
    bias_stability = positional_bias_response_stability(
        centered_bias_records,
        head_alignment,
    )
    contribution = content_position_contribution_agreement(
        centered_content_records,
        centered_bias_records,
        head_alignment,
    )
    fixed_edge_stability_table, fixed_edge_stability_report = (
        _fixed_edge_stability_table(
            ordered,
            receiver_group_ids=receiver_group_ids,
        )
    )
    mutual_rows, mutual_report = _mutual_stability_rows(ordered, batches)
    attention_gradient_rows, attention_gradient_arrays, attention_gradient_report = (
        _gradient_stability_outputs(
            ordered,
            value_name="d_attention_d_source_feature",
        )
    )
    prediction_gradient_rows, prediction_gradient_arrays, prediction_gradient_report = (
        _gradient_stability_outputs(
            ordered,
            value_name="d_prediction_d_source_feature",
        )
    )
    raw_gradient_rows = [
        dict(row) for extraction in ordered for row in extraction.gradient_rows
    ]

    training_rows: list[dict[str, Any]] = []
    for member in member_order:
        training_rows.extend(
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "global_epoch": index,
                "completed_global_epoch": index + 1,
                "equal_core_mean_masked_huber": float(loss),
                "final_plateau_epoch": member.completed_global_epochs,
            }
            for index, loss in enumerate(member.loss_curve)
        )
    final_losses = [float(member.loss_curve[-1]) for member in member_order]
    metric_names = tuple(member_order[0].fixed_metrics)
    fixed_metric_rows = [
        {
            "seed": member.seed,
            "run_id": member.run_id,
            "completed_global_epochs": member.completed_global_epochs,
            **{name: float(member.fixed_metrics[name]) for name in metric_names},
        }
        for member in member_order
    ]
    fixed_metric_summary_rows = [
        {
            "metric": name,
            **scalar_summary([member.fixed_metrics[name] for member in member_order]),
        }
        for name in metric_names
    ]

    report = {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "scope": "four_seed_ensemble_spread_seeds_0_1_2_3",
        "active_model_seeds": list(ACTIVE_SEEDS),
        "deferred_model_seeds": list(DEFERRED_SEEDS),
        "five_seed_campaign_completion_claim_allowed": False,
        "spread_label": "four-seed ensemble spread",
        "uncertainty_label": "seed uncertainty",
        "calibrated_biological_confidence_interval": False,
        "fit_scope": "all_cells_six_cancer_cores_transductive",
        "validation_or_test_partition_present": False,
        "generalization_claim_supported": False,
        "causal_claim_supported": False,
        "differing_final_epochs": {
            "allowed": True,
            "common_final_epoch_required": False,
            "exposure_duration_can_contribute_to_seed_spread": True,
            "epochs_by_seed": {
                str(member.seed): member.completed_global_epochs
                for member in member_order
            },
        },
        "protocol": _json_safe(protocol_provenance),
        "members": [
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "run_root": member.run_root.as_posix(),
                "checkpoint_path": member.checkpoint_path.as_posix(),
                "checkpoint_sha256": member.checkpoint_sha256,
                "model_state_sha256": member.model_state_sha256,
                "history_sha256": member.history_sha256,
                "strict_receipt_path": member.receipt_path.as_posix(),
                "strict_receipt_file_sha256": member.receipt_file_sha256,
                "strict_receipt_content_sha256": member.receipt_content_sha256,
                "completed_global_epochs": member.completed_global_epochs,
                "parameter_count": member.parameter_count,
                "model_construction_sha256": member.model_construction_sha256,
                "mask_base_seed": member.mask_base_seed,
                "core_order_seed": member.core_order_seed,
                "training_schedule_sha256": canonical_sha256(
                    member.training_schedule_epoch_sha256
                ),
            }
            for member in member_order
        ],
        "shared_model_and_training_schedule": {
            "parameter_count": LOCKED_PARAMETER_COUNT,
            "model_construction": dict(LOCKED_MODEL_CONSTRUCTION),
            "model_construction_sha256": member_order[0].model_construction_sha256,
            "mask_base_seed": member_order[0].mask_base_seed,
            "core_order_seed": member_order[0].core_order_seed,
            "shared_prefix_epochs": min(
                member.completed_global_epochs for member in member_order
            ),
            "shared_prefix_schedule_sha256": canonical_sha256(
                member_order[0].training_schedule_epoch_sha256[
                    : min(member.completed_global_epochs for member in member_order)
                ]
            ),
        },
        "fixed_input_identity": {
            "core_aliases": list(CANCER_ALIASES),
            "receiver_probes_per_core": RECEIVER_PROBES_PER_CORE,
            "receiver_selection": "endpoint-inclusive evenly spaced node indices",
            "final_layer": FINAL_LAYER,
            "node_count": len(ordered[0].embedding_ids),
            "node_identity_sha256": canonical_sha256(ordered[0].embedding_ids),
            "fixed_directed_edge_count": len(ordered[0].fixed_edge_ids),
            "fixed_edge_identity_sha256": canonical_sha256(
                ordered[0].fixed_edge_ids
            ),
            "fixed_masks": _json_safe(ordered[0].fixed_mask_receipts),
        },
        "training_loss": {
            "metric": "equal_core_mean_training_masked_huber",
            "final_by_seed": {
                str(member.seed): float(member.loss_curve[-1])
                for member in member_order
            },
            "final_summary": scalar_summary(final_losses),
            "curves_table": "tables/training_curves.parquet",
        },
        "fixed_held_in_metrics": {
            "role": "held_in_fit_diagnostic_not_validation_or_test",
            "per_seed": fixed_metric_rows,
            "summary": {
                row["metric"]: {key: value for key, value in row.items() if key != "metric"}
                for row in fixed_metric_summary_rows
            },
        },
        "embedding_stability": {
            "method": "linear CKA and optional orthogonal Procrustes",
            "seeds": list(embedding_report.seeds),
            "linear_cka": _json_safe(embedding_report.linear_cka),
            "orthogonal_procrustes_similarity": _json_safe(
                embedding_report.orthogonal_procrustes_similarity
            ),
        },
        "attention_head_stability": {
            "reference_seed": REFERENCE_SEED,
            "matching_method": head_alignment.matching_method,
            "head_top_edge_fraction": HEAD_TOP_EDGE_FRACTION,
            "head_top_edge_count": attention_stability.top_k,
            "logit_comparison_gauge": (
                "content and positional-bias logits are centered within each "
                "receiver/head before signatures and contribution diagnostics"
            ),
            "receiver_softmax_null_offsets_removed": True,
            "reference_to_seed_head": _json_safe(
                head_alignment.reference_to_seed_head
            ),
            "matched_signature_spearman": _json_safe(
                head_alignment.matched_signature_spearman
            ),
            "matched_attention_spearman": _json_safe(
                attention_stability.matched_head_spearman
            ),
            "matched_attention_top_edge_jaccard": _json_safe(
                attention_stability.matched_head_top_edge_jaccard
            ),
            "matched_positional_bias_spearman": _json_safe(
                bias_stability.matched_head_spearman
            ),
            "content_vs_positional_bias": {
                "within_seed_spearman": _json_safe(contribution.within_seed_spearman),
                "same_sign_fraction": _json_safe(contribution.same_sign_fraction),
                "content_absolute_fraction": _json_safe(
                    contribution.content_absolute_fraction
                ),
            },
        },
        "fixed_probe_edge_stability": {
            "summary_table": "tables/fixed_edge_summary.parquet",
            "score_scope": (
                "per-edge permutation-invariant mean-head scores on locked receivers"
            ),
            "logit_gauge": "receiver/head centered before mean-head reduction",
            **fixed_edge_stability_report,
        },
        "mutual_attention_routing_stability": {
            "definition": (
                "minimum reciprocal receiver-degree-adjusted mean-head attention"
            ),
            "selection": "union of each seed's top 100 pairs within each core",
            "per_core": mutual_report,
            "summary_table": "tables/mutual_pair_stability.parquet",
            "causal_influence": False,
        },
        "selected_gradient_stability": {
            "request_count": EXPECTED_GRADIENT_REQUEST_COUNT,
            "attention_derivative": attention_gradient_report,
            "prediction_derivative": prediction_gradient_report,
            "selection_uses_model_outputs": False,
            "exhaustive_jacobian": False,
            "probe_scope": "24 sparse prespecified requests, not representative of all edges or genes",
            "support_definition": "absolute gradient strictly greater than zero",
            "tiny_finite_nonzero_gradients_count_as_support": True,
            "source_derivative_unit": "one standardized-log1p expression unit",
            "prediction_derivative_unit": (
                "standardized-log1p target prediction units per "
                "standardized-log1p source expression unit"
            ),
        },
        "limitations": [
            "Four model seeds quantify ensemble spread, not a calibrated confidence interval.",
            "All diagnostics are held-in and transductive; they do not estimate generalization.",
            "Attention is computational routing and is not direct signaling or causality.",
            "Gradients are local, scale-dependent model sensitivities and are not causality.",
            "Head alignment is derived from fixed held-in probes and may not be unique.",
            "The axial orientation representation cannot distinguish opposite polarity.",
            "The 24 selected gradient probes are sparse and are not representative of all edges or genes.",
            "Any finite nonzero gradient, however small, counts as support under the locked protocol.",
            "Models stopped at different plateau epochs, so exposure duration can contribute to observed seed spread.",
        ],
    }

    tables = {
        "training_curves": pa.Table.from_pylist(training_rows),
        "fixed_metrics": pa.Table.from_pylist(fixed_metric_rows),
        "fixed_metric_summary": pa.Table.from_pylist(fixed_metric_summary_rows),
        "fixed_probe_edges": _fixed_edge_table(ordered),
        "fixed_edge_summary": fixed_edge_stability_table,
        "mutual_pair_stability": pa.Table.from_pylist(mutual_rows),
        "selected_gradients": pa.Table.from_pylist(raw_gradient_rows),
        "selected_gradient_stability": pa.Table.from_pylist(
            attention_gradient_rows + prediction_gradient_rows
        ),
        "node_identity": pa.table(
            {
                "node_identity": list(ordered[0].embedding_ids),
            }
        ),
    }
    arrays = {
        "node_embeddings": {
            "node_identity": np.asarray(ordered[0].embedding_ids, dtype="U32"),
            **{
                f"seed_{value.seed}": np.asarray(value.embeddings, dtype=np.float32)
                for value in ordered
            },
        },
        "embedding_stability": {
            "seeds": np.asarray(embedding_report.seeds, dtype=np.int64),
            "linear_cka": embedding_report.linear_cka,
            "orthogonal_procrustes_similarity": (
                embedding_report.orthogonal_procrustes_similarity
            ),
        },
        "attention_head_stability": {
            "seeds": np.asarray(head_alignment.seeds, dtype=np.int64),
            "reference_to_seed_head": head_alignment.reference_to_seed_head,
            "signature_spearman": head_alignment.signature_spearman,
            "matched_signature_spearman": head_alignment.matched_signature_spearman,
            "matched_attention_spearman": (
                attention_stability.matched_head_spearman
            ),
            "matched_attention_top_edge_jaccard": (
                attention_stability.matched_head_top_edge_jaccard
            ),
            "matched_positional_bias_spearman": bias_stability.matched_head_spearman,
            "content_position_spearman": contribution.within_seed_spearman,
            "content_position_same_sign_fraction": contribution.same_sign_fraction,
            "content_absolute_fraction": contribution.content_absolute_fraction,
        },
        "selected_gradient_stability": {
            **attention_gradient_arrays,
            **prediction_gradient_arrays,
        },
    }
    return FourSeedAnalysisResult(
        report=_json_safe(report),
        tables=tables,
        arrays=arrays,
    )


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the canonical JSON report."""

    training = _mapping(report.get("training_loss"), "report training loss")
    training_summary = _mapping(training.get("final_summary"), "training summary")
    fixed = _mapping(report.get("fixed_held_in_metrics"), "fixed metrics")
    fixed_summary = _mapping(fixed.get("summary"), "fixed metric summary")
    embedding = _mapping(report.get("embedding_stability"), "embedding stability")
    attention = _mapping(
        report.get("attention_head_stability"), "attention stability"
    )
    mutual = _mapping(
        report.get("mutual_attention_routing_stability"), "mutual stability"
    )
    gradients = _mapping(
        report.get("selected_gradient_stability"), "gradient stability"
    )
    lines = [
        "# Four-seed Relative-QKV stability report",
        "",
        "This is a **four-seed ensemble-spread** analysis over model seeds "
        "0, 1, 2, and 3. Seed 4 and five-seed campaign completion remain deferred. "
        "The spread is not a calibrated biological confidence interval.",
        "",
        "All computations are held-in and transductive. Attention, gradients, and "
        "Jacobians are model-derived quantities and do not establish direct "
        "signaling, biological mechanism, or causality.",
        "",
        "## Members and plateau epochs",
        "",
        "| Seed | Run ID | Final epoch | Checkpoint SHA-256 |",
        "| ---: | --- | ---: | --- |",
    ]
    for member in report["members"]:
        lines.append(
            f"| {member['seed']} | `{member['run_id']}` | "
            f"{member['completed_global_epochs']} | "
            f"`{member['checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Different final epochs are permitted because every seed independently "
            "satisfied the locked training-loss plateau rule. No validation or test "
            "metric selected a checkpoint.",
            "",
            "## Final training loss",
            "",
            "Metric: equal-core mean masked Huber over the training masks.",
            "",
            "| Mean | Sample SD | Minimum | Maximum | Range |",
            "| ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {training_summary['mean']:.8g} | "
                f"{training_summary['sample_standard_deviation']:.8g} | "
                f"{training_summary['minimum']:.8g} | "
                f"{training_summary['maximum']:.8g} | "
                f"{training_summary['range']:.8g} |"
            ),
            "",
            "Full per-seed numeric curves are in `tables/training_curves.parquet`.",
            "",
            "## Fixed held-in fit diagnostics",
            "",
            "These are diagnostics on identical fixed masks, not validation or test "
            "evaluation.",
            "",
            "| Metric | Mean | Sample SD | Minimum | Maximum |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, raw in fixed_summary.items():
        values = _mapping(raw, f"fixed summary {name}")
        lines.append(
            f"| `{name}` | {values['mean']:.8g} | "
            f"{values['sample_standard_deviation']:.8g} | "
            f"{values['minimum']:.8g} | {values['maximum']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Representation and attention stability",
            "",
            f"- Full aligned final-node embeddings: "
            f"{report['fixed_input_identity']['node_count']} cells.",
            f"- Linear CKA matrix: `{embedding['linear_cka']}`.",
            "- Orthogonal Procrustes similarities are recorded in "
            "`arrays/embedding_stability.npz`.",
            f"- Attention heads were aligned to seed 0 using "
            f"{attention['matching_method']}.",
            f"- Matched top-edge overlap uses fraction "
            f"{attention['head_top_edge_fraction']} "
            f"({attention['head_top_edge_count']} fixed edges per head).",
            "- Matched attention Spearman/Jaccard, positional-bias correlations, and "
            "content-versus-bias rank/sign/magnitude statistics are preserved in "
            "`arrays/attention_head_stability.npz` and `report.json`.",
            "- Per-edge four-seed mean-head attention and receiver-centered "
            "content/bias/combined summaries are in "
            "`tables/fixed_edge_summary.parquet`.",
            "",
            "## Mutual routing and selected gradients",
            "",
            f"- Mutual routing selection: {mutual['selection']}.",
            "- The mutual score is descriptive reciprocal degree-adjusted attention, "
            "not causal influence.",
            f"- Selected gradient requests: {gradients['request_count']}; selection "
            "was independent of model outputs.",
            "- Both attention and prediction derivatives have mean, sample SD, "
            "median, min/max, empirical quantiles, support, sign consistency, and "
            "pairwise seed Spearman statistics.",
            "- No exhaustive Jacobian was materialized.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in report["limitations"])
    return "\n".join(lines) + "\n"


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite JSON: {path}")
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return files


def _atomic_publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing a concurrent destination."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise FourSeedStabilityError(
            "Atomic no-replace directory publication is unavailable."
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_no_replace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_no_replace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Refusing to replace a concurrently created analysis bundle",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def write_four_seed_analysis_bundle(
    result: FourSeedAnalysisResult,
    destination: str | Path,
    *,
    provenance_files: Mapping[str, Path],
    provenance_expected_sha256: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Atomically publish a new compact report bundle and refuse overwrite."""

    output = Path(destination).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        table_dir = staging / "tables"
        array_dir = staging / "arrays"
        provenance_dir = staging / "provenance"
        table_dir.mkdir()
        array_dir.mkdir()
        provenance_dir.mkdir()
        _write_json_new(staging / "report.json", result.report)
        (staging / "report.md").write_text(
            render_report_markdown(result.report),
            encoding="utf-8",
        )
        for name, table in sorted(result.tables.items()):
            path = table_dir / f"{name}.parquet"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite table: {path}")
            pq.write_table(table, path, compression="zstd")
        for name, arrays in sorted(result.arrays.items()):
            write_deterministic_npz(array_dir / f"{name}.npz", arrays)
        if provenance_expected_sha256 is not None and set(
            provenance_expected_sha256
        ) != set(provenance_files):
            raise FourSeedStabilityError(
                "Expected provenance checksums must cover every copied file exactly."
            )
        for name, source in sorted(provenance_files.items()):
            if not name or Path(name).name != name:
                raise FourSeedStabilityError(
                    f"Provenance output name must be a basename: {name!r}."
                )
            resolved = Path(source).expanduser().resolve(strict=True)
            expected_sha256 = (
                None
                if provenance_expected_sha256 is None
                else provenance_expected_sha256[name]
            )
            if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
                raise FourSeedStabilityError(
                    f"Validated provenance source changed before publication: {name}."
                )
            target = provenance_dir / name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite provenance: {target}")
            shutil.copyfile(resolved, target)
            if expected_sha256 is not None and sha256_file(target) != expected_sha256:
                raise FourSeedStabilityError(
                    f"Published provenance copy changed during publication: {name}."
                )
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "analysis_schema": result.report["schema"],
            "status": result.report["status"],
            "campaign_id": result.report["campaign_id"],
            "active_model_seeds": list(ACTIVE_SEEDS),
            "deferred_model_seeds": list(DEFERRED_SEEDS),
            "spread_label": "four-seed ensemble spread",
            "calibrated_biological_confidence_interval": False,
            "files": _file_manifest(staging),
        }
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
        _write_json_new(staging / "manifest.json", manifest)
        staging_verification = verify_four_seed_analysis_bundle(staging)
        if staging_verification["manifest_content_sha256"] != manifest[
            "manifest_content_sha256"
        ]:
            raise FourSeedStabilityError(
                "Staging verification returned a different manifest checksum."
            )
        _atomic_publish_directory_no_replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_four_seed_analysis_bundle(
    bundle_path: str | Path,
) -> Mapping[str, Any]:
    """Semantically verify every required four-seed report artifact."""

    root = Path(bundle_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FourSeedStabilityError("Analysis bundle must be a directory.")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise FourSeedStabilityError("Analysis bundle may not contain symbolic links.")
    manifest = _load_json(root / "manifest.json", "analysis manifest")
    content = dict(manifest)
    observed_content_sha256 = str(content.pop("manifest_content_sha256", ""))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("analysis_schema") != ANALYSIS_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or tuple(manifest.get("active_model_seeds", ())) != ACTIVE_SEEDS
        or tuple(manifest.get("deferred_model_seeds", ())) != DEFERRED_SEEDS
        or not hmac.compare_digest(
            observed_content_sha256,
            canonical_sha256(content),
        )
    ):
        raise FourSeedStabilityError("Analysis manifest identity/self-hash is invalid.")
    expected_files = _mapping(manifest.get("files"), "analysis manifest files")
    observed_files = _file_manifest(root)
    if expected_files != observed_files:
        raise FourSeedStabilityError("Analysis bundle file checksums or sizes drifted.")
    required_table_names = (
        "training_curves",
        "fixed_metrics",
        "fixed_metric_summary",
        "fixed_probe_edges",
        "fixed_edge_summary",
        "mutual_pair_stability",
        "selected_gradients",
        "selected_gradient_stability",
        "node_identity",
    )
    required_array_names = (
        "node_embeddings",
        "embedding_stability",
        "attention_head_stability",
        "selected_gradient_stability",
    )
    required_provenance_names = {
        "analysis_protocol_selected_gradient_stability_v1.yaml",
        "analysis_protocol_selected_gradient_stability_v1.sha256",
        "selected_gradient_requests_v1.csv",
        "selected_gradient_requests_v1.sha256",
        "cohort_manifest.json",
        "graph_manifest.json",
        "analysis_git_dirty.diff",
        "analysis_execution_provenance.json",
        *(f"analysis_source_{Path(source).name}" for source in ANALYSIS_SOURCE_FILES),
        *(
            f"seed_{seed}_strict_checkpoint_verification.json"
            for seed in ACTIVE_SEEDS
        ),
    }
    required_files = {
        "report.json",
        "report.md",
        *(f"tables/{name}.parquet" for name in required_table_names),
        *(f"arrays/{name}.npz" for name in required_array_names),
        *(f"provenance/{name}" for name in required_provenance_names),
    }
    if set(observed_files) != required_files:
        missing = sorted(required_files - set(observed_files))
        extra = sorted(set(observed_files) - required_files)
        raise FourSeedStabilityError(
            f"Analysis bundle required-file schema drifted; missing={missing}, extra={extra}."
        )
    locked_provenance = {
        "analysis_protocol_selected_gradient_stability_v1.yaml": (
            LOCKED_ANALYSIS_PROTOCOL_SHA256
        ),
        "selected_gradient_requests_v1.csv": (
            LOCKED_GRADIENT_REQUEST_TABLE_SHA256
        ),
    }
    for name, expected_sha256 in locked_provenance.items():
        if observed_files[f"provenance/{name}"]["sha256"] != expected_sha256:
            raise FourSeedStabilityError(f"Locked provenance bytes drifted: {name}.")
    expected_sidecars = {
        "analysis_protocol_selected_gradient_stability_v1.sha256": (
            f"{LOCKED_ANALYSIS_PROTOCOL_SHA256}  "
            "analysis_protocol_selected_gradient_stability_v1.yaml\n"
        ),
        "selected_gradient_requests_v1.sha256": (
            f"{LOCKED_GRADIENT_REQUEST_TABLE_SHA256}  "
            "selected_gradient_requests_v1.csv\n"
        ),
    }
    for name, expected_text in expected_sidecars.items():
        if (root / "provenance" / name).read_text(encoding="ascii") != expected_text:
            raise FourSeedStabilityError(f"Locked checksum sidecar drifted: {name}.")
    report = _load_json(root / "report.json", "analysis report")
    if (
        report.get("schema") != ANALYSIS_SCHEMA
        or report.get("status") != manifest.get("status")
        or report.get("campaign_id") != manifest.get("campaign_id")
        or tuple(report.get("active_model_seeds", ())) != ACTIVE_SEEDS
        or tuple(report.get("deferred_model_seeds", ())) != DEFERRED_SEEDS
        or report.get("five_seed_campaign_completion_claim_allowed") is not False
        or report.get("calibrated_biological_confidence_interval") is not False
        or report.get("generalization_claim_supported") is not False
        or report.get("causal_claim_supported") is not False
    ):
        raise FourSeedStabilityError("Analysis report claim scope/identity drifted.")

    raw_members = report.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != EXPECTED_SEED_COUNT:
        raise FourSeedStabilityError("Analysis report must contain four members.")
    members = tuple(
        _mapping(value, f"analysis member {index}")
        for index, value in enumerate(raw_members)
    )
    if tuple(int(member.get("seed", -1)) for member in members) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Analysis member seed order drifted.")
    final_epochs: list[int] = []
    run_ids: list[str] = []
    copied_masks_by_seed: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    core_node_counts: dict[str, int] | None = None
    for seed, member in zip(ACTIVE_SEEDS, members, strict=True):
        final_epoch = int(member.get("completed_global_epochs", 0))
        run_id = str(member.get("run_id", ""))
        if (
            final_epoch < PLATEAU_FIRST_ALLOWED_STOP_EPOCH
            or final_epoch % 25 != 0
            or not run_id
            or not _is_lower_sha256(member.get("checkpoint_sha256"))
            or not _is_lower_sha256(member.get("model_state_sha256"))
            or not _is_lower_sha256(member.get("history_sha256"))
            or not _is_lower_sha256(member.get("strict_receipt_file_sha256"))
            or not _is_lower_sha256(member.get("strict_receipt_content_sha256"))
            or int(member.get("parameter_count", 0)) != LOCKED_PARAMETER_COUNT
        ):
            raise FourSeedStabilityError(f"Seed {seed} member contract drifted.")
        receipt_name = f"seed_{seed}_strict_checkpoint_verification.json"
        receipt_relative = f"provenance/{receipt_name}"
        if observed_files[receipt_relative]["sha256"] != member.get(
            "strict_receipt_file_sha256"
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} copied strict-receipt checksum drifted."
            )
        copied_receipt = _load_json(
            root / receipt_relative, f"seed {seed} copied strict receipt"
        )
        if (
            _validate_receipt_content(copied_receipt, seed=seed)
            != member.get("strict_receipt_content_sha256")
            or copied_receipt.get("run_id") != run_id
            or _mapping(
                copied_receipt.get("checkpoint"),
                f"seed {seed} copied receipt checkpoint",
            ).get("file_sha256")
            != member.get("checkpoint_sha256")
            or int(
                _mapping(
                    copied_receipt.get("checkpoint"),
                    f"seed {seed} copied receipt checkpoint",
                ).get("completed_global_epochs", 0)
            )
            != final_epoch
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} copied strict-receipt content drifted."
            )
        held_in_replay = _mapping(
            copied_receipt.get("held_in_fit_replay"),
            f"seed {seed} copied held-in replay",
        )
        raw_core_rows = held_in_replay.get("per_core")
        if (
            held_in_replay.get("status") != "passed"
            or held_in_replay.get("role")
            != "held_in_fit_diagnostic_not_validation_or_test"
            or held_in_replay.get("fixed_masks_identical_across_reloads") is not True
            or not isinstance(raw_core_rows, list)
            or len(raw_core_rows) != len(CANCER_ALIASES)
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} copied held-in replay scope drifted."
            )
        seed_masks: dict[str, Mapping[str, Any]] = {}
        seed_node_counts: dict[str, int] = {}
        for alias, raw_core in zip(CANCER_ALIASES, raw_core_rows, strict=True):
            core_row = _mapping(
                raw_core, f"seed {seed} copied held-in core {alias}"
            )
            if core_row.get("alias") != alias:
                raise FourSeedStabilityError(
                    f"Seed {seed} copied held-in aliases drifted."
                )
            _validate_core_replay_receipt(core_row, seed=seed, alias=alias)
            mask_seed = core_row.get("mask_seed")
            masked_entries = core_row.get("n_masked_entries")
            mask_checksum = core_row.get("mask_checksum")
            if (
                isinstance(mask_seed, bool)
                or not isinstance(mask_seed, (int, np.integer))
                or int(mask_seed) < 0
                or isinstance(masked_entries, bool)
                or not isinstance(masked_entries, (int, np.integer))
                or int(masked_entries) <= 0
                or not _is_lower_sha256(mask_checksum)
            ):
                raise FourSeedStabilityError(
                    f"Seed {seed} {alias} copied fixed-mask receipt drifted."
                )
            seed_masks[alias] = {
                "effective_seed": int(mask_seed),
                "checksum_sha256": str(mask_checksum),
                "masked_entry_count": int(masked_entries),
            }
            seed_node_counts[alias] = int(core_row["n_nodes"])
        copied_masks_by_seed[seed] = seed_masks
        if core_node_counts is None:
            core_node_counts = seed_node_counts
        elif seed_node_counts != core_node_counts:
            raise FourSeedStabilityError(
                "Copied strict receipts disagree on per-core node counts."
            )
        final_epochs.append(final_epoch)
        run_ids.append(run_id)
    if len(set(run_ids)) != EXPECTED_SEED_COUNT:
        raise FourSeedStabilityError("Analysis member run IDs are not distinct.")
    shared_training = _mapping(
        report.get("shared_model_and_training_schedule"),
        "shared model/training schedule",
    )
    if (
        int(shared_training.get("parameter_count", 0)) != LOCKED_PARAMETER_COUNT
        or _mapping(
            shared_training.get("model_construction"), "shared model construction"
        )
        != LOCKED_MODEL_CONSTRUCTION
        or len({member.get("model_construction_sha256") for member in members}) != 1
        or shared_training.get("model_construction_sha256")
        != members[0].get("model_construction_sha256")
        or int(shared_training.get("shared_prefix_epochs", 0)) != min(final_epochs)
    ):
        raise FourSeedStabilityError("Shared model/training schedule report drifted.")

    fixed_identity = _mapping(
        report.get("fixed_input_identity"), "fixed-input identity"
    )
    node_count = int(fixed_identity.get("node_count", 0))
    fixed_edge_count = int(fixed_identity.get("fixed_directed_edge_count", 0))
    if (
        node_count != LOCKED_NODE_COUNT
        or fixed_edge_count <= 0
        or fixed_identity.get("receiver_probes_per_core")
        != RECEIVER_PROBES_PER_CORE
        or tuple(fixed_identity.get("core_aliases", ())) != CANCER_ALIASES
        or not _is_lower_sha256(fixed_identity.get("node_identity_sha256"))
        or not _is_lower_sha256(fixed_identity.get("fixed_edge_identity_sha256"))
    ):
        raise FourSeedStabilityError("Fixed input identity contract drifted.")
    if core_node_counts is None or sum(core_node_counts.values()) != node_count:
        raise FourSeedStabilityError(
            "Copied strict-receipt node counts do not match fixed-input identity."
        )
    reported_fixed_masks = _mapping(
        fixed_identity.get("fixed_masks"), "reported fixed masks"
    )
    reference_fixed_masks = copied_masks_by_seed[REFERENCE_SEED]
    if (
        tuple(reported_fixed_masks) != CANCER_ALIASES
        or reported_fixed_masks != reference_fixed_masks
        or any(
            copied_masks_by_seed[seed] != reference_fixed_masks
            for seed in ACTIVE_SEEDS[1:]
        )
    ):
        raise FourSeedStabilityError(
            "Copied strict receipts/report do not share identical fixed masks."
        )
    differing_epochs = _mapping(
        report.get("differing_final_epochs"), "differing final epochs"
    )
    if (
        differing_epochs.get("allowed") is not True
        or differing_epochs.get("common_final_epoch_required") is not False
        or differing_epochs.get("exposure_duration_can_contribute_to_seed_spread")
        is not True
        or {
            str(seed): final_epochs[seed]
            for seed in ACTIVE_SEEDS
        }
        != differing_epochs.get("epochs_by_seed")
    ):
        raise FourSeedStabilityError("Differing-plateau-epoch disclosure drifted.")
    selected_gradient_report = _mapping(
        report.get("selected_gradient_stability"), "selected-gradient report"
    )
    if (
        selected_gradient_report.get("request_count")
        != EXPECTED_GRADIENT_REQUEST_COUNT
        or selected_gradient_report.get("selection_uses_model_outputs") is not False
        or selected_gradient_report.get("exhaustive_jacobian") is not False
        or selected_gradient_report.get("tiny_finite_nonzero_gradients_count_as_support")
        is not True
        or selected_gradient_report.get("support_definition")
        != "absolute gradient strictly greater than zero"
        or "standardized-log1p" not in str(
            selected_gradient_report.get("prediction_derivative_unit")
        )
    ):
        raise FourSeedStabilityError("Selected-gradient scope disclosure drifted.")
    fixed_edge_report = _mapping(
        report.get("fixed_probe_edge_stability"), "fixed-edge report"
    )
    if (
        fixed_edge_report.get("summary_table")
        != "tables/fixed_edge_summary.parquet"
        or int(fixed_edge_report.get("row_count", 0)) != fixed_edge_count
        or fixed_edge_report.get("standard_deviation_ddof") != 1
        or fixed_edge_report.get("quantile_method")
        != "numpy linear interpolation"
    ):
        raise FourSeedStabilityError("Fixed-edge report schema drifted.")

    execution = _mapping(report.get("execution"), "analysis execution")
    if (
        execution.get("deterministic_seed") != ANALYSIS_DETERMINISTIC_SEED
        or execution.get("deterministic_algorithms_enabled") is not True
        or execution.get("deterministic_warn_only") is not False
        or execution.get("cublas_workspace_config") != ":4096:8"
        or execution.get("matmul_tf32_enabled") is not False
        or execution.get("cudnn_tf32_enabled") is not False
        or execution.get("attention_and_embedding_replay_amp_enabled") is not True
        or execution.get("attention_and_embedding_replay_amp_dtype") != "float16"
        or execution.get("selected_derivative_replay_amp_enabled") is not False
        or execution.get("selected_derivative_replay_dtype") != "float32"
        or execution.get("receiver_chunk_size") != DEFAULT_RECEIVER_CHUNK_SIZE
        or execution.get("max_edges_per_chunk") != DEFAULT_MAX_EDGES_PER_CHUNK
        or execution.get("receiver_wise_softmax_exact") is not True
        or execution.get("neighbor_sampling") is not False
        or execution.get("models_processed_concurrently") != 1
        or execution.get("cores_staged_on_cuda_concurrently") != 1
        or execution.get("cuda_device_order") != "PCI_BUS_ID"
        or not isinstance(execution.get("cuda_visible_devices"), str)
        or "," in str(execution.get("cuda_visible_devices"))
        or execution.get("device") != "cuda:0"
    ):
        raise FourSeedStabilityError("Analysis numerical execution contract drifted.")
    gpu_binding = _mapping(execution.get("gpu_binding"), "analysis GPU binding")
    selected_gpu = _mapping(
        gpu_binding.get("selected_physical_gpu"), "selected analysis GPU"
    )
    after_cuda = gpu_binding.get("compute_applications_after_cuda")
    if (
        gpu_binding.get("logical_device") != "cuda:0"
        or gpu_binding.get("cuda_device_order") != "PCI_BUS_ID"
        or gpu_binding.get("cuda_visible_devices")
        != execution.get("cuda_visible_devices")
        or gpu_binding.get("compute_applications_before_cuda") != []
        or gpu_binding.get("torch_device_uuid") != selected_gpu.get("uuid")
        or execution.get("device_uuid") != selected_gpu.get("uuid")
        or not isinstance(after_cuda, list)
        or len(after_cuda) != 1
        or int(_mapping(after_cuda[0], "analysis GPU process").get("pid", -1))
        != int(gpu_binding.get("analysis_pid", -2))
        or _mapping(after_cuda[0], "analysis GPU process").get("gpu_uuid")
        != selected_gpu.get("uuid")
    ):
        raise FourSeedStabilityError("Analysis physical/logical GPU binding drifted.")
    source = _mapping(execution.get("source"), "analysis source provenance")
    if (
        source.get("git_dirty") is not False
        or source.get("git_status_porcelain_v1") != []
        or not isinstance(source.get("git_commit"), str)
        or len(str(source.get("git_commit"))) != 40
        or set(
            _mapping(
                source.get("analysis_source_files"),
                "analysis source files",
            )
        )
        != set(ANALYSIS_SOURCE_FILES)
    ):
        raise FourSeedStabilityError("Analysis source provenance is not clean/locked.")
    for relative, raw_record in _mapping(
        source.get("analysis_source_files"), "analysis source files"
    ).items():
        record = _mapping(raw_record, f"analysis source {relative}")
        copied_relative = f"provenance/analysis_source_{Path(relative).name}"
        if (
            record.get("sha256") != observed_files[copied_relative]["sha256"]
            or int(record.get("size_bytes", -1))
            != int(observed_files[copied_relative]["size_bytes"])
        ):
            raise FourSeedStabilityError(
                f"Analysis executing/captured source drifted: {relative}."
            )
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if (
        source.get("tracked_dirty_diff_sha256") != empty_sha256
        or observed_files["provenance/analysis_git_dirty.diff"]["sha256"]
        != empty_sha256
        or observed_files["provenance/analysis_git_dirty.diff"]["size_bytes"] != 0
    ):
        raise FourSeedStabilityError("Analysis dirty-diff provenance is not empty.")
    execution_copy = _load_json(
        root / "provenance/analysis_execution_provenance.json",
        "analysis execution provenance copy",
    )
    if execution_copy != execution:
        raise FourSeedStabilityError("Execution provenance copy/report drifted.")
    protocol_provenance = _mapping(report.get("protocol"), "report protocol provenance")
    if (
        "execution" in protocol_provenance
        or protocol_provenance.get("analysis_protocol_file_sha256")
        != LOCKED_ANALYSIS_PROTOCOL_SHA256
        or protocol_provenance.get("gradient_request_table_file_sha256")
        != LOCKED_GRADIENT_REQUEST_TABLE_SHA256
        or protocol_provenance.get("analysis_protocol_sidecar_file_sha256")
        != observed_files[
            "provenance/analysis_protocol_selected_gradient_stability_v1.sha256"
        ]["sha256"]
        or protocol_provenance.get("gradient_request_sidecar_file_sha256")
        != observed_files["provenance/selected_gradient_requests_v1.sha256"][
            "sha256"
        ]
        or protocol_provenance.get("cohort_manifest_file_sha256")
        != observed_files["provenance/cohort_manifest.json"]["sha256"]
        or protocol_provenance.get("graph_manifest_file_sha256")
        != observed_files["provenance/graph_manifest.json"]["sha256"]
    ):
        raise FourSeedStabilityError("Report protocol provenance drifted.")

    markdown = (root / "report.md").read_text(encoding="utf-8")
    for required_text in (
        "four-seed ensemble-spread",
        "Seed 4 and five-seed campaign completion remain deferred",
        "do not establish direct signaling",
        "held-in and transductive",
    ):
        if required_text not in markdown:
            raise FourSeedStabilityError(
                f"Human-readable report omits required disclosure: {required_text}."
            )

    def parquet_contract(
        name: str,
        *,
        expected_rows: int,
        required_columns: set[str],
        required_types: Mapping[str, pa.DataType] | None = None,
    ) -> pa.Schema:
        path = root / f"tables/{name}.parquet"
        try:
            metadata = pq.read_metadata(path)
            schema = pq.read_schema(path)
        except (OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise FourSeedStabilityError(f"Cannot verify table {name}.") from exc
        if metadata.num_rows != expected_rows or not required_columns.issubset(
            set(schema.names)
        ):
            raise FourSeedStabilityError(
                f"Table {name} row count or columns drifted."
            )
        type_contract = dict(required_types or {})
        if set(type_contract) - required_columns:
            raise FourSeedStabilityError(
                f"Internal table type contract for {name} is inconsistent."
            )
        for column, expected_type in type_contract.items():
            if not schema.field(column).type.equals(expected_type):
                raise FourSeedStabilityError(
                    f"Table {name} column {column} has an invalid Arrow type."
                )
        try:
            required_table = pq.read_table(path, columns=sorted(required_columns))
        except (OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise FourSeedStabilityError(
                f"Cannot read required columns from table {name}."
            ) from exc
        null_columns = [
            column
            for column in required_table.column_names
            if required_table[column].null_count != 0
        ]
        if null_columns:
            raise FourSeedStabilityError(
                f"Table {name} contains null required columns: {null_columns}."
            )
        return schema

    def numeric_column(table: pa.Table, name: str) -> np.ndarray:
        values = np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise FourSeedStabilityError(f"Table column {name} is nonfinite.")
        return values

    training_row_count = sum(final_epochs)
    parquet_contract(
        "training_curves",
        expected_rows=training_row_count,
        required_columns={
            "seed",
            "run_id",
            "global_epoch",
            "completed_global_epoch",
            "equal_core_mean_masked_huber",
            "final_plateau_epoch",
        },
        required_types={
            "seed": pa.int64(),
            "run_id": pa.string(),
            "global_epoch": pa.int64(),
            "completed_global_epoch": pa.int64(),
            "equal_core_mean_masked_huber": pa.float64(),
            "final_plateau_epoch": pa.int64(),
        },
    )
    training_table = pq.read_table(
        root / "tables/training_curves.parquet",
        columns=[
            "seed",
            "run_id",
            "global_epoch",
            "completed_global_epoch",
            "equal_core_mean_masked_huber",
            "final_plateau_epoch",
        ],
    )
    training_seeds = np.asarray(
        training_table["seed"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    training_report = _mapping(report.get("training_loss"), "training report")
    final_by_seed = _mapping(training_report.get("final_by_seed"), "final losses")
    final_losses: list[float] = []
    for seed in ACTIVE_SEEDS:
        selected = np.flatnonzero(training_seeds == seed)
        if len(selected) != final_epochs[seed]:
            raise FourSeedStabilityError("Training-curve per-seed lengths drifted.")
        seed_table = training_table.take(pa.array(selected))
        if (
            tuple(seed_table["run_id"].to_pylist())
            != (run_ids[seed],) * final_epochs[seed]
            or tuple(seed_table["global_epoch"].to_pylist())
            != tuple(range(final_epochs[seed]))
            or tuple(seed_table["completed_global_epoch"].to_pylist())
            != tuple(range(1, final_epochs[seed] + 1))
            or set(seed_table["final_plateau_epoch"].to_pylist())
            != {final_epochs[seed]}
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} training-curve identity/order drifted."
            )
        losses = numeric_column(seed_table, "equal_core_mean_masked_huber")
        final_loss = float(losses[-1])
        if not math.isclose(
            final_loss,
            _finite_float(final_by_seed.get(str(seed)), f"seed {seed} final loss"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FourSeedStabilityError(f"Seed {seed} final loss drifted.")
        final_losses.append(final_loss)
    expected_final_summary = scalar_summary(final_losses)
    observed_final_summary = _mapping(
        training_report.get("final_summary"), "final-loss summary"
    )
    for key in (
        "mean",
        "sample_standard_deviation",
        "median",
        "minimum",
        "maximum",
        "range",
    ):
        if not math.isclose(
            _finite_float(observed_final_summary.get(key), f"final summary {key}"),
            float(expected_final_summary[key]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FourSeedStabilityError(f"Final-loss {key} is not reproducible.")
    parquet_contract(
        "fixed_metrics",
        expected_rows=EXPECTED_SEED_COUNT,
        required_columns={"seed", "run_id", "completed_global_epochs", *LOCKED_FIXED_METRIC_NAMES},
        required_types={
            "seed": pa.int64(),
            "run_id": pa.string(),
            "completed_global_epochs": pa.int64(),
            **{name: pa.float64() for name in LOCKED_FIXED_METRIC_NAMES},
        },
    )
    parquet_contract(
        "fixed_metric_summary",
        expected_rows=len(LOCKED_FIXED_METRIC_NAMES),
        required_columns={
            "metric",
            "mean",
            "sample_standard_deviation",
            "median",
            "minimum",
            "maximum",
            "range",
            "empirical_quantiles",
        },
        required_types={
            "metric": pa.string(),
            "mean": pa.float64(),
            "sample_standard_deviation": pa.float64(),
            "median": pa.float64(),
            "minimum": pa.float64(),
            "maximum": pa.float64(),
            "range": pa.float64(),
            "empirical_quantiles": pa.struct(
                [(f"{level:g}", pa.float64()) for level in QUANTILE_LEVELS]
            ),
        },
    )
    fixed_metric_table = pq.read_table(root / "tables/fixed_metrics.parquet")
    if tuple(
        int(value)
        for value in fixed_metric_table["seed"].combine_chunks().to_pylist()
    ) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Fixed-metric seed order drifted.")
    if (
        tuple(fixed_metric_table["run_id"].to_pylist()) != tuple(run_ids)
        or tuple(
            int(value)
            for value in fixed_metric_table[
                "completed_global_epochs"
            ].combine_chunks().to_pylist()
        )
        != tuple(final_epochs)
    ):
        raise FourSeedStabilityError("Fixed-metric member identity drifted.")
    held_in_metric_report = _mapping(
        report.get("fixed_held_in_metrics"), "fixed held-in metric report"
    )
    reported_per_seed = held_in_metric_report.get("per_seed")
    if (
        held_in_metric_report.get("role")
        != "held_in_fit_diagnostic_not_validation_or_test"
        or not isinstance(reported_per_seed, list)
        or len(reported_per_seed) != EXPECTED_SEED_COUNT
    ):
        raise FourSeedStabilityError("Fixed held-in metric report scope drifted.")
    for row_index, (table_row, reported_row) in enumerate(
        zip(
            fixed_metric_table.to_pylist(),
            reported_per_seed,
            strict=True,
        )
    ):
        report_row = _mapping(reported_row, f"fixed metric report row {row_index}")
        if any(
            (
                not math.isclose(
                    _finite_float(report_row.get(name), name),
                    _finite_float(table_row.get(name), name),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                if name in LOCKED_FIXED_METRIC_NAMES
                else report_row.get(name) != table_row.get(name)
            )
            for name in (
                "seed",
                "run_id",
                "completed_global_epochs",
                *LOCKED_FIXED_METRIC_NAMES,
            )
        ):
            raise FourSeedStabilityError(
                "Fixed held-in per-seed report/table values drifted."
            )
    fixed_summary_rows = pq.read_table(
        root / "tables/fixed_metric_summary.parquet"
    ).to_pylist()
    if tuple(str(row["metric"]) for row in fixed_summary_rows) != LOCKED_FIXED_METRIC_NAMES:
        raise FourSeedStabilityError("Fixed-metric summary order drifted.")
    for metric, summary_row in zip(
        LOCKED_FIXED_METRIC_NAMES, fixed_summary_rows, strict=True
    ):
        values = numeric_column(fixed_metric_table, metric)
        expected = scalar_summary(values)
        for key in (
            "mean",
            "sample_standard_deviation",
            "median",
            "minimum",
            "maximum",
            "range",
        ):
            if not math.isclose(
                float(summary_row[key]),
                float(expected[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise FourSeedStabilityError(
                    f"Fixed-metric {metric} {key} is not reproducible."
                )
        observed_quantiles = _mapping(
            summary_row.get("empirical_quantiles"),
            f"fixed-metric {metric} quantiles",
        )
        if any(
            not math.isclose(
                float(observed_quantiles[level]),
                float(expected["empirical_quantiles"][level]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for level in expected["empirical_quantiles"]
        ):
            raise FourSeedStabilityError(
                f"Fixed-metric {metric} quantiles are not reproducible."
            )
        reported_summary = _mapping(
            _mapping(
                held_in_metric_report.get("summary"),
                "fixed held-in metric summary",
            ).get(metric),
            f"fixed held-in metric summary {metric}",
        )
        for key, expected_value in expected.items():
            if key == "empirical_quantiles":
                observed_quantile_report = _mapping(
                    reported_summary.get(key),
                    f"fixed held-in metric report {metric} quantiles",
                )
                if any(
                    not math.isclose(
                        _finite_float(
                            observed_quantile_report.get(level),
                            f"fixed metric report {metric} quantile {level}",
                        ),
                        float(value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for level, value in expected_value.items()
                ):
                    raise FourSeedStabilityError(
                        f"Fixed held-in metric report {metric} quantiles drifted."
                    )
            elif isinstance(expected_value, (int, np.integer, bool, str)):
                if reported_summary.get(key) != expected_value:
                    raise FourSeedStabilityError(
                        f"Fixed held-in metric report {metric} {key} drifted."
                    )
            elif not math.isclose(
                _finite_float(
                    reported_summary.get(key),
                    f"fixed held-in metric report {metric} {key}",
                ),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise FourSeedStabilityError(
                    f"Fixed held-in metric report {metric} {key} drifted."
                )
    parquet_contract(
        "fixed_probe_edges",
        expected_rows=EXPECTED_SEED_COUNT * fixed_edge_count,
        required_columns={
            "seed",
            "fixed_edge_id",
            "core_alias",
            "edge_id",
            "source_node",
            "receiver_node",
            "attention_mean",
            "content_logit_mean",
            "positional_bias_mean",
            "combined_logit_mean",
            *(f"attention_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
            *(f"content_logit_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
            *(f"positional_bias_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
            *(f"combined_logit_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
        },
        required_types={
            "seed": pa.int8(),
            "fixed_edge_id": pa.string(),
            "core_alias": pa.string(),
            "edge_id": pa.int64(),
            "source_node": pa.int64(),
            "receiver_node": pa.int64(),
            "attention_mean": pa.float32(),
            "content_logit_mean": pa.float32(),
            "positional_bias_mean": pa.float32(),
            "combined_logit_mean": pa.float32(),
            **{
                f"{prefix}_head_{head:02d}": pa.float32()
                for prefix in (
                    "attention",
                    "content_logit",
                    "positional_bias",
                    "combined_logit",
                )
                for head in range(LOCKED_HEAD_COUNT)
            },
        },
    )
    fixed_score_names = (
        "attention_mean_head",
        "receiver_centered_content_logit_mean_head",
        "receiver_centered_positional_bias_mean_head",
        "receiver_centered_combined_logit_mean_head",
    )
    fixed_summary_required = {
        "fixed_edge_id",
        "core_alias",
        "edge_id",
        "source_node",
        "receiver_node",
    }
    for score_name in fixed_score_names:
        fixed_summary_required.update(
            {
                f"{score_name}_{field}"
                for field in (
                    "ensemble_mean",
                    "ensemble_standard_deviation",
                    "median",
                    "minimum",
                    "maximum",
                    "seed_support_count",
                    "empirical_quantile_0.05",
                    "empirical_quantile_0.25",
                    "empirical_quantile_0.75",
                    "empirical_quantile_0.95",
                    "support_definition",
                    "seed_0",
                    "seed_1",
                    "seed_2",
                    "seed_3",
                )
            }
        )
    parquet_contract(
        "fixed_edge_summary",
        expected_rows=fixed_edge_count,
        required_columns=fixed_summary_required,
        required_types={
            "fixed_edge_id": pa.string(),
            "core_alias": pa.string(),
            "edge_id": pa.int64(),
            "source_node": pa.int64(),
            "receiver_node": pa.int64(),
            **{
                f"{score_name}_{field}": (
                    pa.int64()
                    if field == "seed_support_count"
                    else pa.string()
                    if field == "support_definition"
                    else pa.float64()
                )
                for score_name in fixed_score_names
                for field in (
                    "ensemble_mean",
                    "ensemble_standard_deviation",
                    "median",
                    "minimum",
                    "maximum",
                    "seed_support_count",
                    "empirical_quantile_0.05",
                    "empirical_quantile_0.25",
                    "empirical_quantile_0.75",
                    "empirical_quantile_0.95",
                    "support_definition",
                    "seed_0",
                    "seed_1",
                    "seed_2",
                    "seed_3",
                )
            },
        },
    )
    fixed_summary_table = pq.read_table(root / "tables/fixed_edge_summary.parquet")

    for score_name in fixed_score_names:
        seed_values = np.stack(
            [
                numeric_column(fixed_summary_table, f"{score_name}_seed_{seed}")
                for seed in ACTIVE_SEEDS
            ],
            axis=0,
        )
        expected_statistics = {
            "ensemble_mean": seed_values.mean(axis=0),
            "ensemble_standard_deviation": seed_values.std(axis=0, ddof=1),
            "median": np.median(seed_values, axis=0),
            "minimum": seed_values.min(axis=0),
            "maximum": seed_values.max(axis=0),
        }
        quantile_values = np.quantile(
            seed_values,
            np.asarray(QUANTILE_LEVELS),
            axis=0,
            method="linear",
        )
        expected_statistics.update(
            {
                f"empirical_quantile_{level:g}": quantile_values[index]
                for index, level in enumerate(QUANTILE_LEVELS)
            }
        )
        for statistic, expected in expected_statistics.items():
            observed = numeric_column(
                fixed_summary_table, f"{score_name}_{statistic}"
            )
            if not np.allclose(observed, expected, atol=1e-12, rtol=1e-12):
                raise FourSeedStabilityError(
                    f"Fixed-edge {score_name} {statistic} is not reproducible."
                )
        if score_name == "attention_mean_head":
            top_k = max(
                1, int(math.ceil(fixed_edge_count * HEAD_TOP_EDGE_FRACTION))
            )
            support = np.zeros_like(seed_values, dtype=np.int64)
            for seed_index in range(EXPECTED_SEED_COUNT):
                order = np.lexsort(
                    (
                        np.arange(fixed_edge_count, dtype=np.int64),
                        -seed_values[seed_index],
                    )
                )
                support[seed_index, order[:top_k]] = 1
        else:
            support = (np.abs(seed_values) > 0.0).astype(np.int64)
        observed_support = numeric_column(
            fixed_summary_table, f"{score_name}_seed_support_count"
        )
        if not np.array_equal(observed_support.astype(np.int64), support.sum(axis=0)):
            raise FourSeedStabilityError(
                f"Fixed-edge {score_name} support counts are not reproducible."
            )

    fixed_edge_ids = tuple(fixed_summary_table["fixed_edge_id"].to_pylist())
    if (
        len(set(fixed_edge_ids)) != fixed_edge_count
        or canonical_sha256(fixed_edge_ids)
        != fixed_identity.get("fixed_edge_identity_sha256")
    ):
        raise FourSeedStabilityError(
            "Fixed-edge summary identities are not unique/checksum-bound."
        )
    raw_fixed_columns = [
        "seed",
        "fixed_edge_id",
        "core_alias",
        "edge_id",
        "source_node",
        "receiver_node",
        "attention_mean",
        "content_logit_mean",
        "positional_bias_mean",
        "combined_logit_mean",
        *(f"attention_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
        *(f"content_logit_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
        *(f"positional_bias_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
        *(f"combined_logit_head_{head:02d}" for head in range(LOCKED_HEAD_COUNT)),
    ]
    raw_fixed = pq.read_table(
        root / "tables/fixed_probe_edges.parquet", columns=raw_fixed_columns
    )
    raw_seed = np.asarray(
        raw_fixed["seed"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    raw_edge_records: dict[str, list[FixedSeedArray]] = {
        "attention": [],
        "content_logit": [],
        "positional_bias": [],
        "combined_logit": [],
    }
    centered_content_records: list[FixedSeedArray] = []
    centered_bias_records: list[FixedSeedArray] = []
    reference_receiver_groups: tuple[str, ...] | None = None
    for seed in ACTIVE_SEEDS:
        selected = np.flatnonzero(raw_seed == seed)
        if len(selected) != fixed_edge_count:
            raise FourSeedStabilityError(
                f"Seed {seed} raw fixed-edge row count drifted."
            )
        seed_table = raw_fixed.take(pa.array(selected))
        if tuple(seed_table["fixed_edge_id"].to_pylist()) != fixed_edge_ids:
            raise FourSeedStabilityError(
                f"Seed {seed} raw/summary fixed-edge identities drifted."
            )
        for identity_column in (
            "core_alias",
            "edge_id",
            "source_node",
            "receiver_node",
        ):
            if seed_table[identity_column].to_pylist() != fixed_summary_table[
                identity_column
            ].to_pylist():
                raise FourSeedStabilityError(
                    f"Seed {seed} raw/summary {identity_column} drifted."
                )
        core_values = tuple(str(value) for value in seed_table["core_alias"].to_pylist())
        edge_values = np.asarray(seed_table["edge_id"].to_numpy(), dtype=np.int64)
        source_values = np.asarray(
            seed_table["source_node"].to_numpy(), dtype=np.int64
        )
        receiver_values = np.asarray(
            seed_table["receiver_node"].to_numpy(), dtype=np.int64
        )
        if (
            any(alias not in CANCER_ALIASES for alias in core_values)
            or np.any(edge_values < 0)
            or np.any(source_values < 0)
            or np.any(receiver_values < 0)
            or any(
                int(source) >= int(core_node_counts[alias])
                or int(receiver) >= int(core_node_counts[alias])
                for alias, source, receiver in zip(
                    core_values,
                    source_values.tolist(),
                    receiver_values.tolist(),
                    strict=True,
                )
            )
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} raw fixed-edge index domain drifted."
            )
        receiver_groups = tuple(
            f"{core}:receiver:{int(receiver):08d}"
            for core, receiver in zip(
                seed_table["core_alias"].to_pylist(),
                seed_table["receiver_node"].to_pylist(),
                strict=True,
            )
        )
        if reference_receiver_groups is None:
            reference_receiver_groups = receiver_groups
        elif receiver_groups != reference_receiver_groups:
            raise FourSeedStabilityError(
                "Raw fixed-edge receiver groups drifted across seeds."
            )
        for alias in CANCER_ALIASES:
            alias_receivers = {
                receiver
                for core, receiver in zip(
                    core_values, receiver_values.tolist(), strict=True
                )
                if core == alias
            }
            if len(alias_receivers) != RECEIVER_PROBES_PER_CORE:
                raise FourSeedStabilityError(
                    f"Seed {seed} {alias} does not contain exactly 64 fixed receivers."
                )
        raw_channels: dict[str, np.ndarray] = {}
        for prefix in (
            "attention",
            "content_logit",
            "positional_bias",
            "combined_logit",
        ):
            raw_channels[prefix] = np.stack(
                [
                    numeric_column(seed_table, f"{prefix}_head_{head:02d}")
                    for head in range(LOCKED_HEAD_COUNT)
                ],
                axis=1,
            )
            raw_edge_records[prefix].append(
                FixedSeedArray(
                    seed=seed,
                    fixed_input_ids=fixed_edge_ids,
                    values=raw_channels[prefix],
                )
            )
        centered_content_records.append(
            FixedSeedArray(
                seed=seed,
                fixed_input_ids=fixed_edge_ids,
                values=receiver_centered_logits(
                    raw_channels["content_logit"], receiver_groups
                ),
            )
        )
        centered_bias_records.append(
            FixedSeedArray(
                seed=seed,
                fixed_input_ids=fixed_edge_ids,
                values=receiver_centered_logits(
                    raw_channels["positional_bias"], receiver_groups
                ),
            )
        )
        if np.any(raw_channels["attention"] < 0.0) or np.any(
            raw_channels["attention"] > 1.0
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} raw fixed-edge attention domain drifted."
            )
        group_array = np.asarray(receiver_groups)
        for group in np.unique(group_array):
            if not np.allclose(
                raw_channels["attention"][group_array == group].sum(axis=0),
                np.ones(LOCKED_HEAD_COUNT, dtype=np.float64),
                atol=1e-6,
                rtol=0.0,
            ):
                raise FourSeedStabilityError(
                    f"Seed {seed} raw fixed-edge attention is not receiver normalized."
                )
        if not np.allclose(
            raw_channels["combined_logit"],
            raw_channels["content_logit"] + raw_channels["positional_bias"],
            atol=1e-5,
            rtol=0.0,
        ):
            raise FourSeedStabilityError(
                f"Seed {seed} raw fixed-edge logit composition drifted."
            )
        for prefix, mean_column in (
            ("attention", "attention_mean"),
            ("content_logit", "content_logit_mean"),
            ("positional_bias", "positional_bias_mean"),
            ("combined_logit", "combined_logit_mean"),
        ):
            if not np.allclose(
                numeric_column(seed_table, mean_column),
                raw_channels[prefix].mean(axis=1),
                atol=1e-7,
                rtol=1e-7,
            ):
                raise FourSeedStabilityError(
                    f"Seed {seed} raw fixed-edge {mean_column} drifted."
                )
        expected_scores = {
            "attention_mean_head": raw_channels["attention"].mean(axis=1),
            "receiver_centered_content_logit_mean_head": receiver_centered_logits(
                raw_channels["content_logit"], receiver_groups
            ).mean(axis=1),
            "receiver_centered_positional_bias_mean_head": receiver_centered_logits(
                raw_channels["positional_bias"], receiver_groups
            ).mean(axis=1),
            "receiver_centered_combined_logit_mean_head": receiver_centered_logits(
                raw_channels["combined_logit"], receiver_groups
            ).mean(axis=1),
        }
        for score_name, expected in expected_scores.items():
            observed = numeric_column(
                fixed_summary_table, f"{score_name}_seed_{seed}"
            )
            if not np.allclose(observed, expected, atol=1e-12, rtol=1e-12):
                raise FourSeedStabilityError(
                    f"Seed {seed} raw/summary {score_name} drifted."
                )

    mutual_report = _mapping(
        report.get("mutual_attention_routing_stability"), "mutual report"
    )
    per_core_mutual = _mapping(mutual_report.get("per_core"), "mutual per-core")
    if tuple(per_core_mutual) != CANCER_ALIASES:
        raise FourSeedStabilityError("Mutual-pair core order drifted.")
    mutual_row_count = sum(
        int(_mapping(per_core_mutual[alias], alias).get("union_pair_count", 0))
        for alias in CANCER_ALIASES
    )
    if not 600 <= mutual_row_count <= 2400:
        raise FourSeedStabilityError("Mutual-pair union row count is invalid.")
    parquet_contract(
        "mutual_pair_stability",
        expected_rows=mutual_row_count,
        required_columns={
            "relationship_id",
            "score_name",
            "core_alias",
            "pair_position",
            "pair_key",
            "node_low",
            "node_high",
            "ensemble_mean",
            "ensemble_standard_deviation",
            "median",
            "minimum",
            "maximum",
            "seed_support_count",
            "empirical_quantile_0.05",
            "empirical_quantile_0.25",
            "empirical_quantile_0.75",
            "empirical_quantile_0.95",
            *(f"seed_{seed}_score" for seed in ACTIVE_SEEDS),
            *(f"seed_{seed}_top100_support" for seed in ACTIVE_SEEDS),
        },
        required_types={
            "relationship_id": pa.string(),
            "score_name": pa.string(),
            "core_alias": pa.string(),
            "pair_position": pa.int64(),
            "pair_key": pa.int64(),
            "node_low": pa.int64(),
            "node_high": pa.int64(),
            "ensemble_mean": pa.float64(),
            "ensemble_standard_deviation": pa.float64(),
            "median": pa.float64(),
            "minimum": pa.float64(),
            "maximum": pa.float64(),
            "seed_support_count": pa.int64(),
            **{
                f"empirical_quantile_{level:g}": pa.float64()
                for level in QUANTILE_LEVELS
            },
            **{f"seed_{seed}_score": pa.float64() for seed in ACTIVE_SEEDS},
            **{
                f"seed_{seed}_top100_support": pa.bool_()
                for seed in ACTIVE_SEEDS
            },
        },
    )
    mutual_table = pq.read_table(root / "tables/mutual_pair_stability.parquet")
    mutual_seed_values = np.stack(
        [
            numeric_column(mutual_table, f"seed_{seed}_score")
            for seed in ACTIVE_SEEDS
        ],
        axis=0,
    )
    mutual_support = np.stack(
        [
            np.asarray(
                mutual_table[f"seed_{seed}_top100_support"]
                .combine_chunks()
                .to_numpy(zero_copy_only=False),
                dtype=np.bool_,
            )
            for seed in ACTIVE_SEEDS
        ],
        axis=0,
    )
    mutual_statistics = {
        "ensemble_mean": mutual_seed_values.mean(axis=0),
        "ensemble_standard_deviation": mutual_seed_values.std(axis=0, ddof=1),
        "median": np.median(mutual_seed_values, axis=0),
        "minimum": mutual_seed_values.min(axis=0),
        "maximum": mutual_seed_values.max(axis=0),
    }
    mutual_quantiles = np.quantile(
        mutual_seed_values,
        np.asarray(QUANTILE_LEVELS),
        axis=0,
        method="linear",
    )
    mutual_statistics.update(
        {
            f"empirical_quantile_{level:g}": mutual_quantiles[index]
            for index, level in enumerate(QUANTILE_LEVELS)
        }
    )
    for statistic, expected in mutual_statistics.items():
        if not np.allclose(
            numeric_column(mutual_table, statistic),
            expected,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise FourSeedStabilityError(
                f"Mutual-pair {statistic} is not reproducible."
            )
    observed_mutual_support = numeric_column(mutual_table, "seed_support_count")
    if not np.array_equal(
        observed_mutual_support.astype(np.int64), mutual_support.sum(axis=0)
    ):
        raise FourSeedStabilityError("Mutual-pair support counts are not reproducible.")
    mutual_aliases = np.asarray(mutual_table["core_alias"].to_pylist(), dtype="U6")
    mutual_relationship_ids = tuple(
        str(value) for value in mutual_table["relationship_id"].to_pylist()
    )
    mutual_score_names = tuple(
        str(value) for value in mutual_table["score_name"].to_pylist()
    )
    mutual_pair_positions = np.asarray(
        mutual_table["pair_position"].to_numpy(), dtype=np.int64
    )
    mutual_pair_keys = np.asarray(mutual_table["pair_key"].to_numpy(), dtype=np.int64)
    mutual_node_low = np.asarray(mutual_table["node_low"].to_numpy(), dtype=np.int64)
    mutual_node_high = np.asarray(mutual_table["node_high"].to_numpy(), dtype=np.int64)
    if (
        len(set(mutual_relationship_ids)) != mutual_row_count
        or any(
            name != "mutual_attention_routing_score"
            for name in mutual_score_names
        )
        or any(alias not in CANCER_ALIASES for alias in mutual_aliases.tolist())
        or np.any(mutual_pair_positions < 0)
        or np.any(mutual_pair_keys < 0)
        or np.any(mutual_node_low < 0)
        or np.any(mutual_node_high <= mutual_node_low)
    ):
        raise FourSeedStabilityError("Mutual-pair identity/index domain drifted.")
    for alias in CANCER_ALIASES:
        selected = mutual_aliases == alias
        selected_indices = np.flatnonzero(selected)
        alias_report = _mapping(per_core_mutual[alias], f"{alias} mutual report")
        if (
            len(selected_indices) != int(alias_report.get("union_pair_count", 0))
            or int(alias_report.get("top_k_per_seed", 0))
            != MUTUAL_TOP_K_PER_CORE_SEED
            or len(np.unique(mutual_pair_positions[selected])) != len(selected_indices)
            or len(np.unique(mutual_pair_keys[selected])) != len(selected_indices)
            or np.any(mutual_node_high[selected] >= int(core_node_counts[alias]))
            or not np.array_equal(
                mutual_pair_keys[selected],
                mutual_node_low[selected] * int(core_node_counts[alias])
                + mutual_node_high[selected],
            )
            or tuple(mutual_relationship_ids[index] for index in selected_indices)
            != tuple(
                f"{alias}:pair:{int(key):016d}"
                for key in mutual_pair_keys[selected].tolist()
            )
        ):
            raise FourSeedStabilityError(f"{alias} mutual-pair identities drifted.")
        for seed_index in range(EXPECTED_SEED_COUNT):
            local_scores = mutual_seed_values[seed_index, selected]
            local_keys = mutual_pair_keys[selected]
            local_order = np.lexsort((local_keys, -local_scores))
            expected_support = np.zeros(len(selected_indices), dtype=np.bool_)
            expected_support[local_order[:MUTUAL_TOP_K_PER_CORE_SEED]] = True
            if not np.array_equal(
                mutual_support[seed_index, selected], expected_support
            ):
                raise FourSeedStabilityError(
                    f"{alias} seed {seed_index} mutual top-100 membership is not score-ranked."
                )
        alias_support_counts = mutual_support[:, selected].sum(axis=0)
        reported_range = alias_report.get("support_count_range")
        if (
            not isinstance(reported_range, list)
            or reported_range
            != [int(alias_support_counts.min()), int(alias_support_counts.max())]
        ):
            raise FourSeedStabilityError(
                f"{alias} mutual support-count report drifted."
            )
    locked_gradient_rows = _locked_request_metadata_from_csv(
        root / "provenance/selected_gradient_requests_v1.csv"
    )
    locked_gradient_ids = tuple(
        str(row["request_id"]) for row in locked_gradient_rows
    )
    gradient_metadata_columns = {
        "request_id",
        "core_alias",
        "canonical_edge_id",
        "canonical_shell_candidate_index",
        "shell_candidate_count",
        "shell",
        "radial_shell_index",
        "distance_um",
        "source_node",
        "receiver_node",
        "source_feature_index",
        "source_feature_name",
        "target_feature_index",
        "target_feature_name",
        "attention_head",
        "requested_layer",
        "layer_number",
        "graph_sha256",
        "fixed_mask_seed",
        "fixed_mask_sha256",
        "assert_directed_edge",
        "assert_source_feature_observed",
        "assert_target_feature_masked",
        "source_feature_observed",
        "target_feature_masked",
    }
    parquet_contract(
        "selected_gradients",
        expected_rows=EXPECTED_GRADIENT_REQUEST_COUNT * EXPECTED_SEED_COUNT,
        required_columns={
            "seed",
            *gradient_metadata_columns,
            "attention_value",
            "prediction_value",
            "d_attention_d_source_feature",
            "d_prediction_d_source_feature",
        },
        required_types={
            "seed": pa.int64(),
            **{
                name: pa.string()
                for name in (
                    "request_id",
                    "core_alias",
                    "shell",
                    "source_feature_name",
                    "target_feature_name",
                    "attention_head",
                    "graph_sha256",
                    "fixed_mask_sha256",
                )
            },
            **{
                name: pa.int64()
                for name in (
                    "canonical_edge_id",
                    "canonical_shell_candidate_index",
                    "shell_candidate_count",
                    "radial_shell_index",
                    "source_node",
                    "receiver_node",
                    "source_feature_index",
                    "target_feature_index",
                    "requested_layer",
                    "layer_number",
                    "fixed_mask_seed",
                )
            },
            **{
                name: pa.bool_()
                for name in (
                    "assert_directed_edge",
                    "assert_source_feature_observed",
                    "assert_target_feature_masked",
                    "source_feature_observed",
                    "target_feature_masked",
                )
            },
            **{
                name: pa.float64()
                for name in (
                    "distance_um",
                    "attention_value",
                    "prediction_value",
                    "d_attention_d_source_feature",
                    "d_prediction_d_source_feature",
                )
            },
        },
    )
    parquet_contract(
        "selected_gradient_stability",
        expected_rows=EXPECTED_GRADIENT_REQUEST_COUNT * 2,
        required_columns={
            "relationship_id",
            "score_name",
            "core_alias",
            "shell",
            "distance_um",
            "source_node",
            "receiver_node",
            "source_feature_index",
            "source_feature_name",
            "target_feature_index",
            "target_feature_name",
            "attention_head",
            "layer_number",
            "ensemble_mean",
            "ensemble_standard_deviation",
            "median",
            "minimum",
            "maximum",
            "seed_support_count",
            "empirical_quantile_0.05",
            "empirical_quantile_0.25",
            "empirical_quantile_0.75",
            "empirical_quantile_0.95",
            "ensemble_median_sign",
            "consistent_sign_support_count",
            "consistent_sign_fraction",
        },
        required_types={
            "relationship_id": pa.string(),
            "score_name": pa.string(),
            "core_alias": pa.string(),
            "shell": pa.string(),
            "distance_um": pa.float64(),
            "source_node": pa.int64(),
            "receiver_node": pa.int64(),
            "source_feature_index": pa.int64(),
            "source_feature_name": pa.string(),
            "target_feature_index": pa.int64(),
            "target_feature_name": pa.string(),
            "attention_head": pa.string(),
            "layer_number": pa.int64(),
            **{
                name: pa.float64()
                for name in (
                    "ensemble_mean",
                    "ensemble_standard_deviation",
                    "median",
                    "minimum",
                    "maximum",
                    "empirical_quantile_0.05",
                    "empirical_quantile_0.25",
                    "empirical_quantile_0.75",
                    "empirical_quantile_0.95",
                    "ensemble_median_sign",
                    "consistent_sign_fraction",
                )
            },
            "seed_support_count": pa.int64(),
            "consistent_sign_support_count": pa.int64(),
        },
    )
    raw_gradient_table = pq.read_table(root / "tables/selected_gradients.parquet")
    raw_gradient_seed = np.asarray(
        raw_gradient_table["seed"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    gradient_request_ids: tuple[str, ...] | None = None
    raw_gradient_values: dict[str, np.ndarray] = {}
    for value_name in (
        "d_attention_d_source_feature",
        "d_prediction_d_source_feature",
    ):
        per_seed_values: list[np.ndarray] = []
        for seed in ACTIVE_SEEDS:
            selected = np.flatnonzero(raw_gradient_seed == seed)
            if len(selected) != EXPECTED_GRADIENT_REQUEST_COUNT:
                raise FourSeedStabilityError(
                    f"Seed {seed} selected-gradient row count drifted."
                )
            seed_table = raw_gradient_table.take(pa.array(selected))
            request_ids = tuple(seed_table["request_id"].to_pylist())
            if len(set(request_ids)) != EXPECTED_GRADIENT_REQUEST_COUNT:
                raise FourSeedStabilityError(
                    f"Seed {seed} selected-gradient requests are duplicated."
                )
            if gradient_request_ids is None:
                gradient_request_ids = request_ids
            elif request_ids != gradient_request_ids:
                raise FourSeedStabilityError(
                    "Selected-gradient request order differs across seeds."
                )
            if request_ids != locked_gradient_ids:
                raise FourSeedStabilityError(
                    f"Seed {seed} selected-gradient rows do not match the locked CSV order."
                )
            for request_index, (observed_row, locked_row) in enumerate(
                zip(seed_table.to_pylist(), locked_gradient_rows, strict=True)
            ):
                for field, expected_value in locked_row.items():
                    observed_value = observed_row.get(field)
                    if isinstance(expected_value, float):
                        matches = math.isclose(
                            _finite_float(
                                observed_value,
                                f"seed {seed} request {request_index} {field}",
                            ),
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    else:
                        matches = observed_value == expected_value
                    if not matches:
                        raise FourSeedStabilityError(
                            f"Seed {seed} request {request_index} field {field} differs from locked CSV."
                        )
                alias = str(locked_row["core_alias"])
                if (
                    observed_row.get("source_feature_observed") is not True
                    or observed_row.get("target_feature_masked") is not True
                    or int(observed_row.get("layer_number", -1)) != 3
                    or int(observed_row["source_node"])
                    >= int(core_node_counts[alias])
                    or int(observed_row["receiver_node"])
                    >= int(core_node_counts[alias])
                ):
                    raise FourSeedStabilityError(
                        f"Seed {seed} request {request_index} execution metadata drifted."
                    )
            per_seed_values.append(numeric_column(seed_table, value_name))
        raw_gradient_values[value_name] = np.stack(per_seed_values, axis=0)
    if gradient_request_ids is None:
        raise FourSeedStabilityError("Selected-gradient request identities are absent.")
    gradient_summary_table = pq.read_table(
        root / "tables/selected_gradient_stability.parquet"
    )
    gradient_score_names = gradient_summary_table["score_name"].to_pylist()
    for value_name, values in raw_gradient_values.items():
        selected = np.flatnonzero(
            np.asarray(gradient_score_names, dtype=object) == value_name
        )
        if len(selected) != EXPECTED_GRADIENT_REQUEST_COUNT:
            raise FourSeedStabilityError(
                f"Selected-gradient {value_name} summary count drifted."
            )
        summary_table = gradient_summary_table.take(pa.array(selected))
        if tuple(summary_table["relationship_id"].to_pylist()) != gradient_request_ids:
            raise FourSeedStabilityError(
                f"Selected-gradient {value_name} summary identities drifted."
            )
        summary_rows = summary_table.to_pylist()
        for request_index, (summary_row, locked_row) in enumerate(
            zip(summary_rows, locked_gradient_rows, strict=True)
        ):
            summary_expected = {
                key: locked_row[key]
                for key in (
                    "core_alias",
                    "shell",
                    "distance_um",
                    "source_node",
                    "receiver_node",
                    "source_feature_index",
                    "source_feature_name",
                    "target_feature_index",
                    "target_feature_name",
                    "attention_head",
                )
            }
            summary_expected["layer_number"] = 3
            for field, expected_value in summary_expected.items():
                observed_value = summary_row.get(field)
                if isinstance(expected_value, float):
                    matches = math.isclose(
                        _finite_float(
                            observed_value,
                            f"gradient summary {request_index} {field}",
                        ),
                        expected_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                else:
                    matches = observed_value == expected_value
                if not matches:
                    raise FourSeedStabilityError(
                        f"Selected-gradient summary {request_index} field {field} differs from locked CSV."
                    )
        statistics = {
            "ensemble_mean": values.mean(axis=0),
            "ensemble_standard_deviation": values.std(axis=0, ddof=1),
            "median": np.median(values, axis=0),
            "minimum": values.min(axis=0),
            "maximum": values.max(axis=0),
        }
        gradient_quantiles = np.quantile(
            values,
            np.asarray(QUANTILE_LEVELS),
            axis=0,
            method="linear",
        )
        statistics.update(
            {
                f"empirical_quantile_{level:g}": gradient_quantiles[index]
                for index, level in enumerate(QUANTILE_LEVELS)
            }
        )
        for statistic, expected in statistics.items():
            if not np.allclose(
                numeric_column(summary_table, statistic),
                expected,
                atol=1e-12,
                rtol=1e-12,
            ):
                raise FourSeedStabilityError(
                    f"Selected-gradient {value_name} {statistic} is not reproducible."
                )
        support = np.abs(values) > 0.0
        median_sign = np.sign(np.median(values, axis=0))
        sign_count = ((np.sign(values) == median_sign[None, :]) & support).sum(axis=0)
        support_count = support.sum(axis=0)
        sign_fraction = np.divide(
            sign_count,
            support_count,
            out=np.zeros_like(sign_count, dtype=np.float64),
            where=support_count > 0,
        )
        exact_expectations = {
            "seed_support_count": support_count,
            "ensemble_median_sign": median_sign,
            "consistent_sign_support_count": sign_count,
            "consistent_sign_fraction": sign_fraction,
        }
        for statistic, expected in exact_expectations.items():
            observed = numeric_column(summary_table, statistic)
            if not np.array_equal(observed, np.asarray(expected, dtype=np.float64)):
                raise FourSeedStabilityError(
                    f"Selected-gradient {value_name} {statistic} drifted."
                )
    parquet_contract(
        "node_identity",
        expected_rows=node_count,
        required_columns={"node_identity"},
        required_types={"node_identity": pa.string()},
    )
    node_ids = tuple(
        pq.read_table(root / "tables/node_identity.parquet", columns=["node_identity"])[
            "node_identity"
        ].to_pylist()
    )
    if (
        len(set(node_ids)) != node_count
        or canonical_sha256(node_ids) != fixed_identity.get("node_identity_sha256")
        or node_ids
        != tuple(
            f"{alias}:node:{node_index:08d}"
            for alias in CANCER_ALIASES
            for node_index in range(int(core_node_counts[alias]))
        )
    ):
        raise FourSeedStabilityError("Node identity table is not unique/checksum-bound.")

    def verify_npz(
        name: str,
        expected_shapes: Mapping[str, tuple[int, ...]],
        expected_dtypes: Mapping[str, np.dtype[Any] | str],
    ) -> dict[str, np.ndarray]:
        materialized: dict[str, np.ndarray] = {}
        try:
            with np.load(
                root / f"arrays/{name}.npz", allow_pickle=False
            ) as archive:
                if set(archive.files) != set(expected_shapes) or set(
                    expected_dtypes
                ) != set(expected_shapes):
                    raise FourSeedStabilityError(
                        f"NPZ {name} array-name schema drifted."
                    )
                for array_name, expected_shape in expected_shapes.items():
                    array = archive[array_name]
                    if (
                        array.shape != expected_shape
                        or array.dtype != np.dtype(expected_dtypes[array_name])
                        or array.dtype.hasobject
                    ):
                        raise FourSeedStabilityError(
                            f"NPZ {name}/{array_name} shape or dtype drifted."
                        )
                    if array.dtype.kind in "fc" and not np.isfinite(array).all():
                        raise FourSeedStabilityError(
                            f"NPZ {name}/{array_name} contains nonfinite values."
                        )
                    if name == "node_embeddings" and array_name == "node_identity":
                        if tuple(array.tolist()) != node_ids:
                            raise FourSeedStabilityError(
                                "NPZ and Parquet node identities drifted."
                            )
                    if array_name == "seeds" and tuple(
                        int(value) for value in array.tolist()
                    ) != ACTIVE_SEEDS:
                        raise FourSeedStabilityError(
                            f"NPZ {name} seed identity/order drifted."
                        )
                    if array_name == "reference_to_seed_head":
                        expected_heads = list(range(LOCKED_HEAD_COUNT))
                        if any(
                            sorted(int(value) for value in row.tolist())
                            != expected_heads
                            for row in array
                        ) or array[REFERENCE_SEED].tolist() != expected_heads:
                            raise FourSeedStabilityError(
                                "Attention-head alignment is not a seed-0-referenced permutation."
                            )
                    if (
                        array_name
                        in {
                            "linear_cka",
                            "orthogonal_procrustes_similarity",
                        }
                        or array_name.endswith("_pairwise_seed_spearman")
                    ) and (
                        not np.allclose(array, array.T, atol=1e-12, rtol=0.0)
                        or not np.allclose(
                            np.diag(array),
                            np.ones(EXPECTED_SEED_COUNT),
                            atol=1e-12,
                            rtol=0.0,
                        )
                    ):
                        raise FourSeedStabilityError(
                            f"NPZ {name}/{array_name} is not a symmetric self-similarity matrix."
                        )
                    materialized[array_name] = np.ascontiguousarray(array)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise FourSeedStabilityError(f"Cannot verify NPZ {name}.") from exc
        return materialized

    node_embedding_arrays = verify_npz(
        "node_embeddings",
        {
            "node_identity": (node_count,),
            **{
                f"seed_{seed}": (node_count, LOCKED_HIDDEN_DIM)
                for seed in ACTIVE_SEEDS
            },
        },
        {
            "node_identity": np.dtype("<U32"),
            **{f"seed_{seed}": np.float32 for seed in ACTIVE_SEEDS},
        },
    )
    embedding_stability_arrays = verify_npz(
        "embedding_stability",
        {
            "seeds": (EXPECTED_SEED_COUNT,),
            "linear_cka": (EXPECTED_SEED_COUNT, EXPECTED_SEED_COUNT),
            "orthogonal_procrustes_similarity": (
                EXPECTED_SEED_COUNT,
                EXPECTED_SEED_COUNT,
            ),
        },
        {
            "seeds": np.int64,
            "linear_cka": np.float64,
            "orthogonal_procrustes_similarity": np.float64,
        },
    )
    attention_stability_arrays = verify_npz(
        "attention_head_stability",
        {
            "seeds": (EXPECTED_SEED_COUNT,),
            "reference_to_seed_head": (EXPECTED_SEED_COUNT, LOCKED_HEAD_COUNT),
            "signature_spearman": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "matched_signature_spearman": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "matched_attention_spearman": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "matched_attention_top_edge_jaccard": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "matched_positional_bias_spearman": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "content_position_spearman": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "content_position_same_sign_fraction": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
            "content_absolute_fraction": (
                EXPECTED_SEED_COUNT,
                LOCKED_HEAD_COUNT,
            ),
        },
        {
            "seeds": np.int64,
            "reference_to_seed_head": np.int64,
            "signature_spearman": np.float64,
            "matched_signature_spearman": np.float64,
            "matched_attention_spearman": np.float64,
            "matched_attention_top_edge_jaccard": np.float64,
            "matched_positional_bias_spearman": np.float64,
            "content_position_spearman": np.float64,
            "content_position_same_sign_fraction": np.float64,
            "content_absolute_fraction": np.float64,
        },
    )

    def require_recomputed_array(
        observed: np.ndarray,
        expected: np.ndarray,
        *,
        location: str,
    ) -> None:
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if observed_array.shape != expected_array.shape:
            raise FourSeedStabilityError(f"{location} shape drifted.")
        if observed_array.dtype.kind in "iu" or expected_array.dtype.kind in "iu":
            matches = np.array_equal(observed_array, expected_array)
        else:
            matches = np.allclose(
                observed_array,
                expected_array,
                atol=1e-12,
                rtol=1e-12,
            )
        if not matches:
            raise FourSeedStabilityError(f"{location} is not reproducible.")

    def report_array(
        container: Mapping[str, Any],
        key: str,
        *,
        location: str,
    ) -> np.ndarray:
        try:
            value = np.asarray(container.get(key), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FourSeedStabilityError(
                f"{location} is not a numeric array."
            ) from exc
        if not np.isfinite(value).all():
            raise FourSeedStabilityError(f"{location} contains nonfinite values.")
        return value

    embedding_records = tuple(
        FixedSeedArray(
            seed=seed,
            fixed_input_ids=node_ids,
            values=node_embedding_arrays[f"seed_{seed}"],
        )
        for seed in ACTIVE_SEEDS
    )
    recomputed_embedding = embedding_stability(
        embedding_records,
        include_orthogonal_procrustes=True,
        expected_seed_count=EXPECTED_SEED_COUNT,
    )
    embedding_expected = {
        "linear_cka": recomputed_embedding.linear_cka,
        "orthogonal_procrustes_similarity": (
            recomputed_embedding.orthogonal_procrustes_similarity
        ),
    }
    embedding_report = _mapping(
        report.get("embedding_stability"), "embedding stability report"
    )
    if tuple(embedding_report.get("seeds", ())) != ACTIVE_SEEDS:
        raise FourSeedStabilityError("Embedding report seed order drifted.")
    for array_name, expected in embedding_expected.items():
        if expected is None:
            raise FourSeedStabilityError(
                "Orthogonal Procrustes embedding stability is absent."
            )
        require_recomputed_array(
            embedding_stability_arrays[array_name],
            expected,
            location=f"NPZ embedding stability {array_name}",
        )
        require_recomputed_array(
            report_array(
                embedding_report,
                array_name,
                location=f"report embedding stability {array_name}",
            ),
            expected,
            location=f"report embedding stability {array_name}",
        )
    del embedding_records, recomputed_embedding

    if reference_receiver_groups is None:
        raise FourSeedStabilityError("Fixed receiver groups are absent.")
    signature_ids = tuple(
        f"{channel}:{edge_id}"
        for channel in ("attention", "content", "bias")
        for edge_id in fixed_edge_ids
    )
    signature_records = tuple(
        FixedSeedArray(
            seed=seed,
            fixed_input_ids=signature_ids,
            values=attention_head_signature(
                raw_edge_records["attention"][seed].values,
                content_logits=centered_content_records[seed].values,
                positional_bias=centered_bias_records[seed].values,
            ),
        )
        for seed in ACTIVE_SEEDS
    )
    recomputed_alignment = match_attention_heads(
        signature_records,
        reference_seed=REFERENCE_SEED,
        expected_seed_count=EXPECTED_SEED_COUNT,
    )
    recomputed_attention = matched_head_attention_stability(
        tuple(raw_edge_records["attention"]),
        recomputed_alignment,
        top_fraction=HEAD_TOP_EDGE_FRACTION,
    )
    recomputed_bias = positional_bias_response_stability(
        tuple(centered_bias_records), recomputed_alignment
    )
    recomputed_contribution = content_position_contribution_agreement(
        tuple(centered_content_records),
        tuple(centered_bias_records),
        recomputed_alignment,
    )
    attention_expected = {
        "reference_to_seed_head": recomputed_alignment.reference_to_seed_head,
        "signature_spearman": recomputed_alignment.signature_spearman,
        "matched_signature_spearman": (
            recomputed_alignment.matched_signature_spearman
        ),
        "matched_attention_spearman": (
            recomputed_attention.matched_head_spearman
        ),
        "matched_attention_top_edge_jaccard": (
            recomputed_attention.matched_head_top_edge_jaccard
        ),
        "matched_positional_bias_spearman": (
            recomputed_bias.matched_head_spearman
        ),
        "content_position_spearman": recomputed_contribution.within_seed_spearman,
        "content_position_same_sign_fraction": (
            recomputed_contribution.same_sign_fraction
        ),
        "content_absolute_fraction": (
            recomputed_contribution.content_absolute_fraction
        ),
    }
    for array_name, expected in attention_expected.items():
        require_recomputed_array(
            attention_stability_arrays[array_name],
            expected,
            location=f"NPZ attention stability {array_name}",
        )
    attention_report = _mapping(
        report.get("attention_head_stability"), "attention stability report"
    )
    if (
        int(attention_report.get("reference_seed", -1)) != REFERENCE_SEED
        or attention_report.get("matching_method")
        != recomputed_alignment.matching_method
        or not math.isclose(
            _finite_float(
                attention_report.get("head_top_edge_fraction"),
                "attention report top-edge fraction",
            ),
            HEAD_TOP_EDGE_FRACTION,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or int(attention_report.get("head_top_edge_count", 0))
        != recomputed_attention.top_k
        or attention_report.get("receiver_softmax_null_offsets_removed") is not True
    ):
        raise FourSeedStabilityError("Attention stability report contract drifted.")
    report_attention_expected = {
        "reference_to_seed_head": recomputed_alignment.reference_to_seed_head,
        "matched_signature_spearman": (
            recomputed_alignment.matched_signature_spearman
        ),
        "matched_attention_spearman": (
            recomputed_attention.matched_head_spearman
        ),
        "matched_attention_top_edge_jaccard": (
            recomputed_attention.matched_head_top_edge_jaccard
        ),
        "matched_positional_bias_spearman": (
            recomputed_bias.matched_head_spearman
        ),
    }
    for key, expected in report_attention_expected.items():
        require_recomputed_array(
            report_array(
                attention_report,
                key,
                location=f"report attention stability {key}",
            ),
            expected,
            location=f"report attention stability {key}",
        )
    contribution_report = _mapping(
        attention_report.get("content_vs_positional_bias"),
        "content-versus-position report",
    )
    for key, expected in (
        ("within_seed_spearman", recomputed_contribution.within_seed_spearman),
        ("same_sign_fraction", recomputed_contribution.same_sign_fraction),
        ("content_absolute_fraction", recomputed_contribution.content_absolute_fraction),
    ):
        require_recomputed_array(
            report_array(
                contribution_report,
                key,
                location=f"report content-position {key}",
            ),
            expected,
            location=f"report content-position {key}",
        )

    gradient_npz_shapes: dict[str, tuple[int, ...]] = {}
    for value_name in (
        "d_attention_d_source_feature",
        "d_prediction_d_source_feature",
    ):
        gradient_npz_shapes.update(
            {
                f"{value_name}_pairwise_seed_spearman": (
                    EXPECTED_SEED_COUNT,
                    EXPECTED_SEED_COUNT,
                ),
                f"{value_name}_ensemble_median_sign": (
                    EXPECTED_GRADIENT_REQUEST_COUNT,
                ),
                f"{value_name}_consistent_sign_support_count": (
                    EXPECTED_GRADIENT_REQUEST_COUNT,
                ),
                f"{value_name}_consistent_sign_fraction": (
                    EXPECTED_GRADIENT_REQUEST_COUNT,
                ),
            }
        )
    gradient_npz_dtypes = {
        name: (
            np.int64
            if name.endswith("_consistent_sign_support_count")
            else np.float64
        )
        for name in gradient_npz_shapes
    }
    gradient_stability_arrays = verify_npz(
        "selected_gradient_stability",
        gradient_npz_shapes,
        gradient_npz_dtypes,
    )
    for value_name, values in raw_gradient_values.items():
        recomputed = selected_gradient_stability(
            tuple(
                FixedSeedArray(
                    seed=seed,
                    fixed_input_ids=gradient_request_ids,
                    values=values[seed],
                )
                for seed in ACTIVE_SEEDS
            ),
            magnitude_threshold=0.0,
            quantile_levels=QUANTILE_LEVELS,
            expected_seed_count=EXPECTED_SEED_COUNT,
        )
        expected_arrays = {
            f"{value_name}_pairwise_seed_spearman": (
                recomputed.pairwise_seed_spearman
            ),
            f"{value_name}_ensemble_median_sign": (
                recomputed.ensemble_median_sign
            ),
            f"{value_name}_consistent_sign_support_count": (
                recomputed.consistent_sign_support_count
            ),
            f"{value_name}_consistent_sign_fraction": (
                recomputed.consistent_sign_fraction
            ),
        }
        for array_name, expected in expected_arrays.items():
            if not np.array_equal(gradient_stability_arrays[array_name], expected):
                raise FourSeedStabilityError(
                    f"NPZ selected-gradient array {array_name} is not reproducible."
                )
        derivative_key = (
            "attention_derivative"
            if value_name == "d_attention_d_source_feature"
            else "prediction_derivative"
        )
        derivative_report = _mapping(
            selected_gradient_report.get(derivative_key),
            f"selected-gradient {derivative_key} report",
        )
        if (
            int(derivative_report.get("summary_row_count", 0))
            != EXPECTED_GRADIENT_REQUEST_COUNT
            or derivative_report.get("analysis_scope") != recomputed.analysis_scope
            or not np.allclose(
                np.asarray(
                    derivative_report.get("pairwise_seed_spearman"),
                    dtype=np.float64,
                ),
                recomputed.pairwise_seed_spearman,
                atol=1e-12,
                rtol=1e-12,
            )
        ):
            raise FourSeedStabilityError(
                f"Selected-gradient {derivative_key} report drifted."
            )
    return {
        "valid": True,
        "status": "complete",
        "manifest_content_sha256": observed_content_sha256,
        "file_count": len(observed_files),
        "active_model_seeds": list(ACTIVE_SEEDS),
        "deferred_model_seeds": list(DEFERRED_SEEDS),
    }


def _validate_analysis_protocol_inputs(
    protocol: Mapping[str, Any],
    *,
    cohort_manifest_path: Path,
    graph_manifest_path: Path,
) -> None:
    """Bind the additive analysis protocol to the exact prepared inputs."""

    if (
        protocol.get("analysis_protocol_schema")
        != "cancer_6core_relative_qkv_stability_analysis_protocol_v1"
        or protocol.get("campaign_id") != CAMPAIGN_ID
        or protocol.get("role") != "additive_post_training_analysis_protocol"
        or protocol.get("changes_training_contract") is not False
        or protocol.get("uses_validation_or_test_selection") is not False
        or protocol.get("fit_scope")
        != "all_cells_six_cancer_cores_transductive"
        or protocol.get("generalization_claim_supported") is not False
    ):
        raise FourSeedStabilityError(
            "Analysis protocol scientific/additive scope drifted."
        )
    prepared = _mapping(protocol.get("prepared_inputs"), "analysis prepared inputs")
    fixed_mask = _mapping(
        prepared.get("fixed_inference_mask"),
        "analysis fixed inference mask",
    )
    if (
        tuple(prepared.get("core_aliases", ())) != CANCER_ALIASES
        or prepared.get("cohort_manifest_sha256")
        != sha256_file(cohort_manifest_path)
        or prepared.get("graph_manifest_sha256") != sha256_file(graph_manifest_path)
        or prepared.get("coordinates_were_model_inputs") is not False
        or fixed_mask.get("base_seed") != FIXED_INFERENCE_MASK_BASE_SEED
        or fixed_mask.get("namespace") != FIXED_INFERENCE_MASK_NAMESPACE
        or fixed_mask.get("identical_across_model_seeds") is not True
    ):
        raise FourSeedStabilityError(
            "Analysis protocol does not match the exact prepared inputs/mask."
        )
    ensemble = _mapping(protocol.get("ensemble"), "analysis ensemble")
    extraction = _mapping(
        protocol.get("fixed_probe_extraction"),
        "analysis fixed-probe extraction",
    )
    gradients = _mapping(
        protocol.get("selected_gradient_requests"),
        "analysis selected gradients",
    )
    reproducibility = _mapping(
        protocol.get("reproducibility_gate"),
        "analysis reproducibility gate",
    )
    if (
        tuple(ensemble.get("active_model_seeds", ())) != ACTIVE_SEEDS
        or tuple(ensemble.get("deferred_model_seeds", ())) != DEFERRED_SEEDS
        or ensemble.get("expected_active_seed_count") != EXPECTED_SEED_COUNT
        or ensemble.get("seed_4_required_for_this_report") is not False
        or ensemble.get("five_seed_completion_claim_allowed") is not False
        or ensemble.get("common_final_epoch_required") is not False
        or ensemble.get("differing_final_epochs_allowed") is not True
        or ensemble.get("checkpoint_selection_uses_held_in_diagnostics") is not False
        or extraction.get("layer") != FINAL_LAYER
        or extraction.get("receiver_probes_per_core") != RECEIVER_PROBES_PER_CORE
        or extraction.get("top_edge_fraction") != HEAD_TOP_EDGE_FRACTION
        or extraction.get("mutual_top_edges_per_core_per_seed")
        != MUTUAL_TOP_K_PER_CORE_SEED
        or tuple(extraction.get("empirical_quantiles", ())) != QUANTILE_LEVELS
        or gradients.get("request_count") != EXPECTED_GRADIENT_REQUEST_COUNT
        or gradients.get("expected_rows_per_seed")
        != EXPECTED_GRADIENT_REQUEST_COUNT
        or gradients.get("layer") != FINAL_LAYER
        or gradients.get("attention_head") != "mean"
        or gradients.get("model_or_checkpoint_used_for_selection") is not False
        or gradients.get("derivative_scope")
        != "selected_requests_only_no_exhaustive_jacobian"
        or gradients.get("expected_rows_across_active_seeds")
        != EXPECTED_GRADIENT_REQUEST_COUNT * EXPECTED_SEED_COUNT
        or reproducibility.get("verify_protocol_checksum_before_cuda") is not True
        or reproducibility.get("verify_request_checksum_before_cuda") is not True
        or reproducibility.get("regenerate_requests_from_prepared_inputs_before_cuda")
        is not True
        or reproducibility.get("require_exact_csv_byte_equality") is not True
        or reproducibility.get("behavior_on_any_drift")
        != "fail_without_running_model_inference"
    ):
        raise FourSeedStabilityError("Additive analysis protocol settings drifted.")
    if LOCKED_GRADIENT_REQUEST_COUNT != EXPECTED_GRADIENT_REQUEST_COUNT:
        raise FourSeedStabilityError("Locked gradient request constants disagree.")

    gauge = _mapping(protocol.get("logit_gauge"), "analysis logit gauge")
    statistics = _mapping(
        protocol.get("statistical_summary_semantics"),
        "analysis statistical semantics",
    )
    standard_deviation = _mapping(
        statistics.get("standard_deviation"), "analysis standard deviation"
    )
    quantiles = _mapping(
        statistics.get("empirical_quantiles"), "analysis quantiles"
    )
    gradient_support = _mapping(
        statistics.get("selected_gradient_support"),
        "analysis gradient support",
    )
    mutual_support = _mapping(
        statistics.get("mutual_pair_support"), "analysis mutual support"
    )
    fixed_edge_support = _mapping(
        statistics.get("fixed_edge_attention_support"),
        "analysis fixed-edge support",
    )
    numerical = _mapping(
        protocol.get("numerical_execution"), "analysis numerical execution"
    )
    extraction_execution = _mapping(
        numerical.get("attention_and_embedding_extraction"),
        "analysis extraction execution",
    )
    derivative_execution = _mapping(
        numerical.get("selected_derivative_execution"),
        "analysis derivative execution",
    )
    derivative_units = _mapping(
        gradients.get("derivative_units"), "analysis derivative units"
    )
    output_contract = _mapping(
        protocol.get("required_output_contract"), "analysis output contract"
    )
    limits = _mapping(
        protocol.get("interpretation_limits"), "analysis interpretation limits"
    )
    if (
        gauge.get("centering_scope") != "each_receiver_and_attention_head"
        or gauge.get("centering_operation")
        != "subtract_mean_over_all_incoming_edges_of_receiver"
        or tuple(gauge.get("centered_channels", ()))
        != ("content_logit", "relative_positional_bias")
        or gauge.get("attention_probabilities_are_centered") is not False
        or gauge.get("combined_centered_logit_definition")
        != "centered_content_plus_centered_positional_bias"
        or gauge.get("receiver_wise_softmax_null_offsets_removed") is not True
        or statistics.get("ensemble_member_count") != EXPECTED_SEED_COUNT
        or standard_deviation.get("kind") != "sample"
        or standard_deviation.get("ddof") != 1
        or quantiles.get("implementation") != "numpy_quantile"
        or quantiles.get("method") != "linear"
        or tuple(quantiles.get("levels", ())) != QUANTILE_LEVELS
        or gradient_support.get("definition")
        != "absolute_gradient_strictly_greater_than_zero"
        or float(gradient_support.get("magnitude_threshold", -1.0)) != 0.0
        or gradient_support.get("tiny_finite_nonzero_values_count_as_support")
        is not True
        or mutual_support.get("definition")
        != "pair_is_member_of_that_seeds_per_core_top_100"
        or mutual_support.get("score_magnitude_does_not_define_support") is not True
        or fixed_edge_support.get("definition")
        != "edge_is_member_of_that_seeds_top_5_percent_by_mean_head_attention_over_locked_fixed_probe_edge_set"
        or float(fixed_edge_support.get("top_fraction", -1.0))
        != HEAD_TOP_EDGE_FRACTION
    ):
        raise FourSeedStabilityError("Locked statistical/logit semantics drifted.")
    if (
        numerical.get("deterministic_seed") != ANALYSIS_DETERMINISTIC_SEED
        or numerical.get("deterministic_algorithms") is not True
        or numerical.get("deterministic_warn_only") is not False
        or numerical.get("cublas_workspace_config") != ":4096:8"
        or extraction_execution.get("automatic_mixed_precision") is not True
        or extraction_execution.get("autocast_dtype") != "float16"
        or extraction_execution.get("exact_receiver_wise_softmax") is not True
        or extraction_execution.get("attention_accumulation_dtype") != "float32"
        or extraction_execution.get("neighbor_sampling") is not False
        or derivative_execution.get("automatic_mixed_precision") is not False
        or derivative_execution.get("dtype") != "float32"
        or numerical.get("receiver_chunk_size") != DEFAULT_RECEIVER_CHUNK_SIZE
        or numerical.get("maximum_edges_per_chunk")
        != DEFAULT_MAX_EDGES_PER_CHUNK
        or numerical.get("all_incoming_edges_of_receiver_normalized_together")
        is not True
        or numerical.get("execution_overrides_allowed") is not False
        or numerical.get("models_processed_concurrently") != 1
        or numerical.get("complete_cores_staged_on_gpu_concurrently") != 1
    ):
        raise FourSeedStabilityError("Locked numerical execution drifted.")
    if (
        derivative_units.get("source_input_variable")
        != "gene_wise_standardized_log1p_raw_count"
        or derivative_units.get("source_input_perturbation_unit")
        != "one_standardized_log1p_expression_unit"
        or derivative_units.get("attention_derivative")
        != "attention_probability_per_standardized_log1p_source_expression_unit"
        or derivative_units.get("prediction_output_variable")
        != "gene_wise_standardized_log1p_raw_count_prediction"
        or derivative_units.get("prediction_derivative")
        != "standardized_log1p_target_prediction_units_per_standardized_log1p_source_expression_unit"
        or derivative_units.get("library_size_normalization_used") is not False
    ):
        raise FourSeedStabilityError("Locked derivative units drifted.")
    if (
        output_contract.get("analysis_schema") != ANALYSIS_SCHEMA
        or output_contract.get("manifest_schema") != MANIFEST_SCHEMA
        or output_contract.get("status") != "complete"
        or tuple(output_contract.get("active_model_seeds", ())) != ACTIVE_SEEDS
        or tuple(output_contract.get("parquet_tables", ()))
        != (
            "training_curves",
            "fixed_metrics",
            "fixed_metric_summary",
            "fixed_probe_edges",
            "fixed_edge_summary",
            "mutual_pair_stability",
            "selected_gradients",
            "selected_gradient_stability",
            "node_identity",
        )
        or tuple(output_contract.get("numpy_archives", ()))
        != (
            "node_embeddings",
            "embedding_stability",
            "attention_head_stability",
            "selected_gradient_stability",
        )
    ):
        raise FourSeedStabilityError("Locked output schema drifted.")
    if (
        limits.get("selected_gradient_probe_count")
        != EXPECTED_GRADIENT_REQUEST_COUNT
        or limits.get("selected_gradient_probes_are_sparse") is not True
        or limits.get(
            "selected_gradient_probes_are_representative_of_all_edges_or_genes"
        )
        is not False
        or limits.get("tiny_finite_nonzero_gradients_can_count_as_support")
        is not True
        or limits.get("differing_plateau_epochs_allowed") is not True
        or limits.get("exposure_duration_can_contribute_to_observed_seed_spread")
        is not True
        or limits.get("local_orientation_representation") != "axial_tensor"
        or limits.get("opposite_directions_along_same_axis_distinguishable")
        is not False
        or limits.get("attention_gradients_or_jacobians_establish_causality")
        is not False
    ):
        raise FourSeedStabilityError("Locked interpretation limits drifted.")


def _locked_request_metadata(
    requests: Sequence[LockedGradientRequest],
) -> dict[str, Mapping[str, Any]]:
    """Return typed immutable metadata aligned to selected derivative rows."""

    materialized = tuple(requests)
    if len(materialized) != EXPECTED_GRADIENT_REQUEST_COUNT:
        raise FourSeedStabilityError("Exactly 24 locked gradient requests are required.")
    metadata: dict[str, Mapping[str, Any]] = {}
    for request in materialized:
        if request.request_id in metadata:
            raise FourSeedStabilityError("Locked gradient request IDs are duplicated.")
        metadata[request.request_id] = {
            "request_id": request.request_id,
            "core_alias": request.core_alias,
            "canonical_edge_id": request.canonical_edge_id,
            "canonical_shell_candidate_index": (
                request.canonical_shell_candidate_index
            ),
            "shell_candidate_count": request.shell_candidate_count,
            "shell": request.radial_shell,
            "radial_shell_index": request.radial_shell_index,
            "distance_um": request.distance_um,
            "source_node": request.source_node,
            "receiver_node": request.receiver_node,
            "source_feature_index": request.source_feature_index,
            "source_feature_name": request.source_feature_name,
            "target_feature_index": request.target_feature_index,
            "target_feature_name": request.target_feature_name,
            "attention_head": request.attention_head,
            "requested_layer": request.layer,
            "graph_sha256": request.graph_sha256,
            "fixed_mask_seed": request.fixed_mask_seed,
            "fixed_mask_sha256": request.fixed_mask_sha256,
            "assert_directed_edge": True,
            "assert_source_feature_observed": True,
            "assert_target_feature_masked": True,
        }
    return metadata


def _locked_request_metadata_from_csv(
    request_csv_path: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Parse the checksum-locked request table into typed verifier metadata.

    The bundle already binds the CSV bytes to the campaign constant.  This
    second gate deliberately parses every scientific identity field so the
    derivative Parquet rows cannot be relabelled while retaining self-consistent
    table and manifest hashes.
    """

    try:
        with Path(request_csv_path).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUEST_CSV_FIELDS:
                raise FourSeedStabilityError(
                    "Locked selected-gradient request CSV schema drifted."
                )
            raw_rows = tuple(reader)
    except (OSError, csv.Error) as exc:
        raise FourSeedStabilityError(
            "Cannot parse locked selected-gradient request CSV."
        ) from exc
    if len(raw_rows) != EXPECTED_GRADIENT_REQUEST_COUNT:
        raise FourSeedStabilityError(
            "Locked selected-gradient request CSV must contain exactly 24 rows."
        )

    def integer(row: Mapping[str, str], field: str) -> int:
        try:
            text = row[field]
            value = int(text, 10)
        except (KeyError, TypeError, ValueError) as exc:
            raise FourSeedStabilityError(
                f"Locked request field {field} must be an integer."
            ) from exc
        if str(value) != text:
            raise FourSeedStabilityError(
                f"Locked request field {field} is not canonically encoded."
            )
        return value

    expected_order = tuple(
        (alias, shell_index)
        for alias in CANCER_ALIASES
        for shell_index in range(len(RADIAL_SHELLS))
    )
    parsed: list[Mapping[str, Any]] = []
    for row_index, (raw, expected_identity) in enumerate(
        zip(raw_rows, expected_order, strict=True)
    ):
        alias, shell_index = expected_identity
        try:
            distance_um = float(raw["distance_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FourSeedStabilityError(
                "Locked request distance_um must be numeric."
            ) from exc
        source_feature_index = integer(raw, "source_feature_index")
        target_feature_index = integer(raw, "target_feature_index")
        radial_shell_index = integer(raw, "radial_shell_index")
        layer = integer(raw, "layer")
        lower, upper, shell = RADIAL_SHELLS[shell_index]
        request_id = str(raw.get("request_id", ""))
        boolean_fields = (
            "assert_directed_edge",
            "assert_shell_membership",
            "assert_source_feature_observed",
            "assert_target_feature_masked",
            "assert_feature_indices_distinct",
        )
        if (
            raw.get("request_schema") != GRADIENT_REQUEST_SCHEMA
            or raw.get("core_alias") != alias
            or radial_shell_index != shell_index
            or raw.get("radial_shell") != shell
            or layer != FINAL_LAYER
            or raw.get("attention_head") != "mean"
            or not request_id
            or raw.get("source_feature") != str(source_feature_index)
            or raw.get("target_feature") != str(target_feature_index)
            or not math.isfinite(distance_um)
            or not (lower < distance_um <= upper)
            or source_feature_index < 0
            or source_feature_index >= LOCKED_GENE_COUNT
            or target_feature_index < 0
            or target_feature_index >= LOCKED_GENE_COUNT
            or source_feature_index == target_feature_index
            or any(raw.get(field) != "true" for field in boolean_fields)
            or not _is_lower_sha256(raw.get("graph_sha256"))
            or not _is_lower_sha256(raw.get("fixed_mask_sha256"))
        ):
            raise FourSeedStabilityError(
                f"Locked selected-gradient request row {row_index} drifted."
            )
        typed = {
            "request_id": request_id,
            "core_alias": alias,
            "canonical_edge_id": integer(raw, "canonical_edge_id"),
            "canonical_shell_candidate_index": integer(
                raw, "canonical_shell_candidate_index"
            ),
            "shell_candidate_count": integer(raw, "shell_candidate_count"),
            "shell": shell,
            "radial_shell_index": radial_shell_index,
            "distance_um": distance_um,
            "source_node": integer(raw, "source_node"),
            "receiver_node": integer(raw, "receiver_node"),
            "source_feature_index": source_feature_index,
            "source_feature_name": str(raw.get("source_feature_name", "")),
            "target_feature_index": target_feature_index,
            "target_feature_name": str(raw.get("target_feature_name", "")),
            "attention_head": "mean",
            "requested_layer": layer,
            "graph_sha256": str(raw.get("graph_sha256")),
            "fixed_mask_seed": integer(raw, "fixed_mask_seed"),
            "fixed_mask_sha256": str(raw.get("fixed_mask_sha256")),
            "assert_directed_edge": True,
            "assert_source_feature_observed": True,
            "assert_target_feature_masked": True,
        }
        if (
            not typed["source_feature_name"]
            or not typed["target_feature_name"]
            or typed["canonical_edge_id"] < 0
            or typed["canonical_shell_candidate_index"] < 0
            or typed["shell_candidate_count"] <= 0
            or typed["canonical_shell_candidate_index"]
            >= typed["shell_candidate_count"]
            or typed["source_node"] < 0
            or typed["receiver_node"] < 0
        ):
            raise FourSeedStabilityError(
                f"Locked selected-gradient request row {row_index} has invalid bounds."
            )
        parsed.append(typed)
    request_ids = tuple(str(row["request_id"]) for row in parsed)
    if len(set(request_ids)) != EXPECTED_GRADIENT_REQUEST_COUNT:
        raise FourSeedStabilityError(
            "Locked selected-gradient request IDs are duplicated."
        )
    return tuple(parsed)


def _analysis_source_provenance() -> tuple[Mapping[str, Any], bytes]:
    """Capture the exact repository/source state without mutating Git."""

    project_root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=project_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FourSeedStabilityError(
                "Cannot capture analysis Git provenance."
            ) from exc
        return completed.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all").decode(
        "utf-8",
        errors="strict",
    )
    tracked_diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    if status:
        raise FourSeedStabilityError(
            "Analysis must run from a clean committed checkout so executing source "
            "bytes and recorded provenance cannot diverge."
        )
    source_files: dict[str, Mapping[str, Any]] = {}
    for relative in ANALYSIS_SOURCE_FILES:
        path = (project_root / relative).resolve(strict=True)
        source_files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return (
        {
            "project_root": project_root.as_posix(),
            "git_commit": commit,
            "git_dirty": False,
            "git_status_porcelain_v1": status.splitlines(),
            "tracked_dirty_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "analysis_source_files": source_files,
        },
        tracked_diff,
    )


def _nvidia_smi_inventory() -> Mapping[str, Any]:
    """Capture physical GPU UUID/driver inventory for device-binding provenance."""

    query = "index,uuid,pci.bus_id,name,driver_version,memory.total"
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FourSeedStabilityError(
            "Cannot capture the physical NVIDIA GPU inventory."
        ) from exc
    output = completed.stdout.decode("utf-8", errors="strict").strip()
    raw_rows = [line for line in csv.reader(StringIO(output)) if line]
    if not raw_rows:
        raise FourSeedStabilityError("NVIDIA GPU inventory is empty.")
    records: list[dict[str, Any]] = []
    for raw in raw_rows:
        values = [value.strip() for value in raw]
        if (
            len(values) != 6
            or not values[0].isdigit()
            or not values[1].startswith("GPU-")
            or not values[5].isdigit()
        ):
            raise FourSeedStabilityError("NVIDIA GPU inventory row is malformed.")
        records.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "pci_bus_id": values[2],
                "name": values[3],
                "driver_version": values[4],
                "memory_total_mib": int(values[5]),
            }
        )
    return {
        "query": query,
        "records": records,
        "raw_output_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _nvidia_compute_applications() -> tuple[Mapping[str, Any], ...]:
    query = "pid,gpu_uuid,process_name"
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--query-compute-apps={query}",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FourSeedStabilityError("Cannot inspect NVIDIA compute processes.") from exc
    output = completed.stdout.decode("utf-8", errors="strict").strip()
    if not output:
        return ()
    records: list[Mapping[str, Any]] = []
    for raw in csv.reader(StringIO(output)):
        values = [value.strip() for value in raw]
        if len(values) != 3 or not values[0].isdigit() or not values[1]:
            raise FourSeedStabilityError("NVIDIA compute-process row is malformed.")
        records.append(
            {
                "pid": int(values[0]),
                "gpu_uuid": values[1],
                "process_name": values[2],
            }
        )
    return tuple(records)


def _preflight_gpu_binding(
    device: str | torch.device,
) -> tuple[torch.device, Mapping[str, Any]]:
    """Resolve one idle physical GPU before PyTorch can create a CUDA context."""

    resolved = torch.device(device)
    if resolved != torch.device("cuda:0"):
        raise FourSeedStabilityError(
            "The locked analysis requires logical cuda:0 under singleton visibility."
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise FourSeedStabilityError("CUDA_DEVICE_ORDER must be exactly PCI_BUS_ID.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise FourSeedStabilityError(
            "CUDA_VISIBLE_DEVICES must select exactly one physical GPU."
        )
    token = visible.strip()
    inventory = _nvidia_smi_inventory()
    records = tuple(
        _mapping(value, "NVIDIA inventory record")
        for value in inventory["records"]
    )
    selected = tuple(
        value
        for value in records
        if (token.isdigit() and int(value["index"]) == int(token))
        or (token.startswith("GPU-") and value["uuid"] == token)
    )
    if len(selected) != 1:
        raise FourSeedStabilityError(
            "Singleton CUDA visibility does not resolve to one physical GPU."
        )
    applications = _nvidia_compute_applications()
    selected_uuid = str(selected[0]["uuid"])
    occupying = tuple(
        value for value in applications if value["gpu_uuid"] == selected_uuid
    )
    if occupying:
        raise FourSeedStabilityError(
            f"Selected physical GPU {selected_uuid} is not idle before CUDA setup."
        )
    return resolved, {
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": token,
        "logical_device": "cuda:0",
        "selected_physical_gpu": dict(selected[0]),
        "compute_applications_before_cuda": [],
        "nvidia_smi_inventory": inventory,
    }


def _finalize_gpu_binding(
    preflight: Mapping[str, Any],
    *,
    properties: Any,
    device: torch.device,
) -> Mapping[str, Any]:
    """Prove logical CUDA 0 maps to the preflight UUID and only this PID owns it."""

    selected = _mapping(
        preflight.get("selected_physical_gpu"), "selected physical GPU"
    )
    raw_torch_uuid = getattr(properties, "uuid", None)
    if isinstance(raw_torch_uuid, bytes):
        torch_uuid = raw_torch_uuid.decode("ascii", errors="strict")
    else:
        torch_uuid = str(raw_torch_uuid)
    if torch_uuid and not torch_uuid.startswith("GPU-"):
        torch_uuid = f"GPU-{torch_uuid}"
    if torch_uuid != str(selected.get("uuid")):
        raise FourSeedStabilityError(
            "PyTorch logical CUDA device UUID does not match singleton visibility."
        )
    context_probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    del context_probe
    applications = _nvidia_compute_applications()
    selected_applications = tuple(
        value
        for value in applications
        if value["gpu_uuid"] == str(selected.get("uuid"))
    )
    if {int(value["pid"]) for value in selected_applications} != {os.getpid()}:
        raise FourSeedStabilityError(
            "Selected GPU compute ownership is not exclusive to this analysis PID."
        )
    return {
        **dict(preflight),
        "torch_device_uuid": torch_uuid,
        "analysis_pid": os.getpid(),
        "compute_applications_after_cuda": [
            dict(value) for value in selected_applications
        ],
    }


def run_four_seed_stability_pipeline(
    *,
    run_roots: Mapping[int, Path],
    checkpoint_paths: Mapping[int, Path],
    receipt_paths: Mapping[int, Path],
    cohort_dir: str | Path,
    graph_dir: str | Path,
    protocol_path: str | Path,
    protocol_sha256_path: str | Path,
    request_csv_path: str | Path,
    request_sha256_path: str | Path,
    destination: str | Path,
    device: str | torch.device,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
    amp: bool = True,
) -> Mapping[str, Any]:
    """Run the exact compact four-seed analysis and atomically publish it.

    Every checksum, protocol, request-table, run-bundle, strict-receipt, plateau,
    and prepared-input gate is evaluated before CUDA is initialized.  Models are
    then replayed serially; each model processes one complete core at a time.
    """

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    output = Path(destination).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis bundle: {output}")
    if (
        isinstance(receiver_chunk_size, bool)
        or int(receiver_chunk_size) != DEFAULT_RECEIVER_CHUNK_SIZE
        or isinstance(max_edges_per_chunk, bool)
        or int(max_edges_per_chunk) != DEFAULT_MAX_EDGES_PER_CHUNK
        or amp is not True
    ):
        raise FourSeedStabilityError(
            "The locked analysis requires AMP float16 extraction with receiver "
            "chunk size 512 and maximum 200000 edges per chunk."
        )

    cohort_root = Path(cohort_dir).expanduser().resolve(strict=True)
    graph_root = Path(graph_dir).expanduser().resolve(strict=True)
    cohort_manifest = (cohort_root / "manifest.json").resolve(strict=True)
    graph_manifest = (graph_root / "manifest.json").resolve(strict=True)
    resolved_protocol = Path(protocol_path).expanduser().resolve(strict=True)
    resolved_protocol_sha = Path(protocol_sha256_path).expanduser().resolve(strict=True)
    resolved_requests = Path(request_csv_path).expanduser().resolve(strict=True)
    resolved_requests_sha = Path(request_sha256_path).expanduser().resolve(strict=True)

    # Protocol/request regeneration intentionally happens before any checkpoint
    # replay and before the deterministic CUDA contract is installed.
    protocol = verify_locked_gradient_protocol(
        resolved_protocol,
        resolved_protocol_sha,
    )
    _validate_analysis_protocol_inputs(
        protocol,
        cohort_manifest_path=cohort_manifest,
        graph_manifest_path=graph_manifest,
    )
    gradient_inputs = load_prepared_gradient_request_inputs(
        cohort_dir=cohort_root,
        graph_dir=graph_root,
    )
    locked_requests = load_and_verify_locked_gradient_requests(
        resolved_requests,
        resolved_requests_sha,
        cores=gradient_inputs,
        protocol=protocol,
    )
    validated_input_hashes = {
        "analysis_protocol": sha256_file(resolved_protocol),
        "analysis_protocol_sidecar": sha256_file(resolved_protocol_sha),
        "gradient_request_table": sha256_file(resolved_requests),
        "gradient_request_sidecar": sha256_file(resolved_requests_sha),
        "cohort_manifest": sha256_file(cohort_manifest),
        "graph_manifest": sha256_file(graph_manifest),
    }
    if (
        validated_input_hashes["analysis_protocol"]
        != LOCKED_ANALYSIS_PROTOCOL_SHA256
        or validated_input_hashes["gradient_request_table"]
        != LOCKED_GRADIENT_REQUEST_TABLE_SHA256
    ):
        raise FourSeedStabilityError(
            "Analysis protocol or gradient-request table differs from the locked bytes."
        )
    derivative_requests = group_selected_derivative_requests(locked_requests)
    request_metadata = _locked_request_metadata(locked_requests)
    del gradient_inputs

    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_root,
        graph_dir=graph_root,
    )
    members = validate_four_members(
        run_roots,
        receipt_paths,
        cohort_manifest_path=cohort_manifest,
        graph_manifest_path=graph_manifest,
    )
    if tuple(sorted(checkpoint_paths)) != ACTIVE_SEEDS:
        raise FourSeedStabilityError(
            "Checkpoint inputs must contain exactly seeds 0,1,2,3."
        )
    for member in members:
        supplied_checkpoint = checkpoint_paths[member.seed].expanduser().resolve(
            strict=True
        )
        if supplied_checkpoint != member.checkpoint_path:
            raise FourSeedStabilityError(
                f"Seed {member.seed} checkpoint is not the canonical archived last.ckpt."
            )
    for member in members:
        try:
            output.relative_to(member.run_root)
        except ValueError:
            pass
        else:
            raise FourSeedStabilityError(
                "Analysis destination cannot mutate an immutable run bundle."
            )
    source_provenance, tracked_dirty_diff = _analysis_source_provenance()

    preflight_device, gpu_binding_preflight = _preflight_gpu_binding(device)
    resolved_device = install_deterministic_cuda_contract(preflight_device)
    if resolved_device.index >= torch.cuda.device_count():
        raise FourSeedStabilityError("Explicit CUDA device index is unavailable.")
    torch.cuda.set_device(resolved_device)
    torch.cuda.reset_peak_memory_stats(resolved_device)
    properties = torch.cuda.get_device_properties(resolved_device)
    gpu_binding = _finalize_gpu_binding(
        gpu_binding_preflight,
        properties=properties,
        device=resolved_device,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.work-", dir=output.parent)
    )
    try:
        extractions: list[SeedCompactExtraction] = []
        for member in members:
            set_deterministic_seed(
                ANALYSIS_DETERMINISTIC_SEED,
                deterministic=True,
                warn_only=False,
            )
            extractions.append(
                extract_member_compact(
                    member,
                    batches,
                    derivative_requests_by_alias=derivative_requests,
                    request_metadata=request_metadata,
                    device=resolved_device,
                    temporary_dir=temporary,
                    receiver_chunk_size=int(receiver_chunk_size),
                    max_edges_per_chunk=int(max_edges_per_chunk),
                    amp=bool(amp),
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.synchronize(resolved_device)
        extraction_finished_at = datetime.now(timezone.utc)
        execution = {
            "started_at_utc": started_at.isoformat(),
            "extraction_finished_at_utc": extraction_finished_at.isoformat(),
            "extraction_runtime_seconds": float(
                time.perf_counter() - started_monotonic
            ),
            "command_argv": list(sys.argv),
            "working_directory": Path.cwd().resolve().as_posix(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                name: importlib_metadata.version(name)
                for name in ("numpy", "scipy", "pyarrow", "torch")
            },
            "source": source_provenance,
            "cuda_device_order": gpu_binding["cuda_device_order"],
            "cuda_visible_devices": gpu_binding["cuda_visible_devices"],
            "gpu_binding": gpu_binding,
            "device": str(resolved_device),
            "device_name": str(properties.name),
            "device_uuid": (
                gpu_binding["torch_device_uuid"]
            ),
            "device_compute_capability": [
                int(getattr(properties, "major", 0)),
                int(getattr(properties, "minor", 0)),
            ],
            "device_multiprocessor_count": int(
                getattr(properties, "multi_processor_count", 0)
            ),
            "device_total_memory_bytes": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "deterministic_seed": ANALYSIS_DETERMINISTIC_SEED,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "matmul_tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32_enabled": bool(torch.backends.cudnn.allow_tf32),
            "attention_and_embedding_replay_amp_enabled": bool(amp),
            "attention_and_embedding_replay_amp_dtype": (
                "float16" if amp else None
            ),
            "selected_derivative_replay_amp_enabled": False,
            "selected_derivative_replay_dtype": "float32",
            "receiver_chunk_size": int(receiver_chunk_size),
            "max_edges_per_chunk": int(max_edges_per_chunk),
            "receiver_wise_softmax_exact": True,
            "neighbor_sampling": False,
            "models_processed_concurrently": 1,
            "cores_staged_on_cuda_concurrently": 1,
            "peak_cuda_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(resolved_device)
            ),
            "runtime_scope_note": (
                "report records extraction and CPU analysis through report "
                "finalization; publication verification runtime is returned by "
                "the command and is not retroactively inserted into the immutable bundle"
            ),
        }
        protocol_provenance = {
            "analysis_protocol": _json_safe(protocol),
            "analysis_protocol_path": resolved_protocol.as_posix(),
            "analysis_protocol_file_sha256": validated_input_hashes[
                "analysis_protocol"
            ],
            "analysis_protocol_sidecar_file_sha256": validated_input_hashes[
                "analysis_protocol_sidecar"
            ],
            "gradient_request_table_path": resolved_requests.as_posix(),
            "gradient_request_table_file_sha256": validated_input_hashes[
                "gradient_request_table"
            ],
            "gradient_request_sidecar_file_sha256": validated_input_hashes[
                "gradient_request_sidecar"
            ],
            "cohort_manifest_path": cohort_manifest.as_posix(),
            "cohort_manifest_file_sha256": validated_input_hashes[
                "cohort_manifest"
            ],
            "graph_manifest_path": graph_manifest.as_posix(),
            "graph_manifest_file_sha256": validated_input_hashes[
                "graph_manifest"
            ],
        }
        result = compute_four_seed_analysis(
            members,
            extractions,
            batches,
            protocol_provenance=protocol_provenance,
        )
        analysis_compute_finished_at = datetime.now(timezone.utc)
        execution["analysis_compute_finished_at_utc"] = (
            analysis_compute_finished_at.isoformat()
        )
        execution["analysis_compute_runtime_seconds"] = float(
            time.perf_counter() - started_monotonic
        )
        execution["bundle_report_finalized_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )
        report = dict(result.report)
        report["execution"] = execution
        result = FourSeedAnalysisResult(
            report=report,
            tables=result.tables,
            arrays=result.arrays,
        )
        dirty_diff_path = temporary / "analysis_git_dirty.diff"
        dirty_diff_path.write_bytes(tracked_dirty_diff)
        execution_path = temporary / "analysis_execution_provenance.json"
        execution_path.write_bytes(canonical_json_bytes(execution) + b"\n")
        provenance_files: dict[str, Path] = {
            "analysis_protocol_selected_gradient_stability_v1.yaml": (
                resolved_protocol
            ),
            "analysis_protocol_selected_gradient_stability_v1.sha256": (
                resolved_protocol_sha
            ),
            "selected_gradient_requests_v1.csv": resolved_requests,
            "selected_gradient_requests_v1.sha256": resolved_requests_sha,
            "cohort_manifest.json": cohort_manifest,
            "graph_manifest.json": graph_manifest,
            "analysis_git_dirty.diff": dirty_diff_path,
            "analysis_execution_provenance.json": execution_path,
        }
        project_root = Path(__file__).resolve().parents[2]
        provenance_files.update(
            {
                f"analysis_source_{Path(relative).name}": project_root / relative
                for relative in source_provenance["analysis_source_files"]
            }
        )
        provenance_files.update(
            {
                f"seed_{member.seed}_strict_checkpoint_verification.json": (
                    member.receipt_path
                )
                for member in members
            }
        )
        provenance_expected_sha256 = {
            "analysis_protocol_selected_gradient_stability_v1.yaml": (
                validated_input_hashes["analysis_protocol"]
            ),
            "analysis_protocol_selected_gradient_stability_v1.sha256": (
                validated_input_hashes["analysis_protocol_sidecar"]
            ),
            "selected_gradient_requests_v1.csv": validated_input_hashes[
                "gradient_request_table"
            ],
            "selected_gradient_requests_v1.sha256": validated_input_hashes[
                "gradient_request_sidecar"
            ],
            "cohort_manifest.json": validated_input_hashes["cohort_manifest"],
            "graph_manifest.json": validated_input_hashes["graph_manifest"],
            "analysis_git_dirty.diff": hashlib.sha256(
                tracked_dirty_diff
            ).hexdigest(),
            "analysis_execution_provenance.json": sha256_file(execution_path),
            **{
                f"analysis_source_{Path(relative).name}": str(record["sha256"])
                for relative, record in source_provenance[
                    "analysis_source_files"
                ].items()
            },
            **{
                f"seed_{member.seed}_strict_checkpoint_verification.json": (
                    member.receipt_file_sha256
                )
                for member in members
            },
        }
        manifest = write_four_seed_analysis_bundle(
            result,
            output,
            provenance_files=provenance_files,
            provenance_expected_sha256=provenance_expected_sha256,
        )
        verification = verify_four_seed_analysis_bundle(output)
        if verification["manifest_content_sha256"] != manifest[
            "manifest_content_sha256"
        ]:
            raise FourSeedStabilityError(
                "Published analysis verification returned a different manifest hash."
            )
        finished_at = datetime.now(timezone.utc)
        return {
            "status": "complete",
            "analysis_scope": "four_seed_ensemble_spread_seeds_0_1_2_3",
            "destination": output.as_posix(),
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "published_bundle_verified": True,
            "run_ids": {str(member.seed): member.run_id for member in members},
            "final_epochs": {
                str(member.seed): member.completed_global_epochs
                for member in members
            },
            "peak_cuda_memory_allocated_bytes": execution[
                "peak_cuda_memory_allocated_bytes"
            ],
            "finished_at_utc": finished_at.isoformat(),
            "total_runtime_seconds": float(time.perf_counter() - started_monotonic),
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "ACTIVE_SEEDS",
    "ANALYSIS_DETERMINISTIC_SEED",
    "ANALYSIS_SCHEMA",
    "DEFERRED_SEEDS",
    "EXPECTED_SEED_COUNT",
    "FINAL_LAYER",
    "FourSeedAnalysisResult",
    "FourSeedStabilityError",
    "HEAD_TOP_EDGE_FRACTION",
    "MANIFEST_SCHEMA",
    "MUTUAL_TOP_K_PER_CORE_SEED",
    "MutualPairScores",
    "QUANTILE_LEVELS",
    "RECEIVER_PROBES_PER_CORE",
    "SeedCompactExtraction",
    "ValidatedMember",
    "array_sha256",
    "canonical_sha256",
    "compute_four_seed_analysis",
    "evenly_spaced_receivers",
    "extract_member_compact",
    "install_deterministic_cuda_contract",
    "parse_seed_path_specs",
    "render_report_markdown",
    "run_four_seed_stability_pipeline",
    "scalar_summary",
    "top_mutual_pair_positions",
    "validate_four_members",
    "validate_member",
    "vectorized_mutual_pair_scores",
    "verify_four_seed_analysis_bundle",
    "write_deterministic_npz",
    "write_four_seed_analysis_bundle",
]

"""Prespecified synthetic recovery for the multiscale hurdle-count model.

The fixture uses only geometry from a checksum-bound, opaque-alias prepared
artifact.  Expression is generated from scratch.  A fixed-size sender program has
a positive one-hop effect on one receiver target, while a regional feature is
constructed after the target as a correlated, non-causal decoy.  A marginal-
preserving target permutation provides the null-injection counterpart.

This module deliberately keeps data generation, model fitting, attribution,
edge deletion, and gate evaluation separate.  A failed gate is valid diagnostic
evidence and must block real-data graph interpretation; it is not an execution
failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .artifacts import sha256_file
from .hurdle_continuous import hurdle_continuous_loss
from .identifiers import canonical_sha256
from .local_source_permutation import (
    LocalSourcePermutation,
    build_macroblock_spatial_antipode_permutation,
    verify_local_source_permutation_receipt,
)
from .multiscale_graphs import (
    TrueMultiscaleGraphs,
    build_true_multiscale_graphs,
    true_graph_receipt,
)
from .multiscale_hurdle_contract import (
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
)
from .multiscale_hurdle_training import (
    MultiscaleGraphSplitView,
    MultiscaleHurdleFixedMaskResult,
    MultiscaleHurdleTrainingResult,
    evaluate_fixed_multiscale_hurdle_mask,
    fit_full_core_multiscale_hurdle_model,
)
from .multiscale_hybrid import (
    MultiscaleAdditiveHybridModel,
    trainable_parameter_count,
)
from .training import TrainingConfig, set_deterministic_seed


SYNTHETIC_SCHEMA = "multiscale_hurdle_synthetic_recovery_v1"
SYNTHETIC_TASK_FAMILY = "masked_expression_hurdle_count"
SYNTHETIC_PROTOCOL = "held_in_full_core_fixed_budget"
SYNTHETIC_PRIMARY_METRIC = "fit/whole_node/hurdle_loss"
SYNTHETIC_GEOMETRY_ALIAS = "ANC-05"
SYNTHETIC_SELECTED_NODE_COUNT = 160
ARM_SELF_REGIONAL = "self_regional"
ARM_TRUE_LOCAL = "true_local"
ARM_PERMUTED_LOCAL = "permuted_local"
ARM_NULL_TRUE_LOCAL = "null_true_local"
SYNTHETIC_ARMS = (
    ARM_SELF_REGIONAL,
    ARM_TRUE_LOCAL,
    ARM_PERMUTED_LOCAL,
    ARM_NULL_TRUE_LOCAL,
)
_ALIAS = re.compile(r"ANC-(?:0[1-9]|10)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_IDENTIFIER_KEYS = frozenset(
    {
        "patient_id",
        "patient_identifier",
        "donor_id",
        "donor_identifier",
        "subject_id",
        "subject_identifier",
        "core_id",
        "core_identifier",
        "clinical_id",
        "clinical_identifier",
        "cell_id",
        "fov",
        "slide_id",
    }
)


class MultiscaleSyntheticError(RuntimeError):
    """Raised when the synthetic diagnostic contract cannot be verified."""


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def assert_alias_safe_configuration(
    value: Any,
    *,
    path: str = "config",
) -> None:
    """Reject direct biological-unit and row identifier fields recursively."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalized_key(raw_key)
            child = f"{path}.{raw_key}"
            if key in _FORBIDDEN_IDENTIFIER_KEYS:
                raise MultiscaleSyntheticError(
                    f"alias-only synthetic configuration prohibits {child}"
                )
            assert_alias_safe_configuration(item, path=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_alias_safe_configuration(item, path=f"{path}[{index}]")


def _array_sha256(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_correlation(first: Any, second: Any) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise MultiscaleSyntheticError(
            "correlation inputs must be aligned and nontrivial"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise MultiscaleSyntheticError("correlation inputs must be finite")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    result = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(result):
        raise MultiscaleSyntheticError("synthetic correlation is non-finite")
    return result


def _strict_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MultiscaleSyntheticError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise MultiscaleSyntheticError(f"{label} must be a JSON mapping")
    return dict(value)


def _resolved_prepared_directory(
    reference: str | Path,
    *,
    project_root: str | Path,
) -> tuple[Path, str]:
    root = Path(project_root).resolve(strict=True)
    raw_reference = Path(reference)
    path = raw_reference if raw_reference.is_absolute() else root / raw_reference
    if path.is_symlink():
        raise MultiscaleSyntheticError(
            "prepared artifact reference cannot be a symbolic link"
        )
    resolved = path.resolve(strict=True)
    allowed_roots = (
        (root / "data/processed").resolve(strict=False),
        (root / "artifacts").resolve(strict=False),
    )
    if not any(resolved.is_relative_to(allowed) for allowed in allowed_roots):
        raise MultiscaleSyntheticError(
            "observed geometry must come from data/processed or artifacts"
        )
    if not resolved.is_dir():
        raise MultiscaleSyntheticError(
            "prepared artifact reference must be a regular directory"
        )
    try:
        safe_reference = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise MultiscaleSyntheticError(
            "prepared artifact must remain within the project"
        ) from error
    return resolved, safe_reference


@dataclass(frozen=True)
class AliasSafeObservedGeometry:
    """Centered observed geometry with no row or clinical identifiers."""

    biological_unit_alias: str
    coordinates_um: np.ndarray = field(repr=False)
    macroblock_ids: np.ndarray = field(repr=False)
    stable_node_row_index: np.ndarray = field(repr=False)
    full_node_count: int
    selected_node_count: int
    prepared_data_sha256: str
    prepared_manifest_sha256: str
    selection_index_sha256: str
    geometry_sha256: str
    prepared_artifact_reference: str

    def __post_init__(self) -> None:
        if not _ALIAS.fullmatch(self.biological_unit_alias):
            raise ValueError("biological_unit_alias must be an opaque ANC alias")
        coordinates = np.asarray(self.coordinates_um, dtype=np.float64)
        macroblocks = np.asarray(self.macroblock_ids)
        stable_rows = np.asarray(
            self.stable_node_row_index,
            dtype=np.int64,
        )
        if (
            coordinates.shape != (int(self.selected_node_count), 2)
            or not np.isfinite(coordinates).all()
            or int(self.full_node_count) < int(self.selected_node_count)
        ):
            raise ValueError("observed geometry is malformed")
        if (
            macroblocks.shape != (int(self.selected_node_count),)
            or macroblocks.dtype.kind not in "US"
            or stable_rows.shape != (int(self.selected_node_count),)
            or len(np.unique(stable_rows)) != len(stable_rows)
            or bool((stable_rows < 0).any())
            or bool((stable_rows >= int(self.full_node_count)).any())
        ):
            raise ValueError(
                "observed macroblocks/stable row indices are malformed"
            )
        if _array_sha256(coordinates) != self.geometry_sha256:
            raise ValueError("observed geometry checksum does not verify")
        if _array_sha256(stable_rows) != self.selection_index_sha256:
            raise ValueError("stable row-index checksum does not verify")
        for value in (
            self.prepared_data_sha256,
            self.prepared_manifest_sha256,
            self.selection_index_sha256,
            self.geometry_sha256,
        ):
            if not _SHA256.fullmatch(str(value)):
                raise ValueError("observed geometry requires SHA-256 provenance")
        coordinates = np.ascontiguousarray(coordinates)
        macroblocks = np.ascontiguousarray(macroblocks.astype("U", copy=False))
        stable_rows = np.ascontiguousarray(stable_rows)
        coordinates.setflags(write=False)
        macroblocks.setflags(write=False)
        stable_rows.setflags(write=False)
        object.__setattr__(self, "coordinates_um", coordinates)
        object.__setattr__(self, "macroblock_ids", macroblocks)
        object.__setattr__(self, "stable_node_row_index", stable_rows)

    def receipt(self) -> dict[str, Any]:
        """Return identifier-free provenance suitable for a run bundle."""

        return {
            "schema": "alias_safe_observed_geometry_v1",
            "biological_unit_alias": self.biological_unit_alias,
            "full_node_count": self.full_node_count,
            "selected_node_count": self.selected_node_count,
            "prepared_data_sha256": self.prepared_data_sha256,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "selection_index_sha256": self.selection_index_sha256,
            "geometry_sha256": self.geometry_sha256,
            "macroblock_ids_sha256": _array_sha256(self.macroblock_ids),
            "macroblock_count": int(np.unique(self.macroblock_ids).size),
            "prepared_artifact_reference": self.prepared_artifact_reference,
            "coordinates_exported": False,
            "macroblock_ids_exported": False,
            "stable_node_row_indices_exported": False,
            "row_identifiers_loaded": False,
            "direct_identifiers_emitted": False,
        }


def load_alias_safe_observed_geometry(
    prepared_artifact_reference: str | Path,
    *,
    project_root: str | Path,
    biological_unit_alias: str,
    expected_prepared_data_sha256: str,
    selected_node_count: int = 160,
) -> AliasSafeObservedGeometry:
    """Load permitted geometry/null inputs from an opaque-alias artifact."""

    if not _ALIAS.fullmatch(str(biological_unit_alias)):
        raise MultiscaleSyntheticError(
            "biological_unit_alias must be an opaque ANC alias"
        )
    if not _SHA256.fullmatch(str(expected_prepared_data_sha256)):
        raise MultiscaleSyntheticError(
            "expected_prepared_data_sha256 must be a lowercase SHA-256"
        )
    if (
        isinstance(selected_node_count, bool)
        or int(selected_node_count) != selected_node_count
        or not 160 <= int(selected_node_count) <= 512
    ):
        raise MultiscaleSyntheticError(
            "selected_node_count must be an integer in [160, 512]"
        )
    artifact, safe_reference = _resolved_prepared_directory(
        prepared_artifact_reference,
        project_root=project_root,
    )
    manifest_path = artifact / "manifest.json"
    data_path = artifact / "prepared_data.npz"
    if (
        manifest_path.is_symlink()
        or data_path.is_symlink()
        or not manifest_path.is_file()
        or not data_path.is_file()
    ):
        raise MultiscaleSyntheticError(
            "prepared artifact lacks regular manifest/data files"
        )
    manifest = _strict_json_mapping(
        manifest_path,
        label="prepared artifact manifest",
    )
    selection = manifest.get("selection")
    files = manifest.get("files")
    arrays = manifest.get("arrays")
    if (
        not isinstance(selection, Mapping)
        or selection.get("opaque_alias") != biological_unit_alias
        or selection.get("restricted_identifiers_emitted") is not False
    ):
        raise MultiscaleSyntheticError(
            "prepared artifact is not bound to the declared opaque alias"
        )
    if not isinstance(files, Mapping) or not isinstance(arrays, Mapping):
        raise MultiscaleSyntheticError(
            "prepared artifact manifest lacks array/file receipts"
        )
    manifest_data_sha = files.get("prepared_data.npz")
    actual_data_sha = sha256_file(data_path)
    if (
        manifest_data_sha != expected_prepared_data_sha256
        or actual_data_sha != expected_prepared_data_sha256
    ):
        raise MultiscaleSyntheticError(
            "prepared_data.npz checksum does not match the frozen receipt"
        )
    coordinate_spec = arrays.get("coordinates_um")
    macroblock_spec = arrays.get("macroblock_ids")
    if (
        not isinstance(coordinate_spec, Mapping)
        or coordinate_spec.get("shape") is None
        or not isinstance(macroblock_spec, Mapping)
        or macroblock_spec.get("shape") is None
    ):
        raise MultiscaleSyntheticError(
            "prepared manifest lacks coordinates_um or macroblock_ids"
        )
    # np.load is lazy for NPZ members.  Only the two contract-permitted null
    # inputs are requested; no cell/FOV/clinical identifier array is loaded.
    try:
        with np.load(data_path, allow_pickle=False) as bundle:
            coordinates = np.asarray(bundle["coordinates_um"], dtype=np.float64)
            macroblocks = np.asarray(bundle["macroblock_ids"])
    except (OSError, ValueError, KeyError) as error:
        raise MultiscaleSyntheticError(
            "cannot load verified observed geometry/null inputs"
        ) from error
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < int(selected_node_count)
        or not np.isfinite(coordinates).all()
        or macroblocks.shape != (len(coordinates),)
        or macroblocks.dtype.kind not in "US"
    ):
        raise MultiscaleSyntheticError(
            "prepared geometry/null inputs are malformed"
        )
    center = np.median(coordinates, axis=0)
    squared_distance = np.square(coordinates - center).sum(axis=1)
    order = np.lexsort((np.arange(len(coordinates)), squared_distance))
    selected_indices = np.asarray(
        order[: int(selected_node_count)],
        dtype=np.int64,
    )
    selected = np.ascontiguousarray(coordinates[selected_indices])
    selected_macroblocks = np.ascontiguousarray(
        macroblocks[selected_indices].astype("U", copy=False)
    )
    # Remove absolute tissue position before any downstream use or checksum.
    selected -= np.median(selected, axis=0)
    return AliasSafeObservedGeometry(
        biological_unit_alias=str(biological_unit_alias),
        coordinates_um=selected,
        macroblock_ids=selected_macroblocks,
        stable_node_row_index=selected_indices,
        full_node_count=int(len(coordinates)),
        selected_node_count=int(len(selected)),
        prepared_data_sha256=actual_data_sha,
        prepared_manifest_sha256=sha256_file(manifest_path),
        selection_index_sha256=_array_sha256(selected_indices),
        geometry_sha256=_array_sha256(selected),
        prepared_artifact_reference=safe_reference,
    )


@dataclass(frozen=True)
class SyntheticRecoveryConfig:
    """Bounded fixed settings for one Stage-0 recovery execution."""

    seed: int = 0
    null_seed: int = 104729
    selected_node_count: int = SYNTHETIC_SELECTED_NODE_COUNT
    num_genes: int = 4
    active_sender_count: int = 24
    effect_size: float = 4.0
    baseline_positive_probability: float = 0.12
    regional_decoy_max_count: int = 8
    evaluation_node_rate: float = 0.20
    maximum_null_absolute_correlation: float = 0.05
    maximum_permuted_signal_correlation: float = 0.80
    top_edge_count: int = 16
    minimum_loss_advantage: float = 0.0
    minimum_contribution: float = 0.0
    minimum_deletion_advantage: float = 0.0
    max_epochs: int = 96
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    partial_gene_rate: float = 0.35
    node_rate: float = 0.35
    mask_seed: int = 314159
    model_seed: int = 2718
    target_node_batch_size: int = 512
    hidden_dim: int = 32
    decoder_dim: int = 32
    ffn_dim: int = 48
    attention_heads: int = 2
    attention_head_dim: int = 8
    value_head_dim: int = 4
    message_dim: int = 16
    edge_hidden_dim: int = 16
    edge_embedding_dim: int = 8
    receiver_chunk_size: int = 64
    activation_checkpointing: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    amp: bool = False
    amp_dtype: str = "auto"
    device: Optional[str] = None
    query_chunk_size: int = 128
    graph_receiver_chunk_size: int = 128
    mutual_search_chunk_size: int = 100_000
    graph_workers: int = 1

    def __post_init__(self) -> None:
        integer_bounds = {
            "selected_node_count": (160, 512),
            "num_genes": (4, 16),
            "active_sender_count": (1, 32),
            "regional_decoy_max_count": (1, 64),
            "top_edge_count": (1, 256),
            "max_epochs": (1, 256),
            "target_node_batch_size": (1, 4096),
            "hidden_dim": (4, 128),
            "decoder_dim": (4, 128),
            "ffn_dim": (4, 256),
            "attention_heads": (1, 8),
            "attention_head_dim": (1, 32),
            "value_head_dim": (1, 32),
            "message_dim": (1, 64),
            "edge_hidden_dim": (1, 128),
            "edge_embedding_dim": (1, 64),
            "receiver_chunk_size": (1, 512),
            "query_chunk_size": (1, 4096),
            "graph_receiver_chunk_size": (1, 4096),
            "mutual_search_chunk_size": (1, 10_000_000),
            "graph_workers": (1, 16),
        }
        for name, (minimum, maximum) in integer_bounds.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or int(value) != value
                or not minimum <= int(value) <= maximum
            ):
                raise ValueError(
                    f"{name} must be an integer in [{minimum}, {maximum}]"
                )
        if self.active_sender_count >= self.selected_node_count:
            raise ValueError("active_sender_count must be below node count")
        if self.selected_node_count != SYNTHETIC_SELECTED_NODE_COUNT:
            raise ValueError(
                "selected_node_count is frozen at "
                f"{SYNTHETIC_SELECTED_NODE_COUNT}"
            )
        for name in (
            "effect_size",
            "learning_rate",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in (
            "baseline_positive_probability",
            "evaluation_node_rate",
            "partial_gene_rate",
            "node_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if not 0.0 <= self.maximum_null_absolute_correlation < 0.5:
            raise ValueError(
                "maximum_null_absolute_correlation must be in [0, 0.5)"
            )
        if not 0.0 < self.maximum_permuted_signal_correlation < 1.0:
            raise ValueError(
                "maximum_permuted_signal_correlation must be in (0, 1)"
            )
        for name in (
            "minimum_loss_advantage",
            "minimum_contribution",
            "minimum_deletion_advantage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be auto, float16, or bfloat16")

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any]],
        *,
        seed: int,
        trainer: Optional[Mapping[str, Any]] = None,
    ) -> "SyntheticRecoveryConfig":
        settings = {} if value is None else dict(value)
        settings.setdefault("seed", int(seed))
        trainer_values = {} if trainer is None else dict(trainer)
        for destination, source in (
            ("max_epochs", "synthetic_max_epochs"),
            ("learning_rate", "synthetic_learning_rate"),
            ("device", "device"),
            ("amp", "synthetic_amp"),
            ("amp_dtype", "amp_dtype"),
            ("target_node_batch_size", "synthetic_target_node_batch_size"),
        ):
            if destination not in settings and source in trainer_values:
                settings[destination] = trainer_values[source]
        try:
            return cls(**settings)
        except TypeError as error:
            raise MultiscaleSyntheticError(
                f"unknown synthetic recovery setting: {error}"
            ) from error

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            huber_delta=1.0,
            curriculum="P+N",
            warmup_epochs=0,
            partial_gene_rate=self.partial_gene_rate,
            node_rate=self.node_rate,
            block_node_rate=0.10,
            mask_seed=self.mask_seed,
            model_seed=self.model_seed,
            edge_dropout=0.0,
            amp=self.amp,
            amp_dtype=self.amp_dtype,
            deterministic=True,
            deterministic_warn_only=False,
            device=self.device,
            restore_best=False,
        )


@dataclass(frozen=True)
class MultiscaleSyntheticFixture:
    """Planted/null counts and topology with fixed evaluation selections."""

    true_view: MultiscaleGraphSplitView
    permuted_view: MultiscaleGraphSplitView
    null_view: MultiscaleGraphSplitView
    expression_mean: Tensor
    expression_scale: Tensor
    null_expression_mean: Tensor
    null_expression_scale: Tensor
    evaluation_mask: Tensor
    sender_program_active: Tensor
    local_signal: Tensor
    planted_local_edge_mask: Tensor
    source_gene_indices: tuple[int, ...]
    target_gene_index: int
    regional_decoy_gene_index: int
    fixture_checksum: str
    audit: Mapping[str, Any]
    graph_audit: Mapping[str, Any]
    sender_state_permutation_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        n_nodes = self.true_view.num_nodes
        n_genes = self.true_view.num_genes
        if (
            self.permuted_view.num_nodes != n_nodes
            or self.null_view.num_nodes != n_nodes
            or self.permuted_view.num_genes != n_genes
            or self.null_view.num_genes != n_genes
        ):
            raise ValueError("synthetic views are not aligned")
        if (
            self.permuted_view.local_source_index_by_node is None
            or self.true_view.local_source_index_by_node is not None
            or self.null_view.local_source_index_by_node is not None
            or not torch.equal(
                self.true_view.local_edge_index,
                self.permuted_view.local_edge_index,
            )
            or not torch.equal(
                self.true_view.local_edge_attributes,
                self.permuted_view.local_edge_attributes,
            )
            or not torch.equal(
                self.true_view.regional_edge_index,
                self.permuted_view.regional_edge_index,
            )
            or not torch.equal(
                self.true_view.regional_edge_attributes,
                self.permuted_view.regional_edge_attributes,
            )
        ):
            raise ValueError(
                "permuted synthetic arm must change only local source states"
            )
        verify_local_source_permutation_receipt(
            self.sender_state_permutation_audit
        )
        if (
            self.audit.get("active_contract_amendment_sha256")
            != ACTIVE_CONTRACT_AMENDMENT_SHA256
            or self.audit.get("required_contract_supplement_sha256")
            != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
            or self.graph_audit.get("fixed_contract", {}).get(
                "rewired_graph_constructed"
            )
            is not False
        ):
            raise ValueError(
                "synthetic fixture is not bound to the active contracts "
                "and true-only graph receipt"
            )
        if (
            self.evaluation_mask.dtype != torch.bool
            or self.evaluation_mask.shape != (n_nodes, n_genes)
            or not bool(self.evaluation_mask.any())
        ):
            raise ValueError("synthetic evaluation mask is malformed")
        if not bool(
            torch.all(
                self.evaluation_mask
                == self.evaluation_mask[:, :1].expand_as(
                    self.evaluation_mask
                )
            )
        ):
            raise ValueError("synthetic evaluation mask must mask whole nodes")
        if self.sender_program_active.shape != (n_nodes,):
            raise ValueError("sender-program indicator is not node-aligned")
        if self.local_signal.shape != (n_nodes,):
            raise ValueError("local signal is not node-aligned")
        if self.planted_local_edge_mask.shape != (
            self.true_view.local_edge_index.shape[1],
        ):
            raise ValueError("planted edge mask is not edge-aligned")


def _standardization(counts: np.ndarray) -> tuple[Tensor, Tensor]:
    transformed = np.log1p(np.asarray(counts, dtype=np.float64))
    mean = transformed.mean(axis=0)
    scale = transformed.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(scale.astype(np.float32)),
    )


def _regional_decoy(
    target: np.ndarray,
    regional_edge_index: np.ndarray,
    *,
    maximum_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    source, receiver = regional_edge_index
    n_nodes = len(target)
    receiver_degree = np.bincount(receiver, minlength=n_nodes)
    back_projection = np.bincount(
        source,
        weights=target[receiver] / np.maximum(receiver_degree[receiver], 1),
        minlength=n_nodes,
    )
    span = float(np.ptp(back_projection))
    if span <= 0:
        raise MultiscaleSyntheticError(
            "regional graph cannot construct a correlated decoy"
        )
    normalized = (back_projection - back_projection.min()) / span
    base_decoy = np.rint(
        float(maximum_count) * normalized + rng.poisson(0.20, n_nodes)
    ).astype(np.int32)
    # Retain a deliberately imperfect decoy.  It must be correlated enough to
    # challenge the regional control, but not so strong that it substitutes for
    # the planted one-hop mechanism.  Mixing with a marginal-preserving
    # permutation makes that strength explicit and deterministic.
    for attempt in range(1, 101):
        permuted = rng.permutation(base_decoy)
        for mix_fraction in (0.10, 0.15, 0.20, 0.25, 0.30):
            decoy = np.rint(
                mix_fraction * base_decoy
                + (1.0 - mix_fraction) * permuted
            ).astype(np.int32)
            regional_aggregate = np.bincount(
                receiver,
                weights=decoy[source],
                minlength=n_nodes,
            ) / np.maximum(receiver_degree, 1)
            correlation = _safe_correlation(target, regional_aggregate)
            if 0.15 <= correlation <= 0.30:
                return (
                    decoy,
                    regional_aggregate,
                    mix_fraction,
                    attempt,
                )
    raise MultiscaleSyntheticError(
        "could not construct the bounded correlated regional decoy"
    )


def _null_target_permutation(
    target: np.ndarray,
    local_signal: np.ndarray,
    regional_decoy_signal: np.ndarray,
    *,
    seed: int,
    maximum_absolute_correlation: float,
) -> tuple[np.ndarray, int, float, float]:
    generator = np.random.default_rng(int(seed))
    for attempt in range(10_000):
        candidate = generator.permutation(target)
        local_correlation = _safe_correlation(candidate, local_signal)
        regional_correlation = _safe_correlation(
            candidate,
            regional_decoy_signal,
        )
        if (
            abs(local_correlation) <= maximum_absolute_correlation
            and abs(regional_correlation) <= maximum_absolute_correlation
        ):
            return (
                np.ascontiguousarray(candidate),
                attempt + 1,
                local_correlation,
                regional_correlation,
            )
    raise MultiscaleSyntheticError(
        "could not construct a correlation-bounded null injection"
    )


def _evaluation_nodes(
    *,
    sender_active: np.ndarray,
    planted_target: np.ndarray,
    null_target: np.ndarray,
    local_edge_index: np.ndarray,
    planted_edge_mask: np.ndarray,
    rate: float,
    top_edge_count: int,
    seed: int,
) -> np.ndarray:
    candidates = np.flatnonzero(~sender_active)
    requested = max(8, int(math.floor(len(sender_active) * rate + 0.5)))
    requested = min(requested, len(candidates))
    if requested < 4:
        raise MultiscaleSyntheticError(
            "synthetic whole-node mask has too few eligible receivers"
        )
    generator = np.random.default_rng(int(seed))
    receiver = local_edge_index[1]
    for _ in range(10_000):
        selected = np.sort(
            generator.choice(candidates, size=requested, replace=False)
        )
        if (
            np.unique(planted_target[selected] > 0).size == 2
            and np.unique(null_target[selected] > 0).size == 2
            and int(
                np.sum(
                    planted_edge_mask
                    & np.isin(receiver, selected, assume_unique=False)
                )
            )
            >= top_edge_count
        ):
            return selected.astype(np.int64, copy=False)
    raise MultiscaleSyntheticError(
        "cannot construct a stratified whole-node synthetic mask"
    )


def build_multiscale_synthetic_fixture(
    geometry: AliasSafeObservedGeometry,
    config: SyntheticRecoveryConfig,
) -> MultiscaleSyntheticFixture:
    """Build planted and null data on one deterministic observed geometry."""

    if geometry.selected_node_count != config.selected_node_count:
        raise MultiscaleSyntheticError(
            "geometry node count differs from the synthetic configuration"
        )
    graphs: TrueMultiscaleGraphs = build_true_multiscale_graphs(
        geometry.coordinates_um,
        query_chunk_size=config.query_chunk_size,
        receiver_chunk_size=config.graph_receiver_chunk_size,
        mutual_search_chunk_size=config.mutual_search_chunk_size,
        workers=config.graph_workers,
    )
    local_edges, local_attributes = graphs.local.concatenate()
    regional_edges, regional_attributes = graphs.regional.concatenate()
    source_permutation: LocalSourcePermutation = (
        build_macroblock_spatial_antipode_permutation(
            geometry.coordinates_um,
            geometry.macroblock_ids,
            local_edge_index=local_edges,
            local_edge_attributes=local_attributes,
            enforce_qc=True,
        )
    )
    verify_local_source_permutation_receipt(source_permutation.receipt)
    n_nodes = geometry.selected_node_count
    generator = np.random.default_rng(int(config.seed))

    local_source, local_receiver = local_edges
    permuted_source = source_permutation.source_index_by_node[local_source]
    sender_active: np.ndarray | None = None
    local_signal: np.ndarray | None = None
    permuted_signal: np.ndarray | None = None
    permuted_signal_correlation: float | None = None
    sender_pattern_attempt = 0
    for sender_pattern_attempt in range(1, 10_001):
        candidate_active = np.zeros(n_nodes, dtype=bool)
        candidate_active[
            generator.choice(
                n_nodes,
                size=config.active_sender_count,
                replace=False,
            )
        ] = True
        candidate_local = np.bincount(
            local_receiver,
            weights=candidate_active[local_source].astype(np.float64),
            minlength=n_nodes,
        )
        candidate_permuted = np.bincount(
            local_receiver,
            weights=candidate_active[permuted_source].astype(np.float64),
            minlength=n_nodes,
        )
        correlation = _safe_correlation(
            candidate_local,
            candidate_permuted,
        )
        if correlation <= config.maximum_permuted_signal_correlation:
            sender_active = candidate_active
            local_signal = candidate_local
            permuted_signal = candidate_permuted
            permuted_signal_correlation = correlation
            break
    if (
        sender_active is None
        or local_signal is None
        or permuted_signal is None
        or permuted_signal_correlation is None
    ):
        raise MultiscaleSyntheticError(
            "could not construct a sender-state-null-discriminating program"
        )
    if np.unique(local_signal).size < 2:
        raise MultiscaleSyntheticError("planted local signal is constant")

    counts = np.zeros((n_nodes, config.num_genes), dtype=np.int32)
    counts[:, 0] = np.where(
        sender_active,
        8 + generator.poisson(2.0, n_nodes),
        generator.binomial(1, 0.03, n_nodes),
    )
    counts[:, 1] = np.where(
        sender_active,
        4 + generator.poisson(1.0, n_nodes),
        generator.binomial(1, 0.02, n_nodes),
    )
    signal_threshold = float(np.median(local_signal))
    positive_local_excess = np.maximum(
        local_signal - signal_threshold,
        0.0,
    )
    planted_edge_mask = (
        sender_active[local_source]
        & (positive_local_excess[local_receiver] > 0)
    )
    baseline_positive = (
        generator.random(n_nodes) < config.baseline_positive_probability
    )
    baseline_count = baseline_positive * generator.integers(1, 3, n_nodes)
    target_gene = 2
    decoy_gene = 3
    planted_target = np.rint(
        baseline_count + config.effect_size * positive_local_excess
    ).astype(np.int32)
    counts[:, target_gene] = planted_target
    (
        decoy,
        regional_decoy_signal,
        regional_decoy_mix_fraction,
        regional_decoy_attempts,
    ) = _regional_decoy(
        planted_target,
        regional_edges,
        maximum_count=config.regional_decoy_max_count,
        rng=generator,
    )
    counts[:, decoy_gene] = decoy
    for gene in range(4, config.num_genes):
        counts[:, gene] = generator.poisson(
            0.25 + 0.10 * (gene - 3),
            n_nodes,
        ).astype(np.int32)

    null_target, null_attempts, null_local_corr, null_regional_corr = (
        _null_target_permutation(
            planted_target,
            local_signal,
            regional_decoy_signal,
            seed=config.null_seed,
            maximum_absolute_correlation=(
                config.maximum_null_absolute_correlation
            ),
        )
    )
    null_counts = counts.copy()
    null_counts[:, target_gene] = null_target
    evaluation_nodes = _evaluation_nodes(
        sender_active=sender_active,
        planted_target=planted_target,
        null_target=null_target,
        local_edge_index=local_edges,
        planted_edge_mask=planted_edge_mask,
        rate=config.evaluation_node_rate,
        top_edge_count=config.top_edge_count,
        seed=config.seed + 7919,
    )
    evaluation_mask = np.zeros_like(counts, dtype=bool)
    evaluation_mask[evaluation_nodes, :] = True

    mean, scale = _standardization(counts)
    null_mean, null_scale = _standardization(null_counts)
    coordinates = torch.from_numpy(geometry.coordinates_um.copy())
    common = {
        "coordinates_um": coordinates,
        "regional_edge_index": torch.from_numpy(regional_edges.copy()),
        "regional_edge_attributes": torch.from_numpy(
            regional_attributes.copy()
        ),
        "node_covariates": None,
        "block_ids": geometry.macroblock_ids.copy(),
        "name": "fit",
    }
    true_view = MultiscaleGraphSplitView(
        expression=torch.from_numpy(counts.astype(np.float32)),
        local_edge_index=torch.from_numpy(local_edges.copy()),
        local_edge_attributes=torch.from_numpy(local_attributes.copy()),
        **common,
    )
    permuted_view = MultiscaleGraphSplitView(
        expression=torch.from_numpy(counts.astype(np.float32)),
        local_edge_index=torch.from_numpy(local_edges.copy()),
        local_edge_attributes=torch.from_numpy(local_attributes.copy()),
        local_source_index_by_node=torch.from_numpy(
            source_permutation.source_index_by_node.copy()
        ),
        **common,
    )
    null_view = MultiscaleGraphSplitView(
        expression=torch.from_numpy(null_counts.astype(np.float32)),
        local_edge_index=torch.from_numpy(local_edges.copy()),
        local_edge_attributes=torch.from_numpy(local_attributes.copy()),
        **common,
    )
    audit: dict[str, Any] = {
        "schema": SYNTHETIC_SCHEMA,
        "active_contract_amendment_sha256": (
            ACTIVE_CONTRACT_AMENDMENT_SHA256
        ),
        "required_contract_supplement_sha256": (
            REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        ),
        "seed": config.seed,
        "null_seed": config.null_seed,
        "n_nodes": n_nodes,
        "n_genes": config.num_genes,
        "source_gene_indices": [0, 1],
        "target_gene_index": target_gene,
        "regional_decoy_gene_index": decoy_gene,
        "active_sender_count": int(sender_active.sum()),
        "active_sender_fraction": float(sender_active.mean()),
        "sender_selection_rule": (
            "seeded_fixed_size_program_with_bounded_true_permuted_"
            "signal_correlation"
        ),
        "sender_pattern_attempts": sender_pattern_attempt,
        "maximum_permuted_signal_correlation": (
            config.maximum_permuted_signal_correlation
        ),
        "observed_permuted_signal_correlation": (
            permuted_signal_correlation
        ),
        "sender_state_permutation_checksum": (
            source_permutation.receipt["checksum"]
        ),
        "sender_state_permutation_gate_passed": (
            source_permutation.receipt["qc"]["gate_passed"]
        ),
        "local_edge_topology_bit_identical_between_true_and_permuted": True,
        "local_edge_attributes_bit_identical_between_true_and_permuted": (
            True
        ),
        "local_receivers_identical_between_true_and_permuted": True,
        "local_signal_threshold": signal_threshold,
        "effect_sign": "positive",
        "effect_size": config.effect_size,
        "planted_directed_edge_count": int(planted_edge_mask.sum()),
        "evaluation_node_count": int(len(evaluation_nodes)),
        "evaluation_mask_checksum": _array_sha256(evaluation_mask),
        "planted_local_target_correlation": _safe_correlation(
            planted_target,
            local_signal,
        ),
        "regional_decoy_target_correlation": _safe_correlation(
            planted_target,
            regional_decoy_signal,
        ),
        "regional_decoy_base_mix_fraction": regional_decoy_mix_fraction,
        "regional_decoy_permutation_attempts": regional_decoy_attempts,
        "null_local_target_correlation": null_local_corr,
        "null_regional_target_correlation": null_regional_corr,
        "null_permutation_attempts": null_attempts,
        "null_preserves_target_marginal_exactly": bool(
            np.array_equal(np.sort(null_target), np.sort(planted_target))
        ),
        "pre_outcome_geometry_selection": {
            "biological_unit_alias": SYNTHETIC_GEOMETRY_ALIAS,
            "selected_node_count": SYNTHETIC_SELECTED_NODE_COUNT,
            "rejected_candidate": {
                "selected_node_count": 240,
                "node_mapping_changed_fraction": 1.0,
                "node_displacement_above_75um_fraction": (
                    0.8583333333333333
                ),
                "reason": "failed_frozen_0.90_displacement_gate",
            },
            "selected_candidate": {
                "selected_node_count": SYNTHETIC_SELECTED_NODE_COUNT,
                "node_mapping_changed_fraction": 1.0,
                "node_displacement_above_75um_fraction": 0.9625,
            },
            "selection_inputs": [
                "coordinates_um",
                "macroblock_ids",
                "stable_node_row_index",
            ],
            "target_expression_examined": False,
            "model_outcomes_examined": False,
        },
        "observed_geometry": geometry.receipt(),
        "direct_identifiers_emitted": False,
    }
    graph_audit = true_graph_receipt(graphs)
    fixture_payload = {
        "audit": audit,
        "graph_bundle_sha256": graphs.checksums.bundle_sha256,
        "sender_state_permutation_receipt_checksum": (
            source_permutation.receipt["checksum"]
        ),
        "source_index_by_node_sha256": source_permutation.receipt[
            "source_index_by_node_sha256"
        ],
        "planted_counts_sha256": _array_sha256(counts),
        "null_counts_sha256": _array_sha256(null_counts),
        "sender_program_sha256": _array_sha256(sender_active),
        "local_signal_sha256": _array_sha256(local_signal),
    }
    fixture_checksum = canonical_sha256(fixture_payload)
    audit["fixture_checksum"] = fixture_checksum
    return MultiscaleSyntheticFixture(
        true_view=true_view,
        permuted_view=permuted_view,
        null_view=null_view,
        expression_mean=mean,
        expression_scale=scale,
        null_expression_mean=null_mean,
        null_expression_scale=null_scale,
        evaluation_mask=torch.from_numpy(evaluation_mask),
        sender_program_active=torch.from_numpy(sender_active),
        local_signal=torch.from_numpy(local_signal.astype(np.float32)),
        planted_local_edge_mask=torch.from_numpy(planted_edge_mask),
        source_gene_indices=(0, 1),
        target_gene_index=target_gene,
        regional_decoy_gene_index=decoy_gene,
        fixture_checksum=fixture_checksum,
        audit=audit,
        graph_audit=graph_audit,
        sender_state_permutation_audit=source_permutation.receipt,
    )


@dataclass(frozen=True)
class SyntheticArmOutcome:
    """One parameter-matched arm's immutable training/evaluation evidence."""

    arm: str
    regional_routing: str
    local_routing: str
    parameter_count: int
    parameter_structure_sha256: str
    initial_parameter_sha256: str
    training: MultiscaleHurdleTrainingResult
    evaluation: MultiscaleHurdleFixedMaskResult

    @property
    def whole_node_loss(self) -> float:
        return float(self.evaluation.evaluation.metrics["hurdle_loss"])


@dataclass(frozen=True)
class SyntheticDeletionDiagnostic:
    """Signed-contribution and matched-deletion evidence for one fitted model."""

    selected_contribution_mean: float
    selected_contribution_median: float
    selected_contribution_positive_fraction: float
    selected_edge_count: int
    top_edge_count: int
    selected_edge_checksum: str
    matched_null_edge_checksum: str
    mean_top_edge_distance_um: float
    mean_matched_null_edge_distance_um: float
    maximum_absolute_match_distance_difference_um: float
    baseline_target_loss: float
    top_deleted_target_loss: float
    matched_null_deleted_target_loss: float
    top_deletion_loss_delta: float
    matched_null_deletion_loss_delta: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"deletion diagnostic {name} is non-finite")
        if self.selected_edge_count < self.top_edge_count or self.top_edge_count <= 0:
            raise ValueError("deletion diagnostic edge counts are invalid")


@dataclass(frozen=True)
class SyntheticRecoveryGate:
    """Prespecified Stage-0 gate decisions."""

    node_mapping_changed_fraction: float
    node_mapping_changed_sufficient: bool
    node_displacement_above_threshold_fraction: float
    node_displacement_sufficient: bool
    effective_source_edge_slot_identity_changed_fraction: float
    edge_slot_sender_identity_change_sufficient: bool
    sender_state_permutation_qc_passed: bool
    true_local_beats_self_regional: bool
    true_local_beats_permuted_local: bool
    correct_positive_contribution_sign: bool
    top_deletion_exceeds_matched_null: bool
    analogous_null_discovery: bool
    no_analogous_null_discovery: bool
    gate_passed: bool
    failure_reasons: tuple[str, ...]
    thresholds: Mapping[str, float]


@dataclass(frozen=True)
class MultiscaleSyntheticRecoveryResult:
    """All model, attribution, deletion, and gate evidence for one fixture."""

    fixture: MultiscaleSyntheticFixture
    config: SyntheticRecoveryConfig
    arms: Mapping[str, SyntheticArmOutcome]
    planted_diagnostic: SyntheticDeletionDiagnostic
    null_diagnostic: SyntheticDeletionDiagnostic
    gate: SyntheticRecoveryGate
    parameter_match_verified: bool

    def final_metrics(self) -> dict[str, float | int]:
        true_loss = self.arms[ARM_TRUE_LOCAL].whole_node_loss
        self_regional_loss = self.arms[ARM_SELF_REGIONAL].whole_node_loss
        permuted_loss = self.arms[ARM_PERMUTED_LOCAL].whole_node_loss
        null_loss = self.arms[ARM_NULL_TRUE_LOCAL].whole_node_loss
        metrics: dict[str, float | int] = {
            SYNTHETIC_PRIMARY_METRIC: true_loss,
            "synthetic/self_regional_whole_node_hurdle_loss": (
                self_regional_loss
            ),
            "synthetic/permuted_local_whole_node_hurdle_loss": (
                permuted_loss
            ),
            "synthetic/null_true_local_whole_node_hurdle_loss": null_loss,
            "synthetic/true_local_advantage_over_self_regional": (
                self_regional_loss - true_loss
            ),
            "synthetic/true_local_advantage_over_permuted_local": (
                permuted_loss - true_loss
            ),
            "synthetic/planted_selected_contribution_mean": (
                self.planted_diagnostic.selected_contribution_mean
            ),
            "synthetic/planted_top_deletion_target_loss_delta": (
                self.planted_diagnostic.top_deletion_loss_delta
            ),
            "synthetic/planted_matched_null_deletion_target_loss_delta": (
                self.planted_diagnostic.matched_null_deletion_loss_delta
            ),
            "synthetic/planted_deletion_advantage": (
                self.planted_diagnostic.top_deletion_loss_delta
                - self.planted_diagnostic.matched_null_deletion_loss_delta
            ),
            "synthetic/null_selected_contribution_mean": (
                self.null_diagnostic.selected_contribution_mean
            ),
            "synthetic/null_top_deletion_target_loss_delta": (
                self.null_diagnostic.top_deletion_loss_delta
            ),
            "synthetic/null_matched_null_deletion_target_loss_delta": (
                self.null_diagnostic.matched_null_deletion_loss_delta
            ),
            "synthetic/null_deletion_advantage": (
                self.null_diagnostic.top_deletion_loss_delta
                - self.null_diagnostic.matched_null_deletion_loss_delta
            ),
            "synthetic/parameter_match_verified": int(
                self.parameter_match_verified
            ),
            "synthetic/sender_state_node_mapping_changed_fraction": (
                self.gate.node_mapping_changed_fraction
            ),
            "synthetic/sender_state_node_displacement_above_threshold_fraction": (
                self.gate.node_displacement_above_threshold_fraction
            ),
            "synthetic/sender_state_edge_slot_identity_changed_fraction": (
                self.gate.effective_source_edge_slot_identity_changed_fraction
            ),
            "synthetic/gate_sender_state_node_mapping_sufficient": int(
                self.gate.node_mapping_changed_sufficient
            ),
            "synthetic/gate_sender_state_displacement_sufficient": int(
                self.gate.node_displacement_sufficient
            ),
            "synthetic/gate_sender_state_edge_slot_change_sufficient": int(
                self.gate.edge_slot_sender_identity_change_sufficient
            ),
            "synthetic/gate_sender_state_permutation_qc_passed": int(
                self.gate.sender_state_permutation_qc_passed
            ),
            "synthetic/gate_true_local_beats_self_regional": int(
                self.gate.true_local_beats_self_regional
            ),
            "synthetic/gate_true_local_beats_permuted_local": int(
                self.gate.true_local_beats_permuted_local
            ),
            "synthetic/gate_correct_positive_contribution_sign": int(
                self.gate.correct_positive_contribution_sign
            ),
            "synthetic/gate_top_deletion_exceeds_matched_null": int(
                self.gate.top_deletion_exceeds_matched_null
            ),
            "synthetic/gate_no_analogous_null_discovery": int(
                self.gate.no_analogous_null_discovery
            ),
            "synthetic/gate_passed": int(self.gate.gate_passed),
        }
        if not all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            raise FloatingPointError("synthetic final metric is non-finite")
        return metrics


def _parameter_structure(model: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(parameter.shape))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _parameter_structure_sha256(model: nn.Module) -> str:
    return canonical_sha256(
        {
            "parameters": [
                {"name": name, "shape": list(shape)}
                for name, shape in _parameter_structure(model)
            ]
        }
    )


def _parameter_value_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _make_model(
    fixture: MultiscaleSyntheticFixture,
    config: SyntheticRecoveryConfig,
    *,
    regional_routing: str,
    local_routing: str,
    null_injection: bool,
) -> MultiscaleAdditiveHybridModel:
    set_deterministic_seed(
        config.model_seed,
        deterministic=True,
        warn_only=False,
    )
    mean = (
        fixture.null_expression_mean
        if null_injection
        else fixture.expression_mean
    )
    scale = (
        fixture.null_expression_scale
        if null_injection
        else fixture.expression_scale
    )
    return MultiscaleAdditiveHybridModel(
        num_genes=fixture.true_view.num_genes,
        local_edge_attribute_dim=(
            fixture.true_view.local_edge_attributes.shape[1]
        ),
        regional_edge_attribute_dim=(
            fixture.true_view.regional_edge_attributes.shape[1]
        ),
        expression_mean=mean,
        expression_scale=scale,
        node_covariate_dim=0,
        hidden_dim=config.hidden_dim,
        decoder_dim=config.decoder_dim,
        ffn_dim=config.ffn_dim,
        attention_heads=config.attention_heads,
        attention_head_dim=config.attention_head_dim,
        value_head_dim=config.value_head_dim,
        message_dim=config.message_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        output_channels=2,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        receiver_chunk_size=config.receiver_chunk_size,
        activation_checkpointing=config.activation_checkpointing,
        regional_routing=regional_routing,
        local_routing=local_routing,
    )


def _resolved_model_device(model: nn.Module, requested: Optional[str]) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise MultiscaleSyntheticError("CUDA was requested but is unavailable")
    model.to(device)
    return device


def _target_nodes(mask: Tensor, *, device: torch.device) -> Tensor:
    nodes = mask.any(dim=1).nonzero(as_tuple=False).flatten()
    if nodes.numel() == 0:
        raise MultiscaleSyntheticError("synthetic mask contains no target nodes")
    return nodes.to(device=device, dtype=torch.long)


def _target_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    target_gene_index: int,
    expression_mean: Tensor,
    expression_scale: Tensor,
) -> float:
    selected_prediction = prediction[
        :,
        target_gene_index : target_gene_index + 1,
        :,
    ]
    selected_target = target[
        :,
        target_gene_index : target_gene_index + 1,
    ]
    selected_mask = torch.ones_like(selected_target, dtype=torch.bool)
    loss = hurdle_continuous_loss(
        selected_prediction,
        selected_target,
        selected_mask,
        expression_mean=expression_mean[
            target_gene_index : target_gene_index + 1
        ],
        expression_scale=expression_scale[
            target_gene_index : target_gene_index + 1
        ],
        huber_delta=1.0,
    )
    value = float(loss.total.detach().float().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("synthetic target loss is non-finite")
    return value


def _distance_matched_null_edges(
    *,
    edge_index: Tensor,
    coordinates_um: Tensor,
    planted_edge_mask: Tensor,
    top_edge_indices: Tensor,
) -> Tensor:
    edges = edge_index.detach().cpu()
    coordinates = coordinates_um.detach().cpu().double()
    planted = planted_edge_mask.detach().cpu().bool()
    top = top_edge_indices.detach().cpu().long()
    distances = torch.linalg.vector_norm(
        coordinates.index_select(0, edges[0])
        - coordinates.index_select(0, edges[1]),
        dim=1,
    )
    selected: list[int] = []
    used = set(int(value) for value in top.tolist())
    all_candidates = (~planted).nonzero(as_tuple=False).flatten()
    for edge_id in top.tolist():
        receiver = int(edges[1, edge_id])
        same_receiver = all_candidates[
            edges[1].index_select(0, all_candidates) == receiver
        ]
        eligible = [
            int(value)
            for value in same_receiver.tolist()
            if int(value) not in used
        ]
        if not eligible:
            eligible = [
                int(value)
                for value in all_candidates.tolist()
                if int(value) not in used
            ]
        if not eligible:
            raise MultiscaleSyntheticError(
                "no unique non-planted edge remains for distance matching"
            )
        target_distance = float(distances[edge_id])
        chosen = min(
            eligible,
            key=lambda candidate: (
                abs(float(distances[candidate]) - target_distance),
                candidate,
            ),
        )
        selected.append(chosen)
        used.add(chosen)
    return torch.tensor(selected, dtype=torch.long)


def _model_output(
    model: MultiscaleAdditiveHybridModel,
    view: MultiscaleGraphSplitView,
    mask: Tensor,
    *,
    device: torch.device,
    local_edge_index: Tensor,
    local_edge_attributes: Tensor,
    selected_local_edges: Optional[Tensor] = None,
    target_gene_index: Optional[int] = None,
) -> Any:
    expression = view.expression.to(device=device, dtype=torch.float32)
    mask_device = mask.to(device=device, dtype=torch.bool)
    nodes = _target_nodes(mask, device=device)
    masked_expression = expression.masked_fill(mask_device, 0.0)
    return model(
        masked_expression,
        mask_device,
        node_covariates=None,
        regional_edge_index=view.regional_edge_index.to(
            device=device,
            dtype=torch.long,
        ),
        regional_edge_attributes=view.regional_edge_attributes.to(
            device=device,
            dtype=torch.float32,
        ),
        local_edge_index=local_edge_index.to(device=device, dtype=torch.long),
        local_edge_attributes=local_edge_attributes.to(
            device=device,
            dtype=torch.float32,
        ),
        target_nodes=nodes,
        local_contribution_edge_indices=(
            None
            if selected_local_edges is None
            else selected_local_edges.to(device=device, dtype=torch.long)
        ),
        local_contribution_gene_indices=(
            None if target_gene_index is None else [target_gene_index]
        ),
    )


def local_contribution_deletion_diagnostic(
    model: MultiscaleAdditiveHybridModel,
    fixture: MultiscaleSyntheticFixture,
    *,
    null_injection: bool,
    top_edge_count: int,
    device: Optional[str],
) -> SyntheticDeletionDiagnostic:
    """Measure signed planted-edge contributions and a matched deletion test."""

    if model.local_routing != "true":
        raise MultiscaleSyntheticError(
            "deletion diagnostic requires true local routing"
        )
    view = fixture.null_view if null_injection else fixture.true_view
    mean = (
        fixture.null_expression_mean
        if null_injection
        else fixture.expression_mean
    )
    scale = (
        fixture.null_expression_scale
        if null_injection
        else fixture.expression_scale
    )
    resolved_device = _resolved_model_device(model, device)
    mask = fixture.evaluation_mask
    evaluation_receivers = mask.any(dim=1)
    receiver = view.local_edge_index[1]
    selected_mask = (
        fixture.planted_local_edge_mask
        & evaluation_receivers.index_select(0, receiver)
    )
    selected_edges = selected_mask.nonzero(as_tuple=False).flatten()
    if selected_edges.numel() < int(top_edge_count):
        raise MultiscaleSyntheticError(
            "too few planted evaluation edges for deletion diagnostic"
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = _model_output(
                model,
                view,
                mask,
                device=resolved_device,
                local_edge_index=view.local_edge_index,
                local_edge_attributes=view.local_edge_attributes,
                selected_local_edges=selected_edges,
                target_gene_index=fixture.target_gene_index,
            )
            contributions = output.local_contributions
            if (
                contributions is None
                or contributions.shape
                != (selected_edges.numel(), 1, 2)
            ):
                raise MultiscaleSyntheticError(
                    "model did not return aligned selected contributions"
                )
            continuous_contribution = contributions[:, 0, 1].float()
            if not bool(torch.isfinite(continuous_contribution).all()):
                raise FloatingPointError(
                    "selected local contributions are non-finite"
                )
            top_order = torch.argsort(
                continuous_contribution,
                descending=True,
                stable=True,
            )[: int(top_edge_count)]
            top_edges = selected_edges.index_select(
                0,
                top_order.detach().cpu(),
            )
            null_edges = _distance_matched_null_edges(
                edge_index=view.local_edge_index,
                coordinates_um=view.coordinates_um,
                planted_edge_mask=fixture.planted_local_edge_mask,
                top_edge_indices=top_edges,
            )
            target_nodes = _target_nodes(mask, device=resolved_device)
            raw_target = view.expression.index_select(
                0,
                target_nodes.detach().cpu(),
            ).to(device=resolved_device)
            baseline_loss = _target_loss(
                output.prediction,
                raw_target,
                target_gene_index=fixture.target_gene_index,
                expression_mean=mean.to(device=resolved_device),
                expression_scale=scale.to(device=resolved_device),
            )

            deleted_losses: list[float] = []
            for deleted in (top_edges, null_edges):
                keep = torch.ones(
                    view.local_edge_index.shape[1],
                    dtype=torch.bool,
                )
                keep[deleted] = False
                deleted_output = _model_output(
                    model,
                    view,
                    mask,
                    device=resolved_device,
                    local_edge_index=view.local_edge_index[:, keep],
                    local_edge_attributes=view.local_edge_attributes[keep],
                )
                deleted_losses.append(
                    _target_loss(
                        deleted_output.prediction,
                        raw_target,
                        target_gene_index=fixture.target_gene_index,
                        expression_mean=mean.to(device=resolved_device),
                        expression_scale=scale.to(device=resolved_device),
                    )
                )
    finally:
        model.train(was_training)

    coordinates = view.coordinates_um.detach().cpu().double()
    edges = view.local_edge_index.detach().cpu()
    edge_distances = torch.linalg.vector_norm(
        coordinates.index_select(0, edges[0])
        - coordinates.index_select(0, edges[1]),
        dim=1,
    )
    top_distances = edge_distances.index_select(0, top_edges)
    null_distances = edge_distances.index_select(0, null_edges)
    return SyntheticDeletionDiagnostic(
        selected_contribution_mean=float(
            continuous_contribution.mean().detach().cpu()
        ),
        selected_contribution_median=float(
            continuous_contribution.median().detach().cpu()
        ),
        selected_contribution_positive_fraction=float(
            (continuous_contribution > 0).float().mean().detach().cpu()
        ),
        selected_edge_count=int(selected_edges.numel()),
        top_edge_count=int(top_edges.numel()),
        selected_edge_checksum=_array_sha256(top_edges),
        matched_null_edge_checksum=_array_sha256(null_edges),
        mean_top_edge_distance_um=float(top_distances.mean()),
        mean_matched_null_edge_distance_um=float(null_distances.mean()),
        maximum_absolute_match_distance_difference_um=float(
            (top_distances - null_distances).abs().max()
        ),
        baseline_target_loss=baseline_loss,
        top_deleted_target_loss=deleted_losses[0],
        matched_null_deleted_target_loss=deleted_losses[1],
        top_deletion_loss_delta=deleted_losses[0] - baseline_loss,
        matched_null_deletion_loss_delta=deleted_losses[1] - baseline_loss,
    )


def evaluate_synthetic_recovery_gate(
    *,
    node_mapping_changed_fraction: float,
    node_displacement_above_threshold_fraction: float,
    effective_source_edge_slot_identity_changed_fraction: float,
    sender_state_permutation_qc_passed: bool,
    self_regional_loss: float,
    true_local_loss: float,
    permuted_local_loss: float,
    planted_diagnostic: SyntheticDeletionDiagnostic,
    null_diagnostic: SyntheticDeletionDiagnostic,
    minimum_loss_advantage: float = 0.0,
    minimum_contribution: float = 0.0,
    minimum_deletion_advantage: float = 0.0,
) -> SyntheticRecoveryGate:
    """Apply the frozen relational Stage-0 gate without post-outcome tuning."""

    scalar_values = (
        node_mapping_changed_fraction,
        node_displacement_above_threshold_fraction,
        effective_source_edge_slot_identity_changed_fraction,
        self_regional_loss,
        true_local_loss,
        permuted_local_loss,
        minimum_loss_advantage,
        minimum_contribution,
        minimum_deletion_advantage,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("synthetic gate inputs must be finite")
    for name, value in (
        ("node_mapping_changed_fraction", node_mapping_changed_fraction),
        (
            "node_displacement_above_threshold_fraction",
            node_displacement_above_threshold_fraction,
        ),
        (
            "effective_source_edge_slot_identity_changed_fraction",
            effective_source_edge_slot_identity_changed_fraction,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not isinstance(sender_state_permutation_qc_passed, bool):
        raise TypeError("sender_state_permutation_qc_passed must be boolean")
    if any(
        float(value) < 0
        for value in (
            minimum_loss_advantage,
            minimum_contribution,
            minimum_deletion_advantage,
        )
    ):
        raise ValueError("synthetic gate thresholds must be nonnegative")
    beats_self = (
        float(self_regional_loss) - float(true_local_loss)
        > float(minimum_loss_advantage)
    )
    beats_permuted = (
        float(permuted_local_loss) - float(true_local_loss)
        > float(minimum_loss_advantage)
    )
    correct_sign = (
        planted_diagnostic.selected_contribution_mean
        > float(minimum_contribution)
    )
    planted_deletion_threshold = max(
        0.0,
        planted_diagnostic.matched_null_deletion_loss_delta
        + float(minimum_deletion_advantage),
    )
    deletion_pass = (
        planted_diagnostic.top_deletion_loss_delta
        > planted_deletion_threshold
    )
    null_deletion_threshold = max(
        0.0,
        null_diagnostic.matched_null_deletion_loss_delta
        + float(minimum_deletion_advantage),
    )
    analogous_null = (
        null_diagnostic.selected_contribution_mean
        > float(minimum_contribution)
        and null_diagnostic.top_deletion_loss_delta
        > null_deletion_threshold
    )
    node_mapping_sufficient = (
        float(node_mapping_changed_fraction)
        >= LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
    )
    displacement_sufficient = (
        float(node_displacement_above_threshold_fraction)
        >= LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
    )
    edge_slot_sufficient = (
        float(effective_source_edge_slot_identity_changed_fraction)
        >= LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
    )
    checks = {
        "sender_state_permutation_qc_failed": (
            sender_state_permutation_qc_passed
        ),
        "sender_state_node_mapping_changed_fraction_below_0.99": (
            node_mapping_sufficient
        ),
        "sender_state_node_displacement_fraction_below_0.90": (
            displacement_sufficient
        ),
        "sender_state_edge_slot_identity_changed_fraction_below_0.99": (
            edge_slot_sufficient
        ),
        "true_local_did_not_beat_self_regional": beats_self,
        "true_local_did_not_beat_permuted_local": beats_permuted,
        "planted_contribution_sign_was_not_positive": correct_sign,
        "top_edge_deletion_did_not_exceed_matched_null": deletion_pass,
        "analogous_discovery_occurred_under_null_injection": (
            not analogous_null
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return SyntheticRecoveryGate(
        node_mapping_changed_fraction=float(
            node_mapping_changed_fraction
        ),
        node_mapping_changed_sufficient=node_mapping_sufficient,
        node_displacement_above_threshold_fraction=float(
            node_displacement_above_threshold_fraction
        ),
        node_displacement_sufficient=displacement_sufficient,
        effective_source_edge_slot_identity_changed_fraction=float(
            effective_source_edge_slot_identity_changed_fraction
        ),
        edge_slot_sender_identity_change_sufficient=edge_slot_sufficient,
        sender_state_permutation_qc_passed=(
            sender_state_permutation_qc_passed
        ),
        true_local_beats_self_regional=beats_self,
        true_local_beats_permuted_local=beats_permuted,
        correct_positive_contribution_sign=correct_sign,
        top_deletion_exceeds_matched_null=deletion_pass,
        analogous_null_discovery=analogous_null,
        no_analogous_null_discovery=not analogous_null,
        gate_passed=not failures,
        failure_reasons=failures,
        thresholds={
            "minimum_node_mapping_changed_fraction": (
                LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
            ),
            "node_displacement_threshold_um": (
                LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
            ),
            "minimum_node_displacement_above_threshold_fraction": (
                LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
            ),
            "minimum_edge_slot_sender_identity_changed_fraction": (
                LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
            ),
            "minimum_loss_advantage": float(minimum_loss_advantage),
            "minimum_contribution": float(minimum_contribution),
            "minimum_deletion_advantage": float(
                minimum_deletion_advantage
            ),
        },
    )


def _fit_one_arm(
    *,
    arm: str,
    fixture: MultiscaleSyntheticFixture,
    config: SyntheticRecoveryConfig,
) -> tuple[SyntheticArmOutcome, MultiscaleAdditiveHybridModel]:
    specifications = {
        ARM_SELF_REGIONAL: (
            "true",
            "surrogate",
            fixture.true_view,
            False,
        ),
        ARM_TRUE_LOCAL: ("true", "true", fixture.true_view, False),
        ARM_PERMUTED_LOCAL: (
            "true",
            "permuted",
            fixture.permuted_view,
            False,
        ),
        ARM_NULL_TRUE_LOCAL: (
            "true",
            "true",
            fixture.null_view,
            True,
        ),
    }
    if arm not in specifications:
        raise ValueError(f"unknown synthetic arm {arm!r}")
    regional_routing, local_routing, view, null_injection = specifications[arm]
    model = _make_model(
        fixture,
        config,
        regional_routing=regional_routing,
        local_routing=local_routing,
        null_injection=null_injection,
    )
    initial_sha = _parameter_value_sha256(model)
    structure_sha = _parameter_structure_sha256(model)
    mean = (
        fixture.null_expression_mean
        if null_injection
        else fixture.expression_mean
    )
    scale = (
        fixture.null_expression_scale
        if null_injection
        else fixture.expression_scale
    )
    training = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        config.training_config(),
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=config.target_node_batch_size,
    )
    evaluation = evaluate_fixed_multiscale_hurdle_mask(
        model,
        view,
        fixture.evaluation_mask,
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=config.target_node_batch_size,
        device=config.device,
        amp=config.amp,
        amp_dtype=config.amp_dtype,
    )
    outcome = SyntheticArmOutcome(
        arm=arm,
        regional_routing=regional_routing,
        local_routing=local_routing,
        parameter_count=trainable_parameter_count(model),
        parameter_structure_sha256=structure_sha,
        initial_parameter_sha256=initial_sha,
        training=training,
        evaluation=evaluation,
    )
    return outcome, model


def run_multiscale_synthetic_recovery(
    geometry: AliasSafeObservedGeometry,
    config: SyntheticRecoveryConfig,
    *,
    fixture: Optional[MultiscaleSyntheticFixture] = None,
    arm_fit: Callable[
        [
            str,
            MultiscaleSyntheticFixture,
            SyntheticRecoveryConfig,
        ],
        tuple[SyntheticArmOutcome, MultiscaleAdditiveHybridModel],
    ]
    | None = None,
    deletion_diagnostic: Callable[..., SyntheticDeletionDiagnostic]
    | None = None,
) -> MultiscaleSyntheticRecoveryResult:
    """Train four paired arms and apply every prespecified recovery check."""

    selected_fixture = (
        build_multiscale_synthetic_fixture(geometry, config)
        if fixture is None
        else fixture
    )
    fit_callable = _fit_one_arm if arm_fit is None else arm_fit
    diagnostic_callable = (
        local_contribution_deletion_diagnostic
        if deletion_diagnostic is None
        else deletion_diagnostic
    )
    outcomes: dict[str, SyntheticArmOutcome] = {}
    reference_structure: Optional[tuple[tuple[str, tuple[int, ...]], ...]] = None
    reference_count: Optional[int] = None
    reference_initial_sha: Optional[str] = None
    planted_diagnostic: Optional[SyntheticDeletionDiagnostic] = None
    null_diagnostic: Optional[SyntheticDeletionDiagnostic] = None

    for arm in SYNTHETIC_ARMS:
        outcome, model = fit_callable(
            arm=arm,
            fixture=selected_fixture,
            config=config,
        )
        if outcome.arm != arm:
            raise MultiscaleSyntheticError("arm fitter returned the wrong arm")
        structure = _parameter_structure(model)
        count = trainable_parameter_count(model)
        initial_sha = outcome.initial_parameter_sha256
        if reference_structure is None:
            reference_structure = structure
            reference_count = count
            reference_initial_sha = initial_sha
        elif (
            structure != reference_structure
            or count != reference_count
            or initial_sha != reference_initial_sha
        ):
            raise MultiscaleSyntheticError(
                "synthetic arms are not exactly parameter matched at initialization"
            )
        outcomes[arm] = outcome
        if arm == ARM_TRUE_LOCAL:
            planted_diagnostic = diagnostic_callable(
                model,
                selected_fixture,
                null_injection=False,
                top_edge_count=config.top_edge_count,
                device=config.device,
            )
        elif arm == ARM_NULL_TRUE_LOCAL:
            null_diagnostic = diagnostic_callable(
                model,
                selected_fixture,
                null_injection=True,
                top_edge_count=config.top_edge_count,
                device=config.device,
            )
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if planted_diagnostic is None or null_diagnostic is None:
        raise RuntimeError("synthetic diagnostics were not executed")
    permutation_qc = selected_fixture.sender_state_permutation_audit["qc"]
    gate = evaluate_synthetic_recovery_gate(
        node_mapping_changed_fraction=float(
            permutation_qc["node_mapping_changed_fraction"]
        ),
        node_displacement_above_threshold_fraction=float(
            permutation_qc[
                "node_displacement_above_threshold_fraction"
            ]
        ),
        effective_source_edge_slot_identity_changed_fraction=float(
            permutation_qc[
                "effective_source_edge_slot_identity_changed_fraction"
            ]
        ),
        sender_state_permutation_qc_passed=bool(
            permutation_qc["gate_passed"]
        ),
        self_regional_loss=outcomes[ARM_SELF_REGIONAL].whole_node_loss,
        true_local_loss=outcomes[ARM_TRUE_LOCAL].whole_node_loss,
        permuted_local_loss=outcomes[ARM_PERMUTED_LOCAL].whole_node_loss,
        planted_diagnostic=planted_diagnostic,
        null_diagnostic=null_diagnostic,
        minimum_loss_advantage=config.minimum_loss_advantage,
        minimum_contribution=config.minimum_contribution,
        minimum_deletion_advantage=config.minimum_deletion_advantage,
    )
    return MultiscaleSyntheticRecoveryResult(
        fixture=selected_fixture,
        config=config,
        arms=outcomes,
        planted_diagnostic=planted_diagnostic,
        null_diagnostic=null_diagnostic,
        gate=gate,
        parameter_match_verified=True,
    )


__all__ = [
    "ARM_NULL_TRUE_LOCAL",
    "ARM_PERMUTED_LOCAL",
    "ARM_SELF_REGIONAL",
    "ARM_TRUE_LOCAL",
    "AliasSafeObservedGeometry",
    "MultiscaleSyntheticError",
    "MultiscaleSyntheticFixture",
    "MultiscaleSyntheticRecoveryResult",
    "SYNTHETIC_ARMS",
    "SYNTHETIC_GEOMETRY_ALIAS",
    "SYNTHETIC_PRIMARY_METRIC",
    "SYNTHETIC_PROTOCOL",
    "SYNTHETIC_SCHEMA",
    "SYNTHETIC_SELECTED_NODE_COUNT",
    "SYNTHETIC_TASK_FAMILY",
    "SyntheticArmOutcome",
    "SyntheticDeletionDiagnostic",
    "SyntheticRecoveryConfig",
    "SyntheticRecoveryGate",
    "assert_alias_safe_configuration",
    "build_multiscale_synthetic_fixture",
    "evaluate_synthetic_recovery_gate",
    "load_alias_safe_observed_geometry",
    "local_contribution_deletion_diagnostic",
    "run_multiscale_synthetic_recovery",
]

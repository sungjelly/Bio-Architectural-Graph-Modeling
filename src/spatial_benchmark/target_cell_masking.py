"""Deterministic one-mask-per-target sampling and rank partitioning.

This module intentionally accepts only structural identifiers and dimensions;
expression values are not part of either seed derivation or sampling.  Every
target cell receives one independently seeded subset of genes, sampled without
replacement with a size drawn from the discrete uniform distribution
``Uniform{1, ..., G}``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


TRAINING_MASK_SCHEMA = "target_cell_once_training_mask_v1"
VALIDATION_MASK_SCHEMA = "target_cell_once_validation_mask_v1"
PARTITION_SCHEMA = "target_cell_contiguous_partition_v1"
PARTITION_COVERAGE_SCHEMA = "target_cell_partition_coverage_v1"
_MAX_SEED = (1 << 63) - 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    for field in (
        str(name).encode("utf-8"),
        array.dtype.str.encode("ascii"),
        np.asarray(array.shape, dtype=np.int64).tobytes(order="C"),
        array.tobytes(order="C"),
    ):
        digest.update(len(field).to_bytes(8, byteorder="big", signed=False))
        digest.update(field)
    return digest.hexdigest()


def _canonical_alias(core_alias: str) -> str:
    alias = str(core_alias).strip().upper()
    if not alias:
        raise ValueError("core_alias must be non-empty")
    return alias


def _canonical_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace must be non-empty")
    return normalized


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _seed_from_payload(payload: Mapping[str, object]) -> int:
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="little") & _MAX_SEED


def derive_training_target_mask_seed(
    base_seed: int,
    *,
    namespace: str,
    core_alias: str,
    global_epoch: int,
) -> int:
    """Derive the core/epoch seed for stochastic training masks."""

    normalized_namespace = _canonical_namespace(namespace)
    return _seed_from_payload(
        {
            "namespace": normalized_namespace,
            "base_seed": _integer(base_seed, name="base_seed"),
            "role": "training",
            "core_alias": _canonical_alias(core_alias),
            "global_epoch": _integer(global_epoch, name="global_epoch"),
        }
    )


def derive_validation_target_mask_seed(
    namespace: str,
    *,
    core_alias: str,
) -> int:
    """Derive a fixed validation seed without an epoch field."""

    normalized_namespace = _canonical_namespace(namespace)
    return _seed_from_payload(
        {
            "namespace": normalized_namespace,
            "role": "validation",
            "core_alias": _canonical_alias(core_alias),
        }
    )


def derive_target_cell_seed(realization_seed: int, target_index: int) -> int:
    """Give each target cell an independent deterministic random stream."""

    return _seed_from_payload(
        {
            "realization_seed": _integer(
                realization_seed, name="realization_seed"
            ),
            "target_index": _integer(target_index, name="target_index"),
        }
    )


def _target_indices(value: object, *, n_cells: int) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.dtype == np.bool_ or not np.issubdtype(
        indices.dtype, np.integer
    ):
        raise TypeError("target_indices must be a one-dimensional integer array")
    indices = np.array(indices, dtype=np.int64, order="C", copy=True)
    if indices.size:
        if np.any(indices < 0) or np.any(indices >= n_cells):
            raise ValueError("target_indices contains an out-of-range cell")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("target_indices must be unique")
    indices.flags.writeable = False
    return indices


@dataclass(frozen=True, slots=True)
class TargetCellMaskRealization:
    """One independently sampled, nonempty gene subset for every target."""

    role: str
    core_alias: str
    n_cells: int
    n_genes: int
    seed: int
    target_indices: np.ndarray
    mask: np.ndarray
    masked_gene_counts: np.ndarray
    global_epoch: int | None = None
    base_seed: int | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        if role not in {"training", "validation"}:
            raise ValueError("role must be 'training' or 'validation'")
        n_cells = _integer(self.n_cells, name="n_cells")
        n_genes = _integer(self.n_genes, name="n_genes", minimum=1)
        indices = _target_indices(self.target_indices, n_cells=n_cells)
        mask = np.asarray(self.mask)
        counts = np.asarray(self.masked_gene_counts)
        if mask.dtype != np.bool_ or mask.shape != (len(indices), n_genes):
            raise TypeError("mask must be boolean [n_targets, n_genes]")
        if counts.shape != (len(indices),) or counts.dtype != np.int64:
            raise TypeError("masked_gene_counts must be int64 [n_targets]")
        if np.any(counts < 1) or np.any(counts > n_genes):
            raise ValueError("every target must mask between one and G genes")
        if not np.array_equal(mask.sum(axis=1, dtype=np.int64), counts):
            raise ValueError("masked_gene_counts does not match the mask")

        alias = _canonical_alias(self.core_alias)
        seed = _integer(self.seed, name="seed")
        if role == "training":
            if (
                self.base_seed is None
                or self.global_epoch is None
                or self.namespace is None
            ):
                raise ValueError(
                    "training masks require namespace, base_seed, and global_epoch"
                )
            expected_seed = derive_training_target_mask_seed(
                self.base_seed,
                namespace=self.namespace,
                core_alias=alias,
                global_epoch=self.global_epoch,
            )
        else:
            if self.namespace is None:
                raise ValueError("validation masks require a namespace")
            if self.base_seed is not None or self.global_epoch is not None:
                raise ValueError("validation masks must be epoch-independent")
            expected_seed = derive_validation_target_mask_seed(
                self.namespace,
                core_alias=alias,
            )
        if seed != expected_seed:
            raise ValueError("mask seed does not match its derivation fields")

        mask = np.ascontiguousarray(mask.copy())
        counts = np.ascontiguousarray(counts.copy())
        mask.flags.writeable = False
        counts.flags.writeable = False
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "core_alias", alias)
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "n_genes", n_genes)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "target_indices", indices)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "masked_gene_counts", counts)
        if self.global_epoch is not None:
            object.__setattr__(
                self,
                "global_epoch",
                _integer(self.global_epoch, name="global_epoch"),
            )
        if self.base_seed is not None:
            object.__setattr__(
                self, "base_seed", _integer(self.base_seed, name="base_seed")
            )
        if self.namespace is not None:
            object.__setattr__(
                self, "namespace", _canonical_namespace(self.namespace)
            )

    @property
    def n_targets(self) -> int:
        return int(len(self.target_indices))

    @property
    def n_masked_entries(self) -> int:
        return int(self.masked_gene_counts.sum(dtype=np.int64))

    @property
    def mask_sha256(self) -> str:
        return _array_sha256("target_cell_mask", self.mask)

    def _receipt_payload(self) -> dict[str, Any]:
        counts = self.masked_gene_counts
        payload: dict[str, Any] = {
            "schema": (
                TRAINING_MASK_SCHEMA
                if self.role == "training"
                else VALIDATION_MASK_SCHEMA
            ),
            "role": self.role,
            "core_alias": self.core_alias,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_targets": self.n_targets,
            "seed": self.seed,
            "sampling": (
                "one_independent_subset_per_target_with_size_uniform_1_through_G_"
                "inclusive_and_positions_without_replacement"
            ),
            "target_seed_derivation": "sha256(realization_seed,target_index)",
            "all_masks_nonempty": bool(
                not counts.size or np.all(counts >= 1)
            ),
            "targets_unique": True,
            "targets_in_bounds": True,
            "covers_all_cells": bool(
                self.n_targets == self.n_cells
                and np.array_equal(
                    np.sort(self.target_indices),
                    np.arange(self.n_cells, dtype=np.int64),
                )
            ),
            "masked_entry_count": self.n_masked_entries,
            "masked_gene_count_min": int(counts.min()) if counts.size else None,
            "masked_gene_count_max": int(counts.max()) if counts.size else None,
            "target_indices_sha256": _array_sha256(
                "target_indices", self.target_indices
            ),
            "masked_gene_counts_sha256": _array_sha256(
                "masked_gene_counts", counts
            ),
            "mask_sha256": self.mask_sha256,
        }
        if self.role == "training":
            payload["namespace"] = self.namespace
            payload["base_seed"] = self.base_seed
            payload["global_epoch"] = self.global_epoch
        else:
            payload["namespace"] = self.namespace
            payload["fixed_across_epochs"] = True
        return payload

    def to_receipt(self) -> dict[str, Any]:
        payload = self._receipt_payload()
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


def _sample_target_cell_masks(
    *,
    role: str,
    core_alias: str,
    n_cells: int,
    n_genes: int,
    seed: int,
    target_indices: object,
    global_epoch: int | None = None,
    base_seed: int | None = None,
    namespace: str | None = None,
) -> TargetCellMaskRealization:
    n_cells = _integer(n_cells, name="n_cells")
    n_genes = _integer(n_genes, name="n_genes", minimum=1)
    indices = _target_indices(target_indices, n_cells=n_cells)
    mask = np.zeros((len(indices), n_genes), dtype=np.bool_)
    counts = np.empty(len(indices), dtype=np.int64)
    for row, target_index in enumerate(indices):
        generator = np.random.default_rng(
            derive_target_cell_seed(seed, int(target_index))
        )
        count = int(generator.integers(1, n_genes + 1))
        selected = generator.choice(n_genes, size=count, replace=False)
        mask[row, selected] = True
        counts[row] = count
    return TargetCellMaskRealization(
        role=role,
        core_alias=core_alias,
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        target_indices=indices,
        mask=mask,
        masked_gene_counts=counts,
        global_epoch=global_epoch,
        base_seed=base_seed,
        namespace=namespace,
    )


def make_training_target_cell_masks(
    n_cells: int,
    n_genes: int,
    *,
    base_seed: int,
    namespace: str,
    core_alias: str,
    global_epoch: int,
    target_indices: object | None = None,
) -> TargetCellMaskRealization:
    """Sample one stochastic nonempty mask for each requested training cell."""

    n_cells = _integer(n_cells, name="n_cells")
    indices = (
        np.arange(n_cells, dtype=np.int64)
        if target_indices is None
        else target_indices
    )
    seed = derive_training_target_mask_seed(
        base_seed,
        namespace=namespace,
        core_alias=core_alias,
        global_epoch=global_epoch,
    )
    return _sample_target_cell_masks(
        role="training",
        core_alias=core_alias,
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        target_indices=indices,
        global_epoch=global_epoch,
        base_seed=base_seed,
        namespace=namespace,
    )


def make_validation_target_cell_masks(
    n_cells: int,
    n_genes: int,
    *,
    namespace: str,
    core_alias: str,
    target_indices: object | None = None,
) -> TargetCellMaskRealization:
    """Sample fixed nonempty validation masks with no epoch dependency."""

    n_cells = _integer(n_cells, name="n_cells")
    indices = (
        np.arange(n_cells, dtype=np.int64)
        if target_indices is None
        else target_indices
    )
    seed = derive_validation_target_mask_seed(namespace, core_alias=core_alias)
    return _sample_target_cell_masks(
        role="validation",
        core_alias=core_alias,
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        target_indices=indices,
        namespace=namespace,
    )


@dataclass(frozen=True, slots=True)
class TargetCellPartition:
    """One rank's contiguous slice of the canonical cell index range."""

    n_cells: int
    world_size: int
    rank: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        n_cells = _integer(self.n_cells, name="n_cells")
        world_size = _integer(self.world_size, name="world_size", minimum=1)
        rank = _integer(self.rank, name="rank")
        start = _integer(self.start, name="start")
        stop = _integer(self.stop, name="stop")
        if rank >= world_size:
            raise ValueError("rank must be less than world_size")
        if start > stop or stop > n_cells:
            raise ValueError("partition bounds are invalid")
        quotient, remainder = divmod(n_cells, world_size)
        expected_start = rank * quotient + min(rank, remainder)
        expected_stop = expected_start + quotient + int(rank < remainder)
        if (start, stop) != (expected_start, expected_stop):
            raise ValueError("partition bounds are not the canonical contiguous split")
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def target_indices(self) -> np.ndarray:
        result = np.arange(self.start, self.stop, dtype=np.int64)
        result.flags.writeable = False
        return result

    @property
    def n_targets(self) -> int:
        return int(self.stop - self.start)

    def _receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": PARTITION_SCHEMA,
            "n_cells": self.n_cells,
            "world_size": self.world_size,
            "rank": self.rank,
            "start": self.start,
            "stop": self.stop,
            "n_targets": self.n_targets,
            "contiguous": True,
            "target_indices_sha256": _array_sha256(
                "target_indices", self.target_indices
            ),
        }

    def to_receipt(self) -> dict[str, Any]:
        payload = self._receipt_payload()
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


def make_contiguous_target_partitions(
    n_cells: int,
    world_size: int,
) -> tuple[TargetCellPartition, ...]:
    """Split ``range(n_cells)`` into balanced contiguous rank partitions."""

    n_cells = _integer(n_cells, name="n_cells")
    world_size = _integer(world_size, name="world_size", minimum=1)
    quotient, remainder = divmod(n_cells, world_size)
    partitions = []
    for rank in range(world_size):
        start = rank * quotient + min(rank, remainder)
        stop = start + quotient + int(rank < remainder)
        partitions.append(
            TargetCellPartition(
                n_cells=n_cells,
                world_size=world_size,
                rank=rank,
                start=start,
                stop=stop,
            )
        )
    return tuple(partitions)


def make_contiguous_target_partition(
    n_cells: int,
    *,
    world_size: int,
    rank: int,
) -> TargetCellPartition:
    """Return one rank from the canonical balanced partition plan."""

    rank = _integer(rank, name="rank")
    partitions = make_contiguous_target_partitions(n_cells, world_size)
    if rank >= len(partitions):
        raise ValueError("rank must be less than world_size")
    return partitions[rank]


def target_partition_coverage_receipt(
    partitions: Sequence[TargetCellPartition],
) -> dict[str, Any]:
    """Validate and checksum exact, disjoint, exhaustive rank coverage."""

    if not isinstance(partitions, Sequence) or isinstance(partitions, (str, bytes)):
        raise TypeError("partitions must be a sequence")
    parts = tuple(partitions)
    if not parts or any(not isinstance(part, TargetCellPartition) for part in parts):
        raise ValueError("partitions must contain TargetCellPartition values")
    n_cells = parts[0].n_cells
    world_size = parts[0].world_size
    if len(parts) != world_size:
        raise ValueError("partition count must equal world_size")
    by_rank = {part.rank: part for part in parts}
    if len(by_rank) != world_size or set(by_rank) != set(range(world_size)):
        raise ValueError("partitions must cover every rank exactly once")
    ordered = tuple(by_rank[rank] for rank in range(world_size))
    if any(
        part.n_cells != n_cells or part.world_size != world_size
        for part in ordered
    ):
        raise ValueError("all partitions must share n_cells and world_size")
    combined = np.concatenate(
        [part.target_indices for part in ordered], dtype=np.int64
    )
    expected = np.arange(n_cells, dtype=np.int64)
    if not np.array_equal(combined, expected):
        raise ValueError("target partitions are not contiguous and exhaustive")
    counts = [part.n_targets for part in ordered]
    if max(counts) - min(counts) > 1:
        raise ValueError("target partition sizes differ by more than one")
    payload: dict[str, Any] = {
        "schema": PARTITION_COVERAGE_SCHEMA,
        "n_cells": n_cells,
        "world_size": world_size,
        "target_counts_by_rank": counts,
        "total_target_count": int(sum(counts)),
        "unique_target_count": int(len(np.unique(combined))),
        "partitions_contiguous": True,
        "partitions_disjoint": True,
        "coverage_exhaustive": True,
        "maximum_partition_size_difference": int(max(counts) - min(counts)),
        "complete_target_indices_sha256": _array_sha256(
            "complete_target_indices", combined
        ),
        "partition_receipts_sha256": _canonical_sha256(
            [part.to_receipt() for part in ordered]
        ),
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def verify_target_cell_receipt(receipt: Mapping[str, Any]) -> None:
    """Fail when a mask or partition receipt's self-checksum has changed."""

    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    payload = dict(receipt)
    stored = payload.pop("receipt_sha256", None)
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError("receipt_sha256 is missing or malformed")
    if stored != _canonical_sha256(payload):
        raise ValueError("receipt checksum does not match its payload")


__all__ = [
    "PARTITION_COVERAGE_SCHEMA",
    "PARTITION_SCHEMA",
    "TRAINING_MASK_SCHEMA",
    "VALIDATION_MASK_SCHEMA",
    "TargetCellMaskRealization",
    "TargetCellPartition",
    "derive_target_cell_seed",
    "derive_training_target_mask_seed",
    "derive_validation_target_mask_seed",
    "make_contiguous_target_partition",
    "make_contiguous_target_partitions",
    "make_training_target_cell_masks",
    "make_validation_target_cell_masks",
    "target_partition_coverage_receipt",
    "verify_target_cell_receipt",
]

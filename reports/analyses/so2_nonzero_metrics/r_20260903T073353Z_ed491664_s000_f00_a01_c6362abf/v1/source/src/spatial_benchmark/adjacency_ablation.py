"""Scientific core for the grouped Adjacent Normal adjacency ablation.

This module deliberately implements only the frozen comparison variable:
one shared explicit-self mean-adjacency network receives either the fixed
spatial graph, the identity adjacency, or a fixed within-FOV position
permutation of the spatial graph.  It contains no cell annotations, edge
features, or condition-specific trainable paths.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor, nn
from torch.nn import functional as F

from .graphs import GraphQC, build_spatial_graph
from .metrics import evaluate_masked_predictions
from .models import ExpressionDecoder, ModelOutput, NodeEncoder, ResidualFeedForward


CORE_ALIASES: tuple[str, ...] = tuple(f"ANC-{index:02d}" for index in range(1, 11))
GRAPH_K = 12
GRAPH_RADIUS_UM = 50.0
GRAPH_SYMMETRY = "union"
POSITION_PERMUTATION_SEED = 2026080202
TRAINING_MASK_BASE_SEED = 2026080201
EVALUATION_MASK_BASE_SEED = 20260802
STANDARDIZATION_SCALE_FLOOR = 1.0e-6

SPATIAL_ARM = "spatial"
ISOLATED_ARM = "isolated"
POSITION_PERMUTED_NULL_ARM = "position_permuted_null"
ADJACENCY_ARMS = (SPATIAL_ARM, ISOLATED_ARM, POSITION_PERMUTED_NULL_ARM)

TARGET_MASK_BIN_ORDER = (
    "0_to_25",
    "25_to_50",
    "50_to_75",
    "75_to_100",
    "exactly_100",
)
NEIGHBOR_OBSERVED_BIN_ORDER = (
    "0_to_25",
    "25_to_50",
    "50_to_75",
    "75_to_100",
)


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


def ndarray_sha256(value: Any) -> str:
    """Return a shape- and dtype-aware SHA-256 for a non-object ndarray."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays do not have a portable ndarray checksum")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"bagm.ndarray.v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _joined_sha256(namespace: str, fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    for field in fields:
        encoded = str(field).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _derived_uint64_seed(base_seed: int, *parts: object) -> int:
    digest = _joined_sha256(
        "bagm.adjacency_ablation.seed.v1",
        [str(int(base_seed)), *(str(part) for part in parts)],
    )
    # NumPy accepts unsigned 64-bit seeds.  Keeping all 64 bits also makes
    # collisions between fold/core/epoch schedules unnecessarily unlikely.
    return int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=False)


@dataclass(frozen=True)
class FoldSplit:
    """One immutable donor/core-grouped outer fold using opaque aliases."""

    fold_index: int
    train_aliases: tuple[str, ...]
    validation_aliases: tuple[str, ...]
    test_aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_aliases": list(self.train_aliases),
            "validation_aliases": list(self.validation_aliases),
            "test_aliases": list(self.test_aliases),
        }


def build_five_fold_splits() -> tuple[FoldSplit, ...]:
    """Build the frozen 7/1/2 slide-balanced alias folds.

    Fold ``j`` tests position ``j`` in both five-core slide blocks.  For even
    zero-based ``j``, validation is the next alias in the first block; for odd
    ``j``, it is the next alias in the second block.
    """

    folds: list[FoldSplit] = []
    for fold_index in range(5):
        test = (
            CORE_ALIASES[fold_index],
            CORE_ALIASES[5 + fold_index],
        )
        next_position = (fold_index + 1) % 5
        validation = (
            CORE_ALIASES[next_position]
            if fold_index % 2 == 0
            else CORE_ALIASES[5 + next_position]
        ,)
        held_out = set(test + validation)
        train = tuple(alias for alias in CORE_ALIASES if alias not in held_out)
        folds.append(
            FoldSplit(
                fold_index=fold_index,
                train_aliases=train,
                validation_aliases=validation,
                test_aliases=test,
            )
        )
    result = tuple(folds)
    assert_valid_five_fold_splits(result)
    return result


def assert_valid_five_fold_splits(
    splits: Sequence[FoldSplit],
    *,
    require_frozen_assignments: bool = True,
) -> None:
    """Raise if aliases overlap, disappear, or depart from the frozen folds."""

    if len(splits) != 5:
        raise ValueError("the frozen design requires exactly five folds")
    test_counts = {alias: 0 for alias in CORE_ALIASES}
    for expected_index, split in enumerate(splits):
        if split.fold_index != expected_index:
            raise ValueError("fold indices must be unique and ordered 0 through 4")
        train = tuple(split.train_aliases)
        validation = tuple(split.validation_aliases)
        test = tuple(split.test_aliases)
        if (len(train), len(validation), len(test)) != (7, 1, 2):
            raise ValueError("each fold must contain 7 train, 1 validation, and 2 test aliases")
        if len(set(train + validation + test)) != len(CORE_ALIASES):
            raise ValueError("an alias is duplicated across fold partitions")
        if set(train + validation + test) != set(CORE_ALIASES):
            raise ValueError("every fold must partition exactly ANC-01 through ANC-10")
        if not ({test[0], test[1]} & set(CORE_ALIASES[:5])) or not (
            {test[0], test[1]} & set(CORE_ALIASES[5:])
        ):
            raise ValueError("each test fold must contain one alias from each slide block")
        for alias in test:
            test_counts[alias] += 1
    if any(count != 1 for count in test_counts.values()):
        raise ValueError("every donor/core alias must be tested exactly once")

    if require_frozen_assignments:
        expected: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for fold_index in range(5):
            test = (CORE_ALIASES[fold_index], CORE_ALIASES[5 + fold_index])
            next_position = (fold_index + 1) % 5
            validation = (
                CORE_ALIASES[next_position]
                if fold_index % 2 == 0
                else CORE_ALIASES[5 + next_position]
            ,)
            expected.append((validation, test))
        observed = [
            (tuple(split.validation_aliases), tuple(split.test_aliases))
            for split in splits
        ]
        if observed != expected:
            raise ValueError("fold assignments differ from the frozen campaign contract")


@dataclass(frozen=True)
class EqualCoreLog1pStandardizer:
    """Training-only equal-core moments for gene-wise log1p counts."""

    train_aliases: tuple[str, ...]
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    scale_floor: float
    checksum: str

    def __post_init__(self) -> None:
        if not self.train_aliases or len(set(self.train_aliases)) != len(self.train_aliases):
            raise ValueError("train_aliases must be non-empty and unique")
        if not math.isfinite(float(self.scale_floor)) or float(self.scale_floor) <= 0:
            raise ValueError("scale_floor must be finite and positive")
        mean = _readonly_array(self.mean, dtype=np.float64)
        variance = _readonly_array(self.variance, dtype=np.float64)
        scale = _readonly_array(self.scale, dtype=np.float64)
        if mean.ndim != 1 or variance.shape != mean.shape or scale.shape != mean.shape:
            raise ValueError("standardizer arrays must be aligned one-dimensional gene vectors")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(scale).all():
            raise ValueError("standardizer arrays must be finite")
        if np.any(variance < 0) or np.any(scale < self.scale_floor):
            raise ValueError("variance must be nonnegative and scale must respect its floor")
        expected_checksum = _joined_sha256(
            "bagm.equal_core_log1p_standardizer.v1",
            [
                *self.train_aliases,
                repr(float(self.scale_floor)),
                ndarray_sha256(mean),
                ndarray_sha256(variance),
                ndarray_sha256(scale),
            ],
        )
        if self.checksum != expected_checksum:
            raise ValueError("standardizer checksum does not match its fitted state")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "scale", scale)

    @property
    def num_genes(self) -> int:
        return int(self.mean.size)

    def to_metadata(self) -> dict[str, object]:
        return {
            "train_aliases": list(self.train_aliases),
            "num_genes": self.num_genes,
            "scale_floor": self.scale_floor,
            "checksum": self.checksum,
            "weighting": "equal_core_mixture",
            "input_transform": "log1p_raw_count",
        }


def _validate_count_matrix(value: Any, *, name: str, num_genes: int | None = None) -> np.ndarray:
    counts = np.asarray(value)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [cells, genes] matrix")
    if num_genes is not None and counts.shape[1] != num_genes:
        raise ValueError(f"{name} has {counts.shape[1]} genes, expected {num_genes}")
    if not np.issubdtype(counts.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError(f"{name} must contain finite nonnegative raw counts")
    return counts


def fit_equal_core_log1p_standardizer(
    counts_by_alias: Mapping[str, Any],
    *,
    train_aliases: Sequence[str],
    scale_floor: float = STANDARDIZATION_SCALE_FLOOR,
) -> EqualCoreLog1pStandardizer:
    """Fit equal-core log1p moments using only explicit training aliases.

    The mixture mean is the arithmetic mean of per-core means.  Its second
    moment is likewise averaged across cores before variance is derived, so
    larger cores do not receive greater weight.  Values for validation or test
    aliases may be present in ``counts_by_alias`` but are never accessed.
    """

    aliases = tuple(str(alias) for alias in train_aliases)
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("train_aliases must be a non-empty sequence of unique aliases")
    if not math.isfinite(float(scale_floor)) or float(scale_floor) <= 0:
        raise ValueError("scale_floor must be finite and positive")
    missing = [alias for alias in aliases if alias not in counts_by_alias]
    if missing:
        raise KeyError(f"missing training counts for aliases: {missing}")

    core_means: list[np.ndarray] = []
    core_second_moments: list[np.ndarray] = []
    num_genes: int | None = None
    for alias in aliases:
        counts = _validate_count_matrix(
            counts_by_alias[alias], name=f"counts_by_alias[{alias!r}]", num_genes=num_genes
        )
        num_genes = counts.shape[1]
        transformed = np.log1p(counts.astype(np.float64, copy=False))
        core_means.append(transformed.mean(axis=0, dtype=np.float64))
        core_second_moments.append(np.square(transformed).mean(axis=0, dtype=np.float64))

    mean = np.stack(core_means, axis=0).mean(axis=0)
    second_moment = np.stack(core_second_moments, axis=0).mean(axis=0)
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), float(scale_floor))
    checksum = _joined_sha256(
        "bagm.equal_core_log1p_standardizer.v1",
        [
            *aliases,
            repr(float(scale_floor)),
            ndarray_sha256(mean),
            ndarray_sha256(variance),
            ndarray_sha256(scale),
        ],
    )
    return EqualCoreLog1pStandardizer(
        train_aliases=aliases,
        mean=mean,
        variance=variance,
        scale=scale,
        scale_floor=float(scale_floor),
        checksum=checksum,
    )


def transform_log1p_counts(
    expression_counts: Any,
    standardizer: EqualCoreLog1pStandardizer,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Apply a training-fitted gene standardizer without library-size inputs."""

    counts = _validate_count_matrix(
        expression_counts, name="expression_counts", num_genes=standardizer.num_genes
    )
    transformed = (
        np.log1p(counts.astype(np.float64, copy=False)) - standardizer.mean
    ) / standardizer.scale
    return np.asarray(transformed, dtype=dtype)


def inverse_standardized_log1p(
    standardized_values: Any,
    standardizer: EqualCoreLog1pStandardizer,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Return unclipped log1p-count values for scientific error evaluation."""

    values = np.asarray(standardized_values)
    if values.ndim != 2 or values.shape[1] != standardizer.num_genes:
        raise ValueError("standardized_values must have shape [cells, num_genes]")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("standardized_values must be finite and numeric")
    result = values.astype(np.float64, copy=False) * standardizer.scale + standardizer.mean
    return np.asarray(result, dtype=dtype)


@dataclass(frozen=True)
class MaskRealization:
    """One exact per-cell mask and its auditable count/checksum metadata."""

    mask: np.ndarray
    masked_gene_counts: np.ndarray
    seed: int
    checksum: str

    def __post_init__(self) -> None:
        mask = _readonly_array(self.mask, dtype=np.bool_)
        counts = _readonly_array(self.masked_gene_counts, dtype=np.int64)
        if mask.ndim != 2 or counts.shape != (mask.shape[0],):
            raise ValueError("mask must be [cells, genes] and counts must be [cells]")
        if not np.array_equal(mask.sum(axis=1, dtype=np.int64), counts):
            raise ValueError("masked_gene_counts do not equal exact row mask sums")
        if np.any(counts < 0) or np.any(counts > mask.shape[1]):
            raise ValueError("masked gene counts fall outside 0 through G")
        expected = mask_realization_sha256(mask, counts, seed=int(self.seed))
        if self.checksum != expected:
            raise ValueError("mask checksum does not match its arrays and seed")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "masked_gene_counts", counts)

    @property
    def n_cells(self) -> int:
        return int(self.mask.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.mask.shape[1])

    def to_torch(self, *, device: object | None = None) -> tuple[Tensor, Tensor]:
        # from_numpy warns for read-only arrays; copy once into writable host
        # storage before any optional device transfer.
        mask = torch.from_numpy(np.array(self.mask, copy=True)).to(device=device)
        counts = torch.from_numpy(np.array(self.masked_gene_counts, copy=True)).to(device=device)
        return mask, counts


def mask_realization_sha256(mask: Any, masked_gene_counts: Any, *, seed: int) -> str:
    return _joined_sha256(
        "bagm.exact_uniform_mask.v1",
        [
            str(int(seed)),
            ndarray_sha256(np.asarray(mask, dtype=np.bool_)),
            ndarray_sha256(np.asarray(masked_gene_counts, dtype=np.int64)),
        ],
    )


def sample_uniform_mask_numpy(
    n_cells: int,
    num_genes: int,
    *,
    seed: int,
    chunk_cells: int = 2048,
) -> MaskRealization:
    """Sample ``m_i ~ Uniform{0,...,G}`` then exactly ``m_i`` positions.

    Row-wise permutations are generated in bounded chunks.  The returned
    count vector is sampled independently before positions and is verified
    against every row, including the zero- and full-mask endpoints.
    """

    if not isinstance(n_cells, (int, np.integer)) or int(n_cells) <= 0:
        raise ValueError("n_cells must be a positive integer")
    if not isinstance(num_genes, (int, np.integer)) or int(num_genes) <= 0:
        raise ValueError("num_genes must be a positive integer")
    if not isinstance(chunk_cells, (int, np.integer)) or int(chunk_cells) <= 0:
        raise ValueError("chunk_cells must be a positive integer")
    n_cells = int(n_cells)
    num_genes = int(num_genes)
    rng = np.random.default_rng(int(seed))
    counts = rng.integers(0, num_genes + 1, size=n_cells, dtype=np.int64)
    mask = np.empty((n_cells, num_genes), dtype=np.bool_)
    gene_positions = np.arange(num_genes, dtype=np.int64)
    rank_positions = np.arange(num_genes, dtype=np.int64)[None, :]
    for start in range(0, n_cells, int(chunk_cells)):
        stop = min(start + int(chunk_cells), n_cells)
        positions = np.broadcast_to(gene_positions, (stop - start, num_genes)).copy()
        rng.permuted(positions, axis=1, out=positions)
        selected_by_rank = rank_positions < counts[start:stop, None]
        chunk_mask = np.empty((stop - start, num_genes), dtype=np.bool_)
        np.put_along_axis(chunk_mask, positions, selected_by_rank, axis=1)
        mask[start:stop] = chunk_mask
    checksum = mask_realization_sha256(mask, counts, seed=int(seed))
    return MaskRealization(mask=mask, masked_gene_counts=counts, seed=int(seed), checksum=checksum)


def sample_uniform_mask_torch(
    n_cells: int,
    num_genes: int,
    *,
    seed: int,
    device: object | None = None,
    chunk_cells: int = 2048,
) -> tuple[Tensor, Tensor, str]:
    """Torch-ready exact mask using the same audited NumPy realization."""

    realization = sample_uniform_mask_numpy(
        n_cells,
        num_genes,
        seed=seed,
        chunk_cells=chunk_cells,
    )
    mask, counts = realization.to_torch(device=device)
    return mask, counts, realization.checksum


def derive_training_mask_seed(
    *,
    fold_index: int,
    model_seed: int,
    epoch: int,
    core_alias: str,
    base_seed: int = TRAINING_MASK_BASE_SEED,
) -> int:
    """Derive an arm-independent dynamic training-mask seed."""

    if fold_index not in range(5):
        raise ValueError("fold_index must be in 0 through 4")
    if model_seed < 0 or epoch < 0:
        raise ValueError("model_seed and epoch must be nonnegative")
    if core_alias not in CORE_ALIASES:
        raise ValueError("core_alias must be one of the ten opaque campaign aliases")
    return _derived_uint64_seed(base_seed, "train", fold_index, model_seed, epoch, core_alias)


def derive_evaluation_mask_seed(
    *,
    core_alias: str,
    replicate_index: int,
    base_seed: int = EVALUATION_MASK_BASE_SEED,
) -> int:
    """Derive a model-seed- and arm-independent fixed evaluation-mask seed."""

    if core_alias not in CORE_ALIASES:
        raise ValueError("core_alias must be one of the ten opaque campaign aliases")
    if replicate_index not in range(3):
        raise ValueError("replicate_index must be 0, 1, or 2")
    return _derived_uint64_seed(base_seed, "evaluation", core_alias, replicate_index)


def _canonical_fov_labels(fov_labels: Sequence[object] | np.ndarray, n_nodes: int) -> np.ndarray:
    values = np.asarray(fov_labels)
    if values.ndim != 1 or values.shape[0] != n_nodes:
        raise ValueError("fov_labels must align one-to-one with coordinates")
    labels: list[str] = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError("fov_labels cannot be missing")
        labels.append(str(value))
    return np.asarray(labels, dtype="U128")


def _canonical_edge_index(edge_index: Any, *, n_nodes: int) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("edge_index contains an out-of-range node")
    if edges.shape[1]:
        encoded = edges[0] * int(n_nodes) + edges[1]
        if np.unique(encoded).size != encoded.size:
            raise ValueError("edge_index contains duplicate directed edges")
        order = np.lexsort((edges[1], edges[0]))
        edges = edges[:, order]
    return np.ascontiguousarray(edges, dtype=np.int64)


def add_exact_self_adjacency(off_diagonal_edge_index: Any, *, n_nodes: int) -> np.ndarray:
    """Add exactly one explicit self edge per node to loop-free edges."""

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    off_diagonal = _canonical_edge_index(off_diagonal_edge_index, n_nodes=n_nodes)
    if np.any(off_diagonal[0] == off_diagonal[1]):
        raise ValueError("off_diagonal_edge_index must not contain self loops")
    nodes = np.arange(n_nodes, dtype=np.int64)
    self_edges = np.stack([nodes, nodes], axis=0)
    return _canonical_edge_index(
        np.concatenate([off_diagonal, self_edges], axis=1), n_nodes=n_nodes
    )


@dataclass(frozen=True)
class AdjacencyQC:
    arm: str
    n_nodes: int
    n_directed_edges: int
    n_off_diagonal_edges: int
    n_self_loops: int
    duplicate_directed_edges: int
    cross_fov_edges: int
    minimum_in_degree: int
    mean_in_degree: float
    maximum_in_degree: int
    exact_one_self_loop_per_node: bool
    directed_off_diagonal_pairs_are_symmetric: bool
    in_degree_multiset_checksum: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _adjacency_qc(
    arm: str,
    edge_index: np.ndarray,
    fov_labels: np.ndarray,
    *,
    n_nodes: int,
) -> AdjacencyQC:
    edges = np.asarray(edge_index, dtype=np.int64)
    encoded = edges[0] * n_nodes + edges[1]
    duplicate_count = int(encoded.size - np.unique(encoded).size)
    loop_mask = edges[0] == edges[1]
    loops = edges[:, loop_mask]
    self_counts = np.bincount(loops[0], minlength=n_nodes) if loops.shape[1] else np.zeros(n_nodes, dtype=np.int64)
    cross_fov = int(np.sum(fov_labels[edges[0]] != fov_labels[edges[1]]))
    in_degree = np.bincount(edges[1], minlength=n_nodes).astype(np.int64)
    off_diagonal = edges[:, ~loop_mask]
    off_set = set(zip(off_diagonal[0].tolist(), off_diagonal[1].tolist()))
    symmetric = all((receiver, source) in off_set for source, receiver in off_set)
    return AdjacencyQC(
        arm=arm,
        n_nodes=n_nodes,
        n_directed_edges=int(edges.shape[1]),
        n_off_diagonal_edges=int(off_diagonal.shape[1]),
        n_self_loops=int(loop_mask.sum()),
        duplicate_directed_edges=duplicate_count,
        cross_fov_edges=cross_fov,
        minimum_in_degree=int(in_degree.min()),
        mean_in_degree=float(in_degree.mean()),
        maximum_in_degree=int(in_degree.max(initial=0)),
        exact_one_self_loop_per_node=bool(np.all(self_counts == 1)),
        directed_off_diagonal_pairs_are_symmetric=bool(symmetric),
        in_degree_multiset_checksum=ndarray_sha256(np.sort(in_degree)),
    )


@dataclass(frozen=True)
class AdjacencyArm:
    """One fixed model adjacency with exact-self and FOV-safety QC."""

    name: str
    edge_index: np.ndarray
    checksum: str
    qc: AdjacencyQC

    def __post_init__(self) -> None:
        if self.name not in ADJACENCY_ARMS:
            raise ValueError(f"unknown adjacency arm: {self.name}")
        edges = _readonly_array(self.edge_index, dtype=np.int64)
        if self.checksum != ndarray_sha256(edges):
            raise ValueError("adjacency checksum does not match edge_index")
        if self.qc.arm != self.name or self.qc.n_directed_edges != edges.shape[1]:
            raise ValueError("adjacency QC does not align to its arm")
        if (
            self.qc.duplicate_directed_edges
            or self.qc.cross_fov_edges
            or not self.qc.exact_one_self_loop_per_node
        ):
            raise ValueError("adjacency fails duplicate, FOV, or self-loop safety checks")
        object.__setattr__(self, "edge_index", edges)

    def to_torch(self, *, device: object | None = None) -> Tensor:
        return torch.from_numpy(np.array(self.edge_index, copy=True)).to(device=device)

    def to_metadata(self) -> dict[str, object]:
        return {"name": self.name, "checksum": self.checksum, "qc": self.qc.to_dict()}


@dataclass(frozen=True)
class AdjacencyBundle:
    """The three paired adjacencies plus the true off-diagonal graph."""

    spatial: AdjacencyArm
    isolated: AdjacencyArm
    position_permuted_null: AdjacencyArm
    true_off_diagonal_edge_index: np.ndarray
    position_assignment: np.ndarray
    position_assignment_checksum: str
    source_graph_qc: GraphQC
    checksum: str

    def __post_init__(self) -> None:
        off_diagonal = _readonly_array(self.true_off_diagonal_edge_index, dtype=np.int64)
        position_assignment = _readonly_array(self.position_assignment, dtype=np.int64)
        if np.any(off_diagonal[0] == off_diagonal[1]):
            raise ValueError("true_off_diagonal_edge_index contains a self loop")
        if self.position_assignment_checksum != ndarray_sha256(position_assignment):
            raise ValueError("position assignment checksum mismatch")
        if self.spatial.qc.n_nodes != position_assignment.size:
            raise ValueError("position assignment does not align to graph nodes")
        expected_bundle_checksum = _joined_sha256(
            "bagm.adjacency_bundle.v1",
            [
                self.spatial.checksum,
                self.isolated.checksum,
                self.position_permuted_null.checksum,
                ndarray_sha256(off_diagonal),
                self.position_assignment_checksum,
            ],
        )
        if self.checksum != expected_bundle_checksum:
            raise ValueError("adjacency bundle checksum mismatch")
        object.__setattr__(self, "true_off_diagonal_edge_index", off_diagonal)
        object.__setattr__(self, "position_assignment", position_assignment)

    def arm(self, name: str) -> AdjacencyArm:
        if name not in ADJACENCY_ARMS:
            raise KeyError(f"unknown adjacency arm: {name}")
        return getattr(self, name)

    def to_metadata(self) -> dict[str, object]:
        return {
            "checksum": self.checksum,
            "position_assignment_checksum": self.position_assignment_checksum,
            "source_graph_qc": self.source_graph_qc.to_dict(),
            "arms": {name: self.arm(name).to_metadata() for name in ADJACENCY_ARMS},
            "graph_definition": {
                "k": GRAPH_K,
                "radius_um": GRAPH_RADIUS_UM,
                "symmetry": GRAPH_SYMMETRY,
                "grouping": "raw_fov_within_core",
                "self_loops": "exactly_one_per_node_in_all_arms",
            },
        }


def _make_adjacency_arm(
    name: str,
    edge_index: np.ndarray,
    fov_labels: np.ndarray,
    *,
    n_nodes: int,
) -> AdjacencyArm:
    edges = _canonical_edge_index(edge_index, n_nodes=n_nodes)
    qc = _adjacency_qc(name, edges, fov_labels, n_nodes=n_nodes)
    return AdjacencyArm(name=name, edge_index=edges, checksum=ndarray_sha256(edges), qc=qc)


def materialize_fixed_adjacencies(
    coordinates_um: Any,
    fov_labels: Sequence[object] | np.ndarray,
    *,
    null_seed: int = POSITION_PERMUTATION_SEED,
    scope: str = "",
) -> AdjacencyBundle:
    """Materialize frozen k12/r50/union, identity, and position-null arms."""

    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates_um must be non-empty with shape [cells, 2]")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates_um must be finite")
    n_nodes = int(coordinates.shape[0])
    fov = _canonical_fov_labels(fov_labels, n_nodes)
    graph = build_spatial_graph(
        coordinates,
        k=GRAPH_K,
        radius_um=GRAPH_RADIUS_UM,
        symmetry=GRAPH_SYMMETRY,
        group_labels=fov,
        fov=fov,
    )
    if graph.qc.cross_group_edges or graph.qc.fov_seam_edges:
        raise RuntimeError("source graph contains a forbidden cross-FOV edge")
    true_off_diagonal = _canonical_edge_index(graph.edge_index, n_nodes=n_nodes)
    spatial_edges = add_exact_self_adjacency(true_off_diagonal, n_nodes=n_nodes)
    identity_edges = add_exact_self_adjacency(
        np.empty((2, 0), dtype=np.int64), n_nodes=n_nodes
    )

    rng = np.random.default_rng(_derived_uint64_seed(null_seed, "position_null", scope))
    position_assignment = np.arange(n_nodes, dtype=np.int64)
    for label in np.unique(fov):
        positions = np.flatnonzero(fov == label)
        position_assignment[positions] = rng.permutation(positions)
    null_off_diagonal = position_assignment[true_off_diagonal]
    null_edges = add_exact_self_adjacency(null_off_diagonal, n_nodes=n_nodes)

    spatial = _make_adjacency_arm(SPATIAL_ARM, spatial_edges, fov, n_nodes=n_nodes)
    isolated = _make_adjacency_arm(ISOLATED_ARM, identity_edges, fov, n_nodes=n_nodes)
    position_null = _make_adjacency_arm(
        POSITION_PERMUTED_NULL_ARM, null_edges, fov, n_nodes=n_nodes
    )
    if spatial.qc.n_off_diagonal_edges != position_null.qc.n_off_diagonal_edges:
        raise RuntimeError("position null did not preserve graph size")
    if (
        spatial.qc.in_degree_multiset_checksum
        != position_null.qc.in_degree_multiset_checksum
    ):
        raise RuntimeError("position null did not preserve the exact degree distribution")
    if isolated.qc.n_directed_edges != n_nodes or isolated.qc.minimum_in_degree != 1 or isolated.qc.maximum_in_degree != 1:
        raise RuntimeError("isolated adjacency is not literal identity")

    position_checksum = ndarray_sha256(position_assignment)
    bundle_checksum = _joined_sha256(
        "bagm.adjacency_bundle.v1",
        [
            spatial.checksum,
            isolated.checksum,
            position_null.checksum,
            ndarray_sha256(true_off_diagonal),
            position_checksum,
        ],
    )
    return AdjacencyBundle(
        spatial=spatial,
        isolated=isolated,
        position_permuted_null=position_null,
        true_off_diagonal_edge_index=true_off_diagonal,
        position_assignment=position_assignment,
        position_assignment_checksum=position_checksum,
        source_graph_qc=graph.qc,
        checksum=bundle_checksum,
    )


def validate_explicit_self_adjacency(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """Validate and canonicalize an adjacency with one self edge per node."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edge_index.is_floating_point() or edge_index.is_complex() or edge_index.dtype == torch.bool:
        raise TypeError("edge_index must use an integer dtype")
    edges = edge_index.to(dtype=torch.long)
    if edges.numel() == 0:
        raise ValueError("explicit-self adjacency cannot be empty")
    if bool((edges < 0).any()) or bool((edges >= num_nodes).any()):
        raise ValueError("edge_index contains an out-of-range node")
    encoded = edges[0] * int(num_nodes) + edges[1]
    if torch.unique(encoded).numel() != encoded.numel():
        raise ValueError("edge_index contains duplicate directed edges")
    loops = edges[0, edges[0] == edges[1]]
    self_counts = torch.bincount(loops, minlength=num_nodes)
    if self_counts.numel() != num_nodes or not bool((self_counts == 1).all()):
        raise ValueError("every node must have exactly one explicit self edge")
    return edges


def explicit_self_incoming_mean(node_embedding: Tensor, edge_index: Tensor) -> Tensor:
    """Mean all incoming embeddings, including the required explicit self."""

    if node_embedding.ndim != 2:
        raise ValueError("node_embedding must have shape [nodes, hidden_dim]")
    edges = validate_explicit_self_adjacency(edge_index, num_nodes=node_embedding.shape[0]).to(
        device=node_embedding.device
    )
    return _incoming_mean_from_validated_edges(node_embedding, edges)


def _incoming_mean_from_validated_edges(
    node_embedding: Tensor,
    edges: Tensor,
) -> Tensor:
    source, receiver = edges
    aggregate = torch.zeros_like(node_embedding)
    aggregate.index_add_(0, receiver, node_embedding.index_select(0, source))
    degree = node_embedding.new_zeros((node_embedding.shape[0],))
    degree.index_add_(0, receiver, node_embedding.new_ones((edges.shape[1],)))
    return aggregate / degree.unsqueeze(-1)


def _normalise_target_nodes(
    target_nodes: Tensor | Sequence[int] | None,
    *,
    num_nodes: int,
    device: torch.device,
) -> Tensor | None:
    if target_nodes is None:
        return None
    targets = torch.as_tensor(target_nodes, device=device)
    if targets.dtype == torch.bool:
        if targets.shape != (num_nodes,):
            raise ValueError("boolean target_nodes must have shape [num_nodes]")
        return targets.nonzero(as_tuple=False).flatten()
    if targets.ndim != 1 or targets.is_floating_point() or targets.is_complex():
        raise TypeError("target_nodes must be a one-dimensional integer index")
    targets = targets.to(dtype=torch.long)
    if targets.numel() and (bool((targets < 0).any()) or bool((targets >= num_nodes).any())):
        raise ValueError("target_nodes contains an out-of-range node")
    return targets


class ExplicitSelfMeanGraphSAGE(nn.Module):
    """One-hop mean-adjacency network shared literally across all arms.

    The aggregate path remains active for the identity adjacency, where the
    incoming mean equals the same cell's embedding.  Thus arm identity never
    selects a parameter path; only ``edge_index`` changes.
    """

    def __init__(
        self,
        num_genes: int,
        *,
        hidden_dim: int = 128,
        ffn_dim: int = 256,
        decoder_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_genes <= 0 or hidden_dim <= 0 or ffn_dim <= 0 or decoder_dim <= 0:
            raise ValueError("all model dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.num_genes = int(num_genes)
        self.hidden_dim = int(hidden_dim)
        self.encoder = NodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=0,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.aggregate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.adjacency_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.adjacency_normalization = nn.LayerNorm(hidden_dim)
        self.adjacency_dropout = nn.Dropout(dropout)
        self.feed_forward = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )
        self.decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )
        # Validation of a 100k-edge immutable tensor is useful once, but
        # repeating ``torch.unique`` at every optimizer update is not.  Tensor
        # version and storage identity invalidate this non-persistent cache.
        self._validated_adjacencies: dict[
            tuple[object, ...], weakref.ReferenceType[Tensor]
        ] = {}

    def _prepare_adjacency(self, edge_index: Tensor, *, num_nodes: int, device: torch.device) -> Tensor:
        signature = (
            str(edge_index.device),
            str(edge_index.dtype),
            int(edge_index.data_ptr()),
            int(edge_index._version),
            tuple(edge_index.shape),
            int(num_nodes),
        )
        cached = self._validated_adjacencies.get(signature)
        if cached is None or cached() is not edge_index:
            validate_explicit_self_adjacency(edge_index, num_nodes=num_nodes)
            self._validated_adjacencies[signature] = weakref.ref(edge_index)
            if len(self._validated_adjacencies) > 64:
                self._validated_adjacencies = {
                    key: reference
                    for key, reference in self._validated_adjacencies.items()
                    if reference() is not None
                }
        return edge_index.to(device=device, dtype=torch.long)

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Tensor,
        *,
        target_nodes: Tensor | Sequence[int] | None = None,
    ) -> ModelOutput:
        embedding = self.encoder(input_expression, gene_mask)
        prepared_adjacency = self._prepare_adjacency(
            edge_index,
            num_nodes=embedding.shape[0],
            device=embedding.device,
        )
        adjacency_mean = _incoming_mean_from_validated_edges(
            embedding, prepared_adjacency
        )
        embedding = self.adjacency_dropout(
            F.gelu(
                self.adjacency_normalization(
                    self.self_projection(embedding)
                    + self.aggregate_projection(adjacency_mean)
                    + self.adjacency_bias
                )
            )
        )
        embedding = self.feed_forward(embedding)
        targets = _normalise_target_nodes(
            target_nodes,
            num_nodes=embedding.shape[0],
            device=embedding.device,
        )
        if targets is not None:
            embedding = embedding.index_select(0, targets)
        prediction = self.decoder(embedding)
        return ModelOutput(prediction=prediction, node_embedding=embedding)


def build_seeded_explicit_self_model(
    num_genes: int,
    *,
    seed: int,
    hidden_dim: int = 128,
    ffn_dim: int = 256,
    decoder_dim: int = 256,
    dropout: float = 0.1,
) -> ExplicitSelfMeanGraphSAGE:
    """Construct paired initial states without changing the caller RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ExplicitSelfMeanGraphSAGE(
            num_genes,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def state_dict_sha256(model_or_state: nn.Module | Mapping[str, Tensor]) -> str:
    """Return an ordered tensor-name/shape/dtype/value checksum."""

    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    digest.update(b"bagm.torch_state_dict.v1\0")
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        for field in (name, array.dtype.str, repr(tuple(array.shape))):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _compact_masked_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, object]:
    evaluated = evaluate_masked_predictions(
        y_true,
        y_pred,
        mask,
        huber_delta=huber_delta,
    )
    mse = float(evaluated["mse"])
    return {
        "n_cells": int(y_true.shape[0]),
        "n_masked": int(evaluated["n_masked"]),
        "huber": float(evaluated["huber"]),
        "mae": float(evaluated["mae"]),
        "mse": mse,
        "rmse": float(math.sqrt(mse)) if math.isfinite(mse) else float("nan"),
        "gene_pearson_mean": float(evaluated["gene"]["mean_pearson"]),
        "gene_spearman_mean": float(evaluated["gene"]["mean_spearman"]),
        "cell_pearson_mean": float(evaluated["cell"]["mean_pearson"]),
        "cell_spearman_mean": float(evaluated["cell"]["mean_spearman"]),
        "n_valid_gene_pearson": int(evaluated["gene"]["n_valid_pearson"]),
        "n_valid_gene_spearman": int(evaluated["gene"]["n_valid_spearman"]),
        "n_valid_cell_pearson": int(evaluated["cell"]["n_valid_pearson"]),
        "n_valid_cell_spearman": int(evaluated["cell"]["n_valid_spearman"]),
    }


def masked_regression_summary(
    y_true: Any,
    y_pred: Any,
    mask: Any,
    *,
    huber_delta: float = 1.0,
) -> dict[str, object]:
    """Compact scientific regression metrics over finite masked entries."""

    target = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    selected = np.asarray(mask, dtype=np.bool_)
    if target.ndim != 2 or prediction.shape != target.shape or selected.shape != target.shape:
        raise ValueError("y_true, y_pred, and mask must share [cells, genes] shape")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0:
        raise ValueError("huber_delta must be finite and positive")
    return _compact_masked_summary(target, prediction, selected, huber_delta=float(huber_delta))


@dataclass(frozen=True)
class NeighborAvailability:
    """Reusable true-neighbor observation fractions for one fixed mask."""

    fraction: np.ndarray
    off_diagonal_degree: np.ndarray
    target_mask_checksum: str
    edge_index_checksum: str
    checksum: str

    def __post_init__(self) -> None:
        fraction = _readonly_array(self.fraction, dtype=np.float32)
        degree = _readonly_array(self.off_diagonal_degree, dtype=np.int64)
        if fraction.ndim != 2 or degree.shape != (fraction.shape[0],):
            raise ValueError("neighbor fraction must be [cells, genes] and degree [cells]")
        if not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1):
            raise ValueError("neighbor observation fractions must be finite and in [0, 1]")
        if np.any(degree < 0):
            raise ValueError("off-diagonal neighbor degrees cannot be negative")
        if np.any(fraction[degree == 0] != 0):
            raise ValueError("degree-zero cells must have neighbor availability zero")
        expected = _joined_sha256(
            "bagm.true_neighbor_availability.v1",
            [
                ndarray_sha256(fraction),
                ndarray_sha256(degree),
                self.target_mask_checksum,
                self.edge_index_checksum,
            ],
        )
        if self.checksum != expected:
            raise ValueError("neighbor availability checksum does not match its state")
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "off_diagonal_degree", degree)


def compute_true_neighbor_availability(
    target_mask: Any,
    true_spatial_edge_index: Any,
    *,
    gene_chunk_size: int = 256,
) -> NeighborAvailability:
    """Compute reusable true-neighbor observation fractions in bounded memory.

    A sparse receiver-by-source adjacency multiplies chunks of the observed
    indicator matrix.  This avoids the prohibitive ``edges x genes`` temporary
    produced by direct advanced indexing.  A cell with no off-diagonal true
    neighbors receives availability zero for every gene, remains in the
    lowest availability stratum, and is separately countable from ``degree``.
    """

    selected = np.asarray(target_mask, dtype=np.bool_)
    if selected.ndim != 2 or selected.shape[0] == 0 or selected.shape[1] == 0:
        raise ValueError("target_mask must be non-empty with shape [cells, genes]")
    if not isinstance(gene_chunk_size, (int, np.integer)) or int(gene_chunk_size) <= 0:
        raise ValueError("gene_chunk_size must be a positive integer")
    edges = _canonical_edge_index(true_spatial_edge_index, n_nodes=selected.shape[0])
    off_diagonal = edges[:, edges[0] != edges[1]]
    source, receiver = off_diagonal
    adjacency = csr_matrix(
        (
            np.ones(off_diagonal.shape[1], dtype=np.float32),
            (receiver, source),
        ),
        shape=(selected.shape[0], selected.shape[0]),
        dtype=np.float32,
    )
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.int64)
    fraction = np.zeros(selected.shape, dtype=np.float32)
    divisor = degree[:, None].astype(np.float32)
    for start in range(0, selected.shape[1], int(gene_chunk_size)):
        stop = min(start + int(gene_chunk_size), selected.shape[1])
        observed = np.logical_not(selected[:, start:stop]).astype(np.float32, copy=False)
        observed_count = np.asarray(adjacency @ observed, dtype=np.float32)
        np.divide(
            observed_count,
            divisor,
            out=fraction[:, start:stop],
            where=divisor > 0,
        )
    target_mask_checksum = ndarray_sha256(selected)
    edge_index_checksum = ndarray_sha256(off_diagonal)
    checksum = _joined_sha256(
        "bagm.true_neighbor_availability.v1",
        [
            ndarray_sha256(fraction),
            ndarray_sha256(degree),
            target_mask_checksum,
            edge_index_checksum,
        ],
    )
    return NeighborAvailability(
        fraction=fraction,
        off_diagonal_degree=degree,
        target_mask_checksum=target_mask_checksum,
        edge_index_checksum=edge_index_checksum,
        checksum=checksum,
    )


def true_neighbor_observed_fraction(
    target_mask: Any,
    true_spatial_edge_index: Any,
    *,
    gene_chunk_size: int = 256,
) -> np.ndarray:
    """Return the bounded-memory true-neighbor availability array."""

    return compute_true_neighbor_availability(
        target_mask,
        true_spatial_edge_index,
        gene_chunk_size=gene_chunk_size,
    ).fraction


def _loss_only_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, object]:
    valid = selected & np.isfinite(target) & np.isfinite(prediction)
    n_masked = int(valid.sum())
    n_cells = int(np.any(selected, axis=1).sum())
    if n_masked == 0:
        return {
            "n_cells": n_cells,
            "n_masked": 0,
            "huber": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
        }
    difference = prediction[valid] - target[valid]
    absolute = np.abs(difference)
    mse = float(np.mean(np.square(difference)))
    huber = np.where(
        absolute <= huber_delta,
        0.5 * np.square(difference),
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return {
        "n_cells": n_cells,
        "n_masked": n_masked,
        "huber": float(np.mean(huber)),
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def _summary_for_selected_entries(
    target: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
    *,
    huber_delta: float,
    include_correlations: bool,
) -> dict[str, object]:
    if not include_correlations:
        return _loss_only_summary(
            target,
            prediction,
            selected,
            huber_delta=huber_delta,
        )
    if not np.any(selected & np.isfinite(target) & np.isfinite(prediction)):
        summary = _loss_only_summary(
            target,
            prediction,
            selected,
            huber_delta=huber_delta,
        )
        summary.update(
            {
                "gene_pearson_mean": float("nan"),
                "gene_spearman_mean": float("nan"),
                "cell_pearson_mean": float("nan"),
                "cell_spearman_mean": float("nan"),
                "n_valid_gene_pearson": 0,
                "n_valid_gene_spearman": 0,
                "n_valid_cell_pearson": 0,
                "n_valid_cell_spearman": 0,
            }
        )
        return summary
    rows = np.flatnonzero(np.any(selected, axis=1))
    return _compact_masked_summary(
        target[rows], prediction[rows], selected[rows], huber_delta=huber_delta
    )


def evaluate_fixed_mask_strata(
    y_true: Any,
    y_pred: Any,
    target_mask: Any,
    true_spatial_edge_index: Any,
    *,
    huber_delta: float = 1.0,
    neighbor_availability: NeighborAvailability | None = None,
    strata_correlations: bool = False,
) -> dict[str, object]:
    """Evaluate frozen target-mask and true-neighbor availability strata.

    Neighbor availability is computed for every masked target gene from the
    off-diagonal *real* spatial adjacency.  Callers must use this same graph for
    all arms so the strata cannot change with the evaluated condition.
    """

    target = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    selected = np.asarray(target_mask, dtype=np.bool_)
    if target.ndim != 2 or prediction.shape != target.shape or selected.shape != target.shape:
        raise ValueError("y_true, y_pred, and target_mask must share [cells, genes] shape")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0:
        raise ValueError("huber_delta must be finite and positive")
    counts = selected.sum(axis=1, dtype=np.int64)
    genes = target.shape[1]
    target_membership = {
        "0_to_25": (counts > 0) & (4 * counts <= genes),
        "25_to_50": (4 * counts > genes) & (2 * counts <= genes),
        "50_to_75": (2 * counts > genes) & (4 * counts <= 3 * genes),
        "75_to_100": (4 * counts > 3 * genes) & (counts < genes),
        "exactly_100": counts == genes,
    }
    target_bins: dict[str, dict[str, object]] = {}
    for name in TARGET_MASK_BIN_ORDER:
        rows = target_membership[name]
        summary = _summary_for_selected_entries(
            target,
            prediction,
            selected & rows[:, None],
            huber_delta=float(huber_delta),
            include_correlations=bool(strata_correlations),
        )
        summary["n_target_cells"] = int(rows.sum())
        target_bins[name] = summary

    canonical_edges = _canonical_edge_index(
        true_spatial_edge_index, n_nodes=selected.shape[0]
    )
    off_diagonal_edges = canonical_edges[
        :, canonical_edges[0] != canonical_edges[1]
    ]
    if neighbor_availability is None:
        neighbor_availability = compute_true_neighbor_availability(
            selected, off_diagonal_edges
        )
    if neighbor_availability.fraction.shape != selected.shape:
        raise ValueError("neighbor_availability does not align to target_mask")
    if neighbor_availability.target_mask_checksum != ndarray_sha256(selected):
        raise ValueError("neighbor_availability was computed for a different target mask")
    if neighbor_availability.edge_index_checksum != ndarray_sha256(off_diagonal_edges):
        raise ValueError("neighbor_availability was computed for a different true graph")
    availability = neighbor_availability.fraction
    neighbor_membership = {
        "0_to_25": (availability >= 0.0) & (availability <= 0.25),
        "25_to_50": (availability > 0.25) & (availability <= 0.50),
        "50_to_75": (availability > 0.50) & (availability <= 0.75),
        "75_to_100": (availability > 0.75) & (availability <= 1.0),
    }
    neighbor_bins: dict[str, dict[str, object]] = {}
    for name in NEIGHBOR_OBSERVED_BIN_ORDER:
        bin_entries = selected & neighbor_membership[name]
        summary = _summary_for_selected_entries(
            target,
            prediction,
            bin_entries,
            huber_delta=float(huber_delta),
            include_correlations=bool(strata_correlations),
        )
        summary["n_target_entries"] = int(bin_entries.sum())
        neighbor_bins[name] = summary

    return {
        "overall": masked_regression_summary(
            target, prediction, selected, huber_delta=float(huber_delta)
        ),
        "zero_mask_cells": int(np.sum(counts == 0)),
        "target_mask_bins": target_bins,
        "neighbor_observed_bins": neighbor_bins,
        "masked_targets_without_true_neighbors": int(
            np.sum(
                selected
                & (neighbor_availability.off_diagonal_degree == 0)[:, None]
            )
        ),
        "neighbor_zero_degree_policy": "included_as_zero_in_0_to_25_bin",
        "neighbor_availability_checksum": neighbor_availability.checksum,
        "strata_correlations_computed": bool(strata_correlations),
        "target_mask_bin_definition": {
            "0_to_25": "(0,25%]",
            "25_to_50": "(25,50%]",
            "50_to_75": "(50,75%]",
            "75_to_100": "(75,100%)",
            "exactly_100": "100%",
        },
        "neighbor_observed_bin_definition": {
            "0_to_25": "[0,25%]",
            "25_to_50": "(25,50%]",
            "50_to_75": "(50,75%]",
            "75_to_100": "(75,100%]",
        },
    }


__all__ = [
    "ADJACENCY_ARMS",
    "CORE_ALIASES",
    "EVALUATION_MASK_BASE_SEED",
    "GRAPH_K",
    "GRAPH_RADIUS_UM",
    "GRAPH_SYMMETRY",
    "ISOLATED_ARM",
    "NEIGHBOR_OBSERVED_BIN_ORDER",
    "POSITION_PERMUTATION_SEED",
    "POSITION_PERMUTED_NULL_ARM",
    "SPATIAL_ARM",
    "STANDARDIZATION_SCALE_FLOOR",
    "TARGET_MASK_BIN_ORDER",
    "TRAINING_MASK_BASE_SEED",
    "AdjacencyArm",
    "AdjacencyBundle",
    "AdjacencyQC",
    "EqualCoreLog1pStandardizer",
    "ExplicitSelfMeanGraphSAGE",
    "FoldSplit",
    "MaskRealization",
    "NeighborAvailability",
    "add_exact_self_adjacency",
    "assert_valid_five_fold_splits",
    "build_five_fold_splits",
    "build_seeded_explicit_self_model",
    "compute_true_neighbor_availability",
    "derive_evaluation_mask_seed",
    "derive_training_mask_seed",
    "evaluate_fixed_mask_strata",
    "explicit_self_incoming_mean",
    "fit_equal_core_log1p_standardizer",
    "inverse_standardized_log1p",
    "mask_realization_sha256",
    "masked_regression_summary",
    "materialize_fixed_adjacencies",
    "ndarray_sha256",
    "sample_uniform_mask_numpy",
    "sample_uniform_mask_torch",
    "state_dict_sha256",
    "trainable_parameter_count",
    "transform_log1p_counts",
    "true_neighbor_observed_fraction",
    "validate_explicit_self_adjacency",
]

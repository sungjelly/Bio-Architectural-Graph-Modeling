"""Full-core preprocessing and exact high-k mutual graph construction.

This module is intentionally separate from the inductive preparation path.  It
supports the explicitly transductive full-core capacity campaign: every node is
used to fit node and edge transforms, and the graph is a literal exact kNN
graph whose radius is only a validated guard, not a candidate filter.

No cell, FOV, core, donor, or slide identifier is returned.  The prepared
artifact loader verifies the immutable artifact before this module selects the
non-identifying arrays needed by the campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .artifacts import ArtifactContractError, load_prepared_artifact
from .data import ALLOWED_METADATA_COLUMNS
from .splits import TrainOnlyPreprocessor


_PREPARED_ARTIFACT_KINDS = frozenset(
    {
        "normal_true_tissue_spatial_benchmark_preparation",
        "adjacent_normal_tissue_spatial_benchmark_preparation",
    }
)
_RBF_BINS = 8
_GEOMETRY_STATS_CHUNK_EDGES = 1_000_000
EDGE_ATTRIBUTE_NAMES = (
    "distance_um",
    "distance_over_radius",
    "log1p_distance_um",
    "delta_x_over_radius",
    "delta_y_over_radius",
    "cos_theta",
    "sin_theta",
    "cos_2theta",
    "sin_2theta",
    *(f"distance_rbf_{index}" for index in range(_RBF_BINS)),
)
_HASH_CHUNK_BYTES = 64 * 1024 * 1024


class FullCoreContractError(ValueError):
    """Raised when full-core inputs or outputs violate the campaign contract."""


class RadiusGuardError(FullCoreContractError):
    """Raised when a literal kNN graph would be truncated by its radius guard."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _update_array_digest(
    digest: Any,
    name: str,
    array: np.ndarray,
) -> None:
    values = np.asarray(array)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    digest.update(
        _canonical_json(
            {
                "name": name,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
            }
        )
    )
    byte_view = memoryview(values).cast("B")
    for start in range(0, byte_view.nbytes, _HASH_CHUNK_BYTES):
        digest.update(byte_view[start : start + _HASH_CHUNK_BYTES])


def _array_sha256(name: str, array: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array_digest(digest, name, array)
    return digest.hexdigest()


def _update_raw_array_bytes(digest: Any, array: np.ndarray) -> None:
    values = np.asarray(array)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    byte_view = memoryview(values).cast("B")
    for start in range(0, byte_view.nbytes, _HASH_CHUNK_BYTES):
        digest.update(byte_view[start : start + _HASH_CHUNK_BYTES])


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class FullCorePreprocessingQC:
    n_nodes: int
    n_genes: int
    n_measured_metadata_features: int
    n_model_covariates: int
    n_metadata_missing_values: int
    n_constant_expression_features: int
    n_constant_metadata_features: int
    expression_max_abs_fitted_mean: float
    metadata_max_abs_fitted_mean: float
    source_metadata_roundtrip_max_abs_error: float
    fit_scope: str
    metadata_reconstruction: str
    metadata_source_precision: str
    protected_identifier_arrays_returned: bool
    all_outputs_finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FullCorePreprocessingChecksums:
    source_artifact_id: str
    source_prepared_data_sha256: str
    expression_counts_sha256: str
    target_expression_sha256: str
    node_covariates_sha256: str
    coordinates_um_sha256: str
    macroblock_ids_sha256: str
    preprocessing_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FullCoreData:
    """Non-identifying full-core arrays and transforms fitted on every node."""

    expression_counts: np.ndarray = field(repr=False)
    target_expression: np.ndarray = field(repr=False)
    node_covariates: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    macroblock_ids: np.ndarray = field(repr=False)
    gene_names: tuple[str, ...]
    metadata_names: tuple[str, ...]
    expression_mean: np.ndarray = field(repr=False)
    expression_scale: np.ndarray = field(repr=False)
    metadata_median: np.ndarray = field(repr=False)
    metadata_mean: np.ndarray = field(repr=False)
    metadata_scale: np.ndarray = field(repr=False)
    metadata_missing_indicator_indices: np.ndarray = field(repr=False)
    preprocessing_qc: FullCorePreprocessingQC
    checksums: FullCorePreprocessingChecksums

    @property
    def n_nodes(self) -> int:
        return int(self.expression_counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression_counts.shape[1])


def _required_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
) -> np.ndarray:
    try:
        return np.asarray(arrays[name])
    except KeyError as exc:
        raise FullCoreContractError(
            f"Prepared artifact lacks required non-identifying array {name!r}."
        ) from exc


def _metadata_transform_uses_log1p(manifest: Mapping[str, Any]) -> bool:
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise FullCoreContractError("Prepared preprocessing metadata is missing.")
    transform = preprocessing.get("metadata_transform")
    if transform == "median imputation, log1p, standardization":
        return True
    if transform == "median imputation, standardization":
        return False
    raise FullCoreContractError(
        "Prepared metadata transform is not a supported reversible transform."
    )


def _validate_source_statistics(
    arrays: Mapping[str, np.ndarray],
    *,
    n_genes: int,
    n_metadata: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expression_mean = _required_array(arrays, "expression_mean").astype(
        np.float64, copy=False
    )
    expression_scale = _required_array(arrays, "expression_scale").astype(
        np.float64, copy=False
    )
    metadata_median = _required_array(arrays, "metadata_median").astype(
        np.float64, copy=False
    )
    metadata_mean = _required_array(arrays, "metadata_mean").astype(
        np.float64, copy=False
    )
    metadata_scale = _required_array(arrays, "metadata_scale").astype(
        np.float64, copy=False
    )
    raw_missing_indices = _required_array(
        arrays, "metadata_missing_indicator_indices"
    )
    if raw_missing_indices.dtype.kind not in "iu":
        raise FullCoreContractError(
            "Prepared metadata missing-indicator indices must be integral."
        )
    missing_indices = raw_missing_indices.astype(np.int64, copy=False)

    expected_shapes = {
        "expression_mean": (n_genes, expression_mean.shape),
        "expression_scale": (n_genes, expression_scale.shape),
        "metadata_median": (n_metadata, metadata_median.shape),
        "metadata_mean": (n_metadata, metadata_mean.shape),
        "metadata_scale": (n_metadata, metadata_scale.shape),
    }
    for name, (expected_length, shape) in expected_shapes.items():
        if shape != (expected_length,):
            raise FullCoreContractError(
                f"Prepared {name} does not match the declared feature schema."
            )
    if missing_indices.ndim != 1:
        raise FullCoreContractError(
            "Prepared metadata missing-indicator indices must be one-dimensional."
        )
    if len(missing_indices) and (
        missing_indices.min() < 0
        or missing_indices.max() >= n_metadata
        or np.any(missing_indices[1:] <= missing_indices[:-1])
    ):
        raise FullCoreContractError(
            "Prepared metadata missing-indicator indices are invalid."
        )
    statistics = (
        expression_mean,
        expression_scale,
        metadata_median,
        metadata_mean,
        metadata_scale,
    )
    if not all(np.isfinite(values).all() for values in statistics):
        raise FullCoreContractError("Prepared transform statistics must be finite.")
    if np.any(expression_scale <= 0) or np.any(metadata_scale <= 0):
        raise FullCoreContractError("Prepared transform scales must be positive.")
    return (
        expression_mean,
        expression_scale,
        metadata_median,
        metadata_mean,
        metadata_scale,
        missing_indices,
    )


def _reconstruct_measured_metadata(
    standardized_covariates: np.ndarray,
    *,
    source_median: np.ndarray,
    source_mean: np.ndarray,
    source_scale: np.ndarray,
    missing_indices: np.ndarray,
    log1p_metadata: bool,
) -> tuple[np.ndarray, float]:
    n_metadata = len(source_mean)
    expected_width = n_metadata + len(missing_indices)
    if standardized_covariates.ndim != 2 or standardized_covariates.shape[1] != (
        expected_width
    ):
        raise FullCoreContractError(
            "Prepared model covariates do not match the reversible metadata schema."
        )
    if not np.isfinite(standardized_covariates).all():
        raise FullCoreContractError("Prepared model covariates must be finite.")

    base = standardized_covariates[:, :n_metadata].astype(np.float64, copy=False)
    transformed_imputed = base * source_scale + source_mean
    if log1p_metadata:
        measured = np.expm1(transformed_imputed)
        tolerance = 64.0 * np.finfo(np.float32).eps
        if measured.min(initial=0.0) < -tolerance:
            raise FullCoreContractError(
                "Reversing prepared log1p metadata produced negative measurements."
            )
        measured = np.maximum(measured, 0.0)
    else:
        measured = transformed_imputed.copy()

    missing = np.zeros(measured.shape, dtype=bool)
    if len(missing_indices):
        indicators = standardized_covariates[:, n_metadata:].astype(
            np.float64, copy=False
        )
        if np.any((indicators != 0.0) & (indicators != 1.0)):
            raise FullCoreContractError(
                "Prepared metadata missing indicators must be exactly binary."
            )
        missing[:, missing_indices] = indicators.astype(bool)
        measured[missing] = np.nan

    imputed = np.where(missing, source_median, measured)
    if log1p_metadata:
        transformed = np.log1p(imputed)
    else:
        transformed = imputed
    roundtrip = (transformed - source_mean) / source_scale
    if len(missing_indices):
        roundtrip = np.concatenate(
            [roundtrip, missing[:, missing_indices].astype(np.float64)],
            axis=1,
        )
    max_error = float(
        np.max(
            np.abs(
                roundtrip
                - standardized_covariates.astype(np.float64, copy=False)
            ),
            initial=0.0,
        )
    )
    return measured, max_error


def _validate_public_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], str]:
    if manifest.get("artifact_kind") not in _PREPARED_ARTIFACT_KINDS:
        raise FullCoreContractError("Prepared artifact kind is not supported.")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or (
        selection.get("restricted_identifiers_emitted") is not False
    ):
        raise FullCoreContractError(
            "Prepared artifact does not affirm identifier-safe publication."
        )
    features = manifest.get("features")
    if not isinstance(features, Mapping):
        raise FullCoreContractError("Prepared feature manifest is missing.")
    measured_names = tuple(features.get("measured_metadata_names", ()))
    if measured_names != ALLOWED_METADATA_COLUMNS:
        raise FullCoreContractError(
            "Prepared metadata does not use the exact permitted allow-list."
        )
    raw_gene_names = features.get("gene_names")
    if (
        not isinstance(raw_gene_names, Sequence)
        or isinstance(raw_gene_names, (str, bytes))
    ):
        raise FullCoreContractError("Prepared biological probe names are invalid.")
    gene_names = tuple(str(name) for name in raw_gene_names)
    if not gene_names or any(not name for name in gene_names):
        raise FullCoreContractError("Prepared biological probe names are invalid.")
    if len(np.unique(np.asarray(gene_names, dtype="U256"))) != len(gene_names):
        raise FullCoreContractError("Prepared biological probe names are duplicated.")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise FullCoreContractError("Prepared artifact ID is missing.")
    return gene_names, artifact_id


def load_and_refit_full_core(
    prepared_artifact: str | Path,
    *,
    epsilon: float = 1e-8,
) -> FullCoreData:
    """Verify a prepared artifact and refit node transforms on all cells.

    The permitted morphology/imaging measurements are reconstructed by
    reversing the prepared transform.  This reproduces the stored float32
    covariates, including missingness, before fitting new full-core statistics.
    Direct routing identifiers loaded by the underlying verifier are discarded
    and are not represented in :class:`FullCoreData`.
    """

    try:
        manifest, loaded_arrays, _ = load_prepared_artifact(
            prepared_artifact,
            load_arrays=True,
        )
    except ArtifactContractError:
        raise
    if loaded_arrays is None:
        raise FullCoreContractError("Prepared artifact arrays were not loaded.")

    gene_names, artifact_id = _validate_public_manifest(manifest)
    counts = _required_array(loaded_arrays, "expression_counts")
    covariates = _required_array(loaded_arrays, "node_covariates")
    coordinates = _required_array(loaded_arrays, "coordinates_um").astype(
        np.float64, copy=False
    )
    macroblock_ids = _required_array(loaded_arrays, "macroblock_ids")

    if counts.ndim != 2 or counts.shape[1] != len(gene_names):
        raise FullCoreContractError(
            "Prepared expression counts do not match biological probe names."
        )
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise FullCoreContractError(
            "Prepared expression counts must be nonnegative integers."
        )
    n_nodes, n_genes = counts.shape
    if n_nodes == 0:
        raise FullCoreContractError("Prepared full core cannot be empty.")
    if coordinates.shape != (n_nodes, 2) or not np.isfinite(coordinates).all():
        raise FullCoreContractError(
            "Prepared coordinates must be finite and aligned to expression."
        )
    if (
        macroblock_ids.ndim != 1
        or len(macroblock_ids) != n_nodes
        or macroblock_ids.dtype.kind not in "US"
    ):
        raise FullCoreContractError(
            "Prepared macroblock IDs must be non-object strings aligned to nodes."
        )

    (
        _,
        _,
        source_metadata_median,
        source_metadata_mean,
        source_metadata_scale,
        source_missing_indices,
    ) = _validate_source_statistics(
        loaded_arrays,
        n_genes=n_genes,
        n_metadata=len(ALLOWED_METADATA_COLUMNS),
    )
    features = manifest["features"]
    expected_model_names = ALLOWED_METADATA_COLUMNS + tuple(
        f"{ALLOWED_METADATA_COLUMNS[index]}__missing"
        for index in source_missing_indices.tolist()
    )
    if tuple(features.get("model_covariate_names", ())) != expected_model_names:
        raise FullCoreContractError(
            "Prepared model covariate names do not match reversible statistics."
        )

    log1p_metadata = _metadata_transform_uses_log1p(manifest)
    measured_metadata, roundtrip_error = _reconstruct_measured_metadata(
        covariates,
        source_median=source_metadata_median,
        source_mean=source_metadata_mean,
        source_scale=source_metadata_scale,
        missing_indices=source_missing_indices,
        log1p_metadata=log1p_metadata,
    )
    all_fit_labels = np.full(n_nodes, "train", dtype="U5")
    preprocessor = TrainOnlyPreprocessor(
        log1p_metadata=log1p_metadata,
        epsilon=epsilon,
    ).fit(
        counts,
        measured_metadata,
        all_fit_labels,
        metadata_names=ALLOWED_METADATA_COLUMNS,
    )
    nodes = preprocessor.transform(counts, measured_metadata)

    counts = np.asarray(counts)
    target_expression = nodes.expression
    node_covariates = nodes.metadata
    macroblock_ids = np.asarray(macroblock_ids)
    expression_counts_sha256 = _array_sha256("expression_counts", counts)
    target_expression_sha256 = _array_sha256(
        "target_expression", target_expression
    )
    node_covariates_sha256 = _array_sha256(
        "node_covariates", node_covariates
    )
    coordinates_sha256 = _array_sha256("coordinates_um", coordinates)
    macroblock_sha256 = _array_sha256("macroblock_ids", macroblock_ids)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise FullCoreContractError("Prepared manifest file records are missing.")
    source_data_checksum = files.get("prepared_data.npz")
    if not _is_sha256(source_data_checksum):
        raise FullCoreContractError(
            "Prepared manifest lacks a valid prepared-data checksum."
        )

    preprocessing_payload: dict[str, object] = {
        "schema": "full_core_fit_v1",
        "fit_scope": "all nodes (transductive)",
        "source_artifact_id": artifact_id,
        "source_prepared_data_sha256": source_data_checksum,
        "gene_names": list(gene_names),
        "metadata_names": list(nodes.metadata_names),
        "log1p_metadata": log1p_metadata,
        "component_checksums": {
            "expression_counts": expression_counts_sha256,
            "target_expression": target_expression_sha256,
            "node_covariates": node_covariates_sha256,
            "coordinates_um": coordinates_sha256,
            "macroblock_ids": macroblock_sha256,
            "expression_mean": _array_sha256(
                "expression_mean", preprocessor.expression_mean_
            ),
            "expression_scale": _array_sha256(
                "expression_scale", preprocessor.expression_scale_
            ),
            "metadata_median": _array_sha256(
                "metadata_median", preprocessor.metadata_median_
            ),
            "metadata_mean": _array_sha256(
                "metadata_mean", preprocessor.metadata_mean_
            ),
            "metadata_scale": _array_sha256(
                "metadata_scale", preprocessor.metadata_scale_
            ),
            "metadata_missing_indicator_indices": _array_sha256(
                "metadata_missing_indicator_indices",
                preprocessor.missing_indicator_indices_,
            ),
        },
    }
    checksums = FullCorePreprocessingChecksums(
        source_artifact_id=artifact_id,
        source_prepared_data_sha256=source_data_checksum,
        expression_counts_sha256=expression_counts_sha256,
        target_expression_sha256=target_expression_sha256,
        node_covariates_sha256=node_covariates_sha256,
        coordinates_um_sha256=coordinates_sha256,
        macroblock_ids_sha256=macroblock_sha256,
        preprocessing_sha256=_payload_sha256(preprocessing_payload),
    )
    expression_feature_scale = target_expression.astype(
        np.float64, copy=False
    ).std(axis=0, ddof=0)
    metadata_feature_scale = node_covariates[
        :, : len(ALLOWED_METADATA_COLUMNS)
    ].astype(np.float64, copy=False).std(axis=0, ddof=0)
    qc = FullCorePreprocessingQC(
        n_nodes=n_nodes,
        n_genes=n_genes,
        n_measured_metadata_features=len(ALLOWED_METADATA_COLUMNS),
        n_model_covariates=node_covariates.shape[1],
        n_metadata_missing_values=int(np.isnan(measured_metadata).sum()),
        n_constant_expression_features=int(
            np.sum(expression_feature_scale <= epsilon)
        ),
        n_constant_metadata_features=int(
            np.sum(metadata_feature_scale <= epsilon)
        ),
        expression_max_abs_fitted_mean=float(
            np.max(
                np.abs(
                    target_expression.astype(np.float64, copy=False).mean(axis=0)
                ),
                initial=0.0,
            )
        ),
        metadata_max_abs_fitted_mean=float(
            np.max(
                np.abs(
                    node_covariates[
                        :, : len(ALLOWED_METADATA_COLUMNS)
                    ].astype(np.float64, copy=False).mean(axis=0)
                ),
                initial=0.0,
            )
        ),
        source_metadata_roundtrip_max_abs_error=roundtrip_error,
        fit_scope="all nodes (transductive)",
        metadata_reconstruction=(
            "inverse prepared median-imputation/log1p/standardization"
            if log1p_metadata
            else "inverse prepared median-imputation/standardization"
        ),
        metadata_source_precision=str(covariates.dtype),
        protected_identifier_arrays_returned=False,
        all_outputs_finite=bool(
            np.isfinite(target_expression).all()
            and np.isfinite(node_covariates).all()
            and np.isfinite(coordinates).all()
        ),
    )
    if not qc.all_outputs_finite:
        raise FullCoreContractError(
            "Full-core preprocessing produced non-finite output."
        )

    return FullCoreData(
        expression_counts=counts,
        target_expression=target_expression,
        node_covariates=node_covariates,
        coordinates_um=coordinates,
        macroblock_ids=macroblock_ids,
        gene_names=gene_names,
        metadata_names=nodes.metadata_names,
        expression_mean=preprocessor.expression_mean_.astype(
            np.float64, copy=False
        ),
        expression_scale=preprocessor.expression_scale_.astype(
            np.float64, copy=False
        ),
        metadata_median=preprocessor.metadata_median_.astype(
            np.float64, copy=False
        ),
        metadata_mean=preprocessor.metadata_mean_.astype(
            np.float64, copy=False
        ),
        metadata_scale=preprocessor.metadata_scale_.astype(
            np.float64, copy=False
        ),
        metadata_missing_indicator_indices=(
            preprocessor.missing_indicator_indices_.astype(np.int64, copy=False)
        ),
        preprocessing_qc=qc,
        checksums=checksums,
    )


@dataclass(frozen=True)
class HighKGraphQC:
    n_nodes: int
    k: int
    n_directed_candidates: int
    n_directed_edges: int
    n_undirected_edges: int
    n_components: int
    n_isolated_nodes: int
    mean_degree: float
    median_degree: float
    p95_degree: float
    max_degree: int
    mutual_candidate_fraction: float
    candidate_kth_distance_mean_um: float
    candidate_kth_distance_p50_um: float
    candidate_kth_distance_p95_um: float
    candidate_kth_distance_max_um: float
    radius_guard_um: float
    radius_guard_margin_um: float
    edge_distance_mean_um: float
    edge_distance_p50_um: float
    edge_distance_p95_um: float
    edge_distance_max_um: float
    zero_distance_undirected_edges: int
    self_loops: int
    duplicate_directed_edges: int
    directed_edge_pairs_are_symmetric: bool
    receiver_sorted: bool
    edge_attribute_count: int
    edge_standardization_scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HighKGraphChecksums:
    directed_candidates_sha256: str
    receiver_sorted_edge_codes_sha256: str
    edge_index_sha256: str
    standardized_edge_attributes_sha256: str
    graph_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverEdgeShard:
    """All incoming edges for one contiguous half-open receiver range."""

    receiver_start: int
    receiver_stop: int
    edge_index: np.ndarray = field(repr=False)
    edge_attributes: np.ndarray = field(repr=False)
    checksum_sha256: str

    def __post_init__(self) -> None:
        edges = np.asarray(self.edge_index)
        attributes = np.asarray(self.edge_attributes)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise FullCoreContractError("Shard edge_index must have shape [2, E].")
        if attributes.shape != (edges.shape[1], len(EDGE_ATTRIBUTE_NAMES)):
            raise FullCoreContractError(
                "Shard edge attributes must align and contain 17 features."
            )
        if edges.dtype.kind not in "iu":
            raise FullCoreContractError("Shard edge_index must be integral.")
        if not np.isfinite(attributes).all():
            raise FullCoreContractError("Shard edge attributes must be finite.")
        if self.receiver_start < 0 or self.receiver_stop <= self.receiver_start:
            raise FullCoreContractError("Shard receiver range is invalid.")
        if edges.shape[1]:
            source, receiver = edges
            if (
                receiver.min() < self.receiver_start
                or receiver.max() >= self.receiver_stop
                or np.any(source == receiver)
            ):
                raise FullCoreContractError(
                    "Shard edges violate receiver range or no-loop invariants."
                )
            if np.any(receiver[1:] < receiver[:-1]):
                raise FullCoreContractError("Shard edges are not receiver-sorted.")
            same_receiver = receiver[1:] == receiver[:-1]
            if np.any(source[1:][same_receiver] <= source[:-1][same_receiver]):
                raise FullCoreContractError(
                    "Shard sources are not strictly sorted within receivers."
                )
        if not _is_sha256(self.checksum_sha256):
            raise FullCoreContractError("Shard checksum is invalid.")

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class ReceiverSortedGraph:
    """Exact mutual-kNN graph stored once as receiver-aligned shards."""

    n_nodes: int
    k: int
    radius_guard_um: float
    edge_attribute_names: tuple[str, ...]
    edge_attribute_mean: np.ndarray = field(repr=False)
    edge_attribute_scale: np.ndarray = field(repr=False)
    shards: tuple[ReceiverEdgeShard, ...] = field(repr=False)
    qc: HighKGraphQC
    checksums: HighKGraphChecksums

    def __post_init__(self) -> None:
        if self.edge_attribute_names != EDGE_ATTRIBUTE_NAMES:
            raise FullCoreContractError(
                "High-k graph must use the locked 17-feature geometry schema."
            )
        if self.edge_attribute_mean.shape != (len(EDGE_ATTRIBUTE_NAMES),) or (
            self.edge_attribute_scale.shape != (len(EDGE_ATTRIBUTE_NAMES),)
        ):
            raise FullCoreContractError("Edge transform statistics have wrong shape.")
        if (
            not np.isfinite(self.edge_attribute_mean).all()
            or not np.isfinite(self.edge_attribute_scale).all()
            or np.any(self.edge_attribute_scale <= 0)
        ):
            raise FullCoreContractError("Edge transform statistics are invalid.")
        expected_start = 0
        edge_count = 0
        for shard in self.shards:
            if shard.receiver_start != expected_start:
                raise FullCoreContractError(
                    "Receiver shards must be contiguous and ordered."
                )
            expected_start = shard.receiver_stop
            edge_count += shard.n_edges
        if expected_start != self.n_nodes or edge_count != self.qc.n_directed_edges:
            raise FullCoreContractError(
                "Receiver shards do not cover the graph exactly once."
            )

    def iter_shards(self) -> Iterator[ReceiverEdgeShard]:
        """Iterate incoming receiver partitions without concatenating arrays."""

        return iter(self.shards)

    def concatenate(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize all receiver-sorted edges and attributes on explicit request."""

        if not self.shards:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, len(EDGE_ATTRIBUTE_NAMES)), dtype=np.float32),
            )
        if len(self.shards) == 1:
            return self.shards[0].edge_index, self.shards[0].edge_attributes
        return (
            np.concatenate(
                [shard.edge_index for shard in self.shards],
                axis=1,
            ),
            np.concatenate(
                [shard.edge_attributes for shard in self.shards],
                axis=0,
            ),
        )


def _validate_graph_inputs(
    coordinates_um: np.ndarray,
    *,
    k: int,
    radius_guard_um: float,
    query_chunk_size: int,
    receiver_chunk_size: int,
    workers: int,
    epsilon: float,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.isfinite(coordinates).all()
    ):
        raise FullCoreContractError(
            "Graph coordinates must be finite with shape [N, 2] and N >= 2."
        )
    if not isinstance(k, (int, np.integer)) or k <= 0 or k >= len(coordinates):
        raise FullCoreContractError("k must be an integer in [1, N - 1].")
    if not np.isfinite(radius_guard_um) or radius_guard_um <= 0:
        raise FullCoreContractError("radius_guard_um must be positive and finite.")
    if (
        not isinstance(query_chunk_size, (int, np.integer))
        or query_chunk_size <= 0
        or not isinstance(receiver_chunk_size, (int, np.integer))
        or receiver_chunk_size <= 0
    ):
        raise FullCoreContractError("Graph chunk sizes must be positive integers.")
    if (
        not isinstance(workers, (int, np.integer))
        or workers == 0
        or workers < -1
    ):
        raise FullCoreContractError("workers must be -1 or a positive integer.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise FullCoreContractError("epsilon must be positive and finite.")
    if len(coordinates) > int(np.sqrt(np.iinfo(np.int64).max)):
        raise FullCoreContractError("Node count cannot be encoded safely in int64.")
    return coordinates


def _select_exact_neighbors(
    tree: cKDTree,
    coordinates: np.ndarray,
    row_indices: np.ndarray,
    *,
    k: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(coordinates)
    query_count = min(n_nodes, k + 2)
    distances, indices = tree.query(
        coordinates[row_indices],
        k=query_count,
        eps=0.0,
        workers=workers,
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    if (
        distances.shape != indices.shape
        or distances.shape[0] != len(row_indices)
        or not np.isfinite(distances).all()
        or np.any(indices < 0)
        or np.any(indices >= n_nodes)
    ):
        raise FullCoreContractError("Exact kNN query returned invalid candidates.")

    order = np.lexsort((indices, distances), axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    self_matches = indices == row_indices[:, None]
    if np.any(self_matches.sum(axis=1) > 1):
        raise FullCoreContractError("Exact kNN query duplicated a node candidate.")
    has_self = self_matches.any(axis=1)
    self_position = np.argmax(self_matches, axis=1)
    wanted = min(k + 1, n_nodes - 1)
    take_position = np.broadcast_to(
        np.arange(wanted, dtype=np.int64),
        (len(row_indices), wanted),
    ).copy()
    take_position[has_self] += (
        take_position[has_self] >= self_position[has_self, None]
    )
    nonself_indices = np.take_along_axis(indices, take_position, axis=1)
    nonself_distances = np.take_along_axis(distances, take_position, axis=1)
    if np.any(nonself_indices == row_indices[:, None]):
        raise FullCoreContractError(
            "Exact kNN candidate selection retained a self-loop."
        )

    selected_indices = nonself_indices[:, :k].copy()
    selected_distances = nonself_distances[:, :k].copy()
    if wanted > k:
        boundary_tie = nonself_distances[:, k - 1] == nonself_distances[:, k]
        for local_row in np.flatnonzero(boundary_tie).tolist():
            node = int(row_indices[local_row])
            boundary = float(nonself_distances[local_row, k - 1])
            radius = np.nextafter(boundary, np.inf)
            tied_candidates = np.asarray(
                tree.query_ball_point(coordinates[node], radius),
                dtype=np.int64,
            )
            tied_candidates = tied_candidates[tied_candidates != node]
            exact_delta = coordinates[tied_candidates] - coordinates[node]
            exact_distance = np.linalg.norm(exact_delta, axis=1)
            inside = exact_distance <= radius
            tied_candidates = tied_candidates[inside]
            exact_distance = exact_distance[inside]
            exact_order = np.lexsort((tied_candidates, exact_distance))
            if len(exact_order) < k:
                raise FullCoreContractError(
                    "Boundary-tie refinement returned fewer than k neighbors."
                )
            selected = exact_order[:k]
            selected_indices[local_row] = tied_candidates[selected]
            selected_distances[local_row] = exact_distance[selected]
    return selected_indices, selected_distances


def _sorted_exact_candidate_codes(
    coordinates: np.ndarray,
    *,
    k: int,
    query_chunk_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(coordinates)
    tree = cKDTree(coordinates)
    codes = np.empty(n_nodes * k, dtype=np.int64)
    kth_distances = np.empty(n_nodes, dtype=np.float64)
    for start in range(0, n_nodes, query_chunk_size):
        stop = min(start + query_chunk_size, n_nodes)
        rows = np.arange(start, stop, dtype=np.int64)
        neighbors, distances = _select_exact_neighbors(
            tree,
            coordinates,
            rows,
            k=k,
            workers=workers,
        )
        if np.any(neighbors == rows[:, None]):
            raise FullCoreContractError("Exact kNN candidates contain self-loops.")
        flat_start = start * k
        flat_stop = stop * k
        codes[flat_start:flat_stop] = (
            rows[:, None] * n_nodes + neighbors
        ).reshape(-1)
        kth_distances[start:stop] = distances[:, -1]
    codes.sort()
    if np.any(codes[1:] == codes[:-1]):
        raise FullCoreContractError("Exact kNN candidates contain duplicates.")
    return codes, kth_distances


def _mutual_edge_codes(
    sorted_candidate_codes: np.ndarray,
    *,
    n_nodes: int,
    search_chunk_size: int,
) -> np.ndarray:
    mutual = np.empty(len(sorted_candidate_codes), dtype=bool)
    for start in range(0, len(sorted_candidate_codes), search_chunk_size):
        stop = min(start + search_chunk_size, len(sorted_candidate_codes))
        candidate = sorted_candidate_codes[start:stop]
        source = candidate // n_nodes
        receiver = candidate - source * n_nodes
        reverse = receiver * n_nodes + source
        position = np.searchsorted(sorted_candidate_codes, reverse)
        found = position < len(sorted_candidate_codes)
        valid_position = position[found]
        found[found] = (
            sorted_candidate_codes[valid_position] == reverse[found]
        )
        mutual[start:stop] = found
    codes = sorted_candidate_codes[mutual]
    if len(codes) == 0:
        raise FullCoreContractError("Exact mutual kNN graph contains no edges.")
    if len(codes) % 2:
        raise FullCoreContractError("Exact mutual kNN edge count is not symmetric.")
    # The mutual directed-code set is invariant to reversal.  Interpreting its
    # sorted source-major codes as receiver-major codes gives the same edge set
    # already ordered by (receiver, source), without another O(E) sort.
    return codes


def _decode_receiver_codes(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    receiver = receiver_codes // n_nodes
    source = receiver_codes - receiver * n_nodes
    return source.astype(np.int64, copy=False), receiver.astype(
        np.int64, copy=False
    )


def _raw_geometry_attributes(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
    coordinates: np.ndarray,
    radius_guard_um: float,
) -> np.ndarray:
    if len(receiver_codes) == 0:
        return np.empty((0, len(EDGE_ATTRIBUTE_NAMES)), dtype=np.float32)
    source, receiver = _decode_receiver_codes(
        receiver_codes,
        n_nodes=n_nodes,
    )
    delta = coordinates[receiver] - coordinates[source]
    distance = np.linalg.norm(delta, axis=1)
    unit = np.divide(
        delta,
        distance[:, None],
        out=np.zeros_like(delta),
        where=distance[:, None] > 0,
    )
    cos_theta = unit[:, 0]
    sin_theta = unit[:, 1]
    centers = np.linspace(
        0.0,
        radius_guard_um,
        _RBF_BINS,
        dtype=np.float64,
    )
    width = max(
        float(centers[1] - centers[0]),
        np.finfo(np.float64).eps,
    )
    rbf = np.exp(
        -0.5 * ((distance[:, None] - centers[None, :]) / width) ** 2
    )
    attributes = np.column_stack(
        [
            distance,
            distance / radius_guard_um,
            np.log1p(distance),
            delta[:, 0] / radius_guard_um,
            delta[:, 1] / radius_guard_um,
            cos_theta,
            sin_theta,
            cos_theta**2 - sin_theta**2,
            2.0 * cos_theta * sin_theta,
            rbf,
        ]
    )
    return attributes.astype(np.float32, copy=False)


def _receiver_ranges(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
    receiver_chunk_size: int,
) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for receiver_start in range(0, n_nodes, receiver_chunk_size):
        receiver_stop = min(receiver_start + receiver_chunk_size, n_nodes)
        edge_start = int(
            np.searchsorted(receiver_codes, receiver_start * n_nodes)
        )
        edge_stop = int(
            np.searchsorted(receiver_codes, receiver_stop * n_nodes)
        )
        result.append(
            (receiver_start, receiver_stop, edge_start, edge_stop)
        )
    return result


def _fit_edge_standardizer(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
    coordinates: np.ndarray,
    radius_guard_um: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(len(EDGE_ATTRIBUTE_NAMES), dtype=np.float64)
    sum_squared_deviation = np.zeros_like(mean)
    undirected_distances = np.empty(
        len(receiver_codes) // 2,
        dtype=np.float32,
    )
    undirected_offset = 0
    for edge_start in range(0, len(receiver_codes), _GEOMETRY_STATS_CHUNK_EDGES):
        edge_stop = min(
            edge_start + _GEOMETRY_STATS_CHUNK_EDGES,
            len(receiver_codes),
        )
        raw = _raw_geometry_attributes(
            receiver_codes[edge_start:edge_stop],
            n_nodes=n_nodes,
            coordinates=coordinates,
            radius_guard_um=radius_guard_um,
        )
        codes = receiver_codes[edge_start:edge_stop]
        source, receiver = _decode_receiver_codes(codes, n_nodes=n_nodes)
        unique_relation = source < receiver
        unique_distances = raw[unique_relation, 0]
        next_undirected_offset = undirected_offset + len(unique_distances)
        undirected_distances[
            undirected_offset:next_undirected_offset
        ] = unique_distances
        undirected_offset = next_undirected_offset
        if len(raw) == 0:
            continue
        batch = raw.astype(np.float64, copy=False)
        batch_count = len(batch)
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.sum((batch - batch_mean) ** 2, axis=0)
        delta = batch_mean - mean
        combined_count = count + batch_count
        mean += delta * (batch_count / combined_count)
        sum_squared_deviation += (
            batch_m2 + delta**2 * count * batch_count / combined_count
        )
        count = combined_count
    if count != len(receiver_codes):
        raise FullCoreContractError(
            "Edge-standardization pass did not cover every retained edge."
        )
    if undirected_offset != len(undirected_distances):
        raise FullCoreContractError(
            "Edge-QC pass did not find one distance per undirected relation."
        )
    raw_scale = np.sqrt(sum_squared_deviation / count)
    scale = np.where(raw_scale > epsilon, raw_scale, 1.0)
    return mean, scale, undirected_distances


def _component_count(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
    receiver_boundaries: np.ndarray,
) -> int:
    index_dtype = np.int32 if n_nodes <= np.iinfo(np.int32).max else np.int64
    source_indices = np.empty(len(receiver_codes), dtype=index_dtype)
    np.remainder(
        receiver_codes,
        n_nodes,
        out=source_indices,
        casting="unsafe",
    )
    adjacency = csr_matrix(
        (
            np.ones(len(receiver_codes), dtype=np.int8),
            source_indices,
            receiver_boundaries.astype(np.int64, copy=False),
        ),
        shape=(n_nodes, n_nodes),
    )
    return int(
        connected_components(
            adjacency,
            directed=False,
            return_labels=False,
        )
    )


def _build_standardized_shards(
    receiver_codes: np.ndarray,
    receiver_ranges: Sequence[tuple[int, int, int, int]],
    *,
    n_nodes: int,
    coordinates: np.ndarray,
    radius_guard_um: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[
    tuple[ReceiverEdgeShard, ...],
    str,
    str,
]:
    shards: list[ReceiverEdgeShard] = []
    edge_index_digest = hashlib.sha256()
    edge_index_digest.update(
        _canonical_json(
            {
                "name": "receiver_sorted_edge_pairs",
                "shape": [len(receiver_codes), 2],
                "dtype": np.dtype(np.int64).str,
            }
        )
    )
    attribute_digest = hashlib.sha256()
    attribute_digest.update(
        _canonical_json(
            {
                "name": "standardized_edge_attributes",
                "shape": [len(receiver_codes), len(EDGE_ATTRIBUTE_NAMES)],
                "dtype": np.dtype(np.float32).str,
            }
        )
    )
    for receiver_start, receiver_stop, edge_start, edge_stop in receiver_ranges:
        codes = receiver_codes[edge_start:edge_stop]
        source, receiver = _decode_receiver_codes(codes, n_nodes=n_nodes)
        edge_index = np.vstack([source, receiver]).astype(
            np.int64, copy=False
        )
        raw = _raw_geometry_attributes(
            codes,
            n_nodes=n_nodes,
            coordinates=coordinates,
            radius_guard_um=radius_guard_um,
        )
        standardized = (
            (raw.astype(np.float64, copy=False) - mean) / scale
        ).astype(np.float32)
        edge_checksum = _array_sha256("edge_index", edge_index)
        attribute_checksum = _array_sha256(
            "edge_attributes", standardized
        )
        shard_checksum = _payload_sha256(
            {
                "receiver_start": receiver_start,
                "receiver_stop": receiver_stop,
                "edge_index_sha256": edge_checksum,
                "edge_attributes_sha256": attribute_checksum,
            }
        )
        shards.append(
            ReceiverEdgeShard(
                receiver_start=receiver_start,
                receiver_stop=receiver_stop,
                edge_index=edge_index,
                edge_attributes=standardized,
                checksum_sha256=shard_checksum,
            )
        )
        _update_raw_array_bytes(edge_index_digest, edge_index.T)
        _update_raw_array_bytes(attribute_digest, standardized)
    return (
        tuple(shards),
        edge_index_digest.hexdigest(),
        attribute_digest.hexdigest(),
    )


def build_exact_mutual_knn_graph(
    coordinates_um: np.ndarray,
    *,
    k: int = 1000,
    radius_guard_um: float = 650.0,
    query_chunk_size: int = 2048,
    receiver_chunk_size: int = 512,
    mutual_search_chunk_size: int = 4_000_000,
    workers: int = 1,
    epsilon: float = 1e-8,
) -> ReceiverSortedGraph:
    """Build a deterministic literal-k exact mutual graph.

    ``radius_guard_um`` never filters candidates.  The function first finds the
    exact ``k`` nearest non-self neighbors with ``cKDTree`` (``eps=0``), then
    raises :class:`RadiusGuardError` unless every kth-neighbor distance is at
    most the guard.  Equal-distance boundary candidates are deterministically
    resolved by node index.

    Directed pairs are encoded as int64 values and mutuality is computed with
    sorted searches; no Python tuple set is materialized.  Geometry statistics
    are fitted over all retained directed edges.  Returned shards are sorted by
    receiver and contain only edge indices plus standardized 17-feature
    geometry, avoiding a retained duplicate of the raw edge-attribute matrix.
    """

    coordinates = _validate_graph_inputs(
        coordinates_um,
        k=k,
        radius_guard_um=radius_guard_um,
        query_chunk_size=query_chunk_size,
        receiver_chunk_size=receiver_chunk_size,
        workers=workers,
        epsilon=epsilon,
    )
    if (
        not isinstance(mutual_search_chunk_size, (int, np.integer))
        or mutual_search_chunk_size <= 0
    ):
        raise FullCoreContractError(
            "mutual_search_chunk_size must be a positive integer."
        )
    n_nodes = len(coordinates)
    candidates, kth_distances = _sorted_exact_candidate_codes(
        coordinates,
        k=int(k),
        query_chunk_size=int(query_chunk_size),
        workers=int(workers),
    )
    observed_max = float(kth_distances.max())
    guard_tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(radius_guard_um)))
    )
    if observed_max > float(radius_guard_um) + guard_tolerance:
        raise RadiusGuardError(
            "radius_guard_um would truncate exact kNN candidates: "
            f"observed maximum kth-neighbor distance is {observed_max:.6g} um "
            f"but the guard is {float(radius_guard_um):.6g} um."
        )
    candidate_checksum = _array_sha256(
        "sorted_directed_candidate_codes",
        candidates,
    )
    n_directed_candidates = len(candidates)
    receiver_codes = _mutual_edge_codes(
        candidates,
        n_nodes=n_nodes,
        search_chunk_size=int(mutual_search_chunk_size),
    )
    del candidates
    receiver_code_checksum = _array_sha256(
        "receiver_sorted_edge_codes",
        receiver_codes,
    )
    receiver_ranges = _receiver_ranges(
        receiver_codes,
        n_nodes=n_nodes,
        receiver_chunk_size=int(receiver_chunk_size),
    )
    edge_mean, edge_scale, edge_distances = _fit_edge_standardizer(
        receiver_codes,
        n_nodes=n_nodes,
        coordinates=coordinates,
        radius_guard_um=float(radius_guard_um),
        epsilon=float(epsilon),
    )

    receiver_boundaries = np.searchsorted(
        receiver_codes,
        np.arange(n_nodes + 1, dtype=np.int64) * n_nodes,
    ).astype(np.int64, copy=False)
    degree = np.diff(receiver_boundaries)
    n_components = _component_count(
        receiver_codes,
        n_nodes=n_nodes,
        receiver_boundaries=receiver_boundaries,
    )
    n_directed_edges = len(receiver_codes)
    zero_distance_undirected = int(np.sum(edge_distances == 0.0))
    qc = HighKGraphQC(
        n_nodes=n_nodes,
        k=int(k),
        n_directed_candidates=n_directed_candidates,
        n_directed_edges=n_directed_edges,
        n_undirected_edges=n_directed_edges // 2,
        n_components=n_components,
        n_isolated_nodes=int(np.sum(degree == 0)),
        mean_degree=float(degree.mean()),
        median_degree=float(np.median(degree)),
        p95_degree=float(np.quantile(degree, 0.95)),
        max_degree=int(degree.max(initial=0)),
        mutual_candidate_fraction=float(
            n_directed_edges / n_directed_candidates
        ),
        candidate_kth_distance_mean_um=float(kth_distances.mean()),
        candidate_kth_distance_p50_um=float(
            np.quantile(kth_distances, 0.50)
        ),
        candidate_kth_distance_p95_um=float(
            np.quantile(kth_distances, 0.95)
        ),
        candidate_kth_distance_max_um=observed_max,
        radius_guard_um=float(radius_guard_um),
        radius_guard_margin_um=float(radius_guard_um) - observed_max,
        edge_distance_mean_um=float(edge_distances.mean()),
        edge_distance_p50_um=float(np.quantile(edge_distances, 0.50)),
        edge_distance_p95_um=float(np.quantile(edge_distances, 0.95)),
        edge_distance_max_um=float(edge_distances.max()),
        zero_distance_undirected_edges=zero_distance_undirected,
        self_loops=0,
        duplicate_directed_edges=0,
        directed_edge_pairs_are_symmetric=True,
        receiver_sorted=True,
        edge_attribute_count=len(EDGE_ATTRIBUTE_NAMES),
        edge_standardization_scope="all retained directed edges",
    )
    del edge_distances

    shards, edge_index_checksum, attributes_checksum = (
        _build_standardized_shards(
            receiver_codes,
            receiver_ranges,
            n_nodes=n_nodes,
            coordinates=coordinates,
            radius_guard_um=float(radius_guard_um),
            mean=edge_mean,
            scale=edge_scale,
        )
    )
    graph_payload: dict[str, object] = {
        "schema": "exact_mutual_knn_receiver_shards_v1",
        "config": {
            "k": int(k),
            "radius_guard_um": float(radius_guard_um),
            "symmetry": "mutual",
            "self_loops": False,
            "rbf_bins": _RBF_BINS,
            "edge_standardization_scope": "all retained directed edges",
        },
        "edge_attribute_names": list(EDGE_ATTRIBUTE_NAMES),
        "edge_attribute_mean_sha256": _array_sha256(
            "edge_attribute_mean", edge_mean
        ),
        "edge_attribute_scale_sha256": _array_sha256(
            "edge_attribute_scale", edge_scale
        ),
        "qc": qc.to_dict(),
        "component_checksums": {
            "directed_candidates": candidate_checksum,
            "receiver_sorted_edge_codes": receiver_code_checksum,
            "edge_index": edge_index_checksum,
            "standardized_edge_attributes": attributes_checksum,
        },
    }
    checksums = HighKGraphChecksums(
        directed_candidates_sha256=candidate_checksum,
        receiver_sorted_edge_codes_sha256=receiver_code_checksum,
        edge_index_sha256=edge_index_checksum,
        standardized_edge_attributes_sha256=attributes_checksum,
        graph_sha256=_payload_sha256(graph_payload),
    )
    return ReceiverSortedGraph(
        n_nodes=n_nodes,
        k=int(k),
        radius_guard_um=float(radius_guard_um),
        edge_attribute_names=EDGE_ATTRIBUTE_NAMES,
        edge_attribute_mean=edge_mean,
        edge_attribute_scale=edge_scale,
        shards=shards,
        qc=qc,
        checksums=checksums,
    )

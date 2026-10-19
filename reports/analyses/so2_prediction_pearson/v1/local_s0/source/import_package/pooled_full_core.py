"""Alias-safe pooled loading for the ten adjacent-normal full-core artifacts.

The pooled campaign fits one shared expression transform while retaining the
existing independently fitted morphology/imaging transform for each core.  It
does not concatenate cores or construct cross-core edges.  Only the opaque
aliases ``ANC-01`` through ``ANC-10`` are exposed; source routing identifiers
loaded by the verified artifact reader are discarded by :mod:`full_core`.

The exact source receipts below bind each public alias to the prepared
artifact that was frozen for the campaign.  This prevents a valid but wrong
prepared artifact from being silently assigned to an alias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from . import full_core as _full_core
from .artifacts import ArtifactContractError, load_prepared_artifact
from .data import ALLOWED_METADATA_COLUMNS
from .paths import current_paths


FROZEN_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
ANC_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
EXPECTED_TOTAL_NODES = 117_386
EXPECTED_N_GENES = 1_000
EXPECTED_N_MODEL_COVARIATES = 22
EXPRESSION_SCALE_EPSILON = 1e-8
_PREPARED_RELATIVE_ROOT = Path(
    "processed/adjacent_normal_10core_qkv_large_k_v1"
)
_PREPARED_ARTIFACT_KIND = (
    "adjacent_normal_tissue_spatial_benchmark_preparation"
)
_TISSUE_CONTEXT = "pathology_confirmed_adjacent_normal"
_TECHNICAL_CONTROL_PREFIXES = ("Negative", "SystemControl")


class PooledFullCoreContractError(ValueError):
    """Raised when a pooled input violates the frozen campaign contract."""


@dataclass(frozen=True, slots=True)
class PreparedCoreReceipt:
    """Frozen non-identifying receipt for one prepared core."""

    alias: str
    n_nodes: int
    manifest_sha256: str
    prepared_data_sha256: str


EXPECTED_PREPARED_CORES = (
    PreparedCoreReceipt(
        alias="ANC-01",
        n_nodes=14_657,
        manifest_sha256=(
            "38a259380b4b9ca228293ec7d07116e98eba5358cce99c08318a8aa8b8a013b9"
        ),
        prepared_data_sha256=(
            "2d80219073936798c3fea5b939efb2b50130459b39f5395ad09b2b498b9cc1de"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-02",
        n_nodes=9_785,
        manifest_sha256=(
            "1e670cfda427da9251dae47911bc5757c7767a87c2887b03e69c32907c7bbe37"
        ),
        prepared_data_sha256=(
            "e1f697932915f0bac19a34ccab5aa40a3123da9cec6c6cb2a606fd7eada03d15"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-03",
        n_nodes=14_756,
        manifest_sha256=(
            "73b893664417304b851fda0365cc7a5f10cbd9476c132500bcf22be4fc6a59f4"
        ),
        prepared_data_sha256=(
            "c0e65b5643600bcd6cd567145dc0df3879674a577025c9eb45a5a893890bf5ca"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-04",
        n_nodes=7_816,
        manifest_sha256=(
            "9e83f01c0cf8dd651d94a7da6b512c3c00557220cb2e4d3edb05ecbc72462add"
        ),
        prepared_data_sha256=(
            "31ce2f235fe4b7eed4a81185d811931485267dfc9316a96c3087c2046a2e5a04"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-05",
        n_nodes=7_450,
        manifest_sha256=(
            "69075fb19cdbdcd4685a29d2e5d252f0ca89577165f4c173643f6f39d0461b28"
        ),
        prepared_data_sha256=(
            "c0713d5d70fc9597a54b30c1d5264a175e353cdd9d9f7d9c0e47f08c10d0dd3e"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-06",
        n_nodes=13_122,
        manifest_sha256=(
            "f2df39bdcbd39f21b86653c941d39bc39a76273d9bdcb78019a599efe9867038"
        ),
        prepared_data_sha256=(
            "08d2a7f3a45b55ab5bf8f336bf0fd126046f5af69df69085570a4b418fdf89f2"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-07",
        n_nodes=12_155,
        manifest_sha256=(
            "8a2aa70ef58beec5d1ac6ef25ce9864563d72bda2c93f9bcb99f568c36a0b1c8"
        ),
        prepared_data_sha256=(
            "fcc58f91cae83507b73dd3127f53352d5f8c68628ca6239d6125a20129b2998a"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-08",
        n_nodes=14_506,
        manifest_sha256=(
            "8e5f899633992d89d0539ab60dfb46066441c0dc8f3ae564161e59a6d181a676"
        ),
        prepared_data_sha256=(
            "ac423a7afa9301e4dd145c2fe5d97071a3ad8991661cfd7e21a830e4557a6d88"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-09",
        n_nodes=11_696,
        manifest_sha256=(
            "ddf24329686ec926af812ed57c449c9cc7be7b30eab6706bb0a552fd7d4b66ff"
        ),
        prepared_data_sha256=(
            "d41f4d74383d8578a1508e601727e51e1eeb41230bf1120ff3465517eabd0743"
        ),
    ),
    PreparedCoreReceipt(
        alias="ANC-10",
        n_nodes=11_443,
        manifest_sha256=(
            "5182a1fcff688cb77ed54ea364184e81d9d0eeb5e67be1163b33cde855993ffa"
        ),
        prepared_data_sha256=(
            "4b5645ff2f07f2150f032e912cf6efdf665acd89e7d755df3d5ac6759f1bb593"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class PooledCoreChecksums:
    """Content fingerprints for one non-identifying pooled core record."""

    source_manifest_sha256: str
    source_prepared_data_sha256: str
    source_full_core_preprocessing_sha256: str
    expression_counts_sha256: str
    target_expression_sha256: str
    node_covariates_sha256: str
    coordinates_um_sha256: str
    macroblock_ids_sha256: str
    preprocessing_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PooledCoreQC:
    """Safe preprocessing diagnostics for one core."""

    alias: str
    n_nodes: int
    n_genes: int
    n_model_covariates: int
    morphology_fit_scope: str
    expression_fit_scope: str
    expression_moment_weighting: str
    protected_identifier_arrays_returned: bool
    all_outputs_finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PooledCoreData:
    """One complete core graph batch with a shared expression transform."""

    alias: str
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
    preprocessing_qc: PooledCoreQC
    checksums: PooledCoreChecksums

    @property
    def n_nodes(self) -> int:
        return int(self.expression_counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression_counts.shape[1])


@dataclass(frozen=True, slots=True)
class PooledCohortChecksums:
    """Ordered source and shared-transform fingerprints for the cohort."""

    ordered_sources_sha256: str
    ordered_gene_schema_sha256: str
    ordered_metadata_schema_sha256: str
    expression_mean_sha256: str
    expression_scale_sha256: str
    combined_fingerprint_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PooledFullCoreCohort:
    """Ten disconnected core batches sharing only learned model parameters."""

    cores: tuple[PooledCoreData, ...] = field(repr=False)
    aliases: tuple[str, ...]
    gene_names: tuple[str, ...]
    metadata_names: tuple[str, ...]
    expression_mean: np.ndarray = field(repr=False)
    expression_scale: np.ndarray = field(repr=False)
    total_nodes: int
    checksums: PooledCohortChecksums
    _cores_by_alias: Mapping[str, PooledCoreData] = field(
        repr=False,
        compare=False,
    )

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def fingerprint_sha256(self) -> str:
        return self.checksums.combined_fingerprint_sha256

    def core(self, alias: str) -> PooledCoreData:
        """Return one core by its permitted opaque alias."""

        try:
            return self._cores_by_alias[alias]
        except KeyError as exc:
            raise PooledFullCoreContractError(
                "Core lookup requires one of the frozen ANC aliases."
            ) from exc


@dataclass(frozen=True, slots=True)
class _LoadedCore:
    """Temporary identifier-safe core state without the obsolete target array."""

    receipt: PreparedCoreReceipt
    expression_counts: np.ndarray = field(repr=False)
    node_covariates: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    macroblock_ids: np.ndarray = field(repr=False)
    gene_names: tuple[str, ...]
    metadata_names: tuple[str, ...]
    metadata_median: np.ndarray = field(repr=False)
    metadata_mean: np.ndarray = field(repr=False)
    metadata_scale: np.ndarray = field(repr=False)
    metadata_missing_indicator_indices: np.ndarray = field(repr=False)
    source_checksums: _full_core.FullCorePreprocessingChecksums


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    values = np.asarray(array, dtype=dtype)
    values.setflags(write=False)
    return values


def _opaque_macroblock_ids(
    macroblock_ids: np.ndarray,
    *,
    alias: str,
) -> np.ndarray:
    """Preserve block membership without returning source FOV routing values."""

    source = np.asarray(macroblock_ids)
    if source.ndim != 1 or source.dtype.kind not in "US":
        raise PooledFullCoreContractError(
            f"{alias} spatial block labels have an invalid schema."
        )
    _, inverse = np.unique(source, return_inverse=True)
    width = max(4, len(str(int(inverse.max(initial=0)))))
    opaque = np.asarray(
        [
            f"{alias}-BLOCK-{int(block_index):0{width}d}"
            for block_index in inverse
        ],
        dtype=f"U{len(alias) + 7 + width}",
    )
    return _readonly(opaque)


def _validate_counts(counts: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(counts)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise PooledFullCoreContractError(
            f"{label} expression counts must be a nonempty matrix."
        )
    if values.dtype.kind not in "iuf" or values.dtype.kind == "b":
        raise PooledFullCoreContractError(
            f"{label} expression counts must be numeric integer values."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise PooledFullCoreContractError(
            f"{label} expression counts must be finite and nonnegative."
        )
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise PooledFullCoreContractError(
            f"{label} expression counts must be integer-valued."
        )
    return values


def fit_equal_core_log1p_statistics(
    expression_counts: Sequence[np.ndarray],
    *,
    epsilon: float = EXPRESSION_SCALE_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit float64 gene moments for an equally weighted mixture of cores.

    The first moment is the arithmetic mean of the within-core means.  The
    variance is recovered from the arithmetic mean of the within-core second
    moments, so core size never determines normalization weight.
    """

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise PooledFullCoreContractError("Expression epsilon must be positive.")
    if not expression_counts:
        raise PooledFullCoreContractError(
            "At least one core is required for pooled expression statistics."
        )
    means: list[np.ndarray] = []
    second_moments: list[np.ndarray] = []
    expected_genes: int | None = None
    for index, raw_counts in enumerate(expression_counts):
        counts = _validate_counts(raw_counts, label=f"Core {index + 1}")
        if expected_genes is None:
            expected_genes = int(counts.shape[1])
        elif counts.shape[1] != expected_genes:
            raise PooledFullCoreContractError(
                "Core expression matrices have inconsistent ordered widths."
            )
        transformed = np.empty(counts.shape, dtype=np.float64)
        np.log1p(counts, out=transformed)
        means.append(transformed.mean(axis=0, dtype=np.float64))
        np.square(transformed, out=transformed)
        second_moments.append(transformed.mean(axis=0, dtype=np.float64))

    mean = np.mean(np.stack(means, axis=0), axis=0, dtype=np.float64)
    second = np.mean(
        np.stack(second_moments, axis=0),
        axis=0,
        dtype=np.float64,
    )
    variance = np.maximum(second - np.square(mean), 0.0)
    raw_scale = np.sqrt(variance)
    scale = np.where(raw_scale > epsilon, raw_scale, 1.0)
    if not np.isfinite(mean).all() or (
        not np.isfinite(scale).all() or np.any(scale <= 0)
    ):
        raise PooledFullCoreContractError(
            "Pooled expression statistics are not finite and positive."
        )
    return _readonly(mean, dtype=np.float64), _readonly(scale, dtype=np.float64)


def _standardize_counts(
    counts: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    row_chunk_size: int = 2_048,
) -> np.ndarray:
    if row_chunk_size <= 0:
        raise PooledFullCoreContractError(
            "Expression standardization chunk size must be positive."
        )
    output = np.empty(counts.shape, dtype=np.float32)
    for start in range(0, len(counts), row_chunk_size):
        stop = min(start + row_chunk_size, len(counts))
        transformed = np.empty((stop - start, counts.shape[1]), dtype=np.float64)
        np.log1p(counts[start:stop], out=transformed)
        transformed -= mean
        transformed /= scale
        output[start:stop] = transformed
    if not np.isfinite(output).all():
        raise PooledFullCoreContractError(
            "Shared expression standardization produced non-finite values."
        )
    return _readonly(output)


def _normalise_artifact_inputs(
    prepared_artifacts: (
        Mapping[str, str | Path]
        | Sequence[tuple[str, str | Path]]
        | None
    ),
    *,
    data_root: str | Path | None,
) -> tuple[tuple[str, Path], ...]:
    if prepared_artifacts is None:
        resolved_data_root = (
            Path(data_root)
            if data_root is not None
            else current_paths().data_root
        )
        return tuple(
            (
                alias,
                resolved_data_root
                / _PREPARED_RELATIVE_ROOT
                / alias.lower()
                / "prepared_v1",
            )
            for alias in ANC_ALIASES
        )
    if data_root is not None:
        raise PooledFullCoreContractError(
            "data_root cannot be combined with explicit prepared artifacts."
        )
    items = (
        tuple(prepared_artifacts.items())
        if isinstance(prepared_artifacts, Mapping)
        else tuple(prepared_artifacts)
    )
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        for item in items
    ):
        raise PooledFullCoreContractError(
            "Prepared artifacts must be ordered (ANC alias, path) pairs."
        )
    aliases = tuple(item[0] for item in items)
    if len(set(aliases)) != len(aliases):
        raise PooledFullCoreContractError(
            "Prepared artifact aliases contain a duplicate."
        )
    if set(aliases) != set(ANC_ALIASES):
        raise PooledFullCoreContractError(
            "Prepared artifacts must contain every frozen ANC alias exactly once."
        )
    if aliases != ANC_ALIASES:
        raise PooledFullCoreContractError(
            "Prepared artifact aliases do not match the frozen order."
        )
    return tuple((alias, Path(path)) for alias, path in items)


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    receipt: PreparedCoreReceipt,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if manifest.get("artifact_kind") != _PREPARED_ARTIFACT_KIND:
        raise PooledFullCoreContractError(
            f"{receipt.alias} has the wrong prepared artifact kind."
        )
    if manifest.get("manifest_content_sha256") != receipt.manifest_sha256 or (
        manifest.get("artifact_id") != receipt.manifest_sha256[:16]
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} does not match its frozen manifest checksum."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or (
        files.get("prepared_data.npz") != receipt.prepared_data_sha256
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} does not match its frozen prepared-data checksum."
        )
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise PooledFullCoreContractError(
            f"{receipt.alias} lacks an identifier-safe selection record."
        )
    if (
        selection.get("opaque_alias") != receipt.alias
        or selection.get("restricted_identifiers_emitted") is not False
        or selection.get("tissue_context") != _TISSUE_CONTEXT
        or selection.get("n_cells") != receipt.n_nodes
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} selection metadata violates the frozen contract."
        )

    features = manifest.get("features")
    if not isinstance(features, Mapping):
        raise PooledFullCoreContractError(
            f"{receipt.alias} lacks its ordered feature schema."
        )
    gene_names = tuple(features.get("gene_names", ()))
    measured_names = tuple(features.get("measured_metadata_names", ()))
    model_names = tuple(features.get("model_covariate_names", ()))
    if (
        len(gene_names) != EXPECTED_N_GENES
        or len(set(gene_names)) != len(gene_names)
        or any(
            not isinstance(name, str)
            or not name
            or name.startswith(_TECHNICAL_CONTROL_PREFIXES)
            for name in gene_names
        )
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} has an invalid ordered gene schema."
        )
    if measured_names != ALLOWED_METADATA_COLUMNS or (
        model_names != ALLOWED_METADATA_COLUMNS
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} has an invalid ordered covariate schema."
        )
    if (
        features.get("n_biological_probes") != EXPECTED_N_GENES
        or tuple(features.get("technical_control_prefixes_excluded", ()))
        != _TECHNICAL_CONTROL_PREFIXES
        or features.get("coordinates_are_model_covariates") is not False
        or features.get("routing_keys_are_model_covariates") is not False
        or features.get("qc_indicator_is_model_covariate") is not False
    ):
        raise PooledFullCoreContractError(
            f"{receipt.alias} feature policy violates the frozen contract."
        )
    return tuple(str(name) for name in gene_names), tuple(
        str(name) for name in model_names
    )


def _load_verified_core(
    alias: str,
    path: Path,
    *,
    receipt: PreparedCoreReceipt,
    epsilon: float,
) -> tuple[_LoadedCore, tuple[str, ...], tuple[str, ...]]:
    try:
        manifest, _, _ = load_prepared_artifact(path, load_arrays=False)
        gene_names, metadata_names = _validate_manifest(
            manifest,
            receipt=receipt,
        )
        source = _full_core.load_and_refit_full_core(path, epsilon=epsilon)
    except FileNotFoundError:
        raise PooledFullCoreContractError(
            f"Prepared artifact for {alias} was not found."
        ) from None
    except ArtifactContractError:
        # Artifact verifier messages name only relative artifact members.  Do
        # not add the caller-supplied path to the public error.
        raise

    counts = _validate_counts(source.expression_counts, label=alias)
    if (
        source.n_nodes != receipt.n_nodes
        or source.n_genes != EXPECTED_N_GENES
        or source.gene_names != gene_names
        or source.metadata_names != metadata_names
        or source.node_covariates.shape
        != (receipt.n_nodes, EXPECTED_N_MODEL_COVARIATES)
    ):
        raise PooledFullCoreContractError(
            f"{alias} loaded arrays do not match the frozen ordered schema."
        )
    if source.checksums.source_prepared_data_sha256 != (
        receipt.prepared_data_sha256
    ):
        raise PooledFullCoreContractError(
            f"{alias} loaded content differs from its frozen checksum."
        )
    if source.preprocessing_qc.protected_identifier_arrays_returned is not False:
        raise PooledFullCoreContractError(
            f"{alias} preprocessing exposed a prohibited identifier array."
        )
    opaque_macroblocks = _opaque_macroblock_ids(
        source.macroblock_ids,
        alias=alias,
    )
    loaded = _LoadedCore(
        receipt=receipt,
        expression_counts=_readonly(counts),
        node_covariates=_readonly(source.node_covariates),
        coordinates_um=_readonly(source.coordinates_um, dtype=np.float64),
        macroblock_ids=opaque_macroblocks,
        gene_names=source.gene_names,
        metadata_names=source.metadata_names,
        metadata_median=_readonly(source.metadata_median, dtype=np.float64),
        metadata_mean=_readonly(source.metadata_mean, dtype=np.float64),
        metadata_scale=_readonly(source.metadata_scale, dtype=np.float64),
        metadata_missing_indicator_indices=_readonly(
            source.metadata_missing_indicator_indices,
            dtype=np.int64,
        ),
        source_checksums=source.checksums,
    )
    return loaded, gene_names, metadata_names


def load_pooled_full_core_cohort(
    prepared_artifacts: (
        Mapping[str, str | Path]
        | Sequence[tuple[str, str | Path]]
        | None
    ) = None,
    *,
    data_root: str | Path | None = None,
    epsilon: float = EXPRESSION_SCALE_EPSILON,
) -> PooledFullCoreCohort:
    """Load the exact ten-core cohort and fit its shared expression transform.

    Cores remain separate complete graph batches.  The expression mean and
    scale are shared, but morphology/imaging values and their fitted
    statistics remain exactly those from each core's existing all-fit
    transductive preprocessing.
    """

    inputs = _normalise_artifact_inputs(
        prepared_artifacts,
        data_root=data_root,
    )
    receipts = {receipt.alias: receipt for receipt in EXPECTED_PREPARED_CORES}
    if tuple(receipts) != ANC_ALIASES or sum(
        receipt.n_nodes for receipt in EXPECTED_PREPARED_CORES
    ) != EXPECTED_TOTAL_NODES:
        raise PooledFullCoreContractError(
            "Internal frozen prepared-core receipts are inconsistent."
        )

    loaded_cores: list[_LoadedCore] = []
    common_genes: tuple[str, ...] | None = None
    common_metadata: tuple[str, ...] | None = None
    for alias, path in inputs:
        loaded, genes, metadata = _load_verified_core(
            alias,
            path,
            receipt=receipts[alias],
            epsilon=epsilon,
        )
        if common_genes is None:
            common_genes = genes
            common_metadata = metadata
        elif genes != common_genes or metadata != common_metadata:
            raise PooledFullCoreContractError(
                f"{alias} ordered schema differs from the preceding cores."
            )
        loaded_cores.append(loaded)

    if common_genes is None or common_metadata is None:
        raise PooledFullCoreContractError("The pooled cohort is empty.")
    total_nodes = sum(len(core.expression_counts) for core in loaded_cores)
    if total_nodes != EXPECTED_TOTAL_NODES:
        raise PooledFullCoreContractError(
            "Pooled core cell counts do not match the frozen total."
        )
    expression_mean, expression_scale = fit_equal_core_log1p_statistics(
        [core.expression_counts for core in loaded_cores],
        epsilon=epsilon,
    )
    expression_mean_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_mean",
        expression_mean,
    )
    expression_scale_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_scale",
        expression_scale,
    )

    pooled_cores: list[PooledCoreData] = []
    for loaded in loaded_cores:
        target = _standardize_counts(
            loaded.expression_counts,
            mean=expression_mean,
            scale=expression_scale,
        )
        target_sha = _full_core._array_sha256(  # noqa: SLF001
            "target_expression",
            target,
        )
        macroblock_sha = _full_core._array_sha256(  # noqa: SLF001
            "macroblock_ids",
            loaded.macroblock_ids,
        )
        final_payload = {
            "schema": "pooled_full_core_v1",
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "alias": loaded.receipt.alias,
            "n_nodes": loaded.receipt.n_nodes,
            "gene_names": list(common_genes),
            "metadata_names": list(common_metadata),
            "expression_fit_scope": "all ten cores (transductive)",
            "expression_weighting": "equal core mixture",
            "morphology_fit_scope": "each core independently (transductive)",
            "source_manifest_sha256": loaded.receipt.manifest_sha256,
            "source_prepared_data_sha256": (
                loaded.receipt.prepared_data_sha256
            ),
            "source_full_core_preprocessing_sha256": (
                loaded.source_checksums.preprocessing_sha256
            ),
            "component_checksums": {
                "expression_counts": (
                    loaded.source_checksums.expression_counts_sha256
                ),
                "target_expression": target_sha,
                "node_covariates": (
                    loaded.source_checksums.node_covariates_sha256
                ),
                "coordinates_um": (
                    loaded.source_checksums.coordinates_um_sha256
                ),
                "macroblock_ids": (
                    macroblock_sha
                ),
                "expression_mean": expression_mean_sha,
                "expression_scale": expression_scale_sha,
            },
        }
        checksums = PooledCoreChecksums(
            source_manifest_sha256=loaded.receipt.manifest_sha256,
            source_prepared_data_sha256=(
                loaded.receipt.prepared_data_sha256
            ),
            source_full_core_preprocessing_sha256=(
                loaded.source_checksums.preprocessing_sha256
            ),
            expression_counts_sha256=(
                loaded.source_checksums.expression_counts_sha256
            ),
            target_expression_sha256=target_sha,
            node_covariates_sha256=(
                loaded.source_checksums.node_covariates_sha256
            ),
            coordinates_um_sha256=(
                loaded.source_checksums.coordinates_um_sha256
            ),
            macroblock_ids_sha256=(
                macroblock_sha
            ),
            preprocessing_sha256=_canonical_sha256(final_payload),
        )
        qc = PooledCoreQC(
            alias=loaded.receipt.alias,
            n_nodes=loaded.receipt.n_nodes,
            n_genes=len(common_genes),
            n_model_covariates=loaded.node_covariates.shape[1],
            morphology_fit_scope="each core independently (transductive)",
            expression_fit_scope="all ten cores (transductive)",
            expression_moment_weighting="equal core mixture",
            protected_identifier_arrays_returned=False,
            all_outputs_finite=bool(
                np.isfinite(target).all()
                and np.isfinite(loaded.node_covariates).all()
                and np.isfinite(loaded.coordinates_um).all()
            ),
        )
        if not qc.all_outputs_finite:
            raise PooledFullCoreContractError(
                f"{loaded.receipt.alias} pooled output is non-finite."
            )
        pooled_cores.append(
            PooledCoreData(
                alias=loaded.receipt.alias,
                expression_counts=loaded.expression_counts,
                target_expression=target,
                node_covariates=loaded.node_covariates,
                coordinates_um=loaded.coordinates_um,
                macroblock_ids=loaded.macroblock_ids,
                gene_names=common_genes,
                metadata_names=common_metadata,
                expression_mean=expression_mean,
                expression_scale=expression_scale,
                metadata_median=loaded.metadata_median,
                metadata_mean=loaded.metadata_mean,
                metadata_scale=loaded.metadata_scale,
                metadata_missing_indicator_indices=(
                    loaded.metadata_missing_indicator_indices
                ),
                preprocessing_qc=qc,
                checksums=checksums,
            )
        )

    ordered_sources_payload = [
        {
            "alias": receipt.alias,
            "n_nodes": receipt.n_nodes,
            "manifest_sha256": receipt.manifest_sha256,
            "prepared_data_sha256": receipt.prepared_data_sha256,
        }
        for receipt in EXPECTED_PREPARED_CORES
    ]
    ordered_sources_sha = _canonical_sha256(ordered_sources_payload)
    gene_schema_sha = _canonical_sha256(list(common_genes))
    metadata_schema_sha = _canonical_sha256(list(common_metadata))
    cohort_payload = {
        "schema": "pooled_full_core_cohort_v1",
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "aliases": list(ANC_ALIASES),
        "total_nodes": total_nodes,
        "ordered_sources_sha256": ordered_sources_sha,
        "ordered_gene_schema_sha256": gene_schema_sha,
        "ordered_metadata_schema_sha256": metadata_schema_sha,
        "expression_mean_sha256": expression_mean_sha,
        "expression_scale_sha256": expression_scale_sha,
        "core_preprocessing_sha256": [
            core.checksums.preprocessing_sha256 for core in pooled_cores
        ],
    }
    cohort_checksums = PooledCohortChecksums(
        ordered_sources_sha256=ordered_sources_sha,
        ordered_gene_schema_sha256=gene_schema_sha,
        ordered_metadata_schema_sha256=metadata_schema_sha,
        expression_mean_sha256=expression_mean_sha,
        expression_scale_sha256=expression_scale_sha,
        combined_fingerprint_sha256=_canonical_sha256(cohort_payload),
    )
    core_tuple = tuple(pooled_cores)
    return PooledFullCoreCohort(
        cores=core_tuple,
        aliases=ANC_ALIASES,
        gene_names=common_genes,
        metadata_names=common_metadata,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        total_nodes=total_nodes,
        checksums=cohort_checksums,
        _cores_by_alias=MappingProxyType(
            {core.alias: core for core in core_tuple}
        ),
    )


# Compact compatibility spelling for callers that do not need the longer
# campaign-specific name.
load_pooled_full_core = load_pooled_full_core_cohort


__all__ = [
    "ANC_ALIASES",
    "EXPECTED_N_GENES",
    "EXPECTED_N_MODEL_COVARIATES",
    "EXPECTED_PREPARED_CORES",
    "EXPECTED_TOTAL_NODES",
    "EXPRESSION_SCALE_EPSILON",
    "FROZEN_CONTRACT_SHA256",
    "PooledCohortChecksums",
    "PooledCoreChecksums",
    "PooledCoreData",
    "PooledCoreQC",
    "PooledFullCoreCohort",
    "PooledFullCoreContractError",
    "PreparedCoreReceipt",
    "fit_equal_core_log1p_statistics",
    "load_pooled_full_core",
    "load_pooled_full_core_cohort",
]

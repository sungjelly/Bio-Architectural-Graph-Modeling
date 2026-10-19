"""Small, target-isolated SO2 inputs without loading the historical graph cache.

Only the immutable count/covariate/coordinate arrays and the existing train-only
normalization overlay are consumed.  Local context is a one-hop summary of clean
non-self neighbors; changing a target's expression cannot change its own context.
This is partial-gene reconstruction, not whole-region expression imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .fingerprints import sha256_file
from .paths import ProjectPaths, current_paths
from . import so2_nb_data as nb
from .so2_pooled_full_core import EXPECTED_CELL_COUNTS_BY_CORE, EXPECTED_N_GENES
from .so2_relative_graphs import _verify_so2_cohort_manifest
from .target_cell_masking import (
    TargetCellMaskRealization,
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
)


MASK_BASE_SEED = 2_026_092_601
TRAINING_MASK_NAMESPACE = "bagm.so2.target_isolated_nb.training_masks.v1"
VALIDATION_MASK_NAMESPACE = "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"
OVERLAY_RELATIVE = Path("processed/so2_nb_train12_val2_v1")
# The exact historical overlay, as registered in so2_nb_train12_val2_v1.yaml.
EXPECTED_OVERLAY_MANIFEST_SHA256 = "3aea6cfbf7191df221cf226316dbbc2aded6edfaeb193498c7a7a740bc73cabe"
EXPECTED_OVERLAY_CONTENT_SHA256 = "206148ffc8976526118932c7fbed82674f79ff8f2326bbd2a7c54dba8540260c"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LocalGeneCore:
    """Aligned numeric model inputs; aliases are provenance, never features."""

    alias: str
    role: str
    counts: np.ndarray = field(repr=False)
    expression: np.ndarray = field(repr=False)
    covariates: np.ndarray = field(repr=False)
    local_log_mean: np.ndarray = field(repr=False)
    local_positive_fraction: np.ndarray = field(repr=False)
    density: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in {"train", "validation"}:
            raise ValueError("Core role must be train or validation.")
        counts = np.asarray(self.counts)
        if (counts.dtype != np.int32 or counts.ndim != 2
                or min(counts.shape) < 1 or np.any(counts < 0)):
            raise ValueError("Counts must be nonnegative int32 [cells, genes].")
        for name in ("expression", "covariates", "local_log_mean",
                     "local_positive_fraction", "density"):
            array = np.asarray(getattr(self, name))
            expected = counts.shape
            if name == "covariates":
                expected = (len(counts), array.shape[1] if array.ndim == 2 else -1)
            elif name == "density":
                expected = (len(counts), 1)
            if array.dtype != np.float32 or array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"Invalid aligned float32 input: {name}.")
            _readonly(array)
        if np.any(self.local_log_mean < 0) or np.any(self.density < 0):
            raise ValueError("Local means and density must be nonnegative.")
        if np.any(self.local_positive_fraction < 0) or np.any(self.local_positive_fraction > 1):
            raise ValueError("Local positive fractions must be in [0, 1].")
        _readonly(counts)

    @property
    def n_cells(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[1])

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in (
            "counts", "expression", "covariates", "local_log_mean",
            "local_positive_fraction", "density"))


@dataclass(frozen=True, slots=True)
class LocalGeneData:
    training: tuple[LocalGeneCore, ...]
    validation: tuple[LocalGeneCore, ...]
    prevalence: np.ndarray = field(repr=False)
    positive_mean: np.ndarray = field(repr=False)
    statistics: nb.SO2NBPreprocessingStatistics = field(repr=False)
    provenance: Mapping[str, Any]


def local_gene_context(
    counts: np.ndarray,
    coordinates_um: np.ndarray,
    *,
    neighbors: int = 16,
    radius_um: float = 75.0,
    row_chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log1p(raw-count mean), detection fraction, and physical density.

    The first two arrays use at most ``neighbors`` nearest other cells within
    the inclusive radius. Density is the number of *all* non-self cells inside
    the radius, stored as float32 [N,1]; it is not the truncated neighbor degree.
    Empty neighborhoods have three exact zeros. Coordinates are already in um;
    no pixel conversion, expression-dependent edge choice, or fit is performed.
    """

    raw = np.asarray(counts)
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if (raw.dtype != np.int32 or raw.ndim != 2 or min(raw.shape) < 1
            or np.any(raw < 0)):
        raise ValueError("Counts must be nonnegative int32 [cells, genes].")
    if coordinates.shape != (len(raw), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite [cells, 2] micrometers.")
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer.")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be finite and positive.")
    if isinstance(row_chunk_size, bool) or not isinstance(row_chunk_size, int) or row_chunk_size < 1:
        raise ValueError("row_chunk_size must be a positive integer.")
    tree = cKDTree(coordinates)
    # nextafter includes cells on the specified physical boundary in query().
    distance, source = tree.query(
        coordinates, k=min(neighbors + 1, len(raw)),
        distance_upper_bound=np.nextafter(float(radius_um), np.inf), workers=1,
    )
    if distance.ndim == 1:
        distance, source = distance[:, None], source[:, None]
    valid = (source != np.arange(len(raw))[:, None]) & (source < len(raw))
    valid &= np.isfinite(distance) & (distance <= radius_um)
    # If coincident coordinates cause query() to omit self, still cap at K.
    valid &= np.cumsum(valid, axis=1) <= neighbors
    degree = valid.sum(axis=1, dtype=np.int64)
    density = np.maximum(
        tree.query_ball_point(coordinates, r=radius_um, return_length=True, workers=1) - 1,
        0,
    ).astype(np.float32)[:, None]
    log_mean = np.zeros(raw.shape, dtype=np.float32)
    positive_fraction = np.zeros(raw.shape, dtype=np.float32)
    for start in range(0, len(raw), row_chunk_size):
        stop = min(start + row_chunk_size, len(raw))
        total = np.zeros((stop - start, raw.shape[1]), dtype=np.float64)
        positive = np.zeros_like(total)
        for column in range(source.shape[1]):
            selected = np.flatnonzero(valid[start:stop, column])
            if not len(selected):
                continue
            values = raw[source[start:stop, column][selected]]
            total[selected] += values
            positive[selected] += values > 0
        denominator = np.maximum(degree[start:stop], 1)[:, None]
        log_mean[start:stop] = np.log1p(total / denominator)
        positive_fraction[start:stop] = positive / denominator
    return log_mean, positive_fraction, density


def training_gene_initialization(
    training: Sequence[LocalGeneCore],
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-weighted training prevalence and E[count | positive], no refitting.

    These are count-head initialization/baseline statistics, separate from the
    inherited equal-core input normalization. Never inspect validation rows.
    A gene with no training positives gets positive mean 1 and prevalence 0.
    """

    if not training or any(core.role != "train" for core in training):
        raise ValueError("Initialization accepts nonempty training cores only.")
    genes = training[0].n_genes
    total = np.zeros(genes, dtype=np.float64)
    positive = np.zeros(genes, dtype=np.int64)
    cells = 0
    for core in training:
        if core.n_genes != genes:
            raise ValueError("Training gene widths differ.")
        total += core.counts.sum(axis=0, dtype=np.float64)
        positive += np.count_nonzero(core.counts, axis=0)
        cells += core.n_cells
    prevalence = (positive / cells).astype(np.float32)
    positive_mean = np.divide(total, positive, out=np.ones(genes), where=positive > 0).astype(np.float32)
    return _readonly(prevalence), _readonly(positive_mean)


def make_mask(core: LocalGeneCore, epoch: int | None = None) -> TargetCellMaskRealization:
    """Exactly reuse the once-per-target BAGM NB seed and sampling protocol."""

    if core.role == "validation":
        if epoch is not None:
            raise ValueError("Validation masks are fixed; epoch must be None.")
        return make_validation_target_cell_masks(
            core.n_cells, core.n_genes, namespace=VALIDATION_MASK_NAMESPACE,
            core_alias=core.alias,
        )
    if epoch is None:
        raise ValueError("Training masks require the global epoch.")
    return make_training_target_cell_masks(
        core.n_cells, core.n_genes, base_seed=MASK_BASE_SEED,
        namespace=TRAINING_MASK_NAMESPACE, core_alias=core.alias,
        global_epoch=epoch,
    )


def _verify_overlay(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any], nb.SO2NBPreprocessingStatistics]:
    manifest_path = root / "manifest.json"
    if (stat.S_IMODE(root.stat().st_mode) != nb.DIRECTORY_MODE
            or stat.S_IMODE(manifest_path.stat().st_mode) != nb.FILE_MODE):
        raise nb.SO2NBDataContractError("SO2 overlay permissions changed.")
    if sha256_file(manifest_path) != EXPECTED_OVERLAY_MANIFEST_SHA256:
        raise nb.SO2NBDataContractError("SO2 overlay differs from the frozen baseline artifact.")
    manifest = nb._strict_json(manifest_path)
    unsigned = dict(manifest)
    content_sha = unsigned.pop("manifest_content_sha256", None)
    if (content_sha != EXPECTED_OVERLAY_CONTENT_SHA256
            or content_sha != nb._canonical_sha256(unsigned)):
        raise nb.SO2NBDataContractError("SO2 overlay content checksum changed.")
    if manifest.get("artifact_kind") != nb.OVERLAY_SCHEMA or manifest.get("campaign_id") != nb.CAMPAIGN_ID:
        raise nb.SO2NBDataContractError("SO2 overlay identity changed.")
    nb._validate_overlay_semantics(root, manifest)
    split = manifest.get("split")
    expected_split = nb._split_payload()
    if (not isinstance(split, Mapping)
            or {key: split.get(key) for key in expected_split} != expected_split
            or split.get("split_fingerprint") != nb._canonical_sha256(expected_split)):
        raise nb.SO2NBDataContractError("SO2 overlay split changed.")
    source = manifest.get("source_artifacts", {}).get("cohort")
    if not isinstance(source, Mapping):
        raise nb.SO2NBDataContractError("SO2 cohort reference missing.")
    cohort_root = nb._resolve_source_root(root, source, None)
    if sha256_file(cohort_root / "manifest.json") != source.get("manifest_file_sha256"):
        raise nb.SO2NBDataContractError("SO2 cohort manifest file checksum changed.")
    # This helper hashes only the small cohort npz files, never graph arrays.
    cohort = _verify_so2_cohort_manifest(cohort_root)
    cohort_unsigned = dict(cohort)
    cohort_sha = cohort_unsigned.pop("manifest_content_sha256", None)
    if (cohort_sha != nb._canonical_sha256(cohort_unsigned)
            or cohort_sha != source.get("manifest_content_sha256")):
        raise nb.SO2NBDataContractError("SO2 cohort content checksum changed.")
    return manifest, cohort_root, cohort, nb._statistics_from_overlay(root, manifest)


def load_local_gene_data(
    *,
    paths: ProjectPaths | None = None,
    overlay_dir: str | Path | None = None,
    neighbors: int = 16,
    radius_um: float = 75.0,
) -> LocalGeneData:
    """Verify and load SO2 data read-only; never touch historical graph files.

    The optional overlay path resolves under ``paths.data_root`` if relative.
    This loader makes no cache or shared-data writes. Dense arrays cost 16 bytes
    per cell/gene (about 3.94 GB for this cohort), plus small covariate arrays.
    """

    paths = current_paths() if paths is None else paths
    root = Path(overlay_dir) if overlay_dir is not None else OVERLAY_RELATIVE
    root = (root if root.is_absolute() else paths.data_root / root).resolve()
    manifest, cohort_root, cohort, statistics = _verify_overlay(root)
    cores: list[LocalGeneCore] = []
    for alias in nb.SO2_NB_TRAINING_ALIASES + nb.SO2_NB_VALIDATION_ALIASES:
        with np.load(cohort_root / "cores" / f"{alias}.npz", allow_pickle=False) as payload:
            if set(payload.files) != {"expression_counts", "target_expression", "node_covariates", "coordinates_um"}:
                raise nb.SO2NBDataContractError("SO2 source core array schema changed.")
            counts = payload["expression_counts"]
            covariates = payload["node_covariates"]
            coordinates = payload["coordinates_um"]
        nodes = EXPECTED_CELL_COUNTS_BY_CORE[int(alias[-2:])]
        if (counts.dtype != np.int32 or counts.shape != (nodes, EXPECTED_N_GENES)
                or covariates.dtype != np.float32
                or covariates.shape != (nodes, nb.EXPECTED_NODE_COVARIATES)
                or coordinates.shape != (nodes, 2)):
            raise nb.SO2NBDataContractError("SO2 source core alignment changed.")
        log_mean, positive_fraction, density = local_gene_context(
            counts, coordinates, neighbors=neighbors, radius_um=radius_um,
        )
        cores.append(LocalGeneCore(
            alias=alias, role="train" if alias in nb.SO2_NB_TRAINING_ALIASES else "validation",
            counts=counts, expression=nb.standardize_so2_nb_expression_input(counts, statistics),
            covariates=nb.recenter_so2_nb_covariates(covariates, statistics),
            local_log_mean=log_mean, local_positive_fraction=positive_fraction, density=density,
        ))
    training = tuple(core for core in cores if core.role == "train")
    validation = tuple(core for core in cores if core.role == "validation")
    prevalence, positive_mean = training_gene_initialization(training)
    provenance = {
        "schema": "local_gene_so2_data_v1",
        "overlay_manifest_sha256": sha256_file(root / "manifest.json"),
        "overlay_manifest_content_sha256": manifest["manifest_content_sha256"],
        "split_fingerprint": manifest["split"]["split_fingerprint"],
        "preprocessing_fingerprint": statistics.fingerprint,
        "source_cohort_manifest_sha256": sha256_file(cohort_root / "manifest.json"),
        "source_cohort_manifest_content_sha256": cohort["manifest_content_sha256"],
        "source_core_checksums": {alias: cohort["files"][f"cores/{alias}.npz"] for alias in nb.SO2_NB_TRAINING_ALIASES + nb.SO2_NB_VALIDATION_ALIASES},
        "source_core_checksums_verified": True,
        "historical_graph_cache_consumed": False,
        "input_preprocessing_refitted": False,
        "local_neighbors": neighbors,
        "local_radius_um": radius_um,
        "local_pooling": "log1p_raw_count_mean_and_positive_fraction",
        "local_density": "all_nonself_cells_within_physical_radius",
        "local_self_exclusion": True,
        "initialization_fit_scope": "training_cores_only_cell_weighted",
        "mask_base_seed": MASK_BASE_SEED,
        "training_mask_namespace": TRAINING_MASK_NAMESPACE,
        "validation_mask_namespace": VALIDATION_MASK_NAMESPACE,
        "training_cells": sum(core.n_cells for core in training),
        "validation_cells": sum(core.n_cells for core in validation),
        "array_bytes": sum(core.nbytes for core in cores),
    }
    return LocalGeneData(training, validation, prevalence, positive_mean, statistics, provenance)


__all__ = [
    "LocalGeneCore", "LocalGeneData", "local_gene_context", "load_local_gene_data",
    "make_mask", "training_gene_initialization", "MASK_BASE_SEED",
    "TRAINING_MASK_NAMESPACE", "VALIDATION_MASK_NAMESPACE",
]

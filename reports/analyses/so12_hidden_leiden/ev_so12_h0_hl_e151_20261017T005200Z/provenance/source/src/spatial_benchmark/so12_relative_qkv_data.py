"""Frozen SO1/SO2 Pearson targets with masking-safe, visible-count inputs.

The exported Pearson targets use fully observed totals and are never passed to
the model as inputs.  Every input view is recomputed from visible raw counts.
This module reads existing sources only; it does not create a dataset or graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .data import ALLOWED_METADATA_COLUMNS
from .paths import ProjectPaths, current_paths
from .relative_geometry import _array_sha256 as _geometry_array_sha256
from .splits import TrainOnlyPreprocessor


PREPROCESSING_RUN_ID = "r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410"
TRAIN_CORE_NUMBERS = tuple((*range(1, 11), *range(15, 25)))
VALIDATION_CORE_NUMBERS = (11, 12, 13, 14, 25, 26, 27, 28)
EXPECTED_CELLS = 407_659
EXPECTED_TRAIN_CELLS = 301_198
EXPECTED_VALIDATION_CELLS = 106_461
N_GENES = 1_000
PIXEL_SIZE_UM = 0.120281


class SO12DataContractError(ValueError):
    """A public, identifier-free description of a failed data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SO12DataContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"Expected a manifest mapping: {path.name}")
    return value


def _check_file(path: Path, expected: str, verify: bool) -> dict[str, Any]:
    _require(path.is_file(), f"Required source is absent: {path.name}")
    _require(isinstance(expected, str) and len(expected) == 64,
             f"Source lacks a checksum: {path.name}")
    if verify:
        _require(_sha256(path) == expected, f"Source checksum mismatch: {path.name}")
    return {"path": str(path), "sha256": expected,
            "checksum_verified": verify, "size_bytes": path.stat().st_size}


def _check_content_manifest(manifest: Mapping[str, Any], label: str) -> None:
    content = dict(manifest)
    expected = content.pop("manifest_content_sha256", None)
    _require(expected == _canonical_sha256(content),
             f"{label} manifest content checksum mismatch.")


@dataclass(slots=True)
class SO12Core:
    alias: str
    slide: int
    donor_group: int
    split: str
    raw_counts: np.ndarray = field(repr=False)
    target_expression: np.ndarray = field(repr=False)
    node_covariates: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    edge_index: np.ndarray | None = field(default=None, repr=False)
    relative_geometry: np.ndarray | None = field(default=None, repr=False)
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.raw_counts.shape[0])


@dataclass(slots=True)
class SO12Dataset:
    cores: tuple[SO12Core, ...]
    genes: tuple[str, ...]
    gene_p: np.ndarray = field(repr=False)
    residual_mean: np.ndarray = field(repr=False)
    residual_sd: np.ndarray = field(repr=False)
    target_mean: np.ndarray = field(repr=False)
    metadata_statistics: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def train_cores(self) -> tuple[SO12Core, ...]:
        return tuple(core for core in self.cores if core.split == "train")

    @property
    def validation_cores(self) -> tuple[SO12Core, ...]:
        return tuple(core for core in self.cores if core.split == "validation")


def visible_count_pearson_input(
    raw_counts: torch.Tensor,
    gene_mask: torch.Tensor,
    *,
    gene_p: torch.Tensor | np.ndarray,
    residual_mean: torch.Tensor | np.ndarray,
    residual_sd: torch.Tensor | np.ndarray,
    phi: float = 1.01,
    upper_clip: float = 10.0,
) -> torch.Tensor:
    """Return FP32 Pearson inputs depending only on unmasked raw counts.

    True entries in ``gene_mask`` are hidden.  Fully hidden cells return zero.
    Exposure is the visible count sum divided by visible training frequency
    mass, with a minimum of one.  The target's full-cell total is never used.
    The operation also supports autograd; hidden counts have zero derivative.
    """
    _require(isinstance(raw_counts, torch.Tensor) and raw_counts.ndim == 2,
             "Raw counts must be a two-dimensional tensor.")
    _require(isinstance(gene_mask, torch.Tensor) and gene_mask.dtype == torch.bool
             and gene_mask.shape == raw_counts.shape
             and gene_mask.device == raw_counts.device,
             "A same-device boolean mask must align with raw counts.")
    _require(np.isfinite(phi) and phi > 0 and np.isfinite(upper_clip),
             "Pearson transform constants must be finite, with positive phi.")
    # Disabling autocast here prevents a mixed-precision caller from reducing
    # the normalization's precision or the reliability of mask invariance.
    with torch.autocast(device_type=raw_counts.device.type, enabled=False):
        counts = raw_counts.float()
        parameters = [torch.as_tensor(value, device=counts.device, dtype=torch.float32)
                      for value in (gene_p, residual_mean, residual_sd)]
        p, mean, sd = parameters
        _require(all(value.shape == (counts.shape[1],) for value in parameters),
                 "Pearson parameter vectors must match the gene dimension.")
        _require(all(bool(torch.isfinite(value).all()) for value in parameters)
                 and bool((p > 0).all()) and bool((sd > 0).all()),
                 "Pearson parameters must be finite with positive frequencies/scales.")
        _require(bool(torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-5, rtol=1e-5)),
                 "Training gene frequencies must sum to one.")
        visible_counts = torch.where(gene_mask, 0.0, counts)
        _require(bool(torch.isfinite(visible_counts).all())
                 and bool((visible_counts >= 0).all()),
                 "Visible counts must be finite and nonnegative.")
        visible_mass = torch.where(gene_mask, 0.0, p.unsqueeze(0)).sum(dim=1)
        safe_mass = torch.where(visible_mass > 0, visible_mass, 1.0)
        exposure = (visible_counts.sum(dim=1) / safe_mass).clamp_min(1.0)
        mu = exposure.unsqueeze(1) * p.unsqueeze(0)
        residual = (visible_counts - mu) / torch.sqrt(phi * mu)
        standardized = ((residual - mean) / sd).clamp_max(upper_clip)
        return torch.where(gene_mask, 0.0, standardized)


def validate_donor_partition(
    donor_groups: np.ndarray, train: np.ndarray,
) -> None:
    """Check anonymous equality groups without accepting protected values."""
    groups = np.asarray(donor_groups)
    flags = np.asarray(train)
    _require(groups.ndim == flags.ndim == 1 and len(groups) == len(flags)
             and flags.dtype == np.bool_, "Donor partition arrays are invalid.")
    _require(np.issubdtype(groups.dtype, np.integer),
             "Only anonymous integer donor-group aliases are permitted.")
    for group in np.unique(groups):
        _require(np.unique(flags[groups == group]).size == 1,
                 "An anonymous donor group crosses training and validation.")


def fit_training_metadata(
    raw_metadata: np.ndarray, train: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit pooled-cell morphology preprocessing on training rows only.

    The original model has exactly 22 morphology features.  If new missing
    training fields require additional missingness indicators, reject this
    dataset instead of silently changing that architecture.
    """
    values = np.asarray(raw_metadata, dtype=np.float64)
    flags = np.asarray(train)
    _require(values.ndim == 2 and values.shape[1] == len(ALLOWED_METADATA_COLUMNS)
             and flags.shape == (len(values),) and flags.dtype == np.bool_
             and bool(flags.any()), "Morphology/split shapes are invalid.")
    processor = TrainOnlyPreprocessor(log1p_metadata=True).fit(
        np.zeros((len(values), 1), dtype=np.int8), values,
        np.where(flags, "train", "validation"),
        metadata_names=ALLOWED_METADATA_COLUMNS,
    )
    _require(len(processor.missing_indicator_indices_) == 0,
             "Missing training morphology would change the fixed 22-feature model.")
    transformed = processor.transform(
        np.zeros((len(values), 1), dtype=np.int8), values,
    ).metadata
    _require(transformed.shape == values.shape and np.isfinite(transformed).all(),
             "Train-fitted morphology transform is invalid.")
    statistics = {
        "names": list(ALLOWED_METADATA_COLUMNS),
        "fit_scope": "pooled_training_cells_only",
        "training_cells": int(flags.sum()),
        "transform": "training_median_imputation_log1p_then_population_standardization",
        "sd_ddof": 0,
        "median": processor.metadata_median_.tolist(),
        "mean": processor.metadata_mean_.tolist(),
        "scale": processor.metadata_scale_.tolist(),
        "missing_indicator_indices": [],
    }
    return transformed, statistics


def load_so12_training_data(
    config: Mapping[str, Any] | None = None,
    paths: ProjectPaths | None = None,
    *,
    verify_checksums: bool = True,
    load_graphs: bool = True,
) -> SO12Dataset:
    """Load the fixed integrated cohort, frozen targets and train-only metadata.

    ``config`` is accepted for training-dispatch compatibility; the scientific
    data contract is fixed and is not inferred from arbitrary config fields.
    All paths honor ProjectPaths overrides.  Large graph/target arrays stay as
    read-only memory maps.  No raw donor values are read or returned: the
    previously verified protected grouping source is bound by its checksum.
    """
    del config
    paths = paths or current_paths()
    data_root = paths.data_root
    integrated = data_root / "processed" / "so12_pearson_integration_v1"
    manifest_path = integrated / "manifest.json"
    manifest = _json(manifest_path)
    upstream_run = paths.artifact_root / "runs" / "2026" / "10" / PREPROCESSING_RUN_ID
    _require((upstream_run / "_SUCCESS").is_file(),
             "The frozen preprocessing run is not successfully finalized.")
    run_checksums = _json(upstream_run / "checksums.json")
    reference_path = upstream_run / "derived_dataset_reference.json"
    reference_source = _check_file(
        reference_path, run_checksums["derived_dataset_reference.json"], True,
    )
    reference = _json(reference_path)
    integrated_source = _check_file(manifest_path, reference["manifest_sha256"], True)
    _require(manifest.get("run_id") == PREPROCESSING_RUN_ID
             and manifest.get("cells") == EXPECTED_CELLS
             and manifest.get("genes") == N_GENES
             and manifest.get("training_ready_for_masked_counts") is False,
             "Integrated data identity or feature-target contract has changed.")
    _require(manifest["configuration"]["train_cores"] == list(TRAIN_CORE_NUMBERS)
             and manifest["configuration"]["diagnostic_cores"] == list(VALIDATION_CORE_NUMBERS),
             "Frozen integrated donor partition has changed.")
    source_provenance = manifest["provenance"]
    _require(source_provenance["protected_grouping"]["consecutive_core_pairs_verified"]
             and source_provenance["protected_grouping"]["cross_slide_donor_count"] == 0
             and source_provenance["split"]["donor_overlap"] is False,
             "Frozen protected donor grouping is not verified.")
    source_records = {record["path"]: record for record in source_provenance["files"]}
    grouping_record = source_records["raw/Gastric Study_2.xlsx"]
    grouping_source = _check_file(
        data_root / grouping_record["path"], grouping_record["sha256"], verify_checksums,
    )

    genes = json.loads((integrated / "genes.json").read_text())
    _require(isinstance(genes, list) and len(genes) == len(set(genes)) == N_GENES
             and _canonical_sha256(genes) == source_provenance["genes_sha256"]
             and not any(gene.startswith(("Negative", "SystemControl")) for gene in genes),
             "Integrated biological probe order is invalid.")
    parameter_path = integrated / "pearson" / "fitted_parameters.npz"
    parameter_source = _check_file(
        parameter_path, run_checksums["methods/pearson/fitted_parameters.npz"], verify_checksums,
    )
    with np.load(parameter_path, allow_pickle=False) as archive:
        gene_p = archive["gene_frequency"].copy()
        residual_mean = archive["mean"].copy()
        residual_sd = archive["sd"].copy()
    _require(all(value.shape == (N_GENES,) and np.isfinite(value).all()
                 for value in (gene_p, residual_mean, residual_sd))
             and np.all(gene_p > 0) and np.all(residual_sd > 0)
             and np.isclose(gene_p.sum(), 1.0), "Frozen Pearson parameters are invalid.")
    observations_path = integrated / "observations.csv.gz"
    observations = pd.read_csv(observations_path, usecols=[
        "core", "core_number", "slide", "donor_group", "train", "depth",
        "fov", "cell_ID", "x_um", "y_um",
    ])
    _require(len(observations) == EXPECTED_CELLS
             and observations.train.dtype == np.bool_
             and not observations.duplicated(["slide", "fov", "cell_ID"]).any(),
             "Integrated observations have invalid keys or row counts.")
    validate_donor_partition(observations.donor_group.to_numpy(), observations.train.to_numpy())
    _require(observations.train.sum() == EXPECTED_TRAIN_CELLS
             and (~observations.train).sum() == EXPECTED_VALIDATION_CELLS
             and observations.groupby("donor_group").slide.nunique().eq(1).all(),
             "Integrated partition counts or cross-slide grouping changed.")

    target_records = {record["core"]: record for record in manifest["matrices"]
                      if record["method"] == "pearson"}
    _require(len(target_records) == 28, "Expected 28 plain-Pearson target matrices.")
    cores: list[SO12Core] = []
    raw_morphology: list[np.ndarray] = []
    training_flags: list[np.ndarray] = []
    file_sources = [reference_source, integrated_source, parameter_source, grouping_source]
    graph_manifest_sources: list[dict[str, Any]] = []
    for slide in (1, 2):
        cohort_root = data_root / "processed" / f"so{slide}_14core_relative_qkv_v1"
        cohort_path = cohort_root / "manifest.json"
        cohort = _json(cohort_path)
        _check_content_manifest(cohort, f"SO{slide} cohort")
        cohort_record = source_records[f"processed/so{slide}_14core_relative_qkv_v1/manifest.json"]
        file_sources.append(_check_file(cohort_path, cohort_record["sha256"], True))
        _require(cohort["features"]["gene_names"] == genes,
                 "Upstream gene order differs from the integrated targets.")
        metadata_refs = {record["source_artifacts"]["metadata"] for record in cohort["cores"]}
        metadata_hashes = {record["source_checksums"]["metadata_sha256"] for record in cohort["cores"]}
        _require(len(metadata_refs) == len(metadata_hashes) == 1,
                 "Cohort morphology provenance is inconsistent.")
        metadata_path = data_root / "raw" / next(iter(metadata_refs))
        metadata_source = _check_file(metadata_path, next(iter(metadata_hashes)), verify_checksums)
        file_sources.append(metadata_source)
        raw_metadata = pd.read_csv(metadata_path, usecols=[
            "slide_ID", "fov", "cell_ID", "CenterX_global_px", "CenterY_global_px",
            "nCount_RNA", *ALLOWED_METADATA_COLUMNS,
        ])
        _require(set(raw_metadata.slide_ID) == {slide}
                 and not raw_metadata.duplicated(["fov", "cell_ID"]).any(),
                 "Raw morphology keys or slide identity are invalid.")
        metadata_index = raw_metadata.set_index(["fov", "cell_ID"], verify_integrity=True)
        graph_root = data_root / "processed" / f"so{slide}_14core_relative_qkv_graphs_v1"
        graph_records: dict[str, Any] = {}
        if load_graphs:
            graph_manifest_path = graph_root / "manifest.json"
            graph_manifest = _json(graph_manifest_path)
            _check_content_manifest(graph_manifest, f"SO{slide} graph")
            _require(graph_manifest["cohort_manifest_sha256"] == cohort_record["sha256"],
                     "Graph collection was built from a different cohort.")
            graph_records = {record["alias"]: record for record in graph_manifest["cores"]}
            graph_manifest_sources.append({"path": str(graph_manifest_path),
                                           "sha256": _sha256(graph_manifest_path)})
        for record in sorted(cohort["cores"], key=lambda value: value["original_core_number"]):
            number = int(record["original_core_number"])
            alias = f"SO{slide}-C{number:02d}"
            _require(record["alias"] == alias, "Core alias and number disagree.")
            core_obs = observations.loc[observations.core == alias]
            n = int(record["cell_count"])
            expected_train = number in TRAIN_CORE_NUMBERS
            _require(len(core_obs) == n and core_obs.core_number.eq(number).all()
                     and core_obs.slide.eq(slide).all()
                     and core_obs.donor_group.eq((number + 1) // 2).all()
                     and core_obs.train.eq(expected_train).all(),
                     f"{alias} observation identity or partition changed.")
            _require(core_obs[["fov", "cell_ID"]].equals(
                core_obs[["fov", "cell_ID"]].sort_values(["fov", "cell_ID"], kind="mergesort")),
                f"{alias} observation order differs from the prepared row contract.")
            keys = pd.MultiIndex.from_frame(core_obs[["fov", "cell_ID"]])
            _require(bool(keys.isin(metadata_index.index).all()),
                     f"{alias} observations lack raw morphology rows.")
            aligned = metadata_index.loc[keys]
            raw_morphology.append(aligned[list(ALLOWED_METADATA_COLUMNS)].to_numpy(dtype=np.float64))
            training_flags.append(np.full(n, expected_train, dtype=bool))
            raw_path = cohort_root / "cores" / f"{alias}.npz"
            raw_source = _check_file(raw_path, cohort["files"][f"cores/{alias}.npz"], verify_checksums)
            with np.load(raw_path, allow_pickle=False) as archive:
                raw_counts = archive["expression_counts"]
                coordinates = archive["coordinates_um"]
            _require(raw_counts.shape == (n, N_GENES) and raw_counts.dtype == np.int32
                     and np.all(raw_counts >= 0), f"{alias} raw-count contract failed.")
            expected_coordinates = aligned[["CenterX_global_px", "CenterY_global_px"]].to_numpy(
                dtype=np.float64) * PIXEL_SIZE_UM
            _require(coordinates.shape == (n, 2) and np.isfinite(coordinates).all()
                     and np.array_equal(coordinates, expected_coordinates)
                     and np.allclose(coordinates, core_obs[["x_um", "y_um"]].to_numpy(),
                                     rtol=0, atol=1e-9),
                     f"{alias} coordinate rows do not align exactly to raw metadata.")
            totals = raw_counts.sum(axis=1, dtype=np.int64)
            _require(np.array_equal(totals, aligned.nCount_RNA.to_numpy())
                     and np.array_equal(totals, core_obs.depth.to_numpy()),
                     f"{alias} count rows do not align to source totals.")
            target_record = target_records[alias]
            target_path = integrated / target_record["path"]
            target_source = _check_file(target_path, target_record["sha256"], verify_checksums)
            target = np.load(target_path, mmap_mode="r", allow_pickle=False)
            _require(target.shape == (n, N_GENES) and target.dtype == np.float32
                     and np.isfinite(target).all(), f"{alias} Pearson targets are invalid.")
            edge_index = geometry = None
            core_source = {"raw_counts": raw_source, "target": target_source,
                           "raw_morphology": metadata_source}
            if load_graphs:
                graph_record = graph_records[alias]
                record_content = dict(graph_record)
                record_hash = record_content.pop("record_sha256", None)
                _require(record_hash == _canonical_sha256(record_content),
                         f"{alias} graph record checksum mismatch.")
                qc = graph_record["graph"]["qc"]
                _require(graph_record["n_cells"] == n and qc["n_nodes"] == n
                         and qc["self_loops"] == qc["cross_group_edges"] == 0
                         and qc["receiver_major_canonical_order"] is True
                         and qc["directed_edge_pairs_are_symmetric"] is True
                         and graph_record["graph"]["checksums"]["coordinates_um_sha256"]
                         == _geometry_array_sha256("coordinates_um", coordinates),
                         f"{alias} graph coordinates or scientific invariants disagree.")
                graph_sources = {}
                for filename in ("edge_index.npy", "relative_geometry.npy", "orientation.npz"):
                    graph_sources[filename] = _check_file(
                        graph_root / "cores" / alias / filename,
                        graph_record["files"][filename], verify_checksums,
                    )
                edge_index = np.load(graph_sources["edge_index.npy"]["path"], mmap_mode="r", allow_pickle=False)
                geometry = np.load(graph_sources["relative_geometry.npy"]["path"], mmap_mode="r", allow_pickle=False)
                _require(edge_index.shape == (2, qc["n_directed_edges"])
                         and edge_index.dtype == np.int64
                         and geometry.shape == (qc["n_directed_edges"], 70)
                         and geometry.dtype == np.float32,
                         f"{alias} edge/geometry array schema differs from the model contract.")
                core_source["graph"] = graph_sources
                core_source["graph_record_sha256"] = record_hash
            cores.append(SO12Core(
                alias=alias, slide=slide, donor_group=(number + 1) // 2,
                split="train" if expected_train else "validation",
                raw_counts=raw_counts, target_expression=target,
                node_covariates=np.empty((0, len(ALLOWED_METADATA_COLUMNS)), dtype=np.float32),
                coordinates_um=coordinates, edge_index=edge_index, relative_geometry=geometry,
                source=core_source,
            ))
    _require(len(cores) == 28 and sum(core.n_cells for core in cores) == EXPECTED_CELLS,
             "Combined cohort does not contain all 28 expected cores.")
    train_flags = np.concatenate(training_flags)
    transformed_metadata, metadata_statistics = fit_training_metadata(
        np.concatenate(raw_morphology), train_flags,
    )
    offset = 0
    for core in cores:
        core.node_covariates = transformed_metadata[offset:offset + core.n_cells]
        offset += core.n_cells
    target_mean = np.mean([
        np.asarray(core.target_expression).mean(axis=0, dtype=np.float64)
        for core in cores if core.split == "train"
    ], axis=0).astype(np.float32)
    split = {
        "training_core_numbers": list(TRAIN_CORE_NUMBERS),
        "validation_core_numbers": list(VALIDATION_CORE_NUMBERS),
        "training_cells": EXPECTED_TRAIN_CELLS, "validation_cells": EXPECTED_VALIDATION_CELLS,
        "training_donor_groups": 10, "validation_donor_groups": 4,
        "donor_overlap": False, "independent_test": False,
        "scope": "reused_preprocessing_diagnostic_donors_as_exploratory_validation",
    }
    preprocessing = {
        "target": "frozen_train_standardized_full_observation_pearson_upper_clip10",
        "input": "visible_count_over_visible_training_frequency_mass_pearson",
        "phi": 1.01, "upper_clip": 10.0, "full_mask_exposure": 1.0,
        "pearson_parameters_sha256": parameter_source["sha256"],
        "metadata": metadata_statistics,
        "constant_baseline": "equal_core_training_target_mean_per_gene",
        "constant_baseline_sha256": _canonical_sha256(target_mean.tolist()),
    }
    provenance = {
        "schema": "so12_relative_qkv_training_data_v1",
        "preprocessing_run_id": PREPROCESSING_RUN_ID,
        "source_manifest": integrated_source, "source_files": file_sources,
        "graph_manifests": graph_manifest_sources,
        "observations_sha256": _sha256(observations_path),
        "genes_sha256": source_provenance["genes_sha256"],
        "dataset_fingerprint": _canonical_sha256({
            "integrated_manifest_sha256": integrated_source["sha256"],
            "graph_manifests": [item["sha256"] for item in graph_manifest_sources],
        }),
        "split_fingerprint": _canonical_sha256(split),
        "preprocessing_fingerprint": _canonical_sha256(preprocessing),
        "split": split, "preprocessing": preprocessing,
        "source_file_checksums_verified": verify_checksums,
        "graph_arrays_loaded": load_graphs,
        "row_alignment_verified": True,
        "direct_donor_identifiers_returned": False,
        "excluded_unmapped_cells": 340,
        "target_normalization_uses_full_counts_but_inputs_do_not": True,
        "core_sizes": {core.alias: core.n_cells for core in cores},
    }
    return SO12Dataset(
        cores=tuple(cores), genes=tuple(genes), gene_p=gene_p,
        residual_mean=residual_mean, residual_sd=residual_sd,
        target_mean=target_mean, metadata_statistics=metadata_statistics,
        provenance=provenance,
    )


__all__ = [
    "SO12Core", "SO12Dataset", "SO12DataContractError", "PREPROCESSING_RUN_ID",
    "TRAIN_CORE_NUMBERS", "VALIDATION_CORE_NUMBERS", "visible_count_pearson_input",
    "validate_donor_partition", "fit_training_metadata", "load_so12_training_data",
]

"""Train-fitted SO1/SO2 data for the approved paper core/section holdout.

This is a separate dataset: historical Pearson parameters and targets are never
opened. The default role selection cannot open test count or graph arrays.
Test targets are generated in memory from frozen training parameters only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import ALLOWED_METADATA_COLUMNS
from .paths import ProjectPaths, current_paths
from .relative_geometry import _array_sha256 as _geometry_array_sha256
from .so12_relative_qkv_data import (
    N_GENES, PIXEL_SIZE_UM, PREPROCESSING_RUN_ID, SO12Core, SO12Dataset,
    SO12DataContractError, _canonical_sha256, _check_content_manifest,
    _check_file, _json, _require, _sha256, fit_training_metadata,
    visible_count_pearson_input,
)

DATASET_ID = "so12_paper_core_split"
DATASET_VERSION = "v1"
SPLIT_ID = "so12_paper_21train_5val_2test_v1"
PREPROCESSING_VERSION = "train21_quasipoisson_phi1p01_sample_sd_clip10_v1"
CACHE_RELATIVE_PATH = "processed/so12_paper_core_split_v1"
VALIDATION_CORE_NUMBERS = (2, 5, 13, 27, 28)
TEST_CORE_NUMBERS = (15, 23)
TRAIN_CORE_NUMBERS = tuple(i for i in range(1, 29)
                           if i not in VALIDATION_CORE_NUMBERS + TEST_CORE_NUMBERS)
ROLE_CORES = {"train": TRAIN_CORE_NUMBERS, "validation": VALIDATION_CORE_NUMBERS,
              "test": TEST_CORE_NUMBERS}
ROLE_CELLS = {"train": 264491, "validation": 61561, "test": 81607}
EXPECTED_CELLS = sum(ROLE_CELLS.values())
PHI = 1.01
UPPER_CLIP = 10.0
CHUNK_ROWS = 4096
# Recorded by the independent 2026-10-16 preprocessing verification before
# this new split was prepared; the upstream manifest itself omitted this hash.
# reports/analyses/so12_pearson_preprocessing/independent_verification.json
UPSTREAM_OBSERVATIONS_SHA256 = "c388d88d86d5c6edbef182db5b3d7f40a3d60a1807c5e479589f40211fda7dc4"


def core_alias(number: int) -> str:
    _require(1 <= number <= 28, "Core number is outside the approved cohort.")
    return f"SO{1 if number <= 14 else 2}-C{number:02d}"


def split_contract() -> dict[str, Any]:
    return {
        "split_id": SPLIT_ID,
        "training_core_numbers": list(TRAIN_CORE_NUMBERS),
        "validation_core_numbers": list(VALIDATION_CORE_NUMBERS),
        "test_core_numbers": list(TEST_CORE_NUMBERS),
        "training_cells": ROLE_CELLS["train"],
        "validation_cells": ROLE_CELLS["validation"],
        "test_cells": ROLE_CELLS["test"],
        "donor_overlap": True, "donor_overlap_authorized": True,
        "scope": "exploratory_core_section_holdout",
        "historically_untouched_test": False,
        "test_slides": [2], "test_used_for_selection": False,
    }


@dataclass(slots=True)
class SO12PaperDataset(SO12Dataset):
    @property
    def test_cores(self) -> tuple[SO12Core, ...]:
        return tuple(core for core in self.cores if core.split == "test")


def _selected_numbers(roles: Sequence[str]) -> tuple[int, ...]:
    _require(not isinstance(roles, str) and bool(roles)
             and len(set(roles)) == len(roles)
             and set(roles).issubset(ROLE_CORES), "Invalid or duplicate dataset roles.")
    _require("test" not in roles or tuple(roles) == ("test",),
             "Test data must be opened separately with roles=('test',).")
    return tuple(sorted(number for role in roles for number in ROLE_CORES[role]))


def _validate_parameters(p: np.ndarray, mean: np.ndarray, sd: np.ndarray,
                         genes: int) -> None:
    _require(all(np.asarray(x).shape == (genes,) and np.isfinite(x).all()
                 for x in (p, mean, sd)) and np.all(p > 0) and np.all(sd > 0)
             and np.isclose(p.sum(), 1.0), "Frozen Pearson parameters are invalid.")


def _residual_batches(counts: np.ndarray, p: np.ndarray):
    _require(counts.ndim == 2 and counts.shape[1] == len(p)
             and np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0),
             "Pearson targets require nonnegative integer counts.")
    for start in range(0, len(counts), CHUNK_ROWS):
        batch = counts[start:start + CHUNK_ROWS].astype(np.float64)
        exposure = np.maximum(batch.sum(axis=1), 1.)
        mu = exposure[:, None] * np.asarray(p, dtype=np.float64)[None, :]
        yield start, (batch - mu) / np.sqrt(PHI * mu)


def fit_training_pearson(counts_by_core: Mapping[int, np.ndarray],
                         training_numbers: Sequence[int] = TRAIN_CORE_NUMBERS,
                         ) -> dict[str, np.ndarray]:
    """Fit using explicitly selected training arrays; other mapping values are unread.

    FP64 blockwise centered moments avoid cancellation. Saved parameters are
    FP32, matching the masking-safe model adapter. SD uses ddof=1.
    """
    _require(bool(training_numbers) and len(set(training_numbers)) == len(training_numbers),
             "Training membership is empty or duplicated.")
    arrays = [counts_by_core[number] for number in training_numbers]
    genes = arrays[0].shape[1]
    _require(all(x.ndim == 2 and x.shape[1] == genes and len(x) > 0
                 and np.issubdtype(x.dtype, np.integer) and np.all(x >= 0)
                 for x in arrays), "Training counts are invalid.")
    sums = sum((x.sum(axis=0, dtype=np.float64) for x in arrays),
               np.zeros(genes, dtype=np.float64))
    _require(np.all(sums > 0), "A gene has zero training count.")
    p = (sums / sums.sum()).astype(np.float32)
    n = 0
    mean = np.zeros(genes, dtype=np.float64)
    m2 = np.zeros(genes, dtype=np.float64)
    for counts in arrays:
        for _, values in _residual_batches(counts, p):
            batch_n = len(values)
            batch_mean = values.mean(axis=0)
            batch_m2 = np.square(values - batch_mean).sum(axis=0)
            delta = batch_mean - mean
            total = n + batch_n
            mean += delta * (batch_n / total)
            m2 += batch_m2 + delta * delta * (n * batch_n / total)
            n = total
    _require(n > 1, "At least two training cells are required.")
    sd = np.sqrt(m2 / (n - 1)).astype(np.float32)
    mean = mean.astype(np.float32)
    _validate_parameters(p, mean, sd, genes)
    return {"gene_frequency": p, "mean": mean, "sd": sd,
            "training_cells": np.asarray(n, dtype=np.int64)}


def pearson_targets(counts: np.ndarray, p: np.ndarray, mean: np.ndarray,
                     sd: np.ndarray) -> np.ndarray:
    """Fully observed supervised targets, never model inputs."""
    _validate_parameters(p, mean, sd, counts.shape[1])
    output = np.empty(counts.shape, dtype=np.float32)
    for start, values in _residual_batches(counts, p):
        values = np.minimum((values - mean) / sd, UPPER_CLIP)
        _require(np.isfinite(values).all(), "Nonfinite Pearson targets.")
        output[start:start + len(values)] = values
    return output


def apply_frozen_metadata(raw: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    _require(statistics["names"] == list(ALLOWED_METADATA_COLUMNS)
             and statistics["missing_indicator_indices"] == [],
             "Morphology schema has changed.")
    values = np.asarray(raw, dtype=np.float64)
    median, mean, scale = (np.asarray(statistics[key], dtype=np.float64)
                           for key in ("median", "mean", "scale"))
    _require(values.ndim == 2 and values.shape[1] == len(median)
             and all(np.isfinite(x).all() for x in (median, mean, scale))
             and np.all(scale > 0), "Frozen morphology parameters are invalid.")
    values = np.where(np.isnan(values), median, values)
    _require(np.isfinite(values).all() and np.all(values >= 0),
             "Morphology contains invalid values.")
    return ((np.log1p(values) - mean) / scale).astype(np.float32)


def _source_context(paths: ProjectPaths, verify: bool) -> dict[str, Any]:
    """Read alignment records/manifests only; no core count/target/graph arrays."""
    data_root = paths.data_root
    upstream = paths.artifact_root / "runs/2026/10" / PREPROCESSING_RUN_ID
    _require((upstream / "_SUCCESS").is_file(), "Upstream source audit is not finalized.")
    checksums = _json(upstream / "checksums.json")
    reference_path = upstream / "derived_dataset_reference.json"
    _check_file(reference_path, checksums[reference_path.name], True)
    reference = _json(reference_path)
    integrated = data_root / "processed/so12_pearson_integration_v1"
    manifest_path = integrated / "manifest.json"
    _check_file(manifest_path, reference["manifest_sha256"], True)
    manifest = _json(manifest_path)
    _require(manifest["run_id"] == PREPROCESSING_RUN_ID
             and manifest["cells"] == EXPECTED_CELLS and manifest["genes"] == N_GENES,
             "Upstream data identity changed.")
    prior = manifest["provenance"]
    records = {record["path"]: record for record in prior["files"]}
    genes_path = integrated / "genes.json"
    genes = json.loads(genes_path.read_text())
    _require(len(genes) == len(set(genes)) == N_GENES
             and _canonical_sha256(genes) == prior["genes_sha256"]
             and not any(x.startswith(("Negative", "SystemControl")) for x in genes),
             "Biological gene order changed.")
    observations_path = integrated / "observations.csv.gz"
    _check_file(observations_path, UPSTREAM_OBSERVATIONS_SHA256, True)
    observations = pd.read_csv(observations_path, usecols=[
        "core", "core_number", "slide", "donor_group", "depth", "fov", "cell_ID",
        "x_um", "y_um"])
    _require(len(observations) == EXPECTED_CELLS
             and not observations.duplicated(["slide", "fov", "cell_ID"]).any(),
             "Observation keys or counts changed.")
    sources = [{"path": str(path.relative_to(data_root)), "sha256": _sha256(path)}
               for path in (manifest_path, genes_path, observations_path)]
    cohorts, graphs = {}, {}
    for slide in (1, 2):
        cohort_path = data_root / f"processed/so{slide}_14core_relative_qkv_v1/manifest.json"
        cohort = _json(cohort_path)
        _check_content_manifest(cohort, f"SO{slide} cohort")
        relative = str(cohort_path.relative_to(data_root))
        expected = records[relative]["sha256"]
        _check_file(cohort_path, expected, True)
        _require(cohort["features"]["gene_names"] == genes,
                 "Source gene panels differ.")
        cohorts[slide] = cohort
        sources.append({"path": relative, "sha256": expected})
        graph_path = data_root / f"processed/so{slide}_14core_relative_qkv_graphs_v1/manifest.json"
        graph = _json(graph_path)
        _check_content_manifest(graph, f"SO{slide} graph")
        _require(graph["cohort_manifest_sha256"] == expected,
                 "Graph collection belongs to a different cohort.")
        graphs[slide] = graph
        sources.append({"path": str(graph_path.relative_to(data_root)), "sha256": _sha256(graph_path)})
    _require(prior["protected_grouping"]["consecutive_core_pairs_verified"],
             "Anonymous donor grouping is not verified.")
    return {"genes": genes, "observations": observations, "cohorts": cohorts,
            "graphs": graphs, "sources": sources,
            "upstream_alignment_run": PREPROCESSING_RUN_ID}


def _read_source_cores(paths: ProjectPaths, context: Mapping[str, Any],
                        numbers: Sequence[int], verify: bool,
                        load_graphs: bool) -> tuple[list[SO12Core], list[np.ndarray]]:
    """Selection happens before any per-core path is opened."""
    numbers = set(numbers)
    cores, morphology = [], []
    observations = context["observations"]
    for slide in (1, 2):
        selected = [record for record in context["cohorts"][slide]["cores"]
                    if int(record["original_core_number"]) in numbers]
        if not selected:
            continue
        cohort = context["cohorts"][slide]
        metadata_refs = {r["source_artifacts"]["metadata"] for r in selected}
        metadata_hashes = {r["source_checksums"]["metadata_sha256"] for r in selected}
        _require(len(metadata_refs) == len(metadata_hashes) == 1,
                 "Source morphology provenance is inconsistent.")
        metadata_path = paths.data_root / "raw" / next(iter(metadata_refs))
        metadata_source = _check_file(metadata_path, next(iter(metadata_hashes)), verify)
        raw = pd.read_csv(metadata_path, usecols=["slide_ID", "fov", "cell_ID",
            "CenterX_global_px", "CenterY_global_px", "nCount_RNA", *ALLOWED_METADATA_COLUMNS])
        _require(set(raw.slide_ID) == {slide}
                 and not raw.duplicated(["fov", "cell_ID"]).any(),
                 "Raw morphology keys or slide identity changed.")
        indexed = raw.set_index(["fov", "cell_ID"], verify_integrity=True)
        graph_records = {r["alias"]: r for r in context["graphs"][slide]["cores"]}
        for record in sorted(selected, key=lambda r: int(r["original_core_number"])):
            number = int(record["original_core_number"])
            alias = core_alias(number)
            _require(record["alias"] == alias, "Source core identity changed.")
            obs = observations.loc[observations.core == alias]
            n = int(record["cell_count"])
            _require(len(obs) == n and obs.core_number.eq(number).all()
                     and obs.slide.eq(slide).all()
                     and obs.donor_group.eq((number + 1) // 2).all(),
                     f"{alias} observation identity or size changed.")
            _require(obs[["fov", "cell_ID"]].equals(
                obs[["fov", "cell_ID"]].sort_values(["fov", "cell_ID"], kind="mergesort")),
                f"{alias} row order changed.")
            keys = pd.MultiIndex.from_frame(obs[["fov", "cell_ID"]])
            _require(keys.isin(indexed.index).all(), f"{alias} metadata keys are missing.")
            aligned = indexed.loc[keys]
            raw_path = paths.data_root / f"processed/so{slide}_14core_relative_qkv_v1/cores/{alias}.npz"
            raw_source = _check_file(raw_path, cohort["files"][f"cores/{alias}.npz"], verify)
            with np.load(raw_path, allow_pickle=False) as archive:
                counts, coordinates = archive["expression_counts"], archive["coordinates_um"]
            _require(counts.shape == (n, N_GENES) and counts.dtype == np.int32
                     and np.all(counts >= 0), f"{alias} raw counts changed.")
            expected = aligned[["CenterX_global_px", "CenterY_global_px"]].to_numpy(
                dtype=np.float64) * PIXEL_SIZE_UM
            _require(coordinates.shape == (n, 2) and np.isfinite(coordinates).all()
                     and np.array_equal(coordinates, expected)
                     and np.allclose(coordinates, obs[["x_um", "y_um"]].to_numpy(), rtol=0, atol=1e-9),
                     f"{alias} coordinates do not align.")
            totals = counts.sum(axis=1, dtype=np.int64)
            _require(np.array_equal(totals, aligned.nCount_RNA.to_numpy())
                     and np.array_equal(totals, obs.depth.to_numpy()),
                     f"{alias} counts do not align to metadata totals.")
            source = {"raw_counts": raw_source, "raw_morphology": metadata_source}
            edges = geometry = None
            if load_graphs:
                graph_record = graph_records[alias]
                content = dict(graph_record)
                record_hash = content.pop("record_sha256", None)
                _require(record_hash == _canonical_sha256(content), f"{alias} graph record changed.")
                qc = graph_record["graph"]["qc"]
                _require(graph_record["n_cells"] == qc["n_nodes"] == n
                         and qc["self_loops"] == qc["cross_group_edges"] == 0
                         and qc["receiver_major_canonical_order"] is True
                         and qc["directed_edge_pairs_are_symmetric"] is True
                         and graph_record["graph"]["checksums"]["coordinates_um_sha256"]
                         == _geometry_array_sha256("coordinates_um", coordinates),
                         f"{alias} graph invariant or coordinate mismatch.")
                graph_root = paths.data_root / f"processed/so{slide}_14core_relative_qkv_graphs_v1/cores/{alias}"
                files = {name: _check_file(graph_root / name, graph_record["files"][name], verify)
                         for name in ("edge_index.npy", "relative_geometry.npy", "orientation.npz")}
                edges = np.load(files["edge_index.npy"]["path"], mmap_mode="r", allow_pickle=False)
                geometry = np.load(files["relative_geometry.npy"]["path"], mmap_mode="r", allow_pickle=False)
                _require(edges.shape == (2, qc["n_directed_edges"]) and edges.dtype == np.int64
                         and geometry.shape == (qc["n_directed_edges"], 70)
                         and geometry.dtype == np.float32, f"{alias} graph schema mismatch.")
                source.update(graph=files, graph_record_sha256=record_hash)
            role = next(role for role, members in ROLE_CORES.items() if number in members)
            morphology.append(aligned[list(ALLOWED_METADATA_COLUMNS)].to_numpy(dtype=np.float64))
            cores.append(SO12Core(alias, slide, (number + 1) // 2, role, counts,
                                  np.empty((0, N_GENES), np.float32),
                                  np.empty((0, len(ALLOWED_METADATA_COLUMNS)), np.float32),
                                  coordinates, edges, geometry, source))
    _require({int(core.alias[-2:]) for core in cores} == numbers,
             "Selected core coverage is incomplete.")
    return cores, morphology


def _read_cache(paths: ProjectPaths, verify: bool) -> tuple[Path, dict[str, Any]]:
    root = paths.data_root / CACHE_RELATIVE_PATH
    _require((root / "_SUCCESS").is_file(), "Prepare the paper split dataset before loading it.")
    manifest = _json(root / "manifest.json")
    _check_content_manifest(manifest, "Paper split")
    _require((root / "_SUCCESS").read_text().strip() == _sha256(root / "manifest.json")
             and manifest["split"] == split_contract()
             and manifest["dataset_id"] == DATASET_ID
             and manifest["preprocessing_version"] == PREPROCESSING_VERSION,
             "Paper dataset identity or frozen split changed.")
    for name in ("fitted_parameters.npz", "metadata_statistics.json", "genes.json"):
        _check_file(root / name, manifest["files"][name], verify)
    return root, manifest


def prepare_so12_paper_data(paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Create the immutable train/validation cache without opening test arrays.

    Preparation is CPU-only and deterministic. Existing successful caches are
    verified, never overwritten. Interrupted staging directories are preserved.
    """
    paths = paths or current_paths()
    destination = paths.data_root / CACHE_RELATIVE_PATH
    if destination.exists():
        root, manifest = _read_cache(paths, True)
        for name, expected in manifest["files"].items():
            _check_file(root / name, expected, True)
        context = _source_context(paths, True)
        _require(context["sources"] == manifest["alignment_sources"],
                 "Source alignment or graph manifests changed after preparation.")
        for source in manifest["source_files"]:
            _check_file(paths.data_root / source["path"], source["sha256"], True)
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    needed = (ROLE_CELLS["train"] + ROLE_CELLS["validation"]) * N_GENES * 4
    _require(shutil.disk_usage(destination.parent).free > 5 * 1024**3 + needed + 100 * 1024**2,
             "Preparing targets would violate the 5 GiB disk safety floor.")
    staging = destination.with_name(destination.name + ".building")
    _require(not staging.exists(), "An interrupted preparation exists; preserve and inspect it before retrying.")
    context = _source_context(paths, True)
    numbers = _selected_numbers(("train", "validation"))
    cores, raw_metadata = _read_source_cores(paths, context, numbers, True, False)
    for role in ("train", "validation"):
        _require(sum(core.n_cells for core in cores if core.split == role) == ROLE_CELLS[role],
                 "Approved split cell totals changed.")
    parameters = fit_training_pearson({int(core.alias[-2:]): core.raw_counts for core in cores})
    flags = np.concatenate([np.full(core.n_cells, core.split == "train") for core in cores])
    _, metadata_stats = fit_training_metadata(np.concatenate(raw_metadata), flags)
    staging.mkdir()
    (staging / "targets").mkdir()
    matrices, train_means = [], []
    for core in cores:
        target = pearson_targets(core.raw_counts, parameters["gene_frequency"],
                                  parameters["mean"], parameters["sd"])
        relative = f"targets/{core.alias}.npy"
        np.save(staging / relative, target, allow_pickle=False)
        matrices.append({"core": core.alias, "role": core.split, "cells": core.n_cells,
                         "path": relative, "sha256": _sha256(staging / relative)})
        if core.split == "train":
            train_means.append(target.mean(axis=0, dtype=np.float64))
    parameters["target_mean"] = np.mean(train_means, axis=0).astype(np.float32)
    np.savez(staging / "fitted_parameters.npz", **parameters)
    (staging / "metadata_statistics.json").write_text(json.dumps(metadata_stats, indent=2) + "\n")
    (staging / "genes.json").write_text(json.dumps(context["genes"]) + "\n")
    files = {name: _sha256(staging / name) for name in
             ("fitted_parameters.npz", "metadata_statistics.json", "genes.json")}
    files.update({m["path"]: m["sha256"] for m in matrices})
    preprocessing = {
        "fit_scope": "pooled_training_cells_only", "training_core_numbers": list(TRAIN_CORE_NUMBERS),
        "training_cells": ROLE_CELLS["train"], "phi": PHI, "upper_clip": UPPER_CLIP,
        "sample_sd_ddof": 1, "residual_moment_precision": "float64",
        "saved_parameter_dtype": "float32", "target_dtype": "float32",
        "target": "frozen_train_standardized_full_observation_pearson_upper_clip10",
        "input": "visible_count_over_visible_training_frequency_mass_pearson",
        "old_fitted_parameters_used": False, "old_normalized_targets_used": False,
        "fitted_parameters_sha256": files["fitted_parameters.npz"],
        "metadata_statistics_sha256": files["metadata_statistics.json"],
        "constant_baseline": "equal_core_training_target_mean_per_gene",
    }
    source_files = {item["path"]: item for item in context["sources"]}
    for core in cores:
        for kind in ("raw_counts", "raw_morphology"):
            item = core.source[kind]
            relative = str(Path(item["path"]).relative_to(paths.data_root))
            source_files[relative] = {"path": relative, "sha256": item["sha256"]}
    source_files = sorted(source_files.values(), key=lambda x: x["path"])
    manifest = {
        "schema": "so12_paper_core_split_data_v1", "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "split": split_contract(), "split_fingerprint": _canonical_sha256(split_contract()),
        "preprocessing": preprocessing, "preprocessing_fingerprint": _canonical_sha256(preprocessing),
        "source_files": source_files, "alignment_sources": context["sources"],
        "upstream_alignment_run": PREPROCESSING_RUN_ID,
        "genes": N_GENES, "cells": EXPECTED_CELLS, "materialized_cells": sum(core.n_cells for core in cores),
        "matrices": matrices, "files": files, "test_arrays_opened_during_preparation": [],
        "test_target_policy": "in_memory_transform_only_after_checkpoint_selection",
        "excluded_unmapped_cells": 340, "row_alignment_verified": True,
        "direct_donor_identifiers_returned": False,
        "core_sizes": {record["alias"]: int(record["cell_count"])
                       for cohort in context["cohorts"].values() for record in cohort["cores"]},
    }
    manifest["dataset_fingerprint"] = _canonical_sha256({
        "source_files": source_files, "files": files,
        "split_fingerprint": manifest["split_fingerprint"],
        "preprocessing_fingerprint": manifest["preprocessing_fingerprint"]})
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "_SUCCESS").write_text(_sha256(staging / "manifest.json") + "\n")
    os.replace(staging, destination)
    return manifest


def load_so12_paper_data(config: Mapping[str, Any] | None = None,
                          paths: ProjectPaths | None = None,
                          roles: Sequence[str] = ("train", "validation"),
                          verify_checksums: bool = True,
                          load_graphs: bool = True) -> SO12PaperDataset:
    """Load selected roles; test must be requested alone after model selection."""
    numbers = _selected_numbers(roles)
    paths = paths or current_paths()
    cache, manifest = _read_cache(paths, verify_checksums)
    if config:
        declared = config.get("dataset", {})
        for key, expected in (("dataset_id", DATASET_ID), ("version", DATASET_VERSION),
                              ("split_id", SPLIT_ID),
                              ("dataset_fingerprint", manifest["dataset_fingerprint"]),
                              ("split_fingerprint", manifest["split_fingerprint"])):
            _require(key not in declared or declared[key] == expected,
                     f"Configuration {key} disagrees with the frozen dataset.")
    context = _source_context(paths, verify_checksums)
    _require(context["sources"] == manifest["alignment_sources"],
             "Source alignment or graph manifests changed after preparation.")
    genes = json.loads((cache / "genes.json").read_text())
    _require(genes == context["genes"], "Cached gene order changed.")
    with np.load(cache / "fitted_parameters.npz", allow_pickle=False) as archive:
        p, mean, sd, target_mean = (archive[key].copy() for key in
                                   ("gene_frequency", "mean", "sd", "target_mean"))
        _require(int(archive["training_cells"]) == ROLE_CELLS["train"], "Fit membership changed.")
    _validate_parameters(p, mean, sd, N_GENES)
    _require(target_mean.shape == (N_GENES,) and np.isfinite(target_mean).all(),
             "Frozen training baseline is invalid.")
    metadata_stats = _json(cache / "metadata_statistics.json")
    _require(metadata_stats["training_cells"] == ROLE_CELLS["train"], "Morphology fit membership changed.")
    cores, raw_metadata = _read_source_cores(paths, context, numbers, verify_checksums, load_graphs)
    for core, raw in zip(cores, raw_metadata, strict=True):
        core.node_covariates = apply_frozen_metadata(raw, metadata_stats)
        if core.split == "test":
            core.target_expression = pearson_targets(core.raw_counts, p, mean, sd)
            core.source["target"] = {"storage": "memory_only", "fitted_parameters_sha256":
                                    manifest["files"]["fitted_parameters.npz"]}
        else:
            name = f"targets/{core.alias}.npy"
            core.source["target"] = _check_file(cache / name, manifest["files"][name], verify_checksums)
            core.target_expression = np.load(cache / name, mmap_mode="r", allow_pickle=False)
        _require(core.target_expression.shape == core.raw_counts.shape
                 and core.target_expression.dtype == np.float32
                 and np.isfinite(core.target_expression).all(), f"{core.alias} target contract failed.")
    for role in roles:
        _require(sum(core.n_cells for core in cores if core.split == role) == ROLE_CELLS[role],
                 "Loaded split cell count changed.")
    provenance = {key: manifest[key] for key in (
        "dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint", "split",
        "preprocessing", "upstream_alignment_run", "excluded_unmapped_cells")}
    provenance.update({
        "schema": "so12_paper_training_data_v1", "source_manifest": str(cache / "manifest.json"),
        "source_manifest_sha256": _sha256(cache / "manifest.json"),
        "fitted_parameters_sha256": manifest["files"]["fitted_parameters.npz"],
        "source_file_checksums_verified": verify_checksums, "graph_arrays_loaded": load_graphs,
        "row_alignment_verified": True, "direct_donor_identifiers_returned": False,
        "target_normalization_uses_full_counts_but_inputs_do_not": True,
        "loaded_roles": list(roles), "opened_core_aliases": [core.alias for core in cores],
        "test_arrays_opened": "test" in roles,
        "core_sizes": {core.alias: core.n_cells for core in cores},
        "source_files": [core.source for core in cores],
    })
    return SO12PaperDataset(tuple(cores), tuple(genes), p, mean, sd, target_mean,
                            metadata_stats, provenance)


__all__ = ["SO12PaperDataset", "SO12Core", "SO12DataContractError", "DATASET_ID",
           "DATASET_VERSION", "SPLIT_ID", "PREPROCESSING_VERSION", "CACHE_RELATIVE_PATH",
           "TRAIN_CORE_NUMBERS", "VALIDATION_CORE_NUMBERS", "TEST_CORE_NUMBERS", "ROLE_CELLS",
           "core_alias", "split_contract", "prepare_so12_paper_data", "load_so12_paper_data",
           "fit_training_pearson", "pearson_targets", "apply_frozen_metadata",
           "visible_count_pearson_input"]

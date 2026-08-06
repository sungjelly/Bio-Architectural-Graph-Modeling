#!/usr/bin/env python3
"""Prepare the immutable ten-core grouped adjacency-ablation artifact.

Only verified raw count, coordinate, FOV-routing, and vendor-QC arrays are
carried forward from the established per-core artifacts.  Legacy normalized
targets, metadata covariates, cell labels, split labels, and graphs are never
reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.adjacency_ablation import (  # noqa: E402
    CORE_ALIASES,
    build_five_fold_splits,
    derive_evaluation_mask_seed,
    fit_equal_core_log1p_standardizer,
    mask_realization_sha256,
    materialize_fixed_adjacencies,
    ndarray_sha256,
    sample_uniform_mask_numpy,
    validate_explicit_self_adjacency,
)
from spatial_benchmark.adjacent_normal_selection import (  # noqa: E402
    load_adjacent_normal_route,
)
from spatial_benchmark.artifacts import load_prepared_artifact  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


ARTIFACT_KIND = "adjacent_normal_grouped_adjacency_ablation_prepared_v1"
DATASET_ID = "cosmx_adjacent_normal_grouped_adjacency_v1"
DATASET_VERSION = "adjacent_normal_grouped_adjacency_v1"
SPLIT_ID = "adjacent_normal_10donor_slide_balanced_5fold_v1"
EXPECTED_SELECTION_SHA256 = (
    "a460de90ba9998e6817cdf3fbbcd2416eec048ec3fe6c825db8677b70b8b297f"
)
EXPECTED_CELLS = 117_386
EXPECTED_FOV_GROUPS = 139
EXPECTED_GENES = 1000
EXPECTED_QC_PASS = 112_815
EXPECTED_SELECTION = Path(
    "scratch/preprocessing/adjacent_normal_10_core_selection/selection.json"
)
UPSTREAM_ROOT = Path("data/processed/adjacent_normal_10core_qkv_large_k_v1")
DEFAULT_OUTPUT = Path(
    "data/processed/adjacent_normal_grouped_adjacency_ablation_v1"
)


class AdjacencyPreparationError(RuntimeError):
    """Raised when prepared inputs or immutable output violate the contract."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjacencyPreparationError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AdjacencyPreparationError(f"cannot parse JSON: {path}") from error
    return dict(_mapping(value, str(path)))


def _project_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise AdjacencyPreparationError(
            f"artifact path is outside project root: {path}"
        ) from error


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AdjacencyPreparationError(f"refusing to overwrite {path}")
    np.savez_compressed(path, **arrays)


def _selection_routes(selection_path: Path) -> dict[str, Any]:
    if sha256_file(selection_path) != EXPECTED_SELECTION_SHA256:
        raise AdjacencyPreparationError("protected selection file checksum changed")
    mode = stat.S_IMODE(selection_path.stat().st_mode)
    if mode & 0o077:
        raise AdjacencyPreparationError(
            "protected selection manifest is readable outside its owner"
        )
    selection = _strict_json(selection_path)
    aggregate = _mapping(selection.get("aggregate_validation"), "selection aggregate")
    if (
        aggregate.get("selected_count") != 10
        or aggregate.get("selected_distinct_donor_count") != 10
        or aggregate.get("selected_count_by_slide") != {"SO_1": 5, "SO_2": 5}
    ):
        raise AdjacencyPreparationError(
            "selection no longer establishes ten distinct selected donors"
        )
    routes = {
        alias: load_adjacent_normal_route(selection_path, alias)
        for alias in CORE_ALIASES
    }
    return {"selection": selection, "routes": routes}


def _upstream_root(alias: str) -> Path:
    return PROJECT_ROOT / UPSTREAM_ROOT / alias.lower() / "prepared_v1"


def _load_source_cores(
    selection_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], tuple[str, ...], list[dict[str, Any]]]:
    routed = _selection_routes(selection_path)
    routes = routed["routes"]
    cores: dict[str, dict[str, np.ndarray]] = {}
    gene_names: tuple[str, ...] | None = None
    provenance: list[dict[str, Any]] = [
        {
            "role": "protected_ten_core_selection",
            "reference": _project_reference(selection_path),
            "sha256": EXPECTED_SELECTION_SHA256,
            "contains_direct_identifiers": False,
            "contains_protected_core_routing": True,
        }
    ]
    for alias in CORE_ALIASES:
        root = _upstream_root(alias)
        manifest, arrays, _masks = load_prepared_artifact(root, load_arrays=True)
        if arrays is None:
            raise AssertionError("load_arrays=True returned no arrays")
        source_genes = tuple(
            str(value)
            for value in _mapping(manifest.get("features"), "source features").get(
                "gene_names", ()
            )
        )
        if len(source_genes) != EXPECTED_GENES or len(set(source_genes)) != EXPECTED_GENES:
            raise AdjacencyPreparationError(f"{alias} source gene schema is invalid")
        if gene_names is None:
            gene_names = source_genes
        elif source_genes != gene_names:
            raise AdjacencyPreparationError("ordered gene schema differs across cores")
        if any(
            name.lower().startswith(("negative", "systemcontrol"))
            for name in source_genes
        ):
            raise AdjacencyPreparationError("technical probes remain in targets")
        required = {"expression_counts", "coordinates_um", "fov", "qc_passed"}
        if not required.issubset(arrays):
            raise AdjacencyPreparationError(f"{alias} source arrays are incomplete")
        counts = np.asarray(arrays["expression_counts"])
        coordinates = np.asarray(arrays["coordinates_um"], dtype=np.float64)
        raw_fov = np.asarray(arrays["fov"])
        qc = np.asarray(arrays["qc_passed"], dtype=np.bool_)
        n_cells = counts.shape[0]
        if (
            counts.shape != (n_cells, EXPECTED_GENES)
            or coordinates.shape != (n_cells, 2)
            or raw_fov.shape != (n_cells,)
            or qc.shape != (n_cells,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)
            or not np.isfinite(coordinates).all()
        ):
            raise AdjacencyPreparationError(f"{alias} source array contract failed")
        route = routes[alias]
        observed_fovs = tuple(sorted(int(value) for value in np.unique(raw_fov)))
        if observed_fovs != tuple(route.fovs) or n_cells != int(
            next(
                row["n_cells"]
                for row in routed["selection"]["cores"]
                if row["alias"] == alias
            )
        ):
            raise AdjacencyPreparationError(
                f"{alias} source cells/FOVs differ from protected routing"
            )
        local_lookup = {value: index for index, value in enumerate(observed_fovs)}
        fov_group = np.asarray(
            [local_lookup[int(value)] for value in raw_fov], dtype=np.int16
        )
        cores[alias] = {
            "expression_counts": np.asarray(counts, dtype=np.int32),
            "coordinates_um": coordinates,
            "fov_group": fov_group,
            "qc_passed": qc,
        }
        provenance.append(
            {
                "role": "verified_prior_prepared_source",
                "core_alias": alias,
                "reference": _project_reference(root),
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "prepared_data_sha256": sha256_file(root / "prepared_data.npz"),
                "arrays_reused": [
                    "expression_counts",
                    "coordinates_um",
                    "fov",
                    "qc_passed",
                ],
                "arrays_explicitly_rejected": [
                    "target_expression",
                    "node_covariates",
                    "split_labels",
                    "edge_index",
                    "fixed_masks",
                ],
            }
        )
    assert gene_names is not None
    if sum(core["expression_counts"].shape[0] for core in cores.values()) != EXPECTED_CELLS:
        raise AdjacencyPreparationError("ten-core cell count changed")
    if sum(np.unique(core["fov_group"]).size for core in cores.values()) != EXPECTED_FOV_GROUPS:
        raise AdjacencyPreparationError("ten-core FOV count changed")
    if sum(int(core["qc_passed"].sum()) for core in cores.values()) != EXPECTED_QC_PASS:
        raise AdjacencyPreparationError("vendor-QC pass count changed")
    return cores, gene_names, provenance


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": np.asarray(value).dtype.str,
            "sha256": ndarray_sha256(value),
        }
        for name, value in arrays.items()
    }


def _content_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_id", "content_sha256"}
    }


def _fingerprints(manifest: Mapping[str, Any]) -> dict[str, str]:
    cores = _mapping(manifest["cores"], "cores")
    folds = _mapping(manifest["folds"], "folds")
    dataset = canonical_sha256(
        {
            "schema": "adjacency_ablation_dataset_v1",
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "gene_schema": manifest["features"]["gene_schema_sha256"],
            "cores": {
                alias: {
                    "data_sha256": cores[alias]["data_file_sha256"],
                    "n_cells": cores[alias]["n_cells"],
                    "n_fov_groups": cores[alias]["n_fov_groups"],
                }
                for alias in CORE_ALIASES
            },
        }
    )
    split = canonical_sha256(
        {
            "schema": "adjacency_ablation_split_v1",
            "folds": {
                key: {
                    role: folds[key][role]
                    for role in ("train_aliases", "validation_aliases", "test_aliases")
                }
                for key in sorted(folds)
            },
        }
    )
    graph = canonical_sha256(
        {
            "schema": "adjacency_ablation_graph_v1",
            "definition": {"k": 12, "radius_um": 50.0, "symmetry": "union"},
            "cores": {
                alias: cores[alias]["graph_bundle_checksum"]
                for alias in CORE_ALIASES
            },
        }
    )
    masks = canonical_sha256(
        {
            "schema": "adjacency_ablation_masks_v1",
            "cores": {
                alias: cores[alias]["mask_realization_checksums"]
                for alias in CORE_ALIASES
            },
        }
    )
    preprocessing = canonical_sha256(
        {
            "schema": "adjacency_ablation_preprocessing_v1",
            "folds": {
                key: {
                    "train_aliases": folds[key]["train_aliases"],
                    "standardizer_checksum": folds[key]["standardizer_checksum"],
                    "file_sha256": folds[key]["preprocessing_file_sha256"],
                }
                for key in sorted(folds)
            },
        }
    )
    return {
        "dataset": dataset,
        "split": split,
        "graph": graph,
        "masks": masks,
        "preprocessing": preprocessing,
    }


def prepare_artifact(
    *,
    selection_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise AdjacencyPreparationError(
            "prepared output already exists; verification is the only allowed reuse"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".adjacency-ablation-", dir=output_root.parent)
    )
    try:
        cores, gene_names, input_provenance = _load_source_cores(selection_path)
        core_records: dict[str, Any] = {}
        payload_files: dict[str, str] = {}
        for alias in CORE_ALIASES:
            source = cores[alias]
            data_arrays = {
                "expression_counts": source["expression_counts"],
                "coordinates_um": source["coordinates_um"],
                "fov_group": source["fov_group"],
                "qc_passed": source["qc_passed"],
            }
            data_ref = Path("cores") / f"{alias.lower()}.npz"
            _save_npz(temporary / data_ref, **data_arrays)

            bundle = materialize_fixed_adjacencies(
                source["coordinates_um"],
                source["fov_group"],
                scope=alias,
            )
            adjacency_arrays = {
                "spatial_edge_index": bundle.spatial.edge_index,
                "isolated_edge_index": bundle.isolated.edge_index,
                "position_permuted_null_edge_index": (
                    bundle.position_permuted_null.edge_index
                ),
            }
            adjacency_ref = Path("adjacencies") / f"{alias.lower()}.npz"
            _save_npz(temporary / adjacency_ref, **adjacency_arrays)

            realizations = [
                sample_uniform_mask_numpy(
                    source["expression_counts"].shape[0],
                    EXPECTED_GENES,
                    seed=derive_evaluation_mask_seed(
                        core_alias=alias, replicate_index=replicate
                    ),
                )
                for replicate in range(3)
            ]
            masks = np.stack([item.mask for item in realizations], axis=0)
            packed = np.packbits(masks, axis=-1, bitorder="little")
            masked_counts = np.stack(
                [item.masked_gene_counts for item in realizations], axis=0
            ).astype(np.uint16)
            mask_seeds = np.asarray([item.seed for item in realizations], dtype=np.uint64)
            mask_arrays = {
                "packed_masks": packed,
                "masked_counts": masked_counts,
                "mask_seeds": mask_seeds,
            }
            mask_ref = Path("masks") / f"{alias.lower()}.npz"
            _save_npz(temporary / mask_ref, **mask_arrays)

            for reference in (data_ref, adjacency_ref, mask_ref):
                payload_files[reference.as_posix()] = sha256_file(temporary / reference)
            core_records[alias] = {
                "n_cells": int(source["expression_counts"].shape[0]),
                "n_fov_groups": int(np.unique(source["fov_group"]).size),
                "qc_passed_cells": int(source["qc_passed"].sum()),
                "data_reference": data_ref.as_posix(),
                "data_file_sha256": payload_files[data_ref.as_posix()],
                "data_arrays": _array_records(data_arrays),
                "adjacency_reference": adjacency_ref.as_posix(),
                "adjacency_file_sha256": payload_files[adjacency_ref.as_posix()],
                "adjacency_arrays": _array_records(adjacency_arrays),
                "graph_bundle_checksum": bundle.checksum,
                "graph_metadata": bundle.to_metadata(),
                "evaluation_mask_reference": mask_ref.as_posix(),
                "evaluation_mask_file_sha256": payload_files[mask_ref.as_posix()],
                "mask_arrays": _array_records(mask_arrays),
                "mask_realization_checksums": [item.checksum for item in realizations],
                "mask_seeds": [int(item.seed) for item in realizations],
                "mask_bitorder": "little",
            }

        fold_records: dict[str, Any] = {}
        counts_by_alias = {
            alias: core["expression_counts"] for alias, core in cores.items()
        }
        for fold in build_five_fold_splits():
            standardizer = fit_equal_core_log1p_standardizer(
                counts_by_alias, train_aliases=fold.train_aliases
            )
            reference = Path("preprocessing") / f"fold-{fold.fold_index:02d}.npz"
            arrays = {
                "log1p_mean": standardizer.mean,
                "log1p_scale": standardizer.scale,
            }
            _save_npz(temporary / reference, **arrays)
            file_sha = sha256_file(temporary / reference)
            payload_files[reference.as_posix()] = file_sha
            fold_records[str(fold.fold_index)] = {
                **fold.to_dict(),
                "preprocessing_reference": reference.as_posix(),
                "preprocessing_file_sha256": file_sha,
                "preprocessing_arrays": _array_records(arrays),
                "standardizer_checksum": standardizer.checksum,
                "fit_scope": "train_aliases_only",
                "weighting": "equal_core_mixture",
            }

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "dataset": {
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "core_count": 10,
                "distinct_donor_count": 10,
                "cell_count": EXPECTED_CELLS,
                "fov_group_count": EXPECTED_FOV_GROUPS,
                "n_genes": EXPECTED_GENES,
                "tissue": "AdjacentNormal",
                "qc_policy": "retain_all_vendor_qc_flag_not_input",
                "qc_passed_cells": EXPECTED_QC_PASS,
            },
            "split": {
                "split_id": SPLIT_ID,
                "fold_count": 5,
                "train_groups": 7,
                "validation_groups": 1,
                "test_groups": 2,
                "independent_unit": "donor_core_one_to_one",
                "method": "slide_balanced_grouped_five_fold",
            },
            "features": {
                "gene_names": list(gene_names),
                "gene_schema_sha256": canonical_sha256(list(gene_names)),
                "technical_control_prefixes_excluded": [
                    "Negative",
                    "SystemControl",
                ],
                "model_inputs": ["masked_expression", "binary_mask_indicator"],
                "coordinates_as_model_input": False,
                "annotations_as_model_inputs": False,
            },
            "graph": {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "grouping": "raw_fov_within_core",
                "self_loops": "exactly_one_per_cell_all_arms",
                "edge_features": False,
                "arms": ["spatial", "isolated", "position_permuted_null"],
                "null_kind": "within_fov_cell_to_position_permutation",
                "null_seed": 2_026_080_202,
            },
            "masking": {
                "distribution": "exact_uniform_integer_0_through_G_per_cell",
                "positions": "without_replacement",
                "repetitions": 3,
                "base_seed": 20_260_802,
                "packed_bitorder": "little",
                "model_seed_independent": True,
                "arm_independent": True,
            },
            "preprocessing": {
                "expression": "gene_wise_standardized_log1p_raw_count",
                "fit_scope": "seven_training_cores_only",
                "weighting": "equal_core_mixture",
                "scale_floor": 1.0e-6,
                "feature_selection": "none",
                "library_size_normalization": "none",
            },
            "cores": core_records,
            "folds": fold_records,
            "files": dict(sorted(payload_files.items())),
            "input_provenance": input_provenance,
            "provenance": {
                "entry_point": "scripts/train/prepare_adjacency_ablation.py",
                "project_root": ".",
                "selection_sha256": EXPECTED_SELECTION_SHA256,
                "protected_identifiers_exported": False,
                "created_date": "2026-08-02",
            },
            "registry_contract": {
                "explicit_register_flag_required": True,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "split_id": SPLIT_ID,
            },
        }
        fingerprints = _fingerprints(manifest)
        manifest["dataset"]["dataset_fingerprint"] = fingerprints["dataset"]
        manifest["split"]["assignment_fingerprint"] = fingerprints["split"]
        manifest["graph"]["graph_fingerprint"] = fingerprints["graph"]
        manifest["masking"]["mask_fingerprint"] = fingerprints["masks"]
        manifest["preprocessing"]["preprocessing_fingerprint"] = fingerprints[
            "preprocessing"
        ]
        content_sha = canonical_sha256(_content_payload(manifest))
        manifest["content_sha256"] = content_sha
        manifest["artifact_id"] = content_sha[:16]
        _write_json(temporary / "manifest.json", manifest)

        checksum_records = {
            **payload_files,
            "manifest.json": sha256_file(temporary / "manifest.json"),
        }
        lines = [
            f"{checksum_records[relative]}  {relative}\n"
            for relative in sorted(checksum_records)
        ]
        (temporary / "checksums.sha256").write_text("".join(lines), encoding="ascii")
        verify_prepared_artifact(temporary)
        temporary.rename(output_root)
        return verify_prepared_artifact(output_root)
    except BaseException:
        if temporary.is_dir() and temporary.parent == output_root.parent:
            shutil.rmtree(temporary)
        raise


def _verify_checksum_file(root: Path, manifest: Mapping[str, Any]) -> None:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise AdjacencyPreparationError("checksums.sha256 is missing or unsafe")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise AdjacencyPreparationError("invalid checksum line") from error
        if relative in expected or len(digest) != 64 or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise AdjacencyPreparationError("invalid checksum record")
        expected[relative] = digest
    files = _mapping(manifest.get("files"), "manifest files")
    required = {"manifest.json", *files.keys()}
    if set(expected) != required:
        raise AdjacencyPreparationError("checksum file inventory mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual_files != required:
        raise AdjacencyPreparationError("prepared artifact contains unexpected files")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != digest:
            raise AdjacencyPreparationError(f"checksum mismatch: {relative}")
    if dict(files) != {
        relative: expected[relative] for relative in files
    }:
        raise AdjacencyPreparationError("manifest payload hashes differ")


def _verify_array_records(
    arrays: Mapping[str, np.ndarray], records: Mapping[str, Any], label: str
) -> None:
    if set(arrays) != set(records):
        raise AdjacencyPreparationError(f"{label} array inventory mismatch")
    for name, array in arrays.items():
        record = _mapping(records[name], f"{label}.{name}")
        if (
            list(array.shape) != record.get("shape")
            or array.dtype.str != record.get("dtype")
            or ndarray_sha256(array) != record.get("sha256")
        ):
            raise AdjacencyPreparationError(f"{label}.{name} schema/hash mismatch")


def verify_prepared_artifact(path: str | Path) -> dict[str, Any]:
    """Verify every prepared file, array, split, graph, and fixed mask."""

    supplied = Path(path)
    root = supplied.parent if supplied.is_file() else supplied
    manifest_path = root / "manifest.json"
    manifest = _strict_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != ARTIFACT_KIND
    ):
        raise AdjacencyPreparationError("unsupported prepared artifact schema")
    content_sha = canonical_sha256(_content_payload(manifest))
    if (
        manifest.get("content_sha256") != content_sha
        or manifest.get("artifact_id") != content_sha[:16]
    ):
        raise AdjacencyPreparationError("prepared content identity mismatch")
    _verify_checksum_file(root, manifest)
    cores = _mapping(manifest.get("cores"), "manifest cores")
    if set(cores) != set(CORE_ALIASES):
        raise AdjacencyPreparationError("prepared cores differ from ten aliases")
    total_cells = 0
    total_fovs = 0
    total_qc = 0
    for alias in CORE_ALIASES:
        record = _mapping(cores[alias], alias)
        data_path = root / str(record["data_reference"])
        adjacency_path = root / str(record["adjacency_reference"])
        mask_path = root / str(record["evaluation_mask_reference"])
        with np.load(data_path, allow_pickle=False) as archive:
            data = {name: np.asarray(archive[name]) for name in archive.files}
        _verify_array_records(data, _mapping(record["data_arrays"], "data arrays"), alias)
        expected_data = {"expression_counts", "coordinates_um", "fov_group", "qc_passed"}
        if set(data) != expected_data:
            raise AdjacencyPreparationError(f"{alias} data keys changed")
        n_cells = int(data["expression_counts"].shape[0])
        if (
            data["expression_counts"].shape != (n_cells, EXPECTED_GENES)
            or data["coordinates_um"].shape != (n_cells, 2)
            or data["fov_group"].shape != (n_cells,)
            or data["qc_passed"].shape != (n_cells,)
        ):
            raise AdjacencyPreparationError(f"{alias} data shapes changed")
        with np.load(adjacency_path, allow_pickle=False) as archive:
            adj = {name: np.asarray(archive[name]) for name in archive.files}
        _verify_array_records(
            adj, _mapping(record["adjacency_arrays"], "adjacency arrays"), alias
        )
        if set(adj) != {
            "spatial_edge_index",
            "isolated_edge_index",
            "position_permuted_null_edge_index",
        }:
            raise AdjacencyPreparationError(f"{alias} adjacency keys changed")
        for name, edges in adj.items():
            validate_explicit_self_adjacency(
                __import__("torch").from_numpy(np.array(edges, copy=True)),
                num_nodes=n_cells,
            )
            if np.any(data["fov_group"][edges[0]] != data["fov_group"][edges[1]]):
                raise AdjacencyPreparationError(f"{alias} {name} crosses an FOV")
        spatial_degree = np.bincount(adj["spatial_edge_index"][1], minlength=n_cells)
        null_degree = np.bincount(
            adj["position_permuted_null_edge_index"][1], minlength=n_cells
        )
        if not np.array_equal(np.sort(spatial_degree), np.sort(null_degree)):
            raise AdjacencyPreparationError(f"{alias} null degree distribution changed")
        isolated = adj["isolated_edge_index"]
        if isolated.shape != (2, n_cells) or not np.array_equal(isolated[0], isolated[1]):
            raise AdjacencyPreparationError(f"{alias} identity adjacency changed")
        with np.load(mask_path, allow_pickle=False) as archive:
            masks = {name: np.asarray(archive[name]) for name in archive.files}
        _verify_array_records(
            masks, _mapping(record["mask_arrays"], "mask arrays"), alias
        )
        if set(masks) != {"packed_masks", "masked_counts", "mask_seeds"}:
            raise AdjacencyPreparationError(f"{alias} mask keys changed")
        unpacked = np.unpackbits(
            masks["packed_masks"], axis=-1, count=EXPECTED_GENES, bitorder="little"
        ).astype(np.bool_, copy=False)
        if unpacked.shape != (3, n_cells, EXPECTED_GENES):
            raise AdjacencyPreparationError(f"{alias} unpacked mask shape changed")
        checksums: list[str] = []
        for replicate in range(3):
            seed = derive_evaluation_mask_seed(
                core_alias=alias, replicate_index=replicate
            )
            if int(masks["mask_seeds"][replicate]) != seed:
                raise AdjacencyPreparationError(f"{alias} mask seed changed")
            counts = masks["masked_counts"][replicate].astype(np.int64)
            if not np.array_equal(unpacked[replicate].sum(axis=1), counts):
                raise AdjacencyPreparationError(f"{alias} exact mask counts changed")
            checksums.append(
                mask_realization_sha256(unpacked[replicate], counts, seed=seed)
            )
        if checksums != list(record["mask_realization_checksums"]):
            raise AdjacencyPreparationError(f"{alias} mask realization changed")
        total_cells += n_cells
        total_fovs += int(np.unique(data["fov_group"]).size)
        total_qc += int(data["qc_passed"].sum())
    if (total_cells, total_fovs, total_qc) != (
        EXPECTED_CELLS,
        EXPECTED_FOV_GROUPS,
        EXPECTED_QC_PASS,
    ):
        raise AdjacencyPreparationError("aggregate prepared counts changed")

    expected_folds = build_five_fold_splits()
    folds = _mapping(manifest.get("folds"), "manifest folds")
    if set(folds) != {str(index) for index in range(5)}:
        raise AdjacencyPreparationError("fold inventory changed")
    for expected in expected_folds:
        record = _mapping(folds[str(expected.fold_index)], "fold")
        if (
            tuple(record["train_aliases"]) != expected.train_aliases
            or tuple(record["validation_aliases"]) != expected.validation_aliases
            or tuple(record["test_aliases"]) != expected.test_aliases
        ):
            raise AdjacencyPreparationError("fold assignment changed")
        path = root / str(record["preprocessing_reference"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        _verify_array_records(
            arrays,
            _mapping(record["preprocessing_arrays"], "preprocessing arrays"),
            f"fold {expected.fold_index}",
        )
        if set(arrays) != {"log1p_mean", "log1p_scale"} or any(
            value.shape != (EXPECTED_GENES,) for value in arrays.values()
        ):
            raise AdjacencyPreparationError("preprocessing arrays changed")
        if not np.isfinite(arrays["log1p_mean"]).all() or np.any(
            arrays["log1p_scale"] < 1e-6
        ):
            raise AdjacencyPreparationError("preprocessing values are invalid")
    fingerprints = _fingerprints(manifest)
    if (
        manifest["dataset"].get("dataset_fingerprint") != fingerprints["dataset"]
        or manifest["split"].get("assignment_fingerprint") != fingerprints["split"]
        or manifest["graph"].get("graph_fingerprint") != fingerprints["graph"]
        or manifest["masking"].get("mask_fingerprint") != fingerprints["masks"]
        or manifest["preprocessing"].get("preprocessing_fingerprint")
        != fingerprints["preprocessing"]
    ):
        raise AdjacencyPreparationError("scientific fingerprint mismatch")
    return manifest


def register_prepared_artifact(
    manifest: Mapping[str, Any], *, output_root: Path, database_path: Path
) -> None:
    dataset = _mapping(manifest["dataset"], "dataset")
    split = _mapping(manifest["split"], "split")
    registry = Registry(database_path)
    registry.initialize()
    registry.register_dataset(
        DATASET_ID,
        DATASET_VERSION,
        display_name="Ten donor/core Adjacent Normal grouped adjacency ablation",
        protected_source_path=_project_reference(output_root),
        raw_fingerprint=EXPECTED_SELECTION_SHA256,
        preprocessing_version=str(
            _mapping(manifest["preprocessing"], "preprocessing")[
                "preprocessing_fingerprint"
            ]
        ),
        processed_fingerprint=str(dataset["dataset_fingerprint"]),
        aggregate_sample_count=EXPECTED_CELLS,
        graph_count=30,
        node_feature_schema="masked_expression_plus_binary_mask_only",
        edge_feature_schema="none_fixed_adjacency_only",
        creation_date="2026-08-02",
        status="available",
        verification_status="verified_immutable_prepared_artifact",
        metadata={
            "core_count": 10,
            "distinct_donor_count": 10,
            "fov_group_count": EXPECTED_FOV_GROUPS,
            "artifact_id": manifest["artifact_id"],
            "content_sha256": manifest["content_sha256"],
        },
    )
    registry.register_split(
        SPLIT_ID,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        method="slide_balanced_grouped_five_fold",
        unit="donor_core_one_to_one",
        seed=None,
        fold_count=5,
        stratification=["slide_block"],
        fingerprint=str(split["assignment_fingerprint"]),
        protected_path=_project_reference(output_root),
        verification_status="verified_group_disjoint_7_1_2",
        metadata={
            "test_every_group_once": True,
            "preprocessing_fit_scope": "train_groups_only",
            "folds": manifest["folds"],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=paths.project_root / EXPECTED_SELECTION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.project_root / DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if args.verify_only or output.exists():
        manifest = verify_prepared_artifact(output)
        if output.exists() and not (args.verify_only or args.register):
            raise AdjacencyPreparationError(
                "output exists; use --verify-only or --register, never overwrite"
            )
    else:
        manifest = prepare_artifact(
            selection_path=args.selection.resolve(), output_root=output
        )
    if args.register:
        register_prepared_artifact(
            manifest, output_root=output, database_path=args.database.resolve()
        )
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "content_sha256": manifest["content_sha256"],
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "split_id": SPLIT_ID,
                "registered": bool(args.register),
                "output": str(output),
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

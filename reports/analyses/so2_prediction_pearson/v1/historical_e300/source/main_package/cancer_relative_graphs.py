"""Immutable radial-graph and relative-geometry caches for six Cancer cores.

The high-dimensional relative feature cache is written exactly once per core
and is shared read-only by all model seeds.  It is not copied into run bundles.
This is an execution cache: coordinates, orientation tensors, canonical edges,
and locked RBF parameters remain sufficient to reproduce every cached value.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .cancer_pooled_full_core import CANCER_ALIASES
from .relative_geometry import (
    DEFAULT_MAX_DISTANCE_UM,
    DEFAULT_ORIENTATION_NEIGHBOR_CAP,
    DEFAULT_ORIENTATION_RADIUS_UM,
    DEFAULT_ORIENTATION_SIGMA_UM,
    DEFAULT_RADIAL_SHELL_BOUNDS_UM,
    DEFAULT_RADIAL_SHELL_QUOTAS,
    DEFAULT_RBF_COUNT,
    DEFAULT_RBF_WIDTH_UM,
    RELATIVE_GEOMETRY_DIM,
    RELATIVE_GEOMETRY_FEATURE_NAMES,
    iter_invariant_relative_feature_shards,
    local_orientation_tensors,
    radial_stratified_knn,
)


GRAPH_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_RECEIVER_CHUNK_SIZE = 512
DEFAULT_MAX_EDGES_PER_CHUNK = 200_000


class CancerRelativeGraphContractError(ValueError):
    """Raised when a graph cache would violate the six-core contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CancerRelativeGraphContractError(
            f"Cannot read manifest: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CancerRelativeGraphContractError("Manifest must be a mapping.")
    return value


def _verify_cohort_manifest(cohort_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(cohort_dir / "manifest.json")
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise CancerRelativeGraphContractError("Cohort manifest lacks cohort metadata.")
    if tuple(cohort.get("aliases", ())) != CANCER_ALIASES:
        raise CancerRelativeGraphContractError(
            "Cohort artifact does not contain the exact ordered six CAN aliases."
        )
    if cohort.get("validation_or_test_partition_present") is not False:
        raise CancerRelativeGraphContractError(
            "Cancer relative graphs require the fit-only cohort."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CancerRelativeGraphContractError("Cohort manifest lacks file checksums.")
    for alias in CANCER_ALIASES:
        relative = f"cores/{alias}.npz"
        expected = files.get(relative)
        source = cohort_dir / relative
        if not isinstance(expected, str) or _sha256_file(source) != expected:
            raise CancerRelativeGraphContractError(
                f"Prepared cohort checksum mismatch for {alias}."
            )
    return manifest


def _verify_graph_collection(
    graph_root: Path,
    *,
    cohort_manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed on graph/cache drift before any model can use the files."""

    manifest = _load_manifest(graph_root / "manifest.json")
    expected_manifest_checksum = manifest.get("manifest_content_sha256")
    checksum_payload = dict(manifest)
    checksum_payload.pop("manifest_content_sha256", None)
    if expected_manifest_checksum != _payload_sha256(checksum_payload):
        raise CancerRelativeGraphContractError(
            "Graph collection manifest content checksum mismatch."
        )
    if tuple(manifest.get("aliases", ())) != CANCER_ALIASES:
        raise CancerRelativeGraphContractError("Graph manifest aliases are invalid.")
    if manifest.get("cohort_manifest_sha256") != _sha256_file(
        cohort_manifest_path
    ):
        raise CancerRelativeGraphContractError(
            "Graph collection was built from a different cohort manifest."
        )
    records = manifest.get("cores")
    if not isinstance(records, list) or tuple(
        record.get("alias") if isinstance(record, Mapping) else None
        for record in records
    ) != CANCER_ALIASES:
        raise CancerRelativeGraphContractError(
            "Graph collection core records do not match the locked order."
        )
    for record in records:
        assert isinstance(record, Mapping)
        alias = str(record["alias"])
        record_payload = dict(record)
        record_checksum = record_payload.pop("record_sha256", None)
        if record_checksum != _payload_sha256(record_payload):
            raise CancerRelativeGraphContractError(
                f"Graph record checksum mismatch for {alias}."
            )
        graph = record.get("graph")
        relative = record.get("relative_geometry")
        files = record.get("files")
        if not all(isinstance(value, Mapping) for value in (graph, relative, files)):
            raise CancerRelativeGraphContractError(
                f"Graph record sections are incomplete for {alias}."
            )
        assert isinstance(graph, Mapping)
        assert isinstance(relative, Mapping)
        assert isinstance(files, Mapping)
        qc = graph.get("qc")
        eligible_fraction_valid = bool(
            isinstance(qc, Mapping)
            and (
                (
                    int(qc.get("coverage_eligible_cells", -1)) == 0
                    and qc.get("eligible_long_range_coverage_fraction") is None
                )
                or qc.get("eligible_long_range_coverage_fraction") == 1.0
            )
        )
        if not isinstance(qc, Mapping) or any(
            (
                qc.get("self_loops") != 0,
                qc.get("cross_group_edges") != 0,
                qc.get("directed_edge_pairs_are_symmetric") is not True,
                qc.get("receiver_major_canonical_order") is not True,
                qc.get("long_range_coverage_gate_passed") is not True,
                not eligible_fraction_valid,
            )
        ):
            raise CancerRelativeGraphContractError(
                f"Graph scientific gate failed for {alias}."
            )
        core_root = graph_root / "cores" / alias
        for filename in ("edge_index.npy", "orientation.npz", "relative_geometry.npy"):
            expected = files.get(filename)
            if not isinstance(expected, str) or _sha256_file(core_root / filename) != expected:
                raise CancerRelativeGraphContractError(
                    f"Graph cache file checksum mismatch for {alias}/{filename}."
                )
        expected_shape = relative.get("cache_shape")
        if expected_shape != [int(qc["n_directed_edges"]), RELATIVE_GEOMETRY_DIM]:
            raise CancerRelativeGraphContractError(
                f"Relative-geometry shape receipt is invalid for {alias}."
            )
    return manifest


def _relative_parameter_record() -> dict[str, Any]:
    return {
        "schema": "relative_geometry_invariant_v1",
        "feature_dimension": RELATIVE_GEOMETRY_DIM,
        "feature_names": list(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "radial_rbf_count": DEFAULT_RBF_COUNT,
        "radial_rbf_centers_um": np.linspace(
            0.0, DEFAULT_MAX_DISTANCE_UM, DEFAULT_RBF_COUNT
        ).tolist(),
        "radial_rbf_width_um": DEFAULT_RBF_WIDTH_UM,
        "smooth_cutoff": "0.5*(cos(pi*r/500)+1)",
        "maximum_distance_um": DEFAULT_MAX_DISTANCE_UM,
        "orientation_radius_um": DEFAULT_ORIENTATION_RADIUS_UM,
        "orientation_neighbor_cap": DEFAULT_ORIENTATION_NEIGHBOR_CAP,
        "orientation_sigma_um": DEFAULT_ORIENTATION_SIGMA_UM,
        "geometry_role": "attention_logit_bias_only",
    }


def relative_geometry_parameter_record() -> dict[str, Any]:
    """Return a defensive copy of the locked relative-geometry parameters."""

    return deepcopy(_relative_parameter_record())


def verify_cancer_relative_graph_collection(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
) -> dict[str, Any]:
    """Public read-only verifier used before safe cross-cohort cache reuse."""

    cohort_root = Path(cohort_dir)
    _verify_cohort_manifest(cohort_root)
    return _verify_graph_collection(
        Path(graph_dir),
        cohort_manifest_path=cohort_root / "manifest.json",
    )


def materialize_relative_core_graph(
    *,
    alias: str,
    coordinates_um: np.ndarray,
    output_dir: str | Path,
    allowed_aliases: tuple[str, ...],
    artifact_kind: str,
    contract_error: type[ValueError] = ValueError,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
) -> dict[str, Any]:
    """Write one canonical graph plus one shared read-only feature cache.

    This cohort-neutral implementation is shared by the historical six-Cancer
    wrapper and additive cohorts.  Alias allow-lists and artifact kinds remain
    explicit so one cohort can never silently consume another cohort's cache.
    """

    canonical_alias = str(alias).strip().upper()
    canonical_allowed = tuple(str(value).strip().upper() for value in allowed_aliases)
    if not canonical_allowed or len(set(canonical_allowed)) != len(canonical_allowed):
        raise contract_error("Graph alias allow-list must be nonempty and unique.")
    if canonical_alias not in canonical_allowed:
        raise contract_error("Unknown core alias for this graph collection.")
    clean_artifact_kind = str(artifact_kind).strip()
    if not clean_artifact_kind:
        raise contract_error("Graph artifact kind must be nonempty.")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Graph cache already exists and will not be overwritten: {destination}"
        )
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise contract_error(
            "coordinates_um must have shape [n_cells, 2]."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        graph = radial_stratified_knn(
            coordinates,
            shell_bounds_um=DEFAULT_RADIAL_SHELL_BOUNDS_UM,
            shell_quotas=DEFAULT_RADIAL_SHELL_QUOTAS,
            receiver_query_chunk_size=receiver_chunk_size,
            long_range_coverage_gate_min_fraction=1.0,
            fail_on_coverage_gate=True,
        )
        orientation = local_orientation_tensors(
            coordinates,
            radius_um=DEFAULT_ORIENTATION_RADIUS_UM,
            neighbor_cap=DEFAULT_ORIENTATION_NEIGHBOR_CAP,
            sigma_um=DEFAULT_ORIENTATION_SIGMA_UM,
            receiver_query_chunk_size=receiver_chunk_size,
        )

        edge_path = temporary / "edge_index.npy"
        np.save(edge_path, graph.edge_index.astype(np.int64, copy=False))
        orientation_path = temporary / "orientation.npz"
        np.savez_compressed(
            orientation_path,
            tensors=orientation.tensors,
            anisotropy=orientation.anisotropy,
            selected_neighbor_counts=orientation.selected_neighbor_counts,
        )

        feature_path = temporary / "relative_geometry.npy"
        feature_cache = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(graph.edge_index.shape[1], RELATIVE_GEOMETRY_DIM),
        )
        raw_feature_digest = hashlib.sha256()
        raw_feature_digest.update(
            _canonical_json(
                {
                    "name": "relative_geometry",
                    "shape": list(feature_cache.shape),
                    "dtype": np.dtype(np.float32).str,
                }
            )
        )
        observed_edges = 0
        shard_checksums: list[str] = []
        for shard in iter_invariant_relative_feature_shards(
            coordinates,
            graph.edge_index,
            orientation,
            receiver_chunk_size=receiver_chunk_size,
            max_edges_per_chunk=max_edges_per_chunk,
            output_dtype=np.float32,
        ):
            edge_count = shard.n_edges
            feature_cache[observed_edges : observed_edges + edge_count] = shard.features
            raw_feature_digest.update(
                memoryview(np.ascontiguousarray(shard.features)).cast("B")
            )
            observed_edges += edge_count
            shard_checksums.append(shard.checksum_sha256)
        if observed_edges != graph.edge_index.shape[1]:
            raise contract_error(
                "Relative feature cache did not cover every canonical edge."
            )
        feature_cache.flush()
        del feature_cache

        parameters = _relative_parameter_record()
        logical_feature_sha256 = raw_feature_digest.hexdigest()
        relative_geometry_sha256 = _payload_sha256(
            {
                "graph_sha256": graph.checksums.graph_sha256,
                "orientation_sha256": orientation.checksum_sha256,
                "parameters_sha256": _payload_sha256(parameters),
                "logical_feature_sha256": logical_feature_sha256,
                "shard_checksums_sha256": _payload_sha256(shard_checksums),
            }
        )
        files = {
            "edge_index.npy": _sha256_file(edge_path),
            "orientation.npz": _sha256_file(orientation_path),
            "relative_geometry.npy": _sha256_file(feature_path),
        }
        record: dict[str, Any] = {
            "artifact_kind": clean_artifact_kind,
            "format_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
            "alias": canonical_alias,
            "n_cells": int(len(coordinates)),
            "graph": {
                "shell_bounds_um": list(DEFAULT_RADIAL_SHELL_BOUNDS_UM),
                "shell_quotas": list(DEFAULT_RADIAL_SHELL_QUOTAS),
                "nominal_pre_symmetrization_degree": int(graph.nominal_k),
                "maximum_distance_um": DEFAULT_MAX_DISTANCE_UM,
                "symmetry": "bidirectional_union",
                "self_loops": False,
                "qc": graph.qc.to_dict(),
                "checksums": graph.checksums.to_dict(),
            },
            "orientation": {
                "radius_um": orientation.radius_um,
                "neighbor_cap": orientation.neighbor_cap,
                "sigma_um": orientation.sigma_um,
                "minimum_neighbors": orientation.minimum_neighbors,
                "insufficient_neighbor_cells": orientation.insufficient_neighbor_cells,
                "checksum_sha256": orientation.checksum_sha256,
            },
            "relative_geometry": {
                "parameters": parameters,
                "parameters_sha256": _payload_sha256(parameters),
                "logical_feature_sha256": logical_feature_sha256,
                "shard_checksums_sha256": _payload_sha256(shard_checksums),
                "relative_geometry_sha256": relative_geometry_sha256,
                "cache_shape": [int(graph.edge_index.shape[1]), RELATIVE_GEOMETRY_DIM],
                "cache_dtype": "float32",
                "cache_policy": "one_shared_read_only_memory_map_per_core",
                "reproducible_from_minimal_geometry": True,
                "copied_into_per_seed_run_bundles": False,
            },
            "files": files,
        }
        record["record_sha256"] = _payload_sha256(record)
        (temporary / "manifest.json").write_bytes(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        temporary.rename(destination)
        return record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def materialize_cancer_core_relative_graph(
    *,
    alias: str,
    coordinates_um: np.ndarray,
    output_dir: str | Path,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
) -> dict[str, Any]:
    """Backward-compatible six-Cancer-core materialization wrapper."""

    canonical_alias = str(alias).strip().upper()
    if canonical_alias not in CANCER_ALIASES:
        raise CancerRelativeGraphContractError("Unknown Cancer core alias.")
    return materialize_relative_core_graph(
        alias=canonical_alias,
        coordinates_um=coordinates_um,
        output_dir=output_dir,
        allowed_aliases=CANCER_ALIASES,
        artifact_kind="cancer_core_radial_relative_geometry",
        contract_error=CancerRelativeGraphContractError,
        receiver_chunk_size=receiver_chunk_size,
        max_edges_per_chunk=max_edges_per_chunk,
    )


def prepare_cancer_6core_relative_graphs(
    *,
    cohort_dir: str | Path,
    output_dir: str | Path,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
) -> dict[str, Any]:
    """Materialize all six graph caches and a graph-complete cohort manifest."""

    cohort_root = Path(cohort_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            "Six-core graph destination exists; immutable caches are not overwritten."
        )
    cohort_manifest = _verify_cohort_manifest(cohort_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        core_records: list[dict[str, Any]] = []
        for alias in CANCER_ALIASES:
            with np.load(cohort_root / "cores" / f"{alias}.npz", allow_pickle=False) as data:
                coordinates = np.asarray(data["coordinates_um"], dtype=np.float64)
            record = materialize_cancer_core_relative_graph(
                alias=alias,
                coordinates_um=coordinates,
                output_dir=staging / "cores" / alias,
                receiver_chunk_size=receiver_chunk_size,
                max_edges_per_chunk=max_edges_per_chunk,
            )
            core_records.append(record)

        completed_cohort = deepcopy(cohort_manifest)
        by_alias = {record["alias"]: record for record in core_records}
        for core in completed_cohort.get("cores", []):
            alias = str(core.get("alias", ""))
            if alias not in by_alias:
                raise CancerRelativeGraphContractError(
                    "Cohort core manifest does not align to graph aliases."
                )
            core["graph_checksum"] = by_alias[alias]["graph"]["checksums"][
                "graph_sha256"
            ]
            core["relative_geometry_checksum"] = by_alias[alias][
                "relative_geometry"
            ]["relative_geometry_sha256"]
        completed_cohort.pop("manifest_content_sha256", None)
        completed_cohort["manifest_content_sha256"] = _payload_sha256(
            completed_cohort
        )
        completed_path = staging / "cohort_manifest_with_graphs.json"
        completed_path.write_bytes(
            json.dumps(
                completed_cohort, sort_keys=True, indent=2, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )

        manifest: dict[str, Any] = {
            "artifact_kind": "cancer_6core_relative_graph_collection",
            "format_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
            "aliases": list(CANCER_ALIASES),
            "cohort_manifest_sha256": _sha256_file(cohort_root / "manifest.json"),
            "completed_cohort_manifest_sha256": _sha256_file(completed_path),
            "receiver_chunk_size": int(receiver_chunk_size),
            "max_edges_per_chunk": int(max_edges_per_chunk),
            "cores": core_records,
        }
        manifest["manifest_content_sha256"] = _payload_sha256(manifest)
        (staging / "manifest.json").write_bytes(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_cancer_relative_qkv_batches(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
) -> tuple[Any, ...]:
    """Open the six CPU-resident training batches with shared mmap geometry."""

    from .pooled_relative_qkv_training import PooledRelativeQKVCoreBatch

    cohort_root = Path(cohort_dir)
    graph_root = Path(graph_dir)
    cohort_manifest = _verify_cohort_manifest(cohort_root)
    graph_manifest = _verify_graph_collection(
        graph_root,
        cohort_manifest_path=cohort_root / "manifest.json",
    )
    records_by_alias = {
        str(record["alias"]): record for record in graph_manifest["cores"]
    }
    batches = []
    for alias in CANCER_ALIASES:
        with np.load(cohort_root / "cores" / f"{alias}.npz", allow_pickle=False) as data:
            target = np.array(data["target_expression"], dtype=np.float32, copy=True)
            covariates = np.array(data["node_covariates"], dtype=np.float32, copy=True)
        edge_map = np.load(
            graph_root / "cores" / alias / "edge_index.npy", mmap_mode="r"
        )
        geometry_map = np.load(
            graph_root / "cores" / alias / "relative_geometry.npy", mmap_mode="r"
        )
        record = records_by_alias[alias]
        expected_nodes = next(
            int(core["cell_count"])
            for core in cohort_manifest["cores"]
            if core["alias"] == alias
        )
        expected_edges = int(record["graph"]["qc"]["n_directed_edges"])
        if target.shape != (expected_nodes, 1_000):
            raise CancerRelativeGraphContractError(
                f"Prepared target shape changed for {alias}."
            )
        if edge_map.shape != (2, expected_edges) or geometry_map.shape != (
            expected_edges,
            RELATIVE_GEOMETRY_DIM,
        ):
            raise CancerRelativeGraphContractError(
                f"Graph/cache array shape changed for {alias}."
            )
        # Read-only NumPy memory maps are safe because the trainer/model never
        # mutates graph inputs.  ``torch.from_numpy`` preserves shared pages.
        edge_tensor = torch.from_numpy(edge_map)
        geometry_tensor = torch.from_numpy(geometry_map)
        batches.append(
            PooledRelativeQKVCoreBatch(
                alias=alias,
                target_expression=torch.from_numpy(target),
                edge_index=edge_tensor,
                relative_geometry=geometry_tensor,
                node_covariates=torch.from_numpy(covariates),
            )
        )
    return tuple(batches)


__all__ = [
    "CancerRelativeGraphContractError",
    "DEFAULT_MAX_EDGES_PER_CHUNK",
    "DEFAULT_RECEIVER_CHUNK_SIZE",
    "GRAPH_ARTIFACT_SCHEMA_VERSION",
    "load_cancer_relative_qkv_batches",
    "materialize_cancer_core_relative_graph",
    "materialize_relative_core_graph",
    "prepare_cancer_6core_relative_graphs",
    "relative_geometry_parameter_record",
    "verify_cancer_relative_graph_collection",
]

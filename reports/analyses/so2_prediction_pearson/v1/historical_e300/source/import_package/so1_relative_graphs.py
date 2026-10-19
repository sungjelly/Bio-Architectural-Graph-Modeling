"""Radial graphs and relative-geometry caches for SO_1 cores 1--14.

Each SO1 alias owns one disconnected graph.  Graph construction delegates to
the same deterministic radial-stratified implementation used by the six-core
and SO2 campaigns.  Cancer-campaign caches for cores 1, 9, and 13 may be
hard-linked, but only after the source collection, ordered coordinates, graph
parameters, relative-geometry parameters, and every linked file checksum have
all been verified.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .cancer_relative_graphs import (
    DEFAULT_MAX_EDGES_PER_CHUNK,
    DEFAULT_RECEIVER_CHUNK_SIZE,
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    materialize_relative_core_graph,
    relative_geometry_parameter_record,
    verify_cancer_relative_graph_collection,
)
from .relative_geometry import (
    DEFAULT_MAX_DISTANCE_UM,
    DEFAULT_RADIAL_SHELL_BOUNDS_UM,
    DEFAULT_RADIAL_SHELL_QUOTAS,
    RELATIVE_GEOMETRY_DIM,
)
from .so1_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_CELLS,
    SO1_ALIASES,
    SO1_CORE_NUMBERS,
    SO1_SOURCE_SLIDE,
)


class SO1RelativeGraphContractError(ValueError):
    """Raised when an SO1 graph cache violates the locked cohort contract."""


_REUSABLE_CANCER_ALIAS_BY_SO1 = {
    "SO1-C01": "CAN-01",
    "SO1-C09": "CAN-09",
    "SO1-C13": "CAN-13",
}


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


def _coordinate_content_sha256(coordinates: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(b"coordinates_um\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(values.shape)))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SO1RelativeGraphContractError(f"Cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise SO1RelativeGraphContractError("Manifest must be a mapping.")
    return value


def _verify_so1_cohort_manifest(cohort_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(cohort_dir / "manifest.json")
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise SO1RelativeGraphContractError("Cohort manifest lacks cohort metadata.")
    if tuple(cohort.get("aliases", ())) != SO1_ALIASES:
        raise SO1RelativeGraphContractError(
            "Cohort artifact does not contain exact ordered SO1-C01--SO1-C14 aliases."
        )
    if int(cohort.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS:
        raise SO1RelativeGraphContractError(
            "SO_1 cohort manifest does not contain the locked 161,596 cells."
        )
    if cohort.get("source_slide") != SO1_SOURCE_SLIDE:
        raise SO1RelativeGraphContractError("SO1 cohort source slide is invalid.")
    if cohort.get("validation_or_test_partition_present") is not False:
        raise SO1RelativeGraphContractError(
            "SO1 relative graphs require the fit-only cohort."
        )

    audit = manifest.get("routing_audit")
    expected_per_core = {
        str(core): int(EXPECTED_CELL_COUNTS_BY_CORE[core])
        for core in SO1_CORE_NUMBERS
    }
    if not isinstance(audit, Mapping) or any(
        (
            audit.get("source_slide") != SO1_SOURCE_SLIDE,
            audit.get("mapped_core_numbers") != list(SO1_CORE_NUMBERS),
            int(audit.get("mapped_fov_count", -1)) != 205,
            int(audit.get("selected_cell_count", -1)) != EXPECTED_TOTAL_CELLS,
            audit.get("per_core_cell_counts") != expected_per_core,
            int(audit.get("raw_slide_cell_count", -1)) != EXPECTED_TOTAL_CELLS,
            audit.get("selection_is_slide_qualified") is not True,
            audit.get("all_raw_fovs_mapped") is not True,
            audit.get("unmapped_fovs") != [],
            int(audit.get("unmapped_fov_count", -1)) != 0,
            int(audit.get("unmapped_fov_cell_count", -1)) != 0,
            audit.get("unmapped_fov_entered_prepared_arrays") is not False,
        )
    ):
        raise SO1RelativeGraphContractError(
            "SO1 cohort lacks the required complete FOV-routing audit."
        )

    features = manifest.get("features")
    if not isinstance(features, Mapping) or any(
        (
            int(features.get("n_biological_probes", -1)) != EXPECTED_N_GENES,
            features.get("coordinates_are_model_covariates") is not False,
            features.get("routing_keys_are_model_covariates") is not False,
            features.get("core_alias_is_model_covariate") is not False,
            features.get("slide_identity_is_model_covariate") is not False,
            features.get("library_size_is_model_covariate") is not False,
        )
    ):
        raise SO1RelativeGraphContractError(
            "SO1 cohort feature and leakage assurances are incomplete."
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO1RelativeGraphContractError("Cohort manifest lacks file checksums.")
    for alias in SO1_ALIASES:
        relative = f"cores/{alias}.npz"
        expected = files.get(relative)
        source = cohort_dir / relative
        if not isinstance(expected, str) or _sha256_file(source) != expected:
            raise SO1RelativeGraphContractError(
                f"Prepared cohort checksum mismatch for {alias}."
            )
    return manifest


def _verify_so1_graph_collection(
    graph_root: Path,
    *,
    cohort_manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(graph_root / "manifest.json")
    expected_manifest_checksum = manifest.get("manifest_content_sha256")
    checksum_payload = dict(manifest)
    checksum_payload.pop("manifest_content_sha256", None)
    if expected_manifest_checksum != _payload_sha256(checksum_payload):
        raise SO1RelativeGraphContractError(
            "Graph collection manifest content checksum mismatch."
        )
    if tuple(manifest.get("aliases", ())) != SO1_ALIASES:
        raise SO1RelativeGraphContractError("Graph manifest aliases are invalid.")
    if manifest.get("cohort_manifest_sha256") != _sha256_file(
        cohort_manifest_path
    ):
        raise SO1RelativeGraphContractError(
            "Graph collection was built from a different cohort manifest."
        )
    records = manifest.get("cores")
    if not isinstance(records, list) or tuple(
        record.get("alias") if isinstance(record, Mapping) else None
        for record in records
    ) != SO1_ALIASES:
        raise SO1RelativeGraphContractError(
            "Graph collection records do not match the locked SO1 order."
        )
    for record in records:
        assert isinstance(record, Mapping)
        alias = str(record["alias"])
        record_payload = dict(record)
        record_checksum = record_payload.pop("record_sha256", None)
        if record_checksum != _payload_sha256(record_payload):
            raise SO1RelativeGraphContractError(
                f"Graph record checksum mismatch for {alias}."
            )
        graph = record.get("graph")
        relative = record.get("relative_geometry")
        files = record.get("files")
        if not all(isinstance(value, Mapping) for value in (graph, relative, files)):
            raise SO1RelativeGraphContractError(
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
            raise SO1RelativeGraphContractError(
                f"Graph scientific gate failed for {alias}."
            )
        core_root = graph_root / "cores" / alias
        for filename in (
            "edge_index.npy",
            "orientation.npz",
            "relative_geometry.npy",
        ):
            expected = files.get(filename)
            if not isinstance(expected, str) or _sha256_file(
                core_root / filename
            ) != expected:
                raise SO1RelativeGraphContractError(
                    f"Graph cache checksum mismatch for {alias}/{filename}."
                )
        expected_shape = relative.get("cache_shape")
        if expected_shape != [int(qc["n_directed_edges"]), RELATIVE_GEOMETRY_DIM]:
            raise SO1RelativeGraphContractError(
                f"Relative-geometry shape receipt is invalid for {alias}."
            )
    return manifest


def materialize_so1_core_relative_graph(
    *,
    alias: str,
    coordinates_um: np.ndarray,
    output_dir: str | Path,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
) -> dict[str, Any]:
    """Write one SO1 core's canonical graph and shared geometry cache."""

    return materialize_relative_core_graph(
        alias=alias,
        coordinates_um=coordinates_um,
        output_dir=output_dir,
        allowed_aliases=SO1_ALIASES,
        artifact_kind="so1_core_radial_relative_geometry",
        contract_error=SO1RelativeGraphContractError,
        receiver_chunk_size=receiver_chunk_size,
        max_edges_per_chunk=max_edges_per_chunk,
    )


def _reuse_verified_cancer_graph(
    *,
    alias: str,
    source_alias: str,
    cohort_root: Path,
    source_cohort_root: Path,
    source_graph_root: Path,
    source_graph_manifest: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hard-link a cancer graph only after exact content/parameter checks."""

    source_record = next(
        (
            record
            for record in source_graph_manifest["cores"]
            if record["alias"] == source_alias
        ),
        None,
    )
    if not isinstance(source_record, Mapping):
        raise SO1RelativeGraphContractError(
            f"Verified source graph lacks {source_alias}."
        )
    with np.load(
        cohort_root / "cores" / f"{alias}.npz", allow_pickle=False
    ) as current_data, np.load(
        source_cohort_root / "cores" / f"{source_alias}.npz", allow_pickle=False
    ) as source_data:
        current_coordinates = np.asarray(
            current_data["coordinates_um"], dtype=np.float64
        )
        source_coordinates = np.asarray(
            source_data["coordinates_um"], dtype=np.float64
        )
        if current_coordinates.shape != source_coordinates.shape or not np.array_equal(
            current_coordinates, source_coordinates
        ):
            raise SO1RelativeGraphContractError(
                f"Ordered coordinates differ between {alias} and {source_alias}."
            )
        coordinate_sha256 = _coordinate_content_sha256(current_coordinates)

    current_parameters = relative_geometry_parameter_record()
    relative = source_record.get("relative_geometry")
    graph = source_record.get("graph")
    if not isinstance(relative, Mapping) or not isinstance(graph, Mapping):
        raise SO1RelativeGraphContractError(
            f"Source graph record is incomplete for {source_alias}."
        )
    if relative.get("parameters") != current_parameters or relative.get(
        "parameters_sha256"
    ) != _payload_sha256(current_parameters):
        raise SO1RelativeGraphContractError(
            f"Relative-geometry parameters changed for {source_alias}."
        )
    if any(
        (
            graph.get("shell_bounds_um") != list(DEFAULT_RADIAL_SHELL_BOUNDS_UM),
            graph.get("shell_quotas") != list(DEFAULT_RADIAL_SHELL_QUOTAS),
            graph.get("maximum_distance_um") != DEFAULT_MAX_DISTANCE_UM,
            graph.get("symmetry") != "bidirectional_union",
            graph.get("self_loops") is not False,
        )
    ):
        raise SO1RelativeGraphContractError(
            f"Radial graph parameters changed for {source_alias}."
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        source_core_root = source_graph_root / "cores" / source_alias
        files = source_record.get("files")
        if not isinstance(files, Mapping):
            raise SO1RelativeGraphContractError(
                f"Source file receipts are missing for {source_alias}."
            )
        for filename in (
            "edge_index.npy",
            "orientation.npz",
            "relative_geometry.npy",
        ):
            source_path = source_core_root / filename
            target_path = output_dir / filename
            expected_file_sha256 = files.get(filename)
            if not isinstance(expected_file_sha256, str) or _sha256_file(
                source_path
            ) != expected_file_sha256:
                raise SO1RelativeGraphContractError(
                    f"Source cache receipt mismatch for {source_alias}/{filename}."
                )
            os.link(source_path, target_path)
            source_stat = source_path.stat()
            target_stat = target_path.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                target_stat.st_dev,
                target_stat.st_ino,
            ):
                raise SO1RelativeGraphContractError(
                    f"Cache reuse did not create a hard link for {alias}/{filename}."
                )
            if _sha256_file(target_path) != expected_file_sha256:
                raise SO1RelativeGraphContractError(
                    f"Linked cache checksum drifted for {alias}/{filename}."
                )

        record = deepcopy(dict(source_record))
        record["artifact_kind"] = "so1_core_radial_relative_geometry"
        record["alias"] = alias
        record["reuse_provenance"] = {
            "reuse_method": "checksum_verified_hard_link",
            "source_alias": source_alias,
            "source_record_sha256": source_record["record_sha256"],
            "source_collection_manifest_sha256": _sha256_file(
                source_graph_root / "manifest.json"
            ),
            "ordered_coordinates_sha256": coordinate_sha256,
            "relative_parameters_sha256": relative["parameters_sha256"],
            "hard_links_preserve_content_after_source_unlink": True,
        }
        record.pop("record_sha256", None)
        record["record_sha256"] = _payload_sha256(record)
        (output_dir / "manifest.json").write_bytes(
            json.dumps(
                record, sort_keys=True, indent=2, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        return record, {
            "alias": alias,
            "status": "reused",
            "source_alias": source_alias,
            "method": "checksum_verified_hard_link",
            "ordered_coordinates_sha256": coordinate_sha256,
            "relative_parameters_sha256": relative["parameters_sha256"],
        }
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def prepare_so1_14core_relative_graphs(
    *,
    cohort_dir: str | Path,
    output_dir: str | Path,
    receiver_chunk_size: int = DEFAULT_RECEIVER_CHUNK_SIZE,
    max_edges_per_chunk: int = DEFAULT_MAX_EDGES_PER_CHUNK,
    reuse_cancer_cohort_dir: str | Path | None = None,
    reuse_cancer_graph_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize all 14 disconnected SO1 graphs and geometry caches."""

    cohort_root = Path(cohort_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            "SO1 14-core graph destination exists; caches are not overwritten."
        )
    cohort_manifest = _verify_so1_cohort_manifest(cohort_root)
    if (reuse_cancer_cohort_dir is None) != (reuse_cancer_graph_dir is None):
        raise SO1RelativeGraphContractError(
            "Cancer cohort and graph reuse paths must be supplied together."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        source_cohort_root: Path | None = None
        source_graph_root: Path | None = None
        source_graph_manifest: Mapping[str, Any] | None = None
        reuse_audit: dict[str, Any] = {
            "requested": reuse_cancer_cohort_dir is not None,
            "eligible_aliases": sorted(_REUSABLE_CANCER_ALIAS_BY_SO1),
            "results": [],
        }
        if reuse_cancer_cohort_dir is not None and reuse_cancer_graph_dir is not None:
            source_cohort_root = Path(reuse_cancer_cohort_dir)
            source_graph_root = Path(reuse_cancer_graph_dir)
            try:
                source_graph_manifest = verify_cancer_relative_graph_collection(
                    cohort_dir=source_cohort_root,
                    graph_dir=source_graph_root,
                )
                reuse_audit["source_collection_status"] = "verified"
                reuse_audit["source_collection_manifest_sha256"] = _sha256_file(
                    source_graph_root / "manifest.json"
                )
            except (OSError, ValueError) as exc:
                reuse_audit["source_collection_status"] = "rejected_regenerate_all"
                reuse_audit["source_collection_rejection"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                source_cohort_root = None
                source_graph_root = None
                source_graph_manifest = None

        core_records: list[dict[str, Any]] = []
        for alias in SO1_ALIASES:
            with np.load(
                cohort_root / "cores" / f"{alias}.npz", allow_pickle=False
            ) as data:
                coordinates = np.asarray(data["coordinates_um"], dtype=np.float64)
            source_alias = _REUSABLE_CANCER_ALIAS_BY_SO1.get(alias)
            record: dict[str, Any]
            if (
                source_alias is not None
                and source_cohort_root is not None
                and source_graph_root is not None
                and source_graph_manifest is not None
            ):
                try:
                    record, reuse_result = _reuse_verified_cancer_graph(
                        alias=alias,
                        source_alias=source_alias,
                        cohort_root=cohort_root,
                        source_cohort_root=source_cohort_root,
                        source_graph_root=source_graph_root,
                        source_graph_manifest=source_graph_manifest,
                        output_dir=staging / "cores" / alias,
                    )
                    reuse_audit["results"].append(reuse_result)
                except (OSError, ValueError) as exc:
                    reuse_audit["results"].append(
                        {
                            "alias": alias,
                            "status": "rejected_regenerated",
                            "source_alias": source_alias,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    record = materialize_so1_core_relative_graph(
                        alias=alias,
                        coordinates_um=coordinates,
                        output_dir=staging / "cores" / alias,
                        receiver_chunk_size=receiver_chunk_size,
                        max_edges_per_chunk=max_edges_per_chunk,
                    )
            else:
                record = materialize_so1_core_relative_graph(
                    alias=alias,
                    coordinates_um=coordinates,
                    output_dir=staging / "cores" / alias,
                    receiver_chunk_size=receiver_chunk_size,
                    max_edges_per_chunk=max_edges_per_chunk,
                )
                if source_alias is not None:
                    reuse_audit["results"].append(
                        {
                            "alias": alias,
                            "status": "not_requested_or_source_rejected_regenerated",
                            "source_alias": source_alias,
                        }
                    )
            core_records.append(record)

        completed_cohort = deepcopy(cohort_manifest)
        by_alias = {record["alias"]: record for record in core_records}
        for core in completed_cohort.get("cores", []):
            alias = str(core.get("alias", ""))
            if alias not in by_alias:
                raise SO1RelativeGraphContractError(
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
                completed_cohort,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

        manifest: dict[str, Any] = {
            "artifact_kind": "so1_14core_relative_graph_collection",
            "format_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
            "aliases": list(SO1_ALIASES),
            "cohort_manifest_sha256": _sha256_file(cohort_root / "manifest.json"),
            "completed_cohort_manifest_sha256": _sha256_file(completed_path),
            "receiver_chunk_size": int(receiver_chunk_size),
            "max_edges_per_chunk": int(max_edges_per_chunk),
            "reuse_audit": reuse_audit,
            "cores": core_records,
        }
        manifest["manifest_content_sha256"] = _payload_sha256(manifest)
        (staging / "manifest.json").write_bytes(
            json.dumps(
                manifest, sort_keys=True, indent=2, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_so1_relative_qkv_batches(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
) -> tuple[Any, ...]:
    """Open 14 CPU-resident SO1 training batches with mmap geometry."""

    from .pooled_relative_qkv_training_v2 import CohortRelativeQKVCoreBatch

    cohort_root = Path(cohort_dir)
    graph_root = Path(graph_dir)
    cohort_manifest = _verify_so1_cohort_manifest(cohort_root)
    graph_manifest = _verify_so1_graph_collection(
        graph_root,
        cohort_manifest_path=cohort_root / "manifest.json",
    )
    records_by_alias = {
        str(record["alias"]): record for record in graph_manifest["cores"]
    }
    batches = []
    for alias in SO1_ALIASES:
        with np.load(
            cohort_root / "cores" / f"{alias}.npz", allow_pickle=False
        ) as data:
            target = np.array(data["target_expression"], dtype=np.float32, copy=True)
            covariates = np.array(
                data["node_covariates"], dtype=np.float32, copy=True
            )
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
        if target.shape != (expected_nodes, EXPECTED_N_GENES):
            raise SO1RelativeGraphContractError(
                f"Prepared target shape changed for {alias}."
            )
        if edge_map.shape != (2, expected_edges) or geometry_map.shape != (
            expected_edges,
            RELATIVE_GEOMETRY_DIM,
        ):
            raise SO1RelativeGraphContractError(
                f"Graph/cache array shape changed for {alias}."
            )
        batches.append(
            CohortRelativeQKVCoreBatch(
                alias=alias,
                target_expression=torch.from_numpy(target),
                edge_index=torch.from_numpy(edge_map),
                relative_geometry=torch.from_numpy(geometry_map),
                node_covariates=torch.from_numpy(covariates),
            )
        )
    return tuple(batches)


__all__ = [
    "DEFAULT_MAX_EDGES_PER_CHUNK",
    "DEFAULT_RECEIVER_CHUNK_SIZE",
    "GRAPH_ARTIFACT_SCHEMA_VERSION",
    "SO1RelativeGraphContractError",
    "load_so1_relative_qkv_batches",
    "materialize_so1_core_relative_graph",
    "prepare_so1_14core_relative_graphs",
]

"""Identifier-safe preparation for neutral SO_1 cores 1 through 14.

The cohort is selected only through the authoritative slide-qualified core
map.  Every raw SO_1 FOV must be mapped exactly once: unlike SO_2, this slide
has no unmapped field to exclude.  FOV and cell identifiers are temporary
routing keys and are never written to model arrays.  Coordinates are persisted
separately for graph construction and never enter ``node_covariates``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    ALLOWED_METADATA_COLUMNS,
    CONTROL_PREFIXES,
    DEFAULT_PIXEL_SIZE_UM,
    CoreDataset,
    CoreSelection,
    DataContractError,
    discover_slide_raw_path,
    load_selected_core,
    normalize_slide,
)
from .pooled_full_core import fit_equal_core_log1p_statistics
from .splits import TrainOnlyPreprocessor


SO1_CORE_NUMBERS = tuple(range(1, 15))
SO1_ALIASES = tuple(f"SO1-C{number:02d}" for number in SO1_CORE_NUMBERS)
SO1_ALIAS_BY_NUMBER = dict(zip(SO1_CORE_NUMBERS, SO1_ALIASES, strict=True))
SO1_SOURCE_SLIDE = "SO_1"
EXPECTED_MAPPED_FOV_COUNT = 205
EXPECTED_TOTAL_CELLS = 161_596
EXPECTED_CELL_COUNTS_BY_CORE = {
    1: 8_924,
    2: 7_450,
    3: 12_190,
    4: 14_657,
    5: 11_399,
    6: 18_212,
    7: 10_722,
    8: 4_972,
    9: 17_223,
    10: 14_756,
    11: 18_145,
    12: 7_816,
    13: 5_345,
    14: 9_785,
}
EXPECTED_N_GENES = 1_000
FIT_SCOPE = "all_cells_so1_cores_1_through_14_transductive"
PREPARATION_SCHEMA_VERSION = 1


class SO1CohortContractError(ValueError):
    """Raised when SO_1 cohort routing or preprocessing fails closed."""


@dataclass(frozen=True, slots=True)
class SO1CoreRoute:
    """Temporary slide-qualified route; FOVs are not serialized as features."""

    alias: str
    core_number: int
    slide: str
    fovs: tuple[int, ...]

    @property
    def route_sha256(self) -> str:
        """Bind the alias to the complete route and zero-unmapped contract."""

        return _canonical_sha256(
            {
                "alias": self.alias,
                "all_raw_fovs_mapped": True,
                "core_number": self.core_number,
                "fovs": list(self.fovs),
                "slide": self.slide,
            }
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(name: str, array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(_canonical_json(list(values.shape)))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _safe_relative_source(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    np.savez_compressed(path, **arrays)
    return _sha256_file(path)


def resolve_so1_core_routes(core_map_csv: str | Path) -> tuple[SO1CoreRoute, ...]:
    """Resolve cores 1--14 exclusively from explicit ``(slide, fov)`` keys."""

    frame = pd.read_csv(core_map_csv)
    required = {"slide", "core_label", "fov"}
    if not required.issubset(frame.columns):
        raise SO1CohortContractError(
            "Core map must contain slide, core_label, and fov columns."
        )
    frame = frame.loc[:, ["slide", "core_label", "fov"]].copy()
    if frame.isna().any().any():
        raise SO1CohortContractError("Core map routing fields cannot be missing.")
    try:
        frame["slide"] = [normalize_slide(value) for value in frame["slide"]]
        frame["core_label"] = pd.to_numeric(
            frame["core_label"], errors="raise"
        ).astype(int)
        frame["fov"] = pd.to_numeric(frame["fov"], errors="raise").astype(int)
    except (DataContractError, TypeError, ValueError) as exc:
        raise SO1CohortContractError("Core map routing fields are invalid.") from exc
    if (frame[["core_label", "fov"]] <= 0).any().any():
        raise SO1CohortContractError("Core labels and FOVs must be positive.")
    if frame.duplicated(["slide", "fov"]).any():
        raise SO1CohortContractError(
            "Core map contains a duplicate slide-qualified FOV route."
        )

    slide_rows = frame.loc[frame["slide"] == SO1_SOURCE_SLIDE]
    requested_rows = frame.loc[frame["core_label"].isin(SO1_CORE_NUMBERS)]
    if len(requested_rows) != len(slide_rows) or set(
        requested_rows["slide"].tolist()
    ) != {SO1_SOURCE_SLIDE}:
        raise SO1CohortContractError(
            "Core labels 1 through 14 must resolve exclusively to SO_1."
        )
    if set(slide_rows["core_label"].tolist()) != set(SO1_CORE_NUMBERS):
        raise SO1CohortContractError(
            "The SO_1 map must contain exactly core labels 1 through 14."
        )

    routes: list[SO1CoreRoute] = []
    for core_number in SO1_CORE_NUMBERS:
        selected = slide_rows.loc[slide_rows["core_label"] == core_number]
        if selected.empty:
            raise SO1CohortContractError(
                f"Core {core_number} must resolve to a nonempty SO_1 route."
            )
        fovs = tuple(sorted(int(value) for value in selected["fov"].unique()))
        routes.append(
            SO1CoreRoute(
                alias=SO1_ALIAS_BY_NUMBER[core_number],
                core_number=core_number,
                slide=SO1_SOURCE_SLIDE,
                fovs=fovs,
            )
        )
    qualified = [(route.slide, fov) for route in routes for fov in route.fovs]
    if len(qualified) != len(set(qualified)):
        raise SO1CohortContractError("SO_1 core routes overlap.")
    return tuple(routes)


def audit_so1_raw_fov_partition(
    metadata_csv: str | Path,
    routes: Sequence[SO1CoreRoute],
    *,
    chunksize: int = 100_000,
    expected_cell_counts_by_core: Mapping[int, int] | None = None,
    expected_total_cells: int = EXPECTED_TOTAL_CELLS,
    expected_mapped_fov_count: int = EXPECTED_MAPPED_FOV_COUNT,
) -> dict[str, Any]:
    """Audit that mapped routes cover every raw SO_1 FOV and cell exactly."""

    if chunksize <= 0:
        raise SO1CohortContractError("Routing-audit chunk size must be positive.")
    if expected_mapped_fov_count <= 0:
        raise SO1CohortContractError("Expected mapped FOV count must be positive.")
    expected_counts = dict(
        EXPECTED_CELL_COUNTS_BY_CORE
        if expected_cell_counts_by_core is None
        else expected_cell_counts_by_core
    )
    if set(expected_counts) != set(SO1_CORE_NUMBERS):
        raise SO1CohortContractError(
            "Expected cell counts must cover exactly cores 1 through 14."
        )
    if sum(int(value) for value in expected_counts.values()) != int(
        expected_total_cells
    ):
        raise SO1CohortContractError(
            "Expected per-core cell counts do not sum to expected total cells."
        )

    counts_by_fov: dict[int, int] = {}
    try:
        chunks = pd.read_csv(
            metadata_csv,
            usecols=["fov"],
            dtype={"fov": "int32"},
            chunksize=chunksize,
        )
        for chunk in chunks:
            for raw_fov, raw_count in chunk["fov"].value_counts().items():
                fov = int(raw_fov)
                counts_by_fov[fov] = counts_by_fov.get(fov, 0) + int(raw_count)
    except (OSError, TypeError, ValueError) as exc:
        raise SO1CohortContractError(
            "Could not audit raw SO_1 metadata FOV coverage."
        ) from exc

    selected_fovs = {fov for route in routes for fov in route.fovs}
    if len(selected_fovs) != int(expected_mapped_fov_count):
        raise SO1CohortContractError(
            "Mapped SO_1 FOV count changed: "
            f"expected {expected_mapped_fov_count}, observed {len(selected_fovs)}."
        )
    raw_fovs = set(counts_by_fov)
    unmapped_fovs = sorted(raw_fovs.difference(selected_fovs))
    missing_mapped_fovs = sorted(selected_fovs.difference(raw_fovs))
    if unmapped_fovs or missing_mapped_fovs:
        raise SO1CohortContractError(
            "Every raw SO_1 FOV must be mapped exactly once "
            f"(unmapped={unmapped_fovs}, missing={missing_mapped_fovs})."
        )

    observed_by_core = {
        route.core_number: int(sum(counts_by_fov[fov] for fov in route.fovs))
        for route in routes
    }
    if observed_by_core != {
        int(core): int(count) for core, count in expected_counts.items()
    }:
        raise SO1CohortContractError(
            "Mapped SO_1 per-core cell counts changed from the locked contract."
        )
    selected_cells = int(sum(observed_by_core.values()))
    if selected_cells != int(expected_total_cells):
        raise SO1CohortContractError(
            f"Expected {expected_total_cells} selected cells; "
            f"observed {selected_cells}."
        )
    return {
        "source_slide": SO1_SOURCE_SLIDE,
        "mapped_core_numbers": list(SO1_CORE_NUMBERS),
        "mapped_fov_count": len(selected_fovs),
        "selected_cell_count": selected_cells,
        "per_core_cell_counts": {
            str(core): observed_by_core[core] for core in SO1_CORE_NUMBERS
        },
        "raw_slide_cell_count": selected_cells,
        "selection_is_slide_qualified": True,
        "all_raw_fovs_mapped": True,
        "unmapped_fovs": [],
        "unmapped_fov_count": 0,
        "unmapped_fov_cell_count": 0,
        "unmapped_fov_entered_prepared_arrays": False,
    }


def _split_loaded_slide(
    loaded: CoreDataset,
    routes: Sequence[SO1CoreRoute],
    expected_counts: Mapping[int, int],
) -> list[tuple[SO1CoreRoute, CoreDataset]]:
    observed_fovs = set(int(value) for value in loaded.keys["fov"].unique())
    expected_fovs = {fov for route in routes for fov in route.fovs}
    if observed_fovs != expected_fovs:
        raise SO1CohortContractError(
            "Loaded SO_1 routing keys disagree with the complete mapped partition."
        )
    result: list[tuple[SO1CoreRoute, CoreDataset]] = []
    for route in routes:
        selected = loaded.keys["fov"].isin(route.fovs).to_numpy(dtype=bool)
        core = CoreDataset(
            expression=np.ascontiguousarray(loaded.expression[selected]),
            metadata=np.ascontiguousarray(loaded.metadata[selected]),
            coordinates_px=np.ascontiguousarray(loaded.coordinates_px[selected]),
            coordinates_um=np.ascontiguousarray(loaded.coordinates_um[selected]),
            keys=loaded.keys.loc[selected].reset_index(drop=True).copy(),
            qc_passed=np.ascontiguousarray(loaded.qc_passed[selected]),
            gene_names=loaded.gene_names,
            metadata_names=loaded.metadata_names,
        )
        expected = int(expected_counts[route.core_number])
        if core.n_cells != expected:
            raise SO1CohortContractError(
                f"{route.alias} expected {expected} cells; observed {core.n_cells}."
            )
        result.append((route, core))
    if sum(core.n_cells for _, core in result) != loaded.n_cells:
        raise SO1CohortContractError("SO_1 core splitting lost or duplicated cells.")
    return result


def _fit_shared_metadata(
    metadata_by_core: Sequence[np.ndarray],
    *,
    epsilon: float,
) -> TrainOnlyPreprocessor:
    combined = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in metadata_by_core],
        axis=0,
    )
    dummy_counts = np.zeros((len(combined), 1), dtype=np.int8)
    all_fit = np.full(len(combined), "train", dtype="U5")
    try:
        return TrainOnlyPreprocessor(
            log1p_metadata=True,
            epsilon=epsilon,
        ).fit(
            dummy_counts,
            combined,
            all_fit,
            metadata_names=ALLOWED_METADATA_COLUMNS,
        )
    except (DataContractError, ValueError) as exc:
        raise SO1CohortContractError(
            "Shared 14-core metadata preprocessing failed."
        ) from exc


def _standardized_targets(
    counts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    row_chunk_size: int = 2_048,
) -> np.ndarray:
    output = np.empty(counts.shape, dtype=np.float32)
    for start in range(0, len(counts), row_chunk_size):
        stop = min(start + row_chunk_size, len(counts))
        values = np.log1p(counts[start:stop].astype(np.float64, copy=False))
        output[start:stop] = (values - mean) / scale
    if not np.isfinite(output).all():
        raise SO1CohortContractError(
            "Shared expression standardization produced non-finite values."
        )
    return output


def prepare_so1_14core_cohort(
    *,
    raw_dir: str | Path,
    core_map_csv: str | Path,
    output_dir: str | Path,
    chunksize: int = 8_192,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    epsilon: float = 1e-8,
    expected_cell_counts_by_core: Mapping[int, int] | None = None,
    expected_total_cells: int = EXPECTED_TOTAL_CELLS,
    expected_mapped_fov_count: int = EXPECTED_MAPPED_FOV_COUNT,
) -> dict[str, Any]:
    """Materialize the immutable, fit-only SO_1 core-1--14 cohort."""

    raw_root = Path(raw_dir)
    core_map_path = Path(core_map_csv)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            "SO_1 14-core output exists; immutable artifacts are not overwritten."
        )
    if chunksize <= 0 or not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise SO1CohortContractError("Chunk size and pixel size must be positive.")
    expected_counts = dict(
        EXPECTED_CELL_COUNTS_BY_CORE
        if expected_cell_counts_by_core is None
        else expected_cell_counts_by_core
    )
    routes = resolve_so1_core_routes(core_map_path)
    expression_path = discover_slide_raw_path(
        raw_root, SO1_SOURCE_SLIDE, "expression"
    )
    metadata_path = discover_slide_raw_path(raw_root, SO1_SOURCE_SLIDE, "metadata")
    routing_audit = audit_so1_raw_fov_partition(
        metadata_path,
        routes,
        chunksize=chunksize,
        expected_cell_counts_by_core=expected_counts,
        expected_total_cells=expected_total_cells,
        expected_mapped_fov_count=expected_mapped_fov_count,
    )

    all_fovs = tuple(sorted(fov for route in routes for fov in route.fovs))
    try:
        loaded_slide = load_selected_core(
            raw_root,
            CoreSelection(
                slide=SO1_SOURCE_SLIDE,
                fovs=all_fovs,
                label_policy="authoritative_so1_core_map_1_through_14_v1",
            ),
            chunksize=chunksize,
            expected_biological_probes=EXPECTED_N_GENES,
            pixel_size_um=pixel_size_um,
            qc_policy="all",
        )
    except (DataContractError, FileNotFoundError, ValueError) as exc:
        raise SO1CohortContractError("SO_1 raw-count loading failed.") from exc
    if loaded_slide.n_cells != int(expected_total_cells):
        raise SO1CohortContractError(
            f"Expected {expected_total_cells} selected cells; "
            f"loaded {loaded_slide.n_cells}."
        )
    loaded = _split_loaded_slide(loaded_slide, routes, expected_counts)

    gene_names = loaded_slide.gene_names
    if loaded_slide.n_genes != EXPECTED_N_GENES:
        raise SO1CohortContractError(
            "SO_1 cohort does not contain exactly 1,000 biological genes."
        )
    if any(name.startswith(CONTROL_PREFIXES) for name in gene_names) or len(
        set(gene_names)
    ) != EXPECTED_N_GENES:
        raise SO1CohortContractError(
            "SO_1 target schema contains controls or duplicate biological genes."
        )

    try:
        expression_mean, expression_scale = fit_equal_core_log1p_statistics(
            [core.expression for _, core in loaded],
            epsilon=epsilon,
        )
    except ValueError as exc:
        raise SO1CohortContractError(
            "Equal-core expression fitting across 14 cores failed."
        ) from exc
    metadata_preprocessor = _fit_shared_metadata(
        [core.metadata for _, core in loaded],
        epsilon=epsilon,
    )

    source_checksums = {
        "expression_sha256": _sha256_file(expression_path),
        "metadata_sha256": _sha256_file(metadata_path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "cores").mkdir()
        file_checksums: dict[str, str] = {}
        core_records: list[dict[str, Any]] = []
        for route, core in loaded:
            dummy_counts = np.zeros((core.n_cells, 1), dtype=np.int8)
            nodes = metadata_preprocessor.transform(dummy_counts, core.metadata)
            targets = _standardized_targets(
                core.expression,
                expression_mean,
                expression_scale,
            )
            arrays = {
                "expression_counts": np.asarray(core.expression, dtype=np.int32),
                "target_expression": targets,
                "node_covariates": nodes.metadata,
                "coordinates_um": np.asarray(core.coordinates_um, dtype=np.float64),
            }
            relative = Path("cores") / f"{route.alias}.npz"
            file_checksums[str(relative)] = _write_npz(temporary / relative, arrays)
            core_records.append(
                {
                    "alias": route.alias,
                    "original_core_number": route.core_number,
                    "source_slide": route.slide,
                    "n_fovs": len(route.fovs),
                    "route_sha256": route.route_sha256,
                    "cell_count": core.n_cells,
                    "tissue_context_verification": {
                        "status": "not_asserted",
                        "selection_role": "neutral_source_core_number_cohort",
                        "clinical_tissue_label_used_for_selection": False,
                    },
                    "source_artifacts": {
                        "expression": _safe_relative_source(
                            expression_path, base=raw_root
                        ),
                        "metadata": _safe_relative_source(
                            metadata_path, base=raw_root
                        ),
                    },
                    "source_checksums": dict(source_checksums),
                    "component_checksums": {
                        name: _array_sha256(name, values)
                        for name, values in arrays.items()
                    },
                    "coordinate_units": "micrometres",
                    "graph_checksum": None,
                }
            )

        statistics = {
            "expression_mean": np.asarray(expression_mean, dtype=np.float64),
            "expression_scale": np.asarray(expression_scale, dtype=np.float64),
            "metadata_median": metadata_preprocessor.metadata_median_.astype(
                np.float64
            ),
            "metadata_mean": metadata_preprocessor.metadata_mean_.astype(np.float64),
            "metadata_scale": metadata_preprocessor.metadata_scale_.astype(
                np.float64
            ),
            "metadata_missing_indicator_indices": (
                metadata_preprocessor.missing_indicator_indices_.astype(np.int64)
            ),
        }
        stats_relative = Path("cohort_statistics.npz")
        file_checksums[str(stats_relative)] = _write_npz(
            temporary / stats_relative, statistics
        )
        manifest: dict[str, Any] = {
            "artifact_kind": "so1_14core_pooled_full_core_preparation",
            "format_version": PREPARATION_SCHEMA_VERSION,
            "cohort": {
                "aliases": list(SO1_ALIASES),
                "original_core_numbers": list(SO1_CORE_NUMBERS),
                "source_slide": SO1_SOURCE_SLIDE,
                "total_cells": sum(core.n_cells for _, core in loaded),
                "fit_scope": FIT_SCOPE,
                "validation_or_test_partition_present": False,
                "generalization_claim_supported": False,
                "tissue_context": "not_asserted_neutral_core_number_cohort",
            },
            "routing_audit": routing_audit,
            "features": {
                "gene_names": list(gene_names),
                "n_biological_probes": EXPECTED_N_GENES,
                "technical_control_prefixes_excluded": list(CONTROL_PREFIXES),
                "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
                "model_covariate_names": list(
                    metadata_preprocessor.metadata_names_
                ),
                "coordinates_are_model_covariates": False,
                "routing_keys_are_model_covariates": False,
                "core_alias_is_model_covariate": False,
                "slide_identity_is_model_covariate": False,
                "library_size_is_model_covariate": False,
            },
            "preprocessing": {
                "expression_source": "raw_counts",
                "expression_transform": "gene_wise_standardized_log1p",
                "expression_moment_weighting": "equal_core",
                "expression_fit_scope": FIT_SCOPE,
                "metadata_transform": (
                    "median_imputation_log1p_standardization_with_missing_indicators"
                ),
                "metadata_fit_scope": FIT_SCOPE,
                "statistics_checksums": {
                    name: _array_sha256(name, values)
                    for name, values in statistics.items()
                },
            },
            "source": {
                "core_map_path": _safe_relative_source(
                    core_map_path, base=raw_root.parent
                ),
                "core_map_sha256": _sha256_file(core_map_path),
            },
            "cores": core_records,
            "files": file_checksums,
            "assurances": {
                "ordinary_arrays_contain_direct_identifiers": False,
                "raw_coordinates_used_only_for_graph_geometry": True,
                "all_raw_fovs_mapped": True,
                "unmapped_fov_exclusion_required": False,
                "clinical_labels_are_model_inputs": False,
                "protected_source_files_mutated": False,
            },
        }
        manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
        (temporary / "manifest.json").write_bytes(
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "EXPECTED_CELL_COUNTS_BY_CORE",
    "EXPECTED_MAPPED_FOV_COUNT",
    "EXPECTED_N_GENES",
    "EXPECTED_TOTAL_CELLS",
    "FIT_SCOPE",
    "PREPARATION_SCHEMA_VERSION",
    "SO1_ALIASES",
    "SO1_ALIAS_BY_NUMBER",
    "SO1_CORE_NUMBERS",
    "SO1_SOURCE_SLIDE",
    "SO1CohortContractError",
    "SO1CoreRoute",
    "audit_so1_raw_fov_partition",
    "prepare_so1_14core_cohort",
    "resolve_so1_core_routes",
]

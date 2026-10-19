"""Identifier-safe preparation for the six-core gastric-cancer cohort.

This module is deliberately limited to data reconciliation and preprocessing.
It resolves core routes with a slide-qualified join, loads the existing CosMx
raw-count contract, and writes arrays that contain no cell, FOV, patient, or
vendor-derived identifiers.  Coordinates are retained in a separate array for
graph construction only; they are never part of ``node_covariates``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import (
    ALLOWED_METADATA_COLUMNS,
    CONTROL_PREFIXES,
    DEFAULT_PIXEL_SIZE_UM,
    CoreSelection,
    DataContractError,
    discover_slide_raw_path,
    load_selected_core,
    normalize_slide,
)
from .pooled_full_core import fit_equal_core_log1p_statistics
from .splits import TrainOnlyPreprocessor


CORE_NUMBERS = (1, 9, 13, 15, 21, 23)
CANCER_ALIASES = tuple(f"CAN-{number:02d}" for number in CORE_NUMBERS)
CORE_ALIAS_BY_NUMBER = dict(zip(CORE_NUMBERS, CANCER_ALIASES, strict=True))
EXPECTED_N_GENES = 1_000
FIT_SCOPE = "all_cells_six_cancer_cores_transductive"
POLICY_SCHEMA_VERSION = 1
PREPARATION_SCHEMA_VERSION = 1
_ANSWERED_KINDS = frozenset({"cancer"})
_FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "patient_id",
        "patient_ids",
        "donor_id",
        "donor_ids",
        "subject_id",
        "subject_ids",
        "cell_id",
        "cell_ids",
    }
)


class CancerCohortContractError(ValueError):
    """Raised when cohort reconciliation or preparation fails closed."""


@dataclass(frozen=True, slots=True)
class CancerCoreRoute:
    """Temporary route for reading one core; FOVs are never serialized."""

    alias: str
    core_number: int
    slide: str
    fovs: tuple[int, ...]

    @property
    def route_sha256(self) -> str:
        return _canonical_sha256(
            {
                "alias": self.alias,
                "core_number": self.core_number,
                "slide": self.slide,
                "fovs": list(self.fovs),
            }
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReceipt:
    """Validated, non-identifying summary of the user-authorized policy."""

    policy_sha256: str
    resolved_tissue_context: str
    review_status: str
    historical_labels: Mapping[str, Mapping[int, str]]


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CancerCohortContractError(f"{label} must be a mapping.")
    return value


def _load_policy(
    policy: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if isinstance(policy, Mapping):
        value = dict(policy)
        return value, _canonical_sha256(value)
    path = Path(policy)
    raw = path.read_bytes()
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, Mapping):
        raise CancerCohortContractError(
            "Cancer reconciliation YAML must contain a mapping."
        )
    return dict(loaded), hashlib.sha256(raw).hexdigest()


def _assert_no_direct_identifier_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_POLICY_KEYS:
                raise CancerCohortContractError(
                    "Cancer reconciliation policy contains a direct identifier field."
                )
            _assert_no_direct_identifier_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_no_direct_identifier_keys(nested)


def _historical_label_map(value: object, *, label: str) -> dict[int, str]:
    mapping = _mapping(value, label=label)
    parsed: dict[int, str] = {}
    for raw_core, raw_value in mapping.items():
        try:
            core = int(raw_core)
        except (TypeError, ValueError) as exc:
            raise CancerCohortContractError(
                f"{label} contains a non-numeric core key."
            ) from exc
        text = str(raw_value).strip()
        if not text:
            raise CancerCohortContractError(
                f"{label} contains an empty historical label."
            )
        parsed[core] = text
    if tuple(sorted(parsed)) != CORE_NUMBERS:
        raise CancerCohortContractError(
            f"{label} must preserve labels for exactly the six selected cores."
        )
    return {core: parsed[core] for core in CORE_NUMBERS}


def validate_cancer_reconciliation(
    policy: str | Path | Mapping[str, Any],
) -> ReconciliationReceipt:
    """Validate the explicit user-attested Cancer reconciliation policy.

    Historical workbook/review labels are provenance only and are intentionally
    allowed to disagree with the user-authorized rediagnosis.  No source file is
    edited and no clinical label becomes a node feature.
    """

    value, policy_sha256 = _load_policy(policy)
    _assert_no_direct_identifier_keys(value)
    if int(value.get("schema_version", -1)) != POLICY_SCHEMA_VERSION:
        raise CancerCohortContractError(
            "Cancer reconciliation policy has an unsupported schema version."
        )
    if tuple(value.get("selected_core_numbers", ())) != CORE_NUMBERS:
        raise CancerCohortContractError(
            "Cancer reconciliation policy must name the exact six core numbers."
        )
    if tuple(value.get("selected_aliases", ())) != CANCER_ALIASES:
        raise CancerCohortContractError(
            "Cancer reconciliation policy must name the exact ordered aliases."
        )

    attestation = _mapping(value.get("user_attestation"), label="user_attestation")
    resolved = str(attestation.get("resolved_tissue_context", "")).strip()
    if resolved.casefold() not in _ANSWERED_KINDS:
        raise CancerCohortContractError(
            "User attestation must resolve all selected cores to Cancer."
        )
    if attestation.get("applies_to_all_selected_cores") is not True:
        raise CancerCohortContractError(
            "User attestation must explicitly apply to all six selected cores."
        )
    if attestation.get("contains_patient_identifiers") is not False:
        raise CancerCohortContractError(
            "User attestation must be explicitly non-identifying."
        )

    resolution = _mapping(value.get("resolution"), label="resolution")
    required_resolution = {
        "selection_authorized": True,
        "historical_labels_preserved": True,
        "protected_inputs_mutated": False,
        "ordinary_model_inputs_include_clinical_labels": False,
    }
    if any(resolution.get(key) is not expected for key, expected in required_resolution.items()):
        raise CancerCohortContractError(
            "Cancer reconciliation safety assertions are incomplete or false."
        )
    review_status = str(value.get("review_status", "")).strip()
    if review_status != "explicitly_authorized_by_user":
        raise CancerCohortContractError(
            "Cancer reconciliation policy lacks explicit user authorization."
        )

    evidence = _mapping(
        value.get("protected_source_evidence"),
        label="protected_source_evidence",
    )
    historical: dict[str, Mapping[int, str]] = {}
    for source_name in ("legacy_workbook", "pathology_review_workbook"):
        source = _mapping(evidence.get(source_name), label=source_name)
        historical[source_name] = _historical_label_map(
            source.get("observed_nonidentifying_labels"),
            label=f"{source_name}.observed_nonidentifying_labels",
        )
    return ReconciliationReceipt(
        policy_sha256=policy_sha256,
        resolved_tissue_context="Cancer",
        review_status=review_status,
        historical_labels=historical,
    )


def resolve_cancer_core_routes(
    core_map_csv: str | Path,
    policy: str | Path | Mapping[str, Any],
) -> tuple[CancerCoreRoute, ...]:
    """Resolve exact core routes using ``(slide, fov)`` keys, never FOV alone."""

    validate_cancer_reconciliation(policy)
    frame = pd.read_csv(core_map_csv)
    required = {"slide", "core_label", "fov"}
    if not required.issubset(frame.columns):
        raise CancerCohortContractError(
            "Core map must contain slide, core_label, and fov columns."
        )
    frame = frame.loc[:, ["slide", "core_label", "fov"]].copy()
    if frame.isna().any().any():
        raise CancerCohortContractError("Core map routing fields cannot be missing.")
    try:
        frame["slide"] = [normalize_slide(value) for value in frame["slide"]]
        frame["core_label"] = pd.to_numeric(frame["core_label"], errors="raise").astype(int)
        frame["fov"] = pd.to_numeric(frame["fov"], errors="raise").astype(int)
    except (DataContractError, TypeError, ValueError) as exc:
        raise CancerCohortContractError("Core map routing fields are invalid.") from exc
    if frame.duplicated(["slide", "fov"]).any():
        raise CancerCohortContractError(
            "Core map contains a duplicate slide-qualified FOV route."
        )

    routes: list[CancerCoreRoute] = []
    for core_number in CORE_NUMBERS:
        selected = frame.loc[frame["core_label"] == core_number]
        slides = tuple(sorted(selected["slide"].unique().tolist()))
        if len(slides) != 1 or selected.empty:
            raise CancerCohortContractError(
                f"Core {core_number} must resolve to one nonempty source slide."
            )
        fovs = tuple(sorted(int(value) for value in selected["fov"].unique()))
        routes.append(
            CancerCoreRoute(
                alias=CORE_ALIAS_BY_NUMBER[core_number],
                core_number=core_number,
                slide=slides[0],
                fovs=fovs,
            )
        )
    return tuple(routes)


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
        raise CancerCohortContractError(
            "Shared cancer-core metadata preprocessing failed."
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
        raise CancerCohortContractError(
            "Shared expression standardization produced non-finite values."
        )
    return output


def _safe_relative_source(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    np.savez_compressed(path, **arrays)
    return _sha256_file(path)


def prepare_cancer_6core_cohort(
    *,
    raw_dir: str | Path,
    core_map_csv: str | Path,
    reconciliation_yaml: str | Path,
    output_dir: str | Path,
    chunksize: int = 8_192,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Materialize the exact six-core fit-only cohort into a new directory.

    The destination must not already exist.  Work is staged in a sibling
    temporary directory and published with one atomic rename after every file
    checksum and manifest invariant has been established.
    """

    raw_root = Path(raw_dir)
    core_map_path = Path(core_map_csv)
    policy_path = Path(reconciliation_yaml)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            "Cancer cohort output already exists; immutable artifacts are not overwritten."
        )
    if chunksize <= 0 or not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise CancerCohortContractError(
            "Chunk size and pixel size must be positive."
        )
    receipt = validate_cancer_reconciliation(policy_path)
    routes = resolve_cancer_core_routes(core_map_path, policy_path)

    loaded = []
    for route in routes:
        try:
            core = load_selected_core(
                raw_root,
                CoreSelection(
                    slide=route.slide,
                    fovs=route.fovs,
                    label_policy="user_attested_cancer_reconciliation_v1",
                ),
                chunksize=chunksize,
                expected_biological_probes=EXPECTED_N_GENES,
                pixel_size_um=pixel_size_um,
                qc_policy="all",
            )
        except (DataContractError, FileNotFoundError, ValueError) as exc:
            raise CancerCohortContractError(
                f"{route.alias} raw-count loading failed."
            ) from exc
        if core.n_genes != EXPECTED_N_GENES:
            raise CancerCohortContractError(
                f"{route.alias} does not contain exactly 1,000 biological genes."
            )
        loaded.append((route, core))

    gene_names = loaded[0][1].gene_names
    if any(core.gene_names != gene_names for _, core in loaded[1:]):
        raise CancerCohortContractError(
            "The six cancer cores do not share one identical ordered gene schema."
        )
    if any(
        name.startswith(CONTROL_PREFIXES) for name in gene_names
    ) or len(set(gene_names)) != EXPECTED_N_GENES:
        raise CancerCohortContractError(
            "Cancer target schema contains controls or duplicate biological genes."
        )

    try:
        expression_mean, expression_scale = fit_equal_core_log1p_statistics(
            [core.expression for _, core in loaded],
            epsilon=epsilon,
        )
    except ValueError as exc:
        raise CancerCohortContractError(
            "Equal-core expression fitting failed."
        ) from exc
    metadata_preprocessor = _fit_shared_metadata(
        [core.metadata for _, core in loaded],
        epsilon=epsilon,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        core_dir = temporary / "cores"
        core_dir.mkdir()
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
            component_checksums = {
                name: _array_sha256(name, values)
                for name, values in arrays.items()
            }
            expression_path = discover_slide_raw_path(
                raw_root, route.slide, "expression"
            )
            metadata_path = discover_slide_raw_path(
                raw_root, route.slide, "metadata"
            )
            core_records.append(
                {
                    "alias": route.alias,
                    "original_core_number": route.core_number,
                    "source_slide": route.slide,
                    "n_fovs": len(route.fovs),
                    "route_sha256": route.route_sha256,
                    "cell_count": core.n_cells,
                    "tissue_context_verification": {
                        "resolved_tissue_context": receipt.resolved_tissue_context,
                        "policy_sha256": receipt.policy_sha256,
                        "review_status": receipt.review_status,
                    },
                    "source_artifacts": {
                        "expression": _safe_relative_source(expression_path, base=raw_root),
                        "metadata": _safe_relative_source(metadata_path, base=raw_root),
                    },
                    "source_checksums": {
                        "expression_sha256": _sha256_file(expression_path),
                        "metadata_sha256": _sha256_file(metadata_path),
                    },
                    "component_checksums": component_checksums,
                    "coordinate_units": "micrometres",
                    "graph_checksum": None,
                }
            )

        statistics = {
            "expression_mean": np.asarray(expression_mean, dtype=np.float64),
            "expression_scale": np.asarray(expression_scale, dtype=np.float64),
            "metadata_median": metadata_preprocessor.metadata_median_.astype(np.float64),
            "metadata_mean": metadata_preprocessor.metadata_mean_.astype(np.float64),
            "metadata_scale": metadata_preprocessor.metadata_scale_.astype(np.float64),
            "metadata_missing_indicator_indices": (
                metadata_preprocessor.missing_indicator_indices_.astype(np.int64)
            ),
        }
        stats_relative = Path("cohort_statistics.npz")
        file_checksums[str(stats_relative)] = _write_npz(
            temporary / stats_relative,
            statistics,
        )
        stats_checksums = {
            name: _array_sha256(name, values)
            for name, values in statistics.items()
        }

        historical_labels = {
            source: {str(core): label for core, label in labels.items()}
            for source, labels in receipt.historical_labels.items()
        }
        manifest: dict[str, Any] = {
            "artifact_kind": "cancer_6core_pooled_full_core_preparation",
            "format_version": PREPARATION_SCHEMA_VERSION,
            "cohort": {
                "aliases": list(CANCER_ALIASES),
                "original_core_numbers": list(CORE_NUMBERS),
                "total_cells": sum(core.n_cells for _, core in loaded),
                "fit_scope": FIT_SCOPE,
                "validation_or_test_partition_present": False,
                "generalization_claim_supported": False,
                "tissue_context": "Cancer",
            },
            "reconciliation": {
                "policy_sha256": receipt.policy_sha256,
                "review_status": receipt.review_status,
                "historical_nonidentifying_labels": historical_labels,
                "protected_inputs_mutated": False,
                "clinical_labels_are_model_inputs": False,
            },
            "features": {
                "gene_names": list(gene_names),
                "n_biological_probes": EXPECTED_N_GENES,
                "technical_control_prefixes_excluded": list(CONTROL_PREFIXES),
                "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
                "model_covariate_names": list(metadata_preprocessor.metadata_names_),
                "coordinates_are_model_covariates": False,
                "routing_keys_are_model_covariates": False,
                "library_size_is_model_covariate": False,
            },
            "preprocessing": {
                "expression_source": "raw_counts",
                "expression_transform": "gene_wise_standardized_log1p",
                "expression_moment_weighting": "equal_core",
                "expression_fit_scope": FIT_SCOPE,
                "metadata_transform": "median_imputation_log1p_standardization_with_missing_indicators",
                "metadata_fit_scope": FIT_SCOPE,
                "statistics_checksums": stats_checksums,
            },
            "source": {
                "core_map_path": _safe_relative_source(core_map_path, base=raw_root.parent),
                "core_map_sha256": _sha256_file(core_map_path),
                "reconciliation_path": _safe_relative_source(policy_path, base=raw_root.parent),
                "reconciliation_sha256": receipt.policy_sha256,
            },
            "cores": core_records,
            "files": file_checksums,
            "assurances": {
                "ordinary_arrays_contain_direct_identifiers": False,
                "raw_coordinates_used_only_for_graph_geometry": True,
                "protected_source_workbooks_mutated": False,
            },
        }
        manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(
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


# Compact spelling for callers and tests.
prepare_cancer_cohort = prepare_cancer_6core_cohort


__all__ = [
    "CANCER_ALIASES",
    "CORE_ALIAS_BY_NUMBER",
    "CORE_NUMBERS",
    "EXPECTED_N_GENES",
    "FIT_SCOPE",
    "CancerCohortContractError",
    "CancerCoreRoute",
    "ReconciliationReceipt",
    "prepare_cancer_6core_cohort",
    "prepare_cancer_cohort",
    "resolve_cancer_core_routes",
    "validate_cancer_reconciliation",
]

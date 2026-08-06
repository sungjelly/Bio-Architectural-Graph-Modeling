#!/usr/bin/env python3
"""Audit and evaluate the pooled ten-core hybrid-count ensembles.

This entry point is intentionally post-production.  It refuses to load data or
models until the checksum-bound pilot gate and all fourteen production slots
have completed.  Member metrics are read from verified immutable bundles.
Ensemble metrics are recomputed from streamed member predictions: detection
and ordinal probabilities are averaged, continuous standardized predictions
are averaged, and only then are predictions decoded and scored.

The ten opaque adjacent-normal aliases are descriptive outcomes coupled by
shared fitted weights.  This script performs no formal core-level inference
and emits no protected identifiers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.train.run_full_core_capacity import (  # noqa: E402
    _build_graph,
    _fit_view,
    _training_config,
)
from scripts.train.run_hybrid_count_capacity import (  # noqa: E402
    _paired_models,
)
from scripts.train.run_pooled_hybrid_count_capacity import (  # noqa: E402
    _clear_graph_layout_caches,
    _graph_config_for_alias,
    _mask_bundle_for_core,
    _reference_metrics,
    _state_dict_sha256,
    _validate_graph,
    _validate_pooled_contract,
)
from spatial_benchmark import full_core as _full_core  # noqa: E402
from spatial_benchmark import pooled_full_core as _pooled_full_core  # noqa: E402
from spatial_benchmark.full_core import EDGE_ATTRIBUTE_NAMES  # noqa: E402
from spatial_benchmark.hybrid_count import validate_raw_counts  # noqa: E402
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    HybridCountReferences,
    evaluate_hybrid_count_output,
    fit_hybrid_count_references,
    json_safe_metrics,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.pooled_campaign_gates import (  # noqa: E402
    ALIASES,
    ARMS,
    GAT_ARM,
    MASK_MODES,
    REPLICATES,
    SEEDS,
    SELF_ARM,
    collapse_ensemble_replicates,
    collapse_member_replicates,
    evaluate_frozen_gates,
)
from spatial_benchmark.pooled_ensemble import (  # noqa: E402
    HybridCountEnsembleAccumulator,
)
from spatial_benchmark.pooled_full_core import PooledFullCoreCohort  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402
from spatial_benchmark.training import (  # noqa: E402
    _autocast_context,
    _forward_masked_targets,
    _model_dtype,
    _to_device_view,
)


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
FROZEN_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
MATERIALIZATION_KIND = "pooled_hybrid_count_locked_config_materialization_v1"
PILOT_ENQUEUE_KIND = "pooled_hybrid_count_pilot_enqueue_v1"
PILOT_GATE_KIND = "pooled_hybrid_count_pilot_gate_v1"
PRODUCTION_ENQUEUE_KIND = "pooled_hybrid_count_production_enqueue_v1"
EXPECTED_PARAMETER_COUNT = 11_674_880
EXPECTED_GLOBAL_EPOCHS = 200
EXPECTED_OPTIMIZER_STEPS = 2_000
EXPECTED_CHECKPOINT_EPOCH = 199
EXPECTED_PROTOCOL = "held_in_pooled_10core_fixed_budget"
EXPECTED_TASK_FAMILY = "masked_expression_hybrid_count"
EXPECTED_REPRESENTATION_SCHEMA = (
    "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
)
EXPECTED_OBJECTIVE = (
    "equal_weight_balanced_detection_ordinal_positive_huber_within_core"
)
EXPECTED_MEMBER_ROWS = len(ARMS) * len(SEEDS) * len(ALIASES) * len(MASK_MODES) * len(REPLICATES)
EXPECTED_ENSEMBLE_ROWS = len(ARMS) * len(ALIASES) * len(MASK_MODES) * len(REPLICATES)
MODEL_TO_ARM = {
    "hybrid-count-gat": GAT_ARM,
    "hybrid-count-matched-self": SELF_ARM,
}
ARM_TO_MODEL = {arm: model for model, arm in MODEL_TO_ARM.items()}
TABLE_SUFFIXES = (".parquet", ".jsonl", ".csv")
GATE_METRICS = (
    "hybrid_loss",
    "detection_bce",
    "positive_ordinal_mae",
    "positive_continuous_huber",
    "reconstructed_count_log1p_mae",
)
ENSEMBLE_GATE_METRICS = GATE_METRICS + (
    "detection_balanced_accuracy",
    "reference_per_gene_positive_ordinal_mae",
    "reference_per_gene_positive_continuous_huber",
    "reference_per_gene_detection_balanced_accuracy",
    "reference_equal_core_detection_balanced_accuracy",
)
VECTOR_WIDTHS = {
    "state8_support": 8,
    "state8_recall": 8,
    "collapsed4_support": 4,
    "collapsed4_recall": 4,
    "reference_all_zero_state8_support": 8,
    "reference_all_zero_state8_recall": 8,
}
REQUIRED_MODEL_SCALAR_METRICS = (
    "hybrid_loss",
    "detection_bce",
    "ordinal_bce",
    "positive_continuous_huber",
    "detection_balanced_accuracy",
    "detection_sensitivity",
    "detection_specificity",
    "detection_positive_support",
    "detection_zero_support",
    "detection_predicted_positive",
    "state8_exact_accuracy",
    "state8_balanced_accuracy",
    "positive_state_exact_accuracy",
    "positive_ordinal_mae",
    "positive_within_one_state_accuracy",
    "positive_continuous_mae",
    "reconstructed_count_log1p_mae",
    "collapsed4_exact_accuracy",
    "collapsed4_balanced_accuracy",
    "collapsed4_positive_exact_accuracy",
    "reference_per_gene_hybrid_loss",
    "reference_per_gene_detection_bce",
    "reference_per_gene_ordinal_bce",
    "reference_per_gene_positive_continuous_huber",
    "reference_per_gene_detection_balanced_accuracy",
    "reference_per_gene_positive_ordinal_mae",
    "reference_per_gene_positive_continuous_mae",
    "reference_per_gene_state8_exact_accuracy",
    "reference_per_gene_state8_balanced_accuracy",
    "reference_all_zero_state8_exact_accuracy",
    "reference_all_zero_state8_balanced_accuracy",
    "reference_all_zero_collapsed4_exact_accuracy",
)
REQUIRED_EQUAL_CORE_REFERENCE_METRICS = (
    "reference_equal_core_hybrid_loss",
    "reference_equal_core_detection_bce",
    "reference_equal_core_ordinal_bce",
    "reference_equal_core_positive_continuous_huber",
    "reference_equal_core_detection_balanced_accuracy",
    "reference_equal_core_positive_ordinal_mae",
    "reference_equal_core_positive_continuous_mae",
    "reference_equal_core_state8_exact_accuracy",
    "reference_equal_core_reconstructed_count_log1p_mae",
)
ROW_IDENTITY_FIELDS = {
    "arm",
    "seed",
    "core_alias",
    "biological_unit_alias",
    "split",
    "mask_mode",
    "mask_replicate",
    "mask_entry_id",
    "mask_seed",
    "mask_checksum",
    "run_id",
    "prior_run_id",
}
SAFE_GPU_IDS = frozenset({0, 1, 2, 3, 5, 6, 7})
FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "patient",
        "patient_id",
        "patient_identifier",
        "donor",
        "donor_id",
        "donor_identifier",
        "subject",
        "subject_id",
        "subject_identifier",
        "sample_id",
        "sample_identifier",
        "specimen_id",
        "specimen_identifier",
        "cell_id",
        "cell_identifier",
        "fov",
        "fov_id",
        "core_id",
        "core_identifier",
        "clinical",
        "clinical_id",
        "clinical_identifier",
        "slide",
        "slide_id",
    }
)
PRIOR_CATEGORICAL_CAMPAIGN = "cmp_20260726_full_core_g2_count_tokens_multiseed"
PRIOR_CATEGORICAL_COMPARISON_SHA256 = (
    "cdc0fb4e66ae920a6525d19a7291b15f85f85ef877d821d70359da54c687942a"
)
PRIOR_CATEGORICAL_VARIANTS = (
    "g2_tokenized_width512_exact_k1000_full_core",
    "g2_tokenized_width1024_exact_k1000_full_core",
)
DEFAULT_PRIOR_CATEGORICAL_RELATIVE = Path(
    "analyses/full_core_g2_count_tokens_multiseed/comparison/comparison.json"
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "analyses/adjacent_normal_10core_pooled_hybrid_ensemble/comparison"
)
MAXIMUM_CLAIM = (
    "held-in masked-expression capacity of shared and ensembled models across "
    "ten adjacent-normal cores, plus descriptive broad-context graph gain"
)
ANALYSIS_HELPER_RELATIVE_PATHS = (
    Path("scripts/analysis/compare_pooled_hybrid_ensemble.py"),
    Path("scripts/train/run_full_core_capacity.py"),
    Path("scripts/train/run_hybrid_count_capacity.py"),
    Path("scripts/train/run_pooled_hybrid_count_capacity.py"),
)


class PooledEnsembleComparisonError(RuntimeError):
    """Raised when conclusion-bearing evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ProductionRunEvidence:
    arm: str
    seed: int
    run_id: str
    attempt: int
    root: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_file_sha256: str
    state_dict_sha256: str
    encoder_initial_state_sha256: str
    decoder_initial_state_sha256: str
    member_rows: tuple[Mapping[str, Any], ...]
    duration_seconds: float
    peak_vram_gib: float
    peak_host_memory_bytes: int
    convergence: Mapping[str, Any]
    resource_usage: Mapping[str, Any]
    failed_attempt_count: int


@dataclass(frozen=True)
class CampaignAudit:
    selected_jobs: Mapping[tuple[str, int], Mapping[str, Any]]
    attempt_inventory: tuple[Mapping[str, Any], ...]
    registered_failure_inventory: tuple[Mapping[str, Any], ...]
    pilot_attempt_inventory: tuple[Mapping[str, Any], ...] = ()
    pilot_failure_inventory: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class StreamingPooledFullCoreCohort:
    """Shared pooled state without retaining any core-sized arrays.

    Prepared artifacts are deliberately read in separate passes.  The first
    pass fits the equal-core expression moments, the second fits compact
    references and binds the exact cohort fingerprint, and ``iter_cores``
    reloads one core at a time for graph evaluation.  This costs extra I/O but
    prevents the post-production evaluator from retaining all ten expression
    matrices in host memory.
    """

    prepared_artifacts: tuple[tuple[str, Path], ...]
    aliases: tuple[str, ...]
    gene_names: tuple[str, ...]
    metadata_names: tuple[str, ...]
    expression_mean: np.ndarray
    expression_scale: np.ndarray
    total_nodes: int
    checksums: Any
    pooled_references: HybridCountReferences
    core_preprocessing_sha256: Mapping[str, str]
    epsilon: float

    @property
    def n_cores(self) -> int:
        return len(self.aliases)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def fingerprint_sha256(self) -> str:
        return str(self.checksums.combined_fingerprint_sha256)

    def iter_cores(self) -> Iterable[Any]:
        """Yield exactly one verified pooled core at a time."""

        receipts = {
            receipt.alias: receipt
            for receipt in _pooled_full_core.EXPECTED_PREPARED_CORES
        }
        for alias, path in self.prepared_artifacts:
            loaded, genes, metadata = _pooled_full_core._load_verified_core(  # noqa: SLF001
                alias,
                path,
                receipt=receipts[alias],
                epsilon=self.epsilon,
            )
            if genes != self.gene_names or metadata != self.metadata_names:
                raise PooledEnsembleComparisonError(
                    f"streamed schema changed while reloading {alias}"
                )
            core = _pooled_core_from_loaded(
                loaded,
                gene_names=self.gene_names,
                metadata_names=self.metadata_names,
                expression_mean=self.expression_mean,
                expression_scale=self.expression_scale,
            )
            if (
                core.checksums.preprocessing_sha256
                != self.core_preprocessing_sha256[alias]
            ):
                raise PooledEnsembleComparisonError(
                    f"streamed preprocessing identity changed for {alias}"
                )
            del loaded
            yield core


def _pooled_core_from_loaded(
    loaded: Any,
    *,
    gene_names: tuple[str, ...],
    metadata_names: tuple[str, ...],
    expression_mean: np.ndarray,
    expression_scale: np.ndarray,
) -> Any:
    """Construct the exact public pooled-core record from one loaded source."""

    target = _pooled_full_core._standardize_counts(  # noqa: SLF001
        loaded.expression_counts,
        mean=expression_mean,
        scale=expression_scale,
    )
    expression_mean_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_mean",
        expression_mean,
    )
    expression_scale_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_scale",
        expression_scale,
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
        "gene_names": list(gene_names),
        "metadata_names": list(metadata_names),
        "expression_fit_scope": "all ten cores (transductive)",
        "expression_weighting": "equal core mixture",
        "morphology_fit_scope": "each core independently (transductive)",
        "source_manifest_sha256": loaded.receipt.manifest_sha256,
        "source_prepared_data_sha256": loaded.receipt.prepared_data_sha256,
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
            "macroblock_ids": macroblock_sha,
            "expression_mean": expression_mean_sha,
            "expression_scale": expression_scale_sha,
        },
    }
    checksums = _pooled_full_core.PooledCoreChecksums(
        source_manifest_sha256=loaded.receipt.manifest_sha256,
        source_prepared_data_sha256=loaded.receipt.prepared_data_sha256,
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
        macroblock_ids_sha256=macroblock_sha,
        preprocessing_sha256=_pooled_full_core._canonical_sha256(  # noqa: SLF001
            final_payload
        ),
    )
    qc = _pooled_full_core.PooledCoreQC(
        alias=loaded.receipt.alias,
        n_nodes=loaded.receipt.n_nodes,
        n_genes=len(gene_names),
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
        raise PooledEnsembleComparisonError(
            f"streamed pooled output is non-finite for {loaded.receipt.alias}"
        )
    return _pooled_full_core.PooledCoreData(
        alias=loaded.receipt.alias,
        expression_counts=loaded.expression_counts,
        target_expression=target,
        node_covariates=loaded.node_covariates,
        coordinates_um=loaded.coordinates_um,
        macroblock_ids=loaded.macroblock_ids,
        gene_names=gene_names,
        metadata_names=metadata_names,
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


def _reference_array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _combine_equal_core_references(
    references: Mapping[str, HybridCountReferences],
    *,
    expression_mean: np.ndarray,
    expression_scale: np.ndarray,
) -> HybridCountReferences:
    """Combine compact per-core references using the frozen equal-core rule."""

    if tuple(references) != ALIASES:
        raise PooledEnsembleComparisonError(
            "pooled references do not follow the frozen alias order"
        )
    detection = np.mean(
        np.stack(
            [
                references[alias].detection_probability
                for alias in ALIASES
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    ordinal = np.mean(
        np.stack(
            [
                references[alias].positive_ordinal_probability
                for alias in ALIASES
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    median_log = np.median(
        np.stack(
            [
                references[alias].positive_continuous_standardized
                * expression_scale
                + expression_mean
                for alias in ALIASES
            ],
            axis=0,
        ),
        axis=0,
    )
    continuous = (median_log - expression_mean) / expression_scale
    if (
        np.any(detection <= 0)
        or np.any(detection >= 1)
        or np.any(ordinal <= 0)
        or np.any(ordinal >= 1)
        or not np.isfinite(continuous).all()
    ):
        raise PooledEnsembleComparisonError(
            "combined pooled reference is invalid"
        )
    detected_state = detection >= 0.5
    positive_state = 1 + np.sum(ordinal >= 0.5, axis=1)
    count_state = np.where(detected_state, positive_state, 0).astype(
        np.int64,
        copy=False,
    )
    arrays = {
        "detection_probability": detection,
        "positive_ordinal_probability": ordinal,
        "positive_continuous_standardized": continuous,
        "count_state": count_state,
    }
    return HybridCountReferences(
        detection_probability=detection,
        positive_ordinal_probability=ordinal,
        positive_continuous_standardized=continuous,
        detected_state=detected_state,
        positive_state=positive_state.astype(np.int64, copy=False),
        count_state=count_state,
        audit={
            "schema": "hybrid_count_equal_core_pooled_references_v1",
            "fit_scope": "all_nodes_all_ten_cores_transductive",
            "core_weighting": "equal",
            "core_count": len(ALIASES),
            "aliases": list(ALIASES),
            "detection_aggregation": (
                "mean_per_core_jeffreys_probability"
            ),
            "ordinal_aggregation": "mean_per_core_jeffreys_probability",
            "continuous_aggregation": (
                "median_of_per_core_positive_medians"
            ),
            "shared_standardization": True,
            "reference_sha256": _reference_array_digest(arrays),
        },
    )


def load_streaming_pooled_cohort(
    prepared_artifacts: (
        Mapping[str, str | Path]
        | Sequence[tuple[str, str | Path]]
        | None
    ) = None,
    *,
    data_root: str | Path | None = None,
    epsilon: float = _pooled_full_core.EXPRESSION_SCALE_EPSILON,
) -> StreamingPooledFullCoreCohort:
    """Fit pooled state while never retaining more than one core matrix."""

    inputs = _pooled_full_core._normalise_artifact_inputs(  # noqa: SLF001
        prepared_artifacts,
        data_root=data_root,
    )
    receipts = {
        receipt.alias: receipt
        for receipt in _pooled_full_core.EXPECTED_PREPARED_CORES
    }
    expected_total_nodes = sum(
        receipt.n_nodes
        for receipt in _pooled_full_core.EXPECTED_PREPARED_CORES
    )
    if tuple(receipts) != ALIASES:
        raise PooledEnsembleComparisonError(
            "frozen pooled receipts changed alias order"
        )

    means: list[np.ndarray] = []
    second_moments: list[np.ndarray] = []
    common_genes: tuple[str, ...] | None = None
    common_metadata: tuple[str, ...] | None = None
    total_nodes = 0
    for alias, path in inputs:
        loaded, genes, metadata = _pooled_full_core._load_verified_core(  # noqa: SLF001
            alias,
            path,
            receipt=receipts[alias],
            epsilon=epsilon,
        )
        if common_genes is None:
            common_genes = genes
            common_metadata = metadata
        elif genes != common_genes or metadata != common_metadata:
            raise PooledEnsembleComparisonError(
                f"ordered schema changed at {alias}"
            )
        transformed = np.empty(loaded.expression_counts.shape, dtype=np.float64)
        np.log1p(loaded.expression_counts, out=transformed)
        means.append(transformed.mean(axis=0, dtype=np.float64))
        np.square(transformed, out=transformed)
        second_moments.append(
            transformed.mean(axis=0, dtype=np.float64)
        )
        total_nodes += int(loaded.expression_counts.shape[0])
        del loaded, transformed
        gc.collect()
    if common_genes is None or common_metadata is None:
        raise PooledEnsembleComparisonError("streamed pooled cohort is empty")
    if total_nodes != expected_total_nodes:
        raise PooledEnsembleComparisonError(
            "streamed pooled cell total changed"
        )
    expression_mean = np.mean(
        np.stack(means, axis=0),
        axis=0,
        dtype=np.float64,
    )
    second = np.mean(
        np.stack(second_moments, axis=0),
        axis=0,
        dtype=np.float64,
    )
    raw_scale = np.sqrt(
        np.maximum(second - np.square(expression_mean), 0.0)
    )
    expression_scale = np.where(raw_scale > epsilon, raw_scale, 1.0)
    if (
        not np.isfinite(expression_mean).all()
        or not np.isfinite(expression_scale).all()
        or np.any(expression_scale <= 0)
    ):
        raise PooledEnsembleComparisonError(
            "streamed pooled expression moments are invalid"
        )
    expression_mean = _pooled_full_core._readonly(  # noqa: SLF001
        expression_mean,
        dtype=np.float64,
    )
    expression_scale = _pooled_full_core._readonly(  # noqa: SLF001
        expression_scale,
        dtype=np.float64,
    )
    del means, second_moments, second, raw_scale

    per_core_references: dict[str, HybridCountReferences] = {}
    core_preprocessing_sha256: dict[str, str] = {}
    for alias, path in inputs:
        loaded, genes, metadata = _pooled_full_core._load_verified_core(  # noqa: SLF001
            alias,
            path,
            receipt=receipts[alias],
            epsilon=epsilon,
        )
        if genes != common_genes or metadata != common_metadata:
            raise PooledEnsembleComparisonError(
                f"ordered schema changed while fitting references at {alias}"
            )
        core = _pooled_core_from_loaded(
            loaded,
            gene_names=common_genes,
            metadata_names=common_metadata,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
        )
        per_core_references[alias] = fit_hybrid_count_references(
            core.expression_counts,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
        )
        core_preprocessing_sha256[alias] = (
            core.checksums.preprocessing_sha256
        )
        del loaded, core
        gc.collect()

    expression_mean_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_mean",
        expression_mean,
    )
    expression_scale_sha = _full_core._array_sha256(  # noqa: SLF001
        "pooled_expression_scale",
        expression_scale,
    )
    ordered_sources_sha = _pooled_full_core._canonical_sha256(  # noqa: SLF001
        [
            {
                "alias": receipt.alias,
                "n_nodes": receipt.n_nodes,
                "manifest_sha256": receipt.manifest_sha256,
                "prepared_data_sha256": receipt.prepared_data_sha256,
            }
            for receipt in _pooled_full_core.EXPECTED_PREPARED_CORES
        ]
    )
    gene_schema_sha = _pooled_full_core._canonical_sha256(  # noqa: SLF001
        list(common_genes)
    )
    metadata_schema_sha = _pooled_full_core._canonical_sha256(  # noqa: SLF001
        list(common_metadata)
    )
    cohort_payload = {
        "schema": "pooled_full_core_cohort_v1",
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "aliases": list(ALIASES),
        "total_nodes": total_nodes,
        "ordered_sources_sha256": ordered_sources_sha,
        "ordered_gene_schema_sha256": gene_schema_sha,
        "ordered_metadata_schema_sha256": metadata_schema_sha,
        "expression_mean_sha256": expression_mean_sha,
        "expression_scale_sha256": expression_scale_sha,
        "core_preprocessing_sha256": [
            core_preprocessing_sha256[alias] for alias in ALIASES
        ],
    }
    checksums = _pooled_full_core.PooledCohortChecksums(
        ordered_sources_sha256=ordered_sources_sha,
        ordered_gene_schema_sha256=gene_schema_sha,
        ordered_metadata_schema_sha256=metadata_schema_sha,
        expression_mean_sha256=expression_mean_sha,
        expression_scale_sha256=expression_scale_sha,
        combined_fingerprint_sha256=(
            _pooled_full_core._canonical_sha256(cohort_payload)  # noqa: SLF001
        ),
    )
    pooled_references = _combine_equal_core_references(
        per_core_references,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
    )
    return StreamingPooledFullCoreCohort(
        prepared_artifacts=inputs,
        aliases=ALIASES,
        gene_names=common_genes,
        metadata_names=common_metadata,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        total_nodes=total_nodes,
        checksums=checksums,
        pooled_references=pooled_references,
        core_preprocessing_sha256=MappingProxyType(
            dict(core_preprocessing_sha256)
        ),
        epsilon=float(epsilon),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PooledEnsembleComparisonError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PooledEnsembleComparisonError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PooledEnsembleComparisonError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PooledEnsembleComparisonError(f"{label} must be an integer")
    return converted


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PooledEnsembleComparisonError(f"{label} must be finite numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PooledEnsembleComparisonError(
            f"{label} must be finite numeric"
        ) from exc
    if not math.isfinite(converted):
        raise PooledEnsembleComparisonError(f"{label} must be finite numeric")
    return converted


def _validate_required_metric_fields(
    row: Mapping[str, Any],
    *,
    label: str,
    require_equal_core_reference: bool,
) -> None:
    """Fail closed when a required scalar metric is absent or non-finite."""

    required = REQUIRED_MODEL_SCALAR_METRICS
    if require_equal_core_reference:
        required += REQUIRED_EQUAL_CORE_REFERENCE_METRICS
    for field in required:
        _finite(row.get(field), f"{label} {field}")
    precision = row.get("detection_precision")
    if precision is not None:
        value = _finite(precision, f"{label} detection_precision")
        if not 0.0 <= value <= 1.0:
            raise PooledEnsembleComparisonError(
                f"{label} detection_precision must lie in [0, 1]"
            )


def _assert_alias_safe_payload(value: Any, *, label: str) -> None:
    """Reject protected-identifier fields before any report file is written."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = (
                str(raw_key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if key in FORBIDDEN_REPORT_KEYS:
                raise PooledEnsembleComparisonError(
                    f"{label} contains prohibited identifier field {raw_key!r}"
                )
            _assert_alias_safe_payload(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_alias_safe_payload(item, label=label)


def _sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PooledEnsembleComparisonError(f"{label} must be a lowercase SHA-256")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_code_fingerprints(paths: ProjectPaths) -> dict[str, Any]:
    """Hash the entry point, runner helpers, and complete evaluator package."""

    candidates = {
        (paths.project_root / relative).resolve(strict=False)
        for relative in ANALYSIS_HELPER_RELATIVE_PATHS
    }
    candidates.update(
        path.resolve(strict=False)
        for path in (
            paths.project_root / "src" / "spatial_benchmark"
        ).rglob("*.py")
    )
    if not candidates or any(not path.is_file() for path in candidates):
        raise PooledEnsembleComparisonError(
            "analysis source fingerprint set is incomplete"
        )
    files = {
        path.relative_to(paths.project_root).as_posix(): _sha256_file(path)
        for path in sorted(candidates)
    }
    return {
        "scope": (
            "analysis_entry_runner_helpers_and_all_spatial_benchmark_python"
        ),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def _analysis_input_fingerprints(
    *,
    paths: ProjectPaths,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
    prior_categorical_path: Path,
) -> dict[str, str]:
    inputs = {
        "frozen_task_contract": (
            paths.project_root
            / "experiments"
            / "campaigns"
            / CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ),
        "materialization_receipt": materialization_path,
        "pilot_enqueue_receipt": pilot_enqueue_path,
        "pilot_gate_receipt": pilot_gate_path,
        "production_enqueue_receipt": production_enqueue_path,
        "prior_categorical_comparison": prior_categorical_path,
    }
    if any(not path.is_file() for path in inputs.values()):
        raise PooledEnsembleComparisonError(
            "analysis input fingerprint set is incomplete"
        )
    return {
        label: _sha256_file(path.resolve(strict=True))
        for label, path in sorted(inputs.items())
    }


def _analysis_runtime_environment(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_geometric_version": importlib.metadata.version(
            "torch-geometric"
        ),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": device_name,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise PooledEnsembleComparisonError(
                "CUDA analysis device was requested but is unavailable"
            )
        index = (
            torch.cuda.current_device()
            if device.index is None
            else int(device.index)
        )
        properties = torch.cuda.get_device_properties(index)
        result["evaluation_device"] = {
            "type": "cuda",
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    else:
        result["evaluation_device"] = {"type": str(device.type)}
    return result


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PooledEnsembleComparisonError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PooledEnsembleComparisonError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except PooledEnsembleComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PooledEnsembleComparisonError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PooledEnsembleComparisonError(f"{label} is not valid YAML") from exc
    return dict(_mapping(value, label))


def _verified_signed_json(path: Path, *, label: str) -> dict[str, Any]:
    value = _strict_json(path, label=label)
    checksum = _sha256(value.get("checksum"), f"{label} checksum")
    unsigned = dict(value)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise PooledEnsembleComparisonError(f"{label} checksum does not verify")
    return value


def _logical_table_path(root: Path, stem: str) -> Path:
    paths = [
        root / f"{stem}{suffix}"
        for suffix in TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(paths) != 1:
        raise PooledEnsembleComparisonError(
            f"{root.name} requires exactly one {stem} table; found "
            f"{[path.name for path in paths]}"
        )
    return paths[0]


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    rows.append(
                        dict(
                            _mapping(
                                json.loads(line),
                                f"{path.name} row {line_number}",
                            )
                        )
                    )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PooledEnsembleComparisonError(
                f"table is unreadable: {path}"
            ) from exc
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise PooledEnsembleComparisonError(
                f"table is unreadable: {path}"
            ) from exc
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise PooledEnsembleComparisonError(
                f"pyarrow is required to read {path}"
            ) from exc
        try:
            rows = [dict(row) for row in parquet.read_table(path).to_pylist()]
        except Exception as exc:
            raise PooledEnsembleComparisonError(
                f"table is unreadable: {path}"
            ) from exc
    else:
        raise PooledEnsembleComparisonError(f"unsupported table format: {path}")
    if not rows:
        raise PooledEnsembleComparisonError(f"required table is empty: {path}")
    return rows


def _slot_from_config(config: Mapping[str, Any], *, require_production: bool) -> tuple[str, int]:
    try:
        contract = _validate_pooled_contract(config)
    except Exception as exc:
        raise PooledEnsembleComparisonError(
            "resolved pooled configuration violates the frozen runner contract"
        ) from exc
    if require_production and contract.diagnostic_resource_pilot:
        raise PooledEnsembleComparisonError(
            "a resource pilot cannot occupy a production slot"
        )
    if not require_production and not contract.diagnostic_resource_pilot:
        raise PooledEnsembleComparisonError(
            "a production configuration cannot occupy a pilot slot"
        )
    return contract.public_variant, contract.seed


def _normalized_root_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = dict(config)
    normalized["attempt"] = 1
    return canonical_sha256(normalized)


def _production_like(config: Mapping[str, Any]) -> bool:
    campaign = config.get("campaign")
    trainer = config.get("trainer")
    experiment = config.get("experiment")
    metadata = config.get("metadata")
    return bool(
        isinstance(campaign, Mapping)
        and campaign.get("campaign_id") == CAMPAIGN_ID
        and isinstance(trainer, Mapping)
        and trainer.get("diagnostic_resource_pilot") is False
        and isinstance(experiment, Mapping)
        and experiment.get("resource_pilot") is False
        and isinstance(metadata, Mapping)
        and metadata.get("execution_role") == "production"
    )


def _pilot_like(config: Mapping[str, Any]) -> bool:
    campaign = config.get("campaign")
    trainer = config.get("trainer")
    experiment = config.get("experiment")
    metadata = config.get("metadata")
    return bool(
        isinstance(campaign, Mapping)
        and campaign.get("campaign_id") == CAMPAIGN_ID
        and isinstance(trainer, Mapping)
        and trainer.get("diagnostic_resource_pilot") is True
        and isinstance(experiment, Mapping)
        and experiment.get("resource_pilot") is True
        and isinstance(metadata, Mapping)
        and metadata.get("execution_role") == "resource_pilot"
    )


def validate_campaign_receipts(
    *,
    paths: ProjectPaths,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate frozen inputs and the authorization chain before registry reads."""

    materialization = _verified_signed_json(
        materialization_path, label="pooled materialization receipt"
    )
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    counts = _mapping(materialization.get("counts"), "materialization counts")
    assignment = _mapping(
        materialization.get("assignment"), "materialization GPU assignment"
    )
    jobs = materialization.get("production_jobs")
    pilot_materialized_jobs = materialization.get("pilot_jobs")
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or frozen.get("sha256") != FROZEN_CONTRACT_SHA256
        or counts.get("aliases") != 10
        or counts.get("production_configs") != 14
        or counts.get("production_seeds") != 7
        or materialization.get("allowed_gpu_ids") != sorted(SAFE_GPU_IDS)
        or materialization.get("excluded_gpu_ids") != [4]
        or assignment.get("policy") != "fixed_seed_to_gpu_v1"
        or materialization.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or materialization.get("registry_mutation_performed") is not False
        or materialization.get("queue_mutation_performed") is not False
        or materialization.get("training_performed") is not False
        or not isinstance(jobs, list)
        or len(jobs) != 14
        or not all(isinstance(job, Mapping) for job in jobs)
        or not isinstance(pilot_materialized_jobs, list)
        or len(pilot_materialized_jobs) != 2
        or not all(
            isinstance(job, Mapping) for job in pilot_materialized_jobs
        )
    ):
        raise PooledEnsembleComparisonError(
            "materialization receipt does not describe the frozen 14-run campaign"
        )
    contract_path = paths.project_root / str(frozen.get("reference"))
    if _sha256_file(contract_path) != FROZEN_CONTRACT_SHA256:
        raise PooledEnsembleComparisonError("frozen task contract checksum changed")
    component_files = _mapping(
        materialization.get("component_file_sha256"),
        "materialization component file checksums",
    )
    for reference, expected in component_files.items():
        file_path = paths.project_root / str(reference)
        if not file_path.is_file() or _sha256_file(file_path) != _sha256(
            expected, f"component checksum {reference}"
        ):
            raise PooledEnsembleComparisonError(
                f"materialized component file changed: {reference}"
            )
    observed_slots: set[tuple[str, int]] = set()
    for job in jobs:
        arm = str(job.get("arm"))
        seed = _integer(job.get("seed"), "materialized production seed")
        requested_gpu = _integer(
            job.get("requested_gpu"), "materialized requested GPU"
        )
        slot = (arm, seed)
        seed_gpu_map = _mapping(
            assignment.get("seed_gpu_map"), "materialized seed GPU map"
        )
        if (
            arm not in ARMS
            or seed not in SEEDS
            or slot in observed_slots
            or requested_gpu not in SAFE_GPU_IDS
            or _integer(
                seed_gpu_map.get(str(seed)), "materialized seed GPU assignment"
            )
            != requested_gpu
        ):
            raise PooledEnsembleComparisonError(
                "materialization production slot or safe-GPU assignment is invalid"
            )
        observed_slots.add(slot)
        reference = paths.project_root / str(job.get("config"))
        if not reference.is_file():
            raise PooledEnsembleComparisonError(
                f"materialized production config is missing for {slot}"
            )
        if _sha256_file(reference) != _sha256(
            job.get("file_sha256"), f"materialized production file {slot}"
        ):
            raise PooledEnsembleComparisonError(
                f"materialized production config file changed for {slot}"
            )
        config = _yaml_mapping(reference, label=f"materialized config {slot}")
        if _slot_from_config(config, require_production=True) != slot:
            raise PooledEnsembleComparisonError(
                f"materialized production config has wrong slot for {slot}"
            )
        if _normalized_root_config_sha256(config) != _sha256(
            job.get("config_sha256"), f"materialized config digest {slot}"
        ):
            raise PooledEnsembleComparisonError(
                f"materialized production config digest changed for {slot}"
            )
    if observed_slots != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise PooledEnsembleComparisonError(
            "materialization lacks exact two-arm seven-seed coverage"
        )

    materialized_pilot_by_slot: dict[
        tuple[str, int], Mapping[str, Any]
    ] = {}
    for job in pilot_materialized_jobs:
        slot = (
            str(job.get("arm")),
            _integer(job.get("seed"), "materialized pilot seed"),
        )
        if (
            slot not in {(arm, 0) for arm in ARMS}
            or slot in materialized_pilot_by_slot
            or _integer(
                job.get("requested_gpu"), "materialized pilot requested GPU"
            )
            not in SAFE_GPU_IDS
        ):
            raise PooledEnsembleComparisonError(
                "materialization pilot slot or safe-GPU assignment is invalid"
            )
        reference = paths.project_root / str(job.get("config"))
        if not reference.is_file() or _sha256_file(reference) != _sha256(
            job.get("file_sha256"), f"materialized pilot file {slot}"
        ):
            raise PooledEnsembleComparisonError(
                f"materialized pilot configuration changed for {slot}"
            )
        config = _yaml_mapping(reference, label=f"materialized pilot config {slot}")
        if (
            _slot_from_config(config, require_production=False) != slot
            or _normalized_root_config_sha256(config)
            != _sha256(
                job.get("config_sha256"),
                f"materialized pilot config digest {slot}",
            )
        ):
            raise PooledEnsembleComparisonError(
                f"materialized pilot configuration has wrong identity for {slot}"
            )
        materialized_pilot_by_slot[slot] = job
    if set(materialized_pilot_by_slot) != {(arm, 0) for arm in ARMS}:
        raise PooledEnsembleComparisonError(
            "materialization lacks exact paired seed-zero pilot coverage"
        )

    pilot_enqueue = _verified_signed_json(
        pilot_enqueue_path, label="pooled pilot enqueue receipt"
    )
    pilot_enqueue_jobs = pilot_enqueue.get("jobs")
    if (
        pilot_enqueue.get("schema_version") != 1
        or pilot_enqueue.get("receipt_kind") != PILOT_ENQUEUE_KIND
        or pilot_enqueue.get("campaign_id") != CAMPAIGN_ID
        or pilot_enqueue.get("stage") != "pilot"
        or pilot_enqueue.get("materialization_checksum")
        != materialization.get("checksum")
        or pilot_enqueue.get("complete") is not True
        or not isinstance(pilot_enqueue_jobs, list)
        or len(pilot_enqueue_jobs) != 2
        or not all(isinstance(job, Mapping) for job in pilot_enqueue_jobs)
    ):
        raise PooledEnsembleComparisonError(
            "pilot enqueue receipt is incomplete or has the wrong identity"
        )
    pilot_enqueue_by_slot: dict[tuple[str, int], Mapping[str, Any]] = {}
    pilot_root_job_ids: set[str] = set()
    for job in pilot_enqueue_jobs:
        slot = (
            str(job.get("arm")),
            _integer(job.get("seed"), "pilot enqueue seed"),
        )
        root_job_id = job.get("job_id")
        planned = materialized_pilot_by_slot.get(slot)
        if (
            planned is None
            or slot in pilot_enqueue_by_slot
            or not isinstance(root_job_id, str)
            or not root_job_id
            or root_job_id in pilot_root_job_ids
            or job.get("config_sha256") != planned.get("config_sha256")
            or job.get("maximum_attempts") != 2
            or _integer(
                job.get("requested_gpu"), "pilot enqueue requested GPU"
            )
            != _integer(
                planned.get("requested_gpu"),
                "materialized pilot requested GPU",
            )
        ):
            raise PooledEnsembleComparisonError(
                f"pilot enqueue identity changed for {slot}"
            )
        pilot_enqueue_by_slot[slot] = job
        pilot_root_job_ids.add(root_job_id)
    if set(pilot_enqueue_by_slot) != set(materialized_pilot_by_slot):
        raise PooledEnsembleComparisonError(
            "pilot enqueue receipt lacks exact paired seed-zero coverage"
        )

    pilot = _verified_signed_json(pilot_gate_path, label="pooled pilot gate")
    pilot_jobs = pilot.get("jobs")
    required_pilot_flags = (
        "same_frozen_precision_batches_all_cores",
        "same_evaluation_masks",
        "same_verified_graph_bundle",
        "paired_initialization_digests_match",
    )
    if (
        pilot.get("schema_version") != 1
        or pilot.get("receipt_kind") != PILOT_GATE_KIND
        or pilot.get("campaign_id") != CAMPAIGN_ID
        or pilot.get("materialization_checksum") != materialization.get("checksum")
        or pilot.get("pilot_enqueue_receipt_checksum")
        != pilot_enqueue.get("checksum")
        or pilot.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
        or pilot.get("gate_passed") is not True
        or pilot.get("production_authorized") is not True
        or pilot.get("failure_reasons") != []
        or any(pilot.get(field) is not True for field in required_pilot_flags)
        or not isinstance(pilot_jobs, list)
        or len(pilot_jobs) != 2
    ):
        raise PooledEnsembleComparisonError(
            "production lacks the exact passing pooled pilot gate"
        )
    required_pilot_job_flags = (
        "verified_bundle",
        "checkpoint_verified",
        "finite_losses_and_gradients",
        "all_20_optimizer_steps_completed",
        "every_core_once_each_epoch",
        "parameter_match",
        "paired_initialization_match",
        "precision_equivalence_passed",
        "peak_vram_passed",
        "peak_host_memory_passed",
        "projected_runtime_passed",
        "projected_disk_passed",
        "runner_pilot_gate_passed",
    )
    if {
        (str(job.get("arm")), _integer(job.get("seed"), "pilot seed"))
        for job in pilot_jobs
        if isinstance(job, Mapping)
    } != {(arm, 0) for arm in ARMS}:
        raise PooledEnsembleComparisonError(
            "pilot gate lacks the exact paired seed-zero arm coverage"
        )
    for job in pilot_jobs:
        item = _mapping(job, "pilot gate job")
        slot = (
            str(item.get("arm")),
            _integer(item.get("seed"), "pilot gate seed"),
        )
        enqueued = pilot_enqueue_by_slot.get(slot)
        if (
            enqueued is None
            or item.get("original_enqueue_job_id") != enqueued.get("job_id")
            or item.get("job_id") != enqueued.get("job_id")
            or item.get("config_sha256") != enqueued.get("config_sha256")
            or item.get("materialization_checksum")
            != materialization.get("checksum")
            or item.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or any(item.get(field) is not True for field in required_pilot_job_flags)
        ):
            raise PooledEnsembleComparisonError(
                f"pilot gate contains a failed arm check for {item.get('arm')}"
            )

    enqueue = _verified_signed_json(
        production_enqueue_path, label="pooled production enqueue receipt"
    )
    enqueue_jobs = enqueue.get("jobs")
    if (
        enqueue.get("schema_version") != 1
        or enqueue.get("receipt_kind") != PRODUCTION_ENQUEUE_KIND
        or enqueue.get("campaign_id") != CAMPAIGN_ID
        or enqueue.get("stage") != "production"
        or enqueue.get("materialization_checksum") != materialization.get("checksum")
        or enqueue.get("pilot_gate_checksum") != pilot.get("checksum")
        or enqueue.get("complete") is not True
        or not isinstance(enqueue_jobs, list)
        or len(enqueue_jobs) != 14
        or not all(isinstance(job, Mapping) for job in enqueue_jobs)
    ):
        raise PooledEnsembleComparisonError(
            "production enqueue receipt is incomplete or not bound to the pilot gate"
        )
    enqueue_slots = {
        (str(job.get("arm")), _integer(job.get("seed"), "enqueue seed"))
        for job in enqueue_jobs
    }
    if enqueue_slots != observed_slots:
        raise PooledEnsembleComparisonError(
            "production enqueue receipt does not contain the 14 frozen slots"
        )
    materialized_by_slot = {
        (str(job["arm"]), int(job["seed"])): job for job in jobs
    }
    for job in enqueue_jobs:
        slot = (str(job["arm"]), int(job["seed"]))
        if (
            job.get("config_sha256")
            != materialized_by_slot[slot].get("config_sha256")
            or job.get("maximum_attempts") != 2
            or _integer(job.get("requested_gpu"), "enqueue requested GPU")
            != _integer(
                materialized_by_slot[slot].get("requested_gpu"),
                "materialized requested GPU",
            )
        ):
            raise PooledEnsembleComparisonError(
                f"production enqueue identity changed for {slot}"
            )
    return materialization, pilot_enqueue, pilot, enqueue


def _campaign_queue_rows(registry: Registry) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        raw = connection.execute(
            """
            SELECT job_id, campaign_id, canonical_config_json, status,
                   attempt_count, maximum_attempts, requested_gpu, retry_of,
                   run_id, failure_category, last_error, created_at
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    rows: list[dict[str, Any]] = []
    for row in raw:
        item = dict(row)
        try:
            config = json.loads(str(item.pop("canonical_config_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PooledEnsembleComparisonError(
                "campaign queue contains invalid canonical configuration JSON"
            ) from exc
        item["canonical_config"] = dict(_mapping(config, "queue config"))
        rows.append(item)
    return rows


def _campaign_run_rows(registry: Registry) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        raw = connection.execute(
            """
            SELECT run_id, campaign_id, status, attempt, retry_of, config_json
            FROM runs
            WHERE campaign_id = ?
            ORDER BY created_at, run_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    rows: list[dict[str, Any]] = []
    for row in raw:
        item = dict(row)
        try:
            config = json.loads(str(item.pop("config_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PooledEnsembleComparisonError(
                "campaign run inventory contains invalid configuration JSON"
            ) from exc
        item["config"] = dict(_mapping(config, "registered run config"))
        rows.append(item)
    return rows


def validate_registered_production_membership(
    *,
    run_rows: Sequence[Mapping[str, Any]],
    audit: CampaignAudit,
) -> None:
    """Reject unqueued, duplicate, or omitted production-like run records."""

    expected_run_ids = {
        str(row["run_id"])
        for row in audit.attempt_inventory
        if isinstance(row.get("run_id"), str) and row.get("run_id")
    }
    production_rows: list[Mapping[str, Any]] = []
    for row in run_rows:
        config = _mapping(row.get("config"), "registered run config")
        try:
            _slot_from_config(
                config,
                require_production=True,
            )
        except PooledEnsembleComparisonError:
            if _production_like(config):
                raise PooledEnsembleComparisonError(
                    "registry contains a production-like run with invalid semantics"
                )
            continue
        production_rows.append(row)
    observed_run_ids = {str(row["run_id"]) for row in production_rows}
    if observed_run_ids != expected_run_ids:
        raise PooledEnsembleComparisonError(
            "registry production run membership differs from queue attempt lineages"
        )
    completed = [row for row in production_rows if row.get("status") == "completed"]
    if len(completed) != 14:
        raise PooledEnsembleComparisonError(
            "registry must contain exactly fourteen completed production runs"
        )


def resolve_pilot_lineages(
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    materialization: Mapping[str, Any],
    pilot_enqueue: Mapping[str, Any],
    pilot_gate: Mapping[str, Any],
    run_lookup: Callable[[str], Mapping[str, Any] | None],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Reconcile the signed pilot lineage with current queue and run records."""

    by_id = {str(row["job_id"]): row for row in queue_rows}
    if len(by_id) != len(queue_rows):
        raise PooledEnsembleComparisonError("campaign queue has duplicate job IDs")
    planned = {
        (str(job["arm"]), int(job["seed"])): job
        for job in materialization["pilot_jobs"]
    }
    enqueued = {
        (str(job["arm"]), int(job["seed"])): job
        for job in pilot_enqueue["jobs"]
    }
    gated = {
        (str(job["arm"]), int(job["seed"])): job
        for job in pilot_gate["jobs"]
    }
    expected_slots = {(arm, 0) for arm in ARMS}
    if (
        set(planned) != expected_slots
        or set(enqueued) != expected_slots
        or set(gated) != expected_slots
    ):
        raise PooledEnsembleComparisonError(
            "pilot receipts do not have exact paired seed-zero coverage"
        )

    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    recognized_job_ids: set[str] = set()
    for slot in sorted(expected_slots):
        planned_job = planned[slot]
        enqueue_job = enqueued[slot]
        gate_job = _mapping(gated[slot], f"pilot gate job {slot}")
        raw_attempts = gate_job.get("attempts")
        raw_failures = gate_job.get("failed_attempts")
        if (
            not isinstance(raw_attempts, list)
            or not 1 <= len(raw_attempts) <= 2
            or not all(isinstance(row, Mapping) for row in raw_attempts)
            or not isinstance(raw_failures, list)
            or not all(isinstance(row, Mapping) for row in raw_failures)
        ):
            raise PooledEnsembleComparisonError(
                f"pilot gate has an invalid attempt inventory for {slot}"
            )
        attempts = [
            _mapping(row, f"pilot attempt {slot}") for row in raw_attempts
        ]
        expected_attempt_numbers = list(range(1, len(attempts) + 1))
        observed_attempt_numbers = [
            _integer(row.get("attempt"), f"pilot attempt number {slot}")
            for row in attempts
        ]
        if (
            observed_attempt_numbers != expected_attempt_numbers
            or gate_job.get("attempt_count") != len(attempts)
            or gate_job.get("completed_attempt") != len(attempts)
            or gate_job.get("original_enqueue_job_id")
            != enqueue_job.get("job_id")
            or gate_job.get("job_id") != enqueue_job.get("job_id")
            or gate_job.get("config_sha256")
            != planned_job.get("config_sha256")
        ):
            raise PooledEnsembleComparisonError(
                f"pilot gate lineage identity changed for {slot}"
            )
        expected_failed_attempts = [
            dict(row) for row in attempts if not row.get("selected_completed_attempt")
        ]
        if canonical_sha256(raw_failures) != canonical_sha256(
            expected_failed_attempts
        ):
            raise PooledEnsembleComparisonError(
                f"pilot gate failed-attempt inventory changed for {slot}"
            )

        previous_job_id: str | None = None
        previous_run_id: str | None = None
        for index, receipt_attempt in enumerate(attempts):
            job_id = receipt_attempt.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise PooledEnsembleComparisonError(
                    f"pilot attempt lacks a queue job ID for {slot}"
                )
            queue_row = by_id.get(job_id)
            if queue_row is None or job_id in recognized_job_ids:
                raise PooledEnsembleComparisonError(
                    f"pilot attempt is absent or duplicated in the queue for {slot}"
                )
            recognized_job_ids.add(job_id)
            queue_config = _mapping(
                queue_row.get("canonical_config"), "pilot queue config"
            )
            selected = index == len(attempts) - 1
            expected_statuses = {"completed"} if selected else {
                "failed",
                "stale",
                "pruned",
            }
            run_id = queue_row.get("run_id")
            if (
                _slot_from_config(queue_config, require_production=False) != slot
                or _normalized_root_config_sha256(queue_config)
                != planned_job.get("config_sha256")
                or _integer(
                    queue_row.get("attempt_count"), "pilot queue attempt"
                )
                != index + 1
                or _integer(
                    queue_row.get("maximum_attempts"),
                    "pilot queue attempt budget",
                )
                != 2
                or _integer(
                    queue_row.get("requested_gpu"),
                    "pilot queue requested GPU",
                )
                != _integer(
                    planned_job.get("requested_gpu"),
                    "materialized pilot requested GPU",
                )
                or queue_row.get("retry_of") != previous_job_id
                or queue_row.get("status") not in expected_statuses
                or receipt_attempt.get("attempt") != index + 1
                or receipt_attempt.get("maximum_attempts") != 2
                or receipt_attempt.get("retry_of") != previous_job_id
                or receipt_attempt.get("run_id") != run_id
                or receipt_attempt.get("status") != queue_row.get("status")
                or receipt_attempt.get("failure_category")
                != queue_row.get("failure_category")
                or receipt_attempt.get("selected_completed_attempt") is not selected
                or receipt_attempt.get("is_original_enqueue_job")
                is not (index == 0)
            ):
                raise PooledEnsembleComparisonError(
                    f"pilot queue lineage differs from its signed gate for {slot}"
                )
            if not isinstance(run_id, str) or not run_id:
                raise PooledEnsembleComparisonError(
                    f"pilot queue attempt has no registered run for {slot}"
                )
            run = run_lookup(run_id)
            if (
                run is None
                or run.get("campaign_id") != CAMPAIGN_ID
                or run.get("status") != queue_row.get("status")
                or _integer(run.get("attempt"), "registered pilot run attempt")
                != index + 1
                or run.get("retry_of") != previous_run_id
            ):
                raise PooledEnsembleComparisonError(
                    f"pilot run lineage differs from its queue record for {slot}"
                )
            record = {
                "stage": "pilot",
                "arm": slot[0],
                "seed": slot[1],
                "job_id": job_id,
                "attempt": index + 1,
                "status": str(queue_row["status"]),
                "run_id": run_id,
                "run_status": run.get("status"),
                "failure_category": queue_row.get("failure_category"),
                "last_error_present": bool(queue_row.get("last_error")),
                "selected_completed_attempt": selected,
            }
            inventory.append(record)
            if not selected:
                failures.append(record)
            previous_job_id = job_id
            previous_run_id = run_id
        if (
            gate_job.get("completed_job_id") != previous_job_id
            or gate_job.get("run_id") != previous_run_id
        ):
            raise PooledEnsembleComparisonError(
                f"pilot completed attempt changed for {slot}"
            )

    for row in queue_rows:
        config = _mapping(row.get("canonical_config"), "campaign queue config")
        try:
            slot = _slot_from_config(config, require_production=False)
        except PooledEnsembleComparisonError:
            if _pilot_like(config):
                raise PooledEnsembleComparisonError(
                    "campaign contains a pilot-like job with invalid semantics"
                )
            continue
        if (
            slot in expected_slots
            and str(row["job_id"]) not in recognized_job_ids
        ):
            raise PooledEnsembleComparisonError(
                "campaign contains an unexpected or orphaned pilot job"
            )
    if len(inventory) != len(recognized_job_ids):
        raise PooledEnsembleComparisonError(
            "pilot queue inventory did not reconcile exactly"
        )
    return tuple(inventory), tuple(failures)


def resolve_production_lineages(
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    materialization: Mapping[str, Any],
    enqueue: Mapping[str, Any],
    run_lookup: Callable[[str], Mapping[str, Any] | None],
) -> CampaignAudit:
    """Resolve all retry attempts and require one completed run per slot."""

    by_id = {str(row["job_id"]): row for row in queue_rows}
    if len(by_id) != len(queue_rows):
        raise PooledEnsembleComparisonError("campaign queue has duplicate job IDs")
    children: dict[str, list[Mapping[str, Any]]] = {}
    for row in queue_rows:
        parent = row.get("retry_of")
        if parent is not None:
            children.setdefault(str(parent), []).append(row)

    planned = {
        (str(job["arm"]), int(job["seed"])): job
        for job in materialization["production_jobs"]
    }
    receipt = {
        (str(job["arm"]), int(job["seed"])): job
        for job in enqueue["jobs"]
    }
    selected: dict[tuple[str, int], Mapping[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    production_lineage_job_ids: set[str] = set()

    for slot in sorted(planned):
        receipt_job = receipt[slot]
        root_id = str(receipt_job.get("job_id"))
        root = by_id.get(root_id)
        if root is None or root.get("retry_of") is not None:
            raise PooledEnsembleComparisonError(
                f"production slot {slot} lacks its root queue job"
            )
        if _slot_from_config(
            _mapping(root.get("canonical_config"), "root queue config"),
            require_production=True,
        ) != slot:
            raise PooledEnsembleComparisonError(
                f"production root queue slot changed for {slot}"
            )
        if canonical_sha256(root["canonical_config"]) != planned[slot]["config_sha256"]:
            raise PooledEnsembleComparisonError(
                f"production root queue config changed for {slot}"
            )
        if (
            _integer(root.get("requested_gpu"), "root queue requested GPU")
            != _integer(receipt_job.get("requested_gpu"), "enqueue requested GPU")
        ):
            raise PooledEnsembleComparisonError(
                f"production root queue GPU assignment changed for {slot}"
            )

        lineage: list[Mapping[str, Any]] = [root]
        current = root
        while children.get(str(current["job_id"])):
            descendants = children[str(current["job_id"])]
            if len(descendants) != 1:
                raise PooledEnsembleComparisonError(
                    f"production slot {slot} has a branching retry lineage"
                )
            current = descendants[0]
            lineage.append(current)
        if len(lineage) > 2:
            raise PooledEnsembleComparisonError(
                f"production slot {slot} exceeded the frozen two-attempt budget"
            )
        for row in lineage:
            row_config = _mapping(
                row.get("canonical_config"), "production queue config"
            )
            if (
                _slot_from_config(
                    row_config,
                    require_production=True,
                )
                != slot
                or _normalized_root_config_sha256(row_config)
                != planned[slot]["config_sha256"]
                or _integer(
                    row.get("requested_gpu"), "production requested GPU"
                )
                not in SAFE_GPU_IDS
            ):
                raise PooledEnsembleComparisonError(
                    f"production queue lineage changed semantics or used an "
                    f"unsafe GPU for {slot}"
                )
            production_lineage_job_ids.add(str(row["job_id"]))
        attempts = [_integer(row.get("attempt_count"), "queue attempt") for row in lineage]
        if attempts != list(range(1, len(lineage) + 1)):
            raise PooledEnsembleComparisonError(
                f"production slot {slot} retry attempts are not contiguous"
            )
        if any(_integer(row.get("maximum_attempts"), "attempt budget") != 2 for row in lineage):
            raise PooledEnsembleComparisonError(
                f"production slot {slot} changed its attempt budget"
            )
        completed = [row for row in lineage if row.get("status") == "completed"]
        if len(completed) != 1 or completed[0] is not lineage[-1]:
            raise PooledEnsembleComparisonError(
                f"production slot {slot} does not have exactly one terminal completion"
            )
        if any(
            row.get("status") not in {"failed", "stale", "pruned"}
            for row in lineage[:-1]
        ):
            raise PooledEnsembleComparisonError(
                f"production slot {slot} has an invalid pre-completion attempt"
            )
        completed_job = completed[0]
        run_id = completed_job.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise PooledEnsembleComparisonError(
                f"production slot {slot} completion has no run ID"
            )
        run = run_lookup(run_id)
        if (
            run is None
            or run.get("campaign_id") != CAMPAIGN_ID
            or run.get("status") != "completed"
            or _integer(run.get("attempt"), "registered run attempt")
            != attempts[-1]
        ):
            raise PooledEnsembleComparisonError(
                f"production slot {slot} is not a completed registered run"
            )
        selected[slot] = completed_job
        previous_run_id: str | None = None
        for row in lineage:
            attempt_run_id = row.get("run_id")
            attempt_run = (
                run_lookup(str(attempt_run_id))
                if isinstance(attempt_run_id, str) and attempt_run_id
                else None
            )
            if attempt_run is not None:
                if attempt_run.get("retry_of") != previous_run_id:
                    raise PooledEnsembleComparisonError(
                        f"production slot {slot} run retry lineage changed"
                    )
                previous_run_id = str(attempt_run_id)
            record = {
                "stage": "production",
                "arm": slot[0],
                "seed": slot[1],
                "job_id": str(row["job_id"]),
                "attempt": int(row["attempt_count"]),
                "status": str(row["status"]),
                "run_id": attempt_run_id,
                "run_status": None if attempt_run is None else attempt_run.get("status"),
                "failure_category": row.get("failure_category"),
                "last_error_present": bool(row.get("last_error")),
                "selected_completed_attempt": row is completed_job,
            }
            inventory.append(record)
            if row is not completed_job:
                failures.append(record)

    unexpected_production_jobs: list[str] = []
    for row in queue_rows:
        config = _mapping(row.get("canonical_config"), "campaign queue config")
        try:
            slot = _slot_from_config(config, require_production=True)
        except PooledEnsembleComparisonError:
            if _production_like(config):
                raise PooledEnsembleComparisonError(
                    "campaign contains a production-like job with invalid semantics"
                )
            continue
        if slot in planned and str(row["job_id"]) not in production_lineage_job_ids:
            unexpected_production_jobs.append(str(row["job_id"]))
    if unexpected_production_jobs:
        raise PooledEnsembleComparisonError(
            "campaign contains unexpected or orphaned production jobs"
        )
    if set(selected) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise PooledEnsembleComparisonError(
            "registry audit did not resolve all fourteen production slots"
        )
    return CampaignAudit(
        selected_jobs=selected,
        attempt_inventory=tuple(inventory),
        registered_failure_inventory=tuple(failures),
    )


def _expected_mask_entries(
    materialization: Mapping[str, Any], alias: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    sources = _mapping(
        materialization.get("evaluation_mask_sources"),
        "materialization evaluation mask sources",
    )
    source = _mapping(sources.get(alias), f"mask source {alias}")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) != 9:
        raise PooledEnsembleComparisonError(
            f"materialization has no nine-entry mask source for {alias}"
        )
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        item = _mapping(entry, f"mask source entry {alias}")
        key = (
            str(item.get("mode")),
            _integer(item.get("replicate"), f"mask replicate {alias}"),
        )
        if key in indexed:
            raise PooledEnsembleComparisonError(
                f"materialization duplicates mask entry {alias} {key}"
            )
        indexed[key] = item
    expected = {(mode, replicate) for mode in MASK_MODES for replicate in REPLICATES}
    if set(indexed) != expected:
        raise PooledEnsembleComparisonError(
            f"materialization mask coverage changed for {alias}"
        )
    return indexed


def _flatten_vector_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    for field, width in VECTOR_WIDTHS.items():
        value = flattened.pop(field, None)
        if not isinstance(value, (list, tuple)) or len(value) != width:
            raise PooledEnsembleComparisonError(
                f"metric row has malformed {field}; expected width {width}"
            )
        for index, item in enumerate(value):
            if field.endswith("_support"):
                support = _integer(item, f"{field}[{index}]")
                if support < 0:
                    raise PooledEnsembleComparisonError(
                        f"{field}[{index}] cannot be negative"
                    )
                item = support
            elif item is not None:
                recall = _finite(item, f"{field}[{index}]")
                if not 0.0 <= recall <= 1.0:
                    raise PooledEnsembleComparisonError(
                        f"{field}[{index}] must lie in [0, 1]"
                    )
                item = recall
            flattened[f"{field}_{index}"] = item
    return flattened


def _validate_replicate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: str | None,
    seed: int | None,
    materialization: Mapping[str, Any],
    run_id: str,
    prior: bool = False,
) -> list[dict[str, Any]]:
    expected = _expected_mask_entries(materialization, alias)
    observed: set[tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = _flatten_vector_metrics(raw)
        _assert_alias_safe_payload(row, label=f"{run_id} metric row")
        row_alias = row.get("biological_unit_alias")
        if row_alias is not None and row_alias != alias:
            raise PooledEnsembleComparisonError(
                f"{run_id} metric row exposes an unexpected core alias"
            )
        mode = str(row.get("mask_mode"))
        replicate = _integer(row.get("mask_replicate"), "mask replicate")
        key = (mode, replicate)
        if key not in expected or key in observed:
            raise PooledEnsembleComparisonError(
                f"{run_id} metric rows have duplicate or unexpected mask coverage"
            )
        observed.add(key)
        frozen = expected[key]
        if (
            row.get("mask_entry_id") != frozen.get("entry_id")
            or _integer(row.get("mask_seed"), "mask seed")
            != _integer(frozen.get("seed"), "frozen mask seed")
            or row.get("mask_checksum") != frozen.get("mask_checksum")
            or row.get("split") != "fit"
        ):
            raise PooledEnsembleComparisonError(
                f"{run_id} metric row mask identity changed for {alias} {key}"
            )
        _validate_required_metric_fields(
            row,
            label=run_id,
            require_equal_core_reference=not prior,
        )
        normalized = {
            **row,
            "core_alias": alias,
            "mask_mode": mode,
            "mask_replicate": replicate,
        }
        normalized.pop("biological_unit_alias", None)
        if prior:
            normalized["prior_run_id"] = run_id
        else:
            if arm is None or seed is None:
                raise PooledEnsembleComparisonError(
                    "production metric normalization lacks arm or seed"
                )
            normalized.update({"arm": arm, "seed": seed, "run_id": run_id})
        result.append(normalized)
    if observed != set(expected):
        raise PooledEnsembleComparisonError(
            f"{run_id} does not contain exactly nine fixed-mask rows"
        )
    return sorted(result, key=lambda row: (str(row["mask_mode"]), int(row["mask_replicate"])))


def validate_ensemble_replicate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    materialization: Mapping[str, Any],
) -> None:
    """Bind every ensemble metric row to one frozen per-core mask."""

    observed: set[tuple[str, str, str, int]] = set()
    expected_masks = {
        alias: _expected_mask_entries(materialization, alias)
        for alias in ALIASES
    }
    for index, row in enumerate(rows):
        arm = str(row.get("arm"))
        alias = str(row.get("core_alias"))
        mode = str(row.get("mask_mode"))
        replicate = _integer(
            row.get("mask_replicate"), "ensemble mask replicate"
        )
        identity = (arm, alias, mode, replicate)
        if (
            arm not in ARMS
            or alias not in ALIASES
            or (mode, replicate) not in expected_masks.get(alias, {})
            or identity in observed
        ):
            raise PooledEnsembleComparisonError(
                "ensemble metric rows contain unexpected or duplicate coverage"
            )
        observed.add(identity)
        frozen = expected_masks[alias][(mode, replicate)]
        if (
            row.get("mask_entry_id") != frozen.get("entry_id")
            or _integer(row.get("mask_seed"), "ensemble mask seed")
            != _integer(frozen.get("seed"), "frozen ensemble mask seed")
            or row.get("mask_checksum") != frozen.get("mask_checksum")
        ):
            raise PooledEnsembleComparisonError(
                f"ensemble metric row {index} changed the frozen mask identity"
            )
        _assert_alias_safe_payload(row, label=f"ensemble metric row {index}")
        _validate_required_metric_fields(
            row,
            label=f"ensemble metric row {index}",
            require_equal_core_reference=True,
        )
    expected = {
        (arm, alias, mode, replicate)
        for arm in ARMS
        for alias in ALIASES
        for mode in MASK_MODES
        for replicate in REPLICATES
    }
    if observed != expected:
        raise PooledEnsembleComparisonError(
            "ensemble metric rows lack exact 2x10x3x3 frozen coverage"
        )


def _checkpoint_payload(
    path: Path,
    *,
    checkpoint_loader: Callable[..., Any],
) -> Mapping[str, Any]:
    try:
        value = checkpoint_loader(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = checkpoint_loader(path, map_location="cpu")
    except Exception as exc:
        raise PooledEnsembleComparisonError(
            f"checkpoint is unreadable: {path}"
        ) from exc
    return _mapping(value, f"checkpoint {path.name}")


def _validate_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    materialization: Mapping[str, Any],
    expected_config_sha256: str,
) -> tuple[str, Mapping[str, torch.Tensor]]:
    state = _mapping(payload.get("model_state_dict"), "checkpoint model state")
    declared_state = _sha256(
        payload.get("state_dict_sha256"), "checkpoint state dictionary checksum"
    )
    try:
        observed_state = _state_dict_sha256(state)  # type: ignore[arg-type]
    except Exception as exc:
        raise PooledEnsembleComparisonError(
            "checkpoint model state cannot be checksum-verified"
        ) from exc
    graph_sources = _mapping(
        materialization.get("graph_sources"), "materialized graph sources"
    )
    expected_graphs = {
        alias: _mapping(graph_sources.get(alias), f"graph source {alias}").get(
            "graph_sha256"
        )
        for alias in ALIASES
    }
    mask_sources = _mapping(
        materialization.get("evaluation_mask_sources"),
        "materialized mask sources",
    )
    expected_masks = {
        alias: _mapping(mask_sources.get(alias), f"mask source {alias}").get(
            "bundle_checksum"
        )
        for alias in ALIASES
    }
    cohort = _mapping(materialization.get("cohort"), "materialized cohort")
    materialized_cores = _mapping(
        cohort.get("cores"), "materialized cohort cores"
    )
    expected_preprocessing = {
        alias: _mapping(
            materialized_cores.get(alias), f"materialized core {alias}"
        ).get("pooled_preprocessing_sha256")
        for alias in ALIASES
    }
    if (
        observed_state != declared_state
        or payload.get("schema_version") != 1
        or payload.get("checkpoint_role") != "last"
        or payload.get("checkpoint_policy")
        != "final_global_epoch_no_validation_selection"
        or payload.get("public_variant") != arm
        or payload.get("model_name") != ARM_TO_MODEL[arm]
        or payload.get("task_family") != EXPECTED_TASK_FAMILY
        or payload.get("model_seed") != seed
        or payload.get("aliases") != list(ALIASES)
        or payload.get("epoch") != EXPECTED_CHECKPOINT_EPOCH
        or payload.get("completed_global_epochs") != EXPECTED_GLOBAL_EPOCHS
        or payload.get("optimizer_steps_completed") != EXPECTED_OPTIMIZER_STEPS
        or payload.get("fixed_epoch_budget") != EXPECTED_GLOBAL_EPOCHS
        or payload.get("pooled_cohort_fingerprint_sha256")
        != cohort.get("dataset_fingerprint")
        or payload.get("per_core_preprocessing_sha256")
        != expected_preprocessing
        or payload.get("per_core_graph_sha256") != expected_graphs
        or payload.get("per_core_evaluation_mask_bundle_sha256") != expected_masks
        or payload.get("materialization_checksum")
        != materialization.get("checksum")
        or payload.get("config_sha256") != expected_config_sha256
        or payload.get("count_representation_schema")
        != EXPECTED_REPRESENTATION_SCHEMA
        or payload.get("objective") != EXPECTED_OBJECTIVE
        or payload.get("effective_amp") is not True
        or payload.get("selection_policy")
        != "last_epoch_without_validation_selection"
        or payload.get("monitored_metric") is not None
    ):
        raise PooledEnsembleComparisonError(
            f"checkpoint metadata changed for {(arm, seed)}"
        )
    return declared_state, state  # type: ignore[return-value]


def _catalog_checkpoint(
    registry: Registry,
    *,
    run_id: str,
    root: Path,
    checkpoint_path: Path,
) -> Mapping[str, Any]:
    shown = registry.show_run(run_id)
    if shown is None:
        raise PooledEnsembleComparisonError(f"registered run disappeared: {run_id}")
    records = shown.get("checkpoints")
    if not isinstance(records, list) or len(records) != 1:
        raise PooledEnsembleComparisonError(
            f"{run_id} requires exactly one checkpoint catalog record"
        )
    record = _mapping(records[0], "checkpoint catalog record")
    registered_path = Path(str(record.get("path")))
    if not registered_path.is_absolute():
        registered_path = _PROJECT_ROOT / registered_path
    expected_sha = _sha256(record.get("sha256"), "catalog checkpoint checksum")
    if (
        registered_path.resolve(strict=False) != checkpoint_path.resolve(strict=False)
        or record.get("role") != "last"
        or record.get("best_epoch") != EXPECTED_CHECKPOINT_EPOCH
        or record.get("monitored_metric")
        not in {None, "fit/whole_node/hybrid_loss"}
        or record.get("monitored_mode") not in {None, "min"}
        or record.get("verification_status") != "verified"
        or record.get("artifact_status") not in {"present", "verified"}
        or expected_sha != _sha256_file(checkpoint_path)
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} checkpoint catalog identity is invalid"
        )
    physical = sorted((root / "checkpoints").glob("*.ckpt"))
    if physical != [checkpoint_path]:
        raise PooledEnsembleComparisonError(
            f"{run_id} must retain exactly one final checkpoint"
        )
    return record


def _validate_graph_and_mask_diagnostics(
    *,
    root: Path,
    materialization: Mapping[str, Any],
    expected_config_sha256: str,
    expected_runtime_sha256: str,
    attempt: int,
) -> tuple[str, str]:
    graph = _strict_json(
        root / "diagnostics/graph_statistics.json",
        label=f"{root.name} graph diagnostics",
    )
    masks = _strict_json(
        root / "diagnostics/mask_statistics.json",
        label=f"{root.name} mask diagnostics",
    )
    preprocessing = _strict_json(
        root / "diagnostics/pooled_preprocessing.json",
        label=f"{root.name} preprocessing diagnostics",
    )
    representation = _strict_json(
        root / "diagnostics/raw_count_representation.json",
        label=f"{root.name} count representation diagnostics",
    )
    expected_bindings = {
        "materialization_checksum": materialization.get("checksum"),
        "config_sha256": expected_config_sha256,
        "runtime_config_sha256": expected_runtime_sha256,
        "run_attempt": attempt,
    }
    for document, label in (
        (graph, "graph"),
        (masks, "mask"),
        (preprocessing, "preprocessing"),
        (representation, "representation"),
    ):
        if any(document.get(field) != value for field, value in expected_bindings.items()):
            raise PooledEnsembleComparisonError(
                f"{root.name} {label} diagnostics lost config lineage"
            )
    if (
        graph.get("aliases") != list(ALIASES)
        or graph.get("cross_core_edges") is not False
        or graph.get("graph_concatenation_on_gpu") is not False
        or masks.get("aliases") != list(ALIASES)
        or masks.get("model_seed_excluded_from_evaluation_mask_derivation")
        is not True
        or preprocessing.get("aliases") != list(ALIASES)
        or preprocessing.get("expression_moment_weighting") != "equal_core"
        or representation.get("schema") != EXPECTED_REPRESENTATION_SCHEMA
        or representation.get("validation") != "finite_nonnegative_integer"
        or representation.get("total_nodes") != 117386
        or representation.get("n_genes") != 1000
        or representation.get("mask_token_id") != 8
        or representation.get("mask_token_is_output") is not False
        or representation.get("continuous_masked_value") != 0.0
        or representation.get("thresholds_fitted") is not False
    ):
        raise PooledEnsembleComparisonError(
            f"{root.name} pooled graph, mask, preprocessing, or representation "
            "contract changed"
        )
    _sha256(
        representation.get("shared_expression_mean_sha256"),
        f"{root.name} shared expression mean",
    )
    _sha256(
        representation.get("shared_expression_scale_sha256"),
        f"{root.name} shared expression scale",
    )
    graph_sources = _mapping(
        materialization.get("graph_sources"), "materialized graph sources"
    )
    per_core = graph.get("per_core")
    if not isinstance(per_core, list) or len(per_core) != len(ALIASES):
        raise PooledEnsembleComparisonError(
            f"{root.name} graph diagnostics lack ten cores"
        )
    graph_bundle_items: list[dict[str, Any]] = []
    for record in per_core:
        item = _mapping(record, "per-core graph diagnostic")
        alias = str(item.get("alias"))
        frozen = _mapping(graph_sources.get(alias), f"graph source {alias}")
        if (
            alias not in ALIASES
            or item.get("graph_sha256") != frozen.get("graph_sha256")
            or item.get("n_directed_edges") != frozen.get("n_directed_edges")
        ):
            raise PooledEnsembleComparisonError(
                f"{root.name} graph identity changed for {alias}"
            )
        graph_bundle_items.append(
            {
                "alias": alias,
                "graph_sha256": item["graph_sha256"],
                "n_directed_edges": item["n_directed_edges"],
            }
        )
    graph_bundle_sha = canonical_sha256(graph_bundle_items)
    if graph.get("graph_bundle_sha256") != graph_bundle_sha:
        raise PooledEnsembleComparisonError(
            f"{root.name} graph bundle checksum does not reconcile"
        )

    manifests = _mapping(
        masks.get("per_core_manifests"), "per-core mask manifests"
    )
    mask_bundle_items: list[dict[str, str]] = []
    for alias in ALIASES:
        manifest = _mapping(manifests.get(alias), f"mask manifest {alias}")
        source = _mapping(
            _mapping(
                materialization.get("evaluation_mask_sources"),
                "evaluation mask sources",
            ).get(alias),
            f"mask source {alias}",
        )
        if (
            manifest.get("bundle_checksum") != source.get("bundle_checksum")
            or manifest.get("base_seed") != source.get("base_seed")
        ):
            raise PooledEnsembleComparisonError(
                f"{root.name} mask bundle identity changed for {alias}"
            )
        observed_entries = [
            {
                "entry_id": str(entry["entry_id"]),
                "mode": str(entry["spec"]["label"]),
                "replicate": int(entry["replicate"]),
                "seed": int(entry["seed"]),
                "mask_checksum": str(entry["mask_checksum"]),
            }
            for entry in manifest.get("entries", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("spec"), Mapping)
        ]
        if observed_entries != source.get("entries"):
            raise PooledEnsembleComparisonError(
                f"{root.name} fixed mask entries changed for {alias}"
            )
        mask_bundle_items.append(
            {"alias": alias, "mask_bundle_sha256": str(source["bundle_checksum"])}
        )
    mask_bundle_sha = canonical_sha256(mask_bundle_items)
    if masks.get("mask_bundle_sha256") != mask_bundle_sha:
        raise PooledEnsembleComparisonError(
            f"{root.name} mask bundle checksum does not reconcile"
        )
    return graph_bundle_sha, mask_bundle_sha


def audit_production_run(
    *,
    registry: Registry,
    completed_job: Mapping[str, Any],
    slot: tuple[str, int],
    planned_job: Mapping[str, Any],
    materialization: Mapping[str, Any],
    failed_attempt_count: int,
    paths: ProjectPaths,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
    checkpoint_loader: Callable[..., Any] = torch.load,
) -> ProductionRunEvidence:
    arm, seed = slot
    run_id = str(completed_job["run_id"])
    run = registry.get_run(run_id)
    if run is None or not run.get("artifact_path"):
        raise PooledEnsembleComparisonError(f"{run_id} has no registered artifact")
    root = Path(str(run["artifact_path"]))
    if not root.is_absolute():
        root = paths.project_root / root
    root = root.resolve(strict=False)
    verification = bundle_verifier(root)
    if verification.get("valid") is not True or verification.get("status") != "success":
        raise PooledEnsembleComparisonError(f"{run_id} bundle verification failed")
    registry_issues = registry.verify_artifacts(run_id=run_id)
    if registry_issues:
        raise PooledEnsembleComparisonError(
            f"{run_id} registry artifact verification failed: {registry_issues[:3]}"
        )

    config = _yaml_mapping(
        root / "config.resolved.yaml", label=f"{run_id} resolved config"
    )
    if _slot_from_config(config, require_production=True) != slot:
        raise PooledEnsembleComparisonError(
            f"{run_id} resolved config does not match slot {slot}"
        )
    config_sha = _normalized_root_config_sha256(config)
    runtime_sha = canonical_sha256(config)
    attempt = _integer(config.get("attempt"), f"{run_id} attempt")
    if (
        config_sha != planned_job.get("config_sha256")
        or attempt != _integer(completed_job.get("attempt_count"), "queue attempt")
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} resolved config is not bound to materialization"
    )
    summary = _strict_json(root / "summary.json", label=f"{run_id} summary")
    summary_checkpoint = _mapping(
        summary.get("checkpoint"), f"{run_id} summary checkpoint"
    )
    if (
        summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("campaign_id") != CAMPAIGN_ID
        or summary.get("aliases") != list(ALIASES)
        or summary.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or summary.get("evaluation_protocol") != EXPECTED_PROTOCOL
        or summary.get("task_family") != EXPECTED_TASK_FAMILY
        or summary.get("canonical_prediction_split") != "fit"
        or summary.get("model_name") != ARM_TO_MODEL[arm]
        or summary.get("public_variant") != arm
        or summary.get("model_seed") != seed
        or summary.get("diagnostic_resource_pilot") is not False
        or summary.get("conclusion_eligible") is not True
        or summary.get("final_epoch") != EXPECTED_CHECKPOINT_EPOCH
        or summary.get("completed_global_epochs") != EXPECTED_GLOBAL_EPOCHS
        or summary.get("fixed_epoch_budget") != EXPECTED_GLOBAL_EPOCHS
        or summary.get("optimizer_steps_completed") != EXPECTED_OPTIMIZER_STEPS
        or summary.get("checkpoint_role") != "last"
        or summary_checkpoint.get("role") != "last"
        or summary_checkpoint.get("final_epoch") != EXPECTED_CHECKPOINT_EPOCH
        or summary_checkpoint.get("policy")
        != "final_global_epoch_no_validation_selection"
        or summary_checkpoint.get("monitored_metric") is not None
        or summary.get("primary_metric_name")
        != "fit/whole_node/hybrid_loss"
        or summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or summary.get("exact_parameter_match") is not True
        or summary.get("paired_initialization_match") is not True
        or summary.get("generalization_estimate") is not False
        or summary.get("materialization_checksum") != materialization.get("checksum")
        or summary.get("config_sha256") != config_sha
        or summary.get("runtime_config_sha256") != runtime_sha
        or summary.get("run_attempt") != attempt
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} summary violates production identity"
        )
    parameter_audit = _strict_json(
        root / "diagnostics/parameter_structure_audit.json",
        label=f"{run_id} parameter audit",
    )
    encoder_initial_sha = _sha256(
        parameter_audit.get("encoder_initial_state_sha256"),
        f"{run_id} encoder initial state",
    )
    decoder_initial_sha = _sha256(
        parameter_audit.get("decoder_initial_state_sha256"),
        f"{run_id} decoder initial state",
    )
    if (
        parameter_audit.get("schema")
        != "hybrid_count_parameter_structure_audit_v1"
        or parameter_audit.get("trainable_parameter_count_graph")
        != EXPECTED_PARAMETER_COUNT
        or parameter_audit.get("trainable_parameter_count_self")
        != EXPECTED_PARAMETER_COUNT
        or parameter_audit.get("exact_trainable_parameter_match") is not True
        or parameter_audit.get("encoder_initial_state_bit_identical") is not True
        or parameter_audit.get("decoder_initial_state_bit_identical") is not True
        or parameter_audit.get("materialization_checksum")
        != materialization.get("checksum")
        or parameter_audit.get("config_sha256") != config_sha
        or parameter_audit.get("runtime_config_sha256") != runtime_sha
        or parameter_audit.get("run_attempt") != attempt
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} parameter or initialization audit is invalid"
        )
    graph_bundle, mask_bundle = _validate_graph_and_mask_diagnostics(
        root=root,
        materialization=materialization,
        expected_config_sha256=config_sha,
        expected_runtime_sha256=runtime_sha,
        attempt=attempt,
    )
    if (
        summary.get("graph_bundle_sha256") != graph_bundle
        or summary.get("evaluation_mask_bundle_sha256") != mask_bundle
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} summary graph or mask identity changed"
        )
    convergence = _strict_json(
        root / "diagnostics/training_convergence.json",
        label=f"{run_id} convergence",
    )
    if (
        convergence.get("schema") != "pooled_hybrid_count_convergence_v1"
        or convergence.get("objective") != EXPECTED_OBJECTIVE
        or convergence.get("all_global_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
        or convergence.get("every_core_once_each_epoch") is not True
        or convergence.get("final_epoch") != EXPECTED_CHECKPOINT_EPOCH
        or convergence.get("completed_global_epochs") != EXPECTED_GLOBAL_EPOCHS
        or convergence.get("optimizer_steps_completed") != EXPECTED_OPTIMIZER_STEPS
        or convergence.get("materialization_checksum")
        != materialization.get("checksum")
        or convergence.get("config_sha256") != config_sha
        or convergence.get("runtime_config_sha256") != runtime_sha
        or convergence.get("run_attempt") != attempt
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} did not complete finite fixed-budget training"
        )
    resource_usage = _strict_json(
        root / "diagnostics/resource_usage.json",
        label=f"{run_id} resource usage",
    )
    if (
        resource_usage.get("schema")
        != "pooled_hybrid_count_resource_diagnostic_v1"
        or resource_usage.get("diagnostic_resource_pilot") is not False
        or resource_usage.get("public_variant") != arm
        or resource_usage.get("aliases") != list(ALIASES)
        or resource_usage.get("cuda_available") is not True
        or not isinstance(resource_usage.get("cuda_device_name"), str)
        or not str(resource_usage.get("cuda_device_name"))
        or resource_usage.get("effective_training_amp") is not True
        or resource_usage.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or resource_usage.get("finite_losses_and_gradients") is not True
        or resource_usage.get("exact_parameter_match") is not True
        or resource_usage.get("paired_initialization_match") is not True
        or resource_usage.get("completed_global_epochs")
        != EXPECTED_GLOBAL_EPOCHS
        or resource_usage.get("optimizer_steps_completed")
        != EXPECTED_OPTIMIZER_STEPS
        or resource_usage.get("materialization_checksum")
        != materialization.get("checksum")
        or resource_usage.get("config_sha256") != config_sha
    ):
        raise PooledEnsembleComparisonError(
            f"{run_id} resource and environment audit is invalid"
        )
    checkpoint_path = root / "checkpoints/last.ckpt"
    catalog = _catalog_checkpoint(
        registry,
        run_id=run_id,
        root=root,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_sha = _sha256_file(checkpoint_path)
    if checkpoint_sha != catalog.get("sha256"):
        raise PooledEnsembleComparisonError(
            f"{run_id} checkpoint file and catalog checksums differ"
        )
    payload = _checkpoint_payload(checkpoint_path, checkpoint_loader=checkpoint_loader)
    if dict(
        _mapping(
            payload.get("parameter_structure_audit"),
            "checkpoint parameter structure audit",
        )
    ) != parameter_audit:
        raise PooledEnsembleComparisonError(
            f"{run_id} checkpoint and parameter audit differ"
        )
    state_sha, _ = _validate_checkpoint_metadata(
        payload,
        arm=arm,
        seed=seed,
        materialization=materialization,
        expected_config_sha256=config_sha,
    )
    if summary_checkpoint.get("state_dict_sha256") != state_sha:
        raise PooledEnsembleComparisonError(
            f"{run_id} summary and checkpoint state checksums differ"
        )
    del payload
    gc.collect()
    metric_rows = _load_table(
        _logical_table_path(root, "metrics/evaluation_replicates")
    )
    by_alias: dict[str, list[Mapping[str, Any]]] = {alias: [] for alias in ALIASES}
    for row in metric_rows:
        alias = str(row.get("biological_unit_alias"))
        if alias not in by_alias:
            raise PooledEnsembleComparisonError(
                f"{run_id} metric table contains a non-opaque alias"
            )
        by_alias[alias].append(row)
    normalized_rows: list[dict[str, Any]] = []
    for alias in ALIASES:
        normalized_rows.extend(
            _validate_replicate_rows(
                by_alias[alias],
                alias=alias,
                arm=arm,
                seed=seed,
                materialization=materialization,
                run_id=run_id,
            )
        )
    if len(normalized_rows) != 90:
        raise PooledEnsembleComparisonError(
            f"{run_id} must contain 90 fixed-mask metric rows"
        )
    duration = _finite(summary.get("duration_seconds"), f"{run_id} duration")
    peak_vram = _finite(summary.get("peak_vram_gib"), f"{run_id} peak VRAM")
    host_bytes = _integer(
        summary.get("peak_host_memory_bytes"), f"{run_id} peak host memory"
    )
    if duration <= 0 or peak_vram < 0 or host_bytes <= 0:
        raise PooledEnsembleComparisonError(
            f"{run_id} has invalid resource diagnostics"
        )
    return ProductionRunEvidence(
        arm=arm,
        seed=seed,
        run_id=run_id,
        attempt=attempt,
        root=root,
        config=config,
        summary=summary,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_sha,
        state_dict_sha256=state_sha,
        encoder_initial_state_sha256=encoder_initial_sha,
        decoder_initial_state_sha256=decoder_initial_sha,
        member_rows=tuple(normalized_rows),
        duration_seconds=duration,
        peak_vram_gib=peak_vram,
        peak_host_memory_bytes=host_bytes,
        convergence=convergence,
        resource_usage=resource_usage,
        failed_attempt_count=failed_attempt_count,
    )


def audit_prior_independent_seed0(
    *,
    registry: Registry,
    materialization: Mapping[str, Any],
    paths: ProjectPaths,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
) -> list[dict[str, Any]]:
    """Read the exact prior GAT rows named by the pooled mask sources."""

    prior_record = _mapping(
        materialization.get("prior_materialization"),
        "prior materialization reference",
    )
    prior_path = paths.project_root / str(prior_record.get("reference"))
    if (
        not prior_path.is_file()
        or _sha256_file(prior_path)
        != _sha256(prior_record.get("file_sha256"), "prior materialization file")
    ):
        raise PooledEnsembleComparisonError(
            "prior independent campaign materialization changed"
        )
    prior_materialization = _verified_signed_json(
        prior_path, label="prior independent materialization"
    )
    if (
        prior_materialization.get("campaign_id")
        != prior_record.get("campaign_id")
        or prior_materialization.get("checksum")
        != prior_record.get("canonical_checksum")
    ):
        raise PooledEnsembleComparisonError(
            "prior independent materialization identity does not reconcile"
        )

    sources = _mapping(
        materialization.get("evaluation_mask_sources"),
        "materialized prior mask sources",
    )
    output: list[dict[str, Any]] = []
    observed_run_ids: set[str] = set()
    for alias in ALIASES:
        source = _mapping(sources.get(alias), f"prior source {alias}")
        run_id = str(source.get("source_run_id", ""))
        if not run_id or run_id in observed_run_ids:
            raise PooledEnsembleComparisonError(
                "prior independent source run IDs must be one-to-one with aliases"
            )
        observed_run_ids.add(run_id)
        run = registry.get_run(run_id)
        if (
            run is None
            or run.get("status") != "completed"
            or run.get("campaign_id") != prior_record.get("campaign_id")
            or not run.get("artifact_path")
        ):
            raise PooledEnsembleComparisonError(
                f"prior independent run is unavailable for {alias}"
            )
        root = Path(str(run["artifact_path"]))
        if not root.is_absolute():
            root = paths.project_root / root
        root = root.resolve(strict=False)
        verified = bundle_verifier(root)
        if verified.get("valid") is not True or verified.get("status") != "success":
            raise PooledEnsembleComparisonError(
                f"prior independent bundle failed verification for {alias}"
            )
        registry_issues = registry.verify_artifacts(run_id=run_id)
        if registry_issues:
            raise PooledEnsembleComparisonError(
                f"prior independent registry artifacts failed verification "
                f"for {alias}: {registry_issues[:3]}"
            )
        config = _yaml_mapping(
            root / "config.resolved.yaml", label=f"prior config {alias}"
        )
        experiment = _mapping(config.get("experiment"), "prior experiment")
        if (
            _mapping(config.get("model"), "prior model").get("name")
            != "hybrid-count-gat"
            or _integer(config.get("seed"), "prior seed") != 0
            or experiment.get("biological_unit_alias") != alias
            or experiment.get("resource_pilot") is not False
            or experiment.get("conclusion_eligible") is not True
        ):
            raise PooledEnsembleComparisonError(
                f"prior source run has wrong scientific semantics for {alias}"
            )
        rows = _load_table(
            _logical_table_path(root, "metrics/evaluation_replicates")
        )
        output.extend(
            _validate_replicate_rows(
                rows,
                alias=alias,
                arm=None,
                seed=None,
                materialization=materialization,
                run_id=run_id,
                prior=True,
            )
        )
    if len(output) != len(ALIASES) * len(MASK_MODES) * len(REPLICATES):
        raise PooledEnsembleComparisonError(
            "prior independent comparison lacks exact ten-core mask coverage"
        )
    return output


def collapse_rich_replicates(
    rows: Sequence[Mapping[str, Any]],
    *,
    ensemble: bool,
) -> list[dict[str, Any]]:
    """Average every finite scalar across the three technical mask repeats."""

    grouped: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            (str(row.get("arm")), str(row.get("core_alias")), str(row.get("mask_mode")))
            if ensemble
            else (
                str(row.get("arm")),
                int(row.get("seed", -1)),
                str(row.get("core_alias")),
                str(row.get("mask_mode")),
            )
        )
        replicate = int(row.get("mask_replicate", -1))
        if replicate not in REPLICATES or replicate in grouped.setdefault(key, {}):
            raise PooledEnsembleComparisonError(
                "replicate rows contain an invalid or duplicate replicate"
            )
        grouped[key][replicate] = row
    expected_keys = (
        {
            (arm, alias, mode)
            for arm in ARMS
            for alias in ALIASES
            for mode in MASK_MODES
        }
        if ensemble
        else {
            (arm, seed, alias, mode)
            for arm in ARMS
            for seed in SEEDS
            for alias in ALIASES
            for mode in MASK_MODES
        }
    )
    if set(grouped) != expected_keys:
        raise PooledEnsembleComparisonError(
            "replicate rows lack complete frozen arm/seed/core/mode coverage"
        )
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        by_replicate = grouped[key]
        if set(by_replicate) != set(REPLICATES):
            raise PooledEnsembleComparisonError(
                "core/mode rows lack all three mask replicates"
            )
        identity = (
            {
                "arm": key[0],
                "core_alias": key[1],
                "mask_mode": key[2],
            }
            if ensemble
            else {
                "arm": key[0],
                "seed": key[1],
                "core_alias": key[2],
                "mask_mode": key[3],
            }
        )
        item: dict[str, Any] = {**identity, "mask_replicates": 3}
        candidate_fields = sorted(
            set().union(*(set(row) for row in by_replicate.values()))
            - ROW_IDENTITY_FIELDS
        )
        for field in candidate_fields:
            values = [by_replicate[index].get(field) for index in REPLICATES]
            finite: list[float] = []
            for value in values:
                if value in (None, ""):
                    continue
                if isinstance(value, bool):
                    raise PooledEnsembleComparisonError(
                        f"metric {field} cannot be boolean"
                    )
                try:
                    converted = float(value)
                except (TypeError, ValueError, OverflowError):
                    # Non-metric descriptive fields are intentionally omitted.
                    finite = []
                    break
                if not math.isfinite(converted):
                    raise PooledEnsembleComparisonError(
                        f"metric {field} contains a non-finite value"
                    )
                finite.append(converted)
            if finite:
                item[field] = float(np.mean(finite, dtype=np.float64))
                item[f"{field}_defined_replicates"] = len(finite)
            elif all(value in (None, "") for value in values):
                item[field] = None
                item[f"{field}_defined_replicates"] = 0
        output.append(item)
    return output


def _predict_checkpoint_masks(
    *,
    model: torch.nn.Module,
    device_view: Any,
    mask_entries: Sequence[Mapping[str, Any]],
    mask_bundle: Any,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
) -> Iterable[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield one member prediction at a time without retaining member logits."""

    model.to(device)
    model.eval()
    for entry in mask_entries:
        entry_id = str(entry["entry_id"])
        fixed_mask = torch.as_tensor(
            mask_bundle.masks[entry_id],
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad(), _autocast_context(
            enabled=amp,
            device=device,
            dtype_name=amp_dtype,
        ):
            output, target, target_mask = _forward_masked_targets(
                model,
                device_view,
                fixed_mask,
                edge_index=device_view.edge_index,
                edge_attributes=device_view.edge_attributes,
            )
        yield entry_id, output.prediction, target, target_mask


def recompute_prediction_ensembles(
    *,
    evidence: Mapping[tuple[str, int], ProductionRunEvidence],
    cohort: PooledFullCoreCohort | StreamingPooledFullCoreCohort,
    materialization: Mapping[str, Any],
    device_name: str,
    checkpoint_loader: Callable[..., Any] = torch.load,
) -> list[dict[str, Any]]:
    """Recompute both seven-member ensembles one core graph at a time."""

    if set(evidence) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise PooledEnsembleComparisonError(
            "ensemble recomputation requires all fourteen production members"
        )
    cohort_record = _mapping(materialization.get("cohort"), "materialized cohort")
    materialized_checksums = _mapping(
        cohort_record.get("cohort_checksums"),
        "materialized cohort checksums",
    )
    if (
        tuple(cohort.aliases) != ALIASES
        or cohort.fingerprint_sha256
        != cohort_record.get("dataset_fingerprint")
        or cohort.checksums.to_dict() != dict(materialized_checksums)
    ):
        raise PooledEnsembleComparisonError(
            "loaded pooled cohort differs from materialized production data"
        )
    if isinstance(cohort, StreamingPooledFullCoreCohort):
        materialized_cores = _mapping(
            cohort_record.get("cores"),
            "materialized pooled core identities",
        )
        expected_core_preprocessing = {
            alias: str(
                _mapping(
                    materialized_cores.get(alias),
                    f"materialized pooled core {alias}",
                ).get("pooled_preprocessing_sha256")
            )
            for alias in ALIASES
        }
        if dict(cohort.core_preprocessing_sha256) != (
            expected_core_preprocessing
        ):
            raise PooledEnsembleComparisonError(
                "streamed per-core preprocessing differs from materialization"
            )
    device = torch.device(device_name)
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        if index not in SAFE_GPU_IDS:
            raise PooledEnsembleComparisonError(
                f"CUDA device {index} is excluded from this campaign"
            )
        if not torch.cuda.is_available():
            raise PooledEnsembleComparisonError(
                "CUDA ensemble evaluation requested but CUDA is unavailable"
            )

    base_config = evidence[(GAT_ARM, 0)].config
    if isinstance(cohort, StreamingPooledFullCoreCohort):
        pooled_references = cohort.pooled_references
        core_iterator = cohort.iter_cores()
    else:
        # Compatibility path for injected tests and legacy callers.  The
        # production entry point always uses the streaming cohort above.
        from spatial_benchmark.pooled_references import (  # noqa: PLC0415
            fit_equal_core_hybrid_count_references,
        )

        pooled_references = fit_equal_core_hybrid_count_references(
            {core.alias: core.expression_counts for core in cohort.cores},
            expression_mean=cohort.expression_mean,
            expression_scale=cohort.expression_scale,
            expected_aliases=ALIASES,
        )
        core_iterator = iter(cohort.cores)
    rows: list[dict[str, Any]] = []
    for core in core_iterator:
        alias = core.alias
        graph_config = _mapping(base_config.get("graph"), "pooled graph config")
        per_core_graph_config = _graph_config_for_alias(graph_config, alias)
        graph = _build_graph(core, per_core_graph_config)
        _validate_graph(graph, core, per_core_graph_config)
        expression = validate_raw_counts(
            core.expression_counts, name=f"{alias} expression_counts"
        )
        views = {
            GAT_ARM: _fit_view(
                core=core,
                graph=graph,
                uses_graph=True,
                expression=expression,
            ),
            SELF_ARM: _fit_view(
                core=core,
                graph=graph,
                uses_graph=False,
                expression=expression,
            ),
        }
        training = _training_config(base_config)
        mask_bundle = _mask_bundle_for_core(
            config=base_config,
            core=core,
            training=training,
        )
        entries = tuple(mask_bundle.manifest["entries"])
        per_core_references = fit_hybrid_count_references(
            core.expression_counts,
            expression_mean=cohort.expression_mean,
            expression_scale=cohort.expression_scale,
        )

        for arm in ARMS:
            config = evidence[(arm, 0)].config
            trainer = _mapping(config.get("trainer"), "trainer config")
            amp = bool(trainer.get("amp"))
            amp_dtype = str(trainer.get("amp_dtype", "auto")).lower()
            device_view = _to_device_view(
                views[arm],
                device=device,
                dtype=torch.float32,
            )
            accumulators = {
                str(entry["entry_id"]): HybridCountEnsembleAccumulator(
                    expected_member_count=len(SEEDS),
                    accumulation_device="cpu",
                )
                for entry in entries
            }
            targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for seed in SEEDS:
                member = evidence[(arm, seed)]
                payload = _checkpoint_payload(
                    member.checkpoint_path,
                    checkpoint_loader=checkpoint_loader,
                )
                _, state = _validate_checkpoint_metadata(
                    payload,
                    arm=arm,
                    seed=seed,
                    materialization=materialization,
                    expected_config_sha256=_normalized_root_config_sha256(
                        member.config
                    ),
                )
                if (
                    payload.get("pooled_cohort_checksums")
                    != cohort.checksums.to_dict()
                    or payload.get("pooled_reference_sha256")
                    != pooled_references.audit.get("reference_sha256")
                    or _mapping(
                        payload.get("per_core_raw_count_sha256"),
                        "checkpoint per-core raw-count checksums",
                    ).get(alias)
                    != core.checksums.expression_counts_sha256
                    or _mapping(
                        payload.get("per_core_reference_sha256"),
                        "checkpoint per-core reference checksums",
                    ).get(alias)
                    != per_core_references.audit.get("reference_sha256")
                ):
                    raise PooledEnsembleComparisonError(
                        f"checkpoint data or reference identity changed for "
                        f"{(arm, seed, alias)}"
                    )
                model, _, parameter_audit = _paired_models(
                    core=core,
                    graph=SimpleNamespace(
                        edge_attribute_names=EDGE_ATTRIBUTE_NAMES
                    ),
                    model_config=_mapping(member.config.get("model"), "model config"),
                    selected_model_name=ARM_TO_MODEL[arm],
                    seed=seed,
                )
                if (
                    parameter_audit.get("trainable_parameter_count_graph")
                    != EXPECTED_PARAMETER_COUNT
                    or parameter_audit.get("trainable_parameter_count_self")
                    != EXPECTED_PARAMETER_COUNT
                    or parameter_audit.get("exact_trainable_parameter_match")
                    is not True
                    or parameter_audit.get(
                        "encoder_initial_state_bit_identical"
                    )
                    is not True
                    or parameter_audit.get(
                        "decoder_initial_state_bit_identical"
                    )
                    is not True
                ):
                    raise PooledEnsembleComparisonError(
                        f"member architecture parameter count changed for {(arm, seed)}"
                    )
                if (
                    _state_dict_sha256(model.encoder.state_dict())
                    != member.encoder_initial_state_sha256
                    or _state_dict_sha256(model.decoder.state_dict())
                    != member.decoder_initial_state_sha256
                ):
                    raise PooledEnsembleComparisonError(
                        f"member reconstructed initialization changed for "
                        f"{(arm, seed)}"
                    )
                try:
                    model.load_state_dict(state, strict=True)
                except RuntimeError as exc:
                    raise PooledEnsembleComparisonError(
                        f"checkpoint state does not load for {(arm, seed)}"
                    ) from exc
                del payload, state
                _clear_graph_layout_caches(model)
                for entry_id, prediction, target, target_mask in _predict_checkpoint_masks(
                    model=model,
                    device_view=device_view,
                    mask_entries=entries,
                    mask_bundle=mask_bundle,
                    device=device,
                    amp=amp,
                    amp_dtype=amp_dtype,
                ):
                    accumulators[entry_id].update(prediction)
                    cpu_target = target.detach().float().cpu()
                    cpu_mask = target_mask.detach().cpu()
                    previous = targets.get(entry_id)
                    if previous is None:
                        targets[entry_id] = (cpu_target, cpu_mask)
                    elif not (
                        torch.equal(previous[0], cpu_target)
                        and torch.equal(previous[1], cpu_mask)
                    ):
                        raise PooledEnsembleComparisonError(
                            f"member targets changed within {alias} {entry_id}"
                        )
                    del prediction, target, target_mask, cpu_target, cpu_mask
                model.to("cpu")
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            expected_masks = _expected_mask_entries(materialization, alias)
            for entry in entries:
                entry_id = str(entry["entry_id"])
                mode = str(entry["spec"]["label"])
                replicate = int(entry["replicate"])
                frozen_entry = expected_masks[(mode, replicate)]
                if (
                    entry_id != frozen_entry.get("entry_id")
                    or entry.get("mask_checksum") != frozen_entry.get("mask_checksum")
                ):
                    raise PooledEnsembleComparisonError(
                        f"ensemble mask identity changed for {alias} {(mode, replicate)}"
                    )
                prediction = accumulators[entry_id].finalize()
                target, target_mask = targets[entry_id]
                evaluation = evaluate_hybrid_count_output(
                    prediction,
                    target,
                    target_mask,
                    expression_mean=cohort.expression_mean,
                    expression_scale=cohort.expression_scale,
                    references=per_core_references,
                )
                metrics = dict(json_safe_metrics(evaluation.metrics))
                metrics.update(
                    _reference_metrics(
                        SimpleNamespace(target=target, target_mask=target_mask),
                        pooled_references,
                        expression_mean=cohort.expression_mean,
                        expression_scale=cohort.expression_scale,
                        prefix="reference_equal_core",
                    )
                )
                row = _flatten_vector_metrics(
                    {
                        "arm": arm,
                        "core_alias": alias,
                        "mask_mode": mode,
                        "mask_replicate": replicate,
                        "mask_entry_id": entry_id,
                        "mask_seed": int(entry["seed"]),
                        "mask_checksum": str(entry["mask_checksum"]),
                        **metrics,
                    }
                )
                _validate_required_metric_fields(
                    row,
                    label=f"ensemble {arm} {alias} {mode} replicate {replicate}",
                    require_equal_core_reference=True,
                )
                rows.append(row)
                del prediction, evaluation
            del accumulators, targets, device_view
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del views, graph, mask_bundle, per_core_references, core
        gc.collect()
    if len(rows) != EXPECTED_ENSEMBLE_ROWS:
        raise PooledEnsembleComparisonError(
            "ensemble recomputation did not produce exact two-arm mask coverage"
        )
    return rows


def collapse_prior_replicates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("core_alias")), str(row.get("mask_mode")))
        replicate = int(row.get("mask_replicate", -1))
        if (
            key[0] not in ALIASES
            or key[1] not in MASK_MODES
            or replicate not in REPLICATES
            or replicate in grouped.setdefault(key, {})
        ):
            raise PooledEnsembleComparisonError(
                "prior rows contain unexpected or duplicate mask coverage"
            )
        grouped[key][replicate] = row
    if set(grouped) != {(alias, mode) for alias in ALIASES for mode in MASK_MODES}:
        raise PooledEnsembleComparisonError(
            "prior rows lack complete ten-core mask-mode coverage"
        )
    result: list[dict[str, Any]] = []
    for (alias, mode), by_replicate in sorted(grouped.items()):
        if set(by_replicate) != set(REPLICATES):
            raise PooledEnsembleComparisonError(
                "prior core/mode lacks three mask replicates"
            )
        item: dict[str, Any] = {
            "core_alias": alias,
            "mask_mode": mode,
            "mask_replicates": 3,
        }
        for metric in GATE_METRICS:
            item[metric] = float(
                np.mean(
                    [
                        _finite(
                            by_replicate[replicate].get(metric),
                            f"prior {alias} {mode} {metric}",
                        )
                        for replicate in REPLICATES
                    ],
                    dtype=np.float64,
                )
            )
        result.append(item)
    return result


def verify_paired_member_initialization(
    evidence: Mapping[tuple[str, int], ProductionRunEvidence],
) -> None:
    """Verify the common encoder/decoder start state for every arm pair."""

    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(evidence) != expected:
        raise PooledEnsembleComparisonError(
            "paired-initialization audit requires all fourteen members"
        )
    for seed in SEEDS:
        graph = evidence[(GAT_ARM, seed)]
        control = evidence[(SELF_ARM, seed)]
        if (
            graph.encoder_initial_state_sha256
            != control.encoder_initial_state_sha256
            or graph.decoder_initial_state_sha256
            != control.decoder_initial_state_sha256
        ):
            raise PooledEnsembleComparisonError(
                f"paired encoder/decoder initialization differs for seed {seed}"
            )


def _indexed(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in output:
            raise PooledEnsembleComparisonError(f"duplicate comparison key {key!r}")
        output[key] = row
    return output


def _relative_gain(baseline: Any, candidate: Any, *, label: str) -> float:
    base = _finite(baseline, f"{label} baseline")
    value = _finite(candidate, f"{label} candidate")
    if base <= 0:
        raise PooledEnsembleComparisonError(
            f"{label} relative-gain baseline must be positive"
        )
    return (base - value) / base


def build_comparison_tables(
    *,
    member_core_rows: Sequence[Mapping[str, Any]],
    ensemble_core_rows: Sequence[Mapping[str, Any]],
    prior_core_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    member = _indexed(
        member_core_rows, ("arm", "seed", "core_alias", "mask_mode")
    )
    ensemble = _indexed(
        ensemble_core_rows, ("arm", "core_alias", "mask_mode")
    )
    prior = _indexed(prior_core_rows, ("core_alias", "mask_mode"))

    pooled_vs_prior: list[dict[str, Any]] = []
    for alias in ALIASES:
        for metric in (
            "detection_bce",
            "positive_ordinal_mae",
            "reconstructed_count_log1p_mae",
        ):
            baseline = prior[(alias, "whole_node")][metric]
            candidate = member[(GAT_ARM, 0, alias, "whole_node")][metric]
            gain = _relative_gain(
                baseline, candidate, label=f"pooled versus prior {alias} {metric}"
            )
            pooled_vs_prior.append(
                {
                    "core_alias": alias,
                    "metric": metric,
                    "prior_independent_seed0": float(baseline),
                    "pooled_shared_seed0": float(candidate),
                    "relative_improvement": gain,
                    "pooled_favored": gain > 0,
                }
            )

    graph_core: list[dict[str, Any]] = []
    for alias in ALIASES:
        gat = ensemble[(GAT_ARM, alias, "whole_node")]
        control = ensemble[(SELF_ARM, alias, "whole_node")]
        graph_core.append(
            {
                "core_alias": alias,
                "gat_hybrid_loss": gat["hybrid_loss"],
                "matched_self_hybrid_loss": control["hybrid_loss"],
                "gat_relative_hybrid_loss_improvement": _relative_gain(
                    control["hybrid_loss"],
                    gat["hybrid_loss"],
                    label=f"graph {alias}",
                ),
                "gat_positive_ordinal_mae": gat["positive_ordinal_mae"],
                "self_positive_ordinal_mae": control["positive_ordinal_mae"],
                "gat_positive_continuous_huber": gat[
                    "positive_continuous_huber"
                ],
                "self_positive_continuous_huber": control[
                    "positive_continuous_huber"
                ],
                "gat_detection_balanced_accuracy": gat[
                    "detection_balanced_accuracy"
                ],
                "self_detection_balanced_accuracy": control[
                    "detection_balanced_accuracy"
                ],
            }
        )

    seed_pairs: list[dict[str, Any]] = []
    for seed in SEEDS:
        gat_loss = float(
            np.mean(
                [
                    float(member[(GAT_ARM, seed, alias, "whole_node")]["hybrid_loss"])
                    for alias in ALIASES
                ],
                dtype=np.float64,
            )
        )
        self_loss = float(
            np.mean(
                [
                    float(member[(SELF_ARM, seed, alias, "whole_node")]["hybrid_loss"])
                    for alias in ALIASES
                ],
                dtype=np.float64,
            )
        )
        seed_pairs.append(
            {
                "seed": seed,
                "gat_equal_core_hybrid_loss": gat_loss,
                "self_equal_core_hybrid_loss": self_loss,
                "gat_relative_improvement": _relative_gain(
                    self_loss, gat_loss, label=f"seed pair {seed}"
                ),
                "gat_favored": gat_loss < self_loss,
            }
        )

    representation: list[dict[str, Any]] = []
    ensemble_gain: list[dict[str, Any]] = []
    for alias in ALIASES:
        gat = ensemble[(GAT_ARM, alias, "whole_node")]
        representation.append(
            {
                "core_alias": alias,
                "model_positive_ordinal_mae": gat["positive_ordinal_mae"],
                "per_core_reference_positive_ordinal_mae": gat[
                    "reference_per_gene_positive_ordinal_mae"
                ],
                "ordinal_relative_improvement": _relative_gain(
                    gat["reference_per_gene_positive_ordinal_mae"],
                    gat["positive_ordinal_mae"],
                    label=f"representation ordinal {alias}",
                ),
                "model_positive_continuous_huber": gat[
                    "positive_continuous_huber"
                ],
                "per_core_reference_positive_continuous_huber": gat[
                    "reference_per_gene_positive_continuous_huber"
                ],
                "continuous_relative_improvement": _relative_gain(
                    gat["reference_per_gene_positive_continuous_huber"],
                    gat["positive_continuous_huber"],
                    label=f"representation continuous {alias}",
                ),
                "model_detection_balanced_accuracy": gat[
                    "detection_balanced_accuracy"
                ],
                "per_core_reference_detection_balanced_accuracy": gat[
                    "reference_per_gene_detection_balanced_accuracy"
                ],
                "equal_core_reference_detection_balanced_accuracy": gat[
                    "reference_equal_core_detection_balanced_accuracy"
                ],
            }
        )
        for metric in (
            "hybrid_loss",
            "positive_ordinal_mae",
            "positive_continuous_huber",
        ):
            member_mean = float(
                np.mean(
                    [
                        float(member[(GAT_ARM, seed, alias, "whole_node")][metric])
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
            )
            ensemble_value = float(gat[metric])
            ensemble_gain.append(
                {
                    "core_alias": alias,
                    "metric": metric,
                    "prediction_ensemble": ensemble_value,
                    "mean_individual_member_metric": member_mean,
                    "relative_improvement": _relative_gain(
                        member_mean,
                        ensemble_value,
                        label=f"ensemble {alias} {metric}",
                    ),
                    "ensemble_not_worse": ensemble_value <= member_mean,
                }
            )
    return {
        "pooled_seed0_vs_prior": pooled_vs_prior,
        "graph_core_comparison": graph_core,
        "seed_pair_comparison": seed_pairs,
        "representation_core_comparison": representation,
        "ensemble_member_comparison": ensemble_gain,
    }


def _aggregate_equal_core(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["arm"]), str(row["mask_mode"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    for (arm, mode), selected in sorted(grouped.items()):
        if {str(row["core_alias"]) for row in selected} != set(ALIASES):
            raise PooledEnsembleComparisonError(
                f"equal-core aggregate lacks ten aliases for {(arm, mode)}"
            )
        item: dict[str, Any] = {
            "arm": arm,
            "mask_mode": mode,
            "core_count": len(ALIASES),
            "core_weighting": "equal",
        }
        for field in fields:
            item[field] = float(
                np.mean([_finite(row.get(field), field) for row in selected])
            )
        output.append(item)
    return output


def load_categorical_comparison(
    *,
    path: Path,
    gat_ensemble_core_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a clearly descriptive collapsed-state comparison."""

    if _sha256_file(path) != PRIOR_CATEGORICAL_COMPARISON_SHA256:
        raise PooledEnsembleComparisonError(
            "prior one-core categorical comparison checksum changed"
        )
    prior = _strict_json(path, label="prior categorical comparison")
    if (
        prior.get("artifact_kind") != "g2_token_multiseed_comparison"
        or prior.get("campaign_id") != PRIOR_CATEGORICAL_CAMPAIGN
        or prior.get("status") != "complete"
    ):
        raise PooledEnsembleComparisonError(
            "prior one-core categorical comparison has wrong identity or status"
        )
    selected = [
        row
        for row in gat_ensemble_core_rows
        if row.get("arm") == GAT_ARM and row.get("mask_mode") == "whole_node"
    ]
    if {row.get("core_alias") for row in selected} != set(ALIASES):
        raise PooledEnsembleComparisonError(
            "current categorical collapse lacks ten GAT ensemble core rows"
        )
    current: dict[str, Any] = {
        "source": "current_ten_core_pooled_hybrid_gat_ensemble",
        "variant": GAT_ARM,
        "scope": "ten_shared_fit_cores_equal_core",
        "core_count": 10,
        "model_seed_count": 7,
        "collapsed4_exact_accuracy_percent": 100.0
        * float(np.mean([float(row["collapsed4_exact_accuracy"]) for row in selected])),
        "collapsed4_balanced_accuracy_percent": 100.0
        * float(
            np.mean([float(row["collapsed4_balanced_accuracy"]) for row in selected])
        ),
        "collapsed4_positive_exact_accuracy_percent": 100.0
        * float(
            np.mean(
                [float(row["collapsed4_positive_exact_accuracy"]) for row in selected]
            )
        ),
        "all_zero_exact_accuracy_percent": 100.0
        * float(
            np.mean(
                [float(row["reference_all_zero_collapsed4_exact_accuracy"]) for row in selected]
            )
        ),
        "comparison_is_descriptive_only": True,
    }
    for state in range(4):
        current[f"state_{state}_recall_percent"] = 100.0 * float(
            np.mean([float(row[f"collapsed4_recall_{state}"]) for row in selected])
        )
    output = [current]
    aggregates = _mapping(prior.get("variant_aggregates"), "categorical aggregates")
    if set(aggregates) != set(PRIOR_CATEGORICAL_VARIANTS):
        raise PooledEnsembleComparisonError(
            "prior categorical comparison lacks exact two-variant coverage"
        )
    for variant in PRIOR_CATEGORICAL_VARIANTS:
        record = _mapping(aggregates.get(variant), f"categorical variant {variant}")
        if record.get("model_seeds") != [0, 1, 2]:
            raise PooledEnsembleComparisonError(
                f"prior categorical variant {variant} lacks exact seed coverage"
            )
        metrics = _mapping(record.get("metrics"), f"categorical metrics {variant}")
        baselines = _mapping(
            record.get("baselines"), f"categorical baselines {variant}"
        )
        recalls = _mapping(
            record.get("token_recalls_percent"), f"categorical recalls {variant}"
        )
        used_aggregates = [
            _mapping(metrics.get(name), f"categorical {name} aggregate")
            for name in ("exact_percent", "balanced_percent", "nonzero_percent")
        ]
        used_aggregates.append(
            _mapping(
                baselines.get("baseline_always_zero_accuracy_percent"),
                "categorical all-zero aggregate",
            )
        )
        used_aggregates.extend(
            _mapping(
                recalls.get(str(state)),
                f"categorical state-{state} recall aggregate",
            )
            for state in range(4)
        )
        if any(aggregate.get("n") != 3 for aggregate in used_aggregates):
            raise PooledEnsembleComparisonError(
                f"prior categorical variant {variant} lacks three-run aggregates"
            )
        item: dict[str, Any] = {
            "source": "prior_one_core_categorical_gat",
            "variant": variant,
            "scope": "one_core_three_model_seeds",
            "core_count": 1,
            "model_seed_count": 3,
            "collapsed4_exact_accuracy_percent": _finite(
                _mapping(metrics.get("exact_percent"), "exact aggregate").get("mean"),
                "prior exact accuracy",
            ),
            "collapsed4_balanced_accuracy_percent": _finite(
                _mapping(metrics.get("balanced_percent"), "balanced aggregate").get(
                    "mean"
                ),
                "prior balanced accuracy",
            ),
            "collapsed4_positive_exact_accuracy_percent": _finite(
                _mapping(metrics.get("nonzero_percent"), "nonzero aggregate").get(
                    "mean"
                ),
                "prior positive accuracy",
            ),
            "all_zero_exact_accuracy_percent": _finite(
                _mapping(
                    baselines.get("baseline_always_zero_accuracy_percent"),
                    "all-zero aggregate",
                ).get("mean"),
                "prior all-zero accuracy",
            ),
            "comparison_is_descriptive_only": True,
        }
        for state in range(4):
            item[f"state_{state}_recall_percent"] = _finite(
                _mapping(recalls.get(str(state)), f"state {state} recall").get("mean"),
                f"prior state {state} recall",
            )
        output.append(item)
    return output


def _flatten_gate_rows(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            scalar = {
                str(key): item
                for key, item in value.items()
                if not isinstance(item, Mapping)
            }
            if scalar:
                rows.append({"gate_path": "/".join(path), **scalar})
            for key, item in value.items():
                if isinstance(item, Mapping):
                    visit((*path, str(key)), item)

    for name, value in gates.items():
        if isinstance(value, Mapping):
            visit((str(name),), value)
    return rows


def _gate_outcome(gates: Mapping[str, Any]) -> tuple[bool, list[str]]:
    names = (
        "pooled_data_gate",
        "graph_gate",
        "representation_gate",
        "ensemble_gate",
    )
    failures = [
        name
        for name in names
        if _mapping(gates.get(name), f"{name} result").get("passed") is not True
    ]
    return not failures, failures


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_report(
    *,
    analysis: Mapping[str, Any],
    graph_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    categorical_rows: Sequence[Mapping[str, Any]],
    resource_rows: Sequence[Mapping[str, Any]],
) -> str:
    gates = _mapping(analysis["frozen_gates"], "frozen gates")
    all_passed, failed = _gate_outcome(gates)
    outcome = (
        "supported within the frozen exploratory estimand"
        if all_passed
        else "negative under at least one frozen gate"
    )
    lines = [
        "# Pooled ten-core hybrid-count ensemble",
        "",
        f"Status: complete; outcome: **{outcome}**.",
        "",
        "This exploratory, transductive study fitted each ensemble member as one "
        "shared model over all 117,386 cells in ten pathology-confirmed "
        "adjacent-normal cores. These samples are not true Normal. Each core "
        "remained a separate full-core graph batch; no cross-core edges were "
        "created.",
        "",
        "## Architecture and objective",
        "",
        "Both arms used the hybrid raw-count encoder with gene-specific count-token "
        "projections, exact within-bin standardized `log1p(count)`, and 22 permitted "
        "morphology/imaging covariates. Tokens were fixed as `0`, `1`, `2`, `3`, "
        "`4–7`, `8–15`, `16–31`, and `32+`; token 8 was input-only MASK. The GAT "
        "used two 512-wide four-head exact receiver-partitioned edge-conditioned "
        "layers on the mutual k=1,000 graph with 17 measured geometry features. "
        "The self arm had the same 11,674,880 trainable parameters and no graph, "
        "neighbors, coordinates, or edge attributes.",
        "",
        "Every member ran 200 global epochs (2,000 optimizer steps) with AdamW, "
        "learning rate `3e-4`, weight decay `1e-4`, clipping `1.0`, fixed masks, "
        "no validation selection, no early stopping, no neighbor sampling, and no "
        "edge dropout. The loss was the equal mean of balanced detection BCE, "
        "balanced six-threshold ordinal BCE, and positive standardized-log1p Huber.",
        "",
        "The seven-member ensembles averaged detection and ordinal probabilities "
        "and continuous standardized predictions before decoding. Member logits "
        "were streamed and not retained.",
        "",
        "## Frozen descriptive gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    for name in (
        "pooled_data_gate",
        "graph_gate",
        "representation_gate",
        "ensemble_gate",
    ):
        lines.append(f"| `{name}` | {str(bool(gates[name]['passed'])).lower()} |")
    if failed:
        lines.extend(
            [
                "",
                "Failed gates are a valid negative result and were not rescued by "
                "overall exact accuracy: " + ", ".join(f"`{name}`" for name in failed) + ".",
            ]
        )
    pooled_gate = _mapping(gates["pooled_data_gate"], "pooled data gate")
    pooled_metrics = _mapping(
        pooled_gate["metrics"], "pooled data gate metrics"
    )
    representation_gate = _mapping(
        gates["representation_gate"], "representation gate"
    )
    representation_metrics = _mapping(
        representation_gate["positive_metrics"],
        "representation positive metrics",
    )
    detection_check = _mapping(
        representation_gate["detection_balanced_accuracy"],
        "representation detection check",
    )
    ensemble_gate = _mapping(gates["ensemble_gate"], "ensemble gate")
    ensemble_metrics = _mapping(
        ensemble_gate["metrics"], "ensemble gate metrics"
    )
    lines.extend(
        [
            "",
            "Pooled seed-0 versus prior independent seed-0 GAT:",
            "",
            "| Metric | Mean relative gain | Favoring aliases | Passed |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in (
        "detection_bce",
        "positive_ordinal_mae",
        "reconstructed_count_log1p_mae",
    ):
        check = _mapping(pooled_metrics[metric], f"pooled {metric}")
        lines.append(
            f"| `{metric}` | "
            f"{100.0 * float(check['mean_relative_improvement']):.2f}% | "
            f"{int(check['favoring_core_count'])}/10 | "
            f"{str(bool(check['passed'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "Representation checks against the per-gene references:",
            "",
            "| Metric | Mean relative gain | Favoring aliases | Passed |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("positive_ordinal_mae", "positive_continuous_huber"):
        check = _mapping(representation_metrics[metric], metric)
        lines.append(
            f"| `{metric}` | "
            f"{100.0 * float(check['mean_relative_improvement']):.2f}% | "
            f"{int(check['favoring_core_count'])}/10 | "
            f"{str(bool(check['passed'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "Detection balanced accuracy was "
            f"{float(detection_check['model']):.4f} for the GAT ensemble, "
            f"{float(detection_check['per_core_reference']):.4f} for the "
            "per-core reference, and "
            f"{float(detection_check['pooled_reference']):.4f} for the "
            "equal-core pooled reference.",
            "",
            "Prediction-level ensemble versus the mean individual GAT member:",
            "",
            "| Metric | Ensemble | Mean member | Passed non-worsening |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in (
        "hybrid_loss",
        "positive_ordinal_mae",
        "positive_continuous_huber",
    ):
        check = _mapping(ensemble_metrics[metric], metric)
        lines.append(
            f"| `{metric}` | {_fmt(check['ensemble_equal_core_mean'])} | "
            f"{_fmt(check['mean_individual_equal_core_metric'])} | "
            f"{str(bool(check['passed'])).lower()} |"
        )
    graph_gate = _mapping(gates["graph_gate"], "graph gate")
    graph_noninferiority = _mapping(
        graph_gate["positive_metric_noninferiority"],
        "graph positive-metric noninferiority",
    )
    lines.extend(
        [
            "",
            "## Whole-node graph comparison",
            "",
            f"The equal-core mean relative GAT ensemble hybrid-loss improvement was "
            f"{100.0 * float(graph_gate['mean_relative_hybrid_loss_improvement']):.2f}%; "
            f"{int(graph_gate['favoring_core_count'])}/10 aliases and "
            f"{int(graph_gate['favoring_seed_pair_count'])}/7 paired model seeds "
            "favored the GAT. These are descriptive outcomes, not independent "
            "core-level inference, because all aliases share fitted weights.",
            "",
            "Positive ordinal MAE non-worsening: "
            f"{str(bool(graph_noninferiority['positive_ordinal_mae'])).lower()}; "
            "positive continuous Huber non-worsening: "
            f"{str(bool(graph_noninferiority['positive_continuous_huber'])).lower()}.",
            "",
            "| Alias | GAT loss | Self loss | GAT gain | GAT ordinal | Self ordinal | GAT cont. | Self cont. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in graph_rows:
        lines.append(
            f"| {row['core_alias']} | {_fmt(row['gat_hybrid_loss'])} | "
            f"{_fmt(row['matched_self_hybrid_loss'])} | "
            f"{100.0 * float(row['gat_relative_hybrid_loss_improvement']):.2f}% | "
            f"{_fmt(row['gat_positive_ordinal_mae'])} | "
            f"{_fmt(row['self_positive_ordinal_mae'])} | "
            f"{_fmt(row['gat_positive_continuous_huber'])} | "
            f"{_fmt(row['self_positive_continuous_huber'])} |"
        )
    lines.extend(
        [
            "",
            "Paired seeds are technical replicates:",
            "",
            "| Seed | GAT loss | Self loss | GAT gain |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {_fmt(row['gat_equal_core_hybrid_loss'])} | "
            f"{_fmt(row['self_equal_core_hybrid_loss'])} | "
            f"{100.0 * float(row['gat_relative_improvement']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Descriptive comparison with the previous tokenizer",
            "",
            "The previous experiment used one legacy true-Normal core, three seeds, "
            "and categorical `0/1/2/>=3` targets. The current values use ten "
            "adjacent-normal cores, collapse eight-state ensemble predictions after "
            "decoding, and average ten coupled core outcomes. Differences therefore "
            "mix tissue context, data, pooling, objective, architecture, and ensemble "
            "effects; they are descriptive, not an attribution to tokenizer choice.",
            "",
            "| Source | Exact | Balanced | Positive exact | All-zero exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in categorical_rows:
        lines.append(
            f"| {row['source']} / {row['variant']} | "
            f"{_fmt(row['collapsed4_exact_accuracy_percent'], 2)}% | "
            f"{_fmt(row['collapsed4_balanced_accuracy_percent'], 2)}% | "
            f"{_fmt(row['collapsed4_positive_exact_accuracy_percent'], 2)}% | "
            f"{_fmt(row['all_zero_exact_accuracy_percent'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## Runtime and verification",
            "",
            f"All 14 prespecified production members completed, every bundle and "
            "single final checkpoint verified, and all exact graph, data, split, "
            "mask, materialization, pilot-gate, and checkpoint identities reconciled. "
            f"There were {int(analysis['failure_attempt_count'])} failed or stale "
            "attempts before selected completions: "
            f"{int(analysis['pilot_failure_attempt_count'])} pilot and "
            f"{int(analysis['production_failure_attempt_count'])} production; none "
            "were hidden. The two attempt-1 pilot failures were operational "
            "`invalid_configuration` events before model construction or GPU work: "
            "long-lived workers retained a stale protocol validator. Safely reloaded "
            "workers and immutable attempt-2 retries completed.",
            "",
            "| Arm | Seed | Attempt | GPU | Device | Runtime h | Peak VRAM GiB |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in resource_rows:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['attempt']} | "
            f"{row['requested_gpu']} | {row['cuda_device_name']} | "
            f"{float(row['duration_seconds']) / 3600.0:.2f} | "
            f"{float(row['peak_vram_gib']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations and maximum conclusion",
            "",
            "- This is held-in masked-expression reconstruction with transductive "
            "preprocessing and references; it is not held-out-core or patient-held-out "
            "generalization.",
            "- Cells, mask repeats, and model seeds are not biological replicates. "
            "Shared fitted weights couple the ten core outcomes, so no formal "
            "core-level inference is reported.",
            "- Whole-node prediction may infer cell state and return learned means. "
            "Partial-gene masking can be dominated by same-cell co-expression.",
            "- Mutual k=1,000 is broad regional context, not direct cell-cell "
            "communication. The design does not establish a biological mechanism or "
            "causality.",
            "- Adjacent-normal tissue is not true Normal, and the targeted panel limits "
            "lineage interpretation.",
            "- Pooled seed-0 versus prior independent fits also changes cross-core "
            "weight sharing, optimizer-step budget, and expression normalization. "
            "It cannot isolate a causal effect of larger training data or sharing.",
            "- The prior categorical experiment used a legacy true-Normal core, "
            "whereas the current ten cores are adjacent-normal; its collapsed-state "
            "comparison is descriptive only.",
            "",
            "Maximum defensible conclusion: "
            + str(analysis["maximum_defensible_conclusion"])
            + ".",
            "",
            "Complete machine-readable tables and their checksums are listed in "
            "`manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _html_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    caption: str,
    percent_fields: frozenset[str] = frozenset(),
) -> str:
    caption_html = f"<caption>{html.escape(caption)}</caption>"
    head = "".join(
        f'<th scope="col">{html.escape(label)}</th>'
        for _, label in columns
    )
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for field, _ in columns:
            value = row.get(field)
            if isinstance(value, float):
                text = f"{100.0 * value:.2f}%" if field in percent_fields else f"{value:.4f}"
            else:
                text = "NA" if value is None else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table>{caption_html}<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _html_report(
    *,
    analysis: Mapping[str, Any],
    graph_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    categorical_rows: Sequence[Mapping[str, Any]],
    resource_rows: Sequence[Mapping[str, Any]],
) -> str:
    gates = _mapping(analysis["frozen_gates"], "frozen gates")
    _, failed = _gate_outcome(gates)
    gate_rows = [
        {"gate": name, "passed": bool(gates[name]["passed"])}
        for name in (
            "pooled_data_gate",
            "graph_gate",
            "representation_gate",
            "ensemble_gate",
        )
    ]
    pooled_metrics = _mapping(
        _mapping(gates["pooled_data_gate"], "pooled data gate")["metrics"],
        "pooled data metrics",
    )
    representation = _mapping(
        gates["representation_gate"], "representation gate"
    )
    representation_metrics = _mapping(
        representation["positive_metrics"], "representation metrics"
    )
    detail_rows: list[dict[str, Any]] = []
    for metric in (
        "detection_bce",
        "positive_ordinal_mae",
        "reconstructed_count_log1p_mae",
    ):
        check = _mapping(pooled_metrics[metric], metric)
        detail_rows.append(
            {
                "family": "pooled seed-0 vs independent",
                "metric": metric,
                "relative_gain": check["mean_relative_improvement"],
                "favoring_aliases": f"{check['favoring_core_count']}/10",
                "passed": check["passed"],
            }
        )
    for metric in ("positive_ordinal_mae", "positive_continuous_huber"):
        check = _mapping(representation_metrics[metric], metric)
        detail_rows.append(
            {
                "family": "representation vs per-gene reference",
                "metric": metric,
                "relative_gain": check["mean_relative_improvement"],
                "favoring_aliases": f"{check['favoring_core_count']}/10",
                "passed": check["passed"],
            }
        )
    graph_noninferiority = _mapping(
        _mapping(gates["graph_gate"], "graph gate")[
            "positive_metric_noninferiority"
        ],
        "graph noninferiority",
    )
    for metric in ("positive_ordinal_mae", "positive_continuous_huber"):
        detail_rows.append(
            {
                "family": "GAT vs matched self",
                "metric": f"{metric} non-worsening",
                "relative_gain": None,
                "favoring_aliases": "equal-core mean",
                "passed": graph_noninferiority[metric],
            }
        )
    ensemble_metrics = _mapping(
        _mapping(gates["ensemble_gate"], "ensemble gate")["metrics"],
        "ensemble gate metrics",
    )
    for metric in (
        "hybrid_loss",
        "positive_ordinal_mae",
        "positive_continuous_huber",
    ):
        check = _mapping(ensemble_metrics[metric], f"ensemble {metric}")
        detail_rows.append(
            {
                "family": "ensemble vs mean member",
                "metric": f"{metric} non-worsening",
                "relative_gain": None,
                "favoring_aliases": "equal-core mean",
                "passed": check["passed"],
            }
        )
    detection = _mapping(
        representation["detection_balanced_accuracy"],
        "representation detection",
    )
    embedded = json.dumps(
        {
            "campaign_id": CAMPAIGN_ID,
            "frozen_gates": gates,
            "maximum_defensible_conclusion": analysis[
                "maximum_defensible_conclusion"
            ],
            "operational_failures": analysis["operational_failures"],
            "graph_core_comparison": list(graph_rows),
            "seed_pair_comparison": list(seed_rows),
            "categorical_descriptive_comparison": list(categorical_rows),
        },
        sort_keys=True,
        allow_nan=False,
    )
    embedded = (
        embedded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    negative = (
        "<p class='negative'>Failed frozen gates: "
        + html.escape(", ".join(failed))
        + ". High overall exact accuracy does not rescue a failed gate.</p>"
        if failed
        else "<p>All four frozen descriptive gates passed.</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pooled ten-core hybrid-count ensemble</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#53606d; --line:#ccd4dc; --panel:#f5f7f9; --bad:#8a1c1c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:white; }}
main {{ max-width:1120px; margin:0 auto; padding:2rem; }}
h1,h2 {{ line-height:1.2; }} h2 {{ margin-top:2rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }}
.lede {{ color:var(--muted); font-size:1.05rem; }} .panel {{ background:var(--panel); padding:1rem 1.2rem; border-left:4px solid #506b84; }}
.negative {{ color:var(--bad); font-weight:650; }}
table {{ border-collapse:collapse; width:100%; margin:.8rem 0 1.3rem; font-size:.9rem; }}
caption {{ text-align:left; font-weight:650; margin-bottom:.35rem; }}
th,td {{ border:1px solid var(--line); padding:.42rem .52rem; text-align:right; vertical-align:top; }}
th:first-child,td:first-child {{ text-align:left; }} th {{ background:#eaf0f4; }}
code {{ background:#eef2f5; padding:.08rem .25rem; }} li {{ margin:.35rem 0; }}
@media print {{ main {{ max-width:none; padding:.5in; }} table {{ break-inside:avoid; }} }}
</style>
</head>
<body><main>
<h1>Pooled ten-core hybrid-count ensemble</h1>
<p class="lede">Complete exploratory held-in analysis of seven shared GAT members and seven exactly parameter-matched self-only members across ten pathology-confirmed adjacent-normal cores. These samples are not true Normal.</p>
<section class="panel"><strong>Maximum defensible conclusion.</strong> {html.escape(str(analysis["maximum_defensible_conclusion"]))}.</section>
<h2>Architecture, tokenization, and loss</h2>
<p>Each member was one shared model trained on all 117,386 cells as ten separate full-core graph batches. The encoder combined gene-specific fixed raw-count tokens (<code>0</code>, <code>1</code>, <code>2</code>, <code>3</code>, <code>4–7</code>, <code>8–15</code>, <code>16–31</code>, <code>32+</code>), exact standardized <code>log1p(count)</code>, and 22 permitted morphology/imaging covariates. Input-only token 8 represented MASK. The GAT used two 512-wide four-head exact receiver-partitioned layers and the 17-feature mutual k=1,000 broad-context graph. The self arm had the same 11,674,880 parameters and no graph information.</p>
<p>The fixed objective equally averaged balanced detection BCE, six-threshold balanced ordinal BCE, and positive standardized-log1p Huber. Seven-member probabilities and continuous predictions were averaged before decoding and scoring.</p>
<h2>Frozen descriptive gates</h2>
{_html_table(gate_rows, (("gate","Gate"),("passed","Passed")), caption="Frozen gate outcomes")}
{negative}
{_html_table(detail_rows, (("family","Family"),("metric","Metric"),("relative_gain","Mean gain"),("favoring_aliases","Favoring aliases"),("passed","Passed")), caption="Frozen gate metric details", percent_fields=frozenset(("relative_gain",)))}
<p>Detection balanced accuracy: GAT ensemble {float(detection["model"]):.4f}; per-core reference {float(detection["per_core_reference"]):.4f}; equal-core pooled reference {float(detection["pooled_reference"]):.4f}.</p>
<h2>Whole-node graph comparison by opaque alias</h2>
<p>Core outcomes are coupled by shared fitted weights; this table is descriptive and no formal paired core-level inference was performed. Mutual k=1,000 is broad regional context, not direct cell-cell communication.</p>
{_html_table(graph_rows, (("core_alias","Alias"),("gat_hybrid_loss","GAT loss"),("matched_self_hybrid_loss","Self loss"),("gat_relative_hybrid_loss_improvement","GAT gain"),("gat_positive_ordinal_mae","GAT ordinal MAE"),("self_positive_ordinal_mae","Self ordinal MAE"),("gat_positive_continuous_huber","GAT cont. Huber"),("self_positive_continuous_huber","Self cont. Huber")), caption="Whole-node ensemble comparison by opaque alias", percent_fields=frozenset(("gat_relative_hybrid_loss_improvement",)))}
<h2>Paired technical seeds</h2>
{_html_table(seed_rows, (("seed","Seed"),("gat_equal_core_hybrid_loss","GAT loss"),("self_equal_core_hybrid_loss","Self loss"),("gat_relative_improvement","GAT gain")), caption="Equal-core hybrid loss by paired technical seed", percent_fields=frozenset(("gat_relative_improvement",)))}
<h2>Descriptive <code>0/1/2/&gt;=3</code> comparison</h2>
<p>The previous categorical result used one legacy true-Normal core and a different objective. The current result uses ten adjacent-normal cores and collapses decoded eight-state predictions across ten coupled outcomes. Differences mix tissue context, data, pooling, objective, architecture, and ensembling and cannot be attributed to tokenizer choice.</p>
{_html_table(categorical_rows, (("source","Source"),("variant","Variant"),("collapsed4_exact_accuracy_percent","Exact %"),("collapsed4_balanced_accuracy_percent","Balanced %"),("collapsed4_positive_exact_accuracy_percent","Positive exact %"),("all_zero_exact_accuracy_percent","All-zero exact %")), caption="Descriptive comparison with the prior one-core categorical models")}
<h2>Runtime and verification</h2>
<p>All fourteen production bundles, one final checkpoint per member, fixed masks, exact graphs, data fingerprints, pilot authorization, and registry records were checksum-verified. There were {int(analysis["pilot_failure_attempt_count"])} failed pilot attempts and {int(analysis["production_failure_attempt_count"])} failed production attempts. The two attempt-1 pilot <code>invalid_configuration</code> failures occurred before model construction or GPU work because long-lived workers retained a stale protocol validator; safely reloaded workers and immutable attempt-2 retries completed. These were operational failures, not model outcomes.</p>
{_html_table(resource_rows, (("arm","Arm"),("seed","Seed"),("attempt","Attempt"),("requested_gpu","GPU"),("cuda_device_name","Device"),("duration_hours","Runtime h"),("peak_vram_gib","Peak VRAM GiB"),("failed_attempt_count","Earlier failures")), caption="Runtime, hardware, and retry inventory")}
<h2>Limitations</h2>
<ul>
<li>Held-in transductive reconstruction is not held-out-core or patient-held-out generalization.</li>
<li>Cells, masks, and seeds are not biological replicates; shared weights couple core outcomes.</li>
<li>Whole-node prediction may recover cell state and learned means; partial-gene masking may use same-cell co-expression.</li>
<li>The graph cannot establish direct interaction, mechanism, or causality.</li>
<li>Adjacent-normal is not true Normal, and the targeted panel limits interpretation.</li>
<li>Pooled versus prior independent fits also changes cross-core weight sharing, optimizer-step budget, and expression normalization, so it cannot isolate a causal effect of larger training data or sharing.</li>
<li>The prior categorical experiment used one legacy true-Normal core; its comparison with these adjacent-normal cores is descriptive only.</li>
</ul>
<script id="report-data" type="application/json">{embedded}</script>
</main></body></html>
"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise PooledEnsembleComparisonError("report payload contains NaN or infinity")
    return value


def _assert_portable_html(document: str) -> None:
    """Require a standalone HTML document with no external dependencies."""

    if (
        not isinstance(document, str)
        or "<!doctype html" not in document.lower()
        or "<style>" not in document.lower()
        or "<main" not in document.lower()
    ):
        raise PooledEnsembleComparisonError(
            "portable report must be a complete styled HTML document"
        )
    lowered = document.lower()
    if any(
        token in lowered
        for token in ("<link", "<iframe", "<object", "<embed", "@import")
    ):
        raise PooledEnsembleComparisonError(
            "portable report contains an external-capable HTML element"
        )
    external_attribute = re.search(
        r"""\b(?:src|href)\s*=\s*["'](?!data:|#)[^"']+["']""",
        document,
        flags=re.IGNORECASE,
    )
    external_css = re.search(
        r"""url\s*\(\s*["']?(?!data:)[^)]+""",
        document,
        flags=re.IGNORECASE,
    )
    if external_attribute or external_css:
        raise PooledEnsembleComparisonError(
            "portable report contains an external asset reference"
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted(set().union(*(set(row) for row in rows))) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {
                field: (
                    json.dumps(_jsonable(value), sort_keys=True, allow_nan=False)
                    if isinstance(value, (Mapping, list, tuple))
                    else value
                )
                for field, value in row.items()
            }
            writer.writerow(encoded)


def _publish_report_directory(
    *,
    output_dir: Path,
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    markdown: str,
    html_report: str,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write a complete sibling directory and publish it with one rename."""

    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise PooledEnsembleComparisonError(
            f"analysis output already exists and will not be overwritten: {output_dir}"
        )
    _assert_alias_safe_payload(analysis, label="analysis report")
    _assert_alias_safe_payload(tables, label="analysis tables")
    _assert_alias_safe_payload(provenance, label="analysis provenance")
    _assert_portable_html(html_report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "comparison.json", analysis)
        _write_json(temporary / "provenance.json", provenance)
        for name, rows in sorted(tables.items()):
            _write_json(temporary / f"{name}.json", list(rows))
            _write_csv(temporary / f"{name}.csv", list(rows))
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        (temporary / "report.html").write_text(html_report, encoding="utf-8")
        files = {
            path.relative_to(temporary).as_posix(): {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "artifact_kind": "pooled_hybrid_ensemble_comparison",
            "campaign_id": CAMPAIGN_ID,
            "portable_html": True,
            "protected_identifiers_emitted": False,
            "files": files,
            "comparison_sha256": files["comparison.json"]["sha256"],
            "provenance_sha256": files["provenance.json"]["sha256"],
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output_dir)
        return {
            **manifest,
            "manifest_sha256": _sha256_file(output_dir / "manifest.json"),
            "output_dir": output_dir.as_posix(),
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def run_analysis(
    *,
    paths: ProjectPaths,
    database_path: Path,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
    prior_categorical_path: Path,
    output_dir: Path,
    device_name: str,
    registry_factory: Callable[[Path], Registry] = Registry,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
    checkpoint_loader: Callable[..., Any] = torch.load,
    cohort_loader: Callable[
        ..., PooledFullCoreCohort | StreamingPooledFullCoreCohort
    ] = load_streaming_pooled_cohort,
    ensemble_evaluator: Callable[..., list[dict[str, Any]]] = recompute_prediction_ensembles,
) -> Mapping[str, Any]:
    """Run the audited comparison; heavy dependencies are injectable for tests."""

    if output_dir.resolve(strict=False).exists():
        raise PooledEnsembleComparisonError(
            f"analysis output already exists and will not be overwritten: {output_dir}"
        )
    initial_code_fingerprints = _analysis_code_fingerprints(paths)
    initial_input_fingerprints = _analysis_input_fingerprints(
        paths=paths,
        materialization_path=materialization_path,
        pilot_enqueue_path=pilot_enqueue_path,
        pilot_gate_path=pilot_gate_path,
        production_enqueue_path=production_enqueue_path,
        prior_categorical_path=prior_categorical_path,
    )
    runtime_environment = _analysis_runtime_environment(device_name)
    materialization, pilot_enqueue, pilot, enqueue = validate_campaign_receipts(
        paths=paths,
        materialization_path=materialization_path,
        pilot_enqueue_path=pilot_enqueue_path,
        pilot_gate_path=pilot_gate_path,
        production_enqueue_path=production_enqueue_path,
    )
    registry = registry_factory(database_path)
    campaign = registry.get_campaign(CAMPAIGN_ID)
    if campaign is None:
        raise PooledEnsembleComparisonError(
            "pooled campaign is absent from the authoritative registry"
        )
    queue_rows = _campaign_queue_rows(registry)
    audit = resolve_production_lineages(
        queue_rows=queue_rows,
        materialization=materialization,
        enqueue=enqueue,
        run_lookup=registry.get_run,
    )
    pilot_attempts, pilot_failures = resolve_pilot_lineages(
        queue_rows=queue_rows,
        materialization=materialization,
        pilot_enqueue=pilot_enqueue,
        pilot_gate=pilot,
        run_lookup=registry.get_run,
    )
    audit = CampaignAudit(
        selected_jobs=audit.selected_jobs,
        attempt_inventory=audit.attempt_inventory,
        registered_failure_inventory=audit.registered_failure_inventory,
        pilot_attempt_inventory=pilot_attempts,
        pilot_failure_inventory=pilot_failures,
    )
    combined_attempt_inventory = (
        *audit.pilot_attempt_inventory,
        *audit.attempt_inventory,
    )
    combined_failure_inventory = (
        *audit.pilot_failure_inventory,
        *audit.registered_failure_inventory,
    )
    validate_registered_production_membership(
        run_rows=_campaign_run_rows(registry),
        audit=audit,
    )
    planned = {
        (str(job["arm"]), int(job["seed"])): job
        for job in materialization["production_jobs"]
    }
    failed_by_slot: dict[tuple[str, int], int] = {
        slot: 0 for slot in planned
    }
    for row in audit.registered_failure_inventory:
        failed_by_slot[(str(row["arm"]), int(row["seed"]))] += 1

    evidence: dict[tuple[str, int], ProductionRunEvidence] = {}
    for slot in sorted(planned):
        evidence[slot] = audit_production_run(
            registry=registry,
            completed_job=audit.selected_jobs[slot],
            slot=slot,
            planned_job=planned[slot],
            materialization=materialization,
            failed_attempt_count=failed_by_slot[slot],
            paths=paths,
            bundle_verifier=bundle_verifier,
            checkpoint_loader=checkpoint_loader,
        )
    if len(evidence) != 14:
        raise PooledEnsembleComparisonError(
            "exactly fourteen verified production members are required"
        )
    verify_paired_member_initialization(evidence)

    member_rows = [
        dict(row)
        for slot in sorted(evidence)
        for row in evidence[slot].member_rows
    ]
    if len(member_rows) != EXPECTED_MEMBER_ROWS:
        raise PooledEnsembleComparisonError(
            "production bundles lack exact 2x7x10x3x3 member metric coverage"
        )
    member_gate_rows = collapse_member_replicates(
        member_rows, metric_names=GATE_METRICS
    )
    member_core_rows = collapse_rich_replicates(member_rows, ensemble=False)

    prior_rows = audit_prior_independent_seed0(
        registry=registry,
        materialization=materialization,
        paths=paths,
        bundle_verifier=bundle_verifier,
    )
    prior_core_rows = collapse_prior_replicates(prior_rows)

    cohort_record = _mapping(materialization.get("cohort"), "materialized cohort")
    prepared = _mapping(
        cohort_record.get("prepared_artifacts"),
        "materialized prepared artifacts",
    )
    cohort = cohort_loader(
        {
            alias: (
                Path(str(prepared[alias]))
                if Path(str(prepared[alias])).is_absolute()
                else paths.project_root / str(prepared[alias])
            )
            for alias in ALIASES
        }
    )
    ensemble_rows = ensemble_evaluator(
        evidence=evidence,
        cohort=cohort,
        materialization=materialization,
        device_name=device_name,
        checkpoint_loader=checkpoint_loader,
    )
    if len(ensemble_rows) != EXPECTED_ENSEMBLE_ROWS:
        raise PooledEnsembleComparisonError(
            "prediction-level ensemble evaluation lacks exact 2x10x3x3 coverage"
        )
    validate_ensemble_replicate_rows(
        ensemble_rows,
        materialization=materialization,
    )
    ensemble_gate_rows = collapse_ensemble_replicates(
        ensemble_rows, metric_names=ENSEMBLE_GATE_METRICS
    )
    ensemble_core_rows = collapse_rich_replicates(
        ensemble_rows, ensemble=True
    )
    gates = evaluate_frozen_gates(
        member_core_mode_rows=member_gate_rows,
        ensemble_core_mode_rows=ensemble_gate_rows,
        prior_independent_seed0_rows=prior_core_rows,
    )
    if gates.get("formal_core_level_inference") != (
        "not_performed_shared_fitted_weights_couple_core_outcomes"
    ):
        raise PooledEnsembleComparisonError(
            "frozen gate evaluator attempted unsupported core-level inference"
        )
    comparisons = build_comparison_tables(
        member_core_rows=member_core_rows,
        ensemble_core_rows=ensemble_core_rows,
        prior_core_rows=prior_core_rows,
    )
    categorical_rows = load_categorical_comparison(
        path=prior_categorical_path,
        gat_ensemble_core_rows=ensemble_core_rows,
    )
    equal_core_rows = _aggregate_equal_core(
        ensemble_core_rows,
        fields=(
            "hybrid_loss",
            "detection_bce",
            "ordinal_bce",
            "positive_continuous_huber",
            "detection_balanced_accuracy",
            "detection_sensitivity",
            "detection_specificity",
            "state8_exact_accuracy",
            "state8_balanced_accuracy",
            "positive_state_exact_accuracy",
            "positive_ordinal_mae",
            "positive_within_one_state_accuracy",
            "positive_continuous_mae",
            "reconstructed_count_log1p_mae",
            "collapsed4_exact_accuracy",
            "collapsed4_balanced_accuracy",
            "collapsed4_positive_exact_accuracy",
            "reference_all_zero_state8_exact_accuracy",
            "reference_all_zero_state8_balanced_accuracy",
            "reference_all_zero_collapsed4_exact_accuracy",
            "reference_per_gene_hybrid_loss",
            "reference_per_gene_detection_bce",
            "reference_per_gene_ordinal_bce",
            "reference_per_gene_positive_continuous_huber",
            "reference_per_gene_detection_balanced_accuracy",
            "reference_per_gene_state8_exact_accuracy",
            "reference_per_gene_state8_balanced_accuracy",
            "reference_per_gene_positive_ordinal_mae",
            "reference_per_gene_positive_continuous_mae",
            "reference_equal_core_hybrid_loss",
            "reference_equal_core_detection_bce",
            "reference_equal_core_ordinal_bce",
            "reference_equal_core_positive_continuous_huber",
            "reference_equal_core_detection_balanced_accuracy",
            "reference_equal_core_positive_ordinal_mae",
            "reference_equal_core_positive_continuous_mae",
            "reference_equal_core_state8_exact_accuracy",
            "reference_equal_core_reconstructed_count_log1p_mae",
        ),
    )

    resource_rows: list[dict[str, Any]] = []
    run_audit_rows: list[dict[str, Any]] = []
    for slot in sorted(evidence):
        item = evidence[slot]
        requested_gpu = _integer(
            audit.selected_jobs[slot].get("requested_gpu"),
            "selected production requested GPU",
        )
        resource_rows.append(
            {
                "arm": item.arm,
                "seed": item.seed,
                "run_id": item.run_id,
                "attempt": item.attempt,
                "requested_gpu": requested_gpu,
                "training_device": item.resource_usage.get("device"),
                "cuda_device_name": item.resource_usage.get(
                    "cuda_device_name"
                ),
                "torch_version": item.resource_usage.get("torch_version"),
                "torch_cuda_version": item.resource_usage.get(
                    "torch_cuda_version"
                ),
                "effective_amp": item.resource_usage.get(
                    "effective_training_amp"
                ),
                "duration_seconds": item.duration_seconds,
                "duration_hours": item.duration_seconds / 3600.0,
                "peak_vram_gib": item.peak_vram_gib,
                "peak_host_memory_bytes": item.peak_host_memory_bytes,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "final_train_hybrid_loss": item.convergence.get(
                    "final_equal_core_train_hybrid_loss"
                ),
                "minimum_train_hybrid_loss": item.convergence.get(
                    "minimum_observed_global_train_hybrid_loss"
                ),
                "last_20_epoch_loss_slope": item.convergence.get(
                    "last_20_global_epoch_loss_slope"
                ),
                "failed_attempt_count": item.failed_attempt_count,
            }
        )
        run_audit_rows.append(
            {
                "arm": item.arm,
                "seed": item.seed,
                "run_id": item.run_id,
                "attempt": item.attempt,
                "bundle_verified": True,
                "registry_artifacts_verified": True,
                "single_final_checkpoint_verified": True,
                "checkpoint_file_sha256": item.checkpoint_file_sha256,
                "state_dict_sha256": item.state_dict_sha256,
                "encoder_initial_state_sha256": (
                    item.encoder_initial_state_sha256
                ),
                "decoder_initial_state_sha256": (
                    item.decoder_initial_state_sha256
                ),
                "requested_gpu": requested_gpu,
                "materialization_checksum": materialization["checksum"],
                "pilot_gate_checksum": pilot["checksum"],
                "config_sha256": _normalized_root_config_sha256(item.config),
                "completed_global_epochs": EXPECTED_GLOBAL_EPOCHS,
                "optimizer_steps_completed": EXPECTED_OPTIMIZER_STEPS,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
            }
        )

    all_passed, failed_gates = _gate_outcome(gates)
    maximum_conclusion = (
        "Within this exploratory held-in transductive design, the shared "
        "seven-member GAT ensemble met all frozen pooled-data, graph, "
        "representation, and ensemble decision gates across ten adjacent-normal "
        "core aliases; this supports only " + MAXIMUM_CLAIM
        if all_passed
        else (
            "The complete exploratory held-in experiment was technically valid, "
            "but the frozen "
            + ", ".join(failed_gates)
            + " failed; no graph, representation, or ensemble success is claimed "
            "beyond the specific gates that passed"
        )
    )
    analysis = {
        "schema_version": 1,
        "artifact_kind": "pooled_hybrid_ensemble_comparison",
        "campaign_id": CAMPAIGN_ID,
        "status": "complete",
        "outcome": "supported" if all_passed else "negative",
        "exploratory": True,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "tissue_context_is_true_normal": False,
        "training_scope": {
            "one_shared_model_per_member": True,
            "total_fit_cells": 117386,
            "core_aliases": list(ALIASES),
            "cross_core_edges": False,
            "member_count_per_arm": 7,
            "production_run_count": 14,
        },
        "ensemble_contract": {
            "detection": "arithmetic_mean_probability",
            "ordinal": "arithmetic_mean_threshold_probability",
            "continuous": "arithmetic_mean_shared_standardized_prediction",
            "decode_after_combination": True,
            "average_member_metrics_as_ensemble": False,
            "metrics_recomputed_after_prediction_combination": True,
            "member_logits_retained_or_written": False,
        },
        "aggregation_contract": {
            "technical_mask_replicates_averaged_first": True,
            "core_weighting": "equal",
            "cells_are_biological_replicates": False,
            "mask_replicates_are_biological_replicates": False,
            "model_seeds_are_biological_replicates": False,
            "formal_core_level_inference": False,
            "reason": "shared fitted weights couple core outcomes",
        },
        "coverage": {
            "production_slots_expected": 14,
            "production_slots_completed": 14,
            "member_replicate_rows": len(member_rows),
            "ensemble_replicate_rows": len(ensemble_rows),
            "failed_attempts_before_completion": len(
                combined_failure_inventory
            ),
            "failed_pilot_attempts_before_completion": len(
                audit.pilot_failure_inventory
            ),
            "failed_production_attempts_before_completion": len(
                audit.registered_failure_inventory
            ),
        },
        "frozen_gates": gates,
        "failed_gates": failed_gates,
        "negative_results": [
            f"{name} failed its frozen descriptive threshold"
            for name in failed_gates
        ],
        "prior_categorical_comparison": {
            "descriptive_only": True,
            "previous_core_count": 1,
            "previous_tissue_context": "legacy_true_normal",
            "current_core_count": 10,
            "current_tissue_context": "pathology_confirmed_adjacent_normal",
            "differences_not_attributable_to_tokenizer_alone": True,
        },
        "operational_failures": {
            "pilot_invalid_configuration_attempts": len(
                audit.pilot_failure_inventory
            ),
            "production_failed_attempts": len(
                audit.registered_failure_inventory
            ),
            "diagnosis": (
                "The two attempt-1 pilot jobs failed before model construction "
                "or GPU work because long-lived workers retained a stale protocol "
                "validator. Safely reloaded workers and immutable attempt-2 retries "
                "completed. These were operational invalid_configuration failures, "
                "not model outcomes."
            ),
        },
        "limitations": [
            "held-in transductive reconstruction is not generalization",
            "shared fitted weights couple core outcomes",
            "cells masks and seeds are not biological replicates",
            "whole-node prediction may infer state and return learned means",
            "partial-gene masking may use same-cell co-expression",
            "mutual k1000 is broad context not direct interaction",
            "adjacent-normal is not true Normal",
            "targeted-panel coverage limits interpretation",
            (
                "pooled versus prior independent fitting changes cross-core "
                "weight sharing, optimizer-step budget, and expression "
                "normalization, so it does not isolate an effect of data size"
            ),
            (
                "the prior categorical comparison used one legacy true-Normal "
                "core while the current campaign uses adjacent-normal cores"
            ),
        ],
        "maximum_defensible_conclusion": maximum_conclusion,
        "prohibited_claims": [
            "patient-held-out generalization",
            "core-held-out generalization",
            "true-Normal performance",
            "direct cellular interaction",
            "biological mechanism",
            "causality",
        ],
        "failure_attempt_count": len(combined_failure_inventory),
        "pilot_failure_attempt_count": len(audit.pilot_failure_inventory),
        "production_failure_attempt_count": len(
            audit.registered_failure_inventory
        ),
        "tables": [
            "run_audit",
            "attempt_inventory",
            "registered_failures",
            "run_resources_and_convergence",
            "member_replicate_metrics",
            "member_core_mode_metrics",
            "prior_independent_seed0_replicate_metrics",
            "prior_independent_seed0_core_mode_metrics",
            "ensemble_replicate_metrics",
            "ensemble_core_mode_metrics",
            "ensemble_equal_core_aggregates",
            "pooled_seed0_vs_prior",
            "graph_core_comparison",
            "seed_pair_comparison",
            "representation_core_comparison",
            "ensemble_member_comparison",
            "categorical_descriptive_comparison",
            "gate_results",
        ],
    }
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "run_audit": run_audit_rows,
        "attempt_inventory": list(combined_attempt_inventory),
        "registered_failures": list(combined_failure_inventory),
        "run_resources_and_convergence": resource_rows,
        "member_replicate_metrics": member_rows,
        "member_core_mode_metrics": member_core_rows,
        "prior_independent_seed0_replicate_metrics": prior_rows,
        "prior_independent_seed0_core_mode_metrics": prior_core_rows,
        "ensemble_replicate_metrics": ensemble_rows,
        "ensemble_core_mode_metrics": ensemble_core_rows,
        "ensemble_equal_core_aggregates": equal_core_rows,
        "pooled_seed0_vs_prior": comparisons["pooled_seed0_vs_prior"],
        "graph_core_comparison": comparisons["graph_core_comparison"],
        "seed_pair_comparison": comparisons["seed_pair_comparison"],
        "representation_core_comparison": comparisons[
            "representation_core_comparison"
        ],
        "ensemble_member_comparison": comparisons[
            "ensemble_member_comparison"
        ],
        "categorical_descriptive_comparison": categorical_rows,
        "gate_results": _flatten_gate_rows(gates),
    }
    markdown = _markdown_report(
        analysis=analysis,
        graph_rows=comparisons["graph_core_comparison"],
        seed_rows=comparisons["seed_pair_comparison"],
        categorical_rows=categorical_rows,
        resource_rows=resource_rows,
    )
    html_report = _html_report(
        analysis=analysis,
        graph_rows=comparisons["graph_core_comparison"],
        seed_rows=comparisons["seed_pair_comparison"],
        categorical_rows=categorical_rows,
        resource_rows=resource_rows,
    )
    provenance = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_entry_point": Path(__file__).relative_to(
            paths.project_root
        ).as_posix(),
        "analysis_code_sha256": _sha256_file(Path(__file__)),
        "analysis_code_fingerprints": initial_code_fingerprints,
        "analysis_runtime_environment": runtime_environment,
        "analysis_input_file_sha256": initial_input_fingerprints,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "materialization_receipt_sha256": _sha256_file(materialization_path),
        "materialization_checksum": materialization["checksum"],
        "pilot_enqueue_receipt_sha256": _sha256_file(pilot_enqueue_path),
        "pilot_enqueue_checksum": pilot_enqueue["checksum"],
        "pilot_gate_receipt_sha256": _sha256_file(pilot_gate_path),
        "pilot_gate_checksum": pilot["checksum"],
        "production_enqueue_receipt_sha256": _sha256_file(
            production_enqueue_path
        ),
        "production_enqueue_checksum": enqueue["checksum"],
        "prior_categorical_comparison_sha256": _sha256_file(
            prior_categorical_path
        ),
        "cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "run_ids": [evidence[slot].run_id for slot in sorted(evidence)],
        "checkpoint_file_sha256": {
            f"{slot[0]}:seed-{slot[1]}": evidence[slot].checkpoint_file_sha256
            for slot in sorted(evidence)
        },
        "state_dict_sha256": {
            f"{slot[0]}:seed-{slot[1]}": evidence[slot].state_dict_sha256
            for slot in sorted(evidence)
        },
        "device": device_name,
        "read_only_registry_audit": True,
        "protected_identifiers_emitted": False,
    }
    if _analysis_code_fingerprints(paths) != initial_code_fingerprints:
        raise PooledEnsembleComparisonError(
            "analysis source files changed during evaluation"
        )
    if (
        _analysis_input_fingerprints(
            paths=paths,
            materialization_path=materialization_path,
            pilot_enqueue_path=pilot_enqueue_path,
            pilot_gate_path=pilot_gate_path,
            production_enqueue_path=production_enqueue_path,
            prior_categorical_path=prior_categorical_path,
        )
        != initial_input_fingerprints
    ):
        raise PooledEnsembleComparisonError(
            "analysis receipt or prior-comparison input changed during evaluation"
        )
    manifest = _publish_report_directory(
        output_dir=output_dir,
        analysis=analysis,
        tables=tables,
        markdown=markdown,
        html_report=html_report,
        provenance=provenance,
    )
    return {
        "analysis": analysis,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument(
        "--pilot-enqueue",
        type=Path,
        default=locked / "pilot_enqueue_receipt.json",
    )
    parser.add_argument(
        "--pilot-gate",
        type=Path,
        default=locked / "pilot_gate_receipt.json",
    )
    parser.add_argument(
        "--production-enqueue",
        type=Path,
        default=locked / "production_enqueue_receipt.json",
    )
    parser.add_argument(
        "--prior-categorical",
        type=Path,
        default=paths.report_root / DEFAULT_PRIOR_CATEGORICAL_RELATIVE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.report_root / DEFAULT_OUTPUT_RELATIVE,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Evaluation device; CUDA device 4 is prohibited.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = current_paths()
    result = run_analysis(
        paths=paths,
        database_path=args.database.resolve(strict=False),
        materialization_path=args.materialization.resolve(strict=False),
        pilot_enqueue_path=args.pilot_enqueue.resolve(strict=False),
        pilot_gate_path=args.pilot_gate.resolve(strict=False),
        production_enqueue_path=args.production_enqueue.resolve(strict=False),
        prior_categorical_path=args.prior_categorical.resolve(strict=False),
        output_dir=args.output.resolve(strict=False),
        device_name=str(args.device),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "status": result["analysis"]["status"],
                "outcome": result["analysis"]["outcome"],
                "output": result["manifest"]["output_dir"],
                "manifest_sha256": result["manifest"]["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

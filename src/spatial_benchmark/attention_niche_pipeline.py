"""End-to-end post-training extraction for six-core attention-routing niches.

The pipeline is deliberately read-only with respect to trained models and
prepared inputs.  It discovers only successful, catalog-verified ``last``
checkpoints, replays newly generated analysis masks in evaluation mode, and
streams exact complete-receiver attention diagnostics.  Attention-routing
niches are model-defined spatial partitions, not biological or causal claims.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import errno
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sqlite3
import stat
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from .cancer_pooled_full_core import (
    CANCER_ALIASES,
    CORE_NUMBERS,
    resolve_cancer_core_routes,
)
from .data import (
    ALLOWED_METADATA_COLUMNS,
    CoreSelection,
    discover_slide_raw_path,
    load_selected_core,
)
from .fingerprints import sha256_file
from .identifiers import scientific_id
from .paths import ProjectPaths, current_paths
from .pooled_relative_qkv_training import PooledRelativeQKVCoreBatch
from .relative_qkv_post_training import (
    CAMPAIGN_ID as UPSTREAM_CAMPAIGN_ID,
    load_prepared_relative_qkv_batches,
    load_relative_qkv_checkpoint,
    stream_receiver_attention,
)
from .run_archive import RunArchive, verify_run_bundle
from .training import set_deterministic_seed


ANALYSIS_CAMPAIGN_ID = "cmp_20260825_six_core_attention_routing_niches"
EXPECTED_ALIASES = tuple(CANCER_ALIASES)
EXPECTED_CORE_NUMBERS = tuple(CORE_NUMBERS)
EXPECTED_TOTAL_CELLS = 117_996
EXPECTED_TOTAL_DIRECTED_EDGES = 26_961_152
EXPECTED_TOTAL_RETAINED_MUTUAL_EDGES = 681_643
EXPECTED_GENE_SCHEMA_SHA256 = (
    "046eb86c7ea8f1fe6977598a0190132340400fc61802fcde63ab5ac0e9502b03"
)
EXPECTED_METADATA_SCHEMA_SHA256 = (
    "0f49df16640e047f0129ac36895f002e00c53dc56badef37e14fd45fcbca6fae"
)
EXPECTED_COHORT_MANIFEST_SHA256 = (
    "9c4ea4c9445f5230c5f823b4b1cd7a5b230767b43daa7018c99dc27bbf9613ad"
)
EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "20caa27c4d0a816a01475eda321e1910505b3b7849bc8eaa22a56e3cbfd1b48d"
)
EXPECTED_DATASET_FINGERPRINT = (
    "45fe649d1de0f3df3af0f4a1ae69d17249711359d55dde7d1acf45ed1881e77d"
)
EXPECTED_SPLIT_FINGERPRINT = (
    "956931f2cee4d48d33768fa7c6d034e43e4aaf2b40247b527f0755d860805ead"
)
EXPECTED_PREPROCESSING_VERSION = "cancer_6core_equal_core_log1p_metadata_v1"
EXPECTED_VARIANT = "relative_qkv_gat_256_l4_h8_radial_stratified_k200_r500"
EXPECTED_CORE_MAP_SHA256 = (
    "b790074a513fb8af48d991925e08559dd0fe64866509dc97f33c97c56b29e92f"
)
EXPECTED_RECONCILIATION_SHA256 = (
    "9a5fe243d385fb4bf3e2621f2f1b5f31ac749f6571c2592716d9222f2624233c"
)
EXPECTED_POLYGON_SHA256 = {
    "SO_1": "72b89055b30639ec9477b2f1ea9c16a08eeb2b88fd2bb21609da11e1ba2e1ee0",
    "SO_2": "bc1457eca09583557fc36696daa87f0a628d0fa37c0ba073b8ea2abf47f8172e",
}

LIMITATION = (
    "These regions are model-defined attention-routing niches. They describe "
    "stable spatial patterns in how the trained model routes information "
    "during masked-expression reconstruction. They do not by themselves "
    "establish direct molecular signaling or biological causality."
)

REQUIRED_ANALYSIS_OUTPUTS = (
    "six_core_attention_niche_map.png",
    "six_core_attention_niche_map.pdf",
    "six_core_attention_niche_map.svg",
    "six_core_mutual_attention_network_overlay.png",
    "six_core_mutual_attention_network_overlay.pdf",
    *(
        f"core_{core:02d}_attention_niche_map.png"
        for core in EXPECTED_CORE_NUMBERS
    ),
    "cell_attention_niche_assignments.parquet",
    "mutual_attention_edges.parquet",
    "directed_attention_edges.parquet",
    "attention_niche_summary.csv",
    "attention_niche_colors.json",
    "attention_niche_regions.geojson",
    "analysis_manifest.yaml",
    "analysis_qc_report.md",
    "README.md",
)

PROHIBITED_INTERPRETATION_IDENTIFIER_COLUMNS = frozenset(
    {
        "original_cell_identifier",
        "source_slide",
        "fov",
        "cell_ID",
        "patient_id",
    }
)
RENDER_RECOVERY_SOURCE_RUN_ID = (
    "r_20260825T081845Z_0da9fbf2_s000_f00_a01_dc19543d"
)
RENDER_RECOVERY_SOURCE_QUEUE_JOB_ID = "q_07ee75463aae7b7f890e"
RENDER_RECOVERY_FAILED_CONTENT_SHA256 = (
    "403887f52f7b7132409e2efdd2b3c4bef22751fa82f4f9a99c1fccd06254087c"
)
RENDER_RECOVERY_FAILURE_SIGNATURE = (
    "spatial_benchmark.attention_niche_visualization."
    "AttentionNicheVisualizationError: core region 'C01-N1785' ring 3 has "
    "zero signed area."
)
RENDER_RECOVERY_MODE = "verified_failed_run_render_only_v1"
RENDER_RECOVERY_SCIENTIFIC_ID = "sci_0da9fbf2afff6323"
RENDER_RECOVERY_OMITTED_RING_SHA256 = (
    "a93450935e8d0a435f4f4fd911cae8d104afabfda77410783de5b6f1a7d03331"
)
RENDER_RECOVERY_CANONICAL_OUTPUTS = (
    "cell_attention_niche_assignments.parquet",
    "mutual_attention_edges.parquet",
    "directed_attention_edges.parquet",
    "attention_niche_summary.csv",
    "attention_niche_colors.json",
    "attention_niche_regions.geojson",
    "attention_niche_parameter_sensitivity.csv",
)
VISUALIZATION_PATCH_SOURCE_RUN_ID = (
    "r_20260825T110043Z_0da9fbf2_s000_f00_a01_5e477ab9"
)
VISUALIZATION_PATCH_SOURCE_QUEUE_JOB_ID = "q_b549591aeff683de350f"
VISUALIZATION_PATCH_SUCCESS_FILE_SHA256 = (
    "accccfaf2bbde183423fb4580e690e9cbaee48e47b0c98603781ee25eef2fcd4"
)
VISUALIZATION_PATCH_CHECKSUM_MANIFEST_SHA256 = (
    "30f73654a09a135a59b6a3f85b5c8322f2d44f63af8d403a0a0595187951f0fe"
)
VISUALIZATION_PATCH_RENDERER_SHA256 = (
    "d08e6ee4d18c0aab89a437baf3e1c1b40eb29bd986697179b8a6c93bd2d18d02"
)
VISUALIZATION_PATCH_MODE = "verified_completed_run_visualization_only_v1"
VISUALIZATION_PATCH_RENDER_INPUTS = (
    "cell_attention_niche_assignments.parquet",
    "attention_niche_regions.geojson",
    "mutual_attention_edges.parquet",
)
VISUALIZATION_PATCH_OUTPUTS = tuple(
    relative
    for relative in REQUIRED_ANALYSIS_OUTPUTS
    if Path(relative).suffix.lower() in {".png", ".pdf", ".svg"}
)
_LINUX_FICLONE = 0x40049409


class AttentionNichePipelineError(RuntimeError):
    """Raised when a locked input, extraction, or output invariant fails."""


def _validate_interpretation_identifier_minimization(frame: pd.DataFrame) -> None:
    """Fail closed if a public interpretation table exposes source identifiers."""

    exposed = sorted(
        PROHIBITED_INTERPRETATION_IDENTIFIER_COLUMNS.intersection(frame.columns)
    )
    if exposed:
        raise AttentionNichePipelineError(
            "Interpretation export contains prohibited source identifiers: "
            + ", ".join(exposed)
        )


def _validated_locked_analysis_parameters(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete prespecified primary-analysis identity."""

    expected: dict[str, Any] = {
        "execution_role": "posthoc_readout_no_training",
        "upstream_campaign_id": UPSTREAM_CAMPAIGN_ID,
        "discover_completed_catalog_verified_last_checkpoints": True,
        "core_aliases": list(EXPECTED_ALIASES),
        "core_numbers": list(EXPECTED_CORE_NUMBERS),
        "final_graph_layer": True,
        "analysis_mask_seed": 2026082501,
        "analysis_mask_views": 10,
        "analysis_mask_derivation_fields": [
            "analysis_mask_seed",
            "core_alias",
            "mask_view_index",
        ],
        "all_genes_visible_sensitivity": True,
        "uniform_routing_threshold": 1.0,
        "consensus_mutual_score_threshold": 1.0,
        "support_threshold": 0.60,
        "primary_top_neighbors": 8,
        "top_neighbor_sensitivity": [5, 8, 10],
        "primary_leiden_resolution": 1.0,
        "leiden_resolution_sensitivity": [0.5, 1.0, 1.5],
        "leiden_seed": 2026082502,
        "color_seed": 2026082503,
        "polygon_coordinate_alignment_rule": (
            "centroid_tolerance_or_polygon_covers_coordinate"
        ),
        "polygon_centroid_tolerance_um": 5.0,
        "spatial_max_gap_um": 75.0,
        "micro_niche_cell_threshold": 20,
        "minimum_free_disk_gib": 40.0,
        "checkpoint_mutation_allowed": False,
        "required_analysis_outputs": list(REQUIRED_ANALYSIS_OUTPUTS),
    }
    drift = {
        name: {"expected": value, "observed": metadata.get(name)}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if drift:
        raise AttentionNichePipelineError(
            "Resolved locked analysis metadata drifted: "
            + ", ".join(sorted(drift))
        )
    return {
        "analysis_mask_seed": int(expected["analysis_mask_seed"]),
        "analysis_mask_views": int(expected["analysis_mask_views"]),
        "uniform_routing_threshold": float(
            expected["uniform_routing_threshold"]
        ),
        "consensus_mutual_score_threshold": float(
            expected["consensus_mutual_score_threshold"]
        ),
        "support_threshold": float(expected["support_threshold"]),
        "primary_top_neighbors": int(expected["primary_top_neighbors"]),
        "primary_leiden_resolution": float(
            expected["primary_leiden_resolution"]
        ),
        "leiden_seed": int(expected["leiden_seed"]),
        "color_seed": int(expected["color_seed"]),
        "polygon_coordinate_alignment_rule": str(
            expected["polygon_coordinate_alignment_rule"]
        ),
        "polygon_centroid_tolerance_um": float(
            expected["polygon_centroid_tolerance_um"]
        ),
        "spatial_max_gap_um": float(expected["spatial_max_gap_um"]),
        "micro_niche_cell_threshold": int(
            expected["micro_niche_cell_threshold"]
        ),
        "minimum_free_disk_gib": float(expected["minimum_free_disk_gib"]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AttentionNichePipelineError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AttentionNichePipelineError(f"{label} must be a mapping: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(value), handle, sort_keys=True, allow_unicode=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class RenderRecoveryRequest:
    """Explicit, immutable identity of the one approved render-only recovery."""

    mode: str
    source_run_id: str
    source_queue_job_id: str
    source_failed_marker_content_sha256: str
    expected_renderer_failure_signature: str
    minimum_figure_headroom_gib: float


@dataclass(frozen=True, slots=True)
class VerifiedRenderRecoverySource:
    """Verified failed source bundle and its immutable lineage documents."""

    root: Path
    resolved_config: Mapping[str, Any] = field(repr=False)
    bundle_verification: Mapping[str, Any] = field(repr=False)
    failed_marker: Mapping[str, Any] = field(repr=False)
    failed_marker_file_sha256: str
    artifact_checksum_manifest_sha256: str
    artifact_files: Mapping[str, Mapping[str, Any]] = field(repr=False)
    registry_artifacts: Mapping[str, Mapping[str, Any]] = field(repr=False)
    stderr_sha256: str
    scientific_config_sha256: str
    queue_job: Mapping[str, Any] = field(repr=False)
    source_git_identity: Mapping[str, Any] = field(repr=False)
    source_git_provenance: Mapping[str, Mapping[str, Any]] = field(repr=False)


@dataclass(frozen=True, slots=True)
class VisualizationPatchRequest:
    """Exact identity of the approved visualization-only patch source."""

    mode: str
    source_run_id: str
    source_queue_job_id: str
    source_success_marker_file_sha256: str
    source_artifact_checksum_manifest_sha256: str
    expected_renderer_source_sha256: str
    minimum_figure_headroom_gib: float


@dataclass(frozen=True, slots=True)
class VerifiedVisualizationPatchSource:
    """Receipt-bound completed source used without copying scientific tables."""

    root: Path
    resolved_config: Mapping[str, Any] = field(repr=False)
    success_marker: Mapping[str, Any] = field(repr=False)
    artifact_files: Mapping[str, Mapping[str, Any]] = field(repr=False)
    registry_artifacts: Mapping[str, Mapping[str, Any]] = field(repr=False)
    rendering_input_receipts: Mapping[str, Mapping[str, Any]] = field(repr=False)
    queue_job: Mapping[str, Any] = field(repr=False)
    source_success_marker_file_sha256: str
    source_artifact_checksum_manifest_sha256: str
    source_config_sha256: str
    scientific_config_sha256: str


def _validated_visualization_patch_request(
    launcher: Mapping[str, Any],
) -> VisualizationPatchRequest | None:
    """Return the one locked visualization patch, or ``None``."""

    if "visualization_patch" not in launcher:
        return None
    raw = launcher.get("visualization_patch")
    if not isinstance(raw, Mapping):
        raise AttentionNichePipelineError(
            "launcher.visualization_patch must be an explicit mapping."
        )
    expected = {
        "mode": VISUALIZATION_PATCH_MODE,
        "source_run_id": VISUALIZATION_PATCH_SOURCE_RUN_ID,
        "source_queue_job_id": VISUALIZATION_PATCH_SOURCE_QUEUE_JOB_ID,
        "source_success_marker_file_sha256": (
            VISUALIZATION_PATCH_SUCCESS_FILE_SHA256
        ),
        "source_artifact_checksum_manifest_sha256": (
            VISUALIZATION_PATCH_CHECKSUM_MANIFEST_SHA256
        ),
        "expected_renderer_source_sha256": VISUALIZATION_PATCH_RENDERER_SHA256,
        "minimum_figure_headroom_gib": 2.0,
    }
    if set(raw) != set(expected):
        raise AttentionNichePipelineError(
            "launcher.visualization_patch has missing or unrecognized fields."
        )
    drift = {
        key: {"expected": expected_value, "observed": raw.get(key)}
        for key, expected_value in expected.items()
        if raw.get(key) != expected_value
    }
    if drift:
        raise AttentionNichePipelineError(
            "Visualization patch identity drifted: "
            + ", ".join(sorted(drift))
        )
    return VisualizationPatchRequest(
        mode=str(expected["mode"]),
        source_run_id=str(expected["source_run_id"]),
        source_queue_job_id=str(expected["source_queue_job_id"]),
        source_success_marker_file_sha256=str(
            expected["source_success_marker_file_sha256"]
        ),
        source_artifact_checksum_manifest_sha256=str(
            expected["source_artifact_checksum_manifest_sha256"]
        ),
        expected_renderer_source_sha256=str(
            expected["expected_renderer_source_sha256"]
        ),
        minimum_figure_headroom_gib=float(
            expected["minimum_figure_headroom_gib"]
        ),
    )


def _validated_render_recovery_request(
    launcher: Mapping[str, Any],
) -> RenderRecoveryRequest | None:
    """Return the locked recovery request, or ``None`` for a normal extraction."""

    if "recovery" not in launcher:
        return None
    raw = launcher.get("recovery")
    if not isinstance(raw, Mapping):
        raise AttentionNichePipelineError(
            "launcher.recovery must be an explicit mapping."
        )
    expected = {
        "mode": RENDER_RECOVERY_MODE,
        "source_run_id": RENDER_RECOVERY_SOURCE_RUN_ID,
        "source_queue_job_id": RENDER_RECOVERY_SOURCE_QUEUE_JOB_ID,
        "source_failed_marker_content_sha256": (
            RENDER_RECOVERY_FAILED_CONTENT_SHA256
        ),
        "expected_renderer_failure_signature": (
            RENDER_RECOVERY_FAILURE_SIGNATURE
        ),
        "minimum_figure_headroom_gib": 2.0,
    }
    if set(raw) != set(expected):
        raise AttentionNichePipelineError(
            "launcher.recovery has missing or unrecognized fields."
        )
    drift = {
        key: {"expected": expected_value, "observed": raw.get(key)}
        for key, expected_value in expected.items()
        if raw.get(key) != expected_value
    }
    if drift:
        raise AttentionNichePipelineError(
            "Render-only recovery identity drifted: "
            + ", ".join(sorted(drift))
        )
    return RenderRecoveryRequest(
        mode=str(expected["mode"]),
        source_run_id=str(expected["source_run_id"]),
        source_queue_job_id=str(expected["source_queue_job_id"]),
        source_failed_marker_content_sha256=str(
            expected["source_failed_marker_content_sha256"]
        ),
        expected_renderer_failure_signature=str(
            expected["expected_renderer_failure_signature"]
        ),
        minimum_figure_headroom_gib=float(
            expected["minimum_figure_headroom_gib"]
        ),
    )


def _scientific_config_without_recovery_runtime(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only the approved recovery selector and runtime launcher."""

    normalized = json.loads(_canonical_json(dict(config)))
    if not isinstance(normalized, dict):  # pragma: no cover - defensive
        raise AttentionNichePipelineError("Resolved configuration is malformed.")
    normalized.pop("launcher", None)
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        raise AttentionNichePipelineError(
            "Resolved configuration metadata is malformed."
        )
    return normalized


def _verify_render_recovery_source_identity(
    *,
    request: RenderRecoveryRequest,
    database: str | Path,
    paths: ProjectPaths,
    current_config: Mapping[str, Any],
) -> VerifiedRenderRecoverySource:
    """Verify registry identity, full failed bundle, config, and traceback."""

    database_path = Path(database).resolve(strict=True)
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT run_id, campaign_id, status, artifact_path,
                   failure_category, config_json
            FROM runs
            WHERE run_id = ?
            """,
            (request.source_run_id,),
        ).fetchone()
        artifact_rows = connection.execute(
            """
            SELECT path, sha256, size_bytes, status
            FROM artifacts
            WHERE run_id = ?
            ORDER BY path
            """,
            (request.source_run_id,),
        ).fetchall()
        queue_rows = connection.execute(
            """
            SELECT job_id, run_id, campaign_id, status, failure_category,
                   requested_gpu, canonical_config_json
            FROM queue_jobs
            WHERE run_id = ?
            ORDER BY job_id
            """,
            (request.source_run_id,),
        ).fetchall()
    finally:
        connection.close()
    if row is None:
        raise AttentionNichePipelineError(
            f"Recovery source is absent from the registry: {request.source_run_id}."
        )
    if (
        str(row["run_id"]) != request.source_run_id
        or str(row["campaign_id"]) != ANALYSIS_CAMPAIGN_ID
        or str(row["status"]) != "failed"
        or str(row["failure_category"]) != "nonzero_exit"
    ):
        raise AttentionNichePipelineError(
            "Recovery source registry identity/status is not the locked failure."
        )
    if len(queue_rows) != 1:
        raise AttentionNichePipelineError(
            "Recovery source must have exactly one failed queue job."
        )
    queue_row = queue_rows[0]
    if (
        str(queue_row["job_id"]) != request.source_queue_job_id
        or str(queue_row["run_id"]) != request.source_run_id
        or str(queue_row["campaign_id"]) != ANALYSIS_CAMPAIGN_ID
        or str(queue_row["status"]) != "failed"
        or str(queue_row["failure_category"]) != "nonzero_exit"
        or str(queue_row["requested_gpu"]) != "0,2,3"
    ):
        raise AttentionNichePipelineError(
            "Recovery source queue-job identity/status is not the locked failure."
        )

    expected_root = RunArchive.artifact_path_for(request.source_run_id, paths)
    source_root = Path(str(row["artifact_path"]))
    try:
        source_root = source_root.resolve(strict=True)
        canonical_root = expected_root.resolve(strict=True)
    except OSError as exc:
        raise AttentionNichePipelineError(
            "Recovery source artifact bundle is not available."
        ) from exc
    if source_root != canonical_root or source_root.is_symlink():
        raise AttentionNichePipelineError(
            "Recovery source is not the canonical immutable archive path."
        )

    bundle_verification = verify_run_bundle(
        source_root,
        require_success_contract=False,
    )
    if (
        bundle_verification.get("valid") is not True
        or bundle_verification.get("status") != "failed"
        or int(bundle_verification.get("tombstoned_file_count", -1)) != 0
    ):
        raise AttentionNichePipelineError(
            "Full recovery-source bundle verification did not pass intact."
        )

    marker_path = source_root / "_FAILED"
    marker = _load_json(marker_path, "recovery source failed marker")
    if (
        marker.get("run_id") != request.source_run_id
        or marker.get("status") != "failed"
        or marker.get("content_sha256")
        != request.source_failed_marker_content_sha256
    ):
        raise AttentionNichePipelineError(
            "Recovery source _FAILED marker identity or digest drifted."
        )

    checksum_manifest_path = source_root / "provenance/artifact_checksums.json"
    checksum_manifest = _load_json(
        checksum_manifest_path,
        "recovery source artifact checksum manifest",
    )
    artifact_files = checksum_manifest.get("files")
    if not isinstance(artifact_files, Mapping):
        raise AttentionNichePipelineError(
            "Recovery source checksum manifest has no file mapping."
        )
    normalized_artifact_files: dict[str, Mapping[str, Any]] = {
        str(relative): dict(receipt)
        for relative, receipt in artifact_files.items()
        if isinstance(receipt, Mapping)
    }
    if len(normalized_artifact_files) != len(artifact_files):
        raise AttentionNichePipelineError(
            "Recovery source checksum manifest contains malformed receipts."
        )

    try:
        source_config = yaml.safe_load(
            (source_root / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        registry_config = json.loads(str(row["config_json"]))
        queue_config = json.loads(str(queue_row["canonical_config_json"]))
        source_manifest = yaml.safe_load(
            (source_root / "manifest.yaml").read_text(encoding="utf-8")
        )
        source_summary = json.loads(
            (source_root / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AttentionNichePipelineError(
            "Recovery source configuration/lifecycle documents are malformed."
        ) from exc
    if (
        not isinstance(source_config, Mapping)
        or not isinstance(registry_config, Mapping)
        or not isinstance(queue_config, Mapping)
        or source_config != registry_config
        or source_config != queue_config
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("run_id") != request.source_run_id
        or source_manifest.get("status") != "failed"
        or not isinstance(source_summary, Mapping)
        or source_summary.get("run_id") != request.source_run_id
        or source_summary.get("status") != "failed"
    ):
        raise AttentionNichePipelineError(
            "Recovery source registry, config, manifest, or summary identity drifted."
        )
    source_launcher = source_config.get("launcher")
    if not isinstance(source_launcher, Mapping) or "recovery" in source_launcher:
        raise AttentionNichePipelineError(
            "A render-recovery run cannot itself be used as the recovery source."
        )
    source_scientific = _scientific_config_without_recovery_runtime(source_config)
    current_scientific = _scientific_config_without_recovery_runtime(current_config)
    if (
        source_scientific != current_scientific
        or scientific_id(source_config) != RENDER_RECOVERY_SCIENTIFIC_ID
        or scientific_id(current_config) != RENDER_RECOVERY_SCIENTIFIC_ID
    ):
        raise AttentionNichePipelineError(
            "Recovery and source scientific configurations differ outside the "
            "approved recovery/launcher fields or scientific ID drifted."
        )

    stderr_path = source_root / "logs/stderr.log"
    try:
        stderr = stderr_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AttentionNichePipelineError(
            "Recovery source stderr cannot be read."
        ) from exc
    if (
        not stderr.rstrip().endswith(request.expected_renderer_failure_signature)
        or "render_attention_niche_visualizations(" not in stderr
        or "AttentionNicheVisualizationError" not in stderr
    ):
        raise AttentionNichePipelineError(
            "Recovery source does not have the exact approved renderer failure."
        )

    forbidden_source_visuals = (
        "six_core_attention_niche_map.png",
        "six_core_attention_niche_map.pdf",
        "six_core_attention_niche_map.svg",
        "six_core_mutual_attention_network_overlay.png",
        "six_core_mutual_attention_network_overlay.pdf",
        *(f"core_{core:02d}_attention_niche_map.png" for core in EXPECTED_CORE_NUMBERS),
    )
    if any((source_root / relative).exists() for relative in forbidden_source_visuals):
        raise AttentionNichePipelineError(
            "Recovery source unexpectedly contains a completed visualization."
        )

    registry_artifacts: dict[str, Mapping[str, Any]] = {}
    for artifact_row in artifact_rows:
        artifact_path = Path(str(artifact_row["path"]))
        if not artifact_path.is_absolute():
            artifact_path = paths.project_root / artifact_path
        try:
            relative = artifact_path.resolve(strict=False).relative_to(
                source_root
            ).as_posix()
        except ValueError as exc:
            raise AttentionNichePipelineError(
                "A registered recovery-source artifact is outside its bundle."
            ) from exc
        if relative in registry_artifacts:
            raise AttentionNichePipelineError(
                f"Duplicate registered recovery-source artifact: {relative}."
            )
        registry_artifacts[relative] = {
            "sha256": str(artifact_row["sha256"] or ""),
            "size_bytes": int(artifact_row["size_bytes"] or 0),
            "status": str(artifact_row["status"]),
        }
    if (
        len(registry_artifacts) != len(artifact_rows)
        or any(
            receipt.get("status") != "present"
            for receipt in registry_artifacts.values()
        )
    ):
        raise AttentionNichePipelineError(
            "Every registered recovery-source artifact must remain present."
        )
    for relative, expected in normalized_artifact_files.items():
        registered = registry_artifacts.get(relative)
        if (
            not isinstance(registered, Mapping)
            or registered.get("status") != "present"
            or registered.get("sha256") != expected.get("sha256")
            or int(registered.get("size_bytes", -1)) != int(expected.get("size", -2))
        ):
            raise AttentionNichePipelineError(
                f"Registered recovery-source artifact receipt drifted: {relative}."
            )
    for relative in RENDER_RECOVERY_CANONICAL_OUTPUTS:
        source_file = source_root / relative
        expected = normalized_artifact_files.get(relative)
        registered = registry_artifacts.get(relative)
        if (
            source_file.is_symlink()
            or not source_file.is_file()
            or not isinstance(expected, Mapping)
            or expected.get("type") != "file"
            or not isinstance(registered, Mapping)
            or registered.get("status") != "present"
            or registered.get("sha256") != expected.get("sha256")
            or int(registered.get("size_bytes", -1)) != int(expected.get("size", -2))
        ):
            raise AttentionNichePipelineError(
                f"Canonical recovery source is absent or not registry-bound: {relative}."
            )

    provenance_receipts: dict[str, Mapping[str, Any]] = {}
    for relative in (
        "provenance/git.json",
        "provenance/uncommitted_changes.patch",
        "provenance/untracked_files.json",
    ):
        expected = normalized_artifact_files.get(relative)
        if not isinstance(expected, Mapping) or expected.get("type") != "file":
            raise AttentionNichePipelineError(
                f"Source Git/dirty provenance is not checksum-bound: {relative}."
            )
        provenance_receipts[relative] = dict(expected)
    source_git_identity = _load_json(
        source_root / "provenance/git.json",
        "source Git identity",
    )
    if (
        len(str(source_git_identity.get("commit", ""))) != 40
        or not isinstance(source_git_identity.get("dirty"), bool)
        or len(str(source_git_identity.get("dirty_fingerprint", ""))) != 64
        or int(source_git_identity.get("untracked_file_count", -1)) < 0
    ):
        raise AttentionNichePipelineError("Source Git identity is malformed.")

    return VerifiedRenderRecoverySource(
        root=source_root,
        resolved_config=dict(source_config),
        bundle_verification=dict(bundle_verification),
        failed_marker=dict(marker),
        failed_marker_file_sha256=sha256_file(marker_path),
        artifact_checksum_manifest_sha256=sha256_file(checksum_manifest_path),
        artifact_files=normalized_artifact_files,
        registry_artifacts=registry_artifacts,
        stderr_sha256=sha256_file(stderr_path),
        scientific_config_sha256=_canonical_sha256(source_scientific),
        queue_job={
            "job_id": str(queue_row["job_id"]),
            "run_id": str(queue_row["run_id"]),
            "status": str(queue_row["status"]),
            "failure_category": str(queue_row["failure_category"]),
            "requested_gpu": str(queue_row["requested_gpu"]),
        },
        source_git_identity=source_git_identity,
        source_git_provenance=provenance_receipts,
    )


def _verify_visualization_patch_source_identity(
    *,
    request: VisualizationPatchRequest,
    database: str | Path,
    paths: ProjectPaths,
    current_config: Mapping[str, Any],
) -> VerifiedVisualizationPatchSource:
    """Verify the completed source without hashing unused scientific tables."""

    database_path = Path(database).resolve(strict=True)
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        run_row = connection.execute(
            """
            SELECT run_id, campaign_id, scientific_id, status, artifact_path,
                   failure_category, config_json
            FROM runs
            WHERE run_id = ?
            """,
            (request.source_run_id,),
        ).fetchone()
        queue_rows = connection.execute(
            """
            SELECT job_id, run_id, campaign_id, status, failure_category,
                   requested_gpu, canonical_config_json
            FROM queue_jobs
            WHERE run_id = ?
            ORDER BY job_id
            """,
            (request.source_run_id,),
        ).fetchall()
        artifact_rows = connection.execute(
            """
            SELECT path, sha256, size_bytes, status
            FROM artifacts
            WHERE run_id = ?
            ORDER BY path
            """,
            (request.source_run_id,),
        ).fetchall()
    finally:
        connection.close()
    if run_row is None or (
        str(run_row["run_id"]) != request.source_run_id
        or str(run_row["campaign_id"]) != ANALYSIS_CAMPAIGN_ID
        or str(run_row["scientific_id"]) != RENDER_RECOVERY_SCIENTIFIC_ID
        or str(run_row["status"]) != "completed"
        or run_row["failure_category"] is not None
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source is not the exact completed registry run."
        )
    if len(queue_rows) != 1:
        raise AttentionNichePipelineError(
            "Visualization-patch source must have exactly one completed queue job."
        )
    queue_row = queue_rows[0]
    if (
        str(queue_row["job_id"]) != request.source_queue_job_id
        or str(queue_row["run_id"]) != request.source_run_id
        or str(queue_row["campaign_id"]) != ANALYSIS_CAMPAIGN_ID
        or str(queue_row["status"]) != "completed"
        or queue_row["failure_category"] is not None
        or str(queue_row["requested_gpu"]) != "0,2,3"
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch queue identity/status drifted."
        )

    raw_root = Path(str(run_row["artifact_path"]))
    if raw_root.is_symlink():
        raise AttentionNichePipelineError(
            "Visualization-patch source archive cannot be a symlink."
        )
    try:
        source_root = raw_root.resolve(strict=True)
        canonical_root = RunArchive.artifact_path_for(
            request.source_run_id, paths
        ).resolve(strict=True)
    except OSError as exc:
        raise AttentionNichePipelineError(
            "Visualization-patch source archive is unavailable."
        ) from exc
    if source_root != canonical_root:
        raise AttentionNichePipelineError(
            "Visualization-patch source is not its canonical archive path."
        )

    success_path = source_root / "_SUCCESS"
    checksum_manifest_path = source_root / "provenance/artifact_checksums.json"
    if (
        success_path.is_symlink()
        or checksum_manifest_path.is_symlink()
        or not success_path.is_file()
        or not checksum_manifest_path.is_file()
        or sha256_file(success_path)
        != request.source_success_marker_file_sha256
        or sha256_file(checksum_manifest_path)
        != request.source_artifact_checksum_manifest_sha256
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source completion/checksum identity drifted."
        )
    success_marker = _load_json(success_path, "visualization-patch success marker")
    if (
        success_marker.get("run_id") != request.source_run_id
        or success_marker.get("status") != "success"
        or len(str(success_marker.get("content_sha256", ""))) != 64
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source success marker is malformed."
        )
    checksum_manifest = _load_json(
        checksum_manifest_path,
        "visualization-patch source checksum manifest",
    )
    raw_artifact_files = checksum_manifest.get("files")
    if (
        checksum_manifest.get("version") != 1
        or not isinstance(raw_artifact_files, Mapping)
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source checksum manifest lacks files."
        )
    artifact_files: dict[str, Mapping[str, Any]] = {}
    for relative, raw_receipt in raw_artifact_files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(raw_receipt, Mapping)
        ):
            raise AttentionNichePipelineError(
                "Visualization-patch source has a malformed artifact receipt."
            )
        receipt = dict(raw_receipt)
        path = source_root / relative
        try:
            path.relative_to(source_root)
            file_stat = path.lstat()
        except (OSError, ValueError) as exc:
            raise AttentionNichePipelineError(
                f"Visualization-patch source artifact is unavailable: {relative}."
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(file_stat.st_mode)
            or receipt.get("type") != "file"
            or int(receipt.get("size", -1)) != int(file_stat.st_size)
            or len(str(receipt.get("sha256", ""))) != 64
        ):
            raise AttentionNichePipelineError(
                f"Visualization-patch source artifact receipt drifted: {relative}."
            )
        artifact_files[relative] = receipt

    registry_artifacts: dict[str, Mapping[str, Any]] = {}
    for artifact_row in artifact_rows:
        artifact_path = Path(str(artifact_row["path"]))
        if not artifact_path.is_absolute():
            artifact_path = paths.project_root / artifact_path
        try:
            relative = artifact_path.resolve(strict=False).relative_to(
                source_root
            ).as_posix()
        except ValueError as exc:
            raise AttentionNichePipelineError(
                "A visualization-patch registry artifact is outside the source."
            ) from exc
        if relative in registry_artifacts:
            raise AttentionNichePipelineError(
                f"Duplicate visualization-patch artifact receipt: {relative}."
            )
        registry_artifacts[relative] = {
            "sha256": str(artifact_row["sha256"] or ""),
            "size_bytes": int(artifact_row["size_bytes"] or 0),
            "status": str(artifact_row["status"]),
        }
    checksum_manifest_relative = "provenance/artifact_checksums.json"
    expected_registry_paths = set(artifact_files) | {checksum_manifest_relative}
    if (
        set(registry_artifacts) != expected_registry_paths
        or any(
            receipt.get("status") != "present"
            for receipt in registry_artifacts.values()
        )
    ):
        raise AttentionNichePipelineError(
            "All and only source artifact receipts must be registered present."
        )
    for relative, expected in artifact_files.items():
        registered = registry_artifacts[relative]
        if (
            registered.get("sha256") != expected.get("sha256")
            or int(registered.get("size_bytes", -1))
            != int(expected.get("size", -2))
        ):
            raise AttentionNichePipelineError(
                f"Source registry/checksum receipt mismatch: {relative}."
            )
    manifest_registry_receipt = registry_artifacts[checksum_manifest_relative]
    if (
        manifest_registry_receipt.get("sha256")
        != request.source_artifact_checksum_manifest_sha256
        or int(manifest_registry_receipt.get("size_bytes", -1))
        != checksum_manifest_path.stat().st_size
    ):
        raise AttentionNichePipelineError(
            "Source checksum-manifest registry receipt drifted."
        )

    config_path = source_root / "config.resolved.yaml"
    config_receipt = artifact_files.get("config.resolved.yaml")
    if not isinstance(config_receipt, Mapping):
        raise AttentionNichePipelineError(
            "Source resolved configuration is not checksum-bound."
        )
    source_config_sha256 = sha256_file(config_path)
    try:
        source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        registry_config = json.loads(str(run_row["config_json"]))
        queue_config = json.loads(str(queue_row["canonical_config_json"]))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AttentionNichePipelineError(
            "Visualization-patch source configuration is malformed."
        ) from exc
    if (
        not isinstance(source_config, Mapping)
        or not isinstance(registry_config, Mapping)
        or not isinstance(queue_config, Mapping)
        or source_config != registry_config
        or source_config != queue_config
        or source_config_sha256 != config_receipt.get("sha256")
        or _scientific_config_without_recovery_runtime(source_config)
        != _scientific_config_without_recovery_runtime(current_config)
        or scientific_id(source_config) != RENDER_RECOVERY_SCIENTIFIC_ID
        or scientific_id(current_config) != RENDER_RECOVERY_SCIENTIFIC_ID
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source/current configuration identity drifted."
        )

    rendering_input_receipts: dict[str, Mapping[str, Any]] = {}
    for relative in VISUALIZATION_PATCH_RENDER_INPUTS:
        expected = artifact_files.get(relative)
        if not isinstance(expected, Mapping):
            raise AttentionNichePipelineError(
                f"Rendering input lacks a source receipt: {relative}."
            )
        observed_sha256 = sha256_file(source_root / relative)
        if observed_sha256 != expected.get("sha256"):
            raise AttentionNichePipelineError(
                f"Rendering input checksum drifted: {relative}."
            )
        rendering_input_receipts[relative] = {
            "sha256": observed_sha256,
            "size_bytes": int(expected["size"]),
            "source_relative_path": relative,
        }

    return VerifiedVisualizationPatchSource(
        root=source_root,
        resolved_config=dict(source_config),
        success_marker=dict(success_marker),
        artifact_files=artifact_files,
        registry_artifacts=registry_artifacts,
        rendering_input_receipts=rendering_input_receipts,
        queue_job={
            "job_id": str(queue_row["job_id"]),
            "run_id": str(queue_row["run_id"]),
            "status": str(queue_row["status"]),
            "failure_category": queue_row["failure_category"],
            "requested_gpu": str(queue_row["requested_gpu"]),
        },
        source_success_marker_file_sha256=(
            request.source_success_marker_file_sha256
        ),
        source_artifact_checksum_manifest_sha256=(
            request.source_artifact_checksum_manifest_sha256
        ),
        source_config_sha256=source_config_sha256,
        scientific_config_sha256=_canonical_sha256(
            _scientific_config_without_recovery_runtime(source_config)
        ),
    )


def _materialize_ficlone_reflink(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Clone one regular file with Linux FICLONE and no fallback path."""

    if platform.system() != "Linux" or not hasattr(os, "O_NOFOLLOW"):
        raise AttentionNichePipelineError(
            "Render recovery requires Linux FICLONE with O_NOFOLLOW."
        )
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.is_symlink():
        raise AttentionNichePipelineError(
            f"Refusing a symlink recovery source: {source_path}."
        )
    try:
        source_lstat = source_path.lstat()
    except OSError as exc:
        raise AttentionNichePipelineError(
            f"Cannot stat recovery source: {source_path}."
        ) from exc
    if not stat.S_ISREG(source_lstat.st_mode):
        raise AttentionNichePipelineError(
            f"Recovery source is not a regular file: {source_path}."
        )
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite recovery output: {destination_path}"
        )
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise AttentionNichePipelineError("Expected reflink checksum is malformed.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_fd = -1
    destination_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.ficlone-",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        source_fd = os.open(
            source_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        os.fchmod(destination_fd, stat.S_IMODE(source_lstat.st_mode))
        fcntl.ioctl(destination_fd, _LINUX_FICLONE, source_fd)
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = -1
        os.close(source_fd)
        source_fd = -1

        source_stat = source_path.stat()
        temporary_stat = temporary_path.stat()
        observed_sha256 = sha256_file(temporary_path)
        checks = {
            "same_filesystem": source_stat.st_dev == temporary_stat.st_dev,
            "distinct_inode": source_stat.st_ino != temporary_stat.st_ino,
            "source_link_count_one": source_stat.st_nlink == 1,
            "destination_link_count_one": temporary_stat.st_nlink == 1,
            "size_equal": source_stat.st_size == temporary_stat.st_size,
            "checksum_equal": observed_sha256 == expected_sha256,
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise AttentionNichePipelineError(
                "FICLONE materialization verification failed: "
                + ", ".join(failed)
            )

        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise AttentionNichePipelineError(
                "Linux renameat2 is required for no-replace reflink publication."
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(temporary_path),
            -100,
            os.fsencode(destination_path),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    f"Refusing to overwrite recovery output: {destination_path}"
                )
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination_path,
            )
        published = True
        directory_fd = os.open(
            destination_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except AttentionNichePipelineError:
        if published:
            destination_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if published:
            destination_path.unlink(missing_ok=True)
        raise AttentionNichePipelineError(
            f"Linux FICLONE publication failed; no copy fallback is permitted "
            f"for {source_path}."
        ) from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
        temporary_path.unlink(missing_ok=True)

    destination_stat = destination_path.stat()
    return {
        "method": "linux_ficlone_reflink",
        "temporary_destination": "same_directory_o_excl_o_nofollow",
        "publication": "renameat2_rename_noreplace_after_temp_verification",
        "parent_directory_fsynced": True,
        "copy_fallback_permitted": False,
        "source_type": "regular_file_not_symlink",
        "destination_type": "regular_file_not_symlink",
        "source_device": int(source_stat.st_dev),
        "destination_device": int(destination_stat.st_dev),
        "source_inode": int(source_stat.st_ino),
        "destination_inode": int(destination_stat.st_ino),
        "distinct_inode": True,
        "source_link_count": int(source_stat.st_nlink),
        "destination_link_count": int(destination_stat.st_nlink),
        "size_bytes": int(destination_stat.st_size),
        "sha256": observed_sha256,
    }


@dataclass(frozen=True, slots=True)
class CheckpointMember:
    """One immutable eligible completed model member."""

    seed: int
    run_id: str
    run_alias: str
    attempt: int
    completed_global_epochs: int
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str
    parameter_count: int
    bundle_path: Path
    success_marker_content_sha256: str
    catalog_artifact_id: int
    catalog_retention_class: str
    catalog_verification_status: str
    bundle_retention_tombstone_count: int
    bundle_retention_tombstones_sha256: str

    def manifest_record(self, project_root: Path) -> dict[str, Any]:
        def relative(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(project_root.resolve()))
            except ValueError:
                return str(path.resolve())

        value = asdict(self)
        value["checkpoint_path"] = relative(self.checkpoint_path)
        value["bundle_path"] = relative(self.bundle_path)
        return value


def _bundle_retention_tombstones(
    bundle: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Translate audited registry checkpoint deletions for bundle verification."""

    tombstones: dict[str, dict[str, Any]] = {}
    bundle_root = bundle.resolve(strict=True)
    for record in records:
        status = str(record["status"])
        if status == "retention_pending":
            raise AttentionNichePipelineError(
                f"Checkpoint retention is still pending for {record['path']}."
            )
        if status != "deleted_by_retention":
            continue
        target = Path(str(record["path"]))
        if not target.is_absolute():
            target = project_root / target
        try:
            relative = target.resolve(strict=False).relative_to(bundle_root).as_posix()
        except ValueError as exc:
            raise AttentionNichePipelineError(
                f"Retention tombstone is outside checkpoint bundle: {target}."
            ) from exc
        if not relative.startswith("checkpoints/") or relative in tombstones:
            raise AttentionNichePipelineError(
                f"Invalid or duplicate checkpoint tombstone: {relative}."
            )
        sha256 = str(record["sha256"] or "")
        size = record["size_bytes"]
        if (
            len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or size is None
            or int(size) < 0
        ):
            raise AttentionNichePipelineError(
                f"Checkpoint tombstone metadata is incomplete: {relative}."
            )
        tombstones[relative] = {
            "type": "file",
            "size": int(size),
            "sha256": sha256,
        }
    return tombstones


def discover_completed_checkpoint_members(
    database: str | Path,
    *,
    project_root: str | Path,
) -> tuple[CheckpointMember, ...]:
    """Discover successful catalog-verified final checkpoints, fail closed."""

    root = Path(project_root).resolve()
    source = Path(database).resolve(strict=True)
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT r.run_id, r.seed, r.attempt, r.artifact_path, r.config_json,
                   a.artifact_id, a.path AS checkpoint_path,
                   a.sha256 AS checkpoint_sha256, a.status AS artifact_status,
                   c.role, c.retention_class, c.verification_status,
                   ra.alias_id AS preferred_alias
            FROM runs AS r
            JOIN artifacts AS a ON a.run_id = r.run_id
            JOIN checkpoint_catalog AS c
              ON c.artifact_id = a.artifact_id AND c.run_id = r.run_id
            LEFT JOIN run_aliases AS ra
              ON ra.run_id = r.run_id AND ra.preferred = 1
            WHERE r.campaign_id = ?
              AND r.status = 'completed'
              AND c.role = 'last'
              AND c.verification_status = 'verified'
              AND a.status = 'present'
            ORDER BY r.seed, r.attempt
            """,
            (UPSTREAM_CAMPAIGN_ID,),
        ).fetchall()
        run_ids = [str(row["run_id"]) for row in rows]
        retention_rows = (
            connection.execute(
                """
                SELECT run_id, path, sha256, size_bytes, status
                FROM artifacts
                WHERE run_id IN ({})
                  AND status IN ('deleted_by_retention', 'retention_pending')
                ORDER BY run_id, path
                """.format(",".join("?" for _ in run_ids)),
                run_ids,
            ).fetchall()
            if run_ids
            else []
        )
    finally:
        connection.close()
    if not rows:
        raise AttentionNichePipelineError(
            "No completed catalog-verified upstream last checkpoints were found."
        )

    members: list[CheckpointMember] = []
    observed_seeds: set[int] = set()
    reference_bindings: dict[str, Any] | None = None
    retention_by_run: dict[str, list[Mapping[str, Any]]] = {}
    for retention_row in retention_rows:
        retention_by_run.setdefault(str(retention_row["run_id"]), []).append(
            retention_row
        )
    for row in rows:
        run_id = str(row["run_id"])
        seed = int(row["seed"])
        if seed in observed_seeds:
            raise AttentionNichePipelineError(
                f"More than one completed final checkpoint exists for model seed {seed}."
            )
        observed_seeds.add(seed)
        bundle = Path(str(row["artifact_path"])).resolve(strict=True)
        tombstones = _bundle_retention_tombstones(
            bundle,
            retention_by_run.get(run_id, []),
            project_root=root,
        )
        verify_run_bundle(bundle, tombstoned_artifacts=tombstones)
        checkpoint = Path(str(row["checkpoint_path"])).resolve(strict=True)
        expected_checkpoint = bundle / "checkpoints" / "last.ckpt"
        if checkpoint != expected_checkpoint.resolve(strict=True):
            raise AttentionNichePipelineError(
                f"Catalog last checkpoint is not the canonical bundle last.ckpt: {run_id}."
            )
        if any(token in checkpoint.name.casefold() for token in ("smoke", "test")):
            raise AttentionNichePipelineError("A smoke/test checkpoint passed discovery.")
        marker = _load_json(bundle / "_SUCCESS", f"success marker for {run_id}")
        if marker.get("run_id") != run_id or marker.get("status") != "success":
            raise AttentionNichePipelineError(f"Invalid success marker for {run_id}.")
        observed_checkpoint_sha = sha256_file(checkpoint)
        if observed_checkpoint_sha != str(row["checkpoint_sha256"]):
            raise AttentionNichePipelineError(
                f"Checkpoint checksum disagrees with the catalog for {run_id}."
            )
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise AttentionNichePipelineError(
                f"Checkpoint is not loadable: {checkpoint}."
            ) from exc
        if not isinstance(payload, Mapping):
            raise AttentionNichePipelineError("Checkpoint payload is not a mapping.")
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, Mapping):
            raise AttentionNichePipelineError("Checkpoint lacks its resolved config.")
        dataset = resolved.get("dataset")
        classification = resolved.get("classification")
        if not isinstance(dataset, Mapping) or not isinstance(classification, Mapping):
            raise AttentionNichePipelineError("Checkpoint data/classification receipt is absent.")
        bindings = {
            "dataset_fingerprint": dataset.get("dataset_fingerprint"),
            "split_fingerprint": dataset.get("split_fingerprint"),
            "preprocessing_version": dataset.get("preprocessing_version"),
            "cohort_manifest_file_sha256": dataset.get(
                "cohort_manifest_file_sha256"
            ),
            "graph_manifest_file_sha256": dataset.get("graph_manifest_file_sha256"),
            "core_aliases": list(dataset.get("core_aliases", ())),
            "original_core_numbers": dict(dataset.get("original_core_numbers", {})),
            "scientific_variant": classification.get("scientific_variant"),
        }
        expected_bindings = {
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "preprocessing_version": EXPECTED_PREPROCESSING_VERSION,
            "cohort_manifest_file_sha256": EXPECTED_COHORT_MANIFEST_SHA256,
            "graph_manifest_file_sha256": EXPECTED_GRAPH_MANIFEST_SHA256,
            "core_aliases": list(EXPECTED_ALIASES),
            "original_core_numbers": {
                alias: number
                for alias, number in zip(
                    EXPECTED_ALIASES, EXPECTED_CORE_NUMBERS, strict=True
                )
            },
            "scientific_variant": EXPECTED_VARIANT,
        }
        if bindings != expected_bindings:
            raise AttentionNichePipelineError(
                f"Checkpoint {run_id} does not match the locked data/model variant."
            )
        if reference_bindings is None:
            reference_bindings = bindings
        elif bindings != reference_bindings:
            raise AttentionNichePipelineError(
                "Eligible checkpoint preprocessing/graph bindings disagree."
            )
        if (
            payload.get("run_id") != run_id
            or int(payload.get("model_seed", -1)) != seed
            or payload.get("campaign_id") != UPSTREAM_CAMPAIGN_ID
            or not isinstance(payload.get("completed_global_epochs"), int)
            or not isinstance(payload.get("model_state_checksum"), str)
            or payload.get("plateau", {}).get("should_stop") is not True
        ):
            raise AttentionNichePipelineError(
                f"Checkpoint completion/model identity is invalid for {run_id}."
            )
        members.append(
            CheckpointMember(
                seed=seed,
                run_id=run_id,
                run_alias=str(row["preferred_alias"] or run_id),
                attempt=int(row["attempt"]),
                completed_global_epochs=int(payload["completed_global_epochs"]),
                checkpoint_path=checkpoint,
                checkpoint_sha256=observed_checkpoint_sha,
                model_state_sha256=str(payload["model_state_checksum"]),
                parameter_count=int(payload.get("parameter_count", -1)),
                bundle_path=bundle,
                success_marker_content_sha256=str(marker.get("content_sha256", "")),
                catalog_artifact_id=int(row["artifact_id"]),
                catalog_retention_class=str(row["retention_class"]),
                catalog_verification_status=str(row["verification_status"]),
                bundle_retention_tombstone_count=len(tombstones),
                bundle_retention_tombstones_sha256=_canonical_sha256(tombstones),
            )
        )
        del payload
    ordered = tuple(sorted(members, key=lambda item: item.seed))
    if len(ordered) == 1:
        label = "single-model, mask-consensus map"
    else:
        label = f"{len(ordered)}-model ensemble-consensus map"
    if len(ordered) > 5:
        raise AttentionNichePipelineError("More than five eligible seeds are unexpected.")
    # The label is computed here so callers cannot accidentally call one model
    # an ensemble; it is also recomputed in the published manifest.
    _ = label
    return ordered


@dataclass(frozen=True, slots=True)
class PreparedInputContract:
    cohort_dir: Path
    graph_dir: Path
    raw_dir: Path
    core_map_path: Path
    reconciliation_path: Path
    cohort_manifest: Mapping[str, Any] = field(repr=False)
    graph_manifest: Mapping[str, Any] = field(repr=False)
    core_records: Mapping[str, Mapping[str, Any]] = field(repr=False)
    gene_names: tuple[str, ...] = field(repr=False)
    metadata_names: tuple[str, ...] = field(repr=False)
    pre_analysis_file_receipts: Mapping[str, Mapping[str, Any]] = field(repr=False)


def _file_receipt(
    path: Path,
    *,
    root: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    try:
        display = str(path.resolve().relative_to(root.resolve()))
        root_kind = "project_root"
    except ValueError:
        if data_root is not None:
            try:
                display = str(path.resolve().relative_to(data_root.resolve()))
                root_kind = "BAGM_DATA_ROOT"
            except ValueError:
                display = str(path.resolve())
                root_kind = "absolute_external"
        else:
            display = str(path.resolve())
            root_kind = "absolute_external"
    return {
        "path": display,
        "root_kind": root_kind,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _polygon_paths(raw_dir: Path) -> tuple[Path, Path]:
    result: list[Path] = []
    for slide in ("SO_1", "SO_2"):
        candidates = sorted(raw_dir.glob(f"*{slide}*-polygons.csv"))
        resolved: list[Path] = []
        for candidate in candidates:
            nested = candidate / candidate.name
            path = nested if nested.is_file() else candidate
            if path.is_file():
                resolved.append(path)
        if len(resolved) != 1:
            raise AttentionNichePipelineError(
                f"Expected one polygon CSV for {slide}; found {len(resolved)}."
            )
        result.append(resolved[0].resolve())
    return result[0], result[1]


def verify_prepared_input_contract(
    *,
    paths: ProjectPaths,
    cohort_dir: str | Path,
    graph_dir: str | Path,
    raw_dir: str | Path,
    core_map_path: str | Path,
    reconciliation_path: str | Path,
) -> PreparedInputContract:
    """Strictly verify prepared schemas, graph caches, and source polygons."""

    cohort_root = Path(cohort_dir).resolve(strict=True)
    graph_root = Path(graph_dir).resolve(strict=True)
    raw_root = Path(raw_dir).resolve(strict=True)
    map_path = Path(core_map_path).resolve(strict=True)
    policy_path = Path(reconciliation_path).resolve(strict=True)
    cohort_manifest_path = cohort_root / "manifest.json"
    graph_manifest_path = graph_root / "manifest.json"
    if sha256_file(cohort_manifest_path) != EXPECTED_COHORT_MANIFEST_SHA256:
        raise AttentionNichePipelineError("Prepared cohort manifest checksum drifted.")
    if sha256_file(graph_manifest_path) != EXPECTED_GRAPH_MANIFEST_SHA256:
        raise AttentionNichePipelineError("Prepared graph manifest checksum drifted.")
    if sha256_file(map_path) != EXPECTED_CORE_MAP_SHA256:
        raise AttentionNichePipelineError("Core-map checksum drifted.")
    if sha256_file(policy_path) != EXPECTED_RECONCILIATION_SHA256:
        raise AttentionNichePipelineError("Reconciliation-policy checksum drifted.")
    cohort_manifest = _load_json(cohort_manifest_path, "cohort manifest")
    graph_manifest = _load_json(graph_manifest_path, "graph manifest")
    source_binding = cohort_manifest.get("source", {})
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("core_map_sha256") != EXPECTED_CORE_MAP_SHA256
        or source_binding.get("reconciliation_sha256")
        != EXPECTED_RECONCILIATION_SHA256
    ):
        raise AttentionNichePipelineError(
            "Prepared cohort source-map/reconciliation bindings drifted."
        )
    aliases = tuple(cohort_manifest.get("cohort", {}).get("aliases", ()))
    core_numbers = tuple(
        cohort_manifest.get("cohort", {}).get("original_core_numbers", ())
    )
    if aliases != EXPECTED_ALIASES or core_numbers != EXPECTED_CORE_NUMBERS:
        raise AttentionNichePipelineError("Prepared cohort core order drifted.")
    genes = tuple(cohort_manifest.get("features", {}).get("gene_names", ()))
    metadata_names = tuple(
        cohort_manifest.get("features", {}).get("measured_metadata_names", ())
    )
    if (
        len(genes) != 1_000
        or len(set(genes)) != 1_000
        or _canonical_sha256(list(genes)) != EXPECTED_GENE_SCHEMA_SHA256
        or _canonical_sha256(list(metadata_names)) != EXPECTED_METADATA_SCHEMA_SHA256
    ):
        raise AttentionNichePipelineError("Prepared gene/metadata order drifted.")

    # The canonical loader rehashes every cohort, graph, orientation, and
    # relative-geometry file and checks symmetry, order, and cross-core QC.
    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_root,
        graph_dir=graph_root,
    )
    if tuple(batch.alias for batch in batches) != EXPECTED_ALIASES:
        raise AttentionNichePipelineError("Loaded batch order is not the locked order.")
    if sum(batch.n_nodes for batch in batches) != EXPECTED_TOTAL_CELLS:
        raise AttentionNichePipelineError("Prepared cell total drifted.")
    if sum(batch.n_edges for batch in batches) != EXPECTED_TOTAL_DIRECTED_EDGES:
        raise AttentionNichePipelineError("Prepared directed-edge total drifted.")
    del batches

    graph_records = graph_manifest.get("cores")
    if not isinstance(graph_records, list):
        raise AttentionNichePipelineError("Graph manifest core records are absent.")
    by_alias = {
        str(record.get("alias")): record
        for record in graph_records
        if isinstance(record, Mapping)
    }
    if tuple(by_alias) != EXPECTED_ALIASES:
        raise AttentionNichePipelineError("Graph record aliases/order drifted.")
    for alias, number in zip(EXPECTED_ALIASES, EXPECTED_CORE_NUMBERS, strict=True):
        graph_qc = by_alias[alias].get("graph", {}).get("qc", {})
        if (
            int(graph_qc.get("cross_group_edges", -1)) != 0
            or int(graph_qc.get("self_loops", -1)) != 0
            or graph_qc.get("directed_edge_pairs_are_symmetric") is not True
            or graph_qc.get("receiver_major_canonical_order") is not True
            or int(graph_qc.get("n_nodes", -1)) <= 0
            or int(graph_qc.get("n_directed_edges", -1)) <= 0
        ):
            raise AttentionNichePipelineError(f"Graph QC failed for core {number}.")

    routes = resolve_cancer_core_routes(map_path, policy_path)
    cohort_records = cohort_manifest.get("cores")
    if not isinstance(cohort_records, list):
        raise AttentionNichePipelineError("Prepared cohort core receipts are absent.")
    cohort_by_alias = {
        str(record.get("alias")): record
        for record in cohort_records
        if isinstance(record, Mapping)
    }
    if tuple(cohort_by_alias) != EXPECTED_ALIASES:
        raise AttentionNichePipelineError("Prepared cohort core receipts drifted.")
    for route in routes:
        record = cohort_by_alias[route.alias]
        if (
            record.get("route_sha256") != route.route_sha256
            or record.get("source_slide") != route.slide
            or int(record.get("original_core_number", -1)) != route.core_number
        ):
            raise AttentionNichePipelineError(
                f"Prepared route binding drifted for {route.alias}."
            )

    polygon_so1, polygon_so2 = _polygon_paths(raw_root)
    for slide, polygon in (("SO_1", polygon_so1), ("SO_2", polygon_so2)):
        if sha256_file(polygon) != EXPECTED_POLYGON_SHA256[slide]:
            raise AttentionNichePipelineError(
                f"Source segmentation polygon checksum drifted for {slide}."
            )
    raw_source_files: list[Path] = []
    for slide in ("SO_1", "SO_2"):
        expression_path = discover_slide_raw_path(raw_root, slide, "expression")
        metadata_path = discover_slide_raw_path(raw_root, slide, "metadata")
        source_receipt = next(
            record["source_checksums"]
            for record in cohort_records
            if record["source_slide"] == slide
        )
        if (
            sha256_file(expression_path) != source_receipt["expression_sha256"]
            or sha256_file(metadata_path) != source_receipt["metadata_sha256"]
        ):
            raise AttentionNichePipelineError(
                f"Raw expression/metadata checksum drifted for {slide}."
            )
        raw_source_files.extend((expression_path.resolve(), metadata_path.resolve()))
    tracked_files: list[Path] = [
        cohort_manifest_path,
        graph_manifest_path,
        map_path,
        policy_path,
        polygon_so1,
        polygon_so2,
        cohort_root / "cohort_statistics.npz",
        *raw_source_files,
    ]
    tracked_files.extend(sorted((cohort_root / "cores").glob("*.npz")))
    for alias in EXPECTED_ALIASES:
        tracked_files.extend(
            graph_root / "cores" / alias / name
            for name in ("edge_index.npy", "orientation.npz", "relative_geometry.npy")
        )
    receipts = {
        str(path.resolve()): _file_receipt(
            path,
            root=paths.project_root,
            data_root=paths.data_root,
        )
        for path in tracked_files
    }
    return PreparedInputContract(
        cohort_dir=cohort_root,
        graph_dir=graph_root,
        raw_dir=raw_root,
        core_map_path=map_path,
        reconciliation_path=policy_path,
        cohort_manifest=cohort_manifest,
        graph_manifest=graph_manifest,
        core_records=by_alias,
        gene_names=genes,
        metadata_names=metadata_names,
        pre_analysis_file_receipts=receipts,
    )


def verify_inputs_unchanged(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    project_root: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    drift: list[dict[str, Any]] = []
    post: dict[str, Any] = {}
    for absolute, before in receipts.items():
        path = Path(absolute)
        after = _file_receipt(path, root=project_root, data_root=data_root)
        post[absolute] = after
        if (
            after["size_bytes"] != before.get("size_bytes")
            or after["sha256"] != before.get("sha256")
        ):
            drift.append({"path": after["path"], "before": dict(before), "after": after})
    if drift:
        raise AttentionNichePipelineError(
            f"{len(drift)} immutable prepared/source inputs changed during analysis."
        )
    return {"unchanged": True, "file_count": len(post), "post_receipts": post}


def _load_one_core_batch(
    *,
    alias: str,
    cohort_dir: Path,
    graph_dir: Path,
) -> tuple[PooledRelativeQKVCoreBatch, np.ndarray, np.ndarray]:
    if alias not in EXPECTED_ALIASES:
        raise AttentionNichePipelineError(f"Unknown core alias: {alias}")
    with np.load(cohort_dir / "cores" / f"{alias}.npz", allow_pickle=False) as data:
        target = np.array(data["target_expression"], dtype=np.float32, copy=True)
        covariates = np.array(data["node_covariates"], dtype=np.float32, copy=True)
        coordinates = np.array(data["coordinates_um"], dtype=np.float64, copy=True)
        counts = np.array(data["expression_counts"], dtype=np.int32, copy=True)
    edge_map = np.load(
        graph_dir / "cores" / alias / "edge_index.npy", mmap_mode="r"
    )
    geometry_map = np.load(
        graph_dir / "cores" / alias / "relative_geometry.npy", mmap_mode="r"
    )
    batch = PooledRelativeQKVCoreBatch(
        alias=alias,
        target_expression=torch.from_numpy(target),
        edge_index=torch.from_numpy(edge_map),
        relative_geometry=torch.from_numpy(geometry_map),
        node_covariates=torch.from_numpy(covariates),
    )
    return batch, coordinates, counts


def _load_and_verify_raw_core_identity(
    *,
    alias: str,
    cohort_dir: Path,
    raw_dir: Path,
    core_map_path: Path,
    reconciliation_path: Path,
    prepared_coordinates: np.ndarray,
    prepared_counts: np.ndarray,
    prepared_covariates: np.ndarray,
    expected_genes: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Recover original keys/metadata and prove prepared-node alignment.

    The immutable prepared arrays intentionally omit routing identifiers.  This
    function replays the repository's public, slide-qualified raw loader using
    the locked route and then compares every count, coordinate, transformed
    expression value, and transformed metadata value with the prepared-node
    order.  It returns only keys plus measured allow-listed metadata.
    """

    routes = resolve_cancer_core_routes(core_map_path, reconciliation_path)
    matches = [route for route in routes if route.alias == alias]
    if len(matches) != 1:
        raise AttentionNichePipelineError(
            f"The locked core mapping does not resolve exactly one route for {alias}."
        )
    route = matches[0]
    raw_core = load_selected_core(
        raw_dir,
        CoreSelection(
            slide=route.slide,
            fovs=route.fovs,
            label_policy="user_attested_cancer_reconciliation_v1",
        ),
        chunksize=8_192,
        expected_biological_probes=len(expected_genes),
        qc_policy="all",
    )
    if tuple(raw_core.gene_names) != tuple(expected_genes):
        raise AttentionNichePipelineError(f"Raw gene order drifted for {alias}.")
    if not np.array_equal(raw_core.expression, prepared_counts):
        raise AttentionNichePipelineError(
            f"Raw counts no longer align with prepared-node order for {alias}."
        )
    if not np.array_equal(raw_core.coordinates_um, prepared_coordinates):
        raise AttentionNichePipelineError(
            f"Raw physical coordinates no longer align for {alias}."
        )

    with np.load(cohort_dir / "cohort_statistics.npz", allow_pickle=False) as data:
        expression_mean = np.asarray(data["expression_mean"], dtype=np.float64)
        expression_scale = np.asarray(data["expression_scale"], dtype=np.float64)
        metadata_median = np.asarray(data["metadata_median"], dtype=np.float64)
        metadata_mean = np.asarray(data["metadata_mean"], dtype=np.float64)
        metadata_scale = np.asarray(data["metadata_scale"], dtype=np.float64)
        missing_indices = np.asarray(
            data["metadata_missing_indicator_indices"], dtype=np.int64
        )
    reconstructed_target = (
        np.log1p(raw_core.expression.astype(np.float64, copy=False))
        - expression_mean
    ) / expression_scale
    with np.load(cohort_dir / "cores" / f"{alias}.npz", allow_pickle=False) as data:
        prepared_target = np.asarray(data["target_expression"], dtype=np.float32)
    if not np.array_equal(
        reconstructed_target.astype(np.float32, copy=False), prepared_target
    ):
        raise AttentionNichePipelineError(
            f"Expression normalization no longer matches training for {alias}."
        )

    measured_metadata = np.asarray(raw_core.metadata, dtype=np.float64)
    missing = np.isnan(measured_metadata)
    imputed = np.where(missing, metadata_median, measured_metadata)
    if np.any(imputed < 0):
        raise AttentionNichePipelineError(
            f"Raw metadata violates the locked log1p transformation for {alias}."
        )
    transformed_metadata = (np.log1p(imputed) - metadata_mean) / metadata_scale
    if len(missing_indices):
        transformed_metadata = np.concatenate(
            [
                transformed_metadata,
                missing[:, missing_indices].astype(np.float64),
            ],
            axis=1,
        )
    if not np.array_equal(
        transformed_metadata.astype(np.float32, copy=False), prepared_covariates
    ):
        raise AttentionNichePipelineError(
            f"Metadata preprocessing no longer matches training for {alias}."
        )

    keys = raw_core.keys.copy()
    if tuple(keys.columns) != ("slide", "fov", "cell_ID"):
        raise AttentionNichePipelineError("Original cell-key schema drifted.")
    keys["source_slide"] = keys["slide"].astype(str)
    keys["fov"] = keys["fov"].astype(np.int32)
    keys["cell_ID"] = keys["cell_ID"].astype(np.int32)
    keys["source_qc_passed"] = np.asarray(raw_core.qc_passed, dtype=np.bool_)
    measured = np.asarray(raw_core.metadata, dtype=np.float32)
    receipt = {
        "alias": alias,
        "core_number": int(route.core_number),
        "source_slide": route.slide,
        "cell_count": int(len(keys)),
        "gene_order_sha256": _canonical_sha256(list(raw_core.gene_names)),
        "raw_counts_aligned": True,
        "coordinates_um_aligned": True,
        "target_expression_transform_reproduced": True,
        "metadata_transform_reproduced": True,
        "original_key_checksum_sha256": _canonical_sha256(
            keys[["source_slide", "fov", "cell_ID"]].to_dict(orient="records")
        ),
        "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
    }
    return keys, measured, receipt


def _attention_normalization_error(
    attention: np.ndarray,
    receiver: np.ndarray,
) -> float:
    values = np.asarray(attention, dtype=np.float32)
    receivers = np.asarray(receiver, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(receivers) or len(receivers) == 0:
        raise AttentionNichePipelineError("Attention shard shape is invalid.")
    changes = np.r_[0, np.flatnonzero(receivers[1:] != receivers[:-1]) + 1]
    if np.any(receivers[1:] < receivers[:-1]):
        raise AttentionNichePipelineError("Attention shard is not receiver-major.")
    sums = np.add.reduceat(values, changes, axis=0)
    return float(np.max(np.abs(sums - 1.0)))


def _stream_one_view(
    *,
    model: torch.nn.Module,
    batch: PooledRelativeQKVCoreBatch,
    mask: np.ndarray,
    edge_index: np.ndarray,
    indegree: np.ndarray,
    routing_target: np.ndarray,
    channel_targets: Mapping[str, np.ndarray] | None,
    amp: bool,
) -> dict[str, Any]:
    from .attention_routing_niches import validate_attention_shard

    edge_count = edge_index.shape[1]
    coverage = np.zeros(edge_count, dtype=np.bool_)
    max_attention_sum_error = 0.0
    max_logit_composition_error = 0.0
    max_softmax_reconstruction_error = 0.0
    shard_count = 0
    strict_receipt_digest = hashlib.sha256()

    def consume(
        receiver_start: int,
        receiver_stop: int,
        edge_ids: np.ndarray,
        attention: np.ndarray,
        content: np.ndarray,
        bias: np.ndarray,
        combined: np.ndarray,
    ) -> None:
        nonlocal max_attention_sum_error
        nonlocal max_logit_composition_error
        nonlocal max_softmax_reconstruction_error
        nonlocal shard_count
        ids = np.asarray(edge_ids, dtype=np.int64)
        arrays = {
            "attention": np.asarray(attention, dtype=np.float32),
            "content_qk": np.asarray(content, dtype=np.float32),
            "positional_bias": np.asarray(bias, dtype=np.float32),
            "combined_logit": np.asarray(combined, dtype=np.float32),
        }
        shapes = {value.shape for value in arrays.values()}
        if (
            len(shapes) != 1
            or arrays["attention"].ndim != 2
            or arrays["attention"].shape[0] != len(ids)
            or np.any(ids < 0)
            or np.any(ids >= edge_count)
            or bool(coverage[ids].any())
            or not all(np.isfinite(value).all() for value in arrays.values())
        ):
            raise AttentionNichePipelineError("Streamed attention channels misalign.")
        coverage[ids] = True
        source = edge_index[0, ids]
        receiver = edge_index[1, ids]
        if (
            np.any(source == receiver)
            or np.any(receiver < receiver_start)
            or np.any(receiver >= receiver_stop)
        ):
            raise AttentionNichePipelineError("Streamed edge identity is invalid.")
        attention_error = _attention_normalization_error(
            arrays["attention"], receiver
        )
        composition_error = float(
            np.max(
                np.abs(
                    arrays["combined_logit"]
                    - (arrays["content_qk"] + arrays["positional_bias"])
                )
            )
        )
        max_attention_sum_error = max(max_attention_sum_error, attention_error)
        max_logit_composition_error = max(
            max_logit_composition_error, composition_error
        )
        if attention_error > 2e-6:
            raise AttentionNichePipelineError(
                f"Attention normalization error {attention_error:.3g} exceeds tolerance."
            )
        if composition_error > 2e-4:
            raise AttentionNichePipelineError(
                f"Combined-logit composition error {composition_error:.3g} exceeds tolerance."
            )
        strict_audit = validate_attention_shard(
            edge_index[:, ids],
            arrays["attention"],
            arrays["content_qk"],
            arrays["positional_bias"],
            arrays["combined_logit"],
            n_nodes=int(indegree.shape[0]),
            source_indices=source,
            receiver_indices=receiver,
            expected_receiver_in_degrees=indegree,
            normalization_tolerance=2e-6,
            decomposition_atol=2e-4,
            decomposition_rtol=1e-6,
            softmax_atol=2e-6,
            softmax_rtol=1e-5,
        )
        max_softmax_reconstruction_error = max(
            max_softmax_reconstruction_error,
            strict_audit.maximum_softmax_deviation,
        )
        strict_receipt_digest.update(strict_audit.receipt_sha256.encode("ascii"))
        routing_target[ids] = (
            indegree[receiver] * arrays["attention"].mean(axis=1)
        )
        if channel_targets is not None:
            for name, target in channel_targets.items():
                target[ids] += arrays[name]
        shard_count += 1

    layer = stream_receiver_attention(
        model,
        batch,
        mask,
        layer=-1,
        amp=amp,
        consumer=consume,
    )
    if layer != int(model.graph_layers) - 1:
        raise AttentionNichePipelineError("Extraction did not use the final graph layer.")
    if not bool(coverage.all()) or not np.isfinite(routing_target).all():
        raise AttentionNichePipelineError("Attention stream did not cover every edge once.")
    return {
        "layer_number": int(layer),
        "shard_count": int(shard_count),
        "max_attention_sum_error": max_attention_sum_error,
        "max_logit_composition_error": max_logit_composition_error,
        "max_softmax_reconstruction_error": max_softmax_reconstruction_error,
        "strict_attention_shard_receipts_sha256": strict_receipt_digest.hexdigest(),
        "strict_attention_shard_count": int(shard_count),
        "edge_coverage": int(coverage.sum()),
    }


def _fixed_size_list(array: np.ndarray) -> pa.Array:
    values = np.ascontiguousarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise AttentionNichePipelineError("Per-head list array must be two-dimensional.")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=values.shape[1])


def _write_directed_core_table(
    path: Path,
    *,
    alias: str,
    core_number: int,
    seeds: Sequence[int],
    edge_index: np.ndarray,
    coordinates: np.ndarray,
    indegree: np.ndarray,
    routing_samples: np.ndarray,
    all_visible_routing: np.ndarray,
    channel_mask_means: Mapping[str, np.ndarray],
    row_group_size: int = 100_000,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite directed table: {path}")
    source_all = edge_index[0]
    receiver_all = edge_index[1]
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    row_count = 0
    try:
        for start in range(0, edge_index.shape[1], row_group_size):
            stop = min(start + row_group_size, edge_index.shape[1])
            source = source_all[start:stop]
            receiver = receiver_all[start:stop]
            sample = np.asarray(routing_samples[:, :, start:stop], dtype=np.float32)
            seed_values = np.median(sample, axis=1)
            view_values = np.median(sample, axis=0)
            visible = np.asarray(all_visible_routing[:, start:stop], dtype=np.float32)
            delta = coordinates[source] - coordinates[receiver]
            columns: dict[str, Any] = {
                "core_number": np.full(stop - start, core_number, dtype=np.int16),
                "core_alias": np.full(stop - start, alias),
                "edge_index_position": np.arange(start, stop, dtype=np.int64),
                "source_cell_index": source.astype(np.int64, copy=False),
                "receiver_cell_index": receiver.astype(np.int64, copy=False),
                "receiver_in_degree": indegree[receiver].astype(np.int32),
                "distance_um": np.linalg.norm(delta, axis=1).astype(np.float32),
                "head_mean_attention_mean": (
                    sample / indegree[receiver][None, None, :]
                ).mean(axis=(0, 1)).astype(np.float32),
                "degree_adjusted_routing_median": np.median(
                    sample, axis=(0, 1)
                ).astype(np.float32),
                "degree_adjusted_routing_mean": sample.mean(axis=(0, 1)).astype(
                    np.float32
                ),
                "degree_adjusted_routing_sd": sample.std(axis=(0, 1)).astype(
                    np.float32
                ),
                "degree_adjusted_routing_per_seed_mask_view": _fixed_size_list(
                    np.moveaxis(sample, 2, 0).reshape(stop - start, -1)
                ),
                "seed_aggregated_mean": seed_values.mean(axis=0).astype(np.float32),
                "seed_aggregated_sd": (
                    np.full(stop - start, np.nan, dtype=np.float32)
                    if len(seeds) == 1
                    else seed_values.std(axis=0, ddof=1).astype(np.float32)
                ),
                "mask_view_aggregated_mean": view_values.mean(axis=0).astype(
                    np.float32
                ),
                "mask_view_aggregated_sd": view_values.std(axis=0, ddof=1).astype(
                    np.float32
                ),
                "all_visible_routing_median": np.median(visible, axis=0).astype(
                    np.float32
                ),
                "all_visible_degree_adjusted_routing_per_seed": _fixed_size_list(
                    visible.T
                ),
            }
            for seed_position, seed in enumerate(seeds):
                for channel_name in (
                    "attention",
                    "content_qk",
                    "positional_bias",
                    "combined_logit",
                ):
                    columns[
                        f"{channel_name}_per_head_mask_mean_seed_{seed:03d}"
                    ] = _fixed_size_list(
                        channel_mask_means[channel_name][
                            seed_position, start:stop
                        ]
                    )
            table = pa.table(columns)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    path,
                    schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=["core_alias"],
                    write_statistics=True,
                )
            elif table.schema != schema:
                raise AttentionNichePipelineError("Directed Parquet schema drifted.")
            writer.write_table(table, row_group_size=len(table))
            row_count += len(table)
    finally:
        if writer is not None:
            writer.close()
    if row_count != edge_index.shape[1]:
        raise AttentionNichePipelineError("Directed Parquet row coverage drifted.")
    return {
        "path": str(path),
        "row_count": row_count,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "per_head_values": (
            "mask-view means retained separately for every model seed; nominal "
            "head indices are not averaged across seeds"
        ),
        "directional_routing_values": (
            "exact seed-major then mask-view-major degree-adjusted routing values "
            "retained for every directed edge"
        ),
    }


def _write_mutual_core_table(
    path: Path,
    *,
    alias: str,
    core_number: int,
    coordinates: np.ndarray,
    consensus: Any,
    retained_graph: Any,
    row_group_size: int = 100_000,
) -> dict[str, Any]:
    """Write every reciprocal pair, retaining direction and selection status."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite mutual table: {path}")
    pair_count = int(consensus.pairs.pair_count)
    retained = np.zeros(pair_count, dtype=np.bool_)
    endpoint_selection_count = np.zeros(pair_count, dtype=np.int8)
    retained[retained_graph.pair_positions] = True
    endpoint_selection_count[retained_graph.pair_positions] = np.asarray(
        retained_graph.endpoint_selection_count, dtype=np.int8
    )
    seed_sd = consensus.seed_standard_deviation_after_view_median
    mask_sd = consensus.mask_view_standard_deviation_after_seed_median
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    row_count = 0
    try:
        for start in range(0, pair_count, row_group_size):
            stop = min(start + row_group_size, pair_count)
            pairs = consensus.pairs.pair_cells[start:stop]
            first = pairs[:, 0]
            second = pairs[:, 1]
            distance = np.linalg.norm(
                coordinates[first] - coordinates[second], axis=1
            ).astype(np.float32)
            columns: dict[str, Any] = {
                "core_number": np.full(stop - start, core_number, dtype=np.int16),
                "core_alias": np.full(stop - start, alias),
                "mutual_pair_position": np.arange(start, stop, dtype=np.int64),
                "cell_i_index": first.astype(np.int64, copy=False),
                "cell_j_index": second.astype(np.int64, copy=False),
                "distance_um": distance,
                "M_ij": np.asarray(
                    consensus.consensus_mutual_score[start:stop], dtype=np.float32
                ),
                "support_P_ij": np.asarray(
                    consensus.support_fraction[start:stop], dtype=np.float32
                ),
                "i_to_j_degree_adjusted_routing": np.asarray(
                    consensus.i_to_j_directional_median[start:stop],
                    dtype=np.float32,
                ),
                "j_to_i_degree_adjusted_routing": np.asarray(
                    consensus.j_to_i_directional_median[start:stop],
                    dtype=np.float32,
                ),
                "seed_mean_after_mask_median": np.asarray(
                    consensus.seed_mean_after_view_median[start:stop],
                    dtype=np.float32,
                ),
                "seed_spread_sd_after_mask_median": (
                    np.full(stop - start, np.nan, dtype=np.float32)
                    if seed_sd is None
                    else np.asarray(seed_sd[start:stop], dtype=np.float32)
                ),
                "mask_view_mean_after_seed_median": np.asarray(
                    consensus.mask_view_mean_after_seed_median[start:stop],
                    dtype=np.float32,
                ),
                "mask_view_spread_sd_after_seed_median": (
                    np.full(stop - start, np.nan, dtype=np.float32)
                    if mask_sd is None
                    else np.asarray(mask_sd[start:stop], dtype=np.float32)
                ),
                "mutual_score_per_seed_mask_median": _fixed_size_list(
                    np.asarray(
                        consensus.per_seed_view_median[:, start:stop].T,
                        dtype=np.float32,
                    )
                ),
                "mutual_score_per_mask_seed_median": _fixed_size_list(
                    np.asarray(
                        consensus.per_mask_view_seed_median[:, start:stop].T,
                        dtype=np.float32,
                    )
                ),
                "i_to_j_edge_index_position": np.asarray(
                    consensus.pairs.i_to_j_edge_ids[start:stop], dtype=np.int64
                ),
                "j_to_i_edge_index_position": np.asarray(
                    consensus.pairs.j_to_i_edge_ids[start:stop], dtype=np.int64
                ),
                "retained_primary": retained[start:stop],
                "endpoint_selection_count": endpoint_selection_count[start:stop],
            }
            table = pa.table(columns)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    path,
                    schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=["core_alias"],
                    write_statistics=True,
                )
            elif table.schema != schema:
                raise AttentionNichePipelineError("Mutual Parquet schema drifted.")
            writer.write_table(table, row_group_size=len(table))
            row_count += len(table)
    finally:
        if writer is not None:
            writer.close()
    if row_count != pair_count:
        raise AttentionNichePipelineError("Mutual Parquet row coverage drifted.")
    return {
        "path": str(path),
        "row_count": int(row_count),
        "retained_row_count": int(retained.sum()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _incident_median(
    n_nodes: int,
    edge_pairs: np.ndarray,
    values: np.ndarray,
    *,
    default: float,
) -> np.ndarray:
    """Median incident-edge value per node without dense node-edge scans."""

    pairs = np.asarray(edge_pairs, dtype=np.int64)
    edge_values = np.asarray(values, dtype=np.float64)
    if pairs.shape != (len(edge_values), 2):
        raise AttentionNichePipelineError("Incident value arrays are misaligned.")
    result = np.full(int(n_nodes), float(default), dtype=np.float64)
    if not len(pairs):
        return result
    endpoints = np.concatenate((pairs[:, 0], pairs[:, 1]))
    repeated_values = np.concatenate((edge_values, edge_values))
    order = np.argsort(endpoints, kind="stable")
    ordered_endpoints = endpoints[order]
    ordered_values = repeated_values[order]
    starts = np.r_[
        0, np.flatnonzero(ordered_endpoints[1:] != ordered_endpoints[:-1]) + 1
    ]
    for start, stop in zip(
        starts, np.r_[starts[1:], len(ordered_endpoints)], strict=True
    ):
        result[int(ordered_endpoints[start])] = float(
            np.median(ordered_values[start:stop])
        )
    return result


def _integer_partition(labels: Sequence[object] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    unique = sorted({str(value) for value in values.tolist()})
    mapping = {label: position for position, label in enumerate(unique)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64)


def _write_dataframe_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataframe_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".csv", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _descriptive_niche_summaries(
    *,
    alias: str,
    core_number: int,
    final_niche_ids: np.ndarray,
    micro_niche: np.ndarray,
    hub_scores: np.ndarray,
    confidence: np.ndarray,
    counts: np.ndarray,
    gene_names: Sequence[str],
    measured_metadata: np.ndarray,
    metadata_names: Sequence[str],
    retained_pairs: np.ndarray,
    retained_scores: np.ndarray,
    regions: Sequence[Any],
) -> pd.DataFrame:
    """Build concise, explicitly descriptive per-niche summaries."""

    niche_ids = np.asarray(final_niche_ids, dtype=object)
    ordered_ids = sorted({str(value) for value in niche_ids.tolist()})
    code_by_id = {niche_id: position for position, niche_id in enumerate(ordered_ids)}
    codes = np.asarray([code_by_id[str(value)] for value in niche_ids], dtype=np.int64)
    region_by_id = {region.niche_id: region for region in regions}
    log_counts = np.log1p(np.asarray(counts, dtype=np.float32))
    core_gene_sum = log_counts.sum(axis=0, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    pairs = np.asarray(retained_pairs, dtype=np.int64)
    scores = np.asarray(retained_scores, dtype=np.float64)
    for niche_id in ordered_ids:
        niche_code = code_by_id[niche_id]
        selected = codes == niche_code
        indices = np.flatnonzero(selected)
        n_cells = int(len(indices))
        inside_sum = log_counts[selected].sum(axis=0, dtype=np.float64)
        rest_count = len(codes) - n_cells
        if rest_count > 0:
            delta = inside_sum / n_cells - (core_gene_sum - inside_sum) / rest_count
            gene_index = np.arange(len(delta), dtype=np.int64)
            marker_order = np.lexsort((gene_index, -delta))[:5]
            marker_summary: dict[str, Any] = {
                "status": "computed_descriptive_only",
                "scale": "within_niche_minus_rest_of_core_mean_log1p_raw_count",
                "genes": [
                    {
                        "gene": str(gene_names[position]),
                        "mean_difference": float(delta[position]),
                    }
                    for position in marker_order
                ],
            }
        else:
            marker_summary = {
                "status": "not_computed",
                "reason": "niche_contains_entire_core",
            }
        medians = np.nanmedian(measured_metadata[selected], axis=0)
        metadata_summary = {
            str(name): (None if not np.isfinite(value) else float(value))
            for name, value in zip(metadata_names, medians, strict=True)
        }
        if len(pairs):
            inside_edge = selected[pairs[:, 0]] & selected[pairs[:, 1]]
            boundary_edge = selected[pairs[:, 0]] ^ selected[pairs[:, 1]]
        else:
            inside_edge = np.zeros(0, dtype=np.bool_)
            boundary_edge = np.zeros(0, dtype=np.bool_)
        region = region_by_id.get(niche_id)
        if region is None:
            raise AttentionNichePipelineError(
                f"Dissolved region is absent for {niche_id}."
            )
        rows.append(
            {
                "core_number": int(core_number),
                "core_alias": alias,
                "niche_id": niche_id,
                "number_of_cells": n_cells,
                "physical_area_um2": float(region.area_um2),
                "micro_niche": bool(np.all(micro_niche[selected])),
                "median_hub_score": float(np.median(hub_scores[selected])),
                "maximum_hub_score": float(np.max(hub_scores[selected])),
                "median_internal_mutual_routing_score": (
                    float(np.median(scores[inside_edge]))
                    if bool(inside_edge.any())
                    else np.nan
                ),
                "boundary_edge_score": (
                    float(np.median(scores[boundary_edge]))
                    if bool(boundary_edge.any())
                    else np.nan
                ),
                "confidence": float(np.median(confidence[selected])),
                "descriptive_marker_gene_summary_json": _canonical_json(
                    marker_summary
                ),
                "descriptive_metadata_summary_json": _canonical_json(
                    metadata_summary
                ),
                "biological_niche_name_assigned": False,
            }
        )
    return pd.DataFrame.from_records(rows)


@dataclass(frozen=True, slots=True)
class CoreJobSpec:
    alias: str
    core_number: int
    local_device: int
    output_dir: Path
    cohort_dir: Path
    graph_dir: Path
    raw_dir: Path
    core_map_path: Path
    reconciliation_path: Path
    members: tuple[CheckpointMember, ...]
    analysis_mask_seed: int = 2026082501
    mask_views: int = 10
    leiden_seed: int = 2026082502
    color_seed: int = 2026082503
    uniform_routing_threshold: float = 1.0
    score_threshold: float = 1.0
    support_threshold: float = 0.60
    primary_top_k: int = 8
    primary_resolution: float = 1.0
    polygon_coordinate_alignment_rule: str = (
        "centroid_tolerance_or_polygon_covers_coordinate"
    )
    polygon_centroid_tolerance_um: float = 5.0
    max_spatial_gap_um: float = 75.0
    micro_niche_threshold: int = 20
    amp: bool = False
    deterministic_replay: bool = True


def _extract_routing_samples(
    *,
    spec: CoreJobSpec,
    batch: PooledRelativeQKVCoreBatch,
    edge_index: np.ndarray,
    indegree: np.ndarray,
    store_head_channels: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray] | None,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    # Import after process spawn so each child installs its own CUDA context.
    from .attention_routing_niches import make_analysis_mask_view

    seed_count = len(spec.members)
    edge_count = batch.n_edges
    routing = np.empty(
        (seed_count, spec.mask_views, edge_count), dtype=np.float32
    )
    visible = np.empty((seed_count, edge_count), dtype=np.float32)
    channel_sums: dict[str, np.ndarray] | None = None
    head_count: int | None = None
    mask_receipts: dict[str, dict[str, Any]] = {}
    checkpoint_receipts: list[dict[str, Any]] = []
    audit = {
        "view_count": 0,
        "all_visible_view_count": 0,
        "max_attention_sum_error": 0.0,
        "max_logit_composition_error": 0.0,
        "max_softmax_reconstruction_error": 0.0,
        "strict_attention_shard_count": 0,
        "strict_view_receipts": [],
        "total_streamed_edge_observations": 0,
        "layer_numbers": [],
    }
    replay_records: list[dict[str, Any]] = []
    set_deterministic_seed(
        spec.analysis_mask_seed,
        deterministic=True,
        warn_only=False,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{spec.local_device}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    for seed_position, member in enumerate(spec.members):
        loaded = load_relative_qkv_checkpoint(
            member.checkpoint_path,
            num_genes=batch.n_genes,
            node_covariate_dim=int(batch.node_covariates.shape[1]),
            device=device,
        )
        model = loaded.model
        model.eval()
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise AttentionNichePipelineError("Loaded inference model is not frozen/eval.")
        if loaded.checkpoint_sha256 != member.checkpoint_sha256:
            raise AttentionNichePipelineError("Checkpoint hash changed before extraction.")
        observed_heads = int(model.blocks[-1].attention_heads)
        if head_count is None:
            head_count = observed_heads
            if store_head_channels:
                channel_sums = {
                    name: np.zeros(
                        (seed_count, edge_count, observed_heads), dtype=np.float32
                    )
                    for name in (
                        "attention",
                        "content_qk",
                        "positional_bias",
                        "combined_logit",
                    )
                }
        elif observed_heads != head_count:
            raise AttentionNichePipelineError("Attention-head count differs across seeds.")
        checkpoint_receipts.append(
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "model_state_sha256": member.model_state_sha256,
                "completed_global_epochs": member.completed_global_epochs,
                "parameter_count": member.parameter_count,
                "eval_mode": True,
                "gradients_disabled": True,
                "attention_heads": observed_heads,
                "graph_layers": int(model.graph_layers),
            }
        )
        for view_index in range(spec.mask_views):
            mask_view = make_analysis_mask_view(
                n_cells=batch.n_nodes,
                n_genes=batch.n_genes,
                core_alias=spec.alias,
                mask_view_index=view_index,
                analysis_mask_seed=spec.analysis_mask_seed,
            )
            key = str(view_index)
            raw_mask_receipt = mask_view.to_receipt()
            current_receipt = {
                "view_index": int(view_index),
                "derived_seed": int(mask_view.derived_seed),
                "mask_realization_sha256": str(
                    raw_mask_receipt["mask_realization_sha256"]
                ),
                "receipt_sha256": str(mask_view.receipt_sha256),
                "masked_entry_count": int(mask_view.masked_gene_counts.sum()),
                "masked_count_min": int(mask_view.masked_gene_counts.min()),
                "masked_count_max": int(mask_view.masked_gene_counts.max()),
                "zero_mask_cells": int((mask_view.masked_gene_counts == 0).sum()),
                "full_mask_cells": int(
                    (mask_view.masked_gene_counts == batch.n_genes).sum()
                ),
            }
            previous = mask_receipts.setdefault(key, current_receipt)
            if previous != current_receipt:
                raise AttentionNichePipelineError(
                    "Analysis masks differ across model seeds."
                )
            channel_targets = (
                {
                    name: values[seed_position]
                    for name, values in channel_sums.items()
                }
                if channel_sums is not None
                else None
            )
            view_audit = _stream_one_view(
                model=model,
                batch=batch,
                mask=mask_view.mask,
                edge_index=edge_index,
                indegree=indegree,
                routing_target=routing[seed_position, view_index],
                channel_targets=channel_targets,
                amp=spec.amp,
            )
            audit["view_count"] += 1
            audit["total_streamed_edge_observations"] += edge_count
            audit["max_attention_sum_error"] = max(
                audit["max_attention_sum_error"],
                view_audit["max_attention_sum_error"],
            )
            audit["max_logit_composition_error"] = max(
                audit["max_logit_composition_error"],
                view_audit["max_logit_composition_error"],
            )
            audit["max_softmax_reconstruction_error"] = max(
                audit["max_softmax_reconstruction_error"],
                view_audit["max_softmax_reconstruction_error"],
            )
            audit["strict_attention_shard_count"] += view_audit[
                "strict_attention_shard_count"
            ]
            audit["strict_view_receipts"].append(
                {
                    "seed": int(member.seed),
                    "mask_view_index": int(view_index),
                    "all_genes_visible": False,
                    "sha256": view_audit[
                        "strict_attention_shard_receipts_sha256"
                    ],
                }
            )
            audit["layer_numbers"].append(view_audit["layer_number"])
            if spec.deterministic_replay and seed_position == 0:
                replay_mask = make_analysis_mask_view(
                    n_cells=batch.n_nodes,
                    n_genes=batch.n_genes,
                    core_alias=spec.alias,
                    mask_view_index=view_index,
                    analysis_mask_seed=spec.analysis_mask_seed,
                )
                replay_routing = np.empty(edge_count, dtype=np.float32)
                replay_audit = _stream_one_view(
                    model=model,
                    batch=batch,
                    mask=replay_mask.mask,
                    edge_index=edge_index,
                    indegree=indegree,
                    routing_target=replay_routing,
                    channel_targets=None,
                    amp=spec.amp,
                )
                if (
                    replay_mask.receipt_sha256 != mask_view.receipt_sha256
                    or not np.array_equal(
                        replay_routing, routing[seed_position, view_index]
                    )
                ):
                    raise AttentionNichePipelineError(
                        "Deterministic inference replay changed mask or routing values."
                    )
                replay_records.append(
                    {
                        "view_index": int(view_index),
                        "mask_receipt_sha256": replay_mask.receipt_sha256,
                        "routing_sha256": _array_sha256(
                            f"routing_replay_view_{view_index}", replay_routing
                        ),
                        "strict_attention_shard_receipts_sha256": replay_audit[
                            "strict_attention_shard_receipts_sha256"
                        ],
                        "exact_array_equal": True,
                        "max_attention_sum_error": replay_audit[
                            "max_attention_sum_error"
                        ],
                        "max_softmax_reconstruction_error": replay_audit[
                            "max_softmax_reconstruction_error"
                        ],
                    }
                )
        all_visible_mask = np.zeros(
            (batch.n_nodes, batch.n_genes), dtype=np.bool_
        )
        visible_audit = _stream_one_view(
            model=model,
            batch=batch,
            mask=all_visible_mask,
            edge_index=edge_index,
            indegree=indegree,
            routing_target=visible[seed_position],
            channel_targets=None,
            amp=spec.amp,
        )
        audit["all_visible_view_count"] += 1
        audit["total_streamed_edge_observations"] += edge_count
        audit["max_attention_sum_error"] = max(
            audit["max_attention_sum_error"],
            visible_audit["max_attention_sum_error"],
        )
        audit["max_logit_composition_error"] = max(
            audit["max_logit_composition_error"],
            visible_audit["max_logit_composition_error"],
        )
        audit["max_softmax_reconstruction_error"] = max(
            audit["max_softmax_reconstruction_error"],
            visible_audit["max_softmax_reconstruction_error"],
        )
        audit["strict_attention_shard_count"] += visible_audit[
            "strict_attention_shard_count"
        ]
        audit["strict_view_receipts"].append(
            {
                "seed": int(member.seed),
                "mask_view_index": None,
                "all_genes_visible": True,
                "sha256": visible_audit[
                    "strict_attention_shard_receipts_sha256"
                ],
            }
        )
        audit["layer_numbers"].append(visible_audit["layer_number"])
        del loaded, model
        torch.cuda.empty_cache()
    if channel_sums is not None:
        for values in channel_sums.values():
            values /= float(spec.mask_views)
    audit["peak_cuda_allocated_gib"] = float(
        torch.cuda.max_memory_allocated(device) / (1024**3)
    )
    audit["peak_cuda_reserved_gib"] = float(
        torch.cuda.max_memory_reserved(device) / (1024**3)
    )
    audit["routing_samples_sha256"] = _array_sha256("routing_samples", routing)
    audit["all_visible_routing_sha256"] = _array_sha256(
        "all_visible_routing", visible
    )
    audit["deterministic_inference_replay"] = {
        "seed": int(spec.members[0].seed),
        "view_count": len(replay_records),
        "all_ten_mask_views_replayed": len(replay_records) == spec.mask_views == 10,
        "exact_array_equal": bool(
            len(replay_records) == spec.mask_views
            and all(record["exact_array_equal"] for record in replay_records)
        ),
        "records": replay_records,
        "replay_records_sha256": _canonical_sha256(replay_records),
    }
    if len(set(audit["layer_numbers"])) != 1:
        raise AttentionNichePipelineError("Extraction layer drifted across views/seeds.")
    return (
        routing,
        visible,
        channel_sums,
        checkpoint_receipts,
        mask_receipts,
        audit,
    )


def _run_core_job(spec: CoreJobSpec) -> dict[str, Any]:
    """Run one complete core on one assigned CUDA device."""

    from .attention_niche_geometry import (
        construct_local_contiguity_graph,
        deterministic_niche_colors,
        dissolve_niche_regions,
        dissolved_regions_geojson,
        load_aligned_cosmx_polygons,
        niche_adjacency_from_cells,
        split_preliminary_niches,
        verify_niche_connectedness,
    )
    from .attention_routing_niches import (
        build_consensus_routing_partition,
        build_seed_specific_partitions,
        match_seed_partitions_to_consensus,
        mutual_routing_hub_scores,
        partition_similarity,
        retain_top_k_mutual_edges,
        summarize_mutual_routing_consensus,
        summarize_parameter_sensitivity,
        weighted_leiden_partition,
    )

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    core_dir = spec.output_dir
    core_dir.mkdir(parents=True, exist_ok=False)
    batch, coordinates, counts = _load_one_core_batch(
        alias=spec.alias,
        cohort_dir=spec.cohort_dir,
        graph_dir=spec.graph_dir,
    )
    edge_index = np.asarray(batch.edge_index.cpu().numpy(), dtype=np.int64)
    if edge_index.shape != (2, batch.n_edges):
        raise AttentionNichePipelineError("Prepared edge_index shape drifted.")
    if np.any(edge_index[0] == edge_index[1]):
        raise AttentionNichePipelineError("Prepared graph contains self edges.")
    indegree = np.bincount(edge_index[1], minlength=batch.n_nodes).astype(
        np.int64, copy=False
    )
    if np.any(indegree <= 0):
        raise AttentionNichePipelineError(
            f"At least one {spec.alias} receiver has zero incoming degree."
        )

    with (spec.cohort_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        cohort_manifest = json.load(handle)
    gene_names = tuple(cohort_manifest["features"]["gene_names"])
    identity, measured_metadata, identity_receipt = _load_and_verify_raw_core_identity(
        alias=spec.alias,
        cohort_dir=spec.cohort_dir,
        raw_dir=spec.raw_dir,
        core_map_path=spec.core_map_path,
        reconciliation_path=spec.reconciliation_path,
        prepared_coordinates=coordinates,
        prepared_counts=counts,
        prepared_covariates=np.asarray(batch.node_covariates.numpy()),
        expected_genes=gene_names,
    )
    if len(identity) != batch.n_nodes:
        raise AttentionNichePipelineError("Original identity row count drifted.")

    polygon_so1, polygon_so2 = _polygon_paths(spec.raw_dir)
    polygon_alignment = load_aligned_cosmx_polygons(
        {"SO_1": polygon_so1, "SO_2": polygon_so2},
        identity[["slide", "fov", "cell_ID"]],
        prepared_coordinates_um=coordinates,
        centroid_tolerance_um=spec.polygon_centroid_tolerance_um,
    )
    if (
        polygon_alignment.centroid_alignment is None
        or polygon_alignment.centroid_alignment.alignment_rule
        != spec.polygon_coordinate_alignment_rule
    ):
        raise AttentionNichePipelineError(
            "Polygon-coordinate alignment rule drifted from the locked analysis."
        )
    local_graph = construct_local_contiguity_graph(
        coordinates,
        polygons_um=polygon_alignment.polygons_um,
        max_gap_um=spec.max_spatial_gap_um,
        fallback_k=6,
    )

    routing, visible, channel_sums, checkpoint_receipts, mask_receipts, extraction_audit = (
        _extract_routing_samples(
            spec=spec,
            batch=batch,
            edge_index=edge_index,
            indegree=indegree,
            store_head_channels=True,
        )
    )
    if channel_sums is None:
        raise AttentionNichePipelineError("Per-head attention channels were not retained.")
    directed_path = core_dir / "directed_attention_edges.parquet"
    directed_receipt = _write_directed_core_table(
        directed_path,
        alias=spec.alias,
        core_number=spec.core_number,
        seeds=[member.seed for member in spec.members],
        edge_index=edge_index,
        coordinates=coordinates,
        indegree=indegree,
        routing_samples=routing,
        all_visible_routing=visible,
        channel_mask_means=channel_sums,
    )
    del channel_sums

    consensus = summarize_mutual_routing_consensus(
        edge_index,
        routing,
        n_nodes=batch.n_nodes,
        seed_ids=tuple(member.seed for member in spec.members),
        mask_view_ids=tuple(range(spec.mask_views)),
        uniform_threshold=spec.uniform_routing_threshold,
        require_all_edges_reciprocal=True,
    )
    primary = build_consensus_routing_partition(
        consensus,
        n_nodes=batch.n_nodes,
        top_k=spec.primary_top_k,
        score_threshold=spec.score_threshold,
        support_threshold=spec.support_threshold,
        resolution=spec.primary_resolution,
        random_seed=spec.leiden_seed,
        core_alias=spec.alias,
    )
    connected = split_preliminary_niches(
        spec.core_number,
        primary.leiden.labels,
        local_graph.adjacency,
        micro_niche_threshold=spec.micro_niche_threshold,
    )
    verify_niche_connectedness(connected.final_niche_ids, local_graph.adjacency)
    niche_adjacency = niche_adjacency_from_cells(
        connected.final_niche_ids, local_graph.adjacency
    )
    colors = deterministic_niche_colors(
        spec.core_number,
        niche_adjacency,
        color_seed=spec.color_seed,
    )
    color_by_cell = np.asarray(
        [colors[str(value)] for value in connected.final_niche_ids], dtype=object
    )
    regions = dissolve_niche_regions(
        polygon_alignment.polygons_um,
        connected.final_niche_ids,
    )

    # Match each seed-specific, spatially split partition to the equally split
    # consensus.  A single-model analysis instead estimates assignment
    # agreement over the ten individual mask views and leaves seed agreement
    # unavailable.
    consensus_connected_labels = _integer_partition(connected.final_niche_ids)
    seed_partitions = build_seed_specific_partitions(
        consensus,
        n_nodes=batch.n_nodes,
        top_k=spec.primary_top_k,
        score_threshold=spec.score_threshold,
        support_threshold=spec.support_threshold,
        resolution=spec.primary_resolution,
        random_seed=spec.leiden_seed,
        core_alias=spec.alias,
        require_ten_mask_views=True,
    )
    seed_connected_rows: list[np.ndarray] = []
    for seed_partition in seed_partitions:
        seed_connected = split_preliminary_niches(
            spec.core_number,
            seed_partition.graph_partition.leiden.labels,
            local_graph.adjacency,
            micro_niche_threshold=spec.micro_niche_threshold,
        )
        seed_connected_rows.append(_integer_partition(seed_connected.final_niche_ids))
    if len(spec.members) > 1:
        assignment_agreement = match_seed_partitions_to_consensus(
            consensus_connected_labels,
            np.stack(seed_connected_rows),
            seed_ids=tuple(member.seed for member in spec.members),
        )
        agreement_axis = "model_seed"
        seed_assignment_agreement = np.asarray(
            assignment_agreement.cell_assignment_agreement, dtype=np.float64
        )
        mask_assignment_agreement = np.full(batch.n_nodes, np.nan)
        agreement_for_confidence = seed_assignment_agreement
    else:
        mask_connected_rows: list[np.ndarray] = []
        for view_position in range(spec.mask_views):
            view_scores = consensus.mutual_samples[0, view_position]
            view_support = (
                view_scores > spec.uniform_routing_threshold
            ).astype(np.float64)
            view_graph = retain_top_k_mutual_edges(
                consensus.pairs.pair_cells,
                view_scores,
                view_support,
                n_nodes=batch.n_nodes,
                top_k=spec.primary_top_k,
                score_threshold=spec.score_threshold,
                support_threshold=spec.support_threshold,
            )
            view_leiden = weighted_leiden_partition(
                batch.n_nodes,
                view_graph.edge_pairs,
                view_graph.weights,
                resolution=spec.primary_resolution,
                random_seed=spec.leiden_seed,
                core_alias=spec.alias,
            )
            view_connected = split_preliminary_niches(
                spec.core_number,
                view_leiden.labels,
                local_graph.adjacency,
                micro_niche_threshold=spec.micro_niche_threshold,
            )
            mask_connected_rows.append(
                _integer_partition(view_connected.final_niche_ids)
            )
        assignment_agreement = match_seed_partitions_to_consensus(
            consensus_connected_labels,
            np.stack(mask_connected_rows),
            seed_ids=tuple(range(spec.mask_views)),
        )
        agreement_axis = "mask_view"
        seed_assignment_agreement = np.full(batch.n_nodes, np.nan)
        mask_assignment_agreement = np.asarray(
            assignment_agreement.cell_assignment_agreement, dtype=np.float64
        )
        agreement_for_confidence = mask_assignment_agreement

    retained = primary.retained_graph
    hub_scores = mutual_routing_hub_scores(
        batch.n_nodes, retained.edge_pairs, retained.weights
    )
    incident_support = _incident_median(
        batch.n_nodes,
        retained.edge_pairs,
        retained.support_fraction,
        default=0.0,
    )
    retained_positions = retained.pair_positions
    mask_spread_values = consensus.mask_view_standard_deviation_after_seed_median
    if mask_spread_values is None:
        incident_mask_spread = np.zeros(batch.n_nodes, dtype=np.float64)
    else:
        incident_mask_spread = _incident_median(
            batch.n_nodes,
            retained.edge_pairs,
            mask_spread_values[retained_positions],
            default=0.0,
        )
    seed_spread_values = consensus.seed_standard_deviation_after_view_median
    if seed_spread_values is None:
        incident_seed_spread = np.full(batch.n_nodes, np.nan)
        variation_reliability = 1.0 / (1.0 + incident_mask_spread)
    else:
        incident_seed_spread = _incident_median(
            batch.n_nodes,
            retained.edge_pairs,
            seed_spread_values[retained_positions],
            default=0.0,
        )
        variation_reliability = 1.0 / (
            1.0 + incident_seed_spread + incident_mask_spread
        )
    confidence = np.cbrt(
        np.clip(agreement_for_confidence, 0.0, 1.0)
        * np.clip(incident_support, 0.0, 1.0)
        * np.clip(variation_reliability, 0.0, 1.0)
    )

    # Locked parameter sensitivity uses the same spatial split and never
    # substitutes a visually preferred setting for top-8/resolution-1.0.
    sensitivity_partitions: dict[tuple[int, float], np.ndarray] = {}
    sensitivity_details: list[dict[str, Any]] = []
    for top_k in (5, 8, 10):
        top_graph = retain_top_k_mutual_edges(
            consensus.pairs.pair_cells,
            consensus.consensus_mutual_score,
            consensus.support_fraction,
            n_nodes=batch.n_nodes,
            top_k=top_k,
            score_threshold=spec.score_threshold,
            support_threshold=spec.support_threshold,
        )
        for resolution in (0.5, 1.0, 1.5):
            if top_k == spec.primary_top_k and resolution == spec.primary_resolution:
                preliminary = primary.leiden
                final_labels = consensus_connected_labels
                final_niche_count = connected.niche_count
            else:
                preliminary = weighted_leiden_partition(
                    batch.n_nodes,
                    top_graph.edge_pairs,
                    top_graph.weights,
                    resolution=resolution,
                    random_seed=spec.leiden_seed,
                    core_alias=spec.alias,
                )
                sensitivity_connected = split_preliminary_niches(
                    spec.core_number,
                    preliminary.labels,
                    local_graph.adjacency,
                    micro_niche_threshold=spec.micro_niche_threshold,
                )
                final_labels = _integer_partition(
                    sensitivity_connected.final_niche_ids
                )
                final_niche_count = sensitivity_connected.niche_count
            sensitivity_partitions[(top_k, resolution)] = final_labels
            sensitivity_details.append(
                {
                    "core_number": spec.core_number,
                    "core_alias": spec.alias,
                    "top_k": top_k,
                    "resolution": resolution,
                    "retained_edge_count": top_graph.edge_count,
                    "preliminary_community_count": preliminary.community_count,
                    "final_connected_niche_count": final_niche_count,
                }
            )
    sensitivity_comparisons = summarize_parameter_sensitivity(
        sensitivity_partitions,
        primary_top_k=spec.primary_top_k,
        primary_resolution=spec.primary_resolution,
    )
    comparison_by_key = {
        (row.top_k, row.resolution): row for row in sensitivity_comparisons
    }
    for record in sensitivity_details:
        comparison = comparison_by_key[(record["top_k"], record["resolution"])]
        record.update(asdict(comparison))

    visible_consensus = summarize_mutual_routing_consensus(
        edge_index,
        visible[:, None, :],
        n_nodes=batch.n_nodes,
        seed_ids=tuple(member.seed for member in spec.members),
        mask_view_ids=(-1,),
        uniform_threshold=spec.uniform_routing_threshold,
        require_all_edges_reciprocal=True,
    )
    visible_partition = build_consensus_routing_partition(
        visible_consensus,
        n_nodes=batch.n_nodes,
        top_k=spec.primary_top_k,
        score_threshold=spec.score_threshold,
        support_threshold=spec.support_threshold,
        resolution=spec.primary_resolution,
        random_seed=spec.leiden_seed,
        core_alias=spec.alias,
    )
    visible_connected = split_preliminary_niches(
        spec.core_number,
        visible_partition.leiden.labels,
        local_graph.adjacency,
        micro_niche_threshold=spec.micro_niche_threshold,
    )
    visible_similarity = partition_similarity(
        consensus_connected_labels,
        _integer_partition(visible_connected.final_niche_ids),
    )

    # Recompute the deterministic downstream primary partition independently
    # from the immutable routing samples and compare exact identities.
    replay_primary = build_consensus_routing_partition(
        consensus,
        n_nodes=batch.n_nodes,
        top_k=spec.primary_top_k,
        score_threshold=spec.score_threshold,
        support_threshold=spec.support_threshold,
        resolution=spec.primary_resolution,
        random_seed=spec.leiden_seed,
        core_alias=spec.alias,
    )
    replay_connected = split_preliminary_niches(
        spec.core_number,
        replay_primary.leiden.labels,
        local_graph.adjacency,
        micro_niche_threshold=spec.micro_niche_threshold,
    )
    replay_colors = deterministic_niche_colors(
        spec.core_number,
        niche_adjacency_from_cells(
            replay_connected.final_niche_ids, local_graph.adjacency
        ),
        color_seed=spec.color_seed,
    )
    if (
        not np.array_equal(
            replay_primary.retained_graph.edge_pairs, retained.edge_pairs
        )
        or not np.array_equal(replay_primary.leiden.labels, primary.leiden.labels)
        or not np.array_equal(
            replay_connected.final_niche_ids, connected.final_niche_ids
        )
        or replay_colors != colors
    ):
        raise AttentionNichePipelineError(
            "Deterministic downstream replay changed graph, assignments, or colors."
        )

    niche_by_cell_agreement = np.empty(batch.n_nodes, dtype=np.float64)
    for niche_code, niche_agreement in zip(
        assignment_agreement.consensus_community_ids,
        assignment_agreement.niche_assignment_agreement,
        strict=True,
    ):
        niche_by_cell_agreement[consensus_connected_labels == niche_code] = float(
            niche_agreement
        )
    map_label = (
        "single-model, mask-consensus map"
        if len(spec.members) == 1
        else f"{len(spec.members)}-model ensemble-consensus map"
    )
    assignments = pd.DataFrame(
        {
            "core_number": np.full(batch.n_nodes, spec.core_number, dtype=np.int16),
            "core_alias": np.full(batch.n_nodes, spec.alias),
            "cell_index": np.arange(batch.n_nodes, dtype=np.int64),
            "source_qc_passed": identity["source_qc_passed"].to_numpy(dtype=bool),
            "x_um": coordinates[:, 0],
            "y_um": coordinates[:, 1],
            "coordinate_unit": np.full(batch.n_nodes, "micrometres"),
            "preliminary_leiden_community": primary.leiden.labels,
            "final_connected_niche_id": connected.final_niche_ids,
            "final_niche_id": connected.final_niche_ids,
            "micro_niche": connected.micro_niche,
            "niche_color": color_by_cell,
            "mutual_routing_hub_score": hub_scores,
            "S_i": hub_scores,
            "assignment_confidence": confidence,
            "assignment_agreement_axis": np.full(batch.n_nodes, agreement_axis),
            "seed_assignment_agreement": seed_assignment_agreement,
            "mask_view_assignment_agreement": mask_assignment_agreement,
            "niche_assignment_agreement": niche_by_cell_agreement,
            "incident_retained_edge_support_median": incident_support,
            "incident_seed_spread_median": incident_seed_spread,
            "incident_mask_view_spread_median": incident_mask_spread,
            "variation_reliability": variation_reliability,
            "model_seeds_used": np.full(
                batch.n_nodes,
                _canonical_json([member.seed for member in spec.members]),
            ),
            "analysis_mask_views_used": np.full(
                batch.n_nodes, _canonical_json(list(range(spec.mask_views)))
            ),
            "map_label": np.full(batch.n_nodes, map_label),
        }
    )
    _validate_interpretation_identifier_minimization(assignments)
    assignment_path = core_dir / "cell_attention_niche_assignments.parquet"
    _write_dataframe_parquet_atomic(assignment_path, assignments)

    mutual_path = core_dir / "mutual_attention_edges.parquet"
    mutual_receipt = _write_mutual_core_table(
        mutual_path,
        alias=spec.alias,
        core_number=spec.core_number,
        coordinates=coordinates,
        consensus=consensus,
        retained_graph=retained,
    )
    summaries = _descriptive_niche_summaries(
        alias=spec.alias,
        core_number=spec.core_number,
        final_niche_ids=connected.final_niche_ids,
        micro_niche=connected.micro_niche,
        hub_scores=hub_scores,
        confidence=confidence,
        counts=counts,
        gene_names=gene_names,
        measured_metadata=measured_metadata,
        metadata_names=ALLOWED_METADATA_COLUMNS,
        retained_pairs=retained.edge_pairs,
        retained_scores=retained.weights,
        regions=regions,
    )
    summary_path = core_dir / "attention_niche_summary.csv"
    _write_dataframe_csv_atomic(summary_path, summaries)
    sensitivity_path = core_dir / "attention_niche_parameter_sensitivity.csv"
    _write_dataframe_csv_atomic(
        sensitivity_path, pd.DataFrame.from_records(sensitivity_details)
    )

    properties_by_niche = {
        str(row.niche_id): {
            "micro_niche": bool(row.micro_niche),
            "confidence": float(row.confidence),
        }
        for row in summaries.itertuples()
    }
    geojson = dissolved_regions_geojson(
        regions,
        core_number=spec.core_number,
        colors=colors,
        properties_by_niche=properties_by_niche,
    )
    geometry_validation = geojson.get("geometry_validation")
    if not isinstance(geometry_validation, Mapping) or (
        int(geometry_validation.get("feature_count", -1)) != len(regions)
        or int(geometry_validation.get("invalid_after_repair", -1)) != 0
        or float(geometry_validation.get("maximum_reported_area_error_um2", math.inf))
        > 1e-4
    ):
        raise AttentionNichePipelineError(
            f"Serialized region geometry validation failed for {spec.alias}."
        )
    geojson_path = core_dir / "attention_niche_regions.geojson"
    _write_json_atomic(geojson_path, geojson)
    colors_path = core_dir / "attention_niche_colors.json"
    _write_json_atomic(colors_path, colors)

    import resource

    peak_host_rss_gib = float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
    )
    elapsed_seconds = time.monotonic() - started_monotonic
    receipt = {
        "schema": "six_core_attention_niche_core_receipt_v1",
        "status": "complete",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "core_number": spec.core_number,
        "core_alias": spec.alias,
        "local_cuda_device": spec.local_device,
        "peak_host_rss_gib": peak_host_rss_gib,
        "map_label": map_label,
        "cell_count": batch.n_nodes,
        "directed_edge_count": batch.n_edges,
        "reciprocal_pair_count": consensus.pairs.pair_count,
        "eligible_mutual_pair_count": retained.eligible_pair_count,
        "retained_mutual_edge_count": retained.edge_count,
        "preliminary_community_count": primary.leiden.community_count,
        "final_connected_niche_count": connected.niche_count,
        "micro_niche_count": int(summaries["micro_niche"].sum()),
        "serialized_region_geometry_validation": geometry_validation,
        "confidence": {
            "minimum": float(np.min(confidence)),
            "median": float(np.median(confidence)),
            "mean": float(np.mean(confidence)),
            "maximum": float(np.max(confidence)),
            "low_below_0_60_count": int((confidence < 0.60).sum()),
            "agreement_axis": agreement_axis,
        },
        "identity_and_preprocessing": identity_receipt,
        "polygon_alignment": {
            "cell_count": polygon_alignment.n_cells,
            "source_rows_scanned": polygon_alignment.source_rows_scanned,
            "selected_vertex_rows": polygon_alignment.selected_vertex_rows,
            "repaired_polygon_count": int(polygon_alignment.repaired.sum()),
            "centroid_median_distance_um": (
                None
                if polygon_alignment.centroid_alignment is None
                else polygon_alignment.centroid_alignment.median_distance_um
            ),
            "centroid_maximum_distance_um": (
                None
                if polygon_alignment.centroid_alignment is None
                else polygon_alignment.centroid_alignment.maximum_distance_um
            ),
            "alignment_rule": (
                None
                if polygon_alignment.centroid_alignment is None
                else polygon_alignment.centroid_alignment.alignment_rule
            ),
            "centroid_tolerance_um": (
                None
                if polygon_alignment.centroid_alignment is None
                else polygon_alignment.centroid_alignment.tolerance_um
            ),
            "centroid_tolerance_exceeded_count": (
                None
                if polygon_alignment.centroid_alignment is None
                else len(
                    polygon_alignment.centroid_alignment.centroid_tolerance_exceeded_indices
                )
            ),
            "containment_accepted_count": (
                None
                if polygon_alignment.centroid_alignment is None
                else len(
                    polygon_alignment.centroid_alignment.containment_accepted_indices
                )
            ),
            "coordinate_outside_polygon_count": (
                None
                if polygon_alignment.centroid_alignment is None
                else len(
                    polygon_alignment.centroid_alignment.coordinate_outside_polygon_indices
                )
            ),
            "alignment_mismatch_count": (
                None
                if polygon_alignment.centroid_alignment is None
                else len(polygon_alignment.centroid_alignment.mismatch_indices)
            ),
        },
        "local_spatial_contiguity": {
            "method": local_graph.method,
            "max_gap_um": local_graph.max_gap_um,
            "edge_count": local_graph.edge_count,
            "nonisolated_fraction": local_graph.nonisolated_fraction,
            "polygon_nonisolated_fraction": local_graph.polygon_nonisolated_fraction,
            "polygon_edge_count": local_graph.polygon_edge_count,
            "fallback_reason": local_graph.fallback_reason,
            "every_final_niche_connected": True,
        },
        "checkpoint_receipts": checkpoint_receipts,
        "mask_receipts": mask_receipts,
        "extraction_audit": extraction_audit,
        "consensus_receipt_sha256": consensus.receipt_sha256,
        "retained_graph_receipt_sha256": retained.receipt_sha256,
        "leiden_receipt": dict(primary.leiden.receipt),
        "assignment_agreement_receipt_sha256": (
            assignment_agreement.receipt_sha256
        ),
        "all_genes_visible_sensitivity": {
            "retained_mutual_edge_count": visible_partition.retained_graph.edge_count,
            "preliminary_community_count": visible_partition.leiden.community_count,
            "final_connected_niche_count": visible_connected.niche_count,
            "routing_samples_sha256": extraction_audit[
                "all_visible_routing_sha256"
            ],
            "consensus_receipt_sha256": visible_consensus.receipt_sha256,
            "retained_graph_receipt_sha256": (
                visible_partition.retained_graph.receipt_sha256
            ),
            "assignment_sha256": _array_sha256(
                "all_visible_final_connected_niche_ids",
                np.asarray(visible_connected.final_niche_ids, dtype="U16"),
            ),
            **asdict(visible_similarity),
        },
        "deterministic_replay": {
            "inference_view_replayed": bool(spec.deterministic_replay),
            "downstream_assignment_exact": True,
            "assignment_sha256": _array_sha256(
                "final_connected_niche_ids",
                np.asarray(connected.final_niche_ids, dtype="U16"),
            ),
            "color_mapping_sha256": _canonical_sha256(colors),
        },
        "directed_table": directed_receipt,
        "mutual_table": mutual_receipt,
        "outputs": {
            "assignments": str(assignment_path),
            "mutual_edges": str(mutual_path),
            "directed_edges": str(directed_path),
            "summary": str(summary_path),
            "sensitivity": str(sensitivity_path),
            "regions": str(geojson_path),
            "colors": str(colors_path),
        },
    }
    receipt_path = core_dir / "core_analysis_receipt.json"
    _write_json_atomic(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    del routing, visible
    return receipt


def _combine_parquet_files(
    sources: Sequence[Path],
    destination: Path,
    *,
    source_contracts: Sequence[Mapping[str, Any]] | None = None,
    remove_sources_after_write: bool = False,
    staging_root: Path | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite combined table: {destination}")
    if source_contracts is not None and len(source_contracts) != len(sources):
        raise AttentionNichePipelineError(
            "Parquet source contracts do not align with source files."
        )
    if remove_sources_after_write and staging_root is None:
        raise AttentionNichePipelineError(
            "Removing consolidated shards requires an explicit staging root."
        )
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    row_count = 0
    removed_source_bytes = 0
    removed_source_count = 0
    try:
        for source_position, source in enumerate(sources):
            contract = (
                None
                if source_contracts is None
                else source_contracts[source_position]
            )
            parquet_file = pq.ParquetFile(source)
            source_row_count = 0
            try:
                for batch in parquet_file.iter_batches(batch_size=100_000):
                    table = pa.Table.from_batches([batch])
                    if contract is not None:
                        names = set(table.column_names)
                        position_column = str(contract["position_column"])
                        required = {"core_number", "core_alias", position_column}
                        if not required.issubset(names):
                            raise AttentionNichePipelineError(
                                f"Core shard {source} lacks contract columns."
                            )
                        core_numbers = table["core_number"].to_numpy(
                            zero_copy_only=False
                        )
                        core_aliases = table["core_alias"].to_pylist()
                        positions = table[position_column].to_numpy(
                            zero_copy_only=False
                        )
                        expected_positions = np.arange(
                            source_row_count,
                            source_row_count + len(table),
                            dtype=np.int64,
                        )
                        if (
                            not np.all(
                                core_numbers == int(contract["core_number"])
                            )
                            or any(
                                str(value) != str(contract["core_alias"])
                                for value in core_aliases
                            )
                            or not np.array_equal(
                                np.asarray(positions, dtype=np.int64),
                                expected_positions,
                            )
                        ):
                            raise AttentionNichePipelineError(
                                f"Core shard identity/order contract failed: {source}."
                            )
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            destination,
                            schema,
                            compression="zstd",
                            compression_level=6,
                            write_statistics=True,
                        )
                    elif table.schema != schema:
                        raise AttentionNichePipelineError(
                            f"Parquet schema differs in {source}."
                        )
                    writer.write_table(table, row_group_size=len(table))
                    source_row_count += len(table)
                    row_count += len(table)
            finally:
                parquet_file.close()
            if contract is not None and source_row_count != int(
                contract["row_count"]
            ):
                raise AttentionNichePipelineError(
                    f"Core shard row coverage failed: {source}."
                )
            if remove_sources_after_write:
                assert staging_root is not None
                resolved_source = source.resolve()
                resolved_staging = staging_root.resolve()
                if (
                    source.is_symlink()
                    or resolved_source.parent.parent != resolved_staging
                    or resolved_source.suffix != ".parquet"
                ):
                    raise AttentionNichePipelineError(
                        f"Refusing to remove unexpected core shard: {source}."
                    )
                size_bytes = source.stat().st_size
                source.unlink()
                removed_source_bytes += int(size_bytes)
                removed_source_count += 1
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise AttentionNichePipelineError("No Parquet rows were combined.")
    return {
        "path": str(destination),
        "row_count": row_count,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "source_contracts_verified": source_contracts is not None,
        "source_shards_removed_after_streaming": removed_source_count,
        "source_shard_bytes_removed_after_streaming": removed_source_bytes,
    }


def _run_device_job_group(specs: Sequence[CoreJobSpec]) -> list[dict[str, Any]]:
    """Run a deterministic sequence of cores in one spawned GPU process."""

    return [_run_core_job(spec) for spec in specs]


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _portable_file_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index immutable-input receipts by their recorded portable path."""

    portable: dict[str, dict[str, Any]] = {}
    for receipt in receipts.values():
        record = dict(receipt)
        path = str(record.get("path", "")).strip()
        if not path:
            raise AttentionNichePipelineError(
                "An immutable-input receipt has no portable path."
            )
        if path in portable:
            raise AttentionNichePipelineError(
                f"Immutable-input receipt path is duplicated: {path}."
            )
        portable[path] = record
    return portable


def _portable_core_receipts(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove scratch-only paths while retaining per-core evidence.

    Per-core Parquet and geometry files are temporary staging products.  Their
    row counts and checksums remain in this receipt, while canonical paths are
    expressed as bundle-relative outputs plus an explicit core filter.
    """

    portable = json.loads(_canonical_json(list(receipts)))
    for receipt in portable:
        core_number = int(receipt["core_number"])
        transient_outputs = receipt.pop("outputs", {})
        receipt.pop("receipt_path", None)
        for table_key in ("directed_table", "mutual_table"):
            table_receipt = receipt.get(table_key)
            if isinstance(table_receipt, dict):
                table_receipt.pop("path", None)
                table_receipt["temporary_core_shard_retained"] = False
        receipt["temporary_core_outputs"] = {
            "retained": False,
            "files_consolidated": sorted(str(key) for key in transient_outputs),
        }
        receipt["canonical_outputs"] = {
            "assignments": {
                "path": "cell_attention_niche_assignments.parquet",
                "filter": f"core_number == {core_number}",
            },
            "mutual_edges": {
                "path": "mutual_attention_edges.parquet",
                "filter": f"core_number == {core_number}",
            },
            "directed_edges": {
                "path": "directed_attention_edges.parquet",
                "filter": f"core_number == {core_number}",
            },
            "summary": {
                "path": "attention_niche_summary.csv",
                "filter": f"core_number == {core_number}",
            },
            "sensitivity": {
                "path": "attention_niche_parameter_sensitivity.csv",
                "filter": f"core_number == {core_number}",
            },
            "regions": {
                "path": "attention_niche_regions.geojson",
                "filter": f"properties.core_number == {core_number}",
            },
            "colors": {"path": "attention_niche_colors.json"},
        }
    return portable


def _remove_verified_core_work_tree(
    work_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Remove only the known analysis-owned staging tree after consolidation."""

    expected = (
        output_root / "diagnostics" / "attention_niche_core_work"
    ).resolve()
    observed = work_root.resolve()
    if observed != expected or work_root.is_symlink() or not work_root.is_dir():
        raise AttentionNichePipelineError(
            "Refusing to remove an unexpected attention-niche staging path."
        )
    files = [path for path in work_root.rglob("*") if path.is_file()]
    size_bytes = int(sum(path.stat().st_size for path in files))
    file_count = len(files)
    shutil.rmtree(work_root)
    diagnostics = output_root / "diagnostics"
    if diagnostics.is_dir() and not any(diagnostics.iterdir()):
        diagnostics.rmdir()
    return {
        "removed_after_verified_consolidation": True,
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _filesystem_snapshot(path: Path, *, stage: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "stage": stage,
        "total_gib": float(usage.total / (1024**3)),
        "used_gib": float(usage.used / (1024**3)),
        "free_gib": float(usage.free / (1024**3)),
    }


def _required_analysis_free_disk_gib(
    *,
    model_seed_count: int,
    minimum_free_disk_gib: float,
) -> dict[str, float]:
    """Conservative disk estimate for streamed core-shard consolidation."""

    seed_count = int(model_seed_count)
    if seed_count <= 0:
        raise AttentionNichePipelineError("At least one model seed is required.")
    minimum = float(minimum_free_disk_gib)
    if not math.isfinite(minimum) or minimum <= 0:
        raise AttentionNichePipelineError("Minimum free disk must be positive.")
    head_count = 8
    channel_count = 4
    mask_view_count = 10
    # Directed rows retain scalar identities/statistics, exact SxR head-mean
    # routing, all-visible per-seed routing, and per-seed/per-head mask means
    # for all four diagnostic channels. Mutual rows retain exact pair identity,
    # direction, support, axis summaries, and per-seed/per-view medians.
    directed_bytes_per_edge = (
        96
        + seed_count * mask_view_count * 4
        + seed_count * 4
        + seed_count * channel_count * head_count * 4
    )
    mutual_bytes_per_pair = 144 + seed_count * 4 + mask_view_count * 4
    raw_table_gib = (
        EXPECTED_TOTAL_DIRECTED_EDGES * directed_bytes_per_edge
        + (EXPECTED_TOTAL_DIRECTED_EDGES // 2) * mutual_bytes_per_pair
    ) / (1024**3)
    # Core shards are removed immediately after their verified batches are
    # appended, so peak is one compressed representation plus plotting/region
    # outputs and a generous filesystem/codec margin rather than two complete
    # table copies.
    estimated_peak_gib = raw_table_gib * 1.25 + 12.0
    required = max(minimum, estimated_peak_gib)
    return {
        "minimum_configured_gib": minimum,
        "estimated_raw_table_gib": float(raw_table_gib),
        "estimated_peak_with_margin_gib": float(estimated_peak_gib),
        "required_free_gib": float(required),
    }


def _checkpoint_post_receipts(
    members: Sequence[CheckpointMember], *, root: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for member in members:
        observed = sha256_file(member.checkpoint_path)
        marker = _load_json(
            member.bundle_path / "_SUCCESS",
            f"post-analysis success marker for {member.run_id}",
        )
        unchanged = (
            observed == member.checkpoint_sha256
            and marker.get("content_sha256")
            == member.success_marker_content_sha256
        )
        records.append(
            {
                "seed": member.seed,
                "run_id": member.run_id,
                "path": _relative_to_root(member.checkpoint_path, root),
                "pre_sha256": member.checkpoint_sha256,
                "post_sha256": observed,
                "success_marker_content_sha256": marker.get("content_sha256"),
                "unchanged": unchanged,
            }
        )
    if not all(record["unchanged"] for record in records):
        raise AttentionNichePipelineError(
            "At least one trained checkpoint or success marker changed during analysis."
        )
    return {"unchanged": True, "members": records}


def _markdown_core_table(core_receipts: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Core | Cells | Directed edges | Retained mutual edges | Preliminary communities | Final connected niches | Median confidence |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for receipt in sorted(core_receipts, key=lambda value: int(value["core_number"])):
        lines.append(
            "| {core} | {cells:,} | {directed:,} | {retained:,} | {preliminary:,} | {final:,} | {confidence:.4f} |".format(
                core=int(receipt["core_number"]),
                cells=int(receipt["cell_count"]),
                directed=int(receipt["directed_edge_count"]),
                retained=int(receipt["retained_mutual_edge_count"]),
                preliminary=int(receipt["preliminary_community_count"]),
                final=int(receipt["final_connected_niche_count"]),
                confidence=float(receipt["confidence"]["median"]),
            )
        )
    return "\n".join(lines)


def _verified_recovery_core_receipts(
    source: VerifiedRenderRecoverySource,
    *,
    members: Sequence[CheckpointMember],
    mask_views: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify all six completed extraction/QC receipts from the failed source."""

    receipt_root = source.root / "diagnostics" / "attention_niche_core_work"
    receipt_paths = sorted(receipt_root.glob("core_*/core_analysis_receipt.json"))
    expected_paths = [
        receipt_root / f"core_{core:02d}" / "core_analysis_receipt.json"
        for core in EXPECTED_CORE_NUMBERS
    ]
    if receipt_paths != expected_paths:
        raise AttentionNichePipelineError(
            "Recovery source does not contain exactly the six expected core receipts."
        )

    expected_checkpoint_records = [
        {
            "seed": member.seed,
            "run_id": member.run_id,
            "completed_global_epochs": member.completed_global_epochs,
            "checkpoint_sha256": member.checkpoint_sha256,
            "model_state_sha256": member.model_state_sha256,
            "parameter_count": member.parameter_count,
        }
        for member in members
    ]
    receipts: list[dict[str, Any]] = []
    maximum_attention_error = 0.0
    maximum_softmax_error = 0.0
    strict_shard_count = 0
    total_cells = 0
    total_directed = 0
    total_reciprocal = 0
    for alias, core_number, receipt_path in zip(
        EXPECTED_ALIASES,
        EXPECTED_CORE_NUMBERS,
        receipt_paths,
        strict=True,
    ):
        receipt = _load_json(receipt_path, f"Core {core_number} source receipt")
        if (
            receipt.get("schema") != "six_core_attention_niche_core_receipt_v1"
            or receipt.get("status") != "complete"
            or receipt.get("core_alias") != alias
            or int(receipt.get("core_number", -1)) != core_number
            or receipt.get("map_label")
            != f"{len(members)}-model ensemble-consensus map"
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} recovery receipt identity is invalid."
            )

        checkpoint_receipts = receipt.get("checkpoint_receipts")
        if not isinstance(checkpoint_receipts, list):
            raise AttentionNichePipelineError(
                f"Core {core_number} checkpoint receipts are absent."
            )
        observed_checkpoint_records: list[dict[str, Any]] = []
        for checkpoint in checkpoint_receipts:
            if (
                not isinstance(checkpoint, Mapping)
                or checkpoint.get("eval_mode") is not True
                or checkpoint.get("gradients_disabled") is not True
                or int(checkpoint.get("attention_heads", -1)) != 8
                or int(checkpoint.get("graph_layers", -1)) != 4
            ):
                raise AttentionNichePipelineError(
                    f"Core {core_number} checkpoint execution receipt failed."
                )
            observed_checkpoint_records.append(
                {
                    "seed": int(checkpoint["seed"]),
                    "run_id": str(checkpoint["run_id"]),
                    "completed_global_epochs": int(
                        checkpoint["completed_global_epochs"]
                    ),
                    "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
                    "model_state_sha256": str(checkpoint["model_state_sha256"]),
                    "parameter_count": int(checkpoint["parameter_count"]),
                }
            )
        if observed_checkpoint_records != expected_checkpoint_records:
            raise AttentionNichePipelineError(
                f"Core {core_number} checkpoint receipts do not match discovery."
            )

        identity = receipt.get("identity_and_preprocessing")
        if (
            not isinstance(identity, Mapping)
            or identity.get("alias") != alias
            or int(identity.get("core_number", -1)) != core_number
            or identity.get("gene_order_sha256") != EXPECTED_GENE_SCHEMA_SHA256
            or identity.get("target_expression_transform_reproduced") is not True
            or identity.get("metadata_transform_reproduced") is not True
            or identity.get("raw_counts_aligned") is not True
            or identity.get("coordinates_um_aligned") is not True
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} preprocessing identity receipt failed."
            )

        masks = receipt.get("mask_receipts")
        if (
            not isinstance(masks, Mapping)
            or set(masks) != {str(index) for index in range(mask_views)}
            or any(
                not isinstance(mask, Mapping)
                or int(mask.get("view_index", -1)) != index
                or len(str(mask.get("mask_realization_sha256", ""))) != 64
                or len(str(mask.get("receipt_sha256", ""))) != 64
                for index, mask in (
                    (index, masks[str(index)]) for index in range(mask_views)
                )
            )
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} ten-view mask receipts failed."
            )

        extraction = receipt.get("extraction_audit")
        deterministic = receipt.get("deterministic_replay")
        contiguity = receipt.get("local_spatial_contiguity")
        geometry = receipt.get("serialized_region_geometry_validation")
        if not all(
            isinstance(value, Mapping)
            for value in (extraction, deterministic, contiguity, geometry)
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} source QC mappings are absent."
            )
        assert isinstance(extraction, Mapping)
        assert isinstance(deterministic, Mapping)
        assert isinstance(contiguity, Mapping)
        assert isinstance(geometry, Mapping)
        inference_replay = extraction.get("deterministic_inference_replay")
        strict_views = extraction.get("strict_view_receipts")
        layer_numbers = extraction.get("layer_numbers")
        expected_view_keys = {
            (member.seed, False, view)
            for member in members
            for view in range(mask_views)
        } | {(member.seed, True, None) for member in members}
        observed_view_keys = {
            (
                int(view["seed"]),
                bool(view["all_genes_visible"]),
                None
                if view.get("mask_view_index") is None
                else int(view["mask_view_index"]),
            )
            for view in strict_views
            if isinstance(view, Mapping)
        } if isinstance(strict_views, list) else set()
        if (
            float(extraction.get("max_attention_sum_error", math.inf)) > 2e-6
            or float(
                extraction.get("max_softmax_reconstruction_error", math.inf)
            )
            > 2e-6
            or float(extraction.get("max_logit_composition_error", math.inf))
            > 2e-6
            or int(extraction.get("strict_attention_shard_count", 0)) <= 0
            or not isinstance(layer_numbers, list)
            or not layer_numbers
            or set(int(value) for value in layer_numbers) != {3}
            or len(layer_numbers) != len(members) * (mask_views + 1)
            or not isinstance(strict_views, list)
            or len(strict_views) != len(members) * (mask_views + 1)
            or observed_view_keys != expected_view_keys
            or not isinstance(inference_replay, Mapping)
            or inference_replay.get("exact_array_equal") is not True
            or inference_replay.get("all_ten_mask_views_replayed") is not True
            or deterministic.get("inference_view_replayed") is not True
            or deterministic.get("downstream_assignment_exact") is not True
            or contiguity.get("every_final_niche_connected") is not True
            or int(geometry.get("invalid_after_repair", -1)) != 0
            or float(geometry.get("maximum_reported_area_error_um2", math.inf))
            > 1e-4
            or int(geometry.get("feature_count", -1))
            != int(receipt.get("final_connected_niche_count", -2))
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} source extraction/spatial QC failed."
            )

        cell_count = int(receipt.get("cell_count", -1))
        directed_count = int(receipt.get("directed_edge_count", -1))
        reciprocal_count = int(receipt.get("reciprocal_pair_count", -1))
        retained_count = int(receipt.get("retained_mutual_edge_count", -1))
        directed_table = receipt.get("directed_table")
        mutual_table = receipt.get("mutual_table")
        if (
            cell_count <= 0
            or directed_count <= 0
            or reciprocal_count * 2 != directed_count
            or retained_count <= 0
            or not isinstance(directed_table, Mapping)
            or int(directed_table.get("row_count", -1)) != directed_count
            or len(str(directed_table.get("sha256", ""))) != 64
            or not isinstance(mutual_table, Mapping)
            or int(mutual_table.get("row_count", -1)) != reciprocal_count
            or int(mutual_table.get("retained_row_count", -1)) != retained_count
            or len(str(mutual_table.get("sha256", ""))) != 64
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} table/count receipts failed."
            )

        core_dir = receipt_path.parent
        for basename in (
            "attention_niche_summary.csv",
            "attention_niche_parameter_sensitivity.csv",
            "attention_niche_colors.json",
            "attention_niche_regions.geojson",
        ):
            small_output = core_dir / basename
            relative = small_output.relative_to(source.root).as_posix()
            expected = source.artifact_files.get(relative)
            if (
                small_output.is_symlink()
                or not small_output.is_file()
                or not isinstance(expected, Mapping)
                or expected.get("type") != "file"
            ):
                raise AttentionNichePipelineError(
                    f"Core {core_number} retained diagnostic is not bundle-bound."
                )
        receipt_relative = receipt_path.relative_to(source.root).as_posix()
        expected_receipt = source.artifact_files.get(receipt_relative)
        if (
            not isinstance(expected_receipt, Mapping)
            or expected_receipt.get("sha256") != sha256_file(receipt_path)
        ):
            raise AttentionNichePipelineError(
                f"Core {core_number} receipt checksum is not bundle-bound."
            )

        receipt["source_receipt_relative_path"] = receipt_relative
        receipt["source_receipt_sha256"] = str(expected_receipt["sha256"])
        receipts.append(receipt)
        maximum_attention_error = max(
            maximum_attention_error,
            float(extraction["max_attention_sum_error"]),
        )
        maximum_softmax_error = max(
            maximum_softmax_error,
            float(extraction["max_softmax_reconstruction_error"]),
        )
        strict_shard_count += int(extraction["strict_attention_shard_count"])
        total_cells += cell_count
        total_directed += directed_count
        total_reciprocal += reciprocal_count

    if (
        total_cells != EXPECTED_TOTAL_CELLS
        or total_directed != EXPECTED_TOTAL_DIRECTED_EDGES
        or total_reciprocal * 2 != EXPECTED_TOTAL_DIRECTED_EDGES
    ):
        raise AttentionNichePipelineError(
            "Recovery source six-core totals drifted."
        )
    return receipts, {
        "verified": True,
        "core_count": len(receipts),
        "total_cells": total_cells,
        "total_directed_edges": total_directed,
        "total_reciprocal_pairs": total_reciprocal,
        "strict_attention_shard_count": strict_shard_count,
        "maximum_attention_sum_error": maximum_attention_error,
        "maximum_softmax_reconstruction_error": maximum_softmax_error,
        "checkpoint_receipts_match_current_discovery": True,
        "all_ten_masks_replayed_per_core": True,
        "all_final_niches_connected": True,
    }


@dataclass(frozen=True, slots=True)
class RecoveryCanonicalData:
    """Validated immutable source products used only for rendering."""

    assignments: pd.DataFrame = field(repr=False)
    summaries: pd.DataFrame = field(repr=False)
    sensitivity: pd.DataFrame = field(repr=False)
    colors: Mapping[str, str] = field(repr=False)
    regions: Mapping[str, Any] = field(repr=False)
    retained_overlay_edges: pd.DataFrame = field(repr=False)
    audit: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class VisualizationPatchInputs:
    """Three checksum-verified source products consumed by the renderer."""

    assignments: pd.DataFrame = field(repr=False)
    regions: Mapping[str, Any] = field(repr=False)
    retained_mutual_edges: pd.DataFrame = field(repr=False)
    audit: Mapping[str, Any] = field(repr=False)


def _load_visualization_patch_inputs(
    source: VerifiedVisualizationPatchSource,
) -> VisualizationPatchInputs:
    """Load only assignments, regions, and projected retained mutual edges."""

    assignments = pd.read_parquet(
        source.root / "cell_attention_niche_assignments.parquet"
    )
    required_assignment_columns = {
        "core_number",
        "core_alias",
        "cell_index",
        "x_um",
        "y_um",
        "coordinate_unit",
        "final_niche_id",
        "niche_color",
        "mutual_routing_hub_score",
        "assignment_confidence",
        "map_label",
    }
    if not required_assignment_columns.issubset(assignments.columns):
        raise AttentionNichePipelineError(
            "Visualization-patch assignments lack required renderer columns."
        )
    if (
        len(assignments) != EXPECTED_TOTAL_CELLS
        or tuple(sorted(assignments["core_number"].unique().tolist()))
        != tuple(sorted(EXPECTED_CORE_NUMBERS))
        or assignments.duplicated(["core_number", "cell_index"]).any()
        or not assignments["coordinate_unit"].eq("micrometres").all()
        or set(assignments["map_label"].dropna().astype(str))
        != {"4-model ensemble-consensus map"}
        or not np.isfinite(
            assignments[
                [
                    "x_um",
                    "y_um",
                    "mutual_routing_hub_score",
                    "assignment_confidence",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch assignment coverage/QC failed."
        )
    for alias, core_number in zip(
        EXPECTED_ALIASES, EXPECTED_CORE_NUMBERS, strict=True
    ):
        selected = assignments.loc[assignments["core_number"] == core_number]
        if (
            selected.empty
            or not selected["core_alias"].eq(alias).all()
            or not np.array_equal(
                np.sort(selected["cell_index"].to_numpy(dtype=np.int64)),
                np.arange(len(selected), dtype=np.int64),
            )
        ):
            raise AttentionNichePipelineError(
                f"Visualization-patch assignment identity failed for Core {core_number}."
            )

    regions = _load_json(
        source.root / "attention_niche_regions.geojson",
        "visualization-patch source regions",
    )
    features = regions.get("features")
    geometry_validation = regions.get("geometry_validation")
    if (
        regions.get("type") != "FeatureCollection"
        or regions.get("coordinate_unit") != "um"
        or not isinstance(features, list)
        or len(features) != int(assignments["final_niche_id"].nunique())
        or not isinstance(geometry_validation, Mapping)
        or int(geometry_validation.get("invalid_after_repair", -1)) != 0
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source region geometry audit failed."
        )

    retained_columns = [
        "core_number",
        "cell_i_index",
        "cell_j_index",
        "M_ij",
        "support_P_ij",
        "retained_primary",
    ]
    retained_table = pq.read_table(
        source.root / "mutual_attention_edges.parquet",
        columns=retained_columns,
        filters=[("retained_primary", "=", True)],
    )
    retained = retained_table.to_pandas()
    if (
        len(retained) != EXPECTED_TOTAL_RETAINED_MUTUAL_EDGES
        or tuple(sorted(retained["core_number"].unique().tolist()))
        != tuple(sorted(EXPECTED_CORE_NUMBERS))
        or not retained["retained_primary"].all()
        or not (retained["cell_i_index"] < retained["cell_j_index"]).all()
        or not (retained["M_ij"] > 1.0).all()
        or not (retained["support_P_ij"] >= 0.60).all()
        or retained.duplicated(
            ["core_number", "cell_i_index", "cell_j_index"]
        ).any()
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch retained mutual-edge projection failed."
        )

    return VisualizationPatchInputs(
        assignments=assignments,
        regions=regions,
        retained_mutual_edges=retained,
        audit={
            "source_files_read_for_rendering": list(
                VISUALIZATION_PATCH_RENDER_INPUTS
            ),
            "assignment_rows": int(len(assignments)),
            "region_feature_count": int(len(features)),
            "retained_mutual_edge_rows": int(len(retained)),
            "mutual_columns_projected": retained_columns,
            "mutual_full_table_loaded": False,
            "directed_attention_table_opened_or_hashed": False,
        },
    )


def _parquet_core_position_audit(
    path: Path,
    *,
    position_column: str,
    expected_counts: Mapping[int, int],
) -> dict[str, Any]:
    """Use row-group statistics to verify core-scoped contiguous table blocks."""

    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    if "core_number" not in names or position_column not in names:
        raise AttentionNichePipelineError(
            f"Canonical Parquet lacks core/position columns: {path.name}."
        )
    core_index = names.index("core_number")
    position_index = names.index(position_column)
    intervals: dict[int, list[tuple[int, int, int]]] = {
        core: [] for core in expected_counts
    }
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        core_statistics = row_group.column(core_index).statistics
        position_statistics = row_group.column(position_index).statistics
        if (
            core_statistics is None
            or not core_statistics.has_min_max
            or int(core_statistics.min) != int(core_statistics.max)
            or position_statistics is None
            or not position_statistics.has_min_max
        ):
            raise AttentionNichePipelineError(
                f"Canonical Parquet row-group statistics are insufficient: {path.name}."
            )
        core = int(core_statistics.min)
        if core not in intervals:
            raise AttentionNichePipelineError(
                f"Unexpected core in canonical Parquet: {path.name}."
            )
        start = int(position_statistics.min)
        stop = int(position_statistics.max)
        if stop - start + 1 != int(row_group.num_rows):
            raise AttentionNichePipelineError(
                f"Non-contiguous position row group in {path.name}."
            )
        intervals[core].append((start, stop, int(row_group.num_rows)))
    for core, expected_count in expected_counts.items():
        ordered = sorted(intervals[core])
        cursor = 0
        observed_rows = 0
        for start, stop, rows in ordered:
            if start != cursor:
                raise AttentionNichePipelineError(
                    f"Position coverage drifted for Core {core} in {path.name}."
                )
            cursor = stop + 1
            observed_rows += rows
        if cursor != expected_count or observed_rows != expected_count:
            raise AttentionNichePipelineError(
                f"Row count drifted for Core {core} in {path.name}."
            )
    return {
        "path": path.name,
        "row_count": int(parquet.metadata.num_rows),
        "row_group_count": int(parquet.metadata.num_row_groups),
        "position_column": position_column,
        "core_counts": {str(core): int(count) for core, count in expected_counts.items()},
        "core_scoped_contiguous_positions": True,
    }


def _stream_validate_recovery_edge_alignment(
    root: Path,
    *,
    input_contract: PreparedInputContract,
    core_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stream scalar exports and prove exact canonical graph-edge alignment."""

    aliases = {
        int(receipt["core_number"]): str(receipt["core_alias"])
        for receipt in core_receipts
    }
    node_counts = {
        int(receipt["core_number"]): int(receipt["cell_count"])
        for receipt in core_receipts
    }
    directed_counts = {
        int(receipt["core_number"]): int(receipt["directed_edge_count"])
        for receipt in core_receipts
    }
    reciprocal_counts = {
        int(receipt["core_number"]): int(receipt["reciprocal_pair_count"])
        for receipt in core_receipts
    }
    edges: dict[int, np.ndarray] = {}
    indegrees: dict[int, np.ndarray] = {}
    for core_number, alias in aliases.items():
        edge_path = input_contract.graph_dir / "cores" / alias / "edge_index.npy"
        edge_index = np.load(edge_path, mmap_mode="r", allow_pickle=False)
        if edge_index.shape != (2, directed_counts[core_number]):
            raise AttentionNichePipelineError(
                f"Immutable edge_index shape drifted for Core {core_number}."
            )
        edges[core_number] = edge_index
        indegrees[core_number] = np.bincount(
            np.asarray(edge_index[1], dtype=np.int64),
            minlength=node_counts[core_number],
        )

    observed_directed = {core: 0 for core in aliases}
    directed = pq.ParquetFile(root / "directed_attention_edges.parquet")
    directed_columns = [
        "core_number",
        "core_alias",
        "edge_index_position",
        "source_cell_index",
        "receiver_cell_index",
        "receiver_in_degree",
        "head_mean_attention_mean",
        "degree_adjusted_routing_mean",
    ]
    maximum_degree_adjustment_absolute_error = 0.0
    for batch in directed.iter_batches(
        batch_size=250_000,
        columns=directed_columns,
        use_threads=False,
    ):
        frame = batch.to_pandas()
        for core_number in frame["core_number"].unique().tolist():
            core = int(core_number)
            if core not in edges:
                raise AttentionNichePipelineError(
                    "Directed recovery export contains an unexpected core."
                )
            selected = frame.loc[frame["core_number"] == core]
            positions = selected["edge_index_position"].to_numpy(dtype=np.int64)
            sources = selected["source_cell_index"].to_numpy(dtype=np.int64)
            receivers = selected["receiver_cell_index"].to_numpy(dtype=np.int64)
            receiver_degrees = selected["receiver_in_degree"].to_numpy(
                dtype=np.int64
            )
            head_mean_attention = selected[
                "head_mean_attention_mean"
            ].to_numpy(dtype=np.float64)
            degree_adjusted_mean = selected[
                "degree_adjusted_routing_mean"
            ].to_numpy(dtype=np.float64)
            expected_degree_adjusted = receiver_degrees * head_mean_attention
            maximum_degree_adjustment_absolute_error = max(
                maximum_degree_adjustment_absolute_error,
                float(
                    np.max(
                        np.abs(degree_adjusted_mean - expected_degree_adjusted),
                        initial=0.0,
                    )
                ),
            )
            expected_positions = np.arange(
                observed_directed[core],
                observed_directed[core] + len(selected),
                dtype=np.int64,
            )
            if (
                not selected["core_alias"].eq(aliases[core]).all()
                or not np.array_equal(positions, expected_positions)
                or np.any(positions < 0)
                or np.any(positions >= directed_counts[core])
                or not np.array_equal(sources, edges[core][0, positions])
                or not np.array_equal(receivers, edges[core][1, positions])
                or not np.array_equal(
                    receiver_degrees,
                    indegrees[core][receivers],
                )
                or not np.allclose(
                    degree_adjusted_mean,
                    expected_degree_adjusted,
                    rtol=5e-6,
                    atol=1e-6,
                )
                or np.any(sources < 0)
                or np.any(sources >= node_counts[core])
                or np.any(receivers < 0)
                or np.any(receivers >= node_counts[core])
                or np.any(sources == receivers)
            ):
                raise AttentionNichePipelineError(
                    f"Directed edge_index/degree alignment failed for Core {core}."
                )
            observed_directed[core] += len(selected)
    if observed_directed != directed_counts:
        raise AttentionNichePipelineError(
            "Directed scalar alignment did not cover every canonical edge."
        )

    observed_mutual = {core: 0 for core in aliases}
    mutually_covered_directed_positions = {
        core: np.zeros(directed_counts[core], dtype=np.bool_)
        for core in aliases
    }
    mutual = pq.ParquetFile(root / "mutual_attention_edges.parquet")
    mutual_columns = [
        "core_number",
        "core_alias",
        "mutual_pair_position",
        "cell_i_index",
        "cell_j_index",
        "i_to_j_edge_index_position",
        "j_to_i_edge_index_position",
    ]
    for batch in mutual.iter_batches(
        batch_size=250_000,
        columns=mutual_columns,
        use_threads=False,
    ):
        frame = batch.to_pandas()
        for core_number in frame["core_number"].unique().tolist():
            core = int(core_number)
            if core not in edges:
                raise AttentionNichePipelineError(
                    "Mutual recovery export contains an unexpected core."
                )
            selected = frame.loc[frame["core_number"] == core]
            pair_positions = selected["mutual_pair_position"].to_numpy(
                dtype=np.int64
            )
            cells_i = selected["cell_i_index"].to_numpy(dtype=np.int64)
            cells_j = selected["cell_j_index"].to_numpy(dtype=np.int64)
            forward = selected["i_to_j_edge_index_position"].to_numpy(
                dtype=np.int64
            )
            reverse = selected["j_to_i_edge_index_position"].to_numpy(
                dtype=np.int64
            )
            expected_pair_positions = np.arange(
                observed_mutual[core],
                observed_mutual[core] + len(selected),
                dtype=np.int64,
            )
            if (
                not selected["core_alias"].eq(aliases[core]).all()
                or not np.array_equal(pair_positions, expected_pair_positions)
                or np.any(pair_positions < 0)
                or np.any(pair_positions >= reciprocal_counts[core])
                or np.any(cells_i < 0)
                or np.any(cells_j >= node_counts[core])
                or np.any(cells_i >= cells_j)
                or np.any(forward < 0)
                or np.any(reverse < 0)
                or np.any(forward >= directed_counts[core])
                or np.any(reverse >= directed_counts[core])
                or np.any(forward == reverse)
                or np.unique(np.concatenate((forward, reverse))).size
                != 2 * len(selected)
                or not np.array_equal(edges[core][0, forward], cells_i)
                or not np.array_equal(edges[core][1, forward], cells_j)
                or not np.array_equal(edges[core][0, reverse], cells_j)
                or not np.array_equal(edges[core][1, reverse], cells_i)
                or np.any(mutually_covered_directed_positions[core][forward])
                or np.any(mutually_covered_directed_positions[core][reverse])
            ):
                raise AttentionNichePipelineError(
                    f"Mutual reciprocal-edge alignment failed for Core {core}."
                )
            mutually_covered_directed_positions[core][forward] = True
            mutually_covered_directed_positions[core][reverse] = True
            observed_mutual[core] += len(selected)
    if observed_mutual != reciprocal_counts:
        raise AttentionNichePipelineError(
            "Mutual scalar alignment did not cover every reciprocal pair."
        )
    if not all(
        bool(coverage.all())
        for coverage in mutually_covered_directed_positions.values()
    ):
        raise AttentionNichePipelineError(
            "At least one directed edge lacks exactly one reciprocal-pair record."
        )
    return {
        "directed_scalar_columns_streamed": directed_columns,
        "mutual_scalar_columns_streamed": mutual_columns,
        "directed_rows_verified": int(sum(observed_directed.values())),
        "mutual_rows_verified": int(sum(observed_mutual.values())),
        "edge_index_source_receiver_alignment_exact": True,
        "edge_positions_complete_unique_and_ordered": True,
        "receiver_indegree_exact": True,
        "degree_adjustment_uses_receiver_indegree": True,
        "maximum_degree_adjustment_absolute_error": (
            maximum_degree_adjustment_absolute_error
        ),
        "core_local_endpoint_bounds_exact": True,
        "every_mutual_pair_has_two_reversed_exported_edges": True,
        "every_directed_edge_covered_by_exactly_one_mutual_pair": True,
        "mutual_pair_positions_complete_unique_and_ordered": True,
        "one_directional_edges_rejected": True,
    }


def _validated_recovery_canonical_data(
    output_root: Path,
    *,
    core_receipts: Sequence[Mapping[str, Any]],
    input_contract: PreparedInputContract,
) -> RecoveryCanonicalData:
    """Validate canonical tables/regions and read only retained overlay columns."""

    expected_cells = {
        int(receipt["core_number"]): int(receipt["cell_count"])
        for receipt in core_receipts
    }
    expected_directed = {
        int(receipt["core_number"]): int(receipt["directed_edge_count"])
        for receipt in core_receipts
    }
    expected_reciprocal = {
        int(receipt["core_number"]): int(receipt["reciprocal_pair_count"])
        for receipt in core_receipts
    }
    expected_retained = {
        int(receipt["core_number"]): int(receipt["retained_mutual_edge_count"])
        for receipt in core_receipts
    }
    assignments_path = output_root / "cell_attention_niche_assignments.parquet"
    mutual_path = output_root / "mutual_attention_edges.parquet"
    directed_path = output_root / "directed_attention_edges.parquet"
    assignments = pd.read_parquet(assignments_path)
    required_assignment_columns = {
        "core_number",
        "core_alias",
        "cell_index",
        "original_cell_identifier",
        "x_um",
        "y_um",
        "coordinate_unit",
        "preliminary_leiden_community",
        "final_connected_niche_id",
        "final_niche_id",
        "micro_niche",
        "niche_color",
        "assignment_confidence",
        "mutual_routing_hub_score",
        "S_i",
        "model_seeds_used",
        "analysis_mask_views_used",
        "map_label",
    }
    if not required_assignment_columns.issubset(assignments.columns):
        raise AttentionNichePipelineError(
            "Recovery assignment table lacks required canonical columns."
        )
    if (
        len(assignments) != EXPECTED_TOTAL_CELLS
        or tuple(sorted(assignments["core_number"].unique().tolist()))
        != tuple(sorted(EXPECTED_CORE_NUMBERS))
        or assignments.duplicated(["core_number", "cell_index"]).any()
        or assignments["original_cell_identifier"].duplicated().any()
        or not assignments["coordinate_unit"].eq("micrometres").all()
        or not np.isfinite(assignments[["x_um", "y_um"]].to_numpy()).all()
        or not assignments["final_connected_niche_id"].equals(
            assignments["final_niche_id"]
        )
        or not np.allclose(
            assignments["S_i"].to_numpy(dtype=np.float64),
            assignments["mutual_routing_hub_score"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        or not np.isfinite(
            assignments["assignment_confidence"].to_numpy(dtype=np.float64)
        ).all()
        or not assignments["assignment_confidence"].between(0.0, 1.0).all()
        or set(assignments["model_seeds_used"].astype(str)) != {"[0,1,2,3]"}
        or set(assignments["analysis_mask_views_used"].astype(str))
        != {"[0,1,2,3,4,5,6,7,8,9]"}
        or set(assignments["map_label"].astype(str))
        != {"4-model ensemble-consensus map"}
    ):
        raise AttentionNichePipelineError("Recovery assignment coverage/QC failed.")
    for alias, core_number in zip(
        EXPECTED_ALIASES, EXPECTED_CORE_NUMBERS, strict=True
    ):
        selected = assignments.loc[assignments["core_number"] == core_number]
        if (
            len(selected) != expected_cells[core_number]
            or not selected["core_alias"].eq(alias).all()
            or not np.array_equal(
                np.sort(selected["cell_index"].to_numpy(dtype=np.int64)),
                np.arange(expected_cells[core_number], dtype=np.int64),
            )
        ):
            raise AttentionNichePipelineError(
                f"Recovery assignment coverage failed for Core {core_number}."
            )

    directed_audit = _parquet_core_position_audit(
        directed_path,
        position_column="edge_index_position",
        expected_counts=expected_directed,
    )
    mutual_audit = _parquet_core_position_audit(
        mutual_path,
        position_column="mutual_pair_position",
        expected_counts=expected_reciprocal,
    )
    directed_names = set(pq.ParquetFile(directed_path).schema_arrow.names)
    if not {
        "source_cell_index",
        "receiver_cell_index",
        "receiver_in_degree",
        "distance_um",
        "degree_adjusted_routing_per_seed_mask_view",
    }.issubset(directed_names):
        raise AttentionNichePipelineError(
            "Recovery directed table lacks direction/degree routing fields."
        )

    retained_columns = [
        "core_number",
        "cell_i_index",
        "cell_j_index",
        "M_ij",
        "support_P_ij",
        "retained_primary",
    ]
    retained_table = pq.read_table(
        mutual_path,
        columns=retained_columns,
        filters=[("retained_primary", "=", True)],
    )
    retained = retained_table.to_pandas()
    retained_counts = {
        int(core): int(count)
        for core, count in retained.groupby("core_number", sort=True).size().items()
    }
    if (
        retained_counts != expected_retained
        or not retained["retained_primary"].all()
        or not (retained["cell_i_index"] < retained["cell_j_index"]).all()
        or not (retained["M_ij"] > 1.0).all()
        or not (retained["support_P_ij"] >= 0.60).all()
        or retained.duplicated(
            ["core_number", "cell_i_index", "cell_j_index"]
        ).any()
    ):
        raise AttentionNichePipelineError(
            "Filtered retained mutual-edge validation failed."
        )
    for core_number, cell_count in expected_cells.items():
        selected = retained.loc[retained["core_number"] == core_number]
        if (
            selected["cell_i_index"].min() < 0
            or selected["cell_j_index"].max() >= cell_count
        ):
            raise AttentionNichePipelineError(
                f"Retained mutual endpoint escaped Core {core_number}."
            )

    summaries = pd.read_csv(output_root / "attention_niche_summary.csv")
    sensitivity = pd.read_csv(
        output_root / "attention_niche_parameter_sensitivity.csv"
    )
    colors_raw = _load_json(
        output_root / "attention_niche_colors.json",
        "recovery niche colors",
    )
    colors = {str(key): str(value) for key, value in colors_raw.items()}
    regions = _load_json(
        output_root / "attention_niche_regions.geojson",
        "recovery niche regions",
    )
    assignment_niches = set(assignments["final_niche_id"].astype(str))
    summary_niches = set(summaries["niche_id"].astype(str))
    region_features = regions.get("features")
    geometry_validation = regions.get("geometry_validation")
    if not isinstance(region_features, list) or not isinstance(
        geometry_validation, Mapping
    ):
        raise AttentionNichePipelineError("Recovery region GeoJSON audit is absent.")
    region_niches = {str(feature.get("id")) for feature in region_features}
    if (
        assignment_niches != summary_niches
        or assignment_niches != set(colors)
        or assignment_niches != region_niches
        or len(region_features)
        != sum(
            int(receipt["final_connected_niche_count"])
            for receipt in core_receipts
        )
        or regions.get("coordinate_unit") != "um"
        or int(geometry_validation.get("invalid_after_repair", -1)) != 0
        or float(
            geometry_validation.get("maximum_reported_area_error_um2", math.inf)
        )
        > 1e-4
        or not assignments.apply(
            lambda row: colors.get(str(row["final_niche_id"]))
            == str(row["niche_color"]),
            axis=1,
        ).all()
        or int(summaries["number_of_cells"].sum()) != EXPECTED_TOTAL_CELLS
        or bool(summaries["biological_niche_name_assigned"].any())
    ):
        raise AttentionNichePipelineError(
            "Recovery niche/color/summary/region correspondence failed."
        )
    primary_sensitivity = sensitivity.loc[sensitivity["is_primary"].astype(bool)]
    expected_sensitivity_grid = {
        (core, top_k, resolution)
        for core in EXPECTED_CORE_NUMBERS
        for top_k in (5, 8, 10)
        for resolution in (0.5, 1.0, 1.5)
    }
    observed_sensitivity_grid = {
        (int(row.core_number), int(row.top_k), float(row.resolution))
        for row in sensitivity.itertuples()
    }
    if (
        tuple(sorted(sensitivity["core_number"].unique().tolist()))
        != tuple(sorted(EXPECTED_CORE_NUMBERS))
        or len(sensitivity) != len(expected_sensitivity_grid)
        or observed_sensitivity_grid != expected_sensitivity_grid
        or len(primary_sensitivity) != len(EXPECTED_CORE_NUMBERS)
        or not primary_sensitivity["top_k"].eq(8).all()
        or not primary_sensitivity["resolution"].eq(1.0).all()
    ):
        raise AttentionNichePipelineError(
            "Recovery parameter-sensitivity contract failed."
        )
    edge_alignment = _stream_validate_recovery_edge_alignment(
        output_root,
        input_contract=input_contract,
        core_receipts=core_receipts,
    )
    return RecoveryCanonicalData(
        assignments=assignments,
        summaries=summaries,
        sensitivity=sensitivity,
        colors=colors,
        regions=regions,
        retained_overlay_edges=retained,
        audit={
            "assignment_rows": int(len(assignments)),
            "niche_count": int(len(assignment_niches)),
            "retained_mutual_edge_count": int(len(retained)),
            "retained_mutual_columns_read": retained_columns,
            "mutual_table_full_rows_loaded_for_rendering": False,
            "directed_table_rows_loaded_for_rendering": False,
            "directed_table": directed_audit,
            "mutual_table": mutual_audit,
            "geometry_validation": dict(geometry_validation),
            "edge_alignment": edge_alignment,
        },
    )


def _run_attention_niche_visualization_patch(
    *,
    request: VisualizationPatchRequest,
    config: Mapping[str, Any],
    run_id: str,
    output_root: Path,
    database: str | Path,
    selected_paths: ProjectPaths,
) -> dict[str, Any]:
    """Re-render eleven figures from one immutable completed source bundle."""

    import subprocess

    from .attention_niche_visualization import (
        render_attention_niche_visualizations,
    )

    if run_id == request.source_run_id:
        raise AttentionNichePipelineError(
            "Visualization patch requires a new worker-owned run ID."
        )
    if any((output_root / relative).exists() for relative in VISUALIZATION_PATCH_OUTPUTS):
        raise AttentionNichePipelineError(
            "Visualization patch refuses to overwrite an existing figure."
        )
    renderer_path = Path(__file__).with_name("attention_niche_visualization.py")
    renderer_sha256 = sha256_file(renderer_path)
    if renderer_sha256 != request.expected_renderer_source_sha256:
        raise AttentionNichePipelineError(
            "Visualization-patch renderer source identity drifted."
        )
    source = _verify_visualization_patch_source_identity(
        request=request,
        database=database,
        paths=selected_paths,
        current_config=config,
    )
    disk_snapshots = [
        _filesystem_snapshot(output_root, stage="visualization_patch_preflight")
    ]
    if disk_snapshots[-1]["free_gib"] < request.minimum_figure_headroom_gib:
        raise AttentionNichePipelineError(
            "Insufficient free disk headroom for the visualization patch."
        )
    inputs = _load_visualization_patch_inputs(source)
    visualization = render_attention_niche_visualizations(
        inputs.assignments,
        inputs.regions,
        inputs.retained_mutual_edges,
        output_root,
        dpi=300,
        individual_dpi=450,
        include_overlay=True,
        max_edges_per_core=1_000,
        max_edges_total=4_000,
        invert_y=True,
        low_confidence_threshold=0.60,
        allow_cross_core_color_reuse=False,
    )
    visualization_receipt = json.loads(
        _canonical_json(dict(visualization.receipt))
    )
    disk_snapshots.append(
        _filesystem_snapshot(
            output_root, stage="visualization_patch_figures_complete"
        )
    )
    visualization_receipt["output_paths"] = [
        _relative_to_root(Path(path), output_root)
        for path in visualization_receipt.get("output_paths", [])
    ]
    observed_outputs = set(visualization_receipt["output_paths"])
    omitted_rings = visualization_receipt.get(
        "zero_area_interior_rings_omitted_from_render"
    )
    if (
        visualization_receipt.get("schema")
        != "attention_niche_visualization_receipt_v2"
        or visualization_receipt.get("status") != "complete"
        or visualization_receipt.get("core_order")
        != list(EXPECTED_CORE_NUMBERS)
        or int(visualization_receipt.get("total_cell_count", -1))
        != EXPECTED_TOTAL_CELLS
        or observed_outputs != set(VISUALIZATION_PATCH_OUTPUTS)
        or visualization_receipt.get("region_holes_preserved")
        != "all_nondegenerate"
        or visualization_receipt.get("nondegenerate_region_holes_preserved")
        is not True
        or not isinstance(omitted_rings, Mapping)
        or int(omitted_rings.get("count", -1)) != 3
        or omitted_rings.get("identifier_list_sha256")
        != RENDER_RECOVERY_OMITTED_RING_SHA256
        or omitted_rings.get("scientific_geometry_modified") is not False
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch renderer receipt failed its locked contract."
        )

    post_render_hashes: dict[str, str] = {}
    for relative, pre_receipt in source.rendering_input_receipts.items():
        observed = sha256_file(source.root / relative)
        post_render_hashes[relative] = observed
        if observed != pre_receipt.get("sha256"):
            raise AttentionNichePipelineError(
                f"Visualization source changed during rendering: {relative}."
            )
    if (
        sha256_file(source.root / "_SUCCESS")
        != source.source_success_marker_file_sha256
        or sha256_file(source.root / "provenance/artifact_checksums.json")
        != source.source_artifact_checksum_manifest_sha256
        or sha256_file(source.root / "config.resolved.yaml")
        != source.source_config_sha256
    ):
        raise AttentionNichePipelineError(
            "Visualization-patch source lifecycle/config identity changed."
        )
    for relative, receipt in source.artifact_files.items():
        path = source.root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or int(path.stat().st_size) != int(receipt["size"])
        ):
            raise AttentionNichePipelineError(
                f"Visualization-patch source artifact changed: {relative}."
            )

    copied_scientific_outputs = [
        relative
        for relative in RENDER_RECOVERY_CANONICAL_OUTPUTS
        if (output_root / relative).exists()
    ]
    if copied_scientific_outputs:
        raise AttentionNichePipelineError(
            "Visualization patch must not copy canonical scientific products."
        )
    figure_checksums = _artifact_checksums(
        output_root, VISUALIZATION_PATCH_OUTPUTS
    )
    try:
        recovery_git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=selected_paths.project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AttentionNichePipelineError(
            "Cannot record visualization-patch Git commit."
        ) from exc

    qc_checks = {
        "exact_completed_source_run_and_queue": bool(
            source.queue_job.get("job_id") == request.source_queue_job_id
            and source.queue_job.get("status") == "completed"
        ),
        "exact_success_marker_file_sha256": bool(
            source.source_success_marker_file_sha256
            == request.source_success_marker_file_sha256
        ),
        "exact_checksum_manifest_sha256": bool(
            source.source_artifact_checksum_manifest_sha256
            == request.source_artifact_checksum_manifest_sha256
        ),
        "all_source_artifact_receipts_registered_present": bool(
            source.registry_artifacts
            and all(
                receipt.get("status") == "present"
                for receipt in source.registry_artifacts.values()
            )
        ),
        "scientific_configuration_unchanged": bool(
            scientific_id(config) == RENDER_RECOVERY_SCIENTIFIC_ID
        ),
        "exact_title_layout_renderer_source": bool(
            renderer_sha256 == request.expected_renderer_source_sha256
        ),
        "only_three_scientific_inputs_read_for_rendering": bool(
            set(inputs.audit["source_files_read_for_rendering"])
            == set(VISUALIZATION_PATCH_RENDER_INPUTS)
            and inputs.audit["directed_attention_table_opened_or_hashed"]
            is False
        ),
        "source_render_inputs_unchanged": bool(
            all(
                post_render_hashes[relative] == receipt.get("sha256")
                for relative, receipt in source.rendering_input_receipts.items()
            )
        ),
        "exact_core_order_and_all_cells_rendered": bool(
            visualization_receipt.get("core_order")
            == list(EXPECTED_CORE_NUMBERS)
            and visualization_receipt.get("all_eligible_cells_rendered_without_sampling")
            is True
            and visualization_receipt.get("total_cell_count")
            == EXPECTED_TOTAL_CELLS
        ),
        "all_eleven_figures_freshly_rendered": bool(
            observed_outputs == set(VISUALIZATION_PATCH_OUTPUTS)
            and len(figure_checksums) == 11
        ),
        "overlay_is_locked_display_subset": bool(
            visualization_receipt.get("overlay", {}).get("included") is True
            and visualization_receipt.get("overlay", {}).get("max_edges_per_core")
            == 1_000
            and visualization_receipt.get("overlay", {}).get("max_edges_total")
            == 4_000
        ),
        "coordinates_aspect_labels_and_scale_bars_preserved": bool(
            visualization_receipt.get("coordinate_unit") == "micrometres"
            and visualization_receipt.get("equal_physical_aspect") is True
            and visualization_receipt.get("panel_titles_include_core_number")
            is True
            and visualization_receipt.get("in_panel_core_labels")
            == [f"Core {core}" for core in EXPECTED_CORE_NUMBERS]
            and all(
                value.get("present") and value.get("unit") == "µm"
                for value in visualization_receipt.get("scale_bars", {}).values()
            )
        ),
        "all_nondegenerate_region_holes_preserved": bool(
            visualization_receipt.get("nondegenerate_region_holes_preserved")
            is True
            and omitted_rings.get("scientific_geometry_modified") is False
        ),
        "scientific_tables_and_values_not_recomputed": True,
        "source_bundle_not_copied_or_mutated": not copied_scientific_outputs,
    }
    if not all(qc_checks.values()):
        failed = sorted(name for name, passed in qc_checks.items() if not passed)
        raise AttentionNichePipelineError(
            "Visualization-only patch QC failed: " + ", ".join(failed)
        )
    qc_fraction = float(sum(qc_checks.values()) / len(qc_checks))
    if qc_fraction != 1.0:
        raise AttentionNichePipelineError(
            "Visualization-only patch requires QC fraction 1.0."
        )

    requested_gpu = str(config["launcher"].get("requested_gpu", "0,2,3"))
    reproduction_command = (
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 enqueue-experiment "
        f"--campaign-id {ANALYSIS_CAMPAIGN_ID} "
        "--config experiments/campaigns/"
        "cmp_20260825_six_core_attention_routing_niches/"
        "figure_patch_config.yaml --priority 0 --max-attempts 1 "
        f"--gpu {requested_gpu} && "
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 worker "
        "--worker-id attention-niche-visualization-patch "
        f"--gpu {requested_gpu} --min-free-gb 4 --once"
    )
    readme = f"""# Six-core attention-routing niche figure patch

Status: complete visualization-only patch of immutable completed source run
`{request.source_run_id}`. This run corrects the mutual-attention overlay title
layout and freshly renders all eleven static figure files with the current
renderer.

Scientific tables recomputed: `false`. Scientific values recomputed: `false`.
Visualizations recomputed: `true`. The source assignments, model-defined region
geometry, and retained mutual-edge projection were read and checksum-verified;
they were not copied into this bundle. The 21 GB directed-attention table was
not opened, hashed, copied, or modified. No model, checkpoint, prepared input,
or mask was loaded. Existing niche assignments, confidence values, and retained
mutual scores were loaded only as renderer inputs; none of those scientific
values was recomputed.

Source completion marker SHA-256:
`{request.source_success_marker_file_sha256}`. Source checksum-manifest SHA-256:
`{request.source_artifact_checksum_manifest_sha256}`. Queue job:
`{request.source_queue_job_id}`. Scientific ID remains
`{RENDER_RECOVERY_SCIENTIFIC_ID}`.

Reproduce as a new registered analysis-only visualization patch:

```bash
{reproduction_command}
```

> {LIMITATION}
"""
    _write_text_atomic(output_root / "README.md", readme)
    qc_lines = [
        "# Attention-routing niche visualization-patch QC",
        "",
        f"Overall QC pass fraction: `{qc_fraction:.6f}` "
        f"({sum(qc_checks.values())}/{len(qc_checks)}).",
        "",
        f"- Source run: `{request.source_run_id}` (completed).",
        f"- Source queue job: `{request.source_queue_job_id}` (completed).",
        "- Scientific tables recomputed: `false`.",
        "- Scientific values recomputed: `false`.",
        "- Visualizations recomputed: `true`.",
        "- Source bundle copied or mutated: `false`.",
        "- Directed-attention table opened or content-hashed: `false`.",
        f"- Fresh figure count: `{len(figure_checksums)}`.",
        f"- Renderer source SHA-256: `{renderer_sha256}`.",
        "",
        "## Explicit checks",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'} — `{name}`"
            for name, passed in qc_checks.items()
        ],
        "",
        "## Interpretation limit",
        "",
        f"> {LIMITATION}",
        "",
    ]
    _write_text_atomic(
        output_root / "analysis_qc_report.md", "\n".join(qc_lines)
    )

    patch_outputs = (
        *VISUALIZATION_PATCH_OUTPUTS,
        "analysis_qc_report.md",
        "README.md",
    )
    output_artifacts = _artifact_checksums(output_root, patch_outputs)
    canonical_source_references = {
        relative: {
            "source_run_id": request.source_run_id,
            "source_relative_path": relative,
            **dict(source.artifact_files[relative]),
            "copied_to_patch_bundle": False,
        }
        for relative in RENDER_RECOVERY_CANONICAL_OUTPUTS
    }
    source_content_hashed = [
        "_SUCCESS",
        "provenance/artifact_checksums.json",
        "config.resolved.yaml",
        *VISUALIZATION_PATCH_RENDER_INPUTS,
    ]
    source_not_content_hashed = sorted(
        set(source.artifact_files)
        - {"config.resolved.yaml", *VISUALIZATION_PATCH_RENDER_INPUTS}
    )
    analysis_manifest = {
        "schema": "six_core_attention_niche_visualization_patch_v1",
        "status": "complete",
        "run_id": run_id,
        "campaign_id": ANALYSIS_CAMPAIGN_ID,
        "artifact_contract": "analysis_only",
        "execution_mode": VISUALIZATION_PATCH_MODE,
        "patch_reason": "correct_mutual_attention_overlay_title_layout",
        "created_at": _utc_now(),
        "scientific_id": RENDER_RECOVERY_SCIENTIFIC_ID,
        "terminology": "model-defined attention-routing niches",
        "limitation": LIMITATION,
        "core_order": list(EXPECTED_CORE_NUMBERS),
        "scientific_tables_recomputed": False,
        "scientific_values_recomputed": False,
        "visualizations_recomputed": True,
        "attention_extraction_performed": False,
        "model_or_checkpoint_loading_performed": False,
        "source_bundle_copied": False,
        "source_bundle_mutated": False,
        "source_lineage": {
            "source_run_id": request.source_run_id,
            "source_queue_job": dict(source.queue_job),
            "source_bundle": _relative_to_root(
                source.root, selected_paths.project_root
            ),
            "success_marker": dict(source.success_marker),
            "success_marker_file_sha256": (
                source.source_success_marker_file_sha256
            ),
            "artifact_checksum_manifest_sha256": (
                source.source_artifact_checksum_manifest_sha256
            ),
            "source_config_sha256": source.source_config_sha256,
            "scientific_config_sha256": source.scientific_config_sha256,
            "all_registry_artifact_receipts": dict(
                source.registry_artifacts
            ),
            "canonical_scientific_artifact_references": (
                canonical_source_references
            ),
            "source_files_content_hashed": source_content_hashed,
            "source_files_not_content_hashed": source_not_content_hashed,
            "directed_attention_table_opened_or_hashed": False,
            "rendering_input_pre_receipts": dict(
                source.rendering_input_receipts
            ),
            "rendering_input_post_sha256": post_render_hashes,
        },
        "rendering_input_audit": dict(inputs.audit),
        "renderer_provenance": {
            "git_commit": recovery_git_commit,
            "renderer_source_relative_path": _relative_to_root(
                renderer_path, selected_paths.project_root
            ),
            "renderer_source_sha256": renderer_sha256,
            "expected_renderer_source_sha256": (
                request.expected_renderer_source_sha256
            ),
            "title_layout_patch": (
                "mutual-attention overlay suptitle clearance above top-row "
                "panel titles"
            ),
        },
        "resource_execution": {
            "gpu_attention_extraction_performed": False,
            "peak_vram_gib": 0.0,
            "minimum_figure_headroom_gib": (
                request.minimum_figure_headroom_gib
            ),
            "filesystem_snapshots": disk_snapshots,
        },
        "visualization_receipt": visualization_receipt,
        "qc_checks": qc_checks,
        "qc_pass_fraction": qc_fraction,
        "artifacts": output_artifacts,
        "reproduction_command": reproduction_command,
    }
    _write_yaml_atomic(output_root / "analysis_manifest.yaml", analysis_manifest)
    _artifact_checksums(
        output_root,
        (*patch_outputs, "analysis_manifest.yaml"),
    )

    final_metrics = {
        "analysis/attention_niche_qc_pass_fraction": qc_fraction,
        "analysis/visualization_patch_qc_pass_fraction": qc_fraction,
        "analysis/visualization_patch_figure_count": 11.0,
        "analysis/total_cells": float(EXPECTED_TOTAL_CELLS),
        "analysis/retained_mutual_edges": float(
            EXPECTED_TOTAL_RETAINED_MUTUAL_EDGES
        ),
    }
    _write_json_atomic(output_root / "metrics" / "final.json", final_metrics)
    _write_text_atomic(
        output_root / "metrics" / "history.jsonl",
        _canonical_json(
            {"step": 0, **final_metrics, "analysis_complete": True}
        )
        + "\n",
    )
    _write_text_atomic(
        output_root / "metrics" / "events.jsonl",
        "".join(
            _canonical_json({"name": name, "value": value, "step": 0}) + "\n"
            for name, value in final_metrics.items()
        ),
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "analysis_kind": "six_core_attention_niche_visualization_patch",
        "execution_mode": VISUALIZATION_PATCH_MODE,
        "source_run_id": request.source_run_id,
        "scientific_id": RENDER_RECOVERY_SCIENTIFIC_ID,
        "scientific_tables_recomputed": False,
        "scientific_values_recomputed": False,
        "visualizations_recomputed": True,
        "source_bundle_copied_or_mutated": False,
        "primary_metric_name": "analysis/attention_niche_qc_pass_fraction",
        "primary_metric_value": qc_fraction,
        "metrics": final_metrics,
        "figure_count": 11,
        "core_numbers": list(EXPECTED_CORE_NUMBERS),
        "limitation": LIMITATION,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def _run_attention_niche_render_recovery(
    *,
    request: RenderRecoveryRequest,
    config: Mapping[str, Any],
    analysis_parameters: Mapping[str, Any],
    run_id: str,
    output_root: Path,
    database: str | Path,
    selected_paths: ProjectPaths,
    members: Sequence[CheckpointMember],
    input_contract: PreparedInputContract,
) -> dict[str, Any]:
    """Render a new registered bundle from one fully verified failed run."""

    import importlib.metadata
    import resource
    import subprocess

    from .attention_niche_visualization import (
        render_attention_niche_visualizations,
    )

    if run_id == request.source_run_id:
        raise AttentionNichePipelineError(
            "Render recovery requires a new worker-owned run ID."
        )
    source = _verify_render_recovery_source_identity(
        request=request,
        database=database,
        paths=selected_paths,
        current_config=config,
    )
    filesystem_devices = {
        "source_archive": int(source.root.stat().st_dev),
        "owned_scratch": int(output_root.stat().st_dev),
        "final_archive_root": int(selected_paths.artifact_root.stat().st_dev),
    }
    if len(set(filesystem_devices.values())) != 1:
        raise AttentionNichePipelineError(
            "Recovery source, owned scratch, and final archive must share one "
            "filesystem for FICLONE and atomic archive publication."
        )
    disk_snapshots = [_filesystem_snapshot(output_root, stage="recovery_preflight")]
    if (
        disk_snapshots[-1]["free_gib"]
        < request.minimum_figure_headroom_gib
    ):
        raise AttentionNichePipelineError(
            "Insufficient free disk headroom for render-only recovery figures."
        )

    core_receipts, source_core_audit = _verified_recovery_core_receipts(
        source,
        members=members,
        mask_views=int(analysis_parameters["analysis_mask_views"]),
    )
    canonical = _validated_recovery_canonical_data(
        source.root,
        core_receipts=core_receipts,
        input_contract=input_contract,
    )
    visualization = render_attention_niche_visualizations(
        canonical.assignments,
        canonical.regions,
        canonical.retained_overlay_edges,
        output_root,
        dpi=300,
        individual_dpi=450,
        include_overlay=True,
        max_edges_per_core=1_000,
        max_edges_total=4_000,
        invert_y=True,
        low_confidence_threshold=0.60,
        allow_cross_core_color_reuse=False,
    )
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="recovery_visualizations_complete")
    )
    raw_visualization_receipt = json.loads(
        _canonical_json(dict(visualization.receipt))
    )
    omitted_rings = raw_visualization_receipt.get(
        "zero_area_interior_rings_omitted_from_render"
    )
    if (
        raw_visualization_receipt.get("schema")
        != "attention_niche_visualization_receipt_v2"
        or raw_visualization_receipt.get("region_holes_preserved")
        != "all_nondegenerate"
        or raw_visualization_receipt.get("nondegenerate_region_holes_preserved")
        is not True
        or not isinstance(omitted_rings, Mapping)
        or int(omitted_rings.get("count", -1)) != 3
        or len(omitted_rings.get("identifiers", [])) != 3
        or omitted_rings.get("identifier_list_sha256")
        != RENDER_RECOVERY_OMITTED_RING_SHA256
        or omitted_rings.get("scientific_geometry_modified") is not False
        or float(omitted_rings.get("signed_area_um2", math.nan)) != 0.0
        or len(str(omitted_rings.get("identifier_list_sha256", ""))) != 64
    ):
        raise AttentionNichePipelineError(
            "Renderer did not report the exact nondegenerate-hole behavior."
        )
    raw_visualization_receipt["output_paths"] = [
        _relative_to_root(Path(path), output_root)
        for path in raw_visualization_receipt.get("output_paths", [])
    ]

    # Only after source validation and rendering succeed do we materialize the
    # seven canonical scientific products into the new worker-owned bundle.
    materialization: dict[str, dict[str, Any]] = {}
    for relative in RENDER_RECOVERY_CANONICAL_OUTPUTS:
        expected = source.artifact_files[relative]
        receipt = _materialize_ficlone_reflink(
            source.root / relative,
            output_root / relative,
            expected_sha256=str(expected["sha256"]),
        )
        receipt["source_path"] = _relative_to_root(
            source.root / relative,
            selected_paths.project_root,
        )
        receipt["destination_path"] = relative
        receipt["source_bundle_checksum_receipt"] = dict(expected)
        materialization[relative] = receipt
    if not all(
        receipt["method"] == "linux_ficlone_reflink"
        and receipt["distinct_inode"] is True
        and receipt["source_link_count"] == 1
        and receipt["destination_link_count"] == 1
        and receipt["copy_fallback_permitted"] is False
        for receipt in materialization.values()
    ):
        raise AttentionNichePipelineError(
            "Canonical recovery artifacts were not exclusively reflink materialized."
        )
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="reflink_materialization_complete")
    )

    input_post = verify_inputs_unchanged(
        input_contract.pre_analysis_file_receipts,
        project_root=selected_paths.project_root,
        data_root=selected_paths.data_root,
    )
    checkpoint_post = _checkpoint_post_receipts(
        members,
        root=selected_paths.project_root,
    )
    source_post_verification = verify_run_bundle(
        source.root,
        require_success_contract=False,
    )
    if (
        source_post_verification.get("status") != "failed"
        or source_post_verification.get("valid") is not True
        or sha256_file(source.root / "_FAILED")
        != source.failed_marker_file_sha256
        or sha256_file(source.root / "provenance/artifact_checksums.json")
        != source.artifact_checksum_manifest_sha256
    ):
        raise AttentionNichePipelineError(
            "Immutable recovery source changed during rendering."
        )
    for relative, receipt in materialization.items():
        destination = output_root / relative
        source_path = source.root / relative
        if (
            destination.is_symlink()
            or source_path.is_symlink()
            or destination.stat().st_ino == source_path.stat().st_ino
            or sha256_file(destination) != receipt["sha256"]
        ):
            raise AttentionNichePipelineError(
                f"Recovery materialization changed after rendering: {relative}."
            )

    panel_receipt = raw_visualization_receipt
    panel_outputs = {
        str(path)
        for path in panel_receipt.get("output_paths", [])
    }
    expected_visual_outputs = {
        relative
        for relative in REQUIRED_ANALYSIS_OUTPUTS
        if Path(relative).suffix.lower() in {".png", ".pdf", ".svg"}
    }
    source_geometry = canonical.audit["geometry_validation"]
    qc_checks = {
        "verified_failed_source_bundle": bool(
            source.bundle_verification.get("valid")
            and source.bundle_verification.get("status") == "failed"
            and source.bundle_verification.get("tombstoned_file_count") == 0
        ),
        "exact_failed_marker_digest": bool(
            source.failed_marker.get("content_sha256")
            == request.source_failed_marker_content_sha256
        ),
        "source_run_and_queue_failed_identity_verified": bool(
            source.queue_job.get("status") == "failed"
            and source.queue_job.get("failure_category") == "nonzero_exit"
            and source.queue_job.get("job_id") == request.source_queue_job_id
        ),
        "all_registered_source_artifacts_present": bool(
            source.registry_artifacts
            and all(
                receipt.get("status") == "present"
                for receipt in source.registry_artifacts.values()
            )
        ),
        "scientific_config_equal_except_launcher_recovery": True,
        "expected_renderer_failure_signature": True,
        "source_six_core_receipts_verified": bool(source_core_audit["verified"]),
        "source_checkpoint_receipts_match_discovery": bool(
            source_core_audit["checkpoint_receipts_match_current_discovery"]
        ),
        "exact_six_core_set": bool(
            tuple(sorted(canonical.assignments["core_number"].unique().tolist()))
            == tuple(sorted(EXPECTED_CORE_NUMBERS))
        ),
        "every_eligible_cell_exactly_once": bool(
            len(canonical.assignments) == EXPECTED_TOTAL_CELLS
            and not canonical.assignments.duplicated(
                ["core_number", "cell_index"]
            ).any()
        ),
        "no_cross_core_edges": bool(
            canonical.audit["edge_alignment"][
                "core_local_endpoint_bounds_exact"
            ]
        ),
        "attention_edge_index_alignment": bool(
            source_core_audit["strict_attention_shard_count"] > 0
            and canonical.audit["edge_alignment"][
                "edge_index_source_receiver_alignment_exact"
            ]
        ),
        "attention_receiver_head_normalization": bool(
            source_core_audit["maximum_attention_sum_error"] <= 2e-6
        ),
        "attention_matches_combined_logit_softmax": bool(
            source_core_audit["maximum_softmax_reconstruction_error"] <= 2e-6
        ),
        "reciprocal_pairs_and_receiver_degree_adjustment": bool(
            source_core_audit["total_reciprocal_pairs"] * 2
            == source_core_audit["total_directed_edges"]
            and canonical.audit["edge_alignment"]["receiver_indegree_exact"]
            and canonical.audit["edge_alignment"][
                "degree_adjustment_uses_receiver_indegree"
            ]
            and canonical.audit["edge_alignment"][
                "every_mutual_pair_has_two_reversed_exported_edges"
            ]
            and canonical.audit["edge_alignment"][
                "every_directed_edge_covered_by_exactly_one_mutual_pair"
            ]
        ),
        "one_directional_edges_rejected": bool(
            canonical.audit["edge_alignment"]["one_directional_edges_rejected"]
        ),
        "every_final_niche_spatially_connected": bool(
            source_core_audit["all_final_niches_connected"]
        ),
        "disconnected_regions_have_distinct_ids": bool(
            source_core_audit["all_final_niches_connected"]
        ),
        "serialized_region_geometries_valid": bool(
            source_geometry["invalid_after_repair"] == 0
            and source_geometry["maximum_reported_area_error_um2"] <= 1e-4
        ),
        "deterministic_stable_colors": bool(
            all(
                receipt["deterministic_replay"]["downstream_assignment_exact"]
                for receipt in core_receipts
            )
        ),
        "panel_core_labels_and_order": bool(
            panel_receipt.get("core_order") == list(EXPECTED_CORE_NUMBERS)
            and panel_receipt.get("in_panel_core_labels")
            == [f"Core {core}" for core in EXPECTED_CORE_NUMBERS]
            and panel_receipt.get("panel_titles_include_core_number") is True
            and panel_receipt.get("equal_physical_aspect") is True
            and panel_outputs == expected_visual_outputs
        ),
        "micrometre_coordinates_and_scale_bars": bool(
            canonical.assignments["coordinate_unit"].eq("micrometres").all()
            and panel_receipt.get("coordinate_unit") == "micrometres"
            and all(
                value.get("present") and value.get("unit") == "µm"
                for value in panel_receipt.get("scale_bars", {}).values()
            )
        ),
        "deterministic_masks_inference_and_assignments": bool(
            source_core_audit["all_ten_masks_replayed_per_core"]
        ),
        "all_nondegenerate_region_holes_preserved": bool(
            panel_receipt["nondegenerate_region_holes_preserved"]
            and omitted_rings["count"] == 3
            and omitted_rings["scientific_geometry_modified"] is False
        ),
        "canonical_artifacts_are_distinct_inode_ficlone_reflinks": bool(
            all(
                receipt["method"] == "linux_ficlone_reflink"
                and receipt["distinct_inode"]
                and receipt["copy_fallback_permitted"] is False
                for receipt in materialization.values()
            )
        ),
        "source_bundle_unchanged_after_rendering": bool(
            source_post_verification.get("valid")
            and source_post_verification.get("status") == "failed"
        ),
        "checkpoints_unchanged": bool(checkpoint_post["unchanged"]),
        "input_data_unchanged": bool(input_post["unchanged"]),
        "render_only_no_attention_reextraction": True,
    }
    if not all(qc_checks.values()):
        failed = sorted(name for name, passed in qc_checks.items() if not passed)
        raise AttentionNichePipelineError(
            "Render-only recovery QC failed: " + ", ".join(failed)
        )
    qc_fraction = float(sum(qc_checks.values()) / len(qc_checks))
    if qc_fraction != 1.0:
        raise AttentionNichePipelineError(
            "Render-only recovery may publish only with QC fraction 1.0."
        )

    map_label = f"{len(members)}-model ensemble-consensus map"
    launcher = config["launcher"]
    requested_gpu = str(launcher.get("requested_gpu", "0,2,3"))
    reproduction_command = (
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 enqueue-experiment "
        f"--campaign-id {ANALYSIS_CAMPAIGN_ID} "
        "--config experiments/campaigns/"
        "cmp_20260825_six_core_attention_routing_niches/"
        "render_recovery_config.yaml --priority 0 --max-attempts 1 "
        f"--gpu {requested_gpu} && "
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 worker "
        "--worker-id attention-niche-render-recovery "
        f"--gpu {requested_gpu} --min-free-gb 25 --once"
    )
    try:
        recovery_git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=selected_paths.project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AttentionNichePipelineError(
            "Cannot record recovery Git commit."
        ) from exc

    recovery_git_provenance: dict[str, dict[str, Any]] = {}
    for relative in (
        "provenance/git.json",
        "provenance/uncommitted_changes.patch",
        "provenance/untracked_files.json",
    ):
        path = output_root / relative
        if path.is_symlink() or not path.is_file():
            raise AttentionNichePipelineError(
                f"Recovery Git/dirty provenance is absent: {relative}."
            )
        recovery_git_provenance[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    recovery_git_identity = _load_json(
        output_root / "provenance/git.json",
        "recovery Git identity",
    )
    if recovery_git_identity.get("commit") != recovery_git_commit:
        raise AttentionNichePipelineError(
            "Recovery worker Git provenance does not match the live commit."
        )

    core_table = _markdown_core_table(core_receipts)
    omitted_identifiers = ", ".join(
        str(value) for value in omitted_rings["identifiers"]
    )
    readme = f"""# Six-core model-defined attention-routing niche map

Status: complete via a verified render-only recovery. Map type:
`{map_label}`. The canonical assignments, directed attention values, mutual
scores, summaries, colors, regions, and sensitivity results were computed by
failed source run `{request.source_run_id}` and were not recomputed here.
Checkpoints were loaded read-only for identity and loadability verification;
the recovery performed no model instantiation, inference, training, state
mutation, attention extraction, masking, reciprocal-score construction,
clustering, or confidence estimation.
Scientific values recomputed: `false`. Visualizations recomputed: `true`.
Scientific ID: `{RENDER_RECOVERY_SCIENTIFIC_ID}` (unchanged from the source).

{core_table}

The source failed only in the prior renderer, with exact `_FAILED` content
digest `{request.source_failed_marker_content_sha256}`. Its registry run,
queue job `{request.source_queue_job_id}`, complete bundle checksums, six core
receipts, final checkpoints, immutable inputs, and renderer traceback were
verified before rendering directly from the source archive. Only after all
figures succeeded were the seven canonical scientific products materialized
as Linux FICLONE reflinks with distinct inodes into this new run bundle; no
hard links, symlinks, byte-copy fallback, or source mutation was allowed.

All nondegenerate region holes are preserved in the maps. Three exact-zero-area
interior rings have no fill effect and were omitted only from Matplotlib paths:
`{omitted_identifiers}`. Their full renderer receipt and digest are recorded in
`analysis_manifest.yaml`; the GeoJSON scientific geometry was not modified.

Reproduce as a new registered render-only recovery run:

```bash
{reproduction_command}
```

> {LIMITATION}
"""
    _write_text_atomic(output_root / "README.md", readme)

    qc_lines = [
        "# Attention-routing niche render-recovery QC",
        "",
        f"Overall QC pass fraction: `{qc_fraction:.6f}` "
        f"({sum(qc_checks.values())}/{len(qc_checks)}).",
        "",
        core_table,
        "",
        "## Recovery lineage",
        "",
        f"- Failed source run: `{request.source_run_id}`.",
        f"- Failed source queue job: `{request.source_queue_job_id}`.",
        "- Full source bundle verification: PASS; no retention tombstones.",
        f"- Exact `_FAILED` content digest: "
        f"`{request.source_failed_marker_content_sha256}`.",
        f"- Scientific ID: `{RENDER_RECOVERY_SCIENTIFIC_ID}` (unchanged).",
        "- Scientific configuration equality excluding only launcher runtime "
        "and recovery controls: PASS.",
        "- Attention extraction/recomputation in this recovery: `false`.",
        "- Scientific values recomputed: `false`.",
        "- Visualizations recomputed: `true`.",
        "- Seven canonical products: FICLONE reflinks, distinct inodes, "
        "link count one, equal source checksums, and no fallback.",
        "",
        "## Source extraction evidence",
        "",
        f"- Strict receiver-complete shards: "
        f"`{source_core_audit['strict_attention_shard_count']:,}`.",
        f"- Maximum receiver/head normalization error: "
        f"`{source_core_audit['maximum_attention_sum_error']:.3g}`.",
        f"- Maximum softmax reconstruction error: "
        f"`{source_core_audit['maximum_softmax_reconstruction_error']:.3g}`.",
        "- Checkpoint/model-state receipts match fresh catalog discovery: PASS.",
        "- All ten masks, downstream assignments, colors, and connected-niche "
        "checks are inherited from and verified against the source receipts.",
        "",
        "## Rendering geometry",
        "",
        "- All nondegenerate interior rings were preserved.",
        f"- Exact-zero interior rings omitted from render paths: "
        f"`{omitted_rings['count']}`.",
        f"- Omitted-ring identifier digest: "
        f"`{omitted_rings['identifier_list_sha256']}`.",
        "- The source GeoJSON and niche memberships were not modified.",
        "",
        "## Explicit checks",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'} — `{name}`"
            for name, passed in qc_checks.items()
        ],
        "",
        "## Interpretation limit",
        "",
        f"> {LIMITATION}",
        "",
    ]
    _write_text_atomic(
        output_root / "analysis_qc_report.md",
        "\n".join(qc_lines),
    )

    package_versions: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "igraph",
        "leidenalg",
        "shapely",
        "matplotlib",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"

    required_outputs = list(REQUIRED_ANALYSIS_OUTPUTS)
    artifact_records = _artifact_checksums(
        output_root,
        [
            value
            for value in required_outputs
            if value != "analysis_manifest.yaml"
        ]
        + ["attention_niche_parameter_sensitivity.csv"],
    )
    portable_input_post = json.loads(_canonical_json(input_post))
    post_receipts = portable_input_post.get("post_receipts", {})
    if not isinstance(post_receipts, Mapping):
        raise AttentionNichePipelineError(
            "Recovery post-analysis immutable-input receipts are malformed."
        )
    portable_input_post["post_receipts"] = _portable_file_receipts(post_receipts)
    portable_source_verification = dict(source.bundle_verification)
    portable_source_verification["run_path"] = _relative_to_root(
        source.root, selected_paths.project_root
    )
    portable_source_post = dict(source_post_verification)
    portable_source_post["run_path"] = _relative_to_root(
        source.root, selected_paths.project_root
    )
    analysis_manifest = {
        "schema": "six_core_attention_routing_niche_render_recovery_v1",
        "status": "complete",
        "run_id": run_id,
        "campaign_id": ANALYSIS_CAMPAIGN_ID,
        "execution_mode": RENDER_RECOVERY_MODE,
        "map_label": map_label,
        "terminology": "model-defined attention-routing niches",
        "scientific_id": RENDER_RECOVERY_SCIENTIFIC_ID,
        "created_at": _utc_now(),
        "limitation": LIMITATION,
        "core_order": list(EXPECTED_CORE_NUMBERS),
        "core_aliases": list(EXPECTED_ALIASES),
        "model_seeds": [member.seed for member in members],
        "attention_extraction_performed": False,
        "analysis_masks_generated_or_replayed": False,
        "scientific_tables_recomputed": False,
        "scientific_values_recomputed": False,
        "clustering_or_confidence_recomputed": False,
        "visualizations_recomputed": True,
        "rendered_directly_from_verified_source_before_materialization": True,
        "source_lineage": {
            "source_run_id": request.source_run_id,
            "source_queue_job": dict(source.queue_job),
            "source_bundle": _relative_to_root(
                source.root, selected_paths.project_root
            ),
            "source_bundle_pre_verification": portable_source_verification,
            "source_bundle_post_verification": portable_source_post,
            "failed_marker": dict(source.failed_marker),
            "failed_marker_file_sha256": source.failed_marker_file_sha256,
            "artifact_checksum_manifest_sha256": (
                source.artifact_checksum_manifest_sha256
            ),
            "stderr_sha256": source.stderr_sha256,
            "expected_renderer_failure_signature": (
                request.expected_renderer_failure_signature
            ),
            "scientific_config_sha256": source.scientific_config_sha256,
            "scientific_id": RENDER_RECOVERY_SCIENTIFIC_ID,
            "scientific_config_equal_except_launcher": True,
            "source_git_identity": dict(source.source_git_identity),
            "source_git_and_dirty_provenance": dict(
                source.source_git_provenance
            ),
        },
        "recovery_provenance": {
            "git_commit": recovery_git_commit,
            "git_identity": recovery_git_identity,
            "git_and_dirty_provenance": recovery_git_provenance,
            "source_and_recovery_provenance_are_distinct": True,
        },
        "checkpoint_members": [
            member.manifest_record(selected_paths.project_root)
            for member in members
        ],
        "checkpoint_post_recovery": checkpoint_post,
        "input_contract": {
            "cohort_manifest_sha256": EXPECTED_COHORT_MANIFEST_SHA256,
            "graph_manifest_sha256": EXPECTED_GRAPH_MANIFEST_SHA256,
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "gene_order_sha256": EXPECTED_GENE_SCHEMA_SHA256,
            "metadata_order_sha256": EXPECTED_METADATA_SCHEMA_SHA256,
            "preprocessing_version": EXPECTED_PREPROCESSING_VERSION,
            "pre_recovery_file_receipts": _portable_file_receipts(
                input_contract.pre_analysis_file_receipts
            ),
            "post_recovery": portable_input_post,
        },
        "source_core_receipt_audit": source_core_audit,
        "source_core_receipts": _portable_core_receipts(core_receipts),
        "canonical_source_audit": dict(canonical.audit),
        "canonical_materialization": materialization,
        "visualization_receipt": raw_visualization_receipt,
        "filesystem_contract": {
            "devices": filesystem_devices,
            "same_filesystem": True,
            "archive_publication_requires_atomic_rename": True,
        },
        "resource_execution": {
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_attention_extraction_performed": False,
            "peak_vram_gib": 0.0,
            "filesystem_snapshots": disk_snapshots,
            "parent_peak_host_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024
                / (1024**3)
            ),
        },
        "package_versions": package_versions,
        "qc_checks": qc_checks,
        "qc_pass_fraction": qc_fraction,
        "artifacts": artifact_records,
        "reproduction_command": reproduction_command,
    }
    _write_yaml_atomic(output_root / "analysis_manifest.yaml", analysis_manifest)
    _artifact_checksums(output_root, required_outputs)

    final_metrics = {
        "analysis/attention_niche_qc_pass_fraction": qc_fraction,
        "analysis/total_cells": float(EXPECTED_TOTAL_CELLS),
        "analysis/retained_mutual_edges": float(
            sum(int(receipt["retained_mutual_edge_count"]) for receipt in core_receipts)
        ),
        "analysis/final_connected_niches": float(
            sum(int(receipt["final_connected_niche_count"]) for receipt in core_receipts)
        ),
        "analysis/render_only_recovery": 1.0,
    }
    _write_json_atomic(output_root / "metrics" / "final.json", final_metrics)
    _write_text_atomic(
        output_root / "metrics" / "history.jsonl",
        _canonical_json(
            {"step": 0, **final_metrics, "analysis_complete": True}
        )
        + "\n",
    )
    _write_text_atomic(
        output_root / "metrics" / "events.jsonl",
        "".join(
            _canonical_json({"name": name, "value": value, "step": 0}) + "\n"
            for name, value in final_metrics.items()
        ),
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "analysis_kind": "six_core_attention_routing_niche_map",
        "execution_mode": RENDER_RECOVERY_MODE,
        "source_run_id": request.source_run_id,
        "attention_extraction_performed": False,
        "scientific_tables_recomputed": False,
        "scientific_values_recomputed": False,
        "visualizations_recomputed": True,
        "map_label": map_label,
        "primary_metric_name": "analysis/attention_niche_qc_pass_fraction",
        "primary_metric_value": qc_fraction,
        "metrics": final_metrics,
        "model_seed_count": len(members),
        "model_seeds": [member.seed for member in members],
        "core_numbers": list(EXPECTED_CORE_NUMBERS),
        "peak_vram_gib": 0.0,
        "parameter_count": members[0].parameter_count,
        "limitation": LIMITATION,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def run_attention_niche_pipeline(
    *,
    config_path: str | Path,
    run_id: str,
    run_scratch: str | Path,
    database: str | Path,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Execute and publish every in-scratch six-core analysis artifact."""

    import importlib.metadata
    import multiprocessing
    import resource
    import subprocess

    from .attention_niche_geometry import validate_and_repair_regions_geojson
    from .attention_niche_visualization import (
        render_attention_niche_visualizations,
    )
    from .configuration import load_yaml_mapping, validate_experiment_config

    selected_paths = paths or current_paths()
    root = selected_paths.project_root.resolve()
    output_root = Path(run_scratch).resolve()
    expected_output = (selected_paths.scratch_root / "active_runs" / run_id).resolve()
    if output_root != expected_output or not output_root.is_dir():
        raise AttentionNichePipelineError(
            "Analysis output must be the worker-owned active run directory."
        )
    config = load_yaml_mapping(Path(config_path))
    validate_experiment_config(config)
    campaign = config.get("campaign", {})
    metadata = config.get("metadata", {})
    evaluation = config.get("evaluation", {})
    if (
        not isinstance(campaign, Mapping)
        or campaign.get("campaign_id") != ANALYSIS_CAMPAIGN_ID
        or not isinstance(metadata, Mapping)
        or not isinstance(evaluation, Mapping)
        or evaluation.get("protocol") != "posthoc_attention_routing_niche_v1"
        or evaluation.get("artifact_contract") != "analysis_only"
    ):
        raise AttentionNichePipelineError("Resolved analysis configuration drifted.")
    analysis_parameters = _validated_locked_analysis_parameters(metadata)
    launcher = config.get("launcher", {})
    if not isinstance(launcher, Mapping):
        raise AttentionNichePipelineError("Resolved launcher configuration is absent.")
    recovery_request = _validated_render_recovery_request(launcher)
    visualization_patch_request = _validated_visualization_patch_request(launcher)
    if recovery_request is not None and visualization_patch_request is not None:
        raise AttentionNichePipelineError(
            "Render recovery and visualization patch modes are mutually exclusive."
        )
    if visualization_patch_request is not None:
        return _run_attention_niche_visualization_patch(
            request=visualization_patch_request,
            config=config,
            run_id=run_id,
            output_root=output_root,
            database=database,
            selected_paths=selected_paths,
        )

    dataset = config.get("dataset", {})
    if not isinstance(dataset, Mapping):
        raise AttentionNichePipelineError("Resolved dataset configuration is absent.")

    def rooted(value: object, *, label: str) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            candidate = path
        elif path.parts and path.parts[0] == "data":
            candidate = selected_paths.data_root.joinpath(*path.parts[1:])
        else:
            candidate = root / path
        try:
            return candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise AttentionNichePipelineError(
                f"Configured {label} does not exist: {candidate}."
            ) from exc

    cohort_dir = rooted(dataset.get("prepared_artifact"), label="prepared cohort")
    graph_dir = rooted(
        dataset.get("prepared_graph_artifact"), label="prepared graph"
    )
    raw_dir = selected_paths.data_root / "raw"
    core_map_path = selected_paths.data_root / "clinical" / "fov_core_map.csv"
    reconciliation_path = rooted(
        dataset.get("clinical_reconciliation_policy"),
        label="clinical reconciliation policy",
    )
    members = discover_completed_checkpoint_members(
        database,
        project_root=root,
    )
    input_contract = verify_prepared_input_contract(
        paths=selected_paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
        raw_dir=raw_dir,
        core_map_path=core_map_path,
        reconciliation_path=reconciliation_path,
    )
    if recovery_request is not None:
        return _run_attention_niche_render_recovery(
            request=recovery_request,
            config=config,
            analysis_parameters=analysis_parameters,
            run_id=run_id,
            output_root=output_root,
            database=database,
            selected_paths=selected_paths,
            members=members,
            input_contract=input_contract,
        )
    disk_plan = _required_analysis_free_disk_gib(
        model_seed_count=len(members),
        minimum_free_disk_gib=analysis_parameters["minimum_free_disk_gib"],
    )
    disk_snapshots = [_filesystem_snapshot(output_root, stage="preflight")]
    if disk_snapshots[-1]["free_gib"] < disk_plan["required_free_gib"]:
        raise AttentionNichePipelineError(
            "Insufficient free disk for the seed-aware streamed analysis plan: "
            f"{disk_snapshots[-1]['free_gib']:.2f} GiB available, "
            f"{disk_plan['required_free_gib']:.2f} GiB required."
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise AttentionNichePipelineError("Full analysis requires at least one CUDA GPU.")
    requested_count = (
        int(launcher.get("requested_gpu_count", torch.cuda.device_count()))
        if isinstance(launcher, Mapping)
        else torch.cuda.device_count()
    )
    visible_device_count = torch.cuda.device_count()
    if visible_device_count != requested_count:
        raise AttentionNichePipelineError(
            "Worker CUDA visibility does not match launcher.requested_gpu_count "
            f"({visible_device_count} != {requested_count})."
        )
    gpu_records = []
    for device_index in range(visible_device_count):
        properties = torch.cuda.get_device_properties(device_index)
        gpu_records.append(
            {
                "local_device": device_index,
                "name": properties.name,
                "total_memory_gib": properties.total_memory / (1024**3),
                "capability": list(properties.major_minor)
                if hasattr(properties, "major_minor")
                else [properties.major, properties.minor],
            }
        )

    work_root = output_root / "diagnostics" / "attention_niche_core_work"
    work_root.mkdir(parents=True, exist_ok=False)
    mask_seed = analysis_parameters["analysis_mask_seed"]
    mask_views = analysis_parameters["analysis_mask_views"]
    specs_by_device: list[list[CoreJobSpec]] = [
        [] for _ in range(visible_device_count)
    ]
    for core_position, (alias, core_number) in enumerate(
        zip(EXPECTED_ALIASES, EXPECTED_CORE_NUMBERS, strict=True)
    ):
        local_device = core_position % visible_device_count
        specs_by_device[local_device].append(
            CoreJobSpec(
                alias=alias,
                core_number=core_number,
                local_device=local_device,
                output_dir=work_root / f"core_{core_number:02d}",
                cohort_dir=cohort_dir,
                graph_dir=graph_dir,
                raw_dir=raw_dir,
                core_map_path=core_map_path,
                reconciliation_path=reconciliation_path,
                members=members,
                analysis_mask_seed=mask_seed,
                mask_views=mask_views,
                leiden_seed=analysis_parameters["leiden_seed"],
                color_seed=analysis_parameters["color_seed"],
                uniform_routing_threshold=analysis_parameters[
                    "uniform_routing_threshold"
                ],
                score_threshold=analysis_parameters[
                    "consensus_mutual_score_threshold"
                ],
                support_threshold=analysis_parameters["support_threshold"],
                primary_top_k=analysis_parameters["primary_top_neighbors"],
                primary_resolution=analysis_parameters[
                    "primary_leiden_resolution"
                ],
                polygon_coordinate_alignment_rule=analysis_parameters[
                    "polygon_coordinate_alignment_rule"
                ],
                polygon_centroid_tolerance_um=analysis_parameters[
                    "polygon_centroid_tolerance_um"
                ],
                max_spatial_gap_um=analysis_parameters["spatial_max_gap_um"],
                micro_niche_threshold=analysis_parameters[
                    "micro_niche_cell_threshold"
                ],
                amp=False,
                deterministic_replay=True,
            )
        )
    parallel_schedule = {
        str(device): [spec.alias for spec in specs]
        for device, specs in enumerate(specs_by_device)
        if specs
    }
    core_receipts: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=sum(bool(specs) for specs in specs_by_device),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_run_device_job_group, tuple(specs)): device
            for device, specs in enumerate(specs_by_device)
            if specs
        }
        for future in as_completed(futures):
            device = futures[future]
            try:
                core_receipts.extend(future.result())
            except BaseException as exc:
                raise AttentionNichePipelineError(
                    f"Core job group failed on local CUDA device {device}."
                ) from exc
    core_receipts.sort(key=lambda value: int(value["core_number"]))
    if tuple(int(value["core_number"]) for value in core_receipts) != EXPECTED_CORE_NUMBERS:
        raise AttentionNichePipelineError("Parallel core execution did not cover six cores.")
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="core_shards_complete")
    )

    assignment_sources = [
        Path(receipt["outputs"]["assignments"]) for receipt in core_receipts
    ]
    mutual_sources = [
        Path(receipt["outputs"]["mutual_edges"]) for receipt in core_receipts
    ]
    directed_sources = [
        Path(receipt["outputs"]["directed_edges"]) for receipt in core_receipts
    ]
    assignment_contracts = [
        {
            "core_number": receipt["core_number"],
            "core_alias": receipt["core_alias"],
            "row_count": receipt["cell_count"],
            "position_column": "cell_index",
        }
        for receipt in core_receipts
    ]
    mutual_contracts = [
        {
            "core_number": receipt["core_number"],
            "core_alias": receipt["core_alias"],
            "row_count": receipt["reciprocal_pair_count"],
            "position_column": "mutual_pair_position",
        }
        for receipt in core_receipts
    ]
    directed_contracts = [
        {
            "core_number": receipt["core_number"],
            "core_alias": receipt["core_alias"],
            "row_count": receipt["directed_edge_count"],
            "position_column": "edge_index_position",
        }
        for receipt in core_receipts
    ]
    combined_receipts = {
        "assignments": _combine_parquet_files(
            assignment_sources,
            output_root / "cell_attention_niche_assignments.parquet",
            source_contracts=assignment_contracts,
            remove_sources_after_write=True,
            staging_root=work_root,
        ),
        "mutual_edges": _combine_parquet_files(
            mutual_sources,
            output_root / "mutual_attention_edges.parquet",
            source_contracts=mutual_contracts,
            remove_sources_after_write=True,
            staging_root=work_root,
        ),
        "directed_edges": _combine_parquet_files(
            directed_sources,
            output_root / "directed_attention_edges.parquet",
            source_contracts=directed_contracts,
            remove_sources_after_write=True,
            staging_root=work_root,
        ),
    }
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="canonical_tables_complete")
    )
    assignments = pd.read_parquet(
        output_root / "cell_attention_niche_assignments.parquet"
    )
    summaries = pd.concat(
        [pd.read_csv(receipt["outputs"]["summary"]) for receipt in core_receipts],
        ignore_index=True,
    )
    _write_dataframe_csv_atomic(
        output_root / "attention_niche_summary.csv", summaries
    )
    sensitivity = pd.concat(
        [
            pd.read_csv(receipt["outputs"]["sensitivity"])
            for receipt in core_receipts
        ],
        ignore_index=True,
    )
    _write_dataframe_csv_atomic(
        output_root / "attention_niche_parameter_sensitivity.csv", sensitivity
    )
    colors: dict[str, str] = {}
    features: list[dict[str, Any]] = []
    overlay_frames: list[pd.DataFrame] = []
    for receipt in core_receipts:
        current_colors = _load_json(
            Path(receipt["outputs"]["colors"]), "per-core niche colors"
        )
        overlap = set(colors).intersection(current_colors)
        if overlap:
            raise AttentionNichePipelineError("Core-scoped niche IDs collided.")
        colors.update({str(key): str(value) for key, value in current_colors.items()})
        current_geojson = _load_json(
            Path(receipt["outputs"]["regions"]), "per-core niche regions"
        )
        features.extend(current_geojson.get("features", []))
    for core_number in EXPECTED_CORE_NUMBERS:
        overlay_frames.append(
            pd.read_parquet(
                output_root / "mutual_attention_edges.parquet",
                columns=[
                    "core_number",
                    "cell_i_index",
                    "cell_j_index",
                    "M_ij",
                    "support_P_ij",
                    "retained_primary",
                ],
                filters=[
                    ("core_number", "=", core_number),
                    ("retained_primary", "=", True),
                ],
            )
        )
    palette_by_core: dict[int, set[str]] = {}
    for niche_id, color in colors.items():
        core_number = int(niche_id[1:3])
        palette_by_core.setdefault(core_number, set()).add(color.lower())
    for first_core, first_palette in palette_by_core.items():
        for second_core, second_palette in palette_by_core.items():
            if first_core < second_core and first_palette.intersection(second_palette):
                raise AttentionNichePipelineError(
                    "A categorical niche color was reused across cores."
                )
    _write_json_atomic(output_root / "attention_niche_colors.json", colors)
    regions_geojson = {
        "type": "FeatureCollection",
        "coordinate_unit": "um",
        "features": sorted(
            features,
            key=lambda value: (
                int(value["properties"]["core_number"]),
                str(value["properties"]["final_niche_id"]),
            ),
        ),
    }
    regions_geojson, combined_geometry_validation = (
        validate_and_repair_regions_geojson(regions_geojson)
    )
    regions_geojson["geometry_validation"] = combined_geometry_validation
    if (
        int(combined_geometry_validation["feature_count"])
        != sum(int(receipt["final_connected_niche_count"]) for receipt in core_receipts)
        or int(combined_geometry_validation["invalid_after_repair"]) != 0
        or float(combined_geometry_validation["maximum_reported_area_error_um2"])
        > 1e-4
    ):
        raise AttentionNichePipelineError(
            "Combined serialized region geometry validation failed."
        )
    _write_json_atomic(
        output_root / "attention_niche_regions.geojson", regions_geojson
    )
    overlay_edges = pd.concat(overlay_frames, ignore_index=True)
    visualization = render_attention_niche_visualizations(
        assignments,
        regions_geojson,
        overlay_edges,
        output_root,
        dpi=300,
        individual_dpi=450,
        include_overlay=True,
        max_edges_per_core=1_000,
        max_edges_total=4_000,
        invert_y=True,
        low_confidence_threshold=0.60,
        allow_cross_core_color_reuse=False,
    )
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="visualizations_complete")
    )

    input_post = verify_inputs_unchanged(
        input_contract.pre_analysis_file_receipts,
        project_root=root,
        data_root=selected_paths.data_root,
    )
    checkpoint_post = _checkpoint_post_receipts(members, root=root)
    expected_counts = {
        int(receipt["core_number"]): int(receipt["cell_count"])
        for receipt in core_receipts
    }
    observed_cores = tuple(sorted(assignments["core_number"].unique().tolist()))
    if observed_cores != tuple(sorted(EXPECTED_CORE_NUMBERS)):
        raise AttentionNichePipelineError("Combined assignments have wrong cores.")
    for core_number, expected_count in expected_counts.items():
        selected = assignments.loc[assignments["core_number"] == core_number]
        if (
            len(selected) != expected_count
            or selected["cell_index"].nunique() != expected_count
            or not np.array_equal(
                np.sort(selected["cell_index"].to_numpy(dtype=np.int64)),
                np.arange(expected_count, dtype=np.int64),
            )
        ):
            raise AttentionNichePipelineError(
                f"Assignment coverage failed for core {core_number}."
            )
    if len(assignments) != EXPECTED_TOTAL_CELLS:
        raise AttentionNichePipelineError("Combined assignment total drifted.")
    if combined_receipts["directed_edges"]["row_count"] != EXPECTED_TOTAL_DIRECTED_EDGES:
        raise AttentionNichePipelineError("Combined directed-edge total drifted.")
    if combined_receipts["mutual_edges"]["row_count"] * 2 != EXPECTED_TOTAL_DIRECTED_EDGES:
        raise AttentionNichePipelineError("Reciprocal-pair total drifted.")
    core_qc_passed = all(
        receipt["local_spatial_contiguity"]["every_final_niche_connected"]
        and receipt["serialized_region_geometry_validation"][
            "invalid_after_repair"
        ]
        == 0
        and receipt["serialized_region_geometry_validation"][
            "maximum_reported_area_error_um2"
        ]
        <= 1e-4
        and receipt["extraction_audit"]["max_attention_sum_error"] <= 2e-6
        and receipt["extraction_audit"]["max_softmax_reconstruction_error"]
        <= 2e-6
        and receipt["extraction_audit"]["strict_attention_shard_count"] > 0
        and len(receipt["extraction_audit"]["strict_view_receipts"])
        == len(members) * (mask_views + 1)
        and receipt["extraction_audit"]["deterministic_inference_replay"][
            "exact_array_equal"
        ]
        and receipt["extraction_audit"]["deterministic_inference_replay"][
            "all_ten_mask_views_replayed"
        ]
        and receipt["deterministic_replay"]["downstream_assignment_exact"]
        for receipt in core_receipts
    )
    if not core_qc_passed:
        raise AttentionNichePipelineError("One or more core QC gates failed.")
    if visualization.receipt.get("core_order") != list(EXPECTED_CORE_NUMBERS):
        raise AttentionNichePipelineError("Visualization core order drifted.")

    exact_core_set = observed_cores == tuple(sorted(EXPECTED_CORE_NUMBERS))
    exact_assignment_coverage = len(assignments) == EXPECTED_TOTAL_CELLS and all(
        len(assignments.loc[assignments["core_number"] == core_number])
        == expected_count
        for core_number, expected_count in expected_counts.items()
    )
    strict_attention_alignment = all(
        receipt["extraction_audit"]["strict_attention_shard_count"] > 0
        and receipt["extraction_audit"]["max_softmax_reconstruction_error"]
        <= 2e-6
        for receipt in core_receipts
    )
    exact_reciprocal_coverage = all(
        int(receipt["reciprocal_pair_count"]) * 2
        == int(receipt["directed_edge_count"])
        for receipt in core_receipts
    )
    connected_niches = all(
        receipt["local_spatial_contiguity"]["every_final_niche_connected"]
        for receipt in core_receipts
    )
    deterministic_colors = all(
        receipt["deterministic_replay"]["downstream_assignment_exact"]
        and len(receipt["deterministic_replay"]["color_mapping_sha256"]) == 64
        for receipt in core_receipts
    )
    panel_receipt = visualization.receipt
    panel_labels = (
        panel_receipt.get("core_order") == list(EXPECTED_CORE_NUMBERS)
        and panel_receipt.get("in_panel_core_labels")
        == [f"Core {core}" for core in EXPECTED_CORE_NUMBERS]
        and panel_receipt.get("panel_titles_include_core_number") is True
        and panel_receipt.get("equal_physical_aspect") is True
    )
    micrometre_geometry = (
        assignments["coordinate_unit"].eq("micrometres").all()
        and np.isfinite(assignments[["x_um", "y_um"]].to_numpy()).all()
        and panel_receipt.get("coordinate_unit") == "micrometres"
        and all(
            value.get("present") and value.get("unit") == "µm"
            for value in panel_receipt.get("scale_bars", {}).values()
        )
    )
    deterministic_analysis_replay = all(
        receipt["extraction_audit"]["deterministic_inference_replay"][
            "all_ten_mask_views_replayed"
        ]
        and receipt["extraction_audit"]["deterministic_inference_replay"][
            "exact_array_equal"
        ]
        and receipt["deterministic_replay"]["downstream_assignment_exact"]
        for receipt in core_receipts
    )
    qc_checks = {
        "exact_six_core_set": bool(exact_core_set),
        "every_eligible_cell_exactly_once": bool(exact_assignment_coverage),
        "no_cross_core_edges": bool(
            exact_core_set
            and all(
                receipt["core_alias"] in EXPECTED_ALIASES
                for receipt in core_receipts
            )
            and all(
                table["source_contracts_verified"]
                for table in combined_receipts.values()
            )
        ),
        "attention_edge_index_alignment": bool(strict_attention_alignment),
        "attention_receiver_head_normalization": bool(
            all(
                receipt["extraction_audit"]["max_attention_sum_error"] <= 2e-6
                for receipt in core_receipts
            )
        ),
        "attention_matches_combined_logit_softmax": bool(
            strict_attention_alignment
        ),
        "reciprocal_pairs_and_receiver_degree_adjustment": bool(
            exact_reciprocal_coverage
        ),
        "one_directional_edges_rejected": bool(exact_reciprocal_coverage),
        "every_final_niche_spatially_connected": bool(connected_niches),
        "disconnected_regions_have_distinct_ids": bool(connected_niches),
        "serialized_region_geometries_valid": bool(
            combined_geometry_validation["invalid_after_repair"] == 0
            and combined_geometry_validation["feature_count"]
            == sum(
                int(receipt["final_connected_niche_count"])
                for receipt in core_receipts
            )
            and combined_geometry_validation["maximum_reported_area_error_um2"]
            <= 1e-4
        ),
        "deterministic_stable_colors": bool(deterministic_colors),
        "panel_core_labels_and_order": bool(panel_labels),
        "micrometre_coordinates_and_scale_bars": bool(micrometre_geometry),
        "deterministic_masks_inference_and_assignments": bool(
            deterministic_analysis_replay
        ),
        "checkpoints_unchanged": bool(checkpoint_post["unchanged"]),
        "input_data_unchanged": bool(input_post["unchanged"]),
    }
    if not all(qc_checks.values()):
        failed = sorted(name for name, passed in qc_checks.items() if not passed)
        raise AttentionNichePipelineError(
            "Final analysis QC failed: " + ", ".join(failed)
        )
    qc_fraction = float(sum(qc_checks.values()) / len(qc_checks))
    map_label = (
        "single-model, mask-consensus map"
        if len(members) == 1
        else f"{len(members)}-model ensemble-consensus map"
    )
    required_outputs = list(REQUIRED_ANALYSIS_OUTPUTS)
    reproducibility_command = (
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 enqueue-experiment "
        f"--campaign-id {ANALYSIS_CAMPAIGN_ID} "
        "--config experiments/campaigns/"
        "cmp_20260825_six_core_attention_routing_niches/analysis_config.yaml "
        "--priority 0 --max-attempts 1 "
        f"--gpu {str(launcher.get('requested_gpu', '0'))} && "
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark "
        "--database state/tracking/bagm.sqlite3 worker "
        "--worker-id attention-niche-multigpu "
        f"--gpu {str(launcher.get('requested_gpu', '0'))} --once"
    )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AttentionNichePipelineError("Cannot record the analysis Git commit.") from exc
    package_versions: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "igraph",
        "leidenalg",
        "shapely",
        "matplotlib",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"

    # Write narrative files before their presence is included in the manifest
    # gate.  Complete tables remain machine-readable beside this concise view.
    core_table = _markdown_core_table(core_receipts)
    readme = f"""# Six-core model-defined attention-routing niche map

Status: complete. Map type: `{map_label}`. Primary parameters were locked at
ten analysis masks, mutual score `Mij > 1.0`, support `Pij >= 0.60`, per-cell
top-8 undirected union, weighted Leiden resolution `1.0`, and fixed Leiden seed
`2026082502` before plots were inspected.

{core_table}

The primary combined map is `six_core_attention_niche_map.png`; vector versions,
per-core high-resolution maps, the strongest-edge overlay, full directed and
mutual routing exports, assignments, dissolved regions, colors, sensitivity
results, and QC evidence are in this immutable run bundle.

The directed export retains exact seed-by-mask head-mean routing for every
directed edge and per-seed/per-head mask-mean attention, content QK score,
positional bias, and combined logit. Gray cell outlines in the maps mark
assignment confidence below 0.60; cell fill remains the assigned niche color.

Reproduce as a new registered immutable run:

```bash
{reproducibility_command}
```

> {LIMITATION}
"""
    descriptor, temporary_readme = tempfile.mkstemp(
        prefix=".README.md.tmp-", dir=output_root
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(readme)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_readme, output_root / "README.md")
    maximum_attention_error = max(
        float(receipt["extraction_audit"]["max_attention_sum_error"])
        for receipt in core_receipts
    )
    maximum_softmax_error = max(
        float(
            receipt["extraction_audit"][
                "max_softmax_reconstruction_error"
            ]
        )
        for receipt in core_receipts
    )
    strict_shard_count = sum(
        int(receipt["extraction_audit"]["strict_attention_shard_count"])
        for receipt in core_receipts
    )
    qc_lines = [
        "# Attention-routing niche analysis QC",
        "",
        f"Overall QC pass fraction: `{qc_fraction:.6f}` ({sum(qc_checks.values())}/{len(qc_checks)}).",
        "",
        core_table,
        "",
        "## Explicit checks",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'} — `{name}`"
            for name, passed in qc_checks.items()
        ],
        "",
        "## Extraction and reproducibility evidence",
        "",
        f"- Strict receiver-complete attention shards audited: `{strict_shard_count:,}`.",
        f"- Maximum receiver/head normalization error: `{maximum_attention_error:.3g}`.",
        f"- Maximum reconstructed-softmax error: `{maximum_softmax_error:.3g}`.",
        "- All ten analysis-mask views were regenerated and re-extracted for "
        "the fixed reference model in every core with exact routing-array equality.",
        "- Mask receipts were identical across all model seeds; the complete "
        "consensus partition and color mapping were independently rebuilt exactly.",
        "- Canonical Parquet consolidation verified every core number, alias, "
        "row count, and complete edge/pair/cell position range before removing "
        "temporary source shards.",
        "- Input and checkpoint hashes were checked before and after analysis.",
        "",
        "## Scope",
        "",
        "This is a transductive post-hoc model-behavior analysis. Seed and mask "
        "agreement quantify computational stability, not independent biological "
        "replication. The all-genes-visible replay is a masking sensitivity check, "
        "not a mechanism-breaking null.",
        "",
        "## Interpretation limit",
        "",
        f"> {LIMITATION}",
        "",
    ]
    descriptor, temporary_qc = tempfile.mkstemp(
        prefix=".analysis_qc_report.md.tmp-", dir=output_root
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(qc_lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_qc, output_root / "analysis_qc_report.md")

    checksummed_before_manifest = [
        value
        for value in required_outputs
        if value not in {"analysis_manifest.yaml"}
    ] + ["attention_niche_parameter_sensitivity.csv"]
    artifact_records = _artifact_checksums(
        output_root, checksummed_before_manifest
    )
    portable_core_receipts = _portable_core_receipts(core_receipts)
    portable_combined_receipts = json.loads(_canonical_json(combined_receipts))
    for table_receipt in portable_combined_receipts.values():
        table_receipt["path"] = _relative_to_root(
            Path(table_receipt["path"]), output_root
        )
    portable_visualization_receipt = json.loads(
        _canonical_json(dict(visualization.receipt))
    )
    portable_visualization_receipt["output_paths"] = [
        _relative_to_root(Path(path), output_root)
        for path in portable_visualization_receipt.get("output_paths", [])
    ]
    portable_input_post = json.loads(_canonical_json(input_post))
    post_receipts = portable_input_post.get("post_receipts", {})
    if not isinstance(post_receipts, Mapping):
        raise AttentionNichePipelineError(
            "Post-analysis immutable-input receipts are malformed."
        )
    portable_input_post["post_receipts"] = _portable_file_receipts(post_receipts)
    staging_cleanup = _remove_verified_core_work_tree(
        work_root,
        output_root=output_root,
    )
    disk_snapshots.append(
        _filesystem_snapshot(output_root, stage="post_staging_cleanup")
    )
    analysis_manifest = {
        "schema": "six_core_attention_routing_niche_analysis_v1",
        "status": "complete",
        "run_id": run_id,
        "campaign_id": ANALYSIS_CAMPAIGN_ID,
        "map_label": map_label,
        "terminology": "model-defined attention-routing niches",
        "git_commit": git_commit,
        "created_at": _utc_now(),
        "limitation": LIMITATION,
        "core_order": list(EXPECTED_CORE_NUMBERS),
        "core_aliases": list(EXPECTED_ALIASES),
        "model_seeds": [member.seed for member in members],
        "checkpoint_members": [member.manifest_record(root) for member in members],
        "analysis_parameters": {
            "analysis_mask_seed": mask_seed,
            "mask_views": list(range(mask_views)),
            "all_genes_visible_sensitivity": True,
            "final_graph_layer": True,
            "float32_no_autocast": True,
            "uniform_support_event_threshold_strict": analysis_parameters[
                "uniform_routing_threshold"
            ],
            "mutual_score_retention_threshold_strict": analysis_parameters[
                "consensus_mutual_score_threshold"
            ],
            "support_threshold": analysis_parameters["support_threshold"],
            "primary_top_k": analysis_parameters["primary_top_neighbors"],
            "top_k_sensitivity": list(metadata["top_neighbor_sensitivity"]),
            "primary_leiden_resolution": analysis_parameters[
                "primary_leiden_resolution"
            ],
            "leiden_resolution_sensitivity": list(
                metadata["leiden_resolution_sensitivity"]
            ),
            "leiden_seed": analysis_parameters["leiden_seed"],
            "color_seed": analysis_parameters["color_seed"],
            "micro_niche_threshold_cells": analysis_parameters[
                "micro_niche_cell_threshold"
            ],
            "polygon_coordinate_alignment_rule": analysis_parameters[
                "polygon_coordinate_alignment_rule"
            ],
            "polygon_centroid_tolerance_um": analysis_parameters[
                "polygon_centroid_tolerance_um"
            ],
            "maximum_local_gap_um": analysis_parameters["spatial_max_gap_um"],
        },
        "resource_execution": {
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_devices": gpu_records,
            "parallel_schedule": parallel_schedule,
            "disk_preflight_plan": disk_plan,
            "filesystem_snapshots": disk_snapshots,
            "observed_max_filesystem_used_gib_at_sampled_stages": float(
                max(snapshot["used_gib"] for snapshot in disk_snapshots)
            ),
            "observed_min_filesystem_free_gib_at_sampled_stages": float(
                min(snapshot["free_gib"] for snapshot in disk_snapshots)
            ),
            "parent_peak_host_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024
                / (1024**3)
            ),
            "per_core_peak_host_rss_gib": {
                str(receipt["core_number"]): receipt["peak_host_rss_gib"]
                for receipt in core_receipts
            },
            "per_core_peak_cuda_allocated_gib": {
                str(receipt["core_number"]): receipt["extraction_audit"][
                    "peak_cuda_allocated_gib"
                ]
                for receipt in core_receipts
            },
            "per_core_peak_cuda_reserved_gib": {
                str(receipt["core_number"]): receipt["extraction_audit"][
                    "peak_cuda_reserved_gib"
                ]
                for receipt in core_receipts
            },
            "failures": [],
        },
        "package_versions": package_versions,
        "input_contract": {
            "cohort_manifest_sha256": EXPECTED_COHORT_MANIFEST_SHA256,
            "graph_manifest_sha256": EXPECTED_GRAPH_MANIFEST_SHA256,
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "gene_order_sha256": EXPECTED_GENE_SCHEMA_SHA256,
            "metadata_order_sha256": EXPECTED_METADATA_SCHEMA_SHA256,
            "preprocessing_version": EXPECTED_PREPROCESSING_VERSION,
            "pre_analysis_file_receipts": _portable_file_receipts(
                input_contract.pre_analysis_file_receipts
            ),
            "post_analysis": portable_input_post,
        },
        "checkpoint_post_analysis": checkpoint_post,
        "core_receipts": portable_core_receipts,
        "combined_tables": portable_combined_receipts,
        "visualization_receipt": portable_visualization_receipt,
        "combined_region_geometry_validation": combined_geometry_validation,
        "qc_checks": qc_checks,
        "qc_pass_fraction": qc_fraction,
        "artifacts": artifact_records,
        "reproduction_command": reproducibility_command,
        "ephemeral_core_shards_retained": False,
        "staging_cleanup": staging_cleanup,
    }
    _write_yaml_atomic(output_root / "analysis_manifest.yaml", analysis_manifest)
    _artifact_checksums(output_root, required_outputs)

    peak_vram_gib = max(
        float(receipt["extraction_audit"]["peak_cuda_allocated_gib"])
        for receipt in core_receipts
    )
    final_metrics = {
        "analysis/attention_niche_qc_pass_fraction": qc_fraction,
        "analysis/total_cells": float(len(assignments)),
        "analysis/retained_mutual_edges": float(
            sum(int(receipt["retained_mutual_edge_count"]) for receipt in core_receipts)
        ),
        "analysis/final_connected_niches": float(
            sum(int(receipt["final_connected_niche_count"]) for receipt in core_receipts)
        ),
    }
    _write_json_atomic(output_root / "metrics" / "final.json", final_metrics)
    history_record = {
        "step": 0,
        **final_metrics,
        "analysis_complete": True,
    }
    descriptor, temporary_history = tempfile.mkstemp(
        prefix=".history.jsonl.tmp-", dir=output_root / "metrics"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical_json(history_record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_history, output_root / "metrics" / "history.jsonl")
    descriptor, temporary_events = tempfile.mkstemp(
        prefix=".events.jsonl.tmp-", dir=output_root / "metrics"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for name, value in final_metrics.items():
            handle.write(
                _canonical_json({"name": name, "value": value, "step": 0})
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_events, output_root / "metrics" / "events.jsonl")
    summary = {
        "run_id": run_id,
        "status": "success",
        "analysis_kind": "six_core_attention_routing_niche_map",
        "map_label": map_label,
        "primary_metric_name": "analysis/attention_niche_qc_pass_fraction",
        "primary_metric_value": qc_fraction,
        "metrics": final_metrics,
        "model_seed_count": len(members),
        "model_seeds": [member.seed for member in members],
        "core_numbers": list(EXPECTED_CORE_NUMBERS),
        "peak_vram_gib": peak_vram_gib,
        "parameter_count": members[0].parameter_count,
        "limitation": LIMITATION,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def _artifact_checksums(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise AttentionNichePipelineError(f"Required output is absent: {relative}")
        records[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


__all__ = [
    "ANALYSIS_CAMPAIGN_ID",
    "AttentionNichePipelineError",
    "CheckpointMember",
    "CoreJobSpec",
    "EXPECTED_ALIASES",
    "EXPECTED_CORE_NUMBERS",
    "LIMITATION",
    "PreparedInputContract",
    "discover_completed_checkpoint_members",
    "run_attention_niche_pipeline",
    "verify_inputs_unchanged",
    "verify_prepared_input_contract",
]

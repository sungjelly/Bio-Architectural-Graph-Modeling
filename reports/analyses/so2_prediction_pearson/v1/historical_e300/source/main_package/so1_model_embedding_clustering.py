"""CPU-only SO1 h0/hL extraction and direct cosine-kNN Leiden maps.

This module is deliberately locked to the completed seed-0 SO1 fourteen-core
Relative-Geometric QKV run.  It never accepts an active ``latest.ckpt`` and it
does not have a CUDA execution path.  The two clustering graphs are independent
embedding-space graphs: one from the all-node NodeEncoder output ``h0`` and one
from the all-node final graph representation ``hL`` immediately before the
expression decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .checkpoint_catalog import show_checkpoint
from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    KNNGraphResult,
    _array_sha256,
    _atomic_save_figure_pair,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_parquet,
    _atomic_write_text,
    _canonical_sha256,
    _file_manifest,
    _file_record,
    _project_artifact_path,
    _read_json,
    _read_yaml,
    _receipt_with_self_hash,
    _style_spatial_axis,
    _tensor_sha256,
    _verify_self_hash,
    _write_deterministic_npz,
    build_faiss_cosine_knn_graph,
    deterministic_glasbey_palette,
    run_seeded_leiden,
)
from .relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from .relative_qkv_post_training import _tree_sha256
from .run_archive import RunArchive, verify_run_bundle


ANALYSIS_SCHEMA = "so1_14core_relative_qkv_embeddings_direct_knn_clustering_v1"
EXTRACTION_SCHEMA = "so1_14core_h0_hl_extraction_v1"
CORE_EXTRACTION_SCHEMA = "so1_14core_h0_hl_core_extraction_v1"
CLUSTERING_SCHEMA = "so1_14core_joint_h0_hl_direct_clustering_v1"
FIGURE_SCHEMA = "so1_14core_model_embedding_spatial_figures_v1"
CODE_PROVENANCE_SCHEMA = "so1_model_embedding_analysis_code_provenance_v1"

UPSTREAM_TRAINING_CAMPAIGN_ID = (
    "cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150"
)
ANALYSIS_CAMPAIGN_ID = (
    "cmp_20260827_so1_14core_relative_qkv_embeddings_direct_knn_clustering"
)
EXPECTED_RUN_ID = "r_20260826T204925Z_06d12943_s000_f00_a01_94691119"
EXPECTED_MODEL_SEED = 0
EXPECTED_FOLD = 0
EXPECTED_TRAINING_SOURCE_COMMIT = "4b03200af9168c535bde5c291cf5a00af054def7"
EXPECTED_UPSTREAM_STUDY_AXIS = "so1_14core_relative_geometry_qkv_transductive_fit"
EXPECTED_CHECKPOINT_RETENTION_CLASS = "retain_locked_final_model"
CHECKPOINT_SCHEMA = "so1_14core_relative_qkv_resume_v1"
SO1_CORE_NUMBERS = tuple(range(1, 15))
SO1_ALIASES = tuple(f"SO1-C{number:02d}" for number in SO1_CORE_NUMBERS)
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
EXPECTED_TOTAL_CELLS = 161_596
EXPECTED_N_GENES = 1_000
EXPECTED_NODE_COVARIATE_DIM = 22
EXPECTED_HIDDEN_DIM = 256
EXPECTED_GRAPH_LAYERS = 4
EXPECTED_RELATIVE_GEOMETRY_DIM = 70
EXPECTED_COHORT_MANIFEST_FILE_SHA256 = (
    "15f9da492959c35d89020b3956047ec163a5f3537eaaefd5b8a011cb9279b440"
)
EXPECTED_COHORT_MANIFEST_CONTENT_SHA256 = (
    "e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d"
)
EXPECTED_GRAPH_MANIFEST_FILE_SHA256 = (
    "754e98fa1b2d8b488892c4effbf095cf26da6d99c927ee45e9ee01c2d64b978f"
)
EXPECTED_GRAPH_MANIFEST_CONTENT_SHA256 = (
    "5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4"
)
EXPECTED_COMPLETED_COHORT_MANIFEST_SHA256 = (
    "e078588668d9b2285db27da6cda8b7f1d144aa055065c3022e43048fd1951596"
)

DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_CPU_THREADS = 40
DEFAULT_DPI = 300
DEFAULT_RECALL_QUERY_COUNT = 128
DEFAULT_RECALL_QUERY_BATCH_SIZE = 8
MINIMUM_MEAN_RECALL_AT_K = 0.90
DIRECT_PIPELINE_KIND = "direct_embedding_l2_cosine_knn_leiden"
PREDICTION_INVARIANCE_PROBE_COUNT = 64
PREDICTION_INVARIANCE_ATOL = 1.0e-7
PREDICTION_INVARIANCE_RTOL = 1.0e-6
INTRINSIC_LABEL_PREFIX = "S1I"
CONTEXTUAL_LABEL_PREFIX = "S1C"
REPRESENTATIONS = ("intrinsic", "contextual")
REPRESENTATION_ARRAYS = {"intrinsic": "h0", "contextual": "hL"}
REPRESENTATION_PREFIXES = {
    "intrinsic": INTRINSIC_LABEL_PREFIX,
    "contextual": CONTEXTUAL_LABEL_PREFIX,
}
KNN_INSERTION_SEED_XOR = 0x53314B
OUTPUT_NAMESPACE = "so1_14core_model_embedding_direct_knn_clustering"

INTRINSIC_COMBINED_STEM = (
    "intrinsic_direct_h0_leiden_resolution_1p0_spatial_14cores"
)
CONTEXTUAL_COMBINED_STEM = (
    "contextual_direct_hl_leiden_resolution_1p0_spatial_14cores"
)
DELTA_COMBINED_STEM = "delta_h_l2_spatial_14cores"


class SO1ModelEmbeddingClusteringError(ValueError):
    """Raised when the locked SO1 post-training contract is violated."""


@dataclass(frozen=True, slots=True)
class SO1ResolvedInputs:
    run_id: str
    project_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_payload: Mapping[str, Any] = field(repr=False)
    bundle_path: Path
    cohort_dir: Path
    graph_dir: Path
    cohort_manifest: Mapping[str, Any] = field(repr=False)
    graph_manifest: Mapping[str, Any] = field(repr=False)
    completed_cohort_manifest_path: Path
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SO1CoreInputs:
    alias: str
    core_number: int
    target_expression: torch.Tensor = field(repr=False)
    node_covariates: torch.Tensor = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    edge_index: torch.Tensor = field(repr=False)
    relative_geometry: torch.Tensor = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(self.target_expression.shape[0])


@dataclass(frozen=True, slots=True)
class SO1CoreEmbeddings:
    alias: str
    core_number: int
    cell_index: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    h0: np.ndarray = field(repr=False)
    hL: np.ndarray = field(repr=False)
    delta_h: np.ndarray = field(repr=False)
    delta_h_l2: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_index))


@dataclass(frozen=True, slots=True)
class DirectRepresentationResult:
    labels: np.ndarray = field(repr=False)
    edge_pairs: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cpu_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise SO1ModelEmbeddingClusteringError(
            "SO1 model-embedding analysis is CPU-only; --device must be 'cpu'."
        )
    return torch.device("cpu")


def validate_cuda_hidden() -> str:
    observed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed not in {"", "-1"}:
        rendered = "unset" if observed is None else repr(observed)
        raise SO1ModelEmbeddingClusteringError(
            "SO1 model-embedding analysis requires CUDA_VISIBLE_DEVICES='' "
            f"(or '-1'); observed {rendered}."
        )
    return str(observed)


def requested_panel_order() -> tuple[int, ...]:
    return SO1_CORE_NUMBERS


def spatial_plot_spec() -> dict[str, Any]:
    return {
        "panel_order": list(SO1_CORE_NUMBERS),
        "grid_shape": [3, 5],
        "legend_or_colorbar_panel": [2, 4],
        "panel_title_template": "SO1 Core {core_number}",
        "core_numbers_identifiable": True,
        "equal_aspect": True,
        "invert_y_axis": True,
        "coordinate_units": "micrometres",
        "scale_bars": True,
        "one_dot_per_cell": True,
        "lines_between_cells": False,
        "marker_borders": False,
        "point_layer_rasterized_in_pdf": True,
    }


def _manifest_content_is_valid(manifest: Mapping[str, Any]) -> bool:
    content = dict(manifest)
    observed = str(content.pop("manifest_content_sha256", ""))
    return bool(observed) and hmac.compare_digest(observed, _canonical_sha256(content))


def _source_array_sha256(name: str, array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(values.shape),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            f"Cannot load finalized SO1 checkpoint: {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise SO1ModelEmbeddingClusteringError("Checkpoint payload must be a mapping.")
    return dict(payload)


def _validated_model_construction(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    construction = payload.get("model_construction")
    if not isinstance(construction, Mapping):
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint lacks model-construction metadata."
        )
    expected = {
        "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
        "num_genes": EXPECTED_N_GENES,
        "node_covariate_dim": EXPECTED_NODE_COVARIATE_DIM,
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "embedding_dim": EXPECTED_HIDDEN_DIM,
        "graph_layers": EXPECTED_GRAPH_LAYERS,
        "relative_geometry_dim": EXPECTED_RELATIVE_GEOMETRY_DIM,
    }
    for key, expected_value in expected.items():
        value = construction.get(key)
        if isinstance(expected_value, int):
            try:
                observed = int(value)
            except (TypeError, ValueError) as exc:
                raise SO1ModelEmbeddingClusteringError(
                    f"Unexpected checkpoint model field {key}: {value!r}."
                ) from exc
        else:
            observed = value
        if observed != expected_value:
            raise SO1ModelEmbeddingClusteringError(
                f"Unexpected checkpoint model field {key}: {value!r}."
            )
    try:
        attention_dropout = float(construction.get("attention_dropout", math.nan))
    except (TypeError, ValueError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint attention dropout is invalid."
        ) from exc
    if attention_dropout != 0.0:
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint attention dropout is not zero."
        )
    return construction


def _validate_final_checkpoint_semantics(
    payload: Mapping[str, Any], *, expected_run_id: str
) -> Mapping[str, Any]:
    """Validate the terminal plateau contract without inventing completion state.

    The canonical SO1 finalizer deliberately leaves ``completion`` as ``None``;
    immutable finality comes from the published run bundle, registered
    ``last.ckpt``, and terminal plateau.  Therefore ``completion`` is not used
    as a checkpoint-payload gate.
    """

    plateau = payload.get("plateau")
    try:
        model_seed = int(payload.get("model_seed", -1))
        completed_epochs = int(payload.get("completed_global_epochs", -1))
        final_epoch = (
            int(plateau.get("final_epoch", -1))
            if isinstance(plateau, Mapping)
            else -1
        )
    except (TypeError, ValueError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            "Checkpoint terminal plateau fields are invalid."
        ) from exc
    if any(
        (
            payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA,
            payload.get("run_id") != str(expected_run_id),
            payload.get("campaign_id") != UPSTREAM_TRAINING_CAMPAIGN_ID,
            model_seed != EXPECTED_MODEL_SEED,
            completed_epochs < 150,
            not isinstance(plateau, Mapping),
            isinstance(plateau, Mapping)
            and plateau.get("should_stop") is not True,
            final_epoch != completed_epochs,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Checkpoint is not the completed strict-plateau SO1 final payload."
        )
    assert isinstance(plateau, Mapping)
    return {
        "completed_global_epochs": completed_epochs,
        "plateau": dict(plateau),
        "completion_field_is_not_a_finality_gate": True,
    }


def resolve_so1_analysis_inputs(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
) -> SO1ResolvedInputs:
    """Resolve only the completed immutable SO1 ``last.ckpt`` bundle."""

    if checkpoint is not None and Path(checkpoint).expanduser().name == "latest.ckpt":
        raise SO1ModelEmbeddingClusteringError("Active latest.ckpt is prohibited.")
    requested = EXPECTED_RUN_ID if run_id is None else str(run_id)
    canonical_run_id = registry.resolve_run_id(requested)
    if canonical_run_id != EXPECTED_RUN_ID:
        raise SO1ModelEmbeddingClusteringError(
            "This analysis is locked to the prespecified SO1 seed-0 run."
        )
    run_record = registry.show_run(canonical_run_id)
    if not isinstance(run_record, Mapping):
        raise SO1ModelEmbeddingClusteringError("The locked SO1 run is unregistered.")
    if any(
        (
            run_record.get("status") != "completed",
            run_record.get("campaign_id") != UPSTREAM_TRAINING_CAMPAIGN_ID,
            int(run_record.get("seed", -1)) != EXPECTED_MODEL_SEED,
            int(run_record.get("fold", -1)) != EXPECTED_FOLD,
            run_record.get("end_time") in {None, ""},
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "The exact SO1 run is not a completed seed-0/fold-0 immutable run. "
            "Active latest checkpoints are never accepted."
        )
    upstream_campaign = registry.get_campaign(UPSTREAM_TRAINING_CAMPAIGN_ID)
    upstream_campaign_config = (
        upstream_campaign.get("config")
        if isinstance(upstream_campaign, Mapping)
        else None
    )
    if not isinstance(upstream_campaign_config, Mapping) or any(
        (
            upstream_campaign_config.get("campaign_id")
            != UPSTREAM_TRAINING_CAMPAIGN_ID,
            upstream_campaign_config.get("frozen_task_contract_sha256")
            != "837ae83bc33b880ceeb2c736e1e4b021db0abf519bf5c1217aeeb9b71fbc4044",
            upstream_campaign_config.get("cohort", {}).get("total_fit_cells")
            != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "The registered upstream SO1 campaign configuration changed."
        )
    with registry.connect() as connection:
        eligible_completed = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE campaign_id = ? AND seed = ? AND fold = ? AND status = 'completed'
            ORDER BY run_id
            """,
            (UPSTREAM_TRAINING_CAMPAIGN_ID, EXPECTED_MODEL_SEED, EXPECTED_FOLD),
        ).fetchall()
    eligible_run_ids = [str(row["run_id"]) for row in eligible_completed]
    if eligible_run_ids != [canonical_run_id]:
        raise SO1ModelEmbeddingClusteringError(
            "SO1 completed-attempt selection is absent or ambiguous: "
            f"{eligible_run_ids}."
        )

    bundle_path = RunArchive.artifact_path_for(canonical_run_id, paths)
    bundle_verification = verify_run_bundle(bundle_path)
    checkpoint_catalog = show_checkpoint(
        registry, canonical_run_id, paths, role="last", verify=True
    )
    registered_metadata = checkpoint_catalog.get("registry_checkpoint_metadata")
    registered_category = checkpoint_catalog.get("registry_category")
    if any(
        (
            checkpoint_catalog.get("run_id") != canonical_run_id,
            checkpoint_catalog.get("campaign_id")
            != UPSTREAM_TRAINING_CAMPAIGN_ID,
            checkpoint_catalog.get("checkpoint_role") != "last",
            checkpoint_catalog.get("checkpoint_artifact_status") != "present",
            checkpoint_catalog.get("status") != "completed",
            checkpoint_catalog.get("lifecycle_stage") != "locked_final",
            checkpoint_catalog.get("study_axis") != EXPECTED_UPSTREAM_STUDY_AXIS,
            checkpoint_catalog.get("retention_tier")
            != EXPECTED_CHECKPOINT_RETENTION_CLASS,
            checkpoint_catalog.get("verification_status") != "verified",
            not isinstance(registered_metadata, Mapping),
            isinstance(registered_metadata, Mapping)
            and registered_metadata.get("role") != "last",
            isinstance(registered_metadata, Mapping)
            and registered_metadata.get("retention_class")
            != EXPECTED_CHECKPOINT_RETENTION_CLASS,
            isinstance(registered_metadata, Mapping)
            and registered_metadata.get("verification_status") != "verified",
            not isinstance(registered_category, Mapping),
            isinstance(registered_category, Mapping)
            and registered_category.get("lifecycle_stage") != "locked_final",
            isinstance(registered_category, Mapping)
            and registered_category.get("study_axis")
            != EXPECTED_UPSTREAM_STUDY_AXIS,
            isinstance(registered_category, Mapping)
            and registered_category.get("retention_class")
            != EXPECTED_CHECKPOINT_RETENTION_CLASS,
            int(checkpoint_catalog.get("seed", -1)) != EXPECTED_MODEL_SEED,
            int(checkpoint_catalog.get("fold", -1)) != EXPECTED_FOLD,
            int(checkpoint_catalog.get("attempt", -1)) != 1,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Registered SO1 last checkpoint lacks the required locked-final "
            "classification, retention, or verification semantics."
        )
    canonical_checkpoint = Path(
        str(checkpoint_catalog.get("verified_path", ""))
    ).resolve(strict=True)
    if canonical_checkpoint.name != "last.ckpt" or canonical_checkpoint.parent.name != "checkpoints":
        raise SO1ModelEmbeddingClusteringError(
            "Resolved checkpoint is not the immutable checkpoints/last.ckpt."
        )
    if checkpoint is None:
        checkpoint_path = canonical_checkpoint
    else:
        explicit_checkpoint = Path(checkpoint).expanduser()
        if not explicit_checkpoint.is_absolute():
            explicit_checkpoint = paths.project_root / explicit_checkpoint
        explicit_checkpoint = explicit_checkpoint.resolve(strict=True)
        if not hmac.compare_digest(
            sha256_file(explicit_checkpoint), sha256_file(canonical_checkpoint)
        ):
            raise SO1ModelEmbeddingClusteringError(
                "Explicit checkpoint differs from the registered immutable last checkpoint."
            )
        # An explicit checksum-matching path is only a selector.  Inference and
        # provenance always use the canonical registered immutable artifact.
        checkpoint_path = canonical_checkpoint
    if checkpoint_path.name == "latest.ckpt":
        raise SO1ModelEmbeddingClusteringError("Active latest.ckpt is prohibited.")

    payload = _load_checkpoint_payload(checkpoint_path)
    _validate_final_checkpoint_semantics(
        payload, expected_run_id=canonical_run_id
    )
    construction = _validated_model_construction(payload)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or _tree_sha256(state) != payload.get(
        "model_state_checksum"
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint model-state checksum is invalid."
        )

    checkpoint_sha = sha256_file(checkpoint_path)
    if any(
        path.name != "last.ckpt"
        for path in checkpoint_path.parent.iterdir()
    ) or len(tuple(checkpoint_path.parent.iterdir())) != 1:
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint directory does not have last-only retention layout."
        )

    run_manifest_path = bundle_path / "manifest.yaml"
    success_marker_path = bundle_path / "_SUCCESS"
    artifact_checksums_path = (
        bundle_path / "provenance" / "artifact_checksums.json"
    )
    reload_verification_path = (
        bundle_path / "diagnostics" / "final_checkpoint_reload_verification.json"
    )
    training_provenance_path = (
        bundle_path / "provenance" / "so1_relative_qkv_training.json"
    )
    run_manifest = _read_yaml(run_manifest_path, label="SO1 finalized run manifest")
    success_marker = _read_json(success_marker_path, label="SO1 success marker")
    artifact_checksums = _read_json(
        artifact_checksums_path, label="SO1 artifact checksum manifest"
    )
    reload_verification = _read_json(
        reload_verification_path, label="SO1 final checkpoint reload verification"
    )
    training_provenance = _read_json(
        training_provenance_path, label="SO1 training provenance"
    )
    manifest_catalog = run_manifest.get("checkpoint_catalog")
    manifest_artifact_roles = run_manifest.get("artifact_roles")
    checksum_files = artifact_checksums.get("files")
    reload_payload = reload_verification.get("payload")
    replay = reload_verification.get("fixed_prediction_replay")
    terminal_plateau = payload.get("plateau")
    terminal_epoch = int(payload.get("completed_global_epochs", -1))
    plateau_audit_path = (
        bundle_path
        / "diagnostics"
        / f"strict_plateau_audit_epoch_{terminal_epoch:04d}.json"
    )
    plateau_audit = _read_json(
        plateau_audit_path, label="SO1 terminal strict plateau audit"
    )
    if any(
        (
            bundle_verification.get("valid") is not True,
            bundle_verification.get("status") != "success",
            run_manifest.get("run_id") != canonical_run_id,
            run_manifest.get("campaign_id") != UPSTREAM_TRAINING_CAMPAIGN_ID,
            run_manifest.get("lifecycle_status_source")
            != "registry_and_completion_marker",
            not isinstance(manifest_artifact_roles, Mapping),
            isinstance(manifest_artifact_roles, Mapping)
            and manifest_artifact_roles.get("primary_checkpoint") != "last",
            not isinstance(manifest_catalog, Mapping),
            isinstance(manifest_catalog, Mapping)
            and manifest_catalog.get("lifecycle_stage") != "locked_final",
            isinstance(manifest_catalog, Mapping)
            and manifest_catalog.get("study_axis") != EXPECTED_UPSTREAM_STUDY_AXIS,
            isinstance(manifest_catalog, Mapping)
            and manifest_catalog.get("retention_class")
            != EXPECTED_CHECKPOINT_RETENTION_CLASS,
            success_marker.get("run_id") != canonical_run_id,
            success_marker.get("status") != "success",
            artifact_checksums.get("version") != 1,
            not isinstance(checksum_files, Mapping),
            isinstance(checksum_files, Mapping)
            and checksum_files.get("checkpoints/last.ckpt", {}).get("sha256")
            != checkpoint_sha,
            reload_verification.get("schema")
            != "so1_14core_final_checkpoint_reload_verification_v1",
            reload_verification.get("verified") is not True,
            reload_verification.get("checkpoint_file_sha256") != checkpoint_sha,
            not isinstance(reload_payload, Mapping),
            isinstance(reload_payload, Mapping)
            and reload_payload.get("source_checkpoint_sha256") != checkpoint_sha,
            isinstance(reload_payload, Mapping)
            and reload_payload.get("model_state_checksum")
            != payload.get("model_state_checksum"),
            not isinstance(replay, Mapping),
            isinstance(replay, Mapping)
            and replay.get("model_state_checksum")
            != payload.get("model_state_checksum"),
            isinstance(replay, Mapping)
            and replay.get("reloaded_model_state_checksum")
            != payload.get("model_state_checksum"),
            training_provenance.get("campaign_id")
            != UPSTREAM_TRAINING_CAMPAIGN_ID,
            int(training_provenance.get("model_seed", -1)) != EXPECTED_MODEL_SEED,
            training_provenance.get("checkpoint_sha256") != checkpoint_sha,
            training_provenance.get("state_dict_sha256")
            != payload.get("model_state_checksum"),
            int(training_provenance.get("completed_global_epochs", -1))
            != int(payload.get("completed_global_epochs", -2)),
            training_provenance.get("plateau") != terminal_plateau,
            training_provenance.get("checkpoint_reload_verified") is not True,
            training_provenance.get("checkpoint_reload_verification")
            != reload_verification,
            training_provenance.get("checkpoint_layout") != "final_last_only",
            training_provenance.get("model_construction") != dict(construction),
            plateau_audit.get("schema")
            != "so1_14core_strict_plateau_decision_v1",
            int(plateau_audit.get("completed_global_epochs", -1))
            != terminal_epoch,
            int(plateau_audit.get("final_epoch", -1)) != terminal_epoch,
            plateau_audit.get("plateau_conditions_passed") is not True,
            int(plateau_audit.get("plateau_consecutive_passing_audits", -1)) < 2,
            plateau_audit.get("should_stop") is not True,
            plateau_audit.get("plateau_should_stop") is not True,
            plateau_audit.get("training_stop_applied") is not True,
            plateau_audit.get("validation_or_test_metric") is not False,
            plateau_audit.get("checkpoint_selection_metric") is not False,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Finalized SO1 bundle lacks a consistent success, locked-final "
            "catalog, reload-verification, or terminal-training provenance chain."
        )

    resolved_config_path = bundle_path / "config.resolved.yaml"
    config = _read_yaml(
        resolved_config_path, label="SO1 resolved run configuration"
    )
    checkpoint_config = payload.get("resolved_config")
    if not isinstance(checkpoint_config, Mapping) or _canonical_sha256(
        checkpoint_config
    ) != _canonical_sha256(config):
        raise SO1ModelEmbeddingClusteringError(
            "Bundle resolved configuration differs from the checkpoint-bound configuration."
        )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SO1ModelEmbeddingClusteringError(
            "Resolved SO1 configuration lacks its dataset section."
        )
    cohort_dir = _project_artifact_path(dataset.get("prepared_artifact"), paths=paths)
    graph_dir = _project_artifact_path(
        dataset.get("prepared_graph_artifact"), paths=paths
    )
    cohort_manifest_path = cohort_dir / "manifest.json"
    graph_manifest_path = graph_dir / "manifest.json"
    completed_path = graph_dir / "cohort_manifest_with_graphs.json"
    observed_input_hashes = {
        "cohort": sha256_file(cohort_manifest_path),
        "graph": sha256_file(graph_manifest_path),
        "completed": sha256_file(completed_path),
    }
    expected_input_hashes = {
        "cohort": EXPECTED_COHORT_MANIFEST_FILE_SHA256,
        "graph": EXPECTED_GRAPH_MANIFEST_FILE_SHA256,
        "completed": EXPECTED_COMPLETED_COHORT_MANIFEST_SHA256,
    }
    configured_hashes = {
        "cohort": dataset.get("cohort_manifest_file_sha256"),
        "graph": dataset.get("graph_manifest_file_sha256"),
        "completed": dataset.get("completed_cohort_manifest_sha256"),
    }
    if observed_input_hashes != expected_input_hashes or configured_hashes != expected_input_hashes:
        raise SO1ModelEmbeddingClusteringError(
            "SO1 cohort/graph manifests differ from the checkpoint-bound inputs."
        )
    cohort_manifest = _read_json(cohort_manifest_path, label="SO1 cohort manifest")
    graph_manifest = _read_json(graph_manifest_path, label="SO1 graph manifest")
    if not _manifest_content_is_valid(cohort_manifest) or not _manifest_content_is_valid(
        graph_manifest
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 cohort or graph manifest self-checksum is invalid."
        )
    if any(
        (
            cohort_manifest.get("manifest_content_sha256")
            != EXPECTED_COHORT_MANIFEST_CONTENT_SHA256,
            graph_manifest.get("manifest_content_sha256")
            != EXPECTED_GRAPH_MANIFEST_CONTENT_SHA256,
            tuple(cohort_manifest.get("cohort", {}).get("aliases", ()))
            != SO1_ALIASES,
            tuple(cohort_manifest.get("cohort", {}).get(
                "original_core_numbers", ()
            ))
            != SO1_CORE_NUMBERS,
            int(cohort_manifest.get("cohort", {}).get("total_cells", -1))
            != EXPECTED_TOTAL_CELLS,
            tuple(graph_manifest.get("aliases", ())) != SO1_ALIASES,
            dataset.get("dataset_fingerprint")
            != EXPECTED_COHORT_MANIFEST_CONTENT_SHA256,
            dataset.get("graph_manifest_content_sha256")
            != EXPECTED_GRAPH_MANIFEST_CONTENT_SHA256,
        )
    ):
        raise SO1ModelEmbeddingClusteringError("SO1 input identity changed.")

    training_git = _read_json(
        bundle_path / "provenance" / "git.json", label="SO1 training Git provenance"
    )
    training_commit = training_git.get("commit")
    if training_commit != EXPECTED_TRAINING_SOURCE_COMMIT:
        raise SO1ModelEmbeddingClusteringError(
            "SO1 finalized bundle source commit differs from the locked training run."
        )
    provenance = {
        "selection_policy": "exact_completed_campaign_seed0_fold0_last_checkpoint",
        "campaign_id": UPSTREAM_TRAINING_CAMPAIGN_ID,
        "upstream_campaign_registry": {
            "campaign_id": UPSTREAM_TRAINING_CAMPAIGN_ID,
            "configuration_sha256": _canonical_sha256(upstream_campaign_config),
            "frozen_task_contract_sha256": upstream_campaign_config.get(
                "frozen_task_contract_sha256"
            ),
        },
        "run_id": canonical_run_id,
        "model_seed": EXPECTED_MODEL_SEED,
        "fold": EXPECTED_FOLD,
        "attempt": int(checkpoint_catalog["attempt"]),
        "preferred_alias": checkpoint_catalog.get("preferred_alias"),
        "scientific_id": checkpoint_catalog.get("scientific_id"),
        "repro_id": checkpoint_catalog.get("repro_id"),
        "finalized_run_bundle": {
            "path": bundle_path.as_posix(),
            "verification": dict(bundle_verification),
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": int(checkpoint_path.stat().st_size),
            "schema": CHECKPOINT_SCHEMA,
            "completed_global_epochs": int(payload["completed_global_epochs"]),
            "model_state_sha256": str(payload["model_state_checksum"]),
            "role": "last",
        },
        "checkpoint_catalog": {
            "checkpoint_artifact_id": checkpoint_catalog.get(
                "checkpoint_artifact_id"
            ),
            "checkpoint_id": checkpoint_catalog.get("checkpoint_id"),
            "role": checkpoint_catalog.get("checkpoint_role"),
            "best_epoch": checkpoint_catalog.get("best_epoch"),
            "path": checkpoint_catalog.get("checkpoint_path"),
            "sha256": checkpoint_catalog.get("checkpoint_sha256"),
            "size_bytes": checkpoint_catalog.get("checkpoint_size_bytes"),
            "monitored_metric": checkpoint_catalog.get("monitored_metric"),
            "monitored_value": checkpoint_catalog.get("monitored_value"),
            "retention_class": registered_metadata.get("retention_class"),
            "verification_status": registered_metadata.get(
                "verification_status"
            ),
            "lifecycle_stage": checkpoint_catalog.get("lifecycle_stage"),
            "study_axis": checkpoint_catalog.get("study_axis"),
        },
        "resolved_config": {
            "path": resolved_config_path.as_posix(),
            "file": _file_record(resolved_config_path),
            "canonical_content_sha256": _canonical_sha256(config),
            "checkpoint_bound_canonical_content_sha256": _canonical_sha256(
                checkpoint_config
            ),
        },
        "run_manifest": {
            "path": run_manifest_path.as_posix(),
            "file": _file_record(run_manifest_path),
        },
        "success_marker": {
            "path": success_marker_path.as_posix(),
            "file": _file_record(success_marker_path),
            "content_sha256": success_marker.get("content_sha256"),
        },
        "artifact_checksum_manifest": {
            "path": artifact_checksums_path.as_posix(),
            "file": _file_record(artifact_checksums_path),
        },
        "final_checkpoint_reload_verification": {
            "path": reload_verification_path.as_posix(),
            "file": _file_record(reload_verification_path),
        },
        "terminal_plateau_audit": {
            "path": plateau_audit_path.as_posix(),
            "file": _file_record(plateau_audit_path),
        },
        "training_provenance_receipt": {
            "path": training_provenance_path.as_posix(),
            "file": _file_record(training_provenance_path),
        },
        "training_source": training_git,
        "model_construction": dict(construction),
        "preprocessing_version": dataset.get("preprocessing_version"),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "cohort_manifest": {
            "path": cohort_manifest_path.as_posix(),
            "file": _file_record(cohort_manifest_path),
            "content_sha256": cohort_manifest["manifest_content_sha256"],
            "statistics_checksums": cohort_manifest.get("preprocessing", {}).get(
                "statistics_checksums", {}
            ),
        },
        "graph_manifest": {
            "path": graph_manifest_path.as_posix(),
            "file": _file_record(graph_manifest_path),
            "content_sha256": graph_manifest["manifest_content_sha256"],
        },
        "completed_cohort_manifest": {
            "path": completed_path.as_posix(),
            "file": _file_record(completed_path),
        },
    }
    return SO1ResolvedInputs(
        run_id=canonical_run_id,
        project_root=paths.project_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_payload=payload,
        bundle_path=bundle_path,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
        cohort_manifest=cohort_manifest,
        graph_manifest=graph_manifest,
        completed_cohort_manifest_path=completed_path,
        provenance=provenance,
    )


def _integer(construction: Mapping[str, Any], name: str) -> int:
    try:
        return int(construction[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            f"Invalid model-construction field: {name}."
        ) from exc


def load_so1_checkpoint_model(
    inputs: SO1ResolvedInputs,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    payload = inputs.checkpoint_payload
    construction = _validated_model_construction(payload)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise SO1ModelEmbeddingClusteringError("Checkpoint lacks model weights.")
    model = ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=_integer(construction, "num_genes"),
        node_covariate_dim=_integer(construction, "node_covariate_dim"),
        hidden_dim=_integer(construction, "hidden_dim"),
        attention_heads=_integer(construction, "attention_heads"),
        attention_head_dim=_integer(construction, "attention_head_dim"),
        graph_layers=_integer(construction, "graph_layers"),
        ffn_dim=_integer(construction, "ffn_dim"),
        decoder_dim=_integer(construction, "decoder_dim"),
        positional_bias_hidden_dim=_integer(
            construction, "positional_bias_hidden_dim"
        ),
        dropout=float(construction["dropout"]),
        attention_dropout=float(construction["attention_dropout"]),
        relative_geometry_dim=_integer(construction, "relative_geometry_dim"),
        receiver_chunk_size=_integer(construction, "receiver_chunk_size"),
        max_edges_per_chunk=_integer(construction, "max_edges_per_chunk"),
        activation_checkpointing=bool(construction["activation_checkpointing"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise SO1ModelEmbeddingClusteringError("model.eval() did not disable dropout.")
    if {float(block.attention_dropout_probability) for block in model.blocks} != {0.0}:
        raise SO1ModelEmbeddingClusteringError(
            "Attention dropout is not disabled."
        )
    if _tree_sha256(model.state_dict()) != payload.get("model_state_checksum"):
        raise SO1ModelEmbeddingClusteringError("Loaded model state changed.")
    return model


def _expected_core_records(inputs: SO1ResolvedInputs) -> tuple[dict[str, Any], ...]:
    cohort_records = inputs.cohort_manifest.get("cores")
    graph_records = inputs.graph_manifest.get("cores")
    cohort_files = inputs.cohort_manifest.get("files")
    if not isinstance(cohort_records, list) or not isinstance(graph_records, list):
        raise SO1ModelEmbeddingClusteringError("SO1 manifests lack core records.")
    if not isinstance(cohort_files, Mapping):
        raise SO1ModelEmbeddingClusteringError("SO1 cohort lacks file checksums.")
    cohort_by_alias = {
        str(record.get("alias")): record
        for record in cohort_records
        if isinstance(record, Mapping)
    }
    graph_by_alias = {
        str(record.get("alias")): record
        for record in graph_records
        if isinstance(record, Mapping)
    }
    result: list[dict[str, Any]] = []
    for core_number, alias in zip(SO1_CORE_NUMBERS, SO1_ALIASES, strict=True):
        cohort = cohort_by_alias.get(alias)
        graph = graph_by_alias.get(alias)
        if not isinstance(cohort, Mapping) or not isinstance(graph, Mapping):
            raise SO1ModelEmbeddingClusteringError(f"Missing input record for {alias}.")
        expected_cells = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        graph_without_hash = dict(graph)
        observed_record_sha = str(graph_without_hash.pop("record_sha256", ""))
        if any(
            (
                int(cohort.get("cell_count", -1)) != expected_cells,
                int(cohort.get("original_core_number", -1)) != core_number,
                int(graph.get("n_cells", -1)) != expected_cells,
                observed_record_sha != _canonical_sha256(graph_without_hash),
                int(graph.get("relative_geometry", {}).get("parameters", {}).get(
                    "feature_dimension", -1
                ))
                != EXPECTED_RELATIVE_GEOMETRY_DIM,
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Prepared SO1 input record changed for {alias}."
            )
        result.append(
            {
                "alias": alias,
                "core_number": core_number,
                "cell_count": expected_cells,
                "prepared_core_artifact_sha256": str(
                    cohort_files[f"cores/{alias}.npz"]
                ),
                "prepared_component_checksums": dict(
                    cohort.get("component_checksums", {})
                ),
                "graph_record_sha256": observed_record_sha,
                "graph_logical_sha256": str(
                    graph.get("graph", {}).get("checksums", {}).get("graph_sha256")
                ),
                "graph_file_checksums": dict(graph.get("files", {})),
                "edge_count": int(
                    graph.get("graph", {}).get("qc", {}).get("n_directed_edges", -1)
                ),
            }
        )
    if sum(record["cell_count"] for record in result) != EXPECTED_TOTAL_CELLS:
        raise SO1ModelEmbeddingClusteringError("SO1 core counts do not reconcile.")
    return tuple(result)


def _load_core_inputs(
    inputs: SO1ResolvedInputs, expected: Mapping[str, Any]
) -> SO1CoreInputs:
    alias = str(expected["alias"])
    core_number = int(expected["core_number"])
    expected_cells = int(expected["cell_count"])
    cohort_path = inputs.cohort_dir / "cores" / f"{alias}.npz"
    if sha256_file(cohort_path) != expected["prepared_core_artifact_sha256"]:
        raise SO1ModelEmbeddingClusteringError(
            f"Prepared cohort artifact checksum changed for {alias}."
        )
    try:
        with np.load(cohort_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "expression_counts",
                "target_expression",
                "node_covariates",
                "coordinates_um",
            }:
                raise SO1ModelEmbeddingClusteringError(
                    f"Prepared core array schema changed for {alias}."
                )
            expression = np.array(
                archive["target_expression"], dtype=np.float32, copy=True
            )
            covariates = np.array(
                archive["node_covariates"], dtype=np.float32, copy=True
            )
            coordinates = np.array(
                archive["coordinates_um"], dtype=np.float64, copy=True
            )
    except (OSError, ValueError, KeyError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            f"Cannot load prepared SO1 core {alias}."
        ) from exc
    checksums = expected["prepared_component_checksums"]
    for name, value in (
        ("target_expression", expression),
        ("node_covariates", covariates),
        ("coordinates_um", coordinates),
    ):
        if _source_array_sha256(name, value) != checksums.get(name):
            raise SO1ModelEmbeddingClusteringError(
                f"Prepared {name} content changed for {alias}."
            )
    if any(
        (
            expression.shape != (expected_cells, EXPECTED_N_GENES),
            covariates.shape != (expected_cells, EXPECTED_NODE_COVARIATE_DIM),
            coordinates.shape != (expected_cells, 2),
            not np.isfinite(expression).all(),
            not np.isfinite(covariates).all(),
            not np.isfinite(coordinates).all(),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Prepared expression/metadata/coordinates are invalid for {alias}."
        )

    graph_root = inputs.graph_dir / "cores" / alias
    edge_path = graph_root / "edge_index.npy"
    geometry_path = graph_root / "relative_geometry.npy"
    graph_files = expected["graph_file_checksums"]
    if any(
        (
            sha256_file(edge_path) != graph_files.get("edge_index.npy"),
            sha256_file(geometry_path) != graph_files.get("relative_geometry.npy"),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Prepared graph file checksum changed for {alias}."
        )
    edge_map = np.load(edge_path, mmap_mode="r")
    geometry_map = np.load(geometry_path, mmap_mode="r")
    edge_count = int(expected["edge_count"])
    if any(
        (
            edge_map.shape != (2, edge_count),
            edge_map.dtype != np.int64,
            geometry_map.shape != (edge_count, EXPECTED_RELATIVE_GEOMETRY_DIM),
            geometry_map.dtype != np.float32,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Prepared graph/cache schema changed for {alias}."
        )
    return SO1CoreInputs(
        alias=alias,
        core_number=core_number,
        target_expression=torch.from_numpy(expression),
        node_covariates=torch.from_numpy(covariates),
        coordinates_um=np.ascontiguousarray(coordinates),
        edge_index=torch.from_numpy(edge_map),
        relative_geometry=torch.from_numpy(geometry_map),
    )


@torch.inference_mode()
def extract_full_h0_hl(
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    *,
    input_expression: torch.Tensor,
    gene_mask: torch.Tensor,
    edge_index: torch.Tensor,
    relative_geometry: torch.Tensor,
    node_covariates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Return all-node h0/hL and a trained-model prediction-invariance receipt.

    ``h0`` is evaluated directly by the exact trained ``NodeEncoder``.  Two
    complete-graph forwards decode only a deterministic probe of at most 64
    cells: one ordinary forward and one with intermediate outputs enabled.  The
    latter forward's ``full_node_embedding`` is reused as all-node hL, avoiding
    a third graph pass and avoiding an all-cell expression decode.
    """

    if model.training or any(module.training for module in model.modules()):
        raise SO1ModelEmbeddingClusteringError("Extraction requires model.eval().")
    tensors = (
        input_expression,
        gene_mask,
        edge_index,
        relative_geometry,
        node_covariates,
    )
    if any(value.device.type != "cpu" for value in tensors):
        raise SO1ModelEmbeddingClusteringError(
            "Expression, metadata, graph, and geometry must remain on CPU."
        )
    if gene_mask.dtype is not torch.bool or gene_mask.shape != input_expression.shape:
        raise SO1ModelEmbeddingClusteringError("Extraction mask shape/dtype is invalid.")
    if bool(torch.any(gene_mask).item()):
        raise SO1ModelEmbeddingClusteringError(
            "Extraction requires the all-zero binary gene mask."
        )
    if node_covariates.shape != (
        input_expression.shape[0],
        EXPECTED_NODE_COVARIATE_DIM,
    ):
        raise SO1ModelEmbeddingClusteringError("Extraction metadata shape is invalid.")
    h0 = model.encoder(
        input_expression=input_expression,
        gene_mask=gene_mask,
        node_covariates=node_covariates,
    )
    probe_count = min(PREDICTION_INVARIANCE_PROBE_COUNT, int(input_expression.shape[0]))
    probe_indices = torch.from_numpy(
        np.rint(
            np.linspace(
                0,
                int(input_expression.shape[0]) - 1,
                probe_count,
                dtype=np.float64,
            )
        ).astype(np.int64)
    )
    if probe_indices.unique().numel() != probe_count:
        raise SO1ModelEmbeddingClusteringError(
            "Prediction-invariance probe indices are not unique."
        )
    ordinary = model(
        input_expression=input_expression,
        gene_mask=gene_mask,
        edge_index=edge_index,
        relative_geometry=relative_geometry,
        node_covariates=node_covariates,
        target_nodes=probe_indices,
        return_intermediate_embeddings=False,
    )
    extended = model(
        input_expression=input_expression,
        gene_mask=gene_mask,
        edge_index=edge_index,
        relative_geometry=relative_geometry,
        node_covariates=node_covariates,
        target_nodes=probe_indices,
        return_intermediate_embeddings=True,
    )
    hL = extended.full_node_embedding
    expected_shape = (int(input_expression.shape[0]), int(model.hidden_dim))
    if h0.shape != expected_shape or hL is None or hL.shape != expected_shape:
        raise SO1ModelEmbeddingClusteringError("Extracted h0/hL shape is invalid.")
    if not bool(torch.isfinite(h0).all()) or not bool(torch.isfinite(hL).all()):
        raise SO1ModelEmbeddingClusteringError("Extracted h0/hL is non-finite.")
    selected_h0 = h0.index_select(0, probe_indices)
    selected_hL = hL.index_select(0, probe_indices)
    if extended.node_encoder_embedding is None or extended.final_graph_embedding is None:
        raise SO1ModelEmbeddingClusteringError(
            "Intermediate-output forward omitted h0 or hL."
        )
    h0_exact = torch.equal(extended.node_encoder_embedding, selected_h0)
    hL_exact = torch.equal(extended.final_graph_embedding, selected_hL)
    if not h0_exact or not hL_exact:
        raise SO1ModelEmbeddingClusteringError(
            "Intermediate h0/hL rows differ from the all-node representations."
        )
    if ordinary.prediction.shape != (probe_count, model.num_genes) or (
        extended.prediction.shape != ordinary.prediction.shape
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Prediction-invariance probe output shape is invalid."
        )
    differences = torch.abs(ordinary.prediction - extended.prediction)
    maximum_difference = float(differences.max().item()) if differences.numel() else 0.0
    mean_difference = float(differences.mean().item()) if differences.numel() else 0.0
    predictions_exact = torch.equal(ordinary.prediction, extended.prediction)
    predictions_allclose = torch.allclose(
        ordinary.prediction,
        extended.prediction,
        rtol=PREDICTION_INVARIANCE_RTOL,
        atol=PREDICTION_INVARIANCE_ATOL,
        equal_nan=False,
    )
    if not predictions_allclose:
        raise SO1ModelEmbeddingClusteringError(
            "Enabling intermediate outputs changed trained-model predictions."
        )
    invariance = {
        "verified_on_completed_trained_model": True,
        "complete_core_graph_used": True,
        "probe_selection": "deterministic_evenly_spaced_cell_indices",
        "probe_count": int(probe_count),
        "probe_indices_sha256": _tensor_sha256(
            "prediction_invariance_probe_indices", probe_indices
        ),
        "decoded_rows_per_forward": int(probe_count),
        "graph_forward_count": 2,
        "ordinary_return_intermediate_embeddings": False,
        "extended_return_intermediate_embeddings": True,
        "predictions_exact": bool(predictions_exact),
        "predictions_allclose": bool(predictions_allclose),
        "maximum_absolute_prediction_difference": maximum_difference,
        "mean_absolute_prediction_difference": mean_difference,
        "absolute_tolerance": PREDICTION_INVARIANCE_ATOL,
        "relative_tolerance": PREDICTION_INVARIANCE_RTOL,
        "extended_h0_equals_direct_h0_at_probe_rows_exact": bool(h0_exact),
        "extended_hL_equals_full_hL_at_probe_rows_exact": bool(hL_exact),
        "all_node_hL_reused_from_extended_forward": True,
        "all_cell_expression_decode_performed": False,
    }
    return h0, hL, invariance


def _validate_core_embedding_arrays(
    *,
    alias: str,
    core_number: int,
    arrays: Mapping[str, np.ndarray],
    expected_cells: int,
) -> SO1CoreEmbeddings:
    expected_names = {
        "cell_index",
        "core_number",
        "coordinates_um",
        "h0",
        "hL",
        "delta_h",
        "delta_h_l2",
    }
    if set(arrays) != expected_names:
        raise SO1ModelEmbeddingClusteringError(
            f"Unexpected embedding artifact schema for {alias}."
        )
    cell_index = np.asarray(arrays["cell_index"], dtype=np.int64)
    core_value = np.asarray(arrays["core_number"])
    coordinates = np.asarray(arrays["coordinates_um"], dtype=np.float64)
    h0 = np.asarray(arrays["h0"], dtype=np.float32)
    hL = np.asarray(arrays["hL"], dtype=np.float32)
    delta = np.asarray(arrays["delta_h"], dtype=np.float32)
    delta_norm = np.asarray(arrays["delta_h_l2"], dtype=np.float64)
    expected_embedding_shape = (expected_cells, EXPECTED_HIDDEN_DIM)
    if any(
        (
            not np.array_equal(cell_index, np.arange(expected_cells, dtype=np.int64)),
            core_value.size != 1,
            core_value.size == 1
            and int(core_value.reshape(-1)[0]) != int(core_number),
            coordinates.shape != (expected_cells, 2),
            h0.shape != expected_embedding_shape,
            hL.shape != expected_embedding_shape,
            delta.shape != expected_embedding_shape,
            delta_norm.shape != (expected_cells,),
            not np.isfinite(coordinates).all(),
            not np.isfinite(h0).all(),
            not np.isfinite(hL).all(),
            not np.isfinite(delta).all(),
            not np.isfinite(delta_norm).all(),
            bool(np.any(delta_norm < 0.0)),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Embedding coverage/order/finite contract failed for {alias}."
        )
    expected_delta = np.ascontiguousarray(hL - h0, dtype=np.float32)
    expected_norm = np.linalg.norm(expected_delta.astype(np.float64), axis=1)
    if not np.array_equal(delta, expected_delta) or not np.allclose(
        delta_norm, expected_norm, rtol=1e-12, atol=1e-12
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Stored delta_h or delta_h_l2 is inconsistent for {alias}."
        )
    return SO1CoreEmbeddings(
        alias=alias,
        core_number=int(core_number),
        cell_index=np.ascontiguousarray(cell_index),
        coordinates_um=np.ascontiguousarray(coordinates),
        h0=np.ascontiguousarray(h0),
        hL=np.ascontiguousarray(hL),
        delta_h=np.ascontiguousarray(delta),
        delta_h_l2=np.ascontiguousarray(delta_norm),
    )


def _core_embedding_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_embeddings.npz"


def _core_receipt_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_receipt.json"


def load_core_embeddings(
    path: Path, *, alias: str, core_number: int, expected_cells: int
) -> SO1CoreEmbeddings:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise SO1ModelEmbeddingClusteringError(
            f"Cannot load embedding artifact: {path}."
        ) from exc
    return _validate_core_embedding_arrays(
        alias=alias,
        core_number=core_number,
        arrays=arrays,
        expected_cells=expected_cells,
    )


def _verify_core_extraction_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO1ResolvedInputs,
    expected: Mapping[str, Any],
) -> SO1CoreEmbeddings:
    _verify_self_hash(receipt, label="SO1 per-core embedding receipt")
    inference = receipt.get("inference")
    invariance = (
        inference.get("prediction_invariance")
        if isinstance(inference, Mapping)
        else None
    )
    if any(
        (
            receipt.get("schema") != CORE_EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            receipt.get("analysis_code_provenance")
            != _analysis_code_provenance_record(output_root),
            receipt.get("source_core") != dict(expected),
            receipt.get("alias") != expected["alias"],
            int(receipt.get("core_number", -1)) != int(expected["core_number"]),
            int(receipt.get("cell_count", -1)) != int(expected["cell_count"]),
            not isinstance(inference, Mapping),
            isinstance(inference, Mapping) and inference.get("device") != "cpu",
            isinstance(inference, Mapping)
            and inference.get("torch_inference_mode") is not True,
            isinstance(inference, Mapping) and inference.get("model_eval") is not True,
            isinstance(inference, Mapping)
            and inference.get("gene_mask_nonzero_count") != 0,
            isinstance(inference, Mapping)
            and inference.get("complete_core") is not True,
            isinstance(inference, Mapping)
            and inference.get("neighbor_sampling") is not False,
            isinstance(inference, Mapping)
            and inference.get("coordinates_supplied_to_node_encoder") is not False,
            isinstance(inference, Mapping)
            and inference.get("metadata_unchanged") is not True,
            isinstance(inference, Mapping)
            and inference.get("expression_unchanged") is not True,
            isinstance(inference, Mapping)
            and inference.get("all_cell_expression_decode_performed") is not False,
            not isinstance(invariance, Mapping),
            isinstance(invariance, Mapping)
            and invariance.get("verified_on_completed_trained_model") is not True,
            isinstance(invariance, Mapping)
            and invariance.get("predictions_allclose") is not True,
            isinstance(invariance, Mapping)
            and invariance.get(
                "extended_h0_equals_direct_h0_at_probe_rows_exact"
            )
            is not True,
            isinstance(invariance, Mapping)
            and invariance.get("extended_hL_equals_full_hL_at_probe_rows_exact")
            is not True,
            isinstance(invariance, Mapping)
            and invariance.get("probe_count")
            != min(PREDICTION_INVARIANCE_PROBE_COUNT, int(expected["cell_count"])),
            isinstance(invariance, Mapping)
            and invariance.get("absolute_tolerance")
            != PREDICTION_INVARIANCE_ATOL,
            isinstance(invariance, Mapping)
            and invariance.get("relative_tolerance")
            != PREDICTION_INVARIANCE_RTOL,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 per-core extraction receipt identity is invalid."
        )
    relative = str(receipt.get("embedding_file", ""))
    file_path = output_root / relative
    if not file_path.is_file() or receipt.get("file") != _file_record(file_path):
        raise SO1ModelEmbeddingClusteringError(
            f"SO1 per-core embedding file changed: {relative}."
        )
    core = load_core_embeddings(
        file_path,
        alias=str(expected["alias"]),
        core_number=int(expected["core_number"]),
        expected_cells=int(expected["cell_count"]),
    )
    checksums = receipt.get("array_checksums")
    shapes = receipt.get("array_shapes")
    if not isinstance(checksums, Mapping) or not isinstance(shapes, Mapping):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 per-core receipt lacks array records."
        )
    arrays = {
        "cell_index": core.cell_index,
        "coordinates_um": core.coordinates_um,
        "h0": core.h0,
        "hL": core.hL,
        "delta_h": core.delta_h,
        "delta_h_l2": core.delta_h_l2,
    }
    for name, value in arrays.items():
        if checksums.get(name) != _array_sha256(name, value) or shapes.get(name) != list(
            value.shape
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"SO1 per-core array receipt changed: {name}."
            )
    return core


def _verify_extraction_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO1ResolvedInputs,
) -> None:
    _verify_self_hash(receipt, label="SO1 embedding extraction manifest")
    if any(
        (
            receipt.get("schema") != EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            receipt.get("analysis_code_provenance")
            != _analysis_code_provenance_record(output_root),
            tuple(receipt.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(receipt.get("embedding_dimension", -1)) != EXPECTED_HIDDEN_DIM,
            receipt.get("device") != "cpu",
            receipt.get("inference_schedule") != "locked_manifest_core_order",
            receipt.get("all_zero_extraction_masks") is not True,
            receipt.get("all_cell_expression_decode_performed") is not False,
            receipt.get("prediction_invariance", {}).get(
                "verified_on_all_fourteen_completed_trained_cores"
            )
            is not True,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 embedding extraction manifest identity is invalid."
        )
    records = receipt.get("cores")
    expected_records = _expected_core_records(inputs)
    if not isinstance(records, list) or len(records) != len(expected_records):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 extraction manifest is incomplete."
        )
    for observed, expected in zip(records, expected_records, strict=True):
        if not isinstance(observed, Mapping):
            raise SO1ModelEmbeddingClusteringError(
                "SO1 per-core extraction receipt is malformed."
            )
        _verify_core_extraction_receipt(
            output_root=output_root,
            receipt=observed,
            inputs=inputs,
            expected=expected,
        )


def extract_intermediate_embeddings(
    *,
    inputs: SO1ResolvedInputs,
    output_root: Path,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
) -> Mapping[str, Any]:
    """Extract or verify h0/hL/delta for all 14 complete SO1 cores."""

    validate_cpu_device(device)
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise SO1ModelEmbeddingClusteringError("cpu_threads must be positive.")
    receipt_path = output_root / "embeddings" / "extraction_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO1 extraction manifest")
        _verify_extraction_manifest(
            output_root=output_root, receipt=receipt, inputs=inputs
        )
        return receipt

    torch.set_num_threads(int(cpu_threads))
    model = load_so1_checkpoint_model(inputs)
    expected_records = _expected_core_records(inputs)
    output_root.joinpath("embeddings").mkdir(parents=True, exist_ok=True)
    receipts_by_alias: dict[str, Mapping[str, Any]] = {}
    for expected in expected_records:
        alias = str(expected["alias"])
        core_number = int(expected["core_number"])
        file_path = _core_embedding_path(output_root, core_number)
        core_receipt_path = _core_receipt_path(output_root, core_number)
        if core_receipt_path.is_file():
            core_receipt = _read_json(
                core_receipt_path, label=f"{alias} extraction receipt"
            )
            _verify_core_extraction_receipt(
                output_root=output_root,
                receipt=core_receipt,
                inputs=inputs,
                expected=expected,
            )
            receipts_by_alias[alias] = core_receipt
            continue
        if file_path.exists():
            raise SO1ModelEmbeddingClusteringError(
                f"Unreceipted partial embedding file exists for {alias}."
            )

        core_input = _load_core_inputs(inputs, expected)
        expression = core_input.target_expression.to(dtype=torch.float32, device="cpu")
        covariates = core_input.node_covariates.to(dtype=torch.float32, device="cpu")
        gene_mask = torch.zeros_like(expression, dtype=torch.bool, device="cpu")
        expression_before = _tensor_sha256("target_expression", expression)
        metadata_before = _tensor_sha256("node_covariates", covariates)
        started = time.monotonic()
        h0_tensor, hL_tensor, prediction_invariance = extract_full_h0_hl(
            model,
            input_expression=expression,
            gene_mask=gene_mask,
            edge_index=core_input.edge_index,
            relative_geometry=core_input.relative_geometry,
            node_covariates=covariates,
        )
        elapsed = time.monotonic() - started
        expression_after = _tensor_sha256("target_expression", expression)
        metadata_after = _tensor_sha256("node_covariates", covariates)
        if expression_before != expression_after or metadata_before != metadata_after:
            raise SO1ModelEmbeddingClusteringError(
                f"Expression or metadata mutated during extraction for {alias}."
            )
        h0 = np.ascontiguousarray(
            h0_tensor.detach().cpu().float().numpy(), dtype=np.float32
        )
        hL = np.ascontiguousarray(
            hL_tensor.detach().cpu().float().numpy(), dtype=np.float32
        )
        delta = np.ascontiguousarray(hL - h0, dtype=np.float32)
        delta_norm = np.ascontiguousarray(
            np.linalg.norm(delta.astype(np.float64), axis=1), dtype=np.float64
        )
        validated = _validate_core_embedding_arrays(
            alias=alias,
            core_number=core_number,
            arrays={
                "cell_index": np.arange(core_input.n_cells, dtype=np.int64),
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": core_input.coordinates_um,
                "h0": h0,
                "hL": hL,
                "delta_h": delta,
                "delta_h_l2": delta_norm,
            },
            expected_cells=core_input.n_cells,
        )
        _write_deterministic_npz(
            file_path,
            {
                "cell_index": validated.cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": validated.coordinates_um,
                "h0": validated.h0,
                "hL": validated.hL,
                "delta_h": validated.delta_h,
                "delta_h_l2": validated.delta_h_l2,
            },
        )
        array_values = {
            "cell_index": validated.cell_index,
            "coordinates_um": validated.coordinates_um,
            "h0": validated.h0,
            "hL": validated.hL,
            "delta_h": validated.delta_h,
            "delta_h_l2": validated.delta_h_l2,
        }
        core_receipt = _receipt_with_self_hash(
            {
                "schema": CORE_EXTRACTION_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": inputs.run_id,
                "checkpoint_sha256": inputs.checkpoint_sha256,
                "analysis_code_provenance": _analysis_code_provenance_record(
                    output_root
                ),
                "alias": alias,
                "core_number": core_number,
                "cell_count": validated.n_cells,
                "source_core": dict(expected),
                "array_shapes": {
                    name: list(value.shape) for name, value in array_values.items()
                },
                "array_checksums": {
                    name: _array_sha256(name, value)
                    for name, value in array_values.items()
                },
                "source_expression_sha256": expression_before,
                "source_metadata_sha256": metadata_before,
                "inference": {
                    "device": "cpu",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cpu_threads": int(cpu_threads),
                    "model_eval": True,
                    "torch_inference_mode": True,
                    "dropout_disabled": True,
                    "attention_dropout_probability": 0.0,
                    "gene_mask_nonzero_count": 0,
                    "complete_core": True,
                    "neighbor_sampling": False,
                    "decoder_target_rows_per_forward": int(
                        prediction_invariance["probe_count"]
                    ),
                    "all_cell_expression_decode_performed": False,
                    "prediction_invariance": prediction_invariance,
                    "h0_source": "model.encoder_all_nodes_before_graph",
                    "hL_source": "model_output.full_node_embedding_pre_decoder",
                    "coordinates_supplied_to_node_encoder": False,
                    "metadata_unchanged": True,
                    "expression_unchanged": True,
                    "elapsed_seconds": float(elapsed),
                },
                "embedding_file": file_path.relative_to(output_root).as_posix(),
                "file": _file_record(file_path),
            }
        )
        _atomic_write_json(core_receipt_path, core_receipt)
        _verify_core_extraction_receipt(
            output_root=output_root,
            receipt=core_receipt,
            inputs=inputs,
            expected=expected,
        )
        receipts_by_alias[alias] = core_receipt
        del core_input, expression, covariates, gene_mask
        del h0_tensor, hL_tensor, h0, hL, delta, delta_norm, validated
        del prediction_invariance
        model.clear_edge_layout_cache()
        gc.collect()

    ordered_receipts = [receipts_by_alias[alias] for alias in SO1_ALIASES]
    receipt = _receipt_with_self_hash(
        {
            "schema": EXTRACTION_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "analysis_code_provenance": _analysis_code_provenance_record(
                output_root
            ),
            "core_order": list(SO1_CORE_NUMBERS),
            "alias_order": list(SO1_ALIASES),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_dimension": EXPECTED_HIDDEN_DIM,
            "representations": {
                "h0": "all_node_NodeEncoder_output_before_graph",
                "hL": "all_node_final_graph_output_before_decoder",
                "delta_h": "hL_minus_h0",
                "delta_h_l2": "descriptive_contextual_representation_change_magnitude",
            },
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": int(cpu_threads),
            "one_complete_core_at_a_time": True,
            "inference_schedule": "locked_manifest_core_order",
            "stored_core_order": list(SO1_CORE_NUMBERS),
            "all_zero_extraction_masks": True,
            "all_cell_expression_decode_performed": False,
            "prediction_invariance": {
                "verified_on_all_fourteen_completed_trained_cores": True,
                "core_count": len(ordered_receipts),
                "probe_count_per_core": {
                    str(record["core_number"]): int(
                        record["inference"]["prediction_invariance"]["probe_count"]
                    )
                    for record in ordered_receipts
                },
                "all_predictions_allclose": all(
                    bool(
                        record["inference"]["prediction_invariance"][
                            "predictions_allclose"
                        ]
                    )
                    for record in ordered_receipts
                ),
                "all_predictions_exact": all(
                    bool(
                        record["inference"]["prediction_invariance"][
                            "predictions_exact"
                        ]
                    )
                    for record in ordered_receipts
                ),
                "maximum_absolute_prediction_difference": max(
                    float(
                        record["inference"]["prediction_invariance"][
                            "maximum_absolute_prediction_difference"
                        ]
                    )
                    for record in ordered_receipts
                ),
                "absolute_tolerance": PREDICTION_INVARIANCE_ATOL,
                "relative_tolerance": PREDICTION_INVARIANCE_RTOL,
                "all_node_hL_reused_from_extended_forward": True,
                "all_cell_expression_decode_performed": False,
            },
            "input_provenance": inputs.provenance,
            "cores": ordered_receipts,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_extraction_manifest(output_root=output_root, receipt=receipt, inputs=inputs)
    del model
    gc.collect()
    return receipt


def load_all_core_embeddings(
    *, output_root: Path, extraction_receipt: Mapping[str, Any]
) -> tuple[SO1CoreEmbeddings, ...]:
    records = extraction_receipt.get("cores")
    if not isinstance(records, list) or tuple(
        int(record.get("core_number", -1))
        for record in records
        if isinstance(record, Mapping)
    ) != SO1_CORE_NUMBERS:
        raise SO1ModelEmbeddingClusteringError(
            "Extraction does not contain all 14 ordered SO1 cores."
        )
    result = []
    for record in records:
        assert isinstance(record, Mapping)
        result.append(
            load_core_embeddings(
                output_root / str(record["embedding_file"]),
                alias=str(record["alias"]),
                core_number=int(record["core_number"]),
                expected_cells=int(record["cell_count"]),
            )
        )
    if sum(core.n_cells for core in result) != EXPECTED_TOTAL_CELLS:
        raise SO1ModelEmbeddingClusteringError("Embedding extraction dropped cells.")
    return tuple(result)


def concatenate_representation(
    cores: Sequence[SO1CoreEmbeddings], *, representation: str
) -> np.ndarray:
    if representation not in REPRESENTATIONS:
        raise SO1ModelEmbeddingClusteringError("Unknown representation.")
    if tuple(core.core_number for core in cores) != SO1_CORE_NUMBERS:
        raise SO1ModelEmbeddingClusteringError(
            "Direct clustering requires all 14 SO1 cores in locked order."
        )
    name = REPRESENTATION_ARRAYS[representation]
    combined = np.ascontiguousarray(
        np.concatenate([getattr(core, name) for core in cores], axis=0),
        dtype=np.float32,
    )
    if combined.shape != (EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM):
        raise SO1ModelEmbeddingClusteringError(
            f"Joint {representation} embedding shape is invalid."
        )
    if not np.isfinite(combined).all() or not np.any(
        np.var(combined, axis=0, dtype=np.float64) > 0.0
    ):
        raise SO1ModelEmbeddingClusteringError(
            f"Joint {representation} embeddings are non-finite or zero variance."
        )
    return combined


def l2_normalize_for_cosine(
    embeddings: np.ndarray, *, representation: str
) -> tuple[np.ndarray, Mapping[str, Any]]:
    values = np.asarray(embeddings)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SO1ModelEmbeddingClusteringError("Cosine input must be finite 2D data.")
    before = _array_sha256(f"joint_{representation}_embedding", values)
    squared = np.einsum("ij,ij->i", values, values, dtype=np.float64, optimize=True)
    norms = np.sqrt(squared, dtype=np.float64)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise SO1ModelEmbeddingClusteringError(
            f"{representation} contains a zero-norm cell."
        )
    normalized = np.empty(values.shape, dtype=np.float32, order="C")
    np.divide(
        values,
        norms.astype(np.float32, copy=False)[:, None],
        out=normalized,
        casting="unsafe",
    )
    normalized_norms = np.sqrt(
        np.einsum(
            "ij,ij->i", normalized, normalized, dtype=np.float64, optimize=True
        ),
        dtype=np.float64,
    )
    if not np.allclose(normalized_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise SO1ModelEmbeddingClusteringError("Cosine L2 normalization failed.")
    if before != _array_sha256(f"joint_{representation}_embedding", values):
        raise SO1ModelEmbeddingClusteringError("Raw embeddings mutated during setup.")
    return normalized, {
        "operation": "row_l2_normalization_for_cosine_distance_only",
        "representation": representation,
        "representation_learning_or_reduction": False,
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "input_shape": list(values.shape),
        "output_shape": list(normalized.shape),
        "minimum_input_l2_norm": float(norms.min()),
        "maximum_input_l2_norm": float(norms.max()),
        "input_embedding_sha256": before,
        "normalized_sha256": _array_sha256(
            f"{representation}_l2_for_cosine", normalized
        ),
        "raw_embedding_mutated": False,
        "cell_by_cell_matrix_constructed": False,
    }


def _seeded_permuted_knn(
    normalized: np.ndarray,
    *,
    representation: str,
    n_neighbors: int,
    random_seed: int,
) -> KNNGraphResult:
    if representation not in REPRESENTATIONS:
        raise SO1ModelEmbeddingClusteringError("Unknown kNN representation.")
    # Recompute each representation's graph independently, but use the same
    # deterministic insertion realization so h0-vs-hL comparisons do not
    # inherit representation-specific HNSW approximation noise.
    insertion_seed = int(random_seed) ^ KNN_INSERTION_SEED_XOR
    insertion_to_original = np.ascontiguousarray(
        np.random.default_rng(insertion_seed).permutation(len(normalized)),
        dtype=np.int64,
    )
    internal = build_faiss_cosine_knn_graph(
        np.ascontiguousarray(normalized[insertion_to_original]),
        n_neighbors=int(n_neighbors),
        random_seed=int(random_seed),
    )
    mapped_edges = insertion_to_original[internal.edge_pairs]
    low = np.minimum(mapped_edges[:, 0], mapped_edges[:, 1])
    high = np.maximum(mapped_edges[:, 0], mapped_edges[:, 1])
    codes = low * np.int64(len(normalized)) + high
    unique_codes = np.unique(codes)
    edge_pairs = np.ascontiguousarray(
        np.column_stack(
            (
                unique_codes // np.int64(len(normalized)),
                unique_codes % np.int64(len(normalized)),
            )
        ),
        dtype=np.int64,
    )
    if internal.directed_neighbors is None:
        raise SO1ModelEmbeddingClusteringError(
            "FAISS result lacks directed neighbors for recall QC."
        )
    mapped_directed = insertion_to_original[internal.directed_neighbors]
    directed = np.empty_like(mapped_directed)
    directed[insertion_to_original] = mapped_directed
    directed = np.ascontiguousarray(directed, dtype=np.int64)
    receipt = dict(internal.receipt)
    receipt.update(
        {
            "input_order": (
                "deterministic_seeded_permutation_then_mapped_to_locked_global_rows"
            ),
            "insertion_permutation_seed": insertion_seed,
            "insertion_permutation_sha256": _array_sha256(
                "knn_insertion_to_original", insertion_to_original
            ),
            "shared_insertion_seed_across_independent_representations": True,
            "core_labels_consulted_for_insertion_order": False,
            "representation_specific_graph": True,
            "neighbors_sha256": _array_sha256("knn_neighbors", directed),
            "undirected_edges_sha256": _array_sha256(
                "knn_undirected_edges", edge_pairs
            ),
            "faiss_internal_neighbors_sha256": internal.receipt.get(
                "neighbors_sha256"
            ),
            "faiss_internal_edges_sha256": internal.receipt.get(
                "undirected_edges_sha256"
            ),
        }
    )
    return KNNGraphResult(
        edge_pairs=edge_pairs,
        receipt=receipt,
        directed_neighbors=directed,
    )


def exact_neighbor_recall_audit(
    normalized_embeddings: np.ndarray,
    directed_neighbors: np.ndarray,
    *,
    n_neighbors: int,
    query_count: int = DEFAULT_RECALL_QUERY_COUNT,
    query_batch_size: int = DEFAULT_RECALL_QUERY_BATCH_SIZE,
) -> Mapping[str, Any]:
    values = np.ascontiguousarray(normalized_embeddings, dtype=np.float32)
    neighbors = np.asarray(directed_neighbors, dtype=np.int64)
    n_cells = int(values.shape[0])
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SO1ModelEmbeddingClusteringError("Recall-audit embeddings are invalid.")
    if neighbors.shape != (n_cells, int(n_neighbors)):
        raise SO1ModelEmbeddingClusteringError("Recall-audit neighbor rows are invalid.")
    if any(
        (
            isinstance(query_count, bool),
            int(query_count) <= 0,
            isinstance(query_batch_size, bool),
            int(query_batch_size) <= 0,
        )
    ):
        raise SO1ModelEmbeddingClusteringError("Recall-audit sizes must be positive.")
    sampled_count = min(int(query_count), n_cells)
    if sampled_count == n_cells:
        query_indices = np.arange(n_cells, dtype=np.int64)
    else:
        query_indices = np.rint(
            np.linspace(0, n_cells - 1, sampled_count, dtype=np.float64)
        ).astype(np.int64)
    if len(np.unique(query_indices)) != sampled_count:
        raise SO1ModelEmbeddingClusteringError("Recall queries are not unique.")
    exact_rows = np.empty((sampled_count, int(n_neighbors)), dtype=np.int64)
    recalls = np.empty(sampled_count, dtype=np.float64)
    global_indices = np.arange(n_cells, dtype=np.int64)
    maximum_batch_rows = min(int(query_batch_size), sampled_count)
    for start in range(0, sampled_count, int(query_batch_size)):
        stop = min(start + int(query_batch_size), sampled_count)
        batch_indices = query_indices[start:stop]
        pairwise = np.matmul(values[batch_indices], values.T)
        if pairwise.shape != (len(batch_indices), n_cells):
            raise SO1ModelEmbeddingClusteringError("Recall batch shape is invalid.")
        for local_row, query_index in enumerate(batch_indices):
            scores = pairwise[local_row]
            scores[int(query_index)] = -np.inf
            kth_offset = n_cells - int(n_neighbors)
            threshold = float(np.partition(scores, kth_offset)[kth_offset])
            above = np.flatnonzero(scores > threshold).astype(np.int64, copy=False)
            needed = int(n_neighbors) - len(above)
            tied = np.flatnonzero(scores == threshold).astype(np.int64, copy=False)
            if needed < 0 or len(tied) < needed:
                raise SO1ModelEmbeddingClusteringError(
                    "Exact recall tie resolution failed."
                )
            candidates = np.concatenate((above, tied[:needed]))
            order = np.lexsort((global_indices[candidates], -scores[candidates]))
            exact = candidates[order][: int(n_neighbors)]
            exact_rows[start + local_row] = exact
            recalls[start + local_row] = (
                len(np.intersect1d(neighbors[int(query_index)], exact))
                / float(n_neighbors)
            )
        del pairwise
    return {
        "method": "deterministic_evenly_spaced_queries_exact_cosine",
        "purpose": "approximate_HNSW_neighbor_recall_quality_control",
        "query_count_requested": int(query_count),
        "query_count_evaluated": int(sampled_count),
        "query_batch_size": int(query_batch_size),
        "n_neighbors": int(n_neighbors),
        "recall_at_k_mean": float(recalls.mean()),
        "recall_at_k_median": float(np.median(recalls)),
        "recall_at_k_minimum": float(recalls.min()),
        "recall_at_k_maximum": float(recalls.max()),
        "query_indices_sha256": _array_sha256(
            "exact_recall_query_indices", query_indices
        ),
        "exact_neighbors_sha256": _array_sha256(
            "sampled_exact_cosine_neighbors", exact_rows
        ),
        "per_query_recall_sha256": _array_sha256(
            "sampled_exact_recall_at_k", recalls
        ),
        "maximum_explicit_pairwise_array_shape": [maximum_batch_rows, n_cells],
        "cell_by_cell_matrix_constructed": False,
        "exact_similarity_matrix_persisted": False,
    }


def cluster_direct_representation(
    embeddings: np.ndarray,
    *,
    representation: str,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
    minimum_mean_recall: float = MINIMUM_MEAN_RECALL_AT_K,
) -> DirectRepresentationResult:
    if representation not in REPRESENTATIONS:
        raise SO1ModelEmbeddingClusteringError("Unknown direct representation.")
    raw = np.asarray(embeddings)
    domain = f"joint_{representation}_embedding"
    before = _array_sha256(domain, raw)
    normalized, normalization = l2_normalize_for_cosine(
        raw, representation=representation
    )
    knn = _seeded_permuted_knn(
        normalized,
        representation=representation,
        n_neighbors=n_neighbors,
        random_seed=random_seed,
    )
    assert knn.directed_neighbors is not None
    recall = dict(
        exact_neighbor_recall_audit(
            normalized,
            knn.directed_neighbors,
            n_neighbors=n_neighbors,
        )
    )
    recall["minimum_accepted_mean_recall_at_k"] = float(minimum_mean_recall)
    recall["mean_recall_acceptance_passed"] = bool(
        recall["recall_at_k_mean"] >= float(minimum_mean_recall)
    )
    if not recall["mean_recall_acceptance_passed"]:
        raise SO1ModelEmbeddingClusteringError(
            f"{representation} HNSW sampled exact recall is below "
            f"{minimum_mean_recall:.2f}."
        )
    leiden = run_seeded_leiden(
        knn,
        n_cells=len(normalized),
        resolution=leiden_resolution,
        random_seed=random_seed,
    )
    labels = np.ascontiguousarray(leiden.labels, dtype=np.int64)
    after = _array_sha256(domain, raw)
    if before != after:
        raise SO1ModelEmbeddingClusteringError(
            f"Raw {representation} embedding mutated during clustering."
        )
    return DirectRepresentationResult(
        labels=labels,
        edge_pairs=np.ascontiguousarray(knn.edge_pairs, dtype=np.int64),
        receipt={
            "pipeline_kind": DIRECT_PIPELINE_KIND,
            "representation": representation,
            "embedding_array": REPRESENTATION_ARRAYS[representation],
            "joint_embedding_shape": list(raw.shape),
            "joint_embedding_sha256": before,
            "pca": False,
            "mean_center": False,
            "l2_normalize_for_cosine": True,
            "cosine_distance_setup": normalization,
            "knn": dict(knn.receipt),
            "knn_exact_recall_audit": recall,
            "leiden": dict(leiden.receipt),
            "raw_embedding_sha256_after_pipeline": after,
            "raw_embedding_mutated": False,
        },
    )


def _cell_frame(cores: Sequence[SO1CoreEmbeddings]) -> pd.DataFrame:
    frames = []
    offset = 0
    for core in cores:
        expected = EXPECTED_CELL_COUNTS_BY_CORE[core.core_number]
        if core.n_cells != expected:
            raise SO1ModelEmbeddingClusteringError(
                f"Cell-frame count changed for core {core.core_number}."
            )
        frames.append(
            pd.DataFrame(
                {
                    "global_cell_index": np.arange(
                        offset, offset + expected, dtype=np.int64
                    ),
                    "cell_index": core.cell_index,
                    "cell_key": [
                        f"{core.alias}:{int(value):08d}" for value in core.cell_index
                    ],
                    "core_alias": core.alias,
                    "core_number": np.full(expected, core.core_number, dtype=np.int16),
                    "x_um": core.coordinates_um[:, 0],
                    "y_um": core.coordinates_um[:, 1],
                    "delta_h_l2": core.delta_h_l2,
                }
            )
        )
        offset += expected
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO1_CORE_NUMBERS:
        raise SO1ModelEmbeddingClusteringError("Joint cell frame is incomplete.")
    return frame


def _cluster_summary_tables(
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    memberships = np.asarray(labels, dtype=np.int64)
    if memberships.shape != (EXPECTED_TOTAL_CELLS,) or np.any(memberships < 0):
        raise SO1ModelEmbeddingClusteringError("Cluster labels are invalid.")
    unique = np.unique(memberships)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise SO1ModelEmbeddingClusteringError("Cluster numbers are not contiguous.")
    core_values = frame["core_number"].to_numpy(dtype=np.int64)
    core_totals = {
        core: int(np.count_nonzero(core_values == core)) for core in SO1_CORE_NUMBERS
    }
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    dominated: list[str] = []
    for cluster_number in unique:
        selected = memberships == cluster_number
        size = int(np.count_nonzero(selected))
        label = f"{prefix}{int(cluster_number)}"
        counts = {
            core: int(np.count_nonzero(core_values[selected] == core))
            for core in SO1_CORE_NUMBERS
        }
        dominant_core = min(
            SO1_CORE_NUMBERS,
            key=lambda core: (-counts[core], SO1_CORE_NUMBERS.index(core)),
        )
        dominant_proportion = counts[dominant_core] / size
        is_dominated = dominant_proportion > 0.90
        if is_dominated:
            dominated.append(label)
        summary_rows.append(
            {
                "cluster": label,
                "cluster_number": int(cluster_number),
                "size": size,
                "proportion": size / EXPECTED_TOTAL_CELLS,
                "dominant_core": dominant_core,
                "dominant_core_count": counts[dominant_core],
                "dominant_core_proportion": dominant_proportion,
                "core_dominated_gt_90pct": is_dominated,
            }
        )
        for core in SO1_CORE_NUMBERS:
            composition_rows.append(
                {
                    "cluster": label,
                    "cluster_number": int(cluster_number),
                    "core_number": core,
                    "cell_count": counts[core],
                    "proportion_within_cluster": counts[core] / size,
                    "proportion_within_core": counts[core] / core_totals[core],
                    "cluster_size": size,
                    "core_dominated_gt_90pct": is_dominated,
                }
            )
    summary = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(composition_rows)
    if int(summary["size"].sum()) != EXPECTED_TOTAL_CELLS:
        raise SO1ModelEmbeddingClusteringError("Cluster summary dropped cells.")
    return summary, composition, dominated


def _table_values_match(expected: pd.DataFrame, observed: pd.DataFrame) -> bool:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        return False
    for column in expected.columns:
        left = expected[column]
        right = observed[column]
        if pd.api.types.is_numeric_dtype(left.dtype):
            try:
                left_values = left.to_numpy(dtype=np.float64)
                right_values = pd.to_numeric(right, errors="raise").to_numpy(
                    dtype=np.float64
                )
            except (TypeError, ValueError):
                return False
            if not np.allclose(
                left_values,
                right_values,
                rtol=1.0e-12,
                atol=1.0e-12,
                equal_nan=False,
            ):
                return False
        elif not np.array_equal(
            left.astype(str).to_numpy(), right.astype(str).to_numpy()
        ):
            return False
    return True


def _palette(cluster_count: int, *, representation: str) -> dict[str, str]:
    base = deterministic_glasbey_palette(cluster_count, namespace=representation)
    source_prefix = "I" if representation == "intrinsic" else "C"
    target_prefix = REPRESENTATION_PREFIXES[representation]
    result = {
        f"{target_prefix}{int(label[len(source_prefix):])}": color
        for label, color in base.items()
    }
    if len(result) != cluster_count or len(set(result.values())) != cluster_count:
        raise SO1ModelEmbeddingClusteringError("Cluster palette is invalid.")
    return result


def _clustering_configuration(
    *, n_neighbors: int, leiden_resolution: float, random_seed: int
) -> dict[str, Any]:
    if any(
        (
            int(n_neighbors) != DEFAULT_N_NEIGHBORS,
            not math.isclose(
                float(leiden_resolution),
                DEFAULT_LEIDEN_RESOLUTION,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            int(random_seed) != DEFAULT_RANDOM_SEED,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Primary SO1 direct clustering is locked to k=30, resolution=1.0, "
            "seed=20260825."
        )
    return {
        "pipeline": DIRECT_PIPELINE_KIND,
        "representations": ["h0", "hL"],
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "n_neighbors": DEFAULT_N_NEIGHBORS,
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union_unweighted",
        "seeded_insertion_permutation_independent_of_core": True,
        "shared_insertion_seed_across_independent_representations": True,
        "minimum_mean_exact_recall_at_k": MINIMUM_MEAN_RECALL_AT_K,
        "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
        "random_seed": DEFAULT_RANDOM_SEED,
        "cluster_sort": (
            "descending_size_then_minimum_global_cell_index_then_raw_id"
        ),
        "joint_core_order": list(SO1_CORE_NUMBERS),
        "joint_cell_count": EXPECTED_TOTAL_CELLS,
        "intrinsic_label_prefix": INTRINSIC_LABEL_PREFIX,
        "contextual_label_prefix": CONTEXTUAL_LABEL_PREFIX,
        "independent_representation_graphs": True,
        "cross_core_embedding_neighbors_permitted": True,
        "spatial_training_graph_reused_for_clustering": False,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }


def _verify_clustering_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    extraction_sha256: str,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO1 direct clustering manifest")
    if any(
        (
            receipt.get("schema") != CLUSTERING_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != EXPECTED_RUN_ID,
            receipt.get("extraction_manifest_sha256") != extraction_sha256,
            receipt.get("analysis_code_provenance")
            != _analysis_code_provenance_record(output_root),
            receipt.get("configuration") != dict(configuration),
            tuple(receipt.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            receipt.get("independent_representation_graphs") is not True,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 direct clustering manifest identity is invalid."
        )
    pipelines = receipt.get("pipelines")
    intrinsic_knn = (
        pipelines.get("intrinsic", {}).get("knn")
        if isinstance(pipelines, Mapping)
        else None
    )
    contextual_knn = (
        pipelines.get("contextual", {}).get("knn")
        if isinstance(pipelines, Mapping)
        else None
    )
    if any(
        (
            not isinstance(intrinsic_knn, Mapping),
            not isinstance(contextual_knn, Mapping),
            isinstance(intrinsic_knn, Mapping)
            and intrinsic_knn.get(
                "shared_insertion_seed_across_independent_representations"
            )
            is not True,
            isinstance(contextual_knn, Mapping)
            and contextual_knn.get(
                "shared_insertion_seed_across_independent_representations"
            )
            is not True,
            isinstance(intrinsic_knn, Mapping)
            and isinstance(contextual_knn, Mapping)
            and intrinsic_knn.get("insertion_permutation_seed")
            != contextual_knn.get("insertion_permutation_seed"),
            isinstance(intrinsic_knn, Mapping)
            and isinstance(contextual_knn, Mapping)
            and intrinsic_knn.get("insertion_permutation_sha256")
            != contextual_knn.get("insertion_permutation_sha256"),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Independent representation graphs do not share the locked kNN "
            "insertion realization."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO1ModelEmbeddingClusteringError("Clustering manifest lacks files.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO1ModelEmbeddingClusteringError(
                f"Clustering output checksum changed: {relative}."
            )
    intrinsic = np.load(
        output_root / "clustering" / "intrinsic_labels.npy", allow_pickle=False
    )
    contextual = np.load(
        output_root / "clustering" / "contextual_labels.npy", allow_pickle=False
    )
    if intrinsic.shape != (EXPECTED_TOTAL_CELLS,) or contextual.shape != (
        EXPECTED_TOTAL_CELLS,
    ):
        raise SO1ModelEmbeddingClusteringError("Stored cluster labels are incomplete.")
    table = pd.read_parquet(output_root / "tables" / "cell_embedding_clusters.parquet")
    expected_core_numbers = np.concatenate(
        [
            np.full(EXPECTED_CELL_COUNTS_BY_CORE[core], core, dtype=np.int16)
            for core in SO1_CORE_NUMBERS
        ]
    )
    expected_cell_indices = np.concatenate(
        [
            np.arange(EXPECTED_CELL_COUNTS_BY_CORE[core], dtype=np.int64)
            for core in SO1_CORE_NUMBERS
        ]
    )
    if any(
        (
            len(table) != EXPECTED_TOTAL_CELLS,
            tuple(table["core_number"].drop_duplicates().tolist())
            != SO1_CORE_NUMBERS,
            not np.array_equal(
                table["core_number"].to_numpy(dtype=np.int16),
                expected_core_numbers,
            ),
            not np.array_equal(
                table["cell_index"].to_numpy(dtype=np.int64),
                expected_cell_indices,
            ),
            not np.array_equal(
                table["intrinsic_cluster_number"].to_numpy(dtype=np.int64),
                intrinsic,
            ),
            not np.array_equal(
                table["contextual_cluster_number"].to_numpy(dtype=np.int64),
                contextual,
            ),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Stored cluster table and labels are misaligned."
        )
    for representation, labels, prefix in (
        ("intrinsic", intrinsic, INTRINSIC_LABEL_PREFIX),
        ("contextual", contextual, CONTEXTUAL_LABEL_PREFIX),
    ):
        unique_labels = np.unique(labels)
        if not np.array_equal(
            unique_labels, np.arange(len(unique_labels), dtype=np.int64)
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} cluster numbers are not contiguous."
            )
        cluster_count = int(np.max(labels)) + 1
        expected_palette = _palette(cluster_count, representation=representation)
        document = _read_json(
            output_root / "clustering" / f"{representation}_palette.json",
            label=f"SO1 {representation} palette",
        )
        if document.get("colors") != expected_palette:
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} palette is not deterministic."
            )
        expected_labels = np.asarray(
            [f"{prefix}{int(value)}" for value in labels], dtype=object
        )
        if not np.array_equal(
            table[f"{representation}_cluster"].astype(str).to_numpy(),
            expected_labels,
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} label namespace is invalid."
            )
        expected_summary, expected_composition, expected_dominated = (
            _cluster_summary_tables(labels, table, prefix=prefix)
        )
        observed_summary = pd.read_csv(
            output_root / "tables" / f"{representation}_cluster_summary.csv"
        )
        observed_composition = pd.read_csv(
            output_root
            / "tables"
            / f"{representation}_cluster_core_composition.csv"
        )
        if not _table_values_match(expected_summary, observed_summary) or not (
            _table_values_match(expected_composition, observed_composition)
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} summary tables are inconsistent."
            )
        if any(
            (
                receipt.get("cluster_counts", {}).get(representation)
                != cluster_count,
                receipt.get("cluster_size_ranges", {}).get(representation)
                != [
                    int(expected_summary["size"].min()),
                    int(expected_summary["size"].max()),
                ],
                receipt.get("core_dominated_gt_90pct", {}).get(representation)
                != expected_dominated,
                receipt.get("palettes", {}).get(representation)
                != expected_palette,
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} clustering summary receipt is invalid."
            )
        edge_pairs = np.load(
            output_root
            / "clustering"
            / f"{representation}_knn_undirected_edges.npy",
            allow_pickle=False,
        )
        pipeline = receipt.get("pipelines", {}).get(representation)
        if any(
            (
                edge_pairs.ndim != 2,
                edge_pairs.ndim == 2 and edge_pairs.shape[1] != 2,
                not np.issubdtype(edge_pairs.dtype, np.integer),
                edge_pairs.size and int(edge_pairs.min()) < 0,
                edge_pairs.size and int(edge_pairs.max()) >= EXPECTED_TOTAL_CELLS,
                edge_pairs.ndim == 2
                and edge_pairs.shape[1] == 2
                and bool(np.any(edge_pairs[:, 0] >= edge_pairs[:, 1])),
                edge_pairs.ndim == 2
                and edge_pairs.shape[1] == 2
                and len(edge_pairs) > 1
                and bool(
                    np.any(
                        np.diff(
                            edge_pairs[:, 0] * np.int64(EXPECTED_TOTAL_CELLS)
                            + edge_pairs[:, 1]
                        )
                        <= 0
                    )
                ),
                not isinstance(pipeline, Mapping),
                isinstance(pipeline, Mapping)
                and pipeline.get("pipeline_kind") != DIRECT_PIPELINE_KIND,
                isinstance(pipeline, Mapping)
                and pipeline.get("leiden", {}).get("labels_sha256")
                != _array_sha256("sorted_leiden_labels", labels),
                isinstance(pipeline, Mapping)
                and pipeline.get("knn", {}).get("undirected_edges_sha256")
                != _array_sha256("knn_undirected_edges", edge_pairs),
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Stored {representation} sparse graph/receipt is invalid."
            )


def build_clustering_outputs(
    *,
    inputs: SO1ResolvedInputs,
    output_root: Path,
    extraction_receipt: Mapping[str, Any],
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Mapping[str, Any]:
    configuration = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    extraction_path = output_root / "embeddings" / "extraction_manifest.json"
    extraction_sha = sha256_file(extraction_path)
    receipt_path = output_root / "clustering" / "clustering_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO1 clustering manifest")
        _verify_clustering_manifest(
            output_root=output_root,
            receipt=receipt,
            extraction_sha256=extraction_sha,
            configuration=configuration,
        )
        return receipt
    clustering_dir = output_root / "clustering"
    tables_dir = output_root / "tables"
    if clustering_dir.exists() and any(clustering_dir.iterdir()):
        raise SO1ModelEmbeddingClusteringError(
            "Partial clustering outputs exist without a completed receipt."
        )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cores = load_all_core_embeddings(
        output_root=output_root, extraction_receipt=extraction_receipt
    )
    frame = _cell_frame(cores)
    intrinsic_values = concatenate_representation(cores, representation="intrinsic")
    contextual_values = concatenate_representation(cores, representation="contextual")
    intrinsic = cluster_direct_representation(
        intrinsic_values,
        representation="intrinsic",
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    contextual = cluster_direct_representation(
        contextual_values,
        representation="contextual",
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    if intrinsic.edge_pairs is contextual.edge_pairs or intrinsic.labels is contextual.labels:
        raise SO1ModelEmbeddingClusteringError(
            "Intrinsic and contextual clustering pipelines were not independent."
        )
    frame["intrinsic_cluster_number"] = intrinsic.labels.astype(np.int32)
    frame["contextual_cluster_number"] = contextual.labels.astype(np.int32)
    frame["intrinsic_cluster"] = [
        f"{INTRINSIC_LABEL_PREFIX}{int(value)}" for value in intrinsic.labels
    ]
    frame["contextual_cluster"] = [
        f"{CONTEXTUAL_LABEL_PREFIX}{int(value)}" for value in contextual.labels
    ]
    intrinsic_summary, intrinsic_composition, intrinsic_dominated = (
        _cluster_summary_tables(
            intrinsic.labels, frame, prefix=INTRINSIC_LABEL_PREFIX
        )
    )
    contextual_summary, contextual_composition, contextual_dominated = (
        _cluster_summary_tables(
            contextual.labels, frame, prefix=CONTEXTUAL_LABEL_PREFIX
        )
    )
    intrinsic_palette = _palette(len(intrinsic_summary), representation="intrinsic")
    contextual_palette = _palette(
        len(contextual_summary), representation="contextual"
    )

    output_paths = {
        "intrinsic_labels": clustering_dir / "intrinsic_labels.npy",
        "contextual_labels": clustering_dir / "contextual_labels.npy",
        "intrinsic_edges": clustering_dir / "intrinsic_knn_undirected_edges.npy",
        "contextual_edges": clustering_dir / "contextual_knn_undirected_edges.npy",
        "intrinsic_parameters": clustering_dir
        / "intrinsic_clustering_parameters.json",
        "contextual_parameters": clustering_dir
        / "contextual_clustering_parameters.json",
        "intrinsic_palette": clustering_dir / "intrinsic_palette.json",
        "contextual_palette": clustering_dir / "contextual_palette.json",
        "cell_table": tables_dir / "cell_embedding_clusters.parquet",
        "intrinsic_summary": tables_dir / "intrinsic_cluster_summary.csv",
        "contextual_summary": tables_dir / "contextual_cluster_summary.csv",
        "intrinsic_composition": tables_dir
        / "intrinsic_cluster_core_composition.csv",
        "contextual_composition": tables_dir
        / "contextual_cluster_core_composition.csv",
    }
    _atomic_write_npy(output_paths["intrinsic_labels"], intrinsic.labels)
    _atomic_write_npy(output_paths["contextual_labels"], contextual.labels)
    _atomic_write_npy(output_paths["intrinsic_edges"], intrinsic.edge_pairs)
    _atomic_write_npy(output_paths["contextual_edges"], contextual.edge_pairs)
    _atomic_write_json(output_paths["intrinsic_parameters"], intrinsic.receipt)
    _atomic_write_json(output_paths["contextual_parameters"], contextual.receipt)
    _atomic_write_json(
        output_paths["intrinsic_palette"],
        {
            "schema": "so1_intrinsic_palette_v1",
            "deterministic": True,
            "representation": "intrinsic_h0_direct",
            "label_prefix": INTRINSIC_LABEL_PREFIX,
            "colors": intrinsic_palette,
        },
    )
    _atomic_write_json(
        output_paths["contextual_palette"],
        {
            "schema": "so1_contextual_palette_v1",
            "deterministic": True,
            "representation": "contextual_hL_direct",
            "label_prefix": CONTEXTUAL_LABEL_PREFIX,
            "colors": contextual_palette,
        },
    )
    table_columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
        "intrinsic_cluster_number",
        "intrinsic_cluster",
        "contextual_cluster_number",
        "contextual_cluster",
        "delta_h_l2",
    ]
    _atomic_write_parquet(output_paths["cell_table"], frame.loc[:, table_columns])
    _atomic_write_csv(output_paths["intrinsic_summary"], intrinsic_summary)
    _atomic_write_csv(output_paths["contextual_summary"], contextual_summary)
    _atomic_write_csv(output_paths["intrinsic_composition"], intrinsic_composition)
    _atomic_write_csv(output_paths["contextual_composition"], contextual_composition)
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "extraction_manifest_sha256": extraction_sha,
            "analysis_code_provenance": _analysis_code_provenance_record(
                output_root
            ),
            "configuration": configuration,
            "core_order": list(SO1_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "independent_representation_graphs": True,
            "pipelines": {
                "intrinsic": intrinsic.receipt,
                "contextual": contextual.receipt,
            },
            "cluster_counts": {
                "intrinsic": int(len(intrinsic_summary)),
                "contextual": int(len(contextual_summary)),
            },
            "cluster_size_ranges": {
                "intrinsic": [
                    int(intrinsic_summary["size"].min()),
                    int(intrinsic_summary["size"].max()),
                ],
                "contextual": [
                    int(contextual_summary["size"].min()),
                    int(contextual_summary["size"].max()),
                ],
            },
            "core_dominated_gt_90pct": {
                "intrinsic": intrinsic_dominated,
                "contextual": contextual_dominated,
            },
            "palettes": {
                "intrinsic": intrinsic_palette,
                "contextual": contextual_palette,
            },
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in output_paths.values()
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_clustering_manifest(
        output_root=output_root,
        receipt=receipt,
        extraction_sha256=extraction_sha,
        configuration=configuration,
    )
    del cores, frame, intrinsic_values, contextual_values
    gc.collect()
    return receipt


def delta_norm_statistics(cores: Sequence[SO1CoreEmbeddings]) -> dict[str, Any]:
    values = np.concatenate([core.delta_h_l2 for core in cores]).astype(
        np.float64, copy=False
    )
    if len(values) != EXPECTED_TOTAL_CELLS or not np.isfinite(values).all() or np.any(
        values < 0.0
    ):
        raise SO1ModelEmbeddingClusteringError("Delta norms are invalid.")
    p01, p99 = np.quantile(values, [0.01, 0.99])
    if not math.isfinite(float(p01)) or not math.isfinite(float(p99)) or p01 >= p99:
        raise SO1ModelEmbeddingClusteringError("Delta plotting limits are invalid.")
    return {
        "description": "contextual representation-change magnitude",
        "biological_influence_or_causal_effect": False,
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=0)),
        "p01": float(p01),
        "p99": float(p99),
        "plotting_limits": {
            "vmin_global_p01": float(p01),
            "vmax_global_p99": float(p99),
            "shared_across_all_fourteen_cores": True,
            "raw_values_clipped_in_saved_data": False,
        },
    }


def _cluster_number(label: str, prefix: str) -> int:
    if not str(label).startswith(prefix):
        raise SO1ModelEmbeddingClusteringError(f"Invalid cluster label: {label}.")
    try:
        return int(str(label)[len(prefix) :])
    except ValueError as exc:
        raise SO1ModelEmbeddingClusteringError(
            f"Invalid cluster label: {label}."
        ) from exc


def _legend_handles(palette: Mapping[str, str], *, prefix: str) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: _cluster_number(item[0], prefix)
        )
    ]


def _atomic_save_png(figure: Any, *, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "spatial_benchmark.so1_model_embedding_clustering"},
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _render_combined_cluster_map(
    frame: pd.DataFrame,
    *,
    column: str,
    palette: Mapping[str, str],
    prefix: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    for axis, core_number in zip(
        axes.ravel()[:14], SO1_CORE_NUMBERS, strict=True
    ):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected[column].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS_BY_CORE[core_number] or colors.isna().any():
            raise SO1ModelEmbeddingClusteringError(
                f"Spatial cluster panel is incomplete for SO1 core {core_number}."
            )
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.52,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.92,
            rasterized=True,
        )
        axis.set_title(
            f"SO1 Core {core_number}\n$n$ = {len(selected):,}",
            fontsize=12,
            weight="bold",
        )
        _style_spatial_axis(axis, coordinates)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=_legend_handles(palette, prefix=prefix),
        loc="center",
        frameon=False,
        ncol=2 if len(palette) > 14 else 1,
        title="Joint model-derived cluster",
        fontsize=8,
        title_fontsize=10,
    )
    figure.suptitle(title, fontsize=18, weight="bold", y=0.995)
    figure.text(
        0.5,
        0.006,
        "Model-derived clusters; no cell-type or biological annotation is implied.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.925,
        wspace=0.27,
        hspace=0.34,
    )
    _atomic_save_figure_pair(
        figure,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=int(dpi),
        producer="spatial_benchmark.so1_model_embedding_clustering",
    )
    plt.close(figure)


def _render_combined_delta_map(
    frame: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    normalization = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    for axis, core_number in zip(
        axes.ravel()[:14], SO1_CORE_NUMBERS, strict=True
    ):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        values = selected["delta_h_l2"].to_numpy(dtype=np.float64)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.52,
            c=values,
            cmap="viridis",
            norm=normalization,
            marker="o",
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(
            f"SO1 Core {core_number}\n$n$ = {len(selected):,}",
            fontsize=12,
            weight="bold",
        )
        _style_spatial_axis(axis, coordinates)
    color_axis = axes.ravel()[-1]
    color_axis.axis("off")
    inset = color_axis.inset_axes([0.37, 0.16, 0.20, 0.68])
    colorbar = ColorbarBase(inset, cmap="viridis", norm=normalization, orientation="vertical")
    colorbar.set_label("||h_contextual − h_intrinsic||₂", fontsize=10)
    figure.suptitle(
        "SO1 contextual representation-change magnitude\nshared global p1–p99 scale",
        fontsize=18,
        weight="bold",
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.925,
        wspace=0.27,
        hspace=0.34,
    )
    _atomic_save_figure_pair(
        figure,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=int(dpi),
        producer="spatial_benchmark.so1_model_embedding_clustering",
    )
    plt.close(figure)


def _render_per_core_pngs(
    frame: pd.DataFrame,
    *,
    intrinsic_palette: Mapping[str, str],
    contextual_palette: Mapping[str, str],
    vmin: float,
    vmax: float,
    output_root: Path,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    outputs: list[Path] = []
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    definitions = (
        (
            "intrinsic_cluster",
            intrinsic_palette,
            "intrinsic_direct_h0_leiden_resolution_1p0_spatial_core_{core}.png",
            "Intrinsic direct-h0 cluster",
        ),
        (
            "contextual_cluster",
            contextual_palette,
            "contextual_direct_hl_leiden_resolution_1p0_spatial_core_{core}.png",
            "Contextual direct-hL cluster",
        ),
    )
    for core_number in SO1_CORE_NUMBERS:
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        for column, palette, template, title in definitions:
            figure, axis = plt.subplots(figsize=(8.0, 8.0))
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=1.0,
                c=selected[column].map(palette).tolist(),
                linewidths=0,
                edgecolors="none",
                rasterized=True,
            )
            axis.set_title(f"SO1 Core {core_number} — {title}", weight="bold")
            _style_spatial_axis(axis, coordinates)
            path = output_root / "figures" / "per_core" / template.format(
                core=core_number
            )
            _atomic_save_png(figure, path=path, dpi=dpi)
            plt.close(figure)
            outputs.append(path)
        figure, axis = plt.subplots(figsize=(8.0, 8.0))
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=1.0,
            c=selected["delta_h_l2"].to_numpy(dtype=np.float64),
            cmap="viridis",
            norm=norm,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(
            f"SO1 Core {core_number} — contextual representation change",
            weight="bold",
        )
        _style_spatial_axis(axis, coordinates)
        figure.colorbar(scatter, ax=axis, label="||h_contextual − h_intrinsic||₂")
        path = (
            output_root
            / "figures"
            / "per_core"
            / f"delta_h_l2_spatial_core_{core_number}.png"
        )
        _atomic_save_png(figure, path=path, dpi=dpi)
        plt.close(figure)
        outputs.append(path)
    return outputs


def _verify_figure_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    clustering_sha256: str,
    dpi: int,
) -> None:
    _verify_self_hash(receipt, label="SO1 spatial figure manifest")
    expected_files = set(_required_figure_files())
    files = receipt.get("files")
    if any(
        (
            receipt.get("schema") != FIGURE_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("clustering_manifest_sha256") != clustering_sha256,
            receipt.get("analysis_code_provenance")
            != _analysis_code_provenance_record(output_root),
            receipt.get("plot_specification") != spatial_plot_spec(),
            int(receipt.get("dpi", -1)) != int(dpi),
            int(receipt.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            not isinstance(files, Mapping),
            isinstance(files, Mapping) and set(files) != expected_files,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 spatial figure manifest identity is invalid."
        )
    assert isinstance(files, Mapping)
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO1ModelEmbeddingClusteringError(
                f"SO1 spatial figure checksum changed: {relative}."
            )


def render_spatial_outputs(
    *,
    output_root: Path,
    extraction_receipt: Mapping[str, Any],
    clustering_receipt: Mapping[str, Any],
    dpi: int = DEFAULT_DPI,
) -> Mapping[str, Any]:
    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO1ModelEmbeddingClusteringError("Figure DPI must be at least 72.")
    import matplotlib

    matplotlib.use("Agg", force=True)
    clustering_path = output_root / "clustering" / "clustering_manifest.json"
    clustering_sha = sha256_file(clustering_path)
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO1 figure manifest")
        _verify_figure_manifest(
            output_root=output_root,
            receipt=receipt,
            clustering_sha256=clustering_sha,
            dpi=int(dpi),
        )
        return receipt
    frame = pd.read_parquet(output_root / "tables" / "cell_embedding_clusters.parquet")
    cores = load_all_core_embeddings(
        output_root=output_root, extraction_receipt=extraction_receipt
    )
    delta = delta_norm_statistics(cores)
    palettes = clustering_receipt.get("palettes")
    if not isinstance(palettes, Mapping):
        raise SO1ModelEmbeddingClusteringError("Clustering palettes are missing.")
    intrinsic_palette = dict(palettes["intrinsic"])
    contextual_palette = dict(palettes["contextual"])
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_png = figure_dir / f"{INTRINSIC_COMBINED_STEM}.png"
    intrinsic_pdf = figure_dir / f"{INTRINSIC_COMBINED_STEM}.pdf"
    contextual_png = figure_dir / f"{CONTEXTUAL_COMBINED_STEM}.png"
    contextual_pdf = figure_dir / f"{CONTEXTUAL_COMBINED_STEM}.pdf"
    delta_png = figure_dir / f"{DELTA_COMBINED_STEM}.png"
    delta_pdf = figure_dir / f"{DELTA_COMBINED_STEM}.pdf"
    _render_combined_cluster_map(
        frame,
        column="intrinsic_cluster",
        palette=intrinsic_palette,
        prefix=INTRINSIC_LABEL_PREFIX,
        title="SO1 intrinsic h0 direct cosine-kNN Leiden clusters\nresolution 1.0",
        png_path=intrinsic_png,
        pdf_path=intrinsic_pdf,
        dpi=dpi,
    )
    _render_combined_cluster_map(
        frame,
        column="contextual_cluster",
        palette=contextual_palette,
        prefix=CONTEXTUAL_LABEL_PREFIX,
        title="SO1 contextual hL direct cosine-kNN Leiden clusters\nresolution 1.0",
        png_path=contextual_png,
        pdf_path=contextual_pdf,
        dpi=dpi,
    )
    limits = delta["plotting_limits"]
    _render_combined_delta_map(
        frame,
        vmin=float(limits["vmin_global_p01"]),
        vmax=float(limits["vmax_global_p99"]),
        png_path=delta_png,
        pdf_path=delta_pdf,
        dpi=dpi,
    )
    outputs = [
        intrinsic_png,
        intrinsic_pdf,
        contextual_png,
        contextual_pdf,
        delta_png,
        delta_pdf,
    ]
    outputs.extend(
        _render_per_core_pngs(
            frame,
            intrinsic_palette=intrinsic_palette,
            contextual_palette=contextual_palette,
            vmin=float(limits["vmin_global_p01"]),
            vmax=float(limits["vmax_global_p99"]),
            output_root=output_root,
            dpi=dpi,
        )
    )
    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "clustering_manifest_sha256": clustering_sha,
            "analysis_code_provenance": _analysis_code_provenance_record(
                output_root
            ),
            "dpi": int(dpi),
            "point_count": EXPECTED_TOTAL_CELLS,
            "combined_figure_pairs": 3,
            "per_core_png_count": 42,
            "plot_specification": spatial_plot_spec(),
            "delta_h_l2": delta,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in outputs
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_figure_manifest(
        output_root=output_root,
        receipt=receipt,
        clustering_sha256=clustering_sha,
        dpi=dpi,
    )
    del cores, frame
    gc.collect()
    return receipt


def _required_figure_files() -> tuple[str, ...]:
    files = [
        f"figures/{INTRINSIC_COMBINED_STEM}.png",
        f"figures/{INTRINSIC_COMBINED_STEM}.pdf",
        f"figures/{CONTEXTUAL_COMBINED_STEM}.png",
        f"figures/{CONTEXTUAL_COMBINED_STEM}.pdf",
        f"figures/{DELTA_COMBINED_STEM}.png",
        f"figures/{DELTA_COMBINED_STEM}.pdf",
    ]
    for core in SO1_CORE_NUMBERS:
        files.extend(
            (
                "figures/per_core/"
                f"intrinsic_direct_h0_leiden_resolution_1p0_spatial_core_{core}.png",
                "figures/per_core/"
                f"contextual_direct_hl_leiden_resolution_1p0_spatial_core_{core}.png",
                f"figures/per_core/delta_h_l2_spatial_core_{core}.png",
            )
        )
    return tuple(files)


def _required_output_files() -> set[str]:
    files = {
        "README.md",
        "provenance/analysis_code_provenance.json",
        "embeddings/extraction_manifest.json",
        "clustering/clustering_manifest.json",
        "clustering/intrinsic_labels.npy",
        "clustering/contextual_labels.npy",
        "clustering/intrinsic_knn_undirected_edges.npy",
        "clustering/contextual_knn_undirected_edges.npy",
        "clustering/intrinsic_clustering_parameters.json",
        "clustering/contextual_clustering_parameters.json",
        "clustering/intrinsic_palette.json",
        "clustering/contextual_palette.json",
        "tables/cell_embedding_clusters.parquet",
        "tables/intrinsic_cluster_summary.csv",
        "tables/contextual_cluster_summary.csv",
        "tables/intrinsic_cluster_core_composition.csv",
        "tables/contextual_cluster_core_composition.csv",
        "figures/figure_manifest.json",
        *_required_figure_files(),
    }
    files.update(
        f"embeddings/core_{core}_embeddings.npz" for core in SO1_CORE_NUMBERS
    )
    files.update(
        f"embeddings/core_{core}_receipt.json" for core in SO1_CORE_NUMBERS
    )
    return files


def runtime_code_provenance(*, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=False)

    def git_output(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
        )
        return completed.stdout if completed.returncode == 0 else b""

    package_root = Path(__file__).resolve().parent
    relevant = (
        Path(__file__),
        package_root / "models.py",
        package_root / "relative_qkv_graph_transformer.py",
        package_root / "relative_qkv_embedding_clustering.py",
        package_root / "relative_qkv_post_training.py",
    )
    status = git_output("status", "--porcelain=v1", "-z")
    tracked_diff = git_output("diff", "--binary")
    commit = git_output("rev-parse", "HEAD").decode("ascii", errors="replace").strip()
    return {
        "schema": CODE_PROVENANCE_SCHEMA,
        "recorded_at_utc": _utc_now(),
        "workflow": OUTPUT_NAMESPACE,
        "working_directory": Path.cwd().resolve(strict=False).as_posix(),
        "command": shlex.join(str(value) for value in getattr(sys, "orig_argv", sys.argv)),
        "git_commit": commit or None,
        "git_worktree_clean": not bool(status),
        "git_status_path_fingerprint_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "relevant_code": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in relevant
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used": False,
    }


def _ensure_code_provenance(*, output_root: Path, project_root: Path) -> Path:
    path = output_root / "provenance" / "analysis_code_provenance.json"
    current = runtime_code_provenance(project_root=project_root)
    if path.is_file():
        existing = _read_json(path, label="SO1 analysis code provenance")
        drifted = (
            existing.get("relevant_code") != current.get("relevant_code")
            or existing.get("command") != current.get("command")
        )
        if not (output_root / "manifest.json").is_file() and drifted:
            materialized = sorted(
                candidate.relative_to(output_root).as_posix()
                for candidate in output_root.rglob("*")
                if candidate.is_file() and candidate != path
            )
            if materialized:
                raise SO1ModelEmbeddingClusteringError(
                    "Analysis code or command changed after resumable artifacts "
                    "were materialized; refusing a mixed-code bundle: "
                    + ", ".join(materialized[:8])
                )
            _atomic_write_json(path, current)
        return path
    _atomic_write_json(path, current)
    return path


def _analysis_code_provenance_record(output_root: Path) -> dict[str, Any]:
    path = output_root / "provenance" / "analysis_code_provenance.json"
    if not path.is_file():
        raise SO1ModelEmbeddingClusteringError(
            "Analysis code provenance must be frozen before any resumable stage."
        )
    payload = _read_json(path, label="SO1 analysis code provenance")
    if payload.get("schema") != CODE_PROVENANCE_SCHEMA:
        raise SO1ModelEmbeddingClusteringError(
            "Analysis code provenance schema is invalid."
        )
    return _file_record(path)


def _source_artifacts(inputs: SO1ResolvedInputs) -> dict[str, Any]:
    records = {
        "finalized_run_bundle": inputs.provenance["finalized_run_bundle"],
        "checkpoint": {
            "path": inputs.checkpoint_path.as_posix(),
            "file": _file_record(inputs.checkpoint_path),
            "model_state_sha256": inputs.checkpoint_payload["model_state_checksum"],
        },
        "resolved_config": inputs.provenance["resolved_config"],
        "run_manifest": inputs.provenance["run_manifest"],
        "success_marker": inputs.provenance["success_marker"],
        "artifact_checksum_manifest": inputs.provenance[
            "artifact_checksum_manifest"
        ],
        "final_checkpoint_reload_verification": inputs.provenance[
            "final_checkpoint_reload_verification"
        ],
        "terminal_plateau_audit": inputs.provenance["terminal_plateau_audit"],
        "training_provenance_receipt": inputs.provenance[
            "training_provenance_receipt"
        ],
        "cohort_manifest": inputs.provenance["cohort_manifest"],
        "graph_manifest": inputs.provenance["graph_manifest"],
        "completed_cohort_manifest": inputs.provenance[
            "completed_cohort_manifest"
        ],
        "source_cores": {},
    }
    for core in _expected_core_records(inputs):
        records["source_cores"][str(core["alias"])] = dict(core)
    return records


def _render_readme(
    *,
    inputs: SO1ResolvedInputs,
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> str:
    dominated_i = ", ".join(clustering["core_dominated_gt_90pct"]["intrinsic"]) or "None"
    dominated_c = ", ".join(clustering["core_dominated_gt_90pct"]["contextual"]) or "None"
    delta = figures["delta_h_l2"]
    return f"""# SO1 14-core model-embedding direct clustering

This completed post-training report is locked to run `{inputs.run_id}` and its
immutable `last.ckpt`. No retraining, preprocessing refit, batch correction,
neighbor sampling, or GPU execution occurred.

## Representation and clustering method

`h0` is the exact all-node output of the trained NodeEncoder before graph
attention. `hL` is the exact all-node output after the fourth Relative-Geometric
QKV graph layer and immediately before the expression decoder. For each
representation independently, all {EXPECTED_TOTAL_CELLS:,} cells from SO1 cores
1 through 14 were concatenated, row-L2-normalized solely for cosine distance,
inserted into an independent sparse FAISS HNSW 30-nearest-neighbor graph using a
deterministic core-independent permutation, and clustered with seeded Leiden at
resolution 1.0. There was no PCA or mean-centering. Sampled exact recall@30 was
required to be at least {MINIMUM_MEAN_RECALL_AT_K:.2f} for both graphs.

- Intrinsic clusters: {clustering['cluster_counts']['intrinsic']}; size range {clustering['cluster_size_ranges']['intrinsic']}
- Contextual clusters: {clustering['cluster_counts']['contextual']}; size range {clustering['cluster_size_ranges']['contextual']}
- Intrinsic clusters >90% from one core: {dominated_i}
- Contextual clusters >90% from one core: {dominated_c}
- Raw delta-norm mean/median: {delta['mean']:.6g} / {delta['median']:.6g}
- Shared delta plotting limits (global p1/p99): {delta['p01']:.6g} / {delta['p99']:.6g}

## Interpretation constraints

Intrinsic clusters represent patterns in the cell's own standardized expression
and permitted metadata embedding. Contextual clusters represent patterns after
graph-based neighborhood processing by this trained model. `delta_h_l2` is only
the descriptive magnitude of representation change after contextual processing.

None of these quantities independently establishes cell type, signaling,
biological influence, or causality. Cluster names are deliberately limited to
the model-derived `S1I` and `S1C` namespaces. Marker-based and pathological
validation will be conducted separately.

This is a fit-only, transductive, single-seed exploratory readout. Core-dominated
clusters are flagged but not removed or integrated.

## Programmatic reproduction

Run only after the registry marks the exact run complete and the immutable last
checkpoint verifies:

```python
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry
from spatial_benchmark.so1_model_embedding_clustering import run_so1_model_embedding_clustering

paths = ProjectPaths.from_environment(anchor=__file__)
registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
run_so1_model_embedding_clustering(registry=registry, paths=paths)
```

Launch Python with `CUDA_VISIBLE_DEVICES=""`, `PYTHONPATH=src`, and the repository
virtual environment. Per-core extraction, joint clustering, and plotting are
separate checksum-verified resumable stages.
"""


def _verify_external_source_artifacts(source: Mapping[str, Any]) -> None:
    bundle = source.get("finalized_run_bundle")
    if not isinstance(bundle, Mapping):
        raise SO1ModelEmbeddingClusteringError(
            "Final manifest lacks finalized run-bundle provenance."
        )
    bundle_path = Path(str(bundle.get("path", "")))
    observed_bundle_verification = verify_run_bundle(bundle_path)
    if observed_bundle_verification != bundle.get("verification"):
        raise SO1ModelEmbeddingClusteringError(
            "Finalized upstream run-bundle verification changed."
        )
    for name in (
        "checkpoint",
        "resolved_config",
        "run_manifest",
        "success_marker",
        "artifact_checksum_manifest",
        "final_checkpoint_reload_verification",
        "terminal_plateau_audit",
        "training_provenance_receipt",
        "cohort_manifest",
        "graph_manifest",
        "completed_cohort_manifest",
    ):
        record = source.get(name)
        if not isinstance(record, Mapping):
            raise SO1ModelEmbeddingClusteringError(
                f"Final manifest lacks source artifact {name}."
            )
        path = Path(str(record.get("path", "")))
        expected_file = record.get("file")
        if not path.is_file() or not isinstance(expected_file, Mapping) or _file_record(
            path
        ) != dict(expected_file):
            raise SO1ModelEmbeddingClusteringError(
                f"External source artifact changed: {name}."
            )
    resolved_config_record = source["resolved_config"]
    parsed_config = _read_yaml(
        Path(str(resolved_config_record["path"])),
        label="bound SO1 resolved configuration",
    )
    if any(
        (
            resolved_config_record.get("canonical_content_sha256")
            != _canonical_sha256(parsed_config),
            resolved_config_record.get(
                "checkpoint_bound_canonical_content_sha256"
            )
            != _canonical_sha256(parsed_config),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Resolved configuration canonical-content binding changed."
        )
    checkpoint_path = Path(str(source["checkpoint"]["path"]))
    if checkpoint_path.name != "last.ckpt":
        raise SO1ModelEmbeddingClusteringError("Source checkpoint is not last.ckpt.")
    cores = source.get("source_cores")
    if not isinstance(cores, Mapping) or tuple(cores) != SO1_ALIASES:
        raise SO1ModelEmbeddingClusteringError(
            "Final source-core records are incomplete or reordered."
        )
    cohort_root = Path(str(source["cohort_manifest"]["path"])).parent
    graph_root = Path(str(source["graph_manifest"]["path"])).parent
    for alias in SO1_ALIASES:
        record = cores.get(alias)
        if not isinstance(record, Mapping):
            raise SO1ModelEmbeddingClusteringError(
                f"Final source-core record is malformed: {alias}."
            )
        prepared_path = cohort_root / "cores" / f"{alias}.npz"
        if not prepared_path.is_file() or sha256_file(prepared_path) != record.get(
            "prepared_core_artifact_sha256"
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"External prepared core artifact changed: {alias}."
            )
        graph_checksums = record.get("graph_file_checksums")
        required_graph_files = {"edge_index.npy", "relative_geometry.npy"}
        if not isinstance(graph_checksums, Mapping) or not required_graph_files.issubset(
            graph_checksums
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"External graph file records are incomplete: {alias}."
            )
        for name, expected_sha256 in graph_checksums.items():
            safe_name = Path(str(name))
            if safe_name.name != str(name) or safe_name.is_absolute():
                raise SO1ModelEmbeddingClusteringError(
                    f"External graph artifact name is unsafe: {alias}/{name}."
                )
            path = graph_root / "cores" / alias / safe_name
            if not path.is_file() or sha256_file(path) != expected_sha256:
                raise SO1ModelEmbeddingClusteringError(
                    f"External graph artifact changed: {alias}/{name}."
                )


def verify_so1_model_embedding_clustering(
    output_root: str | Path,
) -> Mapping[str, Any]:
    """Strictly verify a completed SO1 embedding-clustering bundle."""

    root = Path(output_root).expanduser().resolve(strict=True)
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 analysis bundle is missing or contains symlinks."
        )
    manifest = _read_json(root / "manifest.json", label="SO1 final analysis manifest")
    _verify_self_hash(manifest, label="SO1 final analysis manifest")
    configuration = manifest.get("configuration")
    expected_configuration = _clustering_configuration(
        n_neighbors=DEFAULT_N_NEIGHBORS,
        leiden_resolution=DEFAULT_LEIDEN_RESOLUTION,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    upstream_binding = manifest.get("upstream_binding")
    bound_campaign = (
        upstream_binding.get("campaign")
        if isinstance(upstream_binding, Mapping)
        else None
    )
    bound_catalog = (
        upstream_binding.get("checkpoint_catalog")
        if isinstance(upstream_binding, Mapping)
        else None
    )
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            manifest.get("campaign_id") != ANALYSIS_CAMPAIGN_ID,
            manifest.get("upstream_training_campaign_id")
            != UPSTREAM_TRAINING_CAMPAIGN_ID,
            not isinstance(upstream_binding, Mapping),
            isinstance(upstream_binding, Mapping)
            and upstream_binding.get("run_id") != EXPECTED_RUN_ID,
            isinstance(upstream_binding, Mapping)
            and int(upstream_binding.get("seed", -1)) != EXPECTED_MODEL_SEED,
            isinstance(upstream_binding, Mapping)
            and int(upstream_binding.get("fold", -1)) != EXPECTED_FOLD,
            isinstance(upstream_binding, Mapping)
            and int(upstream_binding.get("attempt", -1)) != 1,
            not isinstance(bound_campaign, Mapping),
            isinstance(bound_campaign, Mapping)
            and bound_campaign.get("campaign_id")
            != UPSTREAM_TRAINING_CAMPAIGN_ID,
            not isinstance(bound_catalog, Mapping),
            isinstance(bound_catalog, Mapping)
            and bound_catalog.get("role") != "last",
            isinstance(bound_catalog, Mapping)
            and bound_catalog.get("retention_class")
            != EXPECTED_CHECKPOINT_RETENTION_CLASS,
            isinstance(bound_catalog, Mapping)
            and bound_catalog.get("verification_status") != "verified",
            isinstance(bound_catalog, Mapping)
            and bound_catalog.get("lifecycle_stage") != "locked_final",
            isinstance(bound_catalog, Mapping)
            and bound_catalog.get("study_axis")
            != EXPECTED_UPSTREAM_STUDY_AXIS,
            int(manifest.get("model_seed", -1)) != EXPECTED_MODEL_SEED,
            int(manifest.get("fold", -1)) != EXPECTED_FOLD,
            tuple(manifest.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            configuration != expected_configuration,
            manifest.get("embedding_shapes", {}).get("h0")
            != [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
            manifest.get("embedding_shapes", {}).get("hL")
            != [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
            manifest.get("training_source_commit")
            != EXPECTED_TRAINING_SOURCE_COMMIT,
            manifest.get("execution", {}).get("cpu_only") is not True,
            manifest.get("execution", {}).get("gpu_used") is not False,
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "SO1 final analysis manifest identity is invalid."
        )
    files = manifest.get("files")
    observed = _file_manifest(root)
    required = _required_output_files()
    if not isinstance(files, Mapping) or dict(files) != observed or set(files) != required:
        raise SO1ModelEmbeddingClusteringError(
            "SO1 final output inventory/checksums changed."
        )
    _verify_external_source_artifacts(manifest.get("source_artifacts", {}))
    _validated_model_construction(
        {"model_construction": manifest.get("model_construction")}
    )
    source_checkpoint = manifest.get("source_artifacts", {}).get("checkpoint", {})
    if any(
        (
            manifest.get("checkpoint", {}).get("role") != "last",
            manifest.get("checkpoint", {}).get("sha256")
            != source_checkpoint.get("file", {}).get("sha256"),
            manifest.get("checkpoint", {}).get("model_state_sha256")
            != source_checkpoint.get("model_state_sha256"),
            manifest.get("checkpoint", {}).get("sha256")
            != bound_catalog.get("sha256"),
            manifest.get("checkpoint", {}).get("path")
            != bound_catalog.get("path"),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Final checkpoint provenance records do not reconcile."
        )
    provenance_path = root / "provenance" / "analysis_code_provenance.json"
    code_provenance = _read_json(provenance_path, label="SO1 analysis code provenance")
    if any(
        (
            code_provenance.get("schema") != CODE_PROVENANCE_SCHEMA,
            code_provenance.get("workflow") != OUTPUT_NAMESPACE,
            code_provenance.get("gpu_used") is not False,
            not isinstance(code_provenance.get("relevant_code"), Mapping),
            not code_provenance.get("relevant_code"),
        )
    ):
        raise SO1ModelEmbeddingClusteringError("Analysis code provenance is invalid.")

    extraction_path = root / "embeddings" / "extraction_manifest.json"
    clustering_path = root / "clustering" / "clustering_manifest.json"
    figures_path = root / "figures" / "figure_manifest.json"
    if manifest.get("stage_manifests") != {
        "extraction": _file_record(extraction_path),
        "clustering": _file_record(clustering_path),
        "figures": _file_record(figures_path),
    }:
        raise SO1ModelEmbeddingClusteringError("Stage manifest checksums changed.")
    extraction = _read_json(extraction_path, label="SO1 extraction manifest")
    _verify_self_hash(extraction, label="SO1 extraction manifest")
    code_provenance_record = _analysis_code_provenance_record(root)
    if any(
        (
            extraction.get("schema") != EXTRACTION_SCHEMA,
            extraction.get("status") != "complete",
            extraction.get("run_id") != EXPECTED_RUN_ID,
            extraction.get("analysis_code_provenance")
            != code_provenance_record,
            tuple(extraction.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(extraction.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            extraction.get("all_zero_extraction_masks") is not True,
            extraction.get("inference_schedule") != "locked_manifest_core_order",
            extraction.get("all_cell_expression_decode_performed") is not False,
            extraction.get("prediction_invariance", {}).get(
                "verified_on_all_fourteen_completed_trained_cores"
            )
            is not True,
            extraction.get("prediction_invariance", {}).get(
                "all_predictions_allclose"
            )
            is not True,
            extraction.get("device") != "cpu",
        )
    ):
        raise SO1ModelEmbeddingClusteringError("Extraction semantics changed.")
    core_records = extraction.get("cores")
    if not isinstance(core_records, list) or len(core_records) != 14:
        raise SO1ModelEmbeddingClusteringError("Extraction core records are incomplete.")
    cores: list[SO1CoreEmbeddings] = []
    for core_number, alias, record in zip(
        SO1_CORE_NUMBERS, SO1_ALIASES, core_records, strict=True
    ):
        if not isinstance(record, Mapping):
            raise SO1ModelEmbeddingClusteringError("Malformed extraction core record.")
        if record.get("analysis_code_provenance") != code_provenance_record:
            raise SO1ModelEmbeddingClusteringError(
                f"Embedding code provenance changed for core {core_number}."
            )
        path = root / str(record.get("embedding_file", ""))
        if record.get("file") != _file_record(path):
            raise SO1ModelEmbeddingClusteringError(
                f"Embedding file checksum changed for core {core_number}."
            )
        core = load_core_embeddings(
            path,
            alias=alias,
            core_number=core_number,
            expected_cells=EXPECTED_CELL_COUNTS_BY_CORE[core_number],
        )
        checksums = record.get("array_checksums")
        if not isinstance(checksums, Mapping) or any(
            checksums.get(name) != _array_sha256(name, getattr(core, name))
            for name in (
                "cell_index",
                "coordinates_um",
                "h0",
                "hL",
                "delta_h",
                "delta_h_l2",
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"Embedding array checksum changed for core {core_number}."
            )
        cores.append(core)

    clustering = _read_json(clustering_path, label="SO1 clustering manifest")
    _verify_clustering_manifest(
        output_root=root,
        receipt=clustering,
        extraction_sha256=sha256_file(extraction_path),
        configuration=expected_configuration,
    )
    if any(
        (
            manifest.get("cluster_counts") != clustering.get("cluster_counts"),
            manifest.get("cluster_size_ranges")
            != clustering.get("cluster_size_ranges"),
            manifest.get("core_dominated_gt_90pct")
            != clustering.get("core_dominated_gt_90pct"),
        )
    ):
        raise SO1ModelEmbeddingClusteringError(
            "Final and clustering summary records do not reconcile."
        )
    for representation in REPRESENTATIONS:
        pipeline = clustering.get("pipelines", {}).get(representation)
        if not isinstance(pipeline, Mapping) or any(
            (
                pipeline.get("pca") is not False,
                pipeline.get("mean_center") is not False,
                pipeline.get("l2_normalize_for_cosine") is not True,
                pipeline.get("knn", {}).get(
                    "core_labels_consulted_for_insertion_order"
                )
                is not False,
                pipeline.get("knn", {}).get("cell_by_cell_matrix_constructed")
                is not False,
                pipeline.get("knn_exact_recall_audit", {}).get(
                    "mean_recall_acceptance_passed"
                )
                is not True,
                float(
                    pipeline.get("knn_exact_recall_audit", {}).get(
                        "recall_at_k_mean", -1.0
                    )
                )
                < MINIMUM_MEAN_RECALL_AT_K,
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                f"{representation} direct clustering semantics changed."
            )

    figures = _read_json(figures_path, label="SO1 figure manifest")
    _verify_figure_manifest(
        output_root=root,
        receipt=figures,
        clustering_sha256=sha256_file(clustering_path),
        dpi=int(manifest["figure_dpi"]),
    )
    expected_combined = {
        "intrinsic_png": f"figures/{INTRINSIC_COMBINED_STEM}.png",
        "intrinsic_pdf": f"figures/{INTRINSIC_COMBINED_STEM}.pdf",
        "contextual_png": f"figures/{CONTEXTUAL_COMBINED_STEM}.png",
        "contextual_pdf": f"figures/{CONTEXTUAL_COMBINED_STEM}.pdf",
        "delta_png": f"figures/{DELTA_COMBINED_STEM}.png",
        "delta_pdf": f"figures/{DELTA_COMBINED_STEM}.pdf",
    }
    if manifest.get("combined_figures") != expected_combined:
        raise SO1ModelEmbeddingClusteringError(
            "Final combined-figure paths changed."
        )
    observed_delta = delta_norm_statistics(cores)
    if figures.get("delta_h_l2") != observed_delta or manifest.get(
        "delta_h_l2"
    ) != observed_delta:
        raise SO1ModelEmbeddingClusteringError("Delta-norm statistics changed.")
    table = pd.read_parquet(root / "tables" / "cell_embedding_clusters.parquet")
    raw_delta = np.concatenate([core.delta_h_l2 for core in cores])
    if not np.array_equal(table["delta_h_l2"].to_numpy(dtype=np.float64), raw_delta):
        raise SO1ModelEmbeddingClusteringError(
            "Cell table delta norms are misaligned."
        )
    interpretation = manifest.get("interpretation")
    if not isinstance(interpretation, Mapping) or any(
        interpretation.get(name) is not False
        for name in (
            "establishes_cell_type",
            "establishes_signaling",
            "establishes_biological_influence",
            "establishes_causality",
        )
    ) or interpretation.get("marker_and_pathology_validation_separate") is not True:
        raise SO1ModelEmbeddingClusteringError(
            "Final interpretation constraints changed."
        )
    return manifest


verify_so1_model_embedding_clustering_bundle = (
    verify_so1_model_embedding_clustering
)


def _finalize_analysis(
    *,
    inputs: SO1ResolvedInputs,
    output_root: Path,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
    configuration: Mapping[str, Any],
    figure_dpi: int,
) -> Mapping[str, Any]:
    _atomic_write_text(
        output_root / "README.md",
        _render_readme(inputs=inputs, clustering=clustering, figures=figures),
    )
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_scope": "SO1_h0_hL_delta_direct_knn_static_maps",
            "exploratory": True,
            "run_id": inputs.run_id,
            "campaign_id": ANALYSIS_CAMPAIGN_ID,
            "upstream_training_campaign_id": UPSTREAM_TRAINING_CAMPAIGN_ID,
            "upstream_binding": {
                "campaign": inputs.provenance["upstream_campaign_registry"],
                "run_id": inputs.run_id,
                "preferred_alias": inputs.provenance["preferred_alias"],
                "scientific_id": inputs.provenance["scientific_id"],
                "repro_id": inputs.provenance["repro_id"],
                "seed": EXPECTED_MODEL_SEED,
                "fold": EXPECTED_FOLD,
                "attempt": inputs.provenance["attempt"],
                "checkpoint_catalog": inputs.provenance["checkpoint_catalog"],
            },
            "model_seed": EXPECTED_MODEL_SEED,
            "fold": EXPECTED_FOLD,
            "checkpoint": inputs.provenance["checkpoint"],
            "model_construction": inputs.provenance["model_construction"],
            "training_source_commit": EXPECTED_TRAINING_SOURCE_COMMIT,
            "configuration": dict(configuration),
            "figure_dpi": int(figure_dpi),
            "core_order": list(SO1_CORE_NUMBERS),
            "core_cell_counts": {
                str(core): EXPECTED_CELL_COUNTS_BY_CORE[core]
                for core in SO1_CORE_NUMBERS
            },
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_shapes": {
                "h0": [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
                "hL": [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
            },
            "source_artifacts": _source_artifacts(inputs),
            "stage_manifests": {
                "extraction": _file_record(
                    output_root / "embeddings" / "extraction_manifest.json"
                ),
                "clustering": _file_record(
                    output_root / "clustering" / "clustering_manifest.json"
                ),
                "figures": _file_record(
                    output_root / "figures" / "figure_manifest.json"
                ),
            },
            "cluster_counts": clustering["cluster_counts"],
            "cluster_size_ranges": clustering["cluster_size_ranges"],
            "core_dominated_gt_90pct": clustering["core_dominated_gt_90pct"],
            "delta_h_l2": figures["delta_h_l2"],
            "combined_figures": {
                "intrinsic_png": f"figures/{INTRINSIC_COMBINED_STEM}.png",
                "intrinsic_pdf": f"figures/{INTRINSIC_COMBINED_STEM}.pdf",
                "contextual_png": f"figures/{CONTEXTUAL_COMBINED_STEM}.png",
                "contextual_pdf": f"figures/{CONTEXTUAL_COMBINED_STEM}.pdf",
                "delta_png": f"figures/{DELTA_COMBINED_STEM}.png",
                "delta_pdf": f"figures/{DELTA_COMBINED_STEM}.pdf",
            },
            "execution": {
                "cpu_only": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_used": False,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "faiss_backend": "faiss-cpu",
            },
            "interpretation": {
                "intrinsic_is_cell_own_expression_and_metadata_embedding": True,
                "contextual_is_after_graph_neighborhood_processing": True,
                "delta_is_descriptive_representation_change_magnitude": True,
                "establishes_cell_type": False,
                "establishes_signaling": False,
                "establishes_biological_influence": False,
                "establishes_causality": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": _file_manifest(output_root),
        }
    )
    _atomic_write_json(output_root / "manifest.json", manifest)
    verify_so1_model_embedding_clustering(output_root)
    return manifest


def _resolve_output_root(
    *, paths: ProjectPaths, run_id: str, output_dir: str | Path | None
) -> Path:
    namespace = (
        paths.report_root / "analyses" / OUTPUT_NAMESPACE
    ).resolve(strict=False)
    candidate = namespace / run_id if output_dir is None else Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = paths.project_root / candidate
    if candidate.is_symlink():
        raise SO1ModelEmbeddingClusteringError("Analysis output may not be a symlink.")
    root = candidate.resolve(strict=False)
    if root == namespace or not root.is_relative_to(namespace):
        raise SO1ModelEmbeddingClusteringError(
            f"Output must be a run-specific directory beneath {namespace}."
        )
    if root.exists() and not root.is_dir():
        raise SO1ModelEmbeddingClusteringError("Analysis output must be a directory.")
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise SO1ModelEmbeddingClusteringError("Output may not contain symlinks.")
    return root


def run_so1_model_embedding_clustering(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the locked, resumable, CPU-only static SO1 analysis."""

    validate_cpu_device(device)
    validate_cuda_hidden()
    configuration = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    inputs = resolve_so1_analysis_inputs(
        registry=registry, paths=paths, run_id=run_id, checkpoint=checkpoint
    )
    output_root = _resolve_output_root(
        paths=paths,
        run_id=inputs.run_id,
        output_dir=output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    allowed_top_level = {
        "embeddings",
        "clustering",
        "tables",
        "figures",
        "provenance",
        "README.md",
        "manifest.json",
    }
    unexpected = sorted(
        path.name for path in output_root.iterdir() if path.name not in allowed_top_level
    )
    if unexpected:
        raise SO1ModelEmbeddingClusteringError(
            "Analysis output contains unrecognized entries: " + ", ".join(unexpected)
        )
    _ensure_code_provenance(
        output_root=output_root, project_root=paths.project_root
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = verify_so1_model_embedding_clustering(output_root)
        if any(
            (
                manifest.get("configuration") != configuration,
                int(manifest.get("figure_dpi", -1)) != int(dpi),
                manifest.get("checkpoint", {}).get("sha256")
                != inputs.checkpoint_sha256,
            )
        ):
            raise SO1ModelEmbeddingClusteringError(
                "Completed output differs from requested checkpoint or parameters."
            )
    else:
        extraction = extract_intermediate_embeddings(
            inputs=inputs,
            output_root=output_root,
            device=device,
            cpu_threads=cpu_threads,
        )
        clustering = build_clustering_outputs(
            inputs=inputs,
            output_root=output_root,
            extraction_receipt=extraction,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            random_seed=random_seed,
        )
        figures = render_spatial_outputs(
            output_root=output_root,
            extraction_receipt=extraction,
            clustering_receipt=clustering,
            dpi=dpi,
        )
        manifest = _finalize_analysis(
            inputs=inputs,
            output_root=output_root,
            extraction=extraction,
            clustering=clustering,
            figures=figures,
            configuration=configuration,
            figure_dpi=dpi,
        )
    return {
        "status": "complete",
        "run_id": inputs.run_id,
        "device": "cpu",
        "output_root": output_root.as_posix(),
        "total_cells": EXPECTED_TOTAL_CELLS,
        "h0_shape": [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
        "hL_shape": [EXPECTED_TOTAL_CELLS, EXPECTED_HIDDEN_DIM],
        "cluster_counts": manifest["cluster_counts"],
        "combined_intrinsic_png": (
            output_root / "figures" / f"{INTRINSIC_COMBINED_STEM}.png"
        ).as_posix(),
        "combined_contextual_png": (
            output_root / "figures" / f"{CONTEXTUAL_COMBINED_STEM}.png"
        ).as_posix(),
        "combined_delta_png": (
            output_root / "figures" / f"{DELTA_COMBINED_STEM}.png"
        ).as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "CONTEXTUAL_LABEL_PREFIX",
    "DIRECT_PIPELINE_KIND",
    "EXPECTED_CELL_COUNTS_BY_CORE",
    "EXPECTED_RUN_ID",
    "EXPECTED_TOTAL_CELLS",
    "INTRINSIC_LABEL_PREFIX",
    "SO1ModelEmbeddingClusteringError",
    "SO1CoreEmbeddings",
    "SO1ResolvedInputs",
    "build_clustering_outputs",
    "cluster_direct_representation",
    "delta_norm_statistics",
    "exact_neighbor_recall_audit",
    "extract_full_h0_hl",
    "extract_intermediate_embeddings",
    "requested_panel_order",
    "resolve_so1_analysis_inputs",
    "run_so1_model_embedding_clustering",
    "spatial_plot_spec",
    "validate_cpu_device",
    "validate_cuda_hidden",
    "verify_so1_model_embedding_clustering",
    "verify_so1_model_embedding_clustering_bundle",
]

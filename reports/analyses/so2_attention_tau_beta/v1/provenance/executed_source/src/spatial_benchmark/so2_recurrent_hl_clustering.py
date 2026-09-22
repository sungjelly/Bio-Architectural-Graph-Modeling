"""PNG-only joint Leiden map of recurrent SO2 contextual embeddings.

This is a narrowly scoped post-hoc reader for the training-complete recurrent
checkpoint whose run finalization failed after training. It never changes the
source bundle or registry record, and it retains that lifecycle limitation in
every report receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hmac
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import torch

from .fingerprints import sha256_file
from .paths import ProjectPaths
from .pooled_relative_qkv_training_v2 import CohortRelativeQKVCoreBatch
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    KNNGraphResult,
    _array_sha256,
    _atomic_write_json,
    _atomic_write_text,
    _file_manifest,
    _file_record,
    _git_provenance,
    _project_artifact_path,
    _read_json,
    _read_yaml,
    _receipt_with_self_hash,
    _style_spatial_axis,
    _tensor_sha256,
    _verify_self_hash,
    _write_deterministic_npz,
    run_seeded_leiden,
)
from .relative_qkv_graph_transformer import (
    ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer,
)
from .relative_qkv_post_training import _tree_sha256
from .run_archive import RunArchive, verify_run_bundle
from .so2_hl_clustering import (
    SO2ContextualCore,
    SO2ResolvedInputs,
    _clustering_configuration,
    _expected_core_records,
    cluster_joint_contextual_embeddings,
    load_contextual_core,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
)
from .so2_relative_graphs import (
    _verify_so2_cohort_manifest,
    _verify_so2_graph_collection,
)
from .so2_training_observability import CHECKPOINT_SCHEMA


ANALYSIS_SCHEMA = "so2_14core_recurrent_contextual_embedding_clustering_v1"
EXTRACTION_SCHEMA = "so2_14core_recurrent_hl_extraction_v1"
CORE_EXTRACTION_SCHEMA = "so2_14core_recurrent_hl_core_extraction_v1"
FIGURE_SCHEMA = "so2_14core_recurrent_hl_spatial_figure_v1"
DETERMINISM_SCHEMA = "so2_14core_recurrent_hl_leiden_determinism_v1"
CAMPAIGN_ID = "cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2"
EXPECTED_RUN_ID = "r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf"
EXPECTED_CHECKPOINT_SHA256 = (
    "01f3611897557add92b9733bff7e3940d6b19d22881bd8401340e255c9e355e2"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "a9446ea7aa158c21b4c12d69ad82f98bb7b1c934e106ed300aaa23029846386d"
)
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_EXTRACT_DEVICES = "cuda:0,cuda:1,cuda:2,cuda:3"
DEFAULT_CPU_THREADS_PER_WORKER = 4
DEFAULT_DPI = 300
PILOT_CORE_NUMBER = 21
WORST_CASE_CORE_NUMBER = 23
MINIMUM_CUDA_FREE_MARGIN_BYTES = 2 * 1024**3
HL_REPEAT_RTOL = 1.0e-6
HL_REPEAT_ATOL = 1.0e-5


class SO2RecurrentHLClusteringError(RuntimeError):
    """Raised when the locked recurrent hL analysis contract is violated."""


@dataclass(frozen=True, slots=True)
class ExtractionAssignment:
    device: str
    core_records: tuple[Mapping[str, Any], ...]
    repeat_core_numbers: tuple[int, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_extract_devices(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return a validated, unique list of explicit CUDA devices."""

    raw = value.split(",") if isinstance(value, str) else list(value)
    devices = tuple(str(item).strip() for item in raw if str(item).strip())
    if not devices or len(set(devices)) != len(devices):
        raise SO2RecurrentHLClusteringError(
            "Extraction devices must be a non-empty unique list."
        )
    for device in devices:
        resolved = torch.device(device)
        if resolved.type != "cuda" or resolved.index is None:
            raise SO2RecurrentHLClusteringError(
                "Recurrent hL extraction requires explicit CUDA device indices."
            )
    return devices


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO2RecurrentHLClusteringError(
            f"Cannot load recurrent checkpoint: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SO2RecurrentHLClusteringError("Checkpoint payload must be a mapping.")
    return dict(payload)


def _manifest_content_is_valid(manifest: Mapping[str, Any]) -> bool:
    from .relative_qkv_embedding_clustering import _canonical_sha256

    content = dict(manifest)
    observed = str(content.pop("manifest_content_sha256", ""))
    return bool(observed) and hmac.compare_digest(observed, _canonical_sha256(content))


def resolve_recurrent_analysis_inputs(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None,
    checkpoint: str | Path | None,
) -> SO2ResolvedInputs:
    """Resolve and verify the explicit training-complete failed-run exception."""

    requested = EXPECTED_RUN_ID if run_id is None else str(run_id)
    canonical = registry.resolve_run_id(requested)
    if canonical != EXPECTED_RUN_ID:
        raise SO2RecurrentHLClusteringError(
            "This analysis is locked to the recurrent SO2 seed-0 run."
        )
    run_record = registry.show_run(canonical)
    if run_record is None or str(run_record.get("status")) != "failed":
        raise SO2RecurrentHLClusteringError(
            "Expected the documented post-training finalization-failed lifecycle."
        )
    bundle_path = RunArchive.artifact_path_for(canonical, paths)
    bundle_verification = verify_run_bundle(bundle_path)
    if not bundle_verification.get("valid") or bundle_verification.get("status") != "failed":
        raise SO2RecurrentHLClusteringError("The recurrent failed-run bundle is invalid.")

    canonical_checkpoint = (bundle_path / "checkpoints" / "last.ckpt").resolve(
        strict=True
    )
    checkpoint_path = canonical_checkpoint
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint_path)
    if not hmac.compare_digest(checkpoint_sha, EXPECTED_CHECKPOINT_SHA256):
        raise SO2RecurrentHLClusteringError(
            "Explicit checkpoint differs from the verified recurrent checkpoint."
        )
    if not hmac.compare_digest(
        sha256_file(canonical_checkpoint), EXPECTED_CHECKPOINT_SHA256
    ):
        raise SO2RecurrentHLClusteringError("Canonical recurrent checkpoint drifted.")

    payload = _load_checkpoint_payload(checkpoint_path)
    construction = payload.get("model_construction")
    if not isinstance(construction, Mapping):
        raise SO2RecurrentHLClusteringError(
            "Checkpoint lacks model construction metadata."
        )
    identity_checks = (
        payload.get("checkpoint_schema") == CHECKPOINT_SCHEMA,
        payload.get("campaign_id") == CAMPAIGN_ID,
        payload.get("run_id") == canonical,
        int(payload.get("model_seed", -1)) == 0,
        int(payload.get("completed_global_epochs", -1)) == 175,
        payload.get("model_state_checksum") == EXPECTED_MODEL_STATE_SHA256,
        construction.get("class")
        == "ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer",
        construction.get("family")
        == "recurrent_relative_geometry_qkv_graph_transformer",
        int(construction.get("graph_layers", -1)) == 1,
        int(construction.get("unique_graph_blocks", -1)) == 1,
        int(construction.get("recurrent_unroll_steps", -1)) == 4,
        int(construction.get("effective_graph_depth", -1)) == 4,
        construction.get("graph_block_weight_tying") == "all_steps",
    )
    if not all(identity_checks):
        raise SO2RecurrentHLClusteringError(
            "Checkpoint identity or recurrent architecture changed."
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or _tree_sha256(state) != EXPECTED_MODEL_STATE_SHA256:
        raise SO2RecurrentHLClusteringError("Checkpoint model-state checksum is invalid.")

    replay_path = bundle_path / "diagnostics" / "final_checkpoint_reload_verification.json"
    replay = _read_json(replay_path, label="recurrent checkpoint replay receipt")
    fixed_replay = replay.get("fixed_prediction_replay")
    replay_payload = replay.get("payload")
    if not isinstance(fixed_replay, Mapping) or not isinstance(replay_payload, Mapping):
        raise SO2RecurrentHLClusteringError("Checkpoint replay receipt is incomplete.")
    if any(
        (
            replay.get("verified") is not True,
            replay.get("checkpoint_file_sha256") != checkpoint_sha,
            fixed_replay.get("verified") is not True,
            float(fixed_replay.get("maximum_absolute_difference", math.inf)) != 0.0,
            float(fixed_replay.get("mean_absolute_difference", math.inf)) != 0.0,
            replay_payload.get("full_resume_payload_validated") is not True,
            replay_payload.get("plateau_payload_validated") is not True,
            replay_payload.get("model_state_checksum") != EXPECTED_MODEL_STATE_SHA256,
        )
    ):
        raise SO2RecurrentHLClusteringError(
            "Independent checkpoint reload/replay verification did not pass."
        )

    config = _read_yaml(
        bundle_path / "config.resolved.yaml", label="resolved recurrent configuration"
    )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SO2RecurrentHLClusteringError("Resolved run lacks a dataset section.")
    cohort_dir = _project_artifact_path(dataset.get("prepared_artifact"), paths=paths)
    graph_dir = _project_artifact_path(
        dataset.get("prepared_graph_artifact"), paths=paths
    )
    cohort_manifest_path = cohort_dir / "manifest.json"
    graph_manifest_path = graph_dir / "manifest.json"
    cohort_file_sha = sha256_file(cohort_manifest_path)
    graph_file_sha = sha256_file(graph_manifest_path)
    if cohort_file_sha != dataset.get("cohort_manifest_file_sha256"):
        raise SO2RecurrentHLClusteringError("Prepared cohort manifest changed.")
    if graph_file_sha != dataset.get("graph_manifest_file_sha256"):
        raise SO2RecurrentHLClusteringError("Prepared graph manifest changed.")
    cohort_manifest = _read_json(cohort_manifest_path, label="SO2 cohort manifest")
    graph_manifest = _read_json(graph_manifest_path, label="SO2 graph manifest")
    if not _manifest_content_is_valid(cohort_manifest):
        raise SO2RecurrentHLClusteringError("SO2 cohort manifest self-checksum failed.")
    if not _manifest_content_is_valid(graph_manifest):
        raise SO2RecurrentHLClusteringError("SO2 graph manifest self-checksum failed.")
    if tuple(cohort_manifest.get("cohort", {}).get("aliases", ())) != SO2_ALIASES:
        raise SO2RecurrentHLClusteringError("SO2 cohort order changed.")
    if tuple(graph_manifest.get("aliases", ())) != SO2_ALIASES:
        raise SO2RecurrentHLClusteringError("SO2 graph order changed.")

    # These perform the expensive full file-level checksum verification once,
    # before any parallel worker trusts the prepared immutable inputs.
    verified_cohort = _verify_so2_cohort_manifest(cohort_dir)
    verified_graph = _verify_so2_graph_collection(
        graph_dir, cohort_manifest_path=cohort_manifest_path
    )
    if verified_cohort != cohort_manifest or verified_graph != graph_manifest:
        raise SO2RecurrentHLClusteringError("Verified source manifests changed in memory.")

    provenance = {
        "selection_policy": (
            "explicit_training_complete_checkpoint_with_post_training_"
            "artifact_finalization_failure"
        ),
        "source_run_registry_status": "failed",
        "source_run_training_completed": True,
        "source_run_canonical_completion_claimed": False,
        "finalization_failure_stage": "artifact_finalization_failure",
        "campaign_id": CAMPAIGN_ID,
        "run_id": canonical,
        "model_seed": 0,
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "schema": payload["checkpoint_schema"],
            "completed_global_epochs": int(payload["completed_global_epochs"]),
            "model_state_sha256": payload["model_state_checksum"],
            "independent_reload_replay_verified": True,
            "maximum_prediction_replay_difference": 0.0,
        },
        "bundle_verification": dict(bundle_verification),
        "checkpoint_replay_receipt": _file_record(replay_path),
        "training_source": _read_json(
            bundle_path / "provenance" / "git.json", label="training Git provenance"
        ),
        "analysis_source": _git_provenance(paths.project_root),
        "model_construction": dict(construction),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "preprocessing_version": dataset.get("preprocessing_version"),
        "cohort_manifest": {
            "path": cohort_manifest_path.as_posix(),
            "file_sha256": cohort_file_sha,
            "content_sha256": cohort_manifest.get("manifest_content_sha256"),
        },
        "graph_manifest": {
            "path": graph_manifest_path.as_posix(),
            "file_sha256": graph_file_sha,
            "content_sha256": graph_manifest.get("manifest_content_sha256"),
        },
    }
    return SO2ResolvedInputs(
        run_id=canonical,
        project_root=paths.project_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_payload=payload,
        bundle_path=bundle_path,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
        cohort_manifest=cohort_manifest,
        graph_manifest=graph_manifest,
        provenance=provenance,
    )


def _integer(construction: Mapping[str, Any], name: str) -> int:
    try:
        return int(construction[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO2RecurrentHLClusteringError(
            f"Invalid recurrent model construction field: {name}."
        ) from exc


def load_recurrent_checkpoint_model(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    device: str | torch.device,
) -> ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer:
    """Strictly reconstruct the recurrent model and move it to one device."""

    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise SO2RecurrentHLClusteringError("Worker checkpoint checksum changed.")
    payload = _load_checkpoint_payload(checkpoint_path)
    construction = payload.get("model_construction")
    state = payload.get("model_state_dict")
    if not isinstance(construction, Mapping) or not isinstance(state, Mapping):
        raise SO2RecurrentHLClusteringError("Checkpoint model payload is incomplete.")
    if _tree_sha256(state) != EXPECTED_MODEL_STATE_SHA256:
        raise SO2RecurrentHLClusteringError("Worker model-state checksum failed.")
    model = ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer(
        num_genes=_integer(construction, "num_genes"),
        node_covariate_dim=_integer(construction, "node_covariate_dim"),
        hidden_dim=_integer(construction, "hidden_dim"),
        attention_heads=_integer(construction, "attention_heads"),
        attention_head_dim=_integer(construction, "attention_head_dim"),
        ffn_dim=_integer(construction, "ffn_dim"),
        decoder_dim=_integer(construction, "decoder_dim"),
        positional_bias_hidden_dim=_integer(
            construction, "positional_bias_hidden_dim"
        ),
        dropout=float(construction["dropout"]),
        attention_dropout=float(construction["attention_dropout"]),
        relative_geometry_dim=_integer(construction, "relative_geometry_dim"),
        recurrent_unroll_steps=_integer(construction, "recurrent_unroll_steps"),
        receiver_chunk_size=_integer(construction, "receiver_chunk_size"),
        max_edges_per_chunk=_integer(construction, "max_edges_per_chunk"),
        activation_checkpointing=bool(construction["activation_checkpointing"]),
    )
    model.load_state_dict(state, strict=True)
    if _tree_sha256(model.state_dict()) != EXPECTED_MODEL_STATE_SHA256:
        raise SO2RecurrentHLClusteringError("Strictly loaded model state changed.")
    if any(
        (
            model.graph_layers != 1,
            model.unique_graph_blocks != 1,
            model.recurrent_unroll_steps != 4,
            model.effective_graph_depth != 4,
            model.graph_block_weight_tying != "all_steps",
            len(model.blocks) != 1,
        )
    ):
        raise SO2RecurrentHLClusteringError("Loaded recurrent architecture is invalid.")
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise SO2RecurrentHLClusteringError("model.eval() did not disable training mode.")
    if float(model.blocks[0].attention_dropout_probability) != 0.0:
        raise SO2RecurrentHLClusteringError("Attention dropout is not disabled.")
    return model


def extract_full_recurrent_contextual_embedding(
    model: ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer,
    input_expression: torch.Tensor,
    gene_mask: torch.Tensor,
    edge_index: torch.Tensor,
    relative_geometry: torch.Tensor,
    node_covariates: torch.Tensor,
) -> torch.Tensor:
    """Return all-node recurrent hL while decoding zero target rows."""

    if model.training or torch.is_grad_enabled():
        raise SO2RecurrentHLClusteringError(
            "Contextual extraction requires eval and inference mode."
        )
    model_device = next(model.parameters()).device
    if input_expression.device != model_device or node_covariates.device != model_device:
        raise SO2RecurrentHLClusteringError("Node tensors are not on the model device.")
    if gene_mask.dtype is not torch.bool or gene_mask.shape != input_expression.shape:
        raise SO2RecurrentHLClusteringError("Extraction mask shape or dtype is invalid.")
    if bool(torch.any(gene_mask).item()):
        raise SO2RecurrentHLClusteringError("Extraction requires an all-zero mask.")
    if edge_index.device.type != "cpu" or relative_geometry.device.type != "cpu":
        raise SO2RecurrentHLClusteringError(
            "Full graph and relative geometry must remain CPU-resident."
        )
    empty_targets = torch.empty((0,), dtype=torch.long, device=model_device)
    output = model(
        input_expression=input_expression,
        gene_mask=gene_mask,
        edge_index=edge_index,
        relative_geometry=relative_geometry,
        node_covariates=node_covariates,
        target_nodes=empty_targets,
    )
    hL = output.full_node_embedding
    if output.prediction.shape != (0, model.num_genes):
        raise SO2RecurrentHLClusteringError("Decoder target selection was not empty.")
    if hL.shape != (len(input_expression), model.hidden_dim):
        raise SO2RecurrentHLClusteringError("Extracted recurrent hL shape is invalid.")
    if not bool(torch.isfinite(hL).all().item()):
        raise SO2RecurrentHLClusteringError("Extracted recurrent hL is non-finite.")
    return hL


def _source_core_records(inputs: SO2ResolvedInputs) -> tuple[dict[str, Any], ...]:
    base = _expected_core_records(inputs)
    graph_records = {
        str(record["alias"]): record for record in inputs.graph_manifest["cores"]
    }
    result = []
    for record in base:
        graph = graph_records[str(record["alias"])]
        qc = graph["graph"]["qc"]
        enriched = dict(record)
        enriched["directed_edges"] = int(qc["n_directed_edges"])
        enriched["receiver_major_canonical_order"] = bool(
            qc["receiver_major_canonical_order"]
        )
        result.append(enriched)
    return tuple(result)


def _load_core_batch(
    *,
    cohort_dir: Path,
    graph_dir: Path,
    record: Mapping[str, Any],
) -> tuple[CohortRelativeQKVCoreBatch, np.ndarray]:
    alias = str(record["alias"])
    n_cells = int(record["cell_count"])
    n_edges = int(record["directed_edges"])
    with np.load(cohort_dir / "cores" / f"{alias}.npz", allow_pickle=False) as data:
        target = np.array(data["target_expression"], dtype=np.float32, copy=True)
        covariates = np.array(data["node_covariates"], dtype=np.float32, copy=True)
        coordinates = np.array(data["coordinates_um"], dtype=np.float64, copy=True)
    edge_map = np.load(
        graph_dir / "cores" / alias / "edge_index.npy", mmap_mode="r"
    )
    geometry_map = np.load(
        graph_dir / "cores" / alias / "relative_geometry.npy", mmap_mode="r"
    )
    if target.shape != (n_cells, EXPECTED_N_GENES):
        raise SO2RecurrentHLClusteringError(f"Expression shape changed for {alias}.")
    if covariates.shape != (n_cells, 22) or coordinates.shape != (n_cells, 2):
        raise SO2RecurrentHLClusteringError(f"Metadata shape changed for {alias}.")
    if edge_map.shape != (2, n_edges) or geometry_map.shape != (n_edges, 70):
        raise SO2RecurrentHLClusteringError(f"Graph shape changed for {alias}.")
    if not bool(record.get("receiver_major_canonical_order")):
        raise SO2RecurrentHLClusteringError(f"Graph order is not canonical for {alias}.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable", category=UserWarning
        )
        edge_index = torch.from_numpy(edge_map)
        relative_geometry = torch.from_numpy(geometry_map)
    batch = CohortRelativeQKVCoreBatch(
        alias=alias,
        target_expression=torch.from_numpy(target),
        edge_index=edge_index,
        relative_geometry=relative_geometry,
        node_covariates=torch.from_numpy(covariates),
    )
    return batch, coordinates


def _core_embedding_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{core_number}_hL.npz"


def _core_receipt_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{core_number}_receipt.json"


def _verify_core_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    run_id: str,
    checkpoint_sha256: str,
) -> SO2ContextualCore:
    _verify_self_hash(receipt, label="recurrent SO2 core extraction receipt")
    if any(
        (
            receipt.get("schema") != CORE_EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != run_id,
            receipt.get("checkpoint_sha256") != checkpoint_sha256,
            receipt.get("alias") != record["alias"],
            int(receipt.get("core_number", -1)) != int(record["core_number"]),
            int(receipt.get("cell_count", -1)) != int(record["cell_count"]),
            receipt.get("source_core") != dict(record),
        )
    ):
        raise SO2RecurrentHLClusteringError("Core extraction identity is invalid.")
    path = output_root / str(receipt.get("embedding_file", ""))
    if not path.is_file() or _file_record(path) != receipt.get("file"):
        raise SO2RecurrentHLClusteringError("Core hL file checksum changed.")
    core = load_contextual_core(
        path,
        alias=str(record["alias"]),
        core_number=int(record["core_number"]),
        expected_cells=int(record["cell_count"]),
    )
    if list(core.hL.shape) != list(receipt.get("hL_shape", ())):
        raise SO2RecurrentHLClusteringError("Core hL shape receipt changed.")
    if _array_sha256("hL", core.hL) != receipt.get("hL_array_sha256"):
        raise SO2RecurrentHLClusteringError("Core hL content checksum changed.")
    inference = receipt.get("inference")
    if not isinstance(inference, Mapping) or any(
        (
            inference.get("model_eval") is not True,
            inference.get("torch_inference_mode") is not True,
            int(inference.get("gene_mask_nonzero_count", -1)) != 0,
            int(inference.get("unique_graph_blocks", -1)) != 1,
            int(inference.get("recurrent_unroll_steps", -1)) != 4,
            int(inference.get("effective_graph_depth", -1)) != 4,
            inference.get("graph_block_weight_tying") != "all_steps",
            inference.get("edge_index_device_before") != "cpu",
            inference.get("edge_index_device_after") != "cpu",
            inference.get("relative_geometry_device_before") != "cpu",
            inference.get("relative_geometry_device_after") != "cpu",
        )
    ):
        raise SO2RecurrentHLClusteringError("Core inference invariants are invalid.")
    return core


def _extract_assignment(assignment: Mapping[str, Any]) -> list[dict[str, Any]]:
    device_name = str(assignment["device"])
    device = torch.device(device_name)
    torch.set_num_threads(int(assignment["cpu_threads"]))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device)
    model = load_recurrent_checkpoint_model(
        Path(str(assignment["checkpoint_path"])),
        expected_checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        device=device,
    )
    output_root = Path(str(assignment["output_root"]))
    repeat_numbers = {int(value) for value in assignment["repeat_core_numbers"]}
    receipts: list[dict[str, Any]] = []
    for record_value in assignment["core_records"]:
        record = dict(record_value)
        alias = str(record["alias"])
        core_number = int(record["core_number"])
        batch, coordinates = _load_core_batch(
            cohort_dir=Path(str(assignment["cohort_dir"])),
            graph_dir=Path(str(assignment["graph_dir"])),
            record=record,
        )
        expression_sha = _tensor_sha256("target_expression", batch.target_expression)
        metadata_sha = _tensor_sha256("node_covariates", batch.node_covariates)
        edge_device_before = batch.edge_index.device.type
        geometry_device_before = batch.relative_geometry.device.type
        expression = batch.target_expression.to(device=device, dtype=torch.float32)
        covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
        gene_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.inference_mode():
            hL_tensor = extract_full_recurrent_contextual_embedding(
                model,
                expression,
                gene_mask,
                batch.edge_index,
                batch.relative_geometry,
                covariates,
            )
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        hL = np.ascontiguousarray(hL_tensor.float().cpu().numpy(), dtype=np.float32)
        repeat = {
            "performed": False,
            "allclose": None,
            "exact_array_match": None,
            "maximum_absolute_difference": None,
            "repeated_hL_array_sha256": None,
        }
        if core_number in repeat_numbers:
            with torch.inference_mode():
                repeated_tensor = extract_full_recurrent_contextual_embedding(
                    model,
                    expression,
                    gene_mask,
                    batch.edge_index,
                    batch.relative_geometry,
                    covariates,
                )
            torch.cuda.synchronize(device)
            repeated = np.ascontiguousarray(
                repeated_tensor.float().cpu().numpy(), dtype=np.float32
            )
            exact = bool(np.array_equal(hL, repeated))
            maximum = float(np.max(np.abs(hL - repeated), initial=0.0))
            close = bool(
                np.allclose(
                    hL,
                    repeated,
                    rtol=HL_REPEAT_RTOL,
                    atol=HL_REPEAT_ATOL,
                )
            )
            repeat = {
                "performed": True,
                "allclose": close,
                "exact_array_match": exact,
                "maximum_absolute_difference": maximum,
                "relative_tolerance": HL_REPEAT_RTOL,
                "absolute_tolerance": HL_REPEAT_ATOL,
                "repeated_hL_array_sha256": _array_sha256("hL", repeated),
                "note": (
                    "GPU receiver aggregation may vary at floating-point "
                    "roundoff scale across repeated forwards."
                ),
            }
            if not close:
                raise SO2RecurrentHLClusteringError(
                    "Repeated GPU hL extraction exceeded tolerance for "
                    f"{alias}: max_abs={maximum:.9g}, rtol={HL_REPEAT_RTOL:g}, "
                    f"atol={HL_REPEAT_ATOL:g}."
                )
            del repeated_tensor, repeated
        if expression_sha != _tensor_sha256(
            "target_expression", batch.target_expression
        ) or metadata_sha != _tensor_sha256(
            "node_covariates", batch.node_covariates
        ):
            raise SO2RecurrentHLClusteringError(
                f"Prepared node arrays mutated for {alias}."
            )
        if batch.edge_index.device.type != "cpu" or batch.relative_geometry.device.type != "cpu":
            raise SO2RecurrentHLClusteringError(
                f"Full graph tensors moved off CPU for {alias}."
            )
        if hL.shape != (int(record["cell_count"]), 256) or not np.isfinite(hL).all():
            raise SO2RecurrentHLClusteringError(f"Invalid hL output for {alias}.")

        file_path = _core_embedding_path(output_root, core_number)
        receipt_path = _core_receipt_path(output_root, core_number)
        if file_path.exists() or receipt_path.exists():
            raise SO2RecurrentHLClusteringError(
                f"Refusing to overwrite partial core output for {alias}."
            )
        cell_index = np.arange(int(record["cell_count"]), dtype=np.int64)
        _write_deterministic_npz(
            file_path,
            {
                "cell_index": cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": coordinates,
                "hL": hL,
            },
        )
        receipt = _receipt_with_self_hash(
            {
                "schema": CORE_EXTRACTION_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": str(assignment["run_id"]),
                "checkpoint_sha256": str(assignment["checkpoint_sha256"]),
                "alias": alias,
                "core_number": core_number,
                "cell_count": int(record["cell_count"]),
                "directed_edges": int(record["directed_edges"]),
                "hL_shape": list(hL.shape),
                "hL_dtype": str(hL.dtype),
                "hL_array_sha256": _array_sha256("hL", hL),
                "coordinates_array_sha256": _array_sha256(
                    "coordinates_um", coordinates
                ),
                "cell_index_array_sha256": _array_sha256(
                    "cell_index", cell_index
                ),
                "source_expression_sha256": expression_sha,
                "source_metadata_sha256": metadata_sha,
                "source_core": record,
                "inference": {
                    "device": device_name,
                    "device_name": torch.cuda.get_device_name(device),
                    "cpu_threads": int(assignment["cpu_threads"]),
                    "model_eval": True,
                    "torch_inference_mode": True,
                    "full_precision_fp32": True,
                    "autocast_enabled": False,
                    "tf32_enabled": False,
                    "gene_mask_nonzero_count": 0,
                    "complete_core": True,
                    "neighbor_sampling": False,
                    "decoder_target_rows": 0,
                    "hL_source": "model_output.full_node_embedding",
                    "unique_graph_blocks": 1,
                    "recurrent_unroll_steps": 4,
                    "effective_graph_depth": 4,
                    "graph_block_weight_tying": "all_steps",
                    "edge_index_device_before": edge_device_before,
                    "edge_index_device_after": batch.edge_index.device.type,
                    "relative_geometry_device_before": geometry_device_before,
                    "relative_geometry_device_after": batch.relative_geometry.device.type,
                    "elapsed_seconds": float(elapsed),
                    "peak_cuda_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_cuda_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                    "cuda_total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                    "host_max_rss_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                    "repeatability": repeat,
                },
                "embedding_file": file_path.relative_to(output_root).as_posix(),
                "file": _file_record(file_path),
            }
        )
        _atomic_write_json(receipt_path, receipt)
        receipts.append(dict(receipt))
        del expression, covariates, gene_mask, hL_tensor, hL, batch, coordinates
        model.clear_edge_layout_cache()
        torch.cuda.empty_cache()
        gc.collect()
    del model
    torch.cuda.empty_cache()
    return receipts


def _worker_entry(assignment: Mapping[str, Any], queue: Any) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "device": assignment["device"],
                "receipts": _extract_assignment(assignment),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "device": assignment.get("device"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _run_assignments(assignments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not assignments:
        return []
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_worker_entry, args=(dict(assignment), queue))
        for assignment in assignments
    ]
    for process in processes:
        process.start()
    messages = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    failures = [message for message in messages if not message.get("ok")]
    bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
    if failures or bad_exits:
        details = failures[0] if failures else {"exit_codes": bad_exits}
        raise SO2RecurrentHLClusteringError(
            f"GPU extraction worker failed: {details}"
        )
    receipts: list[dict[str, Any]] = []
    for message in messages:
        receipts.extend(dict(receipt) for receipt in message["receipts"])
    return receipts


def _device_inventory(devices: Sequence[str]) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise SO2RecurrentHLClusteringError("CUDA is unavailable for extraction.")
    inventory = []
    for value in devices:
        device = torch.device(value)
        if device.index is None or device.index >= torch.cuda.device_count():
            raise SO2RecurrentHLClusteringError(f"CUDA device is unavailable: {value}")
        properties = torch.cuda.get_device_properties(device)
        inventory.append(
            {
                "device": value,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return inventory


def _balanced_assignments(
    records: Sequence[Mapping[str, Any]], devices: Sequence[str]
) -> dict[str, list[Mapping[str, Any]]]:
    """Greedy longest-processing-time assignment using directed edge count."""

    by_device: dict[str, list[Mapping[str, Any]]] = {device: [] for device in devices}
    loads = {device: 0 for device in devices}
    for record in sorted(
        records,
        key=lambda item: (-int(item["directed_edges"]), int(item["core_number"])),
    ):
        device = min(devices, key=lambda item: (loads[item], devices.index(item)))
        by_device[device].append(record)
        loads[device] += int(record["directed_edges"])
    return by_device


def _assignment_payload(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    device: str,
    records: Sequence[Mapping[str, Any]],
    repeat_core_numbers: Sequence[int],
    cpu_threads: int,
) -> dict[str, Any]:
    return {
        "device": device,
        "core_records": [dict(record) for record in records],
        "repeat_core_numbers": [int(value) for value in repeat_core_numbers],
        "cpu_threads": int(cpu_threads),
        "checkpoint_path": inputs.checkpoint_path.as_posix(),
        "checkpoint_sha256": inputs.checkpoint_sha256,
        "run_id": inputs.run_id,
        "cohort_dir": inputs.cohort_dir.as_posix(),
        "graph_dir": inputs.graph_dir.as_posix(),
        "output_root": output_root.as_posix(),
    }


def _verify_extraction_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO2ResolvedInputs,
    records: Sequence[Mapping[str, Any]],
) -> None:
    _verify_self_hash(receipt, label="recurrent SO2 extraction manifest")
    if any(
        (
            receipt.get("schema") != EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(receipt.get("embedding_dimension", -1)) != 256,
            receipt.get("all_zero_extraction_masks") is not True,
        )
    ):
        raise SO2RecurrentHLClusteringError("Extraction manifest identity is invalid.")
    observed = receipt.get("cores")
    if not isinstance(observed, list) or len(observed) != len(records):
        raise SO2RecurrentHLClusteringError("Extraction manifest is incomplete.")
    for core_receipt, record in zip(observed, records, strict=True):
        if not isinstance(core_receipt, Mapping):
            raise SO2RecurrentHLClusteringError("Core extraction receipt is malformed.")
        _verify_core_receipt(
            output_root=output_root,
            receipt=core_receipt,
            record=record,
            run_id=inputs.run_id,
            checkpoint_sha256=inputs.checkpoint_sha256,
        )


def extract_contextual_embeddings(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    extract_devices: str | Sequence[str] = DEFAULT_EXTRACT_DEVICES,
    cpu_threads_per_worker: int = DEFAULT_CPU_THREADS_PER_WORKER,
) -> Mapping[str, Any]:
    """Extract all recurrent hL arrays after two measured GPU gates."""

    devices = parse_extract_devices(extract_devices)
    if isinstance(cpu_threads_per_worker, bool) or int(cpu_threads_per_worker) <= 0:
        raise SO2RecurrentHLClusteringError("Worker CPU threads must be positive.")
    inventory = _device_inventory(devices)
    records = _source_core_records(inputs)
    receipt_path = output_root / "embeddings" / "extraction_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="recurrent SO2 extraction manifest")
        _verify_extraction_manifest(
            output_root=output_root,
            receipt=receipt,
            inputs=inputs,
            records=records,
        )
        return receipt
    output_root.joinpath("embeddings").mkdir(parents=True, exist_ok=True)

    records_by_core = {int(record["core_number"]): record for record in records}
    completed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        core_number = int(record["core_number"])
        core_path = _core_embedding_path(output_root, core_number)
        core_receipt_path = _core_receipt_path(output_root, core_number)
        if core_receipt_path.is_file():
            receipt = _read_json(
                core_receipt_path, label=f"SO2 Core {core_number} hL receipt"
            )
            _verify_core_receipt(
                output_root=output_root,
                receipt=receipt,
                record=record,
                run_id=inputs.run_id,
                checkpoint_sha256=inputs.checkpoint_sha256,
            )
            completed[core_number] = receipt
        elif core_path.exists():
            raise SO2RecurrentHLClusteringError(
                f"Unreceipted partial hL output exists for SO2 Core {core_number}."
            )

    pilot_records = []
    for core_number in (PILOT_CORE_NUMBER, WORST_CASE_CORE_NUMBER):
        if core_number not in completed:
            record = records_by_core[core_number]
            assignment = _assignment_payload(
                inputs=inputs,
                output_root=output_root,
                device=devices[0],
                records=[record],
                repeat_core_numbers=(
                    [PILOT_CORE_NUMBER] if core_number == PILOT_CORE_NUMBER else []
                ),
                cpu_threads=cpu_threads_per_worker,
            )
            _run_assignments([assignment])
            receipt = _read_json(
                _core_receipt_path(output_root, core_number),
                label=f"SO2 Core {core_number} pilot receipt",
            )
            _verify_core_receipt(
                output_root=output_root,
                receipt=receipt,
                record=record,
                run_id=inputs.run_id,
                checkpoint_sha256=inputs.checkpoint_sha256,
            )
            completed[core_number] = receipt
        pilot_records.append(completed[core_number])
        inference = completed[core_number]["inference"]
        if int(inference["peak_cuda_reserved_bytes"]) > (
            int(inference["cuda_total_memory_bytes"]) - MINIMUM_CUDA_FREE_MARGIN_BYTES
        ):
            raise SO2RecurrentHLClusteringError(
                f"GPU safety-margin gate failed for SO2 Core {core_number}."
            )
    c21_repeat = completed[PILOT_CORE_NUMBER]["inference"]["repeatability"]
    if c21_repeat.get("allclose") is not True:
        raise SO2RecurrentHLClusteringError("C21 repeatability gate did not pass.")

    remaining = [
        record for record in records if int(record["core_number"]) not in completed
    ]
    schedule = _balanced_assignments(remaining, devices)
    assignments = [
        _assignment_payload(
            inputs=inputs,
            output_root=output_root,
            device=device,
            records=device_records,
            repeat_core_numbers=(),
            cpu_threads=cpu_threads_per_worker,
        )
        for device, device_records in schedule.items()
        if device_records
    ]
    _run_assignments(assignments)

    ordered_receipts = []
    for record in records:
        core_number = int(record["core_number"])
        receipt = _read_json(
            _core_receipt_path(output_root, core_number),
            label=f"SO2 Core {core_number} hL receipt",
        )
        _verify_core_receipt(
            output_root=output_root,
            receipt=receipt,
            record=record,
            run_id=inputs.run_id,
            checkpoint_sha256=inputs.checkpoint_sha256,
        )
        ordered_receipts.append(receipt)
    receipt = _receipt_with_self_hash(
        {
            "schema": EXTRACTION_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_dimension": 256,
            "representation": (
                "hL_after_recurrent_unroll_step_4_final_graph_pre_decoder"
            ),
            "all_zero_extraction_masks": True,
            "decoder_target_rows_per_core": 0,
            "full_precision_fp32": True,
            "gpu_inventory": inventory,
            "pilot_gates": [
                {
                    "core_number": int(item["core_number"]),
                    "device": item["inference"]["device"],
                    "elapsed_seconds": item["inference"]["elapsed_seconds"],
                    "peak_cuda_allocated_bytes": item["inference"][
                        "peak_cuda_allocated_bytes"
                    ],
                    "peak_cuda_reserved_bytes": item["inference"][
                        "peak_cuda_reserved_bytes"
                    ],
                    "cuda_total_memory_bytes": item["inference"][
                        "cuda_total_memory_bytes"
                    ],
                    "repeatability": item["inference"]["repeatability"],
                }
                for item in pilot_records
            ],
            "parallel_schedule_after_pilots": {
                device: [int(record["core_number"]) for record in device_records]
                for device, device_records in schedule.items()
            },
            "parallel_schedule_directed_edges": {
                device: int(sum(int(record["directed_edges"]) for record in device_records))
                for device, device_records in schedule.items()
            },
            "input_provenance": inputs.provenance,
            "cores": ordered_receipts,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_extraction_manifest(
        output_root=output_root,
        receipt=receipt,
        inputs=inputs,
        records=records,
    )
    return receipt


def verify_leiden_determinism(
    *,
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    resolution: float,
    random_seed: int,
) -> Mapping[str, Any]:
    """Repeat Leiden on the stored sparse graph and require identical labels."""

    receipt_path = output_root / "diagnostics" / "leiden_determinism.json"
    clustering_manifest_path = output_root / "clustering" / "clustering_manifest.json"
    clustering_sha = sha256_file(clustering_manifest_path)
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="Leiden determinism receipt")
        _verify_self_hash(receipt, label="Leiden determinism receipt")
        if receipt.get("clustering_manifest_sha256") != clustering_sha:
            raise SO2RecurrentHLClusteringError(
                "Leiden determinism source manifest changed."
            )
        if receipt.get("identical_labels") is not True:
            raise SO2RecurrentHLClusteringError("Leiden determinism did not pass.")
        return receipt
    edges = np.load(
        output_root / "clustering" / "contextual_knn_undirected_edges.npy",
        allow_pickle=False,
    )
    stored = np.load(
        output_root / "clustering" / "contextual_labels.npy", allow_pickle=False
    )
    repeated = run_seeded_leiden(
        KNNGraphResult(edge_pairs=edges, receipt={}),
        n_cells=EXPECTED_TOTAL_CELLS,
        resolution=resolution,
        random_seed=random_seed,
    )
    identical = bool(np.array_equal(stored, repeated.labels))
    if not identical:
        raise SO2RecurrentHLClusteringError(
            "Repeated Leiden labels differ from the stored labels."
        )
    receipt = _receipt_with_self_hash(
        {
            "schema": DETERMINISM_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "clustering_manifest_sha256": clustering_sha,
            "resolution": float(resolution),
            "random_seed": int(random_seed),
            "edge_pairs_sha256": _array_sha256("knn_undirected_edges", edges),
            "stored_labels_sha256": _array_sha256("sorted_leiden_labels", stored),
            "repeated_labels_sha256": _array_sha256(
                "sorted_leiden_labels", repeated.labels
            ),
            "identical_labels": identical,
            "cluster_count": int(clustering_receipt["cluster_count"]),
        }
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _cluster_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: int(item[0].removeprefix("C"))
        )
    ]


def _atomic_save_png(figure: Any, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "spatial_benchmark.so2_recurrent_hl_clustering"},
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_spatial_png(
    *,
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    resolution: float,
    dpi: int = DEFAULT_DPI,
) -> Mapping[str, Any]:
    """Render exactly one combined static spatial PNG."""

    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2RecurrentHLClusteringError("PNG DPI must be at least 72.")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    clustering_sha = sha256_file(
        output_root / "clustering" / "clustering_manifest.json"
    )
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="recurrent hL figure receipt")
        _verify_self_hash(receipt, label="recurrent hL figure receipt")
        if receipt.get("clustering_manifest_sha256") != clustering_sha:
            raise SO2RecurrentHLClusteringError("Figure source checksum changed.")
        files = receipt.get("files")
        if not isinstance(files, Mapping) or len(files) != 1:
            raise SO2RecurrentHLClusteringError("PNG-only figure receipt is invalid.")
        for relative, record in files.items():
            path = output_root / str(relative)
            if path.suffix.lower() != ".png" or _file_record(path) != record:
                raise SO2RecurrentHLClusteringError("Spatial PNG checksum changed.")
        return receipt

    frame = pd.read_parquet(
        output_root / "tables" / "cell_contextual_clusters.parquet"
    )
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2RecurrentHLClusteringError("Figure table is incomplete.")
    palette = clustering_receipt.get("palette")
    if not isinstance(palette, Mapping):
        raise SO2RecurrentHLClusteringError("Clustering palette is missing.")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": int(dpi),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    for axis, core_number in zip(
        axes.ravel()[: len(SO2_CORE_NUMBERS)], SO2_CORE_NUMBERS, strict=True
    ):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected["contextual_cluster"].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS_BY_CORE[core_number]:
            raise SO2RecurrentHLClusteringError(
                f"Spatial panel is incomplete for SO2 Core {core_number}."
            )
        if colors.isna().any():
            raise SO2RecurrentHLClusteringError("Spatial palette is incomplete.")
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
            f"SO2 Core {core_number}\n$n$ = {len(selected):,}",
            fontsize=12,
            weight="bold",
            pad=7,
        )
        axis.text(
            0.025,
            0.97,
            f"CORE {core_number}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            weight="bold",
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
            },
            zorder=6,
        )
        _style_spatial_axis(axis, coordinates)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = _cluster_legend_handles(dict(palette))
    legend_axis.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=2 if len(handles) > 14 else 1,
        title="Joint recurrent hL\ncluster",
        fontsize=8,
        title_fontsize=10,
        borderaxespad=0.0,
    )
    figure.suptitle(
        "SO2 recurrent one-block hL Leiden clusters — joint 14-core clustering\n"
        f"resolution {resolution:g}",
        fontsize=18,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Model-derived clusters; no cell-type or biological annotation is implied.",
        ha="center",
        va="bottom",
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
    png_path = (
        output_root
        / "figures"
        / "contextual_leiden_resolution_1p0_spatial_14cores.png"
    )
    _atomic_save_png(figure, png_path, dpi=int(dpi))
    plt.close(figure)
    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "clustering_manifest_sha256": clustering_sha,
            "leiden_resolution": float(resolution),
            "dpi": int(dpi),
            "point_count": EXPECTED_TOTAL_CELLS,
            "one_dot_per_cell": True,
            "marker_borders": False,
            "lines_between_cells": False,
            "png_only": True,
            "interactive_map_created": False,
            "pdf_created": False,
            "plot_specification": {
                "panel_order": list(SO2_CORE_NUMBERS),
                "grid_shape": [3, 5],
                "legend_panel": [2, 4],
                "equal_aspect": True,
                "invert_y_axis": True,
                "coordinate_units": "micrometres",
                "one_shared_joint_cluster_palette": True,
                "palette": dict(palette),
            },
            "files": {
                png_path.relative_to(output_root).as_posix(): _file_record(png_path)
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _render_readme(
    *,
    inputs: SO2ResolvedInputs,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
) -> str:
    dominated = ", ".join(clustering.get("core_dominated_gt_90pct", [])) or "None"
    pilot_lines = "\n".join(
        "- Core {core_number}: {elapsed_seconds:.2f} s, peak reserved "
        "{peak_gib:.2f} GiB on {device}".format(
            core_number=int(item["core_number"]),
            elapsed_seconds=float(item["elapsed_seconds"]),
            peak_gib=float(item["peak_cuda_reserved_bytes"]) / 1024**3,
            device=item["device"],
        )
        for item in extraction["pilot_gates"]
    )
    return f"""# SO2 recurrent one-block contextual hL clustering

This is an exploratory post-hoc readout of checkpoint `{inputs.checkpoint_path}`
(SHA-256 `{inputs.checkpoint_sha256}`). The model completed 175 training epochs
and passed exact reload/prediction replay, but the source run remains status
`failed` because its post-training artifact finalizer rejected the canonical
fit-prediction protocol. This report does not alter or promote that run.

## Result and method

All {EXPECTED_TOTAL_CELLS:,} cells from SO2 cores 15--28 were clustered jointly.
`hL` is the 256-dimensional output after the fourth recurrent application of
one fully weight-tied graph block, immediately before the decoder. Extraction
used complete-core graphs, an all-zero gene mask, eval/inference mode, FP32, and
empty decoder targets. It then used exact mean-centered PCA50, L2 normalization,
a sparse cosine FAISS-HNSW 30-nearest-neighbor graph, and Leiden resolution 1.0
with seed 20260825. The result has {int(clustering['cluster_count'])} clusters,
with sizes {int(clustering['cluster_size_range'][0]):,}--
{int(clustering['cluster_size_range'][1]):,}.
Clusters with >90% of their cells from one core: {dominated}.

GPU pilot gates:

{pilot_lines}

Only one static rendered map was created:
`figures/contextual_leiden_resolution_1p0_spatial_14cores.png`. No interactive
map or PDF was produced.

## Interpretation limit

These are descriptive model-derived groups from one seed and one transductively
fit cohort. Spatial coherence does not establish cell identity, predictive
dependency, faithfulness, patient replication, biological mechanism, or causal
influence. Core/batch structure and spatially smooth covariates remain credible
alternative explanations.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  analyze-so2-recurrent-hl-clusters \\
  --run-id {inputs.run_id} \\
  --checkpoint {inputs.checkpoint_path.relative_to(inputs.project_root)} \\
  --n-neighbors 30 --leiden-resolution 1.0 \\
  --pca-components 50 --random-seed 20260825 \\
  --extract-devices cuda:0,cuda:1,cuda:2,cuda:3
```
"""


def _verify_final_manifest(output_root: Path, manifest: Mapping[str, Any]) -> None:
    _verify_self_hash(manifest, label="recurrent SO2 final analysis manifest")
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            manifest.get("rendered_output_policy") != "one_static_png_only",
            manifest.get("source_run_registry_status") != "failed",
        )
    ):
        raise SO2RecurrentHLClusteringError("Final analysis identity is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2RecurrentHLClusteringError("Final analysis has no file manifest.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2RecurrentHLClusteringError(f"Final output is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2RecurrentHLClusteringError(f"Final checksum changed: {relative}")
    forbidden = [
        relative
        for relative in files
        if Path(str(relative)).suffix.lower() in {".html", ".htm", ".pdf"}
    ]
    if forbidden:
        raise SO2RecurrentHLClusteringError(
            f"Forbidden interactive/PDF output exists: {forbidden}"
        )
    required = {
        "README.md",
        "embeddings/extraction_manifest.json",
        "clustering/clustering_manifest.json",
        "clustering/contextual_labels.npy",
        "tables/cell_contextual_clusters.parquet",
        "tables/contextual_cluster_summary.csv",
        "tables/contextual_cluster_core_composition.csv",
        "diagnostics/leiden_determinism.json",
        "figures/figure_manifest.json",
        "figures/contextual_leiden_resolution_1p0_spatial_14cores.png",
    }
    required.update(
        f"embeddings/core_{number}_hL.npz" for number in SO2_CORE_NUMBERS
    )
    if not required.issubset(files):
        raise SO2RecurrentHLClusteringError(
            f"Required outputs are absent: {sorted(required.difference(files))}"
        )


def _finalize_analysis(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
    determinism: Mapping[str, Any],
    analysis_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    _atomic_write_text(
        output_root / "README.md",
        _render_readme(
            inputs=inputs, extraction=extraction, clustering=clustering
        ),
    )
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_scope": "recurrent_contextual_hL_only",
            "run_id": inputs.run_id,
            "campaign_id": CAMPAIGN_ID,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "source_run_registry_status": "failed",
            "source_training_completed": True,
            "source_canonical_completion_claimed": False,
            "source_finalization_failure_retained": True,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "hL_shape": [EXPECTED_TOTAL_CELLS, 256],
            "analysis_parameters": dict(analysis_parameters),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_inventory": extraction["gpu_inventory"],
                "parallel_schedule_after_pilots": extraction[
                    "parallel_schedule_after_pilots"
                ],
            },
            "input_provenance": inputs.provenance,
            "stage_manifests": {
                "extraction": _file_record(
                    output_root / "embeddings" / "extraction_manifest.json"
                ),
                "clustering": _file_record(
                    output_root / "clustering" / "clustering_manifest.json"
                ),
                "determinism": _file_record(
                    output_root / "diagnostics" / "leiden_determinism.json"
                ),
                "figures": _file_record(
                    output_root / "figures" / "figure_manifest.json"
                ),
            },
            "cluster_count": int(clustering["cluster_count"]),
            "cluster_size_range": clustering["cluster_size_range"],
            "core_dominated_gt_90pct": clustering["core_dominated_gt_90pct"],
            "leiden_repeat_identical": determinism["identical_labels"],
            "rendered_output_policy": "one_static_png_only",
            "figure_paths": sorted(figures["files"]),
            "interactive_map_created": False,
            "pdf_created": False,
            "interpretation": {
                "clusters_are_model_derived": True,
                "cell_types_established": False,
                "predictive_dependency_established": False,
                "faithfulness_established": False,
                "patient_replication_established": False,
                "biological_mechanism_established": False,
                "causality_established": False,
            },
            "files": _file_manifest(output_root),
        }
    )
    _atomic_write_json(output_root / "manifest.json", manifest)
    _verify_final_manifest(output_root, manifest)
    return manifest


def run_so2_recurrent_hl_clustering(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extract_devices: str | Sequence[str] = DEFAULT_EXTRACT_DEVICES,
    cpu_threads_per_worker: int = DEFAULT_CPU_THREADS_PER_WORKER,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run, verify, and atomically publish the recurrent hL static map."""

    if not math.isclose(
        float(leiden_resolution), DEFAULT_LEIDEN_RESOLUTION, rel_tol=0.0, abs_tol=0.0
    ):
        raise SO2RecurrentHLClusteringError(
            "The requested map is locked to Leiden resolution 1.0."
        )
    inputs = resolve_recurrent_analysis_inputs(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    if output_dir is None:
        final_root = (
            paths.report_root
            / "analyses"
            / "so2_14core_recurrent_contextual_embedding_clustering"
            / inputs.run_id
        )
    else:
        final_root = Path(output_dir).expanduser()
        if not final_root.is_absolute():
            final_root = paths.project_root / final_root
        final_root = final_root.resolve(strict=False)
    stage_root = (
        paths.scratch_root
        / "active_runs"
        / inputs.run_id
        / "posthoc_reports"
        / "so2_14core_recurrent_contextual_embedding_clustering"
    )
    parameters = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    parameters.update(
        {
            "extract_devices": list(parse_extract_devices(extract_devices)),
            "cpu_threads_per_worker": int(cpu_threads_per_worker),
            "figure_dpi": int(dpi),
            "rendered_output_policy": "one_static_png_only",
        }
    )
    if final_root.is_dir():
        manifest = _read_json(final_root / "manifest.json", label="final analysis")
        _verify_final_manifest(final_root, manifest)
        if manifest.get("analysis_parameters") != parameters:
            raise SO2RecurrentHLClusteringError(
                "Existing completed analysis used different parameters."
            )
    else:
        stage_root.mkdir(parents=True, exist_ok=True)
        manifest_path = stage_root / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path, label="staged final analysis")
            _verify_final_manifest(stage_root, manifest)
            if manifest.get("analysis_parameters") != parameters:
                raise SO2RecurrentHLClusteringError(
                    "Staged completed analysis used different parameters."
                )
        else:
            extraction = extract_contextual_embeddings(
                inputs=inputs,
                output_root=stage_root,
                extract_devices=extract_devices,
                cpu_threads_per_worker=cpu_threads_per_worker,
            )
            clustering = cluster_joint_contextual_embeddings(
                inputs=inputs,
                output_root=stage_root,
                extraction_receipt=extraction,
                n_neighbors=n_neighbors,
                leiden_resolution=leiden_resolution,
                pca_components=pca_components,
                random_seed=random_seed,
            )
            determinism = verify_leiden_determinism(
                output_root=stage_root,
                clustering_receipt=clustering,
                resolution=leiden_resolution,
                random_seed=random_seed,
            )
            figures = render_spatial_png(
                output_root=stage_root,
                clustering_receipt=clustering,
                resolution=leiden_resolution,
                dpi=dpi,
            )
            manifest = _finalize_analysis(
                inputs=inputs,
                output_root=stage_root,
                extraction=extraction,
                clustering=clustering,
                figures=figures,
                determinism=determinism,
                analysis_parameters=parameters,
            )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            raise SO2RecurrentHLClusteringError(
                "Final report path appeared during publication."
            )
        os.replace(stage_root, final_root)
        _verify_final_manifest(final_root, manifest)

    png = (
        final_root
        / "figures"
        / "contextual_leiden_resolution_1p0_spatial_14cores.png"
    )
    return {
        "status": "complete",
        "run_id": inputs.run_id,
        "source_run_registry_status": "failed",
        "source_training_completed": True,
        "output_root": final_root.as_posix(),
        "total_cells": int(manifest["total_cells"]),
        "hL_shape": manifest["hL_shape"],
        "cluster_count": int(manifest["cluster_count"]),
        "cluster_size_range": manifest["cluster_size_range"],
        "core_dominated_gt_90pct": manifest["core_dominated_gt_90pct"],
        "combined_png": png.as_posix(),
        "interactive_map_created": False,
        "pdf_created": False,
        "manifest": (final_root / "manifest.json").as_posix(),
    }


__all__ = [
    "SO2RecurrentHLClusteringError",
    "extract_contextual_embeddings",
    "extract_full_recurrent_contextual_embedding",
    "load_recurrent_checkpoint_model",
    "parse_extract_devices",
    "render_spatial_png",
    "resolve_recurrent_analysis_inputs",
    "run_so2_recurrent_hl_clustering",
    "verify_leiden_determinism",
]

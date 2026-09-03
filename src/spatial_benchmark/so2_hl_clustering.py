"""CPU-only joint contextual-embedding clustering for the 14-core SO2 cohort.

The analyzed representation is ``hL``: the final graph-block output immediately
before the expression decoder.  Clusters are model-derived descriptive groups;
this module does not assign biological or cell-type identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hmac
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .checkpoint_catalog import resolve_checkpoint
from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    EmbeddingClusterAnalysisError,
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
    _git_provenance,
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
    deterministic_pca,
    run_seeded_leiden,
)
from .relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from .relative_qkv_post_training import _tree_sha256
from .run_archive import RunArchive, verify_run_bundle
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
)
from .so2_relative_graphs import load_so2_relative_qkv_batches
from .so2_training_observability import CHECKPOINT_SCHEMA


ANALYSIS_SCHEMA = "so2_14core_contextual_embedding_clustering_v1"
EXTRACTION_SCHEMA = "so2_14core_hl_extraction_v1"
CLUSTERING_SCHEMA = "so2_14core_joint_hl_clustering_v1"
FIGURE_SCHEMA = "so2_14core_hl_spatial_figure_v1"
CAMPAIGN_ID = "cmp_20260825_so2_14core_relative_qkv_seed0_batch2"
EXPECTED_RUN_ID = "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6"
EXPECTED_MODEL_SEED = 0
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_CPU_THREADS = 40
LABEL_PREFIX = "C"


class SO2HLClusteringError(EmbeddingClusterAnalysisError):
    """Raised when the locked SO2 contextual-clustering contract is violated."""


@dataclass(frozen=True, slots=True)
class SO2ResolvedInputs:
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
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SO2ContextualCore:
    alias: str
    core_number: int
    cell_index: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    hL: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_index))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cpu_device(device: str | torch.device) -> torch.device:
    """Fail closed rather than allowing this analysis to reserve a training GPU."""

    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise SO2HLClusteringError(
            "The SO2 hL analysis is CPU-only; --device must be 'cpu'."
        )
    return torch.device("cpu")


def requested_panel_order() -> tuple[int, ...]:
    return SO2_CORE_NUMBERS


def spatial_plot_spec(palette: Mapping[str, str]) -> dict[str, Any]:
    return {
        "panel_order": list(SO2_CORE_NUMBERS),
        "grid_shape": [3, 5],
        "legend_panel": [2, 4],
        "panel_title_template": "SO2 Core {core_number}",
        "core_numbers_identifiable": True,
        "equal_aspect": True,
        "invert_y_axis": True,
        "coordinate_units": "micrometres",
        "one_shared_joint_cluster_palette": True,
        "palette": dict(palette),
    }


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO2HLClusteringError(f"Cannot load checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SO2HLClusteringError("Checkpoint payload must be a mapping.")
    return dict(payload)


def _manifest_content_is_valid(manifest: Mapping[str, Any]) -> bool:
    content = dict(manifest)
    observed = str(content.pop("manifest_content_sha256", ""))
    return bool(observed) and hmac.compare_digest(observed, _canonical_sha256(content))


def resolve_so2_analysis_inputs(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None,
    checkpoint: str | Path | None,
) -> SO2ResolvedInputs:
    """Resolve the only completed locked SO2 run and verify immutable inputs."""

    requested_run = EXPECTED_RUN_ID if run_id is None else str(run_id)
    canonical_run_id = registry.resolve_run_id(requested_run)
    if canonical_run_id != EXPECTED_RUN_ID:
        raise SO2HLClusteringError(
            "This analysis is locked to the completed SO2 14-core seed-0 run."
        )
    canonical_checkpoint = resolve_checkpoint(
        registry, canonical_run_id, paths, role="last"
    ).resolve(strict=True)
    if checkpoint is None:
        checkpoint_path = canonical_checkpoint
    else:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve(strict=True)
        if not hmac.compare_digest(
            sha256_file(checkpoint_path), sha256_file(canonical_checkpoint)
        ):
            raise SO2HLClusteringError(
                "Explicit checkpoint differs from the verified final checkpoint."
            )
    payload = _load_checkpoint_payload(checkpoint_path)
    if any(
        (
            payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA,
            payload.get("campaign_id") != CAMPAIGN_ID,
            payload.get("run_id") != canonical_run_id,
            int(payload.get("model_seed", -1)) != EXPECTED_MODEL_SEED,
        )
    ):
        raise SO2HLClusteringError("Checkpoint identity is outside the locked contract.")
    run_record = registry.show_run(canonical_run_id)
    if run_record is None or str(run_record.get("status")) != "completed":
        raise SO2HLClusteringError("Selected SO2 run is not completed.")

    bundle_path = RunArchive.artifact_path_for(canonical_run_id, paths)
    verify_run_bundle(bundle_path)
    config = _read_yaml(
        bundle_path / "config.resolved.yaml", label="resolved run configuration"
    )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SO2HLClusteringError("Resolved run lacks a dataset section.")
    cohort_dir = _project_artifact_path(dataset.get("prepared_artifact"), paths=paths)
    graph_dir = _project_artifact_path(
        dataset.get("prepared_graph_artifact"), paths=paths
    )
    cohort_manifest_path = cohort_dir / "manifest.json"
    graph_manifest_path = graph_dir / "manifest.json"
    cohort_file_sha = sha256_file(cohort_manifest_path)
    graph_file_sha = sha256_file(graph_manifest_path)
    if cohort_file_sha != dataset.get("cohort_manifest_file_sha256"):
        raise SO2HLClusteringError("Prepared cohort manifest changed from the run.")
    if graph_file_sha != dataset.get("graph_manifest_file_sha256"):
        raise SO2HLClusteringError("Prepared graph manifest changed from the run.")
    cohort_manifest = _read_json(cohort_manifest_path, label="SO2 cohort manifest")
    graph_manifest = _read_json(graph_manifest_path, label="SO2 graph manifest")
    if not _manifest_content_is_valid(cohort_manifest):
        raise SO2HLClusteringError("SO2 cohort manifest self-checksum is invalid.")
    if not _manifest_content_is_valid(graph_manifest):
        raise SO2HLClusteringError("SO2 graph manifest self-checksum is invalid.")
    if tuple(cohort_manifest.get("cohort", {}).get("aliases", ())) != SO2_ALIASES:
        raise SO2HLClusteringError("SO2 cohort order changed.")
    if tuple(graph_manifest.get("aliases", ())) != SO2_ALIASES:
        raise SO2HLClusteringError("SO2 graph order changed.")

    construction = payload.get("model_construction")
    if not isinstance(construction, Mapping):
        raise SO2HLClusteringError("Checkpoint lacks model construction metadata.")
    training_git = _read_json(
        bundle_path / "provenance/git.json", label="training Git provenance"
    )
    provenance = {
        "selection_policy": "only_completed_matching_run",
        "campaign_id": CAMPAIGN_ID,
        "run_id": canonical_run_id,
        "model_seed": int(payload["model_seed"]),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "schema": payload["checkpoint_schema"],
            "completed_global_epochs": int(payload["completed_global_epochs"]),
            "model_state_sha256": payload.get("model_state_checksum"),
        },
        "training_source": training_git,
        "analysis_source": _git_provenance(paths.project_root),
        "model_construction": dict(construction),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "preprocessing_version": dataset.get("preprocessing_version"),
        "cohort_manifest": {
            "path": cohort_manifest_path.as_posix(),
            "file_sha256": cohort_file_sha,
            "content_sha256": cohort_manifest.get("manifest_content_sha256"),
            "preprocessing_statistics_file_sha256": cohort_manifest.get(
                "files", {}
            ).get("cohort_statistics.npz"),
            "statistics_component_checksums": cohort_manifest.get(
                "preprocessing", {}
            ).get("statistics_checksums", {}),
        },
        "graph_manifest": {
            "path": graph_manifest_path.as_posix(),
            "file_sha256": graph_file_sha,
            "content_sha256": graph_manifest.get("manifest_content_sha256"),
        },
    }
    return SO2ResolvedInputs(
        run_id=canonical_run_id,
        project_root=paths.project_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
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
        raise SO2HLClusteringError(f"Invalid model construction field: {name}.") from exc


def load_so2_checkpoint_model(
    inputs: SO2ResolvedInputs,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    """Construct the exact trained model on CPU and verify its state checksum."""

    payload = inputs.checkpoint_payload
    construction = payload["model_construction"]
    assert isinstance(construction, Mapping)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise SO2HLClusteringError("Checkpoint lacks a model state dictionary.")
    observed_state = _tree_sha256(state)
    if observed_state != payload.get("model_state_checksum"):
        raise SO2HLClusteringError("Checkpoint model-state checksum is invalid.")
    if any(
        (
            construction.get("class")
            != "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
            _integer(construction, "num_genes") != EXPECTED_N_GENES,
            _integer(construction, "node_covariate_dim") != 22,
        )
    ):
        raise SO2HLClusteringError("Checkpoint model construction is unexpected.")
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
    if _tree_sha256(model.state_dict()) != observed_state:
        raise SO2HLClusteringError("Loaded model state changed.")
    model.to(torch.device("cpu"))
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise SO2HLClusteringError("model.eval() did not disable training mode.")
    if {float(block.attention_dropout_probability) for block in model.blocks} != {
        0.0
    }:
        raise SO2HLClusteringError("Attention dropout is not disabled.")
    return model


def extract_full_contextual_embedding(
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    input_expression: torch.Tensor,
    gene_mask: torch.Tensor,
    edge_index: torch.Tensor,
    relative_geometry: torch.Tensor,
    node_covariates: torch.Tensor,
) -> torch.Tensor:
    """Return all-node hL while decoding an empty, zero-cost target selection.

    All graph layers still process the complete core.  ``full_node_embedding``
    is the public pre-decoder all-node result; the decoder receives a ``[0, D]``
    tensor.  This helper is regression-tested against the ordinary public
    forward path's ``final_graph_embedding``.
    """

    if model.training:
        raise SO2HLClusteringError("Contextual extraction requires model.eval().")
    if input_expression.device.type != "cpu" or node_covariates.device.type != "cpu":
        raise SO2HLClusteringError("Contextual extraction is CPU-only.")
    if gene_mask.dtype is not torch.bool or gene_mask.shape != input_expression.shape:
        raise SO2HLClusteringError("Extraction gene mask has an invalid shape or dtype.")
    if bool(torch.any(gene_mask).item()):
        raise SO2HLClusteringError("Contextual extraction requires an all-zero mask.")
    if edge_index.device.type != "cpu" or relative_geometry.device.type != "cpu":
        raise SO2HLClusteringError("Graph and relative geometry must remain on CPU.")
    num_nodes = int(input_expression.shape[0])
    empty_targets = torch.empty((0,), dtype=torch.long, device=input_expression.device)
    output = model(
        input_expression=input_expression,
        gene_mask=gene_mask,
        edge_index=edge_index,
        relative_geometry=relative_geometry,
        node_covariates=node_covariates,
        target_nodes=empty_targets,
    )
    node_embedding = output.full_node_embedding
    if output.prediction.shape != (0, model.num_genes):
        raise SO2HLClusteringError("Decoder-skip target selection was not empty.")
    if node_embedding.shape != (num_nodes, model.hidden_dim):
        raise SO2HLClusteringError("Extracted hL shape is invalid.")
    if not bool(torch.isfinite(node_embedding).all().item()):
        raise SO2HLClusteringError("Extracted hL contains non-finite values.")
    return node_embedding


def _expected_core_records(
    inputs: SO2ResolvedInputs,
) -> tuple[dict[str, Any], ...]:
    cohort_records = inputs.cohort_manifest.get("cores")
    graph_records = inputs.graph_manifest.get("cores")
    cohort_files = inputs.cohort_manifest.get("files")
    if not isinstance(cohort_records, list) or not isinstance(graph_records, list):
        raise SO2HLClusteringError("Prepared manifests lack core records.")
    if not isinstance(cohort_files, Mapping):
        raise SO2HLClusteringError("Prepared cohort manifest lacks file checksums.")
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
    for core_number, alias in zip(SO2_CORE_NUMBERS, SO2_ALIASES, strict=True):
        cohort = cohort_by_alias.get(alias)
        graph = graph_by_alias.get(alias)
        if not isinstance(cohort, Mapping) or not isinstance(graph, Mapping):
            raise SO2HLClusteringError(f"Prepared input record is missing for {alias}.")
        expected_cells = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        if int(cohort.get("cell_count", -1)) != expected_cells:
            raise SO2HLClusteringError(f"Prepared cell count changed for {alias}.")
        result.append(
            {
                "alias": alias,
                "core_number": int(core_number),
                "cell_count": int(expected_cells),
                "prepared_core_artifact_sha256": str(
                    cohort_files[f"cores/{alias}.npz"]
                ),
                "prepared_component_checksums": dict(
                    cohort.get("component_checksums", {})
                ),
                "graph_record_sha256": str(graph.get("record_sha256")),
                "graph_logical_sha256": str(
                    graph.get("graph", {}).get("checksums", {}).get("graph_sha256")
                ),
                "graph_file_checksums": dict(graph.get("files", {})),
            }
        )
    if sum(record["cell_count"] for record in result) != EXPECTED_TOTAL_CELLS:
        raise SO2HLClusteringError("SO2 core records do not sum to 246,063 cells.")
    return tuple(result)


def _core_embedding_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_hL.npz"


def _core_receipt_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_receipt.json"


def _validate_contextual_arrays(
    *,
    alias: str,
    core_number: int,
    arrays: Mapping[str, np.ndarray],
    expected_cells: int,
) -> SO2ContextualCore:
    if set(arrays) != {"cell_index", "core_number", "coordinates_um", "hL"}:
        raise SO2HLClusteringError(f"Unexpected hL artifact schema for {alias}.")
    cell_index = np.asarray(arrays["cell_index"], dtype=np.int64)
    number = np.asarray(arrays["core_number"])
    coordinates = np.asarray(arrays["coordinates_um"], dtype=np.float64)
    hL = np.asarray(arrays["hL"], dtype=np.float32)
    if not np.array_equal(cell_index, np.arange(expected_cells, dtype=np.int64)):
        raise SO2HLClusteringError(f"Cell order changed for {alias}.")
    if number.size != 1 or int(number.reshape(-1)[0]) != int(core_number):
        raise SO2HLClusteringError(f"Core number changed for {alias}.")
    if coordinates.shape != (expected_cells, 2):
        raise SO2HLClusteringError(f"Coordinate shape changed for {alias}.")
    if hL.ndim != 2 or hL.shape[0] != expected_cells or hL.shape[1] <= 0:
        raise SO2HLClusteringError(f"hL shape is invalid for {alias}.")
    if not np.isfinite(coordinates).all() or not np.isfinite(hL).all():
        raise SO2HLClusteringError(f"Non-finite coordinates or hL values for {alias}.")
    return SO2ContextualCore(
        alias=alias,
        core_number=int(core_number),
        cell_index=np.ascontiguousarray(cell_index),
        coordinates_um=np.ascontiguousarray(coordinates),
        hL=np.ascontiguousarray(hL),
    )


def load_contextual_core(
    path: Path, *, alias: str, core_number: int, expected_cells: int
) -> SO2ContextualCore:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise SO2HLClusteringError(f"Cannot load hL artifact: {path}") from exc
    return _validate_contextual_arrays(
        alias=alias,
        core_number=core_number,
        arrays=arrays,
        expected_cells=expected_cells,
    )


def _verify_core_extraction_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO2ResolvedInputs,
    expected: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO2 per-core extraction receipt")
    if any(
        (
            receipt.get("schema") != "so2_14core_hl_core_extraction_v1",
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            receipt.get("alias") != expected["alias"],
            int(receipt.get("core_number", -1)) != expected["core_number"],
            int(receipt.get("cell_count", -1)) != expected["cell_count"],
            receipt.get("source_core") != dict(expected),
        )
    ):
        raise SO2HLClusteringError("Per-core extraction receipt identity is invalid.")
    relative = str(receipt.get("embedding_file", ""))
    record = receipt.get("file")
    path = output_root / relative
    if not isinstance(record, Mapping) or not path.is_file():
        raise SO2HLClusteringError("Per-core extraction output is missing.")
    if _file_record(path) != dict(record):
        raise SO2HLClusteringError("Per-core extraction checksum changed.")
    core = load_contextual_core(
        path,
        alias=str(expected["alias"]),
        core_number=int(expected["core_number"]),
        expected_cells=int(expected["cell_count"]),
    )
    if list(core.hL.shape) != list(receipt.get("hL_shape", ())):
        raise SO2HLClusteringError("Per-core hL shape receipt changed.")
    if _array_sha256("hL", core.hL) != receipt.get("hL_array_sha256"):
        raise SO2HLClusteringError("Per-core hL content checksum changed.")


def _verify_extraction_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO2ResolvedInputs,
) -> None:
    _verify_self_hash(receipt, label="SO2 extraction manifest")
    if any(
        (
            receipt.get("schema") != EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(receipt.get("embedding_dimension", -1))
            != int(inputs.checkpoint_payload["model_construction"]["hidden_dim"]),
            receipt.get("device") != "cpu",
        )
    ):
        raise SO2HLClusteringError("SO2 extraction manifest identity is invalid.")
    records = receipt.get("cores")
    expected_records = _expected_core_records(inputs)
    if not isinstance(records, list) or len(records) != len(expected_records):
        raise SO2HLClusteringError("SO2 extraction manifest is incomplete.")
    for observed, expected in zip(records, expected_records, strict=True):
        if not isinstance(observed, Mapping):
            raise SO2HLClusteringError("SO2 extraction core receipt is malformed.")
        _verify_core_extraction_receipt(
            output_root=output_root,
            receipt=observed,
            inputs=inputs,
            expected=expected,
        )


def extract_contextual_embeddings(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
) -> Mapping[str, Any]:
    """Extract or checksum-verify all 14 hL files, one complete core at a time."""

    validate_cpu_device(device)
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise SO2HLClusteringError("cpu_threads must be positive.")
    receipt_path = output_root / "embeddings" / "extraction_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO2 extraction manifest")
        _verify_extraction_manifest(
            output_root=output_root, receipt=receipt, inputs=inputs
        )
        return receipt

    torch.set_num_threads(int(cpu_threads))
    batches = load_so2_relative_qkv_batches(
        cohort_dir=inputs.cohort_dir, graph_dir=inputs.graph_dir
    )
    if tuple(batch.alias for batch in batches) != SO2_ALIASES:
        raise SO2HLClusteringError("Loaded SO2 batches are missing or reordered.")
    model = load_so2_checkpoint_model(inputs)
    expected_records = _expected_core_records(inputs)
    output_root.joinpath("embeddings").mkdir(parents=True, exist_ok=True)
    receipts_by_alias: dict[str, Mapping[str, Any]] = {}
    inference_pairs = sorted(
        zip(batches, expected_records, strict=True),
        key=lambda pair: (int(pair[1]["cell_count"]), int(pair[1]["core_number"])),
    )

    for batch, expected in inference_pairs:
        core_number = int(expected["core_number"])
        alias = str(expected["alias"])
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
            raise SO2HLClusteringError(
                f"Unreceipted partial extraction exists for {alias}: {file_path}"
            )
        if int(batch.n_nodes) != int(expected["cell_count"]):
            raise SO2HLClusteringError(f"Loaded cell count changed for {alias}.")
        with np.load(
            inputs.cohort_dir / "cores" / f"{alias}.npz", allow_pickle=False
        ) as archive:
            coordinates = np.array(
                archive["coordinates_um"], dtype=np.float64, copy=True
            )
        if coordinates.shape != (int(batch.n_nodes), 2):
            raise SO2HLClusteringError(f"Coordinates do not align for {alias}.")
        expression = batch.target_expression.to(dtype=torch.float32, device="cpu")
        covariates = batch.node_covariates.to(dtype=torch.float32, device="cpu")
        gene_mask = torch.zeros_like(expression, dtype=torch.bool, device="cpu")
        if int(torch.count_nonzero(gene_mask).item()) != 0:
            raise SO2HLClusteringError("Extraction mask is not all-zero.")
        metadata_before = _tensor_sha256("node_covariates", batch.node_covariates)
        expression_before = _tensor_sha256(
            "target_expression", batch.target_expression
        )
        started = time.monotonic()
        with torch.inference_mode():
            hL_tensor = extract_full_contextual_embedding(
                model,
                expression,
                gene_mask,
                batch.edge_index,
                batch.relative_geometry,
                covariates,
            )
        elapsed = time.monotonic() - started
        hL = np.ascontiguousarray(
            hL_tensor.detach().cpu().float().numpy(), dtype=np.float32
        )
        if metadata_before != _tensor_sha256(
            "node_covariates", batch.node_covariates
        ) or expression_before != _tensor_sha256(
            "target_expression", batch.target_expression
        ):
            raise SO2HLClusteringError(
                f"Prepared expression or metadata mutated for {alias}."
            )
        cell_index = np.arange(int(batch.n_nodes), dtype=np.int64)
        validated = _validate_contextual_arrays(
            alias=alias,
            core_number=core_number,
            arrays={
                "cell_index": cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": coordinates,
                "hL": hL,
            },
            expected_cells=int(batch.n_nodes),
        )
        relative = file_path.relative_to(output_root).as_posix()
        _write_deterministic_npz(
            file_path,
            {
                "cell_index": validated.cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": validated.coordinates_um,
                "hL": validated.hL,
            },
        )
        core_receipt = _receipt_with_self_hash(
            {
                "schema": "so2_14core_hl_core_extraction_v1",
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": inputs.run_id,
                "checkpoint_sha256": inputs.checkpoint_sha256,
                "alias": alias,
                "core_number": core_number,
                "cell_count": validated.n_cells,
                "hL_shape": list(validated.hL.shape),
                "hL_dtype": str(validated.hL.dtype),
                "hL_array_sha256": _array_sha256("hL", validated.hL),
                "coordinates_array_sha256": _array_sha256(
                    "coordinates_um", validated.coordinates_um
                ),
                "cell_index_array_sha256": _array_sha256(
                    "cell_index", validated.cell_index
                ),
                "source_expression_sha256": expression_before,
                "source_metadata_sha256": metadata_before,
                "source_core": dict(expected),
                "inference": {
                    "device": "cpu",
                    "cpu_threads": int(cpu_threads),
                    "model_eval": True,
                    "torch_inference_mode": True,
                    "gene_mask_nonzero_count": 0,
                    "dropout_disabled_by_eval": True,
                    "attention_dropout_probability": 0.0,
                    "complete_core": True,
                    "neighbor_sampling": False,
                    "decoder_target_rows": 0,
                    "hL_source": "model_output.full_node_embedding",
                    "coordinates_supplied_to_node_encoder": False,
                    "batch_correction_added": False,
                    "elapsed_seconds": float(elapsed),
                },
                "embedding_file": relative,
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
        del expression, covariates, gene_mask, hL_tensor, hL
        model.clear_edge_layout_cache()
        gc.collect()

    core_receipts = [receipts_by_alias[alias] for alias in SO2_ALIASES]
    receipt = _receipt_with_self_hash(
        {
            "schema": EXTRACTION_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_dimension": int(
                inputs.checkpoint_payload["model_construction"]["hidden_dim"]
            ),
            "representation": "hL_final_graph_pre_decoder",
            "device": "cpu",
            "cpu_threads": int(cpu_threads),
            "one_complete_core_at_a_time": True,
            "inference_schedule": "ascending_cell_count_then_core_number",
            "stored_core_order": list(SO2_CORE_NUMBERS),
            "all_zero_extraction_masks": True,
            "decoder_target_rows_per_core": 0,
            "input_provenance": inputs.provenance,
            "cores": core_receipts,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_extraction_manifest(output_root=output_root, receipt=receipt, inputs=inputs)
    del model, batches
    gc.collect()
    return receipt


def load_all_contextual_cores(
    *, output_root: Path, extraction_receipt: Mapping[str, Any]
) -> tuple[SO2ContextualCore, ...]:
    records = extraction_receipt.get("cores")
    if not isinstance(records, list) or tuple(
        int(record.get("core_number", -1))
        for record in records
        if isinstance(record, Mapping)
    ) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError("Extraction does not contain all ordered SO2 cores.")
    cores = tuple(
        load_contextual_core(
            output_root / str(record["embedding_file"]),
            alias=str(record["alias"]),
            core_number=int(record["core_number"]),
            expected_cells=int(record["cell_count"]),
        )
        for record in records
        if isinstance(record, Mapping)
    )
    if tuple(core.alias for core in cores) != SO2_ALIASES:
        raise SO2HLClusteringError("All 14 SO2 cores are not present in order.")
    return cores


def concatenate_contextual_embeddings(
    cores: Sequence[SO2ContextualCore],
) -> np.ndarray:
    if tuple(core.core_number for core in cores) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError(
            "Joint hL clustering requires all 14 cores in locked order."
        )
    widths = {int(core.hL.shape[1]) for core in cores}
    if len(widths) != 1:
        raise SO2HLClusteringError("hL widths differ across SO2 cores.")
    combined = np.ascontiguousarray(
        np.concatenate([core.hL for core in cores], axis=0), dtype=np.float32
    )
    if combined.shape[0] != EXPECTED_TOTAL_CELLS:
        raise SO2HLClusteringError("Joint hL array dropped cells.")
    if not np.isfinite(combined).all():
        raise SO2HLClusteringError("Joint hL array contains non-finite values.")
    if not np.any(np.var(combined.astype(np.float64), axis=0) > 0.0):
        raise SO2HLClusteringError("Joint hL array has zero variance.")
    return combined


def build_cell_index_frame(
    cores: Sequence[SO2ContextualCore],
) -> pd.DataFrame:
    """Build the canonical one-row-per-cell alignment table."""

    if tuple(core.core_number for core in cores) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError("Cell table requires all 14 ordered SO2 cores.")
    frames: list[pd.DataFrame] = []
    global_start = 0
    for core in cores:
        expected_index = np.arange(core.n_cells, dtype=np.int64)
        if not np.array_equal(core.cell_index, expected_index):
            raise SO2HLClusteringError(f"Cell order changed for {core.alias}.")
        frames.append(
            pd.DataFrame(
                {
                    "global_cell_index": np.arange(
                        global_start,
                        global_start + core.n_cells,
                        dtype=np.int64,
                    ),
                    "cell_index": core.cell_index,
                    "cell_key": [
                        f"{core.alias}:{int(index):08d}" for index in core.cell_index
                    ],
                    "core_alias": core.alias,
                    "core_number": np.full(
                        core.n_cells, core.core_number, dtype=np.int16
                    ),
                    "x_um": core.coordinates_um[:, 0],
                    "y_um": core.coordinates_um[:, 1],
                }
            )
        )
        global_start += core.n_cells
    result = pd.concat(frames, ignore_index=True)
    expected_total = sum(core.n_cells for core in cores)
    if any(
        (
            len(result) != expected_total,
            bool(result["cell_key"].duplicated().any()),
            not np.array_equal(
                result["global_cell_index"].to_numpy(),
                np.arange(expected_total, dtype=np.int64),
            ),
            tuple(result["core_number"].drop_duplicates().tolist())
            != SO2_CORE_NUMBERS,
            not np.isfinite(result[["x_um", "y_um"]].to_numpy()).all(),
        )
    ):
        raise SO2HLClusteringError("SO2 cell alignment table is invalid.")
    return result


def cluster_summary_tables(
    labels: np.ndarray,
    cell_frame: pd.DataFrame,
    *,
    prefix: str = LABEL_PREFIX,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Summarize a joint partition and flag >90%-single-core clusters."""

    memberships = np.asarray(labels, dtype=np.int64)
    if memberships.shape != (len(cell_frame),) or np.any(memberships < 0):
        raise SO2HLClusteringError("Cluster labels do not align with cells.")
    unique = np.unique(memberships)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise SO2HLClusteringError("Cluster labels must be contiguous from zero.")
    core_values = cell_frame["core_number"].to_numpy(dtype=np.int64)
    if tuple(pd.unique(core_values).tolist()) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError("Cluster summary does not contain all 14 cores.")
    core_totals = {
        core: int(np.count_nonzero(core_values == core)) for core in SO2_CORE_NUMBERS
    }
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    dominated: list[str] = []
    for cluster_number in unique:
        selected = memberships == cluster_number
        size = int(np.count_nonzero(selected))
        label = f"{prefix}{int(cluster_number)}"
        counts = {
            core: int(np.count_nonzero(selected & (core_values == core)))
            for core in SO2_CORE_NUMBERS
        }
        dominant_core = min(
            SO2_CORE_NUMBERS,
            key=lambda core: (-counts[core], SO2_CORE_NUMBERS.index(core)),
        )
        dominant_proportion = counts[dominant_core] / size
        is_dominated = bool(dominant_proportion > 0.90)
        if is_dominated:
            dominated.append(label)
        summary_rows.append(
            {
                "cluster": label,
                "cluster_number": int(cluster_number),
                "size": size,
                "proportion": size / len(memberships),
                "dominant_core": int(dominant_core),
                "dominant_core_count": counts[dominant_core],
                "dominant_core_proportion": dominant_proportion,
                "core_dominated_gt_90pct": is_dominated,
            }
        )
        for core in SO2_CORE_NUMBERS:
            composition_rows.append(
                {
                    "cluster": label,
                    "cluster_number": int(cluster_number),
                    "core_number": int(core),
                    "cell_count": counts[core],
                    "proportion_within_cluster": counts[core] / size,
                    "proportion_within_core": counts[core] / core_totals[core],
                    "cluster_size": size,
                    "core_dominated_gt_90pct": is_dominated,
                }
            )
    summary = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(composition_rows)
    if int(summary["size"].sum()) != len(memberships):
        raise SO2HLClusteringError("Cluster summary dropped cells.")
    return summary, composition, dominated


def _clustering_configuration(
    *,
    n_neighbors: int,
    leiden_resolution: float,
    pca_components: int,
    random_seed: int,
) -> dict[str, Any]:
    if isinstance(n_neighbors, bool) or int(n_neighbors) <= 0:
        raise SO2HLClusteringError("n_neighbors must be positive.")
    if isinstance(pca_components, bool) or int(pca_components) <= 0:
        raise SO2HLClusteringError("pca_components must be positive.")
    if not math.isfinite(float(leiden_resolution)) or float(leiden_resolution) <= 0:
        raise SO2HLClusteringError("Leiden resolution must be positive.")
    return {
        "representation": "hL_final_graph_pre_decoder",
        "joint_core_order": list(SO2_CORE_NUMBERS),
        "joint_cell_count": EXPECTED_TOTAL_CELLS,
        "mean_center": True,
        "pca_components_requested": int(pca_components),
        "l2_normalize_after_pca": True,
        "n_neighbors": int(n_neighbors),
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union",
        "leiden_resolution": float(leiden_resolution),
        "random_seed": int(random_seed),
        "label_prefix": LABEL_PREFIX,
        "cluster_sort": "descending_size_then_minimum_global_cell_index_then_raw_id",
        "spatial_training_graph_reused_for_clustering": False,
        "cross_core_embedding_neighbors_permitted": True,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }


def _verify_clustering_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    run_id: str,
    extraction_manifest_sha256: str,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO2 clustering manifest")
    if any(
        (
            receipt.get("schema") != CLUSTERING_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != run_id,
            receipt.get("extraction_manifest_sha256")
            != extraction_manifest_sha256,
            receipt.get("configuration") != dict(configuration),
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HLClusteringError("SO2 clustering manifest identity is invalid.")
    files = receipt.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HLClusteringError("SO2 clustering manifest lacks output files.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLClusteringError(f"Clustering output is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLClusteringError(
                f"Clustering output checksum changed: {relative}"
            )
    labels = np.load(
        output_root / "clustering" / "contextual_labels.npy",
        allow_pickle=False,
    )
    if labels.shape != (EXPECTED_TOTAL_CELLS,) or _array_sha256(
        "sorted_leiden_labels", labels
    ) != receipt.get("pipeline", {}).get("leiden", {}).get("labels_sha256"):
        raise SO2HLClusteringError("Stored contextual labels are invalid.")
    table = pd.read_parquet(output_root / "tables" / "cell_contextual_clusters.parquet")
    if len(table) != EXPECTED_TOTAL_CELLS or tuple(
        table["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError("Stored contextual cell table is incomplete.")
    if not np.array_equal(
        table["contextual_cluster_number"].to_numpy(dtype=np.int64), labels
    ):
        raise SO2HLClusteringError("Stored labels and cell table are misaligned.")


def cluster_joint_contextual_embeddings(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    extraction_receipt: Mapping[str, Any],
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Mapping[str, Any]:
    """Run or verify one joint hL PCA/cosine-kNN/Leiden pipeline."""

    configuration = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    extraction_manifest_path = output_root / "embeddings" / "extraction_manifest.json"
    extraction_sha = sha256_file(extraction_manifest_path)
    receipt_path = output_root / "clustering" / "clustering_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO2 clustering manifest")
        _verify_clustering_manifest(
            output_root=output_root,
            receipt=receipt,
            run_id=inputs.run_id,
            extraction_manifest_sha256=extraction_sha,
            configuration=configuration,
        )
        return receipt
    clustering_dir = output_root / "clustering"
    tables_dir = output_root / "tables"
    if (clustering_dir.exists() and any(clustering_dir.iterdir())) or (
        tables_dir.exists() and any(tables_dir.iterdir())
    ):
        raise SO2HLClusteringError(
            "Partial clustering outputs exist without a complete receipt."
        )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cores = load_all_contextual_cores(
        output_root=output_root, extraction_receipt=extraction_receipt
    )
    cell_frame = build_cell_index_frame(cores)
    combined = concatenate_contextual_embeddings(cores)
    pca = deterministic_pca(combined, n_components=pca_components)
    knn = build_faiss_cosine_knn_graph(
        pca.normalized_scores,
        n_neighbors=n_neighbors,
        random_seed=random_seed,
    )
    leiden = run_seeded_leiden(
        knn,
        n_cells=len(combined),
        resolution=leiden_resolution,
        random_seed=random_seed,
    )
    labels = np.ascontiguousarray(leiden.labels, dtype=np.int64)
    cell_frame["contextual_cluster_number"] = labels.astype(np.int32)
    cell_frame["contextual_cluster"] = [f"C{int(value)}" for value in labels]
    summary, composition, dominated = cluster_summary_tables(labels, cell_frame)
    palette = deterministic_glasbey_palette(len(summary), namespace="contextual")

    labels_path = clustering_dir / "contextual_labels.npy"
    pca_path = clustering_dir / "contextual_pca_l2_normalized.npy"
    edges_path = clustering_dir / "contextual_knn_undirected_edges.npy"
    parameters_path = clustering_dir / "contextual_clustering_parameters.json"
    palette_path = clustering_dir / "contextual_palette.json"
    table_path = tables_dir / "cell_contextual_clusters.parquet"
    summary_path = tables_dir / "contextual_cluster_summary.csv"
    composition_path = tables_dir / "contextual_cluster_core_composition.csv"
    _atomic_write_npy(labels_path, labels)
    _atomic_write_npy(pca_path, pca.normalized_scores)
    _atomic_write_npy(edges_path, knn.edge_pairs)
    pipeline = {
        "representation": "contextual",
        "embedding_array": "hL",
        "joint_embedding_shape": list(combined.shape),
        "joint_embedding_sha256": _array_sha256("joint_contextual_hL", combined),
        "pca": pca.receipt,
        "knn": knn.receipt,
        "leiden": leiden.receipt,
    }
    _atomic_write_json(parameters_path, pipeline)
    _atomic_write_json(
        palette_path,
        {
            "representation": "contextual_hL",
            "label_prefix": LABEL_PREFIX,
            "method": "deterministic_greedy_farthest_point_CIELAB",
            "colors": palette,
        },
    )
    _atomic_write_parquet(
        table_path,
        cell_frame.loc[
            :,
            [
                "global_cell_index",
                "cell_index",
                "cell_key",
                "core_alias",
                "core_number",
                "x_um",
                "y_um",
                "contextual_cluster_number",
                "contextual_cluster",
            ],
        ],
    )
    _atomic_write_csv(summary_path, summary)
    _atomic_write_csv(composition_path, composition)
    stage_paths = (
        labels_path,
        pca_path,
        edges_path,
        parameters_path,
        palette_path,
        table_path,
        summary_path,
        composition_path,
    )
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "extraction_manifest_sha256": extraction_sha,
            "configuration": configuration,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "cluster_count": int(len(summary)),
            "cluster_size_range": [
                int(summary["size"].min()),
                int(summary["size"].max()),
            ],
            "core_dominated_gt_90pct": dominated,
            "pipeline": pipeline,
            "palette": palette,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in stage_paths
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_clustering_manifest(
        output_root=output_root,
        receipt=receipt,
        run_id=inputs.run_id,
        extraction_manifest_sha256=extraction_sha,
        configuration=configuration,
    )
    return receipt


def _cluster_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: int(item[0][1:])
        )
    ]


def _render_combined_map(
    frame: pd.DataFrame,
    *,
    palette: Mapping[str, str],
    resolution: float,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    tissue_axes = axes.ravel()[: len(SO2_CORE_NUMBERS)]
    for axis, core_number in zip(tissue_axes, SO2_CORE_NUMBERS, strict=True):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected["contextual_cluster"].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS_BY_CORE[core_number]:
            raise SO2HLClusteringError(f"Spatial panel is incomplete for core {core_number}.")
        if colors.isna().any():
            raise SO2HLClusteringError("Spatial map palette is incomplete.")
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
    handles = _cluster_legend_handles(palette)
    legend_axis.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=2 if len(handles) > 14 else 1,
        title="Joint model-derived\ncontextual cluster",
        fontsize=8,
        title_fontsize=10,
        borderaxespad=0.0,
    )
    figure.suptitle(
        "SO2 contextualized hL Leiden clusters — joint 14-core clustering\n"
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
    _atomic_save_figure_pair(
        figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi
    )
    plt.close(figure)


def _verify_figure_receipt(
    *, output_root: Path, receipt: Mapping[str, Any], clustering_sha256: str
) -> None:
    _verify_self_hash(receipt, label="SO2 figure receipt")
    if any(
        (
            receipt.get("schema") != FIGURE_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("clustering_manifest_sha256") != clustering_sha256,
            tuple(receipt.get("plot_specification", {}).get("panel_order", ()))
            != SO2_CORE_NUMBERS,
            receipt.get("plot_specification", {}).get("grid_shape") != [3, 5],
        )
    ):
        raise SO2HLClusteringError("SO2 figure receipt identity is invalid.")
    files = receipt.get("files")
    if not isinstance(files, Mapping) or len(files) != 2:
        raise SO2HLClusteringError("SO2 combined figure pair is incomplete.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLClusteringError(f"SO2 spatial figure is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLClusteringError(f"SO2 figure checksum changed: {relative}")


def render_spatial_map(
    *,
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    resolution: float,
    dpi: int = 300,
) -> Mapping[str, Any]:
    """Render or verify the combined 3x5 spatial hL-cluster map."""

    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2HLClusteringError("Figure DPI must be at least 72.")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

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
    clustering_path = output_root / "clustering" / "clustering_manifest.json"
    clustering_sha = sha256_file(clustering_path)
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO2 figure receipt")
        _verify_figure_receipt(
            output_root=output_root,
            receipt=receipt,
            clustering_sha256=clustering_sha,
        )
        return receipt
    frame = pd.read_parquet(
        output_root / "tables" / "cell_contextual_clusters.parquet"
    )
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2HLClusteringError("Figure table lacks all 14 ordered cores.")
    palette = clustering_receipt.get("palette")
    if not isinstance(palette, Mapping):
        raise SO2HLClusteringError("Clustering receipt lacks the contextual palette.")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "contextual_leiden_resolution_1p0_spatial_14cores.png"
    pdf_path = figure_dir / "contextual_leiden_resolution_1p0_spatial_14cores.pdf"
    _render_combined_map(
        frame,
        palette=dict(palette),
        resolution=resolution,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=int(dpi),
    )
    specification = spatial_plot_spec(dict(palette))
    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "clustering_manifest_sha256": clustering_sha,
            "leiden_resolution": float(resolution),
            "dpi": int(dpi),
            "one_dot_per_cell": True,
            "point_count": EXPECTED_TOTAL_CELLS,
            "point_layer_rasterized_in_pdf": True,
            "marker_borders": False,
            "lines_between_cells": False,
            "plot_specification": specification,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in (png_path, pdf_path)
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_figure_receipt(
        output_root=output_root,
        receipt=receipt,
        clustering_sha256=clustering_sha,
    )
    return receipt


def _source_artifact_records(inputs: SO2ResolvedInputs) -> dict[str, Any]:
    records: dict[str, Any] = {
        "checkpoint": {
            "path": inputs.checkpoint_path.as_posix(),
            "sha256": inputs.checkpoint_sha256,
            "size_bytes": int(inputs.checkpoint_path.stat().st_size),
        },
        "cohort_manifest": inputs.provenance["cohort_manifest"],
        "graph_manifest": inputs.provenance["graph_manifest"],
        "source_cores": {},
    }
    for core in _expected_core_records(inputs):
        alias = str(core["alias"])
        records["source_cores"][alias] = {
            "core_number": int(core["core_number"]),
            "cell_count": int(core["cell_count"]),
            "prepared_core_artifact_sha256": core[
                "prepared_core_artifact_sha256"
            ],
            "prepared_component_checksums": core[
                "prepared_component_checksums"
            ],
            "graph_record_sha256": core["graph_record_sha256"],
            "graph_logical_sha256": core["graph_logical_sha256"],
            "graph_file_checksums": core["graph_file_checksums"],
        }
    return records


def _render_readme(
    *,
    inputs: SO2ResolvedInputs,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
) -> str:
    counts = "\n".join(
        f"- SO2 Core {number}: {EXPECTED_CELL_COUNTS_BY_CORE[number]:,} cells"
        for number in SO2_CORE_NUMBERS
    )
    dominated = clustering.get("core_dominated_gt_90pct", [])
    dominated_text = ", ".join(str(value) for value in dominated) or "None"
    model = inputs.checkpoint_payload["model_construction"]
    reproduction = (
        'CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python '
        "-m spatial_benchmark analyze-so2-hl-clusters "
        f"--run-id {inputs.run_id} --n-neighbors 30 "
        "--leiden-resolution 1.0 --pca-components 50 "
        "--random-seed 20260825 --device cpu --cpu-threads 40"
    )
    return f"""# SO2 14-core contextual hL clustering

This report is a CPU-only, post-training readout of the completed locked run
`{inputs.run_id}`. No model retraining occurred, and no GPU was used by the
analysis command.

## Representation and method

`hL` is the final Relative-Geometric QKV graph-layer output immediately before
the expression decoder. All {EXPECTED_TOTAL_CELLS:,} cells were clustered
jointly, so a `C` cluster ID has the same model-derived meaning in every panel.
The workflow mean-centered hL, retained up to 50 exact PCA components,
L2-normalized the PCA scores, built a sparse cosine 30-nearest-neighbor graph,
and ran seeded Leiden at resolution 1.0 (seed 20260825). The original spatial
training graph was not reused as the clustering graph, and no dense cell-by-cell
distance matrix was constructed.

PCA + kNN + Leiden is a standard exploratory clustering workflow in single-cell
analysis. This particular input is a learned graph-contextual representation
rather than a conventional expression matrix, so its clusters are not cell-type
annotations without separate marker and pathological validation.

## Inputs

- Checkpoint: `{inputs.checkpoint_path}`
- Checkpoint SHA-256: `{inputs.checkpoint_sha256}`
- Model seed: `{inputs.checkpoint_payload['model_seed']}`
- Hidden width / graph layers: `{model['hidden_dim']}` / `{model['graph_layers']}`
- hL shape: `[{extraction['total_cells']}, {extraction['embedding_dimension']}]`
- Joint cluster count: `{clustering['cluster_count']}`
- Cluster-size range: `{clustering['cluster_size_range'][0]:,}` to `{clustering['cluster_size_range'][1]:,}` cells
- Clusters with >90% of cells from one core: {dominated_text}

## Core coverage

{counts}

## Interpretation limits

Contextual clusters represent patterns after graph-based neighborhood
processing by this trained model. They do not independently establish cell
type, signaling, biological influence, or causality. No biological names were
assigned. Marker-based and pathological validation will be conducted
separately. The SO2 cohort is fit-only/transductive, and its tissue context is
not asserted uniformly across all 14 cores.

## Reproduction

Run from the repository root:

```bash
{reproduction}
```

Extraction is resumable per core through checksum-verified hL artifacts. A
plotting retry reuses the completed extraction and clustering receipts.
"""


def _verify_final_manifest(output_root: Path, manifest: Mapping[str, Any]) -> None:
    _verify_self_hash(manifest, label="SO2 final analysis manifest")
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            float(manifest.get("analysis_parameters", {}).get(
                "leiden_resolution", math.nan
            ))
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2HLClusteringError("SO2 final manifest identity is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HLClusteringError("SO2 final manifest lacks file checksums.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLClusteringError(f"Final output is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLClusteringError(f"Final output checksum changed: {relative}")
    required = {
        "README.md",
        "embeddings/extraction_manifest.json",
        "clustering/clustering_manifest.json",
        "tables/cell_contextual_clusters.parquet",
        "tables/contextual_cluster_summary.csv",
        "tables/contextual_cluster_core_composition.csv",
        "figures/contextual_leiden_resolution_1p0_spatial_14cores.png",
        "figures/contextual_leiden_resolution_1p0_spatial_14cores.pdf",
    }
    required.update(
        f"embeddings/core_{number}_hL.npz" for number in SO2_CORE_NUMBERS
    )
    if not required.issubset(files):
        missing = sorted(required.difference(files))
        raise SO2HLClusteringError(f"Required SO2 outputs are absent: {missing}")


def _finalize_analysis(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
    analysis_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    readme_path = output_root / "README.md"
    _atomic_write_text(
        readme_path,
        _render_readme(inputs=inputs, extraction=extraction, clustering=clustering),
    )
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_scope": "contextual_hL_only",
            "run_id": inputs.run_id,
            "campaign_id": CAMPAIGN_ID,
            "model_seed": EXPECTED_MODEL_SEED,
            "checkpoint": inputs.provenance["checkpoint"],
            "model_construction": inputs.provenance["model_construction"],
            "core_order": list(SO2_CORE_NUMBERS),
            "core_cell_counts": {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO2_CORE_NUMBERS
            },
            "total_cells": EXPECTED_TOTAL_CELLS,
            "hL_shape": [
                EXPECTED_TOTAL_CELLS,
                int(extraction["embedding_dimension"]),
            ],
            "analysis_parameters": dict(analysis_parameters),
            "cpu_only_execution": {
                "enforced_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_threads": int(extraction["cpu_threads"]),
                "gpu_tensor_or_model_created": False,
            },
            "input_artifacts": _source_artifact_records(inputs),
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
            "cluster_count": int(clustering["cluster_count"]),
            "cluster_size_range": clustering["cluster_size_range"],
            "core_dominated_gt_90pct": clustering[
                "core_dominated_gt_90pct"
            ],
            "figure_paths": sorted(figures["files"]),
            "interpretation": {
                "clusters_are_model_derived": True,
                "cell_types_established": False,
                "biological_influence_established": False,
                "causality_established": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": _file_manifest(output_root),
        }
    )
    manifest_path = output_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    _verify_final_manifest(output_root, manifest)
    return manifest


def run_so2_hl_clustering(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
    dpi: int = 300,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the locked, resumable CPU-only SO2 hL analysis."""

    validate_cpu_device(device)
    if not math.isclose(
        float(leiden_resolution), DEFAULT_LEIDEN_RESOLUTION, rel_tol=0.0, abs_tol=0.0
    ):
        raise SO2HLClusteringError(
            "This requested SO2 map is locked to Leiden resolution 1.0."
        )
    inputs = resolve_so2_analysis_inputs(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    if output_dir is None:
        output_root = (
            paths.report_root
            / "analyses"
            / "so2_14core_contextual_embedding_clustering"
            / inputs.run_id
        )
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    parameters = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    parameters.update({"cpu_threads": int(cpu_threads), "figure_dpi": int(dpi)})
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, label="SO2 final analysis manifest")
        _verify_final_manifest(output_root, manifest)
        if manifest.get("analysis_parameters") != parameters:
            raise SO2HLClusteringError(
                "Existing completed analysis used different parameters."
            )
    else:
        extraction = extract_contextual_embeddings(
            inputs=inputs,
            output_root=output_root,
            device=device,
            cpu_threads=cpu_threads,
        )
        clustering = cluster_joint_contextual_embeddings(
            inputs=inputs,
            output_root=output_root,
            extraction_receipt=extraction,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            pca_components=pca_components,
            random_seed=random_seed,
        )
        figures = render_spatial_map(
            output_root=output_root,
            clustering_receipt=clustering,
            resolution=leiden_resolution,
            dpi=dpi,
        )
        manifest = _finalize_analysis(
            inputs=inputs,
            output_root=output_root,
            extraction=extraction,
            clustering=clustering,
            figures=figures,
            analysis_parameters=parameters,
        )
    png = output_root / "figures" / "contextual_leiden_resolution_1p0_spatial_14cores.png"
    pdf = output_root / "figures" / "contextual_leiden_resolution_1p0_spatial_14cores.pdf"
    return {
        "status": "complete",
        "run_id": inputs.run_id,
        "device": "cpu",
        "output_root": output_root.as_posix(),
        "total_cells": int(manifest["total_cells"]),
        "hL_shape": manifest["hL_shape"],
        "cluster_count": int(manifest["cluster_count"]),
        "cluster_size_range": manifest["cluster_size_range"],
        "core_dominated_gt_90pct": manifest["core_dominated_gt_90pct"],
        "combined_png": png.as_posix(),
        "combined_pdf": pdf.as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "SO2ContextualCore",
    "SO2HLClusteringError",
    "build_cell_index_frame",
    "cluster_joint_contextual_embeddings",
    "cluster_summary_tables",
    "extract_contextual_embeddings",
    "extract_full_contextual_embedding",
    "requested_panel_order",
    "resolve_so2_analysis_inputs",
    "run_so2_hl_clustering",
    "spatial_plot_spec",
    "validate_cpu_device",
]

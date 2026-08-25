"""Bounded post-training readouts for the six-core relative-QKV model.

This module loads the runner's standalone checkpoint directly, reconstructs the
prepared complete-core batches, and replays the runner's deterministic held-in
mask.  Attention export is receiver-streamed; selected derivatives use one
autograd injection variable per requested source entry and never materialize an
exhaustive edge-by-gene or node-by-gene Jacobian.

The returned quantities are computational routing and local model sensitivity.
They are not biological importance, patient replication, or causal effects.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .adjacency_ablation import sample_uniform_mask_numpy
from .cancer_pooled_full_core import CANCER_ALIASES
from .cancer_relative_graphs import load_cancer_relative_qkv_batches
from .fingerprints import sha256_file
from .masking import derive_mask_seed
from .pooled_relative_qkv_training import (
    PooledRelativeQKVCoreBatch,
    _tree_sha256,
)
from .relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    _resolve_explanation_layer,
)


CAMPAIGN_ID = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
CHECKPOINT_SCHEMA = "cancer_6core_relative_qkv_resume_v1"
FIXED_INFERENCE_MASK_BASE_SEED = 2026082491
FIXED_INFERENCE_MASK_NAMESPACE = "relative-qkv-held-in-fit-diagnostic"


class RelativeQKVPostTrainingError(RuntimeError):
    """Raised when post-training inputs or requested readouts are invalid."""


@dataclass(frozen=True, slots=True)
class FixedInferenceMask:
    """One deterministic runner-compatible fixed inference mask."""

    alias: str
    seed: int
    checksum_sha256: str
    mask: np.ndarray
    masked_gene_counts: np.ndarray

    @property
    def n_masked_entries(self) -> int:
        return int(self.masked_gene_counts.sum())


@dataclass(frozen=True, slots=True)
class LoadedRelativeQKVCheckpoint:
    """A verified standalone runner checkpoint and reconstructed model."""

    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer
    payload: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class SelectedDerivativeRequest:
    """One direct-edge attention and receiver-prediction derivative request."""

    request_id: str
    core_alias: str
    source_node: int
    source_feature: int
    receiver_node: int
    target_feature: int
    attention_head: int | None = None
    layer: int = -1


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelativeQKVPostTrainingError(f"{location} must be a mapping.")
    return value


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RelativeQKVPostTrainingError(
            f"Cannot load standalone runner checkpoint: {path}."
        ) from exc
    return dict(_mapping(value, "checkpoint"))


def fixed_inference_mask(
    batch: PooledRelativeQKVCoreBatch,
    *,
    mask_base_seed: int = FIXED_INFERENCE_MASK_BASE_SEED,
) -> FixedInferenceMask:
    """Recreate the exact deterministic mask used by runner diagnostics."""

    seed = derive_mask_seed(
        int(mask_base_seed),
        FIXED_INFERENCE_MASK_NAMESPACE,
        batch.alias,
    )
    realization = sample_uniform_mask_numpy(
        batch.n_nodes,
        batch.n_genes,
        seed=seed,
    )
    retry = 0
    while int(realization.masked_gene_counts.sum()) == 0:
        retry += 1
        seed = derive_mask_seed(seed, "nonzero-held-in", retry)
        realization = sample_uniform_mask_numpy(
            batch.n_nodes,
            batch.n_genes,
            seed=seed,
        )
    mask = np.ascontiguousarray(realization.mask, dtype=np.bool_)
    counts = np.ascontiguousarray(realization.masked_gene_counts, dtype=np.int64)
    mask.setflags(write=False)
    counts.setflags(write=False)
    return FixedInferenceMask(
        alias=batch.alias,
        seed=int(seed),
        checksum_sha256=str(realization.checksum),
        mask=mask,
        masked_gene_counts=counts,
    )


def load_prepared_relative_qkv_batches(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
) -> tuple[PooledRelativeQKVCoreBatch, ...]:
    """Load and checksum-verify the six prepared complete-core batches."""

    batches = tuple(
        load_cancer_relative_qkv_batches(
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )
    )
    if tuple(batch.alias for batch in batches) != CANCER_ALIASES:
        raise RelativeQKVPostTrainingError(
            "Prepared batches do not match the locked six-core order."
        )
    return batches


def _construction_integer(
    construction: Mapping[str, Any], name: str, expected: int
) -> int:
    try:
        observed = int(construction[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise RelativeQKVPostTrainingError(
            f"checkpoint.model_construction.{name} is invalid."
        ) from exc
    if observed != int(expected):
        raise RelativeQKVPostTrainingError(
            f"Checkpoint {name}={observed} does not match prepared data "
            f"({expected})."
        )
    return observed


def load_relative_qkv_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_genes: int,
    node_covariate_dim: int,
    device: str | torch.device = "cpu",
    receiver_chunk_size: int | None = None,
    max_edges_per_chunk: int | None = None,
) -> LoadedRelativeQKVCheckpoint:
    """Verify and load a runner checkpoint without requiring its run bundle."""

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = _load_checkpoint_payload(path)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise RelativeQKVPostTrainingError("Unsupported checkpoint schema.")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise RelativeQKVPostTrainingError(
            "Checkpoint belongs to a different campaign."
        )
    state = _mapping(payload.get("model_state_dict"), "model_state_dict")
    expected_state_checksum = payload.get("model_state_checksum")
    observed_state_checksum = _tree_sha256(state)
    if expected_state_checksum != observed_state_checksum:
        raise RelativeQKVPostTrainingError("Checkpoint model-state checksum mismatch.")
    construction = _mapping(
        payload.get("model_construction"), "model_construction"
    )
    implementation = str(construction.get("class", ""))
    if implementation not in {
        "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
        "ReceiverChunkedRelativeQKVGraphTransformer",
    }:
        raise RelativeQKVPostTrainingError(
            "Checkpoint was not created by the receiver-chunked relative-QKV model."
        )
    _construction_integer(construction, "num_genes", num_genes)
    _construction_integer(
        construction, "node_covariate_dim", node_covariate_dim
    )

    def integer(name: str) -> int:
        try:
            return int(construction[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RelativeQKVPostTrainingError(
                f"checkpoint.model_construction.{name} is invalid."
            ) from exc

    resolved_receiver_chunk_size = (
        integer("receiver_chunk_size")
        if receiver_chunk_size is None
        else int(receiver_chunk_size)
    )
    resolved_max_edges = (
        integer("max_edges_per_chunk")
        if max_edges_per_chunk is None
        else int(max_edges_per_chunk)
    )
    model = ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=int(num_genes),
        node_covariate_dim=int(node_covariate_dim),
        hidden_dim=integer("hidden_dim"),
        attention_heads=integer("attention_heads"),
        attention_head_dim=integer("attention_head_dim"),
        graph_layers=integer("graph_layers"),
        ffn_dim=integer("ffn_dim"),
        decoder_dim=integer("decoder_dim"),
        positional_bias_hidden_dim=integer("positional_bias_hidden_dim"),
        dropout=float(construction["dropout"]),
        attention_dropout=float(construction["attention_dropout"]),
        relative_geometry_dim=integer("relative_geometry_dim"),
        receiver_chunk_size=resolved_receiver_chunk_size,
        max_edges_per_chunk=resolved_max_edges,
        activation_checkpointing=bool(construction["activation_checkpointing"]),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RelativeQKVPostTrainingError(
            "Checkpoint state does not match its model construction receipt."
        ) from exc
    if _tree_sha256(model.state_dict()) != observed_state_checksum:
        raise RelativeQKVPostTrainingError(
            "Loaded model state differs from the checkpoint state."
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RelativeQKVPostTrainingError("CUDA was requested but is unavailable.")
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedRelativeQKVCheckpoint(
        model=model,
        payload=payload,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
    )


def load_core_coordinates_and_genes(
    cohort_dir: str | Path,
    *,
    alias: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load plotting-only coordinates and the ordered biological gene schema."""

    canonical_alias = str(alias).strip().upper()
    if canonical_alias not in CANCER_ALIASES:
        raise RelativeQKVPostTrainingError("Unknown Cancer core alias.")
    root = Path(cohort_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        genes = tuple(manifest["features"]["gene_names"])
        with np.load(
            root / "cores" / f"{canonical_alias}.npz", allow_pickle=False
        ) as values:
            coordinates = np.asarray(values["coordinates_um"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RelativeQKVPostTrainingError(
            f"Cannot load coordinate/gene metadata for {canonical_alias}."
        ) from exc
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not np.isfinite(coordinates).all()
        or len(genes) != 1_000
        or len(set(genes)) != len(genes)
    ):
        raise RelativeQKVPostTrainingError(
            "Prepared coordinate or gene metadata has an invalid schema."
        )
    coordinates = np.ascontiguousarray(coordinates)
    coordinates.setflags(write=False)
    return coordinates, genes


def reciprocal_edge_ids(
    edge_index: np.ndarray | Tensor,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reciprocal edge IDs and canonical unordered integer pair keys.

    Prepared relative-QKV graphs are receiver-major then source-major, so their
    directed key is ``receiver * n_nodes + source``.  A vectorized search joins
    every edge to its reverse without a Python dictionary.
    """

    edges = (
        edge_index.detach().cpu().numpy()
        if isinstance(edge_index, Tensor)
        else np.asarray(edge_index)
    )
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or edges.dtype.kind not in "iu"
        or int(n_nodes) <= 0
    ):
        raise RelativeQKVPostTrainingError(
            "edge_index must be integral [2, edges] with a positive node count."
        )
    source = edges[0].astype(np.int64, copy=False)
    receiver = edges[1].astype(np.int64, copy=False)
    if len(source) == 0 or np.any(source == receiver):
        raise RelativeQKVPostTrainingError(
            "Reciprocal routing requires a non-empty loop-free graph."
        )
    directed = receiver * int(n_nodes) + source
    if np.any(directed[1:] <= directed[:-1]):
        raise RelativeQKVPostTrainingError(
            "Prepared edges must be unique receiver-major/source-major sorted."
        )
    reverse = source * int(n_nodes) + receiver
    reciprocal = np.searchsorted(directed, reverse)
    if np.any(reciprocal >= len(directed)) or not np.array_equal(
        directed[reciprocal], reverse
    ):
        raise RelativeQKVPostTrainingError(
            "Prepared graph is missing at least one reciprocal directed edge."
        )
    low = np.minimum(source, receiver)
    high = np.maximum(source, receiver)
    pair_key = low * int(n_nodes) + high
    return (
        reciprocal.astype(np.int64, copy=False),
        pair_key.astype(np.int64, copy=False),
    )


AttentionShardConsumer = Callable[
    [int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None
]
NodeEmbeddingConsumer = Callable[[np.ndarray], None]


def stream_receiver_attention(
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    batch: PooledRelativeQKVCoreBatch,
    inference_mask: np.ndarray | Tensor,
    *,
    layer: int = -1,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    consumer: AttentionShardConsumer,
    node_embedding_consumer: NodeEmbeddingConsumer | None = None,
) -> int:
    """Stream one layer's exact edge diagnostics in complete receiver shards.

    The callback receives ``(receiver_start, receiver_stop, edge_ids,
    attention, content, bias, combined)`` as CPU NumPy arrays.  The complete
    per-head edge matrix is never retained by this function.  If supplied,
    ``node_embedding_consumer`` receives the final full-core embedding once,
    after every graph layer has completed.
    """

    if model.training:
        raise RelativeQKVPostTrainingError("Attention export requires eval mode.")
    device = next(model.parameters()).device
    selected_layer = _resolve_explanation_layer(
        int(layer), graph_layers=model.graph_layers
    )
    mask = (
        torch.from_numpy(np.array(inference_mask, dtype=np.bool_, copy=True))
        if isinstance(inference_mask, np.ndarray)
        else torch.as_tensor(inference_mask, dtype=torch.bool)
    )
    if tuple(mask.shape) != tuple(batch.target_expression.shape):
        raise RelativeQKVPostTrainingError(
            "Inference mask must match the complete core expression shape."
        )
    if amp and device.type != "cuda":
        raise RelativeQKVPostTrainingError("AMP attention export requires CUDA.")
    autocast = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp
        else nullcontext()
    )
    edge_index = batch.edge_index
    relative_geometry = batch.relative_geometry
    with torch.no_grad(), autocast:
        expression = batch.target_expression.to(device=device)
        covariates = batch.node_covariates.to(device=device)
        device_mask = mask.to(device=device)
        node_embedding, _ = model._encode_and_targets(
            expression,
            device_mask,
            covariates,
            None,
        )
        layout = model._receiver_layout(edge_index, num_nodes=batch.n_nodes)
        for layer_number, block in enumerate(model.blocks):
            queries, keys, values = block.project_nodes(node_embedding)
            outputs: list[Tensor] = []
            for receiver_start, receiver_stop in model._receiver_ranges(
                layout, num_nodes=batch.n_nodes
            ):
                edge_start = layout.receiver_ptr[receiver_start]
                edge_stop = layout.receiver_ptr[receiver_stop]
                edge_ids = model._original_ids(
                    layout, edge_start=edge_start, edge_stop=edge_stop
                )
                chunk_edges = edge_index.index_select(1, edge_ids)
                source = chunk_edges[0].to(device=device, dtype=torch.long)
                receiver = chunk_edges[1].to(device=device, dtype=torch.long)
                geometry = relative_geometry.index_select(0, edge_ids).to(
                    device=device, dtype=node_embedding.dtype
                )
                result = model._attention_partition(
                    block,
                    queries[receiver_start:receiver_stop],
                    keys,
                    values,
                    node_embedding[receiver_start:receiver_stop],
                    source,
                    receiver,
                    geometry,
                    receiver_start=receiver_start,
                )
                output, attention, content, bias, combined = result
                outputs.append(output)
                if layer_number == selected_layer:
                    consumer(
                        receiver_start,
                        receiver_stop,
                        edge_ids.detach().cpu().numpy(),
                        attention.detach().float().cpu().numpy(),
                        content.detach().float().cpu().numpy(),
                        bias.detach().float().cpu().numpy(),
                        combined.detach().float().cpu().numpy(),
                    )
            node_embedding = torch.cat(outputs, dim=0)
        if node_embedding_consumer is not None:
            node_embedding_consumer(
                np.ascontiguousarray(
                    node_embedding.detach().float().cpu().numpy()
                )
            )
    return selected_layer


def _validated_requests(
    requests: Sequence[SelectedDerivativeRequest],
    *,
    batch: PooledRelativeQKVCoreBatch,
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
) -> tuple[SelectedDerivativeRequest, ...]:
    materialized = tuple(requests)
    if not materialized:
        raise RelativeQKVPostTrainingError("At least one derivative is required.")
    identifiers = [request.request_id for request in materialized]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise RelativeQKVPostTrainingError(
            "Derivative request IDs must be non-empty and unique."
        )
    layer_numbers = {
        _resolve_explanation_layer(request.layer, graph_layers=model.graph_layers)
        for request in materialized
    }
    if len(layer_numbers) != 1:
        raise RelativeQKVPostTrainingError(
            "One derivative replay may contain only one attention layer."
        )
    heads = model.blocks[next(iter(layer_numbers))].attention_heads
    for request in materialized:
        if request.core_alias != batch.alias:
            raise RelativeQKVPostTrainingError(
                "Derivative request core does not match the loaded batch."
            )
        if not 0 <= int(request.source_node) < batch.n_nodes or not 0 <= int(
            request.receiver_node
        ) < batch.n_nodes:
            raise RelativeQKVPostTrainingError(
                "Derivative request contains an out-of-range node."
            )
        if not 0 <= int(request.source_feature) < batch.n_genes or not 0 <= int(
            request.target_feature
        ) < batch.n_genes:
            raise RelativeQKVPostTrainingError(
                "Derivative request contains an out-of-range feature."
            )
        if request.source_node == request.receiver_node:
            raise RelativeQKVPostTrainingError(
                "Attention derivative requests cannot use a self edge."
            )
        if request.attention_head is not None and not 0 <= int(
            request.attention_head
        ) < heads:
            raise RelativeQKVPostTrainingError(
                "Derivative request contains an out-of-range attention head."
            )
    return materialized


def _scalar_gradient(
    scalar: Tensor,
    variables: Tensor,
    *,
    retain_graph: bool,
) -> Tensor:
    if not scalar.requires_grad:
        return torch.zeros_like(variables)
    gradient = torch.autograd.grad(
        scalar,
        variables,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    return torch.zeros_like(variables) if gradient is None else gradient


def selected_autograd_derivatives(
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    batch: PooledRelativeQKVCoreBatch,
    inference_mask: np.ndarray | Tensor,
    requests: Sequence[SelectedDerivativeRequest],
) -> list[dict[str, Any]]:
    """Compute selected attention and prediction derivatives in one replay.

    Attention is the requested head, or the head mean when ``attention_head``
    is ``None``.  Each derivative is with respect to the standardized-log1p
    model input at exactly ``(source_node, source_feature)``.  The fixed mask is
    reapplied inside the model; a masked source therefore has exactly zero local
    sensitivity by construction.
    """

    if model.training:
        raise RelativeQKVPostTrainingError("Derivative replay requires eval mode.")
    materialized = _validated_requests(requests, batch=batch, model=model)
    mask = (
        torch.from_numpy(np.array(inference_mask, dtype=np.bool_, copy=True))
        if isinstance(inference_mask, np.ndarray)
        else torch.as_tensor(inference_mask, dtype=torch.bool)
    )
    if tuple(mask.shape) != tuple(batch.target_expression.shape):
        raise RelativeQKVPostTrainingError(
            "Inference mask must match the complete core expression shape."
        )
    device = next(model.parameters()).device
    device_mask = mask.to(device=device)
    base_expression = batch.target_expression.to(device=device)
    covariates = batch.node_covariates.to(device=device)

    source_pairs = tuple(
        dict.fromkeys(
            (int(request.source_node), int(request.source_feature))
            for request in materialized
        )
    )
    source_position = {pair: index for index, pair in enumerate(source_pairs)}
    injections = torch.zeros(
        len(source_pairs),
        dtype=base_expression.dtype,
        device=device,
        requires_grad=True,
    )
    flat_indices = torch.as_tensor(
        [node * batch.n_genes + feature for node, feature in source_pairs],
        dtype=torch.long,
        device=device,
    )
    shifted_flat = base_expression.reshape(-1).index_add(
        0, flat_indices, injections
    )
    shifted_expression = shifted_flat.view_as(base_expression)

    receiver_nodes = tuple(
        dict.fromkeys(int(request.receiver_node) for request in materialized)
    )
    receiver_position = {
        node: index for index, node in enumerate(receiver_nodes)
    }
    selected_layer = _resolve_explanation_layer(
        materialized[0].layer, graph_layers=model.graph_layers
    )
    output = model(
        input_expression=shifted_expression,
        gene_mask=device_mask,
        edge_index=batch.edge_index,
        relative_geometry=batch.relative_geometry,
        node_covariates=covariates,
        return_explanations=True,
        target_nodes=torch.as_tensor(
            receiver_nodes, dtype=torch.long, device=device
        ),
        attention_receivers=torch.as_tensor(
            receiver_nodes, dtype=torch.long, device=batch.edge_index.device
        ),
        explanation_layer=selected_layer,
    )
    if output.edge_index is None or output.attention_weights is None:
        raise RelativeQKVPostTrainingError(
            "Model did not return requested attention diagnostics."
        )
    diagnostic_edges = output.edge_index.detach().cpu().numpy()
    diagnostic_codes = (
        diagnostic_edges[1].astype(np.int64) * batch.n_nodes
        + diagnostic_edges[0].astype(np.int64)
    )
    if np.any(diagnostic_codes[1:] <= diagnostic_codes[:-1]):
        raise RelativeQKVPostTrainingError(
            "Selected diagnostic edges are not in canonical order."
        )

    rows: list[dict[str, Any]] = []
    for request_index, request in enumerate(materialized):
        edge_code = int(request.receiver_node) * batch.n_nodes + int(
            request.source_node
        )
        edge_position = int(np.searchsorted(diagnostic_codes, edge_code))
        if (
            edge_position >= len(diagnostic_codes)
            or int(diagnostic_codes[edge_position]) != edge_code
        ):
            raise RelativeQKVPostTrainingError(
                f"Request {request.request_id!r} is not a directed graph edge."
            )
        edge_attention = output.attention_weights[edge_position]
        attention_scalar = (
            edge_attention.mean()
            if request.attention_head is None
            else edge_attention[int(request.attention_head)]
        )
        prediction_scalar = output.prediction[
            receiver_position[int(request.receiver_node)],
            int(request.target_feature),
        ]
        attention_gradient = _scalar_gradient(
            attention_scalar, injections, retain_graph=True
        )
        is_last = request_index == len(materialized) - 1
        prediction_gradient = _scalar_gradient(
            prediction_scalar, injections, retain_graph=not is_last
        )
        source_index = source_position[
            (int(request.source_node), int(request.source_feature))
        ]
        derivative_attention = float(attention_gradient[source_index].detach().cpu())
        derivative_prediction = float(
            prediction_gradient[source_index].detach().cpu()
        )
        values = (
            float(attention_scalar.detach().cpu()),
            float(prediction_scalar.detach().cpu()),
            derivative_attention,
            derivative_prediction,
        )
        if not all(np.isfinite(value) for value in values):
            raise RelativeQKVPostTrainingError(
                "Selected autograd replay produced a non-finite value."
            )
        rows.append(
            {
                "request_id": request.request_id,
                "core_alias": request.core_alias,
                "layer_number": selected_layer,
                "source_node": int(request.source_node),
                "source_feature_index": int(request.source_feature),
                "receiver_node": int(request.receiver_node),
                "target_feature_index": int(request.target_feature),
                "attention_head": (
                    "mean"
                    if request.attention_head is None
                    else str(int(request.attention_head))
                ),
                "source_feature_observed": not bool(
                    mask[int(request.source_node), int(request.source_feature)]
                ),
                "target_feature_masked": bool(
                    mask[int(request.receiver_node), int(request.target_feature)]
                ),
                "attention_value": values[0],
                "prediction_value": values[1],
                "d_attention_d_source_feature": derivative_attention,
                "d_prediction_d_source_feature": derivative_prediction,
            }
        )
    return rows


def file_sha256(path: str | Path) -> str:
    """Small public checksum helper for post-training manifests."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CAMPAIGN_ID",
    "CHECKPOINT_SCHEMA",
    "FIXED_INFERENCE_MASK_BASE_SEED",
    "FIXED_INFERENCE_MASK_NAMESPACE",
    "FixedInferenceMask",
    "LoadedRelativeQKVCheckpoint",
    "RelativeQKVPostTrainingError",
    "SelectedDerivativeRequest",
    "file_sha256",
    "fixed_inference_mask",
    "load_core_coordinates_and_genes",
    "load_prepared_relative_qkv_batches",
    "load_relative_qkv_checkpoint",
    "reciprocal_edge_ids",
    "selected_autograd_derivatives",
    "stream_receiver_attention",
]

r"""Exact sparse QKV attention with relative geometry used as logit bias only.

For every supplied directed edge ``j -> i`` and attention head ``h`` this
module evaluates

.. math::

    c_{ijh} = Q_h(x_i)^T K_h(x_j) / \sqrt{d_h},

.. math::

    b_{ijh} = f_h(\rho_{ij}), \qquad
    \alpha_{ijh} = \operatorname{softmax}_{j \in N(i)}(c_{ijh} + b_{ijh}),

.. math::

    z_{ih} = \sum_{j \in N(i)} \alpha_{ijh} V_h(x_j).

``rho`` is an already-computed, edge-aligned 70-dimensional relative-geometry
tensor.  Geometry is never added to keys or values and never gates a message.
There are no implicit self loops, learned node/edge identifiers, or dense graph
materializations.

``RelativeGeometryQKVGraphTransformer`` is the transparent full-edge reference.
``ReceiverChunkedRelativeGeometryQKVGraphTransformer`` partitions contiguous
receiver ranges while retaining every incoming edge of each receiver.  In
particular, a CPU-resident ``edge_index`` and relative-geometry tensor stay on
the CPU; only the current exact shard is transferred to the model device.
Chunking therefore changes execution memory, not the attention operator.

When requested, diagnostics are restored to the supplied edge order and expose
attention, content logits, positional bias, and combined logits separately.
These tensors describe model routing and are not biological or causal effects.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Iterator, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import (
    ExpressionDecoder,
    ModelOutput,
    _BaseMaskedExpressionModel,
    _normalize_target_nodes,
    _prepare_edge_index,
    _select_targets,
    _validate_hidden_dimensions,
)


RELATIVE_GEOMETRY_DIM = 70


def _transformer_ffn_dim(hidden_dim: int, ffn_dim: Optional[int]) -> int:
    if ffn_dim is None:
        return 4 * hidden_dim
    if ffn_dim <= 0:
        raise ValueError("ffn_dim must be positive")
    return int(ffn_dim)


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return value


def _prepare_relative_geometry(
    relative_geometry: Optional[Tensor],
    *,
    num_edges: int,
    reference: Tensor,
    move_to_reference: bool,
) -> Tensor:
    """Validate edge-aligned geometry, optionally moving the complete tensor."""

    if relative_geometry is None:
        if num_edges:
            raise ValueError(
                "relative_geometry is required when the graph contains edges"
            )
        return reference.new_empty((0, RELATIVE_GEOMETRY_DIM))
    if not isinstance(relative_geometry, Tensor):
        raise TypeError("relative_geometry must be a torch.Tensor")
    if not relative_geometry.is_floating_point():
        raise TypeError("relative_geometry must be floating point")
    expected = (num_edges, RELATIVE_GEOMETRY_DIM)
    if relative_geometry.ndim != 2 or tuple(relative_geometry.shape) != expected:
        raise ValueError(
            "relative_geometry shape mismatch: expected "
            f"{expected}, got {tuple(relative_geometry.shape)}"
        )
    if move_to_reference:
        return relative_geometry.to(
            device=reference.device,
            dtype=reference.dtype,
        )
    return relative_geometry


def _incoming_softmax(
    scores: Tensor,
    receiver: Tensor,
    *,
    num_receivers: int,
) -> Tensor:
    """Stable sparse per-receiver, per-head softmax in accumulation dtype."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_edges, num_heads]")
    if receiver.ndim != 1 or receiver.shape[0] != scores.shape[0]:
        raise ValueError("receiver must align one-for-one with scores")
    if num_receivers < 0:
        raise ValueError("num_receivers cannot be negative")
    if not scores.shape[0]:
        return scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores

    work = (
        scores.float()
        if scores.dtype in (torch.float16, torch.bfloat16)
        else scores
    )
    expanded_receiver = receiver.view(-1, 1).expand_as(work)
    maxima = work.new_full((num_receivers, work.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0,
        expanded_receiver,
        work,
        reduce="amax",
        include_self=True,
    )
    exponentials = torch.exp(work - maxima.index_select(0, receiver))
    denominator = work.new_zeros((num_receivers, work.shape[1]))
    denominator.index_add_(0, receiver, exponentials)
    return exponentials / denominator.index_select(0, receiver).clamp_min(
        torch.finfo(work.dtype).tiny
    )


def _content_scores(
    receiver_queries: Tensor,
    sender_keys: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return dot-product logits, promoting FP16/BF16 reductions to FP32."""

    if receiver_queries.dtype in (torch.float16, torch.bfloat16):
        receiver_queries = receiver_queries.float()
    if sender_keys.dtype in (torch.float16, torch.bfloat16):
        sender_keys = sender_keys.float()
    return (receiver_queries * sender_keys).sum(dim=-1) * scale


class RelativePositionBiasEncoder(nn.Module):
    """Map fixed 70-D relative geometry to one additive bias per head.

    The final projection is exactly zero-initialized, so geometry has no effect
    on attention at initialization while gradients can immediately update it.
    """

    def __init__(
        self,
        attention_heads: int,
        hidden_dim: int = 128,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
    ) -> None:
        super().__init__()
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if relative_geometry_dim != RELATIVE_GEOMETRY_DIM:
            raise ValueError(
                "relative_geometry_dim is fixed at "
                f"{RELATIVE_GEOMETRY_DIM}"
            )
        self.attention_heads = int(attention_heads)
        self.hidden_dim = int(hidden_dim)
        self.relative_geometry_dim = RELATIVE_GEOMETRY_DIM
        self.input_projection = nn.Linear(RELATIVE_GEOMETRY_DIM, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, attention_heads)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, relative_geometry: Tensor) -> Tensor:
        if relative_geometry.ndim != 2 or relative_geometry.shape[1] != self.relative_geometry_dim:
            raise ValueError(
                "relative_geometry must have shape "
                f"[num_edges, {self.relative_geometry_dim}]"
            )
        hidden = self.input_projection(relative_geometry)
        hidden = F.gelu(self.normalization(hidden))
        return self.output_projection(hidden)


class RelativeGeometryQKVGraphTransformerBlock(nn.Module):
    """Pre-LayerNorm QKV block with relative geometry in logits only."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int] = None,
        positional_bias_hidden_dim: int = 128,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dropout = _validate_probability(dropout, name="dropout")
        attention_dropout = _validate_probability(
            attention_dropout,
            name="attention_dropout",
        )
        head_dim, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim,
            attention_heads,
            attention_head_dim,
        )
        expanded_ffn_dim = _transformer_ffn_dim(hidden_dim, ffn_dim)

        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(head_dim)
        self.raw_attention_dim = int(raw_attention_dim)
        self.attention_scale = 1.0 / sqrt(float(head_dim))
        self.attention_dropout_probability = attention_dropout

        self.attention_normalization = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(
            hidden_dim,
            raw_attention_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            hidden_dim,
            raw_attention_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            hidden_dim,
            raw_attention_dim,
            bias=False,
        )
        self.relative_position_bias_encoder = RelativePositionBiasEncoder(
            attention_heads=attention_heads,
            hidden_dim=positional_bias_hidden_dim,
        )
        self.attention_output_projection = nn.Linear(
            raw_attention_dim,
            hidden_dim,
            bias=False,
        )
        self.residual_dropout = nn.Dropout(dropout)

        self.ffn_normalization = nn.LayerNorm(hidden_dim)
        self.ffn_input = nn.Linear(hidden_dim, expanded_ffn_dim)
        self.ffn_output = nn.Linear(expanded_ffn_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    @property
    def relative_position_bias(self) -> RelativePositionBiasEncoder:
        """Descriptive alias without duplicate state-dict registration."""

        return self.relative_position_bias_encoder

    def project_nodes(
        self,
        node_embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = self.attention_normalization(node_embedding)
        shape = (
            node_embedding.shape[0],
            self.attention_heads,
            self.attention_head_dim,
        )
        return (
            self.query_projection(normalized).view(shape),
            self.key_projection(normalized).view(shape),
            self.value_projection(normalized).view(shape),
        )

    def finish_partition(
        self,
        residual_embedding: Tensor,
        aggregated_values: Tensor,
    ) -> Tensor:
        flattened = aggregated_values.reshape(
            residual_embedding.shape[0],
            self.raw_attention_dim,
        )
        update = self.attention_output_projection(flattened)
        node_embedding = residual_embedding + self.residual_dropout(update)
        ffn_update = self.ffn_output(
            self.ffn_dropout(
                F.gelu(self.ffn_input(self.ffn_normalization(node_embedding)))
            )
        )
        return node_embedding + self.residual_dropout(ffn_update)


@dataclass(frozen=True)
class RelativeGeometryQKVDiagnostics:
    """Edge-aligned routing tensors from one requested graph layer."""

    edge_index: Tensor
    attention_weights: Tensor
    content_logits: Tensor
    positional_bias: Tensor
    combined_logits: Tensor
    source_indices: Tensor
    receiver_indices: Tensor
    layer_number: int
    full_node_embedding: Tensor
    selected_node_embedding: Tensor


@dataclass
class RelativeGeometryQKVModelOutput(ModelOutput):
    """Standard model output plus optional relative-QKV diagnostics."""

    content_logits: Optional[Tensor] = None
    positional_bias: Optional[Tensor] = None
    combined_logits: Optional[Tensor] = None
    source_indices: Optional[Tensor] = None
    receiver_indices: Optional[Tensor] = None
    layer_number: Optional[int] = None
    full_node_embedding: Optional[Tensor] = None
    selected_node_embedding: Optional[Tensor] = None
    diagnostics: Optional[RelativeGeometryQKVDiagnostics] = None
    node_encoder_embedding: Optional[Tensor] = None
    final_graph_embedding: Optional[Tensor] = None


def _resolve_explanation_layer(
    layer: int,
    *,
    graph_layers: int,
) -> int:
    layer = int(layer)
    if layer < 0:
        layer += graph_layers
    if not 0 <= layer < graph_layers:
        raise ValueError(
            f"explanation_layer must be in [-{graph_layers}, {graph_layers - 1}]"
        )
    return layer


class RelativeGeometryQKVGraphTransformer(_BaseMaskedExpressionModel):
    """Full-edge reference relative-geometry QKV graph Transformer."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        positional_bias_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
    ) -> None:
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if positional_bias_hidden_dim <= 0:
            raise ValueError("positional_bias_hidden_dim must be positive")
        if relative_geometry_dim != RELATIVE_GEOMETRY_DIM:
            raise ValueError(
                "relative_geometry_dim is fixed at "
                f"{RELATIVE_GEOMETRY_DIM}"
            )
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.relative_geometry_dim = RELATIVE_GEOMETRY_DIM
        self.graph_layers = int(graph_layers)
        self.blocks = nn.ModuleList(
            [
                RelativeGeometryQKVGraphTransformerBlock(
                    hidden_dim=hidden_dim,
                    attention_heads=attention_heads,
                    attention_head_dim=attention_head_dim,
                    positional_bias_hidden_dim=positional_bias_hidden_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(graph_layers)
            ]
        )
        self.decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

    @staticmethod
    def _explanation_receiver_mask(
        attention_receivers: Optional[Tensor | Sequence[int]],
        *,
        return_explanations: bool,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if attention_receivers is not None and not return_explanations:
            raise ValueError(
                "attention_receivers requires return_explanations=True"
            )
        if not return_explanations:
            return None
        selected = _normalize_target_nodes(
            attention_receivers,
            num_nodes=num_nodes,
            device=device,
        )
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if selected is None:
            mask.fill_(True)
        elif selected.numel():
            mask[selected] = True
        return mask

    def _attention_partition(
        self,
        block: RelativeGeometryQKVGraphTransformerBlock,
        query_projection: Tensor,
        key_projection: Tensor,
        value_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        receiver: Tensor,
        relative_geometry: Tensor,
        *,
        receiver_start: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluate one complete receiver partition and its aligned tensors."""

        num_receivers = residual_embedding.shape[0]
        num_edges = source.shape[0]
        heads = block.attention_heads
        head_dim = block.attention_head_dim
        local_receiver = receiver - receiver_start

        if num_edges:
            receiver_queries = query_projection.index_select(0, local_receiver)
            sender_keys = key_projection.index_select(0, source)
            content_logits = _content_scores(
                receiver_queries,
                sender_keys,
                scale=block.attention_scale,
            )
            positional_bias = block.relative_position_bias_encoder(
                relative_geometry
            )
            if positional_bias.dtype in (torch.float16, torch.bfloat16):
                positional_bias = positional_bias.float()
            combined_logits = content_logits + positional_bias.to(
                dtype=content_logits.dtype
            )
            attention = _incoming_softmax(
                combined_logits,
                local_receiver,
                num_receivers=num_receivers,
            )
            weights = F.dropout(
                attention,
                p=block.attention_dropout_probability,
                training=block.training,
            )
            sender_values = value_projection.index_select(0, source).to(
                dtype=attention.dtype
            )
            messages = sender_values * weights.unsqueeze(-1)
            aggregated = attention.new_zeros(
                (num_receivers, heads, head_dim)
            )
            aggregated.index_add_(0, local_receiver, messages)
        else:
            accumulation_dtype = (
                torch.float32
                if value_projection.dtype in (torch.float16, torch.bfloat16)
                else value_projection.dtype
            )
            shape = (0, heads)
            attention = torch.empty(
                shape,
                device=residual_embedding.device,
                dtype=accumulation_dtype,
            )
            content_logits = attention.clone()
            positional_bias = attention.clone()
            combined_logits = attention.clone()
            aggregated = torch.zeros(
                (num_receivers, heads, head_dim),
                device=residual_embedding.device,
                dtype=accumulation_dtype,
            )

        output = block.finish_partition(residual_embedding, aggregated)
        return (
            output,
            attention,
            content_logits,
            positional_bias,
            combined_logits,
        )

    def _output_partition(
        self,
        block: RelativeGeometryQKVGraphTransformerBlock,
        receiver_start: int,
        *arguments: Tensor,
    ) -> Tensor:
        return self._attention_partition(
            block,
            *arguments,
            receiver_start=receiver_start,
        )[0]

    def _make_output(
        self,
        *,
        prediction: Tensor,
        selected_embedding: Tensor,
        full_embedding: Tensor,
        node_encoder_embedding: Optional[Tensor],
        edge_index: Optional[Tensor],
        attention: Optional[Tensor],
        content_logits: Optional[Tensor],
        positional_bias: Optional[Tensor],
        combined_logits: Optional[Tensor],
        layer_number: Optional[int],
    ) -> RelativeGeometryQKVModelOutput:
        if edge_index is None:
            diagnostics = None
            source_indices = None
            receiver_indices = None
        else:
            assert attention is not None
            assert content_logits is not None
            assert positional_bias is not None
            assert combined_logits is not None
            assert layer_number is not None
            source_indices = edge_index[0]
            receiver_indices = edge_index[1]
            diagnostics = RelativeGeometryQKVDiagnostics(
                edge_index=edge_index,
                attention_weights=attention,
                content_logits=content_logits,
                positional_bias=positional_bias,
                combined_logits=combined_logits,
                source_indices=source_indices,
                receiver_indices=receiver_indices,
                layer_number=layer_number,
                full_node_embedding=full_embedding,
                selected_node_embedding=selected_embedding,
            )
        return RelativeGeometryQKVModelOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            node_encoder_embedding=node_encoder_embedding,
            final_graph_embedding=(
                selected_embedding
                if node_encoder_embedding is not None
                else None
            ),
            attention_weights=attention,
            edge_index=edge_index,
            content_logits=content_logits,
            positional_bias=positional_bias,
            combined_logits=combined_logits,
            source_indices=source_indices,
            receiver_indices=receiver_indices,
            layer_number=layer_number,
            full_node_embedding=full_embedding,
            selected_node_embedding=selected_embedding,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        relative_geometry: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        *,
        attention_receivers: Optional[Tensor | Sequence[int]] = None,
        explanation_layer: int = -1,
        return_diagnostics: Optional[bool] = None,
        return_intermediate_embeddings: bool = False,
    ) -> RelativeGeometryQKVModelOutput:
        if return_diagnostics is not None:
            if return_explanations and not return_diagnostics:
                raise ValueError(
                    "return_diagnostics=False conflicts with "
                    "return_explanations=True"
                )
            return_explanations = bool(return_diagnostics)
        selected_layer = _resolve_explanation_layer(
            explanation_layer,
            graph_layers=self.graph_layers,
        )
        num_nodes = input_expression.shape[0]
        prepared_edges = _prepare_edge_index(
            edge_index,
            num_nodes=num_nodes,
            device=input_expression.device,
        )
        prepared_geometry = _prepare_relative_geometry(
            relative_geometry,
            num_edges=prepared_edges.shape[1],
            reference=input_expression,
            move_to_reference=True,
        )
        node_embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
        )
        node_encoder_embedding = (
            _select_targets(node_embedding, targets)
            if return_intermediate_embeddings
            else None
        )
        receiver_mask = self._explanation_receiver_mask(
            attention_receivers,
            return_explanations=return_explanations,
            num_nodes=num_nodes,
            device=prepared_edges.device,
        )

        final_attention: Optional[Tensor] = None
        final_content: Optional[Tensor] = None
        final_bias: Optional[Tensor] = None
        final_combined: Optional[Tensor] = None
        diagnostic_edges: Optional[Tensor] = None
        source, receiver = prepared_edges
        for layer_number, block in enumerate(self.blocks):
            queries, keys, values = block.project_nodes(node_embedding)
            (
                node_embedding,
                attention,
                content_logits,
                positional_bias,
                combined_logits,
            ) = self._attention_partition(
                block,
                queries,
                keys,
                values,
                node_embedding,
                source,
                receiver,
                prepared_geometry,
            )
            if return_explanations and layer_number == selected_layer:
                assert receiver_mask is not None
                selected_edges = receiver_mask.index_select(0, receiver)
                diagnostic_edges = prepared_edges[:, selected_edges]
                final_attention = attention[selected_edges]
                final_content = content_logits[selected_edges]
                final_bias = positional_bias[selected_edges]
                final_combined = combined_logits[selected_edges]

        selected_embedding = _select_targets(node_embedding, targets)
        prediction = self.decoder(selected_embedding)
        return self._make_output(
            prediction=prediction,
            selected_embedding=selected_embedding,
            full_embedding=node_embedding,
            node_encoder_embedding=node_encoder_embedding,
            edge_index=diagnostic_edges,
            attention=final_attention,
            content_logits=final_content,
            positional_bias=final_bias,
            combined_logits=final_combined,
            layer_number=selected_layer if return_explanations else None,
        )


@dataclass
class _ReceiverLayout:
    edge_index: Tensor
    tensor_version: int
    num_nodes: int
    edge_order: Optional[Tensor]
    receiver_ptr: tuple[int, ...]


def _prepare_chunked_edge_index(
    edge_index: Optional[Tensor],
    *,
    num_nodes: int,
    default_device: torch.device,
) -> Tensor:
    """Validate without moving a caller's complete graph to the model device."""

    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long, device=default_device)
    if not isinstance(edge_index, Tensor):
        raise TypeError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype == torch.bool or edge_index.is_floating_point():
        raise TypeError("edge_index must contain integer node indices")
    prepared = edge_index.to(dtype=torch.long)
    if prepared.shape[1]:
        if bool((prepared < 0).any()) or bool((prepared >= num_nodes).any()):
            raise ValueError("edge_index contains an out-of-range node index")
        if bool((prepared[0] == prepared[1]).any()):
            raise ValueError("explicit self loops are not allowed")
    return prepared


class ReceiverChunkedRelativeGeometryQKVGraphTransformer(
    RelativeGeometryQKVGraphTransformer
):
    """Memory-bounded, exact receiver-chunked relative QKV Transformer."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        positional_bias_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
        *,
        receiver_chunk_size: int = 256,
        max_edges_per_chunk: Optional[int] = None,
        activation_checkpointing: bool = True,
    ) -> None:
        if receiver_chunk_size <= 0:
            raise ValueError("receiver_chunk_size must be positive")
        if max_edges_per_chunk is not None and max_edges_per_chunk <= 0:
            raise ValueError("max_edges_per_chunk must be positive")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be boolean")
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            graph_layers=graph_layers,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            positional_bias_hidden_dim=positional_bias_hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relative_geometry_dim=relative_geometry_dim,
        )
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.max_edges_per_chunk = (
            None
            if max_edges_per_chunk is None
            else int(max_edges_per_chunk)
        )
        self.activation_checkpointing = activation_checkpointing
        self._receiver_layout_cache: Optional[_ReceiverLayout] = None

    def clear_edge_layout_cache(self) -> None:
        self._receiver_layout_cache = None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        self.clear_edge_layout_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()
        state["_receiver_layout_cache"] = None
        return state

    def _receiver_layout(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
    ) -> _ReceiverLayout:
        version = int(edge_index._version)
        cached = self._receiver_layout_cache
        if (
            cached is not None
            and cached.edge_index is edge_index
            and cached.tensor_version == version
            and cached.num_nodes == num_nodes
        ):
            return cached

        receiver = edge_index[1]
        counts = torch.bincount(receiver, minlength=num_nodes)
        cumulative = torch.cumsum(counts, dim=0).cpu().tolist()
        receiver_ptr = (0, *(int(value) for value in cumulative))
        if receiver.numel() < 2 or not bool(
            (receiver[1:] < receiver[:-1]).any()
        ):
            edge_order = None
        else:
            edge_order = torch.argsort(receiver, stable=True)
        layout = _ReceiverLayout(
            edge_index=edge_index,
            tensor_version=version,
            num_nodes=num_nodes,
            edge_order=edge_order,
            receiver_ptr=receiver_ptr,
        )
        self._receiver_layout_cache = layout
        return layout

    def _receiver_ranges(
        self,
        layout: _ReceiverLayout,
        *,
        num_nodes: int,
    ) -> Iterator[tuple[int, int]]:
        receiver_start = 0
        while receiver_start < num_nodes:
            hard_stop = min(
                receiver_start + self.receiver_chunk_size,
                num_nodes,
            )
            receiver_stop = hard_stop
            if self.max_edges_per_chunk is not None:
                edge_start = layout.receiver_ptr[receiver_start]
                edge_limit = edge_start + self.max_edges_per_chunk
                receiver_stop = min(
                    hard_stop,
                    max(
                        receiver_start + 1,
                        bisect_right(layout.receiver_ptr, edge_limit) - 1,
                    ),
                )
            yield receiver_start, receiver_stop
            receiver_start = receiver_stop

    @staticmethod
    def _original_ids(
        layout: _ReceiverLayout,
        *,
        edge_start: int,
        edge_stop: int,
    ) -> Tensor:
        if layout.edge_order is None:
            return torch.arange(
                edge_start,
                edge_stop,
                dtype=torch.long,
                device=layout.edge_index.device,
            )
        return layout.edge_order[edge_start:edge_stop]

    def _chunked_layer(
        self,
        block: RelativeGeometryQKVGraphTransformerBlock,
        node_embedding: Tensor,
        edge_index: Tensor,
        relative_geometry: Tensor,
        layout: _ReceiverLayout,
        explanation_receivers: Optional[Tensor],
    ) -> tuple[
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        num_nodes = node_embedding.shape[0]
        queries, keys, values = block.project_nodes(node_embedding)
        outputs: list[Tensor] = []
        selected_ids: list[Tensor] = []
        selected_attention: list[Tensor] = []
        selected_content: list[Tensor] = []
        selected_bias: list[Tensor] = []
        selected_combined: list[Tensor] = []
        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled()

        for receiver_start, receiver_stop in self._receiver_ranges(
            layout,
            num_nodes=num_nodes,
        ):
            edge_start = layout.receiver_ptr[receiver_start]
            edge_stop = layout.receiver_ptr[receiver_stop]
            original_ids = self._original_ids(
                layout,
                edge_start=edge_start,
                edge_stop=edge_stop,
            )
            chunk_edges_host = edge_index.index_select(1, original_ids)
            chunk_geometry_host = relative_geometry.index_select(0, original_ids)
            source_host = chunk_edges_host[0]
            receiver_host = chunk_edges_host[1]
            source = source_host.to(
                device=node_embedding.device,
                dtype=torch.long,
            )
            receiver = receiver_host.to(
                device=node_embedding.device,
                dtype=torch.long,
            )
            chunk_geometry = chunk_geometry_host.to(
                device=node_embedding.device,
                dtype=node_embedding.dtype,
            )
            query_chunk = queries[receiver_start:receiver_stop]
            residual = node_embedding[receiver_start:receiver_stop]
            common = (
                query_chunk,
                keys,
                values,
                residual,
                source,
                receiver,
                chunk_geometry,
            )
            chunk_function = partial(
                self._output_partition,
                block,
                receiver_start,
            )
            if explanation_receivers is None:
                if use_checkpoint:
                    output = checkpoint(
                        chunk_function,
                        *common,
                        use_reentrant=False,
                    )
                else:
                    output = chunk_function(*common)
                outputs.append(output)
                continue

            selected_host = explanation_receivers.index_select(0, receiver_host)
            if not bool(selected_host.any()):
                # Explanation extraction needs autograd state only for chunks that
                # contain a requested receiver.  Preserve the ordinary activation-
                # checkpointed path everywhere else so a handful of selected
                # derivatives does not retain the full-core edge graph in memory.
                if use_checkpoint:
                    output = checkpoint(
                        chunk_function,
                        *common,
                        use_reentrant=False,
                    )
                else:
                    output = chunk_function(*common)
                outputs.append(output)
                continue

            result = self._attention_partition(
                block,
                *common,
                receiver_start=receiver_start,
            )
            output, attention, content, bias, combined = result
            outputs.append(output)
            selected_device = selected_host.to(device=node_embedding.device)
            selected_ids.append(original_ids[selected_host])
            selected_attention.append(attention[selected_device])
            selected_content.append(content[selected_device])
            selected_bias.append(bias[selected_device])
            selected_combined.append(combined[selected_device])

        block_output = torch.cat(outputs, dim=0) if outputs else node_embedding
        if explanation_receivers is None:
            return block_output, None, None, None, None, None
        if selected_ids:
            edge_ids = torch.cat(selected_ids)
            attention = torch.cat(selected_attention)
            content = torch.cat(selected_content)
            bias = torch.cat(selected_bias)
            combined = torch.cat(selected_combined)
            restore_host = torch.argsort(edge_ids)
            restore_device = restore_host.to(device=node_embedding.device)
            return (
                block_output,
                edge_ids.index_select(0, restore_host),
                attention.index_select(0, restore_device),
                content.index_select(0, restore_device),
                bias.index_select(0, restore_device),
                combined.index_select(0, restore_device),
            )

        empty_ids = edge_index.new_empty((0,), dtype=torch.long)
        accumulation_dtype = (
            torch.float32
            if node_embedding.dtype in (torch.float16, torch.bfloat16)
            else node_embedding.dtype
        )
        empty = torch.empty(
            (0, block.attention_heads),
            dtype=accumulation_dtype,
            device=node_embedding.device,
        )
        return block_output, empty_ids, empty, empty.clone(), empty.clone(), empty.clone()

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        relative_geometry: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        *,
        attention_receivers: Optional[Tensor | Sequence[int]] = None,
        explanation_layer: int = -1,
        return_diagnostics: Optional[bool] = None,
        return_intermediate_embeddings: bool = False,
    ) -> RelativeGeometryQKVModelOutput:
        if return_diagnostics is not None:
            if return_explanations and not return_diagnostics:
                raise ValueError(
                    "return_diagnostics=False conflicts with "
                    "return_explanations=True"
                )
            return_explanations = bool(return_diagnostics)
        selected_layer = _resolve_explanation_layer(
            explanation_layer,
            graph_layers=self.graph_layers,
        )
        num_nodes = input_expression.shape[0]
        prepared_edges = _prepare_chunked_edge_index(
            edge_index,
            num_nodes=num_nodes,
            default_device=input_expression.device,
        )
        prepared_geometry = _prepare_relative_geometry(
            relative_geometry,
            num_edges=prepared_edges.shape[1],
            reference=input_expression,
            move_to_reference=False,
        )
        if prepared_geometry.device != prepared_edges.device:
            raise ValueError(
                "chunked edge_index and relative_geometry must share a device; "
                "keep both CPU-resident for memory-bounded execution"
            )
        node_embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
        )
        node_encoder_embedding = (
            _select_targets(node_embedding, targets)
            if return_intermediate_embeddings
            else None
        )
        layout = self._receiver_layout(prepared_edges, num_nodes=num_nodes)
        receiver_mask = self._explanation_receiver_mask(
            attention_receivers,
            return_explanations=return_explanations,
            num_nodes=num_nodes,
            device=prepared_edges.device,
        )

        edge_ids: Optional[Tensor] = None
        final_attention: Optional[Tensor] = None
        final_content: Optional[Tensor] = None
        final_bias: Optional[Tensor] = None
        final_combined: Optional[Tensor] = None
        for layer_number, block in enumerate(self.blocks):
            diagnostic_mask = (
                receiver_mask
                if return_explanations and layer_number == selected_layer
                else None
            )
            (
                node_embedding,
                layer_ids,
                attention,
                content,
                bias,
                combined,
            ) = self._chunked_layer(
                block,
                node_embedding,
                prepared_edges,
                prepared_geometry,
                layout,
                diagnostic_mask,
            )
            if diagnostic_mask is not None:
                edge_ids = layer_ids
                final_attention = attention
                final_content = content
                final_bias = bias
                final_combined = combined

        if edge_ids is None:
            diagnostic_edges = None
        else:
            diagnostic_edges = prepared_edges.index_select(1, edge_ids)
        selected_embedding = _select_targets(node_embedding, targets)
        prediction = self.decoder(selected_embedding)
        return self._make_output(
            prediction=prediction,
            selected_embedding=selected_embedding,
            full_embedding=node_embedding,
            node_encoder_embedding=node_encoder_embedding,
            edge_index=diagnostic_edges,
            attention=final_attention,
            content_logits=final_content,
            positional_bias=final_bias,
            combined_logits=final_combined,
            layer_number=selected_layer if return_explanations else None,
        )


# Compact aliases for configuration and exploratory callers.
RelativeQKVGraphTransformerBlock = RelativeGeometryQKVGraphTransformerBlock
RelativeQKVGraphTransformer = RelativeGeometryQKVGraphTransformer
ReceiverChunkedRelativeQKVGraphTransformer = (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer
)
DenseRelativeGeometryQKVGraphTransformer = (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer
)


__all__ = [
    "DenseRelativeGeometryQKVGraphTransformer",
    "RELATIVE_GEOMETRY_DIM",
    "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
    "ReceiverChunkedRelativeQKVGraphTransformer",
    "RelativeGeometryQKVDiagnostics",
    "RelativeGeometryQKVGraphTransformer",
    "RelativeGeometryQKVGraphTransformerBlock",
    "RelativeGeometryQKVModelOutput",
    "RelativePositionBiasEncoder",
    "RelativeQKVGraphTransformer",
    "RelativeQKVGraphTransformerBlock",
]

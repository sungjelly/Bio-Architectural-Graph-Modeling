r"""Exact edge-aware QKV graph-transformer models.

This module implements Transformer attention on a supplied sparse graph.  For
every directed edge ``j -> i`` and attention head ``h``, the logit and message
are

.. math::

    s_{ijh} = Q_h(x_i)^T K_h(x_j) / \sqrt{d_h} + b_h^e(e_{ij})

.. math::

    m_{ijh} = \operatorname{softmax}_{j \in N(i)}(s_{ijh})
              g_h^e(e_{ij}) V_h(x_j).

Thus this is true receiver-query/sender-key dot-product attention, rather than
the additive scoring function used by GATv2.  The production ``bias_gate``
mode uses a per-head edge attention bias and bounded value gate.  The optional
``vector`` mode additionally adds full head-dimensional edge vectors to keys
and values.  Both use one shared edge encoder.  There are no learned node or
edge identifiers and no implicit self loops.

``EdgeAwareQKVGraphTransformer`` is the ordinary full-edge implementation and
serves as a transparent reference on small graphs.
``ReceiverChunkedEdgeAwareQKVGraphTransformer`` has the identical parameter
and state-dict layout but partitions execution by contiguous receiver ranges.
Every receiver partition contains *all* of its incoming edges, so chunking is
exact execution, not neighbor sampling.  Only current-chunk edge embeddings
and attention activations are materialized.  Optional non-reentrant activation
checkpointing recomputes those tensors during backward.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Iterator, Literal, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import (
    ExpressionDecoder,
    ModelOutput,
    SharedEdgeEncoder,
    _BaseMaskedExpressionModel,
    _normalize_target_nodes,
    _prepare_edge_attributes,
    _prepare_edge_index,
    _select_targets,
    _validate_hidden_dimensions,
)


def _default_transformer_expansion(
    hidden_dim: int, ffn_dim: Optional[int]
) -> int:
    """Return the conventional four-times-hidden Transformer FFN width."""

    if ffn_dim is None:
        return 4 * hidden_dim
    if ffn_dim <= 0:
        raise ValueError("ffn_dim must be positive")
    return int(ffn_dim)


def _incoming_softmax(
    scores: Tensor,
    receiver: Tensor,
    *,
    num_receivers: int,
) -> Tensor:
    """Stable per-receiver, per-head softmax without graph densification.

    Half and bfloat16 logits are normalized and retained in float32.  This is
    important for high-degree graphs because both the denominator and the
    subsequent weighted-value sum may contain thousands of terms.  Returning
    low-precision weights here corrupts a nominally uniform sum (for example,
    FP16 accumulation can yield approximately 0.5 rather than 1 at k=5000).
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_edges, num_heads]")
    if receiver.ndim != 1 or receiver.shape[0] != scores.shape[0]:
        raise ValueError("receiver must align one-for-one with scores")
    if num_receivers < 0:
        raise ValueError("num_receivers cannot be negative")
    if scores.shape[0] == 0:
        return scores

    work = (
        scores.float()
        if scores.dtype in (torch.float16, torch.bfloat16)
        else scores
    )
    expanded_receiver = receiver.view(-1, 1).expand_as(work)
    maxima = work.new_full(
        (num_receivers, work.shape[1]),
        -torch.inf,
    )
    maxima.scatter_reduce_(
        0,
        expanded_receiver,
        work,
        reduce="amax",
        include_self=True,
    )
    exponentiated = torch.exp(
        work - maxima.index_select(0, receiver)
    )
    denominator = work.new_zeros((num_receivers, work.shape[1]))
    denominator.scatter_add_(
        0,
        expanded_receiver,
        exponentiated,
    )
    return exponentiated / denominator.index_select(
        0, receiver
    ).clamp_min(torch.finfo(work.dtype).tiny)


class QKVGraphTransformerBlock(nn.Module):
    """Pre-LayerNorm edge-aware multi-head graph Transformer block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int],
        edge_embedding_dim: int,
        edge_conditioning_mode: Literal["bias_gate", "vector"],
        ffn_dim: Optional[int],
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if edge_embedding_dim <= 0:
            raise ValueError("edge_embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if edge_conditioning_mode not in {"bias_gate", "vector"}:
            raise ValueError(
                "edge_conditioning_mode must be 'bias_gate' or 'vector'"
            )

        head_dim, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim,
            attention_heads,
            attention_head_dim,
        )
        expanded_ffn_dim = _default_transformer_expansion(
            hidden_dim,
            ffn_dim,
        )
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(head_dim)
        self.raw_attention_dim = int(raw_attention_dim)
        self.attention_scale = 1.0 / sqrt(float(head_dim))
        self.attention_dropout_probability = float(attention_dropout)
        self.edge_conditioning_mode = edge_conditioning_mode

        self.attention_normalization = nn.LayerNorm(hidden_dim)
        # Separate projections make the Q/K/V semantics explicit in both the
        # state dict and architecture audits.
        self.query_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.value_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.edge_attention_bias = nn.Linear(
            edge_embedding_dim, attention_heads, bias=False
        )
        if edge_conditioning_mode == "vector":
            self.edge_key_projection: Optional[nn.Linear] = nn.Linear(
                edge_embedding_dim, raw_attention_dim, bias=False
            )
            self.edge_value_projection: Optional[nn.Linear] = nn.Linear(
                edge_embedding_dim, raw_attention_dim, bias=False
            )
            self.edge_value_gate: Optional[nn.Linear] = None
        else:
            self.edge_key_projection = None
            self.edge_value_projection = None
            self.edge_value_gate = nn.Linear(
                edge_embedding_dim, attention_heads, bias=False
            )
        # A learned W_O is retained even when H*d_h == hidden_dim, matching a
        # standard Transformer and avoiding an accidental low-capacity path.
        self.attention_output_projection = nn.Linear(
            raw_attention_dim, hidden_dim, bias=False
        )
        self.residual_dropout = nn.Dropout(dropout)

        self.ffn_normalization = nn.LayerNorm(hidden_dim)
        self.ffn_input = nn.Linear(hidden_dim, expanded_ffn_dim)
        self.ffn_output = nn.Linear(expanded_ffn_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def project_nodes(
        self, node_embedding: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project pre-normalized nodes to explicit multi-head Q, K, and V."""

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
        """Apply attention and FFN residual paths to one receiver partition."""

        num_receivers = residual_embedding.shape[0]
        flattened = aggregated_values.reshape(
            num_receivers, self.raw_attention_dim
        )
        # Weighted messages are accumulated in FP32 under AMP.  Downcast only
        # at the learned output projection boundary; doing so earlier makes a
        # high-degree uniform sum materially smaller than one.
        attention_update = self.attention_output_projection(
            flattened.to(dtype=residual_embedding.dtype)
        )
        node_embedding = residual_embedding + self.residual_dropout(
            attention_update
        )
        normalized = self.ffn_normalization(node_embedding)
        ffn_update = self.ffn_output(
            self.ffn_dropout(F.gelu(self.ffn_input(normalized)))
        )
        return node_embedding + self.residual_dropout(ffn_update)


class EdgeAwareQKVGraphTransformer(_BaseMaskedExpressionModel):
    """Ordinary full-edge reference implementation of the QKV backbone.

    This class is suitable for sparse and small graphs.  Dense high-k runs
    should use :class:`ReceiverChunkedEdgeAwareQKVGraphTransformer`, whose
    parameters and outputs are equivalent while bounding edge activation
    memory.

    Attention weights returned for explanations are normalized, pre-dropout
    routing weights from the final graph layer.  They are model internals, not
    biological importance or causal effects.
    """

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        edge_hidden_dim: int = 128,
        edge_embedding_dim: int = 64,
        edge_conditioning_mode: Literal[
            "bias_gate", "vector"
        ] = "bias_gate",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if edge_attribute_dim <= 0:
            raise ValueError("edge_attribute_dim must be positive")
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.edge_attribute_dim = int(edge_attribute_dim)
        self.edge_embedding_dim = int(edge_embedding_dim)
        self.edge_conditioning_mode = edge_conditioning_mode
        self.edge_encoder = SharedEdgeEncoder(
            edge_attribute_dim=edge_attribute_dim,
            edge_embedding_dim=edge_embedding_dim,
            edge_hidden_dim=edge_hidden_dim,
        )
        self.blocks = nn.ModuleList(
            [
                QKVGraphTransformerBlock(
                    hidden_dim=hidden_dim,
                    attention_heads=attention_heads,
                    attention_head_dim=attention_head_dim,
                    edge_embedding_dim=edge_embedding_dim,
                    edge_conditioning_mode=edge_conditioning_mode,
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

    def _attention_partition(
        self,
        block: QKVGraphTransformerBlock,
        query_projection: Tensor,
        key_projection: Tensor,
        value_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        receiver: Tensor,
        edge_attributes: Tensor,
        *,
        receiver_start: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate all incoming edges for a complete receiver partition."""

        num_receivers = residual_embedding.shape[0]
        num_edges = source.shape[0]
        heads = block.attention_heads
        head_dim = block.attention_head_dim
        edge_embedding = self.edge_encoder(edge_attributes)
        # For checkpointed dense execution, ``receiver`` is a view into the
        # caller's persistent global edge index.  Constructing this local copy
        # inside the checkpointed function makes it recomputable rather than a
        # separately saved E-length int64 checkpoint input.
        local_receiver = (
            receiver
            if receiver_start == 0
            else receiver - receiver_start
        )

        if num_edges:
            receiver_queries = query_projection.index_select(
                0, local_receiver
            )
            sender_keys = key_projection.index_select(0, source)
            if block.edge_conditioning_mode == "vector":
                assert block.edge_key_projection is not None
                edge_keys = block.edge_key_projection(edge_embedding).view(
                    num_edges, heads, head_dim
                )
                sender_keys = sender_keys + edge_keys
            scores = (
                torch.einsum(
                    "ehd,ehd->eh", receiver_queries, sender_keys
                )
                * block.attention_scale
                + block.edge_attention_bias(edge_embedding)
            )
            attention = _incoming_softmax(
                scores,
                local_receiver,
                num_receivers=num_receivers,
            )
            aggregation_weights = F.dropout(
                attention,
                p=block.attention_dropout_probability,
                training=block.training,
            )
            # Keep the entire weighted-value reduction in the softmax
            # accumulation dtype (FP32 for FP16/BF16 execution).
            sender_values = value_projection.index_select(
                0, source
            ).to(dtype=attention.dtype)
            if block.edge_conditioning_mode == "vector":
                assert block.edge_value_projection is not None
                edge_values = block.edge_value_projection(
                    edge_embedding
                ).view(num_edges, heads, head_dim).to(
                    dtype=attention.dtype
                )
                sender_values = sender_values + edge_values
            else:
                assert block.edge_value_gate is not None
                # The gate is bounded in (0, 2), with its neutral point at 1.
                # It conditions values without an E x H x d edge projection.
                value_gate = 1.0 + torch.tanh(
                    block.edge_value_gate(edge_embedding)
                ).to(dtype=attention.dtype)
                aggregation_weights = aggregation_weights * value_gate
            messages = sender_values * aggregation_weights.unsqueeze(-1)
            aggregated = attention.new_zeros(
                (num_receivers, heads, head_dim)
            )
            aggregated.index_add_(0, local_receiver, messages)
        else:
            accumulation_dtype = (
                torch.float32
                if value_projection.dtype
                in (torch.float16, torch.bfloat16)
                else value_projection.dtype
            )
            attention = torch.empty(
                (0, heads),
                dtype=accumulation_dtype,
                device=query_projection.device,
            )
            aggregated = torch.zeros(
                (num_receivers, heads, head_dim),
                dtype=accumulation_dtype,
                device=value_projection.device,
            )

        output = block.finish_partition(residual_embedding, aggregated)
        return output, attention, edge_embedding

    def _output_partition(
        self,
        block: QKVGraphTransformerBlock,
        receiver_start: int,
        query_projection: Tensor,
        key_projection: Tensor,
        value_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        receiver: Tensor,
        edge_attributes: Tensor,
    ) -> Tensor:
        return self._attention_partition(
            block,
            query_projection,
            key_projection,
            value_projection,
            residual_embedding,
            source,
            receiver,
            edge_attributes,
            receiver_start=receiver_start,
        )[0]

    def _selected_explanation_partition(
        self,
        block: QKVGraphTransformerBlock,
        receiver_start: int,
        query_projection: Tensor,
        key_projection: Tensor,
        value_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        receiver: Tensor,
        edge_attributes: Tensor,
        explanation_receivers: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, attention, edge_embedding = self._attention_partition(
            block,
            query_projection,
            key_projection,
            value_projection,
            residual_embedding,
            source,
            receiver,
            edge_attributes,
            receiver_start=receiver_start,
        )
        selected_edges = explanation_receivers.index_select(0, receiver)
        return (
            output,
            attention[selected_edges],
            edge_embedding[selected_edges],
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
        selected_receivers = _normalize_target_nodes(
            attention_receivers,
            num_nodes=num_nodes,
            device=device,
        )
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if selected_receivers is None:
            mask.fill_(True)
        elif selected_receivers.numel():
            mask[selected_receivers] = True
        return mask

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attributes: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        *,
        attention_receivers: Optional[Tensor | Sequence[int]] = None,
    ) -> ModelOutput:
        num_nodes = input_expression.shape[0]
        prepared_edges = _prepare_edge_index(
            edge_index,
            num_nodes=num_nodes,
            device=input_expression.device,
        )
        prepared_attributes = _prepare_edge_attributes(
            edge_attributes,
            num_edges=prepared_edges.shape[1],
            edge_attribute_dim=self.edge_attribute_dim,
            reference=input_expression,
        )
        node_embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
        )
        explanation_receivers = self._explanation_receiver_mask(
            attention_receivers,
            return_explanations=return_explanations,
            num_nodes=num_nodes,
            device=input_expression.device,
        )

        final_attention: Optional[Tensor] = None
        final_edge_embedding: Optional[Tensor] = None
        for layer_number, block in enumerate(self.blocks):
            queries, keys, values = block.project_nodes(node_embedding)
            node_embedding, attention, encoded_edges = (
                self._attention_partition(
                    block,
                    queries,
                    keys,
                    values,
                    node_embedding,
                    prepared_edges[0],
                    prepared_edges[1],
                    prepared_attributes,
                )
            )
            if (
                return_explanations
                and layer_number == len(self.blocks) - 1
            ):
                final_attention = attention
                final_edge_embedding = encoded_edges

        explanation_edges: Optional[Tensor] = None
        if return_explanations:
            assert explanation_receivers is not None
            selected_edges = explanation_receivers.index_select(
                0, prepared_edges[1]
            )
            explanation_edges = prepared_edges[:, selected_edges]
            assert final_attention is not None
            assert final_edge_embedding is not None
            final_attention = final_attention[selected_edges]
            final_edge_embedding = final_edge_embedding[selected_edges]

        selected_embedding = _select_targets(node_embedding, targets)
        prediction = self.decoder(selected_embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            attention_weights=final_attention,
            edge_embedding=final_edge_embedding,
            edge_index=explanation_edges,
        )


@dataclass
class _ReceiverLayout:
    """Cached stable receiver order for one caller-owned edge tensor."""

    edge_index: Tensor
    tensor_version: int
    num_nodes: int
    edge_order: Optional[Tensor]
    receiver_ptr: tuple[int, ...]


class ReceiverChunkedEdgeAwareQKVGraphTransformer(
    EdgeAwareQKVGraphTransformer
):
    """Memory-bounded exact receiver-chunked QKV graph Transformer.

    ``receiver_chunk_size`` bounds the number of receiver nodes evaluated at
    once.  ``max_edges_per_chunk`` can additionally reduce a partition before
    execution when the receivers have unusually high degree.  A single
    receiver is never split even if its degree exceeds that soft edge bound.
    No neighbor is sampled, truncated, or renormalized outside its receiver's
    complete incoming edge set.

    When ``return_explanations=True``, pass a small ``attention_receivers``
    selection on a dense graph.  Only the selected final-layer incoming edges,
    normalized attention weights, and edge embeddings are retained and
    restored to their positions in the supplied ``edge_index``.
    """

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        edge_hidden_dim: int = 128,
        edge_embedding_dim: int = 64,
        edge_conditioning_mode: Literal[
            "bias_gate", "vector"
        ] = "bias_gate",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        *,
        receiver_chunk_size: int = 256,
        max_edges_per_chunk: Optional[int] = None,
        activation_checkpointing: bool = True,
    ) -> None:
        if receiver_chunk_size <= 0:
            raise ValueError("receiver_chunk_size must be positive")
        if (
            max_edges_per_chunk is not None
            and (
                isinstance(max_edges_per_chunk, bool)
                or max_edges_per_chunk <= 0
            )
        ):
            raise ValueError("max_edges_per_chunk must be positive or None")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be boolean")
        super().__init__(
            num_genes=num_genes,
            edge_attribute_dim=edge_attribute_dim,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            graph_layers=graph_layers,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            edge_hidden_dim=edge_hidden_dim,
            edge_embedding_dim=edge_embedding_dim,
            edge_conditioning_mode=edge_conditioning_mode,
            dropout=dropout,
            attention_dropout=attention_dropout,
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
        """Discard the derived order after replacing or mutating a graph."""

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
            # Stability preserves caller order within each receiver, which
            # keeps softmax and summation order aligned with the reference.
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
        """Yield exact receiver ranges under node and soft edge bounds."""

        receiver_start = 0
        while receiver_start < num_nodes:
            hard_stop = min(
                receiver_start + self.receiver_chunk_size,
                num_nodes,
            )
            receiver_stop = hard_stop
            if self.max_edges_per_chunk is not None:
                edge_limit = (
                    layout.receiver_ptr[receiver_start]
                    + self.max_edges_per_chunk
                )
                receiver_stop = (
                    bisect_right(
                        layout.receiver_ptr,
                        edge_limit,
                        lo=receiver_start + 1,
                        hi=hard_stop + 1,
                    )
                    - 1
                )
                # Never split the incoming edges for one receiver.
                receiver_stop = max(
                    receiver_stop, receiver_start + 1
                )
            yield receiver_start, receiver_stop
            receiver_start = receiver_stop

    def _chunked_block_forward(
        self,
        *,
        block: QKVGraphTransformerBlock,
        node_embedding: Tensor,
        edge_index: Tensor,
        edge_attributes: Tensor,
        layout: _ReceiverLayout,
        explanation_receivers: Optional[Tensor],
    ) -> tuple[
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        num_nodes = node_embedding.shape[0]
        queries, keys, values = block.project_nodes(node_embedding)
        outputs: list[Tensor] = []
        selected_edge_ids: list[Tensor] = []
        selected_attention: list[Tensor] = []
        selected_edge_embeddings: list[Tensor] = []
        use_checkpoint = (
            self.activation_checkpointing and torch.is_grad_enabled()
        )

        for receiver_start, receiver_stop in self._receiver_ranges(
            layout,
            num_nodes=num_nodes,
        ):
            edge_start = layout.receiver_ptr[receiver_start]
            edge_stop = layout.receiver_ptr[receiver_stop]
            if layout.edge_order is None:
                chunk_edges = edge_index[:, edge_start:edge_stop]
                chunk_attributes = edge_attributes[edge_start:edge_stop]
                original_ids: Optional[Tensor] = None
            else:
                original_ids = layout.edge_order[edge_start:edge_stop]
                chunk_edges = edge_index.index_select(1, original_ids)
                chunk_attributes = edge_attributes.index_select(
                    0, original_ids
                )

            source = chunk_edges[0]
            receiver = chunk_edges[1]
            residual_chunk = node_embedding[receiver_start:receiver_stop]
            query_chunk = queries[receiver_start:receiver_stop]
            common_arguments = (
                query_chunk,
                keys,
                values,
                residual_chunk,
                source,
                receiver,
                chunk_attributes,
            )

            if explanation_receivers is None:
                chunk_function = partial(
                    self._output_partition,
                    block,
                    receiver_start,
                )
                if use_checkpoint:
                    output = checkpoint(
                        chunk_function,
                        *common_arguments,
                        use_reentrant=False,
                    )
                else:
                    output = chunk_function(*common_arguments)
                outputs.append(output)
                continue

            chunk_function = partial(
                self._selected_explanation_partition,
                block,
                receiver_start,
            )
            explanation_arguments = (
                *common_arguments,
                explanation_receivers,
            )
            if use_checkpoint:
                output, attention, embeddings = checkpoint(
                    chunk_function,
                    *explanation_arguments,
                    use_reentrant=False,
                )
            else:
                output, attention, embeddings = chunk_function(
                    *explanation_arguments
                )
            outputs.append(output)
            selected_attention.append(attention)
            selected_edge_embeddings.append(embeddings)
            selected_edges = explanation_receivers.index_select(
                0, receiver
            )
            if original_ids is None:
                original_ids = torch.arange(
                    edge_start,
                    edge_stop,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            selected_edge_ids.append(original_ids[selected_edges])

        block_output = (
            torch.cat(outputs, dim=0) if outputs else node_embedding
        )
        if explanation_receivers is None:
            return block_output, None, None, None

        if selected_edge_ids:
            edge_ids = torch.cat(selected_edge_ids, dim=0)
            attention = torch.cat(selected_attention, dim=0)
            embeddings = torch.cat(selected_edge_embeddings, dim=0)
        else:
            edge_ids = edge_index.new_empty((0,))
            attention_dtype = (
                torch.float32
                if node_embedding.dtype
                in (torch.float16, torch.bfloat16)
                else node_embedding.dtype
            )
            attention = torch.empty(
                (0, block.attention_heads),
                dtype=attention_dtype,
                device=node_embedding.device,
            )
            embeddings = node_embedding.new_empty(
                (0, self.edge_embedding_dim)
            )
        restore_order = torch.argsort(edge_ids)
        return (
            block_output,
            attention.index_select(0, restore_order),
            embeddings.index_select(0, restore_order),
            edge_ids.index_select(0, restore_order),
        )

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attributes: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        *,
        attention_receivers: Optional[Tensor | Sequence[int]] = None,
    ) -> ModelOutput:
        num_nodes = input_expression.shape[0]
        prepared_edges = _prepare_edge_index(
            edge_index,
            num_nodes=num_nodes,
            device=input_expression.device,
        )
        prepared_attributes = _prepare_edge_attributes(
            edge_attributes,
            num_edges=prepared_edges.shape[1],
            edge_attribute_dim=self.edge_attribute_dim,
            reference=input_expression,
        )
        node_embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
        )
        layout = self._receiver_layout(
            prepared_edges,
            num_nodes=num_nodes,
        )
        explanation_receivers = self._explanation_receiver_mask(
            attention_receivers,
            return_explanations=return_explanations,
            num_nodes=num_nodes,
            device=input_expression.device,
        )

        final_attention: Optional[Tensor] = None
        final_edge_embedding: Optional[Tensor] = None
        final_edge_ids: Optional[Tensor] = None
        for layer_number, block in enumerate(self.blocks):
            is_last_layer = layer_number == len(self.blocks) - 1
            (
                node_embedding,
                layer_attention,
                layer_edge_embedding,
                layer_edge_ids,
            ) = self._chunked_block_forward(
                block=block,
                node_embedding=node_embedding,
                edge_index=prepared_edges,
                edge_attributes=prepared_attributes,
                layout=layout,
                explanation_receivers=(
                    explanation_receivers if is_last_layer else None
                ),
            )
            if is_last_layer:
                final_attention = layer_attention
                final_edge_embedding = layer_edge_embedding
                final_edge_ids = layer_edge_ids

        explanation_edges: Optional[Tensor] = None
        if return_explanations:
            assert final_edge_ids is not None
            explanation_edges = prepared_edges.index_select(
                1, final_edge_ids
            )
        selected_embedding = _select_targets(node_embedding, targets)
        prediction = self.decoder(selected_embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            attention_weights=final_attention,
            edge_embedding=final_edge_embedding,
            edge_index=explanation_edges,
        )


# Compact names for configuration and training code.
QKVGraphTransformer = EdgeAwareQKVGraphTransformer
DenseQKVGraphTransformer = ReceiverChunkedEdgeAwareQKVGraphTransformer


__all__ = [
    "DenseQKVGraphTransformer",
    "EdgeAwareQKVGraphTransformer",
    "QKVGraphTransformer",
    "QKVGraphTransformerBlock",
    "ReceiverChunkedEdgeAwareQKVGraphTransformer",
]

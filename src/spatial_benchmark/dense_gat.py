"""Exact receiver-chunked execution for the edge-conditioned GATv2 model.

The ordinary :class:`~spatial_benchmark.models.EdgeConditionedGATv2` encodes
every edge before entering a graph layer.  That is convenient for sparse
graphs, but its ``E x edge_embedding_dim`` tensor and GATv2 edge activations
are too large for the full-core high-k campaign.

This module keeps the model parameters and state-dict layout identical to the
ordinary G2 model while partitioning each graph layer by receiver.  A receiver
chunk contains *all* incoming edges for its nodes, so its softmax and aggregate
are the exact full-neighbor GATv2 operations; this is partitioning, not
neighbor sampling.  Raw edge attributes remain materialized, but shared edge
embeddings and attention activations are created only for the current chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter, softmax

from .models import (
    EdgeConditionedGATv2,
    ModelOutput,
    _normalize_target_nodes,
    _prepare_edge_attributes,
    _prepare_edge_index,
    _select_targets,
)


@dataclass
class _ReceiverLayout:
    """Cached stable receiver ordering for one immutable edge-index tensor."""

    edge_index: Tensor
    tensor_version: int
    num_nodes: int
    edge_order: Optional[Tensor]
    receiver_ptr: tuple[int, ...]


class ReceiverChunkedEdgeConditionedGATv2(EdgeConditionedGATv2):
    """G2 with exact, memory-bounded receiver-partitioned graph layers.

    Parameters beyond the ordinary G2 constructor:

    ``receiver_chunk_size``
        Maximum number of receiver nodes processed at once.  Every incoming
        edge for those nodes remains present.
    ``activation_checkpointing``
        Recompute each receiver chunk during backward instead of retaining its
        edge-sized activations.  Non-reentrant PyTorch checkpointing is used.

    The inherited parameters have exactly the same names and shapes as
    :class:`~spatial_benchmark.models.EdgeConditionedGATv2`, so either model's
    state dict can be loaded directly into the other.

    ``return_explanations=True`` preserves the ordinary small-graph behavior
    when ``attention_receivers`` is omitted.  On a dense graph, pass a small
    receiver selection: only its incoming final-layer edges, attention
    weights, and edge embeddings are returned.  Returned edges are restored
    to their order in the supplied ``edge_index``.
    """

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 1,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        edge_hidden_dim: int = 64,
        edge_embedding_dim: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        *,
        receiver_chunk_size: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        if receiver_chunk_size <= 0:
            raise ValueError("receiver_chunk_size must be positive")
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
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.activation_checkpointing = activation_checkpointing
        # This is execution metadata, not model state.  Holding the input
        # tensor by reference makes identity checks safe and avoids repeatedly
        # sorting a fixed 20M-edge graph over hundreds of epochs.
        self._receiver_layout_cache: Optional[_ReceiverLayout] = None

    def clear_edge_layout_cache(self) -> None:
        """Discard cached receiver ordering after replacing the graph."""

        self._receiver_layout_cache = None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        # Cached tensors are deliberately not registered buffers.  They are
        # derived from caller-owned graph tensors and must not follow model
        # device/dtype transformations.
        self.clear_edge_layout_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self) -> dict[str, object]:
        # ``state_dict`` already excludes the cache.  Also keep whole-module
        # pickles from embedding a potentially very large graph permutation.
        state = super().__getstate__()
        state["_receiver_layout_cache"] = None
        return state

    def _receiver_layout(
        self, edge_index: Tensor, *, num_nodes: int
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
        # Boundaries are consumed by Python's chunk loop.  Copying this small
        # N-sized tensor to host once avoids a device synchronization at every
        # receiver chunk.
        cumulative = torch.cumsum(counts, dim=0).cpu().tolist()
        receiver_ptr = (0, *(int(value) for value in cumulative))

        if receiver.numel() < 2 or not bool(
            (receiver[1:] < receiver[:-1]).any()
        ):
            edge_order = None
        else:
            # Stable sorting preserves the supplied order among edges that
            # enter the same receiver, minimizing floating-point differences
            # from the ordinary full-graph aggregate.
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

    def _attention_chunk(
        self,
        block: torch.nn.Module,
        source_projection: Tensor,
        receiver_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        local_receiver: Tensor,
        edge_attributes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate one complete incoming-edge partition."""

        convolution = block.convolution
        num_receivers = residual_embedding.shape[0]
        num_edges = source.shape[0]
        heads = convolution.heads
        head_dim = convolution.out_channels

        edge_embedding = self.edge_encoder(edge_attributes)
        if num_edges:
            projected_source = source_projection.index_select(0, source)
            projected_receiver = receiver_projection.index_select(
                0, local_receiver
            )
            attention_input = projected_source + projected_receiver
            if convolution.lin_edge is not None:
                projected_edge = convolution.lin_edge(edge_embedding).view(
                    num_edges, heads, head_dim
                )
                attention_input = attention_input + projected_edge
            attention_input = F.leaky_relu(
                attention_input, convolution.negative_slope
            )
            attention = (attention_input * convolution.att).sum(dim=-1)
            attention = softmax(
                attention,
                index=local_receiver,
                num_nodes=num_receivers,
            )
            attention = F.dropout(
                attention,
                p=convolution.dropout,
                training=convolution.training,
            )
            messages = projected_source * attention.unsqueeze(-1)
            convolution_output = scatter(
                messages,
                local_receiver,
                dim=0,
                dim_size=num_receivers,
                reduce="sum",
            )
        else:
            attention = residual_embedding.new_empty((0, heads))
            convolution_output = residual_embedding.new_zeros(
                (num_receivers, heads, head_dim)
            )

        if convolution.concat:
            convolution_output = convolution_output.view(
                num_receivers, heads * head_dim
            )
        else:  # Not used by the locked G2, retained for exact PyG semantics.
            convolution_output = convolution_output.mean(dim=1)
        if convolution.res is not None:
            convolution_output = convolution_output + convolution.res(
                residual_embedding
            )
        if convolution.bias is not None:
            convolution_output = convolution_output + convolution.bias

        convolution_output = block.output_projection(convolution_output)
        updated_embedding = block.attention_normalization(
            residual_embedding + block.dropout(convolution_output)
        )
        updated_embedding = block.feed_forward(updated_embedding)
        return updated_embedding, attention, edge_embedding

    def _output_chunk(
        self,
        block: torch.nn.Module,
        source_projection: Tensor,
        receiver_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        local_receiver: Tensor,
        edge_attributes: Tensor,
    ) -> Tensor:
        return self._attention_chunk(
            block,
            source_projection,
            receiver_projection,
            residual_embedding,
            source,
            local_receiver,
            edge_attributes,
        )[0]

    def _selected_explanation_chunk(
        self,
        block: torch.nn.Module,
        source_projection: Tensor,
        receiver_projection: Tensor,
        residual_embedding: Tensor,
        source: Tensor,
        local_receiver: Tensor,
        edge_attributes: Tensor,
        selected_edges: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, attention, edge_embedding = self._attention_chunk(
            block,
            source_projection,
            receiver_projection,
            residual_embedding,
            source,
            local_receiver,
            edge_attributes,
        )
        return (
            output,
            attention[selected_edges],
            edge_embedding[selected_edges],
        )

    def _chunked_block_forward(
        self,
        *,
        block: torch.nn.Module,
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
        convolution = block.convolution
        heads = convolution.heads
        head_dim = convolution.out_channels
        source_projection = convolution.lin_l(node_embedding).view(
            num_nodes, heads, head_dim
        )
        if convolution.share_weights:
            receiver_projection = source_projection
        else:
            receiver_projection = convolution.lin_r(node_embedding).view(
                num_nodes, heads, head_dim
            )

        outputs: list[Tensor] = []
        selected_edge_ids: list[Tensor] = []
        selected_attention: list[Tensor] = []
        selected_edge_embeddings: list[Tensor] = []
        use_checkpoint = (
            self.activation_checkpointing and torch.is_grad_enabled()
        )

        for receiver_start in range(0, num_nodes, self.receiver_chunk_size):
            receiver_end = min(
                receiver_start + self.receiver_chunk_size, num_nodes
            )
            edge_start = layout.receiver_ptr[receiver_start]
            edge_end = layout.receiver_ptr[receiver_end]
            if layout.edge_order is None:
                chunk_edges = edge_index[:, edge_start:edge_end]
                chunk_attributes = edge_attributes[edge_start:edge_end]
                original_ids: Optional[Tensor] = None
            else:
                original_ids = layout.edge_order[edge_start:edge_end]
                chunk_edges = edge_index.index_select(1, original_ids)
                chunk_attributes = edge_attributes.index_select(
                    0, original_ids
                )

            source = chunk_edges[0]
            local_receiver = chunk_edges[1] - receiver_start
            residual_chunk = node_embedding[receiver_start:receiver_end]
            receiver_projection_chunk = receiver_projection[
                receiver_start:receiver_end
            ]
            common_arguments = (
                source_projection,
                receiver_projection_chunk,
                residual_chunk,
                source,
                local_receiver,
                chunk_attributes,
            )

            if explanation_receivers is None:
                chunk_function = partial(self._output_chunk, block)
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

            selected_edges = explanation_receivers.index_select(
                0, chunk_edges[1]
            )
            chunk_function = partial(
                self._selected_explanation_chunk, block
            )
            explanation_arguments = (*common_arguments, selected_edges)
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
            if original_ids is None:
                original_ids = torch.arange(
                    edge_start,
                    edge_end,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            selected_edge_ids.append(original_ids[selected_edges])

        if outputs:
            block_output = torch.cat(outputs, dim=0)
        else:
            block_output = node_embedding
        if explanation_receivers is None:
            return block_output, None, None, None

        if selected_edge_ids:
            edge_ids = torch.cat(selected_edge_ids, dim=0)
            attention = torch.cat(selected_attention, dim=0)
            embeddings = torch.cat(selected_edge_embeddings, dim=0)
        else:
            edge_ids = edge_index.new_empty((0,))
            attention = node_embedding.new_empty((0, heads))
            embeddings = node_embedding.new_empty(
                (0, self.edge_embedding_dim)
            )
        # Receiver partitioning changes global edge order.  Explanations are
        # restored to the caller's original order for unambiguous alignment.
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
        if attention_receivers is not None and not return_explanations:
            raise ValueError(
                "attention_receivers requires return_explanations=True"
            )

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
            prepared_edges, num_nodes=num_nodes
        )

        explanation_mask: Optional[Tensor] = None
        if return_explanations:
            selected_receivers = _normalize_target_nodes(
                attention_receivers,
                num_nodes=num_nodes,
                device=input_expression.device,
            )
            explanation_mask = torch.zeros(
                num_nodes, dtype=torch.bool, device=input_expression.device
            )
            if selected_receivers is None:
                explanation_mask.fill_(True)
            elif selected_receivers.numel():
                explanation_mask[selected_receivers] = True

        attention = None
        edge_embedding = None
        explanation_edge_ids = None
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
                    explanation_mask if is_last_layer else None
                ),
            )
            if is_last_layer:
                attention = layer_attention
                edge_embedding = layer_edge_embedding
                explanation_edge_ids = layer_edge_ids

        selected_embedding = _select_targets(node_embedding, targets)
        prediction = self.decoder(selected_embedding)
        explanation_edges = None
        if return_explanations:
            assert explanation_edge_ids is not None
            explanation_edges = prepared_edges.index_select(
                1, explanation_edge_ids
            )
        return ModelOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            attention_weights=attention,
            edge_embedding=edge_embedding,
            edge_index=explanation_edges,
        )


# Compact alias for campaign configuration code.
DenseG2 = ReceiverChunkedEdgeConditionedGATv2


__all__ = [
    "DenseG2",
    "ReceiverChunkedEdgeConditionedGATv2",
]

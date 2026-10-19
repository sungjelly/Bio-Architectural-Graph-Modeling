r"""Strictly cell-autonomous control for the edge-aware QKV backbone.

The graph Transformer and this control have identical trainable module
layouts when constructed with the same dimensions.  The control deliberately
does not parse, validate, or otherwise consume ``edge_index`` or measured edge
attributes.  Instead, it gives every cell its own Q/K/V routing computation
and derives the input to the shared edge encoder deterministically from that
same cell's embedding.

For one cell and attention head, the within-cell update is

.. math::

    r_h = 2\,\sigma(Q_h(x)^T K'_h(x) / \sqrt{d_h} + b_h(e(x)))

.. math::

    u_h = r_h V'_h(x),

where ``e(x)`` is the shared edge encoder applied to a fixed cyclic view of
the cell embedding.  In ``vector`` mode, the edge key/value projections form
``K'`` and ``V'``; in ``bias_gate`` mode, the edge value projection is a
bounded multiplicative gate on ``V``.  The update then passes through the same
attention output projection, residual LayerNorms, and FFN as the graph model.

The sigmoid is intentional: a one-token softmax is identically one and would
make Q, K, and the attention-bias parameters dead.  This module is therefore
a parameter-count-matched nonlinear single-cell control, not a claim that
self-routing is graph attention.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import (
    ExpressionDecoder,
    ModelOutput,
    SharedEdgeEncoder,
    _BaseMaskedExpressionModel,
    _select_targets,
)
from .qkv_graph_transformer import QKVGraphTransformerBlock


class QKVParameterMatchedSelfControl(_BaseMaskedExpressionModel):
    """Cell-autonomous control exactly parameter-matched to QKV graph models.

    With the same model dimensions, this class and
    :class:`ReceiverChunkedEdgeAwareQKVGraphTransformer` have the same
    trainable parameter names, shapes, and total count.  All graph and
    measured-edge inputs are ignored.  Each attention and edge-conditioning
    parameter is instead active in a within-cell transformation.

    ``receiver_chunk_size`` is retained for constructor parity and bounds the
    number of independent cells evaluated at once.  This changes execution
    memory only; no receivers or edges are consulted.  Likewise,
    ``max_edges_per_chunk`` is accepted and validated for drop-in
    configuration parity but has no operation in a model with no edges.
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
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if edge_attribute_dim <= 0:
            raise ValueError("edge_attribute_dim must be positive")
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
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.max_edges_per_chunk = (
            None
            if max_edges_per_chunk is None
            else int(max_edges_per_chunk)
        )
        self.activation_checkpointing = activation_checkpointing

    def _cell_edge_surrogate(self, node_embedding: Tensor) -> Tensor:
        """Return a fixed, parameter-free within-cell edge-encoder input."""

        indices = torch.arange(
            self.edge_attribute_dim,
            dtype=torch.long,
            device=node_embedding.device,
        ).remainder(node_embedding.shape[1])
        return node_embedding.index_select(1, indices)

    def _self_only_block(
        self,
        block: QKVGraphTransformerBlock,
        node_embedding: Tensor,
    ) -> Tensor:
        """Apply every matched block parameter without inter-cell mixing."""

        queries, keys, values = block.project_nodes(node_embedding)
        num_cells = node_embedding.shape[0]
        heads = block.attention_heads
        head_dim = block.attention_head_dim
        edge_embedding = self.edge_encoder(
            self._cell_edge_surrogate(node_embedding)
        )

        if block.edge_conditioning_mode == "vector":
            assert block.edge_key_projection is not None
            assert block.edge_value_projection is not None
            edge_keys = block.edge_key_projection(edge_embedding).view(
                num_cells, heads, head_dim
            )
            edge_values = block.edge_value_projection(edge_embedding).view(
                num_cells, heads, head_dim
            )
            routed_keys = keys + edge_keys
            routed_values = values + edge_values
        else:
            assert block.edge_value_gate is not None
            routed_keys = keys
            value_gate = 1.0 + torch.tanh(
                block.edge_value_gate(edge_embedding)
            )
            routed_values = values * value_gate.unsqueeze(-1)

        routing_score = (
            torch.einsum("nhd,nhd->nh", queries, routed_keys)
            * block.attention_scale
            + block.edge_attention_bias(edge_embedding)
        )
        # A single-item softmax would erase all Q/K/bias dependence.  The
        # factor of two gives this gate a neutral initial scale near one.
        routing = 2.0 * torch.sigmoid(routing_score)
        routing = F.dropout(
            routing,
            p=block.attention_dropout_probability,
            training=block.training,
        )
        update = routed_values * routing.unsqueeze(-1)
        return block.finish_partition(node_embedding, update)

    def _run_blocks(self, node_embedding: Tensor) -> Tensor:
        for block in self.blocks:
            block_function = partial(self._self_only_block, block)
            if self.activation_checkpointing and torch.is_grad_enabled():
                node_embedding = checkpoint(
                    block_function,
                    node_embedding,
                    use_reentrant=False,
                )
            else:
                node_embedding = block_function(node_embedding)
        return node_embedding

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
        del edge_index, edge_attributes
        if attention_receivers is not None and not return_explanations:
            raise ValueError(
                "attention_receivers requires return_explanations=True"
            )

        node_embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
        )
        # Cells are independent, so selecting first is exactly equivalent to
        # selecting after all blocks and avoids work for non-target cells.
        node_embedding = _select_targets(node_embedding, targets)
        chunks = [
            self._run_blocks(
                node_embedding[start : start + self.receiver_chunk_size]
            )
            for start in range(
                0, node_embedding.shape[0], self.receiver_chunk_size
            )
        ]
        if chunks:
            node_embedding = torch.cat(chunks, dim=0)
        prediction = self.decoder(node_embedding)

        attention_weights: Optional[Tensor] = None
        edge_embedding: Optional[Tensor] = None
        explanation_edges: Optional[Tensor] = None
        if return_explanations:
            # A strict self-only model has no edge-level explanations.
            reference = node_embedding
            attention_weights = reference.new_empty(
                (0, self.blocks[-1].attention_heads)
            )
            edge_embedding = reference.new_empty(
                (0, self.edge_embedding_dim)
            )
            explanation_edges = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=reference.device,
            )

        return ModelOutput(
            prediction=prediction,
            node_embedding=node_embedding,
            attention_weights=attention_weights,
            edge_embedding=edge_embedding,
            edge_index=explanation_edges,
        )


# Compact configuration alias.
QKVMatchedSelfControl = QKVParameterMatchedSelfControl


__all__ = [
    "QKVMatchedSelfControl",
    "QKVParameterMatchedSelfControl",
]

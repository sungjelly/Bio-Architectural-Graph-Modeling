"""Categorical-count variant of the exact receiver-chunked G2 model.

The four prediction classes are the observed count categories ``0``, ``1``,
``2``, and ``>=3``.  Token ``4`` is reserved for masked inputs and is never an
output class.  Although :class:`~spatial_benchmark.training.GraphSplitView`
stores token IDs in a floating tensor, the encoder treats those IDs only as
categories: no token's numeric magnitude enters a projection.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dense_gat import ReceiverChunkedEdgeConditionedGATv2


NUM_COUNT_TOKENS = 4
MASK_TOKEN_ID = 4
NUM_INPUT_TOKENS = NUM_COUNT_TOKENS + 1


class CategoricalNodeEncoder(nn.Module):
    """Encode gene-wise categorical IDs without scalar token magnitudes."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        if node_covariate_dim < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.num_genes = num_genes
        self.node_covariate_dim = node_covariate_dim
        self.hidden_dim = hidden_dim
        self.num_input_tokens = NUM_INPUT_TOKENS
        self.mask_token_id = MASK_TOKEN_ID

        # Each projection receives one N x G indicator channel.  Constructing
        # and consuming channels serially avoids an N x G x K one-hot tensor.
        self.token_projections = nn.ModuleList(
            [
                nn.Linear(num_genes, hidden_dim, bias=False)
                for _ in range(NUM_INPUT_TOKENS)
            ]
        )
        self.covariate_projection = (
            nn.Linear(node_covariate_dim, hidden_dim, bias=False)
            if node_covariate_dim
            else None
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _validated_token_ids(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
    ) -> Tensor:
        if input_expression.ndim != 2:
            raise ValueError(
                "input_expression must have shape [num_nodes, num_genes]"
            )
        if input_expression.dtype != torch.float32:
            raise TypeError(
                "input_expression must be float32 categorical token IDs"
            )
        if input_expression.shape[1] != self.num_genes:
            raise ValueError(
                f"expected {self.num_genes} genes, got "
                f"{input_expression.shape[1]}"
            )
        if not isinstance(gene_mask, Tensor):
            raise TypeError("gene_mask must be a boolean tensor")
        if gene_mask.dtype != torch.bool:
            raise TypeError("gene_mask must be a boolean tensor")
        if gene_mask.shape != input_expression.shape:
            raise ValueError("gene_mask must have the same shape as expression")
        if not bool(torch.isfinite(input_expression).all()):
            raise ValueError("input_expression token IDs must be finite")
        if bool((input_expression != input_expression.round()).any()):
            raise ValueError("input_expression token IDs must be integer-valued")
        if bool((input_expression < 0).any()) or bool(
            (input_expression >= NUM_INPUT_TOKENS).any()
        ):
            raise ValueError(
                "input_expression token IDs must be in the range [0, 4]"
            )

        token_ids = input_expression.to(dtype=torch.long)
        mask_boolean = gene_mask.to(device=input_expression.device)
        # The explicit mask is authoritative even if the caller passes an
        # unmasked token tensor, preventing hidden target leakage.
        return token_ids.masked_fill(mask_boolean, MASK_TOKEN_ID)

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        node_covariates: Optional[Tensor] = None,
        broad_spatial_features: Optional[Tensor] = None,
    ) -> Tensor:
        token_ids = self._validated_token_ids(
            input_expression, gene_mask
        )

        embedding = self.bias
        for token_id, projection in enumerate(self.token_projections):
            indicator = (token_ids == token_id).to(
                dtype=input_expression.dtype
            )
            embedding = embedding + projection(indicator)

        num_nodes = input_expression.shape[0]
        if self.node_covariate_dim:
            if node_covariates is None:
                raise ValueError(
                    "node_covariates are required because "
                    f"node_covariate_dim={self.node_covariate_dim}"
                )
            expected = (num_nodes, self.node_covariate_dim)
            if node_covariates.shape != expected:
                raise ValueError(
                    f"node_covariates must have shape {expected}, got "
                    f"{tuple(node_covariates.shape)}"
                )
            covariates = node_covariates.to(
                device=input_expression.device,
                dtype=input_expression.dtype,
            )
            assert self.covariate_projection is not None
            embedding = embedding + self.covariate_projection(covariates)
        elif node_covariates is not None:
            expected = (num_nodes, 0)
            if node_covariates.shape != expected:
                raise ValueError(
                    "node_covariates were supplied, but this model was "
                    "constructed with node_covariate_dim=0"
                )

        if broad_spatial_features is not None:
            raise ValueError(
                "broad_spatial_features were supplied to a non-spatial encoder"
            )

        return self.dropout(F.gelu(self.normalization(embedding)))


class CategoricalExpressionDecoder(nn.Module):
    """Decode node embeddings into four logits for every gene."""

    def __init__(
        self,
        hidden_dim: int,
        num_genes: int,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        decoder_dim = (
            2 * hidden_dim if decoder_dim is None else decoder_dim
        )
        if decoder_dim <= 0:
            raise ValueError("decoder_dim must be positive")

        self.num_genes = num_genes
        self.num_output_tokens = NUM_COUNT_TOKENS
        self.linear_in = nn.Linear(hidden_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(
            decoder_dim, num_genes * NUM_COUNT_TOKENS
        )

    def forward(self, node_embedding: Tensor) -> Tensor:
        if node_embedding.ndim != 2:
            raise ValueError(
                "node_embedding must have shape [num_nodes, hidden_dim]"
            )
        logits = self.linear_out(
            self.dropout(F.gelu(self.linear_in(node_embedding)))
        )
        return logits.reshape(
            node_embedding.shape[0],
            self.num_genes,
            NUM_COUNT_TOKENS,
        )


class TokenizedReceiverChunkedEdgeConditionedGATv2(
    ReceiverChunkedEdgeConditionedGATv2
):
    """Exact receiver-chunked G2 with categorical inputs and outputs."""

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
        num_expression_tokens: int = NUM_COUNT_TOKENS,
        receiver_chunk_size: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
        if num_expression_tokens != NUM_COUNT_TOKENS:
            raise ValueError(
                "the fixed count vocabulary requires "
                f"num_expression_tokens={NUM_COUNT_TOKENS}"
            )
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
            receiver_chunk_size=receiver_chunk_size,
            activation_checkpointing=activation_checkpointing,
        )
        self.encoder = CategoricalNodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = CategoricalExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )


__all__ = [
    "CategoricalExpressionDecoder",
    "CategoricalNodeEncoder",
    "MASK_TOKEN_ID",
    "NUM_COUNT_TOKENS",
    "NUM_INPUT_TOKENS",
    "TokenizedReceiverChunkedEdgeConditionedGATv2",
]

"""Hybrid raw-count inputs and hurdle/ordinal outputs for full-core G2.

The vocabulary is fixed and requires no fitted token boundaries.  Exact raw
counts are retained in a parallel standardized ``log1p`` channel.  Explicit
gene masks are applied inside the encoder and therefore override whatever raw
value the caller supplies at a masked entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dense_gat import ReceiverChunkedEdgeConditionedGATv2
from .models import EdgeParameterMatchedSelfControl


NUM_COUNT_STATES = 8
MASK_TOKEN_ID = 8
NUM_INPUT_TOKENS = 9
NUM_POSITIVE_ORDINAL_THRESHOLDS = 6
NUM_OUTPUT_CHANNELS = 8
COUNT_BOUNDARIES = (1, 2, 3, 4, 8, 16, 32)
TOKEN_LABELS = (
    "count_0",
    "count_1",
    "count_2",
    "count_3",
    "count_4_to_7",
    "count_8_to_15",
    "count_16_to_31",
    "count_32_or_greater",
)


def validate_raw_counts(value: Any, *, name: str = "counts") -> np.ndarray:
    """Return a validated nonempty two-dimensional raw-count array.

    Integer-valued floating arrays are accepted because graph views store
    expression in floating tensors.  Boolean and complex arrays are rejected.
    """

    counts = np.asarray(value)
    if counts.ndim != 2 or not all(int(size) > 0 for size in counts.shape):
        raise ValueError(f"{name} must be nonempty with shape [cells, genes]")
    if counts.dtype.kind in "bcOSUV":
        raise TypeError(f"{name} must contain real numeric count values")
    if not np.isfinite(counts).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(counts < 0):
        raise ValueError(f"{name} must be nonnegative")
    if np.any(counts != np.rint(counts)):
        raise ValueError(f"{name} must be integer-valued")
    return counts


def tokenize_raw_counts(value: Any) -> np.ndarray:
    """Map raw counts to the frozen eight-state output vocabulary."""

    counts = validate_raw_counts(value)
    tokens = np.zeros(counts.shape, dtype=np.int64)
    for boundary in COUNT_BOUNDARIES:
        tokens += counts >= boundary
    return tokens


def _validate_raw_count_tensor(
    counts: Tensor,
    *,
    name: str,
    expected_genes: int | None = None,
) -> Tensor:
    if not isinstance(counts, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if counts.ndim != 2:
        raise ValueError(f"{name} must have shape [cells, genes]")
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError(f"{name} dimensions must be nonempty")
    if expected_genes is not None and counts.shape[1] != expected_genes:
        raise ValueError(
            f"{name} has {counts.shape[1]} genes; expected {expected_genes}"
        )
    if counts.dtype == torch.bool or counts.is_complex():
        raise TypeError(f"{name} must contain real numeric count values")
    if not counts.is_floating_point():
        counts = counts.to(dtype=torch.float32)
    if not bool(torch.isfinite(counts).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((counts < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    if bool((counts != counts.round()).any()):
        raise ValueError(f"{name} must be integer-valued")
    return counts


def tokenize_raw_count_tensor(counts: Tensor) -> Tensor:
    """Torch equivalent of :func:`tokenize_raw_counts`."""

    validated = _validate_raw_count_tensor(counts, name="counts")
    tokens = torch.zeros_like(validated, dtype=torch.long)
    for boundary in COUNT_BOUNDARIES:
        tokens = tokens + (validated >= boundary).to(dtype=torch.long)
    return tokens


def _validated_standardization(
    expression_mean: Any,
    expression_scale: Any,
    *,
    num_genes: int,
) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(expression_mean, dtype=torch.float32)
    scale = torch.as_tensor(expression_scale, dtype=torch.float32)
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError(
            "expression_mean and expression_scale must have shape [num_genes]"
        )
    if not bool(torch.isfinite(mean).all()) or not bool(
        torch.isfinite(scale).all()
    ):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")
    return mean, scale


def standardized_log1p_counts(
    counts: Tensor,
    expression_mean: Tensor,
    expression_scale: Tensor,
) -> Tensor:
    """Transform validated raw counts to the frozen all-fit gene scale."""

    validated = _validate_raw_count_tensor(
        counts,
        name="counts",
        expected_genes=int(expression_mean.numel()),
    )
    if expression_mean.shape != expression_scale.shape or expression_mean.ndim != 1:
        raise ValueError("expression standardization buffers are misaligned")
    mean = expression_mean.to(device=validated.device, dtype=validated.dtype)
    scale = expression_scale.to(device=validated.device, dtype=validated.dtype)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")
    return (torch.log1p(validated) - mean.unsqueeze(0)) / scale.unsqueeze(0)


class HybridCountNodeEncoder(nn.Module):
    """Encode gene-specific count tokens plus exact standardized log-counts."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int,
        hidden_dim: int,
        expression_mean: Any,
        expression_scale: Any,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_genes <= 0 or hidden_dim <= 0:
            raise ValueError("num_genes and hidden_dim must be positive")
        if node_covariate_dim < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        mean, scale = _validated_standardization(
            expression_mean,
            expression_scale,
            num_genes=num_genes,
        )
        self.num_genes = int(num_genes)
        self.node_covariate_dim = int(node_covariate_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_input_tokens = NUM_INPUT_TOKENS
        self.mask_token_id = MASK_TOKEN_ID
        self.register_buffer("expression_mean", mean, persistent=True)
        self.register_buffer("expression_scale", scale, persistent=True)

        # One G -> H projection per token gives every gene its own vector for
        # every state without materializing an N x G x K one-hot tensor.
        self.token_projections = nn.ModuleList(
            [
                nn.Linear(num_genes, hidden_dim, bias=False)
                for _ in range(NUM_INPUT_TOKENS)
            ]
        )
        self.continuous_projection = nn.Linear(
            num_genes, hidden_dim, bias=False
        )
        self.covariate_projection = (
            nn.Linear(node_covariate_dim, hidden_dim, bias=False)
            if node_covariate_dim
            else None
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def masked_inputs(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return authoritative discrete and continuous masked branches."""

        counts = _validate_raw_count_tensor(
            input_expression,
            name="input_expression",
            expected_genes=self.num_genes,
        )
        if not isinstance(gene_mask, Tensor) or gene_mask.dtype != torch.bool:
            raise TypeError("gene_mask must be a boolean tensor")
        if gene_mask.shape != counts.shape:
            raise ValueError("gene_mask must match input_expression")
        mask = gene_mask.to(device=counts.device)
        tokens = tokenize_raw_count_tensor(counts).masked_fill(
            mask, MASK_TOKEN_ID
        )
        continuous = standardized_log1p_counts(
            counts,
            self.expression_mean,
            self.expression_scale,
        ).masked_fill(mask, 0.0)
        return tokens, continuous

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        node_covariates: Optional[Tensor] = None,
        broad_spatial_features: Optional[Tensor] = None,
    ) -> Tensor:
        tokens, continuous = self.masked_inputs(input_expression, gene_mask)
        embedding = self.bias
        for token_id, projection in enumerate(self.token_projections):
            indicator = (tokens == token_id).to(dtype=self.bias.dtype)
            embedding = embedding + projection(indicator)
        embedding = embedding + self.continuous_projection(
            continuous.to(dtype=self.bias.dtype)
        )

        num_nodes = input_expression.shape[0]
        if self.node_covariate_dim:
            if node_covariates is None:
                raise ValueError("node_covariates are required")
            expected = (num_nodes, self.node_covariate_dim)
            if tuple(node_covariates.shape) != expected:
                raise ValueError(
                    f"node_covariates must have shape {expected}, got "
                    f"{tuple(node_covariates.shape)}"
                )
            covariates = node_covariates.to(
                device=input_expression.device,
                dtype=self.bias.dtype,
            )
            if not bool(torch.isfinite(covariates).all()):
                raise ValueError("node_covariates must be finite")
            assert self.covariate_projection is not None
            embedding = embedding + self.covariate_projection(covariates)
        elif node_covariates is not None and tuple(node_covariates.shape) != (
            num_nodes,
            0,
        ):
            raise ValueError(
                "node_covariates were supplied to a zero-covariate encoder"
            )
        if broad_spatial_features is not None:
            raise ValueError(
                "broad_spatial_features are prohibited in the hybrid encoder"
            )
        return self.dropout(F.gelu(self.normalization(embedding)))


class HybridCountDecoder(nn.Module):
    """Decode detection, six positive ordinal, and continuous channels."""

    def __init__(
        self,
        hidden_dim: int,
        num_genes: int,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_genes <= 0:
            raise ValueError("hidden_dim and num_genes must be positive")
        resolved_decoder = (
            2 * hidden_dim if decoder_dim is None else int(decoder_dim)
        )
        if resolved_decoder <= 0:
            raise ValueError("decoder_dim must be positive")
        self.num_genes = int(num_genes)
        self.linear_in = nn.Linear(hidden_dim, resolved_decoder)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(
            resolved_decoder, num_genes * NUM_OUTPUT_CHANNELS
        )

    def forward(self, node_embedding: Tensor) -> Tensor:
        if node_embedding.ndim != 2:
            raise ValueError("node_embedding must have shape [nodes, hidden]")
        flat = self.linear_out(
            self.dropout(F.gelu(self.linear_in(node_embedding)))
        )
        return flat.reshape(
            node_embedding.shape[0],
            self.num_genes,
            NUM_OUTPUT_CHANNELS,
        )


def split_hybrid_prediction(
    prediction: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split a validated hybrid output tensor into its three heads."""

    if prediction.ndim != 3 or prediction.shape[-1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(
            "prediction must have shape [nodes, genes, 8]"
        )
    return prediction[..., 0], prediction[..., 1:7], prediction[..., 7]


def decode_positive_states(ordinal_logits: Tensor) -> Tensor:
    """Decode positive states 1-7 by cumulative threshold crossings."""

    if ordinal_logits.shape[-1] != NUM_POSITIVE_ORDINAL_THRESHOLDS:
        raise ValueError("ordinal_logits must end in six thresholds")
    return 1 + (torch.sigmoid(ordinal_logits) >= 0.5).sum(dim=-1)


def decode_count_states(prediction: Tensor) -> Tensor:
    """Decode the complete zero plus seven-positive-state vocabulary."""

    detection_logits, ordinal_logits, _ = split_hybrid_prediction(prediction)
    positive = decode_positive_states(ordinal_logits)
    return torch.where(
        torch.sigmoid(detection_logits) >= 0.5,
        positive,
        torch.zeros_like(positive),
    )


class HybridReceiverChunkedEdgeConditionedGATv2(
    ReceiverChunkedEdgeConditionedGATv2
):
    """Exact receiver-partitioned G2 with the frozen hybrid count heads."""

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        expression_mean: Any,
        expression_scale: Any,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 2,
        ffn_dim: Optional[int] = 512,
        decoder_dim: Optional[int] = 512,
        edge_hidden_dim: int = 64,
        edge_embedding_dim: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        *,
        receiver_chunk_size: int = 512,
        activation_checkpointing: bool = True,
    ) -> None:
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
        self.encoder = HybridCountNodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            dropout=dropout,
        )
        self.decoder = HybridCountDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )


class HybridEdgeParameterMatchedSelfControl(EdgeParameterMatchedSelfControl):
    """Strict cell-autonomous control matched to the hybrid G2 parameter count."""

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        expression_mean: Any,
        expression_scale: Any,
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 2,
        ffn_dim: Optional[int] = 512,
        decoder_dim: Optional[int] = 512,
        edge_hidden_dim: int = 64,
        edge_embedding_dim: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
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
        self.encoder = HybridCountNodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            dropout=dropout,
        )
        self.decoder = HybridCountDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )


@dataclass(frozen=True)
class HybridCountLoss:
    """Differentiable equal-weight hurdle objective and its components."""

    total: Tensor
    detection: Tensor
    ordinal: Tensor
    positive_continuous_huber: Tensor
    n_masked: int
    n_zero: int
    n_positive: int


def hybrid_count_hurdle_loss(
    prediction: Tensor,
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Tensor,
    expression_scale: Tensor,
    huber_delta: float = 1.0,
) -> HybridCountLoss:
    """Compute the frozen equal-weight balanced hurdle objective.

    Expected detection and every ordinal stratum must be present.  An absent
    stratum raises instead of silently contributing zero.
    """

    if huber_delta != 1.0:
        raise ValueError("the frozen positive Huber delta is exactly 1.0")
    target = _validate_raw_count_tensor(raw_target, name="raw_target")
    if target_mask.dtype != torch.bool or target_mask.shape != target.shape:
        raise ValueError("target_mask must be boolean and match raw_target")
    expected = (*target.shape, NUM_OUTPUT_CHANNELS)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            f"prediction shape mismatch: expected {expected}, got "
            f"{tuple(prediction.shape)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("hybrid prediction contains non-finite values")
    if expression_mean.shape != (target.shape[1],) or expression_scale.shape != (
        target.shape[1],
    ):
        raise ValueError("expression standardization must match target genes")
    if not bool(target_mask.any()):
        raise ValueError("hybrid loss requires at least one masked target")

    detection_logits, ordinal_logits, continuous_prediction = (
        split_hybrid_prediction(prediction)
    )
    selected_counts = target[target_mask]
    selected_detection = detection_logits[target_mask].float()
    zero = selected_counts == 0
    positive = selected_counts > 0
    if not bool(zero.any()) or not bool(positive.any()):
        raise ValueError(
            "balanced detection loss requires zero and positive strata"
        )
    detection_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            selected_detection[zero],
            torch.zeros_like(selected_detection[zero]),
        )
        + F.binary_cross_entropy_with_logits(
            selected_detection[positive],
            torch.ones_like(selected_detection[positive]),
        )
    )

    selected_tokens = tokenize_raw_count_tensor(target)[target_mask]
    selected_ordinal = ordinal_logits[target_mask].float()
    positive_tokens = selected_tokens[positive]
    positive_ordinal = selected_ordinal[positive]
    threshold_losses: list[Tensor] = []
    for threshold_index in range(NUM_POSITIVE_ORDINAL_THRESHOLDS):
        threshold_state = threshold_index + 1
        above = positive_tokens > threshold_state
        below_or_equal = ~above
        if not bool(above.any()) or not bool(below_or_equal.any()):
            raise ValueError(
                "balanced ordinal loss is missing a stratum at positive "
                f"threshold {threshold_state}"
            )
        logits = positive_ordinal[:, threshold_index]
        threshold_losses.append(
            0.5
            * (
                F.binary_cross_entropy_with_logits(
                    logits[below_or_equal],
                    torch.zeros_like(logits[below_or_equal]),
                )
                + F.binary_cross_entropy_with_logits(
                    logits[above],
                    torch.ones_like(logits[above]),
                )
            )
        )
    ordinal_loss = torch.stack(threshold_losses).mean()

    standardized_target = standardized_log1p_counts(
        target,
        expression_mean,
        expression_scale,
    )[target_mask]
    selected_continuous = continuous_prediction[target_mask].float()
    positive_continuous = F.huber_loss(
        selected_continuous[positive],
        standardized_target[positive].float(),
        reduction="mean",
        delta=1.0,
    )
    total = (detection_loss + ordinal_loss + positive_continuous) / 3.0
    return HybridCountLoss(
        total=total,
        detection=detection_loss,
        ordinal=ordinal_loss,
        positive_continuous_huber=positive_continuous,
        n_masked=int(selected_counts.numel()),
        n_zero=int(zero.sum().detach().cpu()),
        n_positive=int(positive.sum().detach().cpu()),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable scalar parameters without relying on config arithmetic."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def assert_exact_parameter_match(graph_model: nn.Module, self_model: nn.Module) -> int:
    """Fail closed unless graph and self arms have identical trainable counts."""

    graph_count = trainable_parameter_count(graph_model)
    self_count = trainable_parameter_count(self_model)
    if graph_count != self_count:
        raise ValueError(
            "hybrid graph/self parameter mismatch: "
            f"graph={graph_count}, self={self_count}"
        )
    return graph_count


__all__ = [
    "COUNT_BOUNDARIES",
    "HybridCountDecoder",
    "HybridCountLoss",
    "HybridCountNodeEncoder",
    "HybridEdgeParameterMatchedSelfControl",
    "HybridReceiverChunkedEdgeConditionedGATv2",
    "MASK_TOKEN_ID",
    "NUM_COUNT_STATES",
    "NUM_INPUT_TOKENS",
    "NUM_OUTPUT_CHANNELS",
    "NUM_POSITIVE_ORDINAL_THRESHOLDS",
    "TOKEN_LABELS",
    "assert_exact_parameter_match",
    "decode_count_states",
    "decode_positive_states",
    "hybrid_count_hurdle_loss",
    "split_hybrid_prediction",
    "standardized_log1p_counts",
    "tokenize_raw_count_tensor",
    "tokenize_raw_counts",
    "trainable_parameter_count",
    "validate_raw_counts",
]

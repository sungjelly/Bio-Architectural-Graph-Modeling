r"""Geometry-modulated cosine QKV attention with routing-only geometry.

For every directed edge ``j -> i`` and attention head ``h`` this module uses

.. math::

    s_{ijh} = \tau_h \sum_l g_{ijhl}\hat q_{ihl}\hat k_{jhl}
               + \beta_{ijh}.

Queries and keys are L2-normalized per head.  A shared geometry trunk within
each graph block maps the fixed 70-dimensional relative-geometry vector to a
positive, dimension-wise modulation ``g`` and a bounded scalar bias ``beta``.
The modulation is normalized to mean one over the head dimension.  Geometry
never enters values, so it controls routing without changing message content.

The geometry output projections are bias-free and exactly zero-initialized.
Consequently every block starts with ``g == 1`` and ``beta == 0``.  The
positive per-head logit scale is smoothly bounded to avoid runaway cosine
attention.  Q/K normalization and all edge-score arithmetic use FP32.

The full-edge implementation is a transparent reference.  The receiver-
chunked implementation inherits the exact partitioning machinery from the
existing Relative-QKV model; changing chunk size changes memory use, not the
attention operator.
"""

from __future__ import annotations

from math import isfinite, log
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relative_qkv_graph_transformer import (
    RELATIVE_GEOMETRY_DIM,
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    RelativeGeometryQKVGraphTransformer,
    _incoming_softmax,
    _transformer_ffn_dim,
    _validate_probability,
)
from .models import _validate_hidden_dimensions


DEFAULT_QK_NORMALIZATION_EPSILON = 1e-6
DEFAULT_LOGIT_SCALE_INITIAL = 1.8856180831641267
DEFAULT_LOGIT_SCALE_MINIMUM = 0.1
DEFAULT_LOGIT_SCALE_MAXIMUM = 20.0
DEFAULT_MODULATION_AMPLITUDE = 0.5
DEFAULT_BIAS_BOUND = 1.0


def _validate_finite_positive(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _bounded_scale_parameter(
    *,
    initial: float,
    minimum: float,
    maximum: float,
) -> float:
    minimum = _validate_finite_positive(minimum, name="logit_scale_minimum")
    maximum = _validate_finite_positive(maximum, name="logit_scale_maximum")
    initial = _validate_finite_positive(initial, name="logit_scale_initial")
    if maximum <= minimum:
        raise ValueError("logit_scale_maximum must exceed logit_scale_minimum")
    if not minimum < initial < maximum:
        raise ValueError(
            "logit_scale_initial must lie strictly between its bounds"
        )
    fraction = (initial - minimum) / (maximum - minimum)
    return log(fraction / (1.0 - fraction))


class GeometryModulationBiasEncoder(nn.Module):
    """Map relative geometry to mean-one modulation and bounded bias."""

    def __init__(
        self,
        *,
        attention_heads: int,
        attention_head_dim: int,
        hidden_dim: int = 128,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
        modulation_amplitude: float = DEFAULT_MODULATION_AMPLITUDE,
        bias_bound: float = DEFAULT_BIAS_BOUND,
        normalization_epsilon: float = DEFAULT_QK_NORMALIZATION_EPSILON,
    ) -> None:
        super().__init__()
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if relative_geometry_dim != RELATIVE_GEOMETRY_DIM:
            raise ValueError(
                "relative_geometry_dim is fixed at "
                f"{RELATIVE_GEOMETRY_DIM}"
            )
        modulation_amplitude = _validate_finite_positive(
            modulation_amplitude,
            name="modulation_amplitude",
        )
        if modulation_amplitude >= 1.0:
            raise ValueError(
                "modulation_amplitude must be less than one so modulation "
                "remains strictly positive"
            )
        bias_bound = _validate_finite_positive(bias_bound, name="bias_bound")
        normalization_epsilon = _validate_finite_positive(
            normalization_epsilon,
            name="normalization_epsilon",
        )

        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.hidden_dim = int(hidden_dim)
        self.relative_geometry_dim = RELATIVE_GEOMETRY_DIM
        self.modulation_amplitude = modulation_amplitude
        self.bias_bound = bias_bound
        self.normalization_epsilon = normalization_epsilon

        self.input_projection = nn.Linear(RELATIVE_GEOMETRY_DIM, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.modulation_projection = nn.Linear(
            hidden_dim,
            self.attention_heads * self.attention_head_dim,
            bias=False,
        )
        self.bias_projection = nn.Linear(
            hidden_dim,
            self.attention_heads,
            bias=False,
        )
        nn.init.zeros_(self.modulation_projection.weight)
        nn.init.zeros_(self.bias_projection.weight)

    def forward(self, relative_geometry: Tensor) -> tuple[Tensor, Tensor]:
        if (
            relative_geometry.ndim != 2
            or relative_geometry.shape[1] != self.relative_geometry_dim
        ):
            raise ValueError(
                "relative_geometry must have shape "
                f"[num_edges, {self.relative_geometry_dim}]"
            )

        hidden = self.input_projection(relative_geometry)
        hidden = F.gelu(self.normalization(hidden))
        num_edges = relative_geometry.shape[0]
        modulation_logits = self.modulation_projection(hidden).view(
            num_edges,
            self.attention_heads,
            self.attention_head_dim,
        )
        raw_modulation = 1.0 + self.modulation_amplitude * torch.tanh(
            modulation_logits.float()
        )
        modulation_mean = raw_modulation.mean(dim=-1, keepdim=True)
        modulation = raw_modulation / modulation_mean.clamp_min(
            self.normalization_epsilon
        )
        positional_bias = self.bias_bound * torch.tanh(
            self.bias_projection(hidden).float()
        )
        return modulation, positional_bias


class GeometryModulatedRelativeQKVGraphTransformerBlock(nn.Module):
    """Pre-LayerNorm block using geometry-modulated cosine attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int] = None,
        geometry_hidden_dim: int = 128,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        qk_normalization_epsilon: float = DEFAULT_QK_NORMALIZATION_EPSILON,
        logit_scale_initial: float = DEFAULT_LOGIT_SCALE_INITIAL,
        logit_scale_minimum: float = DEFAULT_LOGIT_SCALE_MINIMUM,
        logit_scale_maximum: float = DEFAULT_LOGIT_SCALE_MAXIMUM,
        modulation_amplitude: float = DEFAULT_MODULATION_AMPLITUDE,
        geometry_bias_bound: float = DEFAULT_BIAS_BOUND,
    ) -> None:
        super().__init__()
        dropout = _validate_probability(dropout, name="dropout")
        attention_dropout = _validate_probability(
            attention_dropout,
            name="attention_dropout",
        )
        qk_normalization_epsilon = _validate_finite_positive(
            qk_normalization_epsilon,
            name="qk_normalization_epsilon",
        )
        head_dim, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim,
            attention_heads,
            attention_head_dim,
        )
        expanded_ffn_dim = _transformer_ffn_dim(hidden_dim, ffn_dim)
        raw_initial_scale = _bounded_scale_parameter(
            initial=logit_scale_initial,
            minimum=logit_scale_minimum,
            maximum=logit_scale_maximum,
        )

        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(head_dim)
        self.raw_attention_dim = int(raw_attention_dim)
        self.attention_dropout_probability = attention_dropout
        self.qk_normalization_epsilon = qk_normalization_epsilon
        self.logit_scale_initial = float(logit_scale_initial)
        self.logit_scale_minimum = float(logit_scale_minimum)
        self.logit_scale_maximum = float(logit_scale_maximum)

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
        self.geometry_encoder = GeometryModulationBiasEncoder(
            attention_heads=attention_heads,
            attention_head_dim=head_dim,
            hidden_dim=geometry_hidden_dim,
            modulation_amplitude=modulation_amplitude,
            bias_bound=geometry_bias_bound,
            normalization_epsilon=qk_normalization_epsilon,
        )
        self.raw_attention_logit_scale = nn.Parameter(
            torch.full(
                (self.attention_heads,),
                raw_initial_scale,
                dtype=torch.float32,
            )
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

    def attention_logit_scale(self) -> Tensor:
        """Return smoothly bounded positive per-head logit scales in FP32."""

        raw_scale = self.raw_attention_logit_scale.float()
        return self.logit_scale_minimum + (
            self.logit_scale_maximum - self.logit_scale_minimum
        ) * torch.sigmoid(raw_scale)

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
        queries = self.query_projection(normalized).view(shape).float()
        keys = self.key_projection(normalized).view(shape).float()
        values = self.value_projection(normalized).view(shape)
        return (
            F.normalize(
                queries,
                p=2.0,
                dim=-1,
                eps=self.qk_normalization_epsilon,
            ),
            F.normalize(
                keys,
                p=2.0,
                dim=-1,
                eps=self.qk_normalization_epsilon,
            ),
            values,
        )

    def score_edges(
        self,
        receiver_queries: Tensor,
        sender_keys: Tensor,
        relative_geometry: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return modulated cosine content logits and geometry-only bias."""

        if receiver_queries.shape != sender_keys.shape:
            raise ValueError("receiver queries and sender keys must align")
        expected_tail = (self.attention_heads, self.attention_head_dim)
        if (
            receiver_queries.ndim != 3
            or receiver_queries.shape[1:] != expected_tail
        ):
            raise ValueError(
                "edge queries and keys must have shape "
                f"[num_edges, {self.attention_heads}, "
                f"{self.attention_head_dim}]"
            )
        modulation, positional_bias = self.geometry_encoder(relative_geometry)
        pairwise_products = receiver_queries.float() * sender_keys.float()
        content_logits = self.attention_logit_scale().view(1, -1) * (
            modulation * pairwise_products
        ).sum(dim=-1)
        return content_logits, positional_bias

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


class _GeometryModulatedAttentionMixin:
    """Override only the edge-score calculation of Relative-QKV models."""

    def _attention_partition(
        self,
        block: GeometryModulatedRelativeQKVGraphTransformerBlock,
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
        num_receivers = residual_embedding.shape[0]
        num_edges = source.shape[0]
        heads = block.attention_heads
        head_dim = block.attention_head_dim
        local_receiver = receiver - receiver_start

        if num_edges:
            receiver_queries = query_projection.index_select(0, local_receiver)
            sender_keys = key_projection.index_select(0, source)
            content_logits, positional_bias = block.score_edges(
                receiver_queries,
                sender_keys,
                relative_geometry,
            )
            combined_logits = content_logits + positional_bias
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


def _geometry_modulated_blocks(
    *,
    graph_layers: int,
    hidden_dim: int,
    attention_heads: int,
    attention_head_dim: Optional[int],
    geometry_hidden_dim: int,
    ffn_dim: Optional[int],
    dropout: float,
    attention_dropout: float,
    qk_normalization_epsilon: float,
    logit_scale_initial: float,
    logit_scale_minimum: float,
    logit_scale_maximum: float,
    modulation_amplitude: float,
    geometry_bias_bound: float,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            GeometryModulatedRelativeQKVGraphTransformerBlock(
                hidden_dim=hidden_dim,
                attention_heads=attention_heads,
                attention_head_dim=attention_head_dim,
                geometry_hidden_dim=geometry_hidden_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                qk_normalization_epsilon=qk_normalization_epsilon,
                logit_scale_initial=logit_scale_initial,
                logit_scale_minimum=logit_scale_minimum,
                logit_scale_maximum=logit_scale_maximum,
                modulation_amplitude=modulation_amplitude,
                geometry_bias_bound=geometry_bias_bound,
            )
            for _ in range(graph_layers)
        ]
    )


def _record_geometry_modulated_architecture(
    model: nn.Module,
    *,
    graph_layers: int,
    geometry_hidden_dim: int,
    qk_normalization_epsilon: float,
    logit_scale_initial: float,
    logit_scale_minimum: float,
    logit_scale_maximum: float,
    modulation_amplitude: float,
    geometry_bias_bound: float,
) -> None:
    model.unique_graph_blocks = int(graph_layers)  # type: ignore[attr-defined]
    model.effective_graph_depth = int(graph_layers)  # type: ignore[attr-defined]
    model.graph_block_weight_tying = "none"  # type: ignore[attr-defined]
    model.attention_score_mechanism = (  # type: ignore[attr-defined]
        "geometry_modulated_cosine_qkv_v1"
    )
    model.qk_l2_normalized = True  # type: ignore[attr-defined]
    model.geometry_values_enabled = False  # type: ignore[attr-defined]
    model.geometry_hidden_dim = int(geometry_hidden_dim)  # type: ignore[attr-defined]
    model.qk_normalization_epsilon = float(  # type: ignore[attr-defined]
        qk_normalization_epsilon
    )
    model.logit_scale_initial = float(logit_scale_initial)  # type: ignore[attr-defined]
    model.logit_scale_minimum = float(logit_scale_minimum)  # type: ignore[attr-defined]
    model.logit_scale_maximum = float(logit_scale_maximum)  # type: ignore[attr-defined]
    model.modulation_amplitude = float(modulation_amplitude)  # type: ignore[attr-defined]
    model.geometry_bias_bound = float(geometry_bias_bound)  # type: ignore[attr-defined]


class GeometryModulatedRelativeQKVGraphTransformer(
    _GeometryModulatedAttentionMixin,
    RelativeGeometryQKVGraphTransformer,
):
    """Transparent full-edge geometry-modulated cosine-QKV reference."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = 1024,
        decoder_dim: Optional[int] = 1024,
        geometry_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
        qk_normalization_epsilon: float = DEFAULT_QK_NORMALIZATION_EPSILON,
        logit_scale_initial: float = DEFAULT_LOGIT_SCALE_INITIAL,
        logit_scale_minimum: float = DEFAULT_LOGIT_SCALE_MINIMUM,
        logit_scale_maximum: float = DEFAULT_LOGIT_SCALE_MAXIMUM,
        modulation_amplitude: float = DEFAULT_MODULATION_AMPLITUDE,
        geometry_bias_bound: float = DEFAULT_BIAS_BOUND,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            graph_layers=graph_layers,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            positional_bias_hidden_dim=geometry_hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relative_geometry_dim=relative_geometry_dim,
        )
        self.blocks = _geometry_modulated_blocks(
            graph_layers=graph_layers,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            geometry_hidden_dim=geometry_hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
        )
        _record_geometry_modulated_architecture(
            self,
            graph_layers=graph_layers,
            geometry_hidden_dim=geometry_hidden_dim,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
        )


class ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer(
    _GeometryModulatedAttentionMixin,
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
):
    """Memory-bounded exact geometry-modulated cosine-QKV model."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        graph_layers: int = 4,
        ffn_dim: Optional[int] = 1024,
        decoder_dim: Optional[int] = 1024,
        geometry_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        relative_geometry_dim: int = RELATIVE_GEOMETRY_DIM,
        qk_normalization_epsilon: float = DEFAULT_QK_NORMALIZATION_EPSILON,
        logit_scale_initial: float = DEFAULT_LOGIT_SCALE_INITIAL,
        logit_scale_minimum: float = DEFAULT_LOGIT_SCALE_MINIMUM,
        logit_scale_maximum: float = DEFAULT_LOGIT_SCALE_MAXIMUM,
        modulation_amplitude: float = DEFAULT_MODULATION_AMPLITUDE,
        geometry_bias_bound: float = DEFAULT_BIAS_BOUND,
        *,
        receiver_chunk_size: int = 128,
        max_edges_per_chunk: Optional[int] = 50_000,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            graph_layers=graph_layers,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            positional_bias_hidden_dim=geometry_hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relative_geometry_dim=relative_geometry_dim,
            receiver_chunk_size=receiver_chunk_size,
            max_edges_per_chunk=max_edges_per_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        self.blocks = _geometry_modulated_blocks(
            graph_layers=graph_layers,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            geometry_hidden_dim=geometry_hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
        )
        _record_geometry_modulated_architecture(
            self,
            graph_layers=graph_layers,
            geometry_hidden_dim=geometry_hidden_dim,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
        )


# Compact aliases for exploratory callers.
GeometryModulatedQKVGraphTransformerBlock = (
    GeometryModulatedRelativeQKVGraphTransformerBlock
)
GeometryModulatedQKVGraphTransformer = GeometryModulatedRelativeQKVGraphTransformer
ReceiverChunkedGeometryModulatedQKVGraphTransformer = (
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer
)


__all__ = [
    "DEFAULT_BIAS_BOUND",
    "DEFAULT_LOGIT_SCALE_INITIAL",
    "DEFAULT_LOGIT_SCALE_MAXIMUM",
    "DEFAULT_LOGIT_SCALE_MINIMUM",
    "DEFAULT_MODULATION_AMPLITUDE",
    "DEFAULT_QK_NORMALIZATION_EPSILON",
    "GeometryModulatedQKVGraphTransformer",
    "GeometryModulatedQKVGraphTransformerBlock",
    "GeometryModulatedRelativeQKVGraphTransformer",
    "GeometryModulatedRelativeQKVGraphTransformerBlock",
    "GeometryModulationBiasEncoder",
    "ReceiverChunkedGeometryModulatedQKVGraphTransformer",
    "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer",
]

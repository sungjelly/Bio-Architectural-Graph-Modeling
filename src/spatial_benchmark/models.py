"""Nested model ladder for masked spatial-expression prediction.

All graph tensors in this module use the PyTorch Geometric convention
``edge_index[0] = source`` and ``edge_index[1] = receiver``.  Graph models
disable implicit self loops and reject explicit self loops so attention and
edge-message outputs remain aligned one-for-one with the supplied edges.

The models deliberately contain no node- or edge-identity embeddings.  Cell
metadata enters only through a shared projection in :class:`NodeEncoder` and
is never altered by the expression mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spatial_field import BroadSpatialFieldBasis

try:  # Keep the self-only controls importable in CPU/minimal environments.
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover - exercised only without the dependency.
    GATv2Conv = None  # type: ignore[assignment]


@dataclass
class ModelOutput:
    """Uniform output from every model in the ablation ladder.

    ``prediction`` and ``node_embedding`` contain only ``target_nodes`` when
    that optional argument was supplied.  Edge-level explanation tensors
    describe all edges supplied to the forward pass and are aligned with
    ``edge_index``.  They are populated only when ``return_explanations=True``.
    """

    prediction: Tensor
    node_embedding: Tensor
    attention_weights: Optional[Tensor] = None
    edge_embedding: Optional[Tensor] = None
    edge_message: Optional[Tensor] = None
    self_prediction: Optional[Tensor] = None
    neighbor_prediction: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None


def _default_expansion(hidden_dim: int, value: Optional[int]) -> int:
    return 2 * hidden_dim if value is None else value


def _validate_hidden_dimensions(
    hidden_dim: int,
    attention_heads: int,
    attention_head_dim: Optional[int],
) -> tuple[int, int]:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if attention_heads <= 0:
        raise ValueError("attention_heads must be positive")
    if attention_head_dim is None:
        if hidden_dim % attention_heads:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads when "
                "attention_head_dim is not supplied"
            )
        attention_head_dim = hidden_dim // attention_heads
    if attention_head_dim <= 0:
        raise ValueError("attention_head_dim must be positive")
    return attention_head_dim, attention_heads * attention_head_dim


def _normalize_target_nodes(
    target_nodes: Optional[Tensor | Sequence[int]],
    *,
    num_nodes: int,
    device: torch.device,
) -> Optional[Tensor]:
    if target_nodes is None:
        return None
    if isinstance(target_nodes, Tensor):
        if target_nodes.dtype == torch.bool:
            if target_nodes.ndim != 1 or target_nodes.numel() != num_nodes:
                raise ValueError(
                    "a boolean target_nodes mask must have shape [num_nodes]"
                )
            target_nodes = target_nodes.to(device=device).nonzero(
                as_tuple=False
            ).flatten()
        else:
            if target_nodes.ndim != 1:
                raise ValueError("target_nodes must be one-dimensional")
            target_nodes = target_nodes.to(device=device, dtype=torch.long)
    else:
        target_nodes = torch.as_tensor(
            target_nodes, dtype=torch.long, device=device
        )
        if target_nodes.ndim != 1:
            raise ValueError("target_nodes must be one-dimensional")
    if target_nodes.numel():
        if bool((target_nodes < 0).any()) or bool(
            (target_nodes >= num_nodes).any()
        ):
            raise ValueError("target_nodes contains an out-of-range node index")
    return target_nodes


def _select_targets(value: Tensor, target_nodes: Optional[Tensor]) -> Tensor:
    return value if target_nodes is None else value.index_select(0, target_nodes)


def _prepare_edge_index(
    edge_index: Optional[Tensor],
    *,
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.is_floating_point() or edge_index.is_complex():
        raise TypeError("edge_index must use an integer dtype")
    if edge_index.dtype == torch.bool:
        raise TypeError("edge_index must use an integer dtype")
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.numel():
        if bool((edge_index < 0).any()) or bool((edge_index >= num_nodes).any()):
            raise ValueError("edge_index contains an out-of-range node index")
        if bool((edge_index[0] == edge_index[1]).any()):
            raise ValueError(
                "self loops are prohibited; construct graphs without them"
            )
    return edge_index


def _prepare_edge_attributes(
    edge_attributes: Optional[Tensor],
    *,
    num_edges: int,
    edge_attribute_dim: int,
    reference: Tensor,
) -> Tensor:
    if edge_attributes is None:
        if num_edges:
            raise ValueError(
                "edge_attributes are required when the graph contains edges"
            )
        return reference.new_empty((0, edge_attribute_dim))
    if edge_attributes.ndim != 2:
        raise ValueError("edge_attributes must have shape [num_edges, edge_dim]")
    if edge_attributes.shape != (num_edges, edge_attribute_dim):
        raise ValueError(
            "edge_attributes shape mismatch: expected "
            f"({num_edges}, {edge_attribute_dim}), got "
            f"{tuple(edge_attributes.shape)}"
        )
    return edge_attributes.to(device=reference.device, dtype=reference.dtype)


def mean_incoming_neighbors(
    node_embedding: Tensor,
    edge_index: Optional[Tensor],
) -> Tensor:
    """Return the exact arithmetic mean of incoming source embeddings.

    Isolated receivers receive the zero vector.  ``edge_index`` is validated
    and explicit self loops are rejected.
    """

    if node_embedding.ndim != 2:
        raise ValueError("node_embedding must have shape [num_nodes, hidden_dim]")
    num_nodes = node_embedding.shape[0]
    prepared_edges = _prepare_edge_index(
        edge_index, num_nodes=num_nodes, device=node_embedding.device
    )
    if prepared_edges.shape[1] == 0:
        return torch.zeros_like(node_embedding)

    source, receiver = prepared_edges
    neighbor_sum = torch.zeros_like(node_embedding)
    neighbor_sum.index_add_(0, receiver, node_embedding.index_select(0, source))
    degree = node_embedding.new_zeros((num_nodes,))
    degree.index_add_(
        0, receiver, node_embedding.new_ones((prepared_edges.shape[1],))
    )
    return neighbor_sum / degree.clamp_min(1).unsqueeze(-1)


class NodeEncoder(nn.Module):
    """Shared masked-expression, explicit-mask, and metadata encoder."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int,
        hidden_dim: int,
        broad_spatial_feature_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        if node_covariate_dim < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        if broad_spatial_feature_dim < 0:
            raise ValueError("broad_spatial_feature_dim cannot be negative")
        self.num_genes = num_genes
        self.node_covariate_dim = node_covariate_dim
        self.broad_spatial_feature_dim = broad_spatial_feature_dim
        self.hidden_dim = hidden_dim

        self.expression_projection = nn.Linear(
            num_genes, hidden_dim, bias=False
        )
        self.mask_projection = nn.Linear(num_genes, hidden_dim, bias=False)
        self.covariate_projection = (
            nn.Linear(node_covariate_dim, hidden_dim, bias=False)
            if node_covariate_dim
            else None
        )
        self.broad_spatial_projection = (
            nn.Linear(broad_spatial_feature_dim, hidden_dim, bias=False)
            if broad_spatial_feature_dim
            else None
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        node_covariates: Optional[Tensor] = None,
        broad_spatial_features: Optional[Tensor] = None,
    ) -> Tensor:
        if input_expression.ndim != 2:
            raise ValueError(
                "input_expression must have shape [num_nodes, num_genes]"
            )
        if not input_expression.is_floating_point():
            raise TypeError("input_expression must be floating point")
        if input_expression.shape[1] != self.num_genes:
            raise ValueError(
                f"expected {self.num_genes} genes, got "
                f"{input_expression.shape[1]}"
            )
        if gene_mask.shape != input_expression.shape:
            raise ValueError("gene_mask must have the same shape as expression")

        mask_boolean = gene_mask.to(
            device=input_expression.device, dtype=torch.bool
        )
        mask = mask_boolean.to(dtype=input_expression.dtype)
        # Applying the mask here prevents a hidden target value from leaking
        # into the model even if a caller accidentally passes the unmasked
        # expression tensor. ``masked_fill`` also prevents masked NaN or Inf
        # values from surviving a multiplication by zero.
        masked_expression = input_expression.masked_fill(mask_boolean, 0.0)
        embedding = (
            self.expression_projection(masked_expression)
            + self.mask_projection(mask)
            + self.bias
        )

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
                device=input_expression.device, dtype=input_expression.dtype
            )
            embedding = embedding + self.covariate_projection(covariates)
        elif node_covariates is not None:
            expected = (num_nodes, 0)
            if node_covariates.shape != expected:
                raise ValueError(
                    "node_covariates were supplied, but this model was "
                    "constructed with node_covariate_dim=0"
                )

        if self.broad_spatial_feature_dim:
            if broad_spatial_features is None:
                raise ValueError(
                    "broad_spatial_features are required for the explicitly "
                    "spatial-field-augmented control"
                )
            expected = (num_nodes, self.broad_spatial_feature_dim)
            if broad_spatial_features.shape != expected:
                raise ValueError(
                    f"broad_spatial_features must have shape {expected}, got "
                    f"{tuple(broad_spatial_features.shape)}"
                )
            spatial_features = broad_spatial_features.to(
                device=input_expression.device,
                dtype=input_expression.dtype,
            )
            if not bool(torch.isfinite(spatial_features).all()):
                raise ValueError("broad_spatial_features must be finite")
            embedding = embedding + self.broad_spatial_projection(
                spatial_features
            )
        elif broad_spatial_features is not None:
            raise ValueError(
                "broad_spatial_features were supplied to a non-spatial encoder"
            )

        return self.dropout(F.gelu(self.normalization(embedding)))


class ResidualFeedForward(nn.Module):
    """Predefined two-layer residual feed-forward block."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = _default_expansion(hidden_dim, ffn_dim)
        self.linear_in = nn.Linear(hidden_dim, ffn_dim)
        self.linear_out = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, node_embedding: Tensor) -> Tensor:
        update = self.linear_out(
            self.dropout(F.gelu(self.linear_in(node_embedding)))
        )
        return self.normalization(
            node_embedding + self.dropout(update)
        )


class ExpressionDecoder(nn.Module):
    """Shared nonlinear decoder used by B0, B1, G1, and G2."""

    def __init__(
        self,
        hidden_dim: int,
        num_genes: int,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        decoder_dim = _default_expansion(hidden_dim, decoder_dim)
        self.linear_in = nn.Linear(hidden_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(decoder_dim, num_genes)

    def forward(self, node_embedding: Tensor) -> Tensor:
        return self.linear_out(
            self.dropout(F.gelu(self.linear_in(node_embedding)))
        )


class SharedEdgeEncoder(nn.Module):
    """A shared function of measured edge attributes, never edge identity."""

    def __init__(
        self,
        edge_attribute_dim: int,
        edge_embedding_dim: int = 32,
        edge_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if edge_attribute_dim <= 0:
            raise ValueError("edge_attribute_dim must be positive")
        if edge_embedding_dim <= 0 or edge_hidden_dim <= 0:
            raise ValueError("edge encoder dimensions must be positive")
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(edge_attribute_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_embedding_dim),
            nn.LayerNorm(edge_embedding_dim),
            nn.GELU(),
        )

    def forward(self, edge_attributes: Tensor) -> Tensor:
        if edge_attributes.ndim != 2:
            raise ValueError(
                "edge_attributes must have shape [num_edges, edge_dim]"
            )
        if edge_attributes.shape[1] != self.edge_attribute_dim:
            raise ValueError(
                f"expected edge dimension {self.edge_attribute_dim}, got "
                f"{edge_attributes.shape[1]}"
            )
        if edge_attributes.shape[0] == 0:
            return edge_attributes.new_empty((0, self.edge_embedding_dim))
        return self.network(edge_attributes)


class _BaseMaskedExpressionModel(nn.Module):
    def __init__(
        self,
        *,
        num_genes: int,
        node_covariate_dim: int,
        hidden_dim: int,
        dropout: float,
        broad_spatial_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.node_covariate_dim = node_covariate_dim
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            broad_spatial_feature_dim=broad_spatial_feature_dim,
            dropout=dropout,
        )

    def _encode_and_targets(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        node_covariates: Optional[Tensor],
        target_nodes: Optional[Tensor | Sequence[int]],
        broad_spatial_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        embedding = self.encoder(
            input_expression=input_expression,
            gene_mask=gene_mask,
            node_covariates=node_covariates,
            broad_spatial_features=broad_spatial_features,
        )
        targets = _normalize_target_nodes(
            target_nodes,
            num_nodes=embedding.shape[0],
            device=embedding.device,
        )
        return embedding, targets


class SelfOnlyMLP(_BaseMaskedExpressionModel):
    """B0: cell-autonomous masked-expression MLP with visible metadata."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.self_block = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )
        self.decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
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
    ) -> ModelOutput:
        del edge_index, edge_attributes, return_explanations
        embedding, targets = self._encode_and_targets(
            input_expression, gene_mask, node_covariates, target_nodes
        )
        embedding = _select_targets(embedding, targets)
        embedding = self.self_block(embedding)
        prediction = self.decoder(embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=embedding,
        )


class BroadSpatialFieldControl(_BaseMaskedExpressionModel):
    """B0-like control augmented only with a smooth global spatial field.

    The coordinate transform is locked to the five-term global quadratic
    basis from :class:`BroadSpatialFieldBasis`.  Its center and scale must be
    fitted on training coordinates before forward execution.  No graph,
    neighbor statistic, identity, Fourier frequency, knot, or lookup embedding
    enters this model.
    """

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        broad_spatial_feature_dim = 5
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            broad_spatial_feature_dim=broad_spatial_feature_dim,
        )
        self.broad_spatial_feature_dim = broad_spatial_feature_dim
        self.self_block = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )
        self.decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )
        self.register_buffer(
            "_coordinate_center_um",
            torch.zeros(2, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "_coordinate_scale_um",
            torch.ones(2, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "_coordinate_constant_axes",
            torch.zeros(2, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_coordinate_basis_fitted",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @property
    def coordinate_basis_is_fitted(self) -> bool:
        return bool(self._coordinate_basis_fitted.detach().cpu())

    def set_coordinate_basis(self, basis: BroadSpatialFieldBasis) -> None:
        """Install an already train-fitted basis into persistent buffers."""

        if not isinstance(basis, BroadSpatialFieldBasis):
            raise TypeError("basis must be a BroadSpatialFieldBasis")
        with torch.no_grad():
            self._coordinate_center_um.copy_(
                torch.tensor(
                    basis.center_um,
                    dtype=self._coordinate_center_um.dtype,
                    device=self._coordinate_center_um.device,
                )
            )
            self._coordinate_scale_um.copy_(
                torch.tensor(
                    basis.scale_um,
                    dtype=self._coordinate_scale_um.dtype,
                    device=self._coordinate_scale_um.device,
                )
            )
            self._coordinate_constant_axes.copy_(
                torch.as_tensor(
                    basis.constant_axes,
                    dtype=torch.bool,
                    device=self._coordinate_constant_axes.device,
                )
            )
            self._coordinate_basis_fitted.fill_(True)

    def fit_coordinate_basis(
        self, training_coordinates_um: Tensor | Sequence[Sequence[float]]
    ) -> dict[str, object]:
        """Fit normalization on training coordinates and return provenance."""

        basis = BroadSpatialFieldBasis.fit(training_coordinates_um)
        self.set_coordinate_basis(basis)
        return basis.provenance()

    def spatial_basis_provenance(self) -> dict[str, object]:
        if not self.coordinate_basis_is_fitted:
            raise RuntimeError("the broad spatial coordinate basis is not fitted")
        basis = BroadSpatialFieldBasis(
            center_um=self._coordinate_center_um.detach().cpu().numpy(),
            scale_um=self._coordinate_scale_um.detach().cpu().numpy(),
            constant_axes=tuple(
                bool(value)
                for value in self._coordinate_constant_axes.detach().cpu().tolist()
            ),
        )
        return basis.provenance()

    def coordinate_basis(self, coordinates_um: Tensor) -> Tensor:
        """Apply the persistent global quadratic transform on the model device."""

        if not self.coordinate_basis_is_fitted:
            raise RuntimeError(
                "fit_coordinate_basis must be called on training coordinates "
                "before using the broad spatial-field control"
            )
        if coordinates_um.ndim != 2 or coordinates_um.shape[1] != 2:
            raise ValueError("coordinates_um must have shape [num_nodes, 2]")
        coordinates = coordinates_um.to(
            device=self._coordinate_center_um.device,
            dtype=self._coordinate_center_um.dtype,
        )
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("coordinates_um must contain only finite values")
        standardized = (
            coordinates - self._coordinate_center_um
        ) / self._coordinate_scale_um
        standardized = standardized.masked_fill(
            self._coordinate_constant_axes.unsqueeze(0), 0.0
        )
        x = standardized[:, 0]
        y = standardized[:, 1]
        return torch.stack((x, y, x * x, x * y, y * y), dim=1)

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
        coordinates_um: Optional[Tensor] = None,
    ) -> ModelOutput:
        del edge_index, edge_attributes, return_explanations
        if coordinates_um is None:
            raise ValueError(
                "coordinates_um are required only for the explicitly labeled "
                "broad spatial-field control"
            )
        spatial_features = self.coordinate_basis(coordinates_um).to(
            device=input_expression.device,
            dtype=input_expression.dtype,
        )
        embedding, targets = self._encode_and_targets(
            input_expression,
            gene_mask,
            node_covariates,
            target_nodes,
            broad_spatial_features=spatial_features,
        )
        embedding = _select_targets(embedding, targets)
        embedding = self.self_block(embedding)
        prediction = self.decoder(embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=embedding,
        )


class _MatchedSelfMixingBlock(nn.Module):
    """Self-only block with the same parameter count as one GATv2 block.

    For homogeneous node inputs, a bias-free, non-sharing GATv2 convolution
    owns two ``hidden_dim -> heads * head_dim`` projections and one attention
    vector of length ``heads * head_dim``.  This block uses exactly those
    dimensions, but every operation remains cell autonomous.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int],
        ffn_dim: Optional[int],
        dropout: float,
    ) -> None:
        super().__init__()
        _, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim, attention_heads, attention_head_dim
        )
        self.left_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.right_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.routing_vector = nn.Parameter(torch.empty(raw_attention_dim))
        nn.init.ones_(self.routing_vector)
        self.output_projection = (
            nn.Identity()
            if raw_attention_dim == hidden_dim
            else nn.Linear(raw_attention_dim, hidden_dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.feed_forward = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )

    def forward(self, node_embedding: Tensor) -> Tensor:
        update = F.gelu(
            (
                self.left_projection(node_embedding)
                + self.right_projection(node_embedding)
            )
            * self.routing_vector
        )
        update = self.output_projection(update)
        node_embedding = self.normalization(
            node_embedding + self.dropout(update)
        )
        return self.feed_forward(node_embedding)


class ParameterMatchedSelfControl(_BaseMaskedExpressionModel):
    """Self-only control parameter-matched to topology GATv2.

    With matching constructor dimensions, this model and
    :class:`TopologyGATv2` have exactly the same trainable parameter count.
    """

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 1,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        del attention_dropout  # Kept for constructor parity with G1.
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                _MatchedSelfMixingBlock(
                    hidden_dim=hidden_dim,
                    attention_heads=attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
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

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attributes: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
    ) -> ModelOutput:
        del edge_index, edge_attributes, return_explanations
        embedding, targets = self._encode_and_targets(
            input_expression, gene_mask, node_covariates, target_nodes
        )
        embedding = _select_targets(embedding, targets)
        for block in self.blocks:
            embedding = block(embedding)
        prediction = self.decoder(embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=embedding,
        )


class MeanNeighborModel(_BaseMaskedExpressionModel):
    """B1: uniform incoming-neighbor mean message passing."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # No bias: an isolated node receives exactly zero neighbor update.
        self.neighbor_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.neighbor_normalization = nn.LayerNorm(hidden_dim)
        self.feed_forward = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )
        self.decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
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
    ) -> ModelOutput:
        del edge_attributes, return_explanations
        embedding, targets = self._encode_and_targets(
            input_expression, gene_mask, node_covariates, target_nodes
        )
        neighbor_mean = mean_incoming_neighbors(embedding, edge_index)
        embedding = self.neighbor_normalization(
            embedding + self.neighbor_projection(neighbor_mean)
        )
        embedding = _select_targets(embedding, targets)
        embedding = self.feed_forward(embedding)
        prediction = self.decoder(embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=embedding,
        )


class _GATv2ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int],
        edge_dim: Optional[int],
        ffn_dim: Optional[int],
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if GATv2Conv is None:
            raise ImportError(
                "TopologyGATv2 and EdgeConditionedGATv2 require "
                "torch-geometric"
            )
        head_dim, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim, attention_heads, attention_head_dim
        )
        self.convolution = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=attention_heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=False,
            edge_dim=edge_dim,
            fill_value=0.0,
            bias=False,
            share_weights=False,
            residual=False,
        )
        self.output_projection = (
            nn.Identity()
            if raw_attention_dim == hidden_dim
            else nn.Linear(raw_attention_dim, hidden_dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_normalization = nn.LayerNorm(hidden_dim)
        self.feed_forward = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )

    def forward(
        self,
        node_embedding: Tensor,
        edge_index: Tensor,
        edge_embedding: Optional[Tensor],
        return_attention: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        attention: Optional[Tensor] = None
        if return_attention:
            convolution_output, attention_pair = self.convolution(
                node_embedding,
                edge_index,
                edge_attr=edge_embedding,
                return_attention_weights=True,
            )
            returned_edges, attention = attention_pair
            if not torch.equal(returned_edges, edge_index):
                raise RuntimeError(
                    "GATv2 changed edge order; explanations would be misaligned"
                )
        else:
            convolution_output = self.convolution(
                node_embedding, edge_index, edge_attr=edge_embedding
            )
        convolution_output = self.output_projection(convolution_output)
        node_embedding = self.attention_normalization(
            node_embedding + self.dropout(convolution_output)
        )
        return self.feed_forward(node_embedding), attention


class _BaseGATv2(_BaseMaskedExpressionModel):
    def __init__(
        self,
        *,
        num_genes: int,
        node_covariate_dim: int,
        hidden_dim: int,
        attention_heads: int,
        attention_head_dim: Optional[int],
        graph_layers: int,
        ffn_dim: Optional[int],
        decoder_dim: Optional[int],
        dropout: float,
        attention_dropout: float,
        convolution_edge_dim: Optional[int],
    ) -> None:
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                _GATv2ResidualBlock(
                    hidden_dim=hidden_dim,
                    attention_heads=attention_heads,
                    attention_head_dim=attention_head_dim,
                    edge_dim=convolution_edge_dim,
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

    def _graph_forward(
        self,
        *,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor],
        edge_embedding: Optional[Tensor],
        node_covariates: Optional[Tensor],
        return_explanations: bool,
        target_nodes: Optional[Tensor | Sequence[int]],
    ) -> ModelOutput:
        embedding, targets = self._encode_and_targets(
            input_expression, gene_mask, node_covariates, target_nodes
        )
        prepared_edges = _prepare_edge_index(
            edge_index,
            num_nodes=embedding.shape[0],
            device=embedding.device,
        )
        attention = None
        for layer_number, block in enumerate(self.blocks):
            is_last_layer = layer_number == len(self.blocks) - 1
            embedding, layer_attention = block(
                embedding,
                prepared_edges,
                edge_embedding,
                return_attention=return_explanations and is_last_layer,
            )
            if layer_attention is not None:
                attention = layer_attention
        embedding = _select_targets(embedding, targets)
        prediction = self.decoder(embedding)
        return ModelOutput(
            prediction=prediction,
            node_embedding=embedding,
            attention_weights=attention,
            edge_index=prepared_edges if return_explanations else None,
        )


class TopologyGATv2(_BaseGATv2):
    """G1: topology-only GATv2 with no implicit or explicit self loops."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        graph_layers: int = 1,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
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
            dropout=dropout,
            attention_dropout=attention_dropout,
            convolution_edge_dim=None,
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
    ) -> ModelOutput:
        del edge_attributes
        return self._graph_forward(
            input_expression=input_expression,
            gene_mask=gene_mask,
            edge_index=edge_index,
            edge_embedding=None,
            node_covariates=node_covariates,
            return_explanations=return_explanations,
            target_nodes=target_nodes,
        )


class EdgeConditionedGATv2(_BaseGATv2):
    """G2: GATv2 whose routing scores use shared edge embeddings."""

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
    ) -> None:
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_embedding_dim = edge_embedding_dim
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            graph_layers=graph_layers,
            ffn_dim=ffn_dim,
            decoder_dim=decoder_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            convolution_edge_dim=edge_embedding_dim,
        )
        # One shared edge encoder is reused by every graph layer.
        self.edge_encoder = SharedEdgeEncoder(
            edge_attribute_dim=edge_attribute_dim,
            edge_embedding_dim=edge_embedding_dim,
            edge_hidden_dim=edge_hidden_dim,
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
    ) -> ModelOutput:
        num_nodes = input_expression.shape[0]
        prepared_edges = _prepare_edge_index(
            edge_index, num_nodes=num_nodes, device=input_expression.device
        )
        prepared_attributes = _prepare_edge_attributes(
            edge_attributes,
            num_edges=prepared_edges.shape[1],
            edge_attribute_dim=self.edge_attribute_dim,
            reference=input_expression,
        )
        edge_embedding = self.edge_encoder(prepared_attributes)
        output = self._graph_forward(
            input_expression=input_expression,
            gene_mask=gene_mask,
            edge_index=prepared_edges,
            edge_embedding=edge_embedding,
            node_covariates=node_covariates,
            return_explanations=return_explanations,
            target_nodes=target_nodes,
        )
        output.edge_embedding = (
            edge_embedding if return_explanations else None
        )
        return output


def _segment_softmax(
    scores: Tensor, receiver: Tensor, num_nodes: int
) -> Tensor:
    """Stable softmax over incoming edges, independently for every head."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_edges, num_heads]")
    if scores.shape[0] == 0:
        return scores
    index = receiver.view(-1, 1).expand_as(scores)
    maxima = scores.new_full((num_nodes, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0, index, scores, reduce="amax", include_self=True
    )
    exponentiated = torch.exp(scores - maxima.index_select(0, receiver))
    denominator = scores.new_zeros((num_nodes, scores.shape[1]))
    denominator.scatter_add_(0, index, exponentiated)
    return exponentiated / denominator.index_select(0, receiver).clamp_min(
        torch.finfo(scores.dtype).tiny
    )


class AdditiveEdgeMessageModel(_BaseMaskedExpressionModel):
    """G3: exact additive self plus signed edge-message model.

    The returned prediction is constructed exactly as
    ``self_prediction + neighbor_prediction``.  Edge messages are returned
    only on request; :meth:`edge_gene_contributions` then decodes selected
    edges and/or genes without materializing an unavoidable ``E x G`` tensor.
    """

    def __init__(
        self,
        num_genes: int,
        edge_attribute_dim: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        attention_head_dim: Optional[int] = None,
        message_head_dim: int = 16,
        message_dim: int = 64,
        ffn_dim: Optional[int] = None,
        decoder_dim: Optional[int] = None,
        edge_hidden_dim: int = 64,
        edge_embedding_dim: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        head_dim, raw_attention_dim = _validate_hidden_dimensions(
            hidden_dim, attention_heads, attention_head_dim
        )
        if message_head_dim <= 0 or message_dim <= 0:
            raise ValueError("message dimensions must be positive")
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.attention_heads = attention_heads
        self.attention_head_dim = head_dim
        self.message_head_dim = message_head_dim
        self.message_dim = message_dim
        self.attention_dropout = attention_dropout

        self.self_block = ResidualFeedForward(
            hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout
        )
        self.self_decoder = ExpressionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )
        self.edge_encoder = SharedEdgeEncoder(
            edge_attribute_dim=edge_attribute_dim,
            edge_embedding_dim=edge_embedding_dim,
            edge_hidden_dim=edge_hidden_dim,
        )

        self.query_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_dim, raw_attention_dim, bias=False
        )
        self.edge_attention_projection = nn.Linear(
            edge_embedding_dim, raw_attention_dim, bias=False
        )
        self.attention_vector = nn.Parameter(
            torch.empty(attention_heads, head_dim)
        )
        nn.init.xavier_uniform_(self.attention_vector)

        edge_value_input_dim = 2 * hidden_dim + edge_embedding_dim
        edge_value_hidden_dim = max(
            message_dim, attention_heads * message_head_dim
        )
        self.edge_value_network = nn.Sequential(
            nn.Linear(edge_value_input_dim, edge_value_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                edge_value_hidden_dim,
                attention_heads * message_head_dim,
            ),
        )
        self.message_projection = nn.Linear(
            attention_heads * message_head_dim, message_dim, bias=False
        )
        # Bias must remain absent for an exact sum of edge contributions.
        self.neighbor_decoder = nn.Linear(
            message_dim, num_genes, bias=False
        )

    def copy_self_branch_from(self, model: SelfOnlyMLP) -> None:
        """Initialize the encoder and self path from a compatible B0 model."""

        if not isinstance(model, SelfOnlyMLP):
            raise TypeError("model must be a SelfOnlyMLP")
        if (
            model.num_genes != self.num_genes
            or model.node_covariate_dim != self.node_covariate_dim
            or model.hidden_dim != self.hidden_dim
        ):
            raise ValueError("B0 and G3 self-branch dimensions do not match")
        self.encoder.load_state_dict(model.encoder.state_dict())
        self.self_block.load_state_dict(model.self_block.state_dict())
        self.self_decoder.load_state_dict(model.decoder.state_dict())

    def set_self_branch_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the staged-training self branch."""

        for module in (self.encoder, self.self_block, self.self_decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    def edge_gene_contributions(
        self,
        edge_message: Tensor,
        *,
        gene_indices: Optional[Tensor | Sequence[int]] = None,
        edge_indices: Optional[Tensor | Sequence[int]] = None,
    ) -> Tensor:
        """Decode signed edge contributions for selected edges and genes.

        This method is intentionally explicit and on-demand.  Passing neither
        selector returns the full contribution tensor and should therefore be
        reserved for small graphs.
        """

        if edge_message.ndim != 2 or edge_message.shape[1] != self.message_dim:
            raise ValueError(
                "edge_message must have shape "
                f"[num_edges, {self.message_dim}]"
            )
        messages = edge_message
        if edge_indices is not None:
            edge_indices = torch.as_tensor(
                edge_indices, dtype=torch.long, device=edge_message.device
            )
            messages = messages.index_select(0, edge_indices)

        weight = self.neighbor_decoder.weight
        if gene_indices is not None:
            gene_indices = torch.as_tensor(
                gene_indices, dtype=torch.long, device=weight.device
            )
            weight = weight.index_select(0, gene_indices)
        return F.linear(messages, weight, bias=None)

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attributes: Optional[Tensor] = None,
        node_covariates: Optional[Tensor] = None,
        return_explanations: bool = False,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
    ) -> ModelOutput:
        node_embedding, targets = self._encode_and_targets(
            input_expression, gene_mask, node_covariates, target_nodes
        )
        num_nodes = node_embedding.shape[0]
        prepared_edges = _prepare_edge_index(
            edge_index, num_nodes=num_nodes, device=node_embedding.device
        )
        prepared_attributes = _prepare_edge_attributes(
            edge_attributes,
            num_edges=prepared_edges.shape[1],
            edge_attribute_dim=self.edge_attribute_dim,
            reference=node_embedding,
        )
        edge_embedding = self.edge_encoder(prepared_attributes)

        selected_node_embedding = _select_targets(node_embedding, targets)
        self_embedding = self.self_block(selected_node_embedding)
        self_prediction = self.self_decoder(self_embedding)

        num_edges = prepared_edges.shape[1]
        if num_edges:
            source, receiver = prepared_edges
            query = self.query_projection(node_embedding).view(
                num_nodes, self.attention_heads, self.attention_head_dim
            )
            key = self.key_projection(node_embedding).view(
                num_nodes, self.attention_heads, self.attention_head_dim
            )
            edge_attention = self.edge_attention_projection(
                edge_embedding
            ).view(
                num_edges, self.attention_heads, self.attention_head_dim
            )
            attention_logits = (
                F.leaky_relu(
                    query.index_select(0, receiver)
                    + key.index_select(0, source)
                    + edge_attention,
                    negative_slope=0.2,
                )
                * self.attention_vector
            ).sum(dim=-1)
            attention = _segment_softmax(
                attention_logits, receiver, num_nodes
            )
            routed_attention = F.dropout(
                attention,
                p=self.attention_dropout,
                training=self.training,
            )

            value_input = torch.cat(
                (
                    node_embedding.index_select(0, receiver),
                    node_embedding.index_select(0, source),
                    edge_embedding,
                ),
                dim=-1,
            )
            edge_value = self.edge_value_network(value_input).view(
                num_edges, self.attention_heads, self.message_head_dim
            )
            routed_value = (
                routed_attention.unsqueeze(-1) * edge_value
            ).reshape(num_edges, self.attention_heads * self.message_head_dim)
            edge_message = self.message_projection(routed_value)

            neighbor_latent = node_embedding.new_zeros(
                (num_nodes, self.message_dim)
            )
            neighbor_latent.index_add_(0, receiver, edge_message)
        else:
            attention = node_embedding.new_empty((0, self.attention_heads))
            edge_message = node_embedding.new_empty((0, self.message_dim))
            neighbor_latent = node_embedding.new_zeros(
                (num_nodes, self.message_dim)
            )

        neighbor_prediction = self.neighbor_decoder(
            _select_targets(neighbor_latent, targets)
        )
        prediction = self_prediction + neighbor_prediction

        return ModelOutput(
            prediction=prediction,
            node_embedding=selected_node_embedding,
            attention_weights=attention if return_explanations else None,
            edge_embedding=edge_embedding if return_explanations else None,
            edge_message=edge_message if return_explanations else None,
            self_prediction=self_prediction,
            neighbor_prediction=neighbor_prediction,
            edge_index=prepared_edges if return_explanations else None,
        )


# Compact experimental IDs and descriptive aliases are both public so configs
# can use the ladder notation without coupling downstream code to filenames.
B0 = SelfOnlyMLP
B0SelfMLP = SelfOnlyMLP
ParameterMatchedB0 = ParameterMatchedSelfControl
BroadField = BroadSpatialFieldControl
B1 = MeanNeighborModel
B1MeanNeighbor = MeanNeighborModel
G1 = TopologyGATv2
G1TopologyGATv2 = TopologyGATv2
G2 = EdgeConditionedGATv2
G2EdgeConditionedGATv2 = EdgeConditionedGATv2
G3 = AdditiveEdgeMessageModel
G3AdditiveEdgeMessage = AdditiveEdgeMessageModel


__all__ = [
    "AdditiveEdgeMessageModel",
    "B0",
    "B0SelfMLP",
    "B1",
    "B1MeanNeighbor",
    "BroadField",
    "BroadSpatialFieldControl",
    "EdgeConditionedGATv2",
    "ExpressionDecoder",
    "G1",
    "G1TopologyGATv2",
    "G2",
    "G2EdgeConditionedGATv2",
    "G3",
    "G3AdditiveEdgeMessage",
    "MeanNeighborModel",
    "ModelOutput",
    "NodeEncoder",
    "ParameterMatchedB0",
    "ParameterMatchedSelfControl",
    "ResidualFeedForward",
    "SelfOnlyMLP",
    "SharedEdgeEncoder",
    "TopologyGATv2",
    "mean_incoming_neighbors",
]

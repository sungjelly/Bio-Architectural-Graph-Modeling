"""Constrained signed-program models for the locked Stage-0 diagnostic.

This module is deliberately diagnostic-only.  It reuses the frozen fixture,
training loop, evaluation mask, sender-state permutation, attribution deletion
test, and gate from :mod:`spatial_benchmark.multiscale_synthetic`.  The only
changed object is the model architecture.

The local branch does not normalize all incoming edges with a softmax.  It
preserves an explicit unnormalized sender-count channel and maps fixed,
nonnegative source programs to fixed receiver programs with signed additive
coefficients.  Consequently each selected edge contribution is an exact term
in the local prediction rather than an attention proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .hybrid_count import HybridCountNodeEncoder
from .models import ResidualFeedForward
from .multiscale_hurdle_training import (
    MultiscaleGraphSplitView,
    evaluate_fixed_multiscale_hurdle_mask,
    fit_full_core_multiscale_hurdle_model,
)
from .multiscale_hybrid import trainable_parameter_count
from .multiscale_synthetic import (
    ARM_NULL_TRUE_LOCAL,
    ARM_PERMUTED_LOCAL,
    ARM_SELF_REGIONAL,
    ARM_TRUE_LOCAL,
    MultiscaleSyntheticFixture,
    MultiscaleSyntheticRecoveryResult,
    SyntheticArmOutcome,
    SyntheticRecoveryConfig,
    run_multiscale_synthetic_recovery,
)
from .training import set_deterministic_seed


SignedProgramVariant = Literal[
    "count_sum",
    "count_plus_concentration",
    "radial_count",
]
SIGNED_PROGRAM_VARIANTS: tuple[SignedProgramVariant, ...] = (
    "count_sum",
    "count_plus_concentration",
    "radial_count",
)
_ROUTING_MODES = frozenset({"surrogate", "true", "permuted"})


def _validated_variant(value: str) -> SignedProgramVariant:
    candidate = str(value).strip().lower()
    if candidate not in SIGNED_PROGRAM_VARIANTS:
        raise ValueError(
            f"variant must be one of {list(SIGNED_PROGRAM_VARIANTS)}, "
            f"got {value!r}"
        )
    return candidate  # type: ignore[return-value]


def _validated_routing(value: str, *, name: str) -> str:
    candidate = str(value).strip().lower()
    if candidate not in _ROUTING_MODES:
        raise ValueError(
            f"{name} must be one of {sorted(_ROUTING_MODES)}, got {value!r}"
        )
    return candidate


def _normalized_program_matrix(
    value: Optional[Tensor | Sequence[Sequence[float]]],
    *,
    num_genes: int,
    name: str,
) -> Tensor:
    matrix = (
        torch.eye(num_genes, dtype=torch.float32)
        if value is None
        else torch.as_tensor(value, dtype=torch.float32)
    )
    if matrix.ndim != 2 or matrix.shape[1] != num_genes:
        raise ValueError(f"{name} must have shape [programs, {num_genes}]")
    if matrix.shape[0] <= 0 or not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite and nonempty")
    if bool((matrix < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    row_sum = matrix.sum(dim=1, keepdim=True)
    if bool((row_sum <= 0).any()):
        raise ValueError(f"{name} cannot contain an empty program")
    return (matrix / row_sum).contiguous()


def _normalize_targets(
    target_nodes: Optional[Tensor | Sequence[int]],
    *,
    num_nodes: int,
    device: torch.device,
) -> Optional[Tensor]:
    if target_nodes is None:
        return None
    nodes = torch.as_tensor(target_nodes, device=device)
    if (
        nodes.ndim != 1
        or nodes.dtype == torch.bool
        or nodes.is_floating_point()
        or nodes.is_complex()
    ):
        raise TypeError("target_nodes must be a one-dimensional integer tensor")
    nodes = nodes.to(dtype=torch.long)
    if bool((nodes < 0).any()) or bool((nodes >= num_nodes).any()):
        raise ValueError("target_nodes contains an out-of-range node")
    if torch.unique(nodes).numel() != nodes.numel():
        raise ValueError("target_nodes cannot contain duplicates")
    return nodes


def _normalize_edge_selection(
    value: Tensor | Sequence[int],
    *,
    num_edges: int,
    device: torch.device,
) -> Tensor:
    selected = torch.as_tensor(value, device=device)
    if selected.dtype == torch.bool:
        if selected.ndim != 1 or selected.numel() != num_edges:
            raise ValueError(
                f"boolean edge selection must have shape [{num_edges}]"
            )
        selected = selected.nonzero(as_tuple=False).flatten()
    elif (
        selected.ndim != 1
        or selected.is_floating_point()
        or selected.is_complex()
    ):
        raise TypeError("edge selection must be one-dimensional integer data")
    else:
        selected = selected.to(dtype=torch.long)
    if bool((selected < 0).any()) or bool((selected >= num_edges).any()):
        raise ValueError("edge selection contains an out-of-range edge")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("edge selection cannot contain duplicates")
    return selected


def _normalize_gene_selection(
    value: Optional[Tensor | Sequence[int]],
    *,
    num_genes: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.arange(num_genes, device=device, dtype=torch.long)
    genes = torch.as_tensor(value, device=device)
    if genes.dtype == torch.bool:
        if genes.ndim != 1 or genes.numel() != num_genes:
            raise ValueError(
                f"boolean gene selection must have shape [{num_genes}]"
            )
        genes = genes.nonzero(as_tuple=False).flatten()
    elif (
        genes.ndim != 1
        or genes.is_floating_point()
        or genes.is_complex()
    ):
        raise TypeError("gene selection must be one-dimensional integer data")
    else:
        genes = genes.to(dtype=torch.long)
    if bool((genes < 0).any()) or bool((genes >= num_genes).any()):
        raise ValueError("gene selection contains an out-of-range gene")
    if torch.unique(genes).numel() != genes.numel():
        raise ValueError("gene selection cannot contain duplicates")
    return genes


def _prepare_edge_index(
    value: Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    edge_index = torch.as_tensor(value, device=device)
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype == torch.bool
        or edge_index.is_floating_point()
        or edge_index.is_complex()
    ):
        raise TypeError("edge_index must be integer data with shape [2, edges]")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.shape[1] == 0:
        raise ValueError("graph must contain at least one edge")
    if bool((edge_index < 0).any()) or bool((edge_index >= num_nodes).any()):
        raise ValueError("edge_index contains an out-of-range node")
    if bool((edge_index[0] == edge_index[1]).any()):
        raise ValueError("self edges are prohibited")
    return edge_index


def _prepare_edge_attributes(
    value: Tensor,
    *,
    num_edges: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    attributes = torch.as_tensor(value, device=device)
    if (
        attributes.ndim != 2
        or attributes.shape[0] != num_edges
        or attributes.shape[1] < 2
        or not attributes.is_floating_point()
    ):
        raise TypeError(
            "edge_attributes must be floating with shape [edges, >=2]"
        )
    attributes = attributes.to(dtype=dtype)
    if not bool(torch.isfinite(attributes).all()):
        raise ValueError("edge_attributes must be finite")
    return attributes


def _receiver_sum(
    edge_values: Tensor,
    receiver: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    """Deterministic receiver-sorted segment sum, including isolated nodes."""

    order = torch.argsort(receiver, stable=True)
    sorted_receiver = receiver.index_select(0, order)
    sorted_values = edge_values.index_select(0, order)
    lengths = torch.bincount(sorted_receiver, minlength=num_nodes)
    return torch.segment_reduce(sorted_values, reduce="sum", lengths=lengths)


class _SelfHurdleDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        decoder_dim: int,
        num_genes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, num_genes * 2),
        )
        self.num_genes = int(num_genes)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).reshape(value.shape[0], self.num_genes, 2)


@dataclass
class SignedProgramOutput:
    prediction: Tensor
    node_embedding: Tensor
    self_prediction: Tensor
    regional_prediction: Tensor
    local_prediction: Tensor
    regional_routing: str
    local_routing: str
    local_contribution_edge_indices: Optional[Tensor] = None
    local_contribution_edge_index: Optional[Tensor] = None
    local_contribution_effective_source_index: Optional[Tensor] = None
    local_contributions: Optional[Tensor] = None
    local_contribution_gene_indices: Optional[Tensor] = None


class SignedProgramAdditiveModel(nn.Module):
    """Exact self + regional + signed local program predictor.

    Program matrices are fixed and nonnegative.  Trainable signed coefficients
    are indexed by output channel, receiver program, local basis, and sender
    program.  The identity matrices used by the locked synthetic diagnostic
    make each synthetic gene one source and receiver program.
    """

    def __init__(
        self,
        *,
        num_genes: int,
        expression_mean: Tensor | Sequence[float],
        expression_scale: Tensor | Sequence[float],
        variant: SignedProgramVariant,
        sender_program_matrix: Optional[
            Tensor | Sequence[Sequence[float]]
        ] = None,
        receiver_program_matrix: Optional[
            Tensor | Sequence[Sequence[float]]
        ] = None,
        node_covariate_dim: int = 0,
        hidden_dim: int = 32,
        decoder_dim: int = 32,
        ffn_dim: int = 48,
        dropout: float = 0.0,
        regional_routing: str = "true",
        local_routing: str = "true",
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        if hidden_dim <= 0 or decoder_dim <= 0 or ffn_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if node_covariate_dim < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_genes = int(num_genes)
        self.variant = _validated_variant(variant)
        self.regional_routing = _validated_routing(
            regional_routing, name="regional_routing"
        )
        if self.regional_routing != "true":
            raise ValueError(
                "the diagnostic requires true, separately modeled regional "
                "context in every arm"
            )
        self.local_routing = _validated_routing(
            local_routing, name="local_routing"
        )
        sender = _normalized_program_matrix(
            sender_program_matrix,
            num_genes=self.num_genes,
            name="sender_program_matrix",
        )
        receiver = _normalized_program_matrix(
            receiver_program_matrix,
            num_genes=self.num_genes,
            name="receiver_program_matrix",
        )
        self.register_buffer("sender_program_matrix", sender)
        self.register_buffer("receiver_program_matrix", receiver)
        self.num_sender_programs = int(sender.shape[0])
        self.num_receiver_programs = int(receiver.shape[0])
        self.local_basis_dim = {
            "count_sum": 1,
            "count_plus_concentration": 2,
            "radial_count": 4,
        }[self.variant]

        self.encoder = HybridCountNodeEncoder(
            num_genes=self.num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            dropout=dropout,
        )
        self.self_block = ResidualFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.self_decoder = _SelfHurdleDecoder(
            hidden_dim=hidden_dim,
            decoder_dim=decoder_dim,
            num_genes=self.num_genes,
            dropout=dropout,
        )
        self.regional_coefficients = nn.Parameter(
            torch.zeros(
                2,
                self.num_receiver_programs,
                self.num_sender_programs,
            )
        )
        self.local_coefficients = nn.Parameter(
            torch.zeros(
                2,
                self.num_receiver_programs,
                self.local_basis_dim,
                self.num_sender_programs,
            )
        )

    def _program_scores(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
    ) -> Tensor:
        if (
            input_expression.ndim != 2
            or input_expression.shape[1] != self.num_genes
            or gene_mask.dtype != torch.bool
            or gene_mask.shape != input_expression.shape
        ):
            raise ValueError(
                "input_expression and boolean gene_mask must align with genes"
            )
        if bool((input_expression < 0).any()) or not bool(
            torch.isfinite(input_expression).all()
        ):
            raise ValueError("input_expression must be finite raw counts")
        # Apply the mask again inside the program branch.  This makes the
        # caller's zero fill non-authoritative and prevents hidden values from
        # entering graph messages.
        visible_log_count = torch.log1p(input_expression).masked_fill(
            gene_mask, 0.0
        )
        return visible_log_count @ self.sender_program_matrix.t()

    def _regional_prediction(
        self,
        programs: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        source, receiver = edge_index
        degree = torch.bincount(
            receiver, minlength=programs.shape[0]
        ).clamp_min(1)
        edge_program = programs.index_select(0, source) / degree.index_select(
            0, receiver
        ).to(dtype=programs.dtype).unsqueeze(1)
        aggregate = _receiver_sum(
            edge_program, receiver, num_nodes=programs.shape[0]
        )
        receiver_program_prediction = torch.einsum(
            "np,crp->nrc", aggregate, self.regional_coefficients
        )
        return torch.einsum(
            "nrc,rg->ngc",
            receiver_program_prediction,
            self.receiver_program_matrix,
        )

    def _local_basis(
        self,
        edge_index: Tensor,
        edge_attributes: Tensor,
        *,
        num_nodes: int,
    ) -> Tensor:
        receiver = edge_index[1]
        ones = edge_attributes.new_ones((edge_index.shape[1],))
        if self.variant == "count_sum":
            return ones.unsqueeze(1)
        degree = torch.bincount(receiver, minlength=num_nodes).clamp_min(1)
        concentration = ones / degree.index_select(0, receiver).to(
            dtype=ones.dtype
        )
        if self.variant == "count_plus_concentration":
            return torch.stack((ones, concentration), dim=1)

        # Edge attribute column 1 is the frozen distance/radius value.  The
        # uniform count channel remains first, followed by fixed overlapping
        # triangular near/middle/far bases.  These bases are not learned or
        # selected from the outcome.
        normalized_distance = edge_attributes[:, 1].clamp(0.0, 1.0)
        centers = edge_attributes.new_tensor((0.0, 0.5, 1.0))
        radial = (
            1.0
            - (
                normalized_distance.unsqueeze(1) - centers.unsqueeze(0)
            ).abs()
            / 0.5
        ).clamp_min(0.0)
        return torch.cat((ones.unsqueeze(1), radial), dim=1)

    def _edge_program_values(
        self,
        programs: Tensor,
        edge_index: Tensor,
        basis: Tensor,
        source_index_by_node: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        source = edge_index[0]
        effective_source = source
        if source_index_by_node is not None:
            source_map = torch.as_tensor(
                source_index_by_node,
                device=programs.device,
            )
            if (
                source_map.ndim != 1
                or source_map.numel() != programs.shape[0]
                or source_map.dtype == torch.bool
                or source_map.is_floating_point()
                or source_map.is_complex()
            ):
                raise TypeError(
                    "source_index_by_node must be an integer node permutation"
                )
            source_map = source_map.to(dtype=torch.long)
            if (
                bool((source_map < 0).any())
                or bool((source_map >= programs.shape[0]).any())
                or torch.unique(source_map).numel() != programs.shape[0]
            ):
                raise ValueError(
                    "source_index_by_node must be a complete node permutation"
                )
            effective_source = source_map.index_select(0, source)
        edge_program = programs.index_select(
            0, effective_source
        ).unsqueeze(1) * basis.unsqueeze(2)
        return edge_program, effective_source

    def _decode_program_values(self, value: Tensor) -> Tensor:
        receiver_program_prediction = torch.einsum(
            "...bp,crbp->...rc", value, self.local_coefficients
        )
        return torch.einsum(
            "...rc,rg->...gc",
            receiver_program_prediction,
            self.receiver_program_matrix,
        )

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        *,
        node_covariates: Optional[Tensor] = None,
        regional_edge_index: Optional[Tensor] = None,
        regional_edge_attributes: Optional[Tensor] = None,
        local_edge_index: Optional[Tensor] = None,
        local_edge_attributes: Optional[Tensor] = None,
        local_source_index_by_node: Optional[Tensor] = None,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        local_contribution_edge_indices: Optional[
            Tensor | Sequence[int]
        ] = None,
        local_contribution_gene_indices: Optional[
            Tensor | Sequence[int]
        ] = None,
    ) -> SignedProgramOutput:
        node_embedding = self.encoder(
            input_expression,
            gene_mask,
            node_covariates,
        )
        targets = _normalize_targets(
            target_nodes,
            num_nodes=node_embedding.shape[0],
            device=node_embedding.device,
        )
        selected_embedding = (
            node_embedding
            if targets is None
            else node_embedding.index_select(0, targets)
        )
        self_prediction = self.self_decoder(
            self.self_block(selected_embedding)
        )
        programs = self._program_scores(input_expression, gene_mask)

        if regional_edge_index is None or regional_edge_attributes is None:
            raise ValueError("true regional routing requires graph inputs")
        regional_edges = _prepare_edge_index(
            regional_edge_index,
            num_nodes=node_embedding.shape[0],
            device=node_embedding.device,
        )
        _prepare_edge_attributes(
            regional_edge_attributes,
            num_edges=regional_edges.shape[1],
            device=node_embedding.device,
            dtype=programs.dtype,
        )
        regional_all = self._regional_prediction(programs, regional_edges)
        regional_prediction = (
            regional_all
            if targets is None
            else regional_all.index_select(0, targets)
        )

        contribution_edge_ids: Optional[Tensor] = None
        contribution_edges: Optional[Tensor] = None
        contribution_sources: Optional[Tensor] = None
        contribution_values: Optional[Tensor] = None
        contribution_genes: Optional[Tensor] = None
        if self.local_routing == "surrogate":
            if (
                local_edge_index is not None
                or local_edge_attributes is not None
                or local_source_index_by_node is not None
            ):
                raise ValueError(
                    "disabled local routing prohibits local graph inputs"
                )
            if (
                local_contribution_edge_indices is not None
                or local_contribution_gene_indices is not None
            ):
                raise ValueError(
                    "disabled local routing has no edge contributions"
                )
            local_prediction = torch.zeros_like(regional_prediction)
        else:
            if local_edge_index is None or local_edge_attributes is None:
                raise ValueError("active local routing requires graph inputs")
            if (
                self.local_routing == "true"
                and local_source_index_by_node is not None
            ):
                raise ValueError(
                    "true local routing prohibits a sender permutation"
                )
            if (
                self.local_routing == "permuted"
                and local_source_index_by_node is None
            ):
                raise ValueError(
                    "permuted local routing requires a sender permutation"
                )
            local_edges = _prepare_edge_index(
                local_edge_index,
                num_nodes=node_embedding.shape[0],
                device=node_embedding.device,
            )
            local_attributes = _prepare_edge_attributes(
                local_edge_attributes,
                num_edges=local_edges.shape[1],
                device=node_embedding.device,
                dtype=programs.dtype,
            )
            basis = self._local_basis(
                local_edges,
                local_attributes,
                num_nodes=node_embedding.shape[0],
            )
            edge_program, effective_source = self._edge_program_values(
                programs,
                local_edges,
                basis,
                local_source_index_by_node,
            )
            aggregate = _receiver_sum(
                edge_program,
                local_edges[1],
                num_nodes=node_embedding.shape[0],
            )
            local_all = self._decode_program_values(aggregate)
            local_prediction = (
                local_all
                if targets is None
                else local_all.index_select(0, targets)
            )

            if local_contribution_edge_indices is not None:
                contribution_edge_ids = _normalize_edge_selection(
                    local_contribution_edge_indices,
                    num_edges=local_edges.shape[1],
                    device=node_embedding.device,
                )
                contribution_genes = _normalize_gene_selection(
                    local_contribution_gene_indices,
                    num_genes=self.num_genes,
                    device=node_embedding.device,
                )
                selected_edge_program = edge_program.index_select(
                    0, contribution_edge_ids
                )
                all_gene_contributions = self._decode_program_values(
                    selected_edge_program
                )
                contribution_values = all_gene_contributions.index_select(
                    1, contribution_genes
                )
                contribution_edges = local_edges.index_select(
                    1, contribution_edge_ids
                )
                contribution_sources = effective_source.index_select(
                    0, contribution_edge_ids
                )
            elif local_contribution_gene_indices is not None:
                raise ValueError(
                    "contribution genes require selected local edges"
                )

        prediction = self_prediction + regional_prediction + local_prediction
        return SignedProgramOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            self_prediction=self_prediction,
            regional_prediction=regional_prediction,
            local_prediction=local_prediction,
            regional_routing=self.regional_routing,
            local_routing=self.local_routing,
            local_contribution_edge_indices=contribution_edge_ids,
            local_contribution_edge_index=contribution_edges,
            local_contribution_effective_source_index=contribution_sources,
            local_contributions=contribution_values,
            local_contribution_gene_indices=contribution_genes,
        )


def _parameter_structure_sha256(model: nn.Module) -> str:
    payload = "\n".join(
        f"{name}:{','.join(str(size) for size in parameter.shape)}"
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parameter_value_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def make_signed_program_model(
    fixture: MultiscaleSyntheticFixture,
    config: SyntheticRecoveryConfig,
    *,
    variant: SignedProgramVariant,
    local_routing: str,
    null_injection: bool,
) -> SignedProgramAdditiveModel:
    """Construct one deterministically initialized parameter-matched arm."""

    set_deterministic_seed(
        config.model_seed,
        deterministic=True,
        warn_only=False,
    )
    mean = (
        fixture.null_expression_mean
        if null_injection
        else fixture.expression_mean
    )
    scale = (
        fixture.null_expression_scale
        if null_injection
        else fixture.expression_scale
    )
    identity = torch.eye(fixture.true_view.num_genes, dtype=torch.float32)
    return SignedProgramAdditiveModel(
        num_genes=fixture.true_view.num_genes,
        expression_mean=mean,
        expression_scale=scale,
        variant=variant,
        sender_program_matrix=identity,
        receiver_program_matrix=identity,
        node_covariate_dim=0,
        hidden_dim=config.hidden_dim,
        decoder_dim=config.decoder_dim,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        regional_routing="true",
        local_routing=local_routing,
    )


def _fit_signed_program_arm(
    *,
    arm: str,
    fixture: MultiscaleSyntheticFixture,
    config: SyntheticRecoveryConfig,
    variant: SignedProgramVariant,
) -> tuple[SyntheticArmOutcome, SignedProgramAdditiveModel]:
    specifications: Mapping[
        str, tuple[str, MultiscaleGraphSplitView, bool]
    ] = {
        ARM_SELF_REGIONAL: ("surrogate", fixture.true_view, False),
        ARM_TRUE_LOCAL: ("true", fixture.true_view, False),
        ARM_PERMUTED_LOCAL: (
            "permuted",
            fixture.permuted_view,
            False,
        ),
        ARM_NULL_TRUE_LOCAL: ("true", fixture.null_view, True),
    }
    if arm not in specifications:
        raise ValueError(f"unknown synthetic arm {arm!r}")
    local_routing, view, null_injection = specifications[arm]
    model = make_signed_program_model(
        fixture,
        config,
        variant=variant,
        local_routing=local_routing,
        null_injection=null_injection,
    )
    mean = (
        fixture.null_expression_mean
        if null_injection
        else fixture.expression_mean
    )
    scale = (
        fixture.null_expression_scale
        if null_injection
        else fixture.expression_scale
    )
    initial_sha = _parameter_value_sha256(model)
    structure_sha = _parameter_structure_sha256(model)
    training = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        config.training_config(),
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=config.target_node_batch_size,
    )
    evaluation = evaluate_fixed_multiscale_hurdle_mask(
        model,
        view,
        fixture.evaluation_mask,
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=config.target_node_batch_size,
        device=config.device,
        amp=config.amp,
        amp_dtype=config.amp_dtype,
    )
    return (
        SyntheticArmOutcome(
            arm=arm,
            regional_routing="true",
            local_routing=local_routing,
            parameter_count=trainable_parameter_count(model),
            parameter_structure_sha256=structure_sha,
            initial_parameter_sha256=initial_sha,
            training=training,
            evaluation=evaluation,
        ),
        model,
    )


def run_signed_program_variant(
    fixture: MultiscaleSyntheticFixture,
    config: SyntheticRecoveryConfig,
    *,
    variant: SignedProgramVariant,
) -> MultiscaleSyntheticRecoveryResult:
    """Run all four frozen arms for one prespecified architecture variant."""

    selected_variant = _validated_variant(variant)

    def arm_fit(
        *,
        arm: str,
        fixture: MultiscaleSyntheticFixture,
        config: SyntheticRecoveryConfig,
    ) -> tuple[SyntheticArmOutcome, SignedProgramAdditiveModel]:
        return _fit_signed_program_arm(
            arm=arm,
            fixture=fixture,
            config=config,
            variant=selected_variant,
        )

    return run_multiscale_synthetic_recovery(
        # The fixture is supplied, so geometry is not dereferenced by the
        # existing orchestration function.  Its identity remains embedded in
        # and verified by the fixture checksum/audit.
        geometry=fixture.audit["observed_geometry"],  # type: ignore[arg-type]
        config=config,
        fixture=fixture,
        arm_fit=arm_fit,
    )


__all__ = [
    "SIGNED_PROGRAM_VARIANTS",
    "SignedProgramAdditiveModel",
    "SignedProgramOutput",
    "SignedProgramVariant",
    "make_signed_program_model",
    "run_signed_program_variant",
]

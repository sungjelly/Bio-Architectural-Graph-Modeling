"""Relaxed categorical input sensitivity for the tokenized G2 campaign.

Token IDs are discrete, so differentiating with respect to their integer
encoding is meaningless.  This module instead treats the four observed token
channels as coordinates of a relaxed one-hot vector.  It differentiates
centered output logits with respect to those coordinates and projects the
result onto the per-gene simplex tangent space.

The full input VJP has ``num_nodes * num_genes * 4`` entries.  Pairwise
Frobenius statistics are therefore accumulated in node chunks from the
gradient at the categorical encoder preactivation and the encoder projection
weights.  This is algebraically exact and does not materialize the full
categorical VJP.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masking import FixedMaskBundle
from .models import (
    _normalize_target_nodes,
    _prepare_edge_attributes,
    _prepare_edge_index,
    _select_targets,
)
from .tokenized_g2 import (
    CategoricalNodeEncoder,
    NUM_COUNT_TOKENS,
    TokenizedReceiverChunkedEdgeConditionedGATv2,
)


PROTOCOL_VERSION = "g2_relaxed_categorical_sensitivity_v1"
PROTOCOL_BASE_SEED = 20260726
EXPECTED_MASK_COUNT = 3
PROBES_PER_MASK = 32
BOOTSTRAP_REPLICATES = 2_000


class CategoricalSensitivityError(RuntimeError):
    """Raised when the locked sensitivity protocol cannot be evaluated."""


@dataclass(frozen=True)
class ProbeSufficientStatistics:
    """Exact pairwise sufficient statistics for one common output probe."""

    mask_entry_id: str
    probe_index: int
    reference_squared_norm: float
    candidate_squared_norm: float
    cross_inner_product: float

    def __post_init__(self) -> None:
        if not self.mask_entry_id:
            raise ValueError("mask_entry_id cannot be empty")
        if self.probe_index < 0:
            raise ValueError("probe_index must be non-negative")
        values = (
            self.reference_squared_norm,
            self.candidate_squared_norm,
            self.cross_inner_product,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("probe sufficient statistics must be finite")
        if self.reference_squared_norm < 0:
            raise ValueError("reference_squared_norm cannot be negative")
        if self.candidate_squared_norm < 0:
            raise ValueError("candidate_squared_norm cannot be negative")

        bound = math.sqrt(
            self.reference_squared_norm * self.candidate_squared_norm
        )
        tolerance = 1e-10 * max(
            1.0,
            abs(self.cross_inner_product),
            bound,
        )
        if abs(self.cross_inner_product) > bound + tolerance:
            raise ValueError(
                "cross_inner_product violates the Cauchy-Schwarz bound"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairMetrics:
    """Metrics derived after aggregating common-probe sufficient statistics."""

    cosine: float
    relative_discrepancy: float
    norm_ratio: float
    reference_squared_norm: float
    candidate_squared_norm: float
    cross_inner_product: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PercentileInterval:
    """Two-sided percentile interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PairEstimate:
    """Point estimate and locked hierarchical-bootstrap intervals."""

    reference_label: str
    candidate_label: str
    point: PairMetrics
    cosine_interval: PercentileInterval
    relative_discrepancy_interval: PercentileInterval
    norm_ratio_interval: PercentileInterval
    bootstrap_replicates: int
    bootstrap_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_label": self.reference_label,
            "candidate_label": self.candidate_label,
            "point": self.point.to_dict(),
            "intervals": {
                "cosine": self.cosine_interval.to_dict(),
                "relative_discrepancy": (
                    self.relative_discrepancy_interval.to_dict()
                ),
                "norm_ratio": self.norm_ratio_interval.to_dict(),
            },
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
        }


def locked_protocol_record() -> dict[str, object]:
    """Return the complete prespecified numerical protocol."""

    return {
        "schema_version": 1,
        "protocol": PROTOCOL_VERSION,
        "estimand": (
            "local model-implied relaxed categorical sensitivity, including "
            "direct and two-layer multihop paths"
        ),
        "mask_scope": {
            "mode": "whole_node",
            "entry_count": EXPECTED_MASK_COUNT,
            "input_derivative_support": "observed_entries_only",
            "output_probe_support": "masked_receiver_predictions_only",
        },
        "output": {
            "classes": NUM_COUNT_TOKENS,
            "centering": (
                "subtract the four-class mean logit independently for every "
                "masked receiver and gene"
            ),
        },
        "input": {
            "coordinates": "four observed relaxed one-hot token channels",
            "projection": (
                "subtract the four-channel gradient mean independently for "
                "every node and gene"
            ),
            "mask_only_channel_is_differentiated": False,
        },
        "execution": {
            "floating_point": "float32_vjp_float64_reduction",
            "dropout": "disabled_eval_mode",
            "amp": False,
            "tf32": False,
            "float32_matmul_precision": "highest",
            "parameter_gradients": False,
            "full_jacobian_materialized": False,
        },
        "probes": {
            "distribution": "iid_rademacher_minus_one_plus_one",
            "count_per_mask": PROBES_PER_MASK,
            "base_seed": PROTOCOL_BASE_SEED,
            "seed_payload": [
                PROTOCOL_BASE_SEED,
                "<mask_entry_id>",
                "<probe_index>",
            ],
            "seed_derivation": (
                "first_8_sha256_bytes_little_endian_masked_to_63_bits_of_"
                "canonical_compact_json_array"
            ),
            "generator": "numpy.random.Generator(PCG64(seed))",
            "generation_dtype": "cpu_int8",
            "checksum": "sha256_dtype_shape_and_contiguous_bytes",
        },
        "sufficient_statistics": {
            "A2": "sum((J_A_transpose_u)**2)",
            "B2": "sum((J_B_transpose_u)**2)",
            "AB": "sum((J_A_transpose_u)*(J_B_transpose_u))",
            "aggregation": (
                "equal-weight mean of A2/B2/AB over 32 probes within each "
                "mask, then equal-weight mean over the 3 masks"
            ),
        },
        "metrics": {
            "cosine": "AB / sqrt(A2 * B2)",
            "relative_discrepancy": (
                "sqrt(max(A2 + B2 - 2*AB, 0)) / (A2 * B2)**0.25"
            ),
            "norm_ratio": "sqrt(B2 / A2)",
            "primary_orientation": "B_wider_over_A_current_same_seed",
            "within_width_orientation": "B_higher_seed_over_A_lower_seed",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_payload": [PROTOCOL_BASE_SEED, "bootstrap"],
            "resampling": (
                "sample 3 masks with replacement, then independently sample "
                "32 probes with replacement within each selected mask"
            ),
            "interval": "two_sided_linear_percentile_95",
            "interval_interpretation": (
                "technical mask-plus-Rademacher-trace-probe variability "
                "only; not biological, patient, or training-seed uncertainty"
            ),
        },
        "resource_pilot": {
            "model": "wider_seed_0",
            "mask": "whole_node_replicate_0",
            "probe_index": 0,
            "floating_point": "fp32_no_amp_no_tf32",
            "maximum_peak_allocated_vram_gib": 20.5,
            "require_finite_nonzero_vjp": True,
            "project_full_vjp_workload": 2_304,
            "time_cutoff": None,
            "explicit_review_required_before_full_shards": True,
        },
        "controls": {
            "identical": (
                "current-width seed0 checkpoint loaded into two independent "
                "instances with every VJP recomputed"
            ),
            "within_current": "all_three_unordered_seed_pairs",
            "within_wider": "all_three_unordered_seed_pairs",
            "randomized": {
                "seeds": list(range(9100, 9108)),
                "pairing": (
                    "current and wider models separately instantiated with "
                    "the same listed paired seed"
                ),
                "paired_rng_prefix_limitation": (
                    "draws can share an RNG prefix where unequal-width "
                    "parameter shapes align; this can raise randomized-control "
                    "similarity and makes required separation conservative"
                ),
                "pooled_cosine_bound": "maximum_of_8_cosine_CI_uppers",
                "pooled_discrepancy_bound": (
                    "minimum_of_8_relative_discrepancy_CI_lowers"
                ),
            },
        },
        "thresholds": {
            "identical": {
                "cosine_ci_lower_minimum": 0.9999,
                "relative_discrepancy_ci_upper_maximum": 0.001,
                "norm_ratio_ci_required_range": [0.999, 1.001],
            },
            "primary": {
                "cosine_ci_lower_minimum": 0.95,
                "relative_discrepancy_ci_upper_maximum": 0.25,
                "norm_ratio_ci_required_range": [0.80, 1.25],
                "within_current_point_control": {
                    "cosine": "at_least_minimum",
                    "relative_discrepancy": "at_most_maximum",
                    "abs_log_norm_ratio": "at_most_maximum",
                },
                "randomized_separation": {
                    "cosine_ci_lower": "strictly_above_pooled_upper",
                    "relative_discrepancy_ci_upper": (
                        "strictly_below_pooled_lower"
                    ),
                },
            },
        },
        "failure_policy": "zero_or_nonfinite_aggregate_norm_invalidates_analysis",
        "maximum_claim": (
            "paired local functional similarity of trained models on one "
            "transductive core; not a causal effect or biological mechanism"
        ),
    }


def _canonical_compact_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def locked_seed(*parts: object) -> int:
    """Derive the locked 63-bit seed from a canonical JSON array."""

    digest = hashlib.sha256(_canonical_compact_json(list(parts))).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _tensor_checksum(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def make_rademacher_probe(
    shape: Sequence[int],
    *,
    mask_entry_id: str,
    probe_index: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, dict[str, object]]:
    """Create one common CPU-PCG64 Rademacher probe and its provenance."""

    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("probe shape dimensions must all be positive")
    if not mask_entry_id:
        raise ValueError("mask_entry_id cannot be empty")
    if int(probe_index) < 0:
        raise ValueError("probe_index must be non-negative")

    seed = locked_seed(
        PROTOCOL_BASE_SEED,
        str(mask_entry_id),
        int(probe_index),
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    cpu_values = generator.integers(
        0,
        2,
        size=normalized_shape,
        dtype=np.int8,
    )
    cpu_values = cpu_values * np.int8(2) - np.int8(1)
    probe = torch.from_numpy(cpu_values)
    provenance = {
        "mask_entry_id": str(mask_entry_id),
        "probe_index": int(probe_index),
        "seed": seed,
        "shape": list(normalized_shape),
        "generation_dtype": "int8",
        "checksum_sha256": _tensor_checksum(probe),
    }
    return probe.to(device=device, dtype=torch.float32), provenance


def center_output_logits(logits: Tensor) -> Tensor:
    """Center each gene's four output logits across its class dimension."""

    if logits.ndim != 3 or logits.shape[-1] != NUM_COUNT_TOKENS:
        raise ValueError(
            "logits must have shape [target_nodes, genes, 4]"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    return logits - logits.mean(dim=-1, keepdim=True)


def whole_node_targets(gene_mask: Tensor) -> Tensor:
    """Return target rows after enforcing a literal whole-node mask."""

    if not isinstance(gene_mask, Tensor) or gene_mask.dtype != torch.bool:
        raise TypeError("gene_mask must be a boolean tensor")
    if gene_mask.ndim != 2 or gene_mask.shape[1] <= 0:
        raise ValueError("gene_mask must have shape [nodes, genes]")
    row_counts = gene_mask.sum(dim=1)
    invalid = (row_counts != 0) & (row_counts != gene_mask.shape[1])
    if bool(invalid.any()):
        raise ValueError("the locked analysis accepts whole-node masks only")
    targets = torch.nonzero(row_counts > 0, as_tuple=False).flatten()
    if not targets.numel():
        raise ValueError("whole-node mask selects no target nodes")
    return targets


def select_locked_whole_node_masks(
    bundle: FixedMaskBundle,
) -> tuple[tuple[str, Tensor], ...]:
    """Select and validate the three locked whole-node mask entries."""

    entries = bundle.manifest.get("entries")
    if not isinstance(entries, Sequence):
        raise CategoricalSensitivityError(
            "fixed-mask bundle manifest lacks entries"
        )
    selected: list[tuple[str, Tensor]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CategoricalSensitivityError(
                "fixed-mask manifest entries must be mappings"
            )
        spec = entry.get("spec")
        if not isinstance(spec, Mapping):
            raise CategoricalSensitivityError(
                "fixed-mask entry lacks a spec mapping"
            )
        if str(spec.get("mode")) != "node":
            continue
        entry_id = str(entry.get("entry_id", ""))
        if not entry_id or entry_id not in bundle.masks:
            raise CategoricalSensitivityError(
                "whole-node fixed-mask entry is not materialized"
            )
        mask = torch.from_numpy(
            np.array(bundle.masks[entry_id], dtype=np.bool_, copy=True)
        )
        whole_node_targets(mask)
        selected.append((entry_id, mask))
    if len(selected) != EXPECTED_MASK_COUNT:
        raise CategoricalSensitivityError(
            f"expected exactly {EXPECTED_MASK_COUNT} whole-node masks, "
            f"found {len(selected)}"
        )
    return tuple(selected)


def observed_token_projection_weights(
    encoder: CategoricalNodeEncoder,
) -> Tensor:
    """Stack the four observed-token input projection matrices as K x H x G."""

    if not isinstance(encoder, CategoricalNodeEncoder):
        raise TypeError("encoder must be a CategoricalNodeEncoder")
    if len(encoder.token_projections) != NUM_COUNT_TOKENS + 1:
        raise CategoricalSensitivityError(
            "categorical encoder does not have four observed plus one mask "
            "projection"
        )
    weights = torch.stack(
        [
            encoder.token_projections[token_id].weight
            for token_id in range(NUM_COUNT_TOKENS)
        ],
        dim=0,
    )
    if weights.shape != (
        NUM_COUNT_TOKENS,
        encoder.hidden_dim,
        encoder.num_genes,
    ):
        raise CategoricalSensitivityError(
            "categorical projection weight shapes violate the model contract"
        )
    return weights


def categorical_encoder_preactivation(
    encoder: CategoricalNodeEncoder,
    input_expression: Tensor,
    gene_mask: Tensor,
    node_covariates: Tensor | None,
) -> Tensor:
    """Reproduce the encoder affine sum and expose it as an FP32 leaf."""

    if input_expression.dtype != torch.float32:
        raise TypeError("locked categorical VJPs require float32 token IDs")
    token_ids = encoder._validated_token_ids(input_expression, gene_mask)
    num_nodes = input_expression.shape[0]
    with torch.no_grad():
        preactivation = encoder.bias.detach().expand(num_nodes, -1).clone()
        for token_id, projection in enumerate(encoder.token_projections):
            indicator = (token_ids == token_id).to(dtype=torch.float32)
            preactivation.add_(F.linear(indicator, projection.weight, None))

        if encoder.node_covariate_dim:
            if node_covariates is None:
                raise ValueError("node_covariates are required by the encoder")
            expected = (num_nodes, encoder.node_covariate_dim)
            if tuple(node_covariates.shape) != expected:
                raise ValueError(
                    f"node_covariates must have shape {expected}"
                )
            if node_covariates.dtype != torch.float32:
                raise TypeError(
                    "locked categorical VJPs require float32 covariates"
                )
            assert encoder.covariate_projection is not None
            preactivation.add_(
                F.linear(
                    node_covariates,
                    encoder.covariate_projection.weight,
                    None,
                )
            )
        elif node_covariates is not None and node_covariates.shape != (
            num_nodes,
            0,
        ):
            raise ValueError(
                "node_covariates were supplied to a zero-covariate encoder"
            )
    return preactivation.detach().requires_grad_(True)


def _forward_from_encoder_preactivation(
    model: TokenizedReceiverChunkedEdgeConditionedGATv2,
    preactivation: Tensor,
    *,
    edge_index: Tensor,
    edge_attributes: Tensor,
    target_nodes: Tensor,
) -> Tensor:
    """Run the existing exact graph backbone from the encoder affine sum."""

    num_nodes = preactivation.shape[0]
    prepared_edges = _prepare_edge_index(
        edge_index,
        num_nodes=num_nodes,
        device=preactivation.device,
    )
    prepared_attributes = _prepare_edge_attributes(
        edge_attributes,
        num_edges=prepared_edges.shape[1],
        edge_attribute_dim=model.edge_attribute_dim,
        reference=preactivation,
    )
    targets = _normalize_target_nodes(
        target_nodes,
        num_nodes=num_nodes,
        device=preactivation.device,
    )
    if targets is None:
        raise CategoricalSensitivityError("target_nodes are required")

    node_embedding = model.encoder.dropout(
        F.gelu(model.encoder.normalization(preactivation))
    )
    layout = model._receiver_layout(prepared_edges, num_nodes=num_nodes)
    for block in model.blocks:
        node_embedding, _, _, _ = model._chunked_block_forward(
            block=block,
            node_embedding=node_embedding,
            edge_index=prepared_edges,
            edge_attributes=prepared_attributes,
            layout=layout,
            explanation_receivers=None,
        )
    selected_embedding = _select_targets(node_embedding, targets)
    return model.decoder(selected_embedding)


@contextmanager
def _frozen_eval(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, required in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(required)
        model.train(was_training)


def preactivation_probe_vjp(
    model: TokenizedReceiverChunkedEdgeConditionedGATv2,
    *,
    input_expression: Tensor,
    gene_mask: Tensor,
    edge_index: Tensor,
    edge_attributes: Tensor,
    node_covariates: Tensor | None,
    probe: Tensor,
) -> Tensor:
    """Compute one FP32 centered-logit VJP at encoder preactivation."""

    if not isinstance(
        model, TokenizedReceiverChunkedEdgeConditionedGATv2
    ):
        raise TypeError("model must be the tokenized receiver-chunked G2")
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration as exc:  # pragma: no cover - model always has params.
        raise CategoricalSensitivityError("model has no parameters") from exc
    if model_dtype != torch.float32:
        raise TypeError("locked categorical VJPs require an FP32 model")
    if torch.is_autocast_enabled():
        raise CategoricalSensitivityError(
            "autocast must be disabled for locked FP32 VJPs"
        )

    targets = whole_node_targets(gene_mask).to(
        device=input_expression.device
    )
    expected_probe_shape = (
        int(targets.numel()),
        int(input_expression.shape[1]),
        NUM_COUNT_TOKENS,
    )
    if tuple(probe.shape) != expected_probe_shape:
        raise ValueError(
            f"probe must have shape {expected_probe_shape}, got "
            f"{tuple(probe.shape)}"
        )
    if probe.dtype != torch.float32:
        raise TypeError("probe must be float32")
    if probe.device != input_expression.device:
        raise ValueError("probe and input_expression must share a device")
    if not bool(torch.isfinite(probe).all()):
        raise ValueError("probe must be finite")

    with _frozen_eval(model), torch.enable_grad():
        preactivation = categorical_encoder_preactivation(
            model.encoder,
            input_expression,
            gene_mask,
            node_covariates,
        )
        logits = _forward_from_encoder_preactivation(
            model,
            preactivation,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            target_nodes=targets,
        )
        centered = center_output_logits(logits)
        scalar = torch.sum(centered * probe)
        gradient = torch.autograd.grad(
            scalar,
            preactivation,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

    if gradient.dtype != torch.float32:
        raise CategoricalSensitivityError("VJP was not computed in FP32")
    if not bool(torch.isfinite(gradient).all()):
        raise CategoricalSensitivityError(
            "encoder-preactivation VJP contains non-finite values"
        )
    return gradient.detach()


def materialize_tangent_input_vjp(
    preactivation_gradient: Tensor,
    projection_weights: Tensor,
    observed_mask: Tensor,
) -> Tensor:
    """Materialize a small-case N x G x 4 tangent VJP for validation/tests."""

    _validate_projection_inputs(
        preactivation_gradient,
        projection_weights,
        observed_mask,
    )
    raw = torch.einsum(
        "nh,khg->nkg",
        preactivation_gradient,
        projection_weights,
    )
    tangent = raw - raw.mean(dim=1, keepdim=True)
    tangent = tangent.masked_fill(~observed_mask.unsqueeze(1), 0.0)
    return tangent.permute(0, 2, 1).contiguous()


def _validate_projection_inputs(
    gradient: Tensor,
    weights: Tensor,
    observed_mask: Tensor,
) -> None:
    if gradient.ndim != 2 or not gradient.is_floating_point():
        raise ValueError("preactivation gradient must have shape [nodes, hidden]")
    if (
        weights.ndim != 3
        or weights.shape[0] != NUM_COUNT_TOKENS
        or not weights.is_floating_point()
    ):
        raise ValueError("projection weights must have shape [4, hidden, genes]")
    if gradient.shape[1] != weights.shape[1]:
        raise ValueError("gradient and projection hidden dimensions differ")
    if (
        observed_mask.dtype != torch.bool
        or observed_mask.shape != (gradient.shape[0], weights.shape[2])
    ):
        raise ValueError("observed_mask must have shape [nodes, genes]")
    if gradient.device != weights.device or gradient.device != observed_mask.device:
        raise ValueError("gradient, weights, and observed_mask must share a device")
    if gradient.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("locked tangent projections require FP32 tensors")
    if not bool(torch.isfinite(gradient).all()) or not bool(
        torch.isfinite(weights).all()
    ):
        raise ValueError("gradient and weights must be finite")


def paired_tangent_vjp_statistics(
    reference_gradient: Tensor,
    reference_weights: Tensor,
    candidate_gradient: Tensor,
    candidate_weights: Tensor,
    observed_mask: Tensor,
    *,
    mask_entry_id: str,
    probe_index: int,
    node_chunk_size: int = 64,
) -> ProbeSufficientStatistics:
    """Accumulate exact pair statistics without a full categorical VJP."""

    _validate_projection_inputs(
        reference_gradient,
        reference_weights,
        observed_mask,
    )
    _validate_projection_inputs(
        candidate_gradient,
        candidate_weights,
        observed_mask,
    )
    if reference_gradient.shape[0] != candidate_gradient.shape[0]:
        raise ValueError("paired gradients have different node counts")
    if reference_weights.shape[2] != candidate_weights.shape[2]:
        raise ValueError("paired projection weights have different gene counts")
    if (
        candidate_gradient.device != reference_gradient.device
        or candidate_weights.device != reference_gradient.device
    ):
        raise ValueError("all paired tensors must share a device")
    if node_chunk_size <= 0:
        raise ValueError("node_chunk_size must be positive")

    reference_squared_norm = 0.0
    candidate_squared_norm = 0.0
    cross_inner_product = 0.0
    num_nodes = reference_gradient.shape[0]
    for start in range(0, num_nodes, node_chunk_size):
        stop = min(start + node_chunk_size, num_nodes)
        reference = torch.einsum(
            "nh,khg->nkg",
            reference_gradient[start:stop],
            reference_weights,
        )
        candidate = torch.einsum(
            "nh,khg->nkg",
            candidate_gradient[start:stop],
            candidate_weights,
        )
        reference = reference - reference.mean(dim=1, keepdim=True)
        candidate = candidate - candidate.mean(dim=1, keepdim=True)
        visible = observed_mask[start:stop].unsqueeze(1)
        reference = reference.masked_fill(~visible, 0.0)
        candidate = candidate.masked_fill(~visible, 0.0)

        reference64 = reference.to(dtype=torch.float64)
        candidate64 = candidate.to(dtype=torch.float64)
        reference_squared_norm += float(
            torch.sum(reference64 * reference64).item()
        )
        candidate_squared_norm += float(
            torch.sum(candidate64 * candidate64).item()
        )
        cross_inner_product += float(
            torch.sum(reference64 * candidate64).item()
        )

    return ProbeSufficientStatistics(
        mask_entry_id=str(mask_entry_id),
        probe_index=int(probe_index),
        reference_squared_norm=reference_squared_norm,
        candidate_squared_norm=candidate_squared_norm,
        cross_inner_product=cross_inner_product,
    )


def multi_tangent_vjp_statistics(
    model_vjps: Mapping[str, tuple[Tensor, Tensor]],
    observed_mask: Tensor,
    pairs: Sequence[tuple[str, str]],
    *,
    mask_entry_id: str,
    probe_index: int,
    node_chunk_size: int = 64,
) -> dict[tuple[str, str], ProbeSufficientStatistics]:
    """Project each model once per chunk and score several requested pairs."""

    if not model_vjps:
        raise ValueError("model_vjps cannot be empty")
    if not pairs:
        raise ValueError("pairs cannot be empty")
    labels = set(model_vjps)
    if any(
        not reference
        or not candidate
        or reference == candidate
        or reference not in labels
        or candidate not in labels
        for reference, candidate in pairs
    ):
        raise ValueError(
            "every pair must name two distinct entries in model_vjps"
        )
    if len(set(pairs)) != len(pairs):
        raise ValueError("pairs cannot contain duplicates")
    if node_chunk_size <= 0:
        raise ValueError("node_chunk_size must be positive")

    first_gradient, first_weights = next(iter(model_vjps.values()))
    _validate_projection_inputs(
        first_gradient,
        first_weights,
        observed_mask,
    )
    num_nodes = first_gradient.shape[0]
    num_genes = first_weights.shape[2]
    for gradient, weights in model_vjps.values():
        _validate_projection_inputs(gradient, weights, observed_mask)
        if gradient.shape[0] != num_nodes or weights.shape[2] != num_genes:
            raise ValueError("all model VJPs must share node and gene counts")
        if (
            gradient.device != first_gradient.device
            or weights.device != first_gradient.device
        ):
            raise ValueError("all model VJPs must share a device")

    accumulators = {
        pair: [0.0, 0.0, 0.0] for pair in pairs
    }
    for start in range(0, num_nodes, node_chunk_size):
        stop = min(start + node_chunk_size, num_nodes)
        visible = observed_mask[start:stop].unsqueeze(1)
        projected: dict[str, Tensor] = {}
        for label, (gradient, weights) in model_vjps.items():
            value = torch.einsum(
                "nh,khg->nkg",
                gradient[start:stop],
                weights,
            )
            value = value - value.mean(dim=1, keepdim=True)
            projected[label] = value.masked_fill(~visible, 0.0).to(
                dtype=torch.float64
            )

        for pair in pairs:
            reference = projected[pair[0]]
            candidate = projected[pair[1]]
            accumulators[pair][0] += float(
                torch.sum(reference * reference).item()
            )
            accumulators[pair][1] += float(
                torch.sum(candidate * candidate).item()
            )
            accumulators[pair][2] += float(
                torch.sum(reference * candidate).item()
            )

    return {
        pair: ProbeSufficientStatistics(
            mask_entry_id=str(mask_entry_id),
            probe_index=int(probe_index),
            reference_squared_norm=values[0],
            candidate_squared_norm=values[1],
            cross_inner_product=values[2],
        )
        for pair, values in accumulators.items()
    }


def metrics_from_sufficient_statistics(
    reference_squared_norm: float,
    candidate_squared_norm: float,
    cross_inner_product: float,
) -> PairMetrics:
    """Apply the locked symmetric metrics to aggregated statistics."""

    values = (
        float(reference_squared_norm),
        float(candidate_squared_norm),
        float(cross_inner_product),
    )
    if not all(math.isfinite(value) for value in values):
        raise CategoricalSensitivityError(
            "aggregate sufficient statistics must be finite"
        )
    a2, b2, ab = values
    if a2 <= 0.0 or b2 <= 0.0:
        raise CategoricalSensitivityError(
            "zero or negative aggregate VJP norm invalidates the analysis"
        )
    bound = math.sqrt(a2 * b2)
    tolerance = 1e-10 * max(1.0, abs(ab), bound)
    if abs(ab) > bound + tolerance:
        raise CategoricalSensitivityError(
            "aggregate inner product violates Cauchy-Schwarz"
        )
    clipped_ab = min(max(ab, -bound), bound)
    difference_squared = max(a2 + b2 - 2.0 * clipped_ab, 0.0)
    return PairMetrics(
        cosine=clipped_ab / bound,
        relative_discrepancy=(
            math.sqrt(difference_squared) / ((a2 * b2) ** 0.25)
        ),
        norm_ratio=math.sqrt(b2 / a2),
        reference_squared_norm=a2,
        candidate_squared_norm=b2,
        cross_inner_product=clipped_ab,
    )


def _statistics_grid(
    statistics: Sequence[ProbeSufficientStatistics],
    *,
    mask_entry_ids: Sequence[str],
) -> np.ndarray:
    if len(mask_entry_ids) != EXPECTED_MASK_COUNT:
        raise CategoricalSensitivityError(
            f"expected {EXPECTED_MASK_COUNT} mask entry IDs"
        )
    if len(set(mask_entry_ids)) != EXPECTED_MASK_COUNT:
        raise CategoricalSensitivityError("mask entry IDs must be unique")
    by_key = {
        (row.mask_entry_id, row.probe_index): row for row in statistics
    }
    expected = {
        (str(mask_id), probe_index)
        for mask_id in mask_entry_ids
        for probe_index in range(PROBES_PER_MASK)
    }
    if set(by_key) != expected or len(by_key) != len(statistics):
        raise CategoricalSensitivityError(
            "statistics do not form the locked 3-mask by 32-probe grid"
        )
    grid = np.empty(
        (EXPECTED_MASK_COUNT, PROBES_PER_MASK, 3),
        dtype=np.float64,
    )
    for mask_index, mask_id in enumerate(mask_entry_ids):
        for probe_index in range(PROBES_PER_MASK):
            row = by_key[(str(mask_id), probe_index)]
            grid[mask_index, probe_index] = (
                row.reference_squared_norm,
                row.candidate_squared_norm,
                row.cross_inner_product,
            )
    return grid


def estimate_pair(
    statistics: Sequence[ProbeSufficientStatistics],
    *,
    reference_label: str,
    candidate_label: str,
    mask_entry_ids: Sequence[str],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> PairEstimate:
    """Aggregate one pair and compute the locked hierarchical bootstrap."""

    if not reference_label or not candidate_label:
        raise ValueError("pair labels cannot be empty")
    if int(bootstrap_replicates) != BOOTSTRAP_REPLICATES:
        raise CategoricalSensitivityError(
            f"locked analysis requires {BOOTSTRAP_REPLICATES} bootstraps"
        )
    grid = _statistics_grid(
        statistics,
        mask_entry_ids=mask_entry_ids,
    )
    point_values = grid.mean(axis=(0, 1), dtype=np.float64)
    point = metrics_from_sufficient_statistics(*point_values.tolist())

    bootstrap_seed = locked_seed(PROTOCOL_BASE_SEED, "bootstrap")
    generator = np.random.Generator(np.random.PCG64(bootstrap_seed))
    samples = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_masks = generator.integers(
            0,
            EXPECTED_MASK_COUNT,
            size=EXPECTED_MASK_COUNT,
        )
        accumulated = np.zeros(3, dtype=np.float64)
        for mask_index in selected_masks:
            selected_probes = generator.integers(
                0,
                PROBES_PER_MASK,
                size=PROBES_PER_MASK,
            )
            accumulated += grid[mask_index, selected_probes].mean(
                axis=0,
                dtype=np.float64,
            )
        averaged = accumulated / float(EXPECTED_MASK_COUNT)
        metrics = metrics_from_sufficient_statistics(*averaged.tolist())
        samples[replicate] = (
            metrics.cosine,
            metrics.relative_discrepancy,
            metrics.norm_ratio,
        )

    bounds = np.percentile(
        samples,
        [2.5, 97.5],
        axis=0,
        method="linear",
    )
    return PairEstimate(
        reference_label=reference_label,
        candidate_label=candidate_label,
        point=point,
        cosine_interval=PercentileInterval(
            lower=float(bounds[0, 0]),
            upper=float(bounds[1, 0]),
        ),
        relative_discrepancy_interval=PercentileInterval(
            lower=float(bounds[0, 1]),
            upper=float(bounds[1, 1]),
        ),
        norm_ratio_interval=PercentileInterval(
            lower=float(bounds[0, 2]),
            upper=float(bounds[1, 2]),
        ),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_operational_match(
    *,
    primary_by_seed: Mapping[int, PairEstimate],
    within_current: Mapping[str, PairEstimate],
    within_wider: Mapping[str, PairEstimate],
    identical: PairEstimate,
    randomized_by_seed: Mapping[int, PairEstimate],
) -> dict[str, object]:
    """Apply all locked numerical, calibration, and null thresholds."""

    if set(primary_by_seed) != {0, 1, 2}:
        raise CategoricalSensitivityError(
            "primary comparisons must contain seeds 0, 1, and 2"
        )
    if len(within_current) != 3 or len(within_wider) != 3:
        raise CategoricalSensitivityError(
            "each within-width calibration requires three seed pairs"
        )
    if set(randomized_by_seed) != set(range(9100, 9108)):
        raise CategoricalSensitivityError(
            "randomized controls must contain seeds 9100 through 9107"
        )

    identical_checks = {
        "cosine_ci_lower_at_least_0_9999": (
            identical.cosine_interval.lower >= 0.9999
        ),
        "relative_discrepancy_ci_upper_at_most_0_001": (
            identical.relative_discrepancy_interval.upper <= 0.001
        ),
        "norm_ratio_ci_within_0_999_1_001": (
            identical.norm_ratio_interval.lower >= 0.999
            and identical.norm_ratio_interval.upper <= 1.001
        ),
    }
    identical_valid = all(identical_checks.values())

    within_cosine_floor = min(
        estimate.point.cosine for estimate in within_current.values()
    )
    within_discrepancy_ceiling = max(
        estimate.point.relative_discrepancy
        for estimate in within_current.values()
    )
    within_abs_log_norm_ratio_ceiling = max(
        abs(math.log(estimate.point.norm_ratio))
        for estimate in within_current.values()
    )
    randomized_cosine_upper = max(
        estimate.cosine_interval.upper
        for estimate in randomized_by_seed.values()
    )
    randomized_discrepancy_lower = min(
        estimate.relative_discrepancy_interval.lower
        for estimate in randomized_by_seed.values()
    )

    primary_checks: dict[str, object] = {}
    for seed, estimate in sorted(primary_by_seed.items()):
        checks = {
            "cosine_ci_lower_at_least_0_95": (
                estimate.cosine_interval.lower >= 0.95
            ),
            "relative_discrepancy_ci_upper_at_most_0_25": (
                estimate.relative_discrepancy_interval.upper <= 0.25
            ),
            "norm_ratio_ci_within_0_80_1_25": (
                estimate.norm_ratio_interval.lower >= 0.80
                and estimate.norm_ratio_interval.upper <= 1.25
            ),
            "cosine_point_no_worse_than_within_current": (
                estimate.point.cosine >= within_cosine_floor
            ),
            "relative_discrepancy_point_no_worse_than_within_current": (
                estimate.point.relative_discrepancy
                <= within_discrepancy_ceiling
            ),
            "abs_log_norm_ratio_no_worse_than_within_current": (
                abs(math.log(estimate.point.norm_ratio))
                <= within_abs_log_norm_ratio_ceiling
            ),
            "cosine_separated_from_randomized_controls": (
                estimate.cosine_interval.lower > randomized_cosine_upper
            ),
            "relative_discrepancy_separated_from_randomized_controls": (
                estimate.relative_discrepancy_interval.upper
                < randomized_discrepancy_lower
            ),
        }
        primary_checks[str(seed)] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    primary_valid = all(
        bool(value["passed"])
        for value in primary_checks.values()
        if isinstance(value, Mapping)
    )
    operational_match = identical_valid and primary_valid
    return {
        "analysis_numerically_valid": identical_valid,
        "operational_match": operational_match,
        "identical_control": {
            "passed": identical_valid,
            "checks": identical_checks,
            "estimate": identical.to_dict(),
        },
        "within_current_point_control_bounds": {
            "cosine_minimum": within_cosine_floor,
            "relative_discrepancy_maximum": (
                within_discrepancy_ceiling
            ),
            "abs_log_norm_ratio_maximum": (
                within_abs_log_norm_ratio_ceiling
            ),
        },
        "randomized_control_pooled_bounds": {
            "cosine_ci_upper_maximum": randomized_cosine_upper,
            "relative_discrepancy_ci_lower_minimum": (
                randomized_discrepancy_lower
            ),
        },
        "primary": primary_checks,
        "calibration_estimates": {
            "within_current": {
                label: estimate.to_dict()
                for label, estimate in sorted(within_current.items())
            },
            "within_wider": {
                label: estimate.to_dict()
                for label, estimate in sorted(within_wider.items())
            },
            "randomized": {
                str(seed): estimate.to_dict()
                for seed, estimate in sorted(randomized_by_seed.items())
            },
        },
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CategoricalSensitivityError",
    "EXPECTED_MASK_COUNT",
    "PROBES_PER_MASK",
    "PROTOCOL_BASE_SEED",
    "PROTOCOL_VERSION",
    "PairEstimate",
    "PairMetrics",
    "PercentileInterval",
    "ProbeSufficientStatistics",
    "categorical_encoder_preactivation",
    "center_output_logits",
    "estimate_pair",
    "evaluate_operational_match",
    "locked_protocol_record",
    "locked_seed",
    "make_rademacher_probe",
    "materialize_tangent_input_vjp",
    "multi_tangent_vjp_statistics",
    "observed_token_projection_weights",
    "paired_tangent_vjp_statistics",
    "preactivation_probe_vjp",
    "select_locked_whole_node_masks",
    "whole_node_targets",
]

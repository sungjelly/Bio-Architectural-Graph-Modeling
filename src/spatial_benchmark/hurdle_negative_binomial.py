"""Bernoulli hurdle plus zero-truncated NB2 outputs and likelihoods.

The hurdle gate models whether a masked expression count is positive.  Given a
positive count, the count component is an NB2 distribution conditioned on
``Y > 0``.  The two terms form one normalized likelihood; the count component
is never evaluated for observed zeros.

All likelihood arithmetic is forced to FP32 outside autocast.  Raw targets are
accepted only by the loss helpers and never by a model forward method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .negative_binomial import (
    _fp32_autocast_disabled,
    _masked_nb2_inputs,
    _validate_mu_and_theta,
    nb2_mean_from_logits,
)
from .relative_qkv_graph_transformer import (
    RelativeGeometryQKVModelOutput,
    RelativeGeometryQKVGraphTransformer,
)
from .target_isolated_negative_binomial import (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)


HURDLE_DETECTION_PROBABILITY_EPSILON = 1e-4


def _validate_probability_epsilon(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 0.5:
        raise ValueError("detection probability epsilon must be in (0, 0.5)")
    return value


def _nb2_log_zero_probability_values(mu: Tensor, theta: Tensor) -> Tensor:
    """Return elementwise log P_NB(Y=0) for aligned positive FP32 tensors."""

    return -theta * F.softplus(torch.log(mu) - torch.log(theta))


def log1mexp(log_probability: Tensor) -> Tensor:
    """Stably compute ``log(1 - exp(x))`` for finite ``x < 0``.

    ``log(-expm1(x))`` is accurate when the probability is close to one and
    remains safe when it is close to zero, avoiding eager evaluation of an
    unsafe alternate branch through ``torch.where``.
    """

    if not isinstance(log_probability, Tensor):
        raise TypeError("log_probability must be a tensor")
    if not log_probability.is_floating_point() or log_probability.is_complex():
        raise TypeError("log_probability must be a real floating tensor")
    with _fp32_autocast_disabled(log_probability):
        value = log_probability.float()
        if not bool(torch.isfinite(value).all()) or not bool((value < 0.0).all()):
            raise ValueError("log_probability must be finite and strictly negative")
        result = torch.log(-torch.expm1(value))
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("log1mexp produced a non-finite value")
    return result


def nb2_log_nonzero_probability(mu: Tensor, theta: Tensor) -> Tensor:
    """Return log P_NB(Y>0) for a full cell-by-gene NB2 parameter matrix."""

    _validate_mu_and_theta(mu, theta)
    with _fp32_autocast_disabled(mu):
        work_mu = mu.float()
        work_theta = theta.float().view(1, -1)
        if not bool(torch.isfinite(work_mu).all()) or not bool((work_mu > 0).all()):
            raise ValueError("mu values must be finite and strictly positive")
        if not bool(torch.isfinite(work_theta).all()) or not bool(
            (work_theta > 0).all()
        ):
            raise ValueError("theta values must be finite and strictly positive")
        result = log1mexp(_nb2_log_zero_probability_values(work_mu, work_theta))
    return result


def zero_truncated_nb2_conditional_mean(mu: Tensor, theta: Tensor) -> Tensor:
    """Return E[Y | Y>0] for NB2 mean/inverse-dispersion parameters."""

    log_nonzero = nb2_log_nonzero_probability(mu, theta)
    with _fp32_autocast_disabled(mu):
        result = torch.exp(torch.log(mu.float()) - log_nonzero)
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("zero-truncated NB2 mean is non-finite")
    if not bool((result >= 1.0 - 1e-5).all()):
        raise FloatingPointError("zero-truncated NB2 mean fell below one")
    return result


def zero_truncated_negative_binomial_nll_elementwise(
    mu: Tensor,
    theta: Tensor,
    positive_target: Tensor,
) -> Tensor:
    """Return full-constant zero-truncated NB2 NLL for positive observations."""

    if not all(isinstance(value, Tensor) for value in (mu, theta, positive_target)):
        raise TypeError("mu, theta, and positive_target must be tensors")
    if mu.shape != theta.shape or mu.shape != positive_target.shape or mu.numel() == 0:
        raise ValueError("positive NB2 tensors must be nonempty and shape-aligned")
    if mu.device != theta.device or mu.device != positive_target.device:
        raise ValueError("positive NB2 tensors must share a device")
    if not mu.is_floating_point() or mu.is_complex():
        raise TypeError("mu must be a real floating tensor")
    if not theta.is_floating_point() or theta.is_complex():
        raise TypeError("theta must be a real floating tensor")
    if positive_target.dtype == torch.bool or positive_target.is_complex():
        raise TypeError("positive_target must contain real numeric counts")
    with _fp32_autocast_disabled(mu):
        work_mu = mu.float()
        work_theta = theta.float()
        work_target = positive_target.float()
        if not bool(torch.isfinite(work_mu).all()) or not bool((work_mu > 0).all()):
            raise ValueError("mu must be finite and strictly positive")
        if not bool(torch.isfinite(work_theta).all()) or not bool(
            (work_theta > 0).all()
        ):
            raise ValueError("theta must be finite and strictly positive")
        if not bool(torch.isfinite(work_target).all()) or not bool(
            (work_target > 0).all()
        ):
            raise ValueError("zero-truncated targets must be finite and positive")
        if bool((work_target != work_target.round()).any()):
            raise ValueError("zero-truncated targets must be integer-valued")
        log_nonzero = log1mexp(
            _nb2_log_zero_probability_values(work_mu, work_theta)
        )
        log_mu = torch.log(work_mu)
        log_theta = torch.log(work_theta)
        logit = log_mu - log_theta
        log_combinatorial = (
            torch.lgamma(work_target + work_theta)
            - torch.lgamma(work_theta)
            - torch.lgamma(work_target + 1.0)
        )

        # Direct FP32 log-gamma subtraction loses the entire small gamma ratio
        # when theta is large.  Counts are exact integers, so for only that
        # uncommon branch evaluate the identity
        #   log Gamma(theta+y)-log Gamma(theta)
        #     = y log(theta) + sum_{k=0}^{y-1} log1p(k/theta).
        # Chunking bounds temporary storage even for a large positive shard.
        flat_theta = work_theta.reshape(-1)
        flat_target = work_target.reshape(-1)
        high_indices = torch.nonzero(
            flat_theta >= 4096.0, as_tuple=False
        ).flatten()
        if high_indices.numel():
            flat_combinatorial = log_combinatorial.reshape(-1)
            for start in range(0, int(high_indices.numel()), 4096):
                indices = high_indices[start : start + 4096]
                chunk_theta = flat_theta.index_select(0, indices)
                chunk_target = flat_target.index_select(0, indices)
                maximum_count = int(chunk_target.max().item())
                offsets = torch.arange(
                    maximum_count,
                    dtype=torch.float32,
                    device=work_theta.device,
                )
                active = offsets.view(1, -1) < chunk_target.view(-1, 1)
                correction = torch.where(
                    active,
                    torch.log1p(
                        offsets.view(1, -1) / chunk_theta.view(-1, 1)
                    ),
                    torch.zeros((), dtype=torch.float32, device=work_theta.device),
                ).sum(dim=1, dtype=torch.float32)
                log_rising = chunk_target * torch.log(chunk_theta) + correction
                stable = log_rising - torch.lgamma(chunk_target + 1.0)
                flat_combinatorial = flat_combinatorial.index_copy(
                    0, indices, stable
                )
            log_combinatorial = flat_combinatorial.view_as(log_combinatorial)
        nb_nll = (
            -log_combinatorial
            + work_theta * F.softplus(logit)
            + work_target * F.softplus(-logit)
        )
        values = nb_nll + log_nonzero
        # A conditional probability cannot exceed one.  Permit only FP32-scale
        # cancellation below zero, then remove that numerical artifact.
        if bool((values < -1e-5).any()):
            raise FloatingPointError("zero-truncated NB2 NLL became materially negative")
        values = values.clamp_min(0.0)
    if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("zero-truncated NB2 NLL is non-finite")
    return values


@dataclass(frozen=True)
class MaskedHurdleNegativeBinomialLossSums:
    """Additive FP32 likelihood components over one masked target shard."""

    masked_entries: int
    positive_entries: int
    joint_nll_sum: Tensor
    detection_nll_sum: Tensor
    positive_count_nll_sum: Tensor


def masked_hurdle_negative_binomial_nll_sums(
    nonzero_logits: Tensor,
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> MaskedHurdleNegativeBinomialLossSums:
    """Return exact additive hurdle likelihood components on masked entries.

    The detection term is evaluated for every masked entry.  The count term is
    evaluated only for observed positives and is normalized as a zero-truncated
    NB2 distribution.  A differentiable zero connection materializes zero
    gradients for count parameters on an all-zero local shard, which keeps DDP
    synchronization well-defined without changing the objective.
    """

    if not isinstance(nonzero_logits, Tensor):
        raise TypeError("nonzero_logits must be a tensor")
    if nonzero_logits.shape != mu.shape or nonzero_logits.device != mu.device:
        raise ValueError("nonzero_logits must be shape- and device-aligned with mu")
    if not nonzero_logits.is_floating_point() or nonzero_logits.is_complex():
        raise TypeError("nonzero_logits must be a real floating tensor")
    selected_mu, selected_theta, selected_target = _masked_nb2_inputs(
        mu,
        theta,
        raw_target,
        mask,
    )
    with _fp32_autocast_disabled(mu):
        selected_logits = nonzero_logits.masked_select(mask).float()
        if not bool(torch.isfinite(selected_logits).all()):
            raise ValueError("masked nonzero logits must be finite")
        positive = selected_target > 0.0
        binary_target = positive.float()
        detection_values = F.binary_cross_entropy_with_logits(
            selected_logits,
            binary_target,
            reduction="none",
        )
        detection_sum = detection_values.sum(dtype=torch.float32)
        if bool(positive.any()):
            positive_values = zero_truncated_negative_binomial_nll_elementwise(
                selected_mu[positive],
                selected_theta[positive],
                selected_target[positive],
            )
            positive_sum = positive_values.sum(dtype=torch.float32)
        else:
            positive_sum = (selected_mu.sum() + selected_theta.sum()) * 0.0
        joint_sum = detection_sum + positive_sum
    for name, value in (
        ("joint", joint_sum),
        ("detection", detection_sum),
        ("positive count", positive_sum),
    ):
        if value.dtype != torch.float32 or not bool(torch.isfinite(value)):
            raise FloatingPointError(f"masked hurdle {name} NLL sum is non-finite")
    return MaskedHurdleNegativeBinomialLossSums(
        masked_entries=int(selected_target.numel()),
        positive_entries=int(positive.sum().item()),
        joint_nll_sum=joint_sum,
        detection_nll_sum=detection_sum,
        positive_count_nll_sum=positive_sum,
    )


def masked_hurdle_negative_binomial_nll(
    nonzero_logits: Tensor,
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return entry-pooled joint hurdle NLL over exactly the masked entries."""

    sums = masked_hurdle_negative_binomial_nll_sums(
        nonzero_logits,
        mu,
        theta,
        raw_target,
        mask,
    )
    loss = sums.joint_nll_sum / float(sums.masked_entries)
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("masked hurdle NLL is non-finite")
    return loss


class _SharedHurdleExpressionDecoder(nn.Module):
    """Reuse the count decoder trunk and add one gene-wise detection projection."""

    def __init__(
        self,
        count_decoder: nn.Module,
        *,
        num_genes: int,
        initial_nonzero_probability: Tensor,
    ) -> None:
        super().__init__()
        for field in ("linear_in", "dropout", "linear_out"):
            if not hasattr(count_decoder, field):
                raise TypeError("count decoder does not expose the expected shared trunk")
        linear_in = count_decoder.linear_in
        linear_out = count_decoder.linear_out
        if not isinstance(linear_in, nn.Linear) or not isinstance(linear_out, nn.Linear):
            raise TypeError("count decoder projections must be linear layers")
        if linear_out.out_features != num_genes:
            raise ValueError("count decoder output is not gene-aligned")
        if initial_nonzero_probability.shape != (num_genes,):
            raise ValueError("initial nonzero probability must have shape [genes]")
        probability = initial_nonzero_probability.detach().float()
        if not bool(torch.isfinite(probability).all()) or not bool(
            ((probability > 0.0) & (probability < 1.0)).all()
        ):
            raise ValueError("initial nonzero probabilities must be finite in (0, 1)")

        self.linear_in = linear_in
        self.dropout = count_decoder.dropout
        self.linear_out = linear_out
        self.detection_linear_out = nn.Linear(linear_out.in_features, num_genes)
        nn.init.zeros_(self.detection_linear_out.weight)
        with torch.no_grad():
            self.detection_linear_out.bias.copy_(torch.logit(probability))

    def forward(self, node_embedding: Tensor) -> Tensor:
        hidden = self.dropout(F.gelu(self.linear_in(node_embedding)))
        count_logits = self.linear_out(hidden)
        nonzero_logits = self.detection_linear_out(hidden)
        return torch.cat((count_logits, nonzero_logits), dim=-1)


@dataclass(kw_only=True)
class HurdleNegativeBinomialModelOutput(RelativeGeometryQKVModelOutput):
    """Target output for a Bernoulli hurdle and zero-truncated NB2 count."""

    nb_mean: Tensor
    theta: Tensor
    nonzero_logits: Tensor
    nonzero_probability: Tensor
    predicted_zero_probability: Tensor
    positive_conditional_mean: Tensor
    marginal_mean: Tensor


class TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer(
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer
):
    """Four-block target-isolated geometry model with a proper hurdle-NB2 head."""

    observation_model = "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"

    def __init__(
        self,
        *args: object,
        detection_initial_nonzero_probability: Optional[Tensor] = None,
        detection_probability_epsilon: float = (
            HURDLE_DETECTION_PROBABILITY_EPSILON
        ),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        epsilon = _validate_probability_epsilon(detection_probability_epsilon)
        if detection_initial_nonzero_probability is None:
            probability = torch.full(
                (self.num_genes,),
                0.5,
                dtype=torch.float32,
            )
        else:
            probability = detection_initial_nonzero_probability.detach().float().clone()
        probability.clamp_(epsilon, 1.0 - epsilon)
        self.register_buffer(
            "detection_initial_nonzero_probability",
            probability.clone(),
            persistent=True,
        )
        self.decoder = _SharedHurdleExpressionDecoder(
            self.decoder,
            num_genes=self.num_genes,
            initial_nonzero_probability=probability,
        )
        self.detection_probability_epsilon = epsilon
        self.observation_model = (
            "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"
        )

    def _make_output(
        self,
        *,
        prediction: Tensor,
        selected_embedding: Tensor,
        full_embedding: Tensor,
        node_encoder_embedding: Optional[Tensor],
        graph_step_embeddings: Optional[tuple[Tensor, ...]],
        edge_index: Optional[Tensor],
        attention: Optional[Tensor],
        content_logits: Optional[Tensor],
        positional_bias: Optional[Tensor],
        combined_logits: Optional[Tensor],
        layer_number: Optional[int],
    ) -> HurdleNegativeBinomialModelOutput:
        expected_shape = (selected_embedding.shape[0], 2 * self.num_genes)
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"hurdle decoder must return {expected_shape}, got {tuple(prediction.shape)}"
            )
        count_logits, nonzero_logits = prediction.split(self.num_genes, dim=-1)
        nb_mean = nb2_mean_from_logits(count_logits, epsilon=self.mean_epsilon)
        theta = self.theta
        if theta.device != nb_mean.device or theta.shape != (self.num_genes,):
            raise RuntimeError("hurdle theta must be device- and gene-aligned")
        with _fp32_autocast_disabled(nonzero_logits):
            work_logits = nonzero_logits.float()
            if not bool(torch.isfinite(work_logits).all()):
                raise FloatingPointError("hurdle detection logits are non-finite")
            nonzero_probability = torch.sigmoid(work_logits)
        positive_mean = zero_truncated_nb2_conditional_mean(nb_mean, theta)
        marginal_mean = nonzero_probability * positive_mean
        predicted_zero_probability = 1.0 - nonzero_probability
        for name, value in (
            ("nonzero probability", nonzero_probability),
            ("positive conditional mean", positive_mean),
            ("marginal mean", marginal_mean),
        ):
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"hurdle {name} is non-finite")

        base_output = RelativeGeometryQKVGraphTransformer._make_output(
            self,
            prediction=marginal_mean,
            selected_embedding=selected_embedding,
            full_embedding=full_embedding,
            node_encoder_embedding=node_encoder_embedding,
            graph_step_embeddings=graph_step_embeddings,
            edge_index=edge_index,
            attention=attention,
            content_logits=content_logits,
            positional_bias=positional_bias,
            combined_logits=combined_logits,
            layer_number=layer_number,
        )
        return HurdleNegativeBinomialModelOutput(
            **vars(base_output),
            nb_mean=nb_mean,
            theta=theta,
            nonzero_logits=work_logits,
            nonzero_probability=nonzero_probability,
            predicted_zero_probability=predicted_zero_probability,
            positive_conditional_mean=positive_mean,
            marginal_mean=marginal_mean,
        )


__all__ = [
    "HURDLE_DETECTION_PROBABILITY_EPSILON",
    "HurdleNegativeBinomialModelOutput",
    "MaskedHurdleNegativeBinomialLossSums",
    "TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer",
    "log1mexp",
    "masked_hurdle_negative_binomial_nll",
    "masked_hurdle_negative_binomial_nll_sums",
    "nb2_log_nonzero_probability",
    "zero_truncated_nb2_conditional_mean",
    "zero_truncated_negative_binomial_nll_elementwise",
]

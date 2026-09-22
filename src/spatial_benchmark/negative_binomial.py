"""NB2 output and masked raw-count metrics for the SO2 geometry model.

The graph encoder and decoder in this module are exactly the existing
four-block geometry-modulated relative-QKV implementation.  The only learned
addition is one cell-shared inverse-dispersion parameter per gene.  Decoder
logits are mapped to positive raw-count means and raw inverse dispersions are
mapped to positive values with ``softplus + 1e-4``.

Raw counts are deliberately absent from the model forward signature.  They are
accepted only by the likelihood and descriptive metric helpers below.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import expm1, isfinite, log
from typing import ContextManager, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry_modulated_relative_qkv_graph_transformer import (
    DEFAULT_BIAS_BOUND,
    DEFAULT_LOGIT_SCALE_INITIAL,
    DEFAULT_LOGIT_SCALE_MAXIMUM,
    DEFAULT_LOGIT_SCALE_MINIMUM,
    DEFAULT_MODULATION_AMPLITUDE,
    DEFAULT_QK_NORMALIZATION_EPSILON,
    GeometryModulatedRelativeQKVGraphTransformer,
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from .relative_qkv_graph_transformer import (
    RELATIVE_GEOMETRY_DIM,
    RelativeGeometryQKVModelOutput,
)


NB2_MEAN_EPSILON = 1e-4
NB2_INVERSE_DISPERSION_EPSILON = 1e-4
NB2_INITIAL_INVERSE_DISPERSION = 1.0
NB2_GRAPH_LAYERS = 4


def _fp32_autocast_disabled(reference: Tensor) -> ContextManager[object]:
    if reference.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=reference.device.type, enabled=False)
    return nullcontext()


def _validate_positive_epsilon(value: float, *, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def nb2_mean_from_logits(
    decoder_logits: Tensor,
    *,
    epsilon: float = NB2_MEAN_EPSILON,
) -> Tensor:
    """Map arbitrary decoder logits to strictly positive FP32 means."""

    if not isinstance(decoder_logits, Tensor):
        raise TypeError("decoder_logits must be a tensor")
    if decoder_logits.ndim != 2 or not all(decoder_logits.shape):
        raise ValueError("decoder_logits must be nonempty with shape [cells, genes]")
    if not decoder_logits.is_floating_point() or decoder_logits.is_complex():
        raise TypeError("decoder_logits must be a real floating tensor")
    epsilon = _validate_positive_epsilon(epsilon, name="mean epsilon")
    with _fp32_autocast_disabled(decoder_logits):
        logits = decoder_logits.float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("decoder_logits must contain only finite values")
        mean = F.softplus(logits) + epsilon
    if mean.dtype != torch.float32 or not bool(torch.isfinite(mean).all()):
        raise FloatingPointError(
            "NB2 mean transform did not produce finite FP32 values"
        )
    return mean


def nb2_inverse_dispersion_from_raw(
    raw_theta: Tensor,
    *,
    epsilon: float = NB2_INVERSE_DISPERSION_EPSILON,
) -> Tensor:
    """Map a raw per-gene parameter to positive FP32 inverse dispersion."""

    if not isinstance(raw_theta, Tensor):
        raise TypeError("raw_theta must be a tensor")
    if raw_theta.ndim != 1 or raw_theta.numel() == 0:
        raise ValueError("raw_theta must be nonempty with shape [genes]")
    if not raw_theta.is_floating_point() or raw_theta.is_complex():
        raise TypeError("raw_theta must be a real floating tensor")
    epsilon = _validate_positive_epsilon(
        epsilon,
        name="inverse-dispersion epsilon",
    )
    with _fp32_autocast_disabled(raw_theta):
        raw = raw_theta.float()
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("raw_theta must contain only finite values")
        theta = F.softplus(raw) + epsilon
    if theta.dtype != torch.float32 or not bool(torch.isfinite(theta).all()):
        raise FloatingPointError(
            "NB2 inverse-dispersion transform did not produce finite FP32 values"
        )
    return theta


@dataclass(kw_only=True)
class NegativeBinomialModelOutput(RelativeGeometryQKVModelOutput):
    """Relative-QKV output with an NB2 mean and per-gene inverse dispersion."""

    mu: Tensor
    theta: Tensor


class _NegativeBinomialOutputMixin:
    raw_theta: nn.Parameter
    mean_epsilon: float
    inverse_dispersion_epsilon: float
    inverse_dispersion_initial_value: float

    def _initialize_negative_binomial_output(
        self,
        *,
        mean_epsilon: float,
        inverse_dispersion_epsilon: float,
        inverse_dispersion_initial_value: float,
    ) -> None:
        mean_epsilon = _validate_positive_epsilon(
            mean_epsilon,
            name="mean_epsilon",
        )
        inverse_dispersion_epsilon = _validate_positive_epsilon(
            inverse_dispersion_epsilon,
            name="inverse_dispersion_epsilon",
        )
        inverse_dispersion_initial_value = _validate_positive_epsilon(
            inverse_dispersion_initial_value,
            name="inverse_dispersion_initial_value",
        )
        if mean_epsilon != NB2_MEAN_EPSILON:
            raise ValueError("mean_epsilon is frozen at 1e-4")
        if inverse_dispersion_epsilon != NB2_INVERSE_DISPERSION_EPSILON:
            raise ValueError("inverse_dispersion_epsilon is frozen at 1e-4")
        if inverse_dispersion_initial_value != NB2_INITIAL_INVERSE_DISPERSION:
            raise ValueError("inverse_dispersion_initial_value is frozen at 1.0")
        if int(self.graph_layers) != NB2_GRAPH_LAYERS:  # type: ignore[attr-defined]
            raise ValueError(
                "the NB2 geometry model requires exactly four graph blocks"
            )
        num_genes = int(self.num_genes)  # type: ignore[attr-defined]
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")

        # softplus(raw) + epsilon == the frozen initial theta.  Register this
        # after the backbone so its fresh-seed initialization stays identical.
        positive_before_epsilon = (
            inverse_dispersion_initial_value - inverse_dispersion_epsilon
        )
        if positive_before_epsilon <= 0.0:
            raise ValueError(
                "inverse_dispersion_initial_value must exceed its epsilon"
            )
        raw_initial = log(expm1(positive_before_epsilon))
        self.raw_theta = nn.Parameter(
            torch.full((num_genes,), raw_initial, dtype=torch.float32)
        )
        self.mean_epsilon = mean_epsilon
        self.inverse_dispersion_epsilon = inverse_dispersion_epsilon
        self.inverse_dispersion_initial_value = inverse_dispersion_initial_value
        self.observation_model = "negative_binomial_nb2_mean_inverse_dispersion"

    @property
    def theta(self) -> Tensor:
        return nb2_inverse_dispersion_from_raw(
            self.raw_theta,
            epsilon=self.inverse_dispersion_epsilon,
        )

    def _make_output(
        self,
        *,
        prediction: Tensor,
        selected_embedding: Tensor,
        full_embedding: Tensor,
        node_encoder_embedding: Optional[Tensor],
        edge_index: Optional[Tensor],
        attention: Optional[Tensor],
        content_logits: Optional[Tensor],
        positional_bias: Optional[Tensor],
        combined_logits: Optional[Tensor],
        layer_number: Optional[int],
    ) -> NegativeBinomialModelOutput:
        # Avoid a separate full-matrix finite-reduction synchronization on every
        # training forward.  The masked likelihood validates exactly the values
        # it consumes, while preflight validates the complete output matrix.
        with _fp32_autocast_disabled(prediction):
            mu = F.softplus(prediction.float()) + self.mean_epsilon
        theta = self.theta
        if theta.device != mu.device or theta.shape != (mu.shape[1],):
            raise RuntimeError("NB2 theta must be device- and gene-aligned with mu")
        base_output = super()._make_output(  # type: ignore[misc]
            prediction=mu,
            selected_embedding=selected_embedding,
            full_embedding=full_embedding,
            node_encoder_embedding=node_encoder_embedding,
            edge_index=edge_index,
            attention=attention,
            content_logits=content_logits,
            positional_bias=positional_bias,
            combined_logits=combined_logits,
            layer_number=layer_number,
        )
        return NegativeBinomialModelOutput(
            **vars(base_output),
            mu=mu,
            theta=theta,
        )


class GeometryModulatedNegativeBinomialGraphTransformer(
    _NegativeBinomialOutputMixin,
    GeometryModulatedRelativeQKVGraphTransformer,
):
    """Full-edge four-block geometry-modulated NB2 reference model."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        graph_layers: int = NB2_GRAPH_LAYERS,
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
        mean_epsilon: float = NB2_MEAN_EPSILON,
        inverse_dispersion_epsilon: float = NB2_INVERSE_DISPERSION_EPSILON,
        inverse_dispersion_initial_value: float = NB2_INITIAL_INVERSE_DISPERSION,
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
            geometry_hidden_dim=geometry_hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relative_geometry_dim=relative_geometry_dim,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
        )
        self._initialize_negative_binomial_output(
            mean_epsilon=mean_epsilon,
            inverse_dispersion_epsilon=inverse_dispersion_epsilon,
            inverse_dispersion_initial_value=inverse_dispersion_initial_value,
        )


class ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
    _NegativeBinomialOutputMixin,
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
):
    """Exact receiver-chunked four-block geometry-modulated NB2 model."""

    def __init__(
        self,
        num_genes: int,
        node_covariate_dim: int = 0,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        graph_layers: int = NB2_GRAPH_LAYERS,
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
        mean_epsilon: float = NB2_MEAN_EPSILON,
        inverse_dispersion_epsilon: float = NB2_INVERSE_DISPERSION_EPSILON,
        inverse_dispersion_initial_value: float = NB2_INITIAL_INVERSE_DISPERSION,
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
            geometry_hidden_dim=geometry_hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relative_geometry_dim=relative_geometry_dim,
            qk_normalization_epsilon=qk_normalization_epsilon,
            logit_scale_initial=logit_scale_initial,
            logit_scale_minimum=logit_scale_minimum,
            logit_scale_maximum=logit_scale_maximum,
            modulation_amplitude=modulation_amplitude,
            geometry_bias_bound=geometry_bias_bound,
            receiver_chunk_size=receiver_chunk_size,
            max_edges_per_chunk=max_edges_per_chunk,
            activation_checkpointing=activation_checkpointing,
        )
        self._initialize_negative_binomial_output(
            mean_epsilon=mean_epsilon,
            inverse_dispersion_epsilon=inverse_dispersion_epsilon,
            inverse_dispersion_initial_value=inverse_dispersion_initial_value,
        )


def _validate_mu_and_theta(mu: Tensor, theta: Tensor) -> None:
    if not isinstance(mu, Tensor) or not isinstance(theta, Tensor):
        raise TypeError("mu and theta must be tensors")
    if mu.ndim != 2 or not all(mu.shape):
        raise ValueError("mu must be nonempty with shape [cells, genes]")
    if theta.ndim != 1 or theta.shape[0] != mu.shape[1]:
        raise ValueError("theta must have shape [genes] aligned with mu")
    if not mu.is_floating_point() or mu.is_complex():
        raise TypeError("mu must be a real floating tensor")
    if not theta.is_floating_point() or theta.is_complex():
        raise TypeError("theta must be a real floating tensor")
    if theta.device != mu.device:
        raise ValueError("mu and theta must be on the same device")


def _validate_target_and_mask(
    raw_target: Tensor,
    mask: Tensor,
    *,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> None:
    if not isinstance(raw_target, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("raw_target and mask must be tensors")
    if raw_target.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError("mu, raw_target, and mask shapes must match")
    if raw_target.device != expected_device or mask.device != expected_device:
        raise ValueError("mu, raw_target, theta, and mask must share a device")
    if raw_target.dtype == torch.bool or raw_target.is_complex():
        raise TypeError("raw_target must contain real numeric counts")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")


def _validate_raw_count_values(raw: Tensor) -> None:
    if raw.is_floating_point() and not bool(torch.isfinite(raw).all()):
        raise ValueError("masked raw targets must be finite")
    if bool((raw < 0).any()):
        raise ValueError("masked raw targets must be nonnegative")
    if raw.is_floating_point() and bool((raw != raw.round()).any()):
        raise ValueError("masked raw targets must be integer-valued")


def _masked_mu_target(
    mu: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if not isinstance(mu, Tensor):
        raise TypeError("mu must be a tensor")
    if mu.ndim != 2 or not all(mu.shape):
        raise ValueError("mu must be nonempty with shape [cells, genes]")
    if not mu.is_floating_point() or mu.is_complex():
        raise TypeError("mu must be a real floating tensor")
    _validate_target_and_mask(
        raw_target,
        mask,
        expected_shape=mu.shape,
        expected_device=mu.device,
    )
    selected_mu = mu.masked_select(mask)
    selected_target = raw_target.masked_select(mask)
    if selected_mu.numel() == 0:
        raise ValueError("masked metrics require at least one selected target")
    _validate_raw_count_values(selected_target)
    with _fp32_autocast_disabled(mu):
        selected_mu = selected_mu.float()
        selected_target = selected_target.float()
        if not bool(torch.isfinite(selected_mu).all()) or not bool(
            (selected_mu > 0).all()
        ):
            raise ValueError("masked mu values must be finite and strictly positive")
    return selected_mu, selected_target


def _masked_nb2_inputs(
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_mu_and_theta(mu, theta)
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    selected_theta = theta.view(1, -1).expand_as(mu).masked_select(mask)
    with _fp32_autocast_disabled(mu):
        selected_theta = selected_theta.float()
        if not bool(torch.isfinite(selected_theta).all()) or not bool(
            (selected_theta > 0).all()
        ):
            raise ValueError("theta values must be finite and strictly positive")
    return selected_mu, selected_theta, selected_target


def _nb2_nll_values(mu: Tensor, theta: Tensor, raw_target: Tensor) -> Tensor:
    log_mu = torch.log(mu)
    log_theta = torch.log(theta)
    logit = log_mu - log_theta
    log_combinatorial = (
        torch.lgamma(raw_target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(raw_target + 1.0)
    )
    log_probability = (
        log_combinatorial
        - theta * F.softplus(logit)
        - raw_target * F.softplus(-logit)
    )
    return -log_probability


def negative_binomial_nll_elementwise(
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
) -> Tensor:
    """Return full-constant NB2 NLL values in FP32 without reduction."""

    _validate_mu_and_theta(mu, theta)
    if not isinstance(raw_target, Tensor):
        raise TypeError("raw_target must be a tensor")
    if raw_target.shape != mu.shape or raw_target.device != mu.device:
        raise ValueError("raw_target must be shape- and device-aligned with mu")
    if raw_target.dtype == torch.bool or raw_target.is_complex():
        raise TypeError("raw_target must contain real numeric counts")
    _validate_raw_count_values(raw_target)
    with _fp32_autocast_disabled(mu):
        work_mu = mu.float()
        work_theta = theta.float().view(1, -1)
        work_target = raw_target.float()
        if not bool(torch.isfinite(work_mu).all()) or not bool((work_mu > 0).all()):
            raise ValueError("mu values must be finite and strictly positive")
        if not bool(torch.isfinite(work_theta).all()) or not bool(
            (work_theta > 0).all()
        ):
            raise ValueError("theta values must be finite and strictly positive")
        values = _nb2_nll_values(work_mu, work_theta, work_target)
    if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("NB2 NLL produced non-finite values")
    return values


def masked_negative_binomial_nll(
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return mean full-constant NB2 NLL over exactly the masked entries."""

    selected_mu, selected_theta, selected_target = _masked_nb2_inputs(
        mu,
        theta,
        raw_target,
        mask,
    )
    with _fp32_autocast_disabled(mu):
        values = _nb2_nll_values(selected_mu, selected_theta, selected_target)
        loss = values.sum(dtype=torch.float32) / float(values.numel())
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("masked NB2 NLL is non-finite")
    return loss


def negative_binomial_zero_probability(mu: Tensor, theta: Tensor) -> Tensor:
    """Return ``P(Y=0)`` under the NB2 mean/inverse-dispersion convention."""

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
        logit = torch.log(work_mu) - torch.log(work_theta)
        probability = torch.exp(-work_theta * F.softplus(logit)).clamp_(
            0.0,
            1.0,
        )
    return probability


def masked_raw_count_mae(mu: Tensor, raw_target: Tensor, mask: Tensor) -> Tensor:
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    return torch.mean(torch.abs(selected_mu - selected_target), dtype=torch.float32)


def masked_raw_count_rmse(mu: Tensor, raw_target: Tensor, mask: Tensor) -> Tensor:
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    squared = torch.square(selected_mu - selected_target)
    return torch.sqrt(torch.mean(squared, dtype=torch.float32))


def masked_log1p_mae(mu: Tensor, raw_target: Tensor, mask: Tensor) -> Tensor:
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    difference = torch.log1p(selected_mu) - torch.log1p(selected_target)
    return torch.mean(torch.abs(difference), dtype=torch.float32)


def masked_log1p_rmse(mu: Tensor, raw_target: Tensor, mask: Tensor) -> Tensor:
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    difference = torch.log1p(selected_mu) - torch.log1p(selected_target)
    return torch.sqrt(torch.mean(torch.square(difference), dtype=torch.float32))


def masked_poisson_deviance(
    mu: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    selected_mu, selected_target = _masked_mu_target(mu, raw_target, mask)
    values = 2.0 * (
        torch.xlogy(selected_target, selected_target / selected_mu)
        - selected_target
        + selected_mu
    )
    return torch.mean(values.clamp_min(0.0), dtype=torch.float32)


def masked_observed_zero_rate(
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    if not isinstance(raw_target, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("raw_target and mask must be tensors")
    if raw_target.ndim != 2 or not all(raw_target.shape):
        raise ValueError("raw_target must be nonempty with shape [cells, genes]")
    if mask.shape != raw_target.shape or mask.device != raw_target.device:
        raise ValueError("mask must be shape- and device-aligned with raw_target")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if raw_target.dtype == torch.bool or raw_target.is_complex():
        raise TypeError("raw_target must contain real numeric counts")
    selected_target = raw_target.masked_select(mask)
    if selected_target.numel() == 0:
        raise ValueError("masked metrics require at least one selected target")
    _validate_raw_count_values(selected_target)
    return torch.mean((selected_target == 0).float(), dtype=torch.float32)


def masked_predicted_zero_probability_mean(
    mu: Tensor,
    theta: Tensor,
    mask: Tensor,
) -> Tensor:
    _validate_mu_and_theta(mu, theta)
    if (
        not isinstance(mask, Tensor)
        or mask.shape != mu.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("mask must be boolean and shape-aligned with mu")
    if mask.device != mu.device:
        raise ValueError("mask and mu must share a device")
    if not bool(mask.any()):
        raise ValueError("masked metrics require at least one selected target")
    selected_mu = mu.masked_select(mask).float()
    selected_theta = theta.view(1, -1).expand_as(mu).masked_select(mask).float()
    if not bool(torch.isfinite(selected_mu).all()) or not bool(
        (selected_mu > 0).all()
    ):
        raise ValueError("masked mu values must be finite and strictly positive")
    if not bool(torch.isfinite(selected_theta).all()) or not bool(
        (selected_theta > 0).all()
    ):
        raise ValueError("theta values must be finite and strictly positive")
    logit = torch.log(selected_mu) - torch.log(selected_theta)
    p0 = torch.exp(-selected_theta * F.softplus(logit)).clamp_(0.0, 1.0)
    return torch.mean(p0, dtype=torch.float32)


def masked_zero_brier_score(
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    selected_mu, selected_theta, selected_target = _masked_nb2_inputs(
        mu,
        theta,
        raw_target,
        mask,
    )
    logit = torch.log(selected_mu) - torch.log(selected_theta)
    p0 = torch.exp(-selected_theta * F.softplus(logit)).clamp_(0.0, 1.0)
    observed_zero = (selected_target == 0).float()
    return torch.mean(torch.square(p0 - observed_zero), dtype=torch.float32)


@dataclass(frozen=True)
class MaskedNegativeBinomialMetrics:
    n_masked_entries: int
    negative_binomial_nll: Tensor
    raw_count_mae: Tensor
    raw_count_rmse: Tensor
    log1p_mae: Tensor
    log1p_rmse: Tensor
    poisson_deviance: Tensor
    observed_zero_rate: Tensor
    predicted_zero_probability_mean: Tensor
    zero_brier_score: Tensor


def masked_negative_binomial_metrics(
    mu: Tensor,
    theta: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> MaskedNegativeBinomialMetrics:
    """Compute all frozen masked raw-count metrics from one validated selection."""

    selected_mu, selected_theta, selected_target = _masked_nb2_inputs(
        mu,
        theta,
        raw_target,
        mask,
    )
    with _fp32_autocast_disabled(mu):
        nll_values = _nb2_nll_values(selected_mu, selected_theta, selected_target)
        raw_error = selected_mu - selected_target
        log_error = torch.log1p(selected_mu) - torch.log1p(selected_target)
        poisson_values = 2.0 * (
            torch.xlogy(selected_target, selected_target / selected_mu)
            - selected_target
            + selected_mu
        )
        logit = torch.log(selected_mu) - torch.log(selected_theta)
        p0 = torch.exp(-selected_theta * F.softplus(logit)).clamp_(0.0, 1.0)
        observed_zero = (selected_target == 0).float()

        def mean(value: Tensor) -> Tensor:
            return value.sum(dtype=torch.float32) / float(value.numel())

        result = MaskedNegativeBinomialMetrics(
            n_masked_entries=int(selected_target.numel()),
            negative_binomial_nll=mean(nll_values),
            raw_count_mae=mean(torch.abs(raw_error)),
            raw_count_rmse=torch.sqrt(mean(torch.square(raw_error))),
            log1p_mae=mean(torch.abs(log_error)),
            log1p_rmse=torch.sqrt(mean(torch.square(log_error))),
            poisson_deviance=mean(poisson_values.clamp_min(0.0)),
            observed_zero_rate=mean(observed_zero),
            predicted_zero_probability_mean=mean(p0),
            zero_brier_score=mean(torch.square(p0 - observed_zero)),
        )
    for name, value in vars(result).items():
        if name != "n_masked_entries" and not bool(torch.isfinite(value)):
            raise FloatingPointError(f"masked NB2 metric {name} is non-finite")
    return result


__all__ = [
    "GeometryModulatedNegativeBinomialGraphTransformer",
    "MaskedNegativeBinomialMetrics",
    "NB2_GRAPH_LAYERS",
    "NB2_INITIAL_INVERSE_DISPERSION",
    "NB2_INVERSE_DISPERSION_EPSILON",
    "NB2_MEAN_EPSILON",
    "NegativeBinomialModelOutput",
    "ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer",
    "masked_log1p_mae",
    "masked_log1p_rmse",
    "masked_negative_binomial_metrics",
    "masked_negative_binomial_nll",
    "masked_observed_zero_rate",
    "masked_poisson_deviance",
    "masked_predicted_zero_probability_mean",
    "masked_raw_count_mae",
    "masked_raw_count_rmse",
    "masked_zero_brier_score",
    "nb2_inverse_dispersion_from_raw",
    "nb2_mean_from_logits",
    "negative_binomial_nll_elementwise",
    "negative_binomial_zero_probability",
]

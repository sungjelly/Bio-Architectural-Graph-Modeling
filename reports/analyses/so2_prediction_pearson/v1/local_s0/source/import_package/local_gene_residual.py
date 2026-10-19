"""A masked-cell MLP with a direct, same-gene neighborhood residual.

Expression passed to this module is already standardized with training-only
statistics.  ``gene_mask=True`` means hidden, and hidden values are removed
inside ``forward`` before any arithmetic.  Neighborhood inputs are computed
separately from non-self cells: log1p of each gene's local raw-count mean and
its local positive fraction.  Their alignment with the output gene axis is
preserved by an elementwise linear residual, with no neighborhood encoder.

The output is a normalized Bernoulli hurdle / zero-truncated NB2 distribution.
The model is independent of the BAGM graph architecture; only its established
count-distribution transforms are reused.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .hurdle_negative_binomial import (
    HURDLE_DETECTION_PROBABILITY_EPSILON,
    zero_truncated_nb2_conditional_mean,
)
from .negative_binomial import (
    NB2_INVERSE_DISPERSION_EPSILON,
    NB2_MEAN_EPSILON,
    nb2_inverse_dispersion_from_raw,
    nb2_mean_from_logits,
)


@dataclass(frozen=True)
class LocalGeneResidualOutput:
    """Parameters and distinct point predictions for the hurdle distribution."""

    nonzero_logits: Tensor
    nb_mean: Tensor
    theta: Tensor
    positive_conditional_mean: Tensor
    marginal_mean: Tensor

    @property
    def nonzero_probability(self) -> Tensor:
        return torch.sigmoid(self.nonzero_logits)

    @property
    def predicted_zero_probability(self) -> Tensor:
        return 1.0 - self.nonzero_probability


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _initial_gene_vector(
    value: Tensor | None,
    *,
    default: float,
    num_genes: int,
    name: str,
) -> Tensor:
    result = (
        torch.full((num_genes,), default, dtype=torch.float32)
        if value is None
        else torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
    )
    if result.shape != (num_genes,):
        raise ValueError(f"{name} must have shape [genes]")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


class LocalGeneResidualCountModel(nn.Module):
    """Two hidden layers for target coexpression plus four weights per local gene.

    ``initial_nonzero_probability`` and ``initial_positive_conditional_mean``
    must be fit on training cells only.  Both output projections and the local
    residual start at zero weights, so initial predictions reproduce those
    gene-wise priors (up to the documented numerical floors).  Initial NB2
    inverse dispersion is one: its positive conditional mean is ``1 + mu``.

    With ``use_local=False``, local inputs have no path to the output and local
    parameters are frozen.  The target MLP is identical for the ablation.
    There is no expression-derived library-size input or per-cell identifier.
    """

    observation_model = "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"

    def __init__(
        self,
        num_genes: int,
        *,
        covariate_dim: int = 22,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_local: bool = True,
        initial_nonzero_probability: Tensor | None = None,
        initial_positive_conditional_mean: Tensor | None = None,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("num_genes", num_genes, 1),
            ("covariate_dim", covariate_dim, 0),
            ("hidden_dim", hidden_dim, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite in [0, 1)")
        if not isinstance(use_local, bool):
            raise TypeError("use_local must be a bool")

        self.num_genes = num_genes
        self.covariate_dim = covariate_dim
        self.hidden_dim = hidden_dim
        self.use_local = use_local
        probability = _initial_gene_vector(
            initial_nonzero_probability,
            default=0.1,
            num_genes=num_genes,
            name="initial_nonzero_probability",
        )
        if bool(((probability < 0.0) | (probability > 1.0)).any()):
            raise ValueError("initial_nonzero_probability must be in [0, 1]")
        probability.clamp_(
            HURDLE_DETECTION_PROBABILITY_EPSILON,
            1.0 - HURDLE_DETECTION_PROBABILITY_EPSILON,
        )
        positive_mean = _initial_gene_vector(
            initial_positive_conditional_mean,
            default=2.0,
            num_genes=num_genes,
            name="initial_positive_conditional_mean",
        )
        if bool((positive_mean < 1.0).any()):
            raise ValueError("initial_positive_conditional_mean must be >= 1")
        self.register_buffer("initial_nonzero_probability", probability)
        self.register_buffer("initial_positive_conditional_mean", positive_mean)

        self.trunk = nn.Sequential(
            nn.Linear(2 * num_genes + covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.detection_head = nn.Linear(hidden_dim, num_genes)
        self.count_head = nn.Linear(hidden_dim, num_genes)
        self.local_detection_weights = nn.Parameter(
            torch.zeros(num_genes, 2), requires_grad=use_local
        )
        self.local_count_weights = nn.Parameter(
            torch.zeros(num_genes, 2), requires_grad=use_local
        )
        self.raw_theta = nn.Parameter(
            _inverse_softplus(
                torch.full((num_genes,), 1.0 - NB2_INVERSE_DISPERSION_EPSILON)
            )
        )
        nn.init.zeros_(self.detection_head.weight)
        nn.init.zeros_(self.count_head.weight)
        with torch.no_grad():
            self.detection_head.bias.copy_(torch.logit(probability))
            initial_mu = (positive_mean - 1.0).clamp_min(2.0 * NB2_MEAN_EPSILON)
            self.count_head.bias.copy_(
                _inverse_softplus(initial_mu - NB2_MEAN_EPSILON)
            )

    @staticmethod
    def _validate_matrix(
        value: Tensor,
        *,
        shape: tuple[int, int],
        reference: Tensor,
        name: str,
    ) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.device != reference.device:
            raise ValueError(f"{name} must share the expression device")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be real floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        node_covariates: Tensor | None = None,
        local_log_mean: Tensor | None = None,
        local_positive_fraction: Tensor | None = None,
    ) -> LocalGeneResidualOutput:
        """Reconstruct a target batch with ``True`` marking hidden entries.

        Every local summary must exclude the corresponding target cell.  This
        module cannot infer that provenance from an already pooled matrix; the
        neighborhood builder owns and tests explicit self exclusion.  A target
        without neighbors has both local summaries set to zero.
        """

        if not isinstance(input_expression, Tensor):
            raise TypeError("input_expression must be a tensor")
        if (
            input_expression.ndim != 2
            or input_expression.shape[0] == 0
            or input_expression.shape[1] != self.num_genes
        ):
            raise ValueError("input_expression must have nonempty shape [cells, genes]")
        if not input_expression.is_floating_point() or input_expression.is_complex():
            raise TypeError("input_expression must be real floating point")
        if not isinstance(gene_mask, Tensor) or gene_mask.dtype != torch.bool:
            raise TypeError("gene_mask must be a boolean tensor")
        if gene_mask.shape != input_expression.shape or gene_mask.device != input_expression.device:
            raise ValueError("gene_mask must be shape- and device-aligned with expression")

        # Multiplication would propagate hidden NaN/Inf values.  Mask them before
        # validation or any model computation so neither values nor gradients leak.
        visible_expression = input_expression.masked_fill(gene_mask, 0.0)
        shape = (int(input_expression.shape[0]), self.num_genes)
        self._validate_matrix(
            visible_expression,
            shape=shape,
            reference=input_expression,
            name="visible input_expression",
        )
        inputs = [visible_expression, gene_mask.to(dtype=input_expression.dtype)]
        if node_covariates is None:
            if self.covariate_dim:
                raise ValueError("node_covariates are required for covariate_dim > 0")
        else:
            self._validate_matrix(
                node_covariates,
                shape=(shape[0], self.covariate_dim),
                reference=input_expression,
                name="node_covariates",
            )
            inputs.append(node_covariates.to(dtype=input_expression.dtype))

        hidden = self.trunk(torch.cat(inputs, dim=-1))
        # The distribution uses FP32 even under mixed-precision trunk execution.
        nonzero_logits = self.detection_head(hidden).float()
        count_logits = self.count_head(hidden).float()
        if self.use_local:
            if local_log_mean is None or local_positive_fraction is None:
                raise ValueError("both local summaries are required when use_local=True")
            self._validate_matrix(
                local_log_mean,
                shape=shape,
                reference=input_expression,
                name="local_log_mean",
            )
            self._validate_matrix(
                local_positive_fraction,
                shape=shape,
                reference=input_expression,
                name="local_positive_fraction",
            )
            if bool((local_log_mean < 0.0).any()):
                raise ValueError("local_log_mean must be nonnegative")
            if bool(((local_positive_fraction < 0.0) | (local_positive_fraction > 1.0)).any()):
                raise ValueError("local_positive_fraction must be in [0, 1]")
            log_mean = local_log_mean.float()
            fraction = local_positive_fraction.float()
            nonzero_logits = (
                nonzero_logits
                + self.local_detection_weights[:, 0] * log_mean
                + self.local_detection_weights[:, 1] * fraction
            )
            count_logits = (
                count_logits
                + self.local_count_weights[:, 0] * log_mean
                + self.local_count_weights[:, 1] * fraction
            )

        nb_mean = nb2_mean_from_logits(count_logits)
        theta = nb2_inverse_dispersion_from_raw(self.raw_theta)
        positive_mean = zero_truncated_nb2_conditional_mean(nb_mean, theta)
        marginal_mean = torch.sigmoid(nonzero_logits) * positive_mean
        return LocalGeneResidualOutput(
            nonzero_logits=nonzero_logits,
            nb_mean=nb_mean,
            theta=theta,
            positive_conditional_mean=positive_mean,
            marginal_mean=marginal_mean,
        )

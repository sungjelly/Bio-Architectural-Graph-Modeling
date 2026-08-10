"""Nonlinear additive neighbor model for same-gene sensitivity replication.

The model deliberately has no receiver-expression input.  Its neighbor branch
    is a full linear map plus a one-hidden-layer exact-GELU residual, which makes
    the population-mean Jacobian
available in closed form after averaging the hidden activation derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SameGeneNonlinearError(RuntimeError):
    """Raised when the nonlinear replication contract is violated."""


def exact_gelu_derivative(value: torch.Tensor) -> torch.Tensor:
    """Return the derivative of ``torch.nn.functional.gelu(..., approximate='none')``."""

    if not value.is_floating_point():
        raise SameGeneNonlinearError("GELU derivative requires a floating tensor")
    root_two = math.sqrt(2.0)
    root_two_pi = math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(value / root_two)) + (
        value * torch.exp(-0.5 * value.square()) / root_two_pi
    )


class AdditiveNeighborMLP(nn.Module):
    """Morphology predictor plus an optional nonlinear neighbor-expression branch."""

    def __init__(
        self,
        *,
        gene_count: int = 1000,
        morphology_count: int = 22,
        hidden_count: int = 64,
        use_neighbor: bool = True,
    ) -> None:
        super().__init__()
        if gene_count < 1 or morphology_count < 1 or hidden_count < 1:
            raise ValueError("model dimensions must be positive")
        self.gene_count = int(gene_count)
        self.morphology_count = int(morphology_count)
        self.hidden_count = int(hidden_count)
        self.use_neighbor = bool(use_neighbor)
        self.morphology = nn.Linear(self.morphology_count, self.gene_count)
        if self.use_neighbor:
            self.neighbor_linear = nn.Linear(
                self.gene_count, self.gene_count, bias=False
            )
            self.neighbor_in = nn.Linear(self.gene_count, self.hidden_count)
            self.neighbor_out = nn.Linear(
                self.hidden_count, self.gene_count, bias=False
            )
        else:
            self.neighbor_linear = None
            self.neighbor_in = None
            self.neighbor_out = None

    def forward(
        self,
        morphology: torch.Tensor,
        neighbor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if morphology.ndim != 2 or morphology.shape[1] != self.morphology_count:
            raise SameGeneNonlinearError("morphology has the wrong shape")
        prediction = self.morphology(morphology)
        if self.use_neighbor:
            if neighbor is None:
                raise SameGeneNonlinearError("neighbor input is required")
            if (
                neighbor.ndim != 2
                or neighbor.shape[0] != morphology.shape[0]
                or neighbor.shape[1] != self.gene_count
            ):
                raise SameGeneNonlinearError("neighbor has the wrong shape")
            assert (
                self.neighbor_linear is not None
                and self.neighbor_in is not None
                and self.neighbor_out is not None
            )
            prediction = prediction + self.neighbor_linear(neighbor) + self.neighbor_out(
                F.gelu(self.neighbor_in(neighbor), approximate="none")
            )
        elif neighbor is not None:
            raise SameGeneNonlinearError(
                "morphology-only model refuses an unused neighbor tensor"
            )
        if not bool(torch.isfinite(prediction).all().item()):
            raise SameGeneNonlinearError("model prediction is nonfinite")
        return prediction

    @torch.no_grad()
    def mean_neighbor_jacobian(
        self,
        neighbor: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        chunk_size: int = 8192,
    ) -> np.ndarray:
        """Exact mean ``d output_gene / d neighbor_gene`` over supplied rows."""

        return self.mean_neighbor_jacobian_parts(
            neighbor,
            weights=weights,
            chunk_size=chunk_size,
        )["total"]

    @torch.no_grad()
    def mean_neighbor_jacobian_parts(
        self,
        neighbor: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        chunk_size: int = 8192,
    ) -> dict[str, np.ndarray]:
        """Return total, linear, and nonlinear mean Jacobian components."""

        if (
            not self.use_neighbor
            or self.neighbor_linear is None
            or self.neighbor_in is None
            or self.neighbor_out is None
        ):
            raise SameGeneNonlinearError("model has no neighbor Jacobian")
        if (
            neighbor.ndim != 2
            or neighbor.shape[1] != self.gene_count
            or neighbor.shape[0] < 1
        ):
            raise SameGeneNonlinearError("neighbor Jacobian population is invalid")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if weights is None:
            normalized_weights = torch.full(
                (neighbor.shape[0],),
                1.0 / neighbor.shape[0],
                dtype=torch.float64,
                device=neighbor.device,
            )
        else:
            if weights.shape != (neighbor.shape[0],):
                raise SameGeneNonlinearError("Jacobian weights have the wrong shape")
            normalized_weights = weights.to(
                device=neighbor.device, dtype=torch.float64
            )
            if (
                not bool(torch.isfinite(normalized_weights).all().item())
                or bool((normalized_weights < 0).any().item())
                or float(normalized_weights.sum().item()) <= 0
            ):
                raise SameGeneNonlinearError("Jacobian weights are invalid")
            normalized_weights = normalized_weights / normalized_weights.sum()
        derivative_sum = torch.zeros(
            self.hidden_count,
            dtype=torch.float64,
            device=neighbor.device,
        )
        for start in range(0, neighbor.shape[0], chunk_size):
            batch = neighbor[start : start + chunk_size]
            hidden = F.linear(
                batch,
                self.neighbor_in.weight,
                self.neighbor_in.bias,
            )
            batch_weights = normalized_weights[start : start + len(batch)]
            derivative_sum += (
                exact_gelu_derivative(hidden).double()
                * batch_weights.unsqueeze(1)
            ).sum(dim=0)
        mean_derivative = derivative_sum
        output_weight = self.neighbor_out.weight.detach().cpu().double()
        input_weight = self.neighbor_in.weight.detach().cpu().double()
        linear_weight = self.neighbor_linear.weight.detach().cpu().double()
        linear = linear_weight.numpy()
        nonlinear = (
            (output_weight * mean_derivative.cpu().unsqueeze(0)) @ input_weight
        ).numpy()
        result = linear + nonlinear
        if result.shape != (self.gene_count, self.gene_count):
            raise SameGeneNonlinearError("mean Jacobian has the wrong shape")
        if not np.isfinite(result).all():
            raise SameGeneNonlinearError("mean Jacobian is nonfinite")
        mean_derivative_result = mean_derivative.cpu().numpy()
        for value in (result, linear, nonlinear, mean_derivative_result):
            value.setflags(write=False)
        return {
            "total": result,
            "linear": linear,
            "nonlinear": nonlinear,
            "mean_hidden_derivative": mean_derivative_result,
        }


@dataclass(frozen=True)
class NonlinearControlResult:
    maximum_autograd_error: float
    maximum_finite_difference_error: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_autograd_error": self.maximum_autograd_error,
            "maximum_finite_difference_error": self.maximum_finite_difference_error,
            "passed": self.passed,
        }


def planted_nonlinear_jacobian_control() -> dict[str, Any]:
    """Check the closed-form population Jacobian against autograd and differences."""

    torch.manual_seed(19)
    model = AdditiveNeighborMLP(
        gene_count=4,
        morphology_count=3,
        hidden_count=5,
        use_neighbor=True,
    ).double()
    model.eval()
    morphology = torch.randn(7, 3, dtype=torch.float64)
    neighbor = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    analytical = model.mean_neighbor_jacobian(neighbor.detach(), chunk_size=3)
    explicit = np.zeros((4, 4), dtype=np.float64)
    prediction = model(morphology, neighbor)
    for target in range(4):
        gradient = torch.autograd.grad(
            prediction[:, target].mean(),
            neighbor,
            retain_graph=True,
        )[0]
        explicit[target] = gradient.sum(dim=0).detach().numpy()
    autograd_error = float(np.max(np.abs(analytical - explicit)))

    epsilon = 1e-5
    finite = np.zeros((4, 4), dtype=np.float64)
    with torch.no_grad():
        for source in range(4):
            shift = torch.zeros_like(neighbor)
            shift[:, source] = epsilon
            plus = model(morphology, neighbor.detach() + shift).mean(dim=0)
            minus = model(morphology, neighbor.detach() - shift).mean(dim=0)
            finite[:, source] = ((plus - minus) / (2 * epsilon)).numpy()
    finite_error = float(np.max(np.abs(analytical - finite)))
    return NonlinearControlResult(
        maximum_autograd_error=autograd_error,
        maximum_finite_difference_error=finite_error,
        passed=autograd_error <= 1e-10 and finite_error <= 1e-8,
    ).as_dict()


__all__ = [
    "AdditiveNeighborMLP",
    "NonlinearControlResult",
    "SameGeneNonlinearError",
    "exact_gelu_derivative",
    "planted_nonlinear_jacobian_control",
]

"""Frozen gradient estimands for the MyJJu GeneMAE audit.

The functions in this module operate on an already loaded, deterministic
GeneMAE-like forward.  They do not select checkpoints, mutate model state, or
interpret a derivative as a biological or causal effect.

There are deliberately two gradient APIs:

* :func:`summed_output_input_gradients` computes a single VJP for a normalized
  sum of receiver outputs.  It reports only a global directional derivative.
* :func:`decomposed_masked_input_gradients` computes one VJP per receiver.
  Receiver identity is therefore retained, making the same-cell/other-cell
  split and the pre-cancellation L1 mass exact.

A VJP of a summed output cannot recover the exact decomposition: each input
row can affect several receivers, and receiver-specific derivatives can cancel
before an absolute value is taken.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias

import numpy as np
import torch
from torch import Tensor


FROZEN_MARKER_MODULES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "epithelial": (
            "EPCAM",
            "KRT8",
            "KRT18",
            "KRT19",
            "CDH1",
            "KRT7",
            "KRT17",
            "KRT5",
        ),
        "proliferation": ("MKI67", "PCNA", "TOP2A", "BIRC5"),
        "immune_T": (
            "PTPRC",
            "CD3D",
            "CD3E",
            "CD8A",
            "CD4",
            "FOXP3",
            "NKG7",
        ),
        "immune_myeloid_B": ("CD68", "CD163", "MS4A1", "CD79A"),
        "stroma": (
            "COL1A1",
            "COL1A2",
            "COL3A1",
            "ACTA2",
            "DCN",
            "LUM",
            "PDGFRB",
        ),
        "endothelial": ("PECAM1", "VWF"),
        "cancer_candidate": (
            "PSCA",
            "OLFM4",
            "CEACAM6",
            "CLDN4",
            "LGR5",
            "SOX9",
            "MYC",
        ),
    }
)
FROZEN_MARKER_GENES = tuple(
    gene
    for module_genes in FROZEN_MARKER_MODULES.values()
    for gene in module_genes
)
FROZEN_SOURCE_SELECTED_PAIRS: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            "KRT8": ("KRT19", "KRT18", "KRT7"),
            "COL1A1": ("COL3A1", "COL1A2", "DCN"),
            "EPCAM": ("PSCA", "OLFM4", "KRT19", "CLDN4", "CDH1"),
            "OLFM4": ("PSCA", "EPCAM", "KRT19", "CDH1"),
            "CEACAM6": ("KRT19", "KRT8", "CLDN4", "KRT18"),
        }
    )
)
FROZEN_TARGET_GENES = tuple(FROZEN_SOURCE_SELECTED_PAIRS)
FROZEN_LOCKED_DIRECTED_PAIRS = tuple(
    (target, source)
    for target, sources in FROZEN_SOURCE_SELECTED_PAIRS.items()
    for source in sources
)
FROZEN_PERTURBATION_SCALES = (0.10, 0.25)
FROZEN_CLIP_QUANTILES = (0.01, 0.99)
FROZEN_MAXIMUM_HOPS = 4
FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE = 8
FROZEN_RECEIVER_SAMPLING_NAMESPACE = (
    "myjju-gradient-audit-receiver-v1"
)
FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT = 1e-6


class GradientAuditError(RuntimeError):
    """Raised when a frozen gradient estimand is invalid or unverifiable."""


class GeneMAELikeForward(Protocol):
    """Callable shape used by the real model and analytical test doubles."""

    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor | None = None,
        entry_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(reconstruction, effective_entry_mask)``."""


@dataclass(frozen=True)
class GlobalGradientResult:
    """One normalized summed-output VJP, without same/cross labels.

    ``input_gradient[t, u, s]`` is the derivative of

    ``sum(reconstruction[receivers[t], target[t]]) / divisor[t]``

    with respect to input node ``u`` and selected source gene ``s``.
    ``global_vjp_l1_mass`` is the L1 norm after receiver outputs have been
    summed.  It is not receiver-resolved L1 mass.
    """

    target_gene_indices: tuple[int, ...]
    source_gene_indices: tuple[int, ...]
    receiver_indices: tuple[tuple[int, ...], ...]
    normalization_divisors: np.ndarray
    output_values: np.ndarray
    input_gradient: np.ndarray
    signed_total: np.ndarray
    global_vjp_l1_mass: np.ndarray
    masked_source_max_abs: float


@dataclass(frozen=True)
class ReceiverGradientDecomposition:
    """Exact receiver-resolved masked-target gradient decomposition."""

    target_gene_indices: tuple[int, ...]
    source_gene_indices: tuple[int, ...]
    receiver_indices: tuple[tuple[int, ...], ...]
    receiver_counts: np.ndarray
    normalization_divisors: np.ndarray
    output_values: np.ndarray
    input_gradient: np.ndarray
    signed_total: np.ndarray
    signed_same_cell: np.ndarray
    signed_other_cell: np.ndarray
    l1_total: np.ndarray
    l1_same_cell: np.ndarray
    l1_other_cell: np.ndarray
    global_vjp_l1_mass: np.ndarray
    masked_source_max_abs: float
    outside_receptive_field_max_abs: float


@dataclass(frozen=True)
class BoundedSourcePerturbation:
    """One frozen within-core source-gene perturbation pair."""

    source_gene_index: int
    scale_in_sd: float
    population_sd: float
    lower_quantile: float
    upper_quantile: float
    lower_bound: float
    upper_bound: float
    visible_count: int
    baseline_source_values: Tensor
    plus_input: Tensor
    minus_input: Tensor
    plus_direction: Tensor
    minus_direction: Tensor


@dataclass(frozen=True)
class CentralPerturbationComparison:
    """Gradient predictions and centered frozen-model changes."""

    target_gene_indices: tuple[int, ...]
    source_gene_index: int
    scale_in_sd: float
    baseline_output: np.ndarray
    plus_output: np.ndarray
    minus_output: np.ndarray
    predicted_plus_change: np.ndarray
    predicted_minus_change: np.ndarray
    actual_plus_change: np.ndarray
    actual_minus_change: np.ndarray
    predicted_centered_change: np.ndarray
    actual_centered_change: np.ndarray
    curvature_residual: np.ndarray


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """Fail-closed deterministic faithfulness statistics."""

    spearman: float
    sign_agreement: float
    sign_comparison_count: int
    median_absolute_error: float
    median_absolute_actual_change: float
    median_absolute_error_ratio: float


GradientResult: TypeAlias = (
    GlobalGradientResult | ReceiverGradientDecomposition
)


def _readonly(array: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_x(x: Tensor) -> tuple[int, int]:
    if not isinstance(x, Tensor):
        raise GradientAuditError("x must be a torch.Tensor")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise GradientAuditError("x must have shape [nodes, genes]")
    if not x.is_floating_point():
        raise GradientAuditError("x must have a floating dtype")
    if not bool(torch.isfinite(x).all().item()):
        raise GradientAuditError("x contains nonfinite values")
    return int(x.shape[0]), int(x.shape[1])


def _validate_mask(entry_mask: Tensor, x: Tensor) -> None:
    if not isinstance(entry_mask, Tensor):
        raise GradientAuditError("entry_mask must be a torch.Tensor")
    if entry_mask.dtype != torch.bool:
        raise GradientAuditError("entry_mask must have dtype torch.bool")
    if entry_mask.shape != x.shape:
        raise GradientAuditError("entry_mask must have the same shape as x")
    if entry_mask.device != x.device:
        raise GradientAuditError("entry_mask and x must be on the same device")


def _validate_edge_inputs(
    edge_index: Tensor,
    edge_attr: Tensor | None,
    *,
    num_nodes: int,
    device: torch.device,
) -> None:
    if not isinstance(edge_index, Tensor):
        raise GradientAuditError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GradientAuditError("edge_index must have shape [2, edges]")
    if edge_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise GradientAuditError("edge_index must have an integer dtype")
    if edge_index.device != device:
        raise GradientAuditError("edge_index and x must be on the same device")
    if edge_index.numel():
        low = int(edge_index.min().item())
        high = int(edge_index.max().item())
        if low < 0 or high >= num_nodes:
            raise GradientAuditError("edge_index contains an invalid node index")
    if edge_attr is not None:
        if not isinstance(edge_attr, Tensor):
            raise GradientAuditError("edge_attr must be a torch.Tensor")
        if edge_attr.ndim < 1 or edge_attr.shape[0] != edge_index.shape[1]:
            raise GradientAuditError(
                "edge_attr first dimension must equal the edge count"
            )
        if edge_attr.device != device:
            raise GradientAuditError(
                "edge_attr and x must be on the same device"
            )
        if not bool(torch.isfinite(edge_attr).all().item()):
            raise GradientAuditError("edge_attr contains nonfinite values")


def _validated_indices(
    values: Sequence[int],
    *,
    upper: int,
    label: str,
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise GradientAuditError(f"{label} must contain integer indices")
        index = int(value)
        if not 0 <= index < upper:
            raise GradientAuditError(f"{label} contains an out-of-range index")
        result.append(index)
    if not result:
        raise GradientAuditError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise GradientAuditError(f"{label} must contain unique indices")
    return tuple(result)


def _validated_receivers(
    receiver_indices: Sequence[Sequence[int]],
    *,
    target_count: int,
    num_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    if len(receiver_indices) != target_count:
        raise GradientAuditError(
            "receiver_indices must align one-for-one with targets"
        )
    output: list[tuple[int, ...]] = []
    for target_position, values in enumerate(receiver_indices):
        receivers = _validated_indices(
            values,
            upper=num_nodes,
            label=f"receiver_indices[{target_position}]",
        )
        if tuple(sorted(receivers)) != receivers:
            raise GradientAuditError(
                "receiver lists must be deterministic strictly increasing "
                "sequences"
            )
        output.append(receivers)
    return tuple(output)


def _validated_divisors(
    divisors: Sequence[float] | None,
    receivers: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    if divisors is None:
        return tuple(float(len(indices)) for indices in receivers)
    if len(divisors) != len(receivers):
        raise GradientAuditError(
            "normalization_divisors must align one-for-one with targets"
        )
    output: list[float] = []
    for value in divisors:
        if isinstance(value, bool):
            raise GradientAuditError(
                "normalization divisors must be positive finite numbers"
            )
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise GradientAuditError(
                "normalization divisors must be positive finite numbers"
            )
        output.append(number)
    return tuple(output)


def _prepare_forward(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    edge_attr: Tensor | None,
    entry_mask: Tensor,
    source_gene_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    if isinstance(model, torch.nn.Module) and model.training:
        raise GradientAuditError(
            "model must already be in evaluation mode; this module does not "
            "mutate model state"
        )
    source_index = torch.as_tensor(
        source_gene_indices, dtype=torch.long, device=x.device
    )
    source_values = (
        x.detach().index_select(1, source_index).clone().requires_grad_(True)
    )
    model_input = x.detach().clone().index_copy(
        1, source_index, source_values
    )
    try:
        result = model(
            model_input,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
        )
    except TypeError as exc:
        raise GradientAuditError(
            "model must accept x, edge_index, edge_attr, and entry_mask"
        ) from exc
    if not isinstance(result, tuple) or len(result) != 2:
        raise GradientAuditError(
            "model forward must return (reconstruction, effective_mask)"
        )
    reconstruction, effective_mask = result
    if not isinstance(reconstruction, Tensor) or reconstruction.shape != x.shape:
        raise GradientAuditError(
            "reconstruction must be a tensor with the same shape as x"
        )
    if reconstruction.device != x.device:
        raise GradientAuditError(
            "reconstruction and x must be on the same device"
        )
    if not reconstruction.is_floating_point():
        raise GradientAuditError("reconstruction must have a floating dtype")
    if not bool(torch.isfinite(reconstruction).all().item()):
        raise GradientAuditError("reconstruction contains nonfinite values")
    if (
        not isinstance(effective_mask, Tensor)
        or effective_mask.dtype != torch.bool
        or effective_mask.shape != entry_mask.shape
        or effective_mask.device != entry_mask.device
        or not torch.equal(effective_mask, entry_mask)
    ):
        raise GradientAuditError(
            "model did not return the exact supplied entry_mask"
        )
    return reconstruction, source_values


def _input_gradient(
    output: Tensor,
    source_values: Tensor,
) -> Tensor:
    if output.numel() != 1:
        raise GradientAuditError("autograd output must be scalar")
    if not output.requires_grad:
        return torch.zeros_like(source_values)
    gradient = torch.autograd.grad(
        output,
        source_values,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(source_values)
    if not bool(torch.isfinite(gradient).all().item()):
        raise GradientAuditError("autograd produced a nonfinite gradient")
    return gradient


def _masked_gradient_max_abs(
    gradient: Tensor,
    selected_source_mask: Tensor,
) -> float:
    if not bool(selected_source_mask.any().item()):
        return 0.0
    return float(gradient[selected_source_mask].abs().max().item())


def _assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    label: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if not torch.allclose(
        actual,
        expected,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        maximum = float((actual - expected).abs().max().item())
        raise GradientAuditError(
            f"{label} failed; maximum absolute discrepancy is {maximum:.6g}"
        )


def summed_output_input_gradients(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    entry_mask: Tensor,
    target_gene_indices: Sequence[int],
    source_gene_indices: Sequence[int],
    receiver_indices: Sequence[Sequence[int]],
    normalization_divisors: Sequence[float] | None = None,
    edge_attr: Tensor | None = None,
    masked_gradient_tolerance: float = 1e-7,
) -> GlobalGradientResult:
    """Compute normalized summed-output VJPs.

    This routine intentionally does not expose same-cell or other-cell fields.
    Those quantities require receiver-resolved VJPs; use
    :func:`decomposed_masked_input_gradients`.
    """

    num_nodes, num_genes = _validate_x(x)
    _validate_mask(entry_mask, x)
    _validate_edge_inputs(
        edge_index, edge_attr, num_nodes=num_nodes, device=x.device
    )
    targets = _validated_indices(
        target_gene_indices, upper=num_genes, label="target_gene_indices"
    )
    sources = _validated_indices(
        source_gene_indices, upper=num_genes, label="source_gene_indices"
    )
    receivers = _validated_receivers(
        receiver_indices,
        target_count=len(targets),
        num_nodes=num_nodes,
    )
    divisors = _validated_divisors(normalization_divisors, receivers)
    if not math.isfinite(masked_gradient_tolerance) or (
        masked_gradient_tolerance < 0.0
    ):
        raise GradientAuditError(
            "masked_gradient_tolerance must be finite and nonnegative"
        )

    reconstruction, source_values = _prepare_forward(
        model,
        x,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=entry_mask,
        source_gene_indices=sources,
    )
    selected_mask = entry_mask.index_select(
        1,
        torch.as_tensor(sources, dtype=torch.long, device=x.device),
    )
    gradients: list[Tensor] = []
    output_values: list[float] = []
    masked_max = 0.0
    for target, target_receivers, divisor in zip(
        targets, receivers, divisors, strict=True
    ):
        receiver_tensor = torch.as_tensor(
            target_receivers, dtype=torch.long, device=x.device
        )
        output = reconstruction[receiver_tensor, target].sum() / divisor
        gradient = _input_gradient(output, source_values)
        masked_max = max(
            masked_max, _masked_gradient_max_abs(gradient, selected_mask)
        )
        gradients.append(gradient)
        output_values.append(float(output.detach().item()))
    if masked_max > masked_gradient_tolerance:
        raise GradientAuditError(
            "masked source input has a nonzero gradient: "
            f"{masked_max:.6g} > {masked_gradient_tolerance:.6g}"
        )

    stacked = torch.stack(gradients, dim=0)
    signed_total = stacked.sum(dim=1)
    global_l1 = stacked.abs().sum(dim=1)
    return GlobalGradientResult(
        target_gene_indices=targets,
        source_gene_indices=sources,
        receiver_indices=receivers,
        normalization_divisors=_readonly(
            np.asarray(divisors), dtype=np.float64
        ),
        output_values=_readonly(np.asarray(output_values), dtype=np.float64),
        input_gradient=_readonly(
            stacked.detach().cpu().numpy(), dtype=np.float64
        ),
        signed_total=_readonly(
            signed_total.detach().cpu().numpy(), dtype=np.float64
        ),
        global_vjp_l1_mass=_readonly(
            global_l1.detach().cpu().numpy(), dtype=np.float64
        ),
        masked_source_max_abs=masked_max,
    )


def source_style_all_unmasked_gradients(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    target_gene_indices: Sequence[int],
    source_gene_indices: Sequence[int],
    edge_attr: Tensor | None = None,
    masked_gradient_tolerance: float = 1e-7,
) -> GlobalGradientResult:
    """Reproduce the frozen source-style all-unmasked signed statistic.

    Each target output is summed over all output cells, differentiated with
    respect to the selected source gene in all input cells, and divided by the
    number of input cells.  Because the same cells are used on both sides,
    differentiating the output mean is algebraically identical.
    """

    num_nodes, _ = _validate_x(x)
    entry_mask = torch.zeros_like(x, dtype=torch.bool)
    targets = tuple(target_gene_indices)
    all_nodes = tuple(range(num_nodes))
    return summed_output_input_gradients(
        model,
        x,
        edge_index,
        entry_mask=entry_mask,
        target_gene_indices=targets,
        source_gene_indices=source_gene_indices,
        receiver_indices=tuple(all_nodes for _ in targets),
        normalization_divisors=tuple(float(num_nodes) for _ in targets),
        edge_attr=edge_attr,
        masked_gradient_tolerance=masked_gradient_tolerance,
    )


def _predecessor_adjacency(
    edge_index: Tensor,
    *,
    num_nodes: int,
    flow: str,
) -> tuple[tuple[int, ...], ...]:
    """Build the directed predecessor lists once for repeated support checks."""

    _validate_edge_inputs(
        edge_index, None, num_nodes=num_nodes, device=edge_index.device
    )
    edges = edge_index.detach().cpu().numpy()
    predecessor: list[list[int]] = [[] for _ in range(num_nodes)]
    sources, targets = (
        (edges[0], edges[1])
        if flow == "source_to_target"
        else (edges[1], edges[0])
    )
    for source, target in zip(sources.tolist(), targets.tolist(), strict=True):
        predecessor[int(target)].append(int(source))
    return tuple(tuple(values) for values in predecessor)


def _receptive_field_from_predecessors(
    predecessor: Sequence[Sequence[int]],
    receiver: int,
    *,
    maximum_hops: int,
) -> np.ndarray:
    """Traverse an already validated predecessor adjacency."""

    reached = {receiver}
    frontier = {receiver}
    for _ in range(maximum_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(predecessor[node])
        next_frontier.difference_update(reached)
        if not next_frontier:
            break
        reached.update(next_frontier)
        frontier = next_frontier
    return _readonly(np.asarray(sorted(reached), dtype=np.int64))


def receptive_field_nodes(
    edge_index: Tensor,
    receiver_index: int,
    *,
    num_nodes: int,
    maximum_hops: int = FROZEN_MAXIMUM_HOPS,
    flow: str = "source_to_target",
) -> np.ndarray:
    """Return nodes whose messages can reach a receiver within fixed hops."""

    if isinstance(receiver_index, bool) or not isinstance(
        receiver_index, (int, np.integer)
    ):
        raise GradientAuditError("receiver_index must be an integer")
    receiver = int(receiver_index)
    if num_nodes < 1 or not 0 <= receiver < num_nodes:
        raise GradientAuditError("receiver_index is out of range")
    if isinstance(maximum_hops, bool) or not isinstance(
        maximum_hops, (int, np.integer)
    ):
        raise GradientAuditError("maximum_hops must be a nonnegative integer")
    hops = int(maximum_hops)
    if hops < 0:
        raise GradientAuditError("maximum_hops must be a nonnegative integer")
    if flow not in {"source_to_target", "target_to_source"}:
        raise GradientAuditError(
            "flow must be 'source_to_target' or 'target_to_source'"
        )
    predecessor = _predecessor_adjacency(
        edge_index,
        num_nodes=num_nodes,
        flow=flow,
    )
    return _receptive_field_from_predecessors(
        predecessor,
        receiver,
        maximum_hops=hops,
    )


def decomposed_masked_input_gradients(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    entry_mask: Tensor,
    target_gene_indices: Sequence[int],
    source_gene_indices: Sequence[int],
    receiver_indices: Sequence[Sequence[int]],
    edge_attr: Tensor | None = None,
    maximum_hops: int | None = FROZEN_MAXIMUM_HOPS,
    masked_gradient_tolerance: float = 1e-7,
    outside_receptive_tolerance: float = 1e-7,
    decomposition_absolute_tolerance: float = 1e-7,
    decomposition_relative_tolerance: float = 1e-6,
) -> ReceiverGradientDecomposition:
    """Compute exact same-cell and other-cell gradients per masked receiver.

    One scalar output is differentiated per supplied receiver.  The receiver
    lists must be deterministic, strictly increasing, and contain only cells
    whose corresponding target entry is masked.
    """

    num_nodes, num_genes = _validate_x(x)
    _validate_mask(entry_mask, x)
    _validate_edge_inputs(
        edge_index, edge_attr, num_nodes=num_nodes, device=x.device
    )
    targets = _validated_indices(
        target_gene_indices, upper=num_genes, label="target_gene_indices"
    )
    sources = _validated_indices(
        source_gene_indices, upper=num_genes, label="source_gene_indices"
    )
    receivers = _validated_receivers(
        receiver_indices,
        target_count=len(targets),
        num_nodes=num_nodes,
    )
    tolerances = {
        "masked_gradient_tolerance": masked_gradient_tolerance,
        "outside_receptive_tolerance": outside_receptive_tolerance,
        "decomposition_absolute_tolerance": decomposition_absolute_tolerance,
        "decomposition_relative_tolerance": decomposition_relative_tolerance,
    }
    for label, value in tolerances.items():
        if not math.isfinite(value) or value < 0.0:
            raise GradientAuditError(
                f"{label} must be finite and nonnegative"
            )
    if maximum_hops is not None and (
        isinstance(maximum_hops, bool)
        or not isinstance(maximum_hops, (int, np.integer))
        or int(maximum_hops) < 0
    ):
        raise GradientAuditError(
            "maximum_hops must be None or a nonnegative integer"
        )

    for target, target_receivers in zip(targets, receivers, strict=True):
        receiver_tensor = torch.as_tensor(
            target_receivers, dtype=torch.long, device=x.device
        )
        if not bool(entry_mask[receiver_tensor, target].all().item()):
            raise GradientAuditError(
                "every decomposed receiver target entry must be masked"
            )

    reconstruction, source_values = _prepare_forward(
        model,
        x,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=entry_mask,
        source_gene_indices=sources,
    )
    source_index = torch.as_tensor(
        sources, dtype=torch.long, device=x.device
    )
    selected_mask = entry_mask.index_select(1, source_index)
    source_count = len(sources)
    mean_gradients: list[Tensor] = []
    output_values: list[float] = []
    signed_total_rows: list[Tensor] = []
    signed_same_rows: list[Tensor] = []
    signed_other_rows: list[Tensor] = []
    l1_total_rows: list[Tensor] = []
    l1_same_rows: list[Tensor] = []
    l1_other_rows: list[Tensor] = []
    global_l1_rows: list[Tensor] = []
    masked_max = 0.0
    outside_max = 0.0
    predecessor = (
        _predecessor_adjacency(
            edge_index,
            num_nodes=num_nodes,
            flow="source_to_target",
        )
        if maximum_hops is not None
        else None
    )

    for target, target_receivers in zip(targets, receivers, strict=True):
        count = len(target_receivers)
        mean_gradient = torch.zeros_like(source_values)
        signed_same = torch.zeros(
            source_count, dtype=x.dtype, device=x.device
        )
        signed_other = torch.zeros_like(signed_same)
        l1_same = torch.zeros_like(signed_same)
        l1_other = torch.zeros_like(signed_same)

        for receiver in target_receivers:
            gradient = _input_gradient(
                reconstruction[receiver, target], source_values
            )
            masked_max = max(
                masked_max,
                _masked_gradient_max_abs(gradient, selected_mask),
            )
            if maximum_hops is not None:
                assert predecessor is not None
                support = _receptive_field_from_predecessors(
                    predecessor,
                    receiver,
                    maximum_hops=int(maximum_hops),
                )
                outside = torch.ones(
                    num_nodes, dtype=torch.bool, device=x.device
                )
                outside[
                    torch.tensor(
                        support.tolist(), dtype=torch.long, device=x.device
                    )
                ] = False
                if bool(outside.any().item()):
                    outside_max = max(
                        outside_max,
                        float(gradient[outside].abs().max().item()),
                    )

            same = gradient[receiver]
            absolute = gradient.abs()
            same_absolute = absolute[receiver]
            mean_gradient += gradient
            signed_same += same
            signed_other += gradient.sum(dim=0) - same
            l1_same += same_absolute
            l1_other += absolute.sum(dim=0) - same_absolute

        mean_gradient /= float(count)
        signed_same /= float(count)
        signed_other /= float(count)
        l1_same /= float(count)
        l1_other /= float(count)
        signed_total = signed_same + signed_other
        l1_total = l1_same + l1_other

        receiver_tensor = torch.as_tensor(
            target_receivers, dtype=torch.long, device=x.device
        )
        normalized_output = (
            reconstruction[receiver_tensor, target].sum() / float(count)
        )
        summed_vjp = _input_gradient(normalized_output, source_values)
        _assert_close(
            summed_vjp,
            mean_gradient,
            label="receiver VJP accumulation",
            absolute_tolerance=decomposition_absolute_tolerance,
            relative_tolerance=decomposition_relative_tolerance,
        )
        _assert_close(
            signed_total,
            mean_gradient.sum(dim=0),
            label="signed total decomposition",
            absolute_tolerance=decomposition_absolute_tolerance,
            relative_tolerance=decomposition_relative_tolerance,
        )
        _assert_close(
            l1_total,
            l1_same + l1_other,
            label="L1 total decomposition",
            absolute_tolerance=decomposition_absolute_tolerance,
            relative_tolerance=decomposition_relative_tolerance,
        )

        mean_gradients.append(mean_gradient)
        output_values.append(float(normalized_output.detach().item()))
        signed_total_rows.append(signed_total)
        signed_same_rows.append(signed_same)
        signed_other_rows.append(signed_other)
        l1_total_rows.append(l1_total)
        l1_same_rows.append(l1_same)
        l1_other_rows.append(l1_other)
        global_l1_rows.append(summed_vjp.abs().sum(dim=0))

    if masked_max > masked_gradient_tolerance:
        raise GradientAuditError(
            "masked source input has a nonzero receiver-resolved gradient: "
            f"{masked_max:.6g} > {masked_gradient_tolerance:.6g}"
        )
    if outside_max > outside_receptive_tolerance:
        raise GradientAuditError(
            "input outside the configured receptive field has a nonzero "
            f"gradient: {outside_max:.6g} > "
            f"{outside_receptive_tolerance:.6g}"
        )

    def stack_numpy(rows: list[Tensor]) -> np.ndarray:
        return _readonly(
            torch.stack(rows, dim=0).detach().cpu().numpy(),
            dtype=np.float64,
        )

    counts = np.asarray([len(values) for values in receivers], dtype=np.int64)
    return ReceiverGradientDecomposition(
        target_gene_indices=targets,
        source_gene_indices=sources,
        receiver_indices=receivers,
        receiver_counts=_readonly(counts, dtype=np.int64),
        normalization_divisors=_readonly(counts, dtype=np.float64),
        output_values=_readonly(np.asarray(output_values), dtype=np.float64),
        input_gradient=stack_numpy(mean_gradients),
        signed_total=stack_numpy(signed_total_rows),
        signed_same_cell=stack_numpy(signed_same_rows),
        signed_other_cell=stack_numpy(signed_other_rows),
        l1_total=stack_numpy(l1_total_rows),
        l1_same_cell=stack_numpy(l1_same_rows),
        l1_other_cell=stack_numpy(l1_other_rows),
        global_vjp_l1_mass=stack_numpy(global_l1_rows),
        masked_source_max_abs=masked_max,
        outside_receptive_field_max_abs=outside_max,
    )


def source_absolute_symmetric_transform(directed: np.ndarray) -> np.ndarray:
    """Apply the historical source report's published transform."""

    array = np.asarray(directed, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise GradientAuditError("directed gradient matrix must be square")
    if not np.isfinite(array).all():
        raise GradientAuditError(
            "directed gradient matrix contains nonfinite values"
        )
    return _readonly(0.5 * (np.abs(array) + np.abs(array.T)))


def make_bounded_source_perturbation(
    x: Tensor,
    entry_mask: Tensor,
    *,
    source_gene_index: int,
    scale_in_sd: float,
    clip_quantiles: tuple[float, float] = FROZEN_CLIP_QUANTILES,
) -> BoundedSourcePerturbation:
    """Create plus/minus shifts for visible source entries in one full core.

    The population SD and linear-interpolation quantiles are computed from the
    complete source column before applying the entry mask.  The exact clipped
    directions are retained for the gradient dot product.
    """

    _, num_genes = _validate_x(x)
    _validate_mask(entry_mask, x)
    source = _validated_indices(
        (source_gene_index,),
        upper=num_genes,
        label="source_gene_index",
    )[0]
    scale = float(scale_in_sd)
    if scale not in FROZEN_PERTURBATION_SCALES:
        raise GradientAuditError(
            "scale_in_sd must be one of the frozen perturbation scales"
        )
    quantiles = tuple(float(value) for value in clip_quantiles)
    if quantiles != FROZEN_CLIP_QUANTILES:
        raise GradientAuditError(
            "clip_quantiles must equal the frozen (0.01, 0.99) bounds"
        )

    source_values = (
        x.detach()[:, source].to(dtype=torch.float64).cpu().numpy()
    )
    population_sd = float(np.std(source_values, ddof=0))
    if not math.isfinite(population_sd) or population_sd <= 0.0:
        raise GradientAuditError(
            "source gene must have a positive finite within-core SD"
        )
    lower_bound, upper_bound = (
        float(value)
        for value in np.quantile(
            source_values,
            FROZEN_CLIP_QUANTILES,
            method="linear",
        )
    )
    if not (
        math.isfinite(lower_bound)
        and math.isfinite(upper_bound)
        and lower_bound < upper_bound
    ):
        raise GradientAuditError(
            "source gene must have distinct finite clipping bounds"
        )
    visible = (~entry_mask[:, source]).detach().cpu().numpy()
    visible_count = int(visible.sum())
    if visible_count < 1:
        raise GradientAuditError(
            "source gene must have at least one visible input entry"
        )

    step = scale * population_sd
    plus_values = source_values.copy()
    minus_values = source_values.copy()
    # A literal clip(x +/- step, lower, upper) can reverse the nominal
    # direction for baseline outliers.  The locked protocol leaves baseline
    # values outside [lower, upper] unchanged in both directions.  For values
    # inside the interval it clips the shift, not the baseline.
    within_bounds = (source_values >= lower_bound) & (
        source_values <= upper_bound
    )
    plus_shift = np.minimum(
        step, np.maximum(upper_bound - source_values, 0.0)
    )
    minus_shift = -np.minimum(
        step, np.maximum(source_values - lower_bound, 0.0)
    )
    plus_shift[~within_bounds] = 0.0
    minus_shift[~within_bounds] = 0.0
    plus_values[visible] += plus_shift[visible]
    minus_values[visible] += minus_shift[visible]
    plus_source = torch.as_tensor(
        plus_values, dtype=x.dtype, device=x.device
    )
    minus_source = torch.as_tensor(
        minus_values, dtype=x.dtype, device=x.device
    )
    plus_input = x.detach().clone()
    minus_input = x.detach().clone()
    plus_input[:, source] = plus_source
    minus_input[:, source] = minus_source
    baseline_source = x.detach()[:, source].clone()
    plus_direction = plus_source - baseline_source
    minus_direction = minus_source - baseline_source

    if bool(
        (
            plus_direction[entry_mask[:, source]] != 0
        ).any().item()
    ) or bool(
        (
            minus_direction[entry_mask[:, source]] != 0
        ).any().item()
    ):
        raise GradientAuditError("perturbation changed a masked source entry")
    return BoundedSourcePerturbation(
        source_gene_index=source,
        scale_in_sd=scale,
        population_sd=population_sd,
        lower_quantile=FROZEN_CLIP_QUANTILES[0],
        upper_quantile=FROZEN_CLIP_QUANTILES[1],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        visible_count=visible_count,
        baseline_source_values=baseline_source,
        plus_input=plus_input,
        minus_input=minus_input,
        plus_direction=plus_direction,
        minus_direction=minus_direction,
    )


def _no_grad_outputs(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    edge_attr: Tensor | None,
    entry_mask: Tensor,
) -> Tensor:
    if isinstance(model, torch.nn.Module) and model.training:
        raise GradientAuditError("model must already be in evaluation mode")
    with torch.no_grad():
        try:
            result = model(
                x,
                edge_index,
                edge_attr=edge_attr,
                entry_mask=entry_mask,
            )
        except TypeError as exc:
            raise GradientAuditError(
                "model must accept x, edge_index, edge_attr, and entry_mask"
            ) from exc
    if not isinstance(result, tuple) or len(result) != 2:
        raise GradientAuditError(
            "model forward must return (reconstruction, effective_mask)"
        )
    reconstruction, effective_mask = result
    if (
        not isinstance(reconstruction, Tensor)
        or reconstruction.shape != x.shape
        or reconstruction.device != x.device
        or not reconstruction.is_floating_point()
        or not bool(torch.isfinite(reconstruction).all().item())
    ):
        raise GradientAuditError("model returned an invalid reconstruction")
    if (
        not isinstance(effective_mask, Tensor)
        or effective_mask.dtype != torch.bool
        or effective_mask.shape != entry_mask.shape
        or not torch.equal(effective_mask, entry_mask)
    ):
        raise GradientAuditError(
            "model did not return the exact supplied entry_mask"
        )
    return reconstruction


def compare_central_bounded_perturbation(
    model: GeneMAELikeForward | Callable[..., tuple[Tensor, Tensor]],
    x: Tensor,
    edge_index: Tensor,
    *,
    entry_mask: Tensor,
    gradient_result: GradientResult,
    perturbation: BoundedSourcePerturbation,
    edge_attr: Tensor | None = None,
    replay_absolute_tolerance: float = 1e-6,
    replay_relative_tolerance: float = 1e-6,
) -> CentralPerturbationComparison:
    """Compare a gradient dot direction with centered frozen-model changes."""

    num_nodes, _ = _validate_x(x)
    _validate_mask(entry_mask, x)
    _validate_edge_inputs(
        edge_index, edge_attr, num_nodes=num_nodes, device=x.device
    )
    if perturbation.plus_input.shape != x.shape or (
        perturbation.minus_input.shape != x.shape
    ):
        raise GradientAuditError("perturbation input shapes do not match x")
    if (
        perturbation.plus_input.device != x.device
        or perturbation.minus_input.device != x.device
        or perturbation.plus_direction.device != x.device
        or perturbation.minus_direction.device != x.device
    ):
        raise GradientAuditError("perturbation tensors must share x's device")
    if not torch.equal(
        x.detach()[:, perturbation.source_gene_index],
        perturbation.baseline_source_values,
    ):
        raise GradientAuditError(
            "perturbation was not constructed from the supplied baseline x"
        )
    try:
        source_position = gradient_result.source_gene_indices.index(
            perturbation.source_gene_index
        )
    except ValueError as exc:
        raise GradientAuditError(
            "perturbed source gene is absent from gradient_result"
        ) from exc
    expected_shape = (
        len(gradient_result.target_gene_indices),
        num_nodes,
        len(gradient_result.source_gene_indices),
    )
    if gradient_result.input_gradient.shape != expected_shape:
        raise GradientAuditError(
            "gradient_result.input_gradient has an invalid shape"
        )
    if len(gradient_result.normalization_divisors) != len(
        gradient_result.target_gene_indices
    ):
        raise GradientAuditError(
            "gradient_result normalization metadata is invalid"
        )

    baseline_reconstruction = _no_grad_outputs(
        model,
        x.detach(),
        edge_index,
        edge_attr=edge_attr,
        entry_mask=entry_mask,
    )
    plus_reconstruction = _no_grad_outputs(
        model,
        perturbation.plus_input,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=entry_mask,
    )
    minus_reconstruction = _no_grad_outputs(
        model,
        perturbation.minus_input,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=entry_mask,
    )

    baseline: list[float] = []
    plus: list[float] = []
    minus: list[float] = []
    for target, receivers, divisor in zip(
        gradient_result.target_gene_indices,
        gradient_result.receiver_indices,
        gradient_result.normalization_divisors.tolist(),
        strict=True,
    ):
        receiver_tensor = torch.as_tensor(
            receivers, dtype=torch.long, device=x.device
        )
        baseline.append(
            float(
                (
                    baseline_reconstruction[receiver_tensor, target].sum()
                    / divisor
                ).item()
            )
        )
        plus.append(
            float(
                (
                    plus_reconstruction[receiver_tensor, target].sum()
                    / divisor
                ).item()
            )
        )
        minus.append(
            float(
                (
                    minus_reconstruction[receiver_tensor, target].sum()
                    / divisor
                ).item()
            )
        )
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if not np.allclose(
        baseline_array,
        gradient_result.output_values,
        atol=replay_absolute_tolerance,
        rtol=replay_relative_tolerance,
    ):
        raise GradientAuditError(
            "baseline replay does not match the gradient forward"
        )
    plus_array = np.asarray(plus, dtype=np.float64)
    minus_array = np.asarray(minus, dtype=np.float64)
    gradient = gradient_result.input_gradient[:, :, source_position]
    plus_direction = (
        perturbation.plus_direction.detach().cpu().numpy().astype(np.float64)
    )
    minus_direction = (
        perturbation.minus_direction.detach().cpu().numpy().astype(np.float64)
    )
    predicted_plus = gradient @ plus_direction
    predicted_minus = gradient @ minus_direction
    actual_plus = plus_array - baseline_array
    actual_minus = minus_array - baseline_array
    predicted_centered = 0.5 * (predicted_plus - predicted_minus)
    actual_centered = 0.5 * (plus_array - minus_array)
    curvature_residual = 0.5 * (
        (actual_plus - predicted_plus)
        + (actual_minus - predicted_minus)
    )
    return CentralPerturbationComparison(
        target_gene_indices=gradient_result.target_gene_indices,
        source_gene_index=perturbation.source_gene_index,
        scale_in_sd=perturbation.scale_in_sd,
        baseline_output=_readonly(baseline_array),
        plus_output=_readonly(plus_array),
        minus_output=_readonly(minus_array),
        predicted_plus_change=_readonly(predicted_plus),
        predicted_minus_change=_readonly(predicted_minus),
        actual_plus_change=_readonly(actual_plus),
        actual_minus_change=_readonly(actual_minus),
        predicted_centered_change=_readonly(predicted_centered),
        actual_centered_change=_readonly(actual_centered),
        curvature_residual=_readonly(curvature_residual),
    )


def _paired_finite_vectors(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    *,
    minimum_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(actual, dtype=np.float64)
    if first.shape != second.shape:
        raise GradientAuditError(
            "predicted and actual arrays must have the same shape"
        )
    if first.ndim != 1 or first.size < minimum_size:
        raise GradientAuditError(
            f"metric inputs must be one-dimensional with at least "
            f"{minimum_size} observations"
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise GradientAuditError("metric inputs contain nonfinite values")
    return first, second


def deterministic_average_ranks(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return stable one-based average ranks, assigning equal values ties."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise GradientAuditError(
            "rank input must be a nonempty one-dimensional array"
        )
    if not np.isfinite(array).all():
        raise GradientAuditError("rank input contains nonfinite values")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    return _readonly(ranks)


def deterministic_spearman(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
) -> float:
    """Compute deterministic Spearman correlation with average tie ranks."""

    first, second = _paired_finite_vectors(
        predicted, actual, minimum_size=2
    )
    first_rank = deterministic_average_ranks(first)
    second_rank = deterministic_average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator == 0.0:
        raise GradientAuditError(
            "Spearman correlation is undefined for a constant rank vector"
        )
    return float(np.dot(first_centered, second_centered) / denominator)


def sign_agreement_fraction(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    *,
    minimum_absolute_actual_change: float = (
        FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT
    ),
) -> tuple[float, int]:
    """Return sign agreement on prespecified non-negligible actual changes."""

    first, second = _paired_finite_vectors(
        predicted, actual, minimum_size=1
    )
    threshold = float(minimum_absolute_actual_change)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise GradientAuditError(
            "minimum_absolute_actual_change must be finite and nonnegative"
        )
    selected = np.abs(second) > threshold
    count = int(selected.sum())
    if count == 0:
        raise GradientAuditError(
            "no actual changes exceed the prespecified sign threshold"
        )
    agreement = np.sign(first[selected]) == np.sign(second[selected])
    return float(np.mean(agreement)), count


def median_absolute_error_ratio(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Return median error, median absolute actual change, and their ratio."""

    first, second = _paired_finite_vectors(
        predicted, actual, minimum_size=1
    )
    median_error = float(np.median(np.abs(first - second)))
    median_actual = float(np.median(np.abs(second)))
    if median_actual == 0.0:
        raise GradientAuditError(
            "median absolute actual change is zero; ratio is undefined"
        )
    return median_error, median_actual, median_error / median_actual


def faithfulness_metrics(
    predicted: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    *,
    minimum_absolute_actual_change: float = (
        FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT
    ),
) -> FaithfulnessMetrics:
    """Compute all frozen faithfulness metrics without dropping bad values."""

    spearman = deterministic_spearman(predicted, actual)
    sign_agreement, count = sign_agreement_fraction(
        predicted,
        actual,
        minimum_absolute_actual_change=minimum_absolute_actual_change,
    )
    median_error, median_actual, ratio = median_absolute_error_ratio(
        predicted, actual
    )
    return FaithfulnessMetrics(
        spearman=spearman,
        sign_agreement=sign_agreement,
        sign_comparison_count=count,
        median_absolute_error=median_error,
        median_absolute_actual_change=median_actual,
        median_absolute_error_ratio=ratio,
    )

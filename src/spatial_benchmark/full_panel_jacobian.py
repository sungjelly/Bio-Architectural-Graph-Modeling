"""Memory-bounded full-panel Jacobians for frozen GeneMAE models.

The source-style statistic sums derivatives over every input cell.  Introducing
one broadcast shift variable per selected source gene computes that sum without
materialising a ``targets x cells x sources`` gradient tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


class FullPanelJacobianError(RuntimeError):
    """Raised when a full-panel Jacobian would violate its numerical contract."""


@dataclass(frozen=True, slots=True)
class UniformShiftJacobianResult:
    """Compact source-style Jacobian result for one graph tile."""

    target_gene_indices: tuple[int, ...]
    source_gene_indices: tuple[int, ...]
    output_means: np.ndarray
    signed_jacobian: np.ndarray
    node_count: int
    directed_edge_count: int
    target_batch_size: int


@dataclass(frozen=True, slots=True)
class GraphDegreeSummary:
    """Realized directed-degree audit for a symmetric graph."""

    node_count: int
    directed_edge_count: int
    minimum: int
    maximum: int
    mean: float
    median: float
    quantiles: tuple[float, float, float, float, float]


def _validated_gene_indices(
    values: Sequence[int] | None,
    *,
    gene_count: int,
    label: str,
) -> tuple[int, ...]:
    if values is None:
        return tuple(range(gene_count))
    result: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or int(raw) != raw:
            raise ValueError(f"{label} must contain only integers")
        value = int(raw)
        if value < 0 or value >= gene_count:
            raise ValueError(f"{label} contains an out-of-range index")
        result.append(value)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(result)


def _validate_inputs(
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None,
) -> tuple[int, int]:
    if not isinstance(x, Tensor) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor with shape [nodes, genes]")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("x must contain only finite values")
    node_count, gene_count = (int(x.shape[0]), int(x.shape[1]))
    if node_count < 1 or gene_count < 1:
        raise ValueError("x must contain at least one node and one gene")
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or tuple(edge_index.shape[:1]) != (2,)
        or edge_index.dtype != torch.long
        or edge_index.device != x.device
    ):
        raise ValueError(
            "edge_index must be a long tensor with shape [2, edges] on x.device"
        )
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0
        or int(edge_index.max().item()) >= node_count
    ):
        raise ValueError("edge_index contains an out-of-range node")
    if edge_attr is not None and (
        not isinstance(edge_attr, Tensor)
        or edge_attr.ndim != 2
        or int(edge_attr.shape[0]) != int(edge_index.shape[1])
        or edge_attr.device != x.device
        or not edge_attr.is_floating_point()
        or not bool(torch.isfinite(edge_attr).all().item())
    ):
        raise ValueError(
            "edge_attr must be a finite floating [edges, features] tensor on x.device"
        )
    return node_count, gene_count


def _model_forward(
    model: Any,
    shifted_x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None,
) -> Tensor:
    entry_mask = torch.zeros_like(shifted_x, dtype=torch.bool)
    try:
        reconstruction, effective_mask = model(
            shifted_x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
        )
    except TypeError as exc:
        raise FullPanelJacobianError(
            "model must accept x, edge_index, edge_attr, and entry_mask"
        ) from exc
    if (
        not isinstance(reconstruction, Tensor)
        or reconstruction.shape != shifted_x.shape
        or reconstruction.device != shifted_x.device
        or not reconstruction.is_floating_point()
    ):
        raise FullPanelJacobianError(
            "model reconstruction must be a floating tensor matching x"
        )
    if not bool(torch.isfinite(reconstruction).all().item()):
        raise FullPanelJacobianError("model reconstruction contains nonfinite values")
    if (
        not isinstance(effective_mask, Tensor)
        or effective_mask.dtype != torch.bool
        or effective_mask.shape != entry_mask.shape
        or effective_mask.device != entry_mask.device
        or not torch.equal(effective_mask, entry_mask)
    ):
        raise FullPanelJacobianError(
            "model did not return the exact supplied all-false entry mask"
        )
    return reconstruction


def uniform_shift_jacobian(
    model: Any,
    x: Tensor,
    edge_index: Tensor,
    *,
    edge_attr: Tensor | None = None,
    target_gene_indices: Sequence[int] | None = None,
    source_gene_indices: Sequence[int] | None = None,
    target_batch_size: int = 1,
) -> UniformShiftJacobianResult:
    """Compute the exact source-style Jacobian with one variable per source.

    Let ``delta[s]`` be added to source gene ``s`` in every input cell.  The
    derivative of each target's mean reconstruction with respect to ``delta``
    is exactly the sum of its derivatives over input cells divided by the
    number of output cells.  This is the established source-style statistic
    when the same tile supplies both input and output cells.

    ``target_batch_size`` controls vectorised reverse-mode rows.  A value of
    one minimizes memory and is the production-safe default.
    """

    node_count, gene_count = _validate_inputs(x, edge_index, edge_attr)
    targets = _validated_gene_indices(
        target_gene_indices,
        gene_count=gene_count,
        label="target_gene_indices",
    )
    sources = _validated_gene_indices(
        source_gene_indices,
        gene_count=gene_count,
        label="source_gene_indices",
    )
    if (
        isinstance(target_batch_size, bool)
        or int(target_batch_size) != target_batch_size
        or int(target_batch_size) < 1
    ):
        raise ValueError("target_batch_size must be a positive integer")
    batch_size = min(int(target_batch_size), len(targets))
    if getattr(model, "training", False):
        raise FullPanelJacobianError("model must be in evaluation mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise FullPanelJacobianError(
            "model parameter gradients must be disabled before Jacobian replay"
        )

    source_tensor = torch.as_tensor(sources, dtype=torch.long, device=x.device)
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=x.device)
    shift_values = torch.zeros(
        len(sources), dtype=x.dtype, device=x.device, requires_grad=True
    )
    if len(sources) == gene_count and sources == tuple(range(gene_count)):
        full_shift = shift_values
    else:
        full_shift = torch.zeros(gene_count, dtype=x.dtype, device=x.device)
        full_shift = full_shift.index_copy(0, source_tensor, shift_values)
    shifted_x = x + full_shift.unsqueeze(0)
    reconstruction = _model_forward(model, shifted_x, edge_index, edge_attr)
    outputs = reconstruction.index_select(1, target_tensor).mean(dim=0)
    if not bool(outputs.requires_grad):
        jacobian = torch.zeros(
            (len(targets), len(sources)), dtype=x.dtype, device=x.device
        )
    else:
        rows: list[Tensor] = []
        for start in range(0, len(targets), batch_size):
            stop = min(start + batch_size, len(targets))
            is_last = stop == len(targets)
            if stop - start == 1:
                gradient = torch.autograd.grad(
                    outputs[start],
                    shift_values,
                    retain_graph=not is_last,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if gradient is None:
                    gradient = torch.zeros_like(shift_values)
                rows.append(gradient.unsqueeze(0))
            else:
                basis = torch.zeros(
                    (stop - start, len(targets)),
                    dtype=outputs.dtype,
                    device=outputs.device,
                )
                positions = torch.arange(stop - start, device=outputs.device)
                basis[positions, positions + start] = 1
                gradient = torch.autograd.grad(
                    outputs,
                    shift_values,
                    grad_outputs=basis,
                    retain_graph=not is_last,
                    create_graph=False,
                    allow_unused=True,
                    is_grads_batched=True,
                )[0]
                if gradient is None:
                    gradient = torch.zeros(
                        (stop - start, len(sources)),
                        dtype=x.dtype,
                        device=x.device,
                    )
                rows.append(gradient)
        jacobian = torch.cat(rows, dim=0)
    if jacobian.shape != (len(targets), len(sources)):
        raise FullPanelJacobianError("autograd returned an unexpected Jacobian shape")
    if not bool(torch.isfinite(jacobian).all().item()):
        raise FullPanelJacobianError("autograd returned a nonfinite Jacobian")

    output_array = outputs.detach().cpu().numpy().astype(np.float64, copy=False)
    jacobian_array = (
        jacobian.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    output_array.setflags(write=False)
    jacobian_array.setflags(write=False)
    return UniformShiftJacobianResult(
        target_gene_indices=targets,
        source_gene_indices=sources,
        output_means=output_array,
        signed_jacobian=jacobian_array,
        node_count=node_count,
        directed_edge_count=int(edge_index.shape[1]),
        target_batch_size=batch_size,
    )


def summarize_symmetric_directed_degree(
    edge_index: np.ndarray | Tensor,
    *,
    node_count: int,
    required_minimum: int | None = None,
) -> GraphDegreeSummary:
    """Validate symmetry and summarize realized directed out-degree."""

    if isinstance(node_count, bool) or int(node_count) != node_count or node_count < 1:
        raise ValueError("node_count must be a positive integer")
    nodes = int(node_count)
    edges = (
        edge_index.detach().cpu().numpy()
        if isinstance(edge_index, Tensor)
        else np.asarray(edge_index)
    )
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or not np.issubdtype(edges.dtype, np.integer)
    ):
        raise ValueError("edge_index must be an integer array with shape [2, edges]")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= nodes):
        raise ValueError("edge_index contains an out-of-range node")
    if np.any(edges[0] == edges[1]):
        raise FullPanelJacobianError("constructed graph contains a self-edge")
    # Encode pairs as int64 so dense million-edge graphs can be audited without
    # constructing a prohibitively large Python set of tuple objects.
    codes = (
        edges[0].astype(np.int64, copy=False) * np.int64(nodes)
        + edges[1].astype(np.int64, copy=False)
    )
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    if sorted_codes.size > 1 and np.any(sorted_codes[1:] == sorted_codes[:-1]):
        raise FullPanelJacobianError("constructed graph contains duplicate edges")
    reverse_codes = (
        edges[1].astype(np.int64, copy=False) * np.int64(nodes)
        + edges[0].astype(np.int64, copy=False)
    )
    locations = np.searchsorted(sorted_codes, reverse_codes)
    if np.any(locations == sorted_codes.size):
        raise FullPanelJacobianError("constructed graph is not symmetric")
    if not np.array_equal(sorted_codes[locations], reverse_codes):
        raise FullPanelJacobianError("constructed graph is not symmetric")
    degree = np.bincount(edges[0], minlength=nodes).astype(np.int64, copy=False)
    minimum = int(degree.min())
    if required_minimum is not None:
        if (
            isinstance(required_minimum, bool)
            or int(required_minimum) != required_minimum
            or int(required_minimum) < 0
        ):
            raise ValueError("required_minimum must be a nonnegative integer")
        if minimum < int(required_minimum):
            raise FullPanelJacobianError(
                f"realized minimum degree {minimum} is below {int(required_minimum)}"
            )
    quantiles = np.quantile(degree, [0.0, 0.25, 0.5, 0.75, 1.0])
    return GraphDegreeSummary(
        node_count=nodes,
        directed_edge_count=int(edges.shape[1]),
        minimum=minimum,
        maximum=int(degree.max()),
        mean=float(degree.mean()),
        median=float(np.median(degree)),
        quantiles=tuple(float(value) for value in quantiles),
    )


__all__ = [
    "FullPanelJacobianError",
    "GraphDegreeSummary",
    "UniformShiftJacobianResult",
    "summarize_symmetric_directed_degree",
    "uniform_shift_jacobian",
]

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from spatial_benchmark import myjju_gradient_audit as audit


class LinearGraphForward(nn.Module):
    """Analytical GeneMAE-like forward with a fixed source-gene Jacobian."""

    def __init__(self, weights: Tensor, *, source_gene_index: int = 0) -> None:
        super().__init__()
        if weights.ndim != 3 or weights.shape[1] != weights.shape[2]:
            raise ValueError("weights must have shape [genes, nodes, nodes]")
        self.register_buffer("weights", weights)
        self.source_gene_index = source_gene_index

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor | None = None,
        entry_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del edge_index, edge_attr
        if entry_mask is None:
            raise ValueError("test forward requires an explicit mask")
        visible = torch.where(entry_mask, torch.zeros_like(x), x)
        source = visible[:, self.source_gene_index]
        columns = [
            self.weights[gene] @ source
            for gene in range(self.weights.shape[0])
        ]
        return torch.stack(columns, dim=1), entry_mask


class WrongReturnedMask(LinearGraphForward):
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor | None = None,
        entry_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        reconstruction, mask = super().forward(
            x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
        )
        return reconstruction, ~mask


class IgnoresInputMask(LinearGraphForward):
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor | None = None,
        entry_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del edge_index, edge_attr
        if entry_mask is None:
            raise ValueError("test forward requires an explicit mask")
        source = x[:, self.source_gene_index]
        columns = [
            self.weights[gene] @ source
            for gene in range(self.weights.shape[0])
        ]
        return torch.stack(columns, dim=1), entry_mask


def _model(weights: Tensor) -> LinearGraphForward:
    return LinearGraphForward(weights).eval()


def _empty_edges(dtype: torch.dtype = torch.long) -> Tensor:
    return torch.empty((2, 0), dtype=dtype)


def _all_directed_edges(num_nodes: int) -> Tensor:
    source = torch.arange(num_nodes).repeat_interleave(num_nodes)
    target = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack((source, target), dim=0)


def test_frozen_marker_modules_and_pairs_match_contract() -> None:
    assert tuple(audit.FROZEN_MARKER_MODULES) == (
        "epithelial",
        "proliferation",
        "immune_T",
        "immune_myeloid_B",
        "stroma",
        "endothelial",
        "cancer_candidate",
    )
    assert len(audit.FROZEN_MARKER_GENES) == 39
    assert len(set(audit.FROZEN_MARKER_GENES)) == 39
    assert audit.FROZEN_SOURCE_SELECTED_PAIRS == {
        "KRT8": ("KRT19", "KRT18", "KRT7"),
        "COL1A1": ("COL3A1", "COL1A2", "DCN"),
        "EPCAM": ("PSCA", "OLFM4", "KRT19", "CLDN4", "CDH1"),
        "OLFM4": ("PSCA", "EPCAM", "KRT19", "CDH1"),
        "CEACAM6": ("KRT19", "KRT8", "CLDN4", "KRT18"),
    }
    assert len(audit.FROZEN_LOCKED_DIRECTED_PAIRS) == 19
    assert len(set(audit.FROZEN_LOCKED_DIRECTED_PAIRS)) == 19
    assert audit.FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE == 8
    assert (
        audit.FROZEN_RECEIVER_SAMPLING_NAMESPACE
        == "myjju-gradient-audit-receiver-v1"
    )
    assert audit.FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT == 1e-6


def test_exact_single_receiver_signed_l1_and_outside_support() -> None:
    x = torch.tensor(
        [[1.0, 0.0, 2.0], [2.0, 1.0, 3.0],
         [3.0, 2.0, 4.0], [4.0, 3.0, 5.0]],
        dtype=torch.float64,
    )
    weights = torch.zeros((3, 4, 4), dtype=torch.float64)
    weights[1, 1] = torch.tensor(
        [3.0, 2.0, -4.0, 0.0], dtype=torch.float64
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[1, 1] = True
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)

    result = audit.decomposed_masked_input_gradients(
        _model(weights),
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((1,),),
    )

    np.testing.assert_allclose(result.input_gradient[0, :, 0], [3, 2, -4, 0])
    np.testing.assert_allclose(result.signed_same_cell, [[2]])
    np.testing.assert_allclose(result.signed_other_cell, [[-1]])
    np.testing.assert_allclose(result.signed_total, [[1]])
    np.testing.assert_allclose(result.l1_same_cell, [[2]])
    np.testing.assert_allclose(result.l1_other_cell, [[7]])
    np.testing.assert_allclose(result.l1_total, [[9]])
    np.testing.assert_allclose(result.global_vjp_l1_mass, [[9]])
    assert result.masked_source_max_abs == 0.0
    assert result.outside_receptive_field_max_abs == 0.0
    assert not result.signed_total.flags.writeable


def test_masked_source_derivative_is_zero_before_aggregation() -> None:
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    weights = torch.zeros((3, 4, 4), dtype=torch.float64)
    weights[1, 1] = torch.tensor(
        [3.0, 2.0, -4.0, 0.0], dtype=torch.float64
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[1, 1] = True
    mask[0, 0] = True
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)

    result = audit.decomposed_masked_input_gradients(
        _model(weights),
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((1,),),
    )

    np.testing.assert_allclose(result.input_gradient[0, :, 0], [0, 2, -4, 0])
    np.testing.assert_allclose(result.signed_same_cell, [[2]])
    np.testing.assert_allclose(result.signed_other_cell, [[-4]])
    np.testing.assert_allclose(result.l1_other_cell, [[4]])
    assert result.masked_source_max_abs == 0.0


def test_ignoring_a_masked_source_fails_closed() -> None:
    x = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    weights = torch.zeros((2, 4, 4), dtype=torch.float64)
    weights[1, 1, 0] = 3.0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[1, 1] = True
    mask[0, 0] = True
    model = IgnoresInputMask(weights).eval()

    with pytest.raises(
        audit.GradientAuditError, match="masked source input"
    ):
        audit.decomposed_masked_input_gradients(
            model,
            x,
            torch.tensor([[0], [1]], dtype=torch.long),
            entry_mask=mask,
            target_gene_indices=(1,),
            source_gene_indices=(0,),
            receiver_indices=((1,),),
        )


def test_summed_vjp_row_selection_is_not_an_exact_same_cell_split() -> None:
    x = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    weights = torch.zeros((2, 2, 2), dtype=torch.float64)
    weights[1] = torch.tensor([[1.0, 10.0], [20.0, 1.0]])
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 1] = True
    edge_index = _all_directed_edges(2)

    exact = audit.decomposed_masked_input_gradients(
        _model(weights),
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((0, 1),),
    )
    global_result = audit.summed_output_input_gradients(
        _model(weights),
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((0, 1),),
    )

    np.testing.assert_allclose(exact.signed_same_cell, [[1]])
    np.testing.assert_allclose(exact.signed_other_cell, [[15]])
    np.testing.assert_allclose(exact.signed_total, [[16]])
    np.testing.assert_allclose(global_result.signed_total, [[16]])
    naive_same_from_selected_global_rows = global_result.input_gradient[
        0, [0, 1], 0
    ].sum()
    assert naive_same_from_selected_global_rows == pytest.approx(16.0)
    assert naive_same_from_selected_global_rows != pytest.approx(
        exact.signed_same_cell[0, 0]
    )
    assert not hasattr(global_result, "signed_same_cell")


def test_receiver_resolved_l1_prevents_vjp_cancellation() -> None:
    x = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    weights = torch.zeros((2, 4, 4), dtype=torch.float64)
    weights[1, 0] = torch.tensor([2.0, 3.0, 0.0, 0.0])
    weights[1, 2] = torch.tensor([0.0, -3.0, -4.0, 0.0])
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[[0, 2], 1] = True

    result = audit.decomposed_masked_input_gradients(
        _model(weights),
        x,
        _all_directed_edges(4),
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((0, 2),),
    )

    np.testing.assert_allclose(result.signed_same_cell, [[-1]])
    np.testing.assert_allclose(result.signed_other_cell, [[0]])
    np.testing.assert_allclose(result.signed_total, [[-1]])
    np.testing.assert_allclose(result.l1_same_cell, [[3]])
    np.testing.assert_allclose(result.l1_other_cell, [[3]])
    np.testing.assert_allclose(result.l1_total, [[6]])
    np.testing.assert_allclose(result.global_vjp_l1_mass, [[3]])


def test_source_style_wrapper_is_all_unmasked_signed_and_directed() -> None:
    x = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    weights = torch.zeros((2, 4, 4), dtype=torch.float64)
    weights[1, 0] = torch.tensor([2.0, 3.0, 0.0, 0.0])
    weights[1, 2] = torch.tensor([0.0, -3.0, -4.0, 0.0])

    result = audit.source_style_all_unmasked_gradients(
        _model(weights),
        x,
        _all_directed_edges(4),
        target_gene_indices=(1,),
        source_gene_indices=(0,),
    )

    # d(sum receiver outputs)/d(all source inputs), divided by four cells.
    np.testing.assert_allclose(result.signed_total, [[-0.5]])
    assert result.receiver_indices == ((0, 1, 2, 3),)
    np.testing.assert_allclose(result.normalization_divisors, [4.0])
    assert result.masked_source_max_abs == 0.0


def test_bounded_perturbation_matches_linear_model_exactly() -> None:
    nodes = 100
    x = torch.zeros((nodes, 2), dtype=torch.float64)
    x[:, 0] = torch.linspace(0.0, 10.0, nodes, dtype=torch.float64)
    weights = torch.zeros((2, nodes, nodes), dtype=torch.float64)
    weights[1, 50] = torch.linspace(
        -1.0, 2.0, nodes, dtype=torch.float64
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[50, 1] = True
    edge_index = torch.stack(
        (torch.arange(nodes), torch.full((nodes,), 50)), dim=0
    )
    model = _model(weights)
    gradient = audit.decomposed_masked_input_gradients(
        model,
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((50,),),
    )
    perturbation = audit.make_bounded_source_perturbation(
        x,
        mask,
        source_gene_index=0,
        scale_in_sd=0.10,
    )
    comparison = audit.compare_central_bounded_perturbation(
        model,
        x,
        edge_index,
        entry_mask=mask,
        gradient_result=gradient,
        perturbation=perturbation,
    )

    np.testing.assert_allclose(
        comparison.predicted_plus_change,
        comparison.actual_plus_change,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        comparison.predicted_minus_change,
        comparison.actual_minus_change,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        comparison.predicted_centered_change,
        comparison.actual_centered_change,
        atol=1e-12,
    )
    np.testing.assert_allclose(comparison.curvature_residual, [0.0], atol=1e-12)
    assert perturbation.visible_count == nodes
    assert torch.all(
        perturbation.plus_input[:, 1] == x[:, 1]
    )


def test_bounded_perturbation_leaves_masked_source_inputs_unchanged() -> None:
    x = torch.zeros((100, 2), dtype=torch.float64)
    x[:, 0] = torch.arange(100, dtype=torch.float64)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[[1, 3, 5], 0] = True

    perturbation = audit.make_bounded_source_perturbation(
        x,
        mask,
        source_gene_index=0,
        scale_in_sd=0.25,
    )

    assert perturbation.visible_count == 97
    assert torch.equal(
        perturbation.plus_input[mask[:, 0], 0],
        x[mask[:, 0], 0],
    )
    assert torch.equal(
        perturbation.minus_input[mask[:, 0], 0],
        x[mask[:, 0], 0],
    )
    # Baselines outside the frozen bounds remain unchanged in both directions.
    assert perturbation.minus_input[0, 0] == x[0, 0]
    assert perturbation.plus_input[0, 0] == x[0, 0]
    assert perturbation.plus_input[-1, 0] == x[-1, 0]
    assert perturbation.minus_input[-1, 0] == x[-1, 0]
    assert torch.all(perturbation.plus_direction >= 0)
    assert torch.all(perturbation.minus_direction <= 0)


def test_four_hop_receptive_field_excludes_distance_five() -> None:
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    support = audit.receptive_field_nodes(
        edge_index, 5, num_nodes=6, maximum_hops=4
    )
    np.testing.assert_array_equal(support, [1, 2, 3, 4, 5])

    x = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    weights = torch.zeros((2, 6, 6), dtype=torch.float64)
    weights[1, 5, 1] = 7.0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[5, 1] = True
    result = audit.decomposed_masked_input_gradients(
        _model(weights),
        x,
        edge_index,
        entry_mask=mask,
        target_gene_indices=(1,),
        source_gene_indices=(0,),
        receiver_indices=((5,),),
    )
    np.testing.assert_allclose(
        result.input_gradient[0, :, 0], [0, 7, 0, 0, 0, 0]
    )
    assert result.outside_receptive_field_max_abs == 0.0


def test_nonzero_gradient_outside_receptive_field_fails() -> None:
    x = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    weights = torch.zeros((2, 6, 6), dtype=torch.float64)
    weights[1, 5, 0] = 1.0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[5, 1] = True
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )

    with pytest.raises(
        audit.GradientAuditError, match="outside the configured receptive"
    ):
        audit.decomposed_masked_input_gradients(
            _model(weights),
            x,
            edge_index,
            entry_mask=mask,
            target_gene_indices=(1,),
            source_gene_indices=(0,),
            receiver_indices=((5,),),
        )


def test_source_published_transform_preserves_expected_symmetry() -> None:
    directed = np.asarray([[1.0, -2.0], [3.0, -4.0]])
    transformed = audit.source_absolute_symmetric_transform(directed)
    np.testing.assert_allclose(transformed, [[1.0, 2.5], [2.5, 4.0]])
    assert not transformed.flags.writeable


def test_deterministic_metrics_with_ties_sign_and_error() -> None:
    np.testing.assert_allclose(
        audit.deterministic_average_ranks([20, 10, 20, 30]),
        [2.5, 1.0, 2.5, 4.0],
    )
    assert audit.deterministic_spearman(
        [1, -2, 3, -4], [2, -4, 6, -8]
    ) == pytest.approx(1.0)
    metrics = audit.faithfulness_metrics(
        np.asarray([1, -2, 3, -4], dtype=np.float64),
        np.asarray([2, -4, 6, -8], dtype=np.float64),
    )
    assert metrics.spearman == pytest.approx(1.0)
    assert metrics.sign_agreement == 1.0
    assert metrics.sign_comparison_count == 4
    assert metrics.median_absolute_error == 2.5
    assert metrics.median_absolute_actual_change == 5.0
    assert metrics.median_absolute_error_ratio == 0.5


@pytest.mark.parametrize(
    ("predicted", "actual", "match"),
    [
        ([1.0], [1.0], "at least 2"),
        ([1.0, 1.0], [1.0, 2.0], "constant rank"),
        ([1.0, np.nan], [1.0, 2.0], "nonfinite"),
    ],
)
def test_spearman_invalid_inputs_fail_closed(
    predicted: list[float],
    actual: list[float],
    match: str,
) -> None:
    with pytest.raises(audit.GradientAuditError, match=match):
        audit.deterministic_spearman(predicted, actual)


def test_sign_and_relative_error_invalid_cases_fail_closed() -> None:
    with pytest.raises(
        audit.GradientAuditError, match="no actual changes"
    ):
        audit.sign_agreement_fraction(
            [1.0, -1.0], [1e-7, -1e-7]
        )
    with pytest.raises(
        audit.GradientAuditError, match="median absolute actual change is zero"
    ):
        audit.median_absolute_error_ratio([1.0, 2.0], [0.0, 0.0])


@pytest.mark.parametrize(
    "case",
    [
        "nonfinite_x",
        "duplicate_source",
        "unsorted_receivers",
        "unmasked_receiver",
        "wrong_returned_mask",
        "training_model",
    ],
)
def test_gradient_validation_failures(case: str) -> None:
    x = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    weights = torch.zeros((2, 4, 4), dtype=torch.float64)
    weights[1, 1, 0] = 1.0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[1, 1] = True
    model: nn.Module = _model(weights)
    sources = (0,)
    receivers = ((1,),)
    if case == "nonfinite_x":
        x[0, 0] = torch.nan
    elif case == "duplicate_source":
        sources = (0, 0)
    elif case == "unsorted_receivers":
        receivers = ((2, 1),)
        mask[2, 1] = True
    elif case == "unmasked_receiver":
        mask[1, 1] = False
    elif case == "wrong_returned_mask":
        model = WrongReturnedMask(weights).eval()
    elif case == "training_model":
        model = LinearGraphForward(weights)

    with pytest.raises(audit.GradientAuditError):
        audit.decomposed_masked_input_gradients(
            model,
            x,
            _all_directed_edges(4),
            entry_mask=mask,
            target_gene_indices=(1,),
            source_gene_indices=sources,
            receiver_indices=receivers,
        )


def test_bounded_perturbation_validation_failures() -> None:
    constant = torch.ones((10, 2), dtype=torch.float64)
    mask = torch.zeros_like(constant, dtype=torch.bool)
    with pytest.raises(audit.GradientAuditError, match="positive finite"):
        audit.make_bounded_source_perturbation(
            constant,
            mask,
            source_gene_index=0,
            scale_in_sd=0.10,
        )

    variable = constant.clone()
    variable[:, 0] = torch.arange(10, dtype=torch.float64)
    with pytest.raises(audit.GradientAuditError, match="frozen"):
        audit.make_bounded_source_perturbation(
            variable,
            mask,
            source_gene_index=0,
            scale_in_sd=0.50,
        )
    with pytest.raises(audit.GradientAuditError, match="clip_quantiles"):
        audit.make_bounded_source_perturbation(
            variable,
            mask,
            source_gene_index=0,
            scale_in_sd=0.10,
            clip_quantiles=(0.0, 1.0),
        )

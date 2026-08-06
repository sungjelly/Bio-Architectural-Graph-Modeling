from __future__ import annotations

import pytest
import torch

from spatial_benchmark.signed_program_synthetic import (
    SIGNED_PROGRAM_VARIANTS,
    SignedProgramAdditiveModel,
)
from spatial_benchmark.training import set_deterministic_seed


def _model(
    *,
    variant: str = "count_sum",
    local_routing: str = "true",
) -> SignedProgramAdditiveModel:
    set_deterministic_seed(2718, deterministic=True, warn_only=False)
    return SignedProgramAdditiveModel(
        num_genes=3,
        expression_mean=torch.zeros(3),
        expression_scale=torch.ones(3),
        variant=variant,  # type: ignore[arg-type]
        hidden_dim=8,
        decoder_dim=8,
        ffn_dim=12,
        dropout=0.0,
        regional_routing="true",
        local_routing=local_routing,
    )


def _inputs() -> dict[str, torch.Tensor]:
    expression = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    mask = torch.zeros_like(expression, dtype=torch.bool)
    mask[5] = True
    local_edges = torch.tensor(
        [[0, 1, 2, 3], [3, 3, 4, 5]], dtype=torch.long
    )
    regional_edges = torch.tensor(
        [[0, 1, 2], [4, 5, 5]], dtype=torch.long
    )
    local_attributes = torch.zeros((4, 3), dtype=torch.float32)
    local_attributes[:, 0] = torch.tensor([10.0, 30.0, 50.0, 70.0])
    local_attributes[:, 1] = local_attributes[:, 0] / 75.0
    regional_attributes = torch.zeros((3, 3), dtype=torch.float32)
    regional_attributes[:, 0] = torch.tensor([100.0, 150.0, 250.0])
    regional_attributes[:, 1] = regional_attributes[:, 0] / 300.0
    return {
        "input_expression": expression,
        "gene_mask": mask,
        "local_edge_index": local_edges,
        "local_edge_attributes": local_attributes,
        "regional_edge_index": regional_edges,
        "regional_edge_attributes": regional_attributes,
    }


@pytest.mark.parametrize(
    ("variant", "basis_dim"),
    [
        ("count_sum", 1),
        ("count_plus_concentration", 2),
        ("radial_count", 4),
    ],
)
def test_prespecified_variants_have_fixed_basis(
    variant: str,
    basis_dim: int,
) -> None:
    model = _model(variant=variant)
    assert variant in SIGNED_PROGRAM_VARIANTS
    assert model.local_basis_dim == basis_dim
    assert not model.sender_program_matrix.requires_grad
    assert not model.receiver_program_matrix.requires_grad
    assert torch.equal(model.sender_program_matrix, torch.eye(3))
    assert torch.equal(model.receiver_program_matrix, torch.eye(3))


def test_selected_edge_contributions_sum_exactly_to_local_prediction() -> None:
    model = _model()
    with torch.no_grad():
        model.local_coefficients.zero_()
        # Positive gene-0 sender program -> receiver gene-2 continuous channel.
        model.local_coefficients[1, 2, 0, 0] = 2.0
    inputs = _inputs()
    output = model(
        **inputs,
        local_contribution_edge_indices=torch.arange(4),
        local_contribution_gene_indices=[2],
    )
    assert output.local_contributions is not None
    assert output.local_contributions.shape == (4, 1, 2)
    reconstructed = torch.zeros_like(output.local_prediction)
    receiver = inputs["local_edge_index"][1]
    reconstructed[:, 2:3, :].index_add_(
        0, receiver, output.local_contributions
    )
    assert torch.equal(reconstructed, output.local_prediction)
    assert torch.all(output.local_contributions[:, 0, 1] >= 0)


def test_masked_values_cannot_enter_any_prediction_branch() -> None:
    model = _model(variant="radial_count")
    inputs = _inputs()
    first = model(**inputs).prediction
    corrupted = dict(inputs)
    changed = inputs["input_expression"].clone()
    changed[inputs["gene_mask"]] = 1_000_000.0
    corrupted["input_expression"] = changed
    second = model(**corrupted).prediction
    assert torch.equal(first, second)


def test_routing_arms_are_parameter_matched_at_initialization() -> None:
    true_model = _model(local_routing="true")
    disabled_model = _model(local_routing="surrogate")
    permuted_model = _model(local_routing="permuted")
    true_parameters = dict(true_model.named_parameters())
    for candidate in (disabled_model, permuted_model):
        candidate_parameters = dict(candidate.named_parameters())
        assert true_parameters.keys() == candidate_parameters.keys()
        for name in true_parameters:
            assert torch.equal(
                true_parameters[name], candidate_parameters[name]
            )


def test_permuted_routing_reports_effective_source_indices() -> None:
    model = _model(local_routing="permuted")
    inputs = _inputs()
    permutation = torch.tensor([2, 3, 4, 5, 0, 1])
    output = model(
        **inputs,
        local_source_index_by_node=permutation,
        local_contribution_edge_indices=[0, 2],
        local_contribution_gene_indices=[0],
    )
    expected = permutation.index_select(
        0, inputs["local_edge_index"][0, [0, 2]]
    )
    assert torch.equal(
        output.local_contribution_effective_source_index, expected
    )


def test_nonnegative_program_contract_and_permutation_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        SignedProgramAdditiveModel(
            num_genes=2,
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
            variant="count_sum",
            sender_program_matrix=[[1.0, -1.0]],
        )
    model = _model(local_routing="permuted")
    with pytest.raises(ValueError, match="complete node permutation"):
        model(
            **_inputs(),
            local_source_index_by_node=torch.zeros(6, dtype=torch.long),
        )

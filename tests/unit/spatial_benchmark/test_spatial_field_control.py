"""Torch-only integration tests for the broad spatial-field control."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.models import (  # noqa: E402
    BroadSpatialFieldControl,
    SelfOnlyMLP,
)
from spatial_benchmark.spatial_field import (  # noqa: E402
    BroadSpatialFieldBasis,
)
from spatial_benchmark.training import (  # noqa: E402
    GraphSplitView,
    TrainingConfig,
    build_model,
    evaluate_fixed_mask,
    fit_model,
)


def _view(
    coordinates_um: torch.Tensor,
    *,
    expression: torch.Tensor,
    name: str,
) -> GraphSplitView:
    num_nodes = expression.shape[0]
    return GraphSplitView(
        expression=expression,
        coordinates_um=coordinates_um,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_covariates=torch.arange(
            num_nodes * 2, dtype=torch.float32
        ).reshape(num_nodes, 2),
        block_ids=np.arange(num_nodes),
        name=name,
    )


def test_control_is_mask_safe_coordinate_aware_and_graph_independent() -> None:
    torch.manual_seed(17)
    model = BroadSpatialFieldControl(
        num_genes=3,
        node_covariate_dim=2,
        hidden_dim=8,
        ffn_dim=10,
        decoder_dim=9,
        dropout=0.0,
    ).eval()
    expression = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
        ]
    )
    covariates = torch.zeros((4, 2))
    coordinates = torch.tensor(
        [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]]
    )

    with pytest.raises(RuntimeError, match="fit_coordinate_basis"):
        model(
            expression,
            mask,
            node_covariates=covariates,
            coordinates_um=coordinates,
        )
    model.fit_coordinate_basis(coordinates)
    original = model(
        expression,
        mask,
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attributes=torch.ones((2, 3)),
        node_covariates=covariates,
        coordinates_um=coordinates,
    ).prediction

    changed_hidden = expression.clone()
    changed_hidden[mask] = torch.nan
    graph_changed = model(
        changed_hidden,
        mask,
        edge_index=torch.tensor([[3], [2]]),
        edge_attributes=torch.full((1, 7), 999.0),
        node_covariates=covariates,
        coordinates_um=coordinates,
    ).prediction
    torch.testing.assert_close(original, graph_changed, rtol=0.0, atol=0.0)

    moved_coordinates = coordinates.clone()
    moved_coordinates[0] += torch.tensor([1.5, -0.75])
    moved = model(
        expression,
        mask,
        node_covariates=covariates,
        coordinates_um=moved_coordinates,
    ).prediction
    assert not torch.allclose(original[0], moved[0])
    torch.testing.assert_close(original[1:], moved[1:], rtol=0.0, atol=0.0)
    assert not any(
        isinstance(module, torch.nn.Embedding) for module in model.modules()
    )

    ordinary = SelfOnlyMLP(
        num_genes=3,
        node_covariate_dim=2,
        hidden_dim=8,
        dropout=0.0,
    )
    with pytest.raises(TypeError, match="coordinates_um"):
        ordinary(
            expression,
            mask,
            node_covariates=covariates,
            coordinates_um=coordinates,
        )


def test_torch_basis_round_trip_matches_numpy_and_zeros_constant_axes() -> None:
    training_coordinates = np.asarray(
        [[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]]
    )
    query = np.asarray([[102.0, 4.0], [-500.0, 6.0]])
    basis = BroadSpatialFieldBasis.fit(training_coordinates)
    model = BroadSpatialFieldControl(
        num_genes=3,
        node_covariate_dim=0,
        hidden_dim=8,
        dropout=0.0,
    )
    model.set_coordinate_basis(basis)
    expected = basis.transform(query)
    actual = model.coordinate_basis(
        torch.from_numpy(query)
    ).detach().cpu().numpy()
    np.testing.assert_array_equal(actual.astype(np.float32), expected)
    assert not actual[:, [0, 2, 3]].any()

    restored = BroadSpatialFieldControl(
        num_genes=3,
        node_covariate_dim=0,
        hidden_dim=8,
        dropout=0.0,
    )
    restored.load_state_dict(model.state_dict())
    assert restored.coordinate_basis_is_fitted
    assert restored.spatial_basis_provenance() == (
        model.spatial_basis_provenance()
    )
    torch.testing.assert_close(
        restored.coordinate_basis(torch.from_numpy(query)),
        model.coordinate_basis(torch.from_numpy(query)),
        rtol=0.0,
        atol=0.0,
    )


def test_training_fits_only_train_coordinates_and_preserves_pairing() -> None:
    generator = torch.Generator().manual_seed(23)
    train_expression = torch.randn((6, 4), generator=generator)
    validation_expression = torch.randn((6, 4), generator=generator)
    train_coordinates = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [4.0, 2.0],
            [6.0, 3.0],
            [8.0, 4.0],
            [10.0, 5.0],
        ]
    )
    validation_coordinates = train_coordinates + torch.tensor(
        [10_000.0, -7_000.0]
    )
    train_view = _view(
        train_coordinates,
        expression=train_expression,
        name="train",
    )
    validation_view = _view(
        validation_coordinates,
        expression=validation_expression,
        name="validation",
    )
    model_kwargs = {
        "num_genes": train_view.num_genes,
        "node_covariate_dim": train_view.node_covariate_dim,
        "hidden_dim": 8,
        "ffn_dim": 10,
        "decoder_dim": 9,
        "dropout": 0.0,
        "seed": 41,
    }
    broad = build_model("broad-field", **model_kwargs)
    ordinary = build_model("b0", **model_kwargs)
    config = TrainingConfig(
        max_epochs=1,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=1,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=719,
        model_seed=41,
        edge_dropout=0.25,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_expression, dtype=torch.bool
    )
    validation_mask[:, 0] = True

    broad_result = fit_model(
        broad,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )
    ordinary_result = fit_model(
        ordinary,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )
    provenance = broad_result.spatial_control_provenance
    assert provenance is not None
    np.testing.assert_allclose(
        provenance["center_um"],
        train_coordinates.numpy().mean(axis=0),
    )
    np.testing.assert_allclose(
        provenance["scale_um"],
        train_coordinates.numpy().std(axis=0, ddof=0),
    )
    assert provenance["fit_scope"] == "training split coordinates only"
    assert (
        broad_result.history[0].mask_checksum
        == ordinary_result.history[0].mask_checksum
    )
    assert (
        broad_result.history[0].edge_checksum
        == ordinary_result.history[0].edge_checksum
    )
    assert broad_result.graph_execution == (
        "cell_autonomous_broad_spatial_field_no_graph"
    )

    evaluation = evaluate_fixed_mask(
        broad,
        validation_view,
        validation_mask,
        device="cpu",
    )
    assert evaluation.predictions.shape == validation_expression.shape

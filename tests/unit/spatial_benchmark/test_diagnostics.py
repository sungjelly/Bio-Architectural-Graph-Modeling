from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.diagnostics import (  # noqa: E402
    DiagnosticContractError,
    NearestSpatialNeighborCopyPredictor,
    TrainGlobalMeanPredictor,
    nearest_spatial_neighbor_copy_prediction,
    train_global_mean_prediction,
)
from spatial_benchmark.metrics import evaluate_masked_predictions  # noqa: E402


def test_train_global_mean_is_train_fitted_and_metrics_ready() -> None:
    training = np.asarray(
        [[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]],
        dtype=np.float32,
    )
    mask = np.asarray(
        [[True, False, True], [False, True, False], [True, True, False]]
    )
    target = np.asarray(
        [
            [9001.0, 12.0, 9002.0],
            [13.0, 9003.0, 14.0],
            [9004.0, 9005.0, 15.0],
        ]
    )

    predictor = TrainGlobalMeanPredictor.fit(training)
    result = predictor.predict(mask, split_name="validation")

    np.testing.assert_allclose(predictor.gene_mean, [2.0, 20.0, 200.0])
    np.testing.assert_allclose(
        result.predictions,
        np.broadcast_to([2.0, 20.0, 200.0], mask.shape),
    )
    assert result.control == "train_global_gene_mean"
    assert result.split_name == "validation"
    assert result.fit_scope == "training cells only"
    assert result.n_training_cells == 2
    assert result.n_evaluated_entries == int(mask.sum())
    assert result.n_copied_entries == 0
    assert result.n_fallback_entries == int(mask.sum())
    assert result.copy_rate == 0.0
    assert np.all(result.source_node_index == -1)
    assert not result.predictions.flags.writeable
    assert not predictor.gene_mean.flags.writeable

    metrics = evaluate_masked_predictions(**result.metrics_inputs(target))
    assert metrics["n_masked"] == int(mask.sum())

    # Held-out target values never enter fitting or mean prediction.
    changed_target = target.copy()
    changed_target[mask] *= -1000
    repeated = predictor.predict(mask, split_name="validation")
    np.testing.assert_array_equal(result.predictions, repeated.predictions)

    convenience = train_global_mean_prediction(
        training, mask, split_name="validation"
    )
    np.testing.assert_array_equal(convenience.predictions, result.predictions)


def test_nearest_copy_uses_nearest_visible_gene_and_never_hidden_values() -> None:
    training = np.asarray(
        [[0.0, 40.0, 400.0, 4000.0], [10.0, 60.0, 600.0, 6000.0]]
    )
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    expression = np.asarray(
        [
            [9001.0, 10.0, 9002.0, 9003.0],
            [111.0, 9004.0, 9005.0, 9006.0],
            [222.0, 20.0, 333.0, 9007.0],
        ]
    )
    mask = np.asarray(
        [
            [True, False, True, True],
            [False, True, True, True],
            [False, False, False, True],
        ]
    )
    original_expression = expression.copy()
    original_mask = mask.copy()

    predictor = NearestSpatialNeighborCopyPredictor.fit(training)
    result = predictor.predict(
        coordinates,
        expression,
        mask,
        split_name="validation",
        distance_block_size=1,
    )
    assert result.control == "nearest_spatial_neighbor_copy"

    # Cell 0/gene 0 uses the immediate visible neighbor.  Its gene 2 is
    # hidden there, so the next-nearest visible source is used instead.
    assert result.predictions[0, 0] == 111.0
    assert result.source_node_index[0, 0] == 1
    assert result.source_distance_um[0, 0] == pytest.approx(1.0)
    assert result.predictions[0, 2] == 333.0
    assert result.source_node_index[0, 2] == 2
    assert result.source_distance_um[0, 2] == pytest.approx(2.0)

    # Cell 1/gene 1 uses cell 0; cell 1/gene 2 uses cell 2 because cell 0's
    # copy is hidden.  Gene 3 is hidden everywhere and falls back to train.
    assert result.predictions[1, 1] == 10.0
    assert result.source_node_index[1, 1] == 0
    assert result.predictions[1, 2] == 333.0
    assert result.source_node_index[1, 2] == 2
    np.testing.assert_array_equal(
        result.predictions[mask[:, 3], 3],
        np.full(int(mask[:, 3].sum()), 5000.0),
    )
    assert np.all(result.source_node_index[mask[:, 3], 3] == -1)

    copied_rows, copied_genes = np.nonzero(result.copied_mask)
    copied_sources = result.source_node_index[copied_rows, copied_genes]
    assert np.all(copied_sources != copied_rows)
    assert np.all(~mask[copied_sources, copied_genes])

    # The predictor may receive a full stored target matrix, but changing every
    # hidden value cannot affect sources or predictions.
    changed = expression.copy()
    changed[mask] = np.arange(mask.sum()) + 1_000_000
    repeated = predictor.predict(
        coordinates,
        changed,
        mask,
        split_name="validation",
        distance_block_size=2,
    )
    np.testing.assert_array_equal(
        repeated.source_node_index, result.source_node_index
    )
    np.testing.assert_array_equal(repeated.predictions, result.predictions)
    np.testing.assert_array_equal(expression, original_expression)
    np.testing.assert_array_equal(mask, original_mask)


def test_ties_self_exclusion_and_minimum_distance_are_deterministic() -> None:
    training = np.asarray([[5.0], [7.0]])
    coordinates = np.asarray(
        [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    )
    expression = np.asarray([[999.0], [10.0], [20.0], [30.0]])
    mask = np.asarray([[True], [False], [False], [False]])
    predictor = NearestSpatialNeighborCopyPredictor.fit(training)

    tied = predictor.predict(
        coordinates,
        expression,
        mask,
        distance_block_size=1,
    )
    assert tied.source_node_index[0, 0] == 1
    assert tied.predictions[0, 0] == 10.0
    assert tied.source_node_index[0, 0] != 0

    repeated = predictor.predict(
        coordinates,
        expression,
        mask,
        distance_block_size=4,
    )
    np.testing.assert_array_equal(
        tied.source_node_index, repeated.source_node_index
    )
    np.testing.assert_array_equal(tied.predictions, repeated.predictions)

    inclusive_minimum = predictor.predict(
        coordinates,
        expression,
        mask,
        min_distance_um=1.0,
    )
    assert inclusive_minimum.source_node_index[0, 0] == 1

    minimum_separation = predictor.predict(
        coordinates,
        expression,
        mask,
        min_distance_um=1.5,
        distance_block_size=2,
    )
    assert minimum_separation.source_node_index[0, 0] == 3
    assert minimum_separation.source_distance_um[0, 0] == pytest.approx(3.0)
    assert minimum_separation.predictions[0, 0] == 30.0

    no_source = predictor.predict(
        coordinates,
        expression,
        mask,
        min_distance_um=10.0,
    )
    assert no_source.source_node_index[0, 0] == -1
    assert no_source.predictions[0, 0] == 6.0
    assert no_source.fallback_mask[0, 0]


def test_prediction_calls_are_structurally_split_local() -> None:
    training = np.asarray([[0.0], [2.0]])
    predictor = NearestSpatialNeighborCopyPredictor.fit(training)
    validation_coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    validation_expression = np.asarray([[999.0], [20.0]])
    validation_mask = np.asarray([[True], [False]])

    validation = predictor.predict(
        validation_coordinates,
        validation_expression,
        validation_mask,
        split_name="validation",
    )
    assert validation.predictions[0, 0] == 20.0
    assert validation.source_node_index[0, 0] == 1
    assert validation.source_node_index.max() < len(validation_coordinates)

    # A different split may contain a much closer and more extreme value, but
    # it is never supplied as a candidate to the validation call.
    test_coordinates = np.asarray([[0.01, 0.0], [0.02, 0.0]])
    test_expression = np.asarray([[999_999.0], [888_888.0]])
    test_mask = np.asarray([[True], [False]])
    test_result = predictor.predict(
        test_coordinates,
        test_expression,
        test_mask,
        split_name="test",
    )
    assert test_result.source_node_index[0, 0] == 1
    assert test_result.predictions[0, 0] == 888_888.0
    assert validation.predictions[0, 0] == 20.0
    assert validation.split_name == "validation"
    assert test_result.split_name == "test"


def test_singleton_fallback_and_input_contracts() -> None:
    training = np.asarray([[1.0, 10.0], [3.0, 30.0]])
    result = nearest_spatial_neighbor_copy_prediction(
        training,
        np.asarray([[0.0, 0.0]]),
        np.asarray([[999.0, 888.0]]),
        np.asarray([[True, True]]),
        split_name="test",
    )
    np.testing.assert_array_equal(result.predictions, [[2.0, 20.0]])
    assert result.n_fallback_entries == 2
    assert result.n_copied_entries == 0

    predictor = NearestSpatialNeighborCopyPredictor.fit(training)
    with pytest.raises(DiagnosticContractError, match="boolean"):
        predictor.predict(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2), dtype=np.int8),
        )
    with pytest.raises(DiagnosticContractError, match="shape"):
        predictor.predict(
            np.zeros((2, 3)),
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
        )
    with pytest.raises(DiagnosticContractError, match="finite"):
        predictor.predict(
            np.asarray([[0.0, 0.0], [np.nan, 1.0]]),
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
        )
    with pytest.raises(DiagnosticContractError, match="non-negative"):
        predictor.predict(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            min_distance_um=-0.1,
        )

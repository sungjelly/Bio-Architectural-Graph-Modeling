from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diagnostics import preflight_so2_target_isolated_nb_ddp as preflight
from scripts.train import run_so2_target_isolated_nb as runner
from spatial_benchmark.so2_nb_training import (
    EarlyStoppingState,
    SO2NBTrainingError,
    update_early_stopping,
)
from spatial_benchmark.target_cell_masking import verify_target_cell_receipt


def _tiny_batch(
    *, alias: str = "SO2-C15", n_nodes: int = 11, n_genes: int = 7
) -> SimpleNamespace:
    return SimpleNamespace(alias=alias, n_nodes=n_nodes, n_genes=n_genes)


def test_partition_masks_and_coverage_receipt_prove_exact_training_visits() -> None:
    batch = _tiny_batch()
    records = []
    for rank in range(runner.WORLD_SIZE):
        partition, realization = runner._partition_and_mask(
            batch,
            rank=rank,
            global_epoch=3,
        )
        assert np.array_equal(
            realization.target_indices, partition.target_indices
        )
        assert np.all(realization.masked_gene_counts >= 1)
        assert realization.to_receipt()["namespace"] == (
            runner.TRAINING_MASK_NAMESPACE
        )
        records.append(
            runner._local_receipt_record(
                rank=rank,
                batch=batch,
                partition=partition,
                realization=realization,
            )
        )

    receipt = runner._coverage_receipt(
        batch=batch,
        role="training",
        global_epoch=3,
        gathered=records,
    )

    assert receipt["target_cells"] == batch.n_nodes
    assert receipt["target_visit_count_min"] == 1
    assert receipt["target_visit_count_max"] == 1
    assert receipt["neighbor_masked_entry_count"] == 0
    assert receipt["masked_entries"] >= batch.n_nodes
    verify_target_cell_receipt(receipt["partition_coverage_receipt"])


def test_validation_partition_masks_are_fixed_and_epoch_free() -> None:
    batch = _tiny_batch(alias="SO2-C27")
    first_partition, first = runner._partition_and_mask(
        batch,
        rank=2,
        global_epoch=None,
    )
    replay_partition, replay = runner._partition_and_mask(
        batch,
        rank=2,
        global_epoch=None,
    )

    assert first_partition == replay_partition
    assert np.array_equal(first.mask, replay.mask)
    assert first.to_receipt() == replay.to_receipt()
    assert first.to_receipt()["fixed_across_epochs"] is True
    assert "global_epoch" not in first.to_receipt()


def test_target_tensor_builder_masks_only_the_local_copy() -> None:
    clean = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    original = clean.clone()
    mask = torch.tensor(
        [[True, False, False, True, False, False],
         [False, True, False, False, False, True]],
        dtype=torch.bool,
    )
    realization = SimpleNamespace(mask=mask.numpy())

    target, materialized_mask = runner._make_target_tensors(
        clean_input=clean,
        realization=realization,
        target_start=1,
        target_stop=3,
        device=torch.device("cpu"),
    )

    assert torch.equal(clean, original)
    assert torch.equal(materialized_mask, mask)
    assert torch.equal(target.masked_select(mask), torch.zeros(4))
    assert torch.equal(
        target.masked_select(~mask), clean[1:3].masked_select(~mask)
    )


def test_validation_sufficient_statistics_are_entry_pooled() -> None:
    # Field order is declared by the runner.  The squared-error sums below
    # deliberately differ from squared means so an equal-shard average would
    # not satisfy these expectations.
    values = torch.tensor(
        [
            4.0,   # masked entries
            20.0,  # NB NLL sum
            8.0,   # raw absolute-error sum
            36.0,  # raw squared-error sum
            2.0,   # log1p absolute-error sum
            1.0,   # log1p squared-error sum
            6.0,   # Poisson-deviance sum
            1.0,   # observed-zero sum
            1.6,   # predicted-zero-probability sum
            0.4,   # zero-Brier sum
            12.0,  # observed-count sum
            16.0,  # predicted-count sum
            2.0,   # target cells
        ],
        dtype=torch.float64,
    )

    metrics = runner._metrics_from_sums(values)

    assert metrics["pooled_masked_negative_binomial_nll"] == pytest.approx(5.0)
    assert metrics["masked_raw_count_mae"] == pytest.approx(2.0)
    assert metrics["masked_raw_count_rmse"] == pytest.approx(3.0)
    assert metrics["masked_log1p_rmse"] == pytest.approx(0.5)
    assert metrics["target_cells"] == 2
    assert metrics["masked_entries"] == 4


def test_validation_sufficient_statistics_reject_fractional_support() -> None:
    values = torch.ones(len(runner._VALIDATION_SUM_FIELDS), dtype=torch.float64)
    values[0] = 1.5
    with pytest.raises(SO2NBTrainingError, match="not integral"):
        runner._metrics_from_sums(values)


def test_model_step_seed_binds_epoch_update_core_and_rank() -> None:
    reference = runner._model_step_seed(2, 3, "SO2-C15", 1)
    assert reference == runner._model_step_seed(2, 3, "SO2-C15", 1)
    assert reference != runner._model_step_seed(3, 3, "SO2-C15", 1)
    assert reference != runner._model_step_seed(2, 4, "SO2-C15", 1)
    assert reference != runner._model_step_seed(2, 3, "SO2-C16", 1)
    assert reference != runner._model_step_seed(2, 3, "SO2-C15", 2)


def test_literal_lowest_checkpoint_selection_is_independent_of_patience_delta() -> None:
    selection = runner._update_exact_checkpoint_selection(
        EarlyStoppingState(), 1.0, completed_epoch=1
    )
    patience = update_early_stopping(
        EarlyStoppingState(),
        1.0,
        completed_epoch=1,
        minimum_epochs=50,
        patience=25,
        min_delta=1e-4,
        maximum_epochs=300,
    )

    # This is a new literal minimum, but its 5e-5 improvement is deliberately
    # smaller than the patience controller's meaningful-improvement threshold.
    selection = runner._update_exact_checkpoint_selection(
        selection, 0.99995, completed_epoch=2
    )
    patience = update_early_stopping(
        patience,
        0.99995,
        completed_epoch=2,
        minimum_epochs=50,
        patience=25,
        min_delta=1e-4,
        maximum_epochs=300,
    )

    assert selection.improved is True
    assert selection.best_epoch == 2
    assert selection.best_value == pytest.approx(0.99995)
    assert patience.improved is False
    assert patience.best_epoch == 1
    assert patience.best_value == pytest.approx(1.0)


def test_runner_and_preflight_share_the_exact_launch_contract() -> None:
    assert runner.PREFLIGHT_SCHEMA == preflight.PREFLIGHT_SCHEMA
    assert runner.PROTOCOL == preflight.PROTOCOL
    assert runner.WORLD_SIZE == preflight.WORLD_SIZE == 4
    assert runner.EXPECTED_PARAMETER_COUNT == preflight.EXPECTED_PARAMETER_COUNT
    assert runner.PREFLIGHT_REQUIRED_GATES == preflight.REQUIRED_GATE_NAMES
    assert runner.PREFLIGHT_CODE_HASH_SCOPE == preflight.CODE_HASH_SCOPE
    assert (
        runner.PREFLIGHT_CODE_RELATIVE_PATHS
        == preflight.CODE_RELATIVE_PATHS
    )
    assert runner.MINIMUM_FREE_DISK_GIB == preflight.MINIMUM_FREE_DISK_GIB


def test_preflight_fp32_nb2_gate_uses_scale_aware_tolerance() -> None:
    receipt = preflight._full_constant_fp32_nb2_receipt()

    assert receipt["passed"] is True
    assert receipt["dtype"] == "float32"
    assert receipt["includes_full_combinatorial_constant"] is True
    assert receipt["contains_zero_and_count_729"] is True
    assert receipt["absolute_tolerance"] == 2e-5
    assert receipt["relative_tolerance"] == 2e-6
    assert receipt["maximum_scaled_error_ratio"] <= 1.0
    assert (
        receipt["summed_nll_absolute_error"]
        <= receipt["summed_nll_allowed_error"]
    )

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from spatial_benchmark.target_cell_masking import (
    TRAINING_MASK_SCHEMA,
    VALIDATION_MASK_SCHEMA,
    derive_target_cell_seed,
    derive_training_target_mask_seed,
    derive_validation_target_mask_seed,
    make_contiguous_target_partition,
    make_contiguous_target_partitions,
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
    target_partition_coverage_receipt,
    verify_target_cell_receipt,
)


TRAINING_NAMESPACE = "bagm.so2.target_isolated_nb.training_masks.v1"


def test_training_masks_are_deterministic_and_expression_independent() -> None:
    first_expression = np.arange(72, dtype=np.float32).reshape(8, 9)
    second_expression = first_expression[::-1].copy() * -1000.0
    first = make_training_target_cell_masks(
        len(first_expression),
        first_expression.shape[1],
        base_seed=2026082401,
        namespace=TRAINING_NAMESPACE,
        core_alias="so2-c15",
        global_epoch=7,
    )
    second = make_training_target_cell_masks(
        len(second_expression),
        second_expression.shape[1],
        base_seed=2026082401,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C15",
        global_epoch=7,
    )

    assert np.array_equal(first.mask, second.mask)
    assert first.to_receipt() == second.to_receipt()
    assert first.to_receipt()["schema"] == TRAINING_MASK_SCHEMA
    assert first.to_receipt()["namespace"] == TRAINING_NAMESPACE
    assert first.to_receipt()["covers_all_cells"] is True
    assert first_expression.tobytes() != second_expression.tobytes()


def test_training_seed_binds_role_core_epoch_and_base_seed() -> None:
    reference = derive_training_target_mask_seed(
        19,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C15",
        global_epoch=3,
    )
    assert reference == derive_training_target_mask_seed(
        19,
        namespace=TRAINING_NAMESPACE,
        core_alias="so2-c15",
        global_epoch=3,
    )
    assert reference != derive_training_target_mask_seed(
        20,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C15",
        global_epoch=3,
    )
    assert reference != derive_training_target_mask_seed(
        19,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C16",
        global_epoch=3,
    )
    assert reference != derive_training_target_mask_seed(
        19,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C15",
        global_epoch=4,
    )
    assert reference != derive_training_target_mask_seed(
        19,
        namespace=f"{TRAINING_NAMESPACE}.changed",
        core_alias="SO2-C15",
        global_epoch=3,
    )
    assert reference != derive_validation_target_mask_seed(
        "19", core_alias="SO2-C15"
    )


def test_each_target_has_one_nonempty_uniform_without_replacement_subset() -> None:
    realization = make_training_target_cell_masks(
        512,
        11,
        base_seed=41,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C23",
        global_epoch=9,
    )
    counts = realization.masked_gene_counts

    assert counts.min() == 1
    assert counts.max() == 11
    assert set(counts.tolist()) == set(range(1, 12))
    assert np.array_equal(realization.mask.sum(axis=1), counts)
    assert realization.mask.dtype == np.bool_
    assert realization.mask.shape == (512, 11)
    assert realization.n_masked_entries == int(counts.sum())
    target_seeds = {
        derive_target_cell_seed(realization.seed, target_index)
        for target_index in range(512)
    }
    assert len(target_seeds) == 512


def test_target_masks_are_stable_across_partition_and_request_order() -> None:
    full = make_training_target_cell_masks(
        13,
        17,
        base_seed=53,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C19",
        global_epoch=2,
    )
    requested = np.array([12, 3, 8, 0], dtype=np.int64)
    subset = make_training_target_cell_masks(
        13,
        17,
        base_seed=53,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C19",
        global_epoch=2,
        target_indices=requested,
    )

    assert np.array_equal(subset.target_indices, requested)
    assert np.array_equal(subset.mask, full.mask[requested])
    assert np.array_equal(
        subset.masked_gene_counts, full.masked_gene_counts[requested]
    )
    assert subset.to_receipt()["covers_all_cells"] is False


def test_validation_masks_are_fixed_without_an_epoch_field() -> None:
    first = make_validation_target_cell_masks(
        21,
        7,
        namespace="bagm.so2.target-cell-once.validation.v1",
        core_alias="SO2-C27",
    )
    replay = make_validation_target_cell_masks(
        21,
        7,
        namespace="bagm.so2.target-cell-once.validation.v1",
        core_alias="so2-c27",
    )
    changed = make_validation_target_cell_masks(
        21,
        7,
        namespace="bagm.so2.target-cell-once.validation.v2",
        core_alias="SO2-C27",
    )

    assert np.array_equal(first.mask, replay.mask)
    assert first.to_receipt() == replay.to_receipt()
    assert first.seed != changed.seed
    receipt = first.to_receipt()
    assert receipt["schema"] == VALIDATION_MASK_SCHEMA
    assert receipt["fixed_across_epochs"] is True
    assert "global_epoch" not in receipt
    assert "base_seed" not in receipt


def test_mask_receipts_bind_counts_arrays_and_self_checksum() -> None:
    realization = make_training_target_cell_masks(
        10,
        5,
        base_seed=71,
        namespace=TRAINING_NAMESPACE,
        core_alias="SO2-C20",
        global_epoch=11,
    )
    receipt = realization.to_receipt()
    verify_target_cell_receipt(receipt)
    assert receipt["n_targets"] == 10
    assert receipt["masked_entry_count"] == int(
        realization.masked_gene_counts.sum()
    )
    assert receipt["masked_gene_count_min"] >= 1
    assert receipt["masked_gene_count_max"] <= 5
    assert len(receipt["target_indices_sha256"]) == 64
    assert len(receipt["masked_gene_counts_sha256"]) == 64
    assert len(receipt["mask_sha256"]) == 64

    tampered = deepcopy(receipt)
    tampered["masked_entry_count"] += 1
    with pytest.raises(ValueError, match="checksum"):
        verify_target_cell_receipt(tampered)


def test_contiguous_partitions_are_disjoint_exhaustive_and_balanced() -> None:
    partitions = make_contiguous_target_partitions(10, 3)
    assert [(part.start, part.stop) for part in partitions] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]
    assert [part.n_targets for part in partitions] == [4, 3, 3]
    combined = np.concatenate([part.target_indices for part in partitions])
    assert np.array_equal(combined, np.arange(10))
    assert len(np.unique(combined)) == 10

    receipt = target_partition_coverage_receipt(partitions)
    verify_target_cell_receipt(receipt)
    assert receipt["coverage_exhaustive"] is True
    assert receipt["partitions_disjoint"] is True
    assert receipt["partitions_contiguous"] is True
    assert receipt["total_target_count"] == 10
    assert receipt["unique_target_count"] == 10
    assert receipt["maximum_partition_size_difference"] == 1


def test_partition_supports_more_ranks_than_cells() -> None:
    partitions = make_contiguous_target_partitions(2, 4)
    assert [part.n_targets for part in partitions] == [1, 1, 0, 0]
    assert make_contiguous_target_partition(
        2, world_size=4, rank=3
    ) == partitions[3]
    receipt = target_partition_coverage_receipt(partitions)
    assert receipt["target_counts_by_rank"] == [1, 1, 0, 0]
    assert receipt["maximum_partition_size_difference"] == 1


def test_invalid_targets_and_partition_arguments_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_training_target_cell_masks(
            3,
            4,
            base_seed=1,
            namespace=TRAINING_NAMESPACE,
            core_alias="SO2-C15",
            global_epoch=0,
            target_indices=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="out-of-range"):
        make_validation_target_cell_masks(
            3,
            4,
            namespace="validation-v1",
            core_alias="SO2-C27",
            target_indices=np.array([3]),
        )
    with pytest.raises(TypeError, match="integer"):
        make_contiguous_target_partitions(3, True)
    with pytest.raises(ValueError, match="less than world_size"):
        make_contiguous_target_partition(3, world_size=2, rank=2)


def test_realization_arrays_are_immutable_copies() -> None:
    requested = np.array([1, 3], dtype=np.int64)
    realization = make_validation_target_cell_masks(
        5,
        6,
        namespace="validation-v1",
        core_alias="SO2-C28",
        target_indices=requested,
    )
    requested[:] = 0

    assert np.array_equal(realization.target_indices, np.array([1, 3]))
    assert realization.target_indices.flags.writeable is False
    assert realization.mask.flags.writeable is False
    assert realization.masked_gene_counts.flags.writeable is False
    with pytest.raises(ValueError):
        realization.mask[0, 0] = ~realization.mask[0, 0]

"""Tests for the deterministic local sender-state null."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from spatial_benchmark.local_source_permutation import (
    LocalSourcePermutationError,
    build_macroblock_spatial_antipode_permutation,
    verify_local_source_permutation_receipt,
)
from spatial_benchmark.identifiers import canonical_sha256


def _edge_inputs(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(n_nodes, dtype=np.int64)
    receiver = (source + 1) % n_nodes
    edge_index = np.stack((source, receiver), axis=0)
    edge_attributes = np.arange(
        n_nodes * 3,
        dtype=np.float32,
    ).reshape(n_nodes, 3)
    return edge_index, edge_attributes


def test_antipode_permutation_is_repeatable_and_passes_every_qc() -> None:
    nodes_per_block = 100
    x = np.arange(nodes_per_block, dtype=np.float64) * 20.0
    coordinates = np.concatenate(
        (
            np.stack((x, np.zeros_like(x)), axis=1),
            np.stack((x, np.full_like(x, 5_000.0)), axis=1),
        ),
        axis=0,
    )
    macroblocks = np.asarray(
        ["north"] * nodes_per_block + ["south"] * nodes_per_block
    )
    edge_index, edge_attributes = _edge_inputs(len(coordinates))

    first = build_macroblock_spatial_antipode_permutation(
        coordinates,
        macroblocks,
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
    )
    second = build_macroblock_spatial_antipode_permutation(
        coordinates.copy(),
        macroblocks.copy(),
        local_edge_index=edge_index.copy(),
        local_edge_attributes=edge_attributes.copy(),
    )

    np.testing.assert_array_equal(
        first.source_index_by_node,
        second.source_index_by_node,
    )
    assert first.receipt == second.receipt
    assert (
        first.receipt["source_index_by_node_sha256"]
        == second.receipt["source_index_by_node_sha256"]
    )
    qc = first.receipt["qc"]
    assert qc["global_mapping_is_bijection"] is True
    assert qc["mapping_is_bijection_within_every_macroblock"] is True
    assert qc["node_mapping_changed_fraction"] == 1.0
    assert qc["node_displacement_above_threshold_fraction"] == 1.0
    assert (
        qc["effective_source_edge_slot_identity_changed_fraction"]
        == 1.0
    )
    assert qc["effective_permuted_source_equals_receiver_fraction"] == 0.0
    assert (
        qc[
            "effective_permuted_source_equals_receiver_"
            "affected_receiver_count"
        ]
        == 0
    )
    assert (
        qc[
            "effective_permuted_source_equals_receiver_"
            "affected_receiver_fraction"
        ]
        == 0.0
    )
    assert qc["local_edge_count_unchanged"] is True
    assert qc["local_receivers_unchanged"] is True
    assert qc["local_edge_attributes_bit_identical"] is True
    assert qc["gate_passed"] is True
    verify_local_source_permutation_receipt(first.receipt)
    construction = first.receipt["construction"]
    assert construction["covariance_precision"] == "IEEE_754_float64"
    assert construction["covariance_divisor"] == "macroblock_node_count"
    assert construction["tie_tolerance"] == (
        "64_times_float64_epsilon_times_"
        "max_abs_covariance_entry_or_one"
    )


def test_isotropic_group_uses_positive_x_axis_and_node_index_ties() -> None:
    coordinates = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    edge_index, edge_attributes = _edge_inputs(len(coordinates))

    result = build_macroblock_spatial_antipode_permutation(
        coordinates,
        np.asarray(["square"] * 4),
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
        enforce_qc=False,
    )

    np.testing.assert_array_equal(
        result.source_index_by_node,
        np.asarray([1, 0, 3, 2], dtype=np.int64),
    )
    assert result.receipt["qc"]["gate_passed"] is False


def test_effective_self_slots_disclose_slot_and_receiver_prevalence() -> None:
    n_nodes = 100
    coordinates = np.stack(
        (
            np.arange(n_nodes, dtype=np.float64) * 20.0,
            np.zeros(n_nodes, dtype=np.float64),
        ),
        axis=1,
    )
    source = np.arange(n_nodes, dtype=np.int64)
    # The spatial-antipode mapping shifts this fixture by exactly 50 nodes.
    # Make each observed receiver that effective source to audit disclosure.
    receiver = (source + n_nodes // 2) % n_nodes
    edge_index = np.stack((source, receiver), axis=0)
    edge_attributes = np.zeros((n_nodes, 3), dtype=np.float32)

    result = build_macroblock_spatial_antipode_permutation(
        coordinates,
        np.asarray(["one"] * n_nodes),
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
    )

    qc = result.receipt["qc"]
    assert qc["effective_permuted_source_equals_receiver_count"] == n_nodes
    assert qc["effective_permuted_source_equals_receiver_fraction"] == 1.0
    assert (
        qc[
            "effective_permuted_source_equals_receiver_"
            "affected_receiver_count"
        ]
        == n_nodes
    )
    assert (
        qc[
            "effective_permuted_source_equals_receiver_"
            "affected_receiver_fraction"
        ]
        == 1.0
    )
    verify_local_source_permutation_receipt(result.receipt)


def test_singleton_heavy_mapping_fails_before_execution() -> None:
    coordinates = np.stack(
        (
            np.arange(10, dtype=np.float64) * 100.0,
            np.zeros(10, dtype=np.float64),
        ),
        axis=1,
    )
    edge_index, edge_attributes = _edge_inputs(len(coordinates))

    with pytest.raises(LocalSourcePermutationError, match="failed QC"):
        build_macroblock_spatial_antipode_permutation(
            coordinates,
            np.asarray([f"block-{index}" for index in range(10)]),
            local_edge_index=edge_index,
            local_edge_attributes=edge_attributes,
        )


def test_receipt_checksum_and_threshold_tampering_is_rejected() -> None:
    coordinates = np.stack(
        (
            np.arange(100, dtype=np.float64) * 20.0,
            np.zeros(100, dtype=np.float64),
        ),
        axis=1,
    )
    edge_index, edge_attributes = _edge_inputs(len(coordinates))
    result = build_macroblock_spatial_antipode_permutation(
        coordinates,
        np.asarray(["one"] * len(coordinates)),
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
    )
    tampered = deepcopy(dict(result.receipt))
    tampered["qc"]["node_mapping_changed_fraction"] = 0.98

    with pytest.raises(
        LocalSourcePermutationError,
        match="checksum or schema",
    ):
        verify_local_source_permutation_receipt(tampered)


def test_resigned_construction_semantics_are_rejected() -> None:
    coordinates = np.stack(
        (
            np.arange(100, dtype=np.float64) * 20.0,
            np.zeros(100, dtype=np.float64),
        ),
        axis=1,
    )
    edge_index, edge_attributes = _edge_inputs(len(coordinates))
    result = build_macroblock_spatial_antipode_permutation(
        coordinates,
        np.asarray(["one"] * len(coordinates)),
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
    )
    tampered = deepcopy(dict(result.receipt))
    tampered["construction"]["uses_expression_labels_or_covariates"] = True
    tampered.pop("checksum")
    tampered["checksum"] = canonical_sha256(tampered)

    with pytest.raises(
        LocalSourcePermutationError,
        match="construction semantics",
    ):
        verify_local_source_permutation_receipt(tampered)

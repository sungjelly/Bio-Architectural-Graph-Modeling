"""Deterministic spatial-antipode source-state permutations.

The null leaves the observed local edge slots and edge attributes untouched.
For a local edge ``source -> receiver``, only the source node state is gathered
from ``source_index_by_node[source]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .identifiers import canonical_sha256
from .multiscale_hurdle_contract import (
    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM,
    LOCAL_SOURCE_PERMUTATION_SCHEMA,
)


class LocalSourcePermutationError(ValueError):
    """Raised when a sender-state permutation cannot satisfy its contract."""


@dataclass(frozen=True)
class LocalSourcePermutation:
    """A node-index permutation and its checksum-bound pre-GPU receipt."""

    source_index_by_node: np.ndarray
    receipt: Mapping[str, Any]


def _array_sha256(name: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _macroblock_keys(value: Any, *, n_nodes: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or len(labels) != n_nodes:
        raise LocalSourcePermutationError(
            "macroblock_ids must have one entry per node"
        )
    if labels.dtype.kind in "iu":
        normalized = np.asarray(
            [f"integer:{int(item)}" for item in labels],
            dtype=np.str_,
        )
    elif labels.dtype.kind in "US":
        normalized = np.asarray(
            [f"string:{str(item)}" for item in labels],
            dtype=np.str_,
        )
    else:
        raise LocalSourcePermutationError(
            "macroblock_ids must be integer or string values"
        )
    return normalized


def _coordinates(value: Any) -> np.ndarray:
    coordinates = np.asarray(value)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or coordinates.shape[0] == 0
        or coordinates.dtype.kind not in "iuf"
    ):
        raise LocalSourcePermutationError(
            "coordinates_um must be a nonempty numeric [nodes, 2] array"
        )
    result = np.ascontiguousarray(coordinates, dtype="<f8")
    if not np.isfinite(result).all():
        raise LocalSourcePermutationError("coordinates_um must be finite")
    return result


def _edge_inputs(
    edge_index: Any,
    edge_attributes: Any,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.asarray(edge_index)
    if (
        index.ndim != 2
        or index.shape[0] != 2
        or index.dtype.kind not in "iu"
        or index.shape[1] == 0
    ):
        raise LocalSourcePermutationError(
            "local_edge_index must be a nonempty integer [2, edges] array"
        )
    index = np.ascontiguousarray(index, dtype="<i8")
    if np.any(index < 0) or np.any(index >= n_nodes):
        raise LocalSourcePermutationError(
            "local_edge_index contains an out-of-range node"
        )
    attributes = np.asarray(edge_attributes)
    if (
        attributes.ndim != 2
        or attributes.shape[0] != index.shape[1]
        or attributes.shape[1] == 0
        or attributes.dtype.kind != "f"
    ):
        raise LocalSourcePermutationError(
            "local_edge_attributes must be floating [edges, attributes]"
        )
    attributes = np.ascontiguousarray(attributes)
    if not np.isfinite(attributes).all():
        raise LocalSourcePermutationError(
            "local_edge_attributes must be finite"
        )
    return index, attributes


def _principal_axis(centered: np.ndarray) -> np.ndarray:
    """Return a deterministic leading eigenvector of a 2x2 covariance."""

    if len(centered) < 2:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    xx = float(np.mean(centered[:, 0] * centered[:, 0]))
    xy = float(np.mean(centered[:, 0] * centered[:, 1]))
    yy = float(np.mean(centered[:, 1] * centered[:, 1]))
    discriminant = math.hypot(xx - yy, 2.0 * xy)
    scale = max(abs(xx), abs(xy), abs(yy), 1.0)
    if discriminant <= 64.0 * np.finfo(np.float64).eps * scale:
        return np.asarray([1.0, 0.0], dtype=np.float64)

    largest = 0.5 * (xx + yy + discriminant)
    candidate_x = np.asarray([xy, largest - xx], dtype=np.float64)
    candidate_y = np.asarray([largest - yy, xy], dtype=np.float64)
    vector = (
        candidate_x
        if float(candidate_x @ candidate_x)
        >= float(candidate_y @ candidate_y)
        else candidate_y
    )
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    vector = vector / norm
    dominant = 0 if abs(vector[0]) >= abs(vector[1]) else 1
    if vector[dominant] < 0.0:
        vector = -vector
    return vector


def _mapping_checksum(mapping: np.ndarray) -> str:
    return _array_sha256(
        "macroblock_spatial_antipode_source_index_by_node",
        np.asarray(mapping, dtype="<i8"),
    )


def build_macroblock_spatial_antipode_permutation(
    coordinates_um: Any,
    macroblock_ids: Any,
    *,
    local_edge_index: Any,
    local_edge_attributes: Any,
    enforce_qc: bool = True,
) -> LocalSourcePermutation:
    """Build the fixed macroblock-stratified local source-state null.

    Construction uses only coordinates, macroblock membership, and stable
    zero-based node row indices. Expression, labels, and node covariates are
    neither accepted nor inspected.
    """

    coordinates = _coordinates(coordinates_um)
    n_nodes = int(coordinates.shape[0])
    macroblocks = _macroblock_keys(macroblock_ids, n_nodes=n_nodes)
    edge_index, edge_attributes = _edge_inputs(
        local_edge_index,
        local_edge_attributes,
        n_nodes=n_nodes,
    )
    mapping = np.empty(n_nodes, dtype="<i8")
    singleton_count = 0
    failed_within_block_bijections = 0
    unique_macroblocks = np.unique(macroblocks)
    for macroblock in unique_macroblocks:
        nodes = np.flatnonzero(macroblocks == macroblock).astype(
            np.int64,
            copy=False,
        )
        if len(nodes) == 1:
            singleton_count += 1
        centered = coordinates[nodes] - coordinates[nodes].mean(
            axis=0,
            dtype=np.float64,
        )
        axis = _principal_axis(centered)
        projection = centered @ axis
        order = np.lexsort((nodes, projection))
        ordered_nodes = nodes[order]
        shift = len(ordered_nodes) // 2
        mapped_nodes = np.roll(ordered_nodes, -shift)
        mapping[ordered_nodes] = mapped_nodes
        if not np.array_equal(
            np.sort(mapped_nodes),
            np.sort(ordered_nodes),
        ):
            failed_within_block_bijections += 1

    expected_nodes = np.arange(n_nodes, dtype="<i8")
    global_bijection = bool(
        np.array_equal(np.sort(mapping), expected_nodes)
    )
    within_block_bijection = failed_within_block_bijections == 0
    changed = mapping != expected_nodes
    changed_count = int(np.count_nonzero(changed))
    changed_fraction = float(changed_count / n_nodes)
    displacements = np.linalg.norm(
        coordinates[mapping] - coordinates,
        axis=1,
    )
    displaced = (
        displacements
        > LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
    )
    displaced_count = int(np.count_nonzero(displaced))
    displaced_fraction = float(displaced_count / n_nodes)

    source = edge_index[0]
    receiver = edge_index[1]
    effective_source = mapping[source]
    changed_edge_slots = effective_source != source
    edge_count = int(edge_index.shape[1])
    changed_edge_slot_count = int(np.count_nonzero(changed_edge_slots))
    changed_edge_slot_fraction = float(
        changed_edge_slot_count / edge_count
    )
    effective_source_equals_receiver = effective_source == receiver
    effective_source_equals_receiver_count = int(
        np.count_nonzero(effective_source_equals_receiver)
    )
    effective_source_equals_receiver_fraction = float(
        effective_source_equals_receiver_count / edge_count
    )
    affected_receivers = np.unique(
        receiver[effective_source_equals_receiver]
    )
    affected_receiver_count = int(len(affected_receivers))
    affected_receiver_fraction = float(affected_receiver_count / n_nodes)

    edge_index_sha256 = _array_sha256(
        "observed_local_edge_index",
        edge_index,
    )
    receiver_sha256 = _array_sha256(
        "observed_local_receiver_index",
        receiver,
    )
    edge_attributes_sha256 = _array_sha256(
        "observed_local_edge_attributes",
        edge_attributes,
    )
    qc = {
        "global_mapping_is_bijection": global_bijection,
        "mapping_is_bijection_within_every_macroblock": (
            within_block_bijection
        ),
        "macroblocks_failed_bijection_count": (
            failed_within_block_bijections
        ),
        "node_mapping_changed_count": changed_count,
        "node_mapping_changed_fraction": changed_fraction,
        "minimum_node_mapping_changed_fraction": (
            LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
        ),
        "node_displacement_above_threshold_count": displaced_count,
        "node_displacement_above_threshold_fraction": displaced_fraction,
        "displacement_threshold_um": (
            LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
        ),
        "minimum_node_displacement_above_threshold_fraction": (
            LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
        ),
        "local_edge_count": edge_count,
        "local_edge_count_unchanged": True,
        "local_receivers_unchanged": True,
        "local_edge_attributes_bit_identical": True,
        "effective_source_edge_slot_identity_changed_count": (
            changed_edge_slot_count
        ),
        "effective_source_edge_slot_identity_changed_fraction": (
            changed_edge_slot_fraction
        ),
        "minimum_local_edge_slot_sender_identity_changed_fraction": (
            LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
        ),
        "effective_permuted_source_equals_receiver_count": (
            effective_source_equals_receiver_count
        ),
        "effective_permuted_source_equals_receiver_fraction": (
            effective_source_equals_receiver_fraction
        ),
        "effective_permuted_source_equals_receiver_affected_receiver_count": (
            affected_receiver_count
        ),
        "effective_permuted_source_equals_receiver_affected_receiver_fraction": (
            affected_receiver_fraction
        ),
    }
    gate_passed = bool(
        global_bijection
        and within_block_bijection
        and changed_fraction
        >= LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
        and displaced_fraction
        >= LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
        and changed_edge_slot_fraction
        >= LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
        and qc["local_edge_count_unchanged"]
        and qc["local_receivers_unchanged"]
        and qc["local_edge_attributes_bit_identical"]
    )
    qc["gate_passed"] = gate_passed
    receipt: dict[str, Any] = {
        "schema": LOCAL_SOURCE_PERMUTATION_SCHEMA,
        "construction": {
            "grouping": "separately_within_each_macroblock",
            "coordinate_precision": "float64",
            "covariance_precision": "IEEE_754_float64",
            "covariance_divisor": "macroblock_node_count",
            "axis": (
                "deterministic_largest_eigenvector_of_centered_"
                "2d_coordinate_covariance"
            ),
            "eigenvalue_tie_discriminant": (
                "hypot(cov_xx_minus_cov_yy, 2_times_cov_xy)"
            ),
            "tie_tolerance": (
                "64_times_float64_epsilon_times_"
                "max_abs_covariance_entry_or_one"
            ),
            "axis_sign": (
                "largest_absolute_loading_positive_with_x_tie_break"
            ),
            "degenerate_axis": "positive_x",
            "ordering": (
                "stable_lexicographic_projection_then_node_row_index"
            ),
            "mapping": (
                "forward_circular_shift_by_floor_group_size_divided_by_two"
            ),
            "singleton_policy": "fixed_and_explicitly_counted",
            "uses_expression_labels_or_covariates": False,
        },
        "n_nodes": n_nodes,
        "n_macroblocks": int(len(unique_macroblocks)),
        "singleton_macroblock_count": singleton_count,
        "inputs": {
            "coordinates_um_sha256": _array_sha256(
                "coordinates_um_float64",
                coordinates,
            ),
            "macroblock_ids_sha256": hashlib.sha256(
                json.dumps(
                    macroblocks.tolist(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "stable_node_row_index": True,
            "local_edge_index_sha256": edge_index_sha256,
            "local_receiver_index_sha256": receiver_sha256,
            "local_edge_attributes_sha256": edge_attributes_sha256,
        },
        "execution": {
            "self_branch_source_states": "observed_unpermuted",
            "regional_branch_source_states": "observed_unpermuted",
            "local_branch_topology": "observed_true_local",
            "local_branch_receiver_identities": "observed_unpermuted",
            "local_branch_edge_attributes": (
                "observed_true_local_unchanged"
            ),
            "local_branch_sender_states": (
                "gather_from_permuted_source_identity"
            ),
        },
        "source_index_by_node_sha256": _mapping_checksum(mapping),
        "qc": qc,
    }
    receipt["checksum"] = canonical_sha256(receipt)
    if gate_passed:
        verify_local_source_permutation_receipt(receipt)
    if enforce_qc and not gate_passed:
        raise LocalSourcePermutationError(
            "macroblock spatial-antipode sender permutation failed QC"
        )
    mapping.setflags(write=False)
    return LocalSourcePermutation(
        source_index_by_node=mapping,
        receipt=receipt,
    )


def verify_local_source_permutation_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Validate the immutable schema and all frozen pre-GPU thresholds."""

    expected_top_fields = {
        "schema",
        "construction",
        "n_nodes",
        "n_macroblocks",
        "singleton_macroblock_count",
        "inputs",
        "execution",
        "source_index_by_node_sha256",
        "qc",
        "checksum",
    }
    if set(receipt) != expected_top_fields:
        raise LocalSourcePermutationError(
            "local source permutation receipt fields are not exact"
        )
    payload = dict(receipt)
    checksum = payload.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or checksum != canonical_sha256(payload)
        or receipt.get("schema") != LOCAL_SOURCE_PERMUTATION_SCHEMA
    ):
        raise LocalSourcePermutationError(
            "local source permutation receipt checksum or schema is invalid"
        )
    construction = receipt.get("construction")
    expected_construction = {
        "grouping": "separately_within_each_macroblock",
        "coordinate_precision": "float64",
        "covariance_precision": "IEEE_754_float64",
        "covariance_divisor": "macroblock_node_count",
        "axis": (
            "deterministic_largest_eigenvector_of_centered_"
            "2d_coordinate_covariance"
        ),
        "eigenvalue_tie_discriminant": (
            "hypot(cov_xx_minus_cov_yy, 2_times_cov_xy)"
        ),
        "tie_tolerance": (
            "64_times_float64_epsilon_times_"
            "max_abs_covariance_entry_or_one"
        ),
        "axis_sign": (
            "largest_absolute_loading_positive_with_x_tie_break"
        ),
        "degenerate_axis": "positive_x",
        "ordering": (
            "stable_lexicographic_projection_then_node_row_index"
        ),
        "mapping": (
            "forward_circular_shift_by_floor_group_size_divided_by_two"
        ),
        "singleton_policy": "fixed_and_explicitly_counted",
        "uses_expression_labels_or_covariates": False,
    }
    if construction != expected_construction:
        raise LocalSourcePermutationError(
            "local source permutation construction semantics changed"
        )
    execution = receipt.get("execution")
    expected_execution = {
        "self_branch_source_states": "observed_unpermuted",
        "regional_branch_source_states": "observed_unpermuted",
        "local_branch_topology": "observed_true_local",
        "local_branch_receiver_identities": "observed_unpermuted",
        "local_branch_edge_attributes": "observed_true_local_unchanged",
        "local_branch_sender_states": (
            "gather_from_permuted_source_identity"
        ),
    }
    if execution != expected_execution:
        raise LocalSourcePermutationError(
            "local source permutation execution semantics changed"
        )
    counts: dict[str, int] = {}
    for field, minimum in (
        ("n_nodes", 1),
        ("n_macroblocks", 1),
        ("singleton_macroblock_count", 0),
    ):
        value = receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise LocalSourcePermutationError(
                f"local source permutation {field} is invalid"
            )
        counts[field] = value
    if (
        counts["n_macroblocks"] > counts["n_nodes"]
        or counts["singleton_macroblock_count"]
        > counts["n_macroblocks"]
    ):
        raise LocalSourcePermutationError(
            "local source permutation counts are inconsistent"
        )
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "coordinates_um_sha256",
        "macroblock_ids_sha256",
        "stable_node_row_index",
        "local_edge_index_sha256",
        "local_receiver_index_sha256",
        "local_edge_attributes_sha256",
    }:
        raise LocalSourcePermutationError(
            "local source permutation input identities are not exact"
        )
    digests = [
        inputs.get("coordinates_um_sha256"),
        inputs.get("macroblock_ids_sha256"),
        inputs.get("local_edge_index_sha256"),
        inputs.get("local_receiver_index_sha256"),
        inputs.get("local_edge_attributes_sha256"),
        receipt.get("source_index_by_node_sha256"),
    ]
    if (
        inputs.get("stable_node_row_index") is not True
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        )
    ):
        raise LocalSourcePermutationError(
            "local source permutation input checksum is invalid"
        )
    qc = receipt.get("qc")
    if not isinstance(qc, Mapping):
        raise LocalSourcePermutationError(
            "local source permutation receipt lacks QC"
        )
    expected_qc_fields = {
        "global_mapping_is_bijection",
        "mapping_is_bijection_within_every_macroblock",
        "macroblocks_failed_bijection_count",
        "node_mapping_changed_count",
        "node_mapping_changed_fraction",
        "minimum_node_mapping_changed_fraction",
        "node_displacement_above_threshold_count",
        "node_displacement_above_threshold_fraction",
        "displacement_threshold_um",
        "minimum_node_displacement_above_threshold_fraction",
        "local_edge_count",
        "local_edge_count_unchanged",
        "local_receivers_unchanged",
        "local_edge_attributes_bit_identical",
        "effective_source_edge_slot_identity_changed_count",
        "effective_source_edge_slot_identity_changed_fraction",
        "minimum_local_edge_slot_sender_identity_changed_fraction",
        "effective_permuted_source_equals_receiver_count",
        "effective_permuted_source_equals_receiver_fraction",
        "effective_permuted_source_equals_receiver_affected_receiver_count",
        "effective_permuted_source_equals_receiver_affected_receiver_fraction",
        "gate_passed",
    }
    if set(qc) != expected_qc_fields:
        raise LocalSourcePermutationError(
            "local source permutation QC fields are not exact"
        )
    expected = {
        "minimum_node_mapping_changed_fraction": (
            LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
        ),
        "displacement_threshold_um": (
            LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
        ),
        "minimum_node_displacement_above_threshold_fraction": (
            LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
        ),
        "minimum_local_edge_slot_sender_identity_changed_fraction": (
            LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
        ),
        "global_mapping_is_bijection": True,
        "mapping_is_bijection_within_every_macroblock": True,
        "macroblocks_failed_bijection_count": 0,
        "local_edge_count_unchanged": True,
        "local_receivers_unchanged": True,
        "local_edge_attributes_bit_identical": True,
        "gate_passed": True,
    }
    if any(qc.get(key) != value for key, value in expected.items()):
        raise LocalSourcePermutationError(
            "local source permutation receipt failed a frozen QC identity"
        )
    count_fields = {
        "node_mapping_changed_count": counts["n_nodes"],
        "node_displacement_above_threshold_count": counts["n_nodes"],
        "local_edge_count": None,
        "effective_source_edge_slot_identity_changed_count": None,
        "effective_permuted_source_equals_receiver_count": None,
        "effective_permuted_source_equals_receiver_affected_receiver_count": (
            counts["n_nodes"]
        ),
    }
    observed_counts: dict[str, int] = {}
    for field, maximum in count_fields.items():
        value = qc.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (maximum is not None and value > maximum)
        ):
            raise LocalSourcePermutationError(
                f"local source permutation {field} is invalid"
            )
        observed_counts[field] = value
    edge_count = observed_counts["local_edge_count"]
    if edge_count <= 0 or any(
        observed_counts[field] > edge_count
        for field in (
            "effective_source_edge_slot_identity_changed_count",
            "effective_permuted_source_equals_receiver_count",
        )
    ):
        raise LocalSourcePermutationError(
            "local source permutation edge counts are inconsistent"
        )
    for field, minimum in (
        (
            "node_mapping_changed_fraction",
            LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
        ),
        (
            "node_displacement_above_threshold_fraction",
            LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
        ),
        (
            "effective_source_edge_slot_identity_changed_fraction",
            LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
        ),
    ):
        value = qc.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= 1.0
        ):
            raise LocalSourcePermutationError(
                f"local source permutation {field} failed its threshold"
            )
    fraction_pairs = (
        (
            "node_mapping_changed_fraction",
            observed_counts["node_mapping_changed_count"],
            counts["n_nodes"],
        ),
        (
            "node_displacement_above_threshold_fraction",
            observed_counts["node_displacement_above_threshold_count"],
            counts["n_nodes"],
        ),
        (
            "effective_source_edge_slot_identity_changed_fraction",
            observed_counts[
                "effective_source_edge_slot_identity_changed_count"
            ],
            edge_count,
        ),
    )
    for field, numerator, denominator in fraction_pairs:
        if not math.isclose(
            float(qc[field]),
            numerator / denominator,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise LocalSourcePermutationError(
                f"local source permutation {field} disagrees with its count"
            )
    disclosed = qc.get(
        "effective_permuted_source_equals_receiver_fraction"
    )
    if (
        isinstance(disclosed, bool)
        or not isinstance(disclosed, (int, float))
        or not math.isfinite(float(disclosed))
        or not 0.0 <= float(disclosed) <= 1.0
    ):
        raise LocalSourcePermutationError(
            "effective permuted-source/receiver fraction is not disclosed"
        )
    if not math.isclose(
        float(disclosed),
        observed_counts[
            "effective_permuted_source_equals_receiver_count"
        ]
        / edge_count,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise LocalSourcePermutationError(
            "effective permuted-source/receiver fraction disagrees with count"
        )
    affected_fraction = qc.get(
        "effective_permuted_source_equals_receiver_affected_receiver_fraction"
    )
    if (
        isinstance(affected_fraction, bool)
        or not isinstance(affected_fraction, (int, float))
        or not math.isfinite(float(affected_fraction))
        or not 0.0 <= float(affected_fraction) <= 1.0
        or not math.isclose(
            float(affected_fraction),
            observed_counts[
                "effective_permuted_source_equals_receiver_affected_"
                "receiver_count"
            ]
            / counts["n_nodes"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise LocalSourcePermutationError(
            "affected receiver fraction is missing or inconsistent"
        )


__all__ = [
    "LocalSourcePermutation",
    "LocalSourcePermutationError",
    "build_macroblock_spatial_antipode_permutation",
    "verify_local_source_permutation_receipt",
]

from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark import myjju_gradient_reduction as reduction
from spatial_benchmark.myjju_gradient_audit import (
    FROZEN_MARKER_GENES,
    FROZEN_SOURCE_SELECTED_PAIRS,
    FROZEN_TARGET_GENES,
)


def test_mean_huber_and_relative_improvement() -> None:
    assert reduction.mean_huber([0.0, 2.0], [0.0, 0.0]) == pytest.approx(
        0.75
    )
    assert reduction.relative_improvement(2.0, 1.5) == pytest.approx(0.25)
    with pytest.raises(reduction.GradientReductionError):
        reduction.relative_improvement(0.0, 0.0)


def test_structure_statistic_uses_locked_directed_pairs() -> None:
    matrix = np.ones((39, 39), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    for target, sources in FROZEN_SOURCE_SELECTED_PAIRS.items():
        for source in sources:
            matrix[positions[target], positions[source]] = 5.0

    result = reduction.structure_statistic(matrix)

    assert result["locked_median_absolute_gradient"] == 5.0
    assert result["nonlocked_median_absolute_gradient"] == 1.0
    assert result["S"] == 5.0


def _gradient_matrix(*, locked_value: float, scale: float = 1.0) -> np.ndarray:
    row = np.arange(39, dtype=np.float32)[:, None]
    column = np.arange(39, dtype=np.float32)[None, :]
    matrix = (1.0 + row * 0.001 + column * 0.00001) * scale
    np.fill_diagonal(matrix, 0.0)
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    for target, sources in FROZEN_SOURCE_SELECTED_PAIRS.items():
        for source in sources:
            matrix[positions[target], positions[source]] = locked_value * scale
    return matrix.astype(np.float32)


def _synthetic_seed_arrays(seed: int) -> dict[str, np.ndarray]:
    node_count = 100
    offsets = np.arange(0, node_count + 1, 10, dtype=np.int64)
    truth = np.empty((node_count, 5), dtype=np.float32)
    for target in range(5):
        truth[:, target] = (
            np.arange(node_count, dtype=np.float32) % 10
        ) / 5.0 + target * 0.1
    mask = np.ones((node_count, 3, 5), dtype=np.bool_)
    observed_prediction = truth[:, None, :].repeat(3, axis=1)
    permuted_prediction = observed_prediction + np.float32(1.0)
    gene_mean = np.zeros((10, 5), dtype=np.float32)

    observed_matrix = _gradient_matrix(
        locked_value=10.0, scale=1.0 + seed * 0.01
    )
    permuted_matrix = _gradient_matrix(
        locked_value=1.0, scale=1.0 + seed * 0.01
    )
    observed = np.repeat(observed_matrix[None], 10, axis=0)
    permuted = np.repeat(permuted_matrix[None], 10, axis=0)
    target_marker_positions = [
        FROZEN_MARKER_GENES.index(gene) for gene in FROZEN_TARGET_GENES
    ]
    locked_rows = np.empty((10, 3, 5, 39), dtype=np.float32)
    for replicate in range(3):
        locked_rows[:, replicate] = (
            observed[:, target_marker_positions] * (1.0 + replicate * 0.001)
        )

    faithfulness = np.empty((10, 2, 5, 39), dtype=np.float32)
    for core in range(10):
        for scale in range(2):
            for target in range(5):
                for source in range(39):
                    sign = -1.0 if (core + target + source) % 2 else 1.0
                    faithfulness[core, scale, target, source] = sign * (
                        0.001
                        + core * 0.0001
                        + scale * 0.00001
                        + target * 0.000001
                        + source * 0.0000001
                    )

    same_signed = np.full((10, 5, 39), 0.25, dtype=np.float32)
    other_signed = np.full((10, 5, 39), 0.75, dtype=np.float32)
    same_l1 = np.full((10, 5, 39), 0.5, dtype=np.float32)
    other_l1 = np.full((10, 5, 39), 1.5, dtype=np.float32)
    prevalence = np.repeat(
        np.linspace(0.01, 0.99, 39, dtype=np.float32)[None], 10, axis=0
    )
    expression = np.repeat(
        np.linspace(0.1, 3.9, 39, dtype=np.float32)[None], 10, axis=0
    )
    correlation = np.empty((10, 5, 39), dtype=np.float32)
    for target in range(5):
        correlation[:, target] = np.linspace(
            0.0 + target * 0.001,
            0.9 + target * 0.001,
            39,
            dtype=np.float32,
        )
    return {
        "selected_truth": truth,
        "selected_mask": mask,
        "selected_prediction_observed": observed_prediction.copy(),
        "selected_prediction_permuted": permuted_prediction.copy(),
        "core_offsets": offsets,
        "per_core_gene_mean": gene_mean,
        "source_nonzero_prevalence": prevalence,
        "source_mean_expression": expression,
        "source_population_sd": np.ones((10, 39), dtype=np.float32),
        "source_q01": np.zeros((10, 39), dtype=np.float32),
        "source_q99": np.ones((10, 39), dtype=np.float32) * 5.0,
        "abs_target_source_raw_pearson": correlation,
        "masked_rep0_observed_signed": observed,
        "masked_rep0_permuted_signed": permuted,
        "masked_locked_observed_signed": locked_rows,
        "source_unmasked_signed": observed.copy(),
        "faithfulness_predicted": faithfulness.copy(),
        "faithfulness_actual": faithfulness.copy(),
        "randomized_rep0_observed_signed": permuted_matrix.copy(),
        "decomposition_same_signed": same_signed,
        "decomposition_other_signed": other_signed,
        "decomposition_same_l1": same_l1,
        "decomposition_other_l1": other_l1,
    }


def test_reduce_gradient_audit_passes_planted_computational_controls() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]

    result = reduction.reduce_gradient_audit(shards)

    assert result["candidate_set_computational_precursors_supported"] is True
    assert result["mechanism_validation_available"] is False
    assert result["mechanism_claim_supported"] is False
    assert result["eligible_targets"] == list(FROZEN_TARGET_GENES)
    assert result["graph_gradient_null"]["passing_core_count"] == 10
    assert result["matched_pair_null"]["upper_tail_p"] == pytest.approx(
        1 / 10_001
    )
    assert all(
        row["pass"]
        for row in result["bounded_faithfulness"][
            "primary_offdiagonal_rows"
        ]
    )
    source_reproduction = result["source_style_reproduction"]
    directed = np.asarray(
        source_reproduction[
            "equal_core_seven_seed_signed_directed_matrix"
        ]
    )
    symmetric = np.asarray(
        source_reproduction[
            "equal_core_seven_seed_published_symmetric_absolute_matrix"
        ]
    )
    assert source_reproduction["marker_gene_order"] == list(
        FROZEN_MARKER_GENES
    )
    assert directed.shape == (39, 39)
    assert symmetric.shape == (39, 39)
    np.testing.assert_allclose(
        symmetric,
        0.5 * (np.abs(directed) + np.abs(directed.T)),
    )


def _assert_candidate_qualification_rejected(result: dict[str, object]) -> None:
    assert result["candidate_set_computational_precursors_supported"] is False
    assert result["mechanism_validation_available"] is False
    assert result["mechanism_claim_supported"] is False
    assert (
        result["maximum_defensible_claim"]
        == "no_claim_beyond_reported_model_behavior"
    )


def _assert_common_noneligibility_gates_pass(
    result: dict[str, object],
    *,
    skip: str,
) -> None:
    if skip != "seed":
        assert result["seed_rank_stability"]["pass"] is True
    if skip != "faithfulness":
        assert result["bounded_faithfulness"]["pass"] is True
    if skip != "graph":
        assert result["graph_gradient_null"]["pass"] is True
    if skip != "randomization":
        assert result["parameter_randomization"]["pass"] is True
    if skip != "matched":
        assert result["matched_pair_null"]["pass"] is True


def test_predictive_pass_does_not_override_graph_use_failure() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    for arrays in shards:
        truth = arrays["selected_truth"][:, None, :]
        arrays["selected_prediction_observed"][:] = truth + np.float32(0.1)
        arrays["selected_prediction_permuted"][:] = truth + np.float32(0.1005)

    result = reduction.reduce_gradient_audit(shards)

    target_rows = result["eligibility"]["target_rows"]
    assert all(row["predictive_gate_pass"] for row in target_rows)
    assert not any(row["graph_use_gate_pass"] for row in target_rows)
    assert result["eligible_targets"] == []
    _assert_common_noneligibility_gates_pass(result, skip="")
    _assert_candidate_qualification_rejected(result)


def test_mask_stability_failure_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    for arrays in shards:
        arrays["masked_locked_observed_signed"][:, 1:] = 0.0

    result = reduction.reduce_gradient_audit(shards)

    assert not any(
        row["pass"] for row in result["mask_rank_stability"]["target_rows"]
    )
    assert result["signed_pair_stability"]["passing_pair_count"] == 19
    _assert_common_noneligibility_gates_pass(result, skip="")
    _assert_candidate_qualification_rejected(result)


def test_signed_pair_instability_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    target_positions = [
        FROZEN_MARKER_GENES.index(gene) for gene in FROZEN_TARGET_GENES
    ]
    for seed, arrays in enumerate(shards):
        direction = np.float32(1.0 if seed < 4 else -1.0)
        arrays["masked_rep0_observed_signed"] *= direction
        selected_rows = arrays["masked_rep0_observed_signed"][
            :, target_positions
        ]
        arrays["masked_locked_observed_signed"][:, 0] = selected_rows
        arrays["masked_locked_observed_signed"][:, 1] = (
            selected_rows * np.float32(1.001)
        )
        arrays["masked_locked_observed_signed"][:, 2] = (
            selected_rows * np.float32(0.999)
        )

    result = reduction.reduce_gradient_audit(shards)

    assert result["signed_pair_stability"]["passing_pair_count"] == 0
    assert all(
        row["pass"] for row in result["mask_rank_stability"]["target_rows"]
    )
    _assert_common_noneligibility_gates_pass(result, skip="")
    _assert_candidate_qualification_rejected(result)


def test_one_failed_faithfulness_scale_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    for arrays in shards:
        arrays["faithfulness_actual"][:, 1] *= np.float32(-1.0)

    result = reduction.reduce_gradient_audit(shards)

    scale_rows = result["bounded_faithfulness"]["primary_offdiagonal_rows"]
    assert [row["pass"] for row in scale_rows] == [True, False]
    _assert_common_noneligibility_gates_pass(result, skip="faithfulness")
    _assert_candidate_qualification_rejected(result)


def test_graph_structure_null_failure_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    for arrays in shards:
        arrays["masked_rep0_permuted_signed"][:] = arrays[
            "masked_rep0_observed_signed"
        ]

    result = reduction.reduce_gradient_audit(shards)

    assert result["graph_gradient_null"]["pass"] is False
    assert result["parameter_randomization"]["pass"] is True
    assert result["matched_pair_null"]["pass"] is True
    _assert_common_noneligibility_gates_pass(result, skip="graph")
    _assert_candidate_qualification_rejected(result)


def test_parameter_randomization_failure_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    for seed, arrays in enumerate(shards):
        arrays["randomized_rep0_observed_signed"][:] = _gradient_matrix(
            locked_value=20.0,
            scale=1.0 + seed * 0.01,
        )

    result = reduction.reduce_gradient_audit(shards)

    assert result["parameter_randomization"]["pass"] is False
    assert result["graph_gradient_null"]["pass"] is True
    assert result["matched_pair_null"]["pass"] is True
    _assert_common_noneligibility_gates_pass(result, skip="randomization")
    _assert_candidate_qualification_rejected(result)


def test_matched_pair_null_failure_rejects_candidate_qualification() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    target_positions = [
        FROZEN_MARKER_GENES.index(gene) for gene in FROZEN_TARGET_GENES
    ]
    for seed, arrays in enumerate(shards):
        scale = 1.0 + seed * 0.01
        observed = _gradient_matrix(locked_value=0.01, scale=scale)
        permuted = _gradient_matrix(locked_value=0.001, scale=scale)
        arrays["masked_rep0_observed_signed"][:] = observed[None]
        arrays["masked_rep0_permuted_signed"][:] = permuted[None]
        for replicate, factor in enumerate((1.0, 1.001, 0.999)):
            arrays["masked_locked_observed_signed"][:, replicate] = (
                observed[target_positions] * np.float32(factor)
            )
        arrays["randomized_rep0_observed_signed"][:] = _gradient_matrix(
            locked_value=0.001,
            scale=scale,
        )

    result = reduction.reduce_gradient_audit(shards)

    assert result["matched_pair_null"]["pass"] is False
    assert result["matched_pair_null"]["upper_tail_p"] == 1.0
    assert result["graph_gradient_null"]["pass"] is True
    assert result["parameter_randomization"]["pass"] is True
    _assert_common_noneligibility_gates_pass(result, skip="matched")
    _assert_candidate_qualification_rejected(result)


def test_reduce_gradient_audit_rejects_cross_seed_row_misalignment() -> None:
    shards = [_synthetic_seed_arrays(seed) for seed in range(7)]
    shards[3]["selected_truth"] = shards[3]["selected_truth"].copy()
    shards[3]["selected_truth"][[0, 1]] = shards[3]["selected_truth"][[1, 0]]

    with pytest.raises(
        reduction.GradientReductionError,
        match="selected_truth is not exactly invariant",
    ):
        reduction.reduce_gradient_audit(shards)


def test_failed_signed_pair_retains_adverse_prevalence_counts() -> None:
    masked = np.zeros((7, 10, 39, 39), dtype=np.float64)
    target, source = reduction.FROZEN_LOCKED_DIRECTED_PAIRS[0]
    positions = {
        gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)
    }
    values = masked[:, :, positions[target], positions[source]]
    values[:6, :7] = 1.0
    values[:6, 7:] = -0.1
    values[6, :] = -1.0

    result = reduction._signed_pair_stability(masked)
    row = result["pair_rows"][0]

    assert row["pass"] is False
    assert row["direction"] == 0
    assert row["positive_seed_count"] == 6
    assert row["positive_core_count"] == 7
    assert row["negative_seed_count"] == 1
    assert row["negative_core_count"] == 3

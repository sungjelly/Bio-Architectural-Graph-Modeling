from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from spatial_benchmark.categorical_sensitivity import (
    BOOTSTRAP_REPLICATES,
    CategoricalSensitivityError,
    PROBES_PER_MASK,
    PairEstimate,
    PairMetrics,
    PercentileInterval,
    ProbeSufficientStatistics,
    _forward_from_encoder_preactivation,
    categorical_encoder_preactivation,
    center_output_logits,
    estimate_pair,
    evaluate_operational_match,
    locked_protocol_record,
    locked_seed,
    make_rademacher_probe,
    materialize_tangent_input_vjp,
    metrics_from_sufficient_statistics,
    multi_tangent_vjp_statistics,
    observed_token_projection_weights,
    paired_tangent_vjp_statistics,
    preactivation_probe_vjp,
    select_locked_whole_node_masks,
    whole_node_targets,
)
from spatial_benchmark.masking import MaskSpec, create_fixed_mask_bundle
from spatial_benchmark.tokenized_g2 import (
    NUM_COUNT_TOKENS,
    TokenizedReceiverChunkedEdgeConditionedGATv2,
)


def _model(*, hidden_dim: int = 8) -> TokenizedReceiverChunkedEdgeConditionedGATv2:
    torch.manual_seed(17 + hidden_dim)
    return TokenizedReceiverChunkedEdgeConditionedGATv2(
        num_genes=3,
        edge_attribute_dim=2,
        node_covariate_dim=2,
        hidden_dim=hidden_dim,
        attention_heads=2,
        graph_layers=2,
        ffn_dim=hidden_dim,
        decoder_dim=hidden_dim,
        edge_hidden_dim=4,
        edge_embedding_dim=4,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        activation_checkpointing=False,
    )


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    # Every node has two incoming non-self edges.
    edge_index = torch.tensor(
        [
            [1, 2, 0, 2, 0, 3, 1, 2],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    edge_attributes = torch.linspace(
        -1.0,
        1.0,
        steps=edge_index.shape[1] * 2,
        dtype=torch.float32,
    ).reshape(-1, 2)
    return edge_index, edge_attributes


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expression = torch.tensor(
        [
            [0, 1, 2],
            [3, 0, 1],
            [2, 3, 0],
            [1, 2, 3],
        ],
        dtype=torch.float32,
    )
    mask = torch.zeros_like(expression, dtype=torch.bool)
    mask[0] = True
    covariates = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.5, 0.2],
            [0.0, 0.7],
        ],
        dtype=torch.float32,
    )
    return expression, mask, covariates


def _pair_estimate(
    *,
    cosine: float,
    discrepancy: float,
    ratio: float,
    cosine_interval: tuple[float, float] | None = None,
    discrepancy_interval: tuple[float, float] | None = None,
    ratio_interval: tuple[float, float] | None = None,
    reference_label: str = "A",
    candidate_label: str = "B",
) -> PairEstimate:
    cosine_interval = cosine_interval or (cosine, cosine)
    discrepancy_interval = discrepancy_interval or (
        discrepancy,
        discrepancy,
    )
    ratio_interval = ratio_interval or (ratio, ratio)
    return PairEstimate(
        reference_label=reference_label,
        candidate_label=candidate_label,
        point=PairMetrics(
            cosine=cosine,
            relative_discrepancy=discrepancy,
            norm_ratio=ratio,
            reference_squared_norm=1.0,
            candidate_squared_norm=ratio * ratio,
            cross_inner_product=cosine * ratio,
        ),
        cosine_interval=PercentileInterval(*cosine_interval),
        relative_discrepancy_interval=PercentileInterval(
            *discrepancy_interval
        ),
        norm_ratio_interval=PercentileInterval(*ratio_interval),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=locked_seed(20260726, "bootstrap"),
    )


def test_locked_seed_and_rademacher_probe_are_stable() -> None:
    assert locked_seed(20260726, "mask-node-0", 7) == 3_564_455_979_410_831_329
    first, first_record = make_rademacher_probe(
        (2, 3, 4),
        mask_entry_id="mask-node-0",
        probe_index=7,
    )
    second, second_record = make_rademacher_probe(
        (2, 3, 4),
        mask_entry_id="mask-node-0",
        probe_index=7,
    )

    assert torch.equal(first, second)
    assert set(first.unique().tolist()) <= {-1.0, 1.0}
    assert first_record == second_record
    assert first_record["seed"] == 3_564_455_979_410_831_329
    assert (
        first_record["checksum_sha256"]
        == "de2d092be5b0e0f0df3ae4fb82de6bd4dbed47caa9546ba486ebfda2def27bd5"
    )


def test_center_output_logits_centers_each_gene() -> None:
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    centered = center_output_logits(logits)

    torch.testing.assert_close(
        centered.mean(dim=-1),
        torch.zeros((2, 3)),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(
        centered[0, 0],
        torch.tensor([-1.5, -0.5, 0.5, 1.5]),
    )


def test_whole_node_targets_rejects_partial_masks() -> None:
    mask = torch.zeros((4, 3), dtype=torch.bool)
    mask[1] = True
    mask[3] = True
    assert whole_node_targets(mask).tolist() == [1, 3]

    mask[0, 0] = True
    with pytest.raises(ValueError, match="whole-node"):
        whole_node_targets(mask)


def test_select_locked_masks_extracts_three_numpy_backed_node_masks() -> None:
    coordinates = np.stack(
        (np.arange(10, dtype=np.float64), np.zeros(10)),
        axis=1,
    )
    bundle = create_fixed_mask_bundle(
        {"fit": coordinates},
        3,
        [
            MaskSpec(mode="partial"),
            MaskSpec(mode="node", node_rate=0.2),
            MaskSpec(mode="block", block_node_rate=0.2),
        ],
        replicates=3,
        base_seed=19,
    )

    selected = select_locked_whole_node_masks(bundle)
    assert len(selected) == 3
    assert all(mask.dtype == torch.bool for _, mask in selected)
    assert all(whole_node_targets(mask).numel() == 2 for _, mask in selected)


def test_streamed_pair_statistics_equal_materialized_tangent_vjps() -> None:
    generator = torch.Generator().manual_seed(31)
    reference_gradient = torch.randn((7, 5), generator=generator)
    candidate_gradient = torch.randn((7, 9), generator=generator)
    reference_weights = torch.randn((4, 5, 6), generator=generator)
    candidate_weights = torch.randn((4, 9, 6), generator=generator)
    observed = torch.rand((7, 6), generator=generator) > 0.35

    reference = materialize_tangent_input_vjp(
        reference_gradient,
        reference_weights,
        observed,
    ).to(torch.float64)
    candidate = materialize_tangent_input_vjp(
        candidate_gradient,
        candidate_weights,
        observed,
    ).to(torch.float64)
    statistics = paired_tangent_vjp_statistics(
        reference_gradient,
        reference_weights,
        candidate_gradient,
        candidate_weights,
        observed,
        mask_entry_id="node-0",
        probe_index=3,
        node_chunk_size=2,
    )

    assert statistics.reference_squared_norm == pytest.approx(
        float(torch.sum(reference * reference)),
        rel=5e-8,
    )
    assert statistics.candidate_squared_norm == pytest.approx(
        float(torch.sum(candidate * candidate)),
        rel=5e-8,
    )
    assert statistics.cross_inner_product == pytest.approx(
        float(torch.sum(reference * candidate)),
        rel=1e-7,
    )
    # Tangent projection must remove the all-ones class direction.
    torch.testing.assert_close(
        reference.sum(dim=-1),
        torch.zeros_like(reference[..., 0]),
        atol=1e-6,
        rtol=0.0,
    )
    assert not bool(reference[~observed].any())


def test_multi_pair_statistics_project_each_width_consistently() -> None:
    generator = torch.Generator().manual_seed(41)
    observed = torch.rand((8, 5), generator=generator) > 0.2
    model_vjps = {
        "a": (
            torch.randn((8, 4), generator=generator),
            torch.randn((4, 4, 5), generator=generator),
        ),
        "b": (
            torch.randn((8, 7), generator=generator),
            torch.randn((4, 7, 5), generator=generator),
        ),
        "c": (
            torch.randn((8, 3), generator=generator),
            torch.randn((4, 3, 5), generator=generator),
        ),
    }
    pairs = (("a", "b"), ("a", "c"), ("b", "c"))
    multi = multi_tangent_vjp_statistics(
        model_vjps,
        observed,
        pairs,
        mask_entry_id="node-1",
        probe_index=4,
        node_chunk_size=3,
    )

    for pair in pairs:
        direct = paired_tangent_vjp_statistics(
            *model_vjps[pair[0]],
            *model_vjps[pair[1]],
            observed,
            mask_entry_id="node-1",
            probe_index=4,
            node_chunk_size=3,
        )
        assert multi[pair] == direct


def test_preactivation_vjp_matches_explicit_relaxed_one_hot_autograd() -> None:
    model = _model()
    expression, mask, covariates = _inputs()
    edge_index, edge_attributes = _graph()
    targets = whole_node_targets(mask)
    probe, _ = make_rademacher_probe(
        (targets.numel(), expression.shape[1], NUM_COUNT_TOKENS),
        mask_entry_id="node-0",
        probe_index=0,
    )

    compressed_gradient = preactivation_probe_vjp(
        model,
        input_expression=expression,
        gene_mask=mask,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        node_covariates=covariates,
        probe=probe,
    )
    weights = observed_token_projection_weights(model.encoder).detach()
    compressed = materialize_tangent_input_vjp(
        compressed_gradient,
        weights,
        ~mask,
    )

    # Explicitly expose the four relaxed channels for this tiny graph.  The
    # mask-only fifth channel stays fixed and observed-channel derivatives at
    # masked inputs are removed after differentiation.
    token_ids = expression.to(torch.long)
    relaxed = F.one_hot(
        token_ids,
        num_classes=NUM_COUNT_TOKENS,
    ).to(torch.float32)
    relaxed.requires_grad_(True)
    encoder = model.encoder
    preactivation = encoder.bias.expand(expression.shape[0], -1)
    visible = (~mask).to(torch.float32)
    for token_id in range(NUM_COUNT_TOKENS):
        preactivation = preactivation + F.linear(
            relaxed[:, :, token_id] * visible,
            encoder.token_projections[token_id].weight,
            None,
        )
    mask_indicator = mask.to(torch.float32)
    preactivation = preactivation + F.linear(
        mask_indicator,
        encoder.token_projections[NUM_COUNT_TOKENS].weight,
        None,
    )
    assert encoder.covariate_projection is not None
    preactivation = preactivation + F.linear(
        covariates,
        encoder.covariate_projection.weight,
        None,
    )
    logits = _forward_from_encoder_preactivation(
        model,
        preactivation,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        target_nodes=targets,
    )
    explicit = torch.autograd.grad(
        torch.sum(center_output_logits(logits) * probe),
        relaxed,
    )[0]
    explicit = explicit - explicit.mean(dim=-1, keepdim=True)
    explicit = explicit.masked_fill(mask.unsqueeze(-1), 0.0)

    torch.testing.assert_close(compressed, explicit, atol=2e-6, rtol=2e-5)


def test_encoder_preactivation_reproduces_encoder_output() -> None:
    model = _model()
    model.eval()
    expression, mask, covariates = _inputs()
    preactivation = categorical_encoder_preactivation(
        model.encoder,
        expression,
        mask,
        covariates,
    )
    expected = model.encoder(expression, mask, covariates)
    actual = F.gelu(model.encoder.normalization(preactivation))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_estimate_pair_uses_complete_grid_and_is_deterministic() -> None:
    mask_ids = ("node-a", "node-b", "node-c")
    rows = [
        ProbeSufficientStatistics(
            mask_entry_id=mask_id,
            probe_index=probe_index,
            reference_squared_norm=float(2 + mask_index + probe_index / 100),
            candidate_squared_norm=float(2 + mask_index + probe_index / 100),
            cross_inner_product=float(2 + mask_index + probe_index / 100),
        )
        for mask_index, mask_id in enumerate(mask_ids)
        for probe_index in range(PROBES_PER_MASK)
    ]

    first = estimate_pair(
        rows,
        reference_label="same-a",
        candidate_label="same-b",
        mask_entry_ids=mask_ids,
    )
    second = estimate_pair(
        rows,
        reference_label="same-a",
        candidate_label="same-b",
        mask_entry_ids=mask_ids,
    )
    assert first == second
    assert first.point.cosine == pytest.approx(1.0)
    assert first.point.relative_discrepancy == pytest.approx(0.0)
    assert first.point.norm_ratio == pytest.approx(1.0)
    assert first.cosine_interval == PercentileInterval(1.0, 1.0)
    assert first.relative_discrepancy_interval == PercentileInterval(0.0, 0.0)
    assert first.norm_ratio_interval == PercentileInterval(1.0, 1.0)

    with pytest.raises(CategoricalSensitivityError, match="3-mask by 32-probe"):
        estimate_pair(
            rows[:-1],
            reference_label="same-a",
            candidate_label="same-b",
            mask_entry_ids=mask_ids,
        )


def test_locked_pair_metric_formulas_use_symmetric_discrepancy() -> None:
    metrics = metrics_from_sufficient_statistics(4.0, 9.0, 3.0)

    assert metrics.cosine == pytest.approx(0.5)
    assert metrics.relative_discrepancy == pytest.approx(math.sqrt(7.0 / 6.0))
    assert metrics.norm_ratio == pytest.approx(1.5)


def test_operational_match_applies_identical_within_and_random_controls() -> None:
    identical = _pair_estimate(
        cosine=1.0,
        discrepancy=0.0,
        ratio=1.0,
        cosine_interval=(0.99995, 1.0),
        discrepancy_interval=(0.0, 0.0005),
        ratio_interval=(0.9995, 1.0005),
    )
    within_current = {
        "0-1": _pair_estimate(cosine=0.96, discrepancy=0.12, ratio=1.10),
        "0-2": _pair_estimate(cosine=0.97, discrepancy=0.10, ratio=0.92),
        "1-2": _pair_estimate(cosine=0.965, discrepancy=0.11, ratio=1.05),
    }
    within_wider = {
        "0-1": _pair_estimate(cosine=0.95, discrepancy=0.13, ratio=1.08),
        "0-2": _pair_estimate(cosine=0.955, discrepancy=0.12, ratio=0.94),
        "1-2": _pair_estimate(cosine=0.96, discrepancy=0.11, ratio=1.03),
    }
    randomized = {
        seed: _pair_estimate(
            cosine=0.2 + (seed - 9100) / 1000,
            discrepancy=1.1,
            ratio=1.0,
            cosine_interval=(0.1, 0.35),
            discrepancy_interval=(0.9, 1.3),
        )
        for seed in range(9100, 9108)
    }
    primary = {
        seed: _pair_estimate(
            cosine=0.985,
            discrepancy=0.08,
            ratio=1.04,
            cosine_interval=(0.975, 0.99),
            discrepancy_interval=(0.05, 0.12),
            ratio_interval=(0.95, 1.12),
        )
        for seed in range(3)
    }

    result = evaluate_operational_match(
        primary_by_seed=primary,
        within_current=within_current,
        within_wider=within_wider,
        identical=identical,
        randomized_by_seed=randomized,
    )
    assert result["analysis_numerically_valid"] is True
    assert result["operational_match"] is True
    assert all(
        value["passed"] for value in result["primary"].values()
    )

    invalid_identical = replace(
        identical,
        cosine_interval=PercentileInterval(0.99, 1.0),
    )
    invalid = evaluate_operational_match(
        primary_by_seed=primary,
        within_current=within_current,
        within_wider=within_wider,
        identical=invalid_identical,
        randomized_by_seed=randomized,
    )
    assert invalid["analysis_numerically_valid"] is False
    assert invalid["operational_match"] is False


def test_zero_or_nonfinite_norms_fail_closed() -> None:
    with pytest.raises(CategoricalSensitivityError, match="zero or negative"):
        metrics_from_sufficient_statistics(0.0, 1.0, 0.0)

    with pytest.raises(ValueError, match="finite"):
        ProbeSufficientStatistics("mask", 0, 1.0, math.nan, 0.0)


def test_protocol_record_contains_locked_conventions() -> None:
    record = locked_protocol_record()

    assert record["mask_scope"]["mode"] == "whole_node"
    assert record["mask_scope"]["entry_count"] == 3
    assert record["probes"]["count_per_mask"] == 32
    assert record["bootstrap"]["replicates"] == 2_000
    assert (
        record["metrics"]["relative_discrepancy"]
        == "sqrt(max(A2 + B2 - 2*AB, 0)) / (A2 * B2)**0.25"
    )
    assert record["controls"]["randomized"]["seeds"] == list(
        range(9100, 9108)
    )

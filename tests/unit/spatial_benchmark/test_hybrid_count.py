"""Focused CPU contracts for the hybrid raw-count G2 implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch.nn import functional as F  # noqa: E402

from spatial_benchmark.hybrid_count import (  # noqa: E402
    MASK_TOKEN_ID,
    HybridCountNodeEncoder,
    HybridEdgeParameterMatchedSelfControl,
    HybridReceiverChunkedEdgeConditionedGATv2,
    assert_exact_parameter_match,
    decode_count_states,
    decode_positive_states,
    hybrid_count_hurdle_loss,
    split_hybrid_prediction,
    tokenize_raw_count_tensor,
    tokenize_raw_counts,
    validate_raw_counts,
)
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    evaluate_hybrid_count_output,
    fit_hybrid_count_references,
)


_STATE_COUNTS = np.asarray([0, 1, 2, 3, 4, 8, 16, 32], dtype=np.float32)


def _standardization(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logged = np.log1p(counts.astype(np.float64, copy=False))
    mean = logged.mean(axis=0)
    scale = logged.std(axis=0, ddof=0)
    if np.any(scale == 0):
        scale = np.where(scale == 0, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _perfect_ordinal_logits(tokens: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(
        1,
        7,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    above = tokens.unsqueeze(-1) > thresholds
    return torch.where(
        above,
        torch.full((), 12.0, device=tokens.device),
        torch.full((), -12.0, device=tokens.device),
    )


def test_fixed_token_boundaries_and_mask_token_is_input_only() -> None:
    counts = np.asarray(
        [[0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 100]],
        dtype=np.int64,
    )
    expected = np.asarray(
        [[0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7]],
        dtype=np.int64,
    )

    actual = tokenize_raw_counts(counts)
    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(
        tokenize_raw_count_tensor(torch.from_numpy(counts)),
        torch.from_numpy(expected),
        rtol=0,
        atol=0,
    )
    assert int(actual.max()) < MASK_TOKEN_ID


@pytest.mark.parametrize(
    ("counts", "error", "message"),
    [
        (np.asarray([[0.0, np.nan]]), ValueError, "finite"),
        (np.asarray([[0.0, np.inf]]), ValueError, "finite"),
        (np.asarray([[0.0, -1.0]]), ValueError, "nonnegative"),
        (np.asarray([[0.0, 1.5]]), ValueError, "integer-valued"),
        (np.asarray([[False, True]]), TypeError, "real numeric"),
        (np.asarray([[0 + 0j, 1 + 0j]]), TypeError, "real numeric"),
    ],
)
def test_raw_count_validation_fails_closed(
    counts: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_raw_counts(counts)


def test_mask_is_authoritative_for_discrete_and_continuous_branches() -> None:
    mean = np.asarray([0.3, 0.7, 1.1], dtype=np.float32)
    scale = np.asarray([0.8, 1.2, 1.7], dtype=np.float32)
    encoder = HybridCountNodeEncoder(
        num_genes=3,
        node_covariate_dim=2,
        hidden_dim=10,
        expression_mean=mean,
        expression_scale=scale,
        dropout=0.0,
    ).eval()
    counts = torch.tensor(
        [[4.0, 8.0, 16.0], [7.0, 15.0, 31.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, False, True], [False, True, False]],
        dtype=torch.bool,
    )
    changed = counts.clone()
    changed[mask] = torch.tensor([100.0, 1.0, 32.0])
    covariates = torch.tensor([[0.2, -0.4], [0.7, 0.3]])

    tokens, continuous = encoder.masked_inputs(counts, mask)
    changed_tokens, changed_continuous = encoder.masked_inputs(changed, mask)
    assert bool((tokens[mask] == MASK_TOKEN_ID).all())
    assert bool((continuous[mask] == 0).all())
    torch.testing.assert_close(tokens, changed_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        continuous, changed_continuous, rtol=0, atol=0
    )
    torch.testing.assert_close(
        encoder(counts, mask, covariates),
        encoder(changed, mask, covariates),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        encoder(counts.to(torch.int64), mask, covariates),
        encoder(counts, mask, covariates),
        rtol=0,
        atol=0,
    )

    # Counts 4 and 7 share token 4, while their exact continuous values remain
    # distinguishable when the gene is observed.
    within_bin = torch.tensor([[4.0, 8.0, 16.0], [7.0, 8.0, 16.0]])
    observed_mask = torch.zeros_like(within_bin, dtype=torch.bool)
    within_tokens, within_continuous = encoder.masked_inputs(
        within_bin, observed_mask
    )
    assert within_tokens[0, 0].item() == within_tokens[1, 0].item() == 4
    assert within_continuous[0, 0].item() != within_continuous[1, 0].item()


def test_hurdle_loss_balances_strata_and_uses_equal_component_weights() -> None:
    raw_target = torch.tensor(
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 8, 16, 32],
        dtype=torch.float32,
    ).reshape(-1, 1)
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    prediction = torch.zeros(
        (*raw_target.shape, 8), dtype=torch.float32
    )
    prediction[:5, 0, 0] = 2.0
    prediction[5:, 0, 0] = 1.0
    tokens = tokenize_raw_count_tensor(raw_target)
    prediction[..., 1:7] = _perfect_ordinal_logits(tokens)
    prediction[..., 7] = torch.log1p(raw_target)
    mean = torch.zeros(1)
    scale = torch.ones(1)

    loss = hybrid_count_hurdle_loss(
        prediction,
        raw_target,
        mask,
        expression_mean=mean,
        expression_scale=scale,
    )
    selected_detection = prediction[..., 0][mask]
    zero = raw_target[mask] == 0
    positive = ~zero
    expected_detection = 0.5 * (
        F.binary_cross_entropy_with_logits(
            selected_detection[zero],
            torch.zeros_like(selected_detection[zero]),
        )
        + F.binary_cross_entropy_with_logits(
            selected_detection[positive],
            torch.ones_like(selected_detection[positive]),
        )
    )
    assert loss.detection.item() == pytest.approx(expected_detection.item())
    assert loss.ordinal.item() < 1e-4
    assert loss.positive_continuous_huber.item() == pytest.approx(0.0)
    assert loss.total.item() == pytest.approx(
        (
            loss.detection.item()
            + loss.ordinal.item()
            + loss.positive_continuous_huber.item()
        )
        / 3.0
    )


@pytest.mark.parametrize(
    ("raw_target", "message"),
    [
        (torch.ones(8, 1), "zero and positive strata"),
        (
            torch.tensor([0, 1, 2, 3, 4, 8, 16], dtype=torch.float32).reshape(
                -1, 1
            ),
            "threshold 6",
        ),
    ],
)
def test_hurdle_loss_fails_when_an_expected_stratum_is_absent(
    raw_target: torch.Tensor,
    message: str,
) -> None:
    prediction = torch.zeros((*raw_target.shape, 8))
    with pytest.raises(ValueError, match=message):
        hybrid_count_hurdle_loss(
            prediction,
            raw_target,
            torch.ones_like(raw_target, dtype=torch.bool),
            expression_mean=torch.zeros(1),
            expression_scale=torch.ones(1),
        )


def test_ordinal_decoding_and_positive_metrics_ignore_detection_head() -> None:
    raw_target = torch.from_numpy(_STATE_COUNTS.reshape(-1, 1))
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    tokens = tokenize_raw_count_tensor(raw_target)
    prediction = torch.zeros((*raw_target.shape, 8), dtype=torch.float32)
    prediction[..., 0] = -12.0  # Force every full hurdle state to zero.
    prediction[..., 1:7] = _perfect_ordinal_logits(tokens)
    mean, scale = _standardization(raw_target.numpy())
    prediction[..., 7] = (
        torch.log1p(raw_target) - torch.from_numpy(mean)
    ) / torch.from_numpy(scale)

    ordinal = prediction[..., 1:7]
    expected_positive = torch.arange(1, 8)
    torch.testing.assert_close(
        decode_positive_states(ordinal)[1:, 0],
        expected_positive,
        rtol=0,
        atol=0,
    )
    assert bool((decode_count_states(prediction) == 0).all())

    references = fit_hybrid_count_references(
        raw_target.numpy(),
        expression_mean=mean,
        expression_scale=scale,
    )
    result = evaluate_hybrid_count_output(
        prediction,
        raw_target,
        mask,
        expression_mean=mean,
        expression_scale=scale,
        references=references,
    )
    assert result.metrics["state8_exact_accuracy"] == pytest.approx(1 / 8)
    assert result.metrics["positive_state_exact_accuracy"] == pytest.approx(1.0)
    assert result.metrics["positive_ordinal_mae"] == pytest.approx(0.0)
    assert result.metrics["positive_within_one_state_accuracy"] == pytest.approx(
        1.0
    )


def _paired_models() -> tuple[
    HybridReceiverChunkedEdgeConditionedGATv2,
    HybridEdgeParameterMatchedSelfControl,
]:
    counts = np.asarray(
        [[0, 1, 4], [1, 2, 8], [2, 3, 16], [3, 7, 32]],
        dtype=np.float32,
    )
    mean, scale = _standardization(counts)
    common = {
        "num_genes": 3,
        "edge_attribute_dim": 5,
        "expression_mean": mean,
        "expression_scale": scale,
        "node_covariate_dim": 2,
        "hidden_dim": 12,
        "attention_heads": 3,
        "graph_layers": 2,
        "ffn_dim": 17,
        "decoder_dim": 13,
        "edge_hidden_dim": 7,
        "edge_embedding_dim": 6,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    graph = HybridReceiverChunkedEdgeConditionedGATv2(
        **common,
        receiver_chunk_size=2,
        activation_checkpointing=False,
    )
    self_control = HybridEdgeParameterMatchedSelfControl(**common)
    return graph, self_control


def test_output_shapes_and_exact_graph_self_parameter_count() -> None:
    torch.manual_seed(20260729)
    graph, self_control = _paired_models()
    parameter_count = assert_exact_parameter_match(graph, self_control)
    assert parameter_count > 0
    counts = torch.tensor(
        [[0, 1, 4], [1, 2, 8], [2, 3, 16], [3, 7, 32]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
            [False, False, True],
            [True, True, True],
        ]
    )
    covariates = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5], [-1.0, 0.5]]
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 3, 0, 2, 1, 3, 0, 2]],
        dtype=torch.long,
    )
    edge_attributes = torch.linspace(
        -1.0, 1.0, edge_index.shape[1] * 5
    ).reshape(edge_index.shape[1], 5)
    graph.eval()
    self_control.eval()
    with torch.no_grad():
        graph_output = graph(
            counts,
            mask,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            node_covariates=covariates,
            target_nodes=[3, 1],
        )
        self_output = self_control(
            counts,
            mask,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            node_covariates=covariates,
            target_nodes=[3, 1],
        )

    for output in (graph_output, self_output):
        assert output.prediction.shape == (2, 3, 8)
        detection, ordinal, continuous = split_hybrid_prediction(
            output.prediction
        )
        assert detection.shape == (2, 3)
        assert ordinal.shape == (2, 3, 6)
        assert continuous.shape == (2, 3)
        assert output.node_embedding.shape == (2, 12)


def test_small_deterministic_synthetic_recovery() -> None:
    """The masked self arm can recover planted detection and ordered levels."""

    torch.manual_seed(7319)
    repeats = 6
    raw_target = torch.from_numpy(
        np.tile(_STATE_COUNTS, repeats).reshape(-1, 1)
    )
    state = tokenize_raw_count_tensor(raw_target).reshape(-1)
    covariates = F.one_hot(state, num_classes=8).to(dtype=torch.float32)
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    mean, scale = _standardization(raw_target.numpy())
    model = HybridEdgeParameterMatchedSelfControl(
        num_genes=1,
        edge_attribute_dim=4,
        expression_mean=mean,
        expression_scale=scale,
        node_covariate_dim=8,
        hidden_dim=20,
        attention_heads=4,
        graph_layers=1,
        ffn_dim=28,
        decoder_dim=24,
        edge_hidden_dim=8,
        edge_embedding_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-2, weight_decay=0.0
    )
    mean_tensor = torch.from_numpy(mean)
    scale_tensor = torch.from_numpy(scale)

    model.train()
    with torch.no_grad():
        initial = hybrid_count_hurdle_loss(
            model(raw_target, mask, node_covariates=covariates).prediction,
            raw_target,
            mask,
            expression_mean=mean_tensor,
            expression_scale=scale_tensor,
        ).total.item()
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        output = model(raw_target, mask, node_covariates=covariates)
        loss = hybrid_count_hurdle_loss(
            output.prediction,
            raw_target,
            mask,
            expression_mean=mean_tensor,
            expression_scale=scale_tensor,
        )
        loss.total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(
            raw_target, mask, node_covariates=covariates
        ).prediction
        final = hybrid_count_hurdle_loss(
            prediction,
            raw_target,
            mask,
            expression_mean=mean_tensor,
            expression_scale=scale_tensor,
        )
        detection_logits, ordinal_logits, _ = split_hybrid_prediction(
            prediction
        )
        detected = torch.sigmoid(detection_logits.reshape(-1)) >= 0.5
        target_detected = raw_target.reshape(-1) > 0
        sensitivity = (detected[target_detected]).float().mean()
        specificity = (~detected[~target_detected]).float().mean()
        positive_state = decode_positive_states(ordinal_logits).reshape(-1)
        ordinal_mae = (
            positive_state[target_detected] - state[target_detected]
        ).abs().float().mean()

    assert math.isfinite(final.total.item())
    assert final.total.item() < 0.15 * initial
    assert 0.5 * (sensitivity + specificity) >= 0.99
    assert ordinal_mae <= 0.10

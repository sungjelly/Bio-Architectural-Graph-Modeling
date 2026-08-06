"""Recovery check for one model trained over ten disconnected core batches."""

from __future__ import annotations

import math

import numpy as np
import pytest


torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from spatial_benchmark.hybrid_count import (  # noqa: E402
    HybridEdgeParameterMatchedSelfControl,
    decode_positive_states,
    hybrid_count_hurdle_loss,
    split_hybrid_prediction,
    tokenize_raw_count_tensor,
)
from spatial_benchmark.pooled_hybrid_count_training import (  # noqa: E402
    POOLED_CORE_ALIASES,
    PooledCoreBatch,
    fit_pooled_hybrid_count_model,
)
from spatial_benchmark.training import GraphSplitView, TrainingConfig  # noqa: E402


_STATE_COUNTS = np.asarray([0, 1, 2, 3, 4, 8, 16, 32], dtype=np.float32)


def _synthetic_batches() -> tuple[PooledCoreBatch, ...]:
    batches: list[PooledCoreBatch] = []
    for index, alias in enumerate(POOLED_CORE_ALIASES):
        counts = np.tile(np.roll(_STATE_COUNTS, index % 8), 2)
        expression = torch.from_numpy(counts[:, None].copy())
        state = tokenize_raw_count_tensor(expression).reshape(-1)
        covariates = F.one_hot(state, num_classes=8).to(dtype=torch.float32)
        n_nodes = int(expression.shape[0])
        source = torch.arange(n_nodes, dtype=torch.long)
        receiver = torch.roll(source, shifts=-1)
        batches.append(
            PooledCoreBatch(
                alias,
                GraphSplitView(
                    expression=expression,
                    coordinates_um=torch.stack(
                        (
                            torch.arange(n_nodes, dtype=torch.float64),
                            torch.full(
                                (n_nodes,),
                                float(index),
                                dtype=torch.float64,
                            ),
                        ),
                        dim=1,
                    ),
                    edge_index=torch.stack((source, receiver)),
                    node_covariates=covariates,
                    edge_attributes=torch.ones(n_nodes, 4),
                    name="fit",
                ),
            )
        )
    return tuple(batches)


def _mean_loss(
    model: torch.nn.Module,
    batches: tuple[PooledCoreBatch, ...],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            target = batch.view.expression
            mask = torch.ones_like(target, dtype=torch.bool)
            output = model(
                target,
                mask,
                node_covariates=batch.view.node_covariates,
            )
            losses.append(
                float(
                    hybrid_count_hurdle_loss(
                        output.prediction,
                        target,
                        mask,
                        expression_mean=torch.from_numpy(mean),
                        expression_scale=torch.from_numpy(scale),
                    ).total
                )
            )
    return float(np.mean(losses))


def test_shared_model_recovers_detection_and_order_across_ten_graphs() -> None:
    """A shared model can recover a signal repeated across ten graph batches."""

    batches = _synthetic_batches()
    all_counts = np.concatenate(
        [batch.view.expression.numpy() for batch in batches],
        axis=0,
    )
    transformed = np.log1p(all_counts.astype(np.float64))
    mean = transformed.mean(axis=0).astype(np.float32)
    scale = transformed.std(axis=0).astype(np.float32)
    torch.manual_seed(7319)
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
    initial = _mean_loss(model, batches, mean=mean, scale=scale)
    result = fit_pooled_hybrid_count_model(
        model,
        batches,
        TrainingConfig(
            max_epochs=30,
            learning_rate=2e-2,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            huber_delta=1.0,
            patience=0,
            curriculum="P-only",
            warmup_epochs=0,
            partial_gene_rate=1.0,
            node_rate=0.5,
            block_node_rate=0.5,
            mask_seed=314159,
            model_seed=0,
            edge_dropout=0.0,
            amp=False,
            deterministic=True,
            deterministic_warn_only=False,
            device="cpu",
            restore_best=False,
        ),
        expression_mean=mean,
        expression_scale=scale,
    )

    detected_correct: list[float] = []
    ordinal_errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            target = batch.view.expression
            mask = torch.ones_like(target, dtype=torch.bool)
            prediction = model(
                target,
                mask,
                node_covariates=batch.view.node_covariates,
            ).prediction
            detection_logits, ordinal_logits, _ = split_hybrid_prediction(
                prediction
            )
            target_detected = target.reshape(-1) > 0
            detected = torch.sigmoid(detection_logits.reshape(-1)) >= 0.5
            sensitivity = detected[target_detected].float().mean()
            specificity = (~detected[~target_detected]).float().mean()
            detected_correct.append(
                float(0.5 * (sensitivity + specificity))
            )
            states = tokenize_raw_count_tensor(target).reshape(-1)
            decoded = decode_positive_states(ordinal_logits).reshape(-1)
            ordinal_errors.append(
                float(
                    (decoded[target_detected] - states[target_detected])
                    .abs()
                    .float()
                    .mean()
                )
            )

    final = _mean_loss(model, batches, mean=mean, scale=scale)
    assert len(result.core_history) == 300
    assert result.optimizer_steps_completed == 300
    assert math.isfinite(final)
    assert final < 0.20 * initial
    assert float(np.mean(detected_correct)) >= 0.99
    assert float(np.mean(ordinal_errors)) <= 0.10

"""Focused fixed-budget multiscale hurdle-training contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch import nn  # noqa: E402

from spatial_benchmark.multiscale_hurdle_training import (  # noqa: E402
    MultiscaleGraphSplitView,
    compare_multiscale_fp32_amp_loss,
    evaluate_fixed_multiscale_hurdle_mask,
    fit_full_core_multiscale_hurdle_model,
)
from spatial_benchmark.multiscale_hybrid import (  # noqa: E402
    MultiscaleAdditiveHybridModel,
)
from spatial_benchmark.training import TrainingConfig  # noqa: E402


class _TinyMultiscaleModel(nn.Module):
    def __init__(
        self,
        *,
        regional_routing: str,
        local_routing: str,
        bad_output: str | None = None,
    ) -> None:
        super().__init__()
        self.regional_routing = regional_routing
        self.local_routing = local_routing
        self.output_bias = nn.Parameter(torch.tensor([0.0, 0.2]))
        self.bad_output = bad_output
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        *,
        node_covariates: torch.Tensor | None = None,
        regional_edge_index: torch.Tensor | None = None,
        regional_edge_attributes: torch.Tensor | None = None,
        local_edge_index: torch.Tensor | None = None,
        local_edge_attributes: torch.Tensor | None = None,
        local_source_index_by_node: torch.Tensor | None = None,
        target_nodes: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        assert target_nodes is not None
        if self.regional_routing == "surrogate":
            assert regional_edge_index is None
            assert regional_edge_attributes is None
        else:
            assert regional_edge_index is not None
            assert regional_edge_attributes is not None
        if self.local_routing == "surrogate":
            assert local_edge_index is None
            assert local_edge_attributes is None
        else:
            assert local_edge_index is not None
            assert local_edge_attributes is not None
        if self.local_routing == "permuted":
            assert local_source_index_by_node is not None
        else:
            assert local_source_index_by_node is None
        self.calls.append(
            {
                "target_nodes": target_nodes.detach().cpu().clone(),
                "regional_edges": (
                    0
                    if regional_edge_index is None
                    else int(regional_edge_index.shape[1])
                ),
                "local_edges": (
                    0
                    if local_edge_index is None
                    else int(local_edge_index.shape[1])
                ),
                "local_source_index_by_node": (
                    None
                    if local_source_index_by_node is None
                    else local_source_index_by_node.detach().cpu().clone()
                ),
            }
        )
        prediction = self.output_bias.reshape(1, 1, 2).expand(
            target_nodes.numel(),
            input_expression.shape[1],
            2,
        )
        if self.bad_output == "shape":
            prediction = prediction[..., :1]
        elif self.bad_output == "nonfinite":
            prediction = prediction.clone()
            prediction[0, 0, 0] = float("nan")
        return SimpleNamespace(prediction=prediction)


def _view(
    *,
    expression: torch.Tensor | None = None,
    local_edges: torch.Tensor | None = None,
    regional_edges: torch.Tensor | None = None,
    local_source_index_by_node: torch.Tensor | None = None,
) -> MultiscaleGraphSplitView:
    counts = (
        torch.tensor(
            [
                [0, 1],
                [2, 3],
                [4, 8],
                [16, 32],
                [0, 2],
                [1, 4],
                [3, 16],
                [8, 32],
            ],
            dtype=torch.float32,
        )
        if expression is None
        else expression
    )
    local = (
        torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]],
            dtype=torch.long,
        )
        if local_edges is None
        else local_edges
    )
    regional = (
        torch.tensor(
            [[0, 2, 1, 3], [2, 0, 3, 1]],
            dtype=torch.long,
        )
        if regional_edges is None
        else regional_edges
    )
    return MultiscaleGraphSplitView(
        expression=counts,
        coordinates_um=torch.stack(
            (
                torch.arange(counts.shape[0], dtype=torch.float64),
                torch.zeros(counts.shape[0], dtype=torch.float64),
            ),
            dim=1,
        ),
        node_covariates=torch.zeros(counts.shape[0], 2),
        local_edge_index=local,
        local_edge_attributes=torch.linspace(
            -1.0,
            1.0,
            local.shape[1] * 2,
        ).reshape(local.shape[1], 2),
        regional_edge_index=regional,
        regional_edge_attributes=torch.linspace(
            0.0,
            2.0,
            regional.shape[1] * 2,
        ).reshape(regional.shape[1], 2),
        local_source_index_by_node=local_source_index_by_node,
        block_ids=torch.arange(counts.shape[0]) // 2,
        name="fit",
    )


def _config(*, max_epochs: int = 2) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=max_epochs,
        learning_rate=0.0,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        huber_delta=1.0,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=1.0,
        node_rate=0.5,
        block_node_rate=0.5,
        mask_seed=1729,
        model_seed=2718,
        edge_dropout=0.0,
        amp=False,
        deterministic=True,
        deterministic_warn_only=False,
        device="cpu",
        restore_best=False,
    )


def test_fixed_budget_training_batches_targets_and_tracks_resources() -> None:
    view = _view()
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="true",
    )
    result = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        _config(max_epochs=2),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=3,
    )

    assert len(result.history) == 2
    assert result.final_epoch == 1
    assert result.fixed_epoch_budget == 2
    assert result.target_node_batch_size == 3
    assert result.regional_routing == "true"
    assert result.local_routing == "true"
    assert len(result.final_state_checksum) == 64
    assert len(model.calls) == 6
    for record in result.history:
        assert record.mask_mode == "partial"
        assert record.n_masked_entries == 16
        assert record.n_zero_targets == 2
        assert record.n_positive_targets == 14
        assert record.n_target_nodes == 8
        assert record.n_target_batches == 3
        assert record.n_local_edges_used == 4
        assert record.n_regional_edges_used == 4
        assert record.peak_cuda_memory_bytes == 0
        assert record.duration_seconds > 0
        assert record.gradient_norm > 0
        assert record.local_edge_checksum
        assert record.regional_edge_checksum


def test_trainer_routes_node_permutation_only_to_local_permuted_arm() -> None:
    permutation = torch.roll(torch.arange(8), shifts=4)
    view = _view(local_source_index_by_node=permutation)
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="permuted",
    )

    result = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        _config(max_epochs=1),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=8,
    )

    assert result.local_routing == "permuted"
    assert len(model.calls) == 1
    torch.testing.assert_close(
        model.calls[0]["local_source_index_by_node"],
        permutation,
        rtol=0,
        atol=0,
    )


def test_batched_training_loss_matches_unbatched_fixed_evaluation() -> None:
    view = _view()
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="true",
    )
    result = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        _config(max_epochs=1),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=3,
    )
    evaluation = evaluate_fixed_multiscale_hurdle_mask(
        model,
        view,
        torch.ones_like(view.expression, dtype=torch.bool),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=8,
        device="cpu",
    )

    assert result.history[0].train_hurdle_loss == pytest.approx(
        evaluation.evaluation.metrics["hurdle_loss"],
        abs=1e-7,
    )
    assert result.history[0].train_detection_bce == pytest.approx(
        evaluation.evaluation.metrics["detection_bce"],
        abs=1e-7,
    )
    assert result.history[
        0
    ].train_positive_continuous_huber == pytest.approx(
        evaluation.evaluation.metrics["positive_continuous_huber"],
        abs=1e-7,
    )
    assert evaluation.target_nodes.tolist() == list(range(8))
    assert evaluation.evaluation.metrics["n_positive"] == 14
    assert "positive_count_state_mae" in evaluation.evaluation.metrics
    assert "state8_balanced_accuracy" in evaluation.evaluation.metrics
    assert len(evaluation.evaluation.metrics["state8_recall"]) == 8


def test_trainer_executes_real_multiscale_model_public_interface() -> None:
    local = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
        ],
        dtype=torch.long,
    )
    regional = torch.tensor(
        [
            [0, 2, 1, 3, 4, 6, 5, 7],
            [2, 0, 3, 1, 6, 4, 7, 5],
        ],
        dtype=torch.long,
    )
    view = _view(local_edges=local, regional_edges=regional)
    model = MultiscaleAdditiveHybridModel(
        num_genes=2,
        local_edge_attribute_dim=2,
        regional_edge_attribute_dim=2,
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        node_covariate_dim=2,
        hidden_dim=8,
        decoder_dim=7,
        ffn_dim=11,
        attention_heads=2,
        attention_head_dim=3,
        value_head_dim=2,
        message_dim=5,
        edge_hidden_dim=7,
        edge_embedding_dim=4,
        output_channels=2,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=3,
        activation_checkpointing=True,
        regional_routing="true",
        local_routing="true",
    )

    result = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        _config(max_epochs=1),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=3,
    )
    evaluation = evaluate_fixed_multiscale_hurdle_mask(
        model,
        view,
        torch.ones_like(view.expression, dtype=torch.bool),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=3,
        device="cpu",
    )

    assert result.regional_routing == "true"
    assert result.local_routing == "true"
    assert result.history[0].n_local_edges_used == 8
    assert result.history[0].n_regional_edges_used == 8
    assert result.final_train_loss == pytest.approx(
        evaluation.evaluation.metrics["hurdle_loss"],
        abs=1e-7,
    )
    assert torch.isfinite(
        torch.tensor(evaluation.evaluation.metrics["hurdle_loss"])
    )


def test_surrogate_routing_receives_no_graph_and_masks_are_reproducible() -> None:
    view = _view()
    first = _TinyMultiscaleModel(
        regional_routing="surrogate",
        local_routing="surrogate",
    )
    second = _TinyMultiscaleModel(
        regional_routing="surrogate",
        local_routing="surrogate",
    )
    first_result = fit_full_core_multiscale_hurdle_model(
        first,
        view,
        _config(max_epochs=2),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=4,
    )
    second_result = fit_full_core_multiscale_hurdle_model(
        second,
        view,
        _config(max_epochs=2),
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
        target_node_batch_size=4,
    )

    assert all(call["local_edges"] == 0 for call in first.calls)
    assert all(call["regional_edges"] == 0 for call in first.calls)
    assert all(record.n_local_edges_used == 0 for record in first_result.history)
    assert all(
        record.n_regional_edges_used == 0 for record in first_result.history
    )
    assert [row.mask_seed for row in first_result.history] == [
        row.mask_seed for row in second_result.history
    ]
    assert [row.mask_checksum for row in first_result.history] == [
        row.mask_checksum for row in second_result.history
    ]
    assert first_result.final_state_checksum == second_result.final_state_checksum


@pytest.mark.parametrize(
    ("bad_output", "error", "message"),
    [
        ("shape", ValueError, "shape"),
        ("nonfinite", FloatingPointError, "non-finite"),
    ],
)
def test_training_fails_closed_on_invalid_model_output(
    bad_output: str,
    error: type[Exception],
    message: str,
) -> None:
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="true",
        bad_output=bad_output,
    )
    with pytest.raises(error, match=message):
        fit_full_core_multiscale_hurdle_model(
            model,
            _view(),
            _config(max_epochs=1),
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
            target_node_batch_size=3,
        )


def test_training_and_evaluation_fail_without_positive_masked_targets() -> None:
    zero_view = _view(expression=torch.zeros(8, 2))
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="true",
    )
    with pytest.raises(ValueError, match="zero and positive strata"):
        fit_full_core_multiscale_hurdle_model(
            model,
            zero_view,
            _config(max_epochs=1),
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
        )
    with pytest.raises(ValueError, match="zero and positive strata"):
        evaluate_fixed_multiscale_hurdle_mask(
            model,
            zero_view,
            torch.ones_like(zero_view.expression, dtype=torch.bool),
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
            device="cpu",
        )


def test_graph_view_fails_on_overlap_duplicates_and_misalignment() -> None:
    local = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="disjoint"):
        _view(local_edges=local, regional_edges=local.clone())
    duplicate = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="duplicate"):
        _view(local_edges=duplicate)
    with pytest.raises(TypeError, match="local_edge_attributes"):
        MultiscaleGraphSplitView(
            expression=torch.tensor([[0.0], [1.0], [2.0]]),
            coordinates_um=torch.zeros(3, 2),
            local_edge_index=torch.tensor([[0, 1], [1, 0]]),
            local_edge_attributes=torch.zeros(1, 2),
            regional_edge_index=torch.tensor([[0, 2], [2, 0]]),
            regional_edge_attributes=torch.zeros(2, 2),
            name="fit",
        )


def test_training_contract_rejects_nonfixed_or_invalid_settings() -> None:
    view = _view()
    model = _TinyMultiscaleModel(
        regional_routing="true",
        local_routing="true",
    )
    bad_restore = _config(max_epochs=1)
    object.__setattr__(bad_restore, "restore_best", True)
    with pytest.raises(ValueError, match="restore_best=False"):
        fit_full_core_multiscale_hurdle_model(
            model,
            view,
            bad_restore,
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
        )
    with pytest.raises(ValueError, match="positive integer"):
        fit_full_core_multiscale_hurdle_model(
            model,
            view,
            _config(max_epochs=1),
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
            target_node_batch_size=0,
        )


def test_precision_diagnostic_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        compare_multiscale_fp32_amp_loss(
            _TinyMultiscaleModel(
                regional_routing="true",
                local_routing="true",
            ),
            _view(),
            torch.ones(8, 2, dtype=torch.bool),
            expression_mean=torch.zeros(2),
            expression_scale=torch.ones(2),
            device="cpu",
        )

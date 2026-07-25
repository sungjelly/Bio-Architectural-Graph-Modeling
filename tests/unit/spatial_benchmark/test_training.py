"""CPU synthetic smoke tests for deterministic full-graph training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.models import (  # noqa: E402
    AdditiveEdgeMessageModel,
    BroadSpatialFieldControl,
    EdgeConditionedGATv2,
    MeanNeighborModel,
    ParameterMatchedSelfControl,
    SelfOnlyMLP,
    TopologyGATv2,
)
from spatial_benchmark.training import (  # noqa: E402
    GraphSplitView,
    TrainingConfig,
    apply_edge_dropout,
    build_model,
    evaluate_fixed_mask,
    fit_model,
    fit_staged_g3,
    make_epoch_mask,
)


def _synthetic_view(
    *,
    seed: int,
    name: str,
    num_nodes: int = 12,
    num_genes: int = 6,
) -> GraphSplitView:
    generator = torch.Generator().manual_seed(seed)
    covariates = torch.randn(num_nodes, 2, generator=generator)
    base = torch.randn(num_nodes, num_genes, generator=generator)
    # A small neighbor-correlated signal makes this a meaningful graph smoke,
    # without treating its result as a scientific recovery experiment.
    expression = base + 0.25 * torch.roll(base, shifts=1, dims=0)
    coordinates = torch.stack(
        (
            torch.arange(num_nodes, dtype=torch.float32) * 10.0,
            torch.remainder(
                torch.arange(num_nodes, dtype=torch.float32), 3
            )
            * 8.0,
        ),
        dim=1,
    )
    sources: list[int] = []
    receivers: list[int] = []
    for node in range(num_nodes):
        next_node = (node + 1) % num_nodes
        sources.extend((node, next_node))
        receivers.extend((next_node, node))
    edge_index = torch.tensor([sources, receivers], dtype=torch.long)
    source, receiver = edge_index
    delta = coordinates[receiver] - coordinates[source]
    distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    edge_attributes = torch.cat(
        (distance / 50.0, delta / distance.clamp_min(1e-6)),
        dim=1,
    )
    block_ids = np.repeat(
        np.arange(3), repeats=int(np.ceil(num_nodes / 3))
    )[:num_nodes]
    return GraphSplitView(
        expression=expression,
        coordinates_um=coordinates,
        edge_index=edge_index,
        node_covariates=covariates,
        edge_attributes=edge_attributes,
        block_ids=block_ids,
        name=name,
    )


def _small_model_kwargs() -> dict[str, int | float]:
    return {
        "hidden_dim": 12,
        "ffn_dim": 18,
        "decoder_dim": 16,
        "dropout": 0.0,
    }


def test_model_factory_covers_the_complete_ladder() -> None:
    common = {
        "num_genes": 6,
        "node_covariate_dim": 2,
        **_small_model_kwargs(),
    }
    assert isinstance(build_model("B0", seed=1, **common), SelfOnlyMLP)
    assert isinstance(
        build_model(
            "B0-matched",
            seed=1,
            attention_heads=3,
            **common,
        ),
        ParameterMatchedSelfControl,
    )
    assert isinstance(build_model("B1", seed=1, **common), MeanNeighborModel)
    assert isinstance(
        build_model("broad-field", seed=1, **common),
        BroadSpatialFieldControl,
    )
    assert isinstance(
        build_model("G1", seed=1, attention_heads=3, **common),
        TopologyGATv2,
    )
    assert isinstance(
        build_model(
            "G2",
            seed=1,
            edge_attribute_dim=3,
            edge_embedding_dim=5,
            attention_heads=3,
            **common,
        ),
        EdgeConditionedGATv2,
    )
    assert isinstance(
        build_model(
            "G3",
            seed=1,
            edge_attribute_dim=3,
            edge_embedding_dim=5,
            attention_heads=3,
            message_head_dim=2,
            message_dim=6,
            **common,
        ),
        AdditiveEdgeMessageModel,
    )
    with pytest.raises(ValueError, match="requires edge_attribute_dim"):
        build_model("G2", seed=1, **common)
    with pytest.raises(ValueError, match="unknown model_id"):
        build_model("not-a-model", seed=1, **common)


def test_factory_initialization_is_seed_deterministic() -> None:
    kwargs = {
        "num_genes": 6,
        "node_covariate_dim": 2,
        **_small_model_kwargs(),
    }
    first = build_model("B0", seed=42, **kwargs)
    second = build_model("B0", seed=42, **kwargs)
    third = build_model("B0", seed=43, **kwargs)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        torch.testing.assert_close(
            first_parameter,
            second_parameter,
            rtol=0.0,
            atol=0.0,
        )
    assert any(
        not torch.equal(first_parameter, third_parameter)
        for first_parameter, third_parameter in zip(
            first.parameters(), third.parameters()
        )
    )


def test_edge_dropout_is_paired_and_keeps_attributes_aligned() -> None:
    view = _synthetic_view(seed=3, name="train")
    first_edges, first_attributes, first_keep = apply_edge_dropout(
        view.edge_index,
        view.edge_attributes,
        probability=0.35,
        seed=91,
    )
    second_edges, second_attributes, second_keep = apply_edge_dropout(
        view.edge_index,
        view.edge_attributes,
        probability=0.35,
        seed=91,
    )
    assert torch.equal(first_keep, second_keep)
    assert torch.equal(first_edges, second_edges)
    assert torch.equal(first_attributes, second_attributes)
    assert torch.equal(first_edges, view.edge_index[:, first_keep])
    assert torch.equal(first_attributes, view.edge_attributes[first_keep])


def test_epoch_masks_are_paired_across_model_seeds() -> None:
    view = _synthetic_view(seed=4, name="train")
    first_config = TrainingConfig(
        max_epochs=1,
        warmup_epochs=0,
        curriculum="P+N+B",
        mask_seed=1234,
        model_seed=1,
        device="cpu",
    )
    second_config = TrainingConfig(
        max_epochs=1,
        warmup_epochs=0,
        curriculum="P+N+B",
        mask_seed=1234,
        model_seed=999,
        device="cpu",
    )
    for epoch in range(20):
        first = make_epoch_mask(view, first_config, epoch)
        second = make_epoch_mask(view, second_config, epoch)
        assert first.spec.mode == second.spec.mode
        assert first.seed == second.seed
        assert np.array_equal(first.mask, second.mask)


def test_cpu_training_and_fixed_mask_evaluation_smoke() -> None:
    train_view = _synthetic_view(seed=10, name="train")
    validation_view = _synthetic_view(seed=11, name="validation")
    model = build_model(
        "G1",
        num_genes=train_view.num_genes,
        node_covariate_dim=train_view.node_covariate_dim,
        seed=7,
        hidden_dim=12,
        attention_heads=3,
        ffn_dim=18,
        decoder_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
    )
    config = TrainingConfig(
        max_epochs=3,
        learning_rate=5e-3,
        weight_decay=0.0,
        patience=3,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=222,
        model_seed=7,
        edge_dropout=0.25,
        amp=False,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_view.expression, dtype=torch.bool
    )
    validation_mask[:4] = True
    result = fit_model(
        model,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )
    assert len(result.history) == 3
    assert result.graph_execution == "full_split_exact_no_neighbor_sampling"
    assert result.best_epoch in range(3)
    assert np.isfinite(result.best_validation_loss)
    assert all(
        record.n_target_nodes == train_view.num_nodes
        for record in result.history
    )
    assert all(
        0 <= record.n_edges_used <= train_view.num_edges
        for record in result.history
    )
    assert all(np.isfinite(record.train_loss) for record in result.history)
    assert all(
        np.isfinite(record.validation_loss) for record in result.history
    )

    evaluation = evaluate_fixed_mask(
        model,
        validation_view,
        validation_mask,
        device="cpu",
        return_explanations=True,
    )
    assert evaluation.predictions.shape == validation_view.expression.shape
    assert torch.equal(evaluation.mask, validation_mask)
    assert evaluation.metrics["n_masked"] == int(validation_mask.sum())
    assert np.isfinite(evaluation.metrics["huber"])
    assert len(evaluation.metrics["blocks"]) == 3
    assert torch.equal(evaluation.edge_index, validation_view.edge_index)
    assert evaluation.attention_weights.shape == (
        validation_view.num_edges,
        3,
    )


def test_broad_field_uses_train_only_coordinates_with_paired_training() -> None:
    train_view = _synthetic_view(seed=14, name="train", num_nodes=8)
    validation_base = _synthetic_view(
        seed=15, name="validation", num_nodes=8
    )
    validation_view = GraphSplitView(
        expression=validation_base.expression,
        coordinates_um=validation_base.coordinates_um
        + torch.tensor([10_000.0, -7_000.0]),
        edge_index=validation_base.edge_index,
        node_covariates=validation_base.node_covariates,
        edge_attributes=validation_base.edge_attributes,
        block_ids=validation_base.block_ids,
        name="validation",
    )
    common = {
        "num_genes": train_view.num_genes,
        "node_covariate_dim": train_view.node_covariate_dim,
        "seed": 19,
        **_small_model_kwargs(),
    }
    broad_model = build_model("broad-field", **common)
    b0_model = build_model("B0", **common)
    config = TrainingConfig(
        max_epochs=1,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=1,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=919,
        model_seed=19,
        edge_dropout=0.25,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_view.expression, dtype=torch.bool
    )
    validation_mask[:, 0] = True
    broad_result = fit_model(
        broad_model,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )
    b0_result = fit_model(
        b0_model,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )

    provenance = broad_result.spatial_control_provenance
    assert provenance is not None
    np.testing.assert_allclose(
        provenance["center_um"],
        train_view.coordinates_um.numpy().mean(axis=0),
    )
    np.testing.assert_allclose(
        provenance["scale_um"],
        train_view.coordinates_um.numpy().std(axis=0, ddof=0),
    )
    assert broad_result.graph_execution == (
        "cell_autonomous_broad_spatial_field_no_graph"
    )
    assert (
        broad_result.history[0].mask_checksum
        == b0_result.history[0].mask_checksum
    )
    assert (
        broad_result.history[0].edge_checksum
        == b0_result.history[0].edge_checksum
    )
    evaluation = evaluate_fixed_mask(
        broad_model,
        validation_view,
        validation_mask,
        device="cpu",
    )
    assert evaluation.predictions.shape == validation_view.expression.shape

    unshifted_b0 = evaluate_fixed_mask(
        b0_model,
        validation_base,
        validation_mask,
        device="cpu",
    )
    shifted_b0 = evaluate_fixed_mask(
        b0_model,
        validation_view,
        validation_mask,
        device="cpu",
    )
    torch.testing.assert_close(
        unshifted_b0.predictions,
        shifted_b0.predictions,
        rtol=0.0,
        atol=0.0,
    )


def test_training_is_exactly_reproducible_for_one_seed() -> None:
    train_view = _synthetic_view(seed=20, name="train")
    validation_view = _synthetic_view(seed=21, name="validation")
    model_kwargs = {
        "num_genes": train_view.num_genes,
        "node_covariate_dim": train_view.node_covariate_dim,
        "seed": 88,
        **_small_model_kwargs(),
    }
    config = TrainingConfig(
        max_epochs=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=2,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=900,
        model_seed=88,
        edge_dropout=0.2,
        device="cpu",
    )
    mask = torch.zeros_like(validation_view.expression, dtype=torch.bool)
    mask[:, ::2] = True
    first_model = build_model("B1", **model_kwargs)
    second_model = build_model("B1", **model_kwargs)
    first_result = fit_model(
        first_model,
        train_view,
        validation_view,
        config,
        validation_mask=mask,
    )
    second_result = fit_model(
        second_model,
        train_view,
        validation_view,
        config,
        validation_mask=mask,
    )
    assert first_result.history == second_result.history
    first_evaluation = evaluate_fixed_mask(
        first_model, validation_view, mask, device="cpu"
    )
    second_evaluation = evaluate_fixed_mask(
        second_model, validation_view, mask, device="cpu"
    )
    torch.testing.assert_close(
        first_evaluation.predictions,
        second_evaluation.predictions,
        rtol=0.0,
        atol=0.0,
    )


def test_cpu_bfloat16_amp_training_smoke() -> None:
    train_view = _synthetic_view(seed=25, name="train")
    validation_view = _synthetic_view(seed=26, name="validation")
    model = build_model(
        "B0",
        num_genes=train_view.num_genes,
        node_covariate_dim=train_view.node_covariate_dim,
        seed=5,
        **_small_model_kwargs(),
    )
    config = TrainingConfig(
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=1,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=5,
        model_seed=5,
        edge_dropout=0.0,
        amp=True,
        amp_dtype="bfloat16",
        device="cpu",
    )
    mask = torch.zeros_like(validation_view.expression, dtype=torch.bool)
    mask[:, :2] = True
    result = fit_model(
        model,
        train_view,
        validation_view,
        config,
        validation_mask=mask,
    )
    assert len(result.history) == 1
    assert np.isfinite(result.history[0].train_loss)
    assert np.isfinite(result.history[0].validation_loss)


def test_validation_early_stopping_uses_the_fixed_mask() -> None:
    train_view = _synthetic_view(seed=30, name="train")
    validation_view = _synthetic_view(seed=31, name="validation")
    model = build_model(
        "B0",
        num_genes=train_view.num_genes,
        node_covariate_dim=train_view.node_covariate_dim,
        seed=12,
        **_small_model_kwargs(),
    )
    config = TrainingConfig(
        max_epochs=8,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=2,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=55,
        model_seed=12,
        edge_dropout=0.0,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_view.expression, dtype=torch.bool
    )
    validation_mask[:, 0] = True
    result = fit_model(
        model,
        train_view,
        validation_view,
        config,
        validation_mask=validation_mask,
    )
    assert result.stopped_early
    assert len(result.history) == 3
    assert result.best_epoch == 0
    assert all(
        record.validation_loss == result.history[0].validation_loss
        for record in result.history
    )
    assert result.validation_mask_checksum


@pytest.mark.parametrize("checkpoint_wrapper", [False, True])
def test_staged_g3_loads_b0_and_records_both_optimizer_stages(
    checkpoint_wrapper: bool,
) -> None:
    train_view = _synthetic_view(seed=40, name="train", num_nodes=8)
    validation_view = _synthetic_view(
        seed=41, name="validation", num_nodes=8
    )
    common = {
        "num_genes": train_view.num_genes,
        "node_covariate_dim": train_view.node_covariate_dim,
        "hidden_dim": 12,
        "ffn_dim": 18,
        "decoder_dim": 16,
        "dropout": 0.0,
    }
    b0 = build_model("B0", seed=17, **common)
    g3 = build_model(
        "G3",
        seed=99,
        edge_attribute_dim=train_view.edge_attribute_dim,
        attention_heads=3,
        edge_hidden_dim=7,
        edge_embedding_dim=5,
        message_head_dim=2,
        message_dim=6,
        attention_dropout=0.0,
        **common,
    )
    pretrained: object = b0
    if checkpoint_wrapper:
        pretrained = {
            "run_id": "synthetic-b0",
            "state_dict": b0.state_dict(),
        }
    config = TrainingConfig(
        max_epochs=3,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=3,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=77,
        model_seed=99,
        edge_dropout=0.2,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_view.expression, dtype=torch.bool
    )
    validation_mask[:, :2] = True
    result = fit_staged_g3(
        g3,
        pretrained,
        train_view,
        validation_view,
        config,
        frozen_epochs=2,
        joint_learning_rate=0.0,
        validation_mask=validation_mask,
    )

    assert result.training_protocol == "g3_b0_frozen_neighbor_then_joint"
    assert result.pretrained_self_checksum
    assert result.pretrained_self_source == (
        "checkpoint[state_dict]" if checkpoint_wrapper else "SelfOnlyMLP"
    )
    assert [record.stage for record in result.history] == [
        "frozen_neighbor",
        "frozen_neighbor",
        "joint_finetune",
    ]
    for record in result.history:
        expected_mask = make_epoch_mask(train_view, config, record.epoch)
        assert record.mask_seed == expected_mask.seed
        assert record.mask_mode == expected_mask.spec.mode
    assert [
        (stage.name, stage.start_epoch, stage.completed_epochs)
        for stage in result.stage_provenance
    ] == [
        ("frozen_neighbor", 0, 2),
        ("joint_finetune", 2, 1),
    ]
    assert result.stage_provenance[0].learning_rate == 0.0
    assert result.stage_provenance[0].self_branch_trainable is False
    assert result.stage_provenance[0].early_stopping is False
    assert result.stage_provenance[1].learning_rate == 0.0
    assert result.stage_provenance[1].self_branch_trainable is True
    assert result.stage_provenance[1].early_stopping is True

    b0_state = b0.state_dict()
    g3_state = g3.state_dict()
    for name, value in b0_state.items():
        if name.startswith("encoder.") or name.startswith("self_block."):
            torch.testing.assert_close(
                g3_state[name], value, rtol=0.0, atol=0.0
            )
        elif name.startswith("decoder."):
            g3_name = f"self_decoder.{name.removeprefix('decoder.')}"
            torch.testing.assert_close(
                g3_state[g3_name], value, rtol=0.0, atol=0.0
            )
    assert all(parameter.requires_grad for parameter in g3.parameters())


def test_staged_g3_early_stopping_applies_only_after_frozen_stage() -> None:
    train_view = _synthetic_view(seed=50, name="train", num_nodes=8)
    validation_view = _synthetic_view(
        seed=51, name="validation", num_nodes=8
    )
    common = {
        "num_genes": train_view.num_genes,
        "node_covariate_dim": train_view.node_covariate_dim,
        **_small_model_kwargs(),
    }
    b0 = build_model("B0", seed=8, **common)
    g3 = build_model(
        "G3",
        seed=9,
        edge_attribute_dim=train_view.edge_attribute_dim,
        attention_heads=3,
        edge_embedding_dim=5,
        message_head_dim=2,
        message_dim=6,
        attention_dropout=0.0,
        **common,
    )
    config = TrainingConfig(
        max_epochs=6,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=1,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.5,
        mask_seed=80,
        model_seed=9,
        edge_dropout=0.0,
        device="cpu",
    )
    validation_mask = torch.zeros_like(
        validation_view.expression, dtype=torch.bool
    )
    validation_mask[:, 0] = True
    result = fit_staged_g3(
        g3,
        b0,
        train_view,
        validation_view,
        config,
        frozen_epochs=2,
        joint_learning_rate=0.0,
        validation_mask=validation_mask,
    )

    assert result.stopped_early
    assert len(result.history) == 3
    assert [record.stage for record in result.history[:2]] == [
        "frozen_neighbor",
        "frozen_neighbor",
    ]
    assert result.history[2].stage == "joint_finetune"
    assert result.stage_provenance[0].completed_epochs == 2
    assert result.stage_provenance[1].completed_epochs == 1


def test_staged_g3_rejects_an_incomplete_or_impossible_protocol() -> None:
    train_view = _synthetic_view(seed=60, name="train", num_nodes=8)
    validation_view = _synthetic_view(
        seed=61, name="validation", num_nodes=8
    )
    g3 = build_model(
        "G3",
        num_genes=train_view.num_genes,
        node_covariate_dim=train_view.node_covariate_dim,
        edge_attribute_dim=train_view.edge_attribute_dim,
        seed=3,
        hidden_dim=12,
        attention_heads=3,
        edge_embedding_dim=5,
        message_head_dim=2,
        message_dim=6,
    )
    config = TrainingConfig(max_epochs=2, device="cpu")
    with pytest.raises(ValueError, match="smaller than"):
        fit_staged_g3(
            g3,
            {},
            train_view,
            validation_view,
            config,
            frozen_epochs=2,
        )
    with pytest.raises(ValueError, match="missing the 'encoder'"):
        fit_staged_g3(
            g3,
            {"state_dict": {"not_encoder.weight": torch.ones(1)}},
            train_view,
            validation_view,
            config,
            frozen_epochs=1,
        )

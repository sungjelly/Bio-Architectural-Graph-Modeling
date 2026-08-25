from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

import spatial_benchmark.relative_qkv_graph_transformer as relative_qkv_module
from spatial_benchmark.pooled_relative_qkv_training import (
    PooledRelativeQKVCoreBatch,
    _tree_sha256,
)
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.relative_qkv_post_training import (
    CAMPAIGN_ID,
    CHECKPOINT_SCHEMA,
    RelativeQKVPostTrainingError,
    SelectedDerivativeRequest,
    fixed_inference_mask,
    load_relative_qkv_checkpoint,
    reciprocal_edge_ids,
    selected_autograd_derivatives,
    stream_receiver_attention,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ATTENTION_SCRIPT = (
    PROJECT_ROOT / "scripts/analysis/export_relative_qkv_edge_attention.py"
)


def _load_attention_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "export_relative_qkv_edge_attention", ATTENTION_SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _batch() -> PooledRelativeQKVCoreBatch:
    torch.manual_seed(71)
    n_nodes = 4
    pairs = [
        (source, receiver)
        for receiver in range(n_nodes)
        for source in range(n_nodes)
        if source != receiver
    ]
    return PooledRelativeQKVCoreBatch(
        alias="CAN-01",
        target_expression=torch.randn(n_nodes, 5),
        edge_index=torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        relative_geometry=torch.randn(len(pairs), 70),
        node_covariates=torch.randn(n_nodes, 2),
    )


def _model(
    *,
    activation_checkpointing: bool = False,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    torch.manual_seed(91)
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=5,
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        attention_head_dim=4,
        graph_layers=2,
        ffn_dim=20,
        decoder_dim=9,
        positional_bias_hidden_dim=7,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        max_edges_per_chunk=6,
        activation_checkpointing=activation_checkpointing,
    ).eval()


def _construction() -> dict[str, object]:
    return {
        "class": "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
        "num_genes": 5,
        "node_covariate_dim": 2,
        "hidden_dim": 12,
        "attention_heads": 3,
        "attention_head_dim": 4,
        "graph_layers": 2,
        "ffn_dim": 20,
        "decoder_dim": 9,
        "positional_bias_hidden_dim": 7,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "relative_geometry_dim": 70,
        "receiver_chunk_size": 2,
        "max_edges_per_chunk": 6,
        "activation_checkpointing": False,
    }


def test_standalone_checkpoint_loader_verifies_and_reconstructs(
    tmp_path: Path,
) -> None:
    model = _model()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "model_seed": 0,
        "completed_global_epochs": 175,
        "model_construction": _construction(),
        "model_state_dict": state,
        "model_state_checksum": _tree_sha256(state),
    }
    path = tmp_path / "last.ckpt"
    torch.save(payload, path)

    loaded = load_relative_qkv_checkpoint(
        path,
        num_genes=5,
        node_covariate_dim=2,
        receiver_chunk_size=3,
    )
    assert loaded.model.receiver_chunk_size == 3
    assert loaded.model.training is False
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert _tree_sha256(loaded.model.state_dict()) == payload["model_state_checksum"]
    assert len(loaded.checkpoint_sha256) == 64

    corrupted = dict(payload)
    corrupted["model_state_checksum"] = "0" * 64
    bad_path = tmp_path / "bad.ckpt"
    torch.save(corrupted, bad_path)
    with pytest.raises(RelativeQKVPostTrainingError, match="checksum mismatch"):
        load_relative_qkv_checkpoint(
            bad_path,
            num_genes=5,
            node_covariate_dim=2,
        )


def test_fixed_mask_and_streamed_attention_are_deterministic_and_exact() -> None:
    batch = _batch()
    model = _model()
    first = fixed_inference_mask(batch)
    second = fixed_inference_mask(batch)
    assert first.seed == second.seed
    assert first.checksum_sha256 == second.checksum_sha256
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.n_masked_entries > 0

    streamed: list[tuple[np.ndarray, ...]] = []
    streamed_embeddings: list[np.ndarray] = []

    def consume(
        _start: int,
        _stop: int,
        edge_ids: np.ndarray,
        attention: np.ndarray,
        content: np.ndarray,
        bias: np.ndarray,
        combined: np.ndarray,
    ) -> None:
        streamed.append((edge_ids, attention, content, bias, combined))

    layer = stream_receiver_attention(
        model,
        batch,
        first.mask,
        layer=-1,
        consumer=consume,
        node_embedding_consumer=streamed_embeddings.append,
    )
    with torch.no_grad():
        standard = model(
            batch.target_expression,
            torch.from_numpy(np.array(first.mask, copy=True)),
            batch.edge_index,
            batch.relative_geometry,
            batch.node_covariates,
            return_explanations=True,
            explanation_layer=-1,
        )
    edge_ids = np.concatenate([values[0] for values in streamed])
    attention = np.concatenate([values[1] for values in streamed])
    content = np.concatenate([values[2] for values in streamed])
    bias = np.concatenate([values[3] for values in streamed])
    combined = np.concatenate([values[4] for values in streamed])
    order = np.argsort(edge_ids)
    assert layer == 1
    np.testing.assert_allclose(
        attention[order], standard.attention_weights.numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        content[order], standard.content_logits.numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        bias[order], standard.positional_bias.numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        combined[order], standard.combined_logits.numpy(), atol=1e-7
    )
    assert len(streamed_embeddings) == 1
    np.testing.assert_allclose(
        streamed_embeddings[0], standard.full_node_embedding.numpy(), atol=1e-7
    )


def test_edge_export_fields_use_degree_and_reciprocal_pair_contract() -> None:
    module = _load_attention_script()
    batch = _batch()
    edges = batch.edge_index.numpy()
    reciprocal, pair_key = reciprocal_edge_ids(edges, n_nodes=batch.n_nodes)
    for edge_id in range(batch.n_edges):
        reverse_id = int(reciprocal[edge_id])
        assert edges[0, reverse_id] == edges[1, edge_id]
        assert edges[1, reverse_id] == edges[0, edge_id]
        assert pair_key[reverse_id] == pair_key[edge_id]

    edge_ids = np.arange(3, dtype=np.int64)
    attention = np.asarray(
        [[0.2, 0.4], [0.3, 0.5], [0.5, 0.1]], dtype=np.float64
    )
    coordinates = np.asarray(
        [[0.0, 0.0], [3.0, 4.0], [0.0, 4.0], [3.0, 0.0]]
    )
    indegree = np.bincount(edges[1], minlength=batch.n_nodes)
    table, routing = module._edge_table(
        alias=batch.alias,
        layer_number=1,
        edge_ids=edge_ids,
        edge_index=edges,
        coordinates=coordinates,
        indegree=indegree,
        reciprocal_ids=reciprocal,
        pair_keys=pair_key,
        attention=attention,
        content=attention * 2,
        bias=-attention,
        combined=attention,
    )
    expected_mean = attention.mean(axis=1)
    np.testing.assert_allclose(routing, indegree[edges[1, edge_ids]] * expected_mean)
    assert table.column_names[-8:] == [
        "attention_head_00",
        "attention_head_01",
        "content_logit_head_00",
        "content_logit_head_01",
        "positional_bias_head_00",
        "positional_bias_head_01",
        "combined_logit_head_00",
        "combined_logit_head_01",
    ]
    np.testing.assert_allclose(
        table.column("content_logit_head_01").to_numpy(), attention[:, 1] * 2
    )
    np.testing.assert_allclose(
        table.column("positional_bias_head_00").to_numpy(), -attention[:, 0]
    )
    np.testing.assert_allclose(
        table.column("distance_um").to_numpy(),
        np.linalg.norm(
            coordinates[edges[0, edge_ids]] - coordinates[edges[1, edge_ids]],
            axis=1,
        ),
    )


def test_selected_autograd_matches_finite_difference() -> None:
    batch = _batch()
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mask = np.zeros((batch.n_nodes, batch.n_genes), dtype=np.bool_)
    mask[1, 3] = True
    request = SelectedDerivativeRequest(
        request_id="edge-0-to-1",
        core_alias=batch.alias,
        source_node=0,
        source_feature=2,
        receiver_node=1,
        target_feature=3,
        attention_head=None,
        layer=-1,
    )
    row = selected_autograd_derivatives(model, batch, mask, [request])[0]
    assert row["source_feature_observed"] is True
    assert row["target_feature_masked"] is True

    def replay(delta: float) -> tuple[float, float]:
        expression = batch.target_expression.clone()
        expression[0, 2] += delta
        with torch.no_grad():
            output = model(
                expression,
                torch.from_numpy(mask),
                batch.edge_index,
                batch.relative_geometry,
                batch.node_covariates,
                return_explanations=True,
                target_nodes=torch.tensor([1]),
                attention_receivers=torch.tensor([1]),
                explanation_layer=-1,
            )
        edges = output.edge_index.numpy()
        position = int(np.flatnonzero((edges[0] == 0) & (edges[1] == 1))[0])
        return (
            float(output.attention_weights[position].mean()),
            float(output.prediction[0, 3]),
        )

    epsilon = 1e-3
    positive = replay(epsilon)
    negative = replay(-epsilon)
    finite_attention = (positive[0] - negative[0]) / (2 * epsilon)
    finite_prediction = (positive[1] - negative[1]) / (2 * epsilon)
    assert row["d_attention_d_source_feature"] == pytest.approx(
        finite_attention, abs=2e-4, rel=2e-3
    )
    assert row["d_prediction_d_source_feature"] == pytest.approx(
        finite_prediction, abs=2e-4, rel=2e-3
    )

    masked_source = mask.copy()
    masked_source[0, 2] = True
    masked_row = selected_autograd_derivatives(
        model, batch, masked_source, [request]
    )[0]
    assert masked_row["source_feature_observed"] is False
    assert masked_row["d_attention_d_source_feature"] == 0.0
    assert masked_row["d_prediction_d_source_feature"] == 0.0


def test_selected_autograd_checkpointing_matches_uncheckpointed_with_frozen_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch()
    uncheckpointed = _model(activation_checkpointing=False)
    checkpointed = _model(activation_checkpointing=True)
    checkpointed.load_state_dict(uncheckpointed.state_dict(), strict=True)
    for model in (uncheckpointed, checkpointed):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    mask = np.zeros((batch.n_nodes, batch.n_genes), dtype=np.bool_)
    mask[1, 3] = True
    request = SelectedDerivativeRequest(
        request_id="edge-0-to-1-checkpoint-equivalence",
        core_alias=batch.alias,
        source_node=0,
        source_feature=2,
        receiver_node=1,
        target_feature=3,
        attention_head=None,
        layer=-1,
    )
    reference = selected_autograd_derivatives(
        uncheckpointed, batch, mask, [request]
    )[0]
    checkpoint_calls = 0
    original_checkpoint = relative_qkv_module.checkpoint

    def counted_checkpoint(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(*args, **kwargs)

    monkeypatch.setattr(relative_qkv_module, "checkpoint", counted_checkpoint)
    observed = selected_autograd_derivatives(
        checkpointed, batch, mask, [request]
    )[0]

    assert checkpoint_calls == 3
    numeric_fields = (
        "attention_value",
        "prediction_value",
        "d_attention_d_source_feature",
        "d_prediction_d_source_feature",
    )
    for field in numeric_fields:
        assert np.isfinite(reference[field])
        assert np.isfinite(observed[field])
        assert observed[field] == pytest.approx(reference[field], abs=1e-8, rel=1e-6)
    assert abs(observed["d_attention_d_source_feature"]) > 1e-8
    assert abs(observed["d_prediction_d_source_feature"]) > 1e-8

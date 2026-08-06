from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from spatial_benchmark.myjju_genemae import (
    EXPECTED_TRAINABLE_PARAMETERS_1000,
    GeneMAE,
    MaskedRegressionAccumulator,
    build_symmetric_knn_graph,
    count_parameters,
    gaussian_edge_features,
    log1p_cp10k,
    make_source_model,
    masked_regression_metrics,
    permute_graph_node_labels,
    recursive_spatial_tiles,
    sample_entry_mask,
)


class _PinnedReferenceResGATEncoder(nn.Module):
    """Frozen repaired-source reference, independent of the production class."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        embed: int,
        heads: int,
        layers: int,
        dropout: float,
        edge_dim: int,
        drop_edge_p: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_index in range(layers):
            layer_input = in_dim if layer_index == 0 else hidden
            self.convs.append(
                GATv2Conv(
                    layer_input,
                    hidden,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))
        self.jk = nn.Linear(hidden * layers, embed)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states: list[torch.Tensor] = []
        hidden = x
        for index, (convolution, normalisation) in enumerate(
            zip(self.convs, self.norms, strict=True)
        ):
            output = convolution(hidden, edge_index, edge_attr=edge_attr)
            output = normalisation(output)
            if index > 0:
                output = output + hidden
            output = F.elu(output)
            output = F.dropout(
                output,
                p=self.dropout,
                training=self.training,
            )
            hidden = output
            hidden_states.append(hidden)
        return self.jk(torch.cat(hidden_states, dim=1))


class _PinnedReferenceGeneMAE(nn.Module):
    """Selected source MLP-decoder path with only the frozen self fix."""

    def __init__(self, in_dim: int = 1000) -> None:
        super().__init__()
        hidden = 256
        embed = 192
        dropout = 0.2
        self.self_hidden = 512
        self.encoder = _PinnedReferenceResGATEncoder(
            in_dim,
            hidden,
            embed,
            heads=6,
            layers=4,
            dropout=dropout,
            edge_dim=1,
            drop_edge_p=0.2,
        )
        self.self_enc = nn.Sequential(
            nn.Linear(in_dim, self.self_hidden),
            nn.LayerNorm(self.self_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.self_hidden, self.self_hidden),
            nn.LayerNorm(self.self_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.self_hidden, embed),
        )
        self.dec_proj = nn.Linear(2 * embed, hidden)
        self.decoder = nn.Sequential(nn.ELU(), nn.Linear(hidden, in_dim))
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        entry_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_input = torch.where(
            entry_mask,
            self.mask_token.expand_as(x),
            x,
        )
        graph_embedding = self.encoder(
            masked_input,
            edge_index,
            edge_attr=edge_attr,
        )
        embedding = torch.cat(
            [graph_embedding, self.self_enc(masked_input)],
            dim=-1,
        )
        hidden = F.elu(self.dec_proj(embedding))
        return self.decoder(hidden), entry_mask


def test_source_constructor_has_repaired_self_branch_and_exact_parameter_count() -> None:
    model = make_source_model()

    assert model.self_hidden == 512
    assert model.gnn_decoder is False
    assert model.self_branch is True
    assert model.encoder.jk.in_features == 4 * 256
    assert model.encoder.jk.out_features == 192
    assert all(convolution.add_self_loops for convolution in model.encoder.convs)
    assert count_parameters(model) == EXPECTED_TRAINABLE_PARAMETERS_1000
    assert count_parameters(model) == 6_888_016


def test_repaired_source_state_schema_and_forward_are_exact() -> None:
    torch.manual_seed(41)
    reference = _PinnedReferenceGeneMAE()
    local = make_source_model()
    reference_schema = {
        name: tuple(value.shape)
        for name, value in reference.state_dict().items()
    }
    local_schema = {
        name: tuple(value.shape)
        for name, value in local.state_dict().items()
    }

    assert local_schema == reference_schema
    local.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    local.eval()

    x = torch.linspace(0.0, 1.0, steps=4_000).reshape(4, 1_000)
    edge_index = torch.asarray(
        [
            [0, 1, 0, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 3, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.linspace(0.2, 0.9, edge_index.shape[1]).unsqueeze(1)
    entry_mask = torch.zeros_like(x, dtype=torch.bool)
    entry_mask[:, ::7] = True

    with torch.no_grad():
        expected, expected_mask = reference(
            x,
            edge_index,
            edge_attr,
            entry_mask,
        )
        observed, observed_mask = local(
            x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
        )

    assert torch.equal(observed_mask, expected_mask)
    assert torch.equal(observed, expected)


def test_entry_mask_sampling_is_deterministic_and_never_empty() -> None:
    first = sample_entry_mask((7, 11), rate=0.2, seed=14)
    second = sample_entry_mask((7, 11), rate=0.2, seed=14)
    other = sample_entry_mask((7, 11), rate=0.2, seed=15)
    zero_rate = sample_entry_mask((2, 3), rate=0.0, seed=1)

    assert first.dtype == torch.bool
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert int(zero_rate.sum()) == 1


def test_log1p_cp10k_matches_full_cell_normalisation_for_numpy_and_torch() -> None:
    counts = np.asarray(
        [
            [1, 1, 2],
            [0, 0, 0],
            [4, 0, 0],
        ],
        dtype=np.int64,
    )
    expected = np.log1p(
        np.asarray(
            [
                [2500, 2500, 5000],
                [0, 0, 0],
                [10000, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    observed_numpy = log1p_cp10k(counts)
    observed_torch = log1p_cp10k(torch.from_numpy(counts))

    assert isinstance(observed_numpy, np.ndarray)
    assert observed_numpy.dtype == np.float32
    np.testing.assert_allclose(observed_numpy, expected, rtol=1e-6)
    assert isinstance(observed_torch, torch.Tensor)
    assert observed_torch.dtype == torch.float32
    torch.testing.assert_close(observed_torch, torch.from_numpy(expected))

    with pytest.raises(ValueError, match="nonnegative"):
        log1p_cp10k(np.asarray([[1, -1]], dtype=np.float32))


def test_recursive_spatial_tiles_are_deterministic_bounded_partition() -> None:
    coords = np.column_stack(
        [
            np.repeat(np.arange(5, dtype=np.float64), 5),
            np.tile(np.arange(5, dtype=np.float64), 5),
        ]
    )

    first = recursive_spatial_tiles(coords, max_nodes=4)
    second = recursive_spatial_tiles(coords.copy(), max_nodes=4)

    assert len(first) > 1
    assert all(tile.dtype == np.int64 for tile in first)
    assert all(1 <= tile.size <= 4 for tile in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    np.testing.assert_array_equal(
        np.sort(np.concatenate(first)),
        np.arange(coords.shape[0]),
    )
    assert np.unique(np.concatenate(first)).size == coords.shape[0]

    tied = np.zeros((9, 2), dtype=np.float64)
    tied_tiles = recursive_spatial_tiles(tied, max_nodes=3)
    assert all(tile.size <= 3 for tile in tied_tiles)
    np.testing.assert_array_equal(
        np.concatenate(tied_tiles),
        np.arange(tied.shape[0]),
    )


def test_symmetric_knn_graph_and_gaussian_edge_features_obey_invariants() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )

    edge_index, edge_attr = build_symmetric_knn_graph(coords, k=2)
    pairs = {tuple(pair) for pair in edge_index.T.tolist()}

    assert edge_index.dtype == np.int64
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert edge_attr.dtype == np.float32
    assert len(pairs) == edge_index.shape[1]
    assert all(source != target for source, target in pairs)
    assert all((target, source) in pairs for source, target in pairs)
    degree = np.bincount(edge_index[0], minlength=coords.shape[0])
    assert np.all(degree >= 2)
    assert np.all((edge_attr > 0) & (edge_attr <= 1))

    recomputed = gaussian_edge_features(coords, edge_index)
    np.testing.assert_array_equal(edge_attr, recomputed)
    edge_lookup = {
        tuple(pair): float(value)
        for pair, value in zip(edge_index.T, edge_attr[:, 0], strict=True)
    }
    for source, target in pairs:
        assert edge_lookup[(source, target)] == pytest.approx(
            edge_lookup[(target, source)]
        )


def test_graph_node_relabelling_is_deterministic_and_degree_preserving() -> None:
    coords = np.column_stack(
        [
            np.arange(10, dtype=np.float64),
            np.zeros(10, dtype=np.float64),
        ]
    )
    edge_index, _ = build_symmetric_knn_graph(coords, k=2)

    first = permute_graph_node_labels(edge_index, num_nodes=10, seed=23)
    second = permute_graph_node_labels(edge_index, num_nodes=10, seed=23)
    other = permute_graph_node_labels(edge_index, num_nodes=10, seed=24)

    assert isinstance(first, np.ndarray)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    original_degree = np.bincount(edge_index.ravel(), minlength=10)
    permuted_degree = np.bincount(first.ravel(), minlength=10)
    np.testing.assert_array_equal(
        np.sort(original_degree),
        np.sort(permuted_degree),
    )

    permutation = np.random.default_rng(23).permutation(10)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    np.testing.assert_array_equal(inverse[first], edge_index)

    tensor_edges = torch.from_numpy(edge_index)
    tensor_permuted = permute_graph_node_labels(
        tensor_edges,
        num_nodes=10,
        seed=23,
    )
    assert isinstance(tensor_permuted, torch.Tensor)
    torch.testing.assert_close(tensor_permuted, torch.from_numpy(first))


def test_small_model_forward_and_masked_huber_loss_are_finite() -> None:
    torch.manual_seed(3)
    model = GeneMAE(
        in_dim=5,
        hidden=8,
        embed=4,
        heads=2,
        layers=2,
        dropout=0.0,
        drop_edge_p=0.0,
        gnn_decoder=False,
        self_branch=True,
        self_hidden=7,
    )
    model.eval()
    x = torch.rand(6, 5)
    coords = np.column_stack(
        [np.arange(6, dtype=np.float64), np.zeros(6, dtype=np.float64)]
    )
    edge_index_numpy, edge_attr_numpy = build_symmetric_knn_graph(coords, k=2)
    edge_index = torch.from_numpy(edge_index_numpy)
    edge_attr = torch.from_numpy(edge_attr_numpy)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, ::2] = True

    reconstruction, returned_mask = model(
        x,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=mask,
    )
    loss = model.loss(
        x,
        edge_index,
        edge_attr=edge_attr,
        entry_mask=mask,
    )
    expected = F.huber_loss(reconstruction[mask], x[mask], delta=1.0)

    assert reconstruction.shape == x.shape
    assert torch.equal(returned_mask, mask)
    assert torch.isfinite(reconstruction).all()
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_masked_common_metrics_are_streaming_equivalent() -> None:
    target = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 5.0],
            [2.0, 5.0, 8.0],
            [3.0, 7.0, 11.0],
        ],
        dtype=np.float32,
    )
    prediction = target.copy()
    mask = np.ones_like(target, dtype=bool)

    one_shot = masked_regression_metrics(target, prediction, mask)
    streaming = MaskedRegressionAccumulator(num_genes=3)
    streaming.update(target[:2], prediction[:2], mask[:2])
    streaming.update(target[2:], prediction[2:], mask[2:])
    streamed = streaming.finalize()

    assert one_shot == streamed
    assert one_shot["n_masked"] == target.size
    assert one_shot["masked_huber"] == pytest.approx(0.0)
    assert one_shot["masked_mse"] == pytest.approx(0.0)
    assert one_shot["masked_mae"] == pytest.approx(0.0)
    assert one_shot["pooled_pearson"] == pytest.approx(1.0)
    assert one_shot["masked_r2"] == pytest.approx(1.0)
    assert one_shot["gene_pearson_mean"] == pytest.approx(1.0)
    assert one_shot["gene_pearson_median"] == pytest.approx(1.0)
    assert one_shot["cell_pearson_mean"] == pytest.approx(1.0)

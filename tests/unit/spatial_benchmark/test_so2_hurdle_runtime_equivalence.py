"""Numerical controls for changing only the hurdle target-rank partition.

These CPU tests simulate DDP's arithmetic gradient average; they do not test
the communication backend or claim equal stochastic dropout trajectories.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch

from spatial_benchmark.hurdle_negative_binomial import (
    HurdleNegativeBinomialModelOutput,
    TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    masked_hurdle_negative_binomial_nll_sums,
)
from spatial_benchmark.target_cell_masking import (
    make_contiguous_target_partitions,
    make_training_target_cell_masks,
)


@dataclass(frozen=True)
class _Core:
    expression: torch.Tensor
    covariates: torch.Tensor
    edges: torch.Tensor
    geometry: torch.Tensor
    counts: torch.Tensor
    mask: torch.Tensor


def _core(n_cells: int, seed: int) -> _Core:
    generator = torch.Generator().manual_seed(seed)
    receivers = torch.arange(n_cells).repeat_interleave(n_cells - 1)
    senders = torch.tensor(
        [sender for receiver in range(n_cells)
         for sender in range(n_cells) if sender != receiver],
        dtype=torch.long,
    )
    masks = make_training_target_cell_masks(
        n_cells,
        4,
        base_seed=0,
        namespace="runtime_equivalence_synthetic",
        core_alias=f"synthetic-{seed}",
        global_epoch=1,
    )
    counts = torch.arange(n_cells * 4).reshape(n_cells, 4) % 5
    return _Core(
        expression=torch.randn(n_cells, 4, generator=generator),
        covariates=torch.randn(n_cells, 2, generator=generator),
        edges=torch.stack((senders, receivers)),
        geometry=torch.randn(len(senders), 70, generator=generator),
        counts=counts,
        mask=torch.from_numpy(masks.mask.copy()),
    )


def _model() -> TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer:
    torch.manual_seed(51901)
    return TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer(
        num_genes=4,
        node_covariate_dim=2,
        hidden_dim=8,
        attention_heads=2,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=16,
        decoder_dim=12,
        geometry_hidden_dim=6,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        max_edges_per_chunk=16,
        activation_checkpointing=True,
    )


def _forward(
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    core: _Core,
    start: int,
    stop: int,
) -> HurdleNegativeBinomialModelOutput:
    return model(
        input_expression=core.expression,
        target_input_expression=core.expression[start:stop].masked_fill(
            core.mask[start:stop], 0.0
        ),
        target_gene_mask=core.mask[start:stop],
        edge_index=core.edges,
        relative_geometry=core.geometry,
        node_covariates=core.covariates,
        target_start=start,
        target_stop=stop,
    )


def _loss_sum(
    output: HurdleNegativeBinomialModelOutput,
    core: _Core,
    start: int,
    stop: int,
) -> torch.Tensor:
    return masked_hurdle_negative_binomial_nll_sums(
        output.nonzero_logits,
        output.nb_mean,
        output.theta,
        core.counts[start:stop],
        core.mask[start:stop],
    ).joint_nll_sum


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_fixed_weight_predictions_and_paired_pooled_gradient_match_partitions(
    world_size: int,
) -> None:
    """Two unequal cores preserve the pooled objective for 1/2/4 ranks."""

    cores = (_core(5, 19), _core(7, 22))
    base = _model()
    global_count = sum(int(core.mask.sum()) for core in cores)
    assert int(cores[0].mask.sum()) != int(cores[1].mask.sum())
    partitions = [
        make_contiguous_target_partitions(len(core.expression), world_size)
        for core in cores
    ]
    rank_counts = [
        sum(int(core.mask[parts[rank].start:parts[rank].stop].sum())
            for core, parts in zip(cores, partitions, strict=True))
        for rank in range(world_size)
    ]
    assert sum(rank_counts) == global_count
    if world_size > 1:
        assert len(set(rank_counts)) > 1

    # Evaluation must be independent of target partitioning at fixed weights.
    base.eval()
    with torch.no_grad():
        for core, parts in zip(cores, partitions, strict=True):
            full = _forward(base, core, 0, len(core.expression))
            shards = [_forward(base, core, part.start, part.stop) for part in parts]
            for attribute in ("nonzero_logits", "nb_mean", "marginal_mean"):
                torch.testing.assert_close(
                    torch.cat([getattr(shard, attribute) for shard in shards]),
                    getattr(full, attribute),
                    rtol=2e-5,
                    atol=2e-6,
                )

    # Both cores contribute before the optimizer step. The denominator is the
    # combined number of hidden gene entries, not cells, cores, or local means.
    reference = copy.deepcopy(base).train()
    reference_loss = 0.0
    for core in cores:
        summed = _loss_sum(
            _forward(reference, core, 0, len(core.expression)),
            core, 0, len(core.expression),
        )
        reference_loss += float(summed.detach()) / global_count
        (summed / global_count).backward()
    expected_gradients = {}
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, name
        expected_gradients[name] = parameter.grad.detach().clone()

    average_gradients = {
        name: torch.zeros_like(value) for name, value in expected_gradients.items()
    }
    partition_loss = 0.0
    incorrectly_unweighted_loss = 0.0
    for rank in range(world_size):
        replica = copy.deepcopy(base).train()
        local_sum = 0.0
        for core, parts in zip(cores, partitions, strict=True):
            part = parts[rank]
            summed = _loss_sum(
                _forward(replica, core, part.start, part.stop),
                core, part.start, part.stop,
            )
            local_sum += float(summed.detach())
            # DDP averages gradients; this multiplier recovers the pooled sum.
            (summed * (world_size / global_count)).backward()
        partition_loss += local_sum / global_count
        incorrectly_unweighted_loss += local_sum / rank_counts[rank] / world_size
        for name, parameter in replica.named_parameters():
            assert parameter.grad is not None, name
            average_gradients[name] += parameter.grad / world_size

    assert partition_loss == pytest.approx(reference_loss, rel=2e-6, abs=2e-6)
    for name, expected in expected_gradients.items():
        assert bool(torch.isfinite(average_gradients[name]).all()), name
        torch.testing.assert_close(
            average_gradients[name], expected, rtol=3e-5, atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    # This fixture discriminates against the tempting, incorrect local-mean
    # implementation rather than allowing it to pass accidentally.
    if world_size > 1:
        assert abs(incorrectly_unweighted_loss - reference_loss) > 1e-4

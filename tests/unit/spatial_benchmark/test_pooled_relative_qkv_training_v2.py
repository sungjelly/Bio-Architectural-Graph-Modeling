from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import nn

from spatial_benchmark.pooled_relative_qkv_training import (
    CAN_ALIASES,
    PooledRelativeQKVCoreBatch,
    PooledRelativeQKVTrainingConfig,
    fit_pooled_relative_qkv_segment,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (
    COHORT_RESUME_SCHEMA_V2,
    SO2_14CORE_ALIASES,
    CohortRelativeQKVCoreBatch,
    CohortRelativeQKVTrainingConfig,
    cohort_epoch_boundary_resume_from_checkpoint,
    cohort_relative_qkv_core_order,
    cohort_relative_qkv_distributed_assignment,
    cohort_relative_qkv_mask_seed,
    cohort_relative_qkv_model_step_seed,
    fit_cohort_relative_qkv_segment,
    make_cohort_exact_uniform_training_mask,
)


class _TinyModel(nn.Module):
    def __init__(self, n_genes: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.linspace(-0.1, 0.1, n_genes))
        self.dropout = nn.Dropout(dropout)
        self.forward_calls = 0
        self.update_start_weights: list[float] = []

    def forward(
        self,
        *,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        edge_index: torch.Tensor,
        relative_geometry: torch.Tensor,
        node_covariates: torch.Tensor,
        target_nodes: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, relative_geometry, target_nodes
        assert torch.all(input_expression[gene_mask] == 0)
        if self.forward_calls % 20 == 0:
            self.update_start_weights.append(float(self.weight.detach()))
        self.forward_calls += 1
        context = self.dropout(input_expression).mean(dim=1, keepdim=True)
        metadata = node_covariates.mean(dim=1, keepdim=True) * 0.01
        return self.bias.unsqueeze(0) + self.weight * context + metadata


def _batches(
    aliases: tuple[str, ...] = SO2_14CORE_ALIASES,
    *,
    n_nodes: int = 2,
    n_genes: int = 3,
) -> tuple[CohortRelativeQKVCoreBatch, ...]:
    result = []
    for index, alias in enumerate(aliases):
        expression = (
            torch.arange(1, n_nodes * n_genes + 1, dtype=torch.float32)
            .reshape(n_nodes, n_genes)
            .div(10)
            .add(index * 0.01)
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        result.append(
            CohortRelativeQKVCoreBatch(
                alias=alias,
                target_expression=expression,
                edge_index=edge_index,
                relative_geometry=torch.ones((2, 70)),
                node_covariates=torch.full((n_nodes, 2), float(index + 1)),
            )
        )
    return tuple(result)


def _config(
    *, seed: int = 0, start: int = 0, end: int = 1
) -> CohortRelativeQKVTrainingConfig:
    return CohortRelativeQKVTrainingConfig(
        model_seed=seed,
        segment_start_global_epoch=start,
        segment_end_global_epoch=end,
        learning_rate=0.01,
        device="cpu",
    )


def _four_rank_gloo_worker(
    rank: int,
    rendezvous_path: str,
    result_queue: object,
) -> None:
    """Exercise the production partition under a bounded CPU process group."""

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=4,
    )
    try:
        torch.manual_seed(91)
        model = _TinyModel(3)
        result = fit_cohort_relative_qkv_segment(
            model,
            _batches(),
            CohortRelativeQKVTrainingConfig(
                model_seed=0,
                segment_end_global_epoch=1,
                learning_rate=0.01,
                device="cpu",
                distributed_world_size=4,
                distributed_rank=rank,
            ),
        )
        result_queue.put(  # type: ignore[attr-defined]
            (
                rank,
                float(model.weight.detach()),
                model.bias.detach().tolist(),
                result.history_checksum,
                len(result.core_history),
                len(result.optimizer_update_history),
            )
        )
    finally:
        dist.destroy_process_group()


def test_one_epoch_has_14_core_records_7_updates_and_140_views() -> None:
    model = _TinyModel(3)
    result = fit_cohort_relative_qkv_segment(model, _batches(), _config())

    assert len(result.core_history) == 14
    assert len(result.optimizer_update_history) == 7
    assert len(result.global_history) == 1
    assert result.optimizer_updates_completed == 7
    assert result.optimizer_updates_this_segment == 7
    assert result.losses_per_optimizer_update == 20
    assert model.forward_calls == 140
    assert result.maximum_simultaneously_staged_cores == 1
    assert tuple(record.alias for record in result.core_history) == (
        cohort_relative_qkv_core_order(0)
    )
    assert all(record.n_mask_views == 10 for record in result.core_history)
    assert all(update.n_cores == 2 for update in result.optimizer_update_history)
    assert all(
        update.losses_averaged == 20
        for update in result.optimizer_update_history
    )
    assert len(model.update_start_weights) == 7
    assert len(set(model.update_start_weights)) > 1


def test_masks_and_order_are_model_seed_independent_but_views_differ() -> None:
    batch = _batches()[0]
    checksums = []
    for view in range(10):
        left = make_cohort_exact_uniform_training_mask(batch, 8, view_index=view)
        right = make_cohort_exact_uniform_training_mask(batch, 8, view_index=view)
        np.testing.assert_array_equal(left.mask, right.mask)
        assert left.checksum_sha256 == right.checksum_sha256
        checksums.append(left.checksum_sha256)
    assert len(set(checksums)) == 10
    assert cohort_relative_qkv_mask_seed(batch.alias, 8, view_index=0) == (
        cohort_relative_qkv_mask_seed(batch.alias, 8, view_index=0)
    )
    assert cohort_relative_qkv_model_step_seed(0, 8, 0, batch.alias, 0) != (
        cohort_relative_qkv_model_step_seed(1, 8, 0, batch.alias, 0)
    )
    assert cohort_relative_qkv_model_step_seed(0, 8, 0, batch.alias, 0) != (
        cohort_relative_qkv_model_step_seed(0, 8, 0, batch.alias, 1)
    )


def test_mask_counts_cover_the_full_uniform_integer_support() -> None:
    batch = CohortRelativeQKVCoreBatch(
        alias="SO2-C15",
        target_expression=torch.ones((4096, 8)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        relative_geometry=torch.empty((0, 70)),
        node_covariates=torch.ones((4096, 2)),
    )
    realization = make_cohort_exact_uniform_training_mask(batch, 0, view_index=0)
    counts = realization.masked_gene_counts
    assert set(counts.tolist()) == set(range(9))
    assert np.array_equal(realization.mask.sum(axis=1), counts)


def test_four_rank_partition_is_exact_disjoint_20_loss_cover() -> None:
    pair = ("SO2-C15", "SO2-C16")
    assignments = tuple(
        cohort_relative_qkv_distributed_assignment(pair, rank=rank, world_size=4)
        for rank in range(4)
    )
    assert tuple(item.alias for item in assignments) == (
        "SO2-C15",
        "SO2-C15",
        "SO2-C16",
        "SO2-C16",
    )
    assert assignments[0].view_indices == tuple(range(5))
    assert assignments[1].view_indices == tuple(range(5, 10))
    assert assignments[2].view_indices == tuple(range(5))
    assert assignments[3].view_indices == tuple(range(5, 10))
    assert all(item.local_loss_divisor == 5 for item in assignments)
    labelled = {
        (item.alias, view)
        for item in assignments
        for view in item.view_indices
    }
    assert len(labelled) == 20

    # DDP averages four rank-local means. This is algebraically the same as
    # one process averaging all 20 gradients.
    gradients = np.arange(1, 21, dtype=np.float64)
    rank_means = np.asarray(
        [gradients[index * 5 : (index + 1) * 5].mean() for index in range(4)]
    )
    assert rank_means.mean() == pytest.approx(gradients.mean())


def test_four_rank_gloo_matches_single_process_within_reduction_tolerance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spawned children must not inherit visibility of production GPUs for this
    # bounded CPU collective test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rendezvous = tmp_path / "gloo-rendezvous"
    context = torch_mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    torch_mp.spawn(
        _four_rank_gloo_worker,
        args=(str(rendezvous), result_queue),
        nprocs=4,
        join=True,
    )
    rows = sorted(result_queue.get() for _ in range(4))

    assert [row[0] for row in rows] == list(range(4))
    assert all(row[4:] == (14, 7) for row in rows)
    assert len({row[3] for row in rows}) == 1
    for row in rows[1:]:
        assert row[1] == pytest.approx(rows[0][1], rel=0, abs=0)
        np.testing.assert_array_equal(row[2], rows[0][2])

    torch.manual_seed(91)
    single_model = _TinyModel(3)
    single = fit_cohort_relative_qkv_segment(
        single_model, _batches(), _config()
    )
    assert len(single.core_history) == 14
    assert float(single_model.weight.detach()) == pytest.approx(
        rows[0][1], rel=1e-6, abs=1e-7
    )
    np.testing.assert_allclose(
        single_model.bias.detach().numpy(),
        np.asarray(rows[0][2]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_single_process_update_matches_manual_mean_of_20_gradients() -> None:
    batches = _batches()
    order = cohort_relative_qkv_core_order(0)
    by_alias = {batch.alias: batch for batch in batches}
    first_pair = (by_alias[order[0]], by_alias[order[1]])

    manual = _TinyModel(3)
    optimizer = torch.optim.AdamW(manual.parameters(), lr=0.01, weight_decay=1e-5)
    optimizer.zero_grad(set_to_none=True)
    for batch in first_pair:
        for view in range(10):
            realization = make_cohort_exact_uniform_training_mask(
                batch, 0, view_index=view
            )
            mask = torch.from_numpy(np.array(realization.mask, copy=True))
            seed = cohort_relative_qkv_model_step_seed(0, 0, 0, batch.alias, view)
            torch.manual_seed(seed)
            prediction = manual(
                input_expression=batch.target_expression.masked_fill(mask, 0),
                gene_mask=mask,
                edge_index=batch.edge_index,
                relative_geometry=batch.relative_geometry,
                node_covariates=batch.node_covariates,
                target_nodes=torch.arange(batch.n_nodes),
            )
            elementwise = torch.nn.functional.huber_loss(
                prediction, batch.target_expression, reduction="none", delta=1.0
            )
            (elementwise.masked_select(mask).mean() / 20.0).backward()
    torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.0)
    optimizer.step()

    trained = _TinyModel(3)
    fit_cohort_relative_qkv_segment(trained, batches, _config())
    assert trained.update_start_weights[1] == pytest.approx(
        float(manual.weight.detach()), abs=1e-8
    )


def test_v2_epoch_boundary_resume_is_identical_and_serializable() -> None:
    batches = _batches()
    torch.manual_seed(44)
    uninterrupted = fit_cohort_relative_qkv_segment(
        _TinyModel(3, dropout=0.2), batches, _config(seed=3, end=2)
    )

    torch.manual_seed(44)
    first = fit_cohort_relative_qkv_segment(
        _TinyModel(3, dropout=0.2), batches, _config(seed=3, end=1)
    )
    payload = asdict(first.resume)
    payload["schema"] = COHORT_RESUME_SCHEMA_V2
    payload["amp_scaler_state_dict"] = payload.pop("scaler_state_dict")
    payload["amp_scaler_state_checksum"] = payload.pop("scaler_state_checksum")
    restored = cohort_epoch_boundary_resume_from_checkpoint(payload)
    continued = fit_cohort_relative_qkv_segment(
        _TinyModel(3, dropout=0.2),
        batches,
        _config(seed=3, start=1, end=2),
        resume=restored,
    )
    assert continued.final_state_checksum == uninterrupted.final_state_checksum
    assert continued.history_checksum == uninterrupted.history_checksum
    assert continued.resume.resume_checksum == uninterrupted.resume.resume_checksum
    assert continued.core_history == uninterrupted.core_history
    assert continued.optimizer_update_history == uninterrupted.optimizer_update_history


def test_epoch_callback_precedes_checkpoint_callback() -> None:
    events: list[str] = []

    def observe_epoch(
        epoch,
        cores,
        updates,
        duration_seconds,
        peak_vram_gib,
    ) -> None:
        assert epoch.core_records_this_epoch == 14
        assert len(cores) == 14
        assert len(updates) == 7
        assert duration_seconds >= 0
        assert peak_vram_gib == 0
        events.append("epoch")

    fit_cohort_relative_qkv_segment(
        _TinyModel(3),
        _batches(),
        _config(),
        epoch_callback=observe_epoch,
        checkpoint_callback=lambda _resume: events.append("checkpoint"),
    )
    assert events == ["epoch", "checkpoint"]


def test_six_core_v1_contract_still_runs_six_updates() -> None:
    batches = []
    for index, alias in enumerate(CAN_ALIASES):
        cohort = _batches((alias,))[0]
        batches.append(
            PooledRelativeQKVCoreBatch(
                alias=alias,
                target_expression=cohort.target_expression,
                edge_index=cohort.edge_index,
                relative_geometry=cohort.relative_geometry,
                node_covariates=cohort.node_covariates,
            )
        )
    result = fit_pooled_relative_qkv_segment(
        _TinyModel(3),
        tuple(batches),
        PooledRelativeQKVTrainingConfig(
            model_seed=0,
            segment_start_global_epoch=0,
            segment_end_global_epoch=1,
            learning_rate=0.01,
            device="cpu",
        ),
    )
    assert result.optimizer_steps_completed == 6
    assert len(result.core_history) == 6

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from spatial_benchmark.pooled_relative_qkv_training import (
    CAN_ALIASES,
    MASK_BASE_SEED,
    MASK_VIEWS_PER_CORE_STEP,
    PooledRelativeQKVCoreBatch,
    PooledRelativeQKVTrainingConfig,
    PooledRelativeQKVTrainingError,
    epoch_boundary_resume_from_checkpoint,
    fit_pooled_relative_qkv_segment,
    joint_five_seed_plateau_decision,
    make_exact_uniform_training_mask,
    masked_huber_reconstruction_loss,
    relative_qkv_core_order,
    relative_qkv_mask_seed,
    single_seed_plateau_decision,
)


class _TinyRelativeModel(nn.Module):
    def __init__(self, n_genes: int, *, dropout: float = 0.2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.linspace(-0.1, 0.1, n_genes))
        self.dropout = nn.Dropout(dropout)
        self.forward_calls = 0
        self.parameter_ids: list[int] = []
        self.step_start_weights: list[float] = []
        self.step_start_biases: list[torch.Tensor] = []
        self.seen_devices: list[str] = []
        self.seen_edge_devices: list[str] = []
        self.seen_geometry_devices: list[str] = []

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
        assert torch.all(input_expression[gene_mask] == 0)
        assert torch.equal(
            target_nodes,
            torch.arange(len(input_expression), device=target_nodes.device),
        )
        if self.forward_calls % MASK_VIEWS_PER_CORE_STEP == 0:
            self.step_start_weights.append(float(self.weight.detach().cpu()))
            self.step_start_biases.append(self.bias.detach().cpu().clone())
        self.forward_calls += 1
        self.parameter_ids.append(id(self.weight))
        self.seen_devices.append(input_expression.device.type)
        self.seen_edge_devices.append(edge_index.device.type)
        self.seen_geometry_devices.append(relative_geometry.device.type)
        observed_context = self.dropout(input_expression).mean(dim=1, keepdim=True)
        metadata_context = node_covariates.mean(dim=1, keepdim=True) * 0.01
        return (
            self.bias.unsqueeze(0)
            + self.weight * observed_context
            + metadata_context
        )


def _core_batches(
    *,
    n_nodes: int = 2,
    n_genes: int = 3,
) -> tuple[PooledRelativeQKVCoreBatch, ...]:
    batches = []
    for alias_index, alias in enumerate(CAN_ALIASES):
        expression = (
            torch.arange(1, n_nodes * n_genes + 1, dtype=torch.float32)
            .reshape(n_nodes, n_genes)
            .div(10.0)
            .add(alias_index * 0.05)
        )
        if n_nodes == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            source = torch.arange(n_nodes - 1, dtype=torch.long)
            receiver = source + 1
            edge_index = torch.cat(
                [
                    torch.stack([source, receiver]),
                    torch.stack([receiver, source]),
                ],
                dim=1,
            )
        relative_geometry = torch.ones(
            (edge_index.shape[1], 70), dtype=torch.float32
        )
        node_covariates = torch.full(
            (n_nodes, 2), float(alias_index + 1), dtype=torch.float32
        )
        batches.append(
            PooledRelativeQKVCoreBatch(
                alias=alias,
                target_expression=expression,
                edge_index=edge_index,
                relative_geometry=relative_geometry,
                node_covariates=node_covariates,
            )
        )
    return tuple(batches)


def _config(
    *,
    model_seed: int,
    start: int = 0,
    end: int = 1,
) -> PooledRelativeQKVTrainingConfig:
    return PooledRelativeQKVTrainingConfig(
        model_seed=model_seed,
        segment_start_global_epoch=start,
        segment_end_global_epoch=end,
        learning_rate=0.01,
        device="cpu",
        deterministic=True,
        deterministic_warn_only=False,
    )


def test_ten_mask_views_are_paired_across_model_seeds() -> None:
    batch = _core_batches()[0]
    seed_zero_config = _config(model_seed=0)
    seed_four_config = _config(model_seed=4)
    observed_zero = []
    observed_four = []
    for view_index in range(MASK_VIEWS_PER_CORE_STEP):
        first = make_exact_uniform_training_mask(
            batch,
            7,
            view_index=view_index,
            mask_base_seed=seed_zero_config.mask_base_seed,
        )
        second = make_exact_uniform_training_mask(
            batch,
            7,
            view_index=view_index,
            mask_base_seed=seed_four_config.mask_base_seed,
        )
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(
            first.masked_gene_counts, second.masked_gene_counts
        )
        assert first.checksum_sha256 == second.checksum_sha256
        observed_zero.append(first.effective_seed)
        observed_four.append(second.effective_seed)
    assert observed_zero == observed_four
    assert len(set(observed_zero)) == MASK_VIEWS_PER_CORE_STEP


def test_mask_seed_uses_alias_epoch_and_view_but_not_model_seed() -> None:
    seeds = {
        relative_qkv_mask_seed("CAN-01", 2, view_index=view)
        for view in range(MASK_VIEWS_PER_CORE_STEP)
    }
    assert len(seeds) == MASK_VIEWS_PER_CORE_STEP
    assert relative_qkv_mask_seed("CAN-01", 2, view_index=0) != (
        relative_qkv_mask_seed("CAN-09", 2, view_index=0)
    )
    assert relative_qkv_mask_seed("CAN-01", 2, view_index=0) != (
        relative_qkv_mask_seed("CAN-01", 3, view_index=0)
    )
    with pytest.raises(PooledRelativeQKVTrainingError, match="0 through 9"):
        relative_qkv_mask_seed("CAN-01", 2, view_index=10)


def test_exact_uniform_masks_have_full_integer_support() -> None:
    n_cells = 4096
    n_genes = 8
    batch = PooledRelativeQKVCoreBatch(
        alias="CAN-01",
        target_expression=torch.ones((n_cells, n_genes), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        relative_geometry=torch.empty((0, 70), dtype=torch.float32),
        node_covariates=torch.ones((n_cells, 2), dtype=torch.float32),
    )
    realization = make_exact_uniform_training_mask(batch, 0, view_index=0)
    counts = realization.masked_gene_counts
    frequencies = np.bincount(counts, minlength=n_genes + 1)
    expected_frequency = n_cells / float(n_genes + 1)

    assert set(counts.tolist()) == set(range(n_genes + 1))
    assert counts.min() == 0
    assert counts.max() == n_genes
    assert np.array_equal(realization.mask.sum(axis=1), counts)
    assert np.max(np.abs(frequencies - expected_frequency)) < (
        0.2 * expected_frequency
    )


def test_global_epoch_has_six_optimizer_steps_and_ten_views_per_core() -> None:
    batches = _core_batches()
    model = _TinyRelativeModel(3)
    initial_weight = model.weight.detach().clone()
    result = fit_pooled_relative_qkv_segment(model, batches, _config(model_seed=3))

    assert result.optimizer_steps_this_segment == 6
    assert result.optimizer_steps_completed == 6
    assert len(result.core_history) == 6
    assert len(result.global_history) == 1
    assert result.global_history[0].ordered_aliases == relative_qkv_core_order(0)
    assert set(record.alias for record in result.core_history) == set(CAN_ALIASES)
    assert all(record.n_mask_views == 10 for record in result.core_history)
    assert all(len(record.mask_views) == 10 for record in result.core_history)
    assert all(
        tuple(view.view_index for view in record.mask_views) == tuple(range(10))
        for record in result.core_history
    )
    assert model.forward_calls == 60
    assert len(set(model.parameter_ids)) == 1
    assert not torch.equal(initial_weight, model.weight.detach())
    # The same shared parameter changes between complete-core optimizer steps.
    assert len(model.step_start_weights) == 6
    assert len(set(model.step_start_weights)) > 1


def test_only_one_complete_core_is_staged_and_sources_remain_on_cpu() -> None:
    batches = _core_batches()
    model = _TinyRelativeModel(3, dropout=0.0)
    result = fit_pooled_relative_qkv_segment(model, batches, _config(model_seed=1))
    assert result.maximum_simultaneously_staged_cores == 1
    assert set(model.seen_devices) == {"cpu"}
    assert set(model.seen_edge_devices) == {"cpu"}
    assert set(model.seen_geometry_devices) == {"cpu"}
    for batch in batches:
        assert batch.target_expression.device.type == "cpu"
        assert batch.edge_index.device.type == "cpu"
        assert batch.relative_geometry.device.type == "cpu"
        assert batch.node_covariates.device.type == "cpu"


def test_core_step_gradient_is_exact_mean_of_ten_view_gradients() -> None:
    batches = _core_batches()
    first_alias = relative_qkv_core_order(0)[0]
    batch = {candidate.alias: candidate for candidate in batches}[first_alias]
    manual_model = _TinyRelativeModel(3, dropout=0.0)
    manual_optimizer = torch.optim.AdamW(
        manual_model.parameters(), lr=0.01, weight_decay=1e-5
    )
    manual_optimizer.zero_grad(set_to_none=True)

    individual_weight_gradients = []
    individual_bias_gradients = []
    for view_index in range(MASK_VIEWS_PER_CORE_STEP):
        realization = make_exact_uniform_training_mask(
            batch, 0, view_index=view_index
        )
        gene_mask = torch.from_numpy(np.array(realization.mask, copy=True))
        input_expression = batch.target_expression.masked_fill(gene_mask, 0.0)
        prediction = manual_model(
            input_expression=input_expression,
            gene_mask=gene_mask,
            edge_index=batch.edge_index,
            relative_geometry=batch.relative_geometry,
            node_covariates=batch.node_covariates,
            target_nodes=torch.arange(batch.n_nodes),
        )
        loss = masked_huber_reconstruction_loss(
            prediction, batch.target_expression, gene_mask
        )
        (loss / float(MASK_VIEWS_PER_CORE_STEP)).backward()

        view_model = _TinyRelativeModel(3, dropout=0.0)
        view_prediction = view_model(
            input_expression=input_expression,
            gene_mask=gene_mask,
            edge_index=batch.edge_index,
            relative_geometry=batch.relative_geometry,
            node_covariates=batch.node_covariates,
            target_nodes=torch.arange(batch.n_nodes),
        )
        view_loss = masked_huber_reconstruction_loss(
            view_prediction, batch.target_expression, gene_mask
        )
        view_loss.backward()
        individual_weight_gradients.append(view_model.weight.grad.detach().clone())
        individual_bias_gradients.append(view_model.bias.grad.detach().clone())

    expected_weight_gradient = torch.stack(individual_weight_gradients).mean()
    expected_bias_gradient = torch.stack(individual_bias_gradients).mean(dim=0)
    torch.testing.assert_close(
        manual_model.weight.grad, expected_weight_gradient, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        manual_model.bias.grad, expected_bias_gradient, rtol=1e-6, atol=1e-7
    )
    torch.nn.utils.clip_grad_norm_(manual_model.parameters(), 1.0)
    manual_optimizer.step()

    trained_model = _TinyRelativeModel(3, dropout=0.0)
    fit_pooled_relative_qkv_segment(
        trained_model, batches, _config(model_seed=0)
    )
    assert trained_model.step_start_weights[1] == pytest.approx(
        float(manual_model.weight.detach()), abs=1e-8
    )
    torch.testing.assert_close(
        trained_model.step_start_biases[1],
        manual_model.bias.detach(),
        rtol=0,
        atol=0,
    )


def test_zero_total_mask_is_deterministically_resampled() -> None:
    batch = _core_batches(n_nodes=1, n_genes=1)[0]
    resampled = None
    for epoch in range(100):
        candidate = make_exact_uniform_training_mask(
            batch,
            epoch,
            view_index=0,
            mask_base_seed=MASK_BASE_SEED,
        )
        if candidate.zero_total_resample_count > 0:
            resampled = candidate
            break
    assert resampled is not None
    assert resampled.initial_seed != resampled.effective_seed
    assert resampled.n_masked_entries == 1
    repeated = make_exact_uniform_training_mask(
        batch,
        epoch,
        view_index=0,
        mask_base_seed=MASK_BASE_SEED,
    )
    assert repeated.checksum_sha256 == resampled.checksum_sha256
    np.testing.assert_array_equal(repeated.mask, resampled.mask)


def test_masked_huber_rejects_zero_mask_and_accumulates_fp32() -> None:
    prediction = torch.asarray([[0.0, 2.0]], dtype=torch.bfloat16)
    target = torch.asarray([[1.0, 0.0]], dtype=torch.bfloat16)
    mask = torch.asarray([[True, False]])
    loss = masked_huber_reconstruction_loss(prediction, target, mask)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(0.5)
    with pytest.raises(PooledRelativeQKVTrainingError, match="zero-total"):
        masked_huber_reconstruction_loss(
            prediction, target, torch.zeros_like(mask)
        )


def test_epoch_boundary_resume_matches_uninterrupted_training() -> None:
    batches = _core_batches()
    torch.manual_seed(50)
    uninterrupted_model = _TinyRelativeModel(3, dropout=0.25)
    uninterrupted = fit_pooled_relative_qkv_segment(
        uninterrupted_model,
        batches,
        _config(model_seed=5, end=2),
    )

    torch.manual_seed(50)
    first_segment_model = _TinyRelativeModel(3, dropout=0.25)
    first_segment = fit_pooled_relative_qkv_segment(
        first_segment_model,
        batches,
        _config(model_seed=5, end=1),
    )
    continuation_model = _TinyRelativeModel(3, dropout=0.25)
    continued = fit_pooled_relative_qkv_segment(
        continuation_model,
        batches,
        _config(model_seed=5, start=1, end=2),
        resume=first_segment.resume,
    )

    assert uninterrupted.final_state_checksum == continued.final_state_checksum
    assert uninterrupted.history_checksum == continued.history_checksum
    assert uninterrupted.resume.resume_checksum == continued.resume.resume_checksum
    assert uninterrupted.core_history == continued.core_history
    assert uninterrupted.global_history == continued.global_history
    for name, tensor in uninterrupted.final_state_dict.items():
        torch.testing.assert_close(tensor, continued.final_state_dict[name], rtol=0, atol=0)
    # The resume retains the completed ten-view schedule and future masks are
    # re-derived from the same alias/epoch/view tuple.
    assert all(
        len(record.mask_views) == MASK_VIEWS_PER_CORE_STEP
        for record in first_segment.resume.core_history
    )


def test_serialized_epoch_boundary_resume_round_trip(tmp_path) -> None:
    batches = _core_batches()
    torch.manual_seed(71)
    first_model = _TinyRelativeModel(3, dropout=0.25)
    first = fit_pooled_relative_qkv_segment(
        first_model,
        batches,
        _config(model_seed=5, end=1),
    )
    resume = first.resume
    payload = {
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_steps_completed": resume.optimizer_steps_completed,
        "model_seed": resume.model_seed,
        "mask_base_seed": resume.mask_base_seed,
        "core_order_seed": resume.core_order_seed,
        "mask_views_per_core_step": resume.mask_views_per_core_step,
        "model_state_dict": resume.model_state_dict,
        "model_state_checksum": resume.model_state_checksum,
        "optimizer_state_dict": resume.optimizer_state_dict,
        "optimizer_state_checksum": resume.optimizer_state_checksum,
        "amp_scaler_state_dict": resume.scaler_state_dict,
        "amp_scaler_state_checksum": resume.scaler_state_checksum,
        "core_history": [asdict(record) for record in resume.core_history],
        "global_history": [asdict(record) for record in resume.global_history],
        "history_checksum": resume.history_checksum,
        "resume_checksum": resume.resume_checksum,
        "model_step_rng_derivation": resume.model_step_rng_derivation,
    }
    checkpoint = tmp_path / "epoch_0001.ckpt"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = epoch_boundary_resume_from_checkpoint(loaded)
    assert restored.resume_checksum == resume.resume_checksum
    assert restored.core_history == resume.core_history
    assert restored.global_history == resume.global_history

    continued_model = _TinyRelativeModel(3, dropout=0.25)
    continued = fit_pooled_relative_qkv_segment(
        continued_model,
        batches,
        _config(model_seed=5, start=1, end=2),
        resume=restored,
    )
    direct_model = _TinyRelativeModel(3, dropout=0.25)
    direct = fit_pooled_relative_qkv_segment(
        direct_model,
        batches,
        _config(model_seed=5, start=1, end=2),
        resume=resume,
    )
    assert continued.final_state_checksum == direct.final_state_checksum
    assert continued.history_checksum == direct.history_checksum


def test_nonzero_segment_start_requires_matching_resume() -> None:
    with pytest.raises(PooledRelativeQKVTrainingError, match="requires.*resume"):
        fit_pooled_relative_qkv_segment(
            _TinyRelativeModel(3),
            _core_batches(),
            _config(model_seed=0, start=1, end=2),
        )


def test_joint_plateau_requires_two_common_audits_after_epoch_150() -> None:
    constant_150 = {seed: np.ones(150) for seed in range(5)}
    first = joint_five_seed_plateau_decision(
        constant_150, completed_global_epochs=150
    )
    assert first.every_seed_currently_passes
    assert not first.every_seed_previously_passed
    assert not first.should_stop_all_five
    assert first.continue_all_five_for_global_epochs == 25

    constant_175 = {seed: np.ones(175) for seed in range(5)}
    second = joint_five_seed_plateau_decision(
        constant_175, completed_global_epochs=175
    )
    assert second.every_seed_currently_passes
    assert second.every_seed_previously_passed
    assert second.consecutive_joint_passing_audits == 2
    assert second.should_stop_all_five
    assert second.common_final_epoch == 175

    one_declining = {seed: np.ones(175) for seed in range(5)}
    one_declining[4] = np.linspace(2.0, 1.0, 175)
    failed = joint_five_seed_plateau_decision(
        one_declining, completed_global_epochs=175
    )
    assert not failed.every_seed_currently_passes
    assert not failed.should_stop_all_five
    assert failed.continue_all_five_for_global_epochs == 25


def test_active_single_seed_plateau_requires_two_consecutive_audits() -> None:
    assert PooledRelativeQKVTrainingConfig(model_seed=0).segment_end_global_epoch == 150
    first = single_seed_plateau_decision(
        np.ones(150), model_seed=0, completed_global_epochs=150
    )
    assert first.current_audit.qualifying_passed
    assert not first.previous_audit.qualifying_passed
    assert first.consecutive_passing_audits == 1
    assert not first.should_stop
    assert first.continue_for_global_epochs == 25

    second = single_seed_plateau_decision(
        np.ones(175), model_seed=0, completed_global_epochs=175
    )
    assert second.consecutive_passing_audits == 2
    assert second.should_stop
    assert second.final_epoch == 175


def test_joint_plateau_rejects_missing_seed_or_misaligned_history() -> None:
    with pytest.raises(PooledRelativeQKVTrainingError, match="exactly model seeds"):
        joint_five_seed_plateau_decision(
            {seed: np.ones(150) for seed in range(4)},
            completed_global_epochs=150,
        )
    misaligned = {seed: np.ones(150) for seed in range(5)}
    misaligned[2] = np.ones(149)
    with pytest.raises(PooledRelativeQKVTrainingError, match="align"):
        joint_five_seed_plateau_decision(
            misaligned, completed_global_epochs=150
        )

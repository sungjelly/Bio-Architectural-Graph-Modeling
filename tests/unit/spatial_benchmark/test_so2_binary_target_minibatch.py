"""Independent controls for target minibatches and the historical epoch hook.

Fixed-weight gradient equivalence is deliberately tested with dropout disabled.
It does not assert equivalent trajectories after changing optimizer step count.
"""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train import run_so2_target_isolated_nb as base
from spatial_benchmark.binary_expression import (
    TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    masked_binary_expression_bce_sums,
)
from spatial_benchmark.target_cell_masking import make_training_target_cell_masks


def _epoch_arguments() -> dict:
    return {
        "model": object(),
        "training_model": object(),
        "optimizer": object(),
        "scaler": object(),
        "bundle": SimpleNamespace(batches_by_alias={a: object() for a in base.TRAINING_ALIASES}),
        "global_epoch": 2,
        "rank": 3,
        "device": torch.device("cpu"),
        "trainer": {"test": "unchanged frozen trainer"},
        "control_group": object(),
        "global_tracker": object(),
        "block_tracker": object(),
        "theta_tracker": object(),
        "theta_view": object(),
    }


def test_default_epoch_hook_preserves_six_original_paired_updates(monkeypatch) -> None:
    assert base.WORLD_SIZE == 4
    assert base.OPTIMIZER_UPDATES_PER_EPOCH == 6
    arguments = _epoch_arguments()
    expected_order = tuple(reversed(base.TRAINING_ALIASES))
    calls = []

    def core_order(completed_epoch, *, aliases, core_order_seed):
        assert completed_epoch == arguments["global_epoch"] + 1
        assert aliases == base.TRAINING_ALIASES
        assert core_order_seed == base.CORE_ORDER_SEED
        return expected_order

    def paired_update(**kwargs):
        calls.append(kwargs)
        return (
            {alias: (alias, "original paired loss") for alias in kwargs["pair"]},
            tuple({"core_alias": alias} for alias in kwargs["pair"]),
            1.0,
        )

    monkeypatch.setattr(base, "cohort_relative_qkv_core_order", core_order)
    monkeypatch.setattr(base, "_paired_update", paired_update)
    ordered, statistics, receipts = base._train_epoch(**arguments)

    assert ordered == expected_order
    assert len(calls) == 6
    assert set(statistics) == set(base.TRAINING_ALIASES)
    assert [row["core_alias"] for row in receipts] == list(expected_order)
    for index, call in enumerate(calls):
        assert call["pair"] == expected_order[2 * index:2 * index + 2]
        assert call["update_index"] == index
        assert call["batches"] is arguments["bundle"].batches_by_alias
        for key, value in arguments.items():
            if key != "bundle":
                assert call[key] is value or call[key] == value


def test_default_epoch_hook_rejects_duplicate_core_statistics(monkeypatch) -> None:
    alias = base.TRAINING_ALIASES[0]
    monkeypatch.setattr(
        base, "_paired_update", lambda **_: ({alias: (1.0,)}, (), 1.0)
    )
    with pytest.raises(base.SO2NBTrainingError, match="visited twice"):
        base._train_epoch(**_epoch_arguments())


def test_default_epoch_hook_rejects_incomplete_core_order(monkeypatch) -> None:
    monkeypatch.setattr(
        base, "cohort_relative_qkv_core_order", lambda *_, **__: base.TRAINING_ALIASES[:-1]
    )
    with pytest.raises(base.SO2NBTrainingError, match="complete permutation"):
        base._train_epoch(**_epoch_arguments())


def _model() -> TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer:
    torch.manual_seed(7719)
    model = TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
        num_genes=4, node_covariate_dim=2, hidden_dim=8,
        attention_heads=2, attention_head_dim=4, graph_layers=4,
        ffn_dim=16, decoder_dim=12, geometry_hidden_dim=6,
        dropout=0.0, attention_dropout=0.0,
        receiver_chunk_size=2, max_edges_per_chunk=16,
        activation_checkpointing=True,
    )
    # Use nonzero fixed output weights to test all upstream gradients. The
    # production zero-head initialization is covered separately and unchanged.
    with torch.no_grad():
        model.detection_head.weight.normal_(mean=0.0, std=0.2)
    return model


def _core(n_cells: int, seed: int) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)
    receiver = torch.arange(n_cells).repeat_interleave(n_cells - 1)
    sender = torch.tensor(
        [s for r in range(n_cells) for s in range(n_cells) if s != r]
    )
    args = dict(
        n_cells=n_cells, n_genes=4, base_seed=2026092601,
        namespace="bagm.so2.target_isolated_nb.training_masks.v1",
        core_alias=f"synthetic-{seed}", global_epoch=3,
    )
    realization = make_training_target_cell_masks(**args)
    return SimpleNamespace(
        alias=args["core_alias"], n_cells=n_cells, mask_arguments=args,
        expression=torch.randn(n_cells, 4, generator=generator),
        covariates=torch.randn(n_cells, 2, generator=generator),
        edges=torch.stack((sender, receiver)),
        geometry=torch.randn(len(sender), 70, generator=generator),
        counts=torch.arange(n_cells * 4).reshape(n_cells, 4) % 5,
        mask=torch.from_numpy(realization.mask.copy()),
    )


def _forward(model, core, start: int, stop: int):
    return model(
        input_expression=core.expression,
        target_input_expression=core.expression[start:stop].masked_fill(core.mask[start:stop], 0.0),
        target_gene_mask=core.mask[start:stop],
        node_covariates=core.covariates,
        edge_index=core.edges,
        relative_geometry=core.geometry,
        target_start=start, target_stop=stop,
    )


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_shuffled_windows_preserve_masks_predictions_and_weighted_bce_gradients(batch_size) -> None:
    from spatial_benchmark.target_minibatch import target_window_plan

    cores = (_core(7, 31), _core(11, 47))
    reference = _model().train()
    partitioned = copy.deepcopy(reference).train()
    denominator = sum(int(core.mask.sum()) for core in cores)
    full_loss = 0.0
    batch_loss = 0.0
    batch_means = []
    for core in cores:
        full = _forward(reference, core, 0, core.n_cells)
        full_sums = masked_binary_expression_bce_sums(full.nonzero_logits, core.counts, core.mask)
        full_loss += float(full_sums.detection_nll_sum.detach()) / denominator
        (full_sums.detection_nll_sum / denominator).backward()
        expected_logits = full.nonzero_logits.detach()

        windows = target_window_plan(
            core.n_cells, batch_size, seed=901, core_alias=core.alias, global_epoch=3
        )
        visited = np.zeros(core.n_cells, dtype=np.int64)
        for start, stop in windows:
            sampled = make_training_target_cell_masks(
                **core.mask_arguments, target_indices=np.arange(start, stop)
            )
            assert np.array_equal(sampled.mask, core.mask[start:stop].numpy())
            visited[start:stop] += 1
            output = _forward(partitioned, core, start, stop)
            torch.testing.assert_close(
                output.nonzero_logits, expected_logits[start:stop], rtol=2e-5, atol=2e-6
            )
            sums = masked_binary_expression_bce_sums(
                output.nonzero_logits, core.counts[start:stop], core.mask[start:stop]
            )
            batch_loss += float(sums.detection_nll_sum.detach()) / denominator
            batch_means.append(float(sums.detection_nll_sum.detach()) / sums.masked_entries)
            # This is the batch mean weighted by its fraction of all masked
            # entries, accumulated at fixed parameters before any optimizer step.
            (sums.detection_nll_sum / denominator).backward()
        assert np.all(visited == 1)

    assert batch_loss == pytest.approx(full_loss, rel=2e-6, abs=2e-6)
    assert abs(sum(batch_means) / len(batch_means) - full_loss) > 1e-4
    actual_parameters = dict(partitioned.named_parameters())
    for name, parameter in reference.named_parameters():
        actual = actual_parameters[name].grad
        assert parameter.grad is not None and actual is not None, name
        assert bool(torch.isfinite(actual).all()), name
        torch.testing.assert_close(
            actual, parameter.grad, rtol=3e-5, atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    assert reference.encoder.expression_projection.weight.grad.abs().sum() > 0
    assert all(block.value_projection.weight.grad.abs().sum() > 0 for block in reference.blocks)


def test_importing_minibatch_runner_does_not_install_a_training_contract() -> None:
    code = """
from scripts.train import run_so2_target_isolated_nb as base
before = dict(base.__dict__)
from scripts.train import run_so2_target_minibatch_binary_expression_bce
changed = [name for name in before if base.__dict__.get(name) is not before[name]]
assert not changed, changed
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def minibatch_runner():
    original = dict(base.__dict__)
    module = importlib.import_module(
        "scripts.train.run_so2_target_minibatch_binary_expression_bce"
    )
    try:
        yield module
    finally:
        for name in set(base.__dict__) - set(original):
            delattr(base, name)
        for name, value in original.items():
            setattr(base, name, value)


def test_epoch_executes_one_real_optimizer_step_per_window_and_reencodes_sources(
    minibatch_runner, monkeypatch,
) -> None:
    runner = minibatch_runner
    cores = []
    for index, alias in enumerate(base.TRAINING_ALIASES):
        core = _core(5, 300 + index)
        core.alias = alias
        core.n_nodes = core.n_cells
        core.n_genes = 4
        core.n_edges = core.edges.shape[1]
        core.input_expression = core.expression
        core.raw_count_target = core.counts
        core.node_covariates = core.covariates
        core.edge_index = core.edges
        core.relative_geometry = core.geometry
        cores.append(core)
    bundle = SimpleNamespace(
        training_batches=tuple(cores), batches_by_alias={c.alias: c for c in cores}
    )
    # Five targets split into balanced 3/2 windows, over all twelve cores.
    expected_updates = 24
    monkeypatch.setattr(base, "WORLD_SIZE", 1)
    monkeypatch.setattr(base, "OPTIMIZER_UPDATES_PER_EPOCH", expected_updates)
    monkeypatch.setattr(base, "_seed_model_step", lambda seed, _: torch.manual_seed(seed))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_EPOCH_RESOURCES", {})
    monkeypatch.setattr(runner, "_CURRENT_EPOCH", 0)
    monkeypatch.setattr(runner, "_resource_snapshot", lambda _: {
        "peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "headroom_bytes": 3,
    })
    progress = []
    monkeypatch.setattr(runner, "_progress", lambda event, **values: progress.append((event, values)))
    model = _model().train()
    # Exercise the production zero-output-weight initialization as well: first
    # graph gradients may be zero; later windows must propagate upstream.
    with torch.no_grad():
        model.detection_head.weight.zero_()

    class CountingAdamW(torch.optim.AdamW):
        step_count = 0
        zero_count = 0

        def zero_grad(self, *args, **kwargs):
            self.zero_count += 1
            return super().zero_grad(*args, **kwargs)

        def step(self, *args, **kwargs):
            self.step_count += 1
            return super().step(*args, **kwargs)

    optimizer = CountingAdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    full_tracker = base.FullGradientDirectionTracker(expected_optimizer_updates_per_epoch=expected_updates)
    block_tracker = base.BlockGradientDirectionTracker(expected_blocks=4, expected_optimizer_updates_per_epoch=expected_updates)
    head_tracker = base.FullGradientDirectionTracker(expected_optimizer_updates_per_epoch=expected_updates)
    encoder_calls = []

    def record_encoder(_module, _args, kwargs):
        encoder_calls.append((kwargs["input_expression"].shape[0], model.encoder.expression_projection.weight._version))

    handle = model.encoder.register_forward_pre_hook(record_encoder, with_kwargs=True)
    try:
        ordered, statistics, receipts = runner._train_epoch(
            model=model, training_model=model, optimizer=optimizer, scaler=scaler,
            bundle=bundle, global_epoch=0, rank=0, device=torch.device("cpu"),
            trainer={"target_cells_per_optimizer_update": 3, "target_window_order_seed": 2026100101,
                     "gradient_clip_norm": 1.0, "amp": False, "amp_dtype": "bfloat16"},
            control_group=None, global_tracker=full_tracker, block_tracker=block_tracker,
            theta_tracker=head_tracker, theta_view=model.detection_head,
        )
    finally:
        handle.remove()
    assert optimizer.step_count == optimizer.zero_count == expected_updates
    assert len(encoder_calls) == expected_updates * 2
    assert all(n == 5 for n, _ in encoder_calls[::2])
    assert all(n in (2, 3) for n, _ in encoder_calls[1::2])
    versions = [version for _, version in encoder_calls[::2]]
    assert all(later > earlier for earlier, later in zip(versions, versions[1:]))
    assert set(ordered) == set(statistics) == set(base.TRAINING_ALIASES)
    assert sum(row[3] for row in statistics.values()) == 60
    assert len(receipts) == 12
    assert all(r["target_visit_count_min"] == r["target_visit_count_max"] == 1 for r in receipts)
    assert all(r["actual_optimizer_updates"] == 2 for r in receipts)
    assert all(r["masked_entries"] == statistics[r["core_alias"]][1] for r in receipts)
    updates = [v for event, v in progress if event == "target_update_completed"]
    assert [r["cumulative_optimizer_update"] for r in updates] == list(range(1, 25))
    assert full_tracker.complete_epoch(0).optimizer_updates_observed == expected_updates
    assert all(r.optimizer_updates_observed == expected_updates for r in block_tracker.complete_epoch(0))
    assert head_tracker.complete_epoch(0).optimizer_updates_observed == expected_updates
    assert runner._EPOCH_RESOURCES[1]["minimum_target_batch_size"] == 2
    assert runner._EPOCH_RESOURCES[1]["maximum_target_batch_size"] == 3

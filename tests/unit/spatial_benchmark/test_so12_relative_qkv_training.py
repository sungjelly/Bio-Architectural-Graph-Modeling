"""CPU scientific and durability checks for the joint-slide validation trainer."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spatial_benchmark.so12_relative_qkv_training import (
    CHECKPOINT_SCHEMA, SelectionState, advance_selection, aggregate_statistics,
    atomic_save, build_model, evaluate, forward_view, make_mask,
    masked_loss, paired_update, stage_core,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (
    cohort_relative_qkv_distributed_assignment,
)


def fixture_pair():
    rng = np.random.default_rng(13)
    cores = []
    for alias, n in (("SO1-C01", 8), ("SO2-C15", 12)):
        raw = rng.poisson(1., size=(n, 16)).astype(np.int32)
        cores.append(SimpleNamespace(
            alias=alias, raw_counts=raw,
            target_expression=(raw.astype(np.float32) - 1.) / 2.,
            node_covariates=rng.normal(size=(n, 3)).astype(np.float32),
            edge_index=np.stack((np.arange(n), (np.arange(n) + 1) % n)),
            relative_geometry=np.zeros((n, 70), dtype=np.float32)))
    normalization = dict(gene_p=torch.full((16,), 1/16),
                         residual_mean=torch.zeros(16), residual_sd=torch.ones(16))
    config = {"model": {"hidden_dim": 16, "attention_heads": 2, "attention_head_dim": 8,
                         "graph_layers": 4, "ffn_dim": 32, "decoder_dim": 32,
                         "dropout": 0., "receiver_chunk_size": 4, "max_edges_per_chunk": 32}}
    return cores, normalization, config


def test_strict_best_is_separate_from_patience():
    s = advance_selection(SelectionState(), 1., minimum=1, patience=2, delta=.1, maximum=20)
    s = advance_selection(s, .99, minimum=1, patience=2, delta=.1, maximum=20)
    assert s.best_epoch == 2 and s.best_value == .99 and s.bad_validations == 1
    s = advance_selection(s, .98, minimum=1, patience=2, delta=.1, maximum=20)
    assert s.best_epoch == 3 and s.stop_reason == "patience"
    assert s.patience_best == 1.


def test_patience_counts_from_start_but_stopping_waits_for_minimum():
    s = SelectionState()
    for _ in range(3):
        s = advance_selection(s, 1., minimum=3, patience=2, maximum=6)
    assert s.bad_validations == 2 and s.stop_reason == "patience" and s.epoch == 3
    s = SelectionState()
    for value in (1., .9, .8, .7, .6, .5):
        s = advance_selection(s, value, minimum=3, patience=2, maximum=6)
    assert s.stop_reason == "maximum_epochs"


def test_fixed_validation_masks_replay_and_are_disjoint_from_training():
    cores, _, _ = fixture_pair()
    a, ar = make_mask(cores[0], epoch=0, view=0, training=False)
    b, br = make_mask(cores[0], epoch=200, view=0, training=False)
    c, cr = make_mask(cores[0], epoch=0, view=0, training=True)
    assert np.array_equal(a, b) and ar == br
    assert ar["seed"] != cr["seed"] and not np.array_equal(a, c)
    assert ar["masked_entries"] == int(a.sum())


def test_validation_macro_and_pooled_are_not_confused():
    # Zero predictions versus targets 2 and 4 have Huber 1.5 and 3.5.
    from spatial_benchmark.so12_relative_qkv_training import sufficient_statistics
    records = []
    for alias, n, value in (("A", 1, 2.), ("B", 9, 4.)):
        y = torch.full((n, 1), value)
        records.append({"alias": alias, "view": 0,
                        "statistics": sufficient_statistics(torch.zeros_like(y), y, torch.ones_like(y, dtype=torch.bool))})
    result = aggregate_statistics(records)
    assert result["equal_core_huber"] == pytest.approx(2.5)
    assert result["pooled_huber"] == pytest.approx(3.3)
    assert result["pooled_pearson"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_statistics(records + records[:1])


def test_four_rank_five_view_gradient_mean_equals_twenty_view_reference():
    torch.manual_seed(2)
    x = torch.randn(20, 3)
    y = torch.randn(20)
    w = torch.randn(3, requires_grad=True)
    reference = torch.autograd.grad(((x @ w - y) ** 2).mean(), w)[0]
    gradients = []
    seen = set()
    for rank in range(4):
        a = cohort_relative_qkv_distributed_assignment(("A", "B"), rank=rank, world_size=4)
        indices = [a.position_in_optimizer_update * 10 + v for v in a.view_indices]
        seen.update(indices)
        gradients.append(torch.autograd.grad(((x[indices] @ w - y[indices]) ** 2).sum()/a.local_loss_divisor, w)[0])
    assert seen == set(range(20))
    torch.testing.assert_close(torch.stack(gradients).mean(0), reference)


def test_original_model_cpu_training_validation_and_checkpoint_replay(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    cores, normalization, config = fixture_pair()
    model = build_model(config, genes=16, covariates=3)
    assert len(model.blocks) == 4 and len({id(b) for b in model.blocks}) == 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    update = paired_update(model, optimizer, tuple(cores), normalization, epoch=0, update=0,
                           rank=0, world=1, device=torch.device("cpu"), amp=False)
    assert len(update["receipts"]) == 20 and update["gradient_norm"] > 0
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    metrics, receipts = evaluate(model, cores, normalization, device=torch.device("cpu"),
                                 rank=0, world=1, views=3, amp=False)
    assert metrics["view_count"] == 6 and np.isfinite(metrics["equal_core_huber"])
    path = tmp_path / "latest.ckpt"
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "selection_state": asdict(SelectionState())}
    atomic_save(path, payload)
    loaded = torch.load(path, weights_only=False)
    other = build_model(config, genes=16, covariates=3)
    other.load_state_dict(loaded["model_state_dict"])
    replay, masks = evaluate(other, cores, normalization, device=torch.device("cpu"),
                             rank=0, world=1, views=3, amp=False)
    assert metrics == replay
    assert [r["receipt"] for r in receipts] == [r["receipt"] for r in masks]
    assert list(tmp_path.iterdir()) == [path]


def test_loss_excludes_visible_targets():
    pred = torch.tensor([[1., 2.]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    loss = masked_loss(pred, torch.tensor([[0., 1e6]]), mask)
    assert loss.item() == .5
    loss.backward()
    assert pred.grad.tolist() == [[1., 0.]]

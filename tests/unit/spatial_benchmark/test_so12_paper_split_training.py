"""Scientific checks for the fresh 21/5/2 core split trainer (synthetic data only)."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from spatial_benchmark.so12_paper_split_training import (
    CHECKPOINT_SCHEMA, PRIMARY, PROTOCOL, TEST_NAMESPACE, VALIDATION_NAMESPACE,
    SelectionState, atomic_save, distributed_view_jobs, epoch_core_batches,
    epoch_equal_core_loss, evaluate, make_mask, paired_update, prediction_rows,
    selected_checkpoint_identity, source_binding, view_gradient_weight,
)


def fixture_cores():
    rng = np.random.default_rng(19)
    cores = []
    for alias, n in (("SO1-C01", 7), ("SO1-C03", 9)):
        raw = rng.poisson(2., size=(n, 4)).astype(np.int32)
        cores.append(SimpleNamespace(
            alias=alias, raw_counts=raw,
            target_expression=(raw.astype(np.float32) - 2.) / 3.,
            node_covariates=np.zeros((n, 2), dtype=np.float32),
            edge_index=np.stack((np.arange(n), (np.arange(n) + 1) % n)),
            relative_geometry=np.zeros((n, 70), dtype=np.float32),
        ))
    normalization = dict(gene_p=torch.full((4,), .25),
                         residual_mean=torch.zeros(4), residual_sd=torch.ones(4))
    return cores, normalization


class TinyPrediction(nn.Module):
    """Minimal differentiable model exercising the actual masked update path."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=True)

    def clear_edge_layout_cache(self):
        pass

    def forward(self, input_expression, **kwargs):
        return SimpleNamespace(prediction=self.linear(input_expression))


def _ddp_worker(rank, init_file, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=f"file://{init_file}")
    try:
        cores, normalization = fixture_cores()
        for count in (1, 2):
            torch.manual_seed(53)
            raw_model = TinyPrediction()
            model = DistributedDataParallel(raw_model, broadcast_buffers=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            record = paired_update(model, optimizer, tuple(cores[:count]), normalization,
                                   epoch=0, update=0, rank=rank, world=4,
                                   device=torch.device("cpu"), amp=False)
            if rank == 0:
                torch.save({"weights": raw_model.state_dict(), "record": record},
                           Path(destination) / f"ddp_{count}.pt")
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_odd_core_schedule_has_exactly_ten_pairs_and_one_singleton():
    aliases = tuple(f"core_{n}" for n in range(21))
    result = epoch_core_batches(0, aliases=aliases, core_order_seed=23)
    assert len(result) == 11
    assert [len(batch) for batch in result] == [2] * 10 + [1]
    assert sorted(x for batch in result for x in batch) == sorted(aliases)
    assert result == epoch_core_batches(0, aliases=aliases, core_order_seed=23)
    assert result != epoch_core_batches(1, aliases=aliases, core_order_seed=23)
    with pytest.raises(ValueError):
        epoch_core_batches(0, aliases=("same", "same"), core_order_seed=23)


@pytest.mark.parametrize("core_count", [1, 2])
def test_distributed_view_weight_equals_serial_mean(core_count):
    torch.manual_seed(7)
    x = torch.randn(10 * core_count, 3, dtype=torch.float64)
    y = torch.randn(10 * core_count, dtype=torch.float64)
    w = torch.randn(3, dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(((x @ w - y) ** 2).mean(), w)[0]
    gradients, coverage, rank_counts = [], [], []
    for rank in range(4):
        jobs = distributed_view_jobs(core_count, rank=rank, world=4)
        indices = [core * 10 + view for core, views in jobs for view in views]
        coverage.extend(indices)
        rank_counts.append(len(indices))
        loss = ((x[indices] @ w - y[indices]) ** 2).sum() * view_gradient_weight(core_count, 4)
        gradients.append(torch.autograd.grad(loss, w)[0])
    assert sorted(coverage) == list(range(10 * core_count))
    assert rank_counts == ([3, 3, 2, 2] if core_count == 1 else [5] * 4)
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected, rtol=1e-12, atol=1e-12)


def test_actual_four_rank_gloo_updates_match_serial_for_singleton_and_pair(tmp_path):
    # Unequal local forward counts exercise DDP no_sync and synchronization,
    # not merely the averaging algebra tested above.
    torch.set_num_threads(1)
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "gloo_init"), str(tmp_path)),
                                nprocs=4, join=True)
    cores, normalization = fixture_cores()
    for count in (1, 2):
        torch.manual_seed(53)
        model = TinyPrediction()
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        serial = paired_update(model, optimizer, tuple(cores[:count]), normalization,
                               epoch=0, update=0, rank=0, world=1,
                               device=torch.device("cpu"), amp=False)
        saved = torch.load(tmp_path / f"ddp_{count}.pt", weights_only=False)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(saved["weights"][key], value, atol=3e-7, rtol=1e-6)
        assert saved["record"]["loss"] == pytest.approx(serial["loss"], rel=1e-6)
        assert saved["record"]["gradient_norm"] == pytest.approx(serial["gradient_norm"], rel=1e-6)
        assert saved["record"]["receipts"] == serial["receipts"]


def test_epoch_report_weights_singleton_by_one_core():
    updates = [{"loss": 1., "core_count": 2}] * 10 + [{"loss": 22., "core_count": 1}]
    assert epoch_equal_core_loss(updates) == 2.
    assert epoch_equal_core_loss(updates) != np.mean([r["loss"] for r in updates])


def test_validation_test_and_training_mask_namespaces_are_distinct_and_fixed():
    core = fixture_cores()[0][0]
    _, a = make_mask(core, epoch=0, view=0, training=False, split="validation")
    _, b = make_mask(core, epoch=100, view=0, training=False, split="validation")
    _, c = make_mask(core, epoch=0, view=0, training=False, split="test")
    _, d = make_mask(core, epoch=0, view=0, training=True)
    assert a == b
    assert len({a["seed"], c["seed"], d["seed"]}) == 3
    assert VALIDATION_NAMESPACE != TEST_NAMESPACE
    assert c["split"] == "test"


def test_selection_freeze_rejects_unfinished_or_historical_checkpoint():
    state = SelectionState(epoch=30, best_epoch=23, best_value=.2, stop_reason="patience")
    best = {"checkpoint_schema": CHECKPOINT_SCHEMA, "protocol": PROTOCOL,
            "checkpoint_role": "best", "epoch": 23,
            "primary_metric_name": PRIMARY, "primary_metric_value": .2}
    receipt = selected_checkpoint_identity(best, state, checkpoint_sha256="f" * 64,
                                           validation_replay_verified=True)
    assert receipt["selection_completed"] and not receipt["test_used_for_selection"]
    with pytest.raises(ValueError, match="frozen"):
        selected_checkpoint_identity(best, replace(state, stop_reason=None),
                                     checkpoint_sha256="f" * 64, validation_replay_verified=True)
    with pytest.raises(ValueError, match="frozen"):
        selected_checkpoint_identity(best, state, checkpoint_sha256="f" * 64,
                                     validation_replay_verified=False)
    with pytest.raises(ValueError, match="frozen"):
        selected_checkpoint_identity({**best, "protocol": "old_protocol"}, state,
                                     checkpoint_sha256="f" * 64, validation_replay_verified=True)


def test_prediction_keys_preserve_split_and_selected_core_view_coverage():
    records = [{"alias": alias, "view": view, "statistics": [10, 1, 2, 3, 4, 5, 6, 7, 8],
                "nodes": 12, "edges": 50}
               for alias in ("SO2-C15", "SO2-C23") for view in range(3)]
    test = prediction_rows(records, run_id="synthetic", dataset_id="synthetic", split="test")
    val = prediction_rows(records, run_id="synthetic", dataset_id="synthetic", split="validation")
    assert len({r["sample_key"] for r in test + val}) == 12
    assert all(r["split"] == "test" and r["masked_entry_count"] == 10 for r in test)


def test_evaluation_has_fixed_test_masks_and_rejects_wrong_partition():
    cores, normalization = fixture_cores()
    for core in cores:
        core.split = "test"
    model = TinyPrediction()
    result, records = evaluate(model, cores, normalization, device=torch.device("cpu"),
                               rank=0, world=1, amp=False, split="test")
    assert result["view_count"] == 6
    assert all(r["receipt"]["split"] == "test" for r in records)
    with pytest.raises(ValueError, match="partition"):
        evaluate(model, cores, normalization, device=torch.device("cpu"),
                 rank=0, world=1, amp=False, split="validation")


def test_new_schema_atomic_checkpoint_and_transitive_source_binding(tmp_path):
    path = tmp_path / "latest.ckpt"
    atomic_save(path, {"checkpoint_schema": CHECKPOINT_SCHEMA, "protocol": PROTOCOL})
    assert torch.load(path, weights_only=False)["protocol"] == PROTOCOL
    sources = source_binding()
    for name in ("so12_paper_split_training.py", "so12_paper_split_data.py",
                 "relative_qkv_graph_transformer.py", "adjacency_ablation.py", "masking.py"):
        assert name in sources and len(sources[name]) == 64

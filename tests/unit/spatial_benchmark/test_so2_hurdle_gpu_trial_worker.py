"""CPU correctness gates for the bounded one-GPU hurdle diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from spatial_benchmark.hurdle_negative_binomial import masked_hurdle_negative_binomial_nll_sums
from spatial_benchmark.target_cell_masking import make_contiguous_target_partitions, make_training_target_cell_masks


@pytest.fixture
def worker():
    script = Path(__file__).resolve().parents[3] / "scripts/diagnostics/so2_hurdle_gpu_trial_worker.py"
    spec = importlib.util.spec_from_file_location("_hurdle_gpu_trial_worker_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_memory_gate_uses_actual_device_capacity(worker) -> None:
    # A nominal 24-GB GPU can expose less than 24 GiB to the framework.
    total = 25296044032
    assert worker.memory_gate(allocated=19 * worker.GIB, reserved=20 * worker.GIB, total=total)["resource_gate_passed"]
    assert not worker.memory_gate(allocated=21 * worker.GIB, reserved=22 * worker.GIB, total=total)["resource_gate_passed"]


def test_physical_headroom_accounts_for_driver_memory(worker) -> None:
    result = worker.memory_gate(allocated=19 * worker.GIB, reserved=20 * worker.GIB,
                                total=24 * worker.GIB, sampled_free=worker.GIB)
    assert result["allocator_headroom_bytes"] == 4 * worker.GIB
    assert result["headroom_bytes"] == worker.GIB
    assert not result["resource_gate_passed"]


@pytest.mark.parametrize("allocated,reserved,total", [(-1, 1, 10), (2, 1, 10), (0, 0, 0)])
def test_invalid_resource_counters_rejected(worker, allocated, reserved, total) -> None:
    with pytest.raises(ValueError, match="Invalid GPU memory counters"):
        worker.memory_gate(allocated=allocated, reserved=reserved, total=total)


def test_json_output_is_exclusive_and_metrics_append(worker, tmp_path: Path) -> None:
    target = tmp_path / "diagnostics/result.json"
    worker.write_json(target, {"step": 1})
    with pytest.raises(FileExistsError): worker.write_json(target, {"step": 2})
    assert json.loads(target.read_text()) == {"step": 1}
    log = tmp_path / "metrics/epochs.jsonl"
    worker.append_json(log, {"step": 1}); first = log.read_bytes()
    worker.append_json(log, {"step": 2})
    assert log.read_bytes().startswith(first)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [1, 2]


def _request(worker, tmp_path, monkeypatch):
    monkeypatch.setenv("BAGM_ROOT", str(tmp_path))
    monkeypatch.setenv("BAGM_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    run_id = "r_gpu_trial_test"
    directory = tmp_path / "scratch/active_runs" / run_id
    directory.mkdir(parents=True)
    source = {"seed": 0, "model": {"name": "target-isolated-geometry-modulated-relative-qkv-gat-hurdle-nb2"},
              "trainer": {"cores_per_optimizer_update": 2, "optimizer_updates_per_global_epoch": 6},
              "dataset": {"test_partition_present": False}}
    config = {"source_configuration": source, "runtime": {"gpu": "3", "epochs": 2, "effective_world_size": 1}}
    return config, run_id, directory


def test_request_uses_explicit_physical_gpu_without_mutating_source(worker, tmp_path, monkeypatch) -> None:
    config, run_id, directory = _request(worker, tmp_path, monkeypatch)
    before = json.dumps(config, sort_keys=True)
    result = worker.validate_request(config, run_id=run_id, run_dir=directory, epochs=2)
    assert result is config["source_configuration"]
    assert json.dumps(config, sort_keys=True) == before


@pytest.mark.parametrize("drift", ["multiple_devices", "wrong_device", "updates", "test", "epochs", "resume"])
def test_scientific_or_execution_drift_is_rejected(worker, tmp_path, monkeypatch, drift) -> None:
    config, run_id, directory = _request(worker, tmp_path, monkeypatch)
    epochs = 2
    if drift == "multiple_devices": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    elif drift == "wrong_device": config["runtime"]["gpu"] = "0"
    elif drift == "updates": config["source_configuration"]["trainer"]["optimizer_updates_per_global_epoch"] = 12
    elif drift == "test": config["source_configuration"]["dataset"]["test_partition_present"] = True
    elif drift == "epochs": epochs = 3
    else:
        (directory / "metrics").mkdir(); (directory / "metrics/runtime_epoch_metrics.jsonl").write_text("{}\n")
    with pytest.raises(ValueError): worker.validate_request(config, run_id=run_id, run_dir=directory, epochs=epochs)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_rank_partition_preserves_target_masks(world_size: int) -> None:
    import numpy as np
    args = dict(n_cells=17, n_genes=11, base_seed=2026092601,
                namespace="bagm.so2.target_isolated_nb.training_masks.v1", core_alias="synthetic", global_epoch=1)
    whole = make_training_target_cell_masks(**args)
    parts = make_contiguous_target_partitions(17, world_size)
    split = [make_training_target_cell_masks(**args, target_indices=p.target_indices) for p in parts]
    assert np.array_equal(np.concatenate([r.mask for r in split]), whole.mask)
    assert sum(r.n_masked_entries for r in split) == whole.n_masked_entries
    assert np.array_equal(np.concatenate([r.target_indices for r in split]), np.arange(17))


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_two_core_pooled_hurdle_gradient_matches_partitioned_ddp_mean(world_size: int) -> None:
    # Two differently sized cores, unequal masked-entry/positive counts per
    # rank, one shared scalar parameter. Compare actual hurdle-loss gradients.
    cores = [
        (torch.tensor([[0., 1., 2.], [3., 0., 1.], [0., 0., 2.], [4., 1., 0.], [0., 2., 0.]]),
         torch.tensor([[1,1,0],[1,0,0],[1,1,1],[0,1,1],[1,1,0]], dtype=torch.bool)),
        (torch.tensor([[1., 0., 0.], [0., 2., 1.], [2., 3., 0.], [0., 1., 4.], [1., 0., 0.], [0., 0., 2.], [1., 1., 0.]]),
         torch.tensor([[1,0,1],[1,1,1],[0,1,0],[1,1,1],[1,0,1],[1,1,0],[1,1,1]], dtype=torch.bool)),
    ]
    count = sum(int(mask.sum()) for _, mask in cores)
    def core_loss(parameter, target, mask):
        logits = parameter.expand_as(target)
        mean = torch.nn.functional.softplus(parameter).expand_as(target) + .2
        theta = torch.nn.functional.softplus(parameter).expand(target.shape[1]) + .5
        return masked_hurdle_negative_binomial_nll_sums(logits, mean, theta, target, mask).joint_nll_sum
    parameter = torch.tensor(.3, requires_grad=True)
    reference = sum(core_loss(parameter, target, mask) for target, mask in cores) / count
    reference_gradient, = torch.autograd.grad(reference, parameter)
    rank_gradients = []
    for rank in range(world_size):
        local_parameter = torch.tensor(.3, requires_grad=True)
        local_sum = torch.zeros(())
        for target, mask in cores:
            part = make_contiguous_target_partitions(len(target), world_size)[rank]
            local_sum = local_sum + core_loss(local_parameter, target[part.start:part.stop], mask[part.start:part.stop])
        gradient, = torch.autograd.grad(local_sum * world_size / count, local_parameter)
        rank_gradients.append(gradient)
    actual = torch.stack(rank_gradients).mean()
    torch.testing.assert_close(actual, reference_gradient, rtol=1e-6, atol=1e-7)


def test_nonfinite_diagnostics_fail_closed(worker) -> None:
    with pytest.raises(ValueError, match="Nonfinite scalar"):
        worker.assert_finite_scalars({"nested": [{"loss": float("nan")}]}, "record")

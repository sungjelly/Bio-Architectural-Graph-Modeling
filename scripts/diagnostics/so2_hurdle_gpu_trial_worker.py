"""Fresh one-GPU, two-epoch hurdle runtime trial; orchestration owns publication.

Contract: experiments/campaigns/cmp_20261001_so2_hurdle_gpu_scaling/README.md.
This diagnostic reuses the original model and numerical update/evaluation
helpers, with one-rank DDP. It neither invokes nor claims the historical
four-rank preflight. The nested source_configuration remains unchanged;
effective rank count is recorded separately. No checkpoints or predictions
are written. The caller must register the run and create its scratch bundle.

Smoke checks from the source repository root::

    PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_hurdle_gpu_trial_worker.py
    PYTHONPATH=src /venv/main/bin/python scripts/diagnostics/so2_hurdle_gpu_trial_worker.py --help

The registered orchestrator launches this worker with --config <resolved JSON>,
--run-dir <existing scratch>, --run-id <immutable ID>, and --epochs 2. Exactly
one physical GPU must be selected externally with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

import torch

from spatial_benchmark.paths import current_paths


RESULT_SCHEMA = "so2_hurdle_gpu_runtime_trial_result_v1"
GIB = 1024 ** 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value)); handle.flush(); os.fsync(handle.fileno())


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(json_bytes(value)); handle.flush(); os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_gate(*, allocated: int, reserved: int, total: int, sampled_free: int | None = None) -> dict[str, Any]:
    require(0 <= allocated <= reserved and total > 0, "Invalid GPU memory counters")
    allocator_headroom = total - max(allocated, reserved)
    require(sampled_free is None or sampled_free >= 0, "Invalid sampled physical memory")
    headroom = allocator_headroom if sampled_free is None else min(allocator_headroom, sampled_free)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "total_device_bytes": total, "headroom_bytes": headroom,
            "allocator_headroom_bytes": allocator_headroom,
            "minimum_sampled_physical_free_bytes": sampled_free,
            "headroom_measurement": "minimum_of_allocator_remainder_and_sampled_physical_free; physical_sampling_is_not_continuous",
            "resource_gate_passed": allocated <= 22 * GIB and reserved <= 22 * GIB and headroom >= 2 * GIB}


def validate_request(config: Mapping[str, Any], *, run_id: str, run_dir: Path, epochs: int) -> dict[str, Any]:
    require(epochs == 2, "This bounded diagnostic requires exactly two epochs")
    require(run_id.startswith("r_") and run_dir.name == run_id, "Run identity/path mismatch")
    expected = current_paths().scratch_root / "active_runs" / run_id
    require(run_dir.resolve() == expected.resolve() and run_dir.is_dir(), "Run directory must be existing canonical active scratch")
    require(config.get("run_id", run_id) == run_id, "Resolved configuration run ID differs")
    source = config.get("source_configuration")
    require(isinstance(source, dict), "source_configuration is required")
    require(source.get("seed") == 0, "Resource trial starts from the original seed zero")
    require(source.get("model", {}).get("name") == "target-isolated-geometry-modulated-relative-qkv-gat-hurdle-nb2",
            "Resource trial requires the hurdle count model")
    trainer = source.get("trainer", {})
    require(trainer.get("cores_per_optimizer_update") == 2 and trainer.get("optimizer_updates_per_global_epoch") == 6,
            "The pooled two-core/six-update objective must be preserved")
    require(source.get("dataset", {}).get("test_partition_present") is False, "No test partition is permitted")
    runtime = config.get("runtime", {})
    require(runtime.get("effective_world_size", 1) == 1 and runtime.get("epochs", epochs) == epochs,
            "Runtime topology or epoch limit differs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible.strip()) and "," not in visible, "Exactly one explicit GPU must be visible")
    if "gpu" in runtime:
        require(str(runtime["gpu"]) == visible, "Selected physical GPU differs from resolved config")
    for relative in ("diagnostics/runtime_trial_result.json", "metrics/runtime_epoch_metrics.jsonl", "metrics/runtime_progress.jsonl"):
        require(not (run_dir / relative).exists(), "This fresh diagnostic cannot resume or overwrite existing worker output")
    return source


def host_snapshot() -> dict[str, Any]:
    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, raw = line.split(":", 1); status[key + "_bytes"] = int(raw.split()[0]) * 1024
    io: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1); io[key] = int(value)
    cgroup: dict[str, Any] = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.events", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            raw = path.read_text().strip(); cgroup[name] = int(raw) if raw.isdecimal() else raw
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"timestamp": utc_now(), "rss": status, "io": io, "cgroup": cgroup,
            "process_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
            "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime}


class ResourceSampler:
    def __init__(self, output: Path, gpu: str) -> None:
        self.output, self.gpu = output, gpu
        self.stop_event = threading.Event(); self.error: str | None = None
        self.minimum_free_bytes: int | None = None
        self.thread = threading.Thread(target=self._run, name="hurdle-resource-sampler", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = host_snapshot()
                result = subprocess.run(["nvidia-smi", "-i", self.gpu,
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8, check=False)
                sample["nvidia_smi_sample"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
                if result.returncode == 0:
                    fields = [item.strip() for item in result.stdout.strip().split(",")]
                    if len(fields) == 7:
                        sample["physical_memory_used_bytes"] = int(float(fields[3]) * 1024 ** 2)
                        sample["physical_memory_free_bytes"] = int(float(fields[4]) * 1024 ** 2)
                        free = sample["physical_memory_free_bytes"]
                        self.minimum_free_bytes = free if self.minimum_free_bytes is None else min(free, self.minimum_free_bytes)
                append_json(self.output, sample)
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
            self.stop_event.wait(5)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set(); self.thread.join(timeout=12)


def assert_finite_scalars(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items(): assert_finite_scalars(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value): assert_finite_scalars(item, f"{label}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite scalar: {label}")


def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve(strict=True)
    config = json.loads(args.config.read_text())
    source = validate_request(config, run_id=args.run_id, run_dir=run_dir, epochs=args.epochs)
    runtime = config.get("runtime", {})
    threads = int(runtime.get("cpu_threads", runtime.get("threads", 4)))
    require(threads >= 1, "CPU thread budget must be positive")
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device is required")
    require(shutil.disk_usage(run_dir).free >= float(runtime.get("min_free_gib", 20)) * GIB,
            "Insufficient free disk for the frozen safety floor")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    initial_cuda_free, initial_cuda_total = torch.cuda.mem_get_info(device)
    require(props.total_memory > 2 * GIB, "Device has inadequate VRAM")
    # These module assignments are local to this separate diagnostic process.
    # The immutable four-rank configuration and historical entry points remain unchanged.
    from scripts.train import run_so2_target_isolated_hurdle_nb as hurdle
    base = hurdle.base
    base.WORLD_SIZE = 1
    base.VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    phase = "initialization"; records: list[dict[str, Any]] = []
    sampler = ResourceSampler(run_dir / "diagnostics/runtime_resources.jsonl", base.VISIBLE_DEVICES)
    started = time.monotonic(); control_group = None

    def progress(event: str, **values: Any) -> None:
        payload = {"timestamp": utc_now(), "run_id": args.run_id, "event": event, **values}
        append_json(run_dir / "metrics/runtime_progress.jsonl", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    def measured_free(cuda_free: int) -> int:
        return min(cuda_free, sampler.minimum_free_bytes) if sampler.minimum_free_bytes is not None else cuda_free

    environment = {"schema": "so2_hurdle_gpu_runtime_environment_v1", "python": sys.version,
                   "platform": platform.platform(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "gpu_name": props.name,
                   "total_device_bytes": props.total_memory, "visible_physical_gpu": base.VISIBLE_DEVICES,
                   "initial_cuda_free_bytes": initial_cuda_free, "initial_cuda_total_bytes": initial_cuda_total,
                   "initial_driver_used_bytes": initial_cuda_total - initial_cuda_free,
                   "physical_memory_sampling": "nvidia-smi used/free every five seconds; CUDA free at update and validation boundaries",
                   "effective_world_size": 1, "source_world_size": source["trainer"]["distributed_world_size"],
                   "cpu_threads": threads, "interop_threads": torch.get_num_interop_threads(),
                   "cpu_affinity_count": len(os.sched_getaffinity(0)), "initial_host": host_snapshot(),
                   "configuration_sha256": sha256_file(args.config), "initialization": "fresh_seed0_no_checkpoint",
                   "original_preflight_claimed": False, "dropout_trajectory_matches_four_rank": False}
    write_json(run_dir / "diagnostics/runtime_environment.json", environment)
    try:
        rendezvous = run_dir / "diagnostics/one_rank_rendezvous"
        require(not rendezvous.exists(), "Distributed rendezvous already exists")
        torch.distributed.init_process_group("nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1,
                                             timeout=timedelta(minutes=20))
        control_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(minutes=20))
        sampler.start(); phase = "load_data"; progress("load_data_started")
        bundle = base._load_data(source, current_paths())
        require(tuple(batch.alias for batch in bundle.training_batches) == base.TRAINING_ALIASES,
                "Training core aliases differ from the frozen cohort")
        require(tuple(batch.alias for batch in bundle.validation_batches) == base.VALIDATION_ALIASES,
                "Validation core aliases differ from the frozen cohort")
        for batch in bundle.training_batches + bundle.validation_batches:
            base._validate_cpu_graph(batch)
        require(sum(b.n_nodes for b in bundle.training_batches) == 208696 and
                sum(b.n_nodes for b in bundle.validation_batches) == 37367, "Dataset cell coverage differs")
        progress("load_data_completed")
        phase = "build_model"
        model = hurdle._build_model(source, bundle)
        training_model, optimizer, scheduler, scaler = base._initialize_training(model, source, device)
        trainer = source["trainer"]
        global_tracker = base.FullGradientDirectionTracker(expected_optimizer_updates_per_epoch=6)
        block_tracker = base.BlockGradientDirectionTracker(expected_blocks=4, expected_optimizer_updates_per_epoch=6)
        theta_tracker = base.FullGradientDirectionTracker(expected_optimizer_updates_per_epoch=6)
        theta_view = base._make_tracked_head_view(model)
        fixed_validation_sha: str | None = None
        minimum_cuda_free = initial_cuda_free
        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1; phase = f"epoch_{epoch}_training"
            torch.cuda.synchronize(); epoch_started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
            order = base.cohort_relative_qkv_core_order(epoch, aliases=base.TRAINING_ALIASES, core_order_seed=base.CORE_ORDER_SEED)
            require(len(order) == 12 and set(order) == set(base.TRAINING_ALIASES), "Core order is incomplete")
            local_training: dict[str, Any] = {}; receipts = []
            epoch_minimum_cuda_free = torch.cuda.mem_get_info(device)[0]
            for update_index in range(6):
                pair = tuple(order[2 * update_index:2 * update_index + 2])
                progress("paired_update_started", epoch=epoch, update=update_index + 1, core_aliases=pair)
                local, coverage, grad_norm = hurdle._paired_update(model=model, training_model=training_model,
                    optimizer=optimizer, scaler=scaler, pair=pair, update_index=update_index, global_epoch=epoch_index,
                    rank=0, device=device, batches=bundle.batches_by_alias, trainer=trainer, control_group=control_group,
                    global_tracker=global_tracker, block_tracker=block_tracker, theta_tracker=theta_tracker, theta_view=theta_view)
                require(not set(local_training).intersection(local), "Duplicate training core")
                require(math.isfinite(grad_norm) and grad_norm > 0, "Invalid update gradient norm")
                local_training.update(local); receipts.extend(coverage)
                torch.cuda.synchronize()
                cuda_free, cuda_total = torch.cuda.mem_get_info(device)
                epoch_minimum_cuda_free = min(epoch_minimum_cuda_free, cuda_free)
                minimum_cuda_free = min(minimum_cuda_free, cuda_free)
                progress("paired_update_completed", epoch=epoch, update=update_index + 1,
                         gradient_norm=grad_norm, elapsed_epoch_seconds=time.monotonic() - epoch_started,
                         cuda_free_bytes=cuda_free, cuda_total_bytes=cuda_total)
                current_memory = memory_gate(allocated=torch.cuda.max_memory_allocated(device),
                    reserved=torch.cuda.max_memory_reserved(device), total=props.total_memory,
                    sampled_free=measured_free(minimum_cuda_free))
                require(current_memory["resource_gate_passed"], "Paired update exceeded VRAM/headroom gate")
                require(cuda_free >= 2 * GIB, "Paired update has less than two GiB measured device headroom")
            train_loss, train_entries, train_targets, per_core = hurdle._aggregate_training_epoch(local_training, device=device)
            train_receipts = base._validate_epoch_coverage(receipts, global_epoch=epoch, control_group=control_group)
            gradients = {"full_model": asdict(global_tracker.complete_epoch(epoch_index)),
                         "blocks": [asdict(x) for x in block_tracker.complete_epoch(epoch_index)],
                         "dispersion": asdict(theta_tracker.complete_epoch(epoch_index))}
            assert_finite_scalars(gradients, "gradient_summaries")
            require(all(row["gradient_norm_mean_before_clip"] > 0 for row in gradients["blocks"]), "A graph block has zero gradients")
            training_seconds = time.monotonic() - epoch_started
            phase = f"epoch_{epoch}_validation"; progress("validation_started", epoch=epoch)
            validation_started = time.monotonic()
            validation = hurdle._evaluate_validation(bundle=bundle, training_model=training_model, rank=0,
                device=device, trainer=trainer, control_group=control_group)
            torch.cuda.synchronize(); validation_seconds = time.monotonic() - validation_started
            cuda_free, _ = torch.cuda.mem_get_info(device)
            epoch_minimum_cuda_free = min(epoch_minimum_cuda_free, cuda_free)
            minimum_cuda_free = min(minimum_cuda_free, cuda_free)
            validation_sha = base._canonical_sha256(validation.coverage_receipts)
            if fixed_validation_sha is None: fixed_validation_sha = validation_sha
            require(validation_sha == fixed_validation_sha, "Fixed validation masks changed across epochs")
            require(validation.target_cells == 37367 and train_targets == 208696, "Epoch target coverage incomplete")
            scheduler.step(validation.pooled_hurdle_negative_binomial_nll)
            coverage_path = run_dir / f"diagnostics/coverage_epoch_{epoch:03d}.json"
            write_json(coverage_path, {"schema": "so2_hurdle_gpu_runtime_coverage_v1", "epoch": epoch,
                       "training": train_receipts, "validation": validation.coverage_receipts})
            memory = memory_gate(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device),
                                 total=props.total_memory, sampled_free=measured_free(minimum_cuda_free))
            row = {"schema": "so2_hurdle_gpu_runtime_epoch_v1", "epoch": epoch, "run_id": args.run_id,
                   "duration_seconds": time.monotonic() - epoch_started, "training_duration_seconds": training_seconds,
                   "validation_duration_seconds": validation_seconds, "train_loss": train_loss,
                   "validation_loss": validation.pooled_hurdle_negative_binomial_nll,
                   "train_masked_entries": train_entries, "validation_masked_entries": validation.masked_entries,
                   "train_target_cells": train_targets, "validation_target_cells": validation.target_cells,
                   "train_positive_entries": sum(int(r["positive_entries"]) for r in per_core.values()),
                   "validation_positive_entries": validation.positive_entries, "optimizer_updates": 6,
                   "learning_rate": float(optimizer.param_groups[0]["lr"]), **memory,
                   "minimum_sampled_cuda_free_bytes": epoch_minimum_cuda_free,
                   "coverage_file": coverage_path.relative_to(run_dir).as_posix(), "coverage_sha256": sha256_file(coverage_path),
                   "validation_mask_receipts_sha256": validation_sha, "finite_gradients": True,
                   "gradient_summaries": gradients, "host": host_snapshot()}
            row["training_targets_per_second"] = train_targets / training_seconds
            row["total_targets_per_second_including_validation"] = (train_targets + validation.target_cells) / row["duration_seconds"]
            assert_finite_scalars(row, "epoch")
            append_json(run_dir / "metrics/runtime_epoch_metrics.jsonl", row); records.append(row)
            progress("epoch_completed", epoch=epoch, train_loss=train_loss, validation_loss=row["validation_loss"],
                     duration_seconds=row["duration_seconds"], peak_allocated_gib=memory["peak_allocated_bytes"] / GIB,
                     peak_reserved_gib=memory["peak_reserved_bytes"] / GIB)
            require(memory["resource_gate_passed"], "Validation exceeded VRAM/headroom gate")
            require(cuda_free >= 2 * GIB, "Validation has less than two GiB measured device headroom")
        phase = "completed"; sampler.stop()
        require(sampler.error is None, f"Resource sampler failed: {sampler.error}")
        result = {"schema": RESULT_SCHEMA, "status": "passed", "run_id": args.run_id,
                  "epochs_requested": args.epochs, "epochs_completed": len(records), "effective_world_size": 1,
                  "correctness_gate_passed": True, "resource_gate_passed": True,
                  "duration_seconds": time.monotonic() - started, "epochs": records,
                  "minimum_sampled_cuda_free_bytes": minimum_cuda_free,
                  **memory_gate(allocated=max(r["peak_allocated_bytes"] for r in records),
                      reserved=max(r["peak_reserved_bytes"] for r in records), total=props.total_memory,
                      sampled_free=measured_free(minimum_cuda_free)),
                  "final_validation_metrics": dict(hurdle._final_metrics(validation)), "final_host": host_snapshot(),
                  "source_data_fingerprints": {k: source["dataset"][k] for k in
                       ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint")},
                  "original_four_rank_preflight_claimed": False, "scientific_replication_claim": False,
                  "dropout_trajectory_equivalence_claim": False, "checkpoint_count": 0, "prediction_matrices_persisted": False}
        require(result["resource_gate_passed"], "Final sampled physical headroom failed")
    except Exception as error:
        if sampler.thread.ident is not None: sampler.stop()
        result = {"schema": RESULT_SCHEMA, "status": "failed", "run_id": args.run_id,
                  "epochs_requested": args.epochs, "epochs_completed": len(records), "effective_world_size": 1,
                  "correctness_gate_passed": False, "resource_gate_passed": False,
                  "duration_seconds": time.monotonic() - started, "epochs": records,
                  "error": {"type": type(error).__name__, "message": str(error), "phase": phase},
                  "final_host": host_snapshot(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "total_device_bytes": props.total_memory}
        traceback.print_exc()
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
    write_json(run_dir / "diagnostics/runtime_trial_result.json", result)
    progress("worker_terminal", status=result["status"], epochs_completed=result["epochs_completed"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        return 0 if run_trial(args)["status"] == "passed" else 2
    except Exception as error:
        print(f"Worker setup failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

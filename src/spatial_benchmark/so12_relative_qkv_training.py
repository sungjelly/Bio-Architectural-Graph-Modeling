"""Joint-slide original Relative-QKV training with explicit validation selection.

The historical fit-only trainers remain unchanged. Inputs are recomputed from
visible counts for every mask; precomputed Pearson targets are loss-only.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .adjacency_ablation import sample_uniform_mask_numpy, mask_realization_sha256
from .masking import derive_mask_seed
from .pooled_relative_qkv_training_v2 import (
    cohort_relative_qkv_core_pairs, cohort_relative_qkv_distributed_assignment,
    cohort_relative_qkv_model_step_seed,
)
from .relative_qkv_graph_transformer import ReceiverChunkedRelativeGeometryQKVGraphTransformer
from .run_archive import RunArchive

PROTOCOL = "so12_pearson_relative_qkv_train_validation_v1"
CHECKPOINT_SCHEMA = "so12_relative_qkv_validation_checkpoint_v1"
PRIMARY = "val/masked_huber"
VALIDATION_NAMESPACE = "so12_fixed_validation_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scientific_binding(config: Mapping[str, Any]) -> str:
    # Launch location and preflight receipt cannot change the scientific run.
    return digest({k: config[k] for k in
                   ("dataset", "features", "graph", "model", "masking", "trainer", "evaluation", "seed")
                   if k in config})


def source_binding() -> dict[str, str]:
    base = Path(__file__).parent
    return {name: file_hash(base / name) for name in
            ("so12_relative_qkv_training.py", "so12_relative_qkv_data.py",
             "relative_qkv_graph_transformer.py", "pooled_relative_qkv_training_v2.py")}


def atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        check = torch.load(temporary, map_location="cpu", weights_only=False)
        if check.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Checkpoint schema mismatch after atomic write")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class SelectionState:
    epoch: int = 0
    best_value: float | None = None
    best_epoch: int = 0
    patience_best: float | None = None
    bad_validations: int = 0
    improved: bool = False
    stop_reason: str | None = None


def advance_selection(state: SelectionState, value: float, *, minimum: int = 50,
                      patience: int = 25, delta: float = 1e-4, maximum: int = 300) -> SelectionState:
    if not math.isfinite(value) or minimum >= maximum or patience <= 0 or state.stop_reason:
        raise ValueError("Invalid validation stopping input")
    epoch = state.epoch + 1
    improved = state.best_value is None or value < state.best_value
    meaningful = state.patience_best is None or value < state.patience_best - delta
    bad = 0 if meaningful else state.bad_validations + 1
    reason = "maximum_epochs" if epoch >= maximum else (
        "patience" if epoch >= minimum and bad >= patience else None)
    return SelectionState(epoch, value if improved else state.best_value,
                          epoch if improved else state.best_epoch,
                          value if meaningful else state.patience_best, bad, improved, reason)


def build_model(config: Mapping[str, Any], *, genes: int, covariates: int) -> nn.Module:
    m = config["model"]
    defaults = dict(hidden_dim=256, attention_heads=8, attention_head_dim=32,
                    graph_layers=4, ffn_dim=1024, decoder_dim=1024,
                    positional_bias_hidden_dim=128, dropout=.1, attention_dropout=0.,
                    relative_geometry_dim=70, receiver_chunk_size=128,
                    max_edges_per_chunk=50000, activation_checkpointing=True)
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=genes, node_covariate_dim=covariates,
        **{k: m.get(k, v) for k, v in defaults.items()})


def make_mask(core: Any, *, epoch: int, view: int, training: bool,
              base_seed: int = 2026082401) -> tuple[np.ndarray, dict[str, Any]]:
    n, g = core.raw_counts.shape
    if training:
        seed = derive_mask_seed(base_seed, "so12_training_v1", core.alias, epoch, view)
    else:
        seed = derive_mask_seed(base_seed, VALIDATION_NAMESPACE, core.alias, view)
    for attempt in range(1025):
        effective = seed if attempt == 0 else derive_mask_seed(seed, "nonempty-view", attempt)
        realization = sample_uniform_mask_numpy(n, g, seed=effective, chunk_cells=2048)
        count = int(realization.masked_gene_counts.sum())
        if count:
            counts = realization.masked_gene_counts
            return realization.mask, {
                "alias": core.alias, "view": view, "epoch": epoch if training else None,
                "seed": int(effective), "masked_entries": count,
                "checksum": mask_realization_sha256(realization.mask, counts, seed=effective),
                "min_masked_genes": int(counts.min()), "max_masked_genes": int(counts.max()),
                "zero_mask_cells": int(np.count_nonzero(counts == 0)),
                "full_mask_cells": int(np.count_nonzero(counts == g)),
                "resample_attempt": attempt,
            }
    raise ValueError("Empty masks exhausted deterministic resampling")


def stage_core(core: Any, device: torch.device, *, stage_graph: bool = False) -> dict[str, torch.Tensor]:
    # The locked campaign stages complete graphs with half-precision geometry;
    # the CPU reference can instead retain memory-mapped geometry.
    return {
        "raw": torch.as_tensor(np.asarray(core.raw_counts)).to(device=device, dtype=torch.int32),
        "target": torch.as_tensor(np.asarray(core.target_expression)).to(device=device, dtype=torch.float32),
        "covariates": torch.as_tensor(np.asarray(core.node_covariates)).to(device=device, dtype=torch.float32),
        "edges": torch.as_tensor(np.asarray(core.edge_index)).to(device if stage_graph else "cpu"),
        "geometry": torch.as_tensor(np.asarray(core.relative_geometry)).to(
            device=device if stage_graph else "cpu", dtype=torch.float16 if stage_graph else torch.float32),
    }


def normalization_tensors(dataset: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(np.asarray(getattr(dataset, name)), device=device, dtype=torch.float32)
            for name in ("gene_p", "residual_mean", "residual_sd")}


def forward_view(model: nn.Module, staged: Mapping[str, torch.Tensor], mask: torch.Tensor,
                 normalization: Mapping[str, torch.Tensor], *, amp: bool) -> torch.Tensor:
    from .so12_relative_qkv_data import visible_count_pearson_input
    visible = visible_count_pearson_input(staged["raw"], mask, **normalization)
    with torch.autocast(device_type=visible.device.type, dtype=torch.bfloat16, enabled=amp):
        output = model(input_expression=visible, gene_mask=mask,
                       edge_index=staged["edges"], relative_geometry=staged["geometry"],
                       node_covariates=staged["covariates"])
    return output.prediction.float()


def masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        raise ValueError("A whole mask view must contain observations")
    return torch.nn.functional.huber_loss(prediction[mask], target[mask], delta=1., reduction="mean")


def sufficient_statistics(prediction: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> list[float]:
    p, y = prediction[mask].double(), target[mask].double()
    e = p - y
    a = e.abs()
    huber = torch.where(a <= 1., .5 * e.square(), a - .5)
    return [float(x) for x in (torch.tensor(y.numel(), device=y.device), huber.sum(),
                              a.sum(), e.square().sum(), p.sum(), y.sum(),
                              p.square().sum(), y.square().sum(), (p * y).sum())]


def aggregate_statistics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Missing validation statistics")
    by_core: dict[str, list[float]] = {}
    keys = set()
    for record in records:
        key = (record["alias"], record["view"])
        if key in keys:
            raise ValueError("Duplicate validation core/view")
        keys.add(key)
        s = record["statistics"]
        if s[0] <= 0 or not all(math.isfinite(float(x)) for x in s):
            raise ValueError("Invalid validation sufficient statistics")
        by_core.setdefault(record["alias"], []).append(s[1] / s[0])
    total = np.sum([r["statistics"] for r in records], axis=0)
    n, hub, absolute, squared, sp, sy, spp, syy, spy = map(float, total)
    denominator = max(0., n * spp - sp * sp) * max(0., n * syy - sy * sy)
    return {
        "equal_core_huber": float(np.mean([np.mean(v) for v in by_core.values()])),
        "pooled_huber": hub / n, "pooled_mae": absolute / n,
        "pooled_mse": squared / n,
        "pooled_pearson": (n * spy - sp * sy) / math.sqrt(denominator) if denominator else None,
        "masked_entries": int(n), "per_core_huber": {k: float(np.mean(v)) for k, v in by_core.items()},
        "view_count": len(records),
    }


def gather(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    result: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def clear_graph(model: nn.Module) -> None:
    underlying = model.module if isinstance(model, DistributedDataParallel) else model
    underlying.clear_edge_layout_cache()


def evaluate(model: nn.Module, cores: list[Any], normalization: Mapping[str, torch.Tensor],
             *, device: torch.device, rank: int, world: int, views: int = 3,
             base_seed: int = 2026082401, amp: bool = True,
             constant: torch.Tensor | None = None, stage_graph: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Use the unwrapped model: unequal numbers of inference views must not enter
    # DDP forward collectives. Every rank participates in the final gather.
    underlying = model.module if isinstance(model, DistributedDataParallel) else model
    underlying.eval()
    local = []
    # The original receiver-layout cache reads tensor version counters.
    with torch.no_grad():
        for i, core in enumerate(cores):
            assigned = [v for v in range(views) if (i * views + v) % world == rank]
            if not assigned:
                continue
            clear_graph(underlying)
            staged = stage_core(core, device, stage_graph=stage_graph)
            for view in assigned:
                raw_mask, receipt = make_mask(core, epoch=0, view=view, training=False, base_seed=base_seed)
                mask = torch.from_numpy(raw_mask).to(device=device, dtype=torch.bool)
                pred = (forward_view(underlying, staged, mask, normalization, amp=amp)
                        if constant is None else constant.unsqueeze(0).expand_as(staged["target"]))
                local.append({"alias": core.alias, "view": view, "receipt": receipt,
                              "statistics": sufficient_statistics(pred, staged["target"], mask),
                              "nodes": len(core.raw_counts), "edges": int(core.edge_index.shape[1])})
                del pred, mask, raw_mask
            clear_graph(underlying)
            del staged
    records = sorted([r for part in gather(local) for r in part], key=lambda r: (r["alias"], r["view"]))
    if {(r["alias"], r["view"]) for r in records} != {(c.alias, v) for c in cores for v in range(views)}:
        raise ValueError("Validation mask coverage failed")
    underlying.train()
    return aggregate_statistics(records), records


def paired_update(model: nn.Module, optimizer: torch.optim.Optimizer, pair: tuple[Any, Any],
                  normalization: Mapping[str, torch.Tensor], *, epoch: int, update: int,
                  rank: int, world: int, device: torch.device, amp: bool = True,
                  base_seed: int = 2026082401, model_seed: int = 0,
                  stage_graph: bool = False, progress: bool = False) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if world == 4:
        assignment = cohort_relative_qkv_distributed_assignment(
            (pair[0].alias, pair[1].alias), rank=rank, world_size=4)
        jobs = [(pair[assignment.position_in_optimizer_update], assignment.view_indices)]
        divisor = 5
    elif world == 1:
        jobs, divisor = [(core, tuple(range(10))) for core in pair], 20
    else:
        raise ValueError("Training requires four DDP ranks or the single-process reference")
    receipts, loss_values = [], []
    done = 0
    for core, views in jobs:
        clear_graph(model)
        staged = stage_core(core, device, stage_graph=stage_graph)
        for view in views:
            seed = cohort_relative_qkv_model_step_seed(model_seed, epoch, update, core.alias, view)
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            raw_mask, receipt = make_mask(core, epoch=epoch, view=view, training=True, base_seed=base_seed)
            mask = torch.from_numpy(raw_mask).to(device=device, dtype=torch.bool)
            done += 1
            context = model.no_sync() if isinstance(model, DistributedDataParallel) and done < divisor else nullcontext()
            with context:
                pred = forward_view(model, staged, mask, normalization, amp=amp)
                loss = masked_loss(pred, staged["target"], mask)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training Huber")
                (loss / divisor).backward()
            loss_values.append(float(loss.detach()))
            receipts.append(receipt)
            if progress:
                print(f"[bagm-preflight] rank={rank} core={core.alias} view={view} "
                      f"huber={float(loss.detach()):.8f} backward=complete", flush=True)
            del pred, loss, mask, raw_mask
        clear_graph(model)
        del staged
    for name, p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    if norm <= 0:
        raise FloatingPointError("Zero full-model gradient")
    norms = gather(norm)
    if max(norms) - min(norms) > 1e-5 * max(norms) + 1e-7:
        raise ValueError("DDP gradient norms disagree")
    optimizer.step()
    all_receipts = [r for part in gather(receipts) for r in part]
    if {(r["alias"], r["view"]) for r in all_receipts} != {(c.alias, v) for c in pair for v in range(10)}:
        raise ValueError("Training mask coverage failed")
    return {"loss": float(np.mean([v for part in gather(loss_values) for v in part])),
            "gradient_norm": norm, "receipts": sorted(all_receipts, key=lambda r: (r["alias"], r["view"]))}


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def append_history(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _rank_setup() -> tuple[int, int, torch.device]:
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    torch.set_num_threads(4)
    return rank, world, device


def _metadata(dataset: Any) -> dict[str, Any]:
    return {"provenance": dataset.provenance,
            "train_aliases": [c.alias for c in dataset.train_cores],
            "validation_aliases": [c.alias for c in dataset.validation_cores],
            "train_cells": sum(len(c.raw_counts) for c in dataset.train_cores),
            "validation_cells": sum(len(c.raw_counts) for c in dataset.validation_cores)}


def _preflight(config: Mapping[str, Any], output: Path, dataset: Any, model: nn.Module,
               training_model: nn.Module, optimizer: Any, normalization: Any,
               rank: int, world: int, device: torch.device) -> dict[str, Any] | None:
    from .so12_relative_qkv_data import visible_count_pearson_input
    started = time.monotonic()
    base_seed = int(config["masking"]["mask_base_seed"])
    stage_graph = bool(config["trainer"].get("stage_complete_core_graph_on_device", False))
    train = sorted(dataset.train_cores, key=lambda c: len(c.raw_counts), reverse=True)[:2]
    val = sorted(dataset.validation_cores, key=lambda c: len(c.raw_counts), reverse=True)[:1]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if rank == 0:
        print(f"[bagm-preflight] pair={[c.alias for c in train]} "
              f"cells={[c.n_cells for c in train]} start", flush=True)
    update = paired_update(training_model, optimizer, tuple(train), normalization,
                           epoch=0, update=0, rank=rank, world=world, device=device,
                           base_seed=base_seed, stage_graph=stage_graph, progress=True)
    if rank == 0:
        print(f"[bagm-preflight] validation={[c.alias for c in val]} start", flush=True)
    metrics, records = evaluate(model, val, normalization, device=device, rank=rank,
                               world=world, views=3, base_seed=base_seed, stage_graph=stage_graph)
    # Bounded numerical and hidden-count tests use real inputs and the trained
    # weights; the complete graphs above independently cover peak-memory risk.
    core = train[0]
    raw = torch.tensor(np.asarray(core.raw_counts[:32]), device=device)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[:, ::2] = True
    other = raw.clone(); other[mask] += 123
    a = visible_count_pearson_input(raw, mask, **normalization)
    b = visible_count_pearson_input(other, mask, **normalization)
    invariant = torch.equal(a, b)
    n = len(raw)
    edges = torch.stack((torch.arange(n), (torch.arange(n) + 1) % n)).long()
    staged = {"raw": raw, "target": torch.tensor(np.asarray(core.target_expression[:n]), device=device),
              "covariates": torch.tensor(np.asarray(core.node_covariates[:n]), device=device),
              "edges": edges, "geometry": torch.zeros((n, 70))}
    clear_graph(model); model.eval()
    gradients, outputs = [], []
    for amp in (False, True):
        prediction = forward_view(model, staged, mask, normalization, amp=amp)
        loss = masked_loss(prediction, staged["target"], mask)
        gradient = torch.autograd.grad(loss, tuple(model.parameters()))
        gradients.append(torch.cat([g.detach().float().flatten() for g in gradient]))
        outputs.append(prediction.detach())
        del prediction, loss, gradient
    difference = (outputs[0] - outputs[1]).abs()
    gradient_relative_l2 = float((gradients[0] - gradients[1]).norm() / gradients[0].norm())
    gradient_cosine = float(torch.nn.functional.cosine_similarity(gradients[0], gradients[1], dim=0))
    equivalence = (float(difference.max()) <= .05 and float(difference.mean()) <= .01
                   and gradient_relative_l2 <= .05 and gradient_cosine >= .99)
    model.train(); clear_graph(model)
    local = {"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.,
             "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
             "amp_max_abs": float(difference.max()), "amp_mean_abs": float(difference.mean()),
             "amp_gradient_relative_l2": gradient_relative_l2, "amp_gradient_cosine": gradient_cosine,
             "hidden_count_invariance": invariant, "amp_equivalence": equivalence}
    resources = gather(local)
    if not all(x["hidden_count_invariance"] and x["amp_equivalence"] and x["peak_allocated_gib"] < 21 for x in resources):
        raise ValueError(f"Preflight resource/numerical gate failed: {resources}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "preflight.ckpt"
        atomic_save(checkpoint, {"checkpoint_schema": CHECKPOINT_SCHEMA, "model_state_dict": cpu_tree(model.state_dict())})
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for k, v in model.state_dict().items():
            if not torch.equal(v.cpu(), loaded["model_state_dict"][k]):
                raise ValueError("Preflight checkpoint reload mismatch")
        checkpoint.unlink()
        receipt = {"protocol": PROTOCOL, "all_required_gates_passed": True,
                   "config_sha256": scientific_binding(config), "source_sha256": source_binding(),
                   "dataset_sha256": digest(_metadata(dataset)), "dataset": _metadata(dataset),
                   "world_size": world, "parameter_count": sum(p.numel() for p in model.parameters()),
                   "training_cores": [c.alias for c in train], "validation_cores": [c.alias for c in val],
                   "training_update": update, "validation": metrics,
                   "validation_masks_sha256": digest([r["receipt"] for r in records]),
                   "resources": resources, "checkpoint_reload_verified": True,
                   "duration_seconds": time.monotonic() - started,
                   "torch": torch.__version__, "cuda": torch.version.cuda}
        write_json(output / "preflight.json", receipt)
        print(json.dumps({"preflight": str(output / 'preflight.json'), "passed": True}), flush=True)
        return receipt
    return None


def run_training(config: Mapping[str, Any], output_dir: Path, paths: Any,
                 *, preflight: bool = False) -> dict[str, Any] | None:
    from .so12_relative_qkv_data import load_so12_training_data
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world, device = _rank_setup()
    seed = int(config.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if rank == 0:
        print(f"[bagm-training] loading dataset ranks={world} preflight={preflight}", flush=True)
    dataset = load_so12_training_data(config, paths, verify_checksums=rank == 0)
    metadata = gather(_metadata(dataset))[0]
    if world > 1:
        dist.barrier()
    if rank == 0:
        print(f"[bagm-training] loaded train_cells={metadata['train_cells']} "
              f"validation_cells={metadata['validation_cells']}; building original model", flush=True)
    model = build_model(config, genes=len(dataset.gene_p),
                        covariates=dataset.train_cores[0].node_covariates.shape[1]).to(device)
    if sum(p.numel() for p in model.parameters()) != 5_003_016:
        raise ValueError("Original four-block architecture parameter count drifted")
    ddp_options = {"broadcast_buffers": False}
    if device.type == "cuda":
        ddp_options.update(device_ids=[device.index], output_device=device.index)
    training_model = DistributedDataParallel(model, **ddp_options) if world > 1 else model
    trainer = config["trainer"]
    base_seed = int(config["masking"]["mask_base_seed"])
    core_order_seed = int(trainer["core_order_seed"])
    stage_graph = bool(trainer.get("stage_complete_core_graph_on_device", False))
    if (config["masking"].get("train_seed_namespace") != "so12_training_v1"
            or config["evaluation"].get("mask_seed_namespace") != VALIDATION_NAMESPACE):
        raise ValueError("Mask namespace differs from locked trainer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    normalization = normalization_tensors(dataset, device)
    output = Path(output_dir)
    if preflight:
        return _preflight(config, output, dataset, model, training_model, optimizer,
                          normalization, rank, world, device)
    if world != 4 or device.type != "cuda":
        raise ValueError("Production requires four CUDA DDP ranks")
    receipt_path = Path(config["launcher"]["preflight_receipt"])
    if not receipt_path.is_absolute():
        receipt_path = paths.project_root / receipt_path
    receipt = json.loads(receipt_path.read_text())
    if (not receipt.get("all_required_gates_passed") or receipt.get("world_size") != 4
            or receipt.get("config_sha256") != scientific_binding(config)
            or receipt.get("source_sha256") != source_binding()
            or receipt.get("dataset_sha256") != digest(metadata)):
        raise ValueError("Production configuration/data/source differ from passing preflight")
    run_id = os.environ.get("BAGM_RUN_ID", output.name)
    archive = RunArchive.attach_active(run_id, paths=paths, scratch_path=output) if rank == 0 else None
    latest_path, best_path = output / "checkpoints/latest.ckpt", output / "checkpoints/best.ckpt"
    history_path = output / "results/epoch_metrics.jsonl"
    state = SelectionState()
    history: list[dict[str, Any]] = []
    best_snapshot = None
    binding = {"config_sha256": scientific_binding(config), "dataset_sha256": digest(metadata),
               "source_sha256": source_binding()}
    configured_resume = config["launcher"].get("resume_checkpoint")
    resume_path = Path(configured_resume) if configured_resume else latest_path
    if not resume_path.is_absolute():
        resume_path = paths.project_root / resume_path
    source_run_id = run_id
    if configured_resume and not resume_path.is_file():
        raise ValueError("Configured resume checkpoint is missing")
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("binding") != binding or saved.get("checkpoint_role") != "latest":
            raise ValueError("Resume fingerprint/ownership mismatch")
        if saved.get("run_id") != run_id and not configured_resume:
            raise ValueError("Cross-run resume requires an explicit checkpoint path")
        source_run_id = str(saved["run_id"])
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        state = SelectionState(**saved["selection_state"])
        history = saved["history"]
        best_snapshot = saved["best_snapshot"]
        restore_rng(saved["rng_states"][rank])
        if rank == 0:
            atomic_save(best_path, best_snapshot)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("".join(json.dumps(row) + "\n" for row in history))
    started = time.monotonic()
    aliases = tuple(c.alias for c in dataset.train_cores)
    core_map = {c.alias: c for c in dataset.train_cores}
    # The constant is fitted on training targets only, with equal core weight.
    constant_np = dataset.target_mean
    constant = torch.tensor(constant_np, device=device, dtype=torch.float32)
    baseline, baseline_records = evaluate(model, dataset.validation_cores, normalization,
                                          device=device, rank=rank, world=world, constant=constant,
                                          base_seed=base_seed, stage_graph=stage_graph)
    fixed_mask_digest = digest([r["receipt"] for r in baseline_records])
    if rank == 0:
        write_json(output / "diagnostics/training_start.json", {
            "run_id": run_id, "source_run_id": source_run_id, "binding": binding,
            "dataset": metadata, "constant_reference": baseline,
            "validation_mask_sha256": fixed_mask_digest})
        print(f"[bagm-training] ready train_cores={len(aliases)} validation_cores={len(dataset.validation_cores)} "
              f"constant_validation_huber={baseline['equal_core_huber']:.8f}", flush=True)
    while not state.stop_reason:
        epoch_started = time.monotonic()
        if shutil.disk_usage(output).free < 5 * 2**30:
            raise RuntimeError("Free disk below frozen 5 GiB floor")
        torch.cuda.reset_peak_memory_stats(device)
        pairs = cohort_relative_qkv_core_pairs(state.epoch, aliases=aliases, core_order_seed=core_order_seed)
        updates = []
        for i, (a, b) in enumerate(pairs):
            update = paired_update(training_model, optimizer, (core_map[a], core_map[b]), normalization,
                                    epoch=state.epoch, update=i, rank=rank, world=world, device=device,
                                    model_seed=seed, base_seed=base_seed, stage_graph=stage_graph)
            updates.append(update)
            if rank == 0:
                print(f"[bagm-training] epoch={state.epoch + 1} update={i + 1}/{len(pairs)} "
                      f"cores={a},{b} huber={update['loss']:.8f}", flush=True)
        validation, records = evaluate(model, dataset.validation_cores, normalization,
                                        device=device, rank=rank, world=world,
                                        base_seed=base_seed, stage_graph=stage_graph)
        if digest([r["receipt"] for r in records]) != fixed_mask_digest:
            raise ValueError("Fixed validation masks drifted")
        state = advance_selection(state, validation["equal_core_huber"],
                                  minimum=int(trainer.get("minimum_global_epochs", 50)),
                                  patience=int(trainer.get("early_stopping_patience", 25)),
                                  delta=float(trainer.get("early_stopping_min_delta", 1e-4)),
                                  maximum=int(trainer.get("max_epochs", 300)))
        states = gather(asdict(state))
        if any(x != states[0] for x in states):
            raise ValueError("Ranks disagree about checkpoint selection")
        memory = gather({"rank": rank, "allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                         "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30})
        row = {"global_epoch": state.epoch, "train_equal_core_huber": float(np.mean([u["loss"] for u in updates])),
               "validation": validation, "selection": asdict(state), "learning_rate": 1e-4,
               "optimizer_steps": state.epoch * len(pairs), "epoch_seconds": time.monotonic() - epoch_started,
               "mean_gradient_norm": float(np.mean([u["gradient_norm"] for u in updates])),
               "memory": memory, "validation_mask_sha256": fixed_mask_digest,
               "training_mask_sha256": digest([u["receipts"] for u in updates])}
        history.append(row)
        rngs = gather(rng_state())
        if rank == 0:
            snapshot = {"checkpoint_schema": CHECKPOINT_SCHEMA, "protocol": PROTOCOL, "run_id": run_id,
                        "source_run_id": source_run_id, "binding": binding, "completed_epoch": state.epoch,
                        "epoch": state.epoch, "best_epoch": state.best_epoch, "primary_metric_name": PRIMARY,
                        "primary_metric_value": state.best_value, "selection_state": asdict(state),
                        "model_state_dict": cpu_tree(model.state_dict()), "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                        "amp_scaler_state_dict": {}, "amp_dtype": "bfloat16", "scheduler_state_dict": None,
                        "rng_states": rngs, "history": history, "validation_masks_sha256": fixed_mask_digest,
                        "model_construction": dict(config["model"]), "baseline": baseline}
            if state.improved:
                best_snapshot = {**snapshot, "checkpoint_role": "best"}
                atomic_save(best_path, best_snapshot)
            atomic_save(latest_path, {**snapshot, "checkpoint_role": "latest", "best_snapshot": best_snapshot})
            append_history(history_path, row)
            assert archive is not None
            archive.append_metric_event({"run_id": run_id, "step": state.epoch, "split": "validation",
                                         "name": PRIMARY, "value": validation["equal_core_huber"]})
            write_json(output / "diagnostics/latest_mask_receipts.json",
                       {"epoch": state.epoch, "training": [u["receipts"] for u in updates],
                        "validation": [r["receipt"] for r in records]})
            print(f"[bagm-training] epoch={state.epoch} train_huber={row['train_equal_core_huber']:.8f} "
                  f"val_huber={validation['equal_core_huber']:.8f} best={state.best_value:.8f} "
                  f"bad={state.bad_validations}/25 seconds={row['epoch_seconds']:.1f}", flush=True)
        dist.barrier()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last_metric = history[-1]["validation"]["equal_core_huber"]
    if rank == 0:
        last = torch.load(latest_path, map_location="cpu", weights_only=False)
        last.pop("best_snapshot", None)
        last.update(checkpoint_role="last", primary_metric_value=last_metric)
        atomic_save(output / "checkpoints/last.ckpt", last)
    model.load_state_dict(best["model_state_dict"])
    replay, records = evaluate(model, dataset.validation_cores, normalization,
                               device=device, rank=rank, world=world,
                               base_seed=base_seed, stage_graph=stage_graph)
    if abs(replay["equal_core_huber"] - float(state.best_value)) > 1e-6:
        raise ValueError("Best checkpoint validation replay failed")
    if digest([r["receipt"] for r in records]) != fixed_mask_digest:
        raise ValueError("Best checkpoint validation mask replay failed")
    if rank != 0:
        dist.barrier()
        return None
    assert archive is not None
    final = {PRIMARY: replay["equal_core_huber"], "val/pooled/masked_huber": replay["pooled_huber"],
             "val/pooled/masked_mae": replay["pooled_mae"], "val/pooled/masked_mse": replay["pooled_mse"],
             "val/pooled/pearson": replay["pooled_pearson"], "val/baseline/equal_core_huber": baseline["equal_core_huber"]}
    archive.write_json("metrics/final.json", final)
    archive.write_table("metrics/history", [{"run_id": run_id, "step": r["global_epoch"],
                                              "split": "validation", "name": PRIMARY,
                                              "value": r["validation"]["equal_core_huber"]} for r in history])
    prediction_rows = []
    for r in records:
        s = r["statistics"]
        prediction_rows.append({"run_id": run_id, "sample_key": digest((PROTOCOL, r["alias"], r["view"])),
                                "dataset_id": config["dataset"]["dataset_id"], "split": "validation",
                                "y_true": s[5] / s[0], "y_pred": s[4] / s[0], "sample_loss": s[1] / s[0],
                                "node_count": r["nodes"], "edge_count": r["edges"],
                                "masked_entry_count": int(s[0]), "aggregation_unit": "core_mask_view_mean"})
    archive.write_predictions("validation", prediction_rows)
    archive.write_json("diagnostics/final_checkpoint_reload_verification.json",
                       {"verified": True, "checkpoint": "checkpoints/best.ckpt", "checkpoint_sha256": file_hash(best_path),
                        "best_epoch": state.best_epoch, "replayed_validation_metric": replay["equal_core_huber"],
                        "validation_mask_sha256": fixed_mask_digest})
    archive.write_json("diagnostics/hardware_preflight.json", receipt)
    archive.write_json("diagnostics/validation_per_core.json", replay)
    archive.write_json("diagnostics/constant_reference.json", baseline)
    archive.write_json("provenance/so12_training.json", {**metadata, "binding": binding})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([r["global_epoch"] for r in history], [r["train_equal_core_huber"] for r in history], label="Train: equal-core Huber")
    axis.plot([r["global_epoch"] for r in history], [r["validation"]["equal_core_huber"] for r in history], label="Validation: fixed masks")
    axis.axvline(state.best_epoch, color="gray", linestyle=":", label=f"Selected epoch {state.best_epoch}")
    axis.set(xlabel="Epoch", ylabel="Masked Huber loss")
    axis.legend(); figure.tight_layout()
    (output / "plots").mkdir(exist_ok=True)
    figure.savefig(output / "plots/training_validation_huber.png", dpi=160)
    plt.close(figure)
    summary = {"run_id": run_id, "status": "success", "protocol": PROTOCOL,
               "model_name": "relative-qkv-gat", "model_seed": seed,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "final_epoch": state.epoch, "best_epoch": state.best_epoch,
               "optimizer_steps": state.epoch * len(aliases) // 2,
               "primary_metric_name": PRIMARY, "primary_metric_value": replay["equal_core_huber"],
               "checkpoint": "checkpoints/best.ckpt", "checkpoint_reload_verified": True,
               "stop_reason": state.stop_reason, "duration_seconds": time.monotonic() - started,
               "peak_vram_gib": max(m["allocated_gib"] for r in history for m in r["memory"]),
               "world_size": world, "generalization_estimate": False, "test_artifacts_present": False}
    summary["checkpoints"] = {
        "best": {"path": "checkpoints/best.ckpt", "epoch": state.best_epoch,
                 "monitored_metric": PRIMARY, "monitored_mode": "min", "monitored_value": state.best_value},
        "last": {"path": "checkpoints/last.ckpt", "epoch": state.epoch,
                 "monitored_metric": PRIMARY, "monitored_mode": "min", "monitored_value": last_metric}}
    summary.update(best_validation_loss=state.best_value, final_validation_loss=last_metric)
    archive.write_summary(summary)
    latest_path.unlink()
    dist.barrier()
    return summary

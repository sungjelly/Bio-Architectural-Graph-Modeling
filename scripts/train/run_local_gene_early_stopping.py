#!/usr/bin/env python3
"""Registered Local Gene Residual follow-up with validation early stopping.

Run under supervisor with BAGM roots set as in the campaign README. This additive
runner reuses registry/archive/masking/likelihood contracts without altering any
historical model or training route.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from scripts.diagnostics.run_so2_hurdle_gpu_trial import global_worker_lease, inspect_gpu
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.hurdle_negative_binomial import masked_hurdle_negative_binomial_nll_sums
from spatial_benchmark.identifiers import canonical_sha256, create_run_id, scientific_id, repro_id
from spatial_benchmark.local_gene_data import load_local_gene_data, make_mask
from spatial_benchmark.local_gene_residual import LocalGeneResidualCountModel
from spatial_benchmark.local_gene_stopping import LocalGeneEarlyStopping
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, _environment_snapshot, _hardware_snapshot
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle

CAMPAIGN = "cmp_20261005_local_gene_early_stopping"
DEPENDENCY_CAMPAIGN = "cmp_20261005_so2_genemae_masking"
DEPENDENCY_RECEIPT_ENV = "BAGM_LOCAL_GENE_DEPENDENCY_RECEIPT"
PROTOCOL = "local_gene_validation_early_stopping_counts_v1"
PRIMARY = "val/count_nll"
GIB = 1024**3
SOURCE_FILES = (
    "scripts/train/run_local_gene_early_stopping.py",
    "src/spatial_benchmark/local_gene_stopping.py",
    "src/spatial_benchmark/local_gene_data.py",
    "src/spatial_benchmark/local_gene_residual.py",
    "src/spatial_benchmark/local_gene_metrics.py",
    "src/spatial_benchmark/target_cell_masking.py",
    "src/spatial_benchmark/hurdle_negative_binomial.py",
    "src/spatial_benchmark/negative_binomial.py",
    "src/spatial_benchmark/so2_nb_data.py",
    "src/spatial_benchmark/so2_relative_graphs.py",
    "src/spatial_benchmark/so2_pooled_full_core.py",
    "src/spatial_benchmark/paths.py",
    "src/spatial_benchmark/fingerprints.py",
    "src/spatial_benchmark/identifiers.py",
    "src/spatial_benchmark/registry.py",
    "src/spatial_benchmark/run_archive.py",
    "src/spatial_benchmark/queueing.py",
    "scripts/diagnostics/run_so2_hurdle_gpu_trial.py",
    "ops/supervisor/bagm-local-gene-early-stopping.sh",
)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def build_config(args):
    dataset = yaml.safe_load((ROOT / "configs/dataset/so2_hurdle_nb_train12_val2_v1.yaml").read_text())["dataset"]
    dataset["task"] = "target_isolated_local_gene_count_reconstruction"
    dataset["source_graph_cache_reused_without_copy"] = False
    dataset["source_graph_artifact_role"] = "historical_reference_only_not_loaded"
    contract = ROOT / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
    return {
        "campaign": {"campaign_id": CAMPAIGN}, "seed": args.seed, "fold": 0, "attempt": args.attempt,
        "dataset": dataset,
        "model": {"family": "local_gene_residual", "variant": args.variant, "hidden_dim": 512,
                  "hidden_layers": 2, "dropout": .1, "use_local": args.variant == "local",
                  "output_distribution": "bernoulli_hurdle_zero_truncated_nb2"},
        "graph": {"kind": "fixed_local_summary" if args.variant == "local" else "none",
                  "neighbor_k": 16 if args.variant == "local" else 0, "radius_um": 75., "self_loops": False},
        "masking": {"kind": "target_isolated_uniform_1_1000", "training_base_seed": 2026092601,
                    "validation_namespace": "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"},
        "trainer": {"epochs": 2 if args.profile == "smoke" else 500, "batch_size": 1024,
                    "minimum_epochs": 50, "maximum_epochs": 500,
                    "learning_rate": .001, "minimum_learning_rate": .00001, "weight_decay": .00001,
                    "gradient_clip": 1., "precision": "bfloat16_autocast_fp32_likelihood",
                    "primary_checkpoint_role": "best", "restore_best": True,
                    "ordering_seed": 2026100401, "scheduler": "reduce_on_plateau",
                    "scheduler_factor": .5, "scheduler_patience": 7,
                    "scheduler_threshold": .0001, "scheduler_threshold_mode": "abs",
                    "early_stopping": {"patience": 30, "min_delta": .0001, "min_epochs": 50, "max_epochs": 500},
                    "initialization": "fresh_seed_matched_not_resumed",
                    "retained_checkpoint_roles": ["best", "last"]},
        "evaluation": {"protocol": PROTOCOL, "primary_metric": PRIMARY,
                       "primary_mode": "min", "canonical_prediction_split": "validation",
                       "artifact_contract": "predictive"},
        "classification": {"lifecycle_stage": "diagnostic" if args.profile == "smoke" else "exploratory_validation",
                           "study_axis": "early_stopping", "variant_label": args.variant,
                           "retention_class": "diagnostic" if args.profile == "smoke" else "validation_confirmation"},
        "metadata": {"profile": args.profile, "frozen_contract_sha256": sha256_file(contract),
                     "analysis_type": "prospective_validation_reuse_exploratory_followup",
                     "prior_campaign_id": "cmp_20261004_local_gene_residual"},
        "launcher": {"requested_gpu": args.gpu, "smoke_run": args.smoke_run},
    }


def verify_pilot(registry, paths, run_id, hashes, config):
    record = registry.get_run(run_id)
    if not record or record["status"] != "completed":
        raise RuntimeError("Full experiment requires a completed registered pilot")
    root = RunArchive.artifact_path_for(run_id, paths)
    if Path(record["artifact_path"]).resolve() != root.resolve() or record["campaign_id"] != CAMPAIGN:
        raise RuntimeError("Pilot campaign or canonical artifact identity differs")
    verification = verify_run_bundle(root)
    if verification.get("ok") is False or verification.get("valid") is False:
        raise RuntimeError(f"Pilot bundle verification failed: {verification}")
    receipt = json.loads((root / "diagnostics/preflight.json").read_text())
    pilot_config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    pilot_summary = json.loads((root / "summary.json").read_text())
    if (pilot_config["metadata"]["profile"] != "smoke" or pilot_summary["final_epoch"] != 2
            or pilot_summary["stop_reason"] != "diagnostic_budget"
            or pilot_summary["early_stopping_triggered"] is not False
            or not receipt.get("terminal_checkpoint_replay_passed")):
        raise RuntimeError("Pilot diagnostic budget or checkpoint gates differ")
    if not receipt["passed"] or receipt["source_hashes"] != hashes:
        raise RuntimeError("Pilot correctness gates/source identity differ")
    if receipt["frozen_contract_sha256"] != config["metadata"]["frozen_contract_sha256"]:
        raise RuntimeError("Frozen contract changed since pilot")
    return {"run_id": run_id, "preflight_sha256": sha256_file(root / "diagnostics/preflight.json")}


def verify_dependency_receipt(path, paths, registry):
    """Recheck all nine completed GeneMAE slots before touching any GPU."""
    receipt_path = Path(path).resolve()
    receipt_sha = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    dependencies = receipt.get("dependencies")
    if (receipt.get("schema") != "local_gene_dependency_receipt_v1"
            or receipt.get("campaign_id") != DEPENDENCY_CAMPAIGN
            or not isinstance(receipt.get("plan_sha256"), str)
            or len(receipt["plan_sha256"]) != 64
            or not isinstance(dependencies, list) or len(dependencies) != 9):
        raise RuntimeError("Dependency receipt does not describe the required nine-slot matrix")
    slots, run_ids, job_ids = set(), set(), set()
    for dependency in dependencies:
        job = registry.get_job(dependency["job_id"])
        run = registry.get_run(dependency["run_id"])
        if not job or not run or job["status"] != "completed" or run["status"] != "completed":
            raise RuntimeError("A required GeneMAE dependency is not completed")
        if (job["campaign_id"] != DEPENDENCY_CAMPAIGN or run["campaign_id"] != DEPENDENCY_CAMPAIGN
                or job["run_id"] != dependency["run_id"]):
            raise RuntimeError("Dependency queue/run campaign identity differs")
        canonical = job.get("canonical_config")
        if canonical is None:
            canonical = job.get("canonical_config_json")
        if isinstance(canonical, str):
            canonical = json.loads(canonical)
        if not isinstance(canonical, dict) or canonical_sha256(canonical) != dependency["config_sha256"]:
            raise RuntimeError("Dependency queue configuration fingerprint differs")
        if (canonical["metadata"]["execution_role"] != "production"
                or canonical["trainer"]["max_epochs"] != 200):
            raise RuntimeError("Dependency is not a full 200-epoch GeneMAE production slot")
        slots.add((canonical["masking"]["method"], canonical["seed"]))
        run_ids.add(dependency["run_id"])
        job_ids.add(dependency["job_id"])
        root = RunArchive.artifact_path_for(dependency["run_id"], paths)
        if any(Path(value).resolve() != root.resolve() for value in (run["artifact_path"], dependency["artifact_path"])):
            raise RuntimeError("Dependency canonical artifact path differs")
        verify_run_bundle(root)
        published_config = yaml.safe_load((root / "config.resolved.yaml").read_text())
        strip_attempt = lambda value: {key: item for key, item in value.items() if key != "attempt"}
        for bound_config in (run.get("config"), published_config):
            if not isinstance(bound_config, dict) or strip_attempt(bound_config) != strip_attempt(canonical):
                raise RuntimeError("Dependency registry/published configuration differs from its exact queue slot")
        for name, field in (("_SUCCESS", "success_sha256"), ("summary.json", "summary_sha256"),
                            ("provenance/artifact_checksums.json", "checksum_manifest_sha256")):
            if sha256_file(root / name) != dependency[field]:
                raise RuntimeError("Dependency immutable artifact fingerprint differs")
    expected = {(method, seed) for method in ("whole_cell", "partial_gene", "spatial_block") for seed in (0, 1, 2)}
    if slots != expected or len(run_ids) != 9 or len(job_ids) != 9:
        raise RuntimeError("Dependency method/seed matrix is incomplete or duplicated")
    with registry.connect() as connection:
        active = connection.execute("SELECT COUNT(*) FROM queue_jobs WHERE campaign_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')", (DEPENDENCY_CAMPAIGN,)).fetchone()[0]
    if active:
        raise RuntimeError("Dependency campaign queue is not empty")
    if sha256_file(receipt_path) != receipt_sha:
        raise RuntimeError("Dependency receipt changed during verification")
    return {"path": str(receipt_path), "sha256": receipt_sha,
            "plan_sha256": receipt["plan_sha256"], "dependency_run_ids": sorted(run_ids)}, receipt


def make_optimizer_scheduler(model, config):
    trainer = config["trainer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer["learning_rate"], weight_decay=trainer["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
        factor=trainer["scheduler_factor"], patience=trainer["scheduler_patience"],
        threshold=trainer["scheduler_threshold"], threshold_mode=trainer["scheduler_threshold_mode"],
        min_lr=trainer["minimum_learning_rate"])
    return optimizer, scheduler


def checkpoint_state(*, archive, config, epoch, model, optimizer, scheduler, policy, metrics, hashes):
    return {"schema": "local_gene_early_stopping_checkpoint_v1", "run_id": archive.run_id, "epoch": epoch,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "early_stopping": policy.state_dict(), "config": config, "metrics": metrics,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "source_hashes": hashes}


def replay_delta(observed, expected):
    if set(observed) != set(expected):
        raise RuntimeError("Checkpoint replay metric schema differs")
    delta = max(abs(observed[key] - expected[key]) for key in observed
                if observed[key] is not None and expected[key] is not None)
    if not math.isfinite(delta) or delta > 1e-5:
        raise RuntimeError(f"Checkpoint replay mismatch: {delta}")
    return delta


def stopping_receipt(policy, profile):
    state = policy.state_dict()
    # A diagnostic budget never claims scientific early stopping or convergence.
    reason = "diagnostic_budget" if profile == "smoke" else policy.stop_reason
    if profile == "full" and reason not in {"patience", "maximum_epochs"}:
        raise RuntimeError("Full run ended without its declared stopping condition")
    return {"stop_reason": reason, "early_stopping_triggered": profile == "full" and reason == "patience",
        "safety_cap_reached": profile == "full" and policy.completed_epoch >= policy.config.max_epochs,
        "policy_state": state}


def commit_validation_epoch(*, archive, config, epoch, model, optimizer, scheduler, policy, metrics, hashes):
    """Apply validation-only controls, then atomically preserve both states."""
    decision = policy.observe(epoch=epoch, count_nll=metrics["count_nll"])
    scheduler.step(metrics["count_nll"])
    state = checkpoint_state(archive=archive, config=config, epoch=epoch, model=model,
        optimizer=optimizer, scheduler=scheduler, policy=policy, metrics=metrics, hashes=hashes)
    if decision.improved_best:
        atomic_checkpoint(archive.scratch_path / "checkpoints/best.ckpt", state)
    atomic_checkpoint(archive.scratch_path / "checkpoints/last.ckpt", state)
    return decision


def resource_gate():
    free, total = torch.cuda.mem_get_info()
    memory_path = Path("/sys/fs/cgroup/memory.current")
    host_bytes = int(memory_path.read_text()) if memory_path.exists() else 0
    limit_path = Path("/sys/fs/cgroup/memory.max")
    raw_limit = limit_path.read_text().strip() if limit_path.exists() else "max"
    physical_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = min(int(raw_limit) if raw_limit != "max" else physical_ram, physical_ram)
    disk = shutil.disk_usage(current_paths().scratch_root).free
    if free < 2*GIB or disk < 5*GIB or host_bytes > .90*limit:
        raise RuntimeError(f"Resource gate failed: GPUfree={free}, diskfree={disk}, host={host_bytes}/{limit}")
    return {"gpu_free_bytes": free, "gpu_total_bytes": total, "disk_free_bytes": disk,
            "cgroup_bytes": host_bytes, "host_limit_bytes": limit,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def stage(core):
    return {name: torch.as_tensor(np.asarray(value).copy(), device="cuda") for name, value in {
        "expression": core.expression, "counts": core.counts, "covariates": core.covariates,
        "local_log_mean": core.local_log_mean, "local_positive_fraction": core.local_positive_fraction,
    }.items()}


def forward(model, tensors, mask, indices):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(tensors["expression"][indices], mask,
                     tensors["covariates"][indices], tensors["local_log_mean"][indices],
                     tensors["local_positive_fraction"][indices])


@torch.no_grad()
def precision_check(model, core):
    model.eval()
    tensors = stage(core)
    realization = make_mask(core, epoch=1)
    ix = slice(0, min(1024, core.n_cells))
    mask = torch.as_tensor(realization.mask[ix].copy(), device="cuda")
    fp32 = model(tensors["expression"][ix], mask, tensors["covariates"][ix],
                 tensors["local_log_mean"][ix], tensors["local_positive_fraction"][ix])
    amp = forward(model, tensors, mask, ix)
    losses = []
    for output in (fp32, amp):
        sums = masked_hurdle_negative_binomial_nll_sums(output.nonzero_logits,
            output.nb_mean, output.theta, tensors["counts"][ix], mask)
        losses.append(float(sums.joint_nll_sum/sums.masked_entries))
    delta = abs(losses[0]-losses[1])
    if delta >= .001:
        raise RuntimeError(f"FP32/BF16 pilot likelihood disagreement: {delta}")
    return {"fp32_nll": losses[0], "bf16_nll": losses[1], "absolute_delta": delta,
            "tolerance": .001, "passed": True}


@torch.no_grad()
def evaluate(model, cores, fixed_masks, batch_size, *, strata=False):
    from spatial_benchmark.local_gene_metrics import CountMetricAccumulator
    model.eval()
    aggregate = CountMetricAccumulator()
    core_metrics = {}
    mask_metrics = {name: CountMetricAccumulator() for name in ("1_250", "251_500", "501_750", "751_1000")} if strata else {}
    for core in cores:
        acc = CountMetricAccumulator()
        tensors = stage(core)
        full_mask = torch.as_tensor(fixed_masks[core.alias].mask.copy(), device="cuda")
        for start in range(0, core.n_cells, batch_size):
            ix = slice(start, min(start+batch_size, core.n_cells))
            mask = full_mask[ix]
            output = forward(model, tensors, mask, ix)
            target = tensors["counts"][ix]
            acc.update(output, target, mask)
            if strata:
                sizes = mask.sum(dim=1)
                for group, low, high in (("1_250", 1, 250), ("251_500", 251, 500),
                                         ("501_750", 501, 750), ("751_1000", 751, 1000)):
                    selected = mask & ((sizes >= low) & (sizes <= high)).unsqueeze(1)
                    if selected.any():
                        mask_metrics[group].update(output, target, selected)
        aggregate.merge(acc)
        core_metrics[core.alias] = acc.compute()
        del tensors, full_mask
    return aggregate.compute(), core_metrics, {name: acc.compute() for name, acc in mask_metrics.items()}


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train(archive, config, registry, *, expected_source_hashes):
    started = time.monotonic()
    if source_hashes() != expected_source_hashes:
        raise RuntimeError("Training source changed before initialization")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_float32_matmul_precision("high")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Runner requires one assigned BF16-capable GPU")
    resource_gate()
    paths = archive.paths
    print(json.dumps({"run_id": archive.run_id, "stage": "loading_data", "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    bundle = load_local_gene_data(paths=paths)
    archive.write_json("provenance/data_fingerprints.json", bundle.provenance)
    archive.write_json("provenance/split_fingerprint.json", {
        "split_fingerprint": config["dataset"]["split_fingerprint"], "training_cells": sum(c.n_cells for c in bundle.training),
        "validation_cells": sum(c.n_cells for c in bundle.validation), "test_cells": 0})
    model = LocalGeneResidualCountModel(1000, covariate_dim=22, hidden_dim=512, dropout=.1,
        use_local=config["model"]["use_local"],
        initial_nonzero_probability=torch.as_tensor(bundle.prevalence.copy()),
        initial_positive_conditional_mean=torch.as_tensor(bundle.positive_mean.copy())).cuda()
    param_count = sum(p.numel() for p in model.parameters())
    registry.transition_run(archive.run_id, "running", parameter_count=param_count)
    archive.write_json("diagnostics/precision_initial.json", precision_check(model, bundle.training[0]))
    validation_masks = {c.alias: make_mask(c, epoch=None) for c in bundle.validation}
    # The October audit retained the inherited NB/hurdle fixed-mask receipts.
    reference = paths.artifact_root / "runs/2026/10/r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980/diagnostics/train_fitted_gene_prevalence_validation.json"
    receipts = json.loads(reference.read_text())["validation_coverage_receipts"]
    for alias, mask in validation_masks.items():
        expected = receipts[alias]["rank_shards"][0]["mask"]
        if mask.mask_sha256 != expected["mask_sha256"] or mask.n_masked_entries != expected["masked_entry_count"]:
            raise RuntimeError(f"Validation mask differs from historical reference: {alias}")
    archive.write_json("diagnostics/validation_masks.json", {a: m.to_receipt() for a, m in validation_masks.items()})
    archive.write_json("provenance/mask_reference.json", {"path": str(reference), "sha256": sha256_file(reference)})
    baseline, baseline_core, _ = evaluate(model, bundle.validation, validation_masks, 1024)
    archive.write_json("diagnostics/train_fitted_constant_baseline.json", {"pooled": baseline, "per_core": baseline_core,
        "fit_scope": "training_only", "dispersion": "fixed_theta_1_not_fitted_MLE",
        "note": "Initial per-gene hurdle constants evaluated under same BF16 precision; head logits round before FP32 likelihood"})
    optimizer, scheduler = make_optimizer_scheduler(model, config)
    policy = LocalGeneEarlyStopping(**config["trainer"]["early_stopping"])
    epochs = config["trainer"]["epochs"]
    history, coverage, resource_rows = [], [], []
    best_epoch, best_metrics = 0, None
    checkpoint = archive.scratch_path / "checkpoints/best.ckpt"
    all_steps = 0
    for epoch in range(1, epochs+1):
        begin = time.monotonic()
        model.train()
        training_loss, entries, visits, steps = 0., 0, 0, 0
        gradient_max, gradient_sum, clipping_count = 0., 0., 0
        order = np.random.default_rng(2026100401 + epoch)
        epoch_coverage = {"epoch": epoch, "cores": {}}
        for core_index in order.permutation(len(bundle.training)):
            core = bundle.training[int(core_index)]
            realization = make_mask(core, epoch=epoch)
            tensors = stage(core)
            masks = torch.as_tensor(realization.mask.copy(), device="cuda")
            permutation = order.permutation(core.n_cells)
            if np.unique(permutation).size != core.n_cells:
                raise RuntimeError("Target coverage is not exhaustive")
            epoch_coverage["cores"][core.alias] = {"mask_sha256": realization.mask_sha256,
                "masked_entries": realization.n_masked_entries, "target_visits": core.n_cells,
                "order_sha256": hashlib.sha256(permutation.tobytes()).hexdigest()}
            for target_batch in np.array_split(permutation, math.ceil(core.n_cells/1024)):
                ix = torch.as_tensor(target_batch, device="cuda")
                mask = masks[ix]
                optimizer.zero_grad(set_to_none=True)
                output = forward(model, tensors, mask, ix)
                sums = masked_hurdle_negative_binomial_nll_sums(output.nonzero_logits, output.nb_mean,
                    output.theta, tensors["counts"][ix], mask)
                loss = sums.joint_nll_sum / sums.masked_entries
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                gradient_max = max(gradient_max, norm)
                gradient_sum += norm
                clipping_count += int(norm > 1.)
                optimizer.step()
                training_loss += float(sums.joint_nll_sum.detach())
                entries += sums.masked_entries
                visits += len(ix)
                steps += 1
            resource_rows.append({"epoch": epoch, **resource_gate()})
            del tensors, masks, realization
        if visits != 208696:
            raise RuntimeError(f"Incomplete epoch coverage: {visits}")
        all_steps += steps
        metrics, per_core, _ = evaluate(model, bundle.validation, validation_masks, 1024)
        lr = optimizer.param_groups[0]["lr"]
        if source_hashes() != expected_source_hashes:
            raise RuntimeError("Training source changed after admission")
        decision = commit_validation_epoch(archive=archive, config=config, epoch=epoch,
            model=model, optimizer=optimizer, scheduler=scheduler, policy=policy,
            metrics=metrics, hashes=expected_source_hashes)
        if decision.improved_best:
            best_epoch, best_metrics = epoch, dict(metrics)
        row = {"epoch": epoch, "train/count_nll": training_loss/entries,
               **{"val/"+k: v for k,v in metrics.items() if v is not None}, "learning_rate": lr,
               "learning_rate_after_scheduler": optimizer.param_groups[0]["lr"],
               "stopping_decision": asdict(decision), "stopping_state": policy.state_dict(),
               "steps": steps, "cumulative_steps": all_steps, "target_visits": visits,
               "seconds": time.monotonic()-begin, "best_epoch": best_epoch,
               "gradient_preclip_max": gradient_max, "gradient_preclip_mean": gradient_sum/steps,
               "gradient_clip_activations": clipping_count}
        history.append(row)
        coverage.append(epoch_coverage)
        for name in ("train/count_nll", "val/count_nll", "val/raw_count_mae", "val/raw_count_rmse"):
            archive.append_metric_event({"name": name, "value": row[name], "step": epoch})
            registry.record_metric(archive.run_id, name=name, value=row[name], step=epoch,
                                   split="validation" if name.startswith("val/") else "train")
        with (archive.scratch_path / "metrics/epoch_history.jsonl").open("a") as handle:
            handle.write(json.dumps({**row, "per_core": per_core})+"\n")
            handle.flush(); os.fsync(handle.fileno())
        with (archive.scratch_path / "diagnostics/coverage.jsonl").open("a") as handle:
            handle.write(json.dumps(epoch_coverage)+"\n")
            handle.flush(); os.fsync(handle.fileno())
        print(json.dumps({"run_id": archive.run_id, **row}), flush=True)
        if decision.should_stop:
            break
    stop = stopping_receipt(policy, config["metadata"]["profile"])
    final_epoch = policy.completed_epoch
    terminal_metrics = dict(metrics)
    last_path = archive.scratch_path / "checkpoints/last.ckpt"
    last = torch.load(last_path, map_location="cuda", weights_only=False)
    if (last["epoch"] != final_epoch or last["early_stopping"] != policy.state_dict()
            or last["source_hashes"] != expected_source_hashes):
        raise RuntimeError("Terminal checkpoint control/source state differs")
    model.load_state_dict(last["model"])
    terminal_replay, terminal_cores, _ = evaluate(model, bundle.validation, validation_masks, 1024)
    terminal_delta = replay_delta(terminal_replay, terminal_metrics)
    archive.write_json("metrics/terminal.json", {"val/"+k: v for k,v in terminal_metrics.items()})
    archive.write_json("results/terminal_per_core.json", terminal_cores)
    archive.write_json("diagnostics/terminal_checkpoint_replay.json", {"passed": True,
        "maximum_absolute_delta": terminal_delta, "final_epoch": final_epoch,
        "checkpoint_sha256": sha256_file(last_path)})
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if saved["epoch"] != best_epoch or saved["source_hashes"] != expected_source_hashes:
        raise RuntimeError("Selected checkpoint identity/source state differs")
    LocalGeneEarlyStopping.from_state_dict(saved["early_stopping"])
    model.load_state_dict(saved["model"])
    archive.write_json("diagnostics/precision_selected.json", precision_check(model, bundle.training[0]))
    final, final_cores, strata = evaluate(model, bundle.validation, validation_masks, 1024, strata=True)
    max_delta = replay_delta(final, best_metrics)
    archive.write_json("metrics/final.json", {"val/"+k: v for k,v in final.items()})
    archive.write_table("metrics/history", history)
    archive.write_json("results/per_core.json", final_cores)
    archive.write_json("results/mask_strata.json", strata)
    archive.write_json("diagnostics/coverage.json", coverage)
    archive.write_json("diagnostics/resources.json", resource_rows)
    archive.write_json("diagnostics/stopping.json", stop)
    recent = history[-4:]
    recent_metrics = {key.removeprefix("val/"): float(np.mean([row[key] for row in recent]))
        for key in recent[0] if key.startswith("val/")}
    archive.write_json("metrics/recent_epochs.json", {"epochs": [row["epoch"] for row in recent],
        "aggregation": "unweighted_epoch_mean_last_up_to_four_complete_epochs", "metrics": recent_metrics})
    archive.write_json("diagnostics/checkpoint_replay.json", {"passed": True, "maximum_absolute_delta": max_delta,
        "best_epoch": best_epoch, "checkpoint_sha256": sha256_file(checkpoint)})
    if source_hashes() != expected_source_hashes:
        raise RuntimeError("Training source changed before finalization")
    archive.write_json("diagnostics/preflight.json", {"passed": True, "source_hashes": source_hashes(),
        "frozen_contract_sha256": config["metadata"]["frozen_contract_sha256"], "coverage_verified": True,
        "checkpoint_replay_passed": True, "terminal_checkpoint_replay_passed": True,
        "epochs": final_epoch, "resources": resource_rows[-1], **stop})
    predictions = [{"run_id": archive.run_id, "sample_key": hashlib.sha256(alias.encode()).hexdigest(),
        "dataset_id": config["dataset"]["dataset_id"], "split": "validation",
        "y_true": m["observed_count_mean"], "y_pred": m["predicted_count_mean"],
        "sample_loss": m["count_nll"], "masked_entry_count": m["masked_entries"],
        "aggregation_unit": "core_masked_entry_mean"} for alias, m in final_cores.items()]
    archive.write_predictions("validation", predictions)
    terminal = float(np.mean([h["val/count_nll"] for h in history[-4:]]))
    summary = {"run_id": archive.run_id, "status": "completed", "best_epoch": best_epoch,
        "final_epoch": final_epoch, "primary_metric_name": PRIMARY, "primary_metric_value": final["count_nll"],
        "terminal_mean_count_nll": terminal, "parameter_count": param_count,
        "duration_seconds": time.monotonic()-started, "checkpoint_reload_verified": True,
        "peak_vram_gb": torch.cuda.max_memory_allocated()/GIB, "validation_donor_groups": 1,
        "optimization_steps": all_steps, "variant": config["model"]["variant"], "seed": config["seed"],
        "profile": config["metadata"]["profile"], "stop_reason": stop["stop_reason"],
        "early_stopping_triggered": stop["early_stopping_triggered"],
        "safety_cap_reached": stop["safety_cap_reached"], "stopping_state": policy.state_dict(),
        "selected_metrics": final, "terminal_metrics": terminal_metrics,
        "terminal_checkpoint_reload_verified": True,
        "last_learning_rate": optimizer.param_groups[0]["lr"],
        "recent_epoch_ids": [row["epoch"] for row in recent],
        "maximum_epochs": config["trainer"]["maximum_epochs"], "minimum_epochs": config["trainer"]["minimum_epochs"]}
    archive.write_summary(summary)
    archive.write_manifest({"run_id": archive.run_id, "campaign_id": CAMPAIGN, "status": "completed",
                            "primary_metric": PRIMARY, "best_epoch": best_epoch,
                            "final_epoch": final_epoch, "stop_reason": stop["stop_reason"]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--variant", choices=("local", "self"), default="local")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--smoke-run")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.attempt < 1 or (args.profile == "full") != bool(args.smoke_run):
        parser.error("full requires --smoke-run; smoke omits it; attempt must be positive")
    if not args.check_only and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the requested physical GPU")
    paths = current_paths()
    config, hashes = build_config(args), source_hashes()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    dependency_path = os.environ.get(DEPENDENCY_RECEIPT_ENV)
    dependency = None
    dependency_payload = None
    if dependency_path:
        dependency, dependency_payload = verify_dependency_receipt(dependency_path, paths, registry)
        config["launcher"]["dependency_receipt"] = dependency
    elif not args.check_only:
        raise RuntimeError(f"Normal admission requires {DEPENDENCY_RECEIPT_ENV}; the GeneMAE matrix must finish first")
    if args.smoke_run:
        config["launcher"]["pilot_receipt"] = verify_pilot(registry, paths, args.smoke_run, hashes, config)
    if args.check_only:
        print(json.dumps({"configuration": config, "admission": {"admitted": False,
            "cpu_only_check": True, "dependency_verified": dependency is not None,
            "gpu_probed": False}, "source_hashes": hashes}), flush=True)
        return 0
    admission = inspect_gpu(args.gpu)
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(paths.state_root / f"locks/bagm-worker-gpu-{args.gpu}.lock"):
        dependency, dependency_payload = verify_dependency_receipt(dependency_path, paths, registry)
        if dependency != config["launcher"]["dependency_receipt"]:
            raise RuntimeError("Dependency admission changed while acquiring GPU locks")
        inspect_gpu(args.gpu)
        environment, environment_text = _environment_snapshot()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True)
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        science = scientific_id(config)
        repro = repro_id(config, git_commit=commit, dirty_fingerprint=canonical_sha256({"source": hashes,"diff":diff}),
            dataset_fingerprint=config["dataset"]["dataset_fingerprint"], split_fingerprint=config["dataset"]["split_fingerprint"],
            preprocessing_version=config["dataset"]["preprocessing_version"], environment_fingerprint=canonical_sha256(environment))
        run_id = create_run_id(seed=args.seed, fold=0, attempt=args.attempt, scientific_id_value=science)
        registry.create_campaign(CAMPAIGN, name="Local gene residual validation early stopping",
            scientific_question="Does longer validation-controlled optimization improve count reconstruction and retain local-context gain?",
            config={"protocol": PROTOCOL}, status="pilot")
        registry.register_variant(science, campaign_id=CAMPAIGN, configuration=config)
        registry.create_run(run_id, campaign_id=CAMPAIGN, scientific_id=science, repro_id=repro, seed=args.seed,
            fold=0, attempt=args.attempt, configuration=config, git_commit=commit, dirty_status=True)
        classification = config["classification"]
        registry.register_run_semantics(run_id, lifecycle_stage=classification["lifecycle_stage"],
            study_axis=classification["study_axis"], source_batch=CAMPAIGN,
            seed_known=True, fold_known=True, attempt_known=True, retention_class=classification["retention_class"],
            category_key=f"{CAMPAIGN}.{args.profile}.{args.variant}", classification_confidence="high",
            timestamp_basis="execution", variant_label=args.variant, model_key="local_gene_residual",
            dataset_key="so2", masking_key="target_isolated_uniform", graph_key="local_k16_r75" if args.variant=="local" else "none")
        archive = RunArchive.create(run_id, paths=paths, resolved_config=config)
        try:
            archive.write_json("provenance/git.json", {"commit": commit, "dirty": True, "status": status})
            archive.write_text("provenance/uncommitted_changes.patch", diff)
            archive.write_text("provenance/environment.txt", environment_text)
            archive.write_json("provenance/hardware.json", {**_hardware_snapshot(), "admission": admission})
            archive.write_text("provenance/command.txt", json.dumps(sys.argv)+"\nworking_directory="+str(ROOT)+"\n")
            archive.write_json("provenance/source_hashes.json", hashes)
            archive.write_json("provenance/dependency_receipt.json", dependency_payload)
            archive.write_json("provenance/dependency_admission.json", dependency)
            for name in SOURCE_FILES:
                archive.copy_file(ROOT/name, "provenance/source_snapshot/"+name)
            archive.copy_file(ROOT/"experiments/campaigns"/CAMPAIGN/"frozen_task_contract.md", "provenance/frozen_task_contract.md")
            stdout, stderr = archive.prepare_log_files()
            registry.transition_run(run_id, "running", start_time=utc(), artifact_path=archive.scratch_path)
            print(json.dumps({"run_id": run_id, "status": "running", "scratch": str(archive.scratch_path)}), flush=True)
            with stdout.open("a", buffering=1) as out, stderr.open("a", buffering=1) as err, redirect_stdout(out), redirect_stderr(err):
                summary = train(archive, config, registry, expected_source_hashes=hashes)
            registry.transition_run(run_id, "finalizing")
            published = archive.publish_success_pending()
            for role in ("best", "last"):
                ckpt = published / f"checkpoints/{role}.ckpt"
                artifact_id = registry.record_artifact(run_id, kind="checkpoint", path=ckpt,
                    sha256=sha256_file(ckpt), size_bytes=ckpt.stat().st_size)
                registry.register_checkpoint_metadata(artifact_id, run_id=run_id, role=role,
                    retention_class=classification["retention_class"], verification_status="verified",
                    best_epoch=summary["best_epoch"] if role == "best" else summary["final_epoch"],
                    monitored_metric=PRIMARY, monitored_mode="min",
                    monitored_value=summary["primary_metric_value"] if role == "best" else summary["terminal_metrics"]["count_nll"],
                    metadata={"reload_verified": True, "optimizer_scheduler_policy_rng_retained": True,
                              "stop_reason": summary["stop_reason"]})
            registry.record_artifact(run_id, kind="run_manifest", path=published/"manifest.yaml",
                sha256=sha256_file(published/"manifest.yaml"), size_bytes=(published/"manifest.yaml").stat().st_size)
            registry.transition_run(run_id, "completed", end_time=utc(), artifact_path=published,
                duration_seconds=summary["duration_seconds"], primary_metric_name=PRIMARY,
                primary_metric_value=summary["primary_metric_value"], peak_vram_gb=summary["peak_vram_gb"])
            archive.mark_success()
            verified = verify_run_bundle(published)
            print(json.dumps({"run_id": run_id, "status": "completed", "summary": summary, "verification": verified}), flush=True)
            return 0
        except BaseException as error:
            trace = traceback.format_exc()
            if archive.scratch_path.exists():
                failed = archive.finalize_failure(error, traceback_text=trace)
            else:
                failed = archive.artifact_path
                if not (failed / "_SUCCESS").exists() and not (failed / "_FAILED").exists():
                    archive.mark_published_failure()
            registry.transition_run(run_id, "failed", end_time=utc(), artifact_path=failed, failure_category="local_gene_early_stopping_execution")
            print(trace, file=sys.stderr, flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())

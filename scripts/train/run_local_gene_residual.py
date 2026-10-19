#!/usr/bin/env python3
"""Registered, isolated Local Gene Residual count experiment.

Run under supervisor with BAGM roots set as in the campaign README. This additive
runner reuses registry/archive/masking/likelihood contracts without altering any
historical model or training route.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
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
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, _environment_snapshot, _hardware_snapshot
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle

CAMPAIGN = "cmp_20261004_local_gene_residual"
PRIMARY = "val/count_nll"
GIB = 1024**3
SOURCE_FILES = (
    "scripts/train/run_local_gene_residual.py",
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
    "ops/supervisor/bagm-local-gene-residual.sh",
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
        "trainer": {"epochs": 2 if args.profile == "smoke" else 32, "batch_size": 1024,
                    "learning_rate": .001, "minimum_learning_rate": .0001, "weight_decay": .00001,
                    "gradient_clip": 1., "precision": "bfloat16_autocast_fp32_likelihood",
                    "primary_checkpoint_role": "best", "restore_best": True,
                    "ordering_seed": 2026100401, "scheduler": "fixed_cosine"},
        "evaluation": {"protocol": "local_gene_residual_counts_v1", "primary_metric": PRIMARY,
                       "primary_mode": "min", "canonical_prediction_split": "validation",
                       "artifact_contract": "predictive"},
        "classification": {"lifecycle_stage": "diagnostic" if args.profile == "smoke" else "validation_confirmation",
                           "study_axis": "simple_count_reconstruction", "variant_label": args.variant,
                           "retention_class": "diagnostic" if args.profile == "smoke" else "validation_confirmation"},
        "metadata": {"profile": args.profile, "frozen_contract_sha256": sha256_file(contract)},
        "launcher": {"requested_gpu": args.gpu, "smoke_run": args.smoke_run},
    }


def verify_pilot(registry, paths, run_id, hashes, config):
    record = registry.get_run(run_id)
    if not record or record["status"] != "completed":
        raise RuntimeError("Full experiment requires a completed registered pilot")
    root = Path(record["artifact_path"])
    verification = verify_run_bundle(root)
    if verification.get("ok") is False or verification.get("valid") is False:
        raise RuntimeError(f"Pilot bundle verification failed: {verification}")
    receipt = json.loads((root / "diagnostics/preflight.json").read_text())
    if not receipt["passed"] or receipt["source_hashes"] != hashes:
        raise RuntimeError("Pilot correctness gates/source identity differ")
    if receipt["frozen_contract_sha256"] != config["metadata"]["frozen_contract_sha256"]:
        raise RuntimeError("Frozen contract changed since pilot")
    return {"run_id": run_id, "preflight_sha256": sha256_file(root / "diagnostics/preflight.json")}


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


def train(archive, config, registry):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_float32_matmul_precision("high")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Runner requires one assigned BF16-capable GPU")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    epochs = config["trainer"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.0001)
    history, coverage, resource_rows = [], [], []
    best, best_epoch, best_metrics = float("inf"), 0, None
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
        scheduler.step()
        state = {"schema": "local_gene_residual_checkpoint_v1", "run_id": archive.run_id, "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": config, "metrics": metrics, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "source_hashes": source_hashes()}
        if metrics["count_nll"] < best:
            best, best_epoch, best_metrics = metrics["count_nll"], epoch, metrics
            atomic_checkpoint(checkpoint, state)
        atomic_checkpoint(archive.scratch_path / "checkpoints/latest.ckpt", state)
        row = {"epoch": epoch, "train/count_nll": training_loss/entries,
               **{"val/"+k: v for k,v in metrics.items() if v is not None}, "learning_rate": lr,
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
        print(json.dumps({"run_id": archive.run_id, **row}), flush=True)
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    archive.write_json("diagnostics/precision_selected.json", precision_check(model, bundle.training[0]))
    final, final_cores, strata = evaluate(model, bundle.validation, validation_masks, 1024, strata=True)
    max_delta = max(abs(final[k]-best_metrics[k]) for k in final if final[k] is not None and best_metrics[k] is not None)
    if max_delta > 1e-5:
        raise RuntimeError(f"Selected checkpoint replay mismatch: {max_delta}")
    archive.write_json("metrics/final.json", {"val/"+k: v for k,v in final.items()})
    archive.write_table("metrics/history", history)
    archive.write_json("results/per_core.json", final_cores)
    archive.write_json("results/mask_strata.json", strata)
    archive.write_json("diagnostics/coverage.json", coverage)
    archive.write_json("diagnostics/resources.json", resource_rows)
    archive.write_json("diagnostics/checkpoint_replay.json", {"passed": True, "maximum_absolute_delta": max_delta,
        "best_epoch": best_epoch, "checkpoint_sha256": sha256_file(checkpoint)})
    archive.write_json("diagnostics/preflight.json", {"passed": True, "source_hashes": source_hashes(),
        "frozen_contract_sha256": config["metadata"]["frozen_contract_sha256"], "coverage_verified": True,
        "checkpoint_replay_passed": True, "epochs": epochs, "resources": resource_rows[-1]})
    predictions = [{"run_id": archive.run_id, "sample_key": hashlib.sha256(alias.encode()).hexdigest(),
        "dataset_id": config["dataset"]["dataset_id"], "split": "validation",
        "y_true": m["observed_count_mean"], "y_pred": m["predicted_count_mean"],
        "sample_loss": m["count_nll"], "masked_entry_count": m["masked_entries"],
        "aggregation_unit": "core_masked_entry_mean"} for alias, m in final_cores.items()]
    archive.write_predictions("validation", predictions)
    terminal = float(np.mean([h["val/count_nll"] for h in history[-4:]]))
    summary = {"run_id": archive.run_id, "status": "completed", "best_epoch": best_epoch,
        "final_epoch": epochs, "primary_metric_name": PRIMARY, "primary_metric_value": final["count_nll"],
        "terminal_mean_count_nll": terminal, "parameter_count": param_count,
        "duration_seconds": time.monotonic()-started, "checkpoint_reload_verified": True,
        "peak_vram_gb": torch.cuda.max_memory_allocated()/GIB, "validation_donor_groups": 1,
        "optimization_steps": all_steps, "variant": config["model"]["variant"], "seed": config["seed"]}
    archive.write_summary(summary)
    archive.write_manifest({"run_id": archive.run_id, "campaign_id": CAMPAIGN, "status": "completed",
                            "primary_metric": PRIMARY, "best_epoch": best_epoch})
    (archive.scratch_path / "checkpoints/latest.ckpt").unlink()
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the requested physical GPU")
    paths = current_paths()
    config, hashes = build_config(args), source_hashes()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    if args.smoke_run:
        config["launcher"]["pilot_receipt"] = verify_pilot(registry, paths, args.smoke_run, hashes, config)
    admission = inspect_gpu(args.gpu)
    if args.check_only:
        print(json.dumps({"configuration": config, "admission": admission, "source_hashes": hashes}), flush=True)
        return 0
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(paths.state_root / f"locks/bagm-worker-gpu-{args.gpu}.lock"):
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
        registry.create_campaign(CAMPAIGN, name="Local gene residual count reconstruction",
            scientific_question="Can a shallow cell predictor with local gene residuals improve SO2 count reconstruction?",
            config={"protocol": "local_gene_residual_counts_v1"}, status="pilot")
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
        archive.write_json("provenance/git.json", {"commit": commit, "dirty": True, "status": status})
        archive.write_text("provenance/uncommitted_changes.patch", diff)
        archive.write_text("provenance/environment.txt", environment_text)
        archive.write_json("provenance/hardware.json", {**_hardware_snapshot(), "admission": admission})
        archive.write_text("provenance/command.txt", json.dumps(sys.argv)+"\nworking_directory="+str(ROOT)+"\n")
        archive.write_json("provenance/source_hashes.json", hashes)
        for name in SOURCE_FILES:
            archive.copy_file(ROOT/name, "provenance/source_snapshot/"+name)
        archive.copy_file(ROOT/"experiments/campaigns"/CAMPAIGN/"frozen_task_contract.md", "provenance/frozen_task_contract.md")
        stdout, stderr = archive.prepare_log_files()
        registry.transition_run(run_id, "running", start_time=utc(), artifact_path=archive.scratch_path)
        print(json.dumps({"run_id": run_id, "status": "running", "scratch": str(archive.scratch_path)}), flush=True)
        try:
            with stdout.open("a", buffering=1) as out, stderr.open("a", buffering=1) as err, redirect_stdout(out), redirect_stderr(err):
                summary = train(archive, config, registry)
            registry.transition_run(run_id, "finalizing")
            published = archive.publish_success_pending()
            ckpt = published / "checkpoints/best.ckpt"
            artifact_id = registry.record_artifact(run_id, kind="checkpoint", path=ckpt,
                sha256=sha256_file(ckpt), size_bytes=ckpt.stat().st_size)
            registry.register_checkpoint_metadata(artifact_id, run_id=run_id, role="best",
                retention_class=classification["retention_class"], verification_status="verified",
                best_epoch=summary["best_epoch"], monitored_metric=PRIMARY, monitored_mode="min",
                monitored_value=summary["primary_metric_value"], metadata={"reload_verified": True})
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
            registry.transition_run(run_id, "failed", end_time=utc(), artifact_path=failed, failure_category="local_gene_execution")
            print(trace, file=sys.stderr, flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())

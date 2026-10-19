#!/usr/bin/env python3
"""Run the separately registered SO2 BCE optimization audit on one GPU.

Contract: experiments/campaigns/cmp_20261001_so2_bce_optimization_audit/README.md.
This additive entry point preserves the existing models, numerical training
helpers, deterministic masks, validation and atomic checkpoint lifecycle.
Explicit process-local hooks select paired cores or target windows and apply
the new LR/dropout/weight-decay/seed contract and five-GiB disk floor. Importing
this module does not install a training protocol or initialize CUDA.

The exact gene-prevalence reference uses training-fitted probabilities and
fixed validation masks; no validation targets are used to fit it. Optional
selected-best training diagnostics use fixed epoch-one training masks in eval
mode. Raw train/validation gaps remain affected by different class prevalence.

CLI is the registered worker convention: --config <resolved YAML>
--run-scratch <active run> [--resume-checkpoint <that run's latest.ckpt>].
Launch with torchrun --nproc-per-node=1 and one explicit CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

import numpy as np
import torch

from scripts.train import run_so2_target_minibatch_binary_expression_bce as minibatch
from spatial_benchmark.so2_bce_optimization_audit import validate_configuration

base, bce = minibatch.base, minibatch.bce
_DEFAULT_PAIRED_EPOCH = base._train_epoch
_ORIGINAL_RESOURCE_SNAPSHOT = minibatch._resource_snapshot
CAMPAIGN_ID = "cmp_20261001_so2_bce_optimization_audit"
PROTOCOL = "donor_grouped_so2_bce_optimization_audit_v1"
CHECKPOINT_SCHEMA = "so2_bce_optimization_audit_checkpoint_v1"
PREFLIGHT_SCHEMA = "so2_bce_optimization_audit_preflight_v1"
PREFLIGHT_PATH = "diagnostics/optimization_audit_preflight.json"
BASELINE_PATH = "diagnostics/train_fitted_gene_prevalence_validation.json"
TRAIN_EVALUATION_PATH = "diagnostics/selected_best_fixed_training_evaluation.json"
ENDPOINT_PATH = "results/optimization_audit_endpoint.json"
RESOURCE_SAMPLES_PATH = "diagnostics/optimization_audit_resource_samples.jsonl"
PROVENANCE_PATH = "provenance/so2_bce_optimization_audit.json"
_EXTRA_COLUMNS = (
    "optimizer_updates", "cumulative_optimizer_updates", "minimum_target_batch_size",
    "maximum_target_batch_size", "peak_reserved_vram_gib", "minimum_sampled_physical_free_gib",
)
_CONFIG: dict[str, Any] = {}
_BASELINE: dict[str, Any] = {}
_BUNDLE: Any = None
_MODEL: Any = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise base.SO2NBTrainingError(message)


def prevalence_reference_metrics(probability: np.ndarray, positive_counts: np.ndarray,
                                 masked_counts: np.ndarray) -> dict[str, Any]:
    """Exact entry-weighted metrics for one constant probability per gene.

    Counts are sufficient statistics; equal-score genes are combined before
    ROC integration and descending-threshold average precision. This is not
    an unweighted gene-level metric or a histogram approximation.
    """
    p = np.asarray(probability, dtype=np.float64)
    positive = np.asarray(positive_counts)
    total = np.asarray(masked_counts)
    require(p.ndim == 1 and p.shape == positive.shape == total.shape and p.size > 0,
            "Gene probability/count shapes differ")
    require(np.issubdtype(positive.dtype, np.integer) and np.issubdtype(total.dtype, np.integer),
            "Baseline support counts must be integers")
    require(bool(np.isfinite(p).all()) and bool(((p > 0) & (p < 1)).all()), "Invalid baseline probabilities")
    require(bool(((positive >= 0) & (total >= positive)).all()) and int(total.sum()) > 0,
            "Invalid baseline class counts")
    positive = positive.astype(np.int64, copy=False)
    total = total.astype(np.int64, copy=False)
    negative = total - positive
    n = int(total.sum()); positives = int(positive.sum()); negatives = n - positives
    require(0 < positives < n, "Baseline ROC/AP requires both classes")
    predicted = p >= 0.5
    confusion = bce._confusion_metrics(tp=int(positive[predicted].sum()),
        tn=int(negative[~predicted].sum()), fp=int(negative[predicted].sum()),
        fn=int(positive[~predicted].sum()))
    scores = np.unique(p)[::-1]
    true_positive = false_positive = 0
    previous_tpr = previous_fpr = auroc = average_precision = 0.0
    for score in scores:
        group = p == score
        true_positive += int(positive[group].sum())
        false_positive += int(negative[group].sum())
        tpr, fpr = true_positive / positives, false_positive / negatives
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        if true_positive + false_positive:
            average_precision += (tpr - previous_tpr) * true_positive / (true_positive + false_positive)
        previous_tpr, previous_fpr = tpr, fpr
    nll = float((-positive * np.log(p) - negative * np.log1p(-p)).sum())
    brier = float((positive * (1-p)**2 + negative * p**2).sum()) / n
    return {"masked_entries": n, "positive_entries": positives,
            "observed_positive_rate": positives/n,
            "pooled_masked_binary_cross_entropy": nll/n, "binary_cross_entropy_sum": nll,
            "detection_brier_score": brier, "detection_threshold": 0.5,
            "predicted_positive_probability_mean": float((total*p).sum())/n,
            "detection_auroc_exact": auroc, "detection_average_precision_exact": average_precision,
            **confusion}


def terminal_validation_endpoint(epoch_rows: Any, endpoint_epochs: Any) -> dict[str, Any]:
    """Prespecified endpoint mean; never substitute the best checkpoint metric."""
    required = tuple(int(epoch) for epoch in endpoint_epochs)
    require(bool(required) and len(set(required)) == len(required), "Endpoint epochs must be nonempty and unique")
    by_epoch = {}
    for row in epoch_rows:
        epoch = int(row["global_epoch"])
        require(epoch not in by_epoch, "Duplicate validation epoch in endpoint history")
        by_epoch[epoch] = row
    require(all(epoch in by_epoch for epoch in required), "Missing prespecified endpoint epoch")
    values = [float(by_epoch[epoch]["validation_pooled_masked_binary_cross_entropy"]) for epoch in required]
    require(all(math.isfinite(value) and value >= 0 for value in values), "Invalid endpoint BCE")
    return {"endpoint_epochs": list(required), "validation_bce_values": values,
            "mean_validation_bce": math.fsum(values)/len(values),
            "best_checkpoint_metric_used_for_endpoint_selection": False}


def _gene_counts(batch: Any, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros(batch.n_genes, dtype=np.int64)
    total = np.zeros(batch.n_genes, dtype=np.int64)
    for start in range(0, batch.n_nodes, 2048):
        stop = min(start+2048, batch.n_nodes)
        local_mask = mask[start:stop]
        counts = batch.raw_count_target[start:stop].detach().cpu().numpy()
        positive += ((counts > 0) & local_mask).sum(axis=0, dtype=np.int64)
        total += local_mask.sum(axis=0, dtype=np.int64)
    return positive, total


def _prevalence_reference(bundle: Any, model: Any) -> dict[str, Any]:
    p32 = model.detection_initial_nonzero_probability.detach().cpu().numpy().copy()
    require(not bool(torch.count_nonzero(model.detection_head.weight)), "Initial head is not the frozen prevalence predictor")
    positives = np.zeros(len(p32), dtype=np.int64)
    totals = np.zeros(len(p32), dtype=np.int64)
    per_core = {}; receipts = {}
    for batch in bundle.validation_batches:
        partition, realization = base._partition_and_mask(batch, rank=0, global_epoch=None)
        positive, total = _gene_counts(batch, realization.mask)
        positives += positive; totals += total
        per_core[batch.alias] = prevalence_reference_metrics(p32, positive, total)
        receipts[batch.alias] = base._coverage_receipt(batch=batch, role="validation", global_epoch=None,
            gathered=[base._local_receipt_record(rank=0, batch=batch, partition=partition, realization=realization)])
    return {"schema": "so2_train_fitted_gene_prevalence_validation_v1",
            "fit_scope": "training_cores_only", "validation_fitted": False,
            "formula": "(training_positive_count+0.5)/(training_cells+1), clipped to [1e-4,1-1e-4], float32",
            "probability_values_sha256": hashlib.sha256(p32.tobytes(order="C")).hexdigest(),
            "training_initialization": dict(bce._DETECTION_INITIALIZATION_RECEIPT),
            "validation_target_cells": sum(b.n_nodes for b in bundle.validation_batches),
            "metrics": prevalence_reference_metrics(p32, positives, totals), "per_core": per_core,
            "validation_coverage_receipts": receipts,
            "validation_coverage_receipts_sha256": base._canonical_sha256(receipts),
            "metric_arithmetic": "float64_count_weighted_exact_no_histogram_approximation",
            "untrained_model_probability_note": "probabilities equal initialized gene prior up to float32 logit/sigmoid roundoff"}


def _disk_headroom(path: Path) -> float:
    free_gib = shutil.disk_usage(path).free / minibatch.GIB
    require(math.isfinite(free_gib) and free_gib >= 5.0,
            f"Free disk is {free_gib:.3f} GiB; this audit requires at least 5 GiB")
    return free_gib


def _resource_snapshot(device: torch.device) -> Mapping[str, Any]:
    resources = dict(_ORIGINAL_RESOURCE_SNAPSHOT(device))
    require(minibatch._SCRATCH is not None, "Resource check lacks active scratch")
    resources["free_disk_gib"] = _disk_headroom(minibatch._SCRATCH)
    return resources


def _load_preflight(config: Mapping[str, Any], paths: Any, bundle: Any, model: Any) -> Mapping[str, Any]:
    global _BASELINE, _BUNDLE, _MODEL
    _BUNDLE, _MODEL = bundle, model
    scratch = Path(os.environ["BAGM_RUN_SCRATCH"]).resolve(strict=True)
    minibatch._SCRATCH = scratch
    _disk_headroom(scratch)
    minibatch._MIN_CUDA_FREE, minibatch._DEVICE_TOTAL = torch.cuda.mem_get_info()
    require(minibatch._MIN_CUDA_FREE >= 2*minibatch.GIB, "Initial GPU headroom is insufficient")
    scope = set(bce.PREFLIGHT_CODE_RELATIVE_PATHS) | {
        Path("scripts/train/run_so2_target_minibatch_binary_expression_bce.py"),
        Path("scripts/train/run_so2_bce_optimization_audit.py"),
        Path("src/spatial_benchmark/target_minibatch.py"),
        Path("src/spatial_benchmark/so2_binary_target_minibatch.py"),
        Path("src/spatial_benchmark/so2_bce_optimization_audit.py"),
        Path("scripts/diagnostics/so2_hurdle_gpu_trial_worker.py"),
    }
    hashes = {str(p): minibatch.sha256_file(paths.project_root/p) for p in sorted(scope)}
    smoke = config["metadata"].get("smoke_preflight")
    if config["metadata"]["execution_profile"] != "smoke":
        require(isinstance(smoke, Mapping), "Audit screen/confirmation requires a registered compatible smoke")
        receipt_path = Path(smoke["receipt_path"])
        require(minibatch.sha256_file(receipt_path) == smoke["receipt_sha256"], "Smoke checksum changed")
        receipt = json.loads(receipt_path.read_text())
        for key, expected in {
            "schema": PREFLIGHT_SCHEMA, "passed": True,
            "source_configuration_sha256": base._canonical_sha256(config["source_configuration"]),
            "frozen_task_contract_sha256": config["metadata"]["frozen_task_contract_sha256"],
            "optimizer_batch_mode": config["trainer"]["optimizer_batch_mode"],
            "code_hashes": hashes,
        }.items():
            require(receipt.get(key) == expected, f"Smoke incompatibility: {key}")
    _BASELINE = _prevalence_reference(bundle, model)
    initial = {"schema": "so2_bce_optimization_audit_initialization_v1", "protocol": PROTOCOL,
        "configuration_sha256": base._canonical_sha256(config), "code_hashes": hashes,
        "frozen_task_contract_sha256": config["metadata"]["frozen_task_contract_sha256"],
        "source_configuration_sha256": base._canonical_sha256(config["source_configuration"]),
        "model_seed": config["seed"], "world_size": 1, "smoke_preflight": smoke,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "prevalence_reference_sha256": base._canonical_sha256(_BASELINE),
        "minimum_free_disk_gib": 5.0, "original_four_rank_preflight_claimed": False}
    initial["receipt_content_sha256"] = base._canonical_sha256(initial)
    minibatch._INITIALIZATION_RECEIPT = initial
    for relative, payload in (("diagnostics/optimization_audit_initialization.json", initial),
                              ("diagnostics/initial_train_fitted_prevalence_reference.json", _BASELINE)):
        path = scratch/relative
        if path.exists():
            require(json.loads(path.read_text()) == payload, "Initialization/code/reference differs from restart receipt")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False); handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
    minibatch.append_json(scratch/"diagnostics/optimization_audit_environment.jsonl", {
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "visible_physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "world_size": 1,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "initial_cuda_free_bytes": minibatch._MIN_CUDA_FREE, "total_device_bytes": minibatch._DEVICE_TOTAL,
        "initial_host": minibatch.host_snapshot(), "time": time.time(), "model_seed": config["seed"]})
    minibatch._SAMPLER = minibatch.ResourceSampler(scratch/RESOURCE_SAMPLES_PATH,
                                                  os.environ["CUDA_VISIBLE_DEVICES"])
    minibatch._SAMPLER.start()
    minibatch._progress("train_fitted_prevalence_reference", **_BASELINE["metrics"])
    return initial


def _observed_pair_update(**kwargs: Any) -> Any:
    result = bce._paired_update(**kwargs)
    resources = minibatch._resource_snapshot(kwargs["device"])
    cumulative = kwargs["global_epoch"]*6 + kwargs["update_index"] + 1
    if cumulative > 1:
        for block in kwargs["model"].blocks:
            require(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in block.parameters()),
                    "Paired-core graph block has zero gradients after initialization")
    minibatch._progress("paired_update_completed", epoch=kwargs["global_epoch"]+1,
        update=kwargs["update_index"]+1, cumulative_optimizer_update=cumulative,
        core_aliases=kwargs["pair"], gradient_norm=result[2],
        peak_allocated_bytes=resources["peak_allocated_bytes"],
        peak_reserved_bytes=resources["peak_reserved_bytes"], headroom_bytes=resources["headroom_bytes"])
    return result


def _train_epoch(**kwargs: Any) -> Any:
    if _CONFIG["trainer"]["optimizer_batch_mode"] == "target_window":
        return minibatch._train_epoch(**kwargs)
    require(base.OPTIMIZER_UPDATES_PER_EPOCH == 6, "Paired reference needs six updates")
    require(_DEFAULT_PAIRED_EPOCH.__module__ == base.__name__, "Historical default epoch hook was not captured")
    epoch = kwargs["global_epoch"]+1
    minibatch._CURRENT_EPOCH = epoch
    started = time.monotonic()
    minibatch._progress("epoch_training_started", epoch=epoch, optimizer_updates_expected=6,
                        optimizer_batch_mode="core_pair")
    ordered, statistics, receipts = _DEFAULT_PAIRED_EPOCH(**kwargs)
    batches = kwargs["bundle"].batches_by_alias
    sizes = [batches[ordered[i]].n_nodes+batches[ordered[i+1]].n_nodes for i in range(0,12,2)]
    minibatch._EPOCH_RESOURCES[epoch] = {
        **minibatch._resource_snapshot(kwargs["device"]), "epoch": epoch,
        "optimizer_updates": 6, "cumulative_optimizer_updates": epoch*6,
        "minimum_target_batch_size": min(sizes), "maximum_target_batch_size": max(sizes),
        "training_duration_seconds": time.monotonic()-started, "coverage_verified": True,
        "optimizer_batch_mode": "core_pair"}
    return ordered, statistics, receipts


def _make_epoch_row(**kwargs: Any) -> Mapping[str, Any]:
    row = dict(bce._make_epoch_row(**kwargs))
    epoch = int(kwargs["completed_epoch"])
    resources = minibatch._EPOCH_RESOURCES[epoch]
    resources["optimizer_batch_mode"] = _CONFIG["trainer"]["optimizer_batch_mode"]
    row.update(schema="so2_bce_optimization_audit_epoch_metrics_v1",
        **{k: resources[k] for k in _EXTRA_COLUMNS[:4]},
        peak_reserved_vram_gib=resources["peak_reserved_bytes"]/minibatch.GIB,
        minimum_sampled_physical_free_gib=resources["minimum_sampled_physical_free_bytes"]/minibatch.GIB)
    if _CONFIG["trainer"]["optimizer_batch_mode"] == "target_window":
        row["target_minibatch_nominal_size"] = resources["target_minibatch_nominal_size"]
    minibatch.append_json(minibatch._SCRATCH/"metrics/target_minibatch_resources.jsonl", resources)
    return row


def _fixed_training_evaluation() -> Mapping[str, Any]:
    require(_BUNDLE is not None and _MODEL is not None, "Final training diagnostic lacks model/data")
    model, bundle = _MODEL, _BUNDLE
    was_training = model.training
    device = next(model.parameters()).device
    p = model.detection_initial_nonzero_probability.detach().cpu().numpy()
    positives = np.zeros(len(p), dtype=np.int64); totals = np.zeros(len(p), dtype=np.int64)
    per_core = {}; receipts = {}; sums_by_core = []
    started = time.monotonic()
    model.eval()
    try:
        with torch.no_grad():
            for batch in bundle.training_batches:
                partition, realization = base._partition_and_mask(batch, rank=0, global_epoch=1)
                positive, total = _gene_counts(batch, realization.mask)
                positives += positive; totals += total
                base._clear_graph_layout_caches(model)
                clean, raw, covariates = base._stage_core(batch, device=device)
                target, mask = base._make_target_tensors(clean_input=clean, realization=realization,
                    target_start=partition.start, target_stop=partition.stop, device=device)
                output = bce._forward_target_shard(training_model=model, batch=batch, clean_input=clean,
                    covariates=covariates, target_input=target, mask=mask, target_start=partition.start,
                    target_stop=partition.stop, trainer=_CONFIG["trainer"])
                sums = bce._validation_sum_tensor(output=output, raw_target=raw, mask=mask, target_cells=batch.n_nodes).cpu()
                metrics = bce._metrics_from_sums(sums)
                per_core[batch.alias] = {k:v for k,v in metrics.items() if "histogram" not in k or isinstance(v,(float,int))}
                sums_by_core.append(sums)
                receipts[batch.alias] = realization.to_receipt()
                del output, clean, raw, covariates, target, mask, realization
                base._clear_graph_layout_caches(model); torch.cuda.empty_cache()
                minibatch._resource_snapshot(device)
        pooled = bce._metrics_from_sums(torch.stack(sums_by_core).sum(dim=0))
        # Production cohort size is already checked by run_distributed. Bind
        # this diagnostic to the supplied bundle rather than a second literal,
        # allowing its complete model/metric path to be tested on tiny cores.
        require(int(pooled["target_cells"]) == sum(batch.n_nodes for batch in bundle.training_batches),
                "Fixed training diagnostic has incomplete coverage")
        reference = prevalence_reference_metrics(p, positives, totals)
        return {"schema": "so2_selected_best_fixed_training_evaluation_v1",
            "checkpoint_role": "selected_validation_best", "model_eval_mode": True,
            "checkpoint_sha256": minibatch.sha256_file(minibatch._SCRATCH/"checkpoints/best.ckpt"),
            "mask_contract": "fixed_original_training_epoch_one_masks", "used_for_model_selection": False,
            "metrics": dict(pooled), "per_core": per_core, "mask_receipts": receipts,
            "train_fitted_gene_prevalence_reference": reference,
            "bce_reduction_from_train_fitted_gene_prevalence": reference["pooled_masked_binary_cross_entropy"]-float(pooled["pooled_masked_binary_cross_entropy"]),
            "raw_cross_split_bce_gap_is_not_a_controlled_generalization_estimate": True,
            "duration_seconds": time.monotonic()-started}
    finally:
        model.train(was_training); base._clear_graph_layout_caches(model)


def _write_final_artifacts(archive: Any, validation: Any) -> None:
    bce._write_additional_final_artifacts(archive, validation)
    require(base._canonical_sha256(validation.coverage_receipts) == _BASELINE["validation_coverage_receipts_sha256"],
            "Final model and prior reference masks differ")
    archive.write_json(BASELINE_PATH, _BASELINE)
    with (minibatch._SCRATCH/"results/epoch_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    endpoint = terminal_validation_endpoint(rows, _CONFIG["experiment"]["screening_endpoint_epochs"])
    archive.write_json(ENDPOINT_PATH, {"schema": "so2_bce_optimization_audit_endpoint_v1", **endpoint,
        "profile": _CONFIG["metadata"]["execution_profile"],
        "selected_best_checkpoint_bce": validation.pooled_binary_cross_entropy,
        "optimizer_updates_at_final_epoch": int(rows[-1]["cumulative_optimizer_updates"]),
        "training_target_visits": sum(int(row["train_target_cells"]) for row in rows)})
    if _CONFIG["metadata"].get("evaluate_fixed_training_at_best", False):
        archive.write_json(TRAIN_EVALUATION_PATH, dict(_fixed_training_evaluation()))
    if minibatch._SAMPLER is not None:
        minibatch._SAMPLER.stop()
    resources = minibatch._resource_snapshot(torch.device("cuda:0"))
    by_epoch = {}
    for line in (minibatch._SCRATCH/"metrics/target_minibatch_resources.jsonl").read_text().splitlines():
        row = json.loads(line); by_epoch[int(row["epoch"])] = row
    epochs = [by_epoch[k] for k in sorted(by_epoch)]
    expected_epochs = int(_CONFIG["trainer"]["max_epochs"])
    require(sorted(by_epoch) == list(range(1,expected_epochs+1)), "Audit did not finish its fixed exposure budget")
    require(all(r["optimizer_updates"] == base.OPTIMIZER_UPDATES_PER_EPOCH for r in epochs), "Audit update count changed")
    samples = [json.loads(line) for line in
        (minibatch._SCRATCH/RESOURCE_SAMPLES_PATH).read_text().splitlines()]
    require(any("physical_memory_free_bytes" in row for row in samples), "No physical GPU memory samples")
    peak_cgroup = max(minibatch._MAX_CGROUP_MEMORY, *(int(r["cgroup"]["memory.current"]) for r in samples),
                      *(r["maximum_sampled_cgroup_memory_bytes"] for r in epochs))
    gates = minibatch.memory_gate(allocated=max(resources["peak_allocated_bytes"],*(r["peak_allocated_bytes"] for r in epochs)),
        reserved=max(resources["peak_reserved_bytes"],*(r["peak_reserved_bytes"] for r in epochs)),
        total=minibatch._DEVICE_TOTAL,
        sampled_free=min(resources["minimum_sampled_physical_free_bytes"],*(r["minimum_sampled_physical_free_bytes"] for r in epochs)))
    gates["maximum_sampled_cgroup_memory_bytes"] = peak_cgroup
    require(gates["resource_gate_passed"] and peak_cgroup < 160*minibatch.GIB, "Final audit resource gate failed")
    archive.write_json(PREFLIGHT_PATH, {"schema": PREFLIGHT_SCHEMA, "passed": True,
        "run_id": os.environ["BAGM_RUN_ID"], "profile": _CONFIG["metadata"]["execution_profile"],
        "optimizer_batch_mode": _CONFIG["trainer"]["optimizer_batch_mode"],
        "target_cells_per_optimizer_update": _CONFIG["trainer"]["target_cells_per_optimizer_update"],
        "optimizer_updates_per_global_epoch": base.OPTIMIZER_UPDATES_PER_EPOCH,
        "completed_epochs": len(epochs), "coverage_verified": True, "finite_gradients": True,
        "resource_gates": gates, "epoch_resources": epochs, "minimum_free_disk_gib": 5.0,
        "source_configuration_sha256": base._canonical_sha256(_CONFIG["source_configuration"]),
        "frozen_task_contract_sha256": _CONFIG["metadata"]["frozen_task_contract_sha256"],
        "code_hashes": minibatch._INITIALIZATION_RECEIPT["code_hashes"],
        "prevalence_reference_sha256": base._canonical_sha256(_BASELINE),
        "physical_memory_sampling_seconds": 5, "continuous_physical_peak_observed": False})


def _additional_provenance(validation: Any) -> Mapping[str, Any]:
    paired = _CONFIG["trainer"]["optimizer_batch_mode"] == "core_pair"
    return {**bce._additional_provenance(validation),
        "optimizer_batch_mode": _CONFIG["trainer"]["optimizer_batch_mode"],
        "target_partitioning": "single_rank_complete_core_targets" if paired else "balanced_contiguous_windows_shuffled_within_core_single_rank",
        "paired_update_sync": "first_core_no_sync_second_core_synced_one_rank" if paired else "one_target_window_per_optimizer_update",
        "objective_aggregation": "two_core_masked_entry_mean" if paired else "masked_entry_mean_per_target_window_then_entry_weighted_epoch",
        "target_cells_per_optimizer_update": _CONFIG["trainer"]["target_cells_per_optimizer_update"],
        "optimizer_updates_per_epoch": base.OPTIMIZER_UPDATES_PER_EPOCH,
        "model_seed": _CONFIG["seed"], "minimum_free_disk_gib": 5.0,
        "clean_learned_embeddings_recomputed_every_update": True,
        "execution_profile": _CONFIG["metadata"]["execution_profile"],
        "train_fitted_prevalence_reference": BASELINE_PATH,
        "fixed_training_evaluation": TRAIN_EVALUATION_PATH if _CONFIG["metadata"].get("evaluate_fixed_training_at_best",False) else None,
        "fixed_training_evaluation_used_for_selection": False,
        "equal_epochs_do_not_imply_equal_optimizer_updates": True}


def install_contract(config: Mapping[str, Any]) -> None:
    global _CONFIG
    validate_configuration(config)
    _CONFIG = dict(config); minibatch._CONFIG = dict(config)
    minibatch._resource_snapshot = _resource_snapshot
    bce._install_binary_expression_contract()
    trainer = config["trainer"]
    # Execute the real epoch controller before CUDA/data work, so invalid
    # minimum/maximum combinations fail before a costly first epoch.
    base.update_early_stopping(base.EarlyStoppingState(), 0.5, completed_epoch=1,
        minimum_epochs=int(trainer["minimum_global_epochs"]), maximum_epochs=int(trainer["max_epochs"]),
        patience=int(trainer["early_stopping_patience"]), min_delta=float(trainer["early_stopping_min_delta"]))
    extras = _EXTRA_COLUMNS + (("target_minibatch_nominal_size",) if trainer["optimizer_batch_mode"]=="target_window" else ())
    outputs = (BASELINE_PATH,PREFLIGHT_PATH,ENDPOINT_PATH) + ((TRAIN_EVALUATION_PATH,) if config["metadata"].get("evaluate_fixed_training_at_best",False) else ())
    for name, value in {
        "CAMPAIGN_ID": CAMPAIGN_ID, "PROTOCOL": PROTOCOL, "EVALUATION_PROTOCOL": PROTOCOL,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA, "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA,
        "MODEL_SEED": int(config["seed"]), "WORLD_SIZE": 1,
        "VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "OPTIMIZER_UPDATES_PER_EPOCH": int(trainer["optimizer_updates_per_global_epoch"]),
        "MINIMUM_EPOCHS": int(trainer["minimum_global_epochs"]), "MAXIMUM_EPOCHS": int(trainer["max_epochs"]),
        "EARLY_STOPPING_PATIENCE": int(trainer["early_stopping_patience"]),
        "EARLY_STOPPING_MIN_DELTA": float(trainer["early_stopping_min_delta"]),
        "MINIMUM_FREE_DISK_GIB": 5.0, "_require_disk_headroom": _disk_headroom,
        "FROZEN_TASK_CONTRACT_SHA256": config["metadata"]["frozen_task_contract_sha256"],
        "EPOCH_COLUMNS": bce.EPOCH_COLUMNS+extras, "PROVENANCE_RELATIVE_PATH": PROVENANCE_PATH,
        "ADDITIONAL_REQUIRED_FINAL_OUTPUTS": bce.ADDITIONAL_REQUIRED_FINAL_OUTPUTS+outputs,
        "validate_target_isolated_config": validate_configuration,
        "_train_epoch": _train_epoch, "_paired_update": _observed_pair_update,
        "_load_preflight_receipt": _load_preflight, "_make_epoch_row": _make_epoch_row,
        "_evaluate_validation": minibatch._evaluate_validation,
        "_additional_provenance": _additional_provenance,
        "_write_additional_final_artifacts": _write_final_artifacts,
    }.items():
        setattr(base,name,value)


def main(argv: list[str] | None = None) -> int:
    parser = base.build_parser(); parser.description = __doc__
    args = parser.parse_args(argv)
    config = base.load_yaml_mapping(args.config)
    install_contract(config)
    torch.set_num_threads(int(config["launcher"]["cpu_threads"]))
    torch.set_num_interop_threads(1)
    try:
        return base.main(argv)
    finally:
        if minibatch._SAMPLER is not None:
            minibatch._SAMPLER.stop()


if __name__ == "__main__":
    raise SystemExit(main())

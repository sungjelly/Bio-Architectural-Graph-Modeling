"""Post-hoc Pearson replay of frozen Local Gene Residual checkpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.myjju_genemae import MaskedRegressionAccumulator
from spatial_benchmark.paths import current_paths


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def verify_accumulator():
    """Independent metric check, including partial masks and undefined rows."""
    from scipy.stats import pearsonr
    rng = np.random.default_rng(871)
    y = rng.normal(size=(12, 7))
    p = .7*y + rng.normal(size=y.shape)
    mask = rng.random(y.shape) > .3
    mask[0] = False
    mask[1] = False
    mask[1, 0] = True
    y[2] = 0.
    p[3] = 2.
    y[:, 6] = 0.
    a = MaskedRegressionAccumulator(7)
    a.update(y[:5], p[:5], mask[:5])
    a.update(y[5:], p[5:], mask[5:])
    result = a.finalize()

    def valid_r(x, z):
        if len(x) < 2 or ((x-x.mean())**2).sum() <= 1e-12 or ((z-z.mean())**2).sum() <= 1e-12:
            return np.nan
        return float(pearsonr(x, z).statistic)

    rows = np.array([valid_r(y[i, mask[i]], p[i, mask[i]]) for i in range(len(y))])
    cols = np.array([valid_r(y[mask[:, j], j], p[mask[:, j], j]) for j in range(y.shape[1])])
    expected = {
        "pooled_pearson": valid_r(y[mask], p[mask]),
        "cell_pearson_mean": float(np.nanmean(rows)),
        "gene_pearson_mean": float(np.nanmean(cols)),
        "masked_mae": float(np.abs(y[mask]-p[mask]).mean()),
        "masked_mse": float(np.square(y[mask]-p[mask]).mean()),
    }
    differences = {k: abs(result[k]-v) for k, v in expected.items()}
    assert max(differences.values()) < 1e-12, differences
    assert result["n_valid_cells"] == int(np.isfinite(rows).sum())
    assert result["n_valid_genes"] == int(np.isfinite(cols).sum())
    np.testing.assert_allclose(a.per_gene_pearson(), cols, atol=1e-12, rtol=0, equal_nan=True)
    return {"passed": True, "reference": "scipy.stats.pearsonr", "tolerance": 1e-12,
            "maximum_absolute_difference": max(differences.values()),
            "includes_empty_singleton_constant_rows_and_constant_gene": True}


def finalize(acc, n_cells):
    result = acc.finalize()
    result.update(masked_rmse=math.sqrt(result["masked_mse"]), n_cells=n_cells,
                  n_undefined_cells=n_cells-result["n_valid_cells"],
                  n_undefined_genes=acc.num_genes-result["n_valid_genes"])
    return clean(result)


@torch.no_grad()
def evaluate_local(stage_dir, checkpoint_path):
    from spatial_benchmark import local_gene_data, local_gene_residual, local_gene_metrics
    from spatial_benchmark.local_gene_data import load_local_gene_data, make_mask
    from spatial_benchmark.local_gene_residual import LocalGeneResidualCountModel
    from spatial_benchmark.local_gene_metrics import CountMetricAccumulator

    stage_dir, checkpoint_path = Path(stage_dir), Path(checkpoint_path)
    source = checkpoint_path.parent.parent
    began = time.monotonic()
    paths = current_paths()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.cuda.mem_get_info()[0] > 2*1024**3
    assert shutil.disk_usage(paths.scratch_root).free > 5*1024**3
    assert float((torch.ones(4, device="cuda")*2).sum()) == 8.
    torch.cuda.reset_peak_memory_stats()
    fixture = verify_accumulator()
    write(stage_dir/"metric_fixture.json", fixture)
    saved_hashes = json.loads((source/"provenance/source_hashes.json").read_text())
    source_hashes = {}
    for module in (local_gene_data, local_gene_residual, local_gene_metrics):
        path = Path(module.__file__)
        key = "src/spatial_benchmark/" + path.name
        source_hashes[str(path)] = sha256_file(path)
        assert source_hashes[str(path)] == saved_hashes[key], key
    print(json.dumps({"stage": "loading_verified_local_data", "source_run_id": source.name}), flush=True)
    bundle = load_local_gene_data(paths=paths)
    write(stage_dir/"data_provenance.json", dict(bundle.provenance))
    # The trusted local checkpoint contains NumPy/Python RNG state as well as weights.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["run_id"] == source.name and payload["epoch"] == 30
    config = payload["config"]
    model = LocalGeneResidualCountModel(1000, covariate_dim=22,
        hidden_dim=config["model"]["hidden_dim"], dropout=config["model"]["dropout"],
        use_local=config["model"]["use_local"])
    model.load_state_dict(payload["model"], strict=True)
    model.cuda().eval()
    del payload
    archived_masks = json.loads((source/"diagnostics/validation_masks.json").read_text())
    archived_core = json.loads((source/"results/per_core.json").read_text())
    archived_global = json.loads((source/"metrics/final.json").read_text())
    global_acc = {s: MaskedRegressionAccumulator(1000) for s in ("raw_count", "log1p")}
    global_count = CountMetricAccumulator()
    receipts = []
    for core in bundle.validation:
        core_start = time.monotonic()
        realization = make_mask(core, epoch=None)
        expected = archived_masks[core.alias]
        assert realization.mask_sha256 == expected["mask_sha256"]
        assert realization.n_masked_entries == expected["masked_entry_count"]
        tensors = {name: torch.as_tensor(np.asarray(value).copy(), device="cuda") for name, value in {
            "expression": core.expression, "counts": core.counts, "covariates": core.covariates,
            "local_log_mean": core.local_log_mean, "local_positive_fraction": core.local_positive_fraction,
        }.items()}
        masks = torch.as_tensor(realization.mask.copy(), device="cuda")
        per_acc = {s: MaskedRegressionAccumulator(1000) for s in global_acc}
        per_count = CountMetricAccumulator()
        for start in range(0, core.n_cells, 1024):
            ix = slice(start, min(start+1024, core.n_cells))
            mask = masks[ix]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(tensors["expression"][ix], mask, tensors["covariates"][ix],
                               tensors["local_log_mean"][ix], tensors["local_positive_fraction"][ix])
            per_count.update(output, tensors["counts"][ix], mask)
            predicted = output.marginal_mean.float().cpu().numpy().astype(np.float64)
            target = core.counts[ix].astype(np.float64)
            selected = realization.mask[ix]
            for scale, truth, pred in (("raw_count", target, predicted),
                                      ("log1p", np.log1p(target), np.log1p(predicted))):
                per_acc[scale].update(truth, pred, selected)
                global_acc[scale].update(truth, pred, selected)
        native = per_count.compute()
        differences = {k: abs(native[k]-v) for k, v in archived_core[core.alias].items()
                       if isinstance(v, (int, float)) and v is not None}
        assert max(differences.values()) <= 1e-5, (core.alias, differences)
        # Independently verify the float64 regression path against native metric arithmetic.
        metrics = {s: finalize(a, core.n_cells) for s, a in per_acc.items()}
        for scale, prefix in (("raw_count", "raw_count"), ("log1p", "log1p_count")):
            assert abs(metrics[scale]["masked_mae"]-native[prefix+"_mae"]) <= 1e-5
            assert abs(metrics[scale]["masked_rmse"]-native[prefix+"_rmse"]) <= 1e-5
        global_count.merge(per_count)
        receipt = {"core_alias": core.alias, "n_cells": core.n_cells,
            "mask": realization.to_receipt(), "metrics": metrics, "native_metrics": native,
            "replay_max_absolute_difference": max(differences.values()), "replay_verified": True,
            "runtime_seconds": time.monotonic()-core_start}
        write(stage_dir/(core.alias+".json"), receipt)
        receipts.append(receipt)
        print(json.dumps({"stage": "core_completed", "core_alias": core.alias,
                          "replay_max_absolute_difference": max(differences.values())}), flush=True)
        del tensors, masks, output
    native_global = global_count.compute()
    global_differences = {k: abs(native_global[k.removeprefix("val/")]-v)
        for k, v in archived_global.items() if isinstance(v, (int, float))}
    assert max(global_differences.values()) <= 1e-5, global_differences
    n_cells = sum(c.n_cells for c in bundle.validation)
    metrics = {s: finalize(a, n_cells) for s, a in global_acc.items()}
    assert n_cells == 37367 and metrics["raw_count"]["n_masked"] == 18708207
    result = {"status": "completed", "verified": True, "source_run_id": source.name,
        "checkpoint_sha256": sha256_file(checkpoint_path), "epoch": 30, "seed": config["seed"],
        "evaluation_scope": "validation_used_for_checkpoint_selection_one_held_out_donor_two_cores",
        "prediction": "marginal_expected_raw_count; log1p_of_that_mean_for_log_metrics",
        "global": metrics, "per_core": receipts, "native_metrics": native_global,
        "per_gene_pearson": {s: a.per_gene_pearson() for s, a in global_acc.items()},
        "verification": {"passed": True, "native_metric_absolute_tolerance": 1e-5,
            "maximum_global_absolute_difference": max(global_differences.values()), "metric_fixture": fixture},
        "source_hashes": source_hashes, "runtime_seconds": time.monotonic()-began,
        "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    result = clean(result)
    write(stage_dir/"local_metrics.json", result)
    return result

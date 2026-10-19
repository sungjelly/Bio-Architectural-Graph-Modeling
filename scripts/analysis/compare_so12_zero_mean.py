#!/usr/bin/env python3
"""Replay zero/training-mean baselines on the exact SO12 validation masks."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.so12_relative_qkv_training import (
    aggregate_statistics, digest, file_hash, make_mask, scientific_binding,
    source_binding,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fit_constants(targets, training_aliases):
    """Fit both per-gene means exclusively on the explicit training partition."""
    core_means, sums, sizes = [], [], []
    for alias in training_aliases:
        target = targets[alias]
        core_means.append(target.mean(axis=0, dtype=np.float64))
        sums.append(target.sum(axis=0, dtype=np.float64))
        sizes.append(len(target))
    return {
        "zero": np.zeros_like(core_means[0], dtype=np.float32),
        "mean_equal_core": np.mean(core_means, axis=0).astype(np.float32),
        "mean_equal_cell": (np.sum(sums, axis=0) / sum(sizes)).astype(np.float32),
    }


def constant_statistics(target, mask, constant, chunk_cells=1024):
    """Float64 sufficient statistics, bounded in memory and masked-entry only."""
    total = np.zeros(9, dtype=np.float64)
    for start in range(0, len(target), chunk_cells):
        m = mask[start:start + chunk_cells]
        y = np.asarray(target[start:start + chunk_cells])[m].astype(np.float64)
        p = np.broadcast_to(constant, m.shape)[m].astype(np.float64)
        e = p - y
        a = np.abs(e)
        total += np.array([
            len(y), np.where(a <= 1., .5 * e * e, a - .5).sum(), a.sum(),
            np.square(e).sum(), p.sum(), y.sum(), np.square(p).sum(),
            np.square(y).sum(), (p * y).sum(),
        ])
    return total.tolist()


def check_replay(actual, expected, tolerance=1e-8):
    for key in ("equal_core_huber", "pooled_huber", "pooled_mae", "pooled_mse", "pooled_pearson"):
        if expected[key] is None:
            assert actual[key] is None, key
        else:
            assert abs(actual[key] - expected[key]) <= tolerance, (key, actual[key], expected[key])
    assert actual["masked_entries"] == expected["masked_entries"]
    assert actual["view_count"] == expected["view_count"]
    assert set(actual["per_core_huber"]) == set(expected["per_core_huber"])
    for alias, value in expected["per_core_huber"].items():
        assert abs(actual["per_core_huber"][alias] - value) <= tolerance, alias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths, registry = current_paths(), Registry()
    run = registry.get_run(args.run_id)
    if run is None:
        raise ValueError("Unknown source run")
    run_root = paths.scratch_root / "active_runs" / args.run_id
    if not run_root.exists():
        run_root = Path(run["artifact_path"])
    config = yaml.safe_load((run_root / "config.resolved.yaml").read_text())
    startup = json.loads((run_root / "diagnostics/training_start.json").read_text())
    # Read one inode: trainer atomically replaces best.ckpt while it improves.
    checkpoint_bytes = (run_root / "checkpoints/best.ckpt").read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    best = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    assert best["checkpoint_role"] == "best"
    assert best["run_id"] == args.run_id
    assert best["binding"] == startup["binding"]
    assert best["binding"]["config_sha256"] == scientific_binding(config)
    assert best["binding"]["source_sha256"] == source_binding()
    epoch = int(best["completed_epoch"])
    best_row = next(row for row in best["history"] if row["global_epoch"] == epoch)
    model_metrics = best_row["validation"]
    assert abs(model_metrics["equal_core_huber"] - best["primary_metric_value"]) < 1e-12
    assert best_row["validation_mask_sha256"] == startup["validation_mask_sha256"]
    del best, checkpoint_bytes
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eid = f"ev_so12_zero_mean_e{epoch:03d}_{stamp}"
    # A separate evaluation directory cannot race source-run publication or
    # modify an already finalized source bundle.
    stage = paths.scratch_root / "active_runs" / eid
    final = paths.report_root / "analyses/so12_zero_mean" / eid
    assert not stage.exists() and not final.exists()
    stage.mkdir(parents=True)
    provenance = {
        "evaluation_id": eid, "source_run_id": args.run_id, "source_run_status": run["status"],
        "checkpoint_epoch": epoch, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metric_source": "stored_validation_record_for_the_same_checkpoint_epoch",
        "started_at": now(), "command": sys.argv, "device": "cpu", "threads": 2,
        "target_scale": config["dataset"]["target_scale"], "config_sha256": scientific_binding(config),
        "training_source_sha256": source_binding(), "analysis_source_sha256": file_hash(Path(__file__)),
        "mask_sha256": startup["validation_mask_sha256"],
        "split": startup["dataset"]["provenance"]["split"],
        "interpretation": "exploratory_live_checkpoint_validation_comparison",
        "baseline_definitions": {"zero": "zero_standardized_Pearson_feature",
            "mean_equal_core": "per_gene_training_mean_equal_core_weights",
            "mean_equal_cell": "per_gene_training_mean_equal_cell_weights"},
    }
    write_json(stage / "config.resolved.json", config)
    write_json(stage / "model_best_epoch.json", best_row)
    write_json(stage / "provenance.json", provenance)
    (stage / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    registry.create_evaluation(eid, run_id=args.run_id, checkpoint_name=f"best_epoch_{epoch}",
        dataset_id=config["dataset"]["dataset_id"], split_id=config["dataset"]["split_id"],
        status="running", metrics={"checkpoint_sha256": checkpoint_sha}, artifact_path=stage)
    started = time.monotonic()
    try:
        data = paths.data_root / "processed/so12_pearson_integration_v1"
        expected_manifest = startup["dataset"]["provenance"]["source_manifest"]["sha256"]
        assert file_hash(data / "manifest.json") == expected_manifest
        manifest = json.loads((data / "manifest.json").read_text())
        records = {r["core"]: r for r in manifest["matrices"] if r["method"] == "pearson"}
        train = config["dataset"]["training_core_aliases"]
        val = config["dataset"]["validation_core_aliases"]
        assert len(train) == 20 and len(val) == 8 and not set(train) & set(val)
        targets, source_checksums = {}, {}
        for alias in train + val:
            item = records[alias]
            path = data / item["path"]
            assert file_hash(path) == item["sha256"], alias
            target = np.load(path, mmap_mode="r", allow_pickle=False)
            assert target.shape == (item["rows"], 1000) and target.dtype == np.float32
            assert np.isfinite(target).all()
            targets[alias] = target
            source_checksums[item["path"]] = item["sha256"]
        assert sum(len(targets[a]) for a in train) == 301198
        assert sum(len(targets[a]) for a in val) == 106461
        constants = fit_constants(targets, train)
        assert digest(constants["mean_equal_core"].tolist()) == startup["dataset"]["provenance"]["preprocessing"]["constant_baseline_sha256"]
        np.savez(stage / "fitted_constants.npz", **constants)
        accumulated = {name: [] for name in constants}
        receipts = []
        for alias in val:
            target = targets[alias]
            core = SimpleNamespace(alias=alias, raw_counts=target)
            for view in range(config["evaluation"]["validation_mask_views"]):
                mask, receipt = make_mask(core, epoch=0, view=view, training=False,
                                         base_seed=config["masking"]["mask_base_seed"])
                receipts.append(receipt)
                for name, constant in constants.items():
                    stats = constant_statistics(target, mask, constant)
                    assert int(stats[0]) == receipt["masked_entries"]
                    accumulated[name].append({"alias": alias, "view": view, "statistics": stats})
            print(json.dumps({"core": alias, "views_complete": 3}), flush=True)
        receipts.sort(key=lambda r: (r["alias"], r["view"]))
        assert digest(receipts) == startup["validation_mask_sha256"]
        results = {name: aggregate_statistics(rows) for name, rows in accumulated.items()}
        check_replay(results["mean_equal_core"], startup["constant_reference"])
        assert results["zero"]["pooled_pearson"] is None
        for name, metrics in results.items():
            assert metrics["masked_entries"] == model_metrics["masked_entries"]
            assert metrics["view_count"] == model_metrics["view_count"] == 24
        results["model_best"] = model_metrics
        fields = ("equal_core_huber", "pooled_huber", "pooled_mae", "pooled_mse", "pooled_pearson")
        table = [{"predictor": name, "checkpoint_epoch": epoch if name == "model_best" else None,
                  **{key: metrics[key] for key in fields}}
                 for name, metrics in results.items()]
        per_core = [{"core": alias, **{name: value["per_core_huber"][alias] for name, value in results.items()}}
                    for alias in val]
        write_csv(stage / "comparison.csv", table)
        write_csv(stage / "per_core_huber.csv", per_core)
        write_json(stage / "metrics.json", results)
        write_json(stage / "mask_receipts.json", receipts)
        write_json(stage / "baseline_view_statistics.json", accumulated)
        write_json(stage / "verification.json", {
            "passed": True, "mean_baseline_replay_tolerance": 1e-8,
            "masked_entries": model_metrics["masked_entries"], "validation_views": 24,
            "target_manifest_sha256": expected_manifest, "target_files": source_checksums,
            "duration_seconds": time.monotonic() - started, "finished_at": now(),
            "zero_correlation": "undefined_due_to_zero_prediction_variance",
        })
        files = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        write_json(stage / "manifest.json", {"evaluation_id": eid, "source_run_id": args.run_id,
                                             "files": files})
        write_json(stage / "_SUCCESS", {"manifest_sha256": file_hash(stage / "manifest.json")})
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, final)
        for name, sha in files.items():
            assert file_hash(final / name) == sha
        with registry.transaction(immediate=True) as con:
            con.execute("UPDATE evaluations SET status=?,artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=?",
                        ("completed", str(final), json.dumps(results), now(), eid))
        registry.record_artifact(args.run_id, evaluation_id=eid, kind="so12_zero_mean_comparison",
            path=final / "manifest.json", sha256=file_hash(final / "manifest.json"),
            size_bytes=(final / "manifest.json").stat().st_size, status="present")
        print(json.dumps({"report": str(final), "checkpoint_epoch": epoch, "comparison": table}, indent=2))
    except Exception as error:
        failed_root = final if final.exists() else stage
        write_json(failed_root / "_FAILED", {"error": repr(error), "time": now()})
        with registry.transaction(immediate=True) as con:
            con.execute("UPDATE evaluations SET status=?,artifact_path=?,finished_at=? WHERE evaluation_id=?",
                        ("failed", str(failed_root), now(), eid))
        raise


if __name__ == "__main__":
    main()

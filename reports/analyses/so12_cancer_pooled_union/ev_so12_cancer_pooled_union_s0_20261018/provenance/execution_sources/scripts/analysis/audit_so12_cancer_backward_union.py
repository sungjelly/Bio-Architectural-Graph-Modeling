#!/usr/bin/env python3
"""Independently replay native-pixel backward paths and fixed cell retention.

No metric selector or scoring implementation is imported. The only write is a
new independent review receipt inside the designated active evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spatial_benchmark.paths import current_paths

EVALUATION = "ev_so12_cancer_backward_union_s0_20261018"
PARENT_EVALUATION = "ev_so12_cancer_comparison_s0_20261018"
PARENT_SHA = "e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0"
METHODS = ("h0", "h1", "h2", "h3", "h4", "pca", "scvi")
CORES = (15, 23)
N_CELLS = 81607


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def local(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
    result = root / relative
    require(not result.is_symlink() and result.resolve().is_relative_to(root.resolve()), "Artifact escapes bundle")
    return result


def verify_parent(parent):
    require(sha(parent / "manifest.json") == PARENT_SHA, "Parent manifest changed")
    manifest = read(parent / "manifest.json")
    require(manifest["passed"] is True and manifest["status"] == "complete"
            and manifest["evaluation_id"] == PARENT_EVALUATION, "Parent is incomplete")
    require(read(parent / "_SUCCESS")["manifest_sha256"] == PARENT_SHA, "Parent marker changed")
    require(not any(path.is_symlink() for path in parent.rglob("*")), "Parent contains symlinks")
    actual = {str(path.relative_to(parent)) for path in parent.rglob("*") if path.is_file()} - {"manifest.json", "_SUCCESS"}
    require(actual == set(manifest["files"]), "Parent inventory changed")
    for relative, item in manifest["files"].items():
        path = local(parent, relative)
        require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], f"Parent file changed: {relative}")
    return manifest


def equal(actual, expected, name):
    okay = actual is None if expected is None else actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    require(okay, f"Incorrect {name}: {actual}, expected {expected}")


def pixel_counts(owner, roi, valid, n_cells=N_CELLS):
    require(owner.shape == roi.shape == valid.shape and owner.ndim == 2, "Raster dimensions differ")
    require(owner.dtype.kind == "i" and roi.dtype == valid.dtype == np.bool_, "Raster dtypes differ")
    require(np.all((owner >= -1) & (owner < n_cells)), "Invalid canonical owner")
    visible = valid & (owner >= 0)
    return {"area": np.bincount(owner[visible], minlength=n_cells),
            "positive": np.bincount(owner[visible & roi], minlength=n_cells),
            "footprint_pixels": int(visible.sum()), "valid_pixels": int(valid.sum()),
            "full_positive": int((valid & roi).sum())}


def aggregate(labels, counts):
    n = int(labels.max()) + 1
    area, positive = np.zeros(n, np.int64), np.zeros(n, np.int64)
    np.add.at(area, labels, counts["area"])
    np.add.at(positive, labels, counts["positive"])
    visible = counts["area"] > 0
    roi_cells = counts["positive"] > 0
    majority = visible & (2 * counts["positive"] >= counts["area"])
    return {"area": area, "positive": positive,
            "roi_cells": np.bincount(labels[roi_cells], minlength=n),
            "majority": np.bincount(labels[majority], minlength=n),
            "visible": np.bincount(labels[visible], minlength=n)}


def objective(selected, grouped):
    positive = sum(int(v) for v in grouped["positive"])
    tp = sum(int(grouped["positive"][i]) for i in selected)
    area = sum(int(grouped["area"][i]) for i in selected)
    return Fraction(2 * tp, positive + area) if positive else Fraction(0)


def independent_path(grouped):
    """Exact best deletion at each step, lexicographic retained-ID ties."""
    selected = tuple(int(i) for i in np.flatnonzero(grouped["positive"] > 0))
    path = [selected]
    while selected:
        candidates = [selected[:i] + selected[i+1:] for i in range(len(selected))]
        selected = min(candidates, key=lambda ids: (-objective(ids, grouped), ids))
        path.append(selected)
    return path


def best_index(path, grouped, indices=None):
    indices = list(range(len(path))) if indices is None else list(indices)
    return min(indices, key=lambda i: (-objective(path[i], grouped), len(path[i]), path[i])) if indices else None


def check_score(score, tp, predicted, total_positive, support):
    fp, fn = predicted - tp, total_positive - tp
    tn = support - tp - fp - fn
    require(min(tp, fp, fn, tn) >= 0, "Invalid confusion counts")
    for key, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "predicted_pixels": predicted,
                       "positive_pixels": total_positive, "evaluable_pixels": support}.items():
        require(score[key] == value, f"Incorrect native pixel {key}")
    for key, value in {"dice_f1": 2 * tp / (predicted + total_positive) if total_positive else None,
                       "iou": tp / (predicted + total_positive - tp) if total_positive else None,
                       "precision": tp / predicted if predicted else None,
                       "recall": tp / total_positive if total_positive else None}.items():
        equal(score[key], value, key)


def check_path(record, labels, counts, parent_oracle):
    require(record["status"] == "complete", "Backward path incomplete")
    grouped = aggregate(labels, counts)
    path = independent_path(grouped)
    steps = record["steps"]
    initial = path[0]
    require(record["initial_overlap_cluster_ids"] == list(initial), "Starting overlap clusters differ")
    require(len(steps) == len(initial) + 1, "Path omits deletion steps or empty endpoint")
    total_roi = sum(int(v) for v in grouped["roi_cells"])
    total_majority = sum(int(v) for v in grouped["majority"])
    total_visible = sum(int(v) for v in grouped["visible"])
    for name, expected in (("initial_overlap_clusters", len(initial)), ("roi_cells_total", total_roi),
                           ("majority_cells_total", total_majority), ("visible_cells_total", total_visible)):
        require(record[name] == expected, f"Path-level denominator differs: {name}")
    require(record["optimal_fixed_cardinality_subset_claimed"] is False,
            "Greedy path has an unsupported fixed-cardinality optimality claim")
    retained = [sum(int(grouped["roi_cells"][i]) for i in ids) for ids in path]
    values = [objective(ids, grouped) for ids in path]
    for index, (step, ids) in enumerate(zip(steps, path, strict=True)):
        removed = None if index == 0 else next(iter(set(path[index-1]) - set(ids)))
        require(step["step_index"] == index and step["removed_cluster"] == removed
                and step["selected_clusters"] == list(ids) and step["k"] == len(ids), "Greedy exact deletion or tie differs")
        require(step["initial_overlap_clusters"] == len(initial), "Cluster denominator changed")
        equal(step["cluster_retained_pct"], 100 * len(ids) / len(initial) if initial else None, "cluster retained percent")
        for field, total, key in (("roi_cells", total_roi, "roi_cells"), ("majority_cells", total_majority, "majority")):
            count = sum(int(grouped[key][i]) for i in ids)
            require(step[field + "_retained"] == count and step[field + "_total"] == total, f"Incorrect fixed {field} counts")
            equal(step[field + "_retained_pct"], 100 * count / total if total else None, field + " retention percent")
        selected_visible = sum(int(grouped["visible"][i]) for i in ids)
        require(step["selected_visible_cells"] == selected_visible and step["visible_cells_total"] == total_visible, "Visible-cell denominator differs")
        equal(step["selected_visible_cells_pct"], 100 * selected_visible / total_visible if total_visible else None, "selected visible percent")
        tp = sum(int(grouped["positive"][i]) for i in ids)
        predicted = sum(int(grouped["area"][i]) for i in ids)
        check_score(step["footprint"], tp, predicted, int(grouped["positive"].sum()), counts["footprint_pixels"])
        check_score(step["full_region"], tp, predicted, counts["full_positive"], counts["valid_pixels"])
        nondominated = not any(values[j] >= values[index] and retained[j] >= retained[index]
                              and (values[j] > values[index] or retained[j] > retained[index]) for j in range(len(path)))
        require(step["pareto_nondominated"] is nondominated, "Pareto membership differs")
    best = best_index(path, grouped)
    require(record["best_step_index"] == best, "Path-maximum tie rule differs")
    for threshold in (90, 95):
        eligible = [i for i, count in enumerate(retained) if total_roi and 100 * count >= threshold * total_roi]
        require(record[f"best_retention{threshold}_step_index"] == best_index(path, grouped, eligible), "Retention operating point differs")
    parent_ids = tuple(parent_oracle["selected_clusters"])
    require(record["parent_unrestricted_selected_clusters"] == list(parent_ids), "Parent unrestricted IDs differ")
    upper = objective(parent_ids, grouped)
    require(all(value <= upper for value in values), "Path exceeds parent unrestricted bound")
    reference = parent_oracle["footprint"]["dice_f1"]
    equal(record["parent_unrestricted_f1"], reference, "parent unrestricted F1")
    equal(record["greedy_gap_to_parent_unrestricted_f1"], reference - float(values[best]) if reference is not None else None,
          "greedy path gap to unrestricted")
    require(retained == sorted(retained, reverse=True), "Retained cell count increases")
    return {"steps": len(steps), "best_f1": float(values[best]) if reference is not None else None,
            "parent_f1": reference, "best_k": len(path[best]), "best_roi_retained_pct": steps[best]["roi_cells_retained_pct"],
            "attains_parent_exact_fraction": values[best] == upper}


def audit(parent, output, paths=None):
    started = time.monotonic()
    paths = paths or current_paths()
    parent, output = Path(parent).resolve(), Path(output).resolve()
    require(parent == (paths.report_root / "analyses/so12_cancer_comparison" / PARENT_EVALUATION).resolve(), "Wrong parent")
    require(output == (paths.scratch_root / "active_runs" / EVALUATION).resolve() and output.is_dir()
            and not (output / "_SUCCESS").exists(), "Audit requires designated active output")
    inventory = verify_parent(parent)
    metrics_path, summary_path = output / "metrics/backward_union.json", output / "backward_metrics_summary.json"
    metrics, summary = read(metrics_path), read(summary_path)
    require(metrics["passed"] is True and metrics["status"] == "complete"
            and summary["passed"] is True and summary["status"] == "complete", "Metrics incomplete")
    require(metrics["n_cells"] == N_CELLS and metrics["test_core_numbers"] == list(CORES), "Metric population differs")
    require(metrics["all_evaluable_pixels_retained"] is True
            and metrics["cell_retention_unweighted"] is True and metrics["cell_retention_denominators_fixed"] is True
            and metrics["selection_support"] == "segmented_cell_footprint", "Scoring/retention semantics differ")
    require(metrics["parent"]["manifest_sha256"] == PARENT_SHA and metrics["parent"]["evaluation_id"] == PARENT_EVALUATION
            and Path(metrics["parent"]["path"]).resolve() == parent, "Metric parent differs")
    for relative, digest in metrics["input_sha256"].items():
        require(sha(local(parent, relative)) == digest, f"Metric input changed: {relative}")
    for relative, digest in metrics["code_sha256"].items():
        require(sha(local(paths.project_root, relative)) == digest, f"Metric source changed: {relative}")
    for relative, digest in summary["output_sha256"].items():
        require(sha(local(output, relative)) == digest, f"Metric output changed: {relative}")
    cells = pd.read_csv(parent / "cells.csv.gz", usecols=["row_index", "core_number"])
    require(len(cells) == N_CELLS and np.array_equal(cells.row_index, np.arange(N_CELLS))
            and cells.core_number.value_counts().to_dict() == {23: 43462, 15: 38145}, "Canonical cell coverage differs")
    registration = read(parent / "registration/registration_manifest.json")
    require(registration["canonical_cells"]["sha256"] == sha(parent / "cells.csv.gz"), "Registration row identity differs")
    counts = {}
    for item in registration["cores"]:
        core = item["core_number"]
        raster = local(parent, item["raster_path"])
        require(sha(raster) == item["raster_sha256"], "Owner raster changed")
        with np.load(raster, allow_pickle=False) as arrays:
            counts[core] = pixel_counts(arrays["owner"], arrays["roi"], arrays["valid"])
        require(np.all(cells.core_number.to_numpy()[counts[core]["area"] > 0] == core), "Owner raster crosses cores")
    expected = {(m, r, s) for m in METHODS for r in (.25, .5, 1., 2.) for s in (0, 1)}
    def index(rows):
        result = {(r["method"], r["resolution"], r["seed"]): r for r in rows}
        require(len(rows) == 56 and set(result) == expected, "Incomplete or duplicated partition grid")
        return result
    measured = index(metrics["results"])
    partitions = index(read(parent / "clustering_summary.json")["partitions"])
    unrestricted = index(read(parent / "metrics/pixel_agreement.json")["results"])
    checked = []
    for key in sorted(expected):
        item = partitions[key]["labels"]
        path = local(parent, item["path"])
        require(sha(path) == item["sha256"], "Partition changed")
        labels = np.load(path, allow_pickle=False)
        require(labels.shape == (N_CELLS,) and labels.dtype.kind in "iu" and np.all(labels >= 0), "Invalid canonical labels")
        for core in CORES:
            result = check_path(measured[key]["cores"][str(core)], labels, counts[core],
                                unrestricted[key]["evaluation"]["cores"][str(core)]["oracle"])
            checked.append({"method": key[0], "resolution": key[1], "seed": key[2], "core": core, **result})
    require(metrics["partition_count"] == 56 and metrics["core_path_count"] == len(checked)
            and metrics["step_count"] == sum(r["steps"] for r in checked), "Summary path/step coverage differs")
    verify_parent(parent)
    receipt = {"schema": "so12_cancer_backward_union_independent_review_v1", "status": "complete", "passed": True,
        "n_cells": N_CELLS, "partition_count": 56, "core_paths_checked": len(checked),
        "steps_checked": sum(r["steps"] for r in checked), "binary_score_records_checked": 2 * sum(r["steps"] for r in checked),
        "exact_removals_and_ties_verified": True, "fixed_unweighted_cell_denominators_verified": True,
        "retention90_95_operating_points_verified": True, "pareto_points_verified": True,
        "native_pixel_denominators_and_outside_outline_penalties_verified": True,
        "parent_unrestricted_bound_verified": True,
        "paths_attaining_parent_unrestricted_exact_fraction": sum(r["attains_parent_exact_fraction"] for r in checked),
        "parent_files_verified_before_after": len(inventory["files"]), "parent_manifest_sha256": PARENT_SHA,
        "metrics_sha256": sha(metrics_path), "metrics_summary_sha256": sha(summary_path), "audit_source_sha256": sha(Path(__file__)),
        "primary": [r for r in checked if (r["resolution"], r["seed"]) == (1., 0)],
        "created_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Observed agreement with the unrestricted maximum is an empirical audit result; greedy removal is not asserted to optimize every fixed-cardinality subset. Annotation-selected agreement and retention remain descriptive."}
    destination = output / "independent_metrics_review.json"
    require(not destination.exists(), "Refusing to overwrite an independent review")
    temporary = destination.with_suffix(f".json.tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(destination)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = audit(args.parent, args.output)
    print(json.dumps({k: receipt[k] for k in ("passed", "core_paths_checked", "steps_checked", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent native-raster audit of one shared pooled-core cluster union.

Reuse only the independent backward audit's count and exact-path checks. No
scientific selection implementation is imported. The unrestricted pooled optimum
is certified through exact inclusion/exclusion inequalities, independently of
the metric implementation's purity-prefix enumeration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from scripts.analysis import audit_so12_cancer_backward_union as independent
except ModuleNotFoundError:
    import audit_so12_cancer_backward_union as independent

require, read, sha, local = independent.require, independent.read, independent.sha, independent.local
EVALUATION = "ev_so12_cancer_pooled_union_s0_20261018"
ORACLE_ORIGIN = "recomputed_exact_unrestricted_oracle_on_pooled_core_counts"
CORES = (15, 23)


def pool_native(counts):
    require(set(counts) == set(CORES), "Both native core rasters are required")
    left, right = counts[15], counts[23]
    require(left["area"].shape == right["area"].shape
            and not np.any((left["area"] > 0) & (right["area"] > 0)), "Core cell ownership is not disjoint")
    return {"area": left["area"] + right["area"], "positive": left["positive"] + right["positive"],
            **{name: left[name] + right[name] for name in ("footprint_pixels", "valid_pixels", "full_positive")}}


def certify_unrestricted(selected, grouped):
    """Certify the exact global F1 maximum and its minimum-cardinality tie.

For score F, adding an omitted cluster changes the objective according to
2*positive_i - F*area_i. Selected clusters must have strictly positive marginal
value; omitted clusters must have nonpositive value. Equality clusters are
omitted by the minimum-cardinality tie rule. These inequalities certify all
subsets at once, without invoking the metric's optimizer.
"""
    selected = tuple(selected)
    require(selected == tuple(sorted(set(selected)))
            and all(0 <= i < len(grouped["area"]) for i in selected), "Invalid unrestricted cluster IDs")
    value = independent.objective(selected, grouped)
    selected_set = set(selected)
    for index, (area, positive) in enumerate(zip(grouped["area"], grouped["positive"], strict=True)):
        margin = 2 * int(positive) * value.denominator - value.numerator * int(area)
        require(margin > 0 if index in selected_set else margin <= 0,
                "Pooled unrestricted selection fails exact optimality/minimum-cardinality certificate")
    return value


def score_selection(selection, ids, grouped, counts):
    require(selection["status"] == "complete" and selection["selected_clusters"] == list(ids),
            "Per-core prediction differs from the shared selected cluster IDs")
    tp = sum(int(grouped["positive"][i]) for i in ids)
    predicted = sum(int(grouped["area"][i]) for i in ids)
    independent.check_score(selection["footprint"], tp, predicted, int(grouped["positive"].sum()), counts["footprint_pixels"])
    independent.check_score(selection["full_region"], tp, predicted, counts["full_positive"], counts["valid_pixels"])


def check_pooled_path(curve, labels, counts):
    require(curve["shared_cluster_selection"] is True and curve["optimization"] == "pooled_micro_pixel_dice"
            and curve["per_core_optimized_masks_reused"] is False
            and curve["per_core_f1_averaged_for_selection"] is False, "Pooled objective/shared-selection contract differs")
    require(curve["parent_unrestricted_origin"] == ORACLE_ORIGIN, "Pooled bound reuses per-core optimization")
    pooled = pool_native(counts)
    grouped = independent.aggregate(labels, pooled)
    oracle = curve["pooled_unrestricted_oracle"]
    require(oracle["origin"] == ORACLE_ORIGIN, "Pooled oracle origin differs")
    ids = oracle["selected_clusters"]
    upper = certify_unrestricted(ids, grouped)
    score_selection(oracle, ids, grouped, pooled)
    require(curve["pooled_unrestricted_selected_clusters"] == curve["parent_unrestricted_selected_clusters"] == ids,
            "Pooled oracle alias IDs differ")
    independent.equal(curve["pooled_unrestricted_f1"], oracle["footprint"]["dice_f1"], "pooled bound")
    independent.equal(curve["greedy_gap_to_pooled_unrestricted_f1"], curve["greedy_gap_to_parent_unrestricted_f1"], "pooled gap alias")
    result = independent.check_path(curve, labels, pooled, oracle)
    core_groups = {core: independent.aggregate(labels, native) for core, native in counts.items()}
    coverage = curve["pooled_coverage"]
    for field, expected in {"n_cells": len(labels), "source_cores": [15, 23],
                            "footprint_pixels": pooled["footprint_pixels"], "valid_pixels": pooled["valid_pixels"],
                            "footprint_positive_pixels": int(pooled["positive"].sum()),
                            "full_region_positive_pixels": pooled["full_positive"],
                            "uncovered_positive_pixels": pooled["full_positive"] - int(pooled["positive"].sum())}.items():
        require(coverage[field] == expected, f"Pooled coverage differs: {field}")
    independent.equal(coverage["positive_region_coverage"], int(pooled["positive"].sum()) / pooled["full_positive"]
                      if pooled["full_positive"] else None, "pooled positive-region coverage")
    for step in curve["steps"]:
        ids = step["selected_clusters"]
        require(set(step["core_components"]) == {"15", "23"}, "Per-core component coverage differs")
        for core in CORES:
            component = step["core_components"][str(core)]
            groups = core_groups[core]
            score_selection(component, ids, groups, counts[core])
            for field, grouped_name in (("roi_cells", "roi_cells"), ("majority_cells", "majority"), ("visible_cells", "visible")):
                total = sum(int(v) for v in groups[grouped_name])
                retained = sum(int(groups[grouped_name][i]) for i in ids)
                numerator = "selected_visible_cells" if field == "visible_cells" else field + "_retained"
                require(component[numerator] == retained and component[field + "_total"] == total,
                        "Per-core fixed cell denominator/numerator differs")
                independent.equal(component[numerator + "_pct"], 100 * retained / total if total else None,
                                  "per-core cell retention percentage")
        for support in ("footprint", "full_region"):
            for name in ("tp", "fp", "fn", "tn", "predicted_pixels", "positive_pixels", "evaluable_pixels"):
                require(sum(c[support][name] for c in step["core_components"].values()) == step[support][name],
                        "Pooled confusion count is not the sum of the same-union core components")
        for name in ("roi_cells_retained", "roi_cells_total", "majority_cells_retained", "majority_cells_total", "selected_visible_cells", "visible_cells_total"):
            require(sum(c[name] for c in step["core_components"].values()) == step[name], "Pooled cell counts do not reconcile")
        per_core_f1 = [step["core_components"][str(core)]["footprint"]["dice_f1"] for core in CORES]
        independent.equal(step["diagnostic_macro_footprint_f1"], sum(per_core_f1) / 2
                          if all(v is not None for v in per_core_f1) else None, "diagnostic macro F1")
        require(step["core_component_counts_reconciled"] is True, "Per-step reconciliation incomplete")
    require(curve["all_core_component_counts_reconciled"] is True, "Path reconciliation incomplete")
    best = curve["steps"][curve["best_step_index"]]
    return {**result, "pooled_unrestricted_exact_fraction": str(upper),
            "best_macro_diagnostic_f1": best["diagnostic_macro_footprint_f1"],
            "best_same_union_core_f1": {str(core): best["core_components"][str(core)]["footprint"]["dice_f1"] for core in CORES}}


def audit(parent, output, paths=None):
    started = time.monotonic()
    paths = paths or independent.current_paths()
    parent, output = Path(parent).resolve(), Path(output).resolve()
    require(parent == (paths.report_root / "analyses/so12_cancer_comparison" / independent.PARENT_EVALUATION).resolve(), "Wrong parent")
    require(output == (paths.scratch_root / "active_runs" / EVALUATION).resolve() and output.is_dir()
            and not (output / "_SUCCESS").exists(), "Audit requires designated active output")
    inventory = independent.verify_parent(parent)
    metric_path, summary_path = output / "metrics/pooled_union.json", output / "pooled_metrics_summary.json"
    metrics, summary = read(metric_path), read(summary_path)
    require(metrics["passed"] is True and metrics["status"] == "complete"
            and summary["passed"] is True and summary["status"] == "complete", "Pooled metrics incomplete")
    require(metrics["n_cells"] == independent.N_CELLS and metrics["test_core_numbers"] == list(CORES), "Metric population differs")
    require(metrics["shared_cluster_selection"] is True and metrics["optimization"] == "pooled_micro_pixel_dice"
            and metrics["per_core_optimized_masks_reused"] is False and metrics["per_core_f1_averaged_for_selection"] is False
            and metrics["all_evaluable_pixels_retained"] is True and metrics["cell_retention_unweighted"] is True
            and metrics["cell_retention_denominators_fixed"] is True, "Pooled metric semantics differ")
    require(metrics["parent"]["manifest_sha256"] == independent.PARENT_SHA
            and metrics["parent"]["evaluation_id"] == independent.PARENT_EVALUATION
            and Path(metrics["parent"]["path"]).resolve() == parent, "Metric parent differs")
    for relative, digest in metrics["input_sha256"].items():
        require(sha(local(parent, relative)) == digest, f"Metric input changed: {relative}")
    for relative, digest in metrics["code_sha256"].items():
        require(sha(local(paths.project_root, relative)) == digest, f"Metric source changed: {relative}")
    for relative, digest in summary["output_sha256"].items():
        require(sha(local(output, relative)) == digest, f"Metric output changed: {relative}")
    cells = pd.read_csv(parent / "cells.csv.gz", usecols=["row_index", "core_number"])
    require(len(cells) == independent.N_CELLS and np.array_equal(cells.row_index, np.arange(len(cells)))
            and cells.core_number.value_counts().to_dict() == {23: 43462, 15: 38145}, "Canonical cell coverage differs")
    registration = read(parent / "registration/registration_manifest.json")
    require(registration["canonical_cells"]["sha256"] == sha(parent / "cells.csv.gz"), "Registration cell identity differs")
    counts = {}
    for item in registration["cores"]:
        core = item["core_number"]
        raster = local(parent, item["raster_path"])
        require(sha(raster) == item["raster_sha256"], "Owner raster changed")
        with np.load(raster, allow_pickle=False) as arrays:
            counts[core] = independent.pixel_counts(arrays["owner"], arrays["roi"], arrays["valid"])
        require(np.all(cells.core_number.to_numpy()[counts[core]["area"] > 0] == core), "Owner raster crosses cores")
    pool_native(counts)
    expected = {(m, r, s) for m in independent.METHODS for r in (.25, .5, 1., 2.) for s in (0, 1)}
    def index(rows):
        result = {(r["method"], r["resolution"], r["seed"]): r for r in rows}
        require(len(rows) == 56 and set(result) == expected, "Incomplete or duplicate partition grid")
        return result
    measured = index(metrics["results"])
    partitions = index(read(parent / "clustering_summary.json")["partitions"])
    checked = []
    for key in sorted(expected):
        item = partitions[key]["labels"]
        path = local(parent, item["path"])
        require(sha(path) == item["sha256"], "Joint partition changed")
        labels = np.load(path, allow_pickle=False)
        require(labels.shape == (independent.N_CELLS,) and labels.dtype.kind in "iu" and np.all(labels >= 0), "Invalid joint labels")
        result = check_pooled_path(measured[key]["pooled"], labels, counts)
        checked.append({"method": key[0], "resolution": key[1], "seed": key[2], **result})
    steps = sum(r["steps"] for r in checked)
    require(metrics["partition_count"] == metrics["pooled_path_count"] == 56 and metrics["step_count"] == steps,
            "Pooled path/step coverage differs")
    independent.verify_parent(parent)
    receipt = {"schema": "so12_cancer_pooled_union_independent_review_v1", "passed": True, "status": "complete",
        "n_cells": independent.N_CELLS, "partition_count": 56, "pooled_paths_checked": len(checked), "steps_checked": steps,
        "binary_score_records_checked": 6 * steps + 2 * len(checked),
        "same_union_core_component_predictions_checked": 2 * steps,
        "native_pooled_confusion_sums_verified": True, "shared_cluster_IDs_verified_on_both_cores": True,
        "exact_removals_and_ties_verified": True, "fixed_cell_denominators_and_operating_points_verified": True,
        "pooled_unrestricted_exact_optimality_certificates_verified": len(checked),
        "paths_attaining_pooled_unrestricted_exact_fraction": sum(r["attains_parent_exact_fraction"] for r in checked),
        "parent_manifest_sha256": independent.PARENT_SHA, "parent_files_verified_before_after": len(inventory["files"]),
        "metrics_sha256": sha(metric_path), "metrics_summary_sha256": sha(summary_path),
        "audit_source_sha256": sha(Path(__file__)), "independent_helper_sha256": sha(Path(independent.__file__)),
        "primary": [r for r in checked if (r["resolution"], r["seed"]) == (1., 0)],
        "elapsed_seconds": time.monotonic() - started, "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "One shared union is scored by summed native-pixel confusion counts. This is neither independent per-core selection nor mean per-core F1. Native-pixel weighting is not physical-area normalization. Results remain annotation-selected descriptive agreement."}
    destination = output / "independent_metrics_review.json"
    require(not destination.exists(), "Refusing to overwrite independent review")
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
    print(json.dumps({key: receipt[key] for key in ("passed", "pooled_paths_checked", "steps_checked", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

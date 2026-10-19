"""Backward cluster deletion with fixed pixel and unweighted cell denominators.

Every path is an annotation-selected descriptive trajectory, not a guarantee of
the optimal subset at each cardinality. No fitting or reclustering occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from scripts.analysis import so12_cancer_limited_metrics as limited
except ModuleNotFoundError:
    import so12_cancer_limited_metrics as limited

base = limited.base
_require, _sha, _dump = base._require, base._sha, base._dump
METHODS, RESOLUTIONS, SEEDS = limited.METHODS, limited.RESOLUTIONS, limited.SEEDS
EXPECTED_CELLS = 81607
PARENT_MANIFEST_SHA256 = limited.PARENT_MANIFEST_SHA256


def _fraction(step):
    score = step["footprint"]
    return 2 * int(score["tp"]), max(1, int(score["predicted_pixels"]) + int(score["positive_pixels"]))


def _compare(a, b):
    """Sign of exact footprint-Dice difference, using unbounded integers."""
    an, ad = _fraction(a)
    bn, bd = _fraction(b)
    delta = an * bd - bn * ad
    return (delta > 0) - (delta < 0)


def best_step_index(steps, minimum_retention=None):
    eligible = [i for i, step in enumerate(steps) if minimum_retention is None or
        (step["roi_cells_total"] > 0 and 100 * step["roi_cells_retained"] >= minimum_retention * step["roi_cells_total"])]
    best = None
    for i in eligible:
        if best is None or _compare(steps[i], steps[best]) > 0 or (
                _compare(steps[i], steps[best]) == 0 and
                (steps[i]["k"], steps[i]["selected_clusters"]) < (steps[best]["k"], steps[best]["selected_clusters"])):
            best = i
    return best


def backward_path(labels, counts, *, parent_oracle=None):
    """Greedily remove one cluster, retaining all steps through the empty union."""
    _require(counts.get("status") == "complete", "Verified native-pixel counts are required")
    labels = np.asarray(labels)
    area, positive = limited.cluster_counts(labels, counts)
    visible = counts["area_per_cell"] > 0
    roi_cells = counts["positive_per_cell"] > 0
    majority = visible & (counts["positive_per_cell"] >= (counts["area_per_cell"] + 1) // 2)
    grouped_visible = np.bincount(labels[visible], minlength=len(area))
    grouped_roi = np.bincount(labels[roi_cells], minlength=len(area))
    grouped_majority = np.bincount(labels[majority], minlength=len(area))
    initial = tuple(int(i) for i in np.flatnonzero(positive > 0))
    denominators = {"initial_overlap_clusters":len(initial), "roi_cells_total":int(roi_cells.sum()),
                    "majority_cells_total":int(majority.sum()), "visible_cells_total":int(visible.sum())}
    percent = lambda value,total: 100 * value / total if total else None

    def score(ids, removed=None):
        scored = limited.score_selection(ids, area, positive, counts)
        roi_count = sum(int(grouped_roi[i]) for i in ids)
        majority_count = sum(int(grouped_majority[i]) for i in ids)
        visible_count = sum(int(grouped_visible[i]) for i in ids)
        return {**scored, **denominators, "k":len(ids), "removed_cluster":removed,
            "cluster_retained_pct":percent(len(ids),len(initial)),
            "roi_cells_retained":roi_count, "roi_cells_retained_pct":percent(roi_count,denominators["roi_cells_total"]),
            "majority_cells_retained":majority_count,
            "majority_cells_retained_pct":percent(majority_count,denominators["majority_cells_total"]),
            "selected_visible_cells":visible_count,
            "selected_visible_cells_pct":percent(visible_count,denominators["visible_cells_total"])}

    current, steps = initial, [score(initial)]
    while current:
        best = None
        for removed in current:
            candidate = score(tuple(i for i in current if i != removed), removed)
            if best is None or _compare(candidate,best) > 0 or (
                    _compare(candidate,best) == 0 and candidate["selected_clusters"] < best["selected_clusters"]):
                best = candidate
        steps.append(best)
        current = tuple(best["selected_clusters"])
    for index,step in enumerate(steps):
        step["step_index"] = index
        step["pareto_nondominated"] = not any(
            other["roi_cells_retained"] >= step["roi_cells_retained"] and _compare(other,step) >= 0 and
            (other["roi_cells_retained"] > step["roi_cells_retained"] or _compare(other,step) > 0)
            for other in steps)
    best = best_step_index(steps)
    if parent_oracle is None:
        selected = base.optimal_union(area,positive)["selected_clusters"]
        parent_oracle = limited.score_selection(selected,area,positive,counts)
        parent_origin = "synthetic_exact_unrestricted_oracle"
    else:
        replay = limited.score_selection(parent_oracle["selected_clusters"],area,positive,counts)
        limited._verify_parent_oracle(replay,parent_oracle)
        parent_origin = "frozen_parent_exact_unrestricted_oracle"
    parent_step = {"footprint":parent_oracle["footprint"]}
    _require(_compare(steps[best],parent_step) <= 0, "Backward path exceeds the exact unrestricted maximum")
    parent_f1, best_f1 = parent_oracle["footprint"]["dice_f1"], steps[best]["footprint"]["dice_f1"]
    return {"status":"complete", "initial_overlap_cluster_ids":list(initial), **denominators,
        "steps":steps, "best_step_index":best,
        "best_retention90_step_index":best_step_index(steps,90),
        "best_retention95_step_index":best_step_index(steps,95),
        "parent_unrestricted_f1":parent_f1,
        "parent_unrestricted_selected_clusters":parent_oracle["selected_clusters"],
        "parent_unrestricted_origin":parent_origin,
        "greedy_gap_to_parent_unrestricted_f1":parent_f1-best_f1 if parent_f1 is not None and best_f1 is not None else None,
        "removal_tie_rule":"exact_rational_footprint_dice_then_lexicographic_retained_ids",
        "path_maximum_tie_rule":"exact_rational_footprint_dice_then_fewest_clusters_then_ids",
        "roi_cell_definition":"at_least_one_valid_owned_pixel_inside_annotation",
        "majority_cell_definition":"visible_cell_with_at_least_half_valid_owned_pixels_inside_annotation",
        "pareto_objectives":"maximize_footprint_F1_and_unweighted_roi_cell_retention; equal_points_nondominated",
        "optimal_fixed_cardinality_subset_claimed":False}


def _flat_step(identity,core,step):
    flat = {**identity,"core":int(core),**{k:v for k,v in step.items() if k not in ("footprint","full_region","selected_clusters")},
            "selected_clusters":json.dumps(step["selected_clusters"])}
    for support in ("footprint","full_region"):
        flat.update({f"{support}_{k}":v for k,v in step[support].items()})
    return flat


def run(parent,output):
    from spatial_benchmark.paths import current_paths
    started = time.monotonic()
    parent,output = Path(parent).resolve(),Path(output).resolve()
    active = (current_paths().scratch_root/"active_runs").resolve()
    _require(output.is_relative_to(active) and output != active and not output.is_relative_to(parent),
             "Writes require a separate active evaluation directory")
    manifest = limited.verify_parent(parent)
    _require(_sha(parent/"manifest.json") == PARENT_MANIFEST_SHA256, "Unexpected parent identity")
    read = lambda name:json.loads((parent/name).read_text())
    clustering,registration,old = read("clustering_summary.json"),read("registration/registration_manifest.json"),read("metrics/pixel_agreement.json")
    _require(clustering["passed"] and clustering["all_methods_complete"] and old["passed"], "Parent stages incomplete")
    _require(_sha(Path(base.__file__)) == old["code_sha256"], "Parent metric source changed")
    cells = pd.read_csv(parent/"cells.csv.gz")
    cells_sha = _sha(parent/"cells.csv.gz")
    _require(len(cells) == EXPECTED_CELLS == clustering["n_cells"] and np.array_equal(cells.row_index,np.arange(len(cells)))
             and set(cells.core_number) == {15,23}, "Canonical test rows changed")
    _require(registration["canonical_cells"]["sha256"] == cells_sha == old["input_sha256"]["cells.csv.gz"],
             "Canonical cell binding differs between parent stages")
    expected = {(m,r,s) for m in METHODS for r in RESOLUTIONS for s in SEEDS}
    key = lambda row:(row["method"],float(row["resolution"]),int(row["seed"]))
    parts = clustering["partitions"]
    _require(len(parts) == len(expected) and {key(r) for r in parts} == expected, "Incomplete or duplicated partition grid")
    old_by_key = {key(row):row["evaluation"] for row in old["results"]}
    _require(len(old["results"]) == len(expected) and set(old_by_key) == expected, "Parent score grid differs")
    counts,_,raster_sources = base._registration_counts(parent,registration,len(cells))
    _require(all(c["status"] == "complete" for c in counts.values()), "Verified registration required")
    for core,c in counts.items():
        _require(np.all(cells.core_number.to_numpy()[c["area_per_cell"] > 0] == core), "Raster owner crosses core identity")
    inputs = {name:_sha(parent/name) for name in ("cells.csv.gz","clustering_summary.json","registration/registration_manifest.json","metrics/pixel_agreement.json")}
    inputs.update({r["path"]:r["sha256"] for r in raster_sources.values()})
    results,all_steps,best_rows = [],[],[]
    for part in sorted(parts,key=key):
        record = part["labels"]
        _require(record["relative"] is True, "Expected parent-local label arrays")
        path = base._local_path(parent,record["path"])
        _require(_sha(path) == record["sha256"], "Parent label bytes changed")
        labels = np.load(path,allow_pickle=False)
        _require(labels.shape == (len(cells),), "Label row coverage differs")
        identity = {k:part[k] for k in ("method","resolution","seed")}
        row = {**identity,"cores":{}}
        for core in (15,23):
            curve = backward_path(labels,counts[core],parent_oracle=old_by_key[key(part)]["cores"][str(core)]["oracle"])
            row["cores"][str(core)] = curve
            all_steps.extend(_flat_step(identity,core,s) for s in curve["steps"])
            for operating_point,field in (("best","best_step_index"),("retention90","best_retention90_step_index"),("retention95","best_retention95_step_index")):
                index = curve[field]
                best_rows.append({**identity,"core":core,"operating_point":operating_point,
                    "parent_unrestricted_f1":curve["parent_unrestricted_f1"],
                    "greedy_gap_to_parent_unrestricted_f1":curve["greedy_gap_to_parent_unrestricted_f1"],
                    **(_flat_step(identity,core,curve["steps"][index]) if index is not None else {"status":"unavailable"})})
        results.append(row);inputs[record["path"]] = record["sha256"]
    limited.verify_parent(parent)
    folder = output/"metrics";folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_steps).to_csv(folder/"backward_steps.csv",index=False)
    frame = pd.DataFrame(best_rows)
    frame.to_csv(folder/"backward_best_summary.csv",index=False)
    frame[(frame.resolution == 1.) & (frame.seed == 0)].to_csv(folder/"backward_primary_summary.csv",index=False)
    summary = {"schema":"so12_cancer_backward_union_v1","status":"complete","passed":True,
        "n_cells":len(cells),"test_core_numbers":[15,23],"methods":list(METHODS),"resolutions":list(RESOLUTIONS),"seeds":list(SEEDS),
        "primary_resolution":1.,"primary_seed":0,"all_evaluable_pixels_retained":True,
        "cell_retention_unweighted":True,"cell_retention_denominators_fixed":True,
        "selection_support":"segmented_cell_footprint","parent_inventory_verified_before_after":True,
        "parent":{"path":str(parent),"manifest_sha256":PARENT_MANIFEST_SHA256,"evaluation_id":manifest["evaluation_id"]},
        "input_sha256":inputs,"code_sha256":{
            "scripts/analysis/so12_cancer_backward_metrics.py":_sha(Path(__file__)),
            "scripts/analysis/so12_cancer_limited_metrics.py":_sha(Path(limited.__file__)),
            "scripts/analysis/so12_cancer_metrics.py":_sha(Path(base.__file__))},
        "partition_count":len(results),"core_path_count":sum(len(r["cores"]) for r in results),"step_count":len(all_steps),
        "elapsed_seconds":time.monotonic()-started,"results":results,
        "limitations":"Annotation-selected backward deletion is exploratory and may miss optimal fixed-cardinality subsets. Unweighted retained-cell percentages differ from pixel recall. Two cores, approximate section registration and uncertain annotation exhaustiveness preclude independent accuracy or population superiority claims."}
    _dump(folder/"backward_union.json",summary)
    receipt = {k:v for k,v in summary.items() if k != "results"}
    receipt["output_sha256"] = {str(p.relative_to(output)):_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    _dump(output/"backward_metrics_summary.json",receipt)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.parent,args.output)

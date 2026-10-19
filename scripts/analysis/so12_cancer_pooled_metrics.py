"""One shared cluster union on pooled native pixels from the two test cores.

The unchanged backward engine optimizes micro Dice after adding disjoint counts.
Per-core scores diagnose that same prediction; they never select separate masks.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from scripts.analysis import so12_cancer_backward_metrics as backward
except ModuleNotFoundError:
    import so12_cancer_backward_metrics as backward

limited, base = backward.limited, backward.base
_require, _sha, _dump = base._require, base._sha, base._dump
METHODS, RESOLUTIONS, SEEDS = backward.METHODS, backward.RESOLUTIONS, backward.SEEDS
EXPECTED_CELLS = 81607
PARENT_MANIFEST_SHA256 = backward.PARENT_MANIFEST_SHA256
ORACLE_ORIGIN = "recomputed_exact_unrestricted_oracle_on_pooled_core_counts"


def pool_counts(counts_by_core):
    """Pool disjoint global cell rows and sum their native-raster supports."""
    _require(set(counts_by_core) == {15,23}, "Exactly cores15 and23 are required")
    arrays = []
    for core in (15,23):
        counts = counts_by_core[core]
        _require(counts.get("status") == "complete", "Verified core counts required")
        area, positive = np.asarray(counts["area_per_cell"]), np.asarray(counts["positive_per_cell"])
        _require(area.ndim == 1 and area.shape == positive.shape and len(area) == counts["n_cells"],
                 "Canonical global count rows must align")
        _require(np.issubdtype(area.dtype,np.integer) and np.issubdtype(positive.dtype,np.integer)
                 and np.all(area >= positive) and np.all(positive >= 0), "Invalid integer pixel support counts")
        for key in ("valid_pixels","footprint_pixels","footprint_positive_pixels","full_region_positive_pixels","uncovered_positive_pixels"):
            _require(isinstance(counts[key],(int,np.integer)) and not isinstance(counts[key],bool), "Support totals must be integers")
        a, p = sum(map(int,area)), sum(map(int,positive))
        _require(a == counts["footprint_pixels"] and p == counts["footprint_positive_pixels"]
                 and 0 < a <= counts["valid_pixels"], "Footprint support totals disagree")
        _require(p <= counts["full_region_positive_pixels"] <= counts["valid_pixels"]
                 and counts["uncovered_positive_pixels"] == counts["full_region_positive_pixels"]-p,
                 "Full-region support totals disagree")
        arrays.append((area,positive))
    _require(arrays[0][0].shape == arrays[1][0].shape, "Canonical global count rows must align")
    _require(not np.any((arrays[0][0] > 0) & (arrays[1][0] > 0)), "Core cell ownership must be disjoint")
    area = arrays[0][0].astype(np.int64) + arrays[1][0].astype(np.int64)
    positive = arrays[0][1].astype(np.int64) + arrays[1][1].astype(np.int64)
    pooled = {"status":"complete","registration_status":"verified","n_cells":len(area),
        "source_cores":[15,23],"area_per_cell":area,"positive_per_cell":positive,
        "support":"pooled_disjoint_core_single_owner_cell_rasters"}
    for key in ("valid_pixels","footprint_pixels","footprint_positive_pixels","full_region_positive_pixels","uncovered_positive_pixels"):
        pooled[key] = sum(int(c[key]) for c in counts_by_core.values())
    covered, full = pooled["footprint_positive_pixels"], pooled["full_region_positive_pixels"]
    pooled["positive_region_coverage"] = covered/full if full else None
    pooled["full_region_pixel_coverage_dice_ceiling"] = 2*covered/(full+covered) if full else None
    return pooled


def exact_unrestricted_ids(area,positive):
    """Exact rational purity-prefix oracle with fewest-cluster/ID ties."""
    area,positive = np.asarray(area),np.asarray(positive)
    _require(area.ndim == 1 and area.shape == positive.shape and np.issubdtype(area.dtype,np.integer)
             and np.issubdtype(positive.dtype,np.integer) and np.all(area >= positive) and np.all(positive >= 0),
             "Invalid cluster counts for pooled oracle")
    areas,hits = list(map(int,area)),list(map(int,positive))
    order = sorted((i for i,a in enumerate(areas) if a),key=lambda i:(-Fraction(hits[i],areas[i]),i))
    total = sum(hits)
    best_ids,best_value = (),Fraction(0)
    tp,predicted = 0,0
    for size,i in enumerate(order,1):
        tp += hits[i]; predicted += areas[i]
        value = Fraction(2*tp,total+predicted)
        ids = tuple(sorted(order[:size]))
        if value > best_value or (value == best_value and (len(ids),ids) < (len(best_ids),best_ids)):
            best_ids,best_value = ids,value
    return list(best_ids)


def pooled_path(labels,counts_by_core):
    counts = pool_counts(counts_by_core)
    labels = np.asarray(labels)
    area,positive = limited.cluster_counts(labels,counts)
    oracle = limited.score_selection(exact_unrestricted_ids(area,positive),area,positive,counts)
    path = backward.backward_path(labels,counts,parent_oracle=oracle)
    path.update({"shared_cluster_selection":True,"optimization":"pooled_micro_pixel_dice",
        "parent_unrestricted_origin":ORACLE_ORIGIN,
        "pooled_unrestricted_f1":path["parent_unrestricted_f1"],
        "pooled_unrestricted_selected_clusters":path["parent_unrestricted_selected_clusters"],
        "greedy_gap_to_pooled_unrestricted_f1":path["greedy_gap_to_parent_unrestricted_f1"],
        "pooled_unrestricted_oracle":{**oracle,"origin":ORACLE_ORIGIN},
        "pooled_coverage":{k:v for k,v in counts.items() if not isinstance(v,np.ndarray)},
        "per_core_optimized_masks_reused":False,"per_core_f1_averaged_for_selection":False})
    grouped = {}
    for core,c in counts_by_core.items():
        a,p = limited.cluster_counts(labels,c)
        masks = {"roi_cells":c["positive_per_cell"] > 0,
                 "majority_cells":(c["area_per_cell"] > 0) & (c["positive_per_cell"] >= (c["area_per_cell"]+1)//2),
                 "visible_cells":c["area_per_cell"] > 0}
        grouped[core] = (a,p,{name:np.bincount(labels[mask],minlength=len(a)) for name,mask in masks.items()})
    for step in path["steps"]:
        selected = step["selected_clusters"]
        components = {}
        for core,c in sorted(counts_by_core.items()):
            a,p,cell_groups = grouped[core]
            component = limited.score_selection(selected,a,p,c)
            for name,groups in cell_groups.items():
                retained,total = sum(int(groups[i]) for i in selected),sum(map(int,groups))
                numerator = "selected_visible_cells" if name == "visible_cells" else name+"_retained"
                component[numerator] = retained
                component[name+"_total"] = total
                component[numerator+"_pct"] = 100*retained/total if total else None
            components[str(core)] = component
        for support in ("footprint","full_region"):
            for key in ("tp","fp","fn","tn","predicted_pixels","positive_pixels","evaluable_pixels"):
                _require(sum(c[support][key] for c in components.values()) == step[support][key],
                         "Pooled confusion counts do not equal core components")
        for key in ("roi_cells_retained","roi_cells_total","majority_cells_retained","majority_cells_total",
                    "selected_visible_cells","visible_cells_total"):
            _require(sum(c[key] for c in components.values()) == step[key], "Pooled cell counts do not equal disjoint-core totals")
        values = [c["footprint"]["dice_f1"] for c in components.values()]
        step["core_components"] = components
        step["diagnostic_macro_footprint_f1"] = sum(values)/2 if all(v is not None for v in values) else None
        step["core_component_counts_reconciled"] = True
    path["all_core_component_counts_reconciled"] = True
    return path


def _flat(identity,step):
    scalar = {k:v for k,v in step.items() if k != "core_components"}
    row = backward._flat_step(identity,0,scalar)
    row.pop("core")
    row["core_scope"] = "15+23_pooled"
    for core,component in step["core_components"].items():
        for support in ("footprint","full_region"):
            row.update({f"core_{core}_{support}_{key}":value for key,value in component[support].items()})
        for key in ("roi_cells_retained","roi_cells_total","roi_cells_retained_pct","majority_cells_retained",
                    "majority_cells_total","majority_cells_retained_pct","selected_visible_cells","visible_cells_total","selected_visible_cells_pct"):
            row[f"core_{core}_{key}"] = component[key]
    return row


def run(parent,output):
    from spatial_benchmark.paths import current_paths
    started = time.monotonic()
    parent,output = Path(parent).resolve(),Path(output).resolve()
    active = (current_paths().scratch_root/"active_runs").resolve()
    _require(output.is_relative_to(active) and output != active and not output.is_relative_to(parent),
             "Writes require a separate active evaluation directory")
    manifest = limited.verify_parent(parent)
    _require(_sha(parent/"manifest.json") == PARENT_MANIFEST_SHA256,"Unexpected immutable parent")
    read = lambda name:json.loads((parent/name).read_text())
    clustering,registration,old = read("clustering_summary.json"),read("registration/registration_manifest.json"),read("metrics/pixel_agreement.json")
    _require(clustering["passed"] and clustering["all_methods_complete"] and old["passed"],"Parent stages incomplete")
    _require(_sha(Path(base.__file__)) == old["code_sha256"],"Parent metric source changed")
    cells = pd.read_csv(parent/"cells.csv.gz")
    cells_sha = _sha(parent/"cells.csv.gz")
    _require(len(cells) == EXPECTED_CELLS == clustering["n_cells"] and np.array_equal(cells.row_index,np.arange(len(cells)))
             and set(cells.core_number) == {15,23},"Canonical test rows changed")
    _require(registration["canonical_cells"]["sha256"] == cells_sha == old["input_sha256"]["cells.csv.gz"],"Canonical cell bindings differ")
    expected = {(m,r,s) for m in METHODS for r in RESOLUTIONS for s in SEEDS}
    key = lambda row:(row["method"],float(row["resolution"]),int(row["seed"]))
    parts = clustering["partitions"]
    _require(len(parts) == len(expected) and {key(r) for r in parts} == expected,"Incomplete or duplicated grid")
    counts,_,rasters = base._registration_counts(parent,registration,len(cells))
    pool_counts(counts)
    for core,c in counts.items():
        _require(np.all(cells.core_number.to_numpy()[c["area_per_cell"] > 0] == core),"Raster owner crosses core identity")
    inputs = {name:_sha(parent/name) for name in ("cells.csv.gz","clustering_summary.json","registration/registration_manifest.json","metrics/pixel_agreement.json")}
    inputs.update({r["path"]:r["sha256"] for r in rasters.values()})
    results,all_steps,best_rows = [],[],[]
    for part in sorted(parts,key=key):
        record = part["labels"]
        _require(record["relative"] is True,"Expected parent-local labels")
        path = base._local_path(parent,record["path"])
        _require(_sha(path) == record["sha256"],"Parent label bytes changed")
        labels = np.load(path,allow_pickle=False)
        _require(labels.shape == (len(cells),),"Canonical label rows differ")
        identity = {k:part[k] for k in ("method","resolution","seed")}
        curve = pooled_path(labels,counts)
        results.append({**identity,"pooled":curve})
        all_steps.extend(_flat(identity,step) for step in curve["steps"])
        for operating_point,field in (("best","best_step_index"),("retention90","best_retention90_step_index"),("retention95","best_retention95_step_index")):
            index = curve[field]
            best_rows.append({**identity,"operating_point":operating_point,
                "pooled_unrestricted_f1":curve["pooled_unrestricted_f1"],
                "greedy_gap_to_pooled_unrestricted_f1":curve["greedy_gap_to_pooled_unrestricted_f1"],
                **(_flat(identity,curve["steps"][index]) if index is not None else {"status":"unavailable"})})
        inputs[record["path"]] = record["sha256"]
    limited.verify_parent(parent)
    folder = output/"metrics";folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_steps).to_csv(folder/"pooled_steps.csv",index=False)
    frame = pd.DataFrame(best_rows)
    frame.to_csv(folder/"pooled_best_summary.csv",index=False)
    frame[(frame.resolution == 1.) & (frame.seed == 0)].to_csv(folder/"pooled_primary_summary.csv",index=False)
    summary = {"schema":"so12_cancer_pooled_union_v1","status":"complete","passed":True,"n_cells":len(cells),
        "test_core_numbers":[15,23],"methods":list(METHODS),"resolutions":list(RESOLUTIONS),"seeds":list(SEEDS),
        "primary_resolution":1.,"primary_seed":0,"shared_cluster_selection":True,"optimization":"pooled_micro_pixel_dice",
        "per_core_optimized_masks_reused":False,"per_core_f1_averaged_for_selection":False,
        "all_evaluable_pixels_retained":True,"cell_retention_unweighted":True,"cell_retention_denominators_fixed":True,
        "all_core_component_counts_reconciled":True,"parent_inventory_verified_before_after":True,
        "parent":{"path":str(parent),"manifest_sha256":PARENT_MANIFEST_SHA256,"evaluation_id":manifest["evaluation_id"]},
        "input_sha256":inputs,"code_sha256":{
            "scripts/analysis/so12_cancer_pooled_metrics.py":_sha(Path(__file__)),
            "scripts/analysis/so12_cancer_backward_metrics.py":_sha(Path(backward.__file__)),
            "scripts/analysis/so12_cancer_limited_metrics.py":_sha(Path(limited.__file__)),
            "scripts/analysis/so12_cancer_metrics.py":_sha(Path(base.__file__))},
        "partition_count":len(results),"pooled_path_count":len(results),"step_count":len(all_steps),
        "elapsed_seconds":time.monotonic()-started,"results":results,
        "limitations":"Annotation-selected shared union on two cores is descriptive, not cross-core transfer or independent accuracy. Micro F1 weights native evaluated pixels, including different pixel-to-micrometre calibrations; it is neither equal-core averaging nor physical-area normalization. Approximate registration, incomplete segmentation and annotation exhaustiveness limit interpretation."}
    _dump(folder/"pooled_union.json",summary)
    receipt = {k:v for k,v in summary.items() if k != "results"}
    receipt["output_sha256"] = {str(p.relative_to(output)):_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    _dump(output/"pooled_metrics_summary.json",receipt)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.parent,args.output)

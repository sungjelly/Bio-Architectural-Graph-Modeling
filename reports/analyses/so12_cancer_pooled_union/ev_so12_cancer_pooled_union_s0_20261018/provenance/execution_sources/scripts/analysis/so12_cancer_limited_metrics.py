"""Exact one/two-cluster descriptive overlap on immutable parent partitions.

Selection uses source-core footprint Dice. Every evaluable pixel is retained;
other cells remain negative predictions, and full-region gaps remain false
negatives. These annotation-selected optima are exploratory descriptive oracles.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from scripts.analysis import so12_cancer_metrics as base
except ModuleNotFoundError:
    import so12_cancer_metrics as base


PARENT_MANIFEST_SHA256 = "e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0"
EXPECTED_CELLS = 81607
METHODS = ("h0", "h1", "h2", "h3", "h4", "pca", "scvi")
RESOLUTIONS = (.25, .5, 1., 2.)
SEEDS = (0, 1)
SELECTIONS = ("best_single", "best_pair", "best_at_most_two")
_require, _sha, _dump = base._require, base._sha, base._dump


def exact_limited_unions(area, positive):
    """Enumerate eligible singletons/pairs; compare integer fractions exactly.

    At-most-two includes the empty prediction. Undefined empty-reference scores
    are tied at zero for selection, while reported Dice remains null. An exact
    pair is unavailable when fewer than two source-visible clusters exist.
    """
    area, positive = np.asarray(area), np.asarray(positive)
    _require(area.ndim == 1 and area.shape == positive.shape, "Cluster counts must align")
    _require(np.issubdtype(area.dtype, np.integer) and np.issubdtype(positive.dtype, np.integer),
             "Cluster pixel counts must be integers")
    _require(np.all(area >= positive) and np.all(positive >= 0), "Invalid cluster pixel counts")
    areas, hits = [int(v) for v in area], [int(v) for v in positive]
    total = sum(hits)
    eligible = [i for i, value in enumerate(areas) if value > 0]

    def value(ids):
        return 2 * sum(hits[i] for i in ids), total + sum(areas[i] for i in ids)

    def choose(candidates):
        best = None
        for ids in candidates:
            ids = tuple(ids)
            if best is None:
                best = ids
                continue
            num, den = value(ids)
            old_num, old_den = value(best)
            # A zero denominator denotes an empty reference and prediction.
            difference = num * max(1, old_den) - old_num * max(1, den)
            if difference > 0 or (difference == 0 and (len(ids), ids) < (len(best), best)):
                best = ids
        return None if best is None else list(best)

    single = choose((i,) for i in eligible)
    pair = choose(itertools.combinations(eligible, 2))
    at_most = choose(ids for ids in ([], single, pair) if ids is not None)
    return {"best_single": single, "best_pair": pair, "best_at_most_two": at_most,
            "eligible_cluster_ids": eligible, "tie_rule": "exact_rational_dice_then_fewer_clusters_then_ascending_ids",
            "candidate_counts": {"best_single": len(eligible), "best_pair": len(eligible) * (len(eligible)-1)//2,
                                 "best_at_most_two": 1 + len(eligible) * (len(eligible)+1)//2}}


def cluster_counts(labels, counts):
    labels = np.asarray(labels)
    _require(labels.ndim == 1 and len(labels) > 0 and np.issubdtype(labels.dtype, np.integer)
             and np.all(labels >= 0), "Labels must be nonnegative integer cell rows")
    a, p = counts["area_per_cell"], counts["positive_per_cell"]
    _require(labels.shape == a.shape == p.shape, "Labels and canonical pixel counts differ")
    area, positive = np.zeros(int(labels.max())+1, np.int64), np.zeros(int(labels.max())+1, np.int64)
    np.add.at(area, labels, a)
    np.add.at(positive, labels, p)
    return area, positive


def score_selection(selected, area, positive, counts):
    if selected is None:
        return {"status": "unavailable", "selected_clusters": [], "footprint": None, "full_region": None,
                "reason": "Fewer source-visible clusters than the exact requested cardinality"}
    selected = sorted(int(v) for v in selected)
    scores = base._score_union(selected, area, positive, counts)
    for support, total in (("footprint", counts["footprint_pixels"]), ("full_region", counts["valid_pixels"])):
        row = scores[support]
        row["evaluable_pixels"] = int(total)
        row["tn"] = int(total) - row["tp"] - row["fp"] - row["fn"]
        _require(row["tn"] >= 0, "Confusion counts exceed evaluable support")
    return {"status": "complete", "selected_clusters": selected, **scores}


def _verify_parent_oracle(scored, parent):
    for support in ("footprint", "full_region"):
        for key in ("tp", "fp", "fn", "predicted_pixels", "positive_pixels", "dice_f1", "iou", "precision", "recall"):
            _require(scored[support][key] == parent[support][key], "Parent unrestricted score no longer replays")


def evaluate_partition(labels, counts_by_core, *, parent_cores=None, null_repeats=0, null_seed=0):
    """Exact source selection, fixed cross-core transfer and visible-cell nulls."""
    _require(set(counts_by_core) == {15, 23}, "Both canonical test cores are required")
    _require(isinstance(null_repeats, int) and null_repeats >= 0, "Invalid null repetition count")
    labels = np.asarray(labels)
    complete = [c for c in counts_by_core.values() if c.get("status") == "complete"]
    if len(complete) == 2:
        _require(not np.any((complete[0]["area_per_cell"] > 0) & (complete[1]["area_per_cell"] > 0)),
                 "A cell row cannot belong to both core rasters")
    result = {"cores": {}, "transfer": {}, "selection_support": "segmented_cell_footprint"}
    grouped, choices = {}, {}
    for core, counts in sorted(counts_by_core.items()):
        if counts.get("status") != "complete":
            result["cores"][str(core)] = {"status": "withheld", "reason": counts.get("reason", "Registration unverified")}
            continue
        area, positive = cluster_counts(labels, counts)
        grouped[core] = area, positive
        chosen = choices[core] = exact_limited_unions(area, positive)
        row = {"status": "complete", "selection_details": {k: chosen[k] for k in ("eligible_cluster_ids", "tie_rule", "candidate_counts")},
               **{kind: score_selection(chosen[kind], area, positive, counts) for kind in SELECTIONS},
               "all_positive": score_selection(np.flatnonzero(area), area, positive, counts),
               "empty": score_selection([], area, positive, counts),
               "coverage": {k:v for k,v in counts.items() if not isinstance(v, np.ndarray)}}
        parent = parent_cores[str(core)]["oracle"] if parent_cores is not None else None
        unrestricted = parent["selected_clusters"] if parent else base.optimal_union(area, positive)["selected_clusters"]
        row["unrestricted"] = score_selection(unrestricted, area, positive, counts)
        row["unrestricted"]["selection"] = "frozen_parent_annotation_selected_unrestricted_oracle" if parent else "synthetic_or_null_unrestricted_oracle"
        if parent is not None:
            _verify_parent_oracle(row["unrestricted"], parent)
        values = [row[k]["footprint"]["dice_f1"] for k in ("best_single", "best_at_most_two", "unrestricted")]
        if all(v is not None for v in values):
            _require(values[0] <= values[1] + 1e-14 and values[1] <= values[2] + 1e-14,
                     "Limited selection violates the unrestricted bound")
        result["cores"][str(core)] = row
    for source, target in ((15, 23), (23, 15)):
        key = f"{source}_to_{target}"
        if source not in grouped or target not in grouped:
            result["transfer"][key] = {"status": "withheld", "reason": "Both registrations must be verified"}
            continue
        area, positive = grouped[target]
        result["transfer"][key] = {"status": "complete", "source_core": source, "target_core": target,
            "selection_uses_target_annotation": False,
            "unseen_target_pixel_fraction": float(area[grouped[source][0] == 0].sum()/max(1, area.sum())),
            **{kind: score_selection(choices[source][kind], area, positive, counts_by_core[target]) for kind in SELECTIONS}}
    for scope, field in (("oracle", "cores"), ("transfer", "transfer")):
        result[f"equal_core_{scope}_dice"] = {}
        for kind in SELECTIONS:
            rows = list(result[field].values())
            values = [r[kind]["footprint"]["dice_f1"] for r in rows if r["status"] == "complete" and r[kind]["status"] == "complete"]
            result[f"equal_core_{scope}_dice"][kind] = float(np.mean(values)) if len(values) == 2 and all(v is not None for v in values) else None
    if null_repeats:
        rng, nulls = np.random.default_rng(null_seed), []
        for repeat in range(null_repeats):
            shuffled = labels.copy()
            for core in sorted(grouped):
                visible = np.flatnonzero(counts_by_core[core]["area_per_cell"] > 0)
                shuffled[visible] = rng.permutation(shuffled[visible])
            null = evaluate_partition(shuffled, counts_by_core)
            nulls.append({"repeat": repeat, **{k:null[k] for k in ("cores", "transfer", "equal_core_oracle_dice", "equal_core_transfer_dice")}})
        result["shuffled_label_control"] = {"seed": int(null_seed), "repeats": null_repeats,
            "scope": "visible_cell_labels_permuted_independently_within_each_core", "oracle_reselected_each_repeat": True,
            "population_p_value_claimed": False, "results": nulls}
    return result


def fixed_union_sensitivity(labels, evaluation, sensitivity):
    result = {}
    for core, scenarios in sensitivity.items():
        result[str(core)] = {}
        if evaluation["cores"][str(core)]["status"] != "complete":
            continue
        source = 23 if core == 15 else 15
        for name, counts in scenarios.items():
            if counts["status"] != "complete":
                result[str(core)][name] = counts
                continue
            area, positive = cluster_counts(labels, counts)
            row = {"status": "complete", "selection_frozen_from_nominal_masks": True}
            for scope, nominal in (("oracle", evaluation["cores"][str(core)]),
                                   ("transfer", evaluation["transfer"][f"{source}_to_{core}"])):
                row[scope] = {}
                if nominal["status"] != "complete":
                    row[scope] = {"status": "withheld"}
                    continue
                for kind in SELECTIONS:
                    selection = nominal[kind]
                    row[scope][kind] = score_selection(selection["selected_clusters"] if selection["status"] == "complete" else None,
                                                        area, positive, counts)
            result[str(core)][name] = row
    return result


def verify_parent(parent):
    parent = Path(parent)
    manifest_path = parent / "manifest.json"
    digest = _sha(manifest_path)
    _require(digest == PARENT_MANIFEST_SHA256, "Unexpected immutable parent manifest")
    _require(json.loads((parent/"_SUCCESS").read_text())["manifest_sha256"] == digest, "Parent success marker differs")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["passed"] is True and manifest["status"] == "complete", "Parent is not finalized")
    files = manifest["files"]
    actual = {str(f.relative_to(parent)) for f in parent.rglob("*") if f.is_file()}
    _require(actual == set(files) | {"manifest.json", "_SUCCESS"}, "Parent inventory changed")
    _require(not any(f.is_symlink() for f in parent.rglob("*")), "Parent inventory contains symlinks")
    for relative, record in files.items():
        path = base._local_path(parent, relative)
        _require(path.stat().st_size == record["size_bytes"] and _sha(path) == record["sha256"], "Parent file changed: " + relative)
    return manifest


def _flatten(row):
    identity = {k:row[k] for k in ("method", "resolution", "seed")}
    for core, value in row["cores"].items():
        for kind in (*SELECTIONS, "unrestricted", "all_positive", "empty"):
            selection = value[kind]
            for support in ("footprint", "full_region"):
                yield {**identity, "scope": "oracle" if kind in SELECTIONS else "control", "core": int(core),
                    "source_core": int(core), "selection": kind, "support": support, "status": selection["status"],
                    "selected_clusters": json.dumps(selection["selected_clusters"]),
                    **(selection[support] or {})}
    for value in row["transfer"].values():
        for kind in SELECTIONS:
            selection = value[kind]
            for support in ("footprint", "full_region"):
                yield {**identity, "scope": "transfer", "core": value["target_core"], "source_core": value["source_core"],
                    "selection": kind, "support": support, "status": selection["status"],
                    "selected_clusters": json.dumps(selection["selected_clusters"]), **(selection[support] or {})}


def run(parent, output):
    from spatial_benchmark.paths import current_paths
    started = time.monotonic()
    parent, output = Path(parent).resolve(), Path(output).resolve()
    active = (current_paths().scratch_root / "active_runs").resolve()
    _require(output.is_relative_to(active) and output != active and not output.is_relative_to(parent),
             "Writes require a separate active evaluation directory")
    manifest = verify_parent(parent)
    read = lambda name: json.loads((parent/name).read_text())
    clustering, registration, old = read("clustering_summary.json"), read("registration/registration_manifest.json"), read("metrics/pixel_agreement.json")
    _require(clustering["passed"] and clustering["all_methods_complete"] and old["passed"], "Parent scientific stages incomplete")
    _require(_sha(Path(base.__file__)) == old["code_sha256"], "Imported parent metric source changed")
    cells = pd.read_csv(parent/"cells.csv.gz")
    cells_sha = _sha(parent/"cells.csv.gz")
    _require(len(cells) == EXPECTED_CELLS == clustering["n_cells"] and np.array_equal(cells.row_index, np.arange(len(cells)))
             and set(cells.core_number) == {15,23}, "Canonical test cells changed")
    _require(registration["canonical_cells"]["sha256"] == cells_sha == old["input_sha256"]["cells.csv.gz"],
             "Registration and parent metrics bind different canonical cells")
    expected = {(m,r,s) for m in METHODS for r in RESOLUTIONS for s in SEEDS}
    key = lambda r: (r["method"], float(r["resolution"]), int(r["seed"]))
    parts = clustering["partitions"]
    _require({key(r) for r in parts} == expected and len(parts) == len(expected), "Incomplete or duplicate partition grid")
    old_by_key = {key(r):r["evaluation"] for r in old["results"]}
    _require(set(old_by_key) == expected and len(old["results"]) == len(expected), "Parent metric grid differs")
    counts, sensitivity, raster_sources = base._registration_counts(parent, registration, len(cells))
    _require(all(c["status"] == "complete" for c in counts.values()), "Limited-union execution requires verified registration")
    for core,c in counts.items():
        _require(np.all(cells.core_number.to_numpy()[c["area_per_cell"] > 0] == core), "Owner raster crosses core identities")
    inputs = {name:_sha(parent/name) for name in ("cells.csv.gz", "clustering_summary.json", "registration/registration_manifest.json", "metrics/pixel_agreement.json")}
    inputs.update({r["path"]:r["sha256"] for r in raster_sources.values()})
    results, table = [], []
    for part in sorted(parts, key=key):
        record = part["labels"]
        _require(record["relative"] is True, "Expected parent-local labels")
        path = base._local_path(parent, record["path"])
        _require(_sha(path) == record["sha256"], "Parent labels changed")
        labels = np.load(path, allow_pickle=False)
        _require(labels.shape == (len(cells),), "Labels differ from canonical cell rows")
        method,resolution,seed = key(part)
        evaluation = evaluate_partition(labels, counts, parent_cores=old_by_key[key(part)]["cores"],
            null_repeats=20 if (resolution,seed) == (1.,0) else 0, null_seed=0)
        evaluation["fixed_union_boundary_sensitivity"] = fixed_union_sensitivity(labels, evaluation, sensitivity)
        row = {"method":method, "resolution":resolution, "seed":seed, **evaluation}
        results.append(row); table.extend(_flatten(row)); inputs[record["path"]] = record["sha256"]
    verify_parent(parent)
    folder = output/"metrics"; folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(table)
    frame.to_csv(folder/"limited_union.csv", index=False)
    frame[(frame.resolution == 1.) & (frame.seed == 0)].to_csv(folder/"primary_summary.csv", index=False)
    result = {"schema":"so12_cancer_limited_union_v1", "status":"complete", "passed":True,
        "n_cells":len(cells), "test_core_numbers":[15,23], "methods":list(METHODS), "resolutions":list(RESOLUTIONS), "seeds":list(SEEDS),
        "primary_resolution":1., "primary_seed":0, "selection_support":"segmented_cell_footprint",
        "all_evaluable_pixels_retained":True, "parent_inventory_verified_before_after":True,
        "parent":{"path":str(parent), "manifest_sha256":PARENT_MANIFEST_SHA256, "evaluation_id":manifest["evaluation_id"]},
        "input_sha256":inputs, "code_sha256":{"scripts/analysis/so12_cancer_limited_metrics.py":_sha(Path(__file__)),
                                              "scripts/analysis/so12_cancer_metrics.py":_sha(Path(base.__file__))},
        "elapsed_seconds":time.monotonic()-started, "results":results,
        "limitations":"Annotation-selected descriptive oracle; cross-core transfer remains transductive. Two cores, approximate section registration and reference-negative outside outlines do not establish biological accuracy or population superiority."}
    _dump(folder/"limited_union.json", result)
    receipt = {k:v for k,v in result.items() if k != "results"}
    receipt["partition_count"] = len(results)
    receipt["output_sha256"] = {str(f.relative_to(output)):_sha(f) for f in sorted(folder.iterdir()) if f.is_file()}
    _dump(output/"limited_metrics_summary.json", receipt)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output)

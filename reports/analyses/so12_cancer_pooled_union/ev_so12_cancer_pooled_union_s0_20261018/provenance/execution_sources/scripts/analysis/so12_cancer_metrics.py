"""Pixel agreement for fixed partitions, with oracle and cross-core estimands.

Inputs are method-independent owner rasters supplied by verified registration.
Each pixel has one cell owner, so cluster area/overlap counts are additive. No
registration, threshold tuning, annotation reading or clustering occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def build_pixel_counts(owner, roi, valid, *, n_cells, registration_status="verified", registration_reason=None):
    """Compress a native raster into exact integer areas; uncovered pixels=-1.

    Blocked/unverified registration yields an explicit withheld result. Counts
    from disjoint raster tiles may be added before evaluating any partition.
    """
    if registration_status != "verified":
        return {"status": "withheld", "registration_status": registration_status,
                "reason": registration_reason or "Trustworthy registration has not been established"}
    owner, roi, valid = np.asarray(owner), np.asarray(roi), np.asarray(valid)
    _require(owner.ndim == 2 and owner.shape == roi.shape == valid.shape, "Raster shapes must match")
    _require(np.issubdtype(owner.dtype, np.signedinteger), "Owner raster must use signed integers")
    _require(roi.dtype == np.bool_ and valid.dtype == np.bool_, "ROI and valid masks must be boolean")
    _require(n_cells > 0 and np.all((owner >= -1) & (owner < n_cells)), "Invalid global cell owner")
    _require(valid.any(), "Verified registration has no evaluable pixels")
    footprint = valid & (owner >= 0)
    _require(footprint.any(), "Verified registration has no segmented-cell footprint")
    area = np.bincount(owner[footprint], minlength=n_cells).astype(np.int64)
    positive = np.bincount(owner[footprint & roi], minlength=n_cells).astype(np.int64)
    full_positive = int(np.count_nonzero(valid & roi))
    covered_positive = int(positive.sum())
    return {"status": "complete", "registration_status": "verified", "n_cells": int(n_cells),
        "area_per_cell": area, "positive_per_cell": positive,
        "valid_pixels": int(valid.sum()), "footprint_pixels": int(area.sum()),
        "footprint_positive_pixels": covered_positive, "full_region_positive_pixels": full_positive,
        "uncovered_positive_pixels": full_positive - covered_positive,
        "positive_region_coverage": covered_positive / full_positive if full_positive else None,
        "full_region_pixel_coverage_dice_ceiling": 2 * covered_positive / (full_positive + covered_positive)
            if full_positive else None,
        "support": "method_independent_single_owner_cell_raster"}


def binary_scores(tp, predicted_pixels, positive_pixels):
    tp, predicted_pixels, positive_pixels = map(int, (tp, predicted_pixels, positive_pixels))
    _require(0 <= tp <= min(predicted_pixels, positive_pixels), "Impossible binary overlap counts")
    fp, fn = predicted_pixels - tp, positive_pixels - tp
    # A reference with no positive pixels cannot measure positive-region recall.
    return {"tp": tp, "fp": fp, "fn": fn, "predicted_pixels": predicted_pixels,
        "positive_pixels": positive_pixels,
        "dice_f1": 2 * tp / (predicted_pixels + positive_pixels) if positive_pixels else None,
        "iou": tp / (tp + fp + fn) if positive_pixels else None,
        "precision": tp / predicted_pixels if predicted_pixels else None,
        "recall": tp / positive_pixels if positive_pixels else None,
        "reference_has_positive_pixels": positive_pixels > 0}


def optimal_union(area, positive, *, positive_pixels=None):
    """Exact maximal-Dice subset of disjoint clusters, with deterministic ties.

    At the optimum a cluster belongs iff TP/area exceeds half the optimal Dice
    (ties may be included). Thus one purity ordering and all its prefixes are
    sufficient. This is an annotation-selected oracle, never predictive accuracy.
    """
    area, positive = np.asarray(area), np.asarray(positive)
    _require(area.ndim == 1 and area.shape == positive.shape, "Cluster count arrays must align")
    _require(np.issubdtype(area.dtype, np.integer) and np.issubdtype(positive.dtype, np.integer),
             "Pixel counts must be integers")
    _require(np.all(area >= positive) and np.all(positive >= 0), "Invalid cluster pixel counts")
    total = int(positive.sum()) if positive_pixels is None else int(positive_pixels)
    _require(total >= int(positive.sum()), "Reference positives cannot omit covered positive pixels")
    present = np.flatnonzero(area > 0)
    order = present[np.lexsort((present, -positive[present] / area[present]))]
    cum_tp, cum_area = np.r_[0, np.cumsum(positive[order])], np.r_[0, np.cumsum(area[order])]
    best = int(np.argmax(2 * cum_tp / (total + cum_area))) if total else 0
    selected = order[:best].astype(np.int64)
    return {"selected_clusters": selected.tolist(), "scores": binary_scores(cum_tp[best], cum_area[best], total),
            "selection": "annotation_selected_exact_purity_prefix_oracle",
            "tie_rule": "fewest_clusters_then_cluster_id_within_equal_purity"}


def _cluster_counts(labels, counts):
    labels = np.asarray(labels)
    _require(labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer) and np.all(labels >= 0),
             "Labels must be nonnegative integers")
    area, positive = counts["area_per_cell"], counts["positive_per_cell"]
    _require(labels.shape == area.shape == positive.shape, "Cell labels and pixel counts do not align")
    return (np.bincount(labels, weights=area, minlength=int(labels.max()) + 1).astype(np.int64),
            np.bincount(labels, weights=positive, minlength=int(labels.max()) + 1).astype(np.int64))


def _score_union(selected, area, positive, counts):
    indices = np.asarray(selected, dtype=np.int64)
    _require(np.all((indices >= 0) & (indices < len(area))) and len(np.unique(indices)) == len(indices),
             "Selected cluster IDs are invalid or duplicated")
    tp, predicted = int(positive[indices].sum()), int(area[indices].sum())
    return {"footprint": binary_scores(tp, predicted, counts["footprint_positive_pixels"]),
            "full_region": binary_scores(tp, predicted, counts["full_region_positive_pixels"])}


def evaluate_partition(labels, counts_by_core, *, null_repeats=20, null_seed=0):
    """Score oracle unions and fixed unions transferred between exactly two cores.

    Selection always maximizes source-core FOOTPRINT Dice. The same selected
    union receives both footprint and full-region scores; secondary support does
    not trigger a second optimization. Unknown source-core clusters stay negative.
    """
    _require(len(counts_by_core) == 2, "Cross-core transfer requires exactly two cores")
    _require(isinstance(null_repeats, int) and null_repeats >= 0, "Invalid null repetition count")
    labels = np.asarray(labels)
    cores = sorted(counts_by_core)
    complete_counts = [value for value in counts_by_core.values() if value.get("status") == "complete"]
    if len(complete_counts) == 2:
        _require(not np.any((complete_counts[0]["area_per_cell"] > 0) & (complete_counts[1]["area_per_cell"] > 0)),
                 "A global cell row cannot belong to both core rasters")
    result = {"selection_support": "segmented_cell_footprint", "cores": {}, "transfer": {},
        "interpretation": "Oracle overlap is an annotation-selected descriptive upper bound; cross-core transfer remains transductive and exploratory."}
    grouped, selections = {}, {}
    for core in cores:
        counts = counts_by_core[core]
        if counts.get("status") != "complete":
            result["cores"][str(core)] = {"status": "withheld", "reason": counts.get("reason", "Registration unverified")}
            continue
        area, positive = _cluster_counts(labels, counts)
        grouped[core] = area, positive
        oracle = optimal_union(area, positive)
        selections[core] = oracle["selected_clusters"]
        result["cores"][str(core)] = {"status": "complete", "oracle": {
            "selected_clusters": oracle["selected_clusters"], "selection": oracle["selection"],
            **_score_union(oracle["selected_clusters"], area, positive, counts)},
            "all_positive": _score_union(np.flatnonzero(area), area, positive, counts),
            "empty": _score_union([], area, positive, counts),
            "coverage": {key: value for key, value in counts.items() if not isinstance(value, np.ndarray)}}
    for source, target in (cores, cores[::-1]):
        key = f"{source}_to_{target}"
        if source not in grouped or target not in grouped:
            result["transfer"][key] = {"status": "withheld", "reason": "Both registrations must be verified"}
            continue
        area, positive = grouped[target]
        source_area = grouped[source][0]
        result["transfer"][key] = {"status": "complete", "source_core": int(source), "target_core": int(target),
            "selected_clusters": selections[source], "selection_uses_target_annotation": False,
            "unseen_target_pixel_fraction": float(area[source_area == 0].sum() / max(1, area.sum())),
            **_score_union(selections[source], area, positive, counts_by_core[target])}
    for kind in ("oracle", "transfer"):
        rows = [value[kind] for value in result["cores"].values() if value["status"] == "complete"] if kind == "oracle" else [value for value in result["transfer"].values() if value["status"] == "complete"]
        values = [row["footprint"]["dice_f1"] for row in rows]
        result[f"equal_core_{kind}_dice"] = float(np.mean(values)) if len(values) == 2 and all(v is not None for v in values) else None
    if null_repeats:
        rng = np.random.default_rng(null_seed)
        null_rows = []
        for repeat in range(null_repeats):
            shuffled = labels.copy()
            for core in grouped:
                visible = np.flatnonzero(counts_by_core[core]["area_per_cell"] > 0)
                shuffled[visible] = rng.permutation(shuffled[visible])
            null_result = evaluate_partition(shuffled, counts_by_core, null_repeats=0)
            null_rows.append({"repeat": repeat, "equal_core_oracle_dice": null_result["equal_core_oracle_dice"],
                              "equal_core_transfer_dice": null_result["equal_core_transfer_dice"]})
        result["shuffled_label_control"] = {"seed": int(null_seed), "repeats": null_repeats,
            "scope": "visible_cell_labels_permuted_independently_within_each_core",
            "oracle_reselected_each_repeat": True, "population_p_value_claimed": False, "results": null_rows}
    return result


def seed_stability(first, second, cell_cores):
    first, second, cell_cores = map(np.asarray, (first, second, cell_cores))
    _require(first.ndim == 1 and first.shape == second.shape == cell_cores.shape and len(first) > 0,
             "Seed stability arrays must share nonempty cell coverage")
    return {"all_cells_ari": float(adjusted_rand_score(first, second)),
            "per_core_ari": {str(core): float(adjusted_rand_score(first[cell_cores == core], second[cell_cores == core]))
                             for core in np.unique(cell_cores)}}


EXPECTED_CELLS = 81607
METHODS = ("h0", "h1", "h2", "h3", "h4", "pca", "scvi")
RESOLUTIONS = (.25, .5, 1., 2.)
SEEDS = (0, 1)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _local_path(output, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Expected an output-relative artifact path")
    return output / relative


def _registration_counts(output, registration, n_cells):
    rows = registration.get("cores", [])
    if isinstance(rows, dict):
        rows = [dict(value, core_number=int(core)) for core, value in rows.items()]
    _require(sorted(int(row["core_number"]) for row in rows) == [15, 23], "Registration must account for both cores")
    nominal, sensitivity, sources = {}, {}, {}
    for row in rows:
        core = int(row["core_number"])
        if not row.get("passed") or not row.get("status", "").startswith("verified"):
            nominal[core] = build_pixel_counts(None, None, None, n_cells=n_cells,
                registration_status=row.get("status", "unverified"), registration_reason=row.get("reason"))
            sensitivity[core] = {}
            continue
        path = _local_path(output, row["raster_path"])
        _require(_sha(path) == row["raster_sha256"], "Registered raster checksum changed")
        sources[str(core)] = {"path": row["raster_path"], "sha256": row["raster_sha256"], "registration_status": row["status"]}
        with np.load(path, allow_pickle=False) as arrays:
            needed = ("owner", "roi", "valid", "valid50", "roi_eroded50", "roi_dilated50")
            _require(set(needed).issubset(arrays.files), "Registration raster lacks boundary sensitivity masks")
            owner, roi, valid = arrays["owner"], arrays["roi"], arrays["valid"]
            nominal[core] = build_pixel_counts(owner, roi, valid, n_cells=n_cells)
            sensitivity[core] = {}
            for name, altered_roi, altered_valid in (
                ("exclude_boundary_50um", roi, arrays["valid50"]),
                ("erode_roi_50um", arrays["roi_eroded50"], valid),
                ("dilate_roi_50um", arrays["roi_dilated50"], valid)):
                if not np.any(altered_valid & (owner >= 0)):
                    sensitivity[core][name] = {"status": "withheld", "reason": "No segmented footprint remains under this sensitivity mask"}
                else:
                    sensitivity[core][name] = build_pixel_counts(owner, altered_roi, altered_valid, n_cells=n_cells)
    return nominal, sensitivity, sources


def _fixed_union_sensitivity(labels, nominal_result, sensitivity):
    result = {}
    for core, scenarios in sensitivity.items():
        result[str(core)] = {}
        if nominal_result["cores"][str(core)]["status"] != "complete":
            continue
        source = 23 if core == 15 else 15
        transfer = nominal_result["transfer"][f"{source}_to_{core}"]
        for name, counts in scenarios.items():
            if counts["status"] != "complete":
                result[str(core)][name] = counts
                continue
            area, positive = _cluster_counts(labels, counts)
            row = {"selection_frozen_from_nominal_masks": True, "nominal_oracle_union":
                _score_union(nominal_result["cores"][str(core)]["oracle"]["selected_clusters"], area, positive, counts)}
            row["transferred_union"] = (_score_union(transfer["selected_clusters"], area, positive, counts)
                                         if transfer["status"] == "complete" else {"status": "withheld"})
            result[str(core)][name] = row
    return result


def run_metrics(output):
    """Evaluate every locked partition, or explicitly withhold unregistered cores."""
    from spatial_benchmark.paths import current_paths
    output = Path(output).resolve()
    active = (current_paths().scratch_root / "active_runs").resolve()
    _require(output.is_relative_to(active) and output != active, "Metrics writes require an active evaluation directory")
    cluster_path = output / "clustering_summary.json"
    registration_path = output / "registration/registration_manifest.json"
    clustering = json.loads(cluster_path.read_text())
    registration = json.loads(registration_path.read_text())
    _require(clustering.get("passed") and clustering.get("status") == "complete"
             and clustering["n_cells"] == EXPECTED_CELLS, "Complete clustering is required")
    cells_path = output / "cells.csv.gz"
    cells = pd.read_csv(cells_path)
    _require(len(cells) == EXPECTED_CELLS and np.array_equal(cells.row_index, np.arange(len(cells)))
             and set(cells.core_number) == {15, 23}, "Canonical test cell coverage changed")
    expected = {(method, resolution, seed) for method in METHODS for resolution in RESOLUTIONS for seed in SEEDS}
    actual = {(row["method"], row["resolution"], row["seed"]) for row in clustering["partitions"]}
    _require(actual == expected and len(clustering["partitions"]) == len(expected), "Resolution/seed coverage incomplete")
    nominal, sensitivity, raster_sources = _registration_counts(output, registration, len(cells))
    # Ownership must refer to the target core, not merely a numerically valid row.
    for core, counts in nominal.items():
        if counts["status"] == "complete":
            _require(np.all(cells.core_number.to_numpy()[counts["area_per_cell"] > 0] == core), "Owner raster crosses core identities")
    results, table, label_bindings, stability_inputs = [], [], {}, {}
    for partition in sorted(clustering["partitions"], key=lambda x:(x["method"],x["resolution"],x["seed"])):
        record = partition["labels"]
        _require(record.get("relative") is True, "Partition labels must be local immutable outputs")
        path = _local_path(output, record["path"])
        _require(_sha(path) == record["sha256"], "Partition labels changed")
        labels = np.load(path, allow_pickle=False)
        _require(labels.shape == (len(cells),) and np.issubdtype(labels.dtype, np.integer) and np.all(labels >= 0), "Invalid labels")
        method, resolution, seed = partition["method"], partition["resolution"], partition["seed"]
        stability_inputs[(method, resolution, seed)] = labels
        evaluation = evaluate_partition(labels, nominal, null_repeats=20 if resolution == 1.0 and seed == 0 else 0, null_seed=0)
        evaluation["fixed_union_boundary_sensitivity"] = _fixed_union_sensitivity(labels, evaluation, sensitivity)
        result = {"method": method, "resolution": resolution, "seed": seed,
                  "cluster_count": int(np.unique(labels).size), "evaluation": evaluation}
        results.append(result)
        label_bindings[record["path"]] = record["sha256"]
        for core, core_result in evaluation["cores"].items():
            for estimand in ("oracle", "all_positive", "empty"):
                if core_result["status"] != "complete":
                    table.append({"method": method, "resolution": resolution, "seed": seed, "core": int(core),
                                  "estimand": estimand, "status": "withheld", "reason": core_result["reason"]})
                    continue
                for support in ("footprint", "full_region"):
                    table.append({"method": method, "resolution": resolution, "seed": seed, "core": int(core),
                        "estimand": estimand, "support": support, "status": "complete", **core_result[estimand][support]})
        for key, transfer in evaluation["transfer"].items():
            for support in ("footprint", "full_region"):
                table.append({"method": method, "resolution": resolution, "seed": seed, "transfer": key,
                    "estimand": "cross_core_transfer", "support": support, "status": transfer["status"],
                    **(transfer[support] if transfer["status"] == "complete" else {"reason": transfer["reason"]})})
    stability = [{"method": method, "resolution": resolution,
        **seed_stability(stability_inputs[(method,resolution,0)], stability_inputs[(method,resolution,1)], cells.core_number.to_numpy())}
        for method in METHODS for resolution in RESOLUTIONS]
    folder = output / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(folder / "pixel_agreement.csv", index=False)
    pd.DataFrame([{"method": row["method"], "resolution": row["resolution"], "all_cells_ari": row["all_cells_ari"],
        **{f"core_{core}_ari": value for core,value in row["per_core_ari"].items()}} for row in stability]).to_csv(folder / "seed_stability.csv", index=False)
    summary = {"schema": "so12_cancer_pixel_agreement_v1", "status": "complete", "passed": True,
        "n_cells": len(cells), "methods": list(METHODS), "resolutions": list(RESOLUTIONS), "seeds": list(SEEDS),
        "pixel_scores_available": all(value["status"] == "complete" for value in nominal.values()),
        "registration_status_by_core": {str(core): value["status"] for core,value in nominal.items()},
        "selection_support": "segmented_cell_footprint", "primary_resolution": 1., "primary_seed": 0,
        "null_controls": "20 within-core visible-cell label shuffles for each method at resolution1 seed0; oracle reselected for every shuffle",
        "no_population_confidence_interval_or_superiority_claim": True, "results": results, "seed_stability": stability,
        "input_sha256": {"cells.csv.gz": _sha(cells_path), "clustering_summary.json": _sha(cluster_path),
                         "registration/registration_manifest.json": _sha(registration_path), **label_bindings},
        "raster_sources": raster_sources, "code_sha256": _sha(Path(__file__)),
        "limitations": "Annotation-selected oracle is an upper bound, not unbiased accuracy. Transfer is between two jointly clustered cores, with donor overlap allowed. Outside outlines is reference-negative for this exercise, not proof of cancer absence."}
    _dump(folder / "pixel_agreement.json", summary)
    receipt = {key:value for key,value in summary.items() if key not in ("results", "seed_stability", "input_sha256")}
    receipt["output_sha256"] = {str(path.relative_to(output)): _sha(path) for path in sorted(folder.iterdir()) if path.is_file()}
    _dump(output / "metrics_summary.json", receipt)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_metrics(arguments.output)

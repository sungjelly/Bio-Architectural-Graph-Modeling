#!/usr/bin/env python3
"""Independently audit fixed one-/two-cluster pixel scores, without fitting.

This checker does not import any selection or scoring implementation. It derives
integer areas from the native owner rasters and exhaustively compares singleton
and pair objectives as exact fractions. Only its receipt is written, and only
inside the designated active evaluation directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
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

PARENT_EVALUATION = "ev_so12_cancer_comparison_s0_20261018"
EVALUATION = "ev_so12_cancer_limited_union_s0_20261018"
PARENT_SHA256 = "e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0"
METHODS = ("h0", "h1", "h2", "h3", "h4", "pca", "scvi")
KINDS = ("best_single", "best_pair", "best_at_most_two")
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
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe local artifact path")
    target = root / relative
    require(target.resolve().is_relative_to(root.resolve()) and not target.is_symlink(), "Artifact path escapes its bundle")
    return target


def verify_parent(parent):
    require(sha(parent / "manifest.json") == PARENT_SHA256, "Parent manifest identity changed")
    manifest = read(parent / "manifest.json")
    require(manifest["evaluation_id"] == PARENT_EVALUATION
            and manifest["passed"] is True and manifest["status"] == "complete", "Parent is incomplete")
    require(read(parent / "_SUCCESS")["manifest_sha256"] == PARENT_SHA256, "Parent success marker changed")
    require(not any(path.is_symlink() for path in parent.rglob("*")), "Parent contains symlinks")
    actual = {str(path.relative_to(parent)) for path in parent.rglob("*") if path.is_file()} - {"manifest.json", "_SUCCESS"}
    require(actual == set(manifest["files"]), "Parent inventory changed")
    for relative, item in manifest["files"].items():
        path = local(parent, relative)
        require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
                f"Parent file changed: {relative}")
    return manifest


def exact_objective(selected, area, positive):
    total_positive = sum(int(value) for value in positive)
    tp = sum(int(positive[index]) for index in selected)
    predicted = sum(int(area[index]) for index in selected)
    return Fraction(2 * tp, total_positive + predicted) if total_positive else Fraction(0)


def independent_best_selections(area, positive):
    """Exhaustive exact alternatives, with fewer-clusters/ascending-ID ties."""
    require(len(area) == len(positive) and all(0 <= int(p) <= int(a) for a, p in zip(area, positive)),
            "Invalid cluster pixel counts")
    visible = tuple(index for index, value in enumerate(area) if value > 0)
    singletons = [(index,) for index in visible]
    pairs = list(combinations(visible, 2))
    def choose(candidates):
        return min(candidates, key=lambda selected: (-exact_objective(selected, area, positive),
                                                     len(selected), selected)) if candidates else None
    return {"best_single": choose(singletons), "best_pair": choose(pairs),
            "best_at_most_two": choose([()] + singletons + pairs)}


def count_pixels(owner, roi, valid):
    require(owner.ndim == 2 and owner.shape == roi.shape == valid.shape, "Raster shapes differ")
    require(owner.dtype.kind == "i" and roi.dtype == np.bool_ and valid.dtype == np.bool_, "Raster dtypes differ")
    require(np.all((owner >= -1) & (owner < N_CELLS)), "Invalid global cell owner")
    footprint = valid & (owner >= 0)
    return {"area": np.bincount(owner[footprint], minlength=N_CELLS),
            "positive": np.bincount(owner[footprint & roi], minlength=N_CELLS),
            "footprint_pixels": int(footprint.sum()), "valid_pixels": int(valid.sum()),
            "full_positive": int((valid & roi).sum())}


def grouped(labels, counts):
    size = int(labels.max()) + 1
    area, positive = np.zeros(size, np.int64), np.zeros(size, np.int64)
    # Integer accumulation is independent of the metric implementation's weights.
    np.add.at(area, labels, counts["area"])
    np.add.at(positive, labels, counts["positive"])
    return area, positive


class Checker:
    def __init__(self):
        self.scores = self.selections = self.transfers = self.null_repeats = self.sensitivities = 0

    @staticmethod
    def equal(actual, expected, name):
        require(actual is None if expected is None else actual is not None
                and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
                f"Score differs: {name}, got {actual}, expected {expected}")

    def score(self, record, tp, predicted, total_positive, support):
        fp, fn = predicted - tp, total_positive - tp
        tn = support - tp - fp - fn
        require(min(tp, fp, fn, tn) >= 0, "Invalid native confusion counts")
        for key, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "predicted_pixels": predicted, "positive_pixels": total_positive,
                           "evaluable_pixels": support}.items():
            require(record[key] == value, f"Native pixel denominator/count differs: {key}")
        for key, value in {
            "dice_f1": 2 * tp / (predicted + total_positive) if total_positive else None,
            "iou": tp / (predicted + total_positive - tp) if total_positive else None,
            "precision": tp / predicted if predicted else None,
            "recall": tp / total_positive if total_positive else None,
        }.items():
            self.equal(record[key], value, key)
        self.scores += 1

    def selection(self, record, expected, area, positive, counts, *, strict_ids=True):
        if expected is None:
            require(record["status"] == "unavailable" and record["selected_clusters"] == []
                    and record["footprint"] is None and record["full_region"] is None,
                    "An unavailable singleton/pair was assigned a score")
            return
        selected = tuple(record["selected_clusters"])
        require(len(selected) == len(set(selected)) and all(0 <= index < len(area) for index in selected),
                "Invalid selected cluster IDs")
        require(selected == expected if strict_ids else set(selected) == set(expected),
                f"Selected union or exact tie rule differs: {selected} versus {expected}")
        require(record.get("status", "complete") == "complete", "Selected union is incomplete")
        tp = sum(int(positive[index]) for index in selected)
        predicted = sum(int(area[index]) for index in selected)
        self.score(record["footprint"], tp, predicted, int(positive.sum()), counts["footprint_pixels"])
        self.score(record["full_region"], tp, predicted, counts["full_positive"], counts["valid_pixels"])
        self.selections += 1

    def evaluation(self, result, labels, counts, *, unrestricted=None, controls=True):
        choices, groups = {}, {}
        for core in CORES:
            area, positive = grouped(labels, counts[core])
            groups[core] = area, positive
            choices[core] = independent_best_selections(area, positive)
            record = result["cores"][str(core)]
            for kind in KINDS:
                self.selection(record[kind], choices[core][kind], area, positive, counts[core])
            if controls:
                self.selection(record["empty"], (), area, positive, counts[core])
                self.selection(record["all_positive"], tuple(np.flatnonzero(area)), area, positive, counts[core], strict_ids=False)
            if unrestricted is not None:
                parent_union = tuple(unrestricted["cores"][str(core)]["oracle"]["selected_clusters"])
                self.selection(record["unrestricted"], parent_union, area, positive, counts[core], strict_ids=False)
                single = exact_objective(choices[core]["best_single"], area, positive)
                limited = exact_objective(choices[core]["best_at_most_two"], area, positive)
                unlimited = exact_objective(parent_union, area, positive)
                require(single <= limited <= unlimited, "Limited/unrestricted monotonicity failed")
        for source, target in ((15, 23), (23, 15)):
            record = result["transfer"][f"{source}_to_{target}"]
            require(record["source_core"] == source and record["target_core"] == target,
                    "Transfer direction changed")
            require(record["selection_uses_target_annotation"] is False,
                    "Transfer selection uses target annotations")
            area, positive = groups[target]
            for kind in KINDS:
                self.selection(record[kind], choices[source][kind], area, positive, counts[target])
                self.transfers += 1
        for scope, field in (("oracle", "cores"), ("transfer", "transfer")):
            for kind in KINDS:
                values = [row[kind]["footprint"]["dice_f1"] if row[kind]["status"] == "complete" else None
                          for row in result[field].values()]
                expected = sum(values) / 2 if len(values) == 2 and all(value is not None for value in values) else None
                self.equal(result[f"equal_core_{scope}_dice"][kind], expected,
                           f"equal_core_{scope}_{kind}")
        return choices

    def boundary(self, result, labels, counts, choices):
        for core in CORES:
            source = 23 if core == 15 else 15
            for scenario, pixel_counts in counts[core].items():
                record = result["fixed_union_boundary_sensitivity"][str(core)][scenario]
                require(record["selection_frozen_from_nominal_masks"] is True, "Boundary union was reselected")
                area, positive = grouped(labels, pixel_counts)
                for kind in KINDS:
                    self.selection(record["oracle"][kind], choices[core][kind], area, positive, pixel_counts)
                    self.selection(record["transfer"][kind], choices[source][kind], area, positive, pixel_counts)
                    self.sensitivities += 2


def audit(parent, output, paths=None):
    started = time.monotonic()
    paths = paths or current_paths()
    parent, output = Path(parent).resolve(), Path(output).resolve()
    expected_parent = paths.report_root / "analyses/so12_cancer_comparison" / PARENT_EVALUATION
    require(parent == expected_parent.resolve(), "Wrong immutable parent bundle")
    require(output == (paths.scratch_root / "active_runs" / EVALUATION).resolve(), "Audit writes require the designated active evaluation")
    require(output.is_dir() and not (output / "_SUCCESS").exists(), "Output is absent or immutable")
    parent_manifest = verify_parent(parent)
    metric_path = output / "metrics/limited_union.json"
    summary_path = output / "limited_metrics_summary.json"
    metrics, summary = read(metric_path), read(summary_path)
    require(metrics["passed"] is True and metrics["status"] == "complete"
            and summary["passed"] is True and summary["status"] == "complete", "Limited metrics are incomplete")
    require(metrics["n_cells"] == N_CELLS and metrics["test_core_numbers"] == list(CORES)
            and metrics["selection_support"] == "segmented_cell_footprint"
            and metrics["all_evaluable_pixels_retained"] is True
            and metrics["parent_inventory_verified_before_after"] is True,
            "Metric population or pixel support changed")
    require(metrics["parent"]["manifest_sha256"] == PARENT_SHA256
            and Path(metrics["parent"]["path"]).resolve() == parent
            and metrics["parent"]["evaluation_id"] == PARENT_EVALUATION, "Metric parent identity changed")
    for relative, digest in metrics["input_sha256"].items():
        require(sha(local(parent, relative)) == digest, f"Metric source changed: {relative}")
    for relative, digest in summary["output_sha256"].items():
        require(sha(local(output, relative)) == digest, f"Metric output changed: {relative}")
    for relative, digest in metrics["code_sha256"].items():
        require(sha(local(paths.project_root, relative)) == digest, f"Metric code changed: {relative}")
    parent_metrics = read(parent / "metrics/pixel_agreement.json")
    clustering = read(parent / "clustering_summary.json")
    registration = read(parent / "registration/registration_manifest.json")
    cells = pd.read_csv(parent / "cells.csv.gz", usecols=["row_index", "core_number"])
    require(len(cells) == N_CELLS and np.array_equal(cells.row_index, np.arange(N_CELLS))
            and cells.core_number.value_counts().to_dict() == {23: 43462, 15: 38145}, "Parent cell coverage differs")
    require(registration["canonical_cells"]["sha256"] == sha(parent / "cells.csv.gz"), "Parent registration cell order differs")
    expected = {(method, resolution, seed) for method in METHODS for resolution in (.25, .5, 1., 2.) for seed in (0, 1)}
    def index(rows):
        result = {(row["method"], row["resolution"], row["seed"]): row for row in rows}
        require(len(rows) == 56 and set(result) == expected, "Incomplete or duplicate partition coverage")
        return result
    partitions, upper_bounds, measured = index(clustering["partitions"]), index(parent_metrics["results"]), index(metrics["results"])
    counts, sensitivity = {}, {}
    for record in registration["cores"]:
        core = record["core_number"]
        raster = local(parent, record["raster_path"])
        require(sha(raster) == record["raster_sha256"], "Parent owner raster changed")
        with np.load(raster, allow_pickle=False) as arrays:
            owner, roi, valid = arrays["owner"], arrays["roi"], arrays["valid"]
            counts[core] = count_pixels(owner, roi, valid)
            require(np.all(cells.core_number.to_numpy()[counts[core]["area"] > 0] == core), "Owner raster includes cells from another core")
            sensitivity[core] = {
                "exclude_boundary_50um": count_pixels(owner, roi, arrays["valid50"]),
                "erode_roi_50um": count_pixels(owner, arrays["roi_eroded50"], valid),
                "dilate_roi_50um": count_pixels(owner, arrays["roi_dilated50"], valid),
            }
    checker, primary = Checker(), []
    for key in sorted(expected):
        partition, result = partitions[key], measured[key]
        path = local(parent, partition["labels"]["path"])
        require(sha(path) == partition["labels"]["sha256"], "Parent partition labels changed")
        labels = np.load(path, allow_pickle=False)
        require(labels.shape == (N_CELLS,) and labels.dtype.kind in "iu" and np.all(labels >= 0), "Invalid native labels")
        choices = checker.evaluation(result, labels, counts, unrestricted=upper_bounds[key]["evaluation"])
        checker.boundary(result, labels, sensitivity, choices)
        if key[1:] == (1., 0):
            for core in CORES:
                record = result["cores"][str(core)]
                primary.append({"method": key[0], "core": core,
                                **{kind: {"selected_clusters": record[kind]["selected_clusters"],
                                          "dice_f1": record[kind]["footprint"]["dice_f1"]} for kind in KINDS}})
            control = result["shuffled_label_control"]
            require(control["seed"] == 0 and control["repeats"] == 20
                    and len(control["results"]) == 20, "Primary shuffled-control coverage changed")
            rng = np.random.default_rng(0)
            for repeat, null_result in enumerate(control["results"]):
                require(null_result["repeat"] == repeat, "Shuffled-control ordering differs")
                shuffled = labels.copy()
                for core in CORES:
                    visible = np.flatnonzero(counts[core]["area"] > 0)
                    shuffled[visible] = rng.permutation(shuffled[visible])
                checker.evaluation(null_result, shuffled, counts, controls=False)
                checker.null_repeats += 1
    verify_parent(parent)
    result = {"schema": "so12_cancer_limited_union_independent_audit_v1", "status": "complete", "passed": True,
              "parent_evaluation_id": PARENT_EVALUATION, "parent_manifest_sha256": PARENT_SHA256,
              "parent_files_verified_before_after": len(parent_manifest["files"]),
              "metrics_sha256": sha(metric_path), "metrics_summary_sha256": sha(summary_path),
              "audit_source_sha256": sha(Path(__file__)), "n_cells": N_CELLS, "partition_count": 56,
              "binary_score_records_checked": checker.scores, "selected_unions_checked": checker.selections,
              "transfer_unions_checked": checker.transfers, "boundary_unions_checked": checker.sensitivities,
              "null_replicates_independently_recomputed": checker.null_repeats,
              "exact_fraction_objectives_and_ties_verified": True,
              "single_at_most_two_unrestricted_monotonicity_verified": True,
              "native_pixel_denominators_and_outside_outline_penalties_verified": True,
              "primary": primary, "elapsed_seconds": time.monotonic() - started,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "interpretation": "Annotation-selected limited unions remain exploratory oracle agreement; transferred choices use the other core, with no refitting or annotation clipping."}
    destination = output / "independent_metrics_review.json"
    require(not destination.exists(), "Refusing to overwrite an existing independent review")
    temporary = destination.with_suffix(f".json.tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parent, args.output)
    print(json.dumps({key: result[key] for key in ("status", "passed", "partition_count",
                     "binary_score_records_checked", "null_replicates_independently_recomputed", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

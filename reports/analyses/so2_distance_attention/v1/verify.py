"""Independently audit saved distance-attention receipts and aggregates.

Run from the BAGM root with PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_distance_attention/v1/verify.py. --pilot-only checks core 21
without writing verification.json. --rehash-inputs additionally rereads the
large input files; default checks their extraction receipts against manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths


RUN = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
CHECKPOINT_SHA = "aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b"
RANGES = ("10_450_um", "all_edges")
CHANNELS = ("attention", "degree_scaled_attention", "inverse_square_attention",
            "degree_scaled_inverse_square", "score", "content", "beta")
BINS = np.linspace(0, 500, 51)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


class Audit:
    def __init__(self):
        self.maximum_differences = {}

    def close(self, actual, expected, family, *, rtol=5e-12, atol=5e-12):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(actual.shape == expected.shape, f"{family}: shape mismatch")
        require(np.array_equal(np.isnan(actual), np.isnan(expected)), f"{family}: NaN mismatch")
        valid = ~np.isnan(expected)
        require(np.isfinite(actual[valid]).all() and np.isfinite(expected[valid]).all(),
                f"{family}: non-finite values")
        difference = float(np.max(np.abs(actual[valid] - expected[valid]))) if valid.any() else 0.0
        self.maximum_differences[family] = max(self.maximum_differences.get(family, 0.0), difference)
        require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True),
                f"{family}: maximum absolute difference {difference}")


def independently_fit(moments):
    vx, vr, vy, cxy, cry = np.asarray(moments, dtype=np.float64)
    require(np.isfinite(moments).all() and min(vx, vr, vy) > 0, "Degenerate/non-finite moments")
    p, rate = -cxy / vx, -cry / vr
    power_mse = vy + p * p * vx + 2 * p * cxy
    exponential_mse = vy + rate * rate * vr + 2 * rate * cry
    square_mse = vy + 4 * vx + 4 * cxy
    require(power_mse >= -1e-12 and exponential_mse >= -1e-12, "Invalid covariance moments")
    return dict(power_exponent=float(p), power_r2=float(1 - power_mse / vy),
                exponential_rate_per_um=float(rate),
                exponential_length_um=float(1 / rate) if rate > 0 else None,
                exponential_r2=float(1 - exponential_mse / vy), flat_mse=float(vy),
                power_mse=float(max(0, power_mse)), inverse_square_mse=float(square_mse),
                exponential_mse=float(max(0, exponential_mse)),
                inverse_square_r2=float(1 - square_mse / vy))


def compare_fit(audit, record, moments, family):
    expected = independently_fit(moments)
    for key, value in expected.items():
        if value is None:
            require(record[key] is None, f"{family}/{key}: expected null")
        else:
            audit.close(record[key], value, family)


def check_csv(audit, path, records, key_fields):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    indexed = {tuple(r[k] for k in key_fields): r for r in records}
    require(len(rows) == len(records) == len(indexed), f"{path.name}: row count")
    seen = set()
    for row in rows:
        key = tuple(row[k] if k == "range" else int(row[k]) for k in key_fields)
        require(key in indexed and key not in seen, f"{path.name}: key coverage")
        seen.add(key)
        expected = {k: v for k, v in indexed[key].items() if k != "moments"}
        require(set(row) == set(expected), f"{path.name}: columns")
        for field, value in expected.items():
            if value is None:
                require(row[field] == "", f"{path.name}/{field}: expected empty null")
            elif isinstance(value, str):
                require(row[field] == value, f"{path.name}/{field}: text mismatch")
            else:
                audit.close(float(row[field]), value, "csv_vs_json", rtol=0, atol=0)


def verify(*, pilot_only=False, rehash_inputs=False):
    paths = current_paths()
    root = paths.report_root / "analyses/so2_distance_attention/v1"
    previous = paths.report_root / "analyses/so2_attention_tau_beta/v1"
    bundle = paths.artifact_root / "runs/2026/09" / RUN
    cores = [21] if pilot_only else list(range(15, 29))
    receipts = [read_json(root / f"core_{c}.json") for c in cores]
    audit = Audit()
    checked_files = [root / f"core_{c}.json" for c in cores]
    checkpoint = bundle / "checkpoints/last.ckpt"
    require(sha256_file(checkpoint) == CHECKPOINT_SHA, "Checkpoint checksum")
    bundle_checksums = read_json(bundle / "provenance/artifact_checksums.json")["files"]
    config_path = bundle / "config.resolved.yaml"
    require(sha256_file(config_path) == bundle_checksums["config.resolved.yaml"]["sha256"],
            "Frozen configuration checksum")
    config = yaml.safe_load(config_path.read_text())
    cohort = paths.data_root / "processed/so2_14core_relative_qkv_v1"
    graphs = paths.data_root / "processed/so2_14core_relative_qkv_graphs_v1"
    manifests = {}
    for name, directory in (("cohort", cohort), ("graph", graphs)):
        require(sha256_file(directory / "manifest.json") ==
                config["dataset"][f"{name}_manifest_file_sha256"], f"{name} manifest checksum")
        manifests[name] = read_json(directory / "manifest.json")
    source_manifest = root / "provenance/source_hashes.json"
    source_hashes = read_json(source_manifest)
    checked_files.append(source_manifest)
    require(len(source_hashes) == 7, "Expected seven extraction source files")
    for relative, digest in source_hashes.items():
        require(sha256_file(paths.project_root / relative) == digest, f"Current source: {relative}")
        require(sha256_file(root / "provenance/executed_source" / relative) == digest,
                f"Preserved source: {relative}")
    training = read_json(bundle / "provenance/untracked_files.json")
    training = training if isinstance(training, list) else training["files"]
    geometry_source = "src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py"
    require(next(r["sha256"] for r in training if r["path"] == geometry_source) ==
            source_hashes[geometry_source], "Geometry source differs from training")

    all_rows, core_curves, bins_hashes = [], [], {}
    max_attention_error, max_distance_error, max_replay_error = 0.0, 0.0, 0.0
    replay_count, input_count = 0, 0
    for core, receipt in zip(cores, receipts):
        require(receipt["core"] == core and receipt["run_id"] == RUN and
                receipt["checkpoint_sha256"] == CHECKPOINT_SHA, "Receipt identity")
        require(receipt["code_hashes"] == source_hashes, "Receipt source identity")
        cr = next(r for r in manifests["cohort"]["cores"] if r["alias"] == f"SO2-C{core}")
        gr = next(r for r in manifests["graph"]["cores"] if r["alias"] == f"SO2-C{core}")
        require(receipt["cells"] == cr["cell_count"] == gr["n_cells"], "Cell coverage")
        require(cr["coordinate_units"] == "micrometres", "Coordinate units")
        require(receipt["edges"] == gr["graph"]["qc"]["n_directed_edges"], "Edge coverage")
        expected_inputs = {str((cohort / "cores" / f"SO2-C{core}.npz").relative_to(paths.project_root)):
                           manifests["cohort"]["files"][f"cores/SO2-C{core}.npz"]}
        for filename in ("edge_index.npy", "relative_geometry.npy"):
            expected_inputs[str((graphs / "cores" / f"SO2-C{core}" / filename).relative_to(paths.project_root))] = gr["files"][filename]
        require(receipt["input_files"] == expected_inputs, "Input receipt/manifest agreement")
        input_count += len(expected_inputs)
        if rehash_inputs:
            for relative, digest in expected_inputs.items():
                require(sha256_file(paths.project_root / relative) == digest, f"Live input checksum: {relative}")
        old = read_json(previous / f"core_{core}.json")
        require((receipt["mask_seed"], receipt["mask_sha256"]) ==
                (old["mask_seed"], old["mask_sha256"]), "Previous fixed mask identity")
        require(0 <= receipt["attention_sum_max_error"] < 2e-6, "Recorded attention normalization")
        require(0 <= receipt["distance_cache_max_error_um"] < 4e-5, "Recorded distance agreement")
        max_attention_error = max(max_attention_error, receipt["attention_sum_max_error"])
        max_distance_error = max(max_distance_error, receipt["distance_cache_max_error_um"])
        replay = receipt["pilot_public_forward_max_abs_error"]
        if replay is not None:
            require(np.isfinite(replay) and replay >= 0, "Replay error invalid")
            replay_count += 1
            max_replay_error = max(max_replay_error, replay)

        rows = receipt["rows"]
        grid = {(r["core"], r["block"], r["head"], r["range"]) for r in rows}
        expected_grid = {(core, b, h, label) for b in range(1, 5) for h in range(1, 9) for label in RANGES}
        require(len(rows) == len(grid) == 64 and grid == expected_grid, "Core fit grid")
        for r in rows:
            require(0 < r["receivers"] <= receipt["cells"], "Included receiver count")
            if r["range"] == "all_edges":
                require(r["receivers"] == receipt["cells"], "All-edge receiver coverage")
            require(np.asarray(r["moments"]).shape == (5,), "Moment vector shape")
            compare_fit(audit, r, r["moments"], "per_core_fit_recomputed")
            audit.close(r["reference_exponent"], 2, "inverse_square_control_exponent", rtol=0, atol=1e-10)
            audit.close(r["reference_r2"], 1, "inverse_square_control_r2", rtol=0, atol=1e-10)
        for label in RANGES:
            subset = [r for r in rows if r["range"] == label]
            require(len({r["receivers"] for r in subset}) == 1, "Receiver counts differ across heads/blocks")
            audit.close(np.asarray([r["moments"][:2] for r in subset]),
                        np.tile(subset[0]["moments"][:2], (32, 1)), "distance_moments_across_heads_blocks")
        all_rows.extend(rows)

        bins_path = root / f"core_{core}_bins.npz"
        require(sha256_file(bins_path) == receipt["bins_sha256"], "Binned artifact checksum")
        checked_files.append(bins_path)
        bins_hashes[str(core)] = receipt["bins_sha256"]
        with np.load(bins_path, allow_pickle=False) as saved:
            counts, sums = saved["counts"], saved["sums"]
            require(np.array_equal(saved["bin_edges"], BINS), "Bin edges")
        require(counts.shape == (4, 50) and counts.dtype.kind in "iu" and np.all(counts >= 0), "Bin counts")
        require(sums.shape == (4, 7, 50, 8) and np.isfinite(sums).all(), "Bin sums")
        require(np.all(counts.sum(-1) == receipt["edges"]), "Bin edge coverage")
        require(np.array_equal(counts, np.broadcast_to(counts[0], counts.shape)), "Bin counts vary between blocks")
        require(np.all(sums[:, :4] >= 0), "Negative attention-channel sums")
        empty = counts == 0
        require(np.all(sums.transpose(0, 2, 1, 3)[empty] == 0), "Nonzero empty bins")
        tolerance = receipt["attention_sum_max_error"]
        audit.close(sums[:, 0].sum(1), np.full((4, 8), receipt["cells"]),
                    "model_attention_total", rtol=0, atol=tolerance * receipt["cells"] + 1e-8)
        audit.close(sums[:, 1].sum(1), np.full((4, 8), receipt["edges"]),
                    "degree_scaled_model_total", rtol=0, atol=tolerance * receipt["edges"] + 1e-7)
        audit.close(sums[:, 2].sum(1), np.full((4, 8), receipt["cells"]), "reference_attention_total")
        audit.close(sums[:, 3].sum(1), np.full((4, 8), receipt["edges"]), "degree_scaled_reference_total")
        audit.close(sums[:, 2:4], np.broadcast_to(sums[0:1, 2:4, :, 0:1], (4, 2, 50, 8)),
                    "reference_invariance_heads_blocks", rtol=0, atol=0)
        bound = np.broadcast_to(counts[:, :, None], (4, 50, 8)) * 3e-7 + 1e-8
        require(np.all(np.abs(sums[:, 4] - sums[:, 5] - sums[:, 6]) <= bound),
                "Binned content + beta identity")
        curve = np.full_like(sums, np.nan)
        np.divide(sums, counts[:, None, :, None], out=curve, where=counts[:, None, :, None] > 0)
        core_curves.append(curve)
        print(f"Verified core {core}", flush=True)

    require(replay_count > 0, "No public-forward replay receipt")
    if not pilot_only:
        require(len(all_rows) == 896, "Full row count")
        summary_path = root / "summary.json"
        summary = read_json(summary_path)
        require(summary["run_id"] == RUN and summary["checkpoint_sha256"] == CHECKPOINT_SHA, "Summary identity")
        require(summary["total_cells"] == sum(r["cells"] for r in receipts) == 246063, "Summary cells")
        require(summary["total_edges"] == sum(r["edges"] for r in receipts) == 55980536, "Summary edges")
        require(summary["primary_fit_range_um"] == [10, 450], "Primary range")
        heads = {(r["block"], r["head"], r["range"]): r for r in summary["per_head"]}
        expected_heads = {(b, h, label) for b in range(1, 5) for h in range(1, 9) for label in RANGES}
        require(len(summary["per_head"]) == len(heads) == 64 and set(heads) == expected_heads, "Head summary grid")
        for (b, h, label), record in heads.items():
            subset = [r for r in all_rows if (r["block"], r["head"], r["range"]) == (b, h, label)]
            require(len(subset) == 14, "Core contribution count")
            pooled = sum(np.asarray(r["moments"], dtype=np.float64) / 14 for r in subset)
            audit.close(record["moments"], pooled, "per_head_pooled_moments")
            compare_fit(audit, record, pooled, "per_head_fit_recomputed")
            audit.close(record["core_exponent_min"], min(r["power_exponent"] for r in subset), "core_exponent_range")
            audit.close(record["core_exponent_max"], max(r["power_exponent"] for r in subset), "core_exponent_range")
        blocks = {(r["block"], r["range"]): r for r in summary["per_block"]}
        require(len(summary["per_block"]) == len(blocks) == 8 and
                set(blocks) == {(b, label) for b in range(1, 5) for label in RANGES}, "Block summary grid")
        for (b, label), record in blocks.items():
            subset = [r for r in all_rows if (r["block"], r["range"]) == (b, label)]
            pooled = sum(np.asarray(r["moments"], dtype=np.float64) / 112 for r in subset)
            audit.close(record["moments"], pooled, "per_block_pooled_moments")
            compare_fit(audit, record, pooled, "per_block_fit_recomputed")
        curves = np.asarray(core_curves)
        curves_path = root / "curves.npz"
        with np.load(curves_path, allow_pickle=False) as saved:
            require(np.array_equal(saved["bin_edges"], BINS), "Aggregate bins")
            require(tuple(saved["channels"].tolist()) == CHANNELS, "Channel order")
            audit.close(saved["per_core"], curves, "per_core_curves_reconstructed", rtol=0, atol=0)
            for key, operation in (("mean", np.nanmean), ("core_min", np.nanmin), ("core_max", np.nanmax)):
                audit.close(saved[key], operation(curves, axis=0), f"curve_{key}_recomputed", rtol=0, atol=0)
        for name, records, keys in (("per_core_head", all_rows, ("core", "block", "head", "range")),
                                    ("per_head", summary["per_head"], ("block", "head", "range")),
                                    ("per_block", summary["per_block"], ("block", "range"))):
            csv_path = root / f"{name}.csv"
            check_csv(audit, csv_path, records, keys)
            checked_files.append(csv_path)
        checked_files.extend((summary_path, curves_path))

    result = {
        "schema": "so2_distance_attention_verification_v1", "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(), "run_id": RUN,
        "checkpoint_sha256": CHECKPOINT_SHA, "pilot_only": pilot_only,
        "verification_script_sha256": sha256_file(Path(__file__)),
        "counts": {"cores": len(cores), "fit_rows": len(all_rows), "blocks": 4, "heads": 8,
                   "fit_ranges": 2, "cells": sum(r["cells"] for r in receipts),
                   "edges": sum(r["edges"] for r in receipts), "source_files": len(source_hashes),
                   "input_manifest_bindings": input_count, "public_forward_replays": replay_count},
        "input_verification": {"extraction_receipts_match_frozen_manifests": True,
                               "large_input_files_rehashed_this_audit": rehash_inputs},
        "recorded_extraction_maxima": {"attention_sum_error": max_attention_error,
                                       "distance_cache_error_um": max_distance_error,
                                       "public_forward_absolute_error": max_replay_error},
        "maximum_absolute_differences": audit.maximum_differences,
        "source_code_sha256": source_hashes,
        "verified_artifact_sha256": {str(p.relative_to(root)): sha256_file(p) for p in checked_files},
        "checks": ["unique complete core/block/head/range grid", "checkpoint and frozen configuration checksums",
                   "source current/snapshot/training identity", "input manifests and fixed mask identity",
                   "recorded receiver attention normalization and distance/cache tolerance", "inverse-square controls",
                   "independent per-core and pooled fit recomputation", "binned count and component identities",
                   "raw and degree-scaled model/reference attention totals", "all curve channels reaggregated with stated weights",
                   "complete CSV/JSON agreement"],
        "limitations": ["Saved-moment and binned-sum audit; no independent full encoder replay.",
                        "Per-edge normalization, finite distances and public-forward agreement rely on extraction assertions and receipts.",
                        "Default input verification binds extraction-time hashes to frozen manifests without rereading large inputs.",
                        "In-sample distance association in one model/mask; not a transport or causal mechanism."]}
    if pilot_only:
        result["checks"] = [c for c in result["checks"] if c not in
                            ("complete CSV/JSON agreement", "all curve channels reaggregated with stated weights")]
        result["checks"][6] = "independent per-core fit recomputation"
    else:
        with (root / "verification.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "pilot_only", "counts", "maximum_absolute_differences")}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--rehash-inputs", action="store_true")
    arguments = parser.parse_args()
    verify(pilot_only=arguments.pilot_only, rehash_inputs=arguments.rehash_inputs)

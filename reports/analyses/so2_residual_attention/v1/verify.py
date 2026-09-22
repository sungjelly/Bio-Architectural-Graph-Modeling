"""Audit saved residual/attention statistics without repeating inference.

Run from the BAGM root with PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_residual_attention/v1/verify.py. Use --pilot-only to check
core 21 without writing a final receipt. Default input checking binds saved
extraction hashes to frozen manifests and the audited prior distance receipts;
--rehash-inputs additionally rereads the large input files.
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
SHA = "aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b"
METRICS = ("attention_to_residual", "ffn_to_post_attention", "post_attention_to_residual",
           "output_to_residual", "feature_centered_attention_to_residual")
VECTORS = ("residual", "attention", "post_attention", "ffn", "output")
BINS = np.r_[0., np.geomspace(1e-4, 1e3, 141), np.inf]
QUANTILES = {"q01": .01, "q10": .1, "q25": .25, "median": .5,
             "q75": .75, "q90": .9, "q99": .99}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


class Audit:
    def __init__(self):
        self.maximum_differences = {}

    def close(self, actual, expected, family, rtol=5e-12, atol=5e-12):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                f"{family}: invalid shape or finite values")
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.maximum_differences[family] = max(self.maximum_differences.get(family, 0.), error)
        require(np.allclose(a, b, rtol=rtol, atol=atol), f"{family}: difference {error}")


def distribution_check(d, cosine=False):
    require(set(d) == {"mean", "std", "min", "max", *QUANTILES}, "Distribution fields")
    require(np.isfinite(list(d.values())).all(), "Non-finite distribution")
    ordered = [d["min"], *[d[k] for k in QUANTILES], d["max"]]
    require(np.all(np.diff(ordered) >= -1e-12), "Quantile ordering")
    require(d["min"]-1e-12 <= d["mean"] <= d["max"]+1e-12 and d["std"] >= 0,
            "Distribution moments outside support")
    require(d["std"]**2 <= (d["max"]-d["mean"])*(d["mean"]-d["min"])+1e-9,
            "Variance exceeds bounded-support maximum")
    require(d["min"] >= (-1-1e-12 if cosine else 0), "Distribution lower support")
    if cosine:
        require(d["max"] <= 1+1e-12, "Cosine upper support")


def histogram_check(counts, d, cells):
    require(counts.sum() == cells, "Histogram coverage")
    occupied = np.flatnonzero(counts)
    require(len(occupied) > 0, "Empty histogram")
    left = np.maximum(BINS[:-1], d["min"])
    right = np.minimum(BINS[1:], d["max"])
    require(np.all(left[occupied] <= right[occupied]+1e-12), "Histogram outside recorded support")
    lower = float(np.dot(counts[occupied], left[occupied])/cells)
    upper = float(np.dot(counts[occupied], right[occupied])/cells)
    require(lower-1e-11 <= d["mean"] <= upper+1e-11, "Mean outside histogram bounds")
    second = d["std"]**2+d["mean"]**2
    lower2 = float(np.dot(counts[occupied], left[occupied]**2)/cells)
    upper2 = float(np.dot(counts[occupied], right[occupied]**2)/cells)
    require(lower2-1e-10 <= second <= upper2+1e-10, "Second moment outside histogram bounds")
    cdf = np.cumsum(counts)
    for key, q in QUANTILES.items():
        position = (cells-1)*q
        lower_rank, upper_rank = int(np.floor(position)), int(np.ceil(position))
        lo_bin = np.searchsorted(cdf, lower_rank+1)
        hi_bin = np.searchsorted(cdf, upper_rank+1)
        fraction = position-lower_rank
        lo = (1-fraction)*left[lo_bin]+fraction*left[hi_bin]
        hi = (1-fraction)*right[lo_bin]+fraction*right[hi_bin]
        require(lo-1e-11 <= d[key] <= hi+1e-11, f"{key} outside histogram rank bounds")


def ratios_from_moments(ms, centered):
    return {
        "rms_attention_to_residual": np.sqrt(ms["attention"]/ms["residual"]),
        "between_cell_rms_attention_to_residual": np.sqrt(centered["attention"]/centered["residual"]),
        "rms_ffn_to_post_attention": np.sqrt(ms["ffn"]/ms["post_attention"])}


def check_row(audit, row, counts, cells):
    require(row["cells"] == cells and set(row["distributions"]) == set(METRICS), "Core row schema")
    ms, centered = row["mean_squared_norms"], row["between_cell_mean_squared_norms"]
    require(set(ms) == set(centered) == set(VECTORS), "Vector moments schema")
    for name in VECTORS:
        require(np.isfinite(ms[name]) and np.isfinite(centered[name]) and
                ms[name] >= 0 and -1e-12 <= centered[name] <= ms[name]+1e-9, "Centered norm bounds")
    require(ms["residual"] > 0 and ms["post_attention"] > 0 and centered["residual"] > 0,
            "Zero moment denominator")
    for key, expected in ratios_from_moments(ms, centered).items():
        audit.close(row[key], expected, "core_rms_ratios")
    dot = row["mean_dot_residual_attention"]
    require(np.isfinite(dot) and dot*dot <= ms["residual"]*ms["attention"]+1e-8, "Cauchy-Schwarz bound")
    audit.close(ms["post_attention"], ms["residual"]+ms["attention"]+2*dot, "core_norm_energy_identity")
    require(0 <= row["energy_identity_relative_error"] < 1e-12, "Recorded per-cell energy tolerance")
    for moments in (ms, centered):
        a, f, y = [np.sqrt(moments[k]) for k in ("post_attention", "ffn", "output")]
        tolerance = 1e-6*max(1., a+f+y)
        require(abs(a-f)-tolerance <= y <= a+f+tolerance, "Output/FFN norm triangle bounds")
    for key in ("fraction_attention_larger", "fraction_opposing"):
        require(0 <= row[key] <= 1, "Fraction outside [0,1]")
        audit.close(row[key]*cells, round(row[key]*cells), "fraction_integer_cell_counts", rtol=0, atol=1e-8)
    require(isinstance(row["zero_attention_cells"], int) and 0 <= row["zero_attention_cells"] <= cells,
            "Zero attention count")
    distribution_check(row["cosine"], cosine=True)
    for index, metric in enumerate(METRICS):
        d = row["distributions"][metric]
        distribution_check(d)
        histogram_check(counts[index], d, cells)
    raw_counts = counts[0]
    lower_larger = raw_counts[BINS[:-1] > 1].sum()/cells
    upper_larger = raw_counts[BINS[1:] > 1].sum()/cells
    require(lower_larger-1e-12 <= row["fraction_attention_larger"] <= upper_larger+1e-12,
            "Fraction larger inconsistent with histogram")
    require(row["zero_attention_cells"] <= raw_counts[0], "Zeros absent from first ratio bin")
    require((row["zero_attention_cells"] > 0) == (row["distributions"][METRICS[0]]["min"] == 0),
            "Zero attention/minimum mismatch")


def verify(pilot_only=False, rehash_inputs=False):
    paths = current_paths()
    root = paths.report_root / "analyses/so2_residual_attention/v1"
    previous = paths.report_root / "analyses/so2_distance_attention/v1"
    bundle = paths.artifact_root / "runs/2026/09" / RUN
    cores = [21] if pilot_only else list(range(15, 29))
    receipts = [read(root/f"core_{c}.json") for c in cores]
    audit, checked = Audit(), []
    require(sha256_file(bundle/"checkpoints/last.ckpt") == SHA, "Checkpoint identity")
    artifacts = read(bundle/"provenance/artifact_checksums.json")["files"]
    config_path = bundle/"config.resolved.yaml"
    require(sha256_file(config_path) == artifacts["config.resolved.yaml"]["sha256"], "Frozen config checksum")
    config = yaml.safe_load(config_path.read_text())
    directories = {"cohort": paths.data_root/"processed/so2_14core_relative_qkv_v1",
                   "graph": paths.data_root/"processed/so2_14core_relative_qkv_graphs_v1"}
    manifests = {}
    for name, directory in directories.items():
        require(sha256_file(directory/"manifest.json") == config["dataset"][f"{name}_manifest_file_sha256"],
                f"{name} manifest checksum")
        manifests[name] = read(directory/"manifest.json")
    source_path = root/"provenance/source_hashes.json"
    source_hashes = read(source_path)
    require(len(source_hashes) == 7, "Source count")
    checked.append(source_path)
    for relative, digest in source_hashes.items():
        require(sha256_file(paths.project_root/relative) == digest, f"Current source: {relative}")
        require(sha256_file(root/"provenance/executed_source"/relative) == digest, f"Source snapshot: {relative}")
    training = read(bundle/"provenance/untracked_files.json")
    training = training if isinstance(training, list) else training["files"]
    model_source = "src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py"
    require(next(r["sha256"] for r in training if r["path"] == model_source) == source_hashes[model_source],
            "Model source differs from training")
    previous_verification = read(previous/"verification.json")
    require(previous_verification["status"] == "passed" and previous_verification["run_id"] == RUN and
            previous_verification["checkpoint_sha256"] == SHA, "Prior audited analysis identity")
    all_rows, normalized_histograms = [], []
    replay_values, reconstruction_values, message_values, energy_values = [], [], [], []
    for c, receipt in zip(cores, receipts):
        require(receipt["core"] == c and receipt["run_id"] == RUN and receipt["checkpoint_sha256"] == SHA,
                "Core receipt identity")
        require(receipt["code_hashes"] == source_hashes, "Core source binding")
        cm = next(r for r in manifests["cohort"]["cores"] if r["alias"] == f"SO2-C{c}")
        gm = next(r for r in manifests["graph"]["cores"] if r["alias"] == f"SO2-C{c}")
        require(receipt["cells"] == cm["cell_count"] == gm["n_cells"], "Cell count")
        require(receipt["edges"] == gm["graph"]["qc"]["n_directed_edges"], "Edge count")
        expected_inputs = {str((directories["cohort"]/"cores"/f"SO2-C{c}.npz").relative_to(paths.project_root)):
                           manifests["cohort"]["files"][f"cores/SO2-C{c}.npz"]}
        for filename in ("edge_index.npy", "relative_geometry.npy"):
            expected_inputs[str((directories["graph"]/"cores"/f"SO2-C{c}"/filename).relative_to(paths.project_root))] = gm["files"][filename]
        require(receipt["input_files"] == expected_inputs, "Input/manifest binding")
        if rehash_inputs:
            for relative, digest in expected_inputs.items():
                require(sha256_file(paths.project_root/relative) == digest, f"Live input: {relative}")
        prior_path = previous/f"core_{c}.json"
        require(sha256_file(prior_path) == previous_verification["verified_artifact_sha256"][prior_path.name],
                "Prior receipt changed since verification")
        prior = read(prior_path)
        require(prior["input_files"] == expected_inputs and prior["mask_seed"] == receipt["mask_seed"] and
                prior["mask_sha256"] == receipt["mask_sha256"], "Input/mask agreement with prior distance analysis")
        require(receipt["reconstruction_max_abs_error"] == 0 and
                receipt["independent_attention_update_max_abs_error"] == 0, "Recorded exact reconstruction")
        reconstruction_values.append(receipt["reconstruction_max_abs_error"])
        message_values.append(receipt["independent_attention_update_max_abs_error"])
        if receipt["pilot_public_forward_max_abs_error"] is not None:
            replay = receipt["pilot_public_forward_max_abs_error"]
            require(np.isfinite(replay) and 0 <= replay <= 1e-6, "Public-forward absolute replay check")
            replay_values.append(replay)
        rows = receipt["rows"]
        require(len(rows) == 4 and {(r["core"], r["block"]) for r in rows} == {(c,b) for b in range(1,5)},
                "Unique complete core/block grid")
        hist_path = root/f"core_{c}_histograms.npz"
        require(sha256_file(hist_path) == receipt["histograms_sha256"], "Histogram checksum")
        with np.load(hist_path, allow_pickle=False) as saved:
            counts = saved["counts"]
            require(np.array_equal(saved["ratio_bins"], BINS) and tuple(saved["metrics"].tolist()) == METRICS,
                    "Histogram bins/metric order")
        require(counts.shape == (4, 5, 142) and counts.dtype.kind in "iu" and np.all(counts >= 0),
                "Histogram shape/counts")
        require(np.all(counts.sum(-1) == receipt["cells"]), "Full histogram cell coverage")
        for row in rows:
            check_row(audit, row, counts[row["block"]-1], receipt["cells"])
            energy_values.append(row["energy_identity_relative_error"])
        all_rows.extend(rows)
        normalized_histograms.append(counts/receipt["cells"])
        checked.extend((root/f"core_{c}.json", hist_path))
        print(f"Verified core {c}", flush=True)
    require(len(replay_values) >= 1, "Missing public-forward pilot")
    mixture = sum(normalized_histograms)/len(cores)
    audit.close(mixture.sum(-1), np.ones((4,5)), "equal_core_histogram_normalization")

    if not pilot_only:
        require(len(all_rows) == 56, "Full grid coverage")
        summary_path = root/"summary.json"
        summary = read(summary_path)
        require(summary["run_id"] == RUN and summary["checkpoint_sha256"] == SHA, "Summary identity")
        require(summary["total_cells"] == sum(r["cells"] for r in receipts) == 246063, "Total cells")
        require(summary["total_edges"] == sum(r["edges"] for r in receipts) == 55980536, "Total edges")
        require(summary["weighting"] == "equal cell within core, equal core", "Weighting declaration")
        require(summary["quantiles"] == "exact within core; cross-core averages are not pooled quantiles",
                "Quantile aggregation declaration")
        blocks = {r["block"]: r for r in summary["per_block"]}
        require(len(summary["per_block"]) == len(blocks) == 4 and set(blocks) == set(range(1,5)), "Block summary grid")
        for b, record in blocks.items():
            subset = [r for r in all_rows if r["block"] == b]
            ms = {k: sum(r["mean_squared_norms"][k]/14 for r in subset) for k in VECTORS}
            centered = {k: sum(r["between_cell_mean_squared_norms"][k]/14 for r in subset) for k in VECTORS}
            for key, expected in (("mean_squared_norms",ms), ("between_cell_mean_squared_norms",centered)):
                require(set(record[key]) == set(VECTORS), "Summary moment schema")
                audit.close([record[key][k] for k in VECTORS], [expected[k] for k in VECTORS], "aggregate_norm_moments")
            pooled_dot = sum(r["mean_dot_residual_attention"]/14 for r in subset)
            audit.close(ms["post_attention"], ms["residual"]+ms["attention"]+2*pooled_dot, "aggregate_norm_energy_identity")
            for key, value in ratios_from_moments(ms, centered).items():
                audit.close(record[key], value, "aggregate_rms_ratios")
            audit.close(record["mean_cosine"], sum(r["cosine"]["mean"]/14 for r in subset), "aggregate_mean_cosine")
            for key in ("fraction_attention_larger", "fraction_opposing"):
                audit.close(record[key], sum(r[key]/14 for r in subset), "aggregate_fractions")
            for metric in METRICS:
                stats = [r["distributions"][metric] for r in subset]
                expected = {"equal_core_mean": sum(d["mean"]/14 for d in stats),
                            "core_mean_min": min(d["mean"] for d in stats), "core_mean_max": max(d["mean"] for d in stats),
                            "mean_core_median": sum(d["median"]/14 for d in stats),
                            "mean_core_q10": sum(d["q10"]/14 for d in stats), "mean_core_q90": sum(d["q90"]/14 for d in stats)}
                require(set(record[metric]) == set(expected), "Summary distribution schema")
                audit.close(list(record[metric].values()), [expected[k] for k in record[metric]], "aggregate_distribution_summaries")
        csv_path = root/"per_core_block.csv"
        with csv_path.open() as stream:
            table = list(csv.DictReader(stream))
        lookup = {(r["core"],r["block"]): r for r in all_rows}
        require(len(table) == len(lookup) == 56, "CSV row count")
        seen = set()
        for row in table:
            key = int(row["core"]),int(row["block"])
            require(key in lookup and key not in seen, "CSV key coverage")
            seen.add(key)
            source = lookup[key]
            expected = {k: source[k] for k in ("core","block","cells","rms_attention_to_residual",
                        "between_cell_rms_attention_to_residual","rms_ffn_to_post_attention",
                        "fraction_attention_larger","fraction_opposing")}
            expected["mean_cosine"] = source["cosine"]["mean"]
            for metric in METRICS:
                expected.update({f"{metric}_{k}":v for k,v in source["distributions"][metric].items()})
            require(set(row) == set(expected), "CSV columns")
            audit.close([float(row[k]) for k in expected], list(expected.values()), "csv_vs_receipts", rtol=0, atol=0)
        checked.extend((summary_path,csv_path))

    result = {
        "schema":"so2_residual_attention_verification_v1", "status":"passed", "pilot_only":pilot_only,
        "created_at":datetime.now(timezone.utc).isoformat(), "run_id":RUN,"checkpoint_sha256":SHA,
        "verification_script_sha256":sha256_file(Path(__file__)),
        "counts":{"cores":len(cores),"core_block_rows":len(all_rows),"blocks":4,
                  "cells":sum(r["cells"] for r in receipts),"edges":sum(r["edges"] for r in receipts),
                  "histogram_metrics":5,"histogram_bins":142,"source_files":7,"input_manifest_bindings":len(cores)*3,
                  "public_forward_replays":len(replay_values),"zero_attention_cell_block_observations":sum(r["zero_attention_cells"] for r in all_rows)},
        "recorded_extraction_maxima":{"block_reconstruction_absolute_error":max(reconstruction_values),
                                      "independent_weighted_message_absolute_error":max(message_values),
                                      "public_forward_absolute_error":max(replay_values),
                                      "per_cell_energy_relative_error":max(energy_values)},
        "maximum_absolute_differences":audit.maximum_differences,
        "input_files_rehashed":rehash_inputs,"source_code_sha256":source_hashes,
        "prior_distance_verification_sha256":sha256_file(previous/"verification.json"),
        "verified_artifact_sha256":{str(p.relative_to(root)):sha256_file(p) for p in checked},
        "checks":["complete unique core/block grid","checkpoint/configuration and source snapshot/training binding",
                  "input manifest and audited prior input/mask binding","recorded exact branch/message reconstruction",
                  "public-forward replay","norm ratios, h/u dot-product energy identity and Cauchy-Schwarz bounds",
                  "centered-moment bounds and FFN/output triangle bounds","finite ordered distribution statistics",
                  "full histogram counts, bins and ordering","histogram-compatible moments, quantiles and larger-update fractions",
                  "equal-core histogram mixture normalization"],
        "limitations":["Saved-statistic audit; no new full model inference and no raw activations persisted.",
                       "Exact within-core centered variances and quantiles cannot be recomputed from scalar summaries; their bounds and histogram consistency are checked.",
                       "No aggregate histogram file is emitted by analyze.py; this audit reconstructs and checks the equal-core mixture from all core histograms.",
                       "By default, large input bytes are not reread; extraction-time hashes match frozen manifests and audited prior receipts.",
                       "Mean core quantiles are not pooled quantiles. Zero-update cosine is recorded as zero by the extractor.",
                       "Raw norms measure branch magnitude; centered variation is not information or predictive necessity."]}
    if not pilot_only:
        result["checks"].extend(("independent equal-core summary recomputation","CSV exact agreement"))
        with (root/"verification.json").open("x") as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status","pilot_only","counts","maximum_absolute_differences")},indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only",action="store_true")
    parser.add_argument("--rehash-inputs",action="store_true")
    args = parser.parse_args()
    verify(pilot_only=args.pilot_only,rehash_inputs=args.rehash_inputs)

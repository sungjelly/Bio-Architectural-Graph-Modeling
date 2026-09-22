#!/usr/bin/env python3
"""Build the exploratory SO2 diagnostic synthesis from verified local reports."""
from __future__ import annotations
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file

P = current_paths()
RID = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
BASE = P.report_root / "analyses/so2_failure_diagnosis/baselines/v1"
OUT = P.scratch_root / "active_runs" / RID / "posthoc_reports/so2_failure_comparison/v1"
FINAL = P.report_root / "analyses/so2_failure_diagnosis/comparison/v1"
RUNS = {
    "original": "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6",
    "continued": "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2",
    "recurrent": "r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf",
    "geometry": RID,
}
LABELS = {"gene_mean": "Gene mean (all fit)", "zero_count": "Always zero", "huber_constant": "Huber constant (all fit)",
    "local_log_mean": "Local 16: mean log1p", "local_count_mean": "Local 16: mean count", "full_graph_log_mean": "Full graph: mean log1p",
    "original": "Original e175", "continued": "Continued e300", "recurrent": "Recurrent e175", "geometry": "Geometry e200"}
SPECS = {"all_huber": ("all", "huber_standardized"), "all_mse_z": ("all", "mse_standardized"),
    "positive_mse_z": ("positive", "mse_standardized"), "positive_mse_log1p": ("positive", "mse_log1p"),
    "zero_mse_z": ("zero", "mse_standardized")}

def read(path): return json.loads(path.read_text())
def write(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
def table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def verify(root):
    manifest = read(root / "manifest.json")
    for rel, info in manifest["files"].items(): assert sha256_file(root / rel) == info["sha256"], rel
    assert sha256_file(root / "manifest.json") == read(root / "_SUCCESS")["manifest_sha256"]

def build():
    assert OUT.exists() and not FINAL.exists()
    verify(BASE)
    sources = {str((BASE / "manifest.json").relative_to(P.report_root)): sha256_file(BASE / "manifest.json")}
    by_core = {}
    for name in LABELS:
        by_core[name] = []
        if name in RUNS:
            root = P.report_root / "analyses/so2_nonzero_metrics" / RUNS[name] / "v1"
            verify(root); sources[str((root / "manifest.json").relative_to(P.report_root))] = sha256_file(root / "manifest.json")
        for core in range(15, 29):
            r = read((root if name in RUNS else BASE) / f"core_{core}.json")
            m = r["metrics"]["model" if name in RUNS else name]
            by_core[name].append({"core": core, "predictor": name,
                **{key: m["strata"][s][metric] for key, (s, metric) in SPECS.items()},
                "positive_recall": m["detection"]["positive_recall"], "zero_specificity": m["detection"]["zero_specificity"]})
    aggregate = {name: {key: float(np.mean([r[key] for r in rows])) for key in (*SPECS, "positive_recall", "zero_specificity")}
                 for name, rows in by_core.items()}
    table(OUT / "matched_per_core.csv", [r for rows in by_core.values() for r in rows])
    table(OUT / "matched_summary.csv", [{"predictor": n, "label": LABELS[n], **a} for n, a in aggregate.items()])
    pairs = []
    for model in RUNS:
        for baseline in ("gene_mean", "huber_constant", "local_log_mean", "local_count_mean", "full_graph_log_mean"):
            for metric in SPECS:
                pairs.append({"model": model, "baseline": baseline, "metric": metric,
                    "model_minus_baseline": aggregate[model][metric] - aggregate[baseline][metric],
                    "model_relative_difference_pct": 100*(aggregate[model][metric]/aggregate[baseline][metric]-1),
                    "model_lower_error_cores": sum(a[metric] < b[metric] for a,b in zip(by_core[model], by_core[baseline], strict=True)),
                    "cores": 14})
    table(OUT / "paired_differences.csv", pairs)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, (("all_huber", "All entries: training Huber objective"),
        ("all_mse_z", "All entries: standardized MSE"), ("positive_mse_z", "Observed positive entries: standardized MSE"),
        ("positive_mse_log1p", "Observed positive entries: log1p MSE")), strict=True):
        for i, name in enumerate(LABELS):
            color = "#c44938" if name == "geometry" else "#21658b" if name in RUNS else "#69787f"
            ax.scatter([r[metric] for r in by_core[name]], np.full(14,i), color=color, alpha=.20, s=17, edgecolors="none")
            ax.scatter(aggregate[name][metric], i, color=color, marker="D", s=38, zorder=3)
        ax.set_yticks(range(len(LABELS)), LABELS.values(), fontsize=9); ax.invert_yaxis()
        ax.set_title(title, fontsize=11, loc="left"); ax.set_xlabel("Error (lower is better)")
        ax.grid(axis="x", alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Matched masks reveal objective and zero/positive tradeoffs\nDots: 14 fitted cores · diamonds: equal-core means · red: geometry", fontsize=14)
    fig.savefig(OUT / "diagnostic_comparison.png", dpi=180); fig.savefig(OUT / "diagnostic_comparison.pdf"); plt.close(fig)
    a = aggregate; const = read(BASE / "constant_fit.json")
    geometry_bias = float(np.mean([read(P.report_root / "analyses/so2_nonzero_metrics" / RID / "v1" / f"core_{c}.json")["metrics"]["model"]["strata"]["all"]["bias_standardized"] for c in range(15,29)]))
    rows = "\n".join(f"| {LABELS[n]} | {a[n]['all_huber']:.4f} | {a[n]['all_mse_z']:.4f} | {a[n]['positive_mse_z']:.4f} | {a[n]['positive_mse_log1p']:.4f} |" for n in LABELS)
    def improvement(model, base, metric): return 100*(1-a[model][metric]/a[base][metric])
    masked = [read(BASE / f"core_{c}.json") for c in range(15,29)]
    table(OUT / "fallback_frequency.csv", [{"predictor": name,
        "masked_fallback_entries": sum(r["fallback_masked_entries"][name] for r in masked),
        "masked_entries": sum(r["n_masked_entries"] for r in masked),
        "equal_core_fallback_fraction": float(np.mean([r["fallback_masked_entries"][name]/r["n_masked_entries"] for r in masked]))}
        for name in ("local_log_mean", "local_count_mean", "full_graph_log_mean")])
    mask_fraction = np.mean([r["entry_weighted_mask_fraction"] for r in masked])
    hard_fraction = np.mean([r["scored_entries_from_ge900_masked"]/r["n_masked_entries"] for r in masked])
    write(OUT / "synthesis_metrics.json", {"aggregate": a, "geometry_all_entry_bias_z": geometry_bias, "entry_weighted_mask_fraction_equal_core": mask_fraction,
        "entries_from_cells_masking_ge900_equal_core": hard_fraction, "source_report_manifests": sources})
    external = (OUT / "external_comparison.md").read_text().replace("# External reconstruction-model comparison", "## External reconstruction-model comparison", 1)
    report = f"""# Why SO2 reconstruction gains remain limited

Exploratory diagnostic and literature comparison · 2026-09-06. All local numerical
comparisons below use the same 14 fitted cores and original hidden-entry masks.

The evidence supports an **objective/metric mismatch**, and leaves several
architectural explanations unresolved. Geometry is active, but has not improved
the main positive-error comparison over the original lineage. This is not evidence
that graph models in general fail or that changing to NB will fix the problem.

![Errors on four scales for all four endpoints and six fixed baselines. Dots show individual fitted cores; diamonds show equal-core means.](diagnostic_comparison.png)

## Matched performance: the missing simple baselines

All errors below are lower-is-better. Positive means observed raw count >0,
not positive standardized expression. These are distinct metrics, not accuracy
percentages. Full numerical tables and per-core paired differences are included.

| Predictor | All-entry Huber | All-entry z-MSE | Positive z-MSE | Positive log1p MSE |
|---|---:|---:|---:|---:|
{rows}

The spatial baselines average **only observed neighboring values**, including
observed zeros. Local baselines use up to 16 nearest cells within 75 µm. Full-graph
averaging uses every incoming edge of the exact original graph, about 227.5
neighbors/cell globally. A missing neighborhood falls back to the observed core
gene mean, then zero; hidden entries never enter either fallback. The log-mean
and raw-count-mean baselines differ by averaging scale, so their metric tradeoff
must not be attributed solely to spatial scale. They are fixed, untuned references.

**The model beats naive averaging overall, but not on every positive-only metric.**
Geometry's all-entry standardized MSE is 3.17% lower than the original-full-graph
mean, with lower error in 14/14 cores. Its positive standardized MSE is 3.79%
higher than that mean, with higher error in 12/14 cores; its positive log1p MSE
is nevertheless 1.49% lower. The local raw-count average lowers positive
standardized MSE by 11.40% and positive log1p MSE by 10.30% relative to geometry,
in both cases with lower error in all 14 cores. However, its all-entry standardized
MSE rises to 1.2426 versus geometry's 0.9516, and zero-entry MSE rises to 0.4886
versus 0.0390. Thus it exchanges better positive predictions for much worse zero
predictions. These results do not support an unqualified "averaging is better"
claim, or establish that learned attention is oversmoothing.
Per-baseline fallback support is recorded in [fallback_frequency.csv](fallback_frequency.csv).

The constant gene mean and Huber optimum use the all-fit target distribution.
They are descriptive target-derived references, not leakage-free held-out models.
The four neural predictions were replayed and verified in the earlier evaluation;
this analysis reuses those immutable metrics and introduces no new neural fitting.

Geometry improves all-entry Huber over the Huber constant by
{improvement('geometry','huber_constant','all_huber'):.2f}%, compared with
{improvement('geometry','gene_mean','all_huber'):.2f}% over the gene mean.
Its relative error differences against each spatial reference, including signs
and all 14 paired core results, are in [paired_differences.csv](paired_differences.csv).
No single metric establishes overall superiority: lower positive error can be
accompanied by worse zero error and worse all-entry performance.

## What the loss test establishes

For each gene, the constant that minimizes Huber solves
`E[clip(prediction - standardized_log_count, -1, 1)] = 0`.
The MSE-optimal constant is the mean. On our equal-core fitted distributions,
**{const['below_mean_count']}/1,000 Huber-optimal constants fall below their means**;
their average standardized shift is {const['mean_huber_z']:.4f}. The largest
stationarity error is {const['max_stationarity_error']:.2g}; every solution has
Huber risk no worse than either reference constant. This is direct mathematical
and empirical evidence of loss-induced shrinkage in the constant predictor class.
The actual geometry endpoint also has a mean signed error of {geometry_bias:.4f}
over **all masked entries** on the standardized-log scale. This documents a downward
mean bias on that scale, beyond the outcome-conditioned positive-only diagnostic;
it is consistent with, but does not causally isolate, the objective's preference.

It does not prove that the trained graph model has reached its conditional Huber
optimum, or that Huber is wrong for every scientific objective. Huber is deliberately
robust to large residuals. Our implementation applies it to gene-standardized
log1p counts, so three choices affect the target: logarithmic compression,
inverse-gene-scale weighting, and clipping of residual derivatives beyond one
standardized unit. That objective is not designed to estimate expected raw counts
or classify whether a particular noisy count observation will be nonzero.

Even before fitting the new constants, always-zero prediction had lower Huber
than the gene mean (0.263637 versus 0.270998), while having worse standardized
MSE (1.110121 versus 0.999797). Thus the disagreement exists without a decoder,
attention, or graph. Geometry's observed positives contribute about 96.37% of
standardized squared error and 93.47% of Huber loss. **Zeros dominate entry counts,
not the measured error sums.** Neither loss share measures parameter-gradient
share; the required per-entry Jacobians and clipped residuals were not collected.

Negative bias conditional on observed positives is not alone proof of bad
calibration. For example, a correct conditional mean for a count that is one with
probability 0.1 and zero otherwise is 0.1; rounding yields zero even on the positive
outcomes. Conversely, positive-only MSE can favor overprediction on zeros. This is
why rounded-count recall and exact count matching cannot serve as the sole gate.

## Architecture, data exposure and optimization

| Endpoint | Parameters | Backbone difference | Epochs / optimizer updates |
|---|---:|---|---:|
| Original | 5,003,016 | Four independent relative-QKV blocks | 175 / 1,225 |
| Continued | 5,003,016 | Same original checkpoint lineage resumed | 300 total / 2,100 total |
| Recurrent | 2,605,680 | One entire block reused four times | 175 / 1,225 |
| Geometry | 5,134,088 | Normalized Q/K, learned scale, dimension modulation and bounded bias | 200 / 1,400 |

All use 246,063 cells, 1,000 genes, 22 morphology/imaging covariates, a 256-wide
cell state, eight heads, 1,024-wide FFNs and a **256→1,024→1,000 nonlinear decoder**.
The decoder has ordinary learned linear outputs, not a count distribution or
zero/nonzero head. Its two linear layers contain 1,288,168 parameters, about one
quarter of the geometry model. It is not visibly a tiny output bottleneck: the cell representation
is narrower, but a 256-dimensional embedding is not by itself evidence of insufficient
capacity. These sizes neither prove nor rule out a decoder bottleneck. A successful
trained probe can reveal predictive information unused by the current decoder;
an unsuccessful probe cannot establish that the embedding contains no useful information.

The graph has 55,980,536 directed edges, no self/cross-core edges, and mixed radial
shells out to 500 µm. Four message passes can mix local and broad context. Attention
averages learned value vectors, followed by residual and nonlinear updates; it is
not identical to averaging raw neighboring gene measurements. Values derive from
the whole node state (expression, mask, morphology and earlier context); edge
geometry does not enter the value projection directly.

Mask counts are uniform from 0 through 1,000 per cell, with ten masks/core/epoch.
Because the loss scores entries, heavily masked cells contribute more targets.
Measured across the fixed masks, a scored entry belongs to a cell with
**{100*mask_fraction:.2f}% of genes hidden on average**; **{100*hard_fraction:.2f}%**
of scored entries come from cells hiding at least 900 genes. Both senders and
receivers are masked. This is a more difficult information budget than a flat
"50% masked" description implies, and differs from whole-gene panel completion
or fully observed autoencoder reconstruction in much of the literature. Reducing
evaluation masking changes the available information; it cannot be counted as an architecture
improvement. A training-mask curriculum must be tested on the same fixed endpoints.

There are only seven optimizer updates per epoch, each averaging 20 full-core
mask-view losses. Epoch counts therefore do not match minibatch epochs in other
papers. Training losses decrease and all graph blocks receive gradients. Geometry
and recurrent logged preclip gradient maxima remain below clip norm 1, so clipping
is not restricting the recorded updates. Continued training gives small gains,
which prevents treating the early plateau as a proven optimum. Near-plateau gradient
oscillation is a possible optimization issue, not an established cause.

Existing geometry audits show across-edge modulation RMS of 0.139–0.165 and
geometry-bias variation comparable to content-score variation. This contradicts
"geometry never activated," but does not show useful routing. Existing hidden-layer
maps used fully observed inputs, so they cannot establish or exclude oversmoothing
under these heavy reconstruction masks. Attention entropy, effective neighbor count,
message/residual ratios and masked-layer contraction remain unmeasured.

## Ranked diagnosis and the experiment that could change it

| Candidate explanation | Assessment from current evidence | Discriminating test |
|---|---|---|
| Loss/scale differs from desired count reconstruction | Strongly supported mismatch; neural causal effect remains untested | Hold backbone/data/budget fixed; compare Huber-log, MSE-log, Poisson and NB count heads on common metrics |
| Masking leaves too little information for rare counts | Directly documented difficult information budget; information sufficiency unknown | Fixed 15%, 40%, 70%, 90% masks plus whole-node masking; report abundance and mask strata separately |
| Broad graph dilutes local information | Plausible from degree/range; raw averaging results alone cannot establish oversmoothing | Train cell-only, local-graph and multiscale/self-plus-context controls; then measure masked-layer contraction |
| Decoder discards useful latent information | Possible, currently weak direct evidence | Compare frozen-embedding linear/ridge and modest MLP probes with the current decoder on disjoint masks/samples |
| Insufficient optimization or update budget | Small continued gains and possible plateau oscillation; no evidence of catastrophic gradient failure | Small-subset overfit test and matched optimizer-update/LR schedules, before another long full run |
| Too little data or biological diversity | Far less atlas breadth than foundation models; no controlled scaling evidence | Within-task learning curves and grouped patient validation; do not substitute cell count for independent patients |
| Implementation/masking bug | Tested core masking, target transform, checksums and metric replay pass; no identified fault | Retain synthetic recovery, leakage tests and identity/overfit controls; do not claim every possible bug excluded |

NB would alter the output head and likelihood: decode a positive expected count
µ and a positive dispersion θ, optimize raw-count negative log likelihood, and
evaluate its mean/probabilities/intervals. The graph encoder can initially stay
unchanged. NB already assigns probability to zero; add a hurdle or extra-zero
component only if held-out calibration supports it. Library size must be known
independently of hidden targets or inferred solely from permitted observations;
the true total contains hidden-target information even under partial masking.
Log/normalized encoder inputs can coexist with raw-count
likelihood targets. NB can improve likelihood without improving positive-only MSE,
and its dispersion can absorb misspecification, so common point metrics and
calibration must accompany NLL. Raw Huber and NLL values are not comparable scores.

The highest-value next neural experiment is a matched **cell-only versus current
graph × Huber versus MSE** pilot, retaining the same standardized-log targets and
decoder, fixed inputs, equal optimizer updates, repeat seeds and patient-grouped
validation. This isolates the robust-loss choice from graph inclusion. Follow with
NB/Poisson count-head conditions on the same encoders: those jointly change the
output scale, head and likelihood, so their effect cannot be attributed to loss alone.
Use a fixed Poisson deviance on decoded count predictions as a common point metric,
NLL for probabilistic predictions on identical counts/support with full likelihood
constants, per-gene correlations, zero/positive error strata,
abundance strata, count-probability calibration and patient-level uncertainty.
If the cell-only model matches the graph, prioritize proving usable spatial signal
before adding more geometric attention parameters. A decoder probe and small-subset
overfit check cheaply separate representation limitations from optimizer/readout issues.
These proposed neural experiments were not executed as part of this CPU diagnostic.

{external}

## Evidence limits and verification

This is an all-fit transductive comparison of one trained seed and one fixed mask
per core. Original and continued share a lineage; training durations differ.
All cores are shown, without selecting favorable seeds, metrics or regions. There
is no patient generalization estimate, graph-specific neural gain, uncertainty
calibration, attribution faithfulness, realistic spatial null, independent biological
support or perturbational evidence. Fourteen cores are not assumed to be fourteen
independent patients. Observed counts are noisy measurements, not latent ground truth.

The recurrent source completed training but failed artifact finalization; its frozen
checkpoint and subsequent metric replay were independently verified. Its pre-existing
checkpoint-catalog gap remains recorded. No failed run was silently relabeled.

Four GPUs were occupied by independent training, which was left undisturbed. CPU
baselines use exact original masks and checksum-bound counts/graphs. The helper and
existing metric suites passed 29 tests. Numerical acceptance includes both pilot cores,
all 14 complete receipts, risk/stationarity checks, baseline replay, support partitions,
finite values, source hashes, independent summary reconstruction and registry audit.
See [independent_verification.json](independent_verification.json), the baseline
[manifest](../../baselines/v1/manifest.json), and [source_ledger.json](source_ledger.json).

Maximum defensible conclusion: current fitted-cohort endpoints show modest,
metric-dependent reconstruction gains. The training objective demonstrably favors
lower constant predictions than MSE, and the new spatial references expose the
remaining tradeoffs. The evidence does not yet isolate a single neural failure
cause or justify calling NB, a larger decoder, or a different graph a guaranteed fix.
"""
    (OUT / "report.md").write_text(report)
    diagnostic_sources = []
    for rel in ("analyses/so2_attention_tau_beta/v1/README.md", "analyses/so2_attention_tau_beta/v1/per_head.csv",
                "analyses/so2_geometry_modulation/v1/README.md", "analyses/so2_geometry_modulation/v1/summary.json",
                "analyses/so2_epoch_comparison/v1/comparison.md", "analyses/so2_epoch_comparison/v1/per_epoch_comparison.csv"):
        diagnostic_sources.append({"root": "report", "path": rel, "sha256": sha256_file(P.report_root / rel)})
    gradient_audit = {}
    for name in ("geometry", "recurrent"):
        rid = RUNS[name]; root = P.artifact_root / "runs" / rid[2:6] / rid[6:8] / rid
        rel = "results/gradient_direction_metrics.csv"; f = root / rel
        checks = read(root / "provenance/artifact_checksums.json")["files"]
        assert sha256_file(f) == checks[rel]["sha256"]
        diagnostic_sources.append({"root": "artifact", "path": str(f.relative_to(P.artifact_root)), "sha256": sha256_file(f)})
        with f.open() as stream: grad = list(csv.DictReader(stream))
        gradient_audit[name] = {"epochs": len(grad), "max_preclip_norm": max(float(r["gradient_norm_max_before_clip"]) for r in grad),
            "mean_last25_epoch_gradient_cosine": float(np.mean([float(r["epoch_aggregate_gradient_cosine_to_previous_epoch"]) for r in grad[-25:]]))}
        assert gradient_audit[name]["max_preclip_norm"] < 1
    write(OUT / "optimization_audit.json", gradient_audit)
    write(OUT / "local_sources.json", {"report_manifests": sources,
        "diagnostic_sources": diagnostic_sources,
        "campaign": "cmp_20260906_so2_failure_diagnosis", "source_run_ids": list(RUNS.values()),
        "created_at": datetime.now(timezone.utc).isoformat()})
    shutil.copyfile(Path(__file__), OUT / "report_builder.py")
    print(str(OUT / "report.md"))

if __name__ == "__main__": build()

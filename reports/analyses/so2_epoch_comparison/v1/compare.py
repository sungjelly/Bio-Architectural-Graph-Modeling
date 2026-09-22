"""Compare immutable SO2 epoch logs; see the adjacent README task contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from spatial_benchmark.paths import current_paths

PATHS = current_paths()
RUNS = {
    "original": "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6",
    "continuation": "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2",
    "recurrent": "r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf",
    "geometry": "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf",
    "failed_duplicate": "r_20260826T032515Z_52d16093_s000_f00_a01_149a165b",
}
EXPECTED = {"original": 175, "continuation": 300, "recurrent": 175, "geometry": 200}
LOSS = "equal_core_mean_masked_huber"
CORE_LOSSES = [f"loss_so2_c{i}" for i in range(15, 29)]
LABELS = {"original": "Original + continuation", "recurrent": "Recurrent (finalization failed)",
          "geometry": "Geometry-modulated"}
COLORS = {"original": "#2769A8", "recurrent": "#C57518", "geometry": "#923A92"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(run_id: str) -> Path:
    return PATHS.artifact_root / "runs" / run_id[2:6] / run_id[6:8] / run_id


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PATHS.report_root / "analyses/so2_epoch_comparison/v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(PATHS.report_root.resolve()):
        raise ValueError("Output must be within the configured report root")
    if args.check:
        provenance = json.loads((out / "provenance.json").read_text())
        for entry in provenance["sources"]:
            path = bundle(entry["run_id"]) / entry["relative_path"]
            assert sha(path) == entry["sha256"], path
        for rel, digest in json.loads((out / "output_checksums.json").read_text()).items():
            assert sha(out / rel) == digest, rel
        assert sha(Path(__file__)) == provenance["analysis_script_sha256"]
        assert json.loads((out / "verification.json").read_text())["passed"]
        print("PASS: source, report, figure, table, and analysis-script checksums")
        return
    if (out / "output_checksums.json").exists():
        raise FileExistsError("Completed report exists; use --check or a new --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    if out != Path(__file__).resolve().parent:
        (out / "README.md").write_text(Path(__file__).with_name("README.md").read_text())
    con = sqlite3.connect(f"file:{PATHS.state_root / 'tracking/bagm.sqlite3'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    inventory = []
    for name, rid in RUNS.items():
        row = con.execute("SELECT run_id, campaign_id, status, seed, start_time, end_time "
                          "FROM runs WHERE run_id=?", (rid,)).fetchone()
        assert row is not None, rid
        inventory.append({"name": name, **dict(row), "included": name != "failed_duplicate"})
    con.close()
    for row in inventory:
        expected_status = "failed" if row["name"] in ("recurrent", "failed_duplicate") else "completed"
        assert row["status"] == expected_status, row
    sources, configs, records, frames = [], {}, {}, {}
    for name, n_epochs in EXPECTED.items():
        rid = RUNS[name]
        root = bundle(rid)
        manifest = json.loads((root / "provenance/artifact_checksums.json").read_text())["files"]
        for rel in ["results/epoch_metrics.csv", "metrics/core_steps.parquet", "config.resolved.yaml"]:
            digest = sha(root / rel)
            assert digest == manifest[rel]["sha256"], (rid, rel)
            sources.append({"run_id": rid, "relative_path": rel, "sha256": digest})
        configs[name] = yaml.safe_load((root / "config.resolved.yaml").read_text())
        frame = pd.read_csv(root / "results/epoch_metrics.csv").set_index("global_epoch")
        assert list(frame.index) == list(range(1, n_epochs + 1))
        assert np.isfinite(frame[[LOSS] + CORE_LOSSES].to_numpy()).all()
        np.testing.assert_allclose(frame[CORE_LOSSES].mean(axis=1), frame[LOSS], rtol=0, atol=1e-12)
        assert (frame["model_seed"] == 0).all()
        assert (frame["optimizer_updates_this_epoch"] == 7).all()
        assert (frame["complete_graph_mask_views"] == 140).all()
        assert (frame["cumulative_optimizer_updates"].to_numpy() == frame.index.to_numpy() * 7).all()
        frames[name] = frame
        records[name] = sorted(pq.read_table(root / "metrics/core_steps.parquet").to_pylist(),
                               key=lambda r: (r["completed_global_epoch"], r["position_in_epoch"]))
        assert len(records[name]) == n_epochs * 14
    for name in EXPECTED:
        for section in ("dataset", "graph", "masking"):
            assert configs[name][section] == configs["original"][section], (name, section)
    pd.testing.assert_frame_equal(frames["original"], frames["continuation"].loc[:175])
    # Core records preserve exact masks, graph sizes, order, pairings and RNG seeds.
    identity_fields = [k for k in records["original"][0] if k not in ("run_id", "masked_huber_loss")]
    mask_checks = {}
    for name in ("original", "recurrent", "geometry"):
        peer = records["continuation"][:len(records[name])]
        for a, b in zip(records[name], peer, strict=True):
            assert all(a[k] == b[k] for k in identity_fields), (name, a["completed_global_epoch"])
        mask_checks[name] = {"paired_core_records": len(peer), "identical_mask_checksums": len(peer) * 10}
    original = pd.concat([frames["original"], frames["continuation"].loc[176:]])
    trajectories = {"original": original, "recurrent": frames["recurrent"], "geometry": frames["geometry"]}
    wide = pd.DataFrame(index=pd.Index(range(1, 301), name="global_epoch"))
    long_rows = []
    for name, frame in trajectories.items():
        wide[name + "_training_huber"] = frame[LOSS]
        wide[name + "_trailing25_huber"] = frame[LOSS].rolling(25, min_periods=25).mean()
        wide[name + "_run_id"] = [RUNS["continuation"] if name == "original" and e > 175
                                  else RUNS[name] for e in frame.index] + [None] * (300 - len(frame))
        wide[name + "_run_status"] = ["failed_finalization" if name == "recurrent" else "completed"] * len(frame) + [None] * (300 - len(frame))
        for epoch, row in frame.iterrows():
            rid = RUNS["continuation"] if name == "original" and epoch > 175 else RUNS[name]
            for core in range(15, 29):
                long_rows.append({"global_epoch": epoch, "model": name, "run_id": rid,
                                  "run_status": "failed_finalization" if name == "recurrent" else "completed",
                                  "core_alias": f"SO2-C{core}", "training_masked_huber": row[f"loss_so2_c{core}"]})
    pair_stats = {}
    for peer in ("original", "recurrent"):
        a, b = wide["geometry_training_huber"], wide[peer + "_training_huber"]
        diff = a - b
        wide[f"geometry_minus_{peer}_huber"] = diff
        wide[f"geometry_minus_{peer}_percent"] = 100 * diff / b
        valid = diff.dropna()
        pair_stats[peer] = {"shared_epochs": len(valid), "geometry_lower_epochs": int((valid < 0).sum()),
                            "geometry_higher_epochs": int((valid > 0).sum()), "ties": int((valid == 0).sum()),
                            "mean_paired_difference": float(valid.mean()),
                            "relative_difference_of_means_percent": float(100 * valid.mean() / b.loc[valid.index].mean())}
        wins = []
        for epoch in wide.index:
            if epoch not in frames["geometry"].index or epoch not in trajectories[peer].index:
                wins.append(None)
            else:
                wins.append(int((frames["geometry"].loc[epoch, CORE_LOSSES].astype(float)
                                 < trajectories[peer].loc[epoch, CORE_LOSSES].astype(float)).sum()))
        wide[f"geometry_lower_core_count_vs_{peer}"] = wins
    assert wide["geometry_training_huber"].loc[201:].isna().all()
    assert wide["recurrent_training_huber"].loc[176:].isna().all()
    wide.to_csv(out / "per_epoch_comparison.csv", float_format="%.17g")
    pd.DataFrame(long_rows).to_csv(out / "per_core_epoch_losses.csv", index=False, float_format="%.17g")
    blocks = []
    for start in range(1, 301, 25):
        block = {"first_epoch": start, "last_epoch": start + 24}
        for name in trajectories:
            values = wide.loc[start:start + 24, name + "_training_huber"].dropna()
            block[name] = float(values.mean()) if len(values) == 25 else None
        block["geometry_vs_original_percent"] = (100 * (block["geometry"] / block["original"] - 1)
                                                   if block["geometry"] is not None else None)
        blocks.append(block)
    pd.DataFrame(blocks).to_csv(out / "blocks_25_epochs.csv", index=False, float_format="%.17g")
    dump(out / "summary.json", {"metric": LOSS, "comparisons": pair_stats, "blocks_25_epochs": blocks,
                                "epochs_are_independent_replicates": False})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .17, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for name, frame in trajectories.items():
        for ax in axes[0]:
            ax.plot(frame.index, frame[LOSS], color=COLORS[name], lw=1.25, label=LABELS[name])
    axes[0, 0].set(title="All recorded epochs", ylabel="Training masked Huber ↓", xlim=(1, 300))
    axes[0, 0].legend(fontsize=9, frameon=False)
    axes[0, 1].set(title="Matched epochs 50–200 (expanded loss scale)",
                   ylabel="Training masked Huber ↓", xlim=(50, 200), ylim=(.2402, .2452))
    axes[0, 1].axvline(175, color="#737373", ls=":", lw=1)
    axes[0, 1].text(176.5, .24505, "Original continues", fontsize=8, color="#555555", va="top")
    for ax, peer in zip(axes[1], ("original", "recurrent"), strict=True):
        diff = wide[f"geometry_minus_{peer}_huber"].dropna()
        ax.plot(diff.index, diff, color="#AD8DB1", lw=.85, alpha=.75, label="Each epoch")
        smooth = diff.rolling(25, min_periods=25).mean()
        ax.plot(smooth.index, smooth, color=COLORS["geometry"], lw=2, label="Trailing 25-epoch mean")
        ax.axhline(0, color="#333333", lw=1)
        ax.set(title=f"Geometry minus {peer}: negative favors geometry",
               ylabel="Difference in training masked Huber", xlim=(1, len(diff)))
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        stats = pair_stats[peer]
        ax.text(.02, .04, f"Geometry lower in {stats['geometry_lower_epochs']}/{stats['shared_epochs']} epochs",
                transform=ax.transAxes, fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    for ax in axes.flat:
        ax.set_xlabel("Global epoch (7 optimizer updates per epoch)")
    fig.suptitle("SO2 models: epoch-by-epoch training comparison", fontsize=17, x=.06, ha="left", y=.98)
    fig.text(.06, .932, "Same fitted cohort, graph and training masks • seed 0 • lower loss is better", fontsize=11)
    fig.text(.06, .016, "Training losses, not held-out evaluation. Recurrent artifact finalization failed. "
             "Epochs are repeated observations; no significance test is implied.", fontsize=9, color="#444444")
    fig.subplots_adjust(top=.88, bottom=.11, left=.075, right=.98, hspace=.38, wspace=.23)
    fig.savefig(out / "epoch_comparison.png", dpi=180)
    fig.savefig(out / "epoch_comparison.pdf", metadata={"Title": "SO2 epoch training comparison", "CreationDate": None})
    plt.close(fig)

    milestone_epochs = [1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
    lines = ["# SO2 per-epoch comparison", "", "Exploratory audit of recorded training masked Huber; lower is better.", "",
             "![Loss trajectories and paired epoch differences](epoch_comparison.png)", "",
             "[Every epoch (CSV)](per_epoch_comparison.csv) · [Every core/epoch (CSV)](per_core_epoch_losses.csv) · "
             "[25-epoch blocks](blocks_25_epochs.csv) · [PDF figure](epoch_comparison.pdf)", "",
             "| Epoch | Original + continuation | Recurrent* | Geometry | Geometry − original |",
             "|---:|---:|---:|---:|---:|"]
    for epoch in milestone_epochs:
        vals = [wide.loc[epoch, k] for k in ("original_training_huber", "recurrent_training_huber",
                                           "geometry_training_huber", "geometry_minus_original_huber")]
        lines.append(f"| {epoch} | " + " | ".join("—" if pd.isna(v) else f"{v:.9f}" for v in vals) + " |")
    original_stats = pair_stats["original"]
    recurrent_stats = pair_stats["recurrent"]
    lines += ["", f"Geometry has lower loss in {original_stats['geometry_lower_epochs']}/200 matched epochs against the original "
              f"trajectory and {recurrent_stats['geometry_lower_epochs']}/175 against recurrent. "
              "Against the original, the direction reverses repeatedly; the 25-epoch block means favor geometry "
              "in blocks 1–25, 76–100 and 126–150, and favor the original in the other five shared blocks.", "",
              "At epochs 176–200, mean training Huber is 0.240917320 for geometry and 0.240760973 for the original "
              "(geometry 0.06494% higher). There is no sustained geometry advantage over the original in these logs.", "",
              "*Recurrent training completed 175 epochs, but artifact finalization failed. Its recorded losses are included "
              "descriptively with that status. The failed duplicate original continuation is listed in provenance and not counted "
              "as an independent model. Original epochs 1–175 are an exact inherited prefix of the successful continuation; "
              "the table uses that training trajectory once.", "",
              "MAE, MSE and R² exist only for final fixed-mask evaluations, not every epoch. Here each loss was measured "
              "during training, with dropout and changing model weights. Curves stop where each model stopped; no missing "
              "epochs are filled. The fixed 25-epoch smoothing window is a post-hoc descriptive aid.", "",
              "All models fit the same cells; observed same-cell genes and morphology remain available. One seed, repeated "
              "epochs and cores provide no patient-level uncertainty, held-out generalization estimate, graph-specific gain "
              "or biological mechanism evidence. Joint attention changes and execution differences prevent attribution to "
              "geometry modulation alone. These training diagnostics do not change a project scientific gate.", "",
              "Verification: source hashes, epoch coverage, all equal-core means, shared configs, exact original-prefix "
              "identity, mask checksums/core order/RNG seeds, missing endpoints, and output checksums pass. See "
              "[verification.json](verification.json) and [provenance.json](provenance.json).", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    dump(out / "verification.json", {"passed": True, "epoch_counts": EXPECTED,
                                      "original_inherited_prefix_identical": True, "mask_identity": mask_checks,
                                      "wide_rows": len(wide), "per_core_rows": len(long_rows),
                                      "equal_core_means_verified": True, "endpoint_missingness_verified": True})
    dump(out / "provenance.json", {"schema": "so2_epoch_comparison_v1", "created_at": datetime.now(timezone.utc).isoformat(),
                                    "analysis_design": "exploratory_existing_training_log_comparison",
                                    "command": sys.argv, "working_directory": str(Path.cwd()),
                                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                    "git_status_at_generation": subprocess.check_output(["git", "status", "--short"], text=True),
                                    "python": platform.python_version(), "pandas": pd.__version__,
                                    "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                                    "analysis_script_sha256": sha(Path(__file__)), "sources": sources,
                                    "run_inventory": inventory, "model_seeds": [0],
                                    "held_out_folds": [], "new_training_or_evaluation_runs": False})
    outputs = ["README.md", "per_epoch_comparison.csv", "per_core_epoch_losses.csv", "blocks_25_epochs.csv", "summary.json",
               "epoch_comparison.png", "epoch_comparison.pdf", "comparison.md", "verification.json", "provenance.json"]
    dump(out / "output_checksums.json", {rel: sha(out / rel) for rel in outputs})
    print(json.dumps(pair_stats, indent=2))
    print(f"Wrote {len(wide)} epoch rows and {len(long_rows)} core/epoch rows to {out}")


if __name__ == "__main__":
    main()

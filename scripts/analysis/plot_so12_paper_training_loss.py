"""Export a read-only snapshot of the recorded paper-split training losses."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
from PIL import Image

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    run = registry.get_run(args.run_id)
    if run is None:
        raise ValueError("Unknown registered run")
    source = paths.scratch_root / "active_runs" / args.run_id
    if not source.is_dir():
        source = Path(run["artifact_path"])
    metric_path = source / "results/epoch_metrics.jsonl"
    # A single byte snapshot; a concurrent incomplete final line is excluded.
    raw = metric_path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    if not rows:
        raise ValueError("No complete epoch metrics available")
    epochs = np.array([row["global_epoch"] for row in rows])
    train = np.array([row["train_equal_core_huber"] for row in rows])
    val = np.array([row["validation"]["equal_core_huber"] for row in rows])
    assert np.array_equal(epochs, np.arange(1, len(rows) + 1))
    assert np.isfinite(train).all() and np.isfinite(val).all()
    best_index = int(np.argmin(val))
    best_epoch, best_loss = int(epochs[best_index]), float(val[best_index])
    assert best_epoch == rows[-1]["selection"]["best_epoch"]
    assert best_loss == rows[-1]["selection"]["best_value"]
    stamp = datetime.now(timezone.utc)
    name = f"so12_paper_loss_{stamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    out = paths.scratch_root / "active_runs" / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "epoch_metrics.snapshot.jsonl").write_bytes(complete)
    shutil.copyfile(__file__, out / "plot_source.py")
    with (out / "losses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_equal_core_huber", "validation_equal_core_huber", "best_validation_so_far"])
        writer.writerows(zip(epochs, train, val, np.minimum.accumulate(val), strict=True))
    blue, orange, teal = "#2563A6", "#D86A24", "#147D73"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
        fig.subplots_adjust(left=.075, right=.975, bottom=.20, top=.78, wspace=.28)
        fig.text(.075, .94, "SO1 + SO2  |  Training and validation loss", fontsize=20, weight="bold", color="#15263B")
        fig.text(.075, .885, f"{len(rows)} completed epochs  ·  {stamp.strftime('%d %b %Y, %H:%M UTC')} snapshot  ·  Status: {run['status']}", fontsize=11, color="#536477")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#CBD5DF")
            ax.grid(axis="y", color="#E4EAF0", linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Equal-core masked Huber loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        left, right = axes
        left.plot(epochs, train, color=blue, lw=1.7, label="Training · 21 cores")
        left.plot(epochs, val, color=orange, lw=1.7, label="Validation · 5 cores")
        left.scatter([best_epoch], [best_loss], color=teal, marker="*", s=145, zorder=5)
        left.set_title("Full history", loc="left", pad=14)
        left.set_xlim(1, max(2, int(epochs[-1]) + 2))
        left.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        left.legend(loc="upper right", frameon=False, fontsize=10)
        start = max(1, int(epochs[-1]) - 99)
        recent = epochs >= start
        right.plot(epochs[recent], val[recent], color=orange, lw=1.35, label="Validation each epoch")
        right.step(epochs[recent], np.minimum.accumulate(val)[recent], where="post", color=teal,
                   lw=1.8, linestyle="--", label="Best validation so far")
        if best_epoch >= start:
            right.scatter([best_epoch], [best_loss], color=teal, marker="*", s=175, zorder=5)
        right.scatter([epochs[-1]], [val[-1]], color=orange, s=35, edgecolor="white", zorder=5)
        right.set_title(f"Validation detail · epochs {start}–{epochs[-1]}", loc="left", pad=14)
        right.set_xlim(start, max(start + 1, int(epochs[-1]) + 3))
        lo, hi = float(val[recent].min()), float(val[recent].max())
        span = max(hi - lo, 0.0001)
        right.set_ylim(lo - span * .18, hi + span * .55)
        right.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        right.legend(loc="upper right", frameon=False, fontsize=9.5)
        right.text(.025, .96, f"Best: epoch {best_epoch}\n{best_loss:.6f}", transform=right.transAxes,
                   va="top", color=teal, fontsize=11, weight="bold")
        fig.text(.075, .115, "Lower is better. Raw epoch values; no smoothing. The right panel uses a narrower loss scale.", color="#384B60", fontsize=10)
        fig.text(.075, .065, "Training: fresh masks and dropout on. Validation: different cores, fixed masks and dropout off.", color="#536477", fontsize=10)
        fig.savefig(out / "loss_curve.png", dpi=180)
        fig.savefig(out / "loss_curve.pdf")
        plt.close(fig)
    with Image.open(out / "loss_curve.png") as image:
        image.verify()
    assert (out / "loss_curve.pdf").read_bytes().startswith(b"%PDF-")
    snapshot = {
        "run_id": args.run_id, "run_status_at_snapshot": run["status"], "snapshot_utc": stamp.isoformat(),
        "source_metrics": str(metric_path), "complete_metrics_sha256": hashlib.sha256(complete).hexdigest(),
        "completed_epochs": len(rows), "best_epoch": best_epoch, "best_validation_huber": best_loss,
        "latest_training_huber": float(train[-1]), "latest_validation_huber": float(val[-1]),
        "selection": rows[-1]["selection"], "matplotlib_version": matplotlib.__version__,
        "scope": "live_training_diagnostic_only", "normalization_or_model_changed": False,
        "verification": "contiguous_epochs_finite_losses_best_checkpoint_match_valid_png_pdf",
        "files": {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / "snapshot.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    (out / "_SUCCESS").write_text(json.dumps({"snapshot_sha256": sha256_file(out / "snapshot.json")}) + "\n")
    final = paths.report_root / "runtime" / name
    final.parent.mkdir(parents=True, exist_ok=True)
    assert not final.exists() and out.stat().st_dev == final.parent.stat().st_dev
    out.rename(final)
    for filename in ("loss_curve.png", "loss_curve.pdf", "losses.csv", "snapshot.json"):
        file = final / filename
        registry.record_artifact(args.run_id, kind="training_progress_plot", path=file,
                                 sha256=sha256_file(file), size_bytes=file.stat().st_size)
    print(json.dumps({"directory": str(final), "epochs": len(rows), "best_epoch": best_epoch,
                      "best_validation_huber": best_loss}))


if __name__ == "__main__":
    main()

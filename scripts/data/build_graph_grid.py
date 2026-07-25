#!/usr/bin/env python3
"""Build and audit the prespecified physical graph grid without test targets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.artifacts import load_prepared_artifact, sha256_file  # noqa: E402
from spatial_benchmark.graphs import build_spatial_graph  # noqa: E402


def _parse_numbers(value: str, kind: type) -> list:
    return [kind(part.strip()) for part in value.split(",") if part.strip()]


def _graph_id(k: int, radius: float, symmetry: str) -> str:
    return f"k{k}_r{radius:g}_{symmetry}"


def _plot(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"union": "#3465a4", "mutual": "#c45d2c"}
    markers = {30.0: "o", 50.0: "s", 75.0: "^"}
    for (symmetry, radius), group in frame.groupby(["symmetry", "radius_um"]):
        label = f"{symmetry}, {radius:g} µm"
        style = dict(
            color=colors[symmetry],
            marker=markers.get(float(radius), "o"),
            label=label,
        )
        ordered = group.sort_values("k")
        axes[0, 0].plot(
            ordered["k"], ordered["selection_mean_degree"], **style
        )
        axes[0, 1].plot(
            ordered["k"], ordered["selection_cap_hit_rate"], **style
        )
        axes[1, 0].plot(
            ordered["k"], ordered["selection_n_isolated_nodes"], **style
        )
        axes[1, 1].plot(
            ordered["k"],
            ordered["selection_edge_distance_p95_um"],
            **style,
        )
    axes[0, 0].set_ylabel("Mean undirected degree")
    axes[0, 1].set_ylabel("Fraction of nodes hitting k cap")
    axes[1, 0].set_ylabel("Isolated nodes")
    axes[1, 1].set_ylabel("95th percentile edge length (µm)")
    for axis in axes.flat:
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center right", frameon=False)
    figure.suptitle(
        "Normal-core graph geometry grid "
        "(train + validation only; no expression outcomes)"
    )
    figure.tight_layout(rect=(0, 0, 0.84, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", default="8,12,16")
    parser.add_argument("--radius-um", default="30,50,75")
    parser.add_argument("--symmetry", default="union,mutual")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite graph grid: {destination}")
    manifest, arrays, _ = load_prepared_artifact(args.prepared.resolve())
    assert arrays is not None
    ks = _parse_numbers(args.k, int)
    radii = _parse_numbers(args.radius_um, float)
    symmetries = [part.strip() for part in args.symmetry.split(",") if part.strip()]
    if not ks or not radii or not symmetries:
        raise ValueError("k, radius, and symmetry grids must be non-empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        rows: list[dict] = []
        files: dict[str, str] = {}
        selection_mask = np.asarray(arrays["split_labels"]) != "test"
        if not np.any(selection_mask):
            raise RuntimeError("Graph-QC selection scope has no train/validation nodes")
        for k, radius, symmetry in itertools.product(ks, radii, symmetries):
            graph = build_spatial_graph(
                arrays["coordinates_um"],
                k=k,
                radius_um=radius,
                symmetry=symmetry,
                group_labels=arrays["split_labels"],
                fov=arrays["fov"],
                rbf_bins=8,
            )
            selection_graph = build_spatial_graph(
                arrays["coordinates_um"][selection_mask],
                k=k,
                radius_um=radius,
                symmetry=symmetry,
                group_labels=arrays["split_labels"][selection_mask],
                fov=arrays["fov"][selection_mask],
                rbf_bins=8,
            )
            identifier = _graph_id(k, radius, symmetry)
            graph_path = temporary / f"{identifier}.npz"
            np.savez_compressed(
                graph_path,
                edge_index=graph.edge_index,
                edge_attributes_raw=graph.edge_attr,
                edge_attribute_names=np.asarray(graph.edge_attr_names, dtype="U64"),
            )
            files[graph_path.name] = sha256_file(graph_path)
            record = {
                "graph_id": identifier,
                "k": k,
                "radius_um": radius,
                "symmetry": symmetry,
                **graph.qc.to_dict(),
                **{
                    f"selection_{key}": value
                    for key, value in selection_graph.qc.to_dict().items()
                },
            }
            if (
                record["cross_group_edges"] != 0
                or record["self_loops"] != 0
                or record["selection_cross_group_edges"] != 0
                or record["selection_self_loops"] != 0
            ):
                raise RuntimeError(f"Invalid graph topology for {identifier}")
            rows.append(record)
        frame = pd.DataFrame(rows).sort_values(
            ["symmetry", "radius_um", "k"]
        )
        table_path = temporary / "graph_qc.csv"
        frame.to_csv(table_path, index=False)
        figure_path = temporary / "graph_qc_grid.png"
        _plot(frame, figure_path)
        files[table_path.name] = sha256_file(table_path)
        files[figure_path.name] = sha256_file(figure_path)
        grid_manifest = {
            "format_version": 1,
            "artifact_kind": "geometry_only_graph_grid",
            "prepared_artifact_id": manifest["artifact_id"],
            "split_id": manifest["split"]["split_id"],
            "n_cells": int(len(arrays["split_labels"])),
            "selection_scope": "train_and_validation_geometry_only",
            "selection_n_cells": int(np.sum(selection_mask)),
            "test_geometry_used_for_selection_qc": False,
            "test_expression_targets_evaluated": False,
            "grid": {
                "k": ks,
                "radius_um": radii,
                "symmetry": symmetries,
            },
            "graph_count": len(rows),
            "files": files,
        }
        payload = json.dumps(
            grid_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        grid_manifest["artifact_id"] = hashlib.sha256(payload).hexdigest()[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(grid_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "artifact_id": grid_manifest["artifact_id"],
                "graph_count": len(rows),
                "output": str(destination),
                "test_expression_targets_evaluated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

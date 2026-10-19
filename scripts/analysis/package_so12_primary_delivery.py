"""Package verified primary maps without completing the parent stability study."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import so12_hidden_maps as maps
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths


def verify_records(value, verified):
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            path = Path(value["path"])
            maps._require(path.stat().st_size == value["size_bytes"] and sha256_file(path) == value["sha256"],
                          f"Preview provenance changed: {path}")
            verified.append(value)
        for child in value.values():
            verify_records(child, verified)
    elif isinstance(value, list):
        for child in value:
            verify_records(child, verified)


def package(source: Path, destination: Path) -> dict:
    paths = current_paths()
    source, destination = source.resolve(), destination.resolve()
    active = (paths.scratch_root / "active_runs").resolve()
    maps._require(source.is_relative_to(active) and destination.parent == active,
                  "Use isolated active-run staging paths")
    maps._require(source != destination and not destination.exists(), "Delivery must use a new isolated directory")
    maps._require(not (source / "READY_FOR_REVIEW.json").exists(), "Parent is ready; prioritize full final delivery")
    preview_path = source / "primary_preview/preview_receipt.json"
    preview = json.loads(preview_path.read_text())
    maps._require(preview["status"] == "primary_partitions_complete_stability_pending"
                  and preview["n_cells_displayed"] == 117996 and preview["n_cells_clustering"] == 407659,
                  "Unexpected primary preview scope")
    verified = []
    verify_records(preview, verified)
    roots_review = json.loads((source / "diagnostics/primary_preview_root_review.json").read_text())
    independent_review = json.loads((source / "diagnostics/primary_preview_visual_review.json").read_text())
    maps._require(roots_review["status"] == independent_review["status"] == "passed", "Primary visual reviews did not pass")
    destination.mkdir()
    for name in ("figures", "tables", "provenance", "audits"):
        (destination / name).mkdir()
    copied = []
    def copy_from_stage(relative, target):
        start, end = source / relative, destination / target
        end.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(start, end)
        maps._require(sha256_file(start) == sha256_file(end), f"Copy mismatch: {relative}")
        copied.append({"source_relative_path": relative, "delivery_relative_path": target,
                       "sha256": sha256_file(end), "size_bytes": end.stat().st_size})
    for name in preview["files"]:
        copy_from_stage(f"primary_preview/{name}", f"figures/{name}")
    copy_from_stage("primary_preview/preview_receipt.json", "provenance/primary_preview_receipt.json")
    for relative in ("config.resolved.json", "extraction_inputs.json", "embeddings/extraction_manifest.json"):
        copy_from_stage(relative, f"provenance/parent/{relative}")
    for representation in ("h0", "hL"):
        for name in ("leiden_seed0_receipt.json", "graph_receipt.json", "cpu_exact_audit.json"):
            relative = f"clustering/{representation}/{name}"
            copy_from_stage(relative, f"provenance/parent/{relative}")
    for path in sorted((source / "diagnostics").glob("*.json")):
        copy_from_stage(path.relative_to(source).as_posix(), f"audits/{path.name}")
    for path in sorted((source / "provenance/failed_attempts").rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            copy_from_stage(relative, relative)
    for path in (Path(__file__), Path(maps.__file__)):
        end = destination / "provenance" / path.name
        shutil.copy2(path, end)
    cells = pd.read_parquet(source / "cells.parquet")
    labels = {method: maps._validate_labels(np.load(source / "clustering" / f"{method}_labels.npy", allow_pickle=False),
                                           len(cells), method) for method in ("h0", "hL")}
    pca_source = Path(preview["sources"]["PCA"]["cells.csv.gz"]["path"]).parent
    pca_cells = pd.read_csv(pca_source / "cells.csv.gz")
    for method in maps.METHODS[2:]:
        labels[method] = maps.align_source_labels(cells, pca_cells, pd.read_csv(pca_source / "methods" / method / "clusters.csv.gz"))
    selected = cells.core_number.isin(maps.DEFAULT_CORES).to_numpy()
    rows = []
    for method in maps.METHODS:
        palette = {int(key): value for key, value in preview["global_method_palettes"][method].items()}
        total = np.bincount(labels[method], minlength=len(palette))
        visible = np.bincount(labels[method][selected], minlength=len(palette))
        for cluster, color in palette.items():
            rows.append({"method": method, "cluster_id": cluster, "cluster_label": f"{maps.NAMESPACES[method]}-{cluster:02d}",
                         "color_hex": color, "all_cells": int(total[cluster]), "selected_cells": int(visible[cluster])})
    colors = pd.DataFrame(rows).sort_values(["method", "cluster_id"], kind="stable")
    colors.to_csv(destination / "tables/cluster_color_keys.csv", index=False)
    maps._legend_figure(colors, destination / "figures")
    comparison = preview["partition_comparison"]
    # Recompute only descriptive statistics, never representations or clusters.
    repeated = maps.compare_partitions(labels, cells.core_number.to_numpy(), maps.DEFAULT_CORES)
    maps._require(repeated == comparison, "Replayed primary agreement differs from verified preview")
    maps._write_json(destination / "tables/comparison_metrics.json", comparison)
    pair_rows, count_rows = [], []
    for population, values in comparison["populations"].items():
        for i, method in enumerate(maps.METHODS):
            count_rows.append({"population": population, "cells": values["cells"], "method": method,
                               "clusters_present": values["observed_cluster_counts"][method]})
            for j, other in enumerate(maps.METHODS):
                pair_rows.append({"population": population, "cells": values["cells"], "method_1": method,
                                  "method_2": other, "adjusted_rand_index": values["pairwise_ari"][i][j]})
        pd.DataFrame(values["pairwise_ari"], index=maps.METHODS, columns=maps.METHODS).to_csv(
            destination / "tables" / f"ari_{population}.csv", index_label="method")
    pd.DataFrame(pair_rows).to_csv(destination / "tables/comparison_ari.csv", index=False)
    pd.DataFrame(count_rows).to_csv(destination / "tables/comparison_cluster_counts.csv", index=False)
    extraction = json.loads((source / "extraction_inputs.json").read_text())
    created = datetime.now(timezone.utc).isoformat()
    seed1_available = {rep: (source / "clustering" / rep / "leiden_seed1_receipt.json").exists() for rep in ("h0", "hL")}
    maps._require(not all(seed1_available.values()) and not (source / "READY_FOR_REVIEW.json").exists(),
                  "Secondary analysis finished during packaging; prioritize full final delivery")
    metadata = {"schema": "so12_primary_seed0_delivery_v1", "status": "primary_seed0_complete",
        "secondary_seed1_status": "pending", "secondary_seed1_receipts_available": seed1_available,
        "as_of_utc": created, "parent_evaluation_id": source.name, "parent_evaluation_completed": False,
        "does_not_replace_parent_evaluation": True, "source_run_id": extraction["source_run_id"],
        "checkpoint_epoch": extraction["checkpoint_epoch"], "checkpoint_sha256": extraction["checkpoint_sha256"],
        "all_observed_genes": True, "clustering_cells": 407659, "clustering_cores": 28,
        "displayed_cells": 117996, "displayed_cores": list(maps.DEFAULT_CORES), "representations_dimensions": 256,
        "cluster_counts": preview["cluster_counts"], "primary_resolution": 1.0, "primary_seed": 0,
        "representation_definitions": extraction["representation_definitions"],
        "polygon_qc": preview["polygon_qc"], "primary_statistics_replayed_exactly": True,
        "primary_source_records_reverified": len(verified), "preserved_png_bytes": True,
        "copied_stage_artifacts": copied, "pipeline_comparability": comparison["pipeline_comparability"],
        "publication_status": "staged_for_root_review_not_registered_or_published",
        "limitations": "Requested seed-0 maps and descriptive comparisons are complete. Additional fixed-graph seed-1 stability checks continue in the separate parent evaluation. Cluster IDs/colors are method-specific. All-cell clustering is transductive. No biological cell-type, graph-specific predictive-gain or causal claim is established."}
    maps._write_json(destination / "metadata.json", metadata)
    sections = []
    for core in maps.DEFAULT_CORES:
        path = destination / "figures" / f"core_{core:02d}_primary_preview.png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        sections.append(f'<section><h2>Core {core} · {maps.EXPECTED_CORE_COUNTS[core]:,} cells</h2>'
            f'<img src="data:image/png;base64,{encoded}" alt="Core {core}: initial h0, final hL, PCA, RBF kernel PCA and spatial PCA cell segmentation"></section>')
    legends = []
    for method in maps.METHODS:
        body = ''.join(f'<tr><td><span class="swatch" style="background:{row.color_hex}"></span>{row.cluster_label}</td>'
                       f'<td>{row.selected_cells:,}</td><td>{row.all_cells:,}</td></tr>'
                       for row in colors.loc[colors.method == method].itertuples())
        legends.append(f'<section><h3>{html.escape(maps.METHOD_NAMES[method])}</h3><table><thead>'
            '<tr><th>Cluster</th><th>Shown</th><th>All cells</th></tr></thead><tbody>'+body+'</tbody></table></section>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += '<title>SO12 primary h0/hL and PCA segmentation comparison</title><style>'
    document += 'body{font-family:system-ui,sans-serif;color:#172033;max-width:1600px;margin:auto;padding:24px}img{width:100%;height:auto}p{max-width:1100px;line-height:1.5}.notice{background:#fff7d6;border-left:5px solid #b77900;padding:15px}.comparisons{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:24px}.legends{display:grid;grid-template-columns:repeat(5,minmax(170px,1fr));gap:20px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:4px;border-bottom:1px solid #e2e8f0;text-align:right}td:first-child,th:first-child{text-align:left}.swatch{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle}@media(max-width:900px){.comparisons{display:block}.legends{grid-template-columns:repeat(2,minmax(170px,1fr))}}@media print{section{break-inside:avoid}body{padding:0}}'
    document += '</style><main><h1>Initial and final cell representations, compared with PCA methods</h1>'
    document += f'<div class="notice"><strong>Primary seed-0 results complete.</strong> Additional seed-1 stability checks remain pending as of {html.escape(created)}. This delivery does not mark the parent evaluation complete. The preserved image titles say “primary preview” because those exact verified figure bytes are retained.</div>'
    document += '<p>The frozen epoch-151 four-layer SO1 + SO2 model receives all observed genes. h0 is the trained node-encoder output before graph blocks; hL follows four graph blocks and precedes the expression decoder. All 407,659 cells from 28 cores were clustered jointly for each representation. The figures display 117,996 cells from cores 1, 9, 13, 15, 21 and 23.</p>'
    document += '<p>Global primary cluster counts are h0: 16, hL: 22, Pearson PCA: 19, RBF kernel PCA: 21 and spatial PCA: 17. Panel-title counts refer to all 28 cores. All five methods use the same original polygons, bounds, orientation and 100 µm scale bars. Colors and cluster IDs are independent across methods; complete keys are below. Core 13 is model validation; the other displayed cores are training.</p>'
    document += '<p>The three PCA partitions and their colors are reused without refitting. ARI measures agreement, not biological validity. The retained pipeline differences prevent interpreting this as a controlled encoder ablation. These descriptive groups do not establish cell types, graph-specific predictive benefit or causality.</p>'
    document += ''.join(sections) + maps._comparison_html(comparison)
    document += '<h2>Complete cluster color keys</h2><p>Shown counts cover the six displayed cores; all-cell counts cover all 407,659 cells.</p><div class="legends">'+''.join(legends)+'</div></main></html>'
    (destination / "index.html").write_text(document)
    ari = comparison["populations"]["all_cells"]["pairwise_ari"]
    readme = f'''# SO1/SO2 primary cell-representation maps

Primary seed-0 results are complete; secondary seed-1 stability checks remain
pending as of {created}. This isolated delivery does not replace or complete
parent evaluation `{source.name}`.

Open [the portable report](index.html) for all six five-method segmentation maps,
full color keys, cluster counts, pairwise agreement and pipeline definitions.
The [overview](figures/six_core_primary_preview.png) retains exact verified image
bytes. PNG filenames/titles retain “primary preview”; the primary partitions are
complete, while additional stability validation is pending.

The epoch-151 model uses all observed genes. h0 is before graph processing and
hL follows four graph blocks. All 407,659 cells from 28 cores enter each primary
partition; the displayed six cores contain 117,996 original cell polygons.

| Method | Global primary clusters |
|---|---:|
| h0 | 16 |
| hL | 22 |
| Pearson PCA | 19 |
| RBF kernel PCA | 21 |
| Spatial PCA | 17 |

All-cell h0–hL ARI is {ari[0][1]:.6f}. hL agreement with Pearson PCA,
RBF kernel PCA and spatial PCA is {ari[1][2]:.6f}, {ari[1][3]:.6f} and
{ari[1][4]:.6f}, respectively. These measure partition agreement and do not rank
biological validity. Exact tables for all cells, the displayed cores pooled and
each core are under `tables/`.

Hidden-state clustering uses direct 256-dimensional cosine k30 exhaustive
neighbors, unweighted union graphs and Leiden to convergence. The prior PCA
methods use 32 dimensions, Euclidean HNSW k30 neighbors, exponential edge weights,
arithmetic-mean symmetrization and two Leiden iterations. All primary partitions
use resolution 1.0 and seed 0; no subset was reclustered. Representation and
clustering-pipeline differences are therefore not separated.

All five columns share the same original cell polygons and physical geometry.
Cluster colors/IDs are independent across methods; global palettes remain fixed
across cores. Core 13 is model validation; the other five displayed cores are
training. All-cell clustering is transductive, without independent biological
annotations or causal evidence.

`metadata.json` records primary status and the secondary status at creation,
checkpoint and label provenance, image-byte preservation and replayed metrics.
`audits/` retains the verified extraction, graph, label and visual checks.
`provenance/failed_attempts/` retains two operational failures: logger argument
collision before GPU work (004300Z), and an exact-invariance pilot detecting an
omitted deterministic flag before any saved embeddings (004600Z). Their outputs
were not used. No checkpoint, embedding array or unrestricted cell table is
duplicated in this delivery.
'''
    (destination / "README.md").write_text(readme)
    maps._write_json(destination / "provenance/package_verification.json", {
        "status": "passed", "source_preview_sha256": sha256_file(preview_path),
        "seven_pngs_match_source_bytes": all(sha256_file(destination / "figures" / name) == item["sha256"]
            for name, item in preview["files"].items()), "all_primary_statistics_replayed_exactly": True,
        "cluster_color_counts_cover_all_cells": bool(colors.groupby("method").all_cells.sum().eq(407659).all()),
        "cluster_color_counts_cover_selected_cells": bool(colors.groupby("method").selected_cells.sum().eq(117996).all()),
        "secondary_stability_pending": True, "registry_or_publication_performed": False})
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.source_stage, args.output)
    print(json.dumps({"status": result["status"], "secondary_seed1_status": result["secondary_seed1_status"],
                      "output": str(args.output)}), flush=True)

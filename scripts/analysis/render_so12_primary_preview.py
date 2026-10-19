"""Early primary-label maps while prespecified Leiden stability work continues.

This optional display stage does not satisfy or bypass final publication gates.
It reads genuine completed seed-0 receipts and writes only primary_preview/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import so12_hidden_maps as maps
from spatial_benchmark.attention_niche_geometry import load_aligned_cosmx_polygons
from spatial_benchmark.data import DEFAULT_PIXEL_SIZE_UM, resolve_nested_raw_path
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths


PREVIEW_STATUS = "primary_partitions_complete_stability_pending"


def load_verified_primary(out: Path, representation: str, *, expected_cells: int = maps.EXPECTED_CELLS) -> tuple[np.ndarray, dict]:
    """Require completed primary output, not a partially written label array."""
    maps._require(representation in ("h0", "hL"), "Unexpected hidden representation")
    labels_path = out / "clustering" / f"{representation}_labels.npy"
    receipt_path = out / "clustering" / representation / "leiden_seed0_receipt.json"
    graph_path = out / "clustering" / representation / "graph_receipt.json"
    maps._require(all(path.is_file() for path in (labels_path, receipt_path, graph_path)),
                  f"{representation} primary partition and completion receipt are not both available")
    receipt = json.loads(receipt_path.read_text())
    graph = json.loads(graph_path.read_text())
    maps._require(sha256_file(labels_path) == receipt["labels_sha256"],
                  f"{representation} primary label checksum mismatch")
    maps._require(sha256_file(graph_path) == receipt["graph_receipt_sha256"],
                  f"{representation} frozen graph checksum mismatch")
    maps._require(graph["input_sha256"]["cells.parquet"] == sha256_file(out / "cells.parquet"),
                  f"{representation} graph is not aligned with current observation rows")
    details = receipt["leiden"]
    maps._require(details["random_seed"] == 0 and details["resolution"] == 1.0
                  and details["n_iterations"] == -1, "Preview must use the genuine prescribed primary partition")
    labels = maps._validate_labels(np.load(labels_path, allow_pickle=False), expected_cells, representation)
    maps._require(int(labels.max()) + 1 == details["cluster_count"], "Primary cluster count differs from receipt")
    return labels, {"labels": maps._record(labels_path), "seed0_receipt": maps._record(receipt_path),
                    "graph_receipt": maps._record(graph_path), "leiden": details}


def _save_png(figure, path: Path, dpi: int) -> None:
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        maps.plt.close(figure)


def render_primary_preview(out: Path, pca_run_dir: Path) -> dict:
    started = time.monotonic()
    paths = current_paths()
    out, pca_run_dir = Path(out).resolve(), Path(pca_run_dir).resolve()
    active = (paths.scratch_root / "active_runs").resolve()
    maps._require(out.is_relative_to(active) and out != active, "Preview writes require a separate active evaluation")
    destination = out / "primary_preview"
    maps._require(not destination.exists(), "Refusing to overwrite a prior primary preview")
    # Primary receipts are checked before creating any output directory.
    labels, primary_sources = {}, {}
    for representation in ("h0", "hL"):
        labels[representation], primary_sources[representation] = load_verified_primary(out, representation)
    cells = pd.read_parquet(out / "cells.parquet")
    maps._validate_cells(cells, maps.EXPECTED_CELLS)
    maps._require(np.array_equal(cells.row_index.to_numpy(), np.arange(len(cells))), "Observation order changed")
    extraction_path = out / "embeddings/extraction_manifest.json"
    extraction = json.loads(extraction_path.read_text())
    maps._require(extraction["verified"] is True and extraction["n_cells"] == len(cells)
                  and extraction["all_observed"] is True and extraction["dimensions"] == 256
                  and extraction["core_count"] == 28, "All-observed full-cohort extraction is not verified")
    maps._require(extraction["cells_sha256"] == sha256_file(out / "cells.parquet"), "Extraction observation checksum differs")
    pca_checksums = json.loads((pca_run_dir / "checksums.json").read_text())
    pca_sources = {}
    for relative in ("_SUCCESS", "cells.csv.gz", "resolved_config.json",
                     *[f"methods/{method}/clusters.csv.gz" for method in maps.METHODS[2:]]):
        path = pca_run_dir / relative
        record = maps._record(path)
        maps._require(record["sha256"] == pca_checksums[relative]["sha256"]
                      and record["size_bytes"] == pca_checksums[relative]["size_bytes"],
                      f"PCA source checksum mismatch: {relative}")
        pca_sources[relative] = record
    pca_cells = pd.read_csv(pca_run_dir / "cells.csv.gz")
    identities = cells[maps.KEYS + ["alias", "split"]].merge(
        pca_cells[maps.KEYS + ["core", "train"]], on=maps.KEYS, how="outer", validate="one_to_one", indicator=True)
    maps._require(len(identities) == len(cells) and identities["_merge"].eq("both").all()
                  and identities.alias.eq(identities.core).all(), "PCA keyed core identity differs")
    maps._require(np.array_equal(identities.split.eq("train"), identities.train), "PCA keyed split identity differs")
    for method in maps.METHODS[2:]:
        labels[method] = maps.align_source_labels(cells, pca_cells,
            pd.read_csv(pca_run_dir / "methods" / method / "clusters.csv.gz"))
    comparison = maps.compare_partitions(labels, cells.core_number.to_numpy(), maps.DEFAULT_CORES)
    color_path = paths.report_root / "analyses/so12_embeddings_leiden/cluster_color_keys.csv"
    old_colors = pd.read_csv(color_path) if color_path.is_file() else None
    palettes, palette_provenance = {}, {}
    for method in maps.METHODS:
        palettes[method], palette_provenance[method] = maps.color_keys(labels[method], method, old_colors)
    selected_mask = cells.core_number.isin(maps.DEFAULT_CORES).to_numpy()
    selected = cells.loc[selected_mask]
    maps._require(selected.groupby("core_number").size().to_dict() == maps.EXPECTED_CORE_COUNTS,
                  "Requested core cell coverage differs")
    sources = {f"SO_{slide}": resolve_nested_raw_path(paths.data_root / "raw", f"26040302SO_{slide}-polygons.csv")
               for slide in (1, 2)}
    polygons = load_aligned_cosmx_polygons(sources, selected[maps.KEYS],
        prepared_coordinates_um=selected[["x_um", "y_um"]].to_numpy(), centroid_tolerance_um=5.0)
    vertices = maps.polygon_vertices(polygons.polygons_um)
    selection_labels = {method: values[selected_mask] for method, values in labels.items()}
    core_indices, core_vertices, bounds = {}, {}, {}
    for core in maps.DEFAULT_CORES:
        indices = np.flatnonzero(selected.core_number.to_numpy() == core)
        core_indices[core] = indices
        core_vertices[core] = tuple(vertices[index] for index in indices)
        bounds[core] = maps._bounds(core_vertices[core])
    destination.mkdir()
    with maps.plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        for core in maps.DEFAULT_CORES:
            figure, axes = maps.plt.subplots(1, 5, figsize=(22, 4.8))
            for method, axis in zip(maps.METHODS, axes, strict=True):
                maps._draw(axis, core_vertices[core], selection_labels[method][core_indices[core]], palettes[method], bounds[core])
                axis.set_title(f"{maps.METHOD_NAMES[method]} · {len(palettes[method])} clusters", fontsize=11)
            figure.suptitle(f"PRIMARY PREVIEW · Core {core} · {len(core_indices[core]):,} cells", fontsize=16)
            figure.text(.5, .015, "Primary partitions complete · Seed-stability analysis pending · All genes observed · Colors/IDs are independent across methods",
                        ha="center", fontsize=9)
            figure.tight_layout(rect=(0, .05, 1, .93))
            _save_png(figure, destination / f"core_{core:02d}_primary_preview.png", 180)
        figure, axes = maps.plt.subplots(6, 5, figsize=(22, 26), squeeze=False)
        for row, core in enumerate(maps.DEFAULT_CORES):
            for column, method in enumerate(maps.METHODS):
                axis = axes[row, column]
                maps._draw(axis, core_vertices[core], selection_labels[method][core_indices[core]], palettes[method], bounds[core])
                axis.set_title(f"Core {core} · {maps.METHOD_NAMES[method]}", fontsize=10)
        figure.suptitle("PRIMARY PREVIEW · Five-method cell segmentation comparison", fontsize=19)
        figure.text(.5, .007, "Primary partitions complete · Seed-stability analysis pending · All scale bars 100 µm · Colors/IDs are method-specific",
                    ha="center", fontsize=10)
        figure.tight_layout(rect=(0, .02, 1, .975))
        _save_png(figure, destination / "six_core_primary_preview.png", 160)
    # Independent ongoing seed-1 work must not alter primary inputs during rendering.
    for representation in ("h0", "hL"):
        _, rechecked = load_verified_primary(out, representation)
        maps._require(rechecked == primary_sources[representation], "Primary receipt changed during preview rendering")
    alignment = polygons.centroid_alignment
    receipt = {"schema": "so12_primary_partition_preview_v1", "status": PREVIEW_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "evaluation_id": out.name, "full_analysis_complete": False, "publication_gates_satisfied": False,
        "n_cells_clustering": len(cells), "n_cells_displayed": len(selected), "core_order": list(maps.DEFAULT_CORES),
        "method_order": list(maps.METHODS), "cluster_counts": {method: len(palette) for method, palette in palettes.items()},
        "partition_comparison": comparison,
        "global_method_palettes": palettes, "palette_provenance": palette_provenance,
        "sources": {"primary_partitions": primary_sources, "cells": maps._record(out / "cells.parquet"),
            "extraction_manifest": maps._record(extraction_path), "extraction_inputs": maps._record(out / "extraction_inputs.json"),
            "resolved_config": maps._record(out / "config.resolved.json"), "PCA": pca_sources,
            "PCA_checksums": maps._record(pca_run_dir / "checksums.json"),
            "historical_color_keys": maps._record(color_path) if color_path.is_file() else None,
            "raw_polygons": {slide: maps._record(path) for slide, path in sources.items()},
            "preview_code": maps._record(Path(__file__)), "reused_renderer_code": maps._record(Path(maps.__file__))},
        "polygon_qc": {"covered_cells": polygons.n_cells, "vertices": polygons.selected_vertex_rows,
            "source_rows_scanned": polygons.source_rows_scanned, "repaired_cells": int(polygons.repaired.sum()),
            "alignment_mismatches": len(alignment.mismatch_indices), "centroid_tolerance_um": alignment.tolerance_um,
            "centroid_tolerance_exceeded": len(alignment.centroid_tolerance_exceeded_indices),
            "containment_accepted": len(alignment.containment_accepted_indices),
            "coordinate_outside_polygon_count": len(alignment.coordinate_outside_polygon_indices),
            "median_centroid_distance_um": alignment.median_distance_um,
            "maximum_centroid_distance_um": alignment.maximum_distance_um,
            "pixel_size_um": DEFAULT_PIXEL_SIZE_UM, "alignment_rule": alignment.alignment_rule},
        "rendering": {"original_cell_polygons": True, "equal_aspect": True, "invert_y": True,
            "scale_bar_um": 100, "matched_core_bounds": bounds, "cells_sampled": False,
            "independent_method_cluster_namespaces": True},
        "files": {path.name: maps._record(path) for path in sorted(destination.glob("*.png"))},
        "limitations": "Preview only. Both primary partitions are genuine complete seed-0 outputs; seed-1 stability validation and full report publication remain pending. The five pipelines retain their original clustering differences. No biological interpretation is assigned."}
    maps._write_json(destination / "preview_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = render_primary_preview(arguments.output, arguments.pca_run_dir)
    print(json.dumps({"status": result["status"], "cells": result["n_cells_displayed"],
                      "cluster_counts": result["cluster_counts"]}), flush=True)

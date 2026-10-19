"""One static spatial PNG for the completed SO2 geometry-modulated model."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .fingerprints import sha256_file
from .relative_qkv_embedding_clustering import (
    _atomic_write_json,
    _file_record,
    _read_json,
    _receipt_with_self_hash,
    _style_spatial_axis,
    _verify_self_hash,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_CORE_NUMBERS,
)


FIGURE_SCHEMA = "so2_geometry_modulated_hl_spatial_png_v1"
EXPECTED_RUN_ID = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
PNG_NAME = "contextual_leiden_resolution_1p0_spatial_14cores.png"


def _atomic_save_png(figure: Any, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "spatial_benchmark.so2_geometry_hl_figures"},
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_spatial_png(
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    dpi: int = 300,
) -> Mapping[str, Any]:
    """Validate all cell assignments and render one shared-palette 14-core map."""

    if isinstance(dpi, bool) or int(dpi) != dpi or int(dpi) < 72:
        raise ValueError("PNG DPI must be an integer of at least 72.")
    output_root = Path(output_root)
    clustering_path = output_root / "clustering" / "clustering_manifest.json"
    stored_clustering = _read_json(clustering_path, label="hL clustering manifest")
    _verify_self_hash(stored_clustering, label="hL clustering manifest")
    if stored_clustering != dict(clustering_receipt):
        raise ValueError("Supplied clustering receipt differs from its stored source.")
    if clustering_receipt.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Figure source is not the completed geometry-modulated run.")
    resolution = float(clustering_receipt["configuration"]["leiden_resolution"])
    if not math.isclose(resolution, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("This figure is locked to Leiden resolution 1.0.")
    if int(clustering_receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS:
        raise ValueError("Clustering receipt does not contain all SO2 cells.")

    table_relative = "tables/cell_contextual_clusters.parquet"
    table_path = output_root / table_relative
    if _file_record(table_path) != clustering_receipt["files"][table_relative]:
        raise ValueError("Cell-assignment table checksum differs from clustering.")
    frame = pd.read_parquet(table_path)
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise ValueError("Spatial table must contain all 246,063 cells in core order.")
    if frame[["core_number", "cell_index"]].duplicated().any():
        raise ValueError("Spatial table contains duplicate cell assignments.")
    if not np.isfinite(frame[["x_um", "y_um"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Spatial coordinates contain non-finite values.")
    palette = clustering_receipt.get("palette")
    if not isinstance(palette, Mapping) or set(frame["contextual_cluster"]) != set(
        palette
    ):
        raise ValueError("The shared palette must cover every observed cluster exactly.")
    if len(palette) != int(clustering_receipt["cluster_count"]):
        raise ValueError("Palette size differs from the reported cluster count.")
    for number in SO2_CORE_NUMBERS:
        selected = frame.loc[frame["core_number"] == number]
        expected = EXPECTED_CELL_COUNTS_BY_CORE[number]
        if len(selected) != expected or not np.array_equal(
            selected["cell_index"].to_numpy(), np.arange(expected)
        ):
            raise ValueError(f"Cell count or cell order changed for SO2 Core {number}.")

    source_sha = sha256_file(clustering_path)
    png_path = output_root / "figures" / PNG_NAME
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="geometry hL figure manifest")
        _verify_self_hash(receipt, label="geometry hL figure manifest")
        expected_files = {f"figures/{PNG_NAME}": _file_record(png_path)}
        if any(
            (
                receipt.get("schema") != FIGURE_SCHEMA,
                receipt.get("run_id") != EXPECTED_RUN_ID,
                receipt.get("clustering_manifest_sha256") != source_sha,
                receipt.get("dpi") != int(dpi),
                receipt.get("files") != expected_files,
            )
        ):
            raise ValueError("Existing figure does not match the requested source.")
        return receipt
    if png_path.exists():
        raise ValueError("Refusing to overwrite an unreceipted PNG.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
        try:
            for axis, number in zip(axes.ravel()[:14], SO2_CORE_NUMBERS, strict=True):
                selected = frame.loc[frame["core_number"] == number]
                coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
                axis.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=0.52,
                    c=selected["contextual_cluster"].map(palette).tolist(),
                    marker="o",
                    linewidths=0,
                    edgecolors="none",
                    alpha=0.92,
                    rasterized=True,
                )
                axis.set_title(
                    f"SO2 Core {number}\n$n$ = {len(selected):,}",
                    fontsize=12,
                    weight="bold",
                    pad=7,
                )
                _style_spatial_axis(axis, coordinates)
            legend_axis = axes.ravel()[-1]
            legend_axis.axis("off")
            handles = [
                Patch(facecolor=color, edgecolor="none", label=label)
                for label, color in sorted(
                    palette.items(), key=lambda item: int(item[0].removeprefix("C"))
                )
            ]
            legend_axis.legend(
                handles=handles,
                loc="center",
                frameon=False,
                ncol=2 if len(handles) > 14 else 1,
                title="Joint hL cluster",
                fontsize=8,
                title_fontsize=10,
            )
            figure.suptitle(
                "SO2 geometry-modulated hL Leiden clusters — joint 14-core clustering\n"
                "Epoch 200 · model seed 0 · resolution 1.0",
                fontsize=18,
                weight="bold",
                y=0.995,
            )
            figure.text(
                0.5,
                0.006,
                "246,063 cells · Model-derived groups; no cell-type or biological "
                "annotation is implied.",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#475569",
            )
            figure.subplots_adjust(
                left=0.045, right=0.985, bottom=0.045, top=0.925,
                wspace=0.27, hspace=0.34,
            )
            _atomic_save_png(figure, png_path, dpi=int(dpi))
        finally:
            plt.close(figure)

    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": EXPECTED_RUN_ID,
            "model_epoch": 200,
            "model_seed": 0,
            "clustering_manifest_sha256": source_sha,
            "leiden_resolution": resolution,
            "dpi": int(dpi),
            "point_count": EXPECTED_TOTAL_CELLS,
            "one_dot_per_cell": True,
            "marker_borders": False,
            "png_only": True,
            "interactive_map_created": False,
            "pdf_created": False,
            "plot_specification": {
                "panel_order": list(SO2_CORE_NUMBERS),
                "grid_shape": [3, 5],
                "legend_panel": [2, 4],
                "equal_aspect": True,
                "invert_y_axis": True,
                "coordinate_units": "micrometres",
                "one_shared_joint_cluster_palette": True,
                "palette": dict(palette),
            },
            "files": {f"figures/{PNG_NAME}": _file_record(png_path)},
        }
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt

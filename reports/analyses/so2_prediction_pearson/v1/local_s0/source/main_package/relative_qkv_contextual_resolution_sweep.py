"""hL-only Leiden resolution sweep for the completed six-core analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cancer_pooled_full_core import CANCER_ALIASES, CORE_NUMBERS
from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    EXPECTED_CELL_COUNTS,
    EXPECTED_MODEL_SEED,
    EXPECTED_RUN_ID,
    EXPECTED_TOTAL_CELLS,
    EmbeddingClusterAnalysisError,
    KNNGraphResult,
    LeidenResult,
    _array_sha256,
    _atomic_save_figure_pair,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_parquet,
    _atomic_write_text,
    _canonical_sha256,
    _file_manifest,
    _file_record,
    _read_json,
    _receipt_with_self_hash,
    _style_spatial_axis,
    _verify_self_hash,
    _write_deterministic_npz,
    build_faiss_cosine_knn_graph,
    cluster_summary_tables,
    deterministic_glasbey_palette,
    deterministic_pca,
    run_seeded_leiden,
    verify_embedding_cluster_analysis_bundle,
)


SWEEP_SCHEMA = "cancer_6core_contextual_leiden_resolution_sweep_v1"
GRAPH_STAGE_SCHEMA = "contextual_hl_shared_analysis_graph_v1"
PARTITION_STAGE_SCHEMA = "contextual_hl_resolution_partitions_v1"
PLOTTING_STAGE_SCHEMA = "contextual_hl_resolution_spatial_figures_v1"
DEFAULT_RESOLUTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BASELINE_RESOLUTION = 1.0


@dataclass(frozen=True, slots=True)
class ContextualCore:
    alias: str
    core_number: int
    cell_index: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    hL: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_index))


class ContextualResolutionSweepError(EmbeddingClusterAnalysisError):
    """Raised when the locked hL-only sweep contract is violated."""


def _resolution_token(resolution: float) -> str:
    text = f"{float(resolution):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def _normalize_resolutions(values: Sequence[float]) -> tuple[float, ...]:
    resolutions = tuple(sorted(float(value) for value in values))
    if (
        not resolutions
        or any(not math.isfinite(value) or value <= 0.0 for value in resolutions)
        or len(set(resolutions)) != len(resolutions)
        or BASELINE_RESOLUTION not in resolutions
    ):
        raise ContextualResolutionSweepError(
            "Resolutions must be unique positive finite values and include 1.0."
        )
    tokens = tuple(_resolution_token(value) for value in resolutions)
    if len(set(tokens)) != len(tokens):
        raise ContextualResolutionSweepError(
            "Resolution values collide after filename canonicalization."
        )
    return resolutions


def _source_root(paths: ProjectPaths, run_id: str) -> Path:
    return (
        paths.report_root
        / "analyses"
        / "cancer_6core_embedding_clustering"
        / run_id
    ).resolve(strict=True)


def _resolve_output_root(
    paths: ProjectPaths, *, run_id: str, output_dir: str | Path | None
) -> Path:
    namespace = (
        paths.report_root / "analyses" / "cancer_6core_contextual_resolution_sweep"
    ).resolve(strict=False)
    candidate = namespace / run_id if output_dir is None else Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = paths.project_root / candidate
    if candidate.is_symlink():
        raise ContextualResolutionSweepError("Sweep output may not be a symlink.")
    root = candidate.resolve(strict=False)
    if root == namespace or not root.is_relative_to(namespace):
        raise ContextualResolutionSweepError(
            f"Sweep output must be a run-specific directory beneath {namespace}."
        )
    if root.exists() and not root.is_dir():
        raise ContextualResolutionSweepError("Sweep output must be a directory.")
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise ContextualResolutionSweepError("Sweep output may not contain symlinks.")
    return root


def _load_contextual_cores(
    source_root: Path, source_manifest: Mapping[str, Any]
) -> tuple[ContextualCore, ...]:
    result: list[ContextualCore] = []
    shapes = source_manifest.get("embedding_shapes")
    if not isinstance(shapes, Mapping):
        raise ContextualResolutionSweepError("Source embedding shapes are missing.")
    for core_number, alias in zip(CORE_NUMBERS, CANCER_ALIASES, strict=True):
        path = source_root / "embeddings" / f"core_{core_number}_embeddings.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                cell_index = np.asarray(archive["cell_index"], dtype=np.int64)
                core_value = np.asarray(archive["core_number"])
                coordinates = np.asarray(archive["coordinates_um"], dtype=np.float64)
                contextual = np.asarray(archive["hL"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise ContextualResolutionSweepError(
                f"Cannot load contextual source arrays for core {core_number}."
            ) from exc
        n_cells = EXPECTED_CELL_COUNTS[core_number]
        if (
            not np.array_equal(cell_index, np.arange(n_cells, dtype=np.int64))
            or core_value.size != 1
            or int(core_value.reshape(-1)[0]) != core_number
            or coordinates.shape != (n_cells, 2)
            or contextual.shape != (n_cells, 256)
            or shapes.get(str(core_number), {}).get("hL") != [n_cells, 256]
            or not np.isfinite(coordinates).all()
            or not np.isfinite(contextual).all()
        ):
            raise ContextualResolutionSweepError(
                f"Contextual source alignment changed for core {core_number}."
            )
        result.append(
            ContextualCore(
                alias=alias,
                core_number=core_number,
                cell_index=np.ascontiguousarray(cell_index),
                coordinates_um=np.ascontiguousarray(coordinates),
                hL=np.ascontiguousarray(contextual),
            )
        )
    if sum(core.n_cells for core in result) != EXPECTED_TOTAL_CELLS:
        raise ContextualResolutionSweepError("Contextual source coverage changed.")
    return tuple(result)


def _source_cell_frame(source_root: Path) -> pd.DataFrame:
    columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
    ]
    frame = pd.read_parquet(
        source_root / "tables" / "cell_embedding_clusters.parquet",
        columns=columns,
    )
    if (
        list(frame.columns) != columns
        or len(frame) != EXPECTED_TOTAL_CELLS
        or tuple(frame["core_number"].drop_duplicates()) != CORE_NUMBERS
        or frame["cell_key"].duplicated().any()
        or not np.array_equal(
            frame["global_cell_index"].to_numpy(),
            np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
        )
        or not np.isfinite(frame[["x_um", "y_um"]].to_numpy()).all()
    ):
        raise ContextualResolutionSweepError("Source cell table alignment changed.")
    return frame


def _verify_frame_against_cores(
    frame: pd.DataFrame, cores: Sequence[ContextualCore]
) -> None:
    for core in cores:
        selected = frame.loc[frame["core_number"] == core.core_number]
        if (
            not np.array_equal(selected["cell_index"].to_numpy(), core.cell_index)
            or not np.array_equal(
                selected[["x_um", "y_um"]].to_numpy(), core.coordinates_um
            )
        ):
            raise ContextualResolutionSweepError(
                f"Source table/NPZ order changed for core {core.core_number}."
            )


def _graph_configuration(
    source_manifest: Mapping[str, Any],
    source_clustering: Mapping[str, Any],
    *,
    source_manifest_path: Path,
) -> dict[str, Any]:
    configuration = source_clustering.get("configuration")
    pipeline = source_clustering.get("pipelines", {}).get("contextual")
    if not isinstance(configuration, Mapping) or not isinstance(pipeline, Mapping):
        raise ContextualResolutionSweepError("Source contextual pipeline is missing.")
    return {
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "run_id": EXPECTED_RUN_ID,
        "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "representation": "hL",
        "pca_components": int(configuration["pca_components"]),
        "n_neighbors": int(configuration["n_neighbors"]),
        "distance_metric": str(configuration["distance_metric"]),
        "random_seed": int(configuration["random_seed"]),
        "expected_joint_embedding_sha256": pipeline["joint_embedding_sha256"],
        "expected_pca_scores_sha256": pipeline["pca"][
            "normalized_scores_sha256"
        ],
        "expected_knn_edges_sha256": pipeline["knn"][
            "undirected_edges_sha256"
        ],
    }


def _verify_stage_files(root: Path, files: Mapping[str, Any], *, label: str) -> None:
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.is_file() or _file_record(path) != expected:
            raise ContextualResolutionSweepError(
                f"{label} file checksum changed: {relative}."
            )


def build_or_load_shared_contextual_graph(
    *,
    source_root: Path,
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    source_clustering: Mapping[str, Any],
    output_root: Path,
) -> tuple[KNNGraphResult, Mapping[str, Any]]:
    """Build one hL PCA/kNN graph, checksum-match it, and make it resumable."""

    configuration = _graph_configuration(
        source_manifest,
        source_clustering,
        source_manifest_path=source_manifest_path,
    )
    manifest_path = output_root / "clustering" / "contextual_graph_manifest.json"
    graph_path = output_root / "clustering" / "contextual_analysis_graph.npz"
    if manifest_path.is_file():
        receipt = _read_json(manifest_path, label="contextual graph receipt")
        _verify_self_hash(receipt, label="contextual graph receipt")
        if (
            receipt.get("schema") != GRAPH_STAGE_SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("configuration") != configuration
        ):
            raise ContextualResolutionSweepError("Contextual graph receipt changed.")
        files = receipt.get("files")
        if not isinstance(files, Mapping):
            raise ContextualResolutionSweepError("Contextual graph files are missing.")
        _verify_stage_files(output_root, files, label="contextual graph")
        with np.load(graph_path, allow_pickle=False) as archive:
            edge_pairs = np.asarray(archive["edge_pairs"], dtype=np.int64)
        if _array_sha256("knn_undirected_edges", edge_pairs) != configuration[
            "expected_knn_edges_sha256"
        ]:
            raise ContextualResolutionSweepError("Saved contextual kNN graph changed.")
        return KNNGraphResult(edge_pairs=edge_pairs, receipt=receipt["knn"]), receipt

    clustering_dir = output_root / "clustering"
    if clustering_dir.exists() and any(clustering_dir.iterdir()):
        raise ContextualResolutionSweepError(
            "Partial contextual graph outputs exist without a complete receipt."
        )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    cores = _load_contextual_cores(source_root, source_manifest)
    cell_frame = _source_cell_frame(source_root)
    _verify_frame_against_cores(cell_frame, cores)
    contextual = np.ascontiguousarray(
        np.concatenate([core.hL for core in cores], axis=0), dtype=np.float32
    )
    observed_embedding_sha = _array_sha256("joint_contextual_embedding", contextual)
    if observed_embedding_sha != configuration["expected_joint_embedding_sha256"]:
        raise ContextualResolutionSweepError("Joint hL checksum changed from source.")
    pca = deterministic_pca(
        contextual, n_components=int(configuration["pca_components"])
    )
    if pca.receipt["normalized_scores_sha256"] != configuration[
        "expected_pca_scores_sha256"
    ]:
        raise ContextualResolutionSweepError("Contextual PCA checksum changed.")
    graph = build_faiss_cosine_knn_graph(
        pca.normalized_scores,
        n_neighbors=int(configuration["n_neighbors"]),
        random_seed=int(configuration["random_seed"]),
    )
    if graph.receipt["undirected_edges_sha256"] != configuration[
        "expected_knn_edges_sha256"
    ]:
        raise ContextualResolutionSweepError("Contextual kNN checksum changed.")
    _write_deterministic_npz(graph_path, {"edge_pairs": graph.edge_pairs})
    relative = graph_path.relative_to(output_root).as_posix()
    receipt = _receipt_with_self_hash(
        {
            "schema": GRAPH_STAGE_SCHEMA,
            "status": "complete",
            "configuration": configuration,
            "cell_count": EXPECTED_TOTAL_CELLS,
            "core_order": list(CORE_NUMBERS),
            "h0_or_delta_used": False,
            "pca": pca.receipt,
            "knn": graph.receipt,
            "files": {relative: _file_record(graph_path)},
        }
    )
    _atomic_write_json(manifest_path, receipt)
    _verify_stage_files(output_root, receipt["files"], label="contextual graph")
    return graph, receipt


def _resolution_prefix(resolution: float) -> str:
    return f"R{_resolution_token(resolution)}_C"


def _resolution_palette(
    *,
    resolution: float,
    resolution_index: int,
    cluster_count: int,
    source_palette: Mapping[str, str],
) -> dict[str, str]:
    prefix = _resolution_prefix(resolution)
    if resolution == BASELINE_RESOLUTION:
        if set(source_palette) != {f"C{index}" for index in range(cluster_count)}:
            raise ContextualResolutionSweepError(
                "Source resolution-1.0 palette does not match source labels."
            )
        colors = [str(source_palette[f"C{index}"]) for index in range(cluster_count)]
    else:
        base = deterministic_glasbey_palette(cluster_count, namespace="contextual")
        colors = [base[f"C{index}"] for index in range(cluster_count)]
        offset = (resolution_index * 7) % cluster_count
        colors = colors[offset:] + colors[:offset]
    return {f"{prefix}{index}": color for index, color in enumerate(colors)}


def _partition_configuration(
    *,
    graph_receipt: Mapping[str, Any],
    resolutions: Sequence[float],
    random_seed: int,
) -> dict[str, Any]:
    return {
        "run_id": EXPECTED_RUN_ID,
        "representation": "contextual_hL_only",
        "graph_manifest_content_sha256": graph_receipt["manifest_content_sha256"],
        "shared_knn_edges_sha256": graph_receipt["knn"][
            "undirected_edges_sha256"
        ],
        "resolutions": [float(value) for value in resolutions],
        "resolution_tokens": [_resolution_token(value) for value in resolutions],
        "random_seed": int(random_seed),
        "baseline_resolution": BASELINE_RESOLUTION,
    }


def _verify_partition_receipt(
    output_root: Path,
    receipt: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="contextual resolution partition receipt")
    if (
        receipt.get("schema") != PARTITION_STAGE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("configuration") != configuration
        or tuple(receipt.get("core_order", ())) != CORE_NUMBERS
        or int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
        or receipt.get("baseline_labels_exact_match") is not True
        or receipt.get("h0_or_delta_used") is not False
    ):
        raise ContextualResolutionSweepError("Resolution partition receipt changed.")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise ContextualResolutionSweepError("Resolution partition files are missing.")
    _verify_stage_files(output_root, files, label="resolution partition")


def compute_resolution_partitions(
    graph: KNNGraphResult,
    *,
    resolutions: Sequence[float],
    random_seed: int,
    baseline_labels: np.ndarray,
) -> dict[str, LeidenResult]:
    """Run one independent Leiden partition per resolution on one sparse graph."""

    normalized = _normalize_resolutions(resolutions)
    established = np.asarray(baseline_labels, dtype=np.int64)
    if established.shape != (EXPECTED_TOTAL_CELLS,) or np.any(established < 0):
        raise ContextualResolutionSweepError(
            "Established contextual labels are invalid."
        )
    results: dict[str, LeidenResult] = {}
    for resolution in normalized:
        result = run_seeded_leiden(
            graph,
            n_cells=EXPECTED_TOTAL_CELLS,
            resolution=float(resolution),
            random_seed=int(random_seed),
        )
        labels = np.asarray(result.labels, dtype=np.int64)
        if labels.shape != established.shape or np.any(labels < 0):
            raise ContextualResolutionSweepError(
                f"Leiden labels are invalid at resolution {resolution:g}."
            )
        if resolution == BASELINE_RESOLUTION and not np.array_equal(
            labels, established
        ):
            raise ContextualResolutionSweepError(
                "Resolution-1.0 labels do not match the established contextual labels."
            )
        results[_resolution_token(resolution)] = result
    return results


def build_or_load_resolution_partitions(
    *,
    source_root: Path,
    source_clustering: Mapping[str, Any],
    output_root: Path,
    graph: KNNGraphResult,
    graph_receipt: Mapping[str, Any],
    resolutions: Sequence[float],
    random_seed: int,
) -> Mapping[str, Any]:
    """Run Leiden independently at each resolution on exactly one hL graph."""

    resolutions = _normalize_resolutions(resolutions)
    configuration = _partition_configuration(
        graph_receipt=graph_receipt,
        resolutions=resolutions,
        random_seed=random_seed,
    )
    receipt_path = output_root / "clustering" / "resolution_sweep_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="resolution partition receipt")
        _verify_partition_receipt(
            output_root, receipt, configuration=configuration
        )
        return receipt

    allowed_graph_files = {
        "contextual_analysis_graph.npz",
        "contextual_graph_manifest.json",
    }
    unexpected = sorted(
        path.name
        for path in (output_root / "clustering").iterdir()
        if path.name not in allowed_graph_files
    )
    tables_dir = output_root / "tables"
    if unexpected or (tables_dir.exists() and any(tables_dir.iterdir())):
        raise ContextualResolutionSweepError(
            "Partial resolution outputs exist without a complete receipt."
        )
    tables_dir.mkdir(parents=True, exist_ok=True)
    cell_frame = _source_cell_frame(source_root)
    source_labels = np.load(
        source_root / "clustering" / "contextual_labels.npy", allow_pickle=False
    )
    if source_labels.shape != (EXPECTED_TOTAL_CELLS,):
        raise ContextualResolutionSweepError("Source contextual labels are invalid.")
    expected_source_labels_sha = source_clustering.get("pipelines", {}).get(
        "contextual", {}
    ).get("leiden", {}).get("labels_sha256")
    if (
        _array_sha256("sorted_leiden_labels", source_labels)
        != expected_source_labels_sha
    ):
        raise ContextualResolutionSweepError(
            "Source contextual-label checksum changed."
        )
    source_palette_receipt = _read_json(
        source_root / "clustering" / "contextual_palette.json",
        label="source contextual palette",
    )
    source_palette = source_palette_receipt.get("colors")
    if not isinstance(source_palette, Mapping):
        raise ContextualResolutionSweepError("Source contextual palette is missing.")

    stage_files: dict[str, dict[str, Any]] = {}
    partitions: dict[str, dict[str, Any]] = {}
    palettes: dict[str, dict[str, str]] = {}
    summary_frames: list[pd.DataFrame] = []
    composition_frames: list[pd.DataFrame] = []
    results = compute_resolution_partitions(
        graph,
        resolutions=resolutions,
        random_seed=random_seed,
        baseline_labels=source_labels,
    )
    for resolution_index, resolution in enumerate(resolutions):
        token = _resolution_token(resolution)
        prefix = _resolution_prefix(resolution)
        result = results[token]
        labels = result.labels
        summary, composition, dominated = cluster_summary_tables(
            labels, cell_frame, prefix=prefix
        )
        summary.insert(0, "resolution_token", token)
        summary.insert(0, "leiden_resolution", float(resolution))
        composition.insert(0, "resolution_token", token)
        composition.insert(0, "leiden_resolution", float(resolution))
        summary_frames.append(summary)
        composition_frames.append(composition)
        cluster_count = int(result.receipt["cluster_count"])
        palette = _resolution_palette(
            resolution=float(resolution),
            resolution_index=resolution_index,
            cluster_count=cluster_count,
            source_palette=source_palette,
        )
        palettes[token] = palette
        number_column = f"contextual_cluster_number_r{token}"
        label_column = f"contextual_cluster_r{token}"
        cell_frame[number_column] = labels.astype(np.int32)
        cell_frame[label_column] = [f"{prefix}{int(value)}" for value in labels]

        label_path = output_root / "clustering" / f"labels_resolution_{token}.npy"
        parameter_path = (
            output_root
            / "clustering"
            / f"parameters_resolution_{token}.json"
        )
        palette_path = (
            output_root / "clustering" / f"palette_resolution_{token}.json"
        )
        _atomic_write_npy(label_path, labels)
        _atomic_write_json(
            parameter_path,
            {
                "representation": "contextual_hL_only",
                "leiden_resolution": float(resolution),
                "resolution_token": token,
                "shared_knn_edges_sha256": configuration[
                    "shared_knn_edges_sha256"
                ],
                "leiden": result.receipt,
            },
        )
        _atomic_write_json(
            palette_path,
            {
                "representation": "contextual_hL_only",
                "leiden_resolution": float(resolution),
                "resolution_qualified_prefix": prefix,
                "cross_resolution_cluster_identity_implied": False,
                "colors": palette,
            },
        )
        for path in (label_path, parameter_path, palette_path):
            stage_files[path.relative_to(output_root).as_posix()] = _file_record(path)
        partitions[token] = {
            "leiden_resolution": float(resolution),
            "label_prefix": prefix,
            "cluster_count": cluster_count,
            "cluster_size_range": [
                int(summary["size"].min()),
                int(summary["size"].max()),
            ],
            "core_dominated_gt_90pct": dominated,
            "labels_sha256": result.receipt["labels_sha256"],
            "leiden": result.receipt,
        }

    table_path = output_root / "tables" / "cell_contextual_resolution_clusters.parquet"
    summary_path = output_root / "tables" / "contextual_resolution_cluster_summary.csv"
    composition_path = (
        output_root / "tables" / "contextual_resolution_core_composition.csv"
    )
    _atomic_write_parquet(table_path, cell_frame)
    _atomic_write_csv(summary_path, pd.concat(summary_frames, ignore_index=True))
    _atomic_write_csv(
        composition_path, pd.concat(composition_frames, ignore_index=True)
    )
    for path in (table_path, summary_path, composition_path):
        stage_files[path.relative_to(output_root).as_posix()] = _file_record(path)
    receipt = _receipt_with_self_hash(
        {
            "schema": PARTITION_STAGE_SCHEMA,
            "status": "complete",
            "configuration": configuration,
            "core_order": list(CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "shared_graph_object_reused_for_all_resolutions": True,
            "independent_leiden_partition_per_resolution": True,
            "baseline_labels_exact_match": True,
            "source_contextual_labels_sha256": _array_sha256(
                "sorted_leiden_labels", source_labels
            ),
            "h0_or_delta_used": False,
            "partitions": partitions,
            "palettes": palettes,
            "cell_table_columns": list(cell_frame.columns),
            "files": stage_files,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_partition_receipt(
        output_root, receipt, configuration=configuration
    )
    return receipt


def requested_contextual_panel_order() -> tuple[int, ...]:
    """Return the locked tissue-panel order for every resolution figure."""

    return CORE_NUMBERS


def contextual_resolution_plot_spec(
    *, resolution: float, palette: Mapping[str, str]
) -> dict[str, Any]:
    """Describe the invariant hL-only spatial plotting contract."""

    if not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
        raise ContextualResolutionSweepError("Plot resolution must be positive.")
    if not palette or len(set(palette.values())) != len(palette):
        raise ContextualResolutionSweepError(
            "Each contextual cluster requires one distinct plotting color."
        )
    return {
        "representation": "contextual_hL_only",
        "leiden_resolution": float(resolution),
        "panel_order": list(CORE_NUMBERS),
        "grid_shape": [2, 3],
        "equal_aspect": True,
        "invert_y_axis": True,
        "coordinate_units": "micrometres",
        "one_dot_per_cell": True,
        "point_layer_rasterized_in_pdf": True,
        "marker_borders": False,
        "lines_between_cells": False,
        "palette": dict(palette),
    }


def _resolution_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    def cluster_number(label: str) -> int:
        try:
            return int(label.rsplit("_C", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ContextualResolutionSweepError(
                f"Resolution-qualified cluster label is invalid: {label}."
            ) from exc

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: cluster_number(str(item[0]))
        )
    ]


def _render_contextual_resolution_map(
    frame: pd.DataFrame,
    *,
    resolution: float,
    palette: Mapping[str, str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    token = _resolution_token(resolution)
    label_column = f"contextual_cluster_r{token}"
    if label_column not in frame.columns:
        raise ContextualResolutionSweepError(
            f"Contextual labels are missing for resolution {resolution:g}."
        )
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
    for axis, core_number in zip(axes.ravel(), CORE_NUMBERS, strict=True):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected[label_column].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS[core_number] or colors.isna().any():
            raise ContextualResolutionSweepError(
                f"Contextual palette or cell coverage is incomplete for core {core_number}."
            )
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.75,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.90,
            rasterized=True,
        )
        axis.set_title(
            f"Cancer Core {core_number} (n={len(selected):,})", weight="bold"
        )
        _style_spatial_axis(axis, coordinates)
    figure.suptitle(
        "Contextualized hL Leiden clusters — "
        f"joint six-core graph, resolution {resolution:g}",
        fontsize=15,
        weight="bold",
    )
    handles = _resolution_legend_handles(palette)
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.865, 0.5),
        frameon=False,
        ncol=2 if len(handles) > 24 else 1,
        title=f"Model-derived cluster (r={resolution:g})",
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.85,
        bottom=0.07,
        top=0.92,
        wspace=0.25,
        hspace=0.27,
    )
    _atomic_save_figure_pair(
        figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi
    )
    plt.close(figure)


def _required_figure_files(resolutions: Sequence[float]) -> tuple[str, ...]:
    files: list[str] = []
    for resolution in resolutions:
        stem = (
            "figures/contextual_leiden_resolution_"
            f"{_resolution_token(resolution)}_spatial_6cores"
        )
        files.extend(f"{stem}.{suffix}" for suffix in ("png", "pdf"))
    return tuple(files)


def _plotting_configuration(
    *, partition_receipt: Mapping[str, Any], resolutions: Sequence[float], dpi: int
) -> dict[str, Any]:
    return {
        "representation": "contextual_hL_only",
        "partition_manifest_content_sha256": partition_receipt[
            "manifest_content_sha256"
        ],
        "resolutions": [float(value) for value in resolutions],
        "resolution_tokens": [_resolution_token(value) for value in resolutions],
        "core_order": list(CORE_NUMBERS),
        "dpi": int(dpi),
    }


def _verify_plotting_receipt(
    output_root: Path,
    receipt: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="contextual resolution plotting receipt")
    expected_figures = set(_required_figure_files(configuration["resolutions"]))
    files = receipt.get("files")
    if (
        receipt.get("schema") != PLOTTING_STAGE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("configuration") != configuration
        or int(receipt.get("figure_count", -1)) != len(expected_figures)
        or receipt.get("h0_figures_created") is not False
        or receipt.get("delta_h_figures_created") is not False
        or not isinstance(files, Mapping)
        or set(files) != expected_figures
    ):
        raise ContextualResolutionSweepError(
            "Contextual resolution plotting receipt changed."
        )
    _verify_stage_files(output_root, files, label="contextual resolution plotting")


def render_contextual_resolution_figures(
    *,
    output_root: Path,
    partition_receipt: Mapping[str, Any],
    resolutions: Sequence[float],
    dpi: int = 300,
) -> Mapping[str, Any]:
    """Render one hL-only six-core tissue map for every Leiden resolution."""

    normalized = _normalize_resolutions(resolutions)
    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise ContextualResolutionSweepError("Figure DPI must be positive.")
    configuration = _plotting_configuration(
        partition_receipt=partition_receipt,
        resolutions=normalized,
        dpi=int(dpi),
    )
    figure_dir = output_root / "figures"
    receipt_path = figure_dir / "plotting_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="contextual resolution plotting receipt")
        _verify_plotting_receipt(
            output_root, receipt, configuration=configuration
        )
        return receipt

    expected_figures = set(_required_figure_files(normalized))
    if figure_dir.exists():
        unexpected = sorted(
            path.relative_to(output_root).as_posix()
            for path in figure_dir.rglob("*")
            if path.is_file()
            and path.relative_to(output_root).as_posix() not in expected_figures
        )
        if unexpected:
            raise ContextualResolutionSweepError(
                "Figure directory contains unrecognized partial outputs: "
                + ", ".join(unexpected)
            )
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(
        output_root / "tables" / "cell_contextual_resolution_clusters.parquet"
    )
    if (
        len(frame) != EXPECTED_TOTAL_CELLS
        or tuple(frame["core_number"].drop_duplicates().tolist()) != CORE_NUMBERS
        or not np.array_equal(
            frame["global_cell_index"].to_numpy(),
            np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
        )
    ):
        raise ContextualResolutionSweepError(
            "Contextual plotting table does not retain all ordered cells."
        )
    palettes = partition_receipt.get("palettes")
    partitions = partition_receipt.get("partitions")
    if not isinstance(palettes, Mapping) or not isinstance(partitions, Mapping):
        raise ContextualResolutionSweepError(
            "Resolution partitions lack plotting metadata."
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 140,
            "savefig.dpi": int(dpi),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    outputs: list[Path] = []
    specifications: dict[str, Mapping[str, Any]] = {}
    for resolution in normalized:
        token = _resolution_token(resolution)
        palette_value = palettes.get(token)
        partition = partitions.get(token)
        if not isinstance(palette_value, Mapping) or not isinstance(
            partition, Mapping
        ):
            raise ContextualResolutionSweepError(
                f"Plotting metadata is missing at resolution {resolution:g}."
            )
        palette = {str(key): str(value) for key, value in palette_value.items()}
        expected_labels = {
            f"{_resolution_prefix(resolution)}{index}"
            for index in range(int(partition["cluster_count"]))
        }
        if set(palette) != expected_labels:
            raise ContextualResolutionSweepError(
                f"Contextual palette labels changed at resolution {resolution:g}."
            )
        stem = f"contextual_leiden_resolution_{token}_spatial_6cores"
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        _render_contextual_resolution_map(
            frame,
            resolution=float(resolution),
            palette=palette,
            png_path=png_path,
            pdf_path=pdf_path,
            dpi=int(dpi),
        )
        outputs.extend((png_path, pdf_path))
        specifications[token] = contextual_resolution_plot_spec(
            resolution=float(resolution), palette=palette
        )
    if set(path.relative_to(output_root).as_posix() for path in outputs) != expected_figures:
        raise ContextualResolutionSweepError(
            "Requested contextual resolution figures are incomplete."
        )
    receipt = _receipt_with_self_hash(
        {
            "schema": PLOTTING_STAGE_SCHEMA,
            "status": "complete",
            "configuration": configuration,
            "figure_count": len(outputs),
            "combined_six_core_figure_pairs": len(normalized),
            "one_dot_per_cell": True,
            "point_layer_rasterized_in_pdf": True,
            "h0_figures_created": False,
            "delta_h_figures_created": False,
            "plot_specifications": specifications,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in outputs
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_plotting_receipt(
        output_root, receipt, configuration=configuration
    )
    return receipt


def _render_sweep_readme(
    *,
    source_root: Path,
    source_manifest: Mapping[str, Any],
    graph_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    output_root: Path,
) -> str:
    resolutions = tuple(
        float(value)
        for value in partition_receipt["configuration"]["resolutions"]
    )
    project_root = source_root.parents[3]
    output_display = (
        output_root.relative_to(project_root).as_posix()
        if output_root.is_relative_to(project_root)
        else output_root.as_posix()
    )
    counts = ", ".join(
        f"Core {core}: {int(source_manifest['cell_counts'][str(core)]):,}"
        for core in CORE_NUMBERS
    )
    rows: list[str] = []
    for resolution in resolutions:
        token = _resolution_token(resolution)
        partition = partition_receipt["partitions"][token]
        dominated = ", ".join(partition["core_dominated_gt_90pct"]) or "none"
        size_min, size_max = partition["cluster_size_range"]
        rows.append(
            f"| {resolution:g} | {partition['cluster_count']} | "
            f"{int(size_min):,}–{int(size_max):,} | {dominated} |"
        )
    resolution_arguments = " ".join(f"{value:g}" for value in resolutions)
    command = (
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\\n"
        "  analyze-contextual-resolution-sweep \\\n"
        f"  --run-id {source_manifest['run_id']} \\\n"
        f"  --resolutions {resolution_arguments} \\\n"
        f"  --random-seed {partition_receipt['configuration']['random_seed']} \\\n"
        f"  --output-dir {shlex.quote(output_display)}"
    )
    return f"""# Contextual hL Leiden resolution sweep

Status: complete exploratory sensitivity analysis of the locked contextual
embedding. This bundle uses only the final graph embedding `hL`. It creates no
intrinsic `h0` clustering figures and no delta-h figures. The model was not
retrained and model inference was not rerun.

## Source and coverage

- Source run ID: `{source_manifest['run_id']}`
- Model seed: `{source_manifest['model_seed']}`
- Source checkpoint: `{source_manifest['checkpoint_path']}`
- Checkpoint SHA-256: `{source_manifest['checkpoint_sha256']}`
- Source analysis bundle: `{source_root.as_posix()}`
- Cells: {counts}; total {source_manifest['total_cells']:,}
- Contextual embedding width: {source_manifest['model_construction']['embedding_dim']}

All cells from cores 1, 9, 13, 15, 21, and 23 were clustered jointly. One
mean-centered 50-component PCA representation and one cosine 30-nearest-neighbor
FAISS-HNSW graph were computed from `hL`, checksum-matched to the completed
analysis, saved, and reused without alteration for every resolution. Only the
Leiden resolution parameter varies. The shared kNN edge checksum is
`{graph_receipt['knn']['undirected_edges_sha256']}`. Resolution 1.0 labels match
the established contextual labels exactly.

## Resolution results

| Leiden resolution | Clusters | Cluster-size range | Clusters >90% from one core |
|---:|---:|---:|:---|
{chr(10).join(rows)}

Cluster identifiers are resolution-qualified. A similarly numbered cluster at
two resolutions is not asserted to be the same population. Higher resolution
changes clustering granularity; it is not evidence that one setting is more
biologically correct.

## Interpretation constraints

These contextual clusters represent patterns after graph-based neighborhood
processing by the trained model. They are model-derived partitions, not
independently established cell types. Neither a partition nor its spatial
appearance independently establishes cell type, signaling, biological
influence, or causality. Marker-based and pathological validation will be
conducted separately. Core-dominated clusters are reported without removal,
merging, or batch integration.

## Reproduction

From the repository root:

```bash
{command}
```

The workflow is resumable. Checksum-valid shared-graph and partition receipts
skip PCA, kNN construction, and Leiden if plotting is interrupted. Every final
table and figure is checksum-bound by `manifest.json`.
"""


def _required_sweep_files(resolutions: Sequence[float]) -> set[str]:
    required = {
        "README.md",
        "clustering/contextual_analysis_graph.npz",
        "clustering/contextual_graph_manifest.json",
        "clustering/resolution_sweep_manifest.json",
        "tables/cell_contextual_resolution_clusters.parquet",
        "tables/contextual_resolution_cluster_summary.csv",
        "tables/contextual_resolution_core_composition.csv",
        "figures/plotting_manifest.json",
        *_required_figure_files(resolutions),
    }
    for resolution in resolutions:
        token = _resolution_token(resolution)
        required.update(
            {
                f"clustering/labels_resolution_{token}.npy",
                f"clustering/parameters_resolution_{token}.json",
                f"clustering/palette_resolution_{token}.json",
            }
        )
    return required


def _is_sha256(value: object) -> bool:
    text_value = str(value)
    return len(text_value) == 64 and all(
        character in "0123456789abcdef" for character in text_value
    )


def _verify_contextual_table_and_labels(
    *,
    root: Path,
    source_root: Path,
    partition_receipt: Mapping[str, Any],
    resolutions: Sequence[float],
) -> None:
    base_columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
    ]
    expected_columns = list(base_columns)
    for resolution in resolutions:
        token = _resolution_token(resolution)
        expected_columns.extend(
            [
                f"contextual_cluster_number_r{token}",
                f"contextual_cluster_r{token}",
            ]
        )
    frame = pd.read_parquet(
        root / "tables" / "cell_contextual_resolution_clusters.parquet"
    )
    if (
        list(frame.columns) != expected_columns
        or len(frame) != EXPECTED_TOTAL_CELLS
        or tuple(frame["core_number"].drop_duplicates().tolist()) != CORE_NUMBERS
        or frame["cell_key"].duplicated().any()
        or not np.array_equal(
            frame["global_cell_index"].to_numpy(),
            np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
        )
        or not np.isfinite(frame[["x_um", "y_um"]].to_numpy()).all()
        or any(
            prohibited in column.lower()
            for column in frame.columns
            for prohibited in ("intrinsic", "delta", "h0")
        )
    ):
        raise ContextualResolutionSweepError(
            "Contextual resolution cell table identity changed."
        )
    source_labels = np.load(
        source_root / "clustering" / "contextual_labels.npy", allow_pickle=False
    )
    partitions = partition_receipt.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ContextualResolutionSweepError("Resolution partitions are missing.")
    for resolution in resolutions:
        token = _resolution_token(resolution)
        partition = partitions.get(token)
        if not isinstance(partition, Mapping):
            raise ContextualResolutionSweepError(
                f"Partition is missing at resolution {resolution:g}."
            )
        labels = np.load(
            root / "clustering" / f"labels_resolution_{token}.npy",
            allow_pickle=False,
        )
        unique = np.unique(labels)
        cluster_count = int(partition["cluster_count"])
        if (
            labels.shape != (EXPECTED_TOTAL_CELLS,)
            or np.any(labels < 0)
            or not np.array_equal(unique, np.arange(cluster_count, dtype=np.int64))
            or _array_sha256("sorted_leiden_labels", labels)
            != partition["labels_sha256"]
            or not np.array_equal(
                frame[f"contextual_cluster_number_r{token}"].to_numpy(), labels
            )
        ):
            raise ContextualResolutionSweepError(
                f"Labels changed at resolution {resolution:g}."
            )
        expected_text = np.asarray(
            [f"{_resolution_prefix(resolution)}{int(value)}" for value in labels],
            dtype=object,
        )
        if not np.array_equal(
            frame[f"contextual_cluster_r{token}"].to_numpy(dtype=object),
            expected_text,
        ):
            raise ContextualResolutionSweepError(
                f"Text labels changed at resolution {resolution:g}."
            )
        if resolution == BASELINE_RESOLUTION and not np.array_equal(
            labels, source_labels
        ):
            raise ContextualResolutionSweepError(
                "Resolution-1.0 no longer matches source contextual labels."
            )
        parameters = _read_json(
            root / "clustering" / f"parameters_resolution_{token}.json",
            label=f"resolution {resolution:g} parameters",
        )
        palette_receipt = _read_json(
            root / "clustering" / f"palette_resolution_{token}.json",
            label=f"resolution {resolution:g} palette",
        )
        palette = palette_receipt.get("colors")
        if (
            parameters.get("representation") != "contextual_hL_only"
            or float(parameters.get("leiden_resolution", math.nan)) != resolution
            or parameters.get("shared_knn_edges_sha256")
            != partition_receipt["configuration"]["shared_knn_edges_sha256"]
            or parameters.get("leiden") != partition.get("leiden")
            or palette_receipt.get("representation") != "contextual_hL_only"
            or not isinstance(palette, Mapping)
            or set(palette)
            != {
                f"{_resolution_prefix(resolution)}{index}"
                for index in range(cluster_count)
            }
            or len(set(palette.values())) != cluster_count
        ):
            raise ContextualResolutionSweepError(
                f"Parameters or palette changed at resolution {resolution:g}."
            )


def verify_contextual_resolution_sweep_bundle(
    bundle_path: str | Path,
) -> Mapping[str, Any]:
    """Strictly verify the complete hL-only resolution-sweep bundle."""

    root = Path(bundle_path).expanduser().resolve(strict=True)
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ContextualResolutionSweepError(
            "Contextual resolution bundle is missing or has symlinks."
        )
    manifest = _read_json(root / "manifest.json", label="final sweep manifest")
    _verify_self_hash(manifest, label="final sweep manifest")
    try:
        resolutions = _normalize_resolutions(manifest["resolutions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextualResolutionSweepError(
            "Final sweep resolutions are invalid."
        ) from exc
    if (
        manifest.get("schema") != SWEEP_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("run_id") != EXPECTED_RUN_ID
        or int(manifest.get("model_seed", -1)) != EXPECTED_MODEL_SEED
        or manifest.get("representation") != "contextual_hL_only"
        or manifest.get("h0_or_delta_used") is not False
        or tuple(manifest.get("core_order", ())) != CORE_NUMBERS
        or int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
        or int(manifest.get("random_seed", -1)) < 0
        or not _is_sha256(manifest.get("checkpoint_sha256"))
        or not _is_sha256(manifest.get("source_analysis_manifest_file_sha256"))
    ):
        raise ContextualResolutionSweepError(
            "Final contextual resolution manifest identity is invalid."
        )
    expected_files = manifest.get("files")
    observed_files = _file_manifest(root)
    required_files = _required_sweep_files(resolutions)
    if (
        not isinstance(expected_files, Mapping)
        or dict(expected_files) != observed_files
        or set(expected_files) != required_files
        or any(
            prohibited in relative.lower()
            for relative in expected_files
            for prohibited in ("intrinsic", "delta", "h0")
        )
    ):
        raise ContextualResolutionSweepError(
            "Final contextual-only file inventory changed."
        )

    source_root = Path(str(manifest.get("source_analysis_bundle", ""))).resolve(
        strict=True
    )
    source_manifest_path = source_root / "manifest.json"
    if (
        source_root.name != EXPECTED_RUN_ID
        or sha256_file(source_manifest_path)
        != manifest["source_analysis_manifest_file_sha256"]
    ):
        raise ContextualResolutionSweepError("Source analysis identity changed.")
    source_manifest = verify_embedding_cluster_analysis_bundle(source_root)
    if (
        source_manifest.get("run_id") != EXPECTED_RUN_ID
        or source_manifest.get("checkpoint_sha256")
        != manifest.get("checkpoint_sha256")
        or source_manifest.get("manifest_content_sha256")
        != manifest.get("source_analysis_manifest_content_sha256")
        or source_manifest.get("cell_counts") != manifest.get("cell_counts")
    ):
        raise ContextualResolutionSweepError("Source analysis provenance changed.")

    graph_path = root / "clustering" / "contextual_graph_manifest.json"
    partition_path = root / "clustering" / "resolution_sweep_manifest.json"
    plotting_path = root / "figures" / "plotting_manifest.json"
    if (
        sha256_file(graph_path) != manifest.get("graph_manifest_sha256")
        or sha256_file(partition_path) != manifest.get("partition_manifest_sha256")
        or sha256_file(plotting_path) != manifest.get("plotting_manifest_sha256")
    ):
        raise ContextualResolutionSweepError("Sweep stage manifest checksum changed.")
    graph_receipt = _read_json(graph_path, label="contextual graph receipt")
    _verify_self_hash(graph_receipt, label="contextual graph receipt")
    graph_configuration = _graph_configuration(
        source_manifest,
        _read_json(
            source_root / "clustering" / "clustering_manifest.json",
            label="source clustering receipt",
        ),
        source_manifest_path=source_manifest_path,
    )
    if (
        graph_receipt.get("schema") != GRAPH_STAGE_SCHEMA
        or graph_receipt.get("status") != "complete"
        or graph_receipt.get("configuration") != graph_configuration
        or graph_receipt.get("h0_or_delta_used") is not False
        or graph_receipt.get("pca", {}).get("normalized_scores_sha256")
        != graph_configuration["expected_pca_scores_sha256"]
        or graph_receipt.get("knn", {}).get("undirected_edges_sha256")
        != graph_configuration["expected_knn_edges_sha256"]
    ):
        raise ContextualResolutionSweepError("Shared contextual graph identity changed.")
    graph_files = graph_receipt.get("files")
    if not isinstance(graph_files, Mapping):
        raise ContextualResolutionSweepError(
            "Shared contextual graph files are missing."
        )
    _verify_stage_files(root, graph_files, label="shared contextual graph")
    with np.load(
        root / "clustering" / "contextual_analysis_graph.npz", allow_pickle=False
    ) as archive:
        if set(archive.files) != {"edge_pairs"}:
            raise ContextualResolutionSweepError(
                "Shared contextual graph archive contains unexpected arrays."
            )
        edge_pairs = np.asarray(archive["edge_pairs"], dtype=np.int64)
    if (
        edge_pairs.ndim != 2
        or edge_pairs.shape[1] != 2
        or len(edge_pairs) == 0
        or np.any(edge_pairs < 0)
        or np.any(edge_pairs >= EXPECTED_TOTAL_CELLS)
        or np.any(edge_pairs[:, 0] == edge_pairs[:, 1])
        or _array_sha256("knn_undirected_edges", edge_pairs)
        != graph_receipt["knn"]["undirected_edges_sha256"]
        or graph_receipt["knn"].get("cell_by_cell_matrix_constructed") is not False
    ):
        raise ContextualResolutionSweepError("Shared sparse contextual graph changed.")

    partition_receipt = _read_json(
        partition_path, label="contextual resolution partition receipt"
    )
    partition_configuration = _partition_configuration(
        graph_receipt=graph_receipt,
        resolutions=resolutions,
        random_seed=int(manifest["random_seed"]),
    )
    _verify_partition_receipt(
        root, partition_receipt, configuration=partition_configuration
    )
    plotting_receipt = _read_json(
        plotting_path, label="contextual resolution plotting receipt"
    )
    plotting_configuration = _plotting_configuration(
        partition_receipt=partition_receipt,
        resolutions=resolutions,
        dpi=int(manifest["figure_dpi"]),
    )
    _verify_plotting_receipt(
        root, plotting_receipt, configuration=plotting_configuration
    )
    _verify_contextual_table_and_labels(
        root=root,
        source_root=source_root,
        partition_receipt=partition_receipt,
        resolutions=resolutions,
    )
    cluster_counts = manifest.get("cluster_counts")
    size_ranges = manifest.get("cluster_size_ranges")
    dominated = manifest.get("core_dominated_gt_90pct")
    if not all(
        isinstance(value, Mapping)
        for value in (cluster_counts, size_ranges, dominated)
    ):
        raise ContextualResolutionSweepError("Sweep cluster summaries are missing.")
    for resolution in resolutions:
        token = _resolution_token(resolution)
        partition = partition_receipt["partitions"][token]
        if (
            int(cluster_counts.get(token, -1)) != int(partition["cluster_count"])
            or size_ranges.get(token) != partition["cluster_size_range"]
            or dominated.get(token) != partition["core_dominated_gt_90pct"]
        ):
            raise ContextualResolutionSweepError(
                f"Final summaries changed at resolution {resolution:g}."
            )
    interpretation = manifest.get("interpretation")
    if not isinstance(interpretation, Mapping) or any(
        interpretation.get(name) is not False
        for name in (
            "establishes_cell_type",
            "establishes_signaling",
            "establishes_biological_influence",
            "establishes_causality",
        )
    ) or interpretation.get("marker_and_pathology_validation_separate") is not True:
        raise ContextualResolutionSweepError(
            "Sweep interpretation constraints are incomplete."
        )
    return manifest


def run_contextual_resolution_sweep(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None,
    resolutions: Sequence[float] = DEFAULT_RESOLUTIONS,
    random_seed: int = 20260825,
    output_dir: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run the complete resumable hL-only Leiden resolution sweep."""

    normalized = _normalize_resolutions(resolutions)
    if isinstance(random_seed, bool) or int(random_seed) < 0:
        raise ContextualResolutionSweepError("--random-seed must be non-negative.")
    canonical_run_id = registry.resolve_run_id(str(run_id or EXPECTED_RUN_ID))
    if canonical_run_id != EXPECTED_RUN_ID:
        raise ContextualResolutionSweepError(
            "This sweep is locked to the completed six-core seed-0 run."
        )
    run_record = registry.show_run(canonical_run_id)
    if run_record is None or str(run_record.get("status")) != "completed":
        raise ContextualResolutionSweepError("Selected source run is not completed.")
    source_root = _source_root(paths, canonical_run_id)
    source_manifest_path = source_root / "manifest.json"
    source_manifest_sha = sha256_file(source_manifest_path)
    source_manifest = verify_embedding_cluster_analysis_bundle(source_root)
    source_clustering = _read_json(
        source_root / "clustering" / "clustering_manifest.json",
        label="source clustering receipt",
    )
    source_seed = int(source_clustering["configuration"]["random_seed"])
    if int(random_seed) != source_seed:
        raise ContextualResolutionSweepError(
            f"The exact resolution-1.0 anchor requires random seed {source_seed}."
        )
    output_root = _resolve_output_root(
        paths,
        run_id=canonical_run_id,
        output_dir=output_dir,
    )
    if output_root.exists():
        allowed_top_level = {
            "clustering",
            "tables",
            "figures",
            "README.md",
            "manifest.json",
        }
        unexpected = sorted(
            path.name
            for path in output_root.iterdir()
            if path.name not in allowed_top_level
        )
        if unexpected:
            raise ContextualResolutionSweepError(
                "Sweep output contains unrecognized entries: "
                + ", ".join(unexpected)
            )
    final_manifest_path = output_root / "manifest.json"
    if final_manifest_path.is_file():
        manifest = verify_contextual_resolution_sweep_bundle(output_root)
        if (
            tuple(float(value) for value in manifest["resolutions"]) != normalized
            or int(manifest["random_seed"]) != int(random_seed)
            or manifest["source_analysis_manifest_file_sha256"]
            != source_manifest_sha
        ):
            raise ContextualResolutionSweepError(
                "Completed output does not match the requested sweep. "
                "Use a distinct --output-dir."
            )
        return {
            "status": "complete",
            "resumed": True,
            "run_id": canonical_run_id,
            "output_dir": output_root.as_posix(),
            "manifest_sha256": sha256_file(final_manifest_path),
            "resolutions": list(normalized),
            "cluster_counts": manifest["cluster_counts"],
            "figures": [
                (output_root / relative).as_posix()
                for relative in _required_figure_files(normalized)
            ],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    graph, graph_receipt = build_or_load_shared_contextual_graph(
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        source_clustering=source_clustering,
        output_root=output_root,
    )
    partition_receipt = build_or_load_resolution_partitions(
        source_root=source_root,
        source_clustering=source_clustering,
        output_root=output_root,
        graph=graph,
        graph_receipt=graph_receipt,
        resolutions=normalized,
        random_seed=int(random_seed),
    )
    plotting_receipt = render_contextual_resolution_figures(
        output_root=output_root,
        partition_receipt=partition_receipt,
        resolutions=normalized,
        dpi=300,
    )
    readme_path = output_root / "README.md"
    _atomic_write_text(
        readme_path,
        _render_sweep_readme(
            source_root=source_root,
            source_manifest=source_manifest,
            graph_receipt=graph_receipt,
            partition_receipt=partition_receipt,
            output_root=output_root,
        ),
    )
    partitions = partition_receipt["partitions"]
    cluster_counts = {
        token: int(partition["cluster_count"])
        for token, partition in partitions.items()
    }
    cluster_size_ranges = {
        token: partition["cluster_size_range"]
        for token, partition in partitions.items()
    }
    dominated = {
        token: partition["core_dominated_gt_90pct"]
        for token, partition in partitions.items()
    }
    files = _file_manifest(output_root)
    manifest = _receipt_with_self_hash(
        {
            "schema": SWEEP_SCHEMA,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "analysis_classification": "exploratory_contextual_resolution_sensitivity",
            "campaign_id": source_manifest["campaign_id"],
            "run_id": canonical_run_id,
            "model_seed": int(source_manifest["model_seed"]),
            "checkpoint_path": source_manifest["checkpoint_path"],
            "checkpoint_sha256": source_manifest["checkpoint_sha256"],
            "model_construction": source_manifest["model_construction"],
            "representation": "contextual_hL_only",
            "h0_or_delta_used": False,
            "source_analysis_bundle": source_root.as_posix(),
            "source_analysis_manifest_file_sha256": source_manifest_sha,
            "source_analysis_manifest_content_sha256": source_manifest[
                "manifest_content_sha256"
            ],
            "source_analysis_file_count": len(source_manifest["files"]),
            "core_order": list(CORE_NUMBERS),
            "cell_counts": source_manifest["cell_counts"],
            "total_cells": EXPECTED_TOTAL_CELLS,
            "contextual_embedding_shapes": {
                str(core): source_manifest["embedding_shapes"][str(core)]["hL"]
                for core in CORE_NUMBERS
            },
            "resolutions": [float(value) for value in normalized],
            "random_seed": int(random_seed),
            "shared_analysis_graph": {
                "pca_components": graph_receipt["configuration"]["pca_components"],
                "n_neighbors": graph_receipt["configuration"]["n_neighbors"],
                "distance_metric": graph_receipt["configuration"]["distance_metric"],
                "pca_scores_sha256": graph_receipt["pca"][
                    "normalized_scores_sha256"
                ],
                "knn_edges_sha256": graph_receipt["knn"][
                    "undirected_edges_sha256"
                ],
                "reused_for_all_resolutions": True,
                "cell_by_cell_dense_matrix_constructed": False,
            },
            "baseline_resolution_1_labels_exact_match": True,
            "cluster_counts": cluster_counts,
            "cluster_size_ranges": cluster_size_ranges,
            "core_dominated_gt_90pct": dominated,
            "figure_dpi": 300,
            "plotting": plotting_receipt,
            "graph_manifest_sha256": sha256_file(
                output_root / "clustering" / "contextual_graph_manifest.json"
            ),
            "partition_manifest_sha256": sha256_file(
                output_root / "clustering" / "resolution_sweep_manifest.json"
            ),
            "plotting_manifest_sha256": sha256_file(
                output_root / "figures" / "plotting_manifest.json"
            ),
            "interpretation": {
                "contextual": "patterns after graph-based neighborhood processing",
                "resolution_role": "clustering granularity sensitivity parameter",
                "establishes_cell_type": False,
                "establishes_signaling": False,
                "establishes_biological_influence": False,
                "establishes_causality": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": files,
        }
    )
    _atomic_write_json(final_manifest_path, manifest)
    verified = verify_contextual_resolution_sweep_bundle(output_root)
    if sha256_file(source_manifest_path) != source_manifest_sha:
        raise ContextualResolutionSweepError(
            "Completed source analysis changed while creating the sweep."
        )
    verify_embedding_cluster_analysis_bundle(source_root)
    return {
        "status": "complete",
        "resumed": False,
        "run_id": canonical_run_id,
        "output_dir": output_root.as_posix(),
        "manifest_sha256": sha256_file(final_manifest_path),
        "resolutions": list(normalized),
        "cluster_counts": verified["cluster_counts"],
        "cluster_size_ranges": verified["cluster_size_ranges"],
        "core_dominated_gt_90pct": verified["core_dominated_gt_90pct"],
        "figures": [
            (output_root / relative).as_posix()
            for relative in _required_figure_files(normalized)
        ],
    }


__all__ = [
    "BASELINE_RESOLUTION",
    "ContextualResolutionSweepError",
    "DEFAULT_RESOLUTIONS",
    "GRAPH_STAGE_SCHEMA",
    "PARTITION_STAGE_SCHEMA",
    "PLOTTING_STAGE_SCHEMA",
    "SWEEP_SCHEMA",
    "build_or_load_resolution_partitions",
    "build_or_load_shared_contextual_graph",
    "compute_resolution_partitions",
    "contextual_resolution_plot_spec",
    "render_contextual_resolution_figures",
    "requested_contextual_panel_order",
    "run_contextual_resolution_sweep",
    "verify_contextual_resolution_sweep_bundle",
]

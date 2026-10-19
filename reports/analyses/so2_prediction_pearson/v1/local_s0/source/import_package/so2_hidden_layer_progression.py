"""Matched joint clustering across h0--hL for the locked SO2 four-block model.

This is a descriptive post-hoc readout.  Every representation is clustered
independently; equal numeric cluster identifiers across layers do not imply a
shared biological identity or a lineage.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import errno
import gc
import math
import multiprocessing
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    _array_sha256,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_parquet,
    _atomic_write_text,
    _file_manifest,
    _file_record,
    _git_provenance,
    _read_json,
    _receipt_with_self_hash,
    _verify_self_hash,
    build_faiss_cosine_knn_graph,
    deterministic_glasbey_palette,
    deterministic_pca,
    run_seeded_leiden,
)
from .so2_hl_clustering import (
    DEFAULT_LEIDEN_RESOLUTION,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_RANDOM_SEED,
    EXPECTED_RUN_ID,
    SO2ContextualCore,
    build_cell_index_frame,
    cluster_summary_tables,
    resolve_so2_analysis_inputs,
    run_so2_hl_clustering,
    validate_cpu_device,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
)


ANALYSIS_SCHEMA = "so2_14core_hidden_layer_progression_v1"
CLUSTERING_SCHEMA = "so2_hidden_layer_joint_clustering_v1"
ANALYSIS_FAMILY = "so2_14core_hidden_layer_progression"
NEW_LAYERS = ("h0", "h1", "h2", "h3")
ALL_LAYERS = (*NEW_LAYERS, "hL")
REQUESTED_MAP_LAYERS = ("h0", "h1", "h2", "h3")
LAYER_PREFIX = {layer: f"{layer.upper()}C" for layer in ALL_LAYERS}


class SO2HiddenLayerProgressionError(ValueError):
    """Raised when the locked progression-analysis contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_code_records(
    project_root: Path, output_root: Path
) -> dict[str, Mapping[str, Any]]:
    relative_paths = (
        "src/spatial_benchmark/cli.py",
        "src/spatial_benchmark/relative_qkv_graph_transformer.py",
        "src/spatial_benchmark/so2_hidden_layer_progression.py",
        "src/spatial_benchmark/so2_hidden_layer_progression_extraction.py",
        "src/spatial_benchmark/so2_hidden_layer_progression_figures.py",
    )
    records: dict[str, Mapping[str, Any]] = {}
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise SO2HiddenLayerProgressionError(
                f"Analysis source file is missing: {relative}"
            )
        snapshot = output_root / "provenance" / "source" / relative
        _atomic_write_text(snapshot, path.read_text(encoding="utf-8"))
        source_record = _file_record(path)
        snapshot_record = _file_record(snapshot)
        if source_record != snapshot_record:
            raise SO2HiddenLayerProgressionError(
                f"Source snapshot differs from its input: {relative}"
            )
        records[relative] = {
            "working_tree_file": source_record,
            "snapshot_path": snapshot.relative_to(output_root).as_posix(),
            "snapshot_file": snapshot_record,
        }
    return records


def _atomic_publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one report directory without replacing any target."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - platform guard
        raise SO2HiddenLayerProgressionError(
            "Atomic no-replace report publication is unavailable."
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Refusing to replace a concurrently created progression report",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def clustering_configuration(
    *,
    n_neighbors: int,
    leiden_resolution: float,
    pca_components: int,
    random_seed: int,
) -> dict[str, Any]:
    if isinstance(n_neighbors, bool) or int(n_neighbors) <= 0:
        raise SO2HiddenLayerProgressionError("n_neighbors must be positive.")
    if isinstance(pca_components, bool) or int(pca_components) <= 0:
        raise SO2HiddenLayerProgressionError("pca_components must be positive.")
    if not math.isfinite(float(leiden_resolution)) or float(leiden_resolution) <= 0:
        raise SO2HiddenLayerProgressionError("Leiden resolution must be positive.")
    return {
        "joint_core_order": list(SO2_CORE_NUMBERS),
        "joint_cell_count": EXPECTED_TOTAL_CELLS,
        "mean_center": True,
        "pca_components_requested": int(pca_components),
        "l2_normalize_after_pca": True,
        "n_neighbors": int(n_neighbors),
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union",
        "leiden_resolution": float(leiden_resolution),
        "random_seed": int(random_seed),
        "cluster_sort": (
            "descending_size_then_minimum_global_cell_index_then_raw_id"
        ),
        "spatial_training_graph_reused_for_clustering": False,
        "cross_core_embedding_neighbors_permitted": True,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }


def _layer_directory(output_root: Path, layer: str) -> Path:
    if layer not in NEW_LAYERS:
        raise SO2HiddenLayerProgressionError(f"Unsupported new layer: {layer}")
    return output_root / "clustering" / layer


def _load_joint_layer(
    *, extraction_root: Path, layer: str
) -> tuple[np.ndarray, pd.DataFrame]:
    cores: list[SO2ContextualCore] = []
    arrays: list[np.ndarray] = []
    width: int | None = None
    for core_number, alias in zip(SO2_CORE_NUMBERS, SO2_ALIASES, strict=True):
        path = extraction_root / "embeddings" / f"core_{core_number}_hidden_layers.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                names = set(archive.files)
                required = {
                    "cell_index",
                    "core_number",
                    "coordinates_um",
                    "h0",
                    "h1",
                    "h2",
                    "h3",
                }
                if names != required:
                    raise SO2HiddenLayerProgressionError(
                        f"Unexpected hidden-layer schema for {alias}: {sorted(names)}"
                    )
                cell_index = np.asarray(archive["cell_index"], dtype=np.int64)
                stored_number = np.asarray(archive["core_number"])
                coordinates = np.asarray(
                    archive["coordinates_um"], dtype=np.float64
                )
                values = np.asarray(archive[layer], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise SO2HiddenLayerProgressionError(
                f"Cannot load {layer} for {alias}: {path}"
            ) from exc
        expected = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        if any(
            (
                not np.array_equal(cell_index, np.arange(expected, dtype=np.int64)),
                stored_number.size != 1,
                int(stored_number.reshape(-1)[0]) != core_number,
                coordinates.shape != (expected, 2),
                values.ndim != 2,
                values.shape[0] != expected,
                not np.isfinite(coordinates).all(),
                not np.isfinite(values).all(),
            )
        ):
            raise SO2HiddenLayerProgressionError(
                f"Stored {layer} arrays do not align for {alias}."
            )
        if width is None:
            width = int(values.shape[1])
        elif int(values.shape[1]) != width:
            raise SO2HiddenLayerProgressionError(
                f"Hidden width differs across cores for {layer}."
            )
        values = np.ascontiguousarray(values, dtype=np.float32)
        arrays.append(values)
        cores.append(
            SO2ContextualCore(
                alias=alias,
                core_number=core_number,
                cell_index=np.ascontiguousarray(cell_index),
                coordinates_um=np.ascontiguousarray(coordinates),
                hL=values,
            )
        )
    combined = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    if combined.shape != (EXPECTED_TOTAL_CELLS, int(width or -1)):
        raise SO2HiddenLayerProgressionError(f"Joint {layer} array is incomplete.")
    if not np.any(np.var(combined.astype(np.float64), axis=0) > 0.0):
        raise SO2HiddenLayerProgressionError(f"Joint {layer} has zero variance.")
    return combined, build_cell_index_frame(cores)


def _verify_layer_clustering(
    *,
    output_root: Path,
    layer: str,
    receipt: Mapping[str, Any],
    run_id: str,
    extraction_manifest_sha256: str,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label=f"SO2 {layer} clustering manifest")
    if any(
        (
            receipt.get("schema") != CLUSTERING_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != run_id,
            receipt.get("layer") != layer,
            receipt.get("extraction_manifest_sha256")
            != extraction_manifest_sha256,
            receipt.get("configuration") != dict(configuration),
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HiddenLayerProgressionError(
            f"Stored {layer} clustering identity is invalid."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HiddenLayerProgressionError(
            f"Stored {layer} clustering has no files."
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HiddenLayerProgressionError(
                f"Stored {layer} clustering file is missing: {relative}"
            )
        if _file_record(path) != dict(record):
            raise SO2HiddenLayerProgressionError(
                f"Stored {layer} clustering file changed: {relative}"
            )
    labels = np.load(_layer_directory(output_root, layer) / f"{layer}_labels.npy")
    if labels.shape != (EXPECTED_TOTAL_CELLS,) or _array_sha256(
        "sorted_leiden_labels", labels
    ) != receipt.get("pipeline", {}).get("leiden", {}).get("labels_sha256"):
        raise SO2HiddenLayerProgressionError(f"Stored {layer} labels are invalid.")


def cluster_one_hidden_layer(
    *,
    output_root: Path,
    extraction_root: Path,
    layer: str,
    run_id: str,
    extraction_manifest_sha256: str,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Cluster one state in its own directory, or verify a completed result."""

    layer_dir = _layer_directory(output_root, layer)
    receipt_path = layer_dir / "clustering_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label=f"SO2 {layer} clustering")
        _verify_layer_clustering(
            output_root=output_root,
            layer=layer,
            receipt=receipt,
            run_id=run_id,
            extraction_manifest_sha256=extraction_manifest_sha256,
            configuration=configuration,
        )
        return receipt
    if layer_dir.exists() and any(layer_dir.iterdir()):
        raise SO2HiddenLayerProgressionError(
            f"Partial {layer} clustering exists without a complete receipt."
        )
    layer_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_root / "tables" / layer
    if table_dir.exists() and any(table_dir.iterdir()):
        raise SO2HiddenLayerProgressionError(
            f"Partial {layer} tables exist without a complete receipt."
        )
    table_dir.mkdir(parents=True, exist_ok=True)

    combined, cell_frame = _load_joint_layer(
        extraction_root=extraction_root, layer=layer
    )
    pca = deterministic_pca(
        combined, n_components=int(configuration["pca_components_requested"])
    )
    knn = build_faiss_cosine_knn_graph(
        pca.normalized_scores,
        n_neighbors=int(configuration["n_neighbors"]),
        random_seed=int(configuration["random_seed"]),
    )
    leiden = run_seeded_leiden(
        knn,
        n_cells=EXPECTED_TOTAL_CELLS,
        resolution=float(configuration["leiden_resolution"]),
        random_seed=int(configuration["random_seed"]),
    )
    repeated = run_seeded_leiden(
        knn,
        n_cells=EXPECTED_TOTAL_CELLS,
        resolution=float(configuration["leiden_resolution"]),
        random_seed=int(configuration["random_seed"]),
    )
    labels = np.ascontiguousarray(leiden.labels, dtype=np.int64)
    if not np.array_equal(labels, repeated.labels):
        raise SO2HiddenLayerProgressionError(
            f"Seeded Leiden was not deterministic for {layer}."
        )
    prefix = LAYER_PREFIX[layer]
    cell_frame[f"{layer}_cluster_number"] = labels.astype(np.int32)
    cell_frame[f"{layer}_cluster"] = [f"{prefix}{int(value)}" for value in labels]
    summary, composition, dominated = cluster_summary_tables(
        labels, cell_frame, prefix=prefix
    )
    # Reuse the audited CIELAB generator, then layer-qualify and rotate its
    # colors so equal numeric IDs in separate partitions do not look homologous.
    raw_palette = deterministic_glasbey_palette(
        len(summary), namespace="intrinsic" if layer == "h0" else "contextual"
    )
    raw_colors = list(raw_palette.values())
    rotations = {
        "h0": 0,
        "h1": max(1, len(raw_colors) // 5),
        "h2": max(1, 2 * len(raw_colors) // 5),
        "h3": max(1, 3 * len(raw_colors) // 5),
    }
    shift = rotations[layer] % len(raw_colors)
    palette = {
        f"{prefix}{index}": raw_colors[(index + shift) % len(raw_colors)]
        for index in range(len(raw_colors))
    }

    labels_path = layer_dir / f"{layer}_labels.npy"
    pca_path = layer_dir / f"{layer}_pca_l2_normalized.npy"
    edges_path = layer_dir / f"{layer}_knn_undirected_edges.npy"
    parameters_path = layer_dir / f"{layer}_clustering_parameters.json"
    palette_path = layer_dir / f"{layer}_palette.json"
    summary_path = table_dir / f"{layer}_cluster_summary.csv"
    composition_path = table_dir / f"{layer}_cluster_core_composition.csv"
    _atomic_write_npy(labels_path, labels)
    _atomic_write_npy(pca_path, pca.normalized_scores)
    _atomic_write_npy(edges_path, knn.edge_pairs)
    pipeline = {
        "representation": layer,
        "joint_embedding_shape": list(combined.shape),
        "joint_embedding_sha256": _array_sha256(f"joint_{layer}", combined),
        "pca": pca.receipt,
        "knn": knn.receipt,
        "leiden": leiden.receipt,
        "determinism_replay": {
            "identical_labels": True,
            "labels_sha256": _array_sha256("sorted_leiden_labels", repeated.labels),
        },
    }
    _atomic_write_json(parameters_path, pipeline)
    _atomic_write_json(
        palette_path,
        {
            "layer": layer,
            "label_prefix": prefix,
            "method": "deterministic_greedy_farthest_point_CIELAB",
            "colors": palette,
        },
    )
    _atomic_write_csv(summary_path, summary)
    _atomic_write_csv(composition_path, composition)
    stage_paths = (
        labels_path,
        pca_path,
        edges_path,
        parameters_path,
        palette_path,
        summary_path,
        composition_path,
    )
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": run_id,
            "layer": layer,
            "extraction_manifest_sha256": extraction_manifest_sha256,
            "configuration": dict(configuration),
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "cluster_count": int(len(summary)),
            "cluster_size_range": [
                int(summary["size"].min()),
                int(summary["size"].max()),
            ],
            "core_dominated_gt_90pct": dominated,
            "palette": palette,
            "pipeline": pipeline,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in stage_paths
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_layer_clustering(
        output_root=output_root,
        layer=layer,
        receipt=receipt,
        run_id=run_id,
        extraction_manifest_sha256=extraction_manifest_sha256,
        configuration=configuration,
    )
    del combined, cell_frame, pca, knn, leiden, repeated, labels
    gc.collect()
    return receipt


def _cluster_worker(arguments: Mapping[str, Any]) -> dict[str, Any]:
    receipt = cluster_one_hidden_layer(
        output_root=Path(str(arguments["output_root"])),
        extraction_root=Path(str(arguments["extraction_root"])),
        layer=str(arguments["layer"]),
        run_id=str(arguments["run_id"]),
        extraction_manifest_sha256=str(
            arguments["extraction_manifest_sha256"]
        ),
        configuration=dict(arguments["configuration"]),
    )
    return dict(receipt)


def cluster_hidden_layers(
    *,
    output_root: Path,
    extraction_root: Path,
    run_id: str,
    extraction_manifest_sha256: str,
    configuration: Mapping[str, Any],
    workers: int,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(workers, bool) or not 1 <= int(workers) <= len(NEW_LAYERS):
        raise SO2HiddenLayerProgressionError(
            f"cluster_workers must be in [1, {len(NEW_LAYERS)}]."
        )
    base = {
        "output_root": output_root.as_posix(),
        "extraction_root": extraction_root.as_posix(),
        "run_id": run_id,
        "extraction_manifest_sha256": extraction_manifest_sha256,
        "configuration": dict(configuration),
    }
    if int(workers) == 1:
        return {
            layer: _cluster_worker({**base, "layer": layer})
            for layer in NEW_LAYERS
        }
    results: dict[str, Mapping[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
        futures = {
            pool.submit(_cluster_worker, {**base, "layer": layer}): layer
            for layer in NEW_LAYERS
        }
        for future in as_completed(futures):
            layer = futures[future]
            results[layer] = future.result()
    if set(results) != set(NEW_LAYERS):
        raise SO2HiddenLayerProgressionError(
            "Parallel hidden-layer clustering did not complete every layer."
        )
    return {layer: results[layer] for layer in NEW_LAYERS}


def _load_palette(path: Path, *, prefix: str) -> dict[str, str]:
    payload = _read_json(path, label=f"palette {path.name}")
    colors = payload.get("colors")
    if not isinstance(colors, Mapping) or not colors:
        raise SO2HiddenLayerProgressionError(f"Palette is empty: {path}")
    ordered = sorted(
        ((int(str(label).split("C")[-1]), str(color)) for label, color in colors.items()),
        key=lambda item: item[0],
    )
    if [number for number, _ in ordered] != list(range(len(ordered))):
        raise SO2HiddenLayerProgressionError(f"Palette IDs are not contiguous: {path}")
    return {f"{prefix}{number}": color for number, color in ordered}


def assemble_progression_table(
    *,
    output_root: Path,
    extraction_root: Path,
    hl_output_root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, str]]]:
    """Build a canonical cell-aligned table for all five partitions."""

    _, frame = _load_joint_layer(extraction_root=extraction_root, layer="h0")
    labels_by_layer: dict[str, np.ndarray] = {}
    palettes: dict[str, dict[str, str]] = {}
    for layer in NEW_LAYERS:
        layer_dir = _layer_directory(output_root, layer)
        labels = np.asarray(
            np.load(layer_dir / f"{layer}_labels.npy", allow_pickle=False),
            dtype=np.int64,
        )
        if labels.shape != (EXPECTED_TOTAL_CELLS,):
            raise SO2HiddenLayerProgressionError(
                f"{layer} labels do not align with the canonical cell table."
            )
        labels_by_layer[layer] = np.ascontiguousarray(labels)
        prefix = LAYER_PREFIX[layer]
        frame[f"{layer}_cluster_number"] = labels.astype(np.int32)
        frame[f"{layer}_cluster"] = [f"{prefix}{int(value)}" for value in labels]
        palettes[layer] = _load_palette(
            layer_dir / f"{layer}_palette.json", prefix=prefix
        )

    hl_labels = np.asarray(
        np.load(
            hl_output_root / "clustering" / "contextual_labels.npy",
            allow_pickle=False,
        ),
        dtype=np.int64,
    )
    hl_table = pd.read_parquet(
        hl_output_root / "tables" / "cell_contextual_clusters.parquet"
    )
    alignment_columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
    ]
    if any(
        (
            hl_labels.shape != (EXPECTED_TOTAL_CELLS,),
            len(hl_table) != len(frame),
            not frame[alignment_columns].equals(hl_table[alignment_columns]),
            not np.array_equal(
                hl_labels,
                hl_table["contextual_cluster_number"].to_numpy(dtype=np.int64),
            ),
        )
    ):
        raise SO2HiddenLayerProgressionError(
            "Verified hL clustering does not align cell-for-cell with h0-h3."
        )
    labels_by_layer["hL"] = np.ascontiguousarray(hl_labels)
    frame["hL_cluster_number"] = hl_labels.astype(np.int32)
    frame["hL_cluster"] = [f"HLC{int(value)}" for value in hl_labels]
    palettes["hL"] = _load_palette(
        hl_output_root / "clustering" / "contextual_palette.json", prefix="HLC"
    )
    if tuple(frame["core_number"].drop_duplicates().tolist()) != SO2_CORE_NUMBERS:
        raise SO2HiddenLayerProgressionError(
            "Progression table does not retain the locked core order."
        )
    return frame, labels_by_layer, palettes


def partition_agreement_table(
    labels_by_layer: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Return cell-level ARI/NMI for every ordered layer pair."""

    def pair_metrics(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        _, source_codes = np.unique(source, return_inverse=True)
        _, target_codes = np.unique(target, return_inverse=True)
        counts = np.zeros(
            (int(source_codes.max()) + 1, int(target_codes.max()) + 1),
            dtype=np.int64,
        )
        np.add.at(counts, (source_codes, target_codes), 1)
        values = counts.astype(np.float64, copy=False)

        def combination_two_sum(array: np.ndarray) -> float:
            return float(np.sum(array * (array - 1.0) / 2.0))

        total = int(counts.sum())
        if total < 2:
            ari = 1.0
        else:
            index = combination_two_sum(values)
            rows = combination_two_sum(values.sum(axis=1))
            columns = combination_two_sum(values.sum(axis=0))
            total_pairs = float(total * (total - 1) / 2)
            expected = rows * columns / total_pairs
            maximum = 0.5 * (rows + columns)
            denominator = maximum - expected
            ari = (
                1.0
                if math.isclose(denominator, 0.0, abs_tol=1e-15)
                else float(max(-1.0, min(1.0, (index - expected) / denominator)))
            )

        probabilities = values / float(total)
        row_probabilities = probabilities.sum(axis=1)
        column_probabilities = probabilities.sum(axis=0)
        row_nonzero = row_probabilities > 0.0
        column_nonzero = column_probabilities > 0.0
        row_entropy = -float(
            np.sum(row_probabilities[row_nonzero] * np.log(row_probabilities[row_nonzero]))
        )
        column_entropy = -float(
            np.sum(
                column_probabilities[column_nonzero]
                * np.log(column_probabilities[column_nonzero])
            )
        )
        rows_nonzero, columns_nonzero = np.nonzero(counts)
        joint = probabilities[rows_nonzero, columns_nonzero]
        mutual_information = float(
            np.sum(
                joint
                * np.log(
                    joint
                    / (
                        row_probabilities[rows_nonzero]
                        * column_probabilities[columns_nonzero]
                    )
                )
            )
        )
        nmi_denominator = 0.5 * (row_entropy + column_entropy)
        nmi = (
            1.0
            if math.isclose(nmi_denominator, 0.0, abs_tol=1e-15)
            else float(max(0.0, min(1.0, mutual_information / nmi_denominator)))
        )
        return ari, nmi
    rows: list[dict[str, Any]] = []
    for source_index, source_layer in enumerate(ALL_LAYERS):
        source = np.asarray(labels_by_layer[source_layer], dtype=np.int64)
        if source.shape != (EXPECTED_TOTAL_CELLS,):
            raise SO2HiddenLayerProgressionError(
                f"Labels do not align for {source_layer}."
            )
        for target_index in range(source_index + 1, len(ALL_LAYERS)):
            target_layer = ALL_LAYERS[target_index]
            target = np.asarray(labels_by_layer[target_layer], dtype=np.int64)
            ari, nmi = pair_metrics(source, target)
            rows.append(
                {
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "layer_distance": target_index - source_index,
                    "adjacent": target_index == source_index + 1,
                    "cell_count": len(source),
                    "source_cluster_count": int(len(np.unique(source))),
                    "target_cluster_count": int(len(np.unique(target))),
                    "adjusted_rand_index": ari,
                    "normalized_mutual_information": nmi,
                }
            )
    return pd.DataFrame(rows)


def _render_readme(
    *,
    run_id: str,
    checkpoint_sha256: str,
    layer_receipts: Mapping[str, Mapping[str, Any]],
    hl_cluster_count: int,
    agreement: pd.DataFrame,
    cpu_threads: int,
    cluster_workers: int,
) -> str:
    adjacent_lines = []
    for row in agreement.loc[agreement["adjacent"]].itertuples(index=False):
        adjacent_lines.append(
            f"- `{row.source_layer} -> {row.target_layer}`: ARI "
            f"{row.adjusted_rand_index:.4f}; NMI "
            f"{row.normalized_mutual_information:.4f}"
        )
    counts = {layer: int(layer_receipts[layer]["cluster_count"]) for layer in NEW_LAYERS}
    counts["hL"] = int(hl_cluster_count)
    counts_text = ", ".join(f"{layer}={counts[layer]}" for layer in ALL_LAYERS)
    return f"""# SO2 four-block hidden-layer cluster progression

This CPU-only exploratory report uses the completed locked run `{run_id}` and
checkpoint SHA-256 `{checkpoint_sha256}`. It extracts the actual node-encoder
state `h0` and complete post-block states `h1`, `h2`, and `h3`. Captured block-4
output was required to match the existing verified `hL` artifact within the
locked float32 replay tolerance for every core, while matching the same-process
public final state exactly; the existing hL partition was then reused.

## Method

All {EXPECTED_TOTAL_CELLS:,} cells from SO2 cores 15--28 were clustered jointly
and independently at every layer. Each pipeline mean-centered the 256-D state,
performed exact PCA50, L2-normalized the scores, built a sparse cosine FAISS
HNSW 30-nearest-neighbor union graph, and ran seeded Leiden at resolution 1.0
with seed 20260825. Cluster counts are {counts_text}.

The cluster IDs and colors are layer-specific. For example, `H1C0` and `H2C0`
are not asserted to be the same group. Cell-aligned contingency heatmaps and
partition metrics describe splits and merges without treating IDs as a lineage.

## Adjacent-layer agreement

{chr(10).join(adjacent_lines)}

## Figures

The `figures/` directory contains static PNG spatial maps for h0, h1, h2, and
h3 plus a two-row split/merge transition heatmap for h0 -> h1 -> h2 -> h3 ->
hL. The previously verified hL spatial PNG is referenced as an immutable source
artifact rather than copied or modified. No PDF, HTML, or interactive map was
created by this workflow.

## Interpretation limits

These partitions describe this fitted model's representation geometry. Spatial
coherence, cluster continuity, or cluster splitting does not establish a cell
type, biological mechanism, predictive dependency, patient replication, or
causal influence. This fit-only SO2 model does not support a generalization
claim, and no cell-type or disease labels were used to construct the clusters.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-hidden-layer-progression \
  --run-id {run_id} \
  --n-neighbors 30 --leiden-resolution 1.0 --pca-components 50 \
  --random-seed 20260825 --device cpu --cpu-threads {cpu_threads} \
  --cluster-workers {cluster_workers} --dpi 300
```

Extraction and each layer's clustering are independently resumable from
checksum-verified receipts.
"""


def _verify_final_manifest(output_root: Path, manifest: Mapping[str, Any]) -> None:
    _verify_self_hash(manifest, label="SO2 hidden-layer progression manifest")
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            tuple(manifest.get("layers", ())) != ALL_LAYERS,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HiddenLayerProgressionError(
            "Final hidden-layer progression manifest identity is invalid."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HiddenLayerProgressionError("Final manifest contains no files.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HiddenLayerProgressionError(
                f"Final progression output is missing: {relative}"
            )
        if _file_record(path) != dict(record):
            raise SO2HiddenLayerProgressionError(
                f"Final progression output changed: {relative}"
            )
    required = {
        "README.md",
        "embeddings/extraction_manifest.json",
        "tables/cell_hidden_layer_clusters.parquet",
        "tables/partition_agreement.csv",
        "figures/hidden_layer_cluster_transition_heatmaps_h0_to_hL.png",
    }
    required.update(
        f"embeddings/core_{number}_hidden_layers.npz"
        for number in SO2_CORE_NUMBERS
    )
    required.update(
        f"clustering/{layer}/clustering_manifest.json" for layer in NEW_LAYERS
    )
    required.update(
        f"figures/contextual_{layer}_leiden_resolution_1p0_spatial_14cores.png"
        for layer in REQUESTED_MAP_LAYERS
    )
    if not required.issubset(files):
        raise SO2HiddenLayerProgressionError(
            f"Required progression outputs are absent: {sorted(required - set(files))}"
        )
    source_files = manifest.get("analysis_source_files")
    if not isinstance(source_files, Mapping) or len(source_files) != 5:
        raise SO2HiddenLayerProgressionError(
            "Final manifest lacks exact analysis-source snapshots."
        )
    for source_relative, source_record in source_files.items():
        if not isinstance(source_record, Mapping):
            raise SO2HiddenLayerProgressionError(
                f"Malformed source record: {source_relative}"
            )
        snapshot_relative = str(source_record.get("snapshot_path", ""))
        snapshot_record = source_record.get("snapshot_file")
        working_record = source_record.get("working_tree_file")
        if (
            not snapshot_relative.startswith("provenance/source/")
            or not isinstance(snapshot_record, Mapping)
            or not isinstance(working_record, Mapping)
        ):
            raise SO2HiddenLayerProgressionError(
                f"Analysis-source snapshot is invalid: {source_relative}"
            )
        if (
            dict(snapshot_record) != dict(working_record)
            or files.get(snapshot_relative) != snapshot_record
        ):
            raise SO2HiddenLayerProgressionError(
                f"Analysis-source snapshot is invalid: {source_relative}"
            )
    source_hl = manifest.get("source_hL")
    if not isinstance(source_hl, Mapping):
        raise SO2HiddenLayerProgressionError("Final manifest lacks hL provenance.")
    source_manifest = Path(str(source_hl.get("analysis_manifest")))
    if not source_manifest.is_file() or sha256_file(source_manifest) != source_hl.get(
        "analysis_manifest_sha256"
    ):
        raise SO2HiddenLayerProgressionError("Referenced hL manifest changed.")


def _write_tables_and_figures(
    *,
    output_root: Path,
    extraction_root: Path,
    hl_output_root: Path,
    resolution: float,
    dpi: int,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, dict[str, str]],
    Mapping[str, Any],
]:
    from .so2_hidden_layer_progression_figures import (
        compute_adjacent_partition_transitions,
        render_layer_spatial_png,
        render_transition_heatmap_png,
    )

    frame, labels_by_layer, palettes = assemble_progression_table(
        output_root=output_root,
        extraction_root=extraction_root,
        hl_output_root=hl_output_root,
    )
    agreement = partition_agreement_table(labels_by_layer)
    table_path = output_root / "tables" / "cell_hidden_layer_clusters.parquet"
    agreement_path = output_root / "tables" / "partition_agreement.csv"
    _atomic_write_parquet(table_path, frame)
    _atomic_write_csv(agreement_path, agreement)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    spatial_records: dict[str, Any] = {}
    for layer in REQUESTED_MAP_LAYERS:
        png_path = (
            figure_dir
            / f"contextual_{layer}_leiden_resolution_1p0_spatial_14cores.png"
        )
        summary = render_layer_spatial_png(
            frame,
            labels_by_layer[layer],
            palettes[layer],
            png_path,
            layer=layer,
            resolution=resolution,
            dpi=dpi,
            expected_counts_by_core=EXPECTED_CELL_COUNTS_BY_CORE,
        )
        if summary.point_count != EXPECTED_TOTAL_CELLS:
            raise SO2HiddenLayerProgressionError(
                f"Spatial {layer} PNG omitted cells."
            )
        spatial_records[layer] = {
            "path": png_path.relative_to(output_root).as_posix(),
            "file": _file_record(png_path),
            "point_count": int(summary.point_count),
            "cluster_labels": list(summary.cluster_labels),
            "dpi": int(summary.dpi),
        }

    transition_path = (
        figure_dir / "hidden_layer_cluster_transition_heatmaps_h0_to_hL.png"
    )
    transition_summary = render_transition_heatmap_png(
        labels_by_layer,
        transition_path,
        dpi=dpi,
    )
    adjacent = compute_adjacent_partition_transitions(labels_by_layer)
    if transition_summary.point_count != EXPECTED_TOTAL_CELLS:
        raise SO2HiddenLayerProgressionError(
            "Transition heatmap omitted cell-aligned assignments."
        )
    adjacent_records: list[dict[str, Any]] = []
    for transition in adjacent:
        row = agreement.loc[
            (agreement["source_layer"] == transition.source_layer)
            & (agreement["target_layer"] == transition.target_layer)
        ]
        if len(row) != 1 or not (
            math.isclose(
                float(row.iloc[0]["adjusted_rand_index"]),
                transition.adjusted_rand_index,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(row.iloc[0]["normalized_mutual_information"]),
                transition.normalized_mutual_information,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise SO2HiddenLayerProgressionError(
                "Figure and table partition metrics disagree."
            )
        adjacent_records.append(
            {
                "source_layer": transition.source_layer,
                "target_layer": transition.target_layer,
                "source_cluster_count": len(transition.source_labels),
                "target_cluster_count": len(transition.target_labels),
                "adjusted_rand_index": transition.adjusted_rand_index,
                "normalized_mutual_information": (
                    transition.normalized_mutual_information
                ),
                "contingency_counts_sha256": _array_sha256(
                    f"{transition.source_layer}_to_{transition.target_layer}_counts",
                    transition.counts,
                ),
                "split_fractions_sha256": _array_sha256(
                    f"{transition.source_layer}_to_{transition.target_layer}_splits",
                    transition.split_fractions,
                ),
                "merge_fractions_sha256": _array_sha256(
                    f"{transition.source_layer}_to_{transition.target_layer}_merges",
                    transition.merge_fractions,
                ),
            }
        )
    receipt = _receipt_with_self_hash(
        {
            "schema": "so2_hidden_layer_progression_figures_v1",
            "status": "complete",
            "created_at": _utc_now(),
            "png_only": True,
            "pdf_created": False,
            "interactive_created": False,
            "spatial_maps": spatial_records,
            "transition_heatmap": {
                "path": transition_path.relative_to(output_root).as_posix(),
                "file": _file_record(transition_path),
                "point_count": int(transition_summary.point_count),
                "layer_order": list(transition_summary.layer_order),
                "adjacent_transitions": adjacent_records,
            },
            "table_files": {
                table_path.relative_to(output_root).as_posix(): _file_record(table_path),
                agreement_path.relative_to(output_root).as_posix(): _file_record(
                    agreement_path
                ),
            },
        }
    )
    _atomic_write_json(figure_dir / "figure_manifest.json", receipt)
    return agreement, labels_by_layer, palettes, receipt


def _finalize_analysis(
    *,
    inputs: Any,
    output_root: Path,
    extraction: Mapping[str, Any],
    layer_receipts: Mapping[str, Mapping[str, Any]],
    hl_output_root: Path,
    hl_result: Mapping[str, Any],
    agreement: pd.DataFrame,
    figures: Mapping[str, Any],
    configuration: Mapping[str, Any],
    cpu_threads: int,
    cluster_workers: int,
    dpi: int,
) -> Mapping[str, Any]:
    readme_path = output_root / "README.md"
    _atomic_write_text(
        readme_path,
        _render_readme(
            run_id=inputs.run_id,
            checkpoint_sha256=inputs.checkpoint_sha256,
            layer_receipts=layer_receipts,
            hl_cluster_count=int(hl_result["cluster_count"]),
            agreement=agreement,
            cpu_threads=cpu_threads,
            cluster_workers=cluster_workers,
        ),
    )
    h4_records = [
        dict(record["reference_hL"])
        for record in extraction["cores"]
        if isinstance(record, Mapping)
    ]
    maximum_h4_difference = max(
        float(record["maximum_absolute_difference"]) for record in h4_records
    )
    bitwise_equal_cores = sum(
        bool(record["captured_h4_bitwise_equal"]) for record in h4_records
    )
    hl_manifest_path = hl_output_root / "manifest.json"
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_scope": "exploratory_hidden_state_progression",
            "run_id": inputs.run_id,
            "checkpoint": {
                "path": inputs.checkpoint_path.as_posix(),
                "sha256": inputs.checkpoint_sha256,
                "model_state_sha256": inputs.checkpoint_payload.get(
                    "model_state_checksum"
                ),
            },
            "layers": list(ALL_LAYERS),
            "stored_new_layers": list(NEW_LAYERS),
            "hL_reused_from_verified_report": True,
            "core_order": list(SO2_CORE_NUMBERS),
            "core_cell_counts": {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO2_CORE_NUMBERS
            },
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_dimension": int(extraction["embedding_dimension"]),
            "analysis_parameters": {
                **dict(configuration),
                "cpu_threads": int(cpu_threads),
                "cluster_workers": int(cluster_workers),
                "figure_dpi": int(dpi),
            },
            "cpu_only_execution": {
                "enforced_device": "cpu",
                "reason": (
                    "all four RTX 3090 devices were occupied by the active "
                    "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf training run"
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_threads": int(cpu_threads),
                "parallel_clustering_workers": int(cluster_workers),
            },
            "h4_hL_replay_control": {
                "all_cores_within_tolerance": True,
                "absolute_tolerance": 2e-6,
                "relative_tolerance": 1e-6,
                "maximum_absolute_difference_across_cores": maximum_h4_difference,
                "bitwise_equal_core_count": int(bitwise_equal_cores),
                "core_count": len(h4_records),
                "same_process_h4_equals_full_node_embedding_exact": True,
            },
            "source_hL": {
                "output_root": hl_output_root.as_posix(),
                "analysis_manifest": hl_manifest_path.as_posix(),
                "analysis_manifest_sha256": sha256_file(hl_manifest_path),
                "extraction_manifest_sha256": sha256_file(
                    hl_output_root / "embeddings" / "extraction_manifest.json"
                ),
                "clustering_manifest_sha256": sha256_file(
                    hl_output_root / "clustering" / "clustering_manifest.json"
                ),
                "labels": {
                    "path": (
                        hl_output_root / "clustering" / "contextual_labels.npy"
                    ).as_posix(),
                    "sha256": sha256_file(
                        hl_output_root / "clustering" / "contextual_labels.npy"
                    ),
                },
                "cluster_count": int(hl_result["cluster_count"]),
            },
            "stage_manifests": {
                "extraction": _file_record(
                    output_root / "embeddings" / "extraction_manifest.json"
                ),
                "clustering": {
                    layer: _file_record(
                        _layer_directory(output_root, layer)
                        / "clustering_manifest.json"
                    )
                    for layer in NEW_LAYERS
                },
                "figures": _file_record(
                    output_root / "figures" / "figure_manifest.json"
                ),
            },
            "cluster_counts": {
                **{
                    layer: int(layer_receipts[layer]["cluster_count"])
                    for layer in NEW_LAYERS
                },
                "hL": int(hl_result["cluster_count"]),
            },
            "adjacent_partition_agreement": agreement.loc[
                agreement["adjacent"]
            ].to_dict(orient="records"),
            "figure_paths": sorted(
                [
                    value["path"]
                    for value in figures["spatial_maps"].values()
                ]
                + [figures["transition_heatmap"]["path"]]
            ),
            "analysis_source": _git_provenance(inputs.project_root),
            "analysis_source_files": _source_code_records(
                inputs.project_root, output_root
            ),
            "interpretation": {
                "clusters_are_layer_specific_model_derived_groups": True,
                "numeric_ids_are_cross_layer_identities": False,
                "cell_types_established": False,
                "predictive_dependencies_established": False,
                "biological_mechanism_established": False,
                "causality_established": False,
                "generalization_claim_supported": False,
            },
            "files": _file_manifest(output_root),
        }
    )
    manifest_path = output_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    _verify_final_manifest(output_root, manifest)
    return manifest


def _completed_result(output_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "run_id": str(manifest["run_id"]),
        "device": "cpu",
        "output_root": output_root.as_posix(),
        "total_cells": int(manifest["total_cells"]),
        "layers": list(manifest["layers"]),
        "cluster_counts": dict(manifest["cluster_counts"]),
        "h4_hL_replay_control": dict(manifest["h4_hL_replay_control"]),
        "pngs": [
            (output_root / str(relative)).as_posix()
            for relative in manifest["figure_paths"]
        ],
        "source_hL_png": (
            Path(str(manifest["source_hL"]["output_root"]))
            / "figures"
            / "contextual_leiden_resolution_1p0_spatial_14cores.png"
        ).as_posix(),
        "manifest": (output_root / "manifest.json").as_posix(),
    }


def run_so2_hidden_layer_progression(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str | torch.device = "cpu",
    cpu_threads: int = 40,
    cluster_workers: int = 4,
    dpi: int = 300,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the locked, resumable h0--hL progression analysis."""

    validate_cpu_device(device)
    if any(
        (
            int(n_neighbors) != DEFAULT_N_NEIGHBORS,
            float(leiden_resolution) != DEFAULT_LEIDEN_RESOLUTION,
            int(pca_components) != DEFAULT_PCA_COMPONENTS,
            int(random_seed) != DEFAULT_RANDOM_SEED,
        )
    ):
        raise SO2HiddenLayerProgressionError(
            "Progression must match the verified hL PCA50/cosine-k30/Leiden-1.0 "
            "pipeline with seed 20260825."
        )
    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2HiddenLayerProgressionError("Figure DPI must be at least 72.")
    configuration = clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    inputs = resolve_so2_analysis_inputs(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    hl_result = run_so2_hl_clustering(
        registry=registry,
        paths=paths,
        run_id=inputs.run_id,
        checkpoint=inputs.checkpoint_path,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
        device="cpu",
        cpu_threads=cpu_threads,
    )
    hl_output_root = Path(str(hl_result["output_root"])).resolve(strict=True)

    publish_root: Path | None = None
    if output_dir is None:
        publish_root = (
            paths.report_root / "analyses" / ANALYSIS_FAMILY / inputs.run_id
        ).resolve(strict=False)
        output_root = (
            paths.scratch_root
            / "active_runs"
            / inputs.run_id
            / "posthoc_reports"
            / ANALYSIS_FAMILY
        ).resolve(strict=False)
        if publish_root.joinpath("manifest.json").is_file():
            manifest = _read_json(
                publish_root / "manifest.json",
                label="SO2 hidden-layer progression manifest",
            )
            _verify_final_manifest(publish_root, manifest)
            return _completed_result(publish_root, manifest)
        if publish_root.exists():
            raise SO2HiddenLayerProgressionError(
                f"Unreceipted published progression output exists: {publish_root}"
            )
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(
            manifest_path, label="SO2 hidden-layer progression manifest"
        )
        _verify_final_manifest(output_root, manifest)
    else:
        from .so2_hidden_layer_progression_extraction import (
            extract_hidden_layer_progression,
        )

        extraction = extract_hidden_layer_progression(
            inputs=inputs,
            output_root=output_root,
            verified_hl_output_root=hl_output_root,
            device=device,
            cpu_threads=cpu_threads,
        )
        extraction_sha = sha256_file(
            output_root / "embeddings" / "extraction_manifest.json"
        )
        layer_receipts = cluster_hidden_layers(
            output_root=output_root,
            extraction_root=output_root,
            run_id=inputs.run_id,
            extraction_manifest_sha256=extraction_sha,
            configuration=configuration,
            workers=cluster_workers,
        )
        agreement, _, _, figures = _write_tables_and_figures(
            output_root=output_root,
            extraction_root=output_root,
            hl_output_root=hl_output_root,
            resolution=leiden_resolution,
            dpi=int(dpi),
        )
        manifest = _finalize_analysis(
            inputs=inputs,
            output_root=output_root,
            extraction=extraction,
            layer_receipts=layer_receipts,
            hl_output_root=hl_output_root,
            hl_result=hl_result,
            agreement=agreement,
            figures=figures,
            configuration=configuration,
            cpu_threads=int(cpu_threads),
            cluster_workers=int(cluster_workers),
            dpi=int(dpi),
        )

    if publish_root is not None:
        publish_root.parent.mkdir(parents=True, exist_ok=True)
        _atomic_publish_directory_no_replace(output_root, publish_root)
        output_root = publish_root
        manifest = _read_json(
            output_root / "manifest.json",
            label="published SO2 hidden-layer progression manifest",
        )
        _verify_final_manifest(output_root, manifest)
    return _completed_result(output_root, manifest)


__all__ = [
    "ALL_LAYERS",
    "NEW_LAYERS",
    "SO2HiddenLayerProgressionError",
    "assemble_progression_table",
    "cluster_hidden_layers",
    "cluster_one_hidden_layer",
    "clustering_configuration",
    "partition_agreement_table",
    "run_so2_hidden_layer_progression",
]

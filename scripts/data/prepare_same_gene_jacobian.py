#!/usr/bin/env python3
"""Prepare immutable local arrays for the same-gene Jacobian campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from spatial_benchmark.data import (
    CoreSelection,
    load_selected_core,
    normalize_slide,
    resolve_nested_raw_path,
)
from spatial_benchmark.fingerprints import (
    build_path_fingerprint,
    fingerprint_split,
    sha256_file,
)
from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.same_gene_jacobian import (
    COMPONENT_THRESHOLD_MM,
    FOLD_COMPONENT_ORDINALS,
    build_neighbor_aggregates,
    geometry_components,
    planted_recovery_control,
)


CAMPAIGN_ID = "cmp_20260810_same_gene_cross_cell_jacobian"
DATASET_ID = "gastric_cosmx_drive_public"
DATASET_VERSION = "drive_snapshot_20260810"
SPLIT_ID = "opaque_geometry_components_075mm_4fold_v1"
PREPROCESSING_VERSION = "same_gene_neighbor_log1p_v1"
SLIDES = ("SO_1", "SO_2")
SLIDE_CODE = {"SO_1": 1, "SO_2": 2}
ARRAY_NAMES = (
    "expression_log1p",
    "metadata",
    "coordinates_um",
    "fov",
    "geometry_group",
    "fold",
    "qc_passed",
    "neighbor_near_mean",
    "neighbor_annular_mean",
    "neighbor_permuted_near_mean",
    "near_degree",
    "annular_degree",
    "permuted_near_degree",
    "matched_eligible",
)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positions_path(raw_root: Path, slide: str) -> Path:
    candidates: list[Path] = []
    for entry in raw_root.iterdir():
        if not entry.name.endswith("_fov_positions_file.csv"):
            continue
        try:
            if normalize_slide(entry.name) == slide:
                candidates.append(resolve_nested_raw_path(raw_root, entry.name))
        except ValueError:
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one FOV-position file for {slide}; found {len(candidates)}")
    return candidates[0]


def _save_array(directory: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    np.save(path, np.asarray(value), allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prepare_slide(
    raw_root: Path,
    output_root: Path,
    slide: str,
    expected_genes: tuple[str, ...] | None,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    positions_path = _positions_path(raw_root, slide)
    positions = pd.read_csv(
        positions_path,
        usecols=["FOV", "x_global_mm", "y_global_mm"],
        dtype={"FOV": "int32", "x_global_mm": "float64", "y_global_mm": "float64"},
    ).sort_values("FOV", kind="mergesort")
    if positions["FOV"].duplicated().any():
        raise RuntimeError(f"{slide} FOV positions contain duplicate identifiers")
    components = geometry_components(
        slide,
        positions["FOV"].to_numpy(),
        positions[["x_global_mm", "y_global_mm"]].to_numpy(),
    )
    selection = CoreSelection(slide=slide, fovs=tuple(int(v) for v in positions["FOV"]))
    dataset = load_selected_core(
        raw_root,
        selection,
        chunksize=32768,
        expected_biological_probes=1000,
        qc_policy="all",
    )
    if expected_genes is not None and dataset.gene_names != expected_genes:
        raise RuntimeError("The ordered biological panel differs between slides")

    fov_to_ordinal = {
        int(fov): int(ordinal)
        for fov, ordinal in zip(components.fovs, components.component_ordinals, strict=True)
    }
    fov_to_fold = {
        int(fov): int(fold)
        for fov, fold in zip(components.fovs, components.folds, strict=True)
    }
    cell_fov = dataset.keys["fov"].to_numpy(dtype=np.int32, copy=True)
    if set(int(v) for v in np.unique(cell_fov)) != set(fov_to_ordinal):
        raise RuntimeError(f"{slide} expression/metadata FOV coverage differs from positions")
    group = np.asarray(
        [SLIDE_CODE[slide] * 100 + fov_to_ordinal[int(value)] for value in cell_fov],
        dtype=np.int16,
    )
    fold = np.asarray([fov_to_fold[int(value)] for value in cell_fov], dtype=np.int8)
    expression_log1p = np.log1p(dataset.expression.astype(np.float32, copy=False))
    neighbor = build_neighbor_aggregates(
        expression_log1p,
        dataset.coordinates_um,
        cell_fov,
        slide=slide,
    )
    if float(neighbor.audit["matched_eligible_fraction"]) < 0.80:
        raise RuntimeError(f"{slide} matched neighbor coverage is below the frozen 80% gate")

    slide_root = output_root / slide
    slide_root.mkdir()
    arrays = {
        "expression_log1p": expression_log1p.astype(np.float32, copy=False),
        "metadata": dataset.metadata.astype(np.float32, copy=False),
        "coordinates_um": dataset.coordinates_um.astype(np.float32),
        "fov": cell_fov,
        "geometry_group": group,
        "fold": fold,
        "qc_passed": dataset.qc_passed.astype(bool, copy=False),
        "neighbor_near_mean": neighbor.near_mean,
        "neighbor_annular_mean": neighbor.annular_mean,
        "neighbor_permuted_near_mean": neighbor.permuted_near_mean,
        "near_degree": neighbor.near_degree,
        "annular_degree": neighbor.annular_degree,
        "permuted_near_degree": neighbor.permuted_near_degree,
        "matched_eligible": neighbor.matched_eligible,
    }
    inventory = {name: _save_array(slide_root, name, arrays[name]) for name in ARRAY_NAMES}
    component_records: list[dict[str, int | str]] = []
    for fov, ordinal, assigned_fold in zip(
        components.fovs, components.component_ordinals, components.folds, strict=True
    ):
        component_records.append(
            {
                "slide": slide,
                "fov": int(fov),
                "component_ordinal": int(ordinal),
                "geometry_group": SLIDE_CODE[slide] * 100 + int(ordinal),
                "fold": int(assigned_fold),
            }
        )
    _json_write(slide_root / "fov_component_map.json", component_records)
    inventory["fov_component_map"] = {
        "path": "fov_component_map.json",
        "size_bytes": (slide_root / "fov_component_map.json").stat().st_size,
        "sha256": sha256_file(slide_root / "fov_component_map.json"),
    }

    fold_rows = []
    for assigned_fold in range(4):
        mask = fold == assigned_fold
        fold_rows.append(
            {
                "fold": assigned_fold,
                "cells": int(np.sum(mask)),
                "fovs": int(len(np.unique(cell_fov[mask]))),
                "components": int(len(np.unique(group[mask]))),
                "qc_passed_cells": int(np.sum(dataset.qc_passed[mask])),
                "matched_eligible_cells": int(np.sum(neighbor.matched_eligible[mask])),
            }
        )
    summary = {
        "slide": slide,
        "cell_count": dataset.n_cells,
        "gene_count": dataset.n_genes,
        "fov_count": len(positions),
        "component_count": components.component_count,
        "minimum_between_component_fov_origin_distance_mm": (
            components.minimum_between_component_distance_mm
        ),
        "folds": fold_rows,
        "neighbor_audit": neighbor.audit,
        "arrays": inventory,
    }
    return summary, dataset.gene_names, dataset.metadata_names


def _verify(output_root: Path, *, verify_raw: bool = False) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for slide in SLIDES:
        for name, item in manifest["slides"][slide]["arrays"].items():
            path = output_root / slide / item["path"]
            if not path.is_file():
                errors.append(f"missing:{slide}/{item['path']}")
                continue
            observed = sha256_file(path)
            if observed != item["sha256"]:
                errors.append(f"sha256:{slide}/{item['path']}")
            if name in ARRAY_NAMES:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
                    errors.append(f"array_contract:{slide}/{item['path']}")
    if verify_raw:
        raw_root = current_paths().data_root / "raw" / "Gastric_Cancer_Analysis"
        observed_raw = build_path_fingerprint(raw_root).sha256
        if observed_raw != manifest["raw_snapshot"]["fingerprint"]:
            errors.append("raw_snapshot_fingerprint")
    processed_payload = {
        slide: manifest["slides"][slide]["arrays"] for slide in SLIDES
    }
    observed_processed = canonical_sha256(processed_payload)
    if observed_processed != manifest["processed_fingerprint"]:
        errors.append("processed_fingerprint")
    result = {
        "verified": not errors,
        "errors": errors,
        "manifest": str(manifest_path),
        "processed_fingerprint": observed_processed,
    }
    if errors:
        raise RuntimeError(f"Processed-data verification failed: {errors}")
    return result


def _register(output_root: Path, manifest: dict[str, Any]) -> None:
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.initialize()
    registry.create_campaign(
        CAMPAIGN_ID,
        name="Same-gene cross-cell Jacobian pilot",
        scientific_question=(
            "Is a whole-node-masked receiver selectively sensitive to nearby "
            "cells carrying the same named RNA probe?"
        ),
        config={
            "definition": (
                "experiments/campaigns/"
                "cmp_20260810_same_gene_cross_cell_jacobian/README.md"
            ),
            "exploratory": True,
        },
        status="pilot",
    )
    total_cells = sum(manifest["slides"][slide]["cell_count"] for slide in SLIDES)
    total_fovs = sum(manifest["slides"][slide]["fov_count"] for slide in SLIDES)
    registry.register_dataset(
        DATASET_ID,
        DATASET_VERSION,
        display_name="Public Drive CosMx gastric snapshot without clinical mapping",
        protected_source_path=output_root.parent.parent / "raw" / "Gastric_Cancer_Analysis",
        raw_fingerprint=manifest["raw_snapshot"]["fingerprint"],
        preprocessing_version=PREPROCESSING_VERSION,
        processed_fingerprint=manifest["processed_fingerprint"],
        aggregate_sample_count=total_cells,
        graph_count=total_fovs,
        node_feature_schema="1000 log1p probes + fixed 22 morphology variables",
        edge_feature_schema="within-FOV 0-25um and 25-50um capped neighbor means",
        creation_date="2026-08-10",
        status="available",
        verification_status="verified",
        metadata={
            "clinical_mapping_available": False,
            "geometry_component_count": 27,
            "fold_count": 4,
        },
    )
    registry.register_split(
        SPLIT_ID,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        method="same-slide FOV-origin connected components at 0.75mm",
        unit="opaque_geometry_component",
        seed=20260810,
        fold_count=4,
        stratification={"fields": ["slide", "cell_count", "fov_count"]},
        fingerprint=manifest["split_fingerprint"],
        protected_path=output_root / "manifest.json",
        verification_status="verified",
        metadata={"patient_held_out": False, "clinical_labels_used": False},
    )


def prepare(raw_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Prepared output already exists; use --verify-only: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    control = planted_recovery_control()
    if control["passed"] is not True:
        raise RuntimeError(f"Analytical planted recovery failed: {control}")

    raw_result = build_path_fingerprint(raw_root)
    slides: dict[str, Any] = {}
    genes: tuple[str, ...] | None = None
    metadata_names: tuple[str, ...] | None = None
    for slide in SLIDES:
        summary, observed_genes, observed_metadata_names = _prepare_slide(
            raw_root, temporary, slide, genes
        )
        slides[slide] = summary
        genes = observed_genes
        if metadata_names is None:
            metadata_names = observed_metadata_names
        elif metadata_names != observed_metadata_names:
            raise RuntimeError("Metadata allow-list order differs between slides")

    assert genes is not None and metadata_names is not None
    _json_write(temporary / "genes.json", list(genes))
    _json_write(temporary / "metadata_names.json", list(metadata_names))
    component_records = []
    for slide in SLIDES:
        component_records.extend(
            json.loads(
                (temporary / slide / "fov_component_map.json").read_text(encoding="utf-8")
            )
        )
    logical_split_records = [
        {
            "sample_key": f"{row['slide']}:G{int(row['component_ordinal']):02d}",
            "split": f"fold_{int(row['fold'])}",
        }
        for row in component_records
    ]
    split_fingerprint = fingerprint_split(
        logical_split_records,
        split_id=SPLIT_ID,
        method="opaque_fov_origin_components_075mm",
    )
    processed_payload = {
        slide: slides[slide]["arrays"] for slide in SLIDES
    }
    processed_fingerprint = canonical_sha256(processed_payload)
    manifest = {
        "manifest_schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "split_id": SPLIT_ID,
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw_snapshot": raw_result.summary(),
        "processed_fingerprint": processed_fingerprint,
        "split_fingerprint": split_fingerprint,
        "gene_count": len(genes),
        "gene_order_sha256": canonical_sha256(list(genes)),
        "metadata_names": list(metadata_names),
        "geometry_component_threshold_mm": COMPONENT_THRESHOLD_MM,
        "fold_component_ordinals": {
            str(fold): {slide: list(values) for slide, values in mapping.items()}
            for fold, mapping in FOLD_COMPONENT_ORDINALS.items()
        },
        "analytical_control": control,
        "slides": slides,
        "clinical_mapping_available": False,
        "maximum_claim": (
            "stable held-out-geometry graph-alignment-dependent same-gene "
            "model sensitivity"
        ),
    }
    _json_write(temporary / "manifest.json", manifest)
    temporary.rename(output_root)
    verification = _verify(output_root)
    _json_write(output_root / "verification.json", verification)
    _register(output_root, manifest)
    return {
        "output_root": str(output_root),
        "raw_fingerprint": raw_result.sha256,
        "processed_fingerprint": processed_fingerprint,
        "split_fingerprint": split_fingerprint,
        "cells": sum(slides[slide]["cell_count"] for slide in SLIDES),
        "genes": len(genes),
        "matched_eligible_fraction": (
            sum(
                slides[slide]["neighbor_audit"]["matched_eligible_count"]
                for slide in SLIDES
            )
            / sum(slides[slide]["cell_count"] for slide in SLIDES)
        ),
        "verified": verification["verified"],
    }


def parse_args() -> argparse.Namespace:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=paths.data_root / "raw" / "Gastric_Cancer_Analysis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.data_root / "processed" / "same_gene_cross_cell_jacobian_v1",
    )
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = (
        _verify(arguments.output.resolve(), verify_raw=arguments.verify_raw)
        if arguments.verify_only
        else prepare(arguments.raw_root.resolve(strict=True), arguments.output.resolve())
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()

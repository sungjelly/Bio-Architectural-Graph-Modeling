"""Read and verify the immutable SO1/SO2 inputs for preprocessing diagnostics.

No files are written. Protected donor values are used only to verify equality
groups and never returned. The returned cell metadata remains local restricted
analysis data; callers should publish aggregate diagnostics rather than rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VENDOR_LABEL = "RNA_Gastric_testrun_Cell.Typing.InSituType.1_1_clusters"
VENDOR_POSTERIOR = "RNA_Gastric_testrun_Cell.Typing.InSituType.1_1_posterior_probability"
PIXEL_SIZE_UM = 0.120281
EXPECTED_TOTAL_CELLS = 407_659
EXPECTED_GROUPING_SHA256 = (
    "30a1c3de1fee0d0045ac6bbfa849f9bb47de437c1b11428e6e16ac48d06fd5e0"
)
TRAIN_CORES = frozenset((*range(1, 11), *range(15, 25)))


class InputContractError(ValueError):
    """An input failed an identifier-free integrity or alignment check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_record(path: Path, data_root: Path, expected: str | None = None) -> dict:
    observed = _sha256(path)
    _require(expected is None or observed == expected,
             f"Input checksum mismatch: {path.name}")
    return {"path": path.relative_to(data_root).as_posix(),
            "sha256": observed, "size_bytes": path.stat().st_size,
            "verified_against_upstream": expected is not None}


def _verify_donor_groups(path: Path) -> dict:
    """Inspect only core numbers and protected equality values, without logging."""
    try:
        frame = pd.read_excel(path, header=None, usecols=[0, 1])
        numbers = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        selected = frame.loc[numbers.isin(range(1, 29))].copy()
        selected_numbers = pd.to_numeric(selected.iloc[:, 0]).astype(int)
        _require(len(selected) == 28 and not selected_numbers.duplicated().any()
                 and set(selected_numbers) == set(range(1, 29))
                 and not selected.iloc[:, 1].isna().any(),
                 "Protected grouping source lacks unique complete core coverage.")
        groups: dict[Any, list[int]] = {}
        for number, protected_value in zip(selected_numbers, selected.iloc[:, 1]):
            groups.setdefault(protected_value, []).append(int(number))
        observed = sorted(sorted(members) for members in groups.values())
        expected = [[number, number + 1] for number in range(1, 29, 2)]
        _require(observed == expected,
                 "Protected donor equality classes differ from the locked pairing.")
    except InputContractError:
        raise
    except Exception:
        raise InputContractError("Protected grouping source could not be verified.") from None
    return {"group_count": 14, "cores_per_group": 2,
            "consecutive_core_pairs_verified": True,
            "cross_slide_donor_count": 0, "direct_donor_identifiers_returned": False}


def load_inputs(paths) -> tuple[np.ndarray, pd.DataFrame, list[str], dict]:
    """Return raw counts, aligned local metadata, ordered probes, and receipts.

    ``paths`` is a ``spatial_benchmark.paths.ProjectPaths`` (or any object with
    ``data_root``). Counts are int32 with shape (407659, 1000), in increasing
    core order and then stable (fov, cell_ID) order. The fixed exploratory split
    trains on cores 1--10 and 15--24; other mapped cores are diagnostic holdout.
    This routine does not fit or apply any expression transformation.
    """
    data_root = Path(paths.data_root).resolve()
    files: list[dict] = []
    grouping_path = data_root / "raw" / "Gastric Study_2.xlsx"
    files.append(_file_record(grouping_path, data_root, EXPECTED_GROUPING_SHA256))
    grouping = _verify_donor_groups(grouping_path)
    core_map_path = data_root / "clinical" / "fov_core_map.csv"
    core_map = pd.read_csv(core_map_path, usecols=["slide", "fov", "core_label"])
    _require(not core_map.duplicated(["slide", "fov"]).any(),
             "Core routing contains duplicate slide-qualified FOVs.")
    _require(len(core_map) == 450 and set(core_map.core_label) == set(range(1, 29)),
             "Core routing no longer matches the 28-core mapped cohort.")
    core_map_record = _file_record(core_map_path, data_root)
    files.append(core_map_record)
    counts = np.empty((EXPECTED_TOTAL_CELLS, 1000), dtype=np.int32)
    metadata_parts: list[pd.DataFrame] = []
    alignment: list[dict] = []
    genes: list[str] | None = None
    offset = 0
    for slide in (1, 2):
        slide_name = f"SO_{slide}"
        prepared = data_root / "processed" / f"so{slide}_14core_relative_qkv_v1"
        manifest_path = prepared / "manifest.json"
        files.append(_file_record(manifest_path, data_root))
        manifest = json.loads(manifest_path.read_text())
        content = {key: value for key, value in manifest.items()
                   if key != "manifest_content_sha256"}
        _require(_canonical_sha256(content) == manifest["manifest_content_sha256"],
                 f"SO{slide} manifest content checksum failed.")
        _require(core_map_record["sha256"] == manifest["source"]["core_map_sha256"],
                 f"SO{slide} core routing differs from prepared provenance.")
        slide_genes = manifest["features"]["gene_names"]
        _require(len(slide_genes) == 1000 and len(set(slide_genes)) == 1000
                 and not any(gene.startswith(("Negative", "SystemControl"))
                             for gene in slide_genes),
                 f"SO{slide} biological probe panel is invalid.")
        if genes is None:
            genes = list(slide_genes)
        _require(genes == slide_genes, "Slides do not have identical ordered probes.")
        core_records = sorted(manifest["cores"], key=lambda item: item["original_core_number"])
        expected_numbers = list(range(1, 15)) if slide == 1 else list(range(15, 29))
        _require([item["original_core_number"] for item in core_records] == expected_numbers,
                 f"SO{slide} manifest does not contain exactly its expected cores.")
        metadata_references = {item["source_artifacts"]["metadata"] for item in core_records}
        metadata_checksums = {item["source_checksums"]["metadata_sha256"] for item in core_records}
        _require(len(metadata_references) == len(metadata_checksums) == 1,
                 f"SO{slide} prepared cores disagree on metadata provenance.")
        metadata_path = data_root / "raw" / next(iter(metadata_references))
        files.append(_file_record(metadata_path, data_root, next(iter(metadata_checksums))))
        usecols = ["slide_ID", "fov", "cell_ID", "CenterX_global_px", "CenterY_global_px",
                   "nCount_RNA", "qcCellsPassed", VENDOR_LABEL, VENDOR_POSTERIOR]
        raw_meta = pd.read_csv(metadata_path, usecols=usecols,
                               dtype={"qcCellsPassed": "string", VENDOR_LABEL: "string"})
        _require(set(pd.to_numeric(raw_meta.slide_ID)) == {slide},
                 f"SO{slide} raw metadata has an inconsistent slide identity.")
        _require(not raw_meta.duplicated(["fov", "cell_ID"]).any(),
                 f"SO{slide} raw metadata contains duplicate cell keys.")
        slide_routes = core_map.loc[core_map.slide == slide_name]
        unmapped = raw_meta.loc[~raw_meta.fov.isin(slide_routes.fov)]
        _require((slide == 1 and len(unmapped) == 0)
                 or (slide == 2 and len(unmapped) == 340 and set(unmapped.fov) == {246}),
                 f"SO{slide} unmapped-cell exclusion differs from the input contract.")
        for record in core_records:
            core_number = int(record["original_core_number"])
            alias = record["alias"]
            _require(alias == f"SO{slide}-C{core_number:02d}", "Unexpected prepared core alias.")
            fovs = slide_routes.loc[slide_routes.core_label == core_number, "fov"]
            selected = raw_meta.loc[raw_meta.fov.isin(fovs)].sort_values(
                ["fov", "cell_ID"], kind="mergesort").reset_index(drop=True)
            relative_npz = f"cores/{alias}.npz"
            npz_path = prepared / relative_npz
            files.append(_file_record(npz_path, data_root, manifest["files"][relative_npz]))
            with np.load(npz_path, allow_pickle=False) as archive:
                core_counts = archive["expression_counts"]
                coordinates = archive["coordinates_um"]
            n_cells = len(selected)
            _require(n_cells == record["cell_count"] and core_counts.shape == (n_cells, 1000)
                     and core_counts.dtype == np.dtype("int32")
                     and np.min(core_counts) >= 0,
                     f"{alias} raw-count shape, dtype, or range failed validation.")
            expected_coordinates = selected[["CenterX_global_px", "CenterY_global_px"]].to_numpy(
                dtype=np.float64) * PIXEL_SIZE_UM
            _require(np.isfinite(coordinates).all()
                     and np.array_equal(coordinates, expected_coordinates),
                     f"{alias} prepared coordinate rows do not exactly align to metadata.")
            depth = core_counts.sum(axis=1, dtype=np.int64)
            _require(np.array_equal(depth, selected.nCount_RNA.to_numpy()),
                     f"{alias} count-row totals do not exactly align to metadata.")
            qc_strings = selected.qcCellsPassed.str.strip().str.lower()
            _require(bool(qc_strings.isin(["true", "false", "1", "0"]).all()),
                     f"{alias} vendor QC contains missing or unsupported flags.")
            _require(not selected[VENDOR_LABEL].isna().any(),
                     f"{alias} has missing vendor stratum annotations.")
            stop = offset + n_cells
            _require(stop <= EXPECTED_TOTAL_CELLS, "Prepared cohort exceeds expected cell count.")
            counts[offset:stop] = core_counts
            metadata_parts.append(pd.DataFrame({
                "core": alias, "core_number": core_number, "slide": slide,
                "donor_group": (core_number + 1) // 2,
                "train": core_number in TRAIN_CORES,
                "qc_pass": qc_strings.isin(["true", "1"]).to_numpy(dtype=bool),
                "vendor_label": selected[VENDOR_LABEL].to_numpy(dtype=str),
                "vendor_posterior": selected[VENDOR_POSTERIOR].to_numpy(dtype=np.float64),
                "depth": depth, "fov": selected.fov.to_numpy(dtype=np.int32),
                "cell_ID": selected.cell_ID.to_numpy(dtype=np.int32),
                "x_um": coordinates[:, 0], "y_um": coordinates[:, 1],
            }))
            alignment.append({"core": alias, "cell_count": n_cells,
                              "row_start": offset, "row_stop": stop,
                              "coordinate_rows_exact": True, "count_totals_exact": True,
                              "keys_unique": True})
            offset = stop
            del core_counts
        _require(sum(item["cell_count"] for item in core_records) == manifest["cohort"]["total_cells"],
                 f"SO{slide} total cell count disagrees with its manifest.")
    _require(offset == EXPECTED_TOTAL_CELLS, "Combined cohort cell count differs from its contract.")
    metadata = pd.concat(metadata_parts, ignore_index=True)
    _require(not metadata.duplicated(["slide", "fov", "cell_ID"]).any(),
             "Combined slide-qualified cell keys are not unique.")
    _require(metadata.groupby("donor_group")["train"].nunique().eq(1).all(),
             "A donor appears in both training and diagnostic partitions.")
    provenance = {
        "schema": "so12_preprocessing_input_receipt_v1", "files": files,
        "alignment_checks": alignment, "protected_grouping": grouping,
        "total_cells": offset, "biological_probe_count": len(genes),
        "genes_sha256": _canonical_sha256(genes), "pixel_size_um": PIXEL_SIZE_UM,
        "raw_counts_used": True, "upstream_normalized_targets_used": False,
        "vendor_labels_are_exploratory_expression_derived_strata": True,
        "qc_policy": "all_mapped_cells_with_vendor_pass_flag_retained",
        "excluded_unmapped": {"slide": 2, "fov": 246, "cell_count": 340},
        "split": {"training_core_numbers": sorted(TRAIN_CORES),
                  "diagnostic_core_numbers": sorted(set(range(1, 29)) - TRAIN_CORES),
                  "training_cells": int(metadata.train.sum()),
                  "diagnostic_cells": int((~metadata.train).sum()),
                  "training_donors": 10, "diagnostic_donors": 4,
                  "donor_overlap": False, "selection_used_expression_outcomes": False,
                  "claim": "exploratory_preprocessing_diagnostic_not_sealed_test"},
    }
    return counts, metadata, genes, provenance

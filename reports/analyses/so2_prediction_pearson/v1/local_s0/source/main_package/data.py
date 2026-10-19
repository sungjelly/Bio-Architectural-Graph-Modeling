"""Leakage-resistant loading for the single legacy true-Normal CosMx core.

The public selection object deliberately contains only the slide and FOVs
needed to locate raw rows.  The restricted donor value and the legacy core
identifier are never retained, represented, logged, or returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd


CELL_KEY_COLUMNS = ("slide", "fov", "cell_ID")
CONTROL_PREFIXES = ("Negative", "SystemControl")
DEFAULT_PIXEL_SIZE_UM = 0.120281

# This is an allow-list, not a convenient subset of the vendor metadata table.
# All values are independently measured morphology or imaging measurements.
ALLOWED_METADATA_COLUMNS = (
    "Area",
    "Area.um2",
    "AspectRatio",
    "Width",
    "Height",
    "Mean.PanCK",
    "Max.PanCK",
    "Mean.G",
    "Max.G",
    "Mean.Membrane",
    "Max.Membrane",
    "Mean.CD45",
    "Max.CD45",
    "Mean.DAPI",
    "Max.DAPI",
    "SplitRatioToLocal",
    "NucArea",
    "NucAspectRatio",
    "Circularity",
    "Eccentricity",
    "Perimeter",
    "Solidity",
)

_RAW_SUFFIXES = {
    "expression": "_exprMat_file.csv",
    "metadata": "_metadata_file.csv",
}
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "pass", "passed"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "fail", "failed", ""})


class DataContractError(ValueError):
    """Raised when an input violates a declared data or leakage contract."""


@dataclass(frozen=True)
class CoreSelection:
    """Non-sensitive routing information for the selected core.

    No donor or core identifier is present by design.  ``fovs`` is needed for
    streaming and is not a model covariate.
    """

    slide: str
    fovs: tuple[int, ...]
    label_policy: str = "unique exact legacy tissue label"

    def __post_init__(self) -> None:
        canonical = normalize_slide(self.slide)
        clean_fovs = tuple(sorted({int(value) for value in self.fovs}))
        if not clean_fovs or any(value <= 0 for value in clean_fovs):
            raise DataContractError("The selected core must contain positive FOV values.")
        object.__setattr__(self, "slide", canonical)
        object.__setattr__(self, "fovs", clean_fovs)


@dataclass(frozen=True)
class CoreDataset:
    """Arrays aligned to one deterministic cell order.

    ``keys``, ``coordinates_px``, ``coordinates_um``, and ``qc_passed`` are
    routing/QC data and must not be concatenated into model node covariates.
    Only ``metadata`` is the allowed always-visible metadata matrix.
    """

    expression: np.ndarray
    metadata: np.ndarray
    coordinates_px: np.ndarray
    coordinates_um: np.ndarray
    keys: pd.DataFrame
    qc_passed: np.ndarray
    gene_names: tuple[str, ...]
    metadata_names: tuple[str, ...] = ALLOWED_METADATA_COLUMNS

    def __post_init__(self) -> None:
        n_cells = len(self.keys)
        aligned = (
            self.expression.shape[0],
            self.metadata.shape[0],
            self.coordinates_px.shape[0],
            self.coordinates_um.shape[0],
            self.qc_passed.shape[0],
        )
        if any(value != n_cells for value in aligned):
            raise DataContractError("Loaded arrays are not aligned to the cell keys.")
        if self.expression.ndim != 2 or self.expression.shape[1] != len(self.gene_names):
            raise DataContractError("Expression shape does not match the biological probes.")
        if self.metadata.ndim != 2 or self.metadata.shape[1] != len(self.metadata_names):
            raise DataContractError("Metadata shape does not match the fixed allow-list.")
        if tuple(self.metadata_names) != ALLOWED_METADATA_COLUMNS:
            raise DataContractError("Model metadata must use the exact 22-column allow-list.")
        if self.coordinates_px.shape != (n_cells, 2):
            raise DataContractError("Pixel coordinates must have shape [n_cells, 2].")
        if self.coordinates_um.shape != (n_cells, 2):
            raise DataContractError("Physical coordinates must have shape [n_cells, 2].")
        if tuple(self.keys.columns) != CELL_KEY_COLUMNS:
            raise DataContractError("Cell keys must be slide-qualified and kept separate.")
        if any(name.startswith(CONTROL_PREFIXES) for name in self.gene_names):
            raise DataContractError("Technical controls cannot be biological targets.")

    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]


def normalize_slide(value: object) -> str:
    """Normalize an explicit slide value to ``SO_1`` or ``SO_2``."""

    text = str(value).strip().upper()
    # Vendor metadata uses the explicit numeric slide_ID 1/2, while filenames
    # and the core map use SO_1/SO_2.
    if text in {"1", "1.0"}:
        return "SO_1"
    if text in {"2", "2.0"}:
        return "SO_2"
    match = re.search(r"SO[_-]?([12])(?:\D|$)", text)
    if match is None:
        raise DataContractError("Could not normalize an explicit CosMx slide value.")
    return f"SO_{match.group(1)}"


def resolve_nested_raw_path(raw_dir: str | Path, basename: str) -> Path:
    """Resolve both ``raw/name`` and the observed ``raw/name/name`` layout."""

    direct = Path(raw_dir) / basename
    nested = direct / basename
    path = nested if nested.is_file() else direct
    if not path.is_file():
        raise FileNotFoundError(f"Required raw basename was not found: {basename}")
    return path


def discover_slide_raw_path(
    raw_dir: str | Path,
    slide: str,
    kind: Literal["expression", "metadata"],
) -> Path:
    """Find one raw file using an explicit slide token and known suffix."""

    canonical = normalize_slide(slide)
    suffix = _RAW_SUFFIXES[kind]
    candidates: list[Path] = []
    for entry in Path(raw_dir).iterdir():
        if not entry.name.endswith(suffix):
            continue
        try:
            entry_slide = normalize_slide(entry.name)
        except DataContractError:
            continue
        if entry_slide != canonical:
            continue
        candidate = resolve_nested_raw_path(raw_dir, entry.name)
        candidates.append(candidate)
    if len(candidates) != 1:
        raise DataContractError(
            f"Expected one {kind} raw file for the selected slide; found {len(candidates)}."
        )
    return candidates[0]


def _read_csv_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise DataContractError("A required CSV is empty.") from exc


def biological_probe_columns(
    expression_csv: str | Path,
    *,
    expected_count: int | None = 1000,
) -> tuple[str, ...]:
    """Return all non-control probes in source order.

    Controls are prefix-filtered because they are interspersed in the panel.
    Slash-combined probe symbols are intentionally left indivisible.
    """

    header = _read_csv_header(expression_csv)
    required = {"fov", "cell_ID"}
    if not required.issubset(header):
        raise DataContractError("Expression CSV lacks the canonical local cell keys.")
    probes = [
        name
        for name in header
        if name not in required and not name.startswith(CONTROL_PREFIXES)
    ]
    if len(set(probes)) != len(probes):
        raise DataContractError("Biological probe names must be unique.")
    if expected_count is not None and len(probes) != expected_count:
        raise DataContractError(
            f"Expected {expected_count} biological probes; found {len(probes)}."
        )
    return tuple(probes)


def _exact_string_positions(frame: pd.DataFrame, text: str) -> list[tuple[int, int]]:
    target = text.strip()
    positions: list[tuple[int, int]] = []
    values = frame.to_numpy(dtype=object)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if isinstance(value, str) and value.strip() == target:
                positions.append((row_idx, col_idx))
    return positions


def _private_core_value_from_legacy_row(
    legacy: pd.DataFrame,
    row_idx: int,
    tissue_col_idx: int,
) -> int:
    candidates: list[int] = []
    for value in legacy.iloc[row_idx, :tissue_col_idx].tolist():
        if pd.isna(value) or isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric.is_integer() and numeric > 0:
            candidates.append(int(numeric))
    if len(candidates) != 1:
        raise DataContractError(
            "The unique legacy tissue row does not contain one unambiguous core key."
        )
    return candidates[0]


def _validate_pathology_review(
    review_workbook: str | Path,
    private_core_value: int,
) -> None:
    review = pd.read_excel(review_workbook, sheet_name=0)
    if review.empty:
        raise DataContractError("The pathology review workbook is empty.")
    key_columns = [
        column for column in review.columns if str(column).strip() == "슬라이드번호"
    ]
    note_columns = [
        column for column in review.columns if str(column).strip().startswith("비고")
    ]
    if len(key_columns) != 1 or len(note_columns) != 1:
        raise DataContractError("The pathology review workbook has an unexpected schema.")
    numeric_key = pd.to_numeric(review[key_columns[0]], errors="coerce")
    selected = review.loc[numeric_key == private_core_value]
    if len(selected) != 1:
        raise DataContractError(
            "The selected legacy row is not uniquely represented in pathology review."
        )
    note = selected.iloc[0][note_columns[0]]
    if not (pd.isna(note) or (isinstance(note, str) and not note.strip())):
        raise DataContractError(
            "The selected legacy row has a pathology correction note; auto-selection stopped."
        )


def select_unique_legacy_true_normal_core(
    legacy_workbook: str | Path,
    core_map_csv: str | Path,
    *,
    pathology_review_workbook: str | Path | None = None,
    tissue_label: str = "정상",
) -> CoreSelection:
    """Resolve the unique exact legacy true-Normal label without exposing IDs.

    The legacy numeric core value exists only in local variables long enough to
    select the slide-qualified FOV rows.  It never appears in the returned
    object or in exception text.
    """

    legacy = pd.read_excel(legacy_workbook, sheet_name=0, header=None)
    positions = _exact_string_positions(legacy, tissue_label)
    if len(positions) != 1:
        raise DataContractError(
            f"Expected one exact legacy true-Normal tissue label; found {len(positions)}."
        )
    row_idx, tissue_col_idx = positions[0]
    private_core_value = _private_core_value_from_legacy_row(
        legacy, row_idx, tissue_col_idx
    )
    if pathology_review_workbook is not None:
        _validate_pathology_review(pathology_review_workbook, private_core_value)

    core_map = pd.read_csv(
        core_map_csv,
        usecols=["slide", "core_label", "fov"],
        dtype={"slide": "string", "core_label": "int64", "fov": "int64"},
    )
    if core_map.duplicated(["slide", "fov"]).any():
        raise DataContractError("The core map has duplicate slide-qualified FOV keys.")
    selected = core_map.loc[core_map["core_label"] == private_core_value].copy()
    if selected.empty:
        raise DataContractError("The selected legacy tissue row has no mapped FOVs.")
    try:
        selected["slide"] = selected["slide"].map(normalize_slide)
    except DataContractError as exc:
        raise DataContractError("The selected core map rows contain an invalid slide.") from exc
    slides = selected["slide"].drop_duplicates().tolist()
    if len(slides) != 1:
        raise DataContractError("The selected core must map to exactly one slide.")
    fovs = tuple(sorted(selected["fov"].astype(int).unique().tolist()))
    return CoreSelection(slide=slides[0], fovs=fovs)


def _iter_selected_chunks(
    csv_path: str | Path,
    *,
    slide: str,
    fovs: Sequence[int],
    usecols: Sequence[str],
    dtype: dict[str, object],
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    wanted_fovs = frozenset(int(value) for value in fovs)
    source_slide = normalize_slide(Path(csv_path).name)
    canonical_slide = normalize_slide(slide)
    if source_slide != canonical_slide:
        raise DataContractError("A raw file belongs to a different explicit slide.")
    for chunk in pd.read_csv(
        csv_path,
        usecols=list(usecols),
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    ):
        selected = chunk.loc[chunk["fov"].isin(wanted_fovs)].copy()
        if selected.empty:
            continue
        selected.insert(0, "slide", canonical_slide)
        yield selected


def _stream_selected_frame(
    csv_path: str | Path,
    *,
    slide: str,
    fovs: Sequence[int],
    usecols: Sequence[str],
    dtype: dict[str, object],
    chunksize: int,
) -> pd.DataFrame:
    chunks = list(
        _iter_selected_chunks(
            csv_path,
            slide=slide,
            fovs=fovs,
            usecols=usecols,
            dtype=dtype,
            chunksize=chunksize,
        )
    )
    if not chunks:
        raise DataContractError("No raw rows matched the selected core FOVs.")
    result = pd.concat(chunks, axis=0, ignore_index=True)
    if result.duplicated(list(CELL_KEY_COLUMNS)).any():
        raise DataContractError("A raw table has duplicate slide-qualified cell keys.")
    return result


def _coerce_qc_passed(series: pd.Series) -> np.ndarray:
    result = np.zeros(len(series), dtype=bool)
    for idx, value in enumerate(series.to_numpy(dtype=object)):
        if pd.isna(value):
            result[idx] = False
        elif isinstance(value, (bool, np.bool_)):
            result[idx] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            result[idx] = bool(value)
        else:
            normalized = str(value).strip().lower()
            if normalized in _TRUE_STRINGS:
                result[idx] = True
            elif normalized in _FALSE_STRINGS:
                result[idx] = False
            else:
                raise DataContractError("Encountered an unknown vendor QC indicator value.")
    return result


def _validate_metadata_slide(metadata: pd.DataFrame, expected_slide: str) -> None:
    if "slide_ID" not in metadata:
        raise DataContractError("Metadata lacks its explicit slide_ID field.")
    observed: set[str] = set()
    for value in metadata["slide_ID"].dropna().unique().tolist():
        observed.add(normalize_slide(value))
    if observed != {normalize_slide(expected_slide)}:
        raise DataContractError("Metadata slide_ID disagrees with the source filename.")


def load_selected_core(
    raw_dir: str | Path,
    selection: CoreSelection,
    *,
    chunksize: int = 2048,
    expected_biological_probes: int | None = 1000,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    qc_policy: Literal["all", "passed"] = "all",
) -> CoreDataset:
    """Stream, slide-qualify, validate, and align expression plus metadata.

    The function opens only the selected slide's two cell-level files.  It
    parses them in chunks and retains only rows from the selected FOV set.
    """

    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive and finite.")
    if qc_policy not in {"all", "passed"}:
        raise ValueError("qc_policy must be 'all' or 'passed'.")

    expression_path = discover_slide_raw_path(raw_dir, selection.slide, "expression")
    metadata_path = discover_slide_raw_path(raw_dir, selection.slide, "metadata")
    genes = biological_probe_columns(
        expression_path, expected_count=expected_biological_probes
    )

    expression_usecols = ("fov", "cell_ID", *genes)
    expression_dtype: dict[str, object] = {
        "fov": "int32",
        "cell_ID": "int32",
        **{gene: "int32" for gene in genes},
    }
    expression = _stream_selected_frame(
        expression_path,
        slide=selection.slide,
        fovs=selection.fovs,
        usecols=expression_usecols,
        dtype=expression_dtype,
        chunksize=chunksize,
    )

    metadata_header = _read_csv_header(metadata_path)
    metadata_required = {
        "fov",
        "cell_ID",
        "slide_ID",
        "CenterX_global_px",
        "CenterY_global_px",
        "qcCellsPassed",
        *ALLOWED_METADATA_COLUMNS,
    }
    missing = sorted(metadata_required.difference(metadata_header))
    if missing:
        raise DataContractError(
            f"Metadata is missing {len(missing)} required allow-listed/routing fields."
        )
    metadata_usecols = (
        "fov",
        "cell_ID",
        "slide_ID",
        "CenterX_global_px",
        "CenterY_global_px",
        "qcCellsPassed",
        *ALLOWED_METADATA_COLUMNS,
    )
    metadata_dtype: dict[str, object] = {
        "fov": "int32",
        "cell_ID": "int32",
        "slide_ID": "string",
        "CenterX_global_px": "float64",
        "CenterY_global_px": "float64",
        "qcCellsPassed": "string",
        **{column: "float32" for column in ALLOWED_METADATA_COLUMNS},
    }
    metadata = _stream_selected_frame(
        metadata_path,
        slide=selection.slide,
        fovs=selection.fovs,
        usecols=metadata_usecols,
        dtype=metadata_dtype,
        chunksize=chunksize,
    )
    _validate_metadata_slide(metadata, selection.slide)

    coverage = expression.loc[:, CELL_KEY_COLUMNS].merge(
        metadata.loc[:, CELL_KEY_COLUMNS],
        how="outer",
        on=list(CELL_KEY_COLUMNS),
        indicator=True,
        validate="one_to_one",
    )
    if not (coverage["_merge"] == "both").all():
        missing_expression = int((coverage["_merge"] == "right_only").sum())
        missing_metadata = int((coverage["_merge"] == "left_only").sum())
        raise DataContractError(
            "Expression/metadata key coverage is incomplete "
            f"(missing expression={missing_expression}, missing metadata={missing_metadata})."
        )

    joined = expression.merge(
        metadata,
        how="inner",
        on=list(CELL_KEY_COLUMNS),
        validate="one_to_one",
        sort=False,
    )
    joined = joined.sort_values(list(CELL_KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )
    qc_passed = _coerce_qc_passed(joined["qcCellsPassed"])
    if qc_policy == "passed":
        joined = joined.loc[qc_passed].reset_index(drop=True)
        qc_passed = np.ones(len(joined), dtype=bool)
    if joined.empty:
        raise DataContractError("The configured QC policy removed every selected cell.")

    coordinates_px = joined[
        ["CenterX_global_px", "CenterY_global_px"]
    ].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(coordinates_px).all():
        raise DataContractError("Global cell-center coordinates must be finite.")
    coordinates_um = coordinates_px * float(pixel_size_um)

    model_metadata = joined[list(ALLOWED_METADATA_COLUMNS)].to_numpy(
        dtype=np.float32, copy=True
    )
    counts = joined[list(genes)].to_numpy(dtype=np.int32, copy=True)
    if np.any(counts < 0):
        raise DataContractError("Expression counts must be nonnegative.")
    keys = joined.loc[:, CELL_KEY_COLUMNS].copy()
    keys["slide"] = keys["slide"].astype("string")
    keys["fov"] = keys["fov"].astype("int32")
    keys["cell_ID"] = keys["cell_ID"].astype("int32")
    return CoreDataset(
        expression=counts,
        metadata=model_metadata,
        coordinates_px=coordinates_px,
        coordinates_um=coordinates_um,
        keys=keys,
        qc_passed=qc_passed,
        gene_names=genes,
    )


def load_true_normal_core(
    project_root: str | Path,
    *,
    chunksize: int = 2048,
    qc_policy: Literal["all", "passed"] = "all",
    expected_biological_probes: int | None = 1000,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
) -> CoreDataset:
    """Convenience entry point using the repository's declared input layout."""

    root = Path(project_root)
    clinical = root / "data" / "clinical"
    selection = select_unique_legacy_true_normal_core(
        clinical / "Gastric Study_Old.xlsx",
        clinical / "fov_core_map.csv",
        pathology_review_workbook=clinical / "Gastric Study.xlsx",
    )
    return load_selected_core(
        root / "data" / "raw",
        selection,
        chunksize=chunksize,
        qc_policy=qc_policy,
        expected_biological_probes=expected_biological_probes,
        pixel_size_um=pixel_size_um,
    )

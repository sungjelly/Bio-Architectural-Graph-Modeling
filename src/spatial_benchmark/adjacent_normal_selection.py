"""Protected, deterministic selection of ten reviewed adjacent-normal cores.

This module reconciles the legacy donor/core workbook with the current
pathology-review sheet, counts cells from the slide-qualified metadata files,
and selects five eligible cores per slide.  Raw donor identifiers are reduced
to in-memory one-way digests and are never returned, logged, or serialized.

The written manifest is a protected local routing artifact because it contains
the source core keys.  The public receipt deliberately contains only opaque
aliases plus the slide/FOV routing required by downstream data loading.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    DataContractError,
    discover_slide_raw_path,
    normalize_slide,
)
from .fingerprints import sha256_file
from .identifiers import canonical_sha256


SCHEMA_VERSION = 1
POLICY_ID = "adjacent_normal_reviewed_balanced_v1"
SLIDES = ("SO_1", "SO_2")
LEGACY_ADJACENT_LABELS = ("주변조직", "주변조직 (N)")
EXPECTED_LEGACY_LABEL_COUNTS = {"주변조직": 8, "주변조직 (N)": 6}
EXPECTED_ADJACENT_CANDIDATES = 14
EXPECTED_DISTINCT_DONORS = 14
REVIEW_DIAGNOSIS = "Normal"
REVIEW_RESULT = "정상조직확인"
MINIMUM_CELLS = 5_001
SELECTIONS_PER_SLIDE = 5
SELECTION_COUNT = len(SLIDES) * SELECTIONS_PER_SLIDE


class AdjacentNormalSelectionError(DataContractError):
    """Raised when protected inputs violate the locked selection policy."""


@dataclass(frozen=True, slots=True)
class AdjacentNormalRoute:
    """Non-sensitive downstream routing for one selected tissue core."""

    alias: str
    slide: str
    fovs: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (
            self.alias.startswith("ANC-")
            and len(self.alias) == 6
            and self.alias[-2:].isdigit()
        ):
            raise AdjacentNormalSelectionError(
                "Adjacent-normal aliases must use the ANC-## form."
            )
        canonical_slide = normalize_slide(self.slide)
        clean_fovs = tuple(sorted({int(value) for value in self.fovs}))
        if not clean_fovs or any(value <= 0 for value in clean_fovs):
            raise AdjacentNormalSelectionError(
                "Every adjacent-normal route must contain positive FOV values."
            )
        object.__setattr__(self, "slide", canonical_slide)
        object.__setattr__(self, "fovs", clean_fovs)

    def to_public_dict(self) -> dict[str, Any]:
        """Return only the permitted public alias and routing fields."""

        return {
            "alias": self.alias,
            "slide": self.slide,
            "fovs": list(self.fovs),
        }


@dataclass(frozen=True, slots=True)
class AdjacentNormalSelectionReceipt:
    """Public handoff after a protected manifest has been written."""

    routes: tuple[AdjacentNormalRoute, ...]

    def __post_init__(self) -> None:
        if len(self.routes) != SELECTION_COUNT:
            raise AdjacentNormalSelectionError(
                "The public receipt must contain exactly ten routes."
            )
        aliases = [route.alias for route in self.routes]
        if len(set(aliases)) != len(aliases):
            raise AdjacentNormalSelectionError("Public route aliases must be unique.")

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize without protected core or donor identifiers."""

        return {
            "status": "created",
            "routes": [route.to_public_dict() for route in self.routes],
        }


@dataclass(frozen=True, slots=True)
class _LegacyCandidate:
    protected_core_key: int
    donor_digest: str
    source_label: str


@dataclass(frozen=True, slots=True)
class _RoutedCandidate:
    protected_core_key: int
    donor_digest: str
    source_label: str
    slide: str
    fovs: tuple[int, ...]
    n_cells: int


def _clean_exact_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result if result else None


def _is_blank(value: object) -> bool:
    return bool(pd.isna(value)) or (
        isinstance(value, str) and not value.strip()
    )


def _positive_integer(value: object) -> int | None:
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return None
    return int(number)


def _private_core_key(
    legacy: pd.DataFrame,
    *,
    row_index: int,
    tissue_column_index: int,
) -> int:
    candidates = [
        key
        for value in legacy.iloc[row_index, :tissue_column_index].tolist()
        if (key := _positive_integer(value)) is not None
    ]
    if len(candidates) != 1:
        raise AdjacentNormalSelectionError(
            "A legacy adjacent-normal row lacks one unambiguous protected core key."
        )
    return candidates[0]


def _private_donor_digest(
    legacy: pd.DataFrame,
    *,
    row_index: int,
    tissue_column_index: int,
) -> str:
    if tissue_column_index < 2:
        raise AdjacentNormalSelectionError(
            "A legacy adjacent-normal row lacks the expected restricted donor field."
        )
    value = legacy.iat[row_index, tissue_column_index - 1]
    donor = _clean_exact_text(value)
    if donor is None:
        raise AdjacentNormalSelectionError(
            "A legacy adjacent-normal row has a blank restricted donor field."
        )
    # The raw identifier does not leave this function.
    return hashlib.sha256(
        f"bagm-private-donor-v1\0{donor}".encode("utf-8")
    ).hexdigest()


def _load_legacy_candidates(path: str | Path) -> list[_LegacyCandidate]:
    legacy = pd.read_excel(path, sheet_name=0, header=None)
    if legacy.empty:
        raise AdjacentNormalSelectionError("The legacy clinical workbook is empty.")

    label_counts: Counter[str] = Counter()
    positions: list[tuple[int, int, str]] = []
    unexpected_adjacent_variants = 0
    for row_index, row in enumerate(legacy.to_numpy(dtype=object)):
        for column_index, value in enumerate(row):
            text = _clean_exact_text(value)
            if text in LEGACY_ADJACENT_LABELS:
                assert text is not None
                label_counts[text] += 1
                positions.append((row_index, column_index, text))
            elif text is not None and "주변조직" in text:
                unexpected_adjacent_variants += 1

    if unexpected_adjacent_variants:
        raise AdjacentNormalSelectionError(
            "The legacy workbook contains an unapproved adjacent-normal label variant."
        )
    if dict(label_counts) != EXPECTED_LEGACY_LABEL_COUNTS:
        raise AdjacentNormalSelectionError(
            "The legacy adjacent-normal label counts do not match the locked "
            "8 plain plus 6 '(N)' policy."
        )

    candidates = [
        _LegacyCandidate(
            protected_core_key=_private_core_key(
                legacy,
                row_index=row_index,
                tissue_column_index=column_index,
            ),
            donor_digest=_private_donor_digest(
                legacy,
                row_index=row_index,
                tissue_column_index=column_index,
            ),
            source_label=source_label,
        )
        for row_index, column_index, source_label in positions
    ]
    if len(candidates) != EXPECTED_ADJACENT_CANDIDATES:
        raise AdjacentNormalSelectionError(
            "The legacy workbook does not contain fourteen adjacent-normal candidates."
        )
    if len({candidate.protected_core_key for candidate in candidates}) != len(
        candidates
    ):
        raise AdjacentNormalSelectionError(
            "Legacy adjacent-normal protected core keys are not unique."
        )
    if len({candidate.donor_digest for candidate in candidates}) != (
        EXPECTED_DISTINCT_DONORS
    ):
        raise AdjacentNormalSelectionError(
            "Legacy adjacent-normal candidates do not represent fourteen distinct donors."
        )
    return candidates


def _required_review_column(
    review: pd.DataFrame,
    *,
    exact: str | None = None,
    prefix: str | None = None,
) -> object:
    columns = [
        column
        for column in review.columns
        if (
            (exact is not None and str(column).strip() == exact)
            or (
                prefix is not None
                and str(column).strip().startswith(prefix)
            )
        )
    ]
    if len(columns) != 1:
        raise AdjacentNormalSelectionError(
            "The pathology-review workbook has an unexpected schema."
        )
    return columns[0]


def _validate_pathology_review(
    path: str | Path,
    candidates: Sequence[_LegacyCandidate],
) -> None:
    review = pd.read_excel(path, sheet_name=0)
    if review.empty:
        raise AdjacentNormalSelectionError(
            "The pathology-review workbook is empty."
        )
    key_column = _required_review_column(review, exact="슬라이드번호")
    diagnosis_column = _required_review_column(review, exact="진단명")
    result_column = _required_review_column(review, exact="결과")
    correction_column = _required_review_column(review, prefix="비고")

    numeric_keys = pd.to_numeric(review[key_column], errors="coerce")
    if numeric_keys.isna().any() or not np.equal(
        numeric_keys.to_numpy(dtype=np.float64),
        np.floor(numeric_keys.to_numpy(dtype=np.float64)),
    ).all():
        raise AdjacentNormalSelectionError(
            "The pathology-review key column is not entirely integral."
        )
    if numeric_keys.duplicated().any():
        raise AdjacentNormalSelectionError(
            "The pathology-review protected core keys are not unique."
        )
    keyed_rows = {
        int(key): row
        for key, (_, row) in zip(
            numeric_keys.to_numpy(dtype=np.int64),
            review.iterrows(),
        )
    }

    matched = 0
    for candidate in candidates:
        row = keyed_rows.get(candidate.protected_core_key)
        if row is None:
            raise AdjacentNormalSelectionError(
                "A legacy adjacent-normal candidate is absent from pathology review."
            )
        diagnosis = _clean_exact_text(row[diagnosis_column])
        result = _clean_exact_text(row[result_column])
        correction = row[correction_column]
        if (
            diagnosis != REVIEW_DIAGNOSIS
            or result != REVIEW_RESULT
            or not _is_blank(correction)
        ):
            raise AdjacentNormalSelectionError(
                "A legacy adjacent-normal candidate fails the locked current-review "
                "diagnosis/result/blank-correction policy."
            )
        matched += 1
    if matched != EXPECTED_ADJACENT_CANDIDATES:
        raise AdjacentNormalSelectionError(
            "Pathology review did not validate all fourteen candidates."
        )


def _load_core_map(path: str | Path) -> pd.DataFrame:
    core_map = pd.read_csv(path)
    required = {"slide", "core_label", "fov"}
    if not required.issubset(core_map.columns):
        raise AdjacentNormalSelectionError(
            "The core map lacks required slide/core/FOV columns."
        )
    core_map = core_map.loc[:, ["slide", "core_label", "fov"]].copy()
    try:
        core_map["slide"] = core_map["slide"].map(normalize_slide)
    except DataContractError as exc:
        raise AdjacentNormalSelectionError(
            "The core map contains an invalid explicit slide value."
        ) from exc
    for column in ("core_label", "fov"):
        numeric = pd.to_numeric(core_map[column], errors="coerce")
        if numeric.isna().any() or not np.equal(
            numeric.to_numpy(dtype=np.float64),
            np.floor(numeric.to_numpy(dtype=np.float64)),
        ).all():
            raise AdjacentNormalSelectionError(
                "The core map contains a non-integral protected key or FOV."
            )
        core_map[column] = numeric.astype(np.int64)
    if (core_map[["core_label", "fov"]] <= 0).any().any():
        raise AdjacentNormalSelectionError(
            "Core-map protected keys and FOV values must be positive."
        )
    if core_map.duplicated(["slide", "fov"]).any():
        raise AdjacentNormalSelectionError(
            "The core map has duplicate slide-qualified FOV assignments."
        )
    return core_map


def _candidate_routing(
    candidates: Sequence[_LegacyCandidate],
    core_map: pd.DataFrame,
) -> dict[int, tuple[str, tuple[int, ...]]]:
    routing: dict[int, tuple[str, tuple[int, ...]]] = {}
    for candidate in candidates:
        rows = core_map.loc[
            core_map["core_label"] == candidate.protected_core_key
        ]
        if rows.empty:
            raise AdjacentNormalSelectionError(
                "An adjacent-normal candidate has no core-map FOV routing."
            )
        slides = rows["slide"].drop_duplicates().tolist()
        if len(slides) != 1:
            raise AdjacentNormalSelectionError(
                "An adjacent-normal candidate maps to more than one slide."
            )
        fovs = tuple(sorted(rows["fov"].astype(int).unique().tolist()))
        routing[candidate.protected_core_key] = (slides[0], fovs)
    return routing


@dataclass(frozen=True, slots=True)
class _MetadataCount:
    cell_count_by_fov: Mapping[int, int]
    row_count: int
    fovs: tuple[int, ...]


def _iter_metadata_key_chunks(
    path: str | Path,
    *,
    expected_slide: str,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if normalize_slide(Path(path).name) != normalize_slide(expected_slide):
        raise AdjacentNormalSelectionError(
            "A metadata file belongs to a different explicit slide."
        )
    try:
        iterator = pd.read_csv(
            path,
            usecols=["fov", "cell_ID", "slide_ID"],
            dtype={"fov": "int64", "cell_ID": "int64", "slide_ID": "string"},
            chunksize=chunksize,
            low_memory=False,
        )
    except ValueError as exc:
        raise AdjacentNormalSelectionError(
            "A metadata file lacks the required slide-qualified cell keys."
        ) from exc
    yield from iterator


def _count_metadata_cells(
    path: str | Path,
    *,
    slide: str,
    chunksize: int,
) -> _MetadataCount:
    canonical_slide = normalize_slide(slide)
    counts: Counter[int] = Counter()
    seen_keys: set[int] = set()
    row_count = 0
    for chunk in _iter_metadata_key_chunks(
        path,
        expected_slide=canonical_slide,
        chunksize=chunksize,
    ):
        if chunk.empty:
            continue
        if chunk["slide_ID"].isna().any():
            raise AdjacentNormalSelectionError(
                "Metadata contains a missing explicit slide_ID."
            )
        if (chunk[["fov", "cell_ID"]] <= 0).any().any():
            raise AdjacentNormalSelectionError(
                "Metadata FOV and cell keys must be positive."
            )
        if (
            chunk[["fov", "cell_ID"]].to_numpy(dtype=np.uint64).max()
            > np.iinfo(np.uint32).max
        ):
            raise AdjacentNormalSelectionError(
                "Metadata FOV or cell keys exceed the supported key range."
            )
        observed_slides: set[str] = set()
        for value in chunk["slide_ID"].dropna().unique().tolist():
            observed_slides.add(normalize_slide(value))
        if observed_slides != {canonical_slide}:
            raise AdjacentNormalSelectionError(
                "Metadata slide_ID disagrees with its source filename."
            )

        fov_values = chunk["fov"].to_numpy(dtype=np.uint64, copy=False)
        cell_values = chunk["cell_ID"].to_numpy(dtype=np.uint64, copy=False)
        encoded = (fov_values << np.uint64(32)) | cell_values
        chunk_keys = {int(value) for value in encoded.tolist()}
        if len(chunk_keys) != len(chunk) or not seen_keys.isdisjoint(chunk_keys):
            raise AdjacentNormalSelectionError(
                "Metadata contains duplicate slide-qualified cell keys."
            )
        seen_keys.update(chunk_keys)
        counts.update(
            {
                int(fov): int(count)
                for fov, count in chunk.groupby("fov", sort=False).size().items()
            }
        )
        row_count += len(chunk)
    if row_count == 0:
        raise AdjacentNormalSelectionError("A required metadata file is empty.")
    return _MetadataCount(
        cell_count_by_fov=dict(counts),
        row_count=row_count,
        fovs=tuple(sorted(counts)),
    )


def _route_and_count_candidates(
    candidates: Sequence[_LegacyCandidate],
    routing: Mapping[int, tuple[str, tuple[int, ...]]],
    metadata_counts: Mapping[str, _MetadataCount],
) -> list[_RoutedCandidate]:
    routed: list[_RoutedCandidate] = []
    for candidate in candidates:
        slide, fovs = routing[candidate.protected_core_key]
        slide_counts = metadata_counts[slide].cell_count_by_fov
        if any(fov not in slide_counts for fov in fovs):
            raise AdjacentNormalSelectionError(
                "A mapped adjacent-normal FOV has no cells in slide metadata."
            )
        n_cells = sum(int(slide_counts[fov]) for fov in fovs)
        routed.append(
            _RoutedCandidate(
                protected_core_key=candidate.protected_core_key,
                donor_digest=candidate.donor_digest,
                source_label=candidate.source_label,
                slide=slide,
                fovs=fovs,
                n_cells=n_cells,
            )
        )
    return routed


def _select_balanced(
    candidates: Sequence[_RoutedCandidate],
) -> tuple[list[_RoutedCandidate], dict[str, float], dict[int, int]]:
    selected: list[_RoutedCandidate] = []
    medians: dict[str, float] = {}
    ranks: dict[int, int] = {}
    for slide in SLIDES:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.slide == slide and candidate.n_cells >= MINIMUM_CELLS
        ]
        if len(eligible) < SELECTIONS_PER_SLIDE:
            raise AdjacentNormalSelectionError(
                "A slide has fewer than five adjacent-normal candidates with "
                "at least 5,001 metadata cells."
            )
        median = float(np.median([candidate.n_cells for candidate in eligible]))
        medians[slide] = median
        ranked = sorted(
            eligible,
            key=lambda candidate: (
                abs(candidate.n_cells - median),
                candidate.protected_core_key,
            ),
        )
        for rank, candidate in enumerate(
            ranked[:SELECTIONS_PER_SLIDE],
            start=1,
        ):
            ranks[candidate.protected_core_key] = rank
            selected.append(candidate)
    if len(selected) != SELECTION_COUNT:
        raise AdjacentNormalSelectionError(
            "Balanced selection did not produce exactly ten candidates."
        )
    if len({candidate.donor_digest for candidate in selected}) != SELECTION_COUNT:
        raise AdjacentNormalSelectionError(
            "The selected candidates do not represent ten distinct donors."
        )
    return selected, medians, ranks


def _input_provenance(role: str, path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required protected input is missing for role {role}.")
    stat_result = source.stat()
    return {
        "role": role,
        "basename": source.name,
        "size_bytes": int(stat_result.st_size),
        "sha256": sha256_file(source),
    }


def _aggregate_unmapped_metadata(
    core_map: pd.DataFrame,
    metadata_counts: Mapping[str, _MetadataCount],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for slide in SLIDES:
        mapped = set(
            core_map.loc[core_map["slide"] == slide, "fov"].astype(int).tolist()
        )
        observed = set(metadata_counts[slide].fovs)
        unmapped = sorted(observed.difference(mapped))
        result[slide] = {
            "fovs": unmapped,
            "n_fovs": len(unmapped),
            "n_cells": sum(
                int(metadata_counts[slide].cell_count_by_fov[fov])
                for fov in unmapped
            ),
            "handling": "excluded_without_inferred_core_assignment",
        }
    return result


def _build_manifest(
    *,
    selected: Sequence[_RoutedCandidate],
    all_candidates: Sequence[_RoutedCandidate],
    medians: Mapping[str, float],
    ranks: Mapping[int, int],
    core_map: pd.DataFrame,
    metadata_counts: Mapping[str, _MetadataCount],
    provenance: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[AdjacentNormalRoute, ...]]:
    routes: list[AdjacentNormalRoute] = []
    protected_records: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        alias = f"ANC-{index:02d}"
        route = AdjacentNormalRoute(
            alias=alias,
            slide=candidate.slide,
            fovs=candidate.fovs,
        )
        routes.append(route)
        median = medians[candidate.slide]
        protected_records.append(
            {
                "alias": alias,
                "protected_core_key": candidate.protected_core_key,
                "slide": candidate.slide,
                "fovs": list(candidate.fovs),
                "n_fovs": len(candidate.fovs),
                "n_cells": candidate.n_cells,
                "absolute_cell_count_distance_from_slide_median": abs(
                    candidate.n_cells - median
                ),
                "selection_rank_within_slide": ranks[
                    candidate.protected_core_key
                ],
            }
        )

    eligible = [
        candidate
        for candidate in all_candidates
        if candidate.n_cells >= MINIMUM_CELLS
    ]
    label_counts = Counter(
        candidate.source_label for candidate in all_candidates
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "protected_adjacent_normal_core_selection",
        "policy_id": POLICY_ID,
        "confidentiality": {
            "contains_protected_core_keys": True,
            "contains_raw_donor_ids": False,
            "handling": "local protected artifact; do not commit or publish",
        },
        "selection_policy": {
            "legacy_label_mapping": {
                "canonical_label": "AdjacentNormal",
                "strict_exact_allow_list": list(LEGACY_ADJACENT_LABELS),
                "expected_source_counts": EXPECTED_LEGACY_LABEL_COUNTS,
                "reject_other_adjacent_like_variants": True,
            },
            "pathology_review": {
                "diagnosis_exact": REVIEW_DIAGNOSIS,
                "result_exact": REVIEW_RESULT,
                "correction": "blank",
            },
            "cell_count_source": "all rows in slide metadata",
            "minimum_cells_inclusive": MINIMUM_CELLS,
            "selections_per_slide": SELECTIONS_PER_SLIDE,
            "ranking": (
                "absolute distance from the slide-specific median eligible "
                "cell count; protected core key ascending as deterministic tie-break"
            ),
            "alias_order": (
                "SO_1 then SO_2; within slide, selection ranking order"
            ),
        },
        "input_provenance": list(provenance),
        "aggregate_validation": {
            "legacy_source_label_counts": dict(sorted(label_counts.items())),
            "legacy_adjacent_candidate_count": len(all_candidates),
            "pathology_review_match_count": len(all_candidates),
            "distinct_donor_count": len(
                {candidate.donor_digest for candidate in all_candidates}
            ),
            "mapped_candidate_count": len(all_candidates),
            "eligible_candidate_count": len(eligible),
            "eligible_candidate_count_by_slide": {
                slide: sum(
                    candidate.slide == slide
                    and candidate.n_cells >= MINIMUM_CELLS
                    for candidate in all_candidates
                )
                for slide in SLIDES
            },
            "selected_count": len(selected),
            "selected_count_by_slide": {
                slide: sum(candidate.slide == slide for candidate in selected)
                for slide in SLIDES
            },
            "selected_distinct_donor_count": len(
                {candidate.donor_digest for candidate in selected}
            ),
            "slide_specific_eligible_median_cells": dict(medians),
            "metadata_rows_by_slide": {
                slide: metadata_counts[slide].row_count for slide in SLIDES
            },
            "core_map_slide_fov_keys_unique": not core_map.duplicated(
                ["slide", "fov"]
            ).any(),
            "metadata_slide_cell_keys_unique": True,
            "unmapped_metadata": _aggregate_unmapped_metadata(
                core_map,
                metadata_counts,
            ),
        },
        "routes": [route.to_public_dict() for route in routes],
        "cores": protected_records,
        "checksum": {
            "algorithm": "sha256",
            "scope": "canonical JSON with this checksum mapping omitted",
        },
    }
    digest_payload = dict(payload)
    digest_payload.pop("checksum")
    payload["checksum"]["value"] = canonical_sha256(digest_payload)
    return payload, tuple(routes)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _write_protected_json(
    output_path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
    forbidden_directories: Sequence[str | Path],
) -> None:
    output = Path(output_path)
    if output.suffix.lower() != ".json":
        raise AdjacentNormalSelectionError(
            "The protected selection output must use a .json suffix."
        )
    resolved_output = output.resolve(strict=False)
    for directory in forbidden_directories:
        resolved_directory = Path(directory).resolve(strict=False)
        if _is_within(resolved_output, resolved_directory):
            raise AdjacentNormalSelectionError(
                "Protected selection output cannot be written inside immutable "
                "raw or clinical input directories."
            )
    if output.exists() and not overwrite:
        raise FileExistsError(
            "Protected selection output already exists; use explicit overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            os.chmod(temporary_name, 0o600)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(output)
        os.chmod(output, 0o600)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def create_adjacent_normal_selection(
    *,
    legacy_workbook: str | Path,
    pathology_review_workbook: str | Path,
    core_map_csv: str | Path,
    raw_dir: str | Path,
    output_path: str | Path,
    chunksize: int = 100_000,
    overwrite: bool = False,
) -> AdjacentNormalSelectionReceipt:
    """Create one protected ten-core manifest and return safe routing only."""

    legacy_path = Path(legacy_workbook)
    review_path = Path(pathology_review_workbook)
    core_map_path = Path(core_map_csv)
    raw_path = Path(raw_dir)

    candidates = _load_legacy_candidates(legacy_path)
    _validate_pathology_review(review_path, candidates)
    core_map = _load_core_map(core_map_path)
    routing = _candidate_routing(candidates, core_map)

    metadata_paths = {
        slide: discover_slide_raw_path(raw_path, slide, "metadata")
        for slide in SLIDES
    }
    metadata_counts = {
        slide: _count_metadata_cells(
            metadata_paths[slide],
            slide=slide,
            chunksize=chunksize,
        )
        for slide in SLIDES
    }
    routed = _route_and_count_candidates(
        candidates,
        routing,
        metadata_counts,
    )
    selected, medians, ranks = _select_balanced(routed)
    provenance = [
        _input_provenance("legacy_clinical_workbook", legacy_path),
        _input_provenance("pathology_review_workbook", review_path),
        _input_provenance("fov_core_map", core_map_path),
        *[
            _input_provenance(f"{slide}_metadata", metadata_paths[slide])
            for slide in SLIDES
        ],
    ]
    payload, routes = _build_manifest(
        selected=selected,
        all_candidates=routed,
        medians=medians,
        ranks=ranks,
        core_map=core_map,
        metadata_counts=metadata_counts,
        provenance=provenance,
    )
    _write_protected_json(
        output_path,
        payload,
        overwrite=overwrite,
        forbidden_directories=(
            raw_path,
            legacy_path.parent,
            review_path.parent,
            core_map_path.parent,
        ),
    )
    return AdjacentNormalSelectionReceipt(routes=routes)


def _reject_json_constant(value: str) -> None:
    raise AdjacentNormalSelectionError(
        "The protected selection manifest contains a non-finite JSON constant."
    )


def _unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdjacentNormalSelectionError(
                "The protected selection manifest contains a duplicate JSON key."
            )
        result[key] = value
    return result


def _load_manifest_mapping(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except AdjacentNormalSelectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdjacentNormalSelectionError(
            "The protected selection manifest could not be read as strict JSON."
        ) from exc
    if not isinstance(value, dict):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest root must be a mapping."
        )
    return value


def _verify_manifest_checksum(payload: Mapping[str, Any]) -> None:
    checksum = payload.get("checksum")
    if not isinstance(checksum, Mapping):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest lacks its checksum mapping."
        )
    if (
        checksum.get("algorithm") != "sha256"
        or checksum.get("scope")
        != "canonical JSON with this checksum mapping omitted"
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest has an unsupported checksum contract."
        )
    observed = checksum.get("value")
    if not (
        isinstance(observed, str)
        and len(observed) == 64
        and all(character in "0123456789abcdef" for character in observed)
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest checksum is malformed."
        )
    digest_payload = dict(payload)
    digest_payload.pop("checksum", None)
    try:
        expected = canonical_sha256(digest_payload)
    except (TypeError, ValueError) as exc:
        raise AdjacentNormalSelectionError(
            "The protected selection manifest cannot be canonicalized."
        ) from exc
    if not hmac.compare_digest(observed, expected):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest checksum verification failed."
        )


def _validate_manifest_policy(payload: Mapping[str, Any]) -> None:
    policy = payload.get("selection_policy")
    confidentiality = payload.get("confidentiality")
    if not isinstance(policy, Mapping) or not isinstance(
        confidentiality,
        Mapping,
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest lacks its policy or confidentiality "
            "contract."
        )
    legacy = policy.get("legacy_label_mapping")
    review = policy.get("pathology_review")
    if not isinstance(legacy, Mapping) or not isinstance(review, Mapping):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest has an incomplete policy."
        )
    if (
        legacy.get("canonical_label") != "AdjacentNormal"
        or legacy.get("strict_exact_allow_list")
        != list(LEGACY_ADJACENT_LABELS)
        or legacy.get("expected_source_counts")
        != EXPECTED_LEGACY_LABEL_COUNTS
        or legacy.get("reject_other_adjacent_like_variants") is not True
        or review.get("diagnosis_exact") != REVIEW_DIAGNOSIS
        or review.get("result_exact") != REVIEW_RESULT
        or review.get("correction") != "blank"
        or policy.get("cell_count_source")
        != "all rows in slide metadata"
        or policy.get("minimum_cells_inclusive") != MINIMUM_CELLS
        or policy.get("selections_per_slide") != SELECTIONS_PER_SLIDE
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest disagrees with the locked "
            "adjacent-normal policy."
        )
    if (
        confidentiality.get("contains_protected_core_keys") is not True
        or confidentiality.get("contains_raw_donor_ids") is not False
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest has an invalid confidentiality "
            "declaration."
        )


def _route_from_mapping(
    value: object,
    *,
    require_exact_public_fields: bool,
) -> AdjacentNormalRoute:
    if not isinstance(value, Mapping):
        raise AdjacentNormalSelectionError(
            "A protected selection route is not a mapping."
        )
    if require_exact_public_fields and set(value) != {"alias", "slide", "fovs"}:
        raise AdjacentNormalSelectionError(
            "A public route contains fields outside the alias/slide/FOV contract."
        )
    alias = value.get("alias")
    slide = value.get("slide")
    fovs = value.get("fovs")
    if (
        not isinstance(alias, str)
        or not isinstance(slide, str)
        or not isinstance(fovs, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in fovs
        )
    ):
        raise AdjacentNormalSelectionError(
            "A protected selection route has invalid alias/slide/FOV types."
        )
    route = AdjacentNormalRoute(
        alias=alias,
        slide=slide,
        fovs=tuple(fovs),
    )
    if slide != route.slide or fovs != list(route.fovs):
        raise AdjacentNormalSelectionError(
            "A protected selection route is not canonically ordered."
        )
    return route


def _validate_manifest_routes(
    payload: Mapping[str, Any],
) -> tuple[AdjacentNormalRoute, ...]:
    public_records = payload.get("routes")
    protected_records = payload.get("cores")
    if not isinstance(public_records, list) or not isinstance(
        protected_records,
        list,
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest lacks route/core record lists."
        )
    if len(public_records) != SELECTION_COUNT or len(protected_records) != (
        SELECTION_COUNT
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest must contain exactly ten routes."
        )

    public_routes = tuple(
        _route_from_mapping(record, require_exact_public_fields=True)
        for record in public_records
    )
    expected_aliases = tuple(
        f"ANC-{index:02d}" for index in range(1, SELECTION_COUNT + 1)
    )
    if tuple(route.alias for route in public_routes) != expected_aliases:
        raise AdjacentNormalSelectionError(
            "The protected selection manifest does not contain the exact ten aliases."
        )

    protected_routes: list[AdjacentNormalRoute] = []
    protected_core_keys: list[int] = []
    expected_protected_record_keys = {
        "alias",
        "protected_core_key",
        "slide",
        "fovs",
        "n_fovs",
        "n_cells",
        "absolute_cell_count_distance_from_slide_median",
        "selection_rank_within_slide",
    }
    for record in protected_records:
        if not isinstance(record, Mapping):
            raise AdjacentNormalSelectionError(
                "A protected core record is not a mapping."
            )
        if set(record) != expected_protected_record_keys:
            raise AdjacentNormalSelectionError(
                "A protected core record does not match the exact protected schema."
            )
        protected_core_key = record.get("protected_core_key")
        if (
            isinstance(protected_core_key, bool)
            or not isinstance(protected_core_key, int)
            or protected_core_key <= 0
        ):
            raise AdjacentNormalSelectionError(
                "A protected core record has an invalid protected key."
            )
        protected_core_keys.append(protected_core_key)
        protected_routes.append(
            _route_from_mapping(
                record,
                require_exact_public_fields=False,
            )
        )
    if len(set(protected_core_keys)) != SELECTION_COUNT:
        raise AdjacentNormalSelectionError(
            "Protected core keys are not unique across the ten records."
        )
    if tuple(protected_routes) != public_routes:
        raise AdjacentNormalSelectionError(
            "Public routes disagree with their protected core records."
        )

    aggregate = payload.get("aggregate_validation")
    if not isinstance(aggregate, Mapping):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest lacks aggregate validation."
        )
    if (
        aggregate.get("selected_count") != SELECTION_COUNT
        or aggregate.get("selected_distinct_donor_count") != SELECTION_COUNT
        or aggregate.get("selected_count_by_slide")
        != {"SO_1": SELECTIONS_PER_SLIDE, "SO_2": SELECTIONS_PER_SLIDE}
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection aggregate does not certify ten balanced "
            "distinct-donor records."
        )
    if {
        slide: sum(route.slide == slide for route in public_routes)
        for slide in SLIDES
    } != {"SO_1": SELECTIONS_PER_SLIDE, "SO_2": SELECTIONS_PER_SLIDE}:
        raise AdjacentNormalSelectionError(
            "The protected selection routes are not balanced five per slide."
        )
    return public_routes


def load_adjacent_normal_route(
    manifest_path: str | Path,
    alias: str,
) -> AdjacentNormalRoute:
    """Verify a protected manifest and return one deidentified routing record."""

    payload = _load_manifest_mapping(manifest_path)
    _verify_manifest_checksum(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_kind")
        != "protected_adjacent_normal_core_selection"
        or payload.get("policy_id") != POLICY_ID
    ):
        raise AdjacentNormalSelectionError(
            "The protected selection manifest has an unsupported schema, "
            "artifact kind, or policy."
        )
    _validate_manifest_policy(payload)
    routes = _validate_manifest_routes(payload)
    if not isinstance(alias, str):
        raise AdjacentNormalSelectionError(
            "The requested adjacent-normal alias must be a string."
        )
    matches = [route for route in routes if route.alias == alias]
    if len(matches) != 1:
        raise AdjacentNormalSelectionError(
            "The requested adjacent-normal alias is absent or non-unique."
        )
    return matches[0]


__all__ = [
    "AdjacentNormalRoute",
    "AdjacentNormalSelectionError",
    "AdjacentNormalSelectionReceipt",
    "EXPECTED_LEGACY_LABEL_COUNTS",
    "LEGACY_ADJACENT_LABELS",
    "MINIMUM_CELLS",
    "POLICY_ID",
    "SELECTIONS_PER_SLIDE",
    "create_adjacent_normal_selection",
    "load_adjacent_normal_route",
]

#!/usr/bin/env python3
"""Analyze the frozen matched graph-context nested-CV confirmation.

Geometry components, not cells, are the analysis units.  Seeds are averaged
inside component before the deterministic slide-stratified component
bootstrap.  The resulting intervals are descriptive because the 27 geometry
components are nested in only two already-observed slides.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260812_matched_graph_context_nested_cv"
CONTRACT_RELATIVE = (
    "experiments/campaigns/cmp_20260812_matched_graph_context_nested_cv/"
    "frozen_task_contract.yaml"
)
CONTRACT_SHA256 = (
    "4e39e1ef623a5ce4e1d67f4733bdea790000745d523a8845bcd11a0ddea77b73"
)
SELECTION_KIND = "matched_graph_context_selection_receipt"
ARMS = ("no_graph", "observed_near", "permuted_near", "observed_annular")
CANDIDATES: Mapping[str, Mapping[str, float | int | str]] = {
    candidate_id: {
        "candidate_id": candidate_id,
        "hidden_width": hidden,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
    }
    for candidate_id, hidden, learning_rate, weight_decay, dropout in (
        ("c00", 32, 0.0003, 0.0, 0.0),
        ("c01", 32, 0.0003, 0.001, 0.2),
        ("c02", 32, 0.001, 0.0001, 0.1),
        ("c03", 32, 0.003, 0.0, 0.2),
        ("c04", 32, 0.003, 0.001, 0.0),
        ("c05", 64, 0.0003, 0.0, 0.1),
        ("c06", 64, 0.0003, 0.001, 0.0),
        ("c07", 64, 0.001, 0.0, 0.2),
        ("c08", 64, 0.001, 0.0001, 0.1),
        ("c09", 64, 0.003, 0.0001, 0.0),
        ("c10", 64, 0.003, 0.001, 0.2),
        ("c11", 128, 0.0003, 0.0001, 0.2),
        ("c12", 128, 0.0003, 0.001, 0.0),
        ("c13", 128, 0.001, 0.0, 0.0),
        ("c14", 128, 0.001, 0.001, 0.1),
        ("c15", 128, 0.003, 0.0001, 0.1),
    )
}
CANDIDATE_DESIGN_SHA256 = "c495ccc2cb5cdbf98333495d17bd9608b00d749d57d2d8154cb20a3fe61243c6"
SEEDS = (20260812, 20261812, 20262812, 20263812, 20264812)
FOLDS = (0, 1, 2, 3)
SLIDES = ("SO_1", "SO_2")
EXPECTED_COMPONENTS = 27
EXPECTED_GENES = 1000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812
GRAPH_GAIN_MINIMUM = 0.02
PERMUTATION_GAIN_MINIMUM = 0.01
MINIMUM_FAVORING_COMPONENTS = 20
MINIMUM_FAVORING_SEEDS = 4
SUBSTITUTIONS = (
    "native",
    "zero",
    "permuted_near",
    "observed_annular",
    "observed_near_10_25",
)
TARGET_PROGRAMS: Mapping[str, tuple[str, ...]] = {
    "tls_immune": (
        "CXCL13", "CCL19", "CCL21", "LTB", "MS4A1", "CD79A", "CD74",
        "HLA-DRA", "CD3D", "CD3E", "CXCR5", "CCR7",
    ),
    "myeloid": (
        "LYZ", "FCER1G", "TYROBP", "C1QA", "C1QB", "C1QC", "APOE",
        "SPP1", "IL1B", "CXCL8",
    ),
    "stromal_ecm": (
        "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "DCN", "LUM",
        "COL4A1", "COL4A2", "FN1", "FAP", "PDGFRA",
    ),
    "epithelial": (
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "KRT17", "CEACAM6",
        "KRT20", "TACSTD2",
    ),
    "endothelial": ("PECAM1", "VWF", "KDR", "ENG", "RAMP2", "ESAM", "RGCC"),
}
V0_V6_VARIANTS = tuple(f"V{index}" for index in range(7))
DEFAULT_PRIOR_AGGREGATE = Path(
    "reports/analyses/same_gene_robustness_20260811/aggregate_results.json"
)
DEFAULT_PRIOR_MANIFEST = Path(
    "reports/analyses/same_gene_robustness_20260811/analysis_manifest.json"
)
DEFAULT_PRIOR_SUCCESS = Path(
    "reports/analyses/same_gene_robustness_20260811/_SUCCESS"
)
DEFAULT_REPORT_ROOT = Path(
    "reports/analyses/matched_graph_context_nested_cv"
)


class MatchedGraphContextAnalysisError(RuntimeError):
    """Raised before any report is emitted when evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class ConfirmationRun:
    root: Path
    result_path: Path
    result_file_sha256: str
    result: Mapping[str, Any]
    arm: str
    seed: int
    fold: int
    component_rows: tuple[dict[str, Any], ...]
    gene_rows: tuple[dict[str, Any], ...]
    component_gene_rows: tuple[dict[str, Any], ...]
    substitution_rows: tuple[dict[str, Any], ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.arm, self.seed, self.fold


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedGraphContextAnalysisError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise MatchedGraphContextAnalysisError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatchedGraphContextAnalysisError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except MatchedGraphContextAnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatchedGraphContextAnalysisError(f"cannot read {label}: {path}") from error
    return dict(_mapping(value, label))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MatchedGraphContextAnalysisError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _hex(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MatchedGraphContextAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatchedGraphContextAnalysisError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise MatchedGraphContextAnalysisError(f"{label} must be {qualifier}")
    return number


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchedGraphContextAnalysisError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise MatchedGraphContextAnalysisError(f"{label} must be >= {minimum}")
    return int(value)


def _project_path(reference: object, label: str, *, project_root: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise MatchedGraphContextAnalysisError(f"{label} path is missing")
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project_root.resolve())
    except (OSError, ValueError) as error:
        raise MatchedGraphContextAnalysisError(
            f"{label} must resolve beneath the project root"
        ) from error
    return resolved


def _bound_file(
    value: object,
    label: str,
    *,
    project_root: Path,
    default_name: str | None = None,
) -> Path:
    block = _mapping(value, label)
    reference = block.get("path", block.get("reference"))
    if reference is None and default_name is not None:
        reference = default_name
    path = _project_path(reference, label, project_root=project_root)
    expected = _hex(block.get("sha256"), f"{label} sha256")
    if _sha256_file(path) != expected:
        raise MatchedGraphContextAnalysisError(f"{label} checksum mismatch")
    return path


def _contains_forbidden_test_key(value: object) -> bool:
    """Return whether validation-only evidence contains any test-labeled field."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.strip().lower().startswith("test"):
                return True
            if _contains_forbidden_test_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_test_key(item) for item in value)
    return False


def _verify_contract(*, project_root: Path) -> Mapping[str, Any]:
    path = project_root / CONTRACT_RELATIVE
    if _sha256_file(path) != CONTRACT_SHA256:
        raise MatchedGraphContextAnalysisError("frozen task contract checksum changed")
    contract = load_yaml_mapping(path)
    expected_programs = {key: list(value) for key, value in TARGET_PROGRAMS.items()}
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign_id") != CAMPAIGN_ID
        or _mapping(contract.get("model"), "contract model").get("arms") != list(ARMS)
        or _mapping(contract.get("confirmation"), "contract confirmation").get("seeds")
        != list(SEEDS)
        or _mapping(contract.get("confirmation"), "contract confirmation").get("folds")
        != list(FOLDS)
        or contract.get("faithfulness_substitutions") != list(SUBSTITUTIONS[1:])
        or contract.get("target_programs") != expected_programs
    ):
        raise MatchedGraphContextAnalysisError("frozen task contract semantics changed")
    return contract


def _verify_selection_receipt(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    receipt = _strict_json(path, label="selection receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind", receipt.get("receipt_kind")) != SELECTION_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("test_metrics_used_for_selection") is not False
    ):
        raise MatchedGraphContextAnalysisError(
            "selection receipt differs from the frozen selection contract"
        )
    digest_field = "payload_sha256"
    expected_digest = _hex(receipt.get(digest_field), "selection receipt payload digest")
    unsigned = dict(receipt)
    unsigned.pop(digest_field, None)
    if canonical_sha256(unsigned) != expected_digest:
        raise MatchedGraphContextAnalysisError("selection receipt self-digest mismatch")

    if receipt.get("contract_sha256") != CONTRACT_SHA256:
        raise MatchedGraphContextAnalysisError("selection receipt contract binding changed")

    if receipt.get("candidate_design_sha256") != CANDIDATE_DESIGN_SHA256:
        raise MatchedGraphContextAnalysisError("selection candidate design changed")

    if receipt.get("status") != "frozen" or receipt.get("cross_outer_pooling") is not False:
        raise MatchedGraphContextAnalysisError(
            "selection receipt is not frozen independent nested-CV authority"
        )
    for name in ("source_stage_a_plan", "source_stage_a_selection", "source_stage_b_plan"):
        if name not in receipt:
            raise MatchedGraphContextAnalysisError(f"selection receipt omits {name}")
        _bound_file(receipt[name], name, project_root=project_root)
    source_groups = _mapping(
        receipt.get("source_tuning_results_by_outer_fold"),
        "source_tuning_results_by_outer_fold",
    )
    source_hash_groups = _mapping(
        receipt.get("source_tuning_result_sha256s_by_outer_fold"),
        "source_tuning_result_sha256s_by_outer_fold",
    )
    expected_folds = {str(fold) for fold in FOLDS}
    if set(source_groups) != expected_folds or set(source_hash_groups) != expected_folds:
        raise MatchedGraphContextAnalysisError(
            "selection receipt lacks four fold-local tuning source groups"
        )
    for fold in FOLDS:
        rows = source_groups[str(fold)]
        hashes = source_hash_groups[str(fold)]
        if (
            not isinstance(rows, list)
            or len(rows) != 88
            or not isinstance(hashes, list)
            or len(hashes) != 88
        ):
            raise MatchedGraphContextAnalysisError(
                f"selection outer fold {fold} must bind exactly 88 tuning results"
            )
        observed_hashes: list[str] = []
        observed_job_ids: set[str] = set()
        observed_paths: set[Path] = set()
        for index, value in enumerate(rows):
            path = _bound_file(
                value,
                f"tuning result outer fold {fold}/{index}",
                project_root=project_root,
            )
            block = _mapping(value, f"tuning result outer fold {fold}/{index}")
            job_id = block.get("job_id")
            if (
                not isinstance(job_id, str)
                or not job_id.endswith(f".f{fold}")
                or job_id in observed_job_ids
                or path in observed_paths
            ):
                raise MatchedGraphContextAnalysisError(
                    f"tuning result outer fold {fold}/{index} identity is invalid"
                )
            observed_job_ids.add(job_id)
            observed_paths.add(path)
            observed_hashes.append(str(block["sha256"]))
            tuning = _strict_json(path, label=f"tuning result outer fold {fold}/{index}")
            if (
                tuning.get("schema_version") != 1
                or tuning.get("campaign_id") != CAMPAIGN_ID
                or tuning.get("contract_sha256") != CONTRACT_SHA256
                or tuning.get("mode") != "tune"
                or tuning.get("fold") != fold
                or tuning.get("status") != "success"
                or tuning.get("finite_metrics") is not True
                or tuning.get("coverage_complete") is not True
                or _contains_forbidden_test_key(tuning)
            ):
                raise MatchedGraphContextAnalysisError(
                    f"tuning result outer fold {fold}/{index} is not validation-only"
                )
            marker_reference = block.get("success_marker")
            marker_sha = block.get("success_marker_sha256")
            if marker_reference is None or marker_sha is None:
                raise MatchedGraphContextAnalysisError(
                    f"tuning result outer fold {fold}/{index} lacks success marker binding"
                )
            marker = _project_path(
                marker_reference,
                f"tuning success marker outer fold {fold}/{index}",
                project_root=project_root,
            )
            if _sha256_file(marker) != _hex(
                marker_sha, f"tuning success marker outer fold {fold}/{index} sha"
            ):
                raise MatchedGraphContextAnalysisError(
                    f"tuning success marker checksum mismatch: {path}"
                )
        if sorted(observed_hashes) != sorted(
            _hex(value, f"outer fold {fold} tuning result SHA") for value in hashes
        ):
            raise MatchedGraphContextAnalysisError(
                f"outer fold {fold} tuning hash list is inconsistent"
            )

    selected_by_fold = _mapping(
        receipt.get("selected_by_outer_fold"), "selected_by_outer_fold"
    )
    if set(selected_by_fold) != {str(fold) for fold in FOLDS}:
        raise MatchedGraphContextAnalysisError(
            "selection must contain exactly four independent outer folds"
        )
    for fold in FOLDS:
        selected = _mapping(
            selected_by_fold[str(fold)], f"selected_by_outer_fold {fold}"
        )
        if set(selected) != set(ARMS):
            raise MatchedGraphContextAnalysisError(
                f"outer fold {fold} selection must contain exactly four arms"
            )
        hidden_widths: set[int] = set()
        parameter_counts: set[int] = set()
        for arm in ARMS:
            block = _mapping(selected[arm], f"selection fold {fold}/{arm}")
            candidate = block.get("candidate_id")
            if not isinstance(candidate, str) or candidate not in CANDIDATES:
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} candidate_id is missing"
                )
            _hex(
                block.get("config_sha256"),
                f"selection fold {fold}/{arm} config sha256",
            )
            config = _mapping(
                block.get("config", block.get("hyperparameters")),
                f"selection fold {fold}/{arm} config",
            )
            if set(config) != {
                "candidate_id",
                "hidden_width",
                "learning_rate",
                "weight_decay",
                "dropout",
                "epoch",
                "batch_size",
            } or config.get("candidate_id") != candidate:
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} exact config schema changed"
                )
            if canonical_sha256(config) != block.get("config_sha256"):
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} config checksum mismatch"
                )
            frozen_candidate = CANDIDATES[candidate]
            if any(config.get(key) != value for key, value in frozen_candidate.items()):
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} differs from frozen candidate"
                )
            hidden_widths.add(
                _integer(
                    config.get("hidden_width", config.get("hidden")),
                    f"selection fold {fold}/{arm} hidden width",
                    minimum=1,
                )
            )
            _finite(
                config.get("learning_rate"),
                f"selection fold {fold}/{arm} learning rate",
            )
            _finite(
                config.get("weight_decay"),
                f"selection fold {fold}/{arm} weight decay",
                nonnegative=True,
            )
            _finite(
                config.get("dropout"),
                f"selection fold {fold}/{arm} dropout",
                nonnegative=True,
            )
            _integer(
                config.get("epoch", config.get("epochs")),
                f"selection fold {fold}/{arm} epochs",
                minimum=1,
            )
            if config.get("epoch") not in (12, 24, 48, 96, 192):
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} epoch is outside frozen trajectory"
                )
            if config.get("batch_size") != 4096:
                raise MatchedGraphContextAnalysisError(
                    f"selection fold {fold}/{arm} batch size changed"
                )
            count = block.get("parameter_count", config.get("parameter_count"))
            parameter_counts.add(
                _integer(
                    count,
                    f"selection fold {fold}/{arm} parameter_count",
                    minimum=1,
                )
            )
        if len(hidden_widths) != 1 or len(parameter_counts) != 1:
            raise MatchedGraphContextAnalysisError(
                f"selected arms in outer fold {fold} do not share hidden width "
                "and exact parameter count"
            )
        hidden_width = next(iter(hidden_widths))
        if parameter_counts != {1_028_000 + 2_001 * hidden_width}:
            raise MatchedGraphContextAnalysisError(
                f"selected arms in outer fold {fold} have invalid parameter count"
            )
    return receipt


def _read_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatchedGraphContextAnalysisError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(
                            line,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                MatchedGraphContextAnalysisError(
                                    f"{label} contains nonfinite {value}"
                                )
                            ),
                            object_pairs_hook=unique,
                        )
                    except (json.JSONDecodeError, UnicodeError) as error:
                        raise MatchedGraphContextAnalysisError(
                            f"invalid {label} JSONL line {line_number}"
                        ) from error
                    rows.append(dict(_mapping(value, f"{label} row {line_number}")))
        except OSError as error:
            raise MatchedGraphContextAnalysisError(f"cannot read {label}") from error
        return rows
    if suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except (OSError, UnicodeError, csv.Error) as error:
            raise MatchedGraphContextAnalysisError(f"cannot read {label}") from error
    if suffix == ".json":
        value = _strict_json(path, label=label)
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise MatchedGraphContextAnalysisError(f"{label} JSON must contain rows")
        return [dict(_mapping(row, f"{label} row")) for row in rows]
    if suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        except Exception as error:
            raise MatchedGraphContextAnalysisError(f"cannot read {label} parquet") from error
    raise MatchedGraphContextAnalysisError(f"unsupported {label} format: {path.suffix}")


def _output_file(
    outputs: Mapping[str, Any],
    name: str,
    *,
    root: Path,
    project_root: Path,
) -> Path:
    value = outputs.get(name)
    if isinstance(value, str):
        raise MatchedGraphContextAnalysisError(
            f"output {name} must include path and SHA-256, not a bare path"
        )
    block = _mapping(value, f"output {name}")
    reference = block.get("path", block.get("reference"))
    if not isinstance(reference, str):
        raise MatchedGraphContextAnalysisError(f"output {name} path is missing")
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root.resolve())
        path.relative_to(project_root.resolve())
    except (OSError, ValueError) as error:
        raise MatchedGraphContextAnalysisError(
            f"output {name} must resolve inside its immutable bundle"
        ) from error
    expected = _hex(block.get("sha256"), f"output {name} sha256")
    if _sha256_file(path) != expected:
        raise MatchedGraphContextAnalysisError(f"output {name} checksum mismatch")
    return path


def _string_field(row: Mapping[str, Any], name: str, label: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise MatchedGraphContextAnalysisError(f"{label} {name} is missing")
    return value


def _coerce_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as error:
            raise MatchedGraphContextAnalysisError(f"{label} must be integer") from error
        value = parsed
    return _integer(value, label, minimum=minimum)


def _coerce_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as error:
            raise MatchedGraphContextAnalysisError(f"{label} must be numeric") from error
        value = parsed
    return _finite(value, label, nonnegative=nonnegative)


def _expected_component_map(
    prepared_root: Path,
) -> tuple[dict[tuple[str, int], int], dict[int, set[tuple[str, int]]]]:
    component_cells: dict[tuple[str, int], int] = {}
    by_fold: dict[int, set[tuple[str, int]]] = {fold: set() for fold in FOLDS}
    for slide in SLIDES:
        try:
            groups = np.load(prepared_root / slide / "geometry_group.npy", allow_pickle=False)
            folds = np.load(prepared_root / slide / "fold.npy", allow_pickle=False)
            eligible = np.load(prepared_root / slide / "eligible_primary.npy", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise MatchedGraphContextAnalysisError(
                f"cannot load prepared component authority for {slide}"
            ) from error
        if groups.shape != folds.shape or groups.shape != eligible.shape or groups.ndim != 1:
            raise MatchedGraphContextAnalysisError(f"prepared arrays misaligned for {slide}")
        for raw_component in np.unique(groups):
            component = int(raw_component)
            mask = groups == raw_component
            component_folds = np.unique(folds[mask])
            if component_folds.shape != (1,) or int(component_folds[0]) not in FOLDS:
                raise MatchedGraphContextAnalysisError(
                    f"prepared component fold invalid: {slide}/{component}"
                )
            key = (slide, component)
            cells = int(np.count_nonzero(eligible[mask]))
            if cells <= 0 or key in component_cells:
                raise MatchedGraphContextAnalysisError(
                    f"prepared component eligibility invalid: {slide}/{component}"
                )
            component_cells[key] = cells
            by_fold[int(component_folds[0])].add(key)
    if len(component_cells) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError(
            f"prepared component count is {len(component_cells)}, expected {EXPECTED_COMPONENTS}"
        )
    return component_cells, by_fold


def _component_heterogeneity(
    prepared_root: Path,
    graph_vs_no_graph: Mapping[str, Any],
) -> dict[str, Any]:
    covariates: dict[tuple[str, int], dict[str, Any]] = {}
    for slide in SLIDES:
        try:
            groups = np.load(prepared_root / slide / "geometry_group.npy", allow_pickle=False)
            eligible = np.load(prepared_root / slide / "eligible_primary.npy", allow_pickle=False).astype(bool)
            degree = np.load(prepared_root / slide / "near_degree.npy", allow_pickle=False)
            qc_passed = np.load(prepared_root / slide / "qc_passed.npy", allow_pickle=False).astype(bool)
        except (OSError, ValueError) as error:
            raise MatchedGraphContextAnalysisError(
                f"cannot load heterogeneity covariates for {slide}"
            ) from error
        if not (groups.shape == eligible.shape == degree.shape == qc_passed.shape) or groups.ndim != 1:
            raise MatchedGraphContextAnalysisError(
                f"heterogeneity covariate arrays are misaligned for {slide}"
            )
        if not np.isfinite(degree).all() or np.any(degree < 0):
            raise MatchedGraphContextAnalysisError(f"near degree is invalid for {slide}")
        for raw_component in np.unique(groups):
            component = int(raw_component)
            member = groups == raw_component
            receiver = member & eligible
            if not np.any(receiver):
                raise MatchedGraphContextAnalysisError(
                    f"heterogeneity component has no eligible receiver: {slide}/{component}"
                )
            covariates[(slide, component)] = {
                "slide": slide,
                "component": component,
                "eligible_receiver_count": int(np.count_nonzero(receiver)),
                "mean_near_degree_among_eligible_receivers": float(np.mean(degree[receiver])),
                "vendor_qc_pass_fraction_all_prepared_cells": float(np.mean(qc_passed[member])),
                "vendor_qc_fraction_numerator": int(np.count_nonzero(qc_passed[member])),
                "vendor_qc_fraction_denominator": int(np.count_nonzero(member)),
            }
    effects = graph_vs_no_graph.get("component_effects")
    if not isinstance(effects, list) or len(effects) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError("heterogeneity effect axis is incomplete")
    effect_by_component: dict[tuple[str, int], float] = {}
    for row in effects:
        block = _mapping(row, "heterogeneity component effect")
        key = (str(block.get("slide")), int(block.get("component")))
        if key in effect_by_component:
            raise MatchedGraphContextAnalysisError("heterogeneity effect component is duplicated")
        effect_by_component[key] = _finite(
            block.get("difference"), f"heterogeneity effect {key}"
        )
    if set(effect_by_component) != set(covariates) or len(covariates) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError("heterogeneity component join is incomplete")
    table = []
    for key in sorted(covariates):
        table.append(
            {
                **covariates[key],
                "no_graph_minus_observed_near_mse": effect_by_component[key],
            }
        )
    from scipy.stats import spearmanr

    correlations: dict[str, Any] = {}
    for field in (
        "eligible_receiver_count",
        "mean_near_degree_among_eligible_receivers",
        "vendor_qc_pass_fraction_all_prepared_cells",
    ):
        x = np.asarray([float(row[field]) for row in table], dtype=np.float64)
        y = np.asarray(
            [float(row["no_graph_minus_observed_near_mse"]) for row in table],
            dtype=np.float64,
        )
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            correlations[field] = {
                "status": "undefined_constant_vector",
                "spearman_rho": None,
                "two_sided_p_value": None,
            }
        else:
            statistic = spearmanr(x, y)
            correlations[field] = {
                "status": "defined",
                "spearman_rho": _finite(statistic.statistic, f"Spearman {field}"),
                "two_sided_p_value": _finite(statistic.pvalue, f"Spearman p {field}"),
            }
    slide_summary = {
        slide: {
            "component_count": int(sum(row["slide"] == slide for row in table)),
            "mean_no_graph_minus_observed_near_mse": float(
                np.mean(
                    [
                        row["no_graph_minus_observed_near_mse"]
                        for row in table
                        if row["slide"] == slide
                    ]
                )
            ),
        }
        for slide in SLIDES
    }
    return {
        "effect_definition": "seed-averaged no_graph minus observed_near component MSE",
        "vendor_qc_fraction_denominator": (
            "all prepared cells assigned to the geometry component, before primary-receiver eligibility"
        ),
        "slide_summary": slide_summary,
        "descriptive_spearman": correlations,
        "components": table,
        "interpretation": (
            "descriptive heterogeneity screen only; associations are not independent "
            "patient evidence and are not causal covariate effects"
        ),
    }


def _normalize_component_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    fold: int,
    run_id: str,
    expected: set[tuple[str, int]],
    expected_cells: Mapping[tuple[str, int], int],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        label = f"component row {arm}/{seed}/{fold}/{index}"
        row_arm = _string_field(row, "arm", label)
        row_seed = _coerce_int(row.get("seed"), f"{label} seed")
        row_fold = _coerce_int(row.get("fold"), f"{label} fold")
        row_run = _string_field(row, "run_id", label)
        slide = _string_field(row, "slide", label)
        component = _coerce_int(row.get("component"), f"{label} component")
        context = str(row.get("context_variant", "native"))
        key = (slide, component)
        if (
            row_arm != arm
            or row_seed != seed
            or row_fold != fold
            or row_run != run_id
            or context != "native"
            or key in seen
        ):
            raise MatchedGraphContextAnalysisError(f"{label} identity is invalid")
        cells = _coerce_int(row.get("n_cells"), f"{label} n_cells", minimum=1)
        if key not in expected or cells != expected_cells[key]:
            raise MatchedGraphContextAnalysisError(f"{label} component coverage changed")
        seen.add(key)
        normalized.append(
            {
                "run_id": run_id,
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "slide": slide,
                "component": component,
                "n_cells": cells,
                "mse": _coerce_float(row.get("mse"), f"{label} mse", nonnegative=True),
                "mae": _coerce_float(row.get("mae"), f"{label} mae", nonnegative=True),
            }
        )
    if seen != expected:
        raise MatchedGraphContextAnalysisError(
            f"component coverage incomplete for {arm}/{seed}/{fold}"
        )
    return tuple(sorted(normalized, key=lambda row: (row["slide"], row["component"])))


def _normalize_gene_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    fold: int,
    run_id: str,
    genes: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row_number, row in enumerate(rows):
        label = f"gene row {arm}/{seed}/{fold}/{row_number}"
        if (
            _string_field(row, "arm", label) != arm
            or _coerce_int(row.get("seed"), f"{label} seed") != seed
            or _coerce_int(row.get("fold"), f"{label} fold") != fold
            or _string_field(row, "run_id", label) != run_id
        ):
            raise MatchedGraphContextAnalysisError(f"{label} identity is invalid")
        index = _coerce_int(row.get("gene_index"), f"{label} gene_index", minimum=0)
        gene = _string_field(row, "gene", label)
        if index >= len(genes) or gene != genes[index] or index in seen:
            raise MatchedGraphContextAnalysisError(f"{label} gene axis changed")
        seen.add(index)
        normalized.append(
            {
                "run_id": run_id,
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "gene_index": index,
                "gene": gene,
                "mse": _coerce_float(row.get("mse"), f"{label} mse", nonnegative=True),
                "mae": _coerce_float(row.get("mae"), f"{label} mae", nonnegative=True),
                "pearson": _coerce_float(row.get("pearson"), f"{label} pearson"),
                "n_cells": _coerce_int(row.get("n_cells"), f"{label} n_cells", minimum=1),
            }
        )
    if seen != set(range(len(genes))) or len(normalized) != EXPECTED_GENES:
        raise MatchedGraphContextAnalysisError(
            f"gene coverage incomplete for {arm}/{seed}/{fold}"
        )
    return tuple(sorted(normalized, key=lambda row: row["gene_index"]))


def _normalize_component_gene_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    fold: int,
    run_id: str,
    genes: Sequence[str],
    expected: set[tuple[str, int]],
    expected_cells: Mapping[tuple[str, int], int],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row_number, row in enumerate(rows):
        label = f"component-gene row {arm}/{seed}/{fold}/{row_number}"
        if (
            _string_field(row, "arm", label) != arm
            or _coerce_int(row.get("seed"), f"{label} seed") != seed
            or _coerce_int(row.get("fold"), f"{label} fold") != fold
            or _string_field(row, "run_id", label) != run_id
        ):
            raise MatchedGraphContextAnalysisError(f"{label} identity is invalid")
        slide = _string_field(row, "slide", label)
        component = _coerce_int(row.get("component"), f"{label} component")
        index = _coerce_int(row.get("gene_index"), f"{label} gene_index", minimum=0)
        gene = _string_field(row, "gene", label)
        component_key = (slide, component)
        slot = (*component_key, index)
        if (
            component_key not in expected
            or index >= len(genes)
            or gene != genes[index]
            or slot in seen
        ):
            raise MatchedGraphContextAnalysisError(f"{label} axis is invalid")
        cells = _coerce_int(row.get("n_cells"), f"{label} n_cells", minimum=1)
        if cells != expected_cells[component_key]:
            raise MatchedGraphContextAnalysisError(f"{label} cell coverage changed")
        seen.add(slot)
        normalized.append(
            {
                "run_id": run_id,
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "slide": slide,
                "component": component,
                "n_cells": cells,
                "gene_index": index,
                "gene": gene,
                "mse": _coerce_float(row.get("mse"), f"{label} mse", nonnegative=True),
                "mae": _coerce_float(row.get("mae"), f"{label} mae", nonnegative=True),
            }
        )
    expected_slots = {
        (*component, gene_index)
        for component in expected
        for gene_index in range(len(genes))
    }
    if seen != expected_slots:
        raise MatchedGraphContextAnalysisError(
            f"component-gene coverage incomplete for {arm}/{seed}/{fold}"
        )
    return tuple(
        sorted(
            normalized,
            key=lambda row: (row["slide"], row["component"], row["gene_index"]),
        )
    )


def _normalize_substitution_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fold: int,
    run_id: str,
    expected: set[tuple[str, int]],
    expected_cells: Mapping[tuple[str, int], int],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, row in enumerate(rows):
        label = f"substitution row {seed}/{fold}/{index}"
        if (
            _string_field(row, "arm", label) != "observed_near"
            or _coerce_int(row.get("seed"), f"{label} seed") != seed
            or _coerce_int(row.get("fold"), f"{label} fold") != fold
            or _string_field(row, "run_id", label) != run_id
        ):
            raise MatchedGraphContextAnalysisError(f"{label} identity is invalid")
        slide = _string_field(row, "slide", label)
        component = _coerce_int(row.get("component"), f"{label} component")
        context = _string_field(row, "context_variant", label)
        key = (slide, component)
        slot = (*key, context)
        cells = _coerce_int(row.get("n_cells"), f"{label} n_cells", minimum=1)
        if (
            key not in expected
            or context not in SUBSTITUTIONS
            or slot in seen
            or cells != expected_cells[key]
        ):
            raise MatchedGraphContextAnalysisError(f"{label} coverage is invalid")
        seen.add(slot)
        normalized.append(
            {
                "run_id": run_id,
                "arm": "observed_near",
                "seed": seed,
                "fold": fold,
                "slide": slide,
                "component": component,
                "n_cells": cells,
                "context_variant": context,
                "mse": _coerce_float(row.get("mse"), f"{label} mse", nonnegative=True),
                "mae": _coerce_float(row.get("mae"), f"{label} mae", nonnegative=True),
            }
        )
    expected_slots = {(*key, context) for key in expected for context in SUBSTITUTIONS}
    if seen != expected_slots:
        raise MatchedGraphContextAnalysisError(
            f"substitution coverage incomplete for observed_near/{seed}/{fold}"
        )
    return tuple(
        sorted(
            normalized,
            key=lambda row: (row["slide"], row["component"], row["context_variant"]),
        )
    )


def _result_binding(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("selection_receipt")
    if not isinstance(value, Mapping):
        value = _mapping(
            _mapping(result.get("input"), "result input").get("selection_receipt"),
            "result selection_receipt",
        )
    return value


def _selected_receipt_block(
    receipt: Mapping[str, Any], *, fold: int, arm: str
) -> Mapping[str, Any]:
    by_fold = _mapping(
        receipt.get("selected_by_outer_fold"), "selected_by_outer_fold"
    )
    selected = _mapping(by_fold.get(str(fold)), f"selection outer fold {fold}")
    return _mapping(selected.get(arm), f"selection outer fold {fold}/{arm}")


def _load_confirmation_run(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    receipt_file_sha256: str,
    receipt_payload_sha256: str,
    expected_components: Mapping[int, set[tuple[str, int]]],
    component_cells: Mapping[tuple[str, int], int],
    genes: Sequence[str],
    project_root: Path,
) -> ConfirmationRun:
    try:
        verification = verify_run_bundle(root, require_success_contract=False)
    except Exception as error:
        raise MatchedGraphContextAnalysisError(f"bundle verification failed: {root}") from error
    if verification.get("status") != "success":
        raise MatchedGraphContextAnalysisError(f"confirmation bundle is not successful: {root}")
    result_path = root / "results.json"
    result = _strict_json(result_path, label=f"confirmation result {root.name}")
    arm = str(result.get("arm"))
    seed = _integer(result.get("seed"), f"{root.name} seed")
    fold = _integer(result.get("fold"), f"{root.name} fold")
    run_id = result.get("run_id")
    if (
        result.get("schema_version") != 1
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("contract_sha256") != CONTRACT_SHA256
        or result.get("mode") != "confirm"
        or result.get("status") != "success"
        or result.get("finite_metrics") is not True
        or result.get("coverage_complete") is not True
        or arm not in ARMS
        or seed not in SEEDS
        or fold not in FOLDS
        or run_id != root.name
    ):
        raise MatchedGraphContextAnalysisError(
            f"confirmation result identity/status invalid: {root}"
        )
    binding = _result_binding(result)
    bound_reference = _project_path(
        binding.get("path", binding.get("reference")),
        f"{root.name} selection binding",
        project_root=project_root,
    )
    if (
        bound_reference != receipt_path.resolve()
        or binding.get("sha256") != receipt_file_sha256
        or binding.get("payload_sha256") != receipt_payload_sha256
    ):
        raise MatchedGraphContextAnalysisError(
            f"confirmation selection binding changed: {root.name}"
        )
    inputs = _mapping(result.get("input"), f"{root.name} input")
    if (
        inputs.get("processed_fingerprint")
        != "01c525695883784befc1b9ebbe37a6d96b248d5b18450f46a1255e571bd3819e"
        or inputs.get("split_fingerprint")
        != "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"
    ):
        raise MatchedGraphContextAnalysisError(f"confirmation input hash changed: {root.name}")
    _hex(inputs.get("prepared_manifest_sha256"), f"{root.name} manifest sha")
    _hex(inputs.get("integrity_manifest_sha256"), f"{root.name} integrity sha")
    result_config_sha = _hex(result.get("config_sha256"), f"{root.name} config sha")
    selected = _selected_receipt_block(receipt, fold=fold, arm=arm)
    selected_config = _mapping(
        selected.get("config", selected.get("hyperparameters")),
        f"selection outer fold {fold}/{arm} config",
    )
    result_config = _mapping(result.get("config"), f"{root.name} config")
    if (
        result_config_sha != selected.get("config_sha256")
        or canonical_sha256(result_config) != result_config_sha
        or result_config != selected_config
        or result.get("candidate_id") != selected.get("candidate_id")
        or result.get("parameter_count")
        != selected.get("parameter_count", selected_config.get("parameter_count"))
    ):
        raise MatchedGraphContextAnalysisError(
            f"confirmation config does not match outer-specific selection: {root.name}"
        )
    _hex(result.get("normalization_sha256"), f"{root.name} normalization sha")
    _hex(result.get("projection_sha256"), f"{root.name} projection sha")
    _integer(result.get("parameter_count"), f"{root.name} parameter_count", minimum=1)
    roles = _mapping(result.get("split_roles"), f"{root.name} split roles")
    if roles.get("test_fold") != fold or roles.get("validation_fold") is not None:
        raise MatchedGraphContextAnalysisError(f"confirmation split roles invalid: {root.name}")
    train_folds = roles.get("train_folds")
    if train_folds != [value for value in FOLDS if value != fold]:
        raise MatchedGraphContextAnalysisError(f"confirmation train folds invalid: {root.name}")

    outputs = _mapping(result.get("outputs"), f"{root.name} outputs")
    component_path = _output_file(
        outputs, "component_metrics", root=root, project_root=project_root
    )
    gene_path = _output_file(outputs, "gene_metrics", root=root, project_root=project_root)
    component_gene_path = _output_file(
        outputs, "component_gene_metrics", root=root, project_root=project_root
    )
    _output_file(outputs, "checkpoint", root=root, project_root=project_root)
    component_rows = _normalize_component_rows(
        _read_records(component_path, label=f"{root.name} component metrics"),
        arm=arm,
        seed=seed,
        fold=fold,
        run_id=str(run_id),
        expected=expected_components[fold],
        expected_cells=component_cells,
    )
    gene_rows = _normalize_gene_rows(
        _read_records(gene_path, label=f"{root.name} gene metrics"),
        arm=arm,
        seed=seed,
        fold=fold,
        run_id=str(run_id),
        genes=genes,
    )
    component_gene_rows = _normalize_component_gene_rows(
        _read_records(
            component_gene_path, label=f"{root.name} component-gene metrics"
        ),
        arm=arm,
        seed=seed,
        fold=fold,
        run_id=str(run_id),
        genes=genes,
        expected=expected_components[fold],
        expected_cells=component_cells,
    )
    substitutions: tuple[dict[str, Any], ...] = ()
    if arm == "observed_near":
        substitution_path = _output_file(
            outputs, "context_substitution", root=root, project_root=project_root
        )
        substitutions = _normalize_substitution_rows(
            _read_records(substitution_path, label=f"{root.name} context substitution"),
            seed=seed,
            fold=fold,
            run_id=str(run_id),
            expected=expected_components[fold],
            expected_cells=component_cells,
        )
    elif outputs.get("context_substitution") is not None:
        raise MatchedGraphContextAnalysisError(
            f"non-near run contains context substitutions: {root.name}"
        )
    return ConfirmationRun(
        root=root,
        result_path=result_path,
        result_file_sha256=_sha256_file(result_path),
        result=result,
        arm=arm,
        seed=seed,
        fold=fold,
        component_rows=component_rows,
        gene_rows=gene_rows,
        component_gene_rows=component_gene_rows,
        substitution_rows=substitutions,
    )


def _validate_confirmation_coverage(
    runs: Sequence[ConfirmationRun],
) -> list[ConfirmationRun]:
    expected = {(arm, seed, fold) for arm in ARMS for seed in SEEDS for fold in FOLDS}
    observed: dict[tuple[str, int, int], ConfirmationRun] = {}
    for run in runs:
        if run.key in observed:
            raise MatchedGraphContextAnalysisError(f"duplicate confirmation slot: {run.key}")
        observed[run.key] = run
    if set(observed) != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise MatchedGraphContextAnalysisError(
            f"confirmation coverage differs: missing={missing}, extra={extra}"
        )
    for fold in FOLDS:
        fold_runs = [run for run in runs if run.fold == fold]
        parameter_counts = {int(run.result["parameter_count"]) for run in fold_runs}
        hidden_widths = {
            int(
                _mapping(run.result.get("config"), "confirmation config").get(
                    "hidden_width"
                )
            )
            for run in fold_runs
        }
        if len(parameter_counts) != 1 or len(hidden_widths) != 1:
            raise MatchedGraphContextAnalysisError(
                f"confirmation arms in outer fold {fold} do not retain shared "
                "hidden width/parameter count"
            )
    return [observed[key] for key in sorted(expected)]


def _relative_gain(baseline: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if (
        first.ndim != 1
        or first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first < 0)
        or np.any(second < 0)
    ):
        raise MatchedGraphContextAnalysisError("relative-gain inputs are invalid")
    denominator = float(first.mean())
    if denominator <= 0:
        raise MatchedGraphContextAnalysisError("relative-gain baseline is nonpositive")
    return (denominator - float(second.mean())) / denominator


def _slide_stratified_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    slides: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    return_draws: bool = False,
) -> dict[str, Any]:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    slide_array = np.asarray(slides, dtype=str)
    if first.shape != (EXPECTED_COMPONENTS,) or second.shape != first.shape or slide_array.shape != first.shape:
        raise MatchedGraphContextAnalysisError(
            f"bootstrap requires exactly {EXPECTED_COMPONENTS} aligned components"
        )
    if isinstance(resamples, bool) or resamples != BOOTSTRAP_RESAMPLES:
        raise MatchedGraphContextAnalysisError(
            f"bootstrap requires exactly {BOOTSTRAP_RESAMPLES} resamples"
        )
    strata = [np.flatnonzero(slide_array == slide) for slide in SLIDES]
    if any(len(indices) == 0 for indices in strata):
        raise MatchedGraphContextAnalysisError("bootstrap requires both slide strata")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        indices = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        draws[draw] = _relative_gain(first[indices], second[indices])
    result: dict[str, Any] = {
        "resamples": resamples,
        "seed": seed,
        "slide_stratified": True,
        "point": _relative_gain(first, second),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "positive_draw_fraction": float(np.mean(draws > 0)),
    }
    if return_draws:
        result["draws"] = draws
    return result


def _component_arm_arrays(
    component_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> tuple[list[tuple[str, int]], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    keys = sorted({(str(row["slide"]), int(row["component"])) for row in component_rows})
    if len(keys) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError("aggregate component axis is incomplete")
    index = {key: offset for offset, key in enumerate(keys)}
    by_seed: dict[str, dict[int, np.ndarray]] = {
        arm: {seed: np.full(len(keys), np.nan) for seed in SEEDS} for arm in ARMS
    }
    seen: set[tuple[str, int, tuple[str, int]]] = set()
    for row in component_rows:
        key = (str(row["slide"]), int(row["component"]))
        slot = (str(row["arm"]), int(row["seed"]), key)
        if slot in seen:
            raise MatchedGraphContextAnalysisError(f"duplicate aggregate component slot: {slot}")
        seen.add(slot)
        by_seed[slot[0]][slot[1]][index[key]] = float(row[metric])
    arrays: dict[str, np.ndarray] = {}
    for arm in ARMS:
        stack = np.stack([by_seed[arm][seed] for seed in SEEDS])
        if not np.isfinite(stack).all():
            raise MatchedGraphContextAnalysisError(f"aggregate {metric} coverage incomplete: {arm}")
        arrays[arm] = stack.mean(axis=0)
    return keys, arrays, by_seed


def _contrast_summary(
    component_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys, arrays, by_seed = _component_arm_arrays(component_rows, metric=metric)
    first, second = arrays[baseline], arrays[candidate]
    effects = first - second
    slides = [key[0] for key in keys]
    slide_means = {
        slide: float(np.mean(effects[np.asarray(slides) == slide])) for slide in SLIDES
    }
    seed_effects = {
        str(seed): float(
            np.mean(by_seed[baseline][seed] - by_seed[candidate][seed])
        )
        for seed in SEEDS
    }
    bootstrap = _slide_stratified_bootstrap(
        first,
        second,
        slides,
        seed=bootstrap_seed,
    )
    return {
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "difference_definition": f"{baseline} minus {candidate}; positive favors {candidate}",
        "component_equal_baseline": float(first.mean()),
        "component_equal_candidate": float(second.mean()),
        "mean_difference": float(effects.mean()),
        "relative_gain": _relative_gain(first, second),
        "relative_gain_ci95": [bootstrap["lower_95"], bootstrap["upper_95"]],
        "bootstrap": bootstrap,
        "favoring_components": int(np.sum(effects > 0)),
        "favoring_seeds": int(np.sum(np.asarray(list(seed_effects.values())) > 0)),
        "slide_mean_differences": slide_means,
        "component_effects": [
            {"slide": key[0], "component": key[1], "difference": float(effect)}
            for key, effect in zip(keys, effects, strict=True)
        ],
        "seed_mean_differences": seed_effects,
    }


def _substitution_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keys = sorted({(str(row["slide"]), int(row["component"])) for row in rows})
    if len(keys) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError("substitution aggregate component axis incomplete")
    index = {key: offset for offset, key in enumerate(keys)}
    values = {
        context: {seed: np.full(len(keys), np.nan) for seed in SEEDS}
        for context in SUBSTITUTIONS
    }
    for row in rows:
        values[str(row["context_variant"])][int(row["seed"])][
            index[(str(row["slide"]), int(row["component"]))]
        ] = float(row["mse"])
    aggregate: dict[str, np.ndarray] = {}
    for context in SUBSTITUTIONS:
        matrix = np.stack([values[context][seed] for seed in SEEDS])
        if not np.isfinite(matrix).all():
            raise MatchedGraphContextAnalysisError(f"substitution aggregate incomplete: {context}")
        aggregate[context] = matrix.mean(axis=0)
    native = aggregate["native"]
    summaries: dict[str, Any] = {}
    for offset, context in enumerate(SUBSTITUTIONS[1:], start=1):
        substituted = aggregate[context]
        degradation = substituted - native
        summaries[context] = {
            "definition": f"{context} substituted MSE minus native true-model MSE",
            "component_equal_native_mse": float(native.mean()),
            "component_equal_substituted_mse": float(substituted.mean()),
            "mean_mse_degradation": float(degradation.mean()),
            "relative_native_advantage": _relative_gain(substituted, native),
            "favoring_native_components": int(np.sum(degradation > 0)),
            "bootstrap": _slide_stratified_bootstrap(
                substituted,
                native,
                [key[0] for key in keys],
                seed=BOOTSTRAP_SEED + 100 + offset,
            ),
        }
    return {
        "role": "locked true-model feature substitution; no refitting",
        "claim": "prediction faithfulness diagnostic, not a mechanism or causal effect",
        "substitutions": summaries,
    }


def _program_summaries(
    component_gene_rows: Sequence[Mapping[str, Any]],
    *,
    genes: Sequence[str],
) -> dict[str, Any]:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    result: dict[str, Any] = {}
    for program, declared in TARGET_PROGRAMS.items():
        intersection = [gene for gene in declared if gene in gene_index]
        if intersection != list(declared):
            raise MatchedGraphContextAnalysisError(
                f"frozen target-program panel intersection changed: {program}"
            )
        indices = {gene_index[gene] for gene in intersection}
        arm_values: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            selected = [
                row for row in component_gene_rows
                if row["arm"] == arm and int(row["gene_index"]) in indices
            ]
            expected_count = len(indices) * len(SEEDS) * EXPECTED_COMPONENTS
            if len(selected) != expected_count:
                raise MatchedGraphContextAnalysisError(
                    f"program gene coverage incomplete: {program}/{arm}"
                )
            grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
            for row in selected:
                grouped.setdefault((str(row["slide"]), int(row["component"])), []).append(row)
            component_mse = [
                float(np.mean([float(row["mse"]) for row in rows]))
                for rows in grouped.values()
            ]
            component_mae = [
                float(np.mean([float(row["mae"]) for row in rows]))
                for rows in grouped.values()
            ]
            if len(component_mse) != EXPECTED_COMPONENTS:
                raise MatchedGraphContextAnalysisError(
                    f"program component coverage incomplete: {program}/{arm}"
                )
            arm_values[arm] = {
                "component_equal_mse": float(np.mean(component_mse)),
                "component_equal_mae": float(np.mean(component_mae)),
            }
        result[program] = {
            "declared_genes": list(declared),
            "panel_intersection": intersection,
            "intersection_count": len(intersection),
            "arms": arm_values,
            "observed_near_relative_mse_gain_vs_no_graph": (
                arm_values["no_graph"]["component_equal_mse"]
                - arm_values["observed_near"]["component_equal_mse"]
            ) / arm_values["no_graph"]["component_equal_mse"],
            "observed_near_relative_mse_gain_vs_permuted_near": (
                arm_values["permuted_near"]["component_equal_mse"]
                - arm_values["observed_near"]["component_equal_mse"]
            ) / arm_values["permuted_near"]["component_equal_mse"],
            "interpretation": "descriptive fixed target-gene-family summary; not an independent validation",
        }
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise MatchedGraphContextAnalysisError("BH-FDR p-values are invalid")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _gene_level_bh_summary(
    component_gene_rows: Sequence[Mapping[str, Any]],
    *,
    genes: Sequence[str],
) -> dict[str, Any]:
    """Exploratory 1,000-gene paired component analysis with BH adjustment."""

    component_keys = sorted(
        {(str(row["slide"]), int(row["component"])) for row in component_gene_rows}
    )
    if len(component_keys) != EXPECTED_COMPONENTS:
        raise MatchedGraphContextAnalysisError("gene analysis component axis is incomplete")
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    component_index = {key: index for index, key in enumerate(component_keys)}
    mse = np.full(
        (len(ARMS), len(SEEDS), EXPECTED_COMPONENTS, EXPECTED_GENES),
        np.nan,
        dtype=np.float64,
    )
    for row in component_gene_rows:
        slot = (
            arm_index[str(row["arm"])],
            seed_index[int(row["seed"])],
            component_index[(str(row["slide"]), int(row["component"]))],
            int(row["gene_index"]),
        )
        if math.isfinite(mse[slot]):
            raise MatchedGraphContextAnalysisError(
                "gene analysis has duplicate arm/seed/component/gene slot"
            )
        mse[slot] = float(row["mse"])
    if not np.isfinite(mse).all():
        raise MatchedGraphContextAnalysisError("gene analysis coverage is incomplete")
    seed_mean = mse.mean(axis=1)
    baseline = seed_mean[arm_index["no_graph"]]
    candidate = seed_mean[arm_index["observed_near"]]
    differences = baseline - candidate
    means = differences.mean(axis=0)
    standard_deviation = differences.std(axis=0, ddof=1)
    standard_error = standard_deviation / math.sqrt(EXPECTED_COMPONENTS)
    from scipy.stats import t as student_t

    p_values = np.empty(EXPECTED_GENES, dtype=np.float64)
    regular = standard_error > 0
    p_values[regular] = 2.0 * student_t.sf(
        np.abs(means[regular] / standard_error[regular]),
        df=EXPECTED_COMPONENTS - 1,
    )
    p_values[~regular] = np.where(means[~regular] == 0.0, 1.0, 0.0)
    q_values = _benjamini_hochberg(p_values)
    baseline_means = baseline.mean(axis=0)
    candidate_means = candidate.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for index, gene in enumerate(genes):
        relative = (
            (baseline_means[index] - candidate_means[index]) / baseline_means[index]
            if baseline_means[index] > 0
            else None
        )
        rows.append(
            {
                "gene_index": index,
                "gene": gene,
                "component_equal_no_graph_mse": float(baseline_means[index]),
                "component_equal_observed_near_mse": float(candidate_means[index]),
                "mean_paired_mse_difference": float(means[index]),
                "relative_mse_gain": None if relative is None else float(relative),
                "favoring_components": int(np.sum(differences[:, index] > 0)),
                "paired_component_t_p_value": float(p_values[index]),
                "bh_q_value": float(q_values[index]),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["bh_q_value"]),
            -float(row["mean_paired_mse_difference"]),
            int(row["gene_index"]),
        ),
    )
    return {
        "status": "exploratory_descriptive",
        "contrast": "no_graph minus observed_near per-gene MSE; positive favors graph",
        "multiple_testing_family": "all 1,000 fixed panel target genes",
        "test": (
            "two-sided one-sample t test of 27 seed-averaged paired geometry-component "
            "MSE differences followed by Benjamini-Hochberg adjustment"
        ),
        "q_threshold": 0.05,
        "positive_q05_count": int(
            sum(row["bh_q_value"] <= 0.05 and row["mean_paired_mse_difference"] > 0 for row in rows)
        ),
        "negative_q05_count": int(
            sum(row["bh_q_value"] <= 0.05 and row["mean_paired_mse_difference"] < 0 for row in rows)
        ),
        "top_25_by_q_then_graph_favorable_effect": ranked[:25],
        "all_genes": sorted(rows, key=lambda row: int(row["gene_index"])),
        "interpretation": (
            "exploratory gene ranking only; components are nested in two slides and "
            "are not independent patient-level biological replicates"
        ),
    }


def _verified_prior_contextual_sensitivity(
    aggregate_path: Path | None,
    *,
    manifest_path: Path | None,
    success_path: Path | None,
) -> dict[str, Any]:
    if aggregate_path is None:
        return {
            "status": "not_provided",
            "interpretation": "existing V0-V6 contextual sensitivity was not imported",
        }
    if manifest_path is None or success_path is None:
        raise MatchedGraphContextAnalysisError(
            "prior V0-V6 aggregate requires its manifest and success receipt"
        )
    manifest = _strict_json(manifest_path, label="prior analysis manifest")
    success = _strict_json(success_path, label="prior analysis success")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("campaign_id") != "cmp_20260810_same_gene_robustness_multiverse_v1"
        or not isinstance(files, list)
        or canonical_sha256(files) != manifest.get("files_sha256")
        or canonical_sha256(_mapping(success.get("payload"), "prior success payload"))
        != success.get("success_sha256")
        or _mapping(success.get("payload"), "prior success payload").get("files_sha256")
        != manifest.get("files_sha256")
        or _mapping(success.get("payload"), "prior success payload").get(
            "analysis_manifest_sha256"
        ) != _sha256_file(manifest_path)
    ):
        raise MatchedGraphContextAnalysisError("prior V0-V6 analysis authority is invalid")
    entries = {str(item.get("path")): item for item in files if isinstance(item, Mapping)}
    aggregate_entry = entries.get(aggregate_path.name)
    if aggregate_entry is None or aggregate_entry.get("sha256") != _sha256_file(aggregate_path):
        raise MatchedGraphContextAnalysisError("prior aggregate checksum is not manifest-bound")
    aggregate = _strict_json(aggregate_path, label="prior V0-V6 aggregate")
    variants = _mapping(aggregate.get("variants"), "prior variants")
    if set(variants) != set(V0_V6_VARIANTS):
        raise MatchedGraphContextAnalysisError("prior V0-V6 aggregate coverage changed")
    rows: dict[str, Any] = {}
    for variant in V0_V6_VARIANTS:
        prediction = _mapping(_mapping(variants[variant], variant).get("prediction"), f"{variant} prediction")
        near_no_graph = _mapping(
            prediction.get("near_vs_morphology"), f"{variant} near_vs_morphology"
        )
        near_permutation = _mapping(
            prediction.get("near_vs_permutation"), f"{variant} near_vs_permutation"
        )
        rows[variant] = {
            "context_change": {
                "V0": "within-FOV log1p all-cell reference",
                "V1": "cross-FOV edges within geometry component",
                "V2": "QC-pass receivers and graph sources",
                "V3": "panel-CP10k scale",
                "V4": "train-only library-size residualization",
                "V5": "component graph + QC-pass + CP10k",
                "V6": "train-only cell-type and library-size residualization",
            }[variant],
            "near_vs_no_graph_relative_gain": _finite(
                near_no_graph.get("component_equal_relative_gain"),
                f"{variant} prior no-graph gain",
            ),
            "near_vs_permutation_relative_gain": _finite(
                near_permutation.get("component_equal_relative_gain"),
                f"{variant} prior permutation gain",
            ),
        }
    return {
        "status": "verified_existing_aggregate",
        "source_path": aggregate_path.as_posix(),
        "source_sha256": _sha256_file(aggregate_path),
        "variants": rows,
        "interpretation": (
            "existing contextual/preprocessing sensitivity only; different model-selection "
            "protocol and no contribution to the matched campaign verdict"
        ),
    }


def _jacobian_summary(
    runs: Sequence[ConfirmationRun],
    *,
    genes: Sequence[str],
    project_root: Path,
) -> dict[str, Any]:
    near_runs = [run for run in runs if run.arm == "observed_near"]
    jacobian_presence: list[bool] = []
    derivative_presence: list[bool] = []
    for run in near_runs:
        outputs = _mapping(run.result.get("outputs"), "outputs")
        jacobian_presence.append(outputs.get("context_jacobian") is not None)
        derivative_presence.append(
            outputs.get("component_hidden_derivatives") is not None
        )
    if jacobian_presence != derivative_presence:
        raise MatchedGraphContextAnalysisError(
            "Jacobian and component-derivative outputs must be present together"
        )
    if not any(jacobian_presence):
        return {
            "status": "not_available",
            "interpretation": "optional locked-model Jacobian was absent; no sensitivity claim emitted",
        }
    if not all(jacobian_presence):
        raise MatchedGraphContextAnalysisError(
            "partial observed-near Jacobian coverage cannot support sensitivity claims"
        )
    matrices: list[np.ndarray] = []
    for run in near_runs:
        outputs = _mapping(run.result.get("outputs"), "outputs")
        path = _output_file(
            outputs,
            "context_jacobian",
            root=run.root,
            project_root=project_root,
        )
        derivative_path = _output_file(
            outputs,
            "component_hidden_derivatives",
            root=run.root,
            project_root=project_root,
        )
        derivative_rows = _read_records(
            derivative_path, label=f"{run.root.name} component hidden derivatives"
        )
        expected_components = {
            (str(row["slide"]), int(row["component"])): int(row["n_cells"])
            for row in run.component_rows
        }
        hidden_width = int(
            _mapping(run.result.get("config"), f"{run.root.name} config").get(
                "hidden_width"
            )
        )
        derivative_by_component: dict[tuple[str, int], np.ndarray] = {}
        for row_index, row in enumerate(derivative_rows):
            block = _mapping(row, f"{run.root.name} derivative row {row_index}")
            key = (str(block.get("slide")), int(block.get("component")))
            if key not in expected_components or key in derivative_by_component:
                raise MatchedGraphContextAnalysisError(
                    f"component hidden derivative coverage invalid: {run.root.name}"
                )
            if _coerce_int(
                block.get("n_cells"), f"{run.root.name} derivative n_cells", minimum=1
            ) != expected_components[key]:
                raise MatchedGraphContextAnalysisError(
                    f"component hidden derivative cell count changed: {run.root.name}"
                )
            derivative = np.asarray(block.get("mean_hidden_derivative"), dtype=np.float64)
            if derivative.shape != (hidden_width,) or not np.isfinite(derivative).all():
                raise MatchedGraphContextAnalysisError(
                    f"component hidden derivative shape/value invalid: {run.root.name}"
                )
            derivative_by_component[key] = derivative
        if set(derivative_by_component) != set(expected_components):
            raise MatchedGraphContextAnalysisError(
                f"component hidden derivative coverage incomplete: {run.root.name}"
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                required = {
                    "total", "linear", "nonlinear", "mean_hidden_derivative",
                    "genes", "eligible_gene_mask",
                }
                if keys != required:
                    raise MatchedGraphContextAnalysisError(
                        f"Jacobian archive schema changed: {path}"
                    )
                matrix = np.asarray(archive["total"], dtype=np.float64)
                linear = np.asarray(archive["linear"], dtype=np.float64)
                nonlinear = np.asarray(archive["nonlinear"], dtype=np.float64)
                mean_derivative = np.asarray(
                    archive["mean_hidden_derivative"], dtype=np.float64
                )
                archive_genes = tuple(str(value) for value in archive["genes"].tolist())
                eligible = np.asarray(archive["eligible_gene_mask"], dtype=bool)
        except MatchedGraphContextAnalysisError:
            raise
        except (OSError, ValueError) as error:
            raise MatchedGraphContextAnalysisError(f"cannot read Jacobian: {path}") from error
        if (
            matrix.shape != (EXPECTED_GENES, EXPECTED_GENES)
            or linear.shape != matrix.shape
            or nonlinear.shape != matrix.shape
            or mean_derivative.shape != (hidden_width,)
            or archive_genes != tuple(genes)
            or eligible.shape != (EXPECTED_GENES,)
            or not eligible.all()
            or not all(
                np.isfinite(value).all()
                for value in (matrix, linear, nonlinear, mean_derivative)
            )
            or not np.allclose(matrix, linear + nonlinear, rtol=0, atol=2e-6)
            or not np.allclose(
                mean_derivative,
                np.mean(list(derivative_by_component.values()), axis=0),
                rtol=0,
                atol=2e-6,
            )
        ):
            raise MatchedGraphContextAnalysisError(f"Jacobian shape/values invalid: {path}")
        matrices.append(matrix)
    matrix_stack = np.stack(matrices)
    index = {gene: offset for offset, gene in enumerate(genes)}
    names = list(TARGET_PROGRAMS)
    signed = np.empty((len(names), len(names)), dtype=np.float64)
    absolute = np.empty_like(signed)
    for source_offset, source in enumerate(names):
        source_index = [index[gene] for gene in TARGET_PROGRAMS[source]]
        for target_offset, target in enumerate(names):
            target_index = [index[gene] for gene in TARGET_PROGRAMS[target]]
            block = matrix_stack[:, target_index][:, :, source_index]
            signed[target_offset, source_offset] = float(block.mean())
            absolute[target_offset, source_offset] = float(np.abs(block).mean())
    diagonal = np.diag(absolute)
    off_diagonal = absolute[~np.eye(len(names), dtype=bool)]
    off_diagonal_mean = float(off_diagonal.mean())
    return {
        "status": "complete",
        "units": "standardized target output per standardized context input",
        "orientation": "rows are target programs; columns are source programs",
        "program_order": names,
        "signed_mean_sensitivity": signed.tolist(),
        "mean_absolute_sensitivity": absolute.tolist(),
        "same_name_absolute_enrichment": (
            float(diagonal.mean() / off_diagonal_mean)
            if off_diagonal_mean > 0
            else None
        ),
        "run_count": len(matrices),
        "interpretation": (
            "locked-model mean Jacobian sensitivity only; not predictive gain, "
            "a correlation, communication mechanism, or causal effect"
        ),
    }


def _decision(
    graph_vs_no_graph: Mapping[str, Any],
    graph_vs_permutation: Mapping[str, Any],
    mae_vs_no_graph: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "graph_vs_no_graph_gain_at_least_2pct": (
            float(graph_vs_no_graph["relative_gain"]) >= GRAPH_GAIN_MINIMUM
        ),
        "graph_vs_no_graph_ci_lower_above_zero": (
            float(graph_vs_no_graph["relative_gain_ci95"][0]) > 0
        ),
        "both_slide_mean_differences_positive": all(
            float(value) > 0
            for value in _mapping(
                graph_vs_no_graph["slide_mean_differences"], "slide means"
            ).values()
        ),
        "at_least_20_of_27_components_favor_graph": (
            int(graph_vs_no_graph["favoring_components"])
            >= MINIMUM_FAVORING_COMPONENTS
        ),
        "at_least_4_of_5_seeds_favor_graph": (
            int(graph_vs_no_graph["favoring_seeds"]) >= MINIMUM_FAVORING_SEEDS
        ),
        "component_equal_mae_does_not_worsen": (
            float(mae_vs_no_graph["mean_difference"]) >= 0
        ),
        "graph_vs_permutation_gain_at_least_1pct": (
            float(graph_vs_permutation["relative_gain"])
            >= PERMUTATION_GAIN_MINIMUM
        ),
        "graph_vs_permutation_ci_lower_above_zero": (
            float(graph_vs_permutation["relative_gain_ci95"][0]) > 0
        ),
    }
    if all(criteria.values()):
        verdict = "GRAPH CONTEXT SUPPORTED"
    elif float(graph_vs_no_graph["relative_gain_ci95"][1]) < GRAPH_GAIN_MINIMUM:
        verdict = "GRAPH CONTEXT NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "criteria": criteria,
        "claim_ceiling": "graph_alignment_dependent_predictive_dependency_within_two_observed_slides",
        "not_supported_claims": [
            "patient generalization",
            "cell-cell communication",
            "biological mechanism",
            "causal influence",
            "GAT or attention utility",
        ],
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    primary = _mapping(payload["primary_contrasts"], "primary contrasts")
    no_graph = _mapping(primary["graph_vs_no_graph_mse"], "no graph contrast")
    permutation = _mapping(primary["graph_vs_permutation_mse"], "permutation contrast")
    mae = _mapping(primary["graph_vs_no_graph_mae"], "MAE contrast")
    decision = _mapping(payload["decision"], "decision")
    programs = _mapping(payload["target_programs"], "target programs")
    rows = [
        "# Matched graph-context nested-CV analysis",
        "",
        f"Verdict: **{decision['verdict']}**.",
        "",
        "This is an exploratory held-out geometry-component comparison within two "
        "already-observed slides. Seeds were averaged within each component; the "
        "10,000-resample intervals are slide-stratified descriptive intervals.",
        "",
        "## Primary contrasts",
        "",
        f"- Observed near versus no graph: {100*float(no_graph['relative_gain']):.3f}% "
        f"relative MSE gain (95% descriptive interval "
        f"{100*float(no_graph['relative_gain_ci95'][0]):.3f}% to "
        f"{100*float(no_graph['relative_gain_ci95'][1]):.3f}%); "
        f"{no_graph['favoring_components']}/27 components and "
        f"{no_graph['favoring_seeds']}/5 seed aggregates favored near context.",
        f"- Observed near versus independently tuned permutation: "
        f"{100*float(permutation['relative_gain']):.3f}% relative MSE gain "
        f"(95% descriptive interval {100*float(permutation['relative_gain_ci95'][0]):.3f}% "
        f"to {100*float(permutation['relative_gain_ci95'][1]):.3f}%).",
        f"- No-graph minus observed-near MAE: {float(mae['mean_difference']):.6g} "
        "(nonnegative means MAE did not worsen).",
        "",
        "Near-versus-annular is reported as a locality diagnostic and does not alter "
        "the primary verdict. Zero, permuted, annular, and 10–25 µm substitutions "
        "are locked-model faithfulness diagnostics without refitting.",
        "",
        "## Fixed target programs",
        "",
    ]
    for name, value in programs.items():
        block = _mapping(value, f"program {name}")
        rows.append(
            f"- `{name}` ({block['intersection_count']} genes): near-versus-no-graph "
            f"MSE gain {100*float(block['observed_near_relative_mse_gain_vs_no_graph']):.3f}%; "
            f"near-versus-permutation {100*float(block['observed_near_relative_mse_gain_vs_permuted_near']):.3f}%."
        )
    rows.extend(
        [
            "",
            "These program summaries use the fixed panel intersection and are descriptive, "
            "not independent biological validation.",
            "",
            "## Claim limits",
            "",
            "The maximum defensible claim is a graph-alignment-dependent predictive "
            "dependency within these two observed slides. Components are leakage-control "
            "units, not patients. No patient-generalization, communication, mechanism, "
            "attention-utility, or causal claim is supported by this analysis.",
            "",
        ]
    )
    return "\n".join(rows)


def _publish_analysis_file(path: Path, value: bytes) -> None:
    """Publish once, or accept an exactly identical existing authority."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == value:
            return
        raise MatchedGraphContextAnalysisError(
            f"refusing to overwrite a different analysis artifact: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != value:
                raise MatchedGraphContextAnalysisError(
                    f"concurrent different analysis artifact exists: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _discover_bundles(
    references: Sequence[Path],
    *,
    project_root: Path,
) -> list[Path]:
    bundles: list[Path] = []
    for reference in references:
        path = reference if reference.is_absolute() else project_root / reference
        if path.is_dir() and (path / "results.json").is_file():
            bundles.append(path.resolve())
            continue
        if path.is_dir():
            bundles.extend(
                sorted(result.parent.resolve() for result in path.rglob("results.json"))
            )
            continue
        if path.is_file():
            value = _strict_json(path, label=f"confirmation authority {path}")
            if value.get("kind") == "matched_graph_context_job_plan":
                digest = _hex(
                    value.get("plan_payload_sha256"),
                    "confirmation plan payload sha256",
                )
                unsigned = dict(value)
                unsigned.pop("plan_payload_sha256", None)
                if canonical_sha256(unsigned) != digest:
                    raise MatchedGraphContextAnalysisError(
                        "confirmation plan self-digest mismatch"
                    )
                if (
                    value.get("schema_version") != 1
                    or value.get("campaign_id") != CAMPAIGN_ID
                    or value.get("stage") != "confirmation"
                    or value.get("contract_sha256") != CONTRACT_SHA256
                    or value.get("candidate_design_sha256")
                    != CANDIDATE_DESIGN_SHA256
                ):
                    raise MatchedGraphContextAnalysisError(
                        "confirmation plan identity/contract changed"
                    )
            jobs = value.get("jobs", value.get("bundles"))
            if not isinstance(jobs, list):
                raise MatchedGraphContextAnalysisError(
                    f"confirmation authority has no jobs/bundles: {path}"
                )
            for index, item in enumerate(jobs):
                block = _mapping(item, f"confirmation authority item {index}")
                raw = block.get(
                    "bundle_path", block.get("artifact_path", block.get("path"))
                )
                if raw is None and block.get("result_path") is not None:
                    raw = str(Path(str(block["result_path"])).parent)
                bundle = _project_path(
                    raw,
                    f"confirmation bundle authority {index}",
                    project_root=project_root,
                )
                if not bundle.is_dir():
                    raise MatchedGraphContextAnalysisError(
                        f"confirmation authority is not a bundle: {bundle}"
                    )
                expected_sha = block.get("results_sha256", block.get("sha256"))
                if expected_sha is not None:
                    _hex(expected_sha, f"confirmation authority {index} SHA")
                    if _sha256_file(bundle / "results.json") != expected_sha:
                        raise MatchedGraphContextAnalysisError(
                            f"confirmation authority result checksum mismatch: {bundle}"
                        )
                bundles.append(bundle)
            continue
        raise MatchedGraphContextAnalysisError(f"confirmation reference missing: {path}")
    unique = sorted(set(bundles))
    if len(unique) != len(bundles):
        raise MatchedGraphContextAnalysisError("duplicate confirmation bundle authorities")
    return unique


def analyze(
    *,
    selection_receipt_path: Path,
    bundle_paths: Sequence[Path],
    output_root: Path,
    prior_aggregate_path: Path | None = None,
    prior_manifest_path: Path | None = None,
    prior_success_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    _verify_contract(project_root=project_root)
    receipt_path = selection_receipt_path.resolve(strict=True)
    receipt = _verify_selection_receipt(receipt_path, project_root=project_root)
    receipt_file_sha = _sha256_file(receipt_path)
    receipt_payload_sha = str(receipt["payload_sha256"])
    prepared_root = project_root / (
        "data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all"
    )
    manifest_sha = _sha256_file(prepared_root / "manifest.json")
    integrity_sha = _sha256_file(prepared_root / "integrity_manifest.json")
    genes = _strict_gene_list(prepared_root / "genes.json")
    component_cells, expected_components = _expected_component_map(prepared_root)
    runs = [
        _load_confirmation_run(
            path,
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_file_sha256=receipt_file_sha,
            receipt_payload_sha256=receipt_payload_sha,
            expected_components=expected_components,
            component_cells=component_cells,
            genes=genes,
            project_root=project_root,
        )
        for path in bundle_paths
    ]
    runs = _validate_confirmation_coverage(runs)
    for run in runs:
        inputs = _mapping(run.result["input"], "run input")
        if (
            inputs.get("prepared_manifest_sha256") != manifest_sha
            or inputs.get("integrity_manifest_sha256") != integrity_sha
        ):
            raise MatchedGraphContextAnalysisError(
                f"run prepared-file hashes changed: {run.root.name}"
            )

    component_rows = [row for run in runs for row in run.component_rows]
    gene_rows = [row for run in runs for row in run.gene_rows]
    component_gene_rows = [row for run in runs for row in run.component_gene_rows]
    substitution_rows = [row for run in runs for row in run.substitution_rows]
    if len(component_gene_rows) != len(ARMS) * len(SEEDS) * EXPECTED_COMPONENTS * EXPECTED_GENES:
        raise MatchedGraphContextAnalysisError(
            "aggregate component-gene row count is not exactly 540,000"
        )
    graph_no = _contrast_summary(
        component_rows,
        baseline="no_graph",
        candidate="observed_near",
        metric="mse",
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    graph_perm = _contrast_summary(
        component_rows,
        baseline="permuted_near",
        candidate="observed_near",
        metric="mse",
        bootstrap_seed=BOOTSTRAP_SEED + 1,
    )
    graph_annular = _contrast_summary(
        component_rows,
        baseline="observed_annular",
        candidate="observed_near",
        metric="mse",
        bootstrap_seed=BOOTSTRAP_SEED + 2,
    )
    mae_no = _contrast_summary(
        component_rows,
        baseline="no_graph",
        candidate="observed_near",
        metric="mae",
        bootstrap_seed=BOOTSTRAP_SEED + 3,
    )
    prior = _verified_prior_contextual_sensitivity(
        prior_aggregate_path,
        manifest_path=prior_manifest_path,
        success_path=prior_success_path,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "exploratory": True,
        "contract_sha256": CONTRACT_SHA256,
        "selection_receipt": {
            "path": receipt_path.relative_to(project_root).as_posix(),
            "file_sha256": receipt_file_sha,
            "payload_sha256": receipt_payload_sha,
            "test_metrics_used_for_selection": False,
        },
        "coverage": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "bundle_count": len(runs),
            "component_count": EXPECTED_COMPONENTS,
            "gene_count": EXPECTED_GENES,
            "component_metric_rows": len(component_rows),
            "gene_metric_rows": len(gene_rows),
            "component_gene_metric_rows": len(component_gene_rows),
            "substitution_metric_rows": len(substitution_rows),
        },
        "primary_contrasts": {
            "graph_vs_no_graph_mse": graph_no,
            "graph_vs_permutation_mse": graph_perm,
            "graph_vs_no_graph_mae": mae_no,
        },
        "locality_diagnostic": {
            "near_vs_annular_mse": graph_annular,
            "role": "secondary locality diagnostic; cannot rescue or reverse the primary gate",
        },
        "component_heterogeneity": _component_heterogeneity(
            prepared_root, graph_no
        ),
        "true_model_substitution_faithfulness": _substitution_summary(substitution_rows),
        "existing_v0_v6_contextual_sensitivity": prior,
        "target_programs": _program_summaries(component_gene_rows, genes=genes),
        "exploratory_gene_bh_fdr": _gene_level_bh_summary(
            component_gene_rows, genes=genes
        ),
        "jacobian_program_sensitivity": _jacobian_summary(
            runs, genes=genes, project_root=project_root
        ),
        "decision": _decision(graph_no, graph_perm, mae_no),
        "run_evidence": [
            {
                "run_id": run.root.name,
                "arm": run.arm,
                "seed": run.seed,
                "fold": run.fold,
                "results_path": run.result_path.relative_to(project_root).as_posix(),
                "results_sha256": run.result_file_sha256,
            }
            for run in runs
        ],
        "statistical_unit_note": (
            "Seeds are technical replicates averaged within geometry component. "
            "Components are leakage-control units nested in two slides, not patients."
        ),
    }
    markdown = _markdown(payload)
    result_path = output_root / "analysis.json"
    report_path = output_root / "report.md"
    result_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_analysis_file(result_path, result_bytes)
    _publish_analysis_file(report_path, markdown.encode("utf-8"))
    manifest_files = [
        {
            "path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (result_path, report_path)
    ]
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "files": manifest_files,
        "files_sha256": canonical_sha256(manifest_files),
    }
    manifest_path = output_root / "analysis_manifest.json"
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_analysis_file(manifest_path, manifest_bytes)
    sorted_result_hashes = sorted(run.result_file_sha256 for run in runs)
    success_payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "analysis_manifest_sha256": _sha256_file(manifest_path),
        "files_sha256": manifest["files_sha256"],
        "selection_payload_sha256": receipt_payload_sha,
        "confirmation_results_sha256": sorted_result_hashes,
        "confirmation_results_sha256_digest": canonical_sha256(sorted_result_hashes),
    }
    success = {
        "payload": success_payload,
        "success_sha256": canonical_sha256(success_payload),
    }
    success_bytes = (
        json.dumps(success, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_analysis_file(output_root / "_SUCCESS", success_bytes)
    return payload


def _strict_gene_list(path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatchedGraphContextAnalysisError("cannot read ordered gene axis") from error
    if (
        not isinstance(value, list)
        or len(value) != EXPECTED_GENES
        or len(set(value)) != EXPECTED_GENES
        or not all(isinstance(gene, str) and gene for gene in value)
    ):
        raise MatchedGraphContextAnalysisError("ordered gene axis is invalid")
    return tuple(value)


def _optional_existing(path: Path, *, project_root: Path) -> Path | None:
    candidate = path if path.is_absolute() else project_root / path
    return candidate.resolve() if candidate.is_file() else None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument(
        "--confirmation",
        type=Path,
        action="append",
        required=True,
        help="bundle, directory of bundles, or JSON authority; repeatable",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--prior-aggregate", type=Path, default=DEFAULT_PRIOR_AGGREGATE)
    parser.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--prior-success", type=Path, default=DEFAULT_PRIOR_SUCCESS)
    parser.add_argument(
        "--omit-prior-v0-v6",
        action="store_true",
        help="emit not_provided instead of importing the verified existing aggregate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = current_paths(anchor=PROJECT_ROOT)
    project_root = paths.project_root
    bundles = _discover_bundles(args.confirmation, project_root=project_root)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    if args.omit_prior_v0_v6:
        prior_aggregate = prior_manifest = prior_success = None
    else:
        prior_aggregate = _optional_existing(args.prior_aggregate, project_root=project_root)
        prior_manifest = _optional_existing(args.prior_manifest, project_root=project_root)
        prior_success = _optional_existing(args.prior_success, project_root=project_root)
        if any(value is None for value in (prior_aggregate, prior_manifest, prior_success)):
            prior_aggregate = prior_manifest = prior_success = None
    payload = analyze(
        selection_receipt_path=(
            args.selection_receipt
            if args.selection_receipt.is_absolute()
            else project_root / args.selection_receipt
        ),
        bundle_paths=bundles,
        output_root=output_root,
        prior_aggregate_path=prior_aggregate,
        prior_manifest_path=prior_manifest,
        prior_success_path=prior_success,
        project_root=project_root,
    )
    print(json.dumps({"verdict": payload["decision"]["verdict"], "output_root": output_root.as_posix()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatchedGraphContextAnalysisError as error:
        print(f"analysis failed closed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

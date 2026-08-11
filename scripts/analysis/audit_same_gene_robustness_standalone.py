#!/usr/bin/env python3
"""Standalone numerical audit of the published same-gene robustness aggregate.

This audit deliberately does not import the campaign analyzer or the
``spatial_benchmark`` package.  It reconstructs every statistic that is
identifiable from the four published numerical files and labels quantities
whose source arrays are not present in that publication.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
CONTRACT_SHA256 = "8761098cbafa91ae81b53a1c9cd0d8dcd293476d967f74cddde9be7dc5c990e9"
LAUNCH_SHA256 = "38fd482c1343f10efc8ec18507b05f5110f226898f236815a97b9671f7269223"
ELIGIBLE_MASK_SHA256 = "2420a9d160894a78e6a1db1cc4ff2c52757211cf31dec859856f7c3cf231712d"

VARIANTS = tuple(f"V{index}" for index in range(7))
PRIMARY_VARIANTS = VARIANTS[:6]
SEEDS = (20260810, 20261810, 20262810, 20263810, 20264810)
FOLDS = (0, 1, 2, 3)
ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
)
MATRIX_ARMS = ARMS[1:]
LABELS = ("selected", "anchor12")
PARTS = ("total", "linear", "nonlinear")
GATES = (
    "near_vs_morphology_prediction",
    "near_vs_permutation_prediction",
    "same_name_diagonal_enrichment",
    "strict_row_selectivity",
    "near_vs_permutation_diagonal",
    "fold_stability",
    "technical_validity",
)
NULL_FAMILIES = ("full", "prevalence_sd_decile_matched")
NULL_STATISTICS = (
    "median_absolute_diagonal",
    "row_top1_fraction",
    "row_top1_percent_fraction",
)

COMPONENT_FIELDS = (
    "variant_id",
    "model_seed",
    "fold",
    "arm",
    "anchor",
    "geometry_group",
    "slide",
    "cell_count",
    "mse",
    "mae",
)
GENE_FIELDS = (
    "gene_index",
    "gene",
    "signed_diagonal",
    "absolute_diagonal",
    "absolute_row_rank",
    "row_top1",
    "row_top1_percent",
)

RTOL = 1e-12
ATOL = 1e-15


class AuditRecorder:
    """Collect fail-closed checks without losing later diagnostic evidence."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.fail_counts: dict[str, int] = defaultdict(int)
        self.failures: list[dict[str, Any]] = []

    def require(
        self,
        condition: bool,
        name: str,
        *,
        section: str,
        observed: Any = None,
        expected: Any = None,
    ) -> None:
        self.counts[section] += 1
        if condition:
            return
        self.fail_counts[section] += 1
        self.failures.append(
            {
                "section": section,
                "check": name,
                "observed": json_safe(observed),
                "expected": json_safe(expected),
            }
        )

    def equal(
        self, observed: Any, expected: Any, name: str, *, section: str
    ) -> None:
        self.require(
            observed == expected,
            name,
            section=section,
            observed=observed,
            expected=expected,
        )

    def close(
        self,
        observed: Any,
        expected: Any,
        name: str,
        *,
        section: str,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> None:
        try:
            first = float(observed)
            second = float(expected)
            condition = math.isfinite(first) and math.isfinite(second) and bool(
                np.isclose(first, second, rtol=rtol, atol=atol)
            )
        except (TypeError, ValueError, OverflowError):
            condition = False
        self.require(
            condition,
            name,
            section=section,
            observed=observed,
            expected=expected,
        )

    def summary(self) -> dict[str, Any]:
        sections = {
            name: {
                "checks": count,
                "passed": count - self.fail_counts.get(name, 0),
                "failed": self.fail_counts.get(name, 0),
            }
            for name, count in sorted(self.counts.items())
        }
        total = sum(self.counts.values())
        failed = len(self.failures)
        return {
            "status": "pass" if failed == 0 else "fail",
            "checks": total,
            "passed": total - failed,
            "failed": failed,
            "sections": sections,
            "failures": self.failures,
        }


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    return value


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON token: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject,
        object_pairs_hook=unique,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def read_component_rows(path: Path, audit: AuditRecorder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        audit.equal(tuple(reader.fieldnames or ()), COMPONENT_FIELDS, "component CSV header", section="component_schema")
        for number, raw in enumerate(reader, 2):
            audit.equal(set(raw), set(COMPONENT_FIELDS), f"component row {number} fields", section="component_schema")
            try:
                anchor_text = raw["anchor"]
                if anchor_text not in {"True", "False"}:
                    raise ValueError("invalid boolean")
                row = {
                    "variant_id": raw["variant_id"],
                    "model_seed": int(raw["model_seed"]),
                    "fold": int(raw["fold"]),
                    "arm": raw["arm"],
                    "anchor": anchor_text == "True",
                    "geometry_group": int(raw["geometry_group"]),
                    "slide": raw["slide"],
                    "cell_count": int(raw["cell_count"]),
                    "mse": float(raw["mse"]),
                    "mae": float(raw["mae"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                audit.require(False, f"component row {number} parse: {error}", section="component_schema")
                continue
            audit.require(
                row["variant_id"] in VARIANTS
                and row["model_seed"] in SEEDS
                and row["fold"] in FOLDS
                and row["arm"] in ARMS,
                f"component row {number} categorical domain",
                section="component_schema",
                observed={key: row[key] for key in ("variant_id", "model_seed", "fold", "arm")},
            )
            expected_slide = (
                "SO_1" if 100 <= row["geometry_group"] < 200
                else "SO_2" if 200 <= row["geometry_group"] < 300
                else None
            )
            audit.equal(row["slide"], expected_slide, f"component row {number} slide coding", section="component_schema")
            audit.require(
                row["cell_count"] > 0
                and math.isfinite(row["mse"])
                and math.isfinite(row["mae"])
                and row["mse"] >= 0
                and row["mae"] >= 0,
                f"component row {number} numeric domain",
                section="component_schema",
            )
            rows.append(row)
    return rows


def read_gene_rows(path: Path, audit: AuditRecorder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        audit.equal(tuple(reader.fieldnames or ()), GENE_FIELDS, "gene CSV header", section="gene_schema")
        for number, raw in enumerate(reader, 2):
            audit.equal(set(raw), set(GENE_FIELDS), f"gene row {number} fields", section="gene_schema")
            try:
                row = {
                    "gene_index": int(raw["gene_index"]),
                    "gene": raw["gene"],
                    "signed_diagonal": float(raw["signed_diagonal"]),
                    "absolute_diagonal": float(raw["absolute_diagonal"]),
                    "absolute_row_rank": int(raw["absolute_row_rank"]),
                    "row_top1": int(raw["row_top1"]),
                    "row_top1_percent": int(raw["row_top1_percent"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                audit.require(False, f"gene row {number} parse: {error}", section="gene_schema")
                continue
            audit.require(
                0 <= row["gene_index"] < 1000
                and bool(row["gene"])
                and math.isfinite(row["signed_diagonal"])
                and math.isfinite(row["absolute_diagonal"])
                and 1 <= row["absolute_row_rank"] <= 932
                and row["row_top1"] in (0, 1)
                and row["row_top1_percent"] in (0, 1),
                f"gene row {number} domain",
                section="gene_schema",
            )
            rows.append(row)
    return rows


def relative_gain(
    baseline: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if weights is None:
        first_mean = float(np.mean(first))
        second_mean = float(np.mean(second))
    else:
        weight = np.asarray(weights, dtype=np.float64)
        first_mean = float(np.average(first, weights=weight))
        second_mean = float(np.average(second, weights=weight))
    if first_mean <= 0:
        raise ValueError("nonpositive relative-gain baseline")
    return (first_mean - second_mean) / first_mean


def prediction_array(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    arm: str,
    anchor: bool,
    seed: int | None = None,
    field: str = "mse",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["variant_id"] == variant
        and row["arm"] == arm
        and bool(row["anchor"]) == anchor
        and (seed is None or int(row["model_seed"]) == seed)
    ]
    by_group: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_group[int(row["geometry_group"])].append(row)
    groups = np.asarray(sorted(by_group), dtype=np.int64)
    values = np.asarray(
        [np.mean([float(row[field]) for row in by_group[group]]) for group in groups],
        dtype=np.float64,
    )
    counts = np.asarray(
        [np.mean([int(row["cell_count"]) for row in by_group[group]]) for group in groups],
        dtype=np.float64,
    )
    folds = np.asarray([int(by_group[group][0]["fold"]) for group in groups], dtype=np.int8)
    return groups, values, counts, folds


def bootstrap_indices(groups: np.ndarray, *, draws: int, seed: int) -> list[np.ndarray]:
    strata = [np.flatnonzero(groups // 100 == code) for code in (1, 2)]
    if any(len(value) == 0 for value in strata):
        raise ValueError("both slide strata are required")
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([rng.choice(value, size=len(value), replace=True) for value in strata])
        for _ in range(draws)
    ]


def gain_draws(
    baseline: np.ndarray,
    candidate: np.ndarray,
    indices: Sequence[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [relative_gain(baseline[index], candidate[index]) for index in indices],
        dtype=np.float64,
    )


def matrix_summary(matrix: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    source = np.asarray(matrix, dtype=np.float64)
    selected = source[np.ix_(indices, indices)]
    absolute = np.abs(selected)
    diagonal = np.diag(selected)
    absolute_diagonal = np.abs(diagonal)
    offdiagonal = absolute[~np.eye(len(indices), dtype=bool)]
    ranks = np.asarray(
        [1 + np.sum(absolute[row] > absolute[row, row]) for row in range(len(indices))],
        dtype=np.int64,
    )
    median_diagonal = float(np.median(absolute_diagonal))
    median_offdiagonal = float(np.median(offdiagonal))
    ratio = median_diagonal / median_offdiagonal if median_offdiagonal > 0 else math.inf
    positive = diagonal[diagonal > 0]
    negative = diagonal[diagonal < 0]
    threshold = max(1, int(np.ceil(0.01 * len(indices))))
    return {
        "eligible_gene_count": int(len(indices)),
        "median_absolute_diagonal": median_diagonal,
        "median_absolute_offdiagonal": median_offdiagonal,
        "diagonal_offdiagonal_ratio": float(ratio),
        "row_top1_fraction": float(np.mean(ranks <= 1)),
        "row_top10_fraction": float(np.mean(ranks <= min(10, len(indices)))),
        "row_top1_percent_fraction": float(np.mean(ranks <= threshold)),
        "median_diagonal_absolute_rank": float(np.median(ranks)),
        "positive_diagonal_fraction": float(np.mean(diagonal > 0)),
        "signed_diagonal_positive_count": int(len(positive)),
        "signed_diagonal_negative_count": int(len(negative)),
        "signed_diagonal_zero_count": int(np.sum(diagonal == 0)),
        "median_signed_diagonal": float(np.median(diagonal)),
        "median_positive_diagonal": None if len(positive) == 0 else float(np.median(positive)),
        "median_absolute_negative_diagonal": None if len(negative) == 0 else float(np.median(np.abs(negative))),
        "row_top1_count": int(np.sum(ranks <= 1)),
        "row_top1_percent_count": int(np.sum(ranks <= threshold)),
    }


def compare_summary(
    audit: AuditRecorder,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
    section: str,
) -> None:
    audit.equal(set(observed), set(expected), f"{name} summary fields", section=section)
    for field, value in expected.items():
        if value is None or isinstance(value, (bool, int)):
            audit.equal(observed.get(field), value, f"{name}/{field}", section=section)
        else:
            audit.close(observed.get(field), value, f"{name}/{field}", section=section)


def three_way(consensus: bool, seed_passes: int) -> str:
    if consensus and seed_passes >= 4:
        return "robust_pass"
    if not consensus and 5 - seed_passes >= 4:
        return "robust_gate_failure"
    return "seed_sensitive"


def deterministic_derangement(size: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    identity = np.arange(size, dtype=np.int64)
    for _ in range(256):
        candidate = rng.permutation(identity)
        if bool(np.all(candidate != identity)):
            return candidate
    raise RuntimeError("could not produce deterministic derangement")


def expected_npz_keys() -> set[str]:
    keys = {
        f"{variant}_{arm}_{label}_{part}"
        for variant in VARIANTS
        for arm in MATRIX_ARMS
        for label in LABELS
        for part in PARTS
    }
    for variant in VARIANTS:
        keys.add(f"{variant}_observed_near_selected_cell_weighted_total")
        keys.add(f"{variant}_observed_near_selected_slide_equal_total")
    keys.update(
        f"gene_label_null_{family}_{statistic}"
        for family in NULL_FAMILIES
        for statistic in NULL_STATISTICS
    )
    keys.add("V0_budget_selected_minus_anchor12_bootstrap")
    return keys


def audit_component_coverage(
    audit: AuditRecorder,
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    audit.equal(len(rows), 7560, "component row count", section="component_coverage")
    audit.equal(len(rows), payload["exact_row_counts"]["component_metrics_csv"], "component row count vs JSON", section="component_coverage")
    identities = [
        (
            row["variant_id"], row["model_seed"], row["arm"], row["anchor"],
            row["geometry_group"],
        )
        for row in rows
    ]
    audit.equal(len(set(identities)), len(identities), "component row identity uniqueness", section="component_coverage")
    expected_axis = tuple(payload["coverage"]["component_axis"])
    reference_fold: dict[int, int] | None = None
    for variant in VARIANTS:
        for seed in SEEDS:
            for arm in ARMS:
                for anchor in (False, True):
                    selected = [
                        row for row in rows
                        if row["variant_id"] == variant
                        and row["model_seed"] == seed
                        and row["arm"] == arm
                        and row["anchor"] == anchor
                    ]
                    axis = tuple(sorted(int(row["geometry_group"]) for row in selected))
                    audit.equal(axis, expected_axis, f"axis {variant}/{seed}/{arm}/{anchor}", section="component_coverage")
                    fold_map = {int(row["geometry_group"]): int(row["fold"]) for row in selected}
                    if reference_fold is None:
                        reference_fold = fold_map
                    else:
                        audit.equal(fold_map, reference_fold, f"fold map {variant}/{seed}/{arm}/{anchor}", section="component_coverage")
    for variant in VARIANTS:
        for group in expected_axis:
            counts = {
                int(row["cell_count"])
                for row in rows
                if row["variant_id"] == variant and int(row["geometry_group"]) == group
            }
            audit.equal(len(counts), 1, f"cell-count invariance {variant}/{group}", section="component_coverage")


def audit_identity_and_schema(
    audit: AuditRecorder, payload: Mapping[str, Any]
) -> None:
    expected_top = {
        "schema_version", "campaign_id", "report_name", "contract_sha256",
        "launch_manifest_sha256", "coverage", "attempt_history", "scientific_ids",
        "design_notes", "variants", "cross_variant_classification",
        "prepared_graph_audits", "budget_attribution", "gene_label_null",
        "fixed_randomization", "exact_row_counts", "claim_limits",
    }
    audit.equal(set(payload), expected_top, "aggregate top-level schema", section="identity")
    audit.equal(payload.get("schema_version"), 1, "aggregate schema version", section="identity")
    audit.equal(payload.get("campaign_id"), CAMPAIGN_ID, "campaign ID", section="identity")
    audit.equal(payload.get("contract_sha256"), CONTRACT_SHA256, "contract SHA", section="identity")
    audit.equal(payload.get("launch_manifest_sha256"), LAUNCH_SHA256, "launch SHA", section="identity")
    coverage = payload["coverage"]
    expected_coverage = {
        "variant_ids": list(VARIANTS), "model_seeds": list(SEEDS), "folds": list(FOLDS),
        "expected_jobs": 140, "verified_jobs": 140, "declared_attempts": 140,
        "superseded_unsuccessful_attempts": 0, "failed_jobs": 0,
        "duplicate_slots": 0, "component_count": 27,
        "eligible_gene_count": 932, "gene_count": 1000,
    }
    for field, value in expected_coverage.items():
        audit.equal(coverage.get(field), value, f"coverage/{field}", section="identity")
    audit.equal(set(payload["variants"]), set(VARIANTS), "variant inventory", section="identity")
    audit.equal(set(payload["scientific_ids"]), set(VARIANTS), "scientific ID inventory", section="identity")
    audit.equal(len(set(payload["scientific_ids"].values())), 7, "scientific IDs unique", section="identity")
    attempts = payload["attempt_history"]
    audit.equal(len(attempts), 140, "attempt-history count", section="identity")
    slots = set()
    for row in attempts:
        slot = (row.get("variant_id"), row.get("model_seed"), row.get("fold"))
        slots.add(slot)
        audit.require(
            row.get("attempt") == 1 and row.get("selected") is True
            and row.get("registry_status") == "completed"
            and row.get("artifact_status") == "success",
            f"attempt row {slot}", section="identity", observed=row,
        )
    audit.equal(
        slots,
        {(variant, seed, fold) for variant in VARIANTS for seed in SEEDS for fold in FOLDS},
        "attempt slot coverage",
        section="identity",
    )


def audit_predictions(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    reference_groups = prediction_array(rows, variant="V0", arm="morphology_only", anchor=False)[0]
    resamples = bootstrap_indices(reference_groups, draws=20_000, seed=20260821)
    for variant in VARIANTS:
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        encoded = payload["variants"][variant]
        for arm in ARMS:
            groups, values, counts, folds = prediction_array(
                rows, variant=variant, arm=arm, anchor=False
            )
            arrays[arm] = (groups, values, counts, folds)
            expected_arm = {
                "component_equal_mse": float(np.mean(values)),
                "cell_weighted_mse": float(np.average(values, weights=counts)),
                "fold_equal_mse": float(np.mean([np.mean(values[folds == fold]) for fold in FOLDS])),
                "slide_equal_mse": float(np.mean([np.mean(values[groups // 100 == code]) for code in (1, 2)])),
                "component_count": 27,
                "cell_count": int(round(float(np.sum(counts)))),
            }
            observed_arm = encoded["prediction"]["arms"][arm]
            audit.equal(set(observed_arm), set(expected_arm), f"prediction fields {variant}/{arm}", section="prediction")
            for field, value in expected_arm.items():
                if isinstance(value, int):
                    audit.equal(observed_arm.get(field), value, f"prediction {variant}/{arm}/{field}", section="prediction")
                else:
                    audit.close(observed_arm.get(field), value, f"prediction {variant}/{arm}/{field}", section="prediction")
        groups, morphology, counts, folds = arrays["morphology_only"]
        near = arrays["observed_near"][1]
        permuted = arrays["within_fov_permuted_near"][1]
        fold_morph = np.asarray([np.mean(morphology[folds == fold]) for fold in FOLDS])
        fold_near = np.asarray([np.mean(near[folds == fold]) for fold in FOLDS])
        slide_morph = np.asarray([np.mean(morphology[groups // 100 == code]) for code in (1, 2)])
        slide_near = np.asarray([np.mean(near[groups // 100 == code]) for code in (1, 2)])
        morph_values = {
            "component_equal_relative_gain": relative_gain(morphology, near),
            "cell_weighted_relative_gain": relative_gain(morphology, near, counts),
            "fold_equal_relative_gain": relative_gain(fold_morph, fold_near),
            "slide_equal_relative_gain": relative_gain(slide_morph, slide_near),
        }
        perm_gain = relative_gain(permuted, near)
        for field, value in morph_values.items():
            audit.close(encoded["prediction"]["near_vs_morphology"].get(field), value, f"near-morph {variant}/{field}", section="prediction")
        audit.close(encoded["prediction"]["near_vs_permutation"].get("component_equal_relative_gain"), perm_gain, f"near-perm {variant}/gain", section="prediction")
        for name, baseline, candidate in (
            ("near_vs_morphology", morphology, near),
            ("near_vs_permutation", permuted, near),
        ):
            draws = gain_draws(baseline, candidate, resamples)
            expected_bootstrap = {
                "draws": 20_000,
                "seed": 20260821,
                "point": relative_gain(baseline, candidate),
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
                "positive_draw_fraction": float(np.mean(draws > 0)),
                "slide_stratified": True,
            }
            observed_bootstrap = encoded["prediction"][name]["bootstrap"]
            for field, value in expected_bootstrap.items():
                if isinstance(value, (bool, int)):
                    audit.equal(observed_bootstrap.get(field), value, f"bootstrap {variant}/{name}/{field}", section="bootstrap")
                else:
                    audit.close(observed_bootstrap.get(field), value, f"bootstrap {variant}/{name}/{field}", section="bootstrap")
        morph_folds = int(sum(np.mean(near[folds == fold]) < np.mean(morphology[folds == fold]) for fold in FOLDS))
        perm_folds = int(sum(np.mean(near[folds == fold]) < np.mean(permuted[folds == fold]) for fold in FOLDS))
        for gate, gain, fold_count, threshold in (
            ("near_vs_morphology_prediction", morph_values["component_equal_relative_gain"], morph_folds, 0.02),
            ("near_vs_permutation_prediction", perm_gain, perm_folds, 0.01),
        ):
            observed = encoded["consensus_gates"][gate]
            expected_pass = bool(gain >= threshold and fold_count >= 3)
            audit.close(observed.get("observed"), gain, f"consensus gate {variant}/{gate}/observed", section="prediction_gates")
            audit.equal(observed.get("folds_favoring"), fold_count, f"consensus gate {variant}/{gate}/folds", section="prediction_gates")
            audit.equal(observed.get("passed"), expected_pass, f"consensus gate {variant}/{gate}/pass", section="prediction_gates")
            audit.close(observed.get("margin"), gain - threshold, f"consensus gate {variant}/{gate}/margin", section="prediction_gates")
            audit.equal(observed.get("fold_margin"), fold_count - 3, f"consensus gate {variant}/{gate}/fold margin", section="prediction_gates")
        for seed in SEEDS:
            seed_arrays = {
                arm: prediction_array(rows, variant=variant, arm=arm, anchor=False, seed=seed)
                for arm in ("morphology_only", "observed_near", "within_fov_permuted_near")
            }
            seed_groups, seed_morph, _, seed_folds = seed_arrays["morphology_only"]
            seed_near = seed_arrays["observed_near"][1]
            seed_perm = seed_arrays["within_fov_permuted_near"][1]
            audit.require(np.array_equal(seed_groups, groups), f"seed group axis {variant}/{seed}", section="prediction_gates")
            for gate, baseline, threshold in (
                ("near_vs_morphology_prediction", seed_morph, 0.02),
                ("near_vs_permutation_prediction", seed_perm, 0.01),
            ):
                gain = relative_gain(baseline, seed_near)
                fold_count = int(sum(np.mean(seed_near[seed_folds == fold]) < np.mean(baseline[seed_folds == fold]) for fold in FOLDS))
                observed = encoded["seed_specific_gates"][str(seed)][gate]
                audit.close(observed.get("observed"), gain, f"seed gate {variant}/{seed}/{gate}/observed", section="prediction_gates")
                audit.equal(observed.get("folds_favoring"), fold_count, f"seed gate {variant}/{seed}/{gate}/folds", section="prediction_gates")
                audit.equal(observed.get("passed"), bool(gain >= threshold and fold_count >= 3), f"seed gate {variant}/{seed}/{gate}/pass", section="prediction_gates")
        derived[variant] = {
            "near_vs_morphology_component_gain": morph_values["component_equal_relative_gain"],
            "near_vs_morphology_cell_gain": morph_values["cell_weighted_relative_gain"],
            "near_vs_permutation_component_gain": perm_gain,
            "morphology_bootstrap_95": [
                encoded["prediction"]["near_vs_morphology"]["bootstrap"]["lower_95"],
                encoded["prediction"]["near_vs_morphology"]["bootstrap"]["upper_95"],
            ],
        }
    return derived


def audit_npz_and_jacobians(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    archive: Any,
    eligible_indices: np.ndarray,
    gene_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    expected_keys = expected_npz_keys()
    audit.equal(len(expected_keys), 147, "programmed NPZ key count", section="npz_schema")
    audit.equal(set(archive.files), expected_keys, "NPZ exact key inventory", section="npz_schema")
    summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    retained: dict[str, np.ndarray] = {}
    decomposition_errors: dict[str, float] = {}
    for variant in VARIANTS:
        for arm in MATRIX_ARMS:
            for label in LABELS:
                matrices: dict[str, np.ndarray] = {}
                for part in PARTS:
                    key = f"{variant}_{arm}_{label}_{part}"
                    matrix = np.asarray(archive[key])
                    audit.require(matrix.dtype == np.float64, f"{key} dtype", section="npz_schema", observed=str(matrix.dtype), expected="float64")
                    audit.equal(matrix.shape, (1000, 1000), f"{key} shape", section="npz_schema")
                    audit.require(bool(np.isfinite(matrix).all()), f"{key} finite", section="npz_schema")
                    summary = matrix_summary(matrix, eligible_indices)
                    summaries[(variant, arm, label, part)] = summary
                    encoded = payload["variants"][variant]["jacobian"]["consensus"][arm][label][part]
                    compare_summary(audit, encoded, summary, name=key, section="jacobian_summaries")
                    matrices[part] = matrix
                    if key in {
                        "V0_observed_near_selected_total",
                        "V0_observed_near_anchor12_total",
                    }:
                        retained[key] = matrix.copy()
                error = float(np.max(np.abs(matrices["total"] - matrices["linear"] - matrices["nonlinear"])))
                decomposition_errors[f"{variant}/{arm}/{label}"] = error
                audit.require(error <= 1e-12, f"decomposition {variant}/{arm}/{label}", section="jacobian_decomposition", observed=error, expected="<=1e-12")
    weighting_errors: dict[str, Any] = {}
    for variant in VARIANTS:
        mappings = (
            ("fold_equal_cell_weighted", f"{variant}_observed_near_selected_cell_weighted_total"),
            ("slide_equal_component_equal", f"{variant}_observed_near_selected_slide_equal_total"),
        )
        for json_label, key in mappings:
            matrix = np.asarray(archive[key])
            audit.require(matrix.dtype == np.float64 and matrix.shape == (1000, 1000) and bool(np.isfinite(matrix).all()), f"weighting matrix {key}", section="npz_schema")
            summary = matrix_summary(matrix, eligible_indices)
            encoded = payload["variants"][variant]["jacobian"]["weighting_sensitivities"][json_label]
            compare_summary(audit, encoded, summary, name=key, section="jacobian_summaries")
            weighting_errors[key] = summary
        component_summary = summaries[(variant, "observed_near", "selected", "total")]
        compare_summary(
            audit,
            payload["variants"][variant]["jacobian"]["weighting_sensitivities"]["fold_equal_component_equal"],
            component_summary,
            name=f"{variant}/fold_equal_component_equal",
            section="jacobian_summaries",
        )
    v0 = retained["V0_observed_near_selected_total"]
    selected = v0[np.ix_(eligible_indices, eligible_indices)]
    absolute = np.abs(selected)
    for local, row in enumerate(gene_rows):
        diagonal = float(selected[local, local])
        rank = 1 + int(np.sum(absolute[local] > absolute[local, local]))
        audit.close(row["signed_diagonal"], diagonal, f"eligible row {local}/signed", section="eligible_cross_format")
        audit.close(row["absolute_diagonal"], abs(diagonal), f"eligible row {local}/absolute", section="eligible_cross_format")
        audit.equal(row["absolute_row_rank"], rank, f"eligible row {local}/rank", section="eligible_cross_format")
        audit.equal(row["row_top1"], int(rank == 1), f"eligible row {local}/top1", section="eligible_cross_format")
        audit.equal(row["row_top1_percent"], int(rank <= 10), f"eligible row {local}/top1pct", section="eligible_cross_format")
    return summaries, retained, {
        "maximum_total_minus_linear_minus_nonlinear_abs_error": max(decomposition_errors.values()),
        "decomposition_errors": decomposition_errors,
    }


def audit_jacobian_gates_and_classifications(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    summaries: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    for variant in VARIANTS:
        encoded = payload["variants"][variant]
        near = summaries[(variant, "observed_near", "selected", "total")]
        perm = summaries[(variant, "within_fov_permuted_near", "selected", "total")]
        diag_gate = encoded["consensus_gates"]["same_name_diagonal_enrichment"]
        audit.close(diag_gate.get("observed"), near["diagonal_offdiagonal_ratio"], f"diag gate {variant}/observed", section="jacobian_gates")
        audit.equal(diag_gate.get("passed"), bool(near["diagonal_offdiagonal_ratio"] >= 2), f"diag gate {variant}/pass", section="jacobian_gates")
        row_gate = encoded["consensus_gates"]["strict_row_selectivity"]
        row_pass = bool(near["row_top1_fraction"] >= 0.25 and near["row_top1_percent_fraction"] >= 0.50)
        audit.close(row_gate.get("row_top1_fraction"), near["row_top1_fraction"], f"row gate {variant}/top1", section="jacobian_gates")
        audit.close(row_gate.get("row_top1_percent_fraction"), near["row_top1_percent_fraction"], f"row gate {variant}/top1pct", section="jacobian_gates")
        audit.equal(row_gate.get("row_top1_count"), near["row_top1_count"], f"row gate {variant}/top1 count", section="jacobian_gates")
        audit.equal(row_gate.get("row_top1_percent_count"), near["row_top1_percent_count"], f"row gate {variant}/top1pct count", section="jacobian_gates")
        audit.equal(row_gate.get("passed"), row_pass, f"row gate {variant}/pass", section="jacobian_gates")
        ratio = float(near["median_absolute_diagonal"] / perm["median_absolute_diagonal"])
        perm_gate = encoded["consensus_gates"]["near_vs_permutation_diagonal"]
        audit.close(perm_gate.get("observed"), ratio, f"near-perm diag {variant}/ratio", section="jacobian_gates")
        fold_ratios = perm_gate.get("fold_ratios", [])
        folds_at = int(sum(float(value) >= 1.25 for value in fold_ratios))
        audit.equal(len(fold_ratios), 4, f"near-perm diag {variant}/fold-ratio count", section="jacobian_gate_internal")
        audit.equal(perm_gate.get("folds_at_or_above"), folds_at, f"near-perm diag {variant}/fold count", section="jacobian_gate_internal")
        audit.equal(perm_gate.get("passed"), bool(ratio >= 1.25 and folds_at >= 3), f"near-perm diag {variant}/pass", section="jacobian_gate_internal")
        stability = encoded["consensus_gates"]["fold_stability"]
        correlations = stability.get("pairwise_signed_diagonal_spearman", [])
        audit.equal(len(correlations), 6, f"stability {variant}/pair count", section="jacobian_gate_internal")
        if len(correlations) == 6:
            audit.close(stability.get("median_pairwise_signed_diagonal_spearman"), float(np.median(correlations)), f"stability {variant}/median", section="jacobian_gate_internal")
        audit.close(stability.get("sign_consistent_gene_fraction"), float(stability.get("genes_same_sign_in_at_least_3_folds", -1)) / 932, f"stability {variant}/sign fraction", section="jacobian_gate_internal")
        stability_pass = bool(
            float(stability.get("median_pairwise_signed_diagonal_spearman", -math.inf)) >= 0.70
            and float(stability.get("sign_consistent_gene_fraction", -math.inf)) >= 0.75
        )
        audit.equal(stability.get("passed"), stability_pass, f"stability {variant}/pass", section="jacobian_gate_internal")
        for gate in GATES:
            seed_passes = int(sum(bool(encoded["seed_specific_gates"][str(seed)][gate]["passed"]) for seed in SEEDS))
            classification = encoded["gate_classification"][gate]
            consensus_pass = bool(encoded["consensus_gates"][gate]["passed"])
            audit.equal(classification.get("seed_specific_pass_count"), seed_passes, f"classification {variant}/{gate}/seed passes", section="classification")
            audit.equal(classification.get("seed_specific_fail_count"), 5 - seed_passes, f"classification {variant}/{gate}/seed fails", section="classification")
            audit.equal(classification.get("consensus_passed"), consensus_pass, f"classification {variant}/{gate}/consensus", section="classification")
            audit.equal(classification.get("classification"), three_way(consensus_pass, seed_passes), f"classification {variant}/{gate}/label", section="classification")
        derived[variant] = {
            "diagonal_offdiagonal_ratio": near["diagonal_offdiagonal_ratio"],
            "row_top1_count": near["row_top1_count"],
            "row_top1_fraction": near["row_top1_fraction"],
            "row_top1_percent_count": near["row_top1_percent_count"],
            "row_top1_percent_fraction": near["row_top1_percent_fraction"],
            "positive_diagonal_count": near["signed_diagonal_positive_count"],
            "positive_diagonal_fraction": near["positive_diagonal_fraction"],
            "near_permuted_diagonal_ratio": ratio,
            "gate_classification": {
                gate: encoded["gate_classification"][gate]["classification"] for gate in GATES
            },
        }
    for gate in GATES:
        observed = payload["cross_variant_classification"][gate]
        labels = {
            variant: payload["variants"][variant]["gate_classification"][gate]["classification"]
            for variant in PRIMARY_VARIANTS
        }
        unique = set(labels.values())
        expected_label = "robust_across_preprocessing" if len(unique) == 1 else "preprocessing_sensitive"
        shared = next(iter(unique)) if len(unique) == 1 else None
        audit.equal(observed.get("classification"), expected_label, f"cross variant {gate}/classification", section="cross_variant")
        audit.equal(observed.get("shared_gate_classification"), shared, f"cross variant {gate}/shared label", section="cross_variant")
        audit.equal(observed.get("variant_classifications"), labels, f"cross variant {gate}/labels", section="cross_variant")
        audit.equal(observed.get("all_six_identical"), len(unique) == 1, f"cross variant {gate}/identical", section="cross_variant")
        audit.equal(observed.get("majority_vote_used"), False, f"cross variant {gate}/majority", section="cross_variant")
        audit.equal(observed.get("V6_mechanistic_secondary"), payload["variants"]["V6"]["gate_classification"][gate]["classification"], f"cross variant {gate}/V6", section="cross_variant")
    return derived


def audit_budget(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    archive: Any,
    summaries: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    budget = payload["budget_attribution"]
    selected_groups, selected_morph, _, selected_folds = prediction_array(rows, variant="V0", arm="morphology_only", anchor=False)
    selected_near = prediction_array(rows, variant="V0", arm="observed_near", anchor=False)[1]
    anchor_groups, anchor_morph, _, anchor_folds = prediction_array(rows, variant="V0", arm="morphology_only", anchor=True)
    anchor_near = prediction_array(rows, variant="V0", arm="observed_near", anchor=True)[1]
    audit.require(np.array_equal(selected_groups, anchor_groups) and np.array_equal(selected_folds, anchor_folds), "budget selected/anchor axes", section="budget")
    selected_gain = relative_gain(selected_morph, selected_near)
    anchor_gain = relative_gain(anchor_morph, anchor_near)
    audit.close(budget.get("selected_prediction_gain"), selected_gain, "budget selected gain", section="budget")
    audit.close(budget.get("anchor12_prediction_gain"), anchor_gain, "budget anchor gain", section="budget")
    resamples = bootstrap_indices(selected_groups, draws=20_000, seed=20260821)
    recomputed_draws = np.asarray(
        [
            relative_gain(selected_morph[index], selected_near[index])
            - relative_gain(anchor_morph[index], anchor_near[index])
            for index in resamples
        ],
        dtype=np.float64,
    )
    stored_draws = np.asarray(archive["V0_budget_selected_minus_anchor12_bootstrap"])
    audit.require(stored_draws.dtype == np.float64 and stored_draws.shape == (20_000,) and bool(np.isfinite(stored_draws).all()), "budget NPZ shape/dtype/finite", section="budget")
    max_draw_error = float(np.max(np.abs(stored_draws - recomputed_draws)))
    audit.require(max_draw_error <= 1e-15, "budget bootstrap draw reproduction", section="budget", observed=max_draw_error, expected="<=1e-15")
    encoded_bootstrap = budget["selected_minus_anchor12_gain_bootstrap"]
    bootstrap_expected = {
        "draws": 20_000,
        "seed": 20260821,
        "point_selected_minus_anchor12_gain": selected_gain - anchor_gain,
        "lower_95": float(np.quantile(stored_draws, 0.025)),
        "upper_95": float(np.quantile(stored_draws, 0.975)),
        "positive_draw_fraction": float(np.mean(stored_draws > 0)),
        "slide_stratified": True,
    }
    for field, value in bootstrap_expected.items():
        if isinstance(value, (bool, int)):
            audit.equal(encoded_bootstrap.get(field), value, f"budget bootstrap/{field}", section="budget")
        else:
            audit.close(encoded_bootstrap.get(field), value, f"budget bootstrap/{field}", section="budget")
    fold_counts: dict[str, int] = {}
    for seed in SEEDS:
        _, seed_selected, _, seed_folds = prediction_array(rows, variant="V0", arm="observed_near", anchor=False, seed=seed)
        _, seed_anchor, _, anchor_seed_folds = prediction_array(rows, variant="V0", arm="observed_near", anchor=True, seed=seed)
        audit.require(np.array_equal(seed_folds, anchor_seed_folds), f"budget fold axis/{seed}", section="budget")
        fold_counts[str(seed)] = int(sum(np.mean(seed_selected[seed_folds == fold]) < np.mean(seed_anchor[seed_folds == fold]) for fold in FOLDS))
    audit.equal(budget.get("selected_seed_folds_near_better"), fold_counts, "budget selected-near fold support", section="budget")
    prediction_support = sum(value >= 3 for value in fold_counts.values()) >= 4
    prediction_explained = bool(
        payload["variants"]["V0"]["consensus_gates"]["near_vs_morphology_prediction"]["passed"]
        and encoded_bootstrap["lower_95"] > 0
        and prediction_support
    )
    audit.equal(budget.get("prediction_explained"), prediction_explained, "budget prediction explained", section="budget")
    selected_summary = summaries[("V0", "observed_near", "selected", "total")]
    anchor_summary = summaries[("V0", "observed_near", "anchor12", "total")]
    compare_summary(audit, budget["selected_row_summary"], selected_summary, name="budget selected row", section="budget")
    compare_summary(audit, budget["anchor12_row_summary"], anchor_summary, name="budget anchor row", section="budget")
    row_seed_counts = budget.get("seed_folds_improving_both_row_metrics", {})
    audit.equal(set(row_seed_counts), {str(seed) for seed in SEEDS}, "budget row seed inventory", section="budget_internal")
    row_support = sum(int(value) >= 3 for value in row_seed_counts.values()) >= 4
    row_explained = bool(
        payload["variants"]["V0"]["consensus_gates"]["strict_row_selectivity"]["passed"]
        and selected_summary["row_top1_fraction"] > anchor_summary["row_top1_fraction"]
        and selected_summary["row_top1_percent_fraction"] > anchor_summary["row_top1_percent_fraction"]
        and row_support
    )
    audit.equal(budget.get("row_explained"), row_explained, "budget row explained", section="budget_internal")
    trajectories = budget.get("trajectories", [])
    audit.equal(len(trajectories), 40, "budget trajectory count", section="budget_internal")
    trajectory_slots = set()
    saturated_count = 0
    for row in trajectories:
        slot = (row.get("model_seed"), row.get("fold"), row.get("arm"))
        trajectory_slots.add(slot)
        selected_epoch = int(row.get("selected_epoch", -1))
        late_gain = float(row.get("literal_relative_gain_96_to_192", math.nan))
        saturated = bool(selected_epoch < 192 or late_gain <= 0.001)
        saturated_count += int(saturated)
        audit.require(selected_epoch in {12, 24, 48, 96, 192}, f"budget epoch {slot}", section="budget_internal")
        audit.equal(row.get("saturated"), saturated, f"budget saturation {slot}", section="budget_internal")
        audit.close(row.get("saturation_margin"), max(192 - selected_epoch, 0), f"budget saturation margin {slot}", section="budget_internal")
        audit.close(row.get("late_gain_margin"), 0.001 - late_gain, f"budget late margin {slot}", section="budget_internal")
    audit.equal(trajectory_slots, {(seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in ("morphology_only", "observed_near")}, "budget trajectory slots", section="budget_internal")
    audit.equal(budget.get("saturated_trajectory_count"), saturated_count, "budget saturated count", section="budget_internal")
    all_saturated = saturated_count == 40
    audit.equal(budget.get("all_trajectories_saturated"), all_saturated, "budget all saturated", section="budget_internal")
    strong = bool(all_saturated and prediction_explained and row_explained)
    verdict = "strong_budget_explanation" if strong else ("optimization_inconclusive" if not all_saturated else "budget_explanation_not_supported")
    audit.equal(budget.get("strong_budget_explanation"), strong, "budget strong flag", section="budget_internal")
    audit.equal(budget.get("verdict"), verdict, "budget verdict", section="budget_internal")
    return {
        "verdict": verdict,
        "all_trajectories_saturated": all_saturated,
        "saturated_trajectories": saturated_count,
        "selected_prediction_gain": selected_gain,
        "anchor12_prediction_gain": anchor_gain,
        "selected_minus_anchor12_gain": selected_gain - anchor_gain,
        "selected_minus_anchor12_bootstrap_95": [bootstrap_expected["lower_95"], bootstrap_expected["upper_95"]],
        "bootstrap_draw_max_abs_error": max_draw_error,
        "prediction_explained": prediction_explained,
        "row_explained": row_explained,
        "selected_seed_folds_near_better": fold_counts,
        "row_seed_support_source": "JSON internal only; seed/fold matrices are not in the four-file aggregate",
    }


def audit_nulls(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    archive: Any,
    v0_matrix: np.ndarray,
    eligible_indices: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(v0_matrix, dtype=np.float64)[np.ix_(eligible_indices, eligible_indices)]
    absolute = np.abs(selected)
    count = len(eligible_indices)
    ranks = np.empty((count, count), dtype=np.int32)
    for row in range(count):
        order = np.argsort(-absolute[row], kind="stable")
        sorted_values = absolute[row, order]
        rank_values = np.empty(count, dtype=np.int32)
        position = 0
        while position < count:
            end = position + 1
            while end < count and sorted_values[end] == sorted_values[position]:
                end += 1
            rank_values[order[position:end]] = position + 1
            position = end
        ranks[row] = rank_values
    threshold = 10
    observed = {
        "median_absolute_diagonal": float(np.median(np.diag(absolute))),
        "row_top1_fraction": float(np.mean(np.diag(ranks) <= 1)),
        "row_top1_percent_fraction": float(np.mean(np.diag(ranks) <= threshold)),
    }
    distributions: dict[tuple[str, str], np.ndarray] = {}
    derived: dict[str, Any] = {}
    for family in NULL_FAMILIES:
        family_payload = payload["gene_label_null"][family]
        audit.equal(family_payload.get("draws"), 10_000, f"null {family}/draws", section="null_summary")
        audit.equal(family_payload.get("base_seed"), 20261019, f"null {family}/seed", section="null_summary")
        audit.equal(family_payload.get("eligible_gene_count"), 932, f"null {family}/eligible", section="null_summary")
        family_derived: dict[str, Any] = {}
        for statistic in NULL_STATISTICS:
            key = f"gene_label_null_{family}_{statistic}"
            distribution = np.asarray(archive[key])
            distributions[(family, statistic)] = distribution
            audit.require(distribution.dtype == np.float64 and distribution.shape == (10_000,) and bool(np.isfinite(distribution).all()), f"null array {key}", section="npz_schema")
            encoded = family_payload["statistics"][statistic]
            audit.close(encoded.get("observed"), observed[statistic], f"null {family}/{statistic}/observed", section="null_summary")
            audit.close(encoded.get("null_mean"), float(np.mean(distribution)), f"null {family}/{statistic}/mean", section="null_summary")
            audit.close(encoded.get("null_95")[0], float(np.quantile(distribution, 0.025)), f"null {family}/{statistic}/q025", section="null_summary")
            audit.close(encoded.get("null_95")[1], float(np.quantile(distribution, 0.975)), f"null {family}/{statistic}/q975", section="null_summary")
            p_value = float((1 + np.sum(distribution >= observed[statistic])) / 10001)
            audit.close(encoded.get("upper_tail_p"), p_value, f"null {family}/{statistic}/p", section="null_summary")
            if statistic != "median_absolute_diagonal":
                scaled = distribution * 932
                audit.require(float(np.max(np.abs(scaled - np.rint(scaled)))) <= 1e-12, f"null {family}/{statistic} grid", section="null_summary")
            family_derived[statistic] = {
                "observed": observed[statistic],
                "null_mean": float(np.mean(distribution)),
                "null_95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
                "upper_tail_p": p_value,
            }
        derived[family] = family_derived
    rows = np.arange(count)
    full_recomputed = {name: np.empty(10_000, dtype=np.float64) for name in NULL_STATISTICS}
    for draw in range(10_000):
        mapping = deterministic_derangement(count, seed=20261019 + draw * 100_003)
        selected_absolute = absolute[rows, mapping]
        selected_rank = ranks[rows, mapping]
        full_recomputed["median_absolute_diagonal"][draw] = np.median(selected_absolute)
        full_recomputed["row_top1_fraction"][draw] = np.mean(selected_rank <= 1)
        full_recomputed["row_top1_percent_fraction"][draw] = np.mean(selected_rank <= threshold)
    maximum_error = 0.0
    for statistic in NULL_STATISTICS:
        error = float(np.max(np.abs(full_recomputed[statistic] - distributions[("full", statistic)])))
        maximum_error = max(maximum_error, error)
        audit.require(error <= 1e-15, f"full null exact regeneration/{statistic}", section="null_full_regeneration", observed=error, expected="<=1e-15")
    audit.equal(payload["gene_label_null"]["full"].get("stratum_count"), 1, "full null stratum count", section="null_full_regeneration")
    audit.equal(payload["gene_label_null"]["full"].get("minimum_stratum_size"), 932, "full null minimum stratum", section="null_full_regeneration")
    return {
        "families": derived,
        "full_family_distribution_max_abs_error": maximum_error,
        "matched_family_exact_regeneration": "not identifiable from the four files: prevalence, target_std, and stratum membership are not published",
        "null_95_semantics": "2.5% and 97.5% quantiles of the null distribution, not a confidence interval",
    }


def audit_graph_payload(audit: AuditRecorder, payload: Mapping[str, Any]) -> None:
    audits = payload["prepared_graph_audits"]
    audit.equal(set(audits), set(VARIANTS), "graph audit variants", section="graph_internal")
    for variant in VARIANTS:
        row = audits[variant]
        fixed_total = 0
        near_total = 0
        annular_total = 0
        for slide in ("SO_1", "SO_2"):
            item = row["slides"][slide]
            active = int(item["active_source_states"])
            fixed = int(item["fixed_source_states"])
            fixed_total += fixed
            near_total += int(item["near_cross_fov_directed_edges"])
            annular_total += int(item["annular_cross_fov_directed_edges"])
            audit.require(active > 0 and 0 <= fixed <= active, f"graph {variant}/{slide}/counts", section="graph_internal")
            audit.close(item["source_mapping_changed_fraction"], (active - fixed) / active, f"graph {variant}/{slide}/changed", section="graph_internal", rtol=0.0, atol=1e-15)
            audit.equal(sum(item["fixed_source_states_by_fov"].values()), fixed, f"graph {variant}/{slide}/fixed FOV sum", section="graph_internal")
            audit.equal(item["receiver_collisions"], 0, f"graph {variant}/{slide}/collisions", section="graph_internal")
            audit.equal(item["receiver_degree_preserved"], True, f"graph {variant}/{slide}/degree", section="graph_internal")
        audit.equal(row["total_fixed_source_states"], fixed_total, f"graph {variant}/fixed total", section="graph_internal")
        audit.equal(row["total_near_cross_fov_directed_edges"], near_total, f"graph {variant}/near total", section="graph_internal")
        audit.equal(row["total_annular_cross_fov_directed_edges"], annular_total, f"graph {variant}/annular total", section="graph_internal")
        expected_cross = variant in {"V1", "V5"}
        audit.equal(row["cross_fov_within_component_expected"], expected_cross, f"graph {variant}/expected cross", section="graph_internal")
        audit.equal(near_total > 0 and annular_total > 0, expected_cross, f"graph {variant}/cross mechanism", section="graph_internal")
    for variant in ("V2", "V5"):
        so2 = audits[variant]["slides"]["SO_2"]
        audit.equal(so2["active_source_states"], 237946, f"graph {variant}/SO2 active", section="graph_contract")
        audit.equal(so2["fixed_source_states"], 1, f"graph {variant}/SO2 unavoidable fixed", section="graph_contract")
        audit.equal(so2["fixed_source_states_by_fov"], {"245": 1}, f"graph {variant}/SO2 FOV245", section="graph_contract")


def markdown_report(result: Mapping[str, Any]) -> str:
    audit = result["audit"]
    lines = [
        "# Standalone audit: same-gene robustness multiverse v1",
        "",
        f"Overall: **{audit['status'].upper()}** — {audit['passed']}/{audit['checks']} checks passed; {audit['failed']} failed.",
        "",
        "This audit did not import the campaign analyzer or `spatial_benchmark`. It recomputed the identifiable numerical content from the four published aggregate files.",
        "",
        "## Variant-level reconstructed results",
        "",
        "| Variant | near/morph component gain | 95% component bootstrap | cell-weighted gain | near/permutation gain | diag/off | row top-1 | row top-1% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        prediction = result["prediction"][variant]
        jacobian = result["jacobian"][variant]
        interval = prediction["morphology_bootstrap_95"]
        lines.append(
            f"| {variant} | {100*prediction['near_vs_morphology_component_gain']:.4f}% | "
            f"[{100*interval[0]:.4f}%, {100*interval[1]:.4f}%] | "
            f"{100*prediction['near_vs_morphology_cell_gain']:.4f}% | "
            f"{100*prediction['near_vs_permutation_component_gain']:.4f}% | "
            f"{jacobian['diagonal_offdiagonal_ratio']:.4f} | "
            f"{jacobian['row_top1_count']}/932 ({jacobian['row_top1_fraction']:.4f}) | "
            f"{jacobian['row_top1_percent_count']}/932 ({jacobian['row_top1_percent_fraction']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Budget attribution",
            "",
            f"Verdict: `{result['budget']['verdict']}`; saturated trajectories: {result['budget']['saturated_trajectories']}/40.",
            f"Selected-minus-anchor12 prediction-gain difference: {100*result['budget']['selected_minus_anchor12_gain']:.4f}% "
            f"(bootstrap [{100*result['budget']['selected_minus_anchor12_bootstrap_95'][0]:.4f}%, {100*result['budget']['selected_minus_anchor12_bootstrap_95'][1]:.4f}%]).",
            "",
            "## Deterministic label null",
            "",
            "| Family | Statistic | Observed | null 95% | upper-tail p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for family in NULL_FAMILIES:
        for statistic in NULL_STATISTICS:
            item = result["gene_label_null"]["families"][family][statistic]
            lines.append(
                f"| {family} | {statistic} | {item['observed']:.8g} | "
                f"[{item['null_95'][0]:.8g}, {item['null_95'][1]:.8g}] | {item['upper_tail_p']:.8g} |"
            )
    lines.extend(
        [
            "",
            "The reported null 95% range is a reference interval for null draws, not a confidence interval. The label null is secondary and cannot rescue a frozen gate.",
            "",
            "## Identifiability boundary of the four-file audit",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in result["limitations"])
    if audit["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{item['section']}/{item['check']}`: observed={item['observed']!r}, expected={item['expected']!r}"
            for item in audit["failures"]
        )
    lines.append("")
    return "\n".join(lines)


def run(analysis_dir: Path) -> dict[str, Any]:
    audit = AuditRecorder()
    payload = strict_json(analysis_dir / "aggregate_results.json")
    component_rows = read_component_rows(analysis_dir / "component_metrics.csv", audit)
    gene_rows = read_gene_rows(analysis_dir / "eligible_gene_summary.csv", audit)
    audit_identity_and_schema(audit, payload)
    audit_component_coverage(audit, component_rows, payload)
    audit.equal(len(gene_rows), 932, "eligible row count", section="gene_schema")
    audit.equal(len(gene_rows), payload["exact_row_counts"]["eligible_gene_summary_csv"], "eligible row count vs JSON", section="gene_schema")
    gene_indices = np.asarray([row["gene_index"] for row in gene_rows], dtype=np.int64)
    audit.require(np.array_equal(gene_indices, np.sort(gene_indices)), "eligible indices sorted", section="gene_schema")
    audit.equal(len(np.unique(gene_indices)), 932, "eligible indices unique", section="gene_schema")
    audit.equal(len({row["gene"] for row in gene_rows}), 932, "eligible gene labels unique", section="gene_schema")
    mask = np.zeros(1000, dtype=bool)
    mask[gene_indices] = True
    mask_sha = sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
    audit.equal(mask_sha, ELIGIBLE_MASK_SHA256, "eligible mask SHA", section="gene_schema")
    prediction = audit_predictions(audit, payload, component_rows)
    with np.load(analysis_dir / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        summaries, retained, decomposition = audit_npz_and_jacobians(
            audit, payload, archive, gene_indices, gene_rows
        )
        jacobian = audit_jacobian_gates_and_classifications(audit, payload, summaries)
        budget = audit_budget(audit, payload, component_rows, archive, summaries)
        nulls = audit_nulls(
            audit,
            payload,
            archive,
            retained["V0_observed_near_selected_total"],
            gene_indices,
        )
    audit_graph_payload(audit, payload)
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "audit_kind": "standalone_four_file_numerical_audit",
        "independence": {
            "campaign_analyzer_imported": False,
            "spatial_benchmark_imported": False,
            "inputs": [
                "aggregate_results.json",
                "component_metrics.csv",
                "eligible_gene_summary.csv",
                "aggregate_jacobians.npz",
            ],
            "numeric_tolerance": {"rtol": RTOL, "atol": ATOL},
        },
        "audit": audit.summary(),
        "prediction": prediction,
        "jacobian": jacobian,
        "decomposition": decomposition,
        "budget": budget,
        "gene_label_null": nulls,
        "limitations": [
            "Seed/fold Jacobian matrices are not published, so seed-specific Jacobian gates, fold-ratio support, and fold stability cannot be reconstructed from these four files; only their encoded arithmetic and classification logic can be checked.",
            "Per-slide Jacobian matrices are not published, so the two per-slide summaries cannot be independently reconstructed here; the published slide-equal matrix is checked.",
            "The matched-null prevalence, target-standard-deviation covariates, and stratum memberships are not published, so its stored draw summaries are checked but its draw vector needs upstream prepared arrays/checkpoints for exact regeneration.",
            "Validation histories are not published, so each trajectory's 96-to-192 gain is checked for internal saturation logic but not recomputed from validation records.",
            "Seed/fold row matrices are not published, so the budget row-support counts are checked only for downstream logical consistency.",
            "The CSV has only the 932 eligible labels, so its index mask hash is verified but the complete frozen 1000-label gene-order hash requires an upstream checkpoint/prepared axis.",
            "Technical-control truth and component MAE source values require run bundles; this four-file audit checks only their published logical state, schemas, and finiteness.",
        ],
    }
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    result = run(arguments.analysis_dir.resolve(strict=True))
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_md.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    arguments.output_md.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "status": result["audit"]["status"],
        "checks": result["audit"]["checks"],
        "failed": result["audit"]["failed"],
        "output_json": str(arguments.output_json),
        "output_md": str(arguments.output_md),
    }, sort_keys=True))
    return 0 if result["audit"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify and aggregate the frozen V0--V6 same-gene robustness core.

The analyzer accepts the frozen contract, its launch manifest, and one or more
materialized core-full coordinator plans.  Coverage and every job marker are
verified before an output staging directory is created.  It never searches a
registry for a convenient successful run and never substitutes a retry that
was not named by the materialized plans.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = str(PROJECT_ROOT / "src")
if not sys.path or sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

from spatial_benchmark.environment_lock import (
    EnvironmentLockError,
    load_environment_lock,
    verify_environment_observation,
    verify_live_environment,
)
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_json, canonical_sha256, scientific_id
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.same_gene_jacobian import (
    deterministic_derangement,
    diagonal_summary,
)


CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
REPORT_NAME = "same_gene_robustness_multiverse_v1"
ANALYZER_RELATIVE_PATH = "scripts/analysis/analyze_same_gene_robustness.py"
BASE_ANALYZER_RELATIVE_PATH = "scripts/analysis/analyze_same_gene_nonlinear.py"
ENVIRONMENT_LOCK_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/environment_lock.json"
)
ENVIRONMENT_VERIFIER_RELATIVE_PATH = "src/spatial_benchmark/environment_lock.py"
CORE_VARIANTS = tuple(f"V{index}" for index in range(7))
PRIMARY_VARIANTS = CORE_VARIANTS[:6]
MODEL_SEEDS = (20260810, 20261810, 20262810, 20263810, 20264810)
FOLDS = (0, 1, 2, 3)
ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
)
MATRIX_ARMS = ARMS[1:]
MATRIX_PARTS = ("total", "linear", "nonlinear")
ANCHOR_EPOCH = 12
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260821
GENE_NULL_DRAWS = 10_000
GENE_NULL_SEED = 20261019
GENE_NULL_FAMILIES = ("full", "prevalence_sd_decile_matched")
GENE_NULL_STATISTICS = (
    "median_absolute_diagonal",
    "row_top1_fraction",
    "row_top1_percent_fraction",
)
EXPECTED_COMPONENTS = 27
EXPECTED_GENES = 1000
EXPECTED_ELIGIBLE = 932
JACOBIAN_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-12
RUN_VERIFICATION_FIELDS = (
    "variant_id", "model_seed", "fold", "attempt", "run_id",
    "scientific_id", "artifact_path", "config_sha256",
    "checkpoint_sha256", "jacobian_npz_sha256",
    "maximum_jacobian_reconstruction_error", "prediction_rows",
    "bundle_verified", "native_manifest_verified",
    "registry_hashes_verified", "marker_verified",
)
COMPONENT_METRIC_FIELDS = (
    "variant_id", "model_seed", "fold", "arm", "anchor",
    "geometry_group", "slide", "cell_count", "mse", "mae",
)
ELIGIBLE_GENE_FIELDS = (
    "gene_index", "gene", "signed_diagonal", "absolute_diagonal",
    "absolute_row_rank", "row_top1", "row_top1_percent",
)


class RobustnessAnalysisError(RuntimeError):
    """Raised when coverage, input evidence, aggregation, or publication fails."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import required module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_wrapper = _load_module(
    "bagm_robustness_wrapper_for_analysis",
    PROJECT_ROOT / "scripts/train/run_same_gene_robustness.py",
)
launcher = _load_module(
    "bagm_robustness_launcher_for_analysis",
    PROJECT_ROOT / "scripts/train/launch_same_gene_robustness.py",
)
base_analysis = _load_module(
    "bagm_nonlinear_analysis_primitives_for_robustness",
    PROJECT_ROOT / "scripts/analysis/analyze_same_gene_nonlinear.py",
)


@dataclass(frozen=True, slots=True)
class PlannedRun:
    variant_id: str
    model_seed: int
    fold: int
    attempt: int
    config_path: Path
    config_sha256: str
    marker_path: Path
    plan_path: Path
    materialized: Any

    @property
    def key(self) -> tuple[str, int, int]:
        return self.variant_id, self.model_seed, self.fold


@dataclass(slots=True)
class VerifiedRun:
    variant_id: str
    model_seed: int
    fold: int
    attempt: int
    run_id: str
    scientific_id: str
    artifact_path: Path
    result: dict[str, Any]
    configuration: dict[str, Any]
    genes: tuple[str, ...]
    eligible: np.ndarray
    matrices: dict[str, dict[str, dict[str, np.ndarray]]]
    component_matrices: dict[str, dict[int, np.ndarray]]
    target_std: np.ndarray
    check: dict[str, Any]


@dataclass(slots=True)
class PublicationExpectation:
    payload: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    checks: Sequence[Mapping[str, Any]]
    component_rows: Sequence[Mapping[str, Any]]
    gene_rows: Sequence[Mapping[str, Any]]
    provenance: Mapping[str, Any]
    report_markdown: str


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise RobustnessAnalysisError(f"{label} contains nonfinite value {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RobustnessAnalysisError(f"{label} duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except RobustnessAnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustnessAnalysisError(f"cannot read strict {label}: {path}") from error
    if not isinstance(value, dict):
        raise RobustnessAnalysisError(f"{label} must contain an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gene_hash(genes: Iterable[str]) -> str:
    return sha256(canonical_json(list(genes)).encode("utf-8")).hexdigest()


def _mask_hash(mask: np.ndarray) -> str:
    return sha256(
        np.asarray(mask, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobustnessAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RobustnessAnalysisError(f"{label} must be finite")
    return result


def _materialized_arguments(job: Any, *, project_root: Path) -> argparse.Namespace:
    cli = argparse.Namespace(
        config=job.config_path,
        variant_root=None,
        model_seed=None,
        fold=None,
        profile=None,
        attempt=None,
        launch_manifest=None,
        pilot_receipt=None,
        verify_job_marker=True,
    )
    return train_wrapper._materialize_cli_arguments(cli, project_root=project_root)


def _collect_plan_entries(
    plan_paths: Sequence[Path], *, project_root: Path = PROJECT_ROOT
) -> list[PlannedRun]:
    if not plan_paths:
        raise RobustnessAnalysisError("at least one materialized core plan is required")
    entries: list[PlannedRun] = []
    seen_job_ids: set[str] = set()
    for path in plan_paths:
        try:
            plan = launcher.load_plan(path, project_root=project_root)
        except Exception as error:
            raise RobustnessAnalysisError(f"invalid materialized plan: {path}") from error
        if sha256_file(plan.source_manifest_path) != plan.source_manifest_sha256:
            raise RobustnessAnalysisError(f"plan source manifest changed: {path}")
        for job in plan.jobs:
            if job.job_id in seen_job_ids:
                raise RobustnessAnalysisError(f"duplicate plan job_id: {job.job_id}")
            seen_job_ids.add(job.job_id)
            if sha256_file(job.config_path) != job.expected_config_sha256:
                raise RobustnessAnalysisError(f"job config SHA mismatch: {job.job_id}")
            materialized = train_wrapper._load_materialized_job(
                job.config_path, project_root=project_root
            )
            if materialized.profile != "full":
                raise RobustnessAnalysisError(f"non-full job in core plan: {job.job_id}")
            if materialized.job_success_marker != job.success_marker:
                raise RobustnessAnalysisError(f"job marker/config mismatch: {job.job_id}")
            variant = train_wrapper._verify_variant_root(
                materialized.variant_root, project_root=project_root
            )
            if variant.variant_id not in CORE_VARIANTS:
                raise RobustnessAnalysisError(
                    f"non-core variant in core plan: {variant.variant_id}"
                )
            entries.append(
                PlannedRun(
                    variant_id=variant.variant_id,
                    model_seed=int(materialized.model_seed),
                    fold=int(materialized.fold),
                    attempt=int(materialized.attempt),
                    config_path=materialized.config_path,
                    config_sha256=materialized.config_sha256,
                    marker_path=materialized.job_success_marker,
                    plan_path=plan.path,
                    materialized=materialized,
                )
            )
    return entries


def _validate_coverage(
    entries: Sequence[Any],
    *,
    variants: Sequence[str] = CORE_VARIANTS,
    seeds: Sequence[int] = MODEL_SEEDS,
    folds: Sequence[int] = FOLDS,
) -> list[Any]:
    expected = {(variant, int(seed), int(fold)) for variant in variants for seed in seeds for fold in folds}
    histories: dict[tuple[str, int, int], dict[int, Any]] = {}
    for entry in entries:
        key = (str(entry.variant_id), int(entry.model_seed), int(entry.fold))
        attempt = int(getattr(entry, "attempt", 1))
        by_attempt = histories.setdefault(key, {})
        if attempt in by_attempt:
            raise RobustnessAnalysisError(
                f"duplicate core coverage slot/attempt: {key}, attempt={attempt}"
            )
        by_attempt[attempt] = entry
    observed = set(histories)
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    if missing or extra:
        raise RobustnessAnalysisError(
            f"core coverage differs before scientific output: missing={missing}, extra={extra}"
        )
    selected: list[Any] = []
    for key in sorted(expected):
        attempts = sorted(histories[key])
        if attempts != list(range(1, max(attempts) + 1)):
            raise RobustnessAnalysisError(
                f"noncontiguous attempt history for {key}: {attempts}"
            )
        selected.append(histories[key][attempts[-1]])
    return selected


def _verify_superseded_attempts(
    declared: Sequence[PlannedRun],
    selected: Sequence[PlannedRun],
    *,
    registry: Registry,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Verify and report every declared attempt older than the selected one."""

    selected_attempt = {row.key: row.attempt for row in selected}
    evidence: list[dict[str, Any]] = []
    for row in sorted(
        declared,
        key=lambda value: (*value.key, value.attempt),
    ):
        if row.attempt == selected_attempt[row.key]:
            continue
        cli = argparse.Namespace(
            config=row.config_path,
            variant_root=None,
            model_seed=None,
            fold=None,
            profile=None,
            attempt=None,
            launch_manifest=None,
            pilot_receipt=None,
            verify_job_marker=True,
            abandon_incomplete_attempt=False,
        )
        arguments = train_wrapper._materialize_cli_arguments(
            cli, project_root=project_root
        )
        verified = train_wrapper._verify_inputs(
            arguments,
            project_root=project_root,
            verify_live_job_environment=False,
        )
        train_wrapper._patch_base_runner(
            verified,
            project_root=project_root,
            require_live_job_environment=False,
        )
        configuration = train_wrapper.runner._configuration(
            profile=arguments.profile,
            fold=arguments.fold,
            attempt=arguments.attempt,
        )
        rows = train_wrapper.runner._attempt_rows(
            registry,
            configuration=configuration,
            fold=arguments.fold,
            attempt=arguments.attempt,
        )
        if len(rows) != 1 or rows[0].get("status") not in {
            "failed",
            "cancelled",
            "pruned",
        }:
            raise RobustnessAnalysisError(
                f"superseded attempt is not uniquely terminal unsuccessful: "
                f"{row.key}, attempt={row.attempt}"
            )
        registry_row = rows[0]
        run_id = str(registry_row["run_id"])
        artifact = train_wrapper.runner.RunArchive.artifact_path_for(
            run_id, train_wrapper.runner.current_paths()
        )
        registered = Path(str(registry_row.get("artifact_path", ""))).resolve(
            strict=False
        )
        if artifact.is_symlink() or registered != artifact or not artifact.is_dir():
            raise RobustnessAnalysisError(
                f"superseded attempt artifact authority is unsafe: {run_id}"
            )
        try:
            bundle_check = train_wrapper.runner.verify_run_bundle(
                artifact, require_success_contract=False
            )
        except Exception as error:
            raise RobustnessAnalysisError(
                f"superseded attempt bundle verification failed: {run_id}"
            ) from error
        if bundle_check.get("status") not in {"failed", "pruned"}:
            raise RobustnessAnalysisError(
                f"superseded attempt artifact is not unsuccessful: {run_id}"
            )
        evidence.append(
            {
                "variant_id": row.variant_id,
                "model_seed": row.model_seed,
                "fold": row.fold,
                "attempt": row.attempt,
                "selected": False,
                "registry_status": str(registry_row["status"]),
                "artifact_status": str(bundle_check["status"]),
                "run_id": run_id,
                "artifact_path": str(artifact),
                "config_sha256": row.config_sha256,
                "plan_sha256": sha256_file(row.plan_path),
            }
        )
    return evidence


def _declared_attempt_authority(
    row: PlannedRun, *, project_root: Path
) -> dict[str, Any]:
    """Reconstruct one declared plan's exact registry authority offline."""

    cli = argparse.Namespace(
        config=row.config_path,
        variant_root=None,
        model_seed=None,
        fold=None,
        profile=None,
        attempt=None,
        launch_manifest=None,
        pilot_receipt=None,
        verify_job_marker=True,
        abandon_incomplete_attempt=False,
    )
    arguments = train_wrapper._materialize_cli_arguments(
        cli, project_root=project_root
    )
    verified = train_wrapper._verify_inputs(
        arguments,
        project_root=project_root,
        verify_live_job_environment=False,
    )
    train_wrapper._patch_base_runner(
        verified,
        project_root=project_root,
        require_live_job_environment=False,
    )
    configuration = train_wrapper.runner._configuration(
        profile=arguments.profile,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    return {
        "slot": (
            row.variant_id,
            row.model_seed % 1_000_000,
            row.fold,
            row.attempt,
        ),
        "scientific_id": scientific_id(configuration),
        "configuration": configuration,
    }


def _verify_registry_attempt_inventory(
    declared: Sequence[PlannedRun],
    *,
    registry: Registry,
    project_root: Path,
) -> None:
    """Reject full-campaign attempts absent from the immutable plan lineage.

    Pilot/resource-validation rows belong to the separately receipt-bound pilot
    lineage and are ignored here. Every full core row, regardless of status,
    must map to exactly one declared attempt with the same canonical scientific
    configuration. This prevents an unreported failed, orphaned, or successful
    attempt from disappearing merely because its plan was omitted at analysis.
    """

    expected: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in declared:
        authority = _declared_attempt_authority(row, project_root=project_root)
        slot = authority["slot"]
        if slot in expected:
            raise RobustnessAnalysisError(
                f"duplicate declared registry attempt authority: {slot}"
            )
        expected[slot] = authority

    with registry.connect() as connection:
        identities = connection.execute(
            "SELECT run_id FROM runs WHERE campaign_id = ? ORDER BY created_at, run_id",
            (CAMPAIGN_ID,),
        ).fetchall()
    observed: dict[tuple[str, int, int, int], list[str]] = {}
    for identity in identities:
        run_id = str(identity["run_id"])
        registry_row = registry.get_run(run_id)
        if registry_row is None:
            raise RobustnessAnalysisError(
                "registry attempt disappeared during inventory verification"
            )
        configuration = registry_row.get("config")
        if not isinstance(configuration, Mapping):
            raise RobustnessAnalysisError(
                f"campaign registry configuration is malformed: {run_id}"
            )
        evaluation = configuration.get("evaluation")
        protocol = evaluation.get("protocol") if isinstance(evaluation, Mapping) else None
        if protocol == "resource_validation":
            continue
        if protocol != "held_out_geometry_masked_reconstruction":
            raise RobustnessAnalysisError(
                f"campaign registry row has an undeclared execution protocol: {run_id}"
            )
        variant = configuration.get("robustness_variant")
        variant_id = variant.get("variant_id") if isinstance(variant, Mapping) else None
        slot = (
            str(variant_id),
            int(registry_row.get("seed", -1)),
            int(registry_row.get("fold", -1)),
            int(registry_row.get("attempt", -1)),
        )
        authority = expected.get(slot)
        if authority is None:
            raise RobustnessAnalysisError(
                f"registry contains a full attempt absent from declared plans: {slot}"
            )
        if (
            registry_row.get("scientific_id") != authority["scientific_id"]
            or canonical_json(configuration)
            != canonical_json(authority["configuration"])
        ):
            raise RobustnessAnalysisError(
                f"registry attempt differs from its declared plan authority: {slot}"
            )
        observed.setdefault(slot, []).append(run_id)

    missing = sorted(set(expected).difference(observed))
    duplicates = {
        slot: run_ids for slot, run_ids in observed.items() if len(run_ids) != 1
    }
    if missing or duplicates:
        raise RobustnessAnalysisError(
            "declared/registry attempt inventory differs: "
            f"missing={missing}, duplicates={duplicates}"
        )


def _reconstruct_jacobian_parts(
    state_dict: Mapping[str, torch.Tensor], hidden_derivative: np.ndarray
) -> dict[str, np.ndarray]:
    required = {
        "neighbor_linear.weight",
        "neighbor_in.weight",
        "neighbor_out.weight",
    }
    if not required.issubset(state_dict):
        raise RobustnessAnalysisError(
            f"checkpoint neighbor state is missing {sorted(required.difference(state_dict))}"
        )
    linear = state_dict["neighbor_linear.weight"].detach().cpu().double().numpy()
    incoming = state_dict["neighbor_in.weight"].detach().cpu().double().numpy()
    outgoing = state_dict["neighbor_out.weight"].detach().cpu().double().numpy()
    hidden = np.asarray(hidden_derivative, dtype=np.float64)
    if (
        linear.ndim != 2
        or incoming.ndim != 2
        or outgoing.ndim != 2
        or hidden.shape != (incoming.shape[0],)
        or outgoing.shape[1] != len(hidden)
        or linear.shape != (outgoing.shape[0], incoming.shape[1])
    ):
        raise RobustnessAnalysisError("checkpoint weights/hidden derivative shapes differ")
    nonlinear = (outgoing * hidden[None, :]) @ incoming
    total = linear + nonlinear
    if not all(np.isfinite(value).all() for value in (linear, nonlinear, total, hidden)):
        raise RobustnessAnalysisError("reconstructed Jacobian is nonfinite")
    return {"linear": linear, "nonlinear": nonlinear, "total": total}


def _assert_close(first: np.ndarray, second: np.ndarray, *, label: str) -> float:
    one = np.asarray(first, dtype=np.float64)
    two = np.asarray(second, dtype=np.float64)
    if one.shape != two.shape or not np.isfinite(one).all() or not np.isfinite(two).all():
        raise RobustnessAnalysisError(f"{label} shape/finite mismatch")
    maximum = float(np.max(np.abs(one - two))) if one.size else 0.0
    if maximum > JACOBIAN_TOLERANCE:
        raise RobustnessAnalysisError(
            f"{label} reconstruction differs by {maximum:.3g}"
        )
    return maximum


def _contract_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RobustnessAnalysisError(f"contract {label} must be a mapping")
    return value


def _contract_value(
    contract: Mapping[str, Any], path: Sequence[str], expected: Any
) -> None:
    value: Any = contract
    for name in path:
        value = _contract_mapping(value, label=".".join(path[:-1]) or "root").get(name)
    if value != expected:
        raise RobustnessAnalysisError(
            f"contract {'.'.join(path)} changed: {value!r} != {expected!r}"
        )


def _verify_analysis_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed when any analysis-defining contract field drifts."""

    required = {
        "campaign_id",
        "status",
        "launch_authorized",
        "dataset",
        "shared_split",
        "model_seeds",
        "arms_core",
        "variants",
        "production_matrix",
        "estimands",
        "frozen_gate_vector",
        "seed_robustness_classification",
        "cross_variant_classification",
        "budget_attribution",
        "post_core_secondary_analyses",
        "verification_and_publication",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise RobustnessAnalysisError(f"contract omits analysis fields: {missing}")
    if (
        contract["campaign_id"] != CAMPAIGN_ID
        or contract["status"] != "frozen_preoutcome"
        or contract["launch_authorized"] is not True
    ):
        raise RobustnessAnalysisError("contract is not the authorized frozen campaign")

    exact_values: tuple[tuple[tuple[str, ...], Any], ...] = (
        (("dataset", "genes"), EXPECTED_GENES),
        (("dataset", "frozen_common_eligible_gene_count"), EXPECTED_ELIGIBLE),
        (("shared_split", "outer_fold_count"), len(FOLDS)),
        (("shared_split", "outer_folds"), list(FOLDS)),
        (("shared_split", "component_count"), EXPECTED_COMPONENTS),
        (("shared_split", "validation_fold_expression"), "(outer_fold + 1) mod 4"),
        (("shared_split", "canonical_production_prediction_split"), "test"),
        (("model_seeds", "values"), list(MODEL_SEEDS)),
        (("model_seeds", "count"), len(MODEL_SEEDS)),
        (("model_seeds", "fold_seed_expression"), "seed_base + outer_fold"),
        (("arms_core",), list(ARMS)),
        (("estimands", "prediction_primary", "response_genes"), "all_1000"),
        (("estimands", "jacobian_primary", "genes"), "frozen_common_932"),
        (("estimands", "jacobian_primary", "component_weighting_within_fold"), "equal_component"),
        (("estimands", "jacobian_primary", "fold_aggregation"), "equal_fold_signed_mean_before_absolute_value"),
        (("estimands", "jacobian_primary", "seed_consensus"), "equal_seed_signed_mean_before_absolute_value"),
        (("frozen_gate_vector", "near_vs_morphology_prediction", "minimum_relative_component_equal_mse_gain"), 0.02),
        (("frozen_gate_vector", "near_vs_morphology_prediction", "minimum_folds_favoring_near"), 3),
        (("frozen_gate_vector", "near_vs_permutation_prediction", "minimum_relative_component_equal_mse_gain"), 0.01),
        (("frozen_gate_vector", "near_vs_permutation_prediction", "minimum_folds_favoring_near"), 3),
        (("frozen_gate_vector", "same_name_diagonal_enrichment", "minimum_ratio"), 2.0),
        (("frozen_gate_vector", "strict_row_selectivity", "minimum_row_top1_fraction"), 0.25),
        (("frozen_gate_vector", "strict_row_selectivity", "minimum_row_top1_percent_fraction"), 0.50),
        (("frozen_gate_vector", "near_vs_permutation_diagonal", "minimum_ratio"), 1.25),
        (("frozen_gate_vector", "near_vs_permutation_diagonal", "minimum_folds_at_or_above"), 3),
        (("frozen_gate_vector", "fold_stability", "minimum_median_pairwise_signed_diagonal_spearman"), 0.70),
        (("frozen_gate_vector", "fold_stability", "minimum_fraction_genes_same_sign_in_at_least_3_of_4_folds"), 0.75),
        (("frozen_gate_vector", "technical_validity", "all_outputs_finite"), True),
        (("frozen_gate_vector", "technical_validity", "receiver_expression_input"), False),
        (("frozen_gate_vector", "technical_validity", "exact_nonlinear_control"), True),
        (("frozen_gate_vector", "technical_validity", "identity_oracle_executed_row_top1_fraction"), 1.0),
        (("frozen_gate_vector", "technical_validity", "split_and_graph_invariants"), True),
        (("frozen_gate_vector", "technical_validity", "maximum_peak_vram_gb"), 20.5),
        (("production_matrix", "core", "variant_count"), len(CORE_VARIANTS)),
        (("production_matrix", "core", "primary_variant_ids"), list(PRIMARY_VARIANTS)),
        (("production_matrix", "core", "mechanistic_variant_ids"), ["V6"]),
        (("production_matrix", "core", "seeds_per_variant"), len(MODEL_SEEDS)),
        (("production_matrix", "core", "folds_per_seed"), len(FOLDS)),
        (("production_matrix", "core", "arms_per_process"), len(ARMS)),
        (("production_matrix", "core", "process_jobs"), 140),
        (("production_matrix", "core", "scientific_coverage_slots"), 140),
        (("production_matrix", "core", "selected_successful_runs"), 140),
        (
            ("production_matrix", "core", "process_job_semantics"),
            "planned_attempt_one_slots_not_total_retry_invocations",
        ),
        (("production_matrix", "core", "model_arm_fits"), 560),
        (("production_matrix", "totals", "scientific_coverage_slots"), 140),
        (("production_matrix", "totals", "selected_successful_runs"), 140),
        (
            ("pilots", "production_gate", "authorizes_exactly"),
            "140_core_scientific_coverage_slots_and_140_selected_successful_runs",
        ),
        (
            ("pilots", "production_gate", "recovery_attempts"),
            "additional_immutable_invocations_only_when_explicitly_declared",
        ),
        (
            ("pilots", "production_gate", "recovery_attempt_history"),
            "contiguous_and_fully_reported",
        ),
        (("post_core_secondary_analyses", "gene_label_null", "authorized"), True),
        (("post_core_secondary_analyses", "gene_label_null", "execution_gate"), "all_140_core_jobs_verified_before_any_null_outcome_access"),
        (("post_core_secondary_analyses", "gene_label_null", "additional_training_jobs"), 0),
        (("post_core_secondary_analyses", "gene_label_null", "phase_specific_gpu_pilot_required"), False),
        (("post_core_secondary_analyses", "gene_label_null", "source_variant"), "V0"),
        (("post_core_secondary_analyses", "gene_label_null", "source_arm"), "observed_near"),
        (("post_core_secondary_analyses", "gene_label_null", "source_checkpoint_label"), "selected"),
        (("post_core_secondary_analyses", "gene_label_null", "source_jacobian_part"), "total"),
        (("post_core_secondary_analyses", "gene_label_null", "source_matrix_aggregation"), "equal_signed_mean_over_5_seeds_and_4_folds"),
        (("post_core_secondary_analyses", "gene_label_null", "gene_population"), "frozen_common_932_target_rows_and_source_columns"),
        (("post_core_secondary_analyses", "gene_label_null", "draws_per_family"), GENE_NULL_DRAWS),
        (("post_core_secondary_analyses", "gene_label_null", "base_seed"), GENE_NULL_SEED),
        (("post_core_secondary_analyses", "gene_label_null", "derangement_seed_expression"), "base_seed + family_index*10000019 + draw_index*100003 + stratum_index"),
        (("post_core_secondary_analyses", "gene_label_null", "statistics"), list(GENE_NULL_STATISTICS)),
        (("verification_and_publication", "float_policy", "aggregate_dtype"), "float64"),
        (("verification_and_publication", "float_policy", "storage_dtype"), "float64"),
        (("verification_and_publication", "retries", "all_attempts_reported"), True),
        (
            (
                "verification_and_publication",
                "retries",
                "completed_attempt_selected_by_explicit_identity",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "retries",
                "attempt_number_above_one_must_not_poison_discovery",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "retries",
                "exactly_140_selected_successful_scientific_slots",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "retries",
                "additional_recovery_attempts_are_not_additional_scientific_slots",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "retries",
                "every_attempt_must_be_declared_in_an_immutable_plan",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "retries",
                "declared_plan_attempts_must_equal_registry_attempts",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "crash_recovery",
                "reconcile_final_directory_registry_and_success_marker",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "crash_recovery",
                "standalone_runs_supported",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "live_environment",
                "job_visibility",
                "visible_gpu_count",
            ),
            1,
        ),
        (
            (
                "verification_and_publication",
                "live_environment",
                "job_visibility",
                "cuda_smoke_test_required",
            ),
            True,
        ),
        (
            (
                "verification_and_publication",
                "live_environment",
                "launcher_and_analysis_visibility",
                "visible_gpu_count",
            ),
            4,
        ),
        (
            (
                "verification_and_publication",
                "live_environment",
                "scientific_configuration_binding",
                "exclude_process_local_receipt_sha_from_scientific_identity",
            ),
            True,
        ),
        (("budget_attribution", "primary_variant"), "V0"),
        (("budget_attribution", "evaluated_arms"), ["morphology_only", "observed_near"]),
        (("budget_attribution", "trajectory_count"), 40),
        (("budget_attribution", "all_trajectories_must_be_saturated"), True),
    )
    for path, expected in exact_values:
        _contract_value(contract, path, expected)

    dataset = _contract_mapping(contract["dataset"], label="dataset")
    for name in (
        "eligible_mask_sha256_uint8_c_order",
        "gene_order_sha256_canonical_json_utf8",
        "raw_fingerprint",
        "split_fingerprint",
    ):
        value = dataset.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RobustnessAnalysisError(f"contract dataset.{name} is not a SHA-256")

    variants = contract["variants"]
    if (
        not isinstance(variants, list)
        or any(not isinstance(row, Mapping) for row in variants)
        or [row.get("id") for row in variants] != list(CORE_VARIANTS)
    ):
        raise RobustnessAnalysisError("contract V0--V6 ordering changed")
    for index, row in enumerate(variants):
        expected_tier = "primary" if index < 6 else "mechanistic_secondary"
        if (
            not isinstance(row, Mapping)
            or row.get("tier") != expected_tier
            or row.get("seeds") != len(MODEL_SEEDS)
            or row.get("folds") != len(FOLDS)
            or row.get("process_jobs") != len(MODEL_SEEDS) * len(FOLDS)
        ):
            raise RobustnessAnalysisError(f"contract core inventory changed for V{index}")
    v0_equivalence = _contract_mapping(
        variants[0].get("antecedent_data_only_equivalence"),
        label="variants.V0.antecedent_data_only_equivalence",
    )
    if (
        v0_equivalence.get("observed_near_and_annular_arms")
        != "exact_antecedent_inputs"
        or v0_equivalence.get("permutation_control")
        != "corrected_degree_preserving_receiver_collision_free_rebuild"
        or v0_equivalence.get(
            "not_full_exact_replication_due_to_corrected_permutation_control"
        )
        is not True
        or v0_equivalence.get("antecedent_permuted_degree_mismatch_node_count")
        != {"SO_1": 1779, "SO_2": 2407}
        or v0_equivalence.get(
            "graph_native_eligibility_additional_receivers_not_used_by_primary"
        )
        != {"SO_1": 36, "SO_2": 17}
        or v0_equivalence.get("primary_receiver_cohort_remains_exact_antecedent")
        is not True
    ):
        raise RobustnessAnalysisError("contract V0 corrected-control identity changed")
    v6 = _contract_mapping(variants[6].get("residualization"), label="variants.V6.residualization")
    if (
        v6.get("cell_type_levels_expected") != 12
        or v6.get("training_coverage_rule")
        != "every_frozen_cell_type_level_must_occur_in_training"
        or v6.get("missing_training_level_action") != "fail_closed"
        or v6.get("receiver_cell_type_or_library_is_model_input") is not False
    ):
        raise RobustnessAnalysisError("contract V6 mechanism/coverage rule changed")

    bootstrap = _contract_mapping(
        _contract_mapping(contract["estimands"], label="estimands").get("bootstrap"),
        label="estimands.bootstrap",
    )
    if (bootstrap.get("draws"), bootstrap.get("seed")) != (
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    ):
        raise RobustnessAnalysisError("contract bootstrap definition changed")

    null_families = _contract_mapping(
        _contract_mapping(
            contract["post_core_secondary_analyses"],
            label="post_core_secondary_analyses",
        ).get("gene_label_null"),
        label="post_core_secondary_analyses.gene_label_null",
    ).get("families")
    if not isinstance(null_families, Mapping) or tuple(null_families) != GENE_NULL_FAMILIES:
        raise RobustnessAnalysisError("contract gene-label-null families changed")

    seed_rule = _contract_mapping(
        contract["seed_robustness_classification"],
        label="seed_robustness_classification",
    )
    if (
        seed_rule.get("robust_pass")
        != ["consensus_passes", {"minimum_seed_specific_passes": 4}, {"seed_count": 5}]
        or seed_rule.get("robust_gate_failure")
        != ["consensus_fails", {"minimum_seed_specific_failures": 4}, {"seed_count": 5}]
        or seed_rule.get("otherwise") != "seed_sensitive"
        or seed_rule.get("consensus_prediction")
        != "average_per_component_loss_over_seeds_before_ratio"
        or seed_rule.get("consensus_jacobian")
        != "equal_signed_mean_over_5_seeds_and_4_folds_before_absolute_value"
    ):
        raise RobustnessAnalysisError("contract four-of-five classification changed")

    cross = _contract_mapping(
        contract["cross_variant_classification"], label="cross_variant_classification"
    )
    if (
        cross.get("applies_to") != list(PRIMARY_VARIANTS)
        or cross.get("robust_across_preprocessing_requires")
        != "identical_three_way_classification_in_all_six_variants"
        or cross.get("mixed_result_label") != "preprocessing_sensitive"
        or cross.get("V6_use") != "mechanistic_secondary_only"
        or cross.get("majority_vote_forbidden") is not True
        or cross.get("best_variant_selection_forbidden") is not True
    ):
        raise RobustnessAnalysisError("contract cross-variant rule changed")

    budget = _contract_mapping(contract["budget_attribution"], label="budget_attribution")
    if (
        budget.get("saturated_if_either")
        != {"selected_epoch_below": 192, "literal_relative_gain_96_to_192_at_most": 0.001}
        or budget.get("prediction_explained_requires")
        != [
            "selected_consensus_passes_frozen_2_percent_gate",
            "bootstrap_lower_bound_gain_selected_minus_anchor12_above_zero",
            "at_least_4_of_5_seeds_have_selected_near_better_in_at_least_3_of_4_folds",
        ]
        or budget.get("row_explained_requires")
        != [
            "selected_consensus_passes_both_strict_row_thresholds",
            "selected_both_row_metrics_exceed_anchor12",
            "at_least_4_of_5_seeds_improve_both_metrics_in_at_least_3_of_4_fold_matrices",
        ]
    ):
        raise RobustnessAnalysisError("contract budget-attribution rule changed")


def _verify_campaign_inputs(
    contract_path: Path,
    launch_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Verify the immutable launch binding and the analysis-critical contract."""

    try:
        launch = train_wrapper._verify_launch_manifest(
            launch_path, project_root=project_root
        )
    except Exception as error:
        raise RobustnessAnalysisError("launch-manifest verification failed") from error
    try:
        supplied = contract_path.resolve(strict=True)
    except OSError as error:
        raise RobustnessAnalysisError("frozen contract is missing") from error
    if supplied != launch.contract_path or sha256_file(supplied) != launch.contract_sha256:
        raise RobustnessAnalysisError("contract argument differs from frozen launch contract")
    try:
        contract = train_wrapper._strict_yaml(supplied, label="analysis contract")
    except Exception as error:
        raise RobustnessAnalysisError("frozen contract is not strict YAML") from error
    _verify_analysis_contract(contract)
    required = {
        "campaign_id",
        "status",
        "launch_authorized",
        "dataset",
        "shared_split",
        "model_seeds",
        "variants",
        "estimands",
        "frozen_gate_vector",
        "seed_robustness_classification",
        "cross_variant_classification",
        "budget_attribution",
        "verification_and_publication",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise RobustnessAnalysisError(f"contract omits analysis fields: {missing}")
    if (
        contract["campaign_id"] != CAMPAIGN_ID
        or contract["status"] != "frozen_preoutcome"
        or contract["launch_authorized"] is not True
    ):
        raise RobustnessAnalysisError("contract is not the authorized frozen campaign")
    dataset = contract["dataset"]
    split = contract["shared_split"]
    seeds = contract["model_seeds"]
    estimands = contract["estimands"]
    if not isinstance(dataset, Mapping) or not isinstance(split, Mapping):
        raise RobustnessAnalysisError("contract dataset/shared_split must be mappings")
    expected_dataset = {
        "genes": EXPECTED_GENES,
        "frozen_common_eligible_gene_count": EXPECTED_ELIGIBLE,
    }
    for name, expected in expected_dataset.items():
        if dataset.get(name) != expected:
            raise RobustnessAnalysisError(f"contract dataset.{name} changed")
    for name in (
        "eligible_mask_sha256_uint8_c_order",
        "gene_order_sha256_canonical_json_utf8",
        "raw_fingerprint",
        "split_fingerprint",
    ):
        value = dataset.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RobustnessAnalysisError(f"contract dataset.{name} is not a SHA-256")
    if split.get("outer_folds") != list(FOLDS) or split.get("component_count") != EXPECTED_COMPONENTS:
        raise RobustnessAnalysisError("contract fold/component definition changed")
    if not isinstance(seeds, Mapping) or seeds.get("values") != list(MODEL_SEEDS):
        raise RobustnessAnalysisError("contract model seeds changed")
    variants = contract["variants"]
    if not isinstance(variants, list) or [row.get("id") for row in variants] != list(CORE_VARIANTS):
        raise RobustnessAnalysisError("contract V0--V6 ordering changed")
    bootstrap = estimands.get("bootstrap") if isinstance(estimands, Mapping) else None
    if not isinstance(bootstrap, Mapping) or (
        bootstrap.get("draws"), bootstrap.get("seed")
    ) != (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED):
        raise RobustnessAnalysisError("contract bootstrap definition changed")
    declared_sources = {str(row["path"]) for row in launch.sources}
    missing_analysis_sources = {
        ANALYZER_RELATIVE_PATH,
        BASE_ANALYZER_RELATIVE_PATH,
        ENVIRONMENT_LOCK_RELATIVE_PATH,
        ENVIRONMENT_VERIFIER_RELATIVE_PATH,
    }.difference(declared_sources)
    if missing_analysis_sources:
        raise RobustnessAnalysisError(
            "launch source manifest does not bind analysis sources: "
            f"{sorted(missing_analysis_sources)}"
        )
    verification = contract["verification_and_publication"]
    float_policy = verification.get("float_policy") if isinstance(verification, Mapping) else None
    if not isinstance(float_policy, Mapping) or (
        float_policy.get("aggregate_dtype"), float_policy.get("storage_dtype")
    ) != ("float64", "float64"):
        raise RobustnessAnalysisError("contract float64 aggregation/storage policy changed")
    try:
        environment_verification = verify_live_environment(
            project_root / ENVIRONMENT_LOCK_RELATIVE_PATH,
            visibility_mode="analysis",
        )
    except EnvironmentLockError as error:
        raise RobustnessAnalysisError(
            "analysis environment differs from the frozen environment lock"
        ) from error
    return contract, launch, environment_verification


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RobustnessAnalysisError(f"cannot read {label}: {path}") from error
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ))
        except (json.JSONDecodeError, ValueError) as error:
            raise RobustnessAnalysisError(f"invalid {label} row {index}") from error
        if not isinstance(row, dict):
            raise RobustnessAnalysisError(f"{label} row {index} is not an object")
        rows.append(row)
    if not rows:
        raise RobustnessAnalysisError(f"{label} is empty")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise RobustnessAnalysisError(f"cannot load Jacobian archive: {path}") from error


def _registry_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("config_json")
    if not isinstance(value, str):
        value = row.get("configuration_json")
    if not isinstance(value, str):
        raise RobustnessAnalysisError("registry row omits config_json")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise RobustnessAnalysisError("registry config_json is invalid") from error
    if not isinstance(result, dict):
        raise RobustnessAnalysisError("registry config_json is not an object")
    return result


def _verify_predictions_and_metric(
    bundle: Path,
    *,
    result: Mapping[str, Any],
    configuration: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    registry_detail: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = configuration.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise RobustnessAnalysisError("resolved configuration omits evaluation")
    primary_name = evaluation.get("primary_metric")
    if not isinstance(primary_name, str) or not primary_name.startswith("test/"):
        raise RobustnessAnalysisError("primary metric is not canonical test metric")
    final = _strict_json(bundle / "metrics/final.json", label="final metrics")
    if primary_name not in final:
        raise RobustnessAnalysisError("final metrics omit primary metric")
    primary = _finite_number(final[primary_name], label="primary metric")
    observed = result["arms"]["observed_near"]
    recorded = _finite_number(
        observed["evaluation"]["component_equal_mse"],
        label="results observed-near MSE",
    )
    if not np.isclose(primary, recorded, rtol=METRIC_TOLERANCE, atol=0.0):
        raise RobustnessAnalysisError("results.json and final metric disagree")
    rows = _read_jsonl(bundle / "predictions/test.jsonl", label="test predictions")
    groups: list[int] = []
    losses: list[float] = []
    expected_loss = {
        int(row["geometry_group"]): float(row["mse"])
        for row in observed["evaluation"]["per_component"]
    }
    for row in rows:
        try:
            group = int(str(row["sample_key"]).rsplit("_", 1)[1])
        except (KeyError, ValueError, IndexError) as error:
            raise RobustnessAnalysisError("prediction sample_key is invalid") from error
        if (
            row.get("run_id") != result["run_id"]
            or row.get("split") != "test"
            or int(row.get("fold", -1)) != int(result["outer_fold"])
            or group not in expected_loss
        ):
            raise RobustnessAnalysisError("prediction identity/split/component mismatch")
        loss = _finite_number(row.get("sample_loss"), label="prediction sample_loss")
        if not np.isclose(loss, expected_loss[group], rtol=METRIC_TOLERANCE, atol=0.0):
            raise RobustnessAnalysisError("prediction sample_loss differs from results")
        y_true = np.asarray(row.get("y_true"), dtype=np.float64)
        y_pred = np.asarray(row.get("y_pred"), dtype=np.float64)
        if y_true.shape != (EXPECTED_GENES,) or y_pred.shape != y_true.shape:
            raise RobustnessAnalysisError("prediction vectors have wrong gene axis")
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise RobustnessAnalysisError("prediction vectors are nonfinite")
        groups.append(group)
        losses.append(loss)
    if len(rows) != len(expected_loss) or len(set(groups)) != len(expected_loss):
        raise RobustnessAnalysisError("canonical predictions do not cover this fold's components")
    reproduced = float(np.mean(losses))
    if not np.isclose(reproduced, primary, rtol=METRIC_TOLERANCE, atol=0.0):
        raise RobustnessAnalysisError("canonical predictions do not reproduce primary")
    registered_primary = _finite_number(
        registry_row.get("primary_metric_value"), label="registry primary metric"
    )
    if registry_row.get("primary_metric_name") != primary_name or not np.isclose(
        registered_primary, primary, rtol=METRIC_TOLERANCE, atol=0.0
    ):
        raise RobustnessAnalysisError("registry run primary metric differs")
    metric_rows = [
        row
        for row in registry_detail.get("metrics", [])
        if row.get("name") == primary_name and row.get("split") == "test"
    ]
    if len(metric_rows) != 1 or not np.isclose(
        _finite_number(metric_rows[0].get("value"), label="registry metric row"),
        primary,
        rtol=METRIC_TOLERANCE,
        atol=0.0,
    ):
        raise RobustnessAnalysisError("registry metric table is incoherent")
    return {
        "primary_metric_name": primary_name,
        "primary_metric_value": primary,
        "prediction_rows": len(rows),
        "component_groups": sorted(groups),
    }


def _verify_phase_transform(
    variant_id: str,
    *,
    result: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    expected_kind = {"V4": "library", "V6": "cell_type_library"}.get(variant_id)
    frozen_v6_levels: tuple[str, ...] | None = None
    for arm in ARMS:
        result_arm = result["arms"][arm]
        state_arm = checkpoint["arms"][arm]
        result_metadata = result_arm.get("phase_transform")
        state_metadata = state_arm.get("phase_transform")
        if expected_kind is None:
            if result_metadata is not None or state_metadata is not None:
                raise RobustnessAnalysisError(
                    f"unexpected phase-transform metadata for {variant_id}/{arm}"
                )
            continue
        if not isinstance(result_metadata, Mapping) or result_metadata != state_metadata:
            raise RobustnessAnalysisError(
                f"phase-transform result/checkpoint metadata differs: {variant_id}/{arm}"
            )
        if (
            result_metadata.get("kind") != expected_kind
            or result_metadata.get("tuning_fit_mask") != "tuning_train"
            or result_metadata.get("final_fit_mask") != "final_train"
            or result_metadata.get("arm") != arm
        ):
            raise RobustnessAnalysisError(
                f"phase transform is not train-mask fitted: {variant_id}/{arm}"
            )
        for phase in ("tuning", "final"):
            fit = result_metadata.get(phase)
            if (
                not isinstance(fit, Mapping)
                or fit.get("kind") != expected_kind
                or isinstance(fit.get("training_row_count"), bool)
                or int(fit.get("training_row_count", 0)) <= 0
                or int(fit.get("training_component_count", 0)) <= 0
            ):
                raise RobustnessAnalysisError(
                    f"phase-transform fit metadata is incomplete: {variant_id}/{arm}/{phase}"
                )
            slope = np.asarray(fit.get("library_slope"), dtype=np.float64)
            if slope.shape != (EXPECTED_GENES,) or not np.isfinite(slope).all():
                raise RobustnessAnalysisError(
                    f"phase-transform library slope differs: {variant_id}/{arm}/{phase}"
                )
            if expected_kind == "library":
                intercept = np.asarray(fit.get("intercept"), dtype=np.float64)
                if intercept.shape != (EXPECTED_GENES,) or not np.isfinite(intercept).all():
                    raise RobustnessAnalysisError(
                        f"phase-transform intercept differs: {variant_id}/{arm}/{phase}"
                    )
                continue
            levels_value = fit.get("levels")
            levels = (
                tuple(levels_value)
                if isinstance(levels_value, list)
                and all(isinstance(value, str) and value for value in levels_value)
                else ()
            )
            masses = np.asarray(fit.get("type_weight_mass"), dtype=np.float64)
            intercepts = np.asarray(fit.get("type_intercepts"), dtype=np.float64)
            global_intercept = np.asarray(
                fit.get("global_intercept"), dtype=np.float64
            )
            if (
                len(levels) != 12
                or len(set(levels)) != 12
                or masses.shape != (12,)
                or not np.isfinite(masses).all()
                or np.any(masses <= 0)
                or intercepts.shape != (12, EXPECTED_GENES)
                or not np.isfinite(intercepts).all()
                or global_intercept.shape != (EXPECTED_GENES,)
                or not np.isfinite(global_intercept).all()
            ):
                raise RobustnessAnalysisError(
                    f"V6 frozen-level training coverage differs: {arm}/{phase}"
                )
            if frozen_v6_levels is None:
                frozen_v6_levels = levels
            elif frozen_v6_levels != levels:
                raise RobustnessAnalysisError("V6 frozen cell-type levels differ across fits")


def _verify_run_technical_controls(
    controls: Any, *, run_id: str
) -> dict[str, Any]:
    if not isinstance(controls, Mapping):
        raise RobustnessAnalysisError(f"technical controls are missing: {run_id}")
    analytical = controls.get("analytical_nonlinear_jacobian")
    if not isinstance(analytical, Mapping) or analytical.get("passed") is not True:
        raise RobustnessAnalysisError(f"analytical control failed: {run_id}")
    required_exact = {
        "all_outputs_finite": True,
        "train_validation_test_component_overlap": False,
        "receiver_expression_input": False,
        "receiver_rna_or_derived_covariate_model_input": False,
        "identity_oracle_actually_executed": True,
        "identity_oracle_row_top1_fraction": 1.0,
        "graph_specific_invariants": True,
        "source_config_data_hashes_verified": True,
        "checkpoint_replay_device_type": "cuda",
        "canonical_production_split_label": "test",
        "outer_test_untouched": True,
        "peak_vram_gate_passed": True,
        "environment_lock_verified": True,
        "environment_visibility_mode": "job",
    }
    for name, expected in required_exact.items():
        if controls.get(name) != expected:
            raise RobustnessAnalysisError(
                f"technical control {name} failed: {run_id}"
            )
    split = controls.get("split_overlap_control")
    if not isinstance(split, Mapping) or split.get("passed") is not True:
        raise RobustnessAnalysisError(f"split-overlap audit failed: {run_id}")
    bounds = {
        "maximum_autograd_error": (
            analytical.get("maximum_autograd_error"),
            1e-10,
        ),
        "maximum_finite_difference_error": (
            analytical.get("maximum_finite_difference_error"),
            1e-8,
        ),
        "checkpoint_gpu_replay_max_abs_metric_error": (
            controls.get("checkpoint_gpu_replay_max_abs_metric_error"),
            1e-7,
        ),
        "checkpoint_gpu_replay_max_abs_prediction_error": (
            controls.get("checkpoint_gpu_replay_max_abs_prediction_error"),
            1e-7,
        ),
        "peak_vram_gb": (controls.get("peak_vram_gb"), 20.5),
    }
    checked: dict[str, Any] = {}
    for name, (value, maximum) in bounds.items():
        observed = _finite_number(value, label=f"control {name}")
        if observed > maximum:
            raise RobustnessAnalysisError(
                f"technical control {name} exceeds {maximum}: {run_id}"
            )
        checked[name] = observed
    for name in ("environment_lock_sha256", "environment_verification_sha256"):
        value = controls.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RobustnessAnalysisError(
                f"technical control {name} is not a SHA-256: {run_id}"
            )
        checked[name] = value
    checked["environment_visibility_mode"] = "job"
    return checked


def _verify_archived_job_environment(
    bundle: Path,
    *,
    checked_controls: Mapping[str, Any],
    project_root: Path,
    run_id: str,
) -> None:
    """Verify an archived one-GPU receipt without probing analysis CUDA as job CUDA."""

    hardware = _strict_json(
        bundle / "provenance/hardware.json", label="run hardware provenance"
    )
    report = hardware.get("frozen_environment_verification")
    expected_keys = {
        "schema_version",
        "verified",
        "visibility_mode",
        "environment_lock_sha256",
        "observation",
        "verification_sha256",
    }
    if not isinstance(report, Mapping) or set(report) != expected_keys:
        raise RobustnessAnalysisError(
            f"archived job environment receipt is missing/malformed: {run_id}"
        )
    expected_lock_sha = sha256_file(
        project_root / ENVIRONMENT_LOCK_RELATIVE_PATH
    )
    if (
        report.get("schema_version") != 1
        or report.get("verified") is not True
        or report.get("visibility_mode") != "job"
        or report.get("environment_lock_sha256") != expected_lock_sha
        or report.get("environment_lock_sha256")
        != checked_controls["environment_lock_sha256"]
        or report.get("verification_sha256")
        != checked_controls["environment_verification_sha256"]
    ):
        raise RobustnessAnalysisError(
            f"archived job environment receipt identity differs: {run_id}"
        )
    payload = {
        key: value for key, value in report.items() if key != "verification_sha256"
    }
    if canonical_sha256(payload) != report["verification_sha256"]:
        raise RobustnessAnalysisError(
            f"archived job environment receipt digest differs: {run_id}"
        )
    try:
        _lock_path, lock = load_environment_lock(
            project_root / ENVIRONMENT_LOCK_RELATIVE_PATH
        )
        observation = report.get("observation")
        if not isinstance(observation, Mapping):
            raise EnvironmentLockError("archived environment observation is missing")
        verify_environment_observation(
            lock,
            observation,
            visibility_mode="job",
        )
    except EnvironmentLockError as error:
        raise RobustnessAnalysisError(
            f"archived job environment observation differs from the lock: {run_id}"
        ) from error


def _verify_one_run(
    planned: PlannedRun,
    *,
    contract: Mapping[str, Any],
    launch: Any,
    registry: Registry,
    project_root: Path = PROJECT_ROOT,
) -> VerifiedRun:
    """Reverify one planned run and independently replay its numerical evidence."""

    materialized = planned.materialized
    if materialized.launch_manifest != launch.path:
        raise RobustnessAnalysisError("plan job uses a different launch manifest")
    arguments = _materialized_arguments(materialized, project_root=project_root)
    try:
        verified_inputs = train_wrapper._verify_inputs(
            arguments,
            project_root=project_root,
            verify_live_job_environment=False,
        )
    except Exception as error:
        raise RobustnessAnalysisError(
            f"job source/contract/variant/receipt verification failed: {planned.key}"
        ) from error
    try:
        marker_check = train_wrapper.verify_job_marker(
            arguments,
            project_root=project_root,
            verify_live_job_environment=False,
        )
    except Exception as error:
        raise RobustnessAnalysisError(
            f"job marker verification failed: {planned.key}"
        ) from error
    if marker_check.get("config_sha256") != planned.config_sha256:
        raise RobustnessAnalysisError("verified job marker config SHA differs")
    bundle = Path(str(marker_check["artifact_path"])).resolve(strict=True)
    run_id = str(marker_check["run_id"])
    try:
        canonical = verify_run_bundle(bundle)
        native = base_analysis._verify_native_manifest(bundle)
    except Exception as error:
        raise RobustnessAnalysisError(f"run bundle verification failed: {run_id}") from error
    row = registry.get_run(run_id)
    detail = registry.show_run(run_id)
    if row is None or detail is None:
        raise RobustnessAnalysisError(f"registry run is missing: {run_id}")
    if (
        row.get("status") != "completed"
        or row.get("campaign_id") != CAMPAIGN_ID
        or int(row.get("seed", -1)) != planned.model_seed % 1_000_000
        or int(row.get("fold", -1)) != planned.fold
        or int(row.get("attempt", -1)) != planned.attempt
        or Path(str(row.get("artifact_path"))).resolve(strict=True) != bundle
    ):
        raise RobustnessAnalysisError(f"registry execution identity mismatch: {run_id}")
    issues = registry.verify_artifacts(run_id=run_id)
    if issues:
        raise RobustnessAnalysisError(f"registry artifact hashes failed: {issues}")

    result = _strict_json(bundle / "results.json", label="results")
    if (
        result.get("run_id") != run_id
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("profile") != "full"
        or result.get("status") != "completed"
        or int(result.get("outer_fold", -1)) != planned.fold
        or result.get("statistical_evaluation_role") != "outer_geometry_test"
        or set(result.get("arms", {})) != set(ARMS)
    ):
        raise RobustnessAnalysisError(f"results execution/test identity mismatch: {run_id}")
    for arm in ARMS:
        if result["arms"][arm].get("evaluation_role") != "test":
            raise RobustnessAnalysisError(f"arm is not outer-test evaluated: {run_id}/{arm}")
    expected_fingerprints = {
        "raw": verified_inputs.variant.raw_fingerprint,
        "processed": verified_inputs.variant.processed_fingerprint,
        "split": verified_inputs.variant.split_fingerprint,
    }
    if result.get("dataset_fingerprints") != expected_fingerprints:
        raise RobustnessAnalysisError(f"result fingerprint mismatch: {run_id}")

    try:
        checkpoint = torch.load(
            bundle / "checkpoints/last.ckpt", map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RobustnessAnalysisError(f"cannot load checkpoint: {run_id}") from error
    if not isinstance(checkpoint, Mapping):
        raise RobustnessAnalysisError("checkpoint is not a mapping")
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise RobustnessAnalysisError("checkpoint omits resolved configuration")
    try:
        archived_configuration = train_wrapper._strict_yaml(
            bundle / "config.resolved.yaml", label="archived resolved configuration"
        )
    except Exception as error:
        raise RobustnessAnalysisError("archived resolved configuration is invalid") from error
    if canonical_json(configuration) != canonical_json(archived_configuration):
        raise RobustnessAnalysisError("checkpoint and archived configurations differ")
    registry_configuration = _registry_configuration(row)
    if canonical_json(configuration) != canonical_json(registry_configuration):
        raise RobustnessAnalysisError("checkpoint and registry configurations differ")
    variant_config = configuration.get("robustness_variant")
    evaluation_config = configuration.get("evaluation")
    if (
        not isinstance(variant_config, Mapping)
        or variant_config.get("variant_id") != planned.variant_id
        or not isinstance(evaluation_config, Mapping)
        or evaluation_config.get("canonical_prediction_split") != "test"
        or evaluation_config.get("primary_eligibility_file")
        != verified_inputs.variant.eligibility_file
        or evaluation_config.get("frozen_gene_eligibility_file") != "eligible_genes.npy"
        or evaluation_config.get("frozen_gene_eligibility_sha256")
        != contract["dataset"]["eligible_mask_sha256_uint8_c_order"]
        or evaluation_config.get("frozen_gene_eligibility_count") != EXPECTED_ELIGIBLE
        or configuration.get("cohort", {}).get("primary_eligibility_file")
        != verified_inputs.variant.eligibility_file
    ):
        raise RobustnessAnalysisError("resolved robustness/evaluation configuration differs")
    computed_scientific_id = scientific_id(configuration)
    if row.get("scientific_id") != computed_scientific_id:
        raise RobustnessAnalysisError("registry scientific_id is not reproducible")
    if (
        checkpoint.get("run_id") != run_id
        or checkpoint.get("campaign_id") != CAMPAIGN_ID
        or checkpoint.get("frozen_contract_sha256") != launch.contract_sha256
        or set(checkpoint.get("arms", {})) != set(ARMS)
    ):
        raise RobustnessAnalysisError("checkpoint identity differs")
    _verify_phase_transform(
        planned.variant_id, result=result, checkpoint=checkpoint
    )

    genes = tuple(str(value) for value in checkpoint.get("genes", ()))
    dataset = contract["dataset"]
    if (
        len(genes) != EXPECTED_GENES
        or len(set(genes)) != EXPECTED_GENES
        or _gene_hash(genes) != dataset["gene_order_sha256_canonical_json_utf8"]
    ):
        raise RobustnessAnalysisError("checkpoint gene axis differs from frozen axis")
    arrays = _load_npz(bundle / "nonlinear_jacobians.npz")
    archive_genes = tuple(str(value) for value in arrays.get("genes", ()))
    if archive_genes != genes:
        raise RobustnessAnalysisError("NPZ and checkpoint gene axes differ")
    expected_mask_hash = dataset["eligible_mask_sha256_uint8_c_order"]
    frozen_mask_path = verified_inputs.variant.root / "eligible_genes.npy"
    try:
        prepared_frozen_mask = np.load(frozen_mask_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RobustnessAnalysisError(
            f"prepared frozen gene-eligibility mask is unreadable: {frozen_mask_path}"
        ) from error
    if (
        prepared_frozen_mask.shape != (EXPECTED_GENES,)
        or prepared_frozen_mask.dtype != np.bool_
        or int(prepared_frozen_mask.sum()) != EXPECTED_ELIGIBLE
        or _mask_hash(prepared_frozen_mask) != expected_mask_hash
    ):
        raise RobustnessAnalysisError("prepared frozen gene-eligibility mask differs")
    eligible: np.ndarray | None = None
    raw_eligible_counts: dict[str, int] = {}
    for arm in ARMS:
        mask = np.asarray(arrays.get(f"eligible_{arm}"), dtype=bool)
        result_mask = np.asarray(result["arms"][arm].get("eligible_genes"), dtype=bool)
        raw_result_mask = np.asarray(
            result["arms"][arm].get("raw_eligible_genes"), dtype=bool
        )
        gene_mse = np.asarray(result["arms"][arm].get("gene_mse"), dtype=np.float64)
        gene_pearson = np.asarray(
            result["arms"][arm].get("gene_pearson"), dtype=np.float64
        )
        state_arm = checkpoint["arms"][arm]
        state_mask_value = state_arm.get("eligible_genes")
        raw_state_value = state_arm.get("raw_eligible_genes")
        if not isinstance(state_mask_value, torch.Tensor) or not isinstance(
            raw_state_value, torch.Tensor
        ):
            raise RobustnessAnalysisError(
                f"checkpoint eligibility tensors are missing: {run_id}/{arm}"
            )
        state_mask = state_mask_value.detach().cpu().numpy().astype(bool, copy=False)
        raw_state_mask = raw_state_value.detach().cpu().numpy().astype(bool, copy=False)
        if (
            mask.shape != (EXPECTED_GENES,)
            or result_mask.shape != mask.shape
            or raw_result_mask.shape != mask.shape
            or state_mask.shape != mask.shape
            or raw_state_mask.shape != mask.shape
            or gene_mse.shape != (EXPECTED_GENES,)
            or gene_pearson.shape != (EXPECTED_GENES,)
            or not np.isfinite(gene_mse).all()
            or not np.isfinite(gene_pearson).all()
            or not np.array_equal(mask, result_mask)
            or not np.array_equal(mask, state_mask)
            or not np.array_equal(raw_result_mask, raw_state_mask)
            or int(mask.sum()) != EXPECTED_ELIGIBLE
            or _mask_hash(mask) != expected_mask_hash
            or not np.array_equal(mask, prepared_frozen_mask)
            or result["arms"][arm].get("eligibility_mode") != "frozen_common_mask"
            or state_arm.get("eligibility_mode") != "frozen_common_mask"
            or result["arms"][arm].get("eligible_gene_count") != EXPECTED_ELIGIBLE
            or np.any(mask & ~raw_result_mask)
        ):
            raise RobustnessAnalysisError(f"frozen eligible mask differs: {run_id}/{arm}")
        raw_eligible_counts[arm] = int(raw_result_mask.sum())
        if eligible is None:
            eligible = mask.copy()
        elif not np.array_equal(eligible, mask):
            raise RobustnessAnalysisError("arm eligibility masks differ")
    assert eligible is not None

    matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    component_diagonals: dict[str, dict[int, np.ndarray]] = {}
    maximum_reconstruction_error = 0.0
    for arm in MATRIX_ARMS:
        state = checkpoint["arms"][arm]
        if (
            not isinstance(state, Mapping)
            or state.get("fit_mask") != "final_train"
            or state.get("evaluation_mask") != "test"
            or int(state.get("selected_epoch", -1)) != int(result["arms"][arm]["selected_epoch"])
            or int(state.get("refit_epoch", -1)) != int(result["arms"][arm]["refit_epoch"])
        ):
            raise RobustnessAnalysisError(f"checkpoint train/test role differs: {run_id}/{arm}")
        history = result["arms"][arm].get("validation_history")
        if not isinstance(history, list) or [int(row["epoch"]) for row in history] != [12, 24, 48, 96, 192]:
            raise RobustnessAnalysisError(f"selection trajectory differs: {run_id}/{arm}")
        selected = min(history, key=lambda row: (float(row["component_equal_mse"]), int(row["epoch"])))
        if int(selected["epoch"]) != int(state["selected_epoch"]):
            raise RobustnessAnalysisError(f"selected epoch cannot be reproduced: {run_id}/{arm}")
        if int(state.get("anchor_epoch", -1)) != ANCHOR_EPOCH or state.get("anchor_state_dict") is None:
            raise RobustnessAnalysisError(f"anchor-12 state is missing: {run_id}/{arm}")
        arm_matrices: dict[str, dict[str, np.ndarray]] = {}
        component_diagonals[arm] = {}
        for label, state_key, prefix in (
            ("selected", "state_dict", ""),
            ("anchor12", "anchor_state_dict", f"anchor{ANCHOR_EPOCH}_"),
        ):
            hidden_key = f"{arm}_{prefix}mean_hidden_derivative"
            hidden = arrays.get(hidden_key)
            if hidden is None:
                raise RobustnessAnalysisError(f"hidden derivative missing: {run_id}/{hidden_key}")
            reconstructed = _reconstruct_jacobian_parts(state[state_key], hidden)
            arm_matrices[label] = {}
            for part in MATRIX_PARTS:
                array_key = f"{arm}_{prefix}{part}"
                value = np.asarray(arrays.get(array_key), dtype=np.float64)
                if value.shape != (EXPECTED_GENES, EXPECTED_GENES):
                    raise RobustnessAnalysisError(f"Jacobian shape differs: {array_key}")
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    _assert_close(reconstructed[part], value, label=f"{run_id}/{array_key}"),
                )
                arm_matrices[label][part] = value.astype(np.float64, copy=True)
            groups_key = f"{arm}_{prefix}component_geometry_groups"
            derivative_key = f"{arm}_{prefix}component_mean_hidden_derivative"
            groups = np.asarray(arrays.get(groups_key), dtype=np.int64)
            derivatives = np.asarray(arrays.get(derivative_key), dtype=np.float64)
            expected_groups = {
                int(row["geometry_group"])
                for row in result["arms"][arm]["evaluation"]["per_component"]
            }
            if (
                groups.ndim != 1
                or len(groups) != len(expected_groups)
                or set(int(value) for value in groups) != expected_groups
                or derivatives.ndim != 2
                or derivatives.shape[0] != len(groups)
            ):
                raise RobustnessAnalysisError(f"component derivatives differ: {run_id}/{arm}/{label}")
            component_total = []
            for group, derivative in zip(groups, derivatives, strict=True):
                parts = _reconstruct_jacobian_parts(state[state_key], derivative)
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    _assert_close(
                        parts["total"],
                        parts["linear"] + parts["nonlinear"],
                        label=f"{run_id}/{arm}/{label}/component{group}",
                    ),
                )
                component_total.append(parts["total"])
                if label == "selected":
                    component_diagonals[arm][int(group)] = np.diag(parts["total"]).copy()
            component_average = np.mean(component_total, axis=0)
            maximum_reconstruction_error = max(
                maximum_reconstruction_error,
                _assert_close(
                    component_average,
                    arm_matrices[label]["total"],
                    label=f"{run_id}/{arm}/{label}/equal-component-average",
                ),
            )
            if label == "selected" and arm == "observed_near":
                total_stack = np.stack(component_total, axis=0)
                count_by_group = {
                    int(row["geometry_group"]): int(row["cell_count"])
                    for row in result["arms"][arm]["evaluation"]["per_component"]
                }
                cell_weights = np.asarray(
                    [count_by_group[int(group)] for group in groups], dtype=np.float64
                )
                arm_matrices["selected_cell_weighted"] = {
                    "total": np.average(total_stack, axis=0, weights=cell_weights).astype(np.float64)
                }
                for slide_code, slide_name in ((1, "SO_1"), (2, "SO_2")):
                    selected_groups = (groups // 100) == slide_code
                    if not np.any(selected_groups):
                        raise RobustnessAnalysisError(
                            f"fold omits slide {slide_name}: {run_id}/{arm}"
                        )
                    arm_matrices[f"selected_{slide_name}"] = {
                        "total": np.mean(total_stack[selected_groups], axis=0).astype(np.float64)
                    }
        matrices[arm] = arm_matrices
    checked_controls = _verify_run_technical_controls(
        result.get("controls"), run_id=run_id
    )
    _verify_archived_job_environment(
        bundle,
        checked_controls=checked_controls,
        project_root=project_root,
        run_id=run_id,
    )
    campaign_configuration = configuration.get("campaign")
    expected_environment_lock_sha = sha256_file(
        project_root / ENVIRONMENT_LOCK_RELATIVE_PATH
    )
    if (
        not isinstance(campaign_configuration, Mapping)
        or campaign_configuration.get("environment_lock_sha256")
        != expected_environment_lock_sha
        or checked_controls["environment_lock_sha256"]
        != expected_environment_lock_sha
    ):
        raise RobustnessAnalysisError(
            f"run environment-lock provenance differs: {run_id}"
        )
    metric_check = _verify_predictions_and_metric(
        bundle,
        result=result,
        configuration=configuration,
        registry_row=row,
        registry_detail=detail,
    )
    target_std = checkpoint["arms"]["observed_near"].get("target_std")
    if not isinstance(target_std, torch.Tensor) or tuple(target_std.shape) != (EXPECTED_GENES,):
        raise RobustnessAnalysisError("checkpoint target standard deviation differs")
    return VerifiedRun(
        variant_id=planned.variant_id,
        model_seed=planned.model_seed,
        fold=planned.fold,
        attempt=planned.attempt,
        run_id=run_id,
        scientific_id=computed_scientific_id,
        artifact_path=bundle,
        result=result,
        configuration=configuration,
        genes=genes,
        eligible=eligible,
        matrices=matrices,
        component_matrices=component_diagonals,
        target_std=target_std.detach().cpu().double().numpy(),
        check={
            "canonical_bundle": canonical,
            "native_manifest": native,
            "marker": marker_check,
            "metric": metric_check,
            "checkpoint_sha256": sha256_file(bundle / "checkpoints/last.ckpt"),
            "jacobian_npz_sha256": sha256_file(bundle / "nonlinear_jacobians.npz"),
            "maximum_jacobian_reconstruction_error": maximum_reconstruction_error,
            "raw_eligible_gene_counts": raw_eligible_counts,
            "technical_controls": checked_controls,
        },
    )


def _verify_common_scientific_id(runs: Sequence[VerifiedRun]) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for variant in CORE_VARIANTS:
        values = {run.scientific_id for run in runs if run.variant_id == variant}
        if len(values) != 1:
            raise RobustnessAnalysisError(
                f"variant does not share one scientific_id across seed/fold: {variant}"
            )
        identifiers[variant] = next(iter(values))
    if len(set(identifiers.values())) != len(identifiers):
        raise RobustnessAnalysisError("different variants share a scientific_id")
    return identifiers


def _slide_for_group(group: int) -> str:
    if 100 <= int(group) < 200:
        return "SO_1"
    if 200 <= int(group) < 300:
        return "SO_2"
    raise RobustnessAnalysisError(f"unknown geometry-group slide code: {group}")


def _per_component_rows(
    runs: Sequence[VerifiedRun], *, anchor: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for arm in ARMS:
            block = run.result["arms"][arm]
            if anchor:
                block = block.get("anchor")
                if not isinstance(block, Mapping) or int(block.get("epoch", -1)) != ANCHOR_EPOCH:
                    raise RobustnessAnalysisError(f"anchor result missing: {run.run_id}/{arm}")
            evaluation = block.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise RobustnessAnalysisError("arm evaluation is missing")
            component_rows = evaluation.get("per_component")
            if not isinstance(component_rows, list) or not component_rows:
                raise RobustnessAnalysisError("per-component evaluation is empty")
            groups = [int(item["geometry_group"]) for item in component_rows]
            counts = [int(item["cell_count"]) for item in component_rows]
            mse_values = [
                _finite_number(item["mse"], label="component MSE")
                for item in component_rows
            ]
            mae_values = [
                _finite_number(item["mae"], label="component MAE")
                for item in component_rows
            ]
            if (
                len(groups) != len(set(groups))
                or any(value <= 0 for value in counts)
                or any(value < 0 for value in mse_values + mae_values)
                or int(evaluation.get("component_count", -1)) != len(component_rows)
                or int(evaluation.get("cell_count", -1)) != sum(counts)
                or not np.isclose(
                    _finite_number(
                        evaluation.get("component_equal_mse"),
                        label="component-equal MSE",
                    ),
                    float(np.mean(mse_values)),
                    rtol=METRIC_TOLERANCE,
                    atol=0.0,
                )
                or not np.isclose(
                    _finite_number(
                        evaluation.get("component_equal_mae"),
                        label="component-equal MAE",
                    ),
                    float(np.mean(mae_values)),
                    rtol=METRIC_TOLERANCE,
                    atol=0.0,
                )
            ):
                raise RobustnessAnalysisError(
                    f"per-component aggregate differs: {run.run_id}/{arm}/{anchor}"
                )
            for item in component_rows:
                group = int(item["geometry_group"])
                rows.append(
                    {
                        "variant_id": run.variant_id,
                        "model_seed": run.model_seed,
                        "fold": run.fold,
                        "arm": arm,
                        "anchor": anchor,
                        "geometry_group": group,
                        "slide": _slide_for_group(group),
                        "cell_count": int(item["cell_count"]),
                        "mse": _finite_number(item["mse"], label="component MSE"),
                        "mae": _finite_number(item["mae"], label="component MAE"),
                    }
                )
    return rows


def _verify_component_coverage(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    expected: tuple[int, ...] | None = None
    expected_fold_by_group: dict[int, int] | None = None
    expected_counts_by_variant: dict[str, dict[int, int]] = {}
    arms = set(ARMS)
    variants = {str(row["variant_id"]) for row in rows}
    seeds = {int(row["model_seed"]) for row in rows}
    for variant in variants:
        for seed in seeds:
            arm_sets: dict[str, set[int]] = {}
            arm_folds: dict[str, dict[int, int]] = {}
            arm_counts: dict[str, dict[int, int]] = {}
            for arm in arms:
                selected = [
                    row for row in rows
                    if row["variant_id"] == variant
                    and int(row["model_seed"]) == seed
                    and row["arm"] == arm
                ]
                groups = [int(row["geometry_group"]) for row in selected]
                if len(groups) != len(set(groups)):
                    raise RobustnessAnalysisError(
                        f"duplicate component across folds: {variant}/{seed}/{arm}"
                    )
                if len(groups) != EXPECTED_COMPONENTS:
                    raise RobustnessAnalysisError(
                        f"component coverage is not 27: {variant}/{seed}/{arm}"
                    )
                arm_sets[arm] = set(groups)
                arm_folds[arm] = {
                    int(row["geometry_group"]): int(row["fold"]) for row in selected
                }
                arm_counts[arm] = {
                    int(row["geometry_group"]): int(row["cell_count"])
                    for row in selected
                }
            if len({tuple(sorted(value)) for value in arm_sets.values()}) != 1:
                raise RobustnessAnalysisError(f"arm component axes differ: {variant}/{seed}")
            if len({canonical_json(value) for value in arm_folds.values()}) != 1:
                raise RobustnessAnalysisError(
                    f"arm component-to-fold assignments differ: {variant}/{seed}"
                )
            if len({canonical_json(value) for value in arm_counts.values()}) != 1:
                raise RobustnessAnalysisError(
                    f"arm component cell counts differ: {variant}/{seed}"
                )
            axis = tuple(sorted(next(iter(arm_sets.values()))))
            reference_rows = [
                row for row in rows
                if row["variant_id"] == variant
                and int(row["model_seed"]) == seed
                and row["arm"] == "observed_near"
            ]
            fold_by_group = {
                int(row["geometry_group"]): int(row["fold"])
                for row in reference_rows
            }
            if expected is None:
                expected = axis
                expected_fold_by_group = fold_by_group
            elif expected != axis:
                raise RobustnessAnalysisError("component axis differs across variants/seeds")
            elif expected_fold_by_group != fold_by_group:
                raise RobustnessAnalysisError("component-to-fold assignment differs")
            count_by_group = arm_counts["observed_near"]
            if variant not in expected_counts_by_variant:
                expected_counts_by_variant[variant] = count_by_group
            elif expected_counts_by_variant[variant] != count_by_group:
                raise RobustnessAnalysisError(
                    f"component cell counts differ across seeds: {variant}"
                )
    if expected is None:
        raise RobustnessAnalysisError("no component rows")
    return expected


def _prediction_array(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    arm: str,
    field: str = "mse",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row for row in rows
        if row["variant_id"] == variant
        and row["arm"] == arm
        and (seed is None or int(row["model_seed"]) == seed)
    ]
    by_group: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_group.setdefault(int(row["geometry_group"]), []).append(row)
    if len(by_group) != EXPECTED_COMPONENTS:
        raise RobustnessAnalysisError(f"prediction axis incomplete: {variant}/{arm}/{seed}")
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
    if not np.isfinite(values).all() or np.any(counts <= 0):
        raise RobustnessAnalysisError("prediction aggregation inputs are invalid")
    return groups, values, counts, folds


def _relative_gain(baseline: np.ndarray, candidate: np.ndarray, weights: np.ndarray | None = None) -> float:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or np.any(first < 0) or np.any(second < 0):
        raise RobustnessAnalysisError("paired prediction arrays are invalid")
    if weights is None:
        baseline_mean = float(np.mean(first))
        candidate_mean = float(np.mean(second))
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != first.shape or np.any(weight < 0) or float(weight.sum()) <= 0:
            raise RobustnessAnalysisError("prediction weights are invalid")
        baseline_mean = float(np.average(first, weights=weight))
        candidate_mean = float(np.average(second, weights=weight))
    if baseline_mean <= 0:
        raise RobustnessAnalysisError("relative prediction gain has nonpositive baseline")
    return (baseline_mean - candidate_mean) / baseline_mean


def _paired_slide_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    return_draws: bool = False,
) -> dict[str, Any]:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    group_array = np.asarray(groups, dtype=np.int64)
    if first.shape != second.shape or first.shape != group_array.shape or first.ndim != 1:
        raise RobustnessAnalysisError("bootstrap inputs are not aligned")
    if isinstance(draws, bool) or draws < 1:
        raise RobustnessAnalysisError("bootstrap draws must be positive")
    strata = [np.flatnonzero((group_array // 100) == code) for code in (1, 2)]
    if any(len(value) == 0 for value in strata):
        raise RobustnessAnalysisError("bootstrap requires both slide strata")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(value, size=len(value), replace=True) for value in strata]
        )
        samples[draw] = _relative_gain(first[indices], second[indices])
    result: dict[str, Any] = {
        "draws": int(draws),
        "seed": int(seed),
        "point": _relative_gain(first, second),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "positive_draw_fraction": float(np.mean(samples > 0)),
        "slide_stratified": True,
    }
    if return_draws:
        result["samples"] = samples
    return result


def _matrix_consensus(
    runs: Sequence[VerifiedRun],
    *,
    variant: str,
    arm: str,
    label: str = "selected",
    part: str = "total",
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, np.ndarray], np.ndarray]:
    fold_by_seed: dict[int, dict[int, np.ndarray]] = {}
    for seed in MODEL_SEEDS:
        rows = [run for run in runs if run.variant_id == variant and run.model_seed == seed]
        if {run.fold for run in rows} != set(FOLDS):
            raise RobustnessAnalysisError(f"matrix folds incomplete: {variant}/{seed}")
        fold_by_seed[seed] = {
            run.fold: np.asarray(run.matrices[arm][label][part], dtype=np.float64)
            for run in rows
        }
    seed_matrices = {
        seed: np.mean([fold_by_seed[seed][fold] for fold in FOLDS], axis=0)
        for seed in MODEL_SEEDS
    }
    consensus = np.mean([seed_matrices[seed] for seed in MODEL_SEEDS], axis=0)
    if not np.isfinite(consensus).all():
        raise RobustnessAnalysisError("matrix consensus is nonfinite")
    return fold_by_seed, seed_matrices, consensus


def _summary_with_sign(matrix: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    summary = diagonal_summary(matrix, eligible).as_dict()
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in summary.values()
    ):
        raise RobustnessAnalysisError("Jacobian summary contains a nonfinite statistic")
    signed = np.diag(np.asarray(matrix, dtype=np.float64))[eligible]
    positive = signed[signed > 0]
    negative = signed[signed < 0]
    eligible_count = int(eligible.sum())
    summary.update(
        {
            "signed_diagonal_positive_count": int(len(positive)),
            "signed_diagonal_negative_count": int(len(negative)),
            "signed_diagonal_zero_count": int(np.sum(signed == 0)),
            "median_signed_diagonal": float(np.median(signed)),
            "median_positive_diagonal": (
                None if len(positive) == 0 else float(np.median(positive))
            ),
            "median_absolute_negative_diagonal": (
                None if len(negative) == 0 else float(np.median(np.abs(negative)))
            ),
            "row_top1_count": int(round(float(summary["row_top1_fraction"]) * eligible_count)),
            "row_top1_percent_count": int(
                round(float(summary["row_top1_percent_fraction"]) * eligible_count)
            ),
        }
    )
    return summary


def _fold_stability(fold_matrices: Sequence[np.ndarray], eligible: np.ndarray) -> dict[str, Any]:
    if len(fold_matrices) != 4:
        raise RobustnessAnalysisError("fold stability requires exactly four folds")
    diagonals = np.asarray([np.diag(value)[eligible] for value in fold_matrices])
    correlations = []
    for first in range(4):
        for second in range(first + 1, 4):
            value = float(spearmanr(diagonals[first], diagonals[second]).statistic)
            if not math.isfinite(value):
                raise RobustnessAnalysisError("fold diagonal Spearman is undefined")
            correlations.append(value)
    positive = np.sum(diagonals > 0, axis=0)
    negative = np.sum(diagonals < 0, axis=0)
    fraction = float(np.mean(np.maximum(positive, negative) >= 3))
    return {
        "pairwise_signed_diagonal_spearman": correlations,
        "median_pairwise_signed_diagonal_spearman": float(np.median(correlations)),
        "genes_same_sign_in_at_least_3_folds": int(np.sum(np.maximum(positive, negative) >= 3)),
        "sign_consistent_gene_fraction": fraction,
    }


def _gate_vector(
    *,
    morphology_loss: np.ndarray,
    near_loss: np.ndarray,
    permuted_loss: np.ndarray,
    component_folds: np.ndarray,
    near_matrix: np.ndarray,
    permuted_matrix: np.ndarray,
    near_folds: Sequence[np.ndarray],
    permuted_folds: Sequence[np.ndarray],
    eligible: np.ndarray,
    technical_pass: bool,
) -> dict[str, dict[str, Any]]:
    near_summary = _summary_with_sign(near_matrix, eligible)
    permuted_summary = _summary_with_sign(permuted_matrix, eligible)
    morph_gain = _relative_gain(morphology_loss, near_loss)
    perm_gain = _relative_gain(permuted_loss, near_loss)
    morph_folds = int(sum(
        float(np.mean(near_loss[component_folds == fold]))
        < float(np.mean(morphology_loss[component_folds == fold]))
        for fold in FOLDS
    ))
    perm_folds = int(sum(
        float(np.mean(near_loss[component_folds == fold]))
        < float(np.mean(permuted_loss[component_folds == fold]))
        for fold in FOLDS
    ))
    if float(permuted_summary["median_absolute_diagonal"]) <= 0:
        raise RobustnessAnalysisError("permuted diagonal magnitude is nonpositive")
    near_perm_ratio = float(
        near_summary["median_absolute_diagonal"]
        / permuted_summary["median_absolute_diagonal"]
    )
    fold_ratios = []
    for near_fold, permuted_fold in zip(near_folds, permuted_folds, strict=True):
        numerator = float(np.median(np.abs(np.diag(near_fold)[eligible])))
        denominator = float(np.median(np.abs(np.diag(permuted_fold)[eligible])))
        if denominator <= 0:
            raise RobustnessAnalysisError("fold permuted diagonal magnitude is nonpositive")
        fold_ratios.append(numerator / denominator)
    stability = _fold_stability(near_folds, eligible)
    gates = {
        "near_vs_morphology_prediction": {
            "passed": bool(morph_gain >= 0.02 and morph_folds >= 3),
            "observed": morph_gain,
            "minimum": 0.02,
            "margin": morph_gain - 0.02,
            "folds_favoring": morph_folds,
            "minimum_folds": 3,
            "fold_margin": morph_folds - 3,
        },
        "near_vs_permutation_prediction": {
            "passed": bool(perm_gain >= 0.01 and perm_folds >= 3),
            "observed": perm_gain,
            "minimum": 0.01,
            "margin": perm_gain - 0.01,
            "folds_favoring": perm_folds,
            "minimum_folds": 3,
            "fold_margin": perm_folds - 3,
        },
        "same_name_diagonal_enrichment": {
            "passed": bool(near_summary["diagonal_offdiagonal_ratio"] >= 2.0),
            "observed": near_summary["diagonal_offdiagonal_ratio"],
            "minimum": 2.0,
            "margin": near_summary["diagonal_offdiagonal_ratio"] - 2.0,
        },
        "strict_row_selectivity": {
            "passed": bool(
                near_summary["row_top1_fraction"] >= 0.25
                and near_summary["row_top1_percent_fraction"] >= 0.50
            ),
            "row_top1_fraction": near_summary["row_top1_fraction"],
            "row_top1_count": near_summary["row_top1_count"],
            "row_top1_minimum": 0.25,
            "row_top1_margin": near_summary["row_top1_fraction"] - 0.25,
            "row_top1_percent_fraction": near_summary["row_top1_percent_fraction"],
            "row_top1_percent_count": near_summary["row_top1_percent_count"],
            "row_top1_percent_minimum": 0.50,
            "row_top1_percent_margin": near_summary["row_top1_percent_fraction"] - 0.50,
            "eligible_rows": int(eligible.sum()),
        },
        "near_vs_permutation_diagonal": {
            "passed": bool(near_perm_ratio >= 1.25 and sum(value >= 1.25 for value in fold_ratios) >= 3),
            "observed": near_perm_ratio,
            "minimum": 1.25,
            "margin": near_perm_ratio - 1.25,
            "fold_ratios": fold_ratios,
            "folds_at_or_above": int(sum(value >= 1.25 for value in fold_ratios)),
            "minimum_folds": 3,
        },
        "fold_stability": {
            "passed": bool(
                stability["median_pairwise_signed_diagonal_spearman"] >= 0.70
                and stability["sign_consistent_gene_fraction"] >= 0.75
            ),
            **stability,
            "minimum_median_spearman": 0.70,
            "spearman_margin": stability["median_pairwise_signed_diagonal_spearman"] - 0.70,
            "minimum_sign_fraction": 0.75,
            "sign_fraction_margin": stability["sign_consistent_gene_fraction"] - 0.75,
        },
        "technical_validity": {"passed": bool(technical_pass)},
    }
    return gates


def _three_way_classification(consensus: bool, seed_passes: int) -> str:
    if consensus and seed_passes >= 4:
        return "robust_pass"
    if not consensus and (5 - seed_passes) >= 4:
        return "robust_gate_failure"
    return "seed_sensitive"


def _weighted_matrix_consensus(
    runs: Sequence[VerifiedRun], *, variant: str, label: str
) -> np.ndarray:
    seed_values = []
    for seed in MODEL_SEEDS:
        selected_runs = sorted(
            (
                run for run in runs
                if run.variant_id == variant and run.model_seed == seed
            ),
            key=lambda run: run.fold,
        )
        if len(selected_runs) != 4:
            raise RobustnessAnalysisError(f"weighting matrix coverage differs: {variant}/{seed}/{label}")
        fold_values = [
            np.asarray(run.matrices["observed_near"][label]["total"], dtype=np.float64)
            for run in selected_runs
        ]
        if label == "selected_cell_weighted":
            weights = [
                sum(
                    int(row["cell_count"])
                    for row in run.result["arms"]["observed_near"]["evaluation"]["per_component"]
                )
                for run in selected_runs
            ]
        elif label.startswith("selected_SO_"):
            slide_code = 1 if label.endswith("SO_1") else 2
            weights = [
                sum(
                    int(row["geometry_group"]) // 100 == slide_code
                    for row in run.result["arms"]["observed_near"]["evaluation"]["per_component"]
                )
                for run in selected_runs
            ]
        else:
            raise RobustnessAnalysisError(f"unknown weighting matrix label: {label}")
        if any(weight <= 0 for weight in weights):
            raise RobustnessAnalysisError("weighting sensitivity has an empty fold stratum")
        seed_values.append(np.average(np.stack(fold_values), axis=0, weights=weights))
    return np.mean(seed_values, axis=0)


def _prediction_sensitivities(
    groups: np.ndarray,
    morphology: np.ndarray,
    near: np.ndarray,
    counts: np.ndarray,
    folds: np.ndarray,
) -> dict[str, float]:
    fold_morph = np.asarray([np.mean(morphology[folds == fold]) for fold in FOLDS])
    fold_near = np.asarray([np.mean(near[folds == fold]) for fold in FOLDS])
    slide_codes = groups // 100
    slide_morph = np.asarray([np.mean(morphology[slide_codes == code]) for code in (1, 2)])
    slide_near = np.asarray([np.mean(near[slide_codes == code]) for code in (1, 2)])
    return {
        "component_equal_relative_gain": _relative_gain(morphology, near),
        "cell_weighted_relative_gain": _relative_gain(morphology, near, counts),
        "fold_equal_relative_gain": _relative_gain(fold_morph, fold_near),
        "slide_equal_relative_gain": _relative_gain(slide_morph, slide_near),
    }


def _analyze_variant(
    runs: Sequence[VerifiedRun],
    component_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    eligible = next(run.eligible for run in runs if run.variant_id == variant)
    prediction: dict[str, Any] = {"arms": {}}
    prediction_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm in ARMS:
        groups, values, counts, folds = _prediction_array(
            component_rows, variant=variant, arm=arm
        )
        prediction_arrays[arm] = (groups, values, counts, folds)
        prediction["arms"][arm] = {
            "component_equal_mse": float(np.mean(values)),
            "cell_weighted_mse": float(np.average(values, weights=counts)),
            "fold_equal_mse": float(np.mean([np.mean(values[folds == fold]) for fold in FOLDS])),
            "slide_equal_mse": float(np.mean([np.mean(values[(groups // 100) == code]) for code in (1, 2)])),
            "component_count": int(len(groups)),
            "cell_count": int(round(float(np.sum(counts)))),
        }
    groups, morphology, counts, component_folds = prediction_arrays["morphology_only"]
    near = prediction_arrays["observed_near"][1]
    permuted = prediction_arrays["within_fov_permuted_near"][1]
    prediction["near_vs_morphology"] = {
        **_prediction_sensitivities(groups, morphology, near, counts, component_folds),
        "bootstrap": _paired_slide_bootstrap(morphology, near, groups),
    }
    prediction["near_vs_permutation"] = {
        "component_equal_relative_gain": _relative_gain(permuted, near),
        "bootstrap": _paired_slide_bootstrap(permuted, near, groups),
    }

    fold_by_seed_near, seed_near, consensus_near = _matrix_consensus(
        runs, variant=variant, arm="observed_near"
    )
    fold_by_seed_perm, seed_perm, consensus_perm = _matrix_consensus(
        runs, variant=variant, arm="within_fov_permuted_near"
    )
    fold_consensus_near = [
        np.mean([fold_by_seed_near[seed][fold] for seed in MODEL_SEEDS], axis=0)
        for fold in FOLDS
    ]
    fold_consensus_perm = [
        np.mean([fold_by_seed_perm[seed][fold] for seed in MODEL_SEEDS], axis=0)
        for fold in FOLDS
    ]
    consensus_gates = _gate_vector(
        morphology_loss=morphology,
        near_loss=near,
        permuted_loss=permuted,
        component_folds=component_folds,
        near_matrix=consensus_near,
        permuted_matrix=consensus_perm,
        near_folds=fold_consensus_near,
        permuted_folds=fold_consensus_perm,
        eligible=eligible,
        technical_pass=True,
    )
    seed_gates: dict[str, dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        seed_groups, seed_morph, _, seed_folds = _prediction_array(
            component_rows, variant=variant, arm="morphology_only", seed=seed
        )
        if not np.array_equal(seed_groups, groups):
            raise RobustnessAnalysisError("seed component axis differs")
        seed_near_loss = _prediction_array(
            component_rows, variant=variant, arm="observed_near", seed=seed
        )[1]
        seed_perm_loss = _prediction_array(
            component_rows,
            variant=variant,
            arm="within_fov_permuted_near",
            seed=seed,
        )[1]
        seed_gates[str(seed)] = _gate_vector(
            morphology_loss=seed_morph,
            near_loss=seed_near_loss,
            permuted_loss=seed_perm_loss,
            component_folds=seed_folds,
            near_matrix=seed_near[seed],
            permuted_matrix=seed_perm[seed],
            near_folds=[fold_by_seed_near[seed][fold] for fold in FOLDS],
            permuted_folds=[fold_by_seed_perm[seed][fold] for fold in FOLDS],
            eligible=eligible,
            technical_pass=True,
        )
    classifications: dict[str, Any] = {}
    for gate in consensus_gates:
        passes = int(sum(seed_gates[str(seed)][gate]["passed"] for seed in MODEL_SEEDS))
        classifications[gate] = {
            "classification": _three_way_classification(
                bool(consensus_gates[gate]["passed"]), passes
            ),
            "consensus_passed": bool(consensus_gates[gate]["passed"]),
            "seed_specific_pass_count": passes,
            "seed_specific_fail_count": 5 - passes,
            "required_seed_count": 4,
        }

    arm_summaries: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for arm in MATRIX_ARMS:
        arm_summaries[arm] = {}
        for label in ("selected", "anchor12"):
            arm_summaries[arm][label] = {}
            for part in MATRIX_PARTS:
                _, _, matrix = _matrix_consensus(
                    runs, variant=variant, arm=arm, label=label, part=part
                )
                arrays[f"{variant}_{arm}_{label}_{part}"] = matrix.astype(np.float64)
                arm_summaries[arm][label][part] = _summary_with_sign(matrix, eligible)
    cell_matrix = _weighted_matrix_consensus(
        runs, variant=variant, label="selected_cell_weighted"
    )
    slide_matrices = {
        slide: _weighted_matrix_consensus(
            runs, variant=variant, label=f"selected_{slide}"
        )
        for slide in ("SO_1", "SO_2")
    }
    slide_equal_matrix = np.mean(list(slide_matrices.values()), axis=0)
    jacobian_sensitivities = {
        "fold_equal_component_equal": _summary_with_sign(consensus_near, eligible),
        "fold_equal_cell_weighted": _summary_with_sign(cell_matrix, eligible),
        "slide_equal_component_equal": _summary_with_sign(slide_equal_matrix, eligible),
        "per_slide": {
            slide: _summary_with_sign(matrix, eligible)
            for slide, matrix in slide_matrices.items()
        },
    }
    arrays[f"{variant}_observed_near_selected_cell_weighted_total"] = cell_matrix.astype(np.float64)
    arrays[f"{variant}_observed_near_selected_slide_equal_total"] = slide_equal_matrix.astype(np.float64)
    return (
        {
            "prediction": prediction,
            "jacobian": {
                "consensus": arm_summaries,
                "weighting_sensitivities": jacobian_sensitivities,
            },
            "consensus_gates": consensus_gates,
            "seed_specific_gates": seed_gates,
            "gate_classification": classifications,
        },
        arrays,
    )


def _matched_strata(prevalence: np.ndarray, target_std: np.ndarray) -> list[np.ndarray]:
    first = np.asarray(prevalence, dtype=np.float64)
    second = np.asarray(target_std, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise RobustnessAnalysisError("matched-null covariates are invalid")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RobustnessAnalysisError("matched-null covariates are nonfinite")
    order_first = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    order_second = np.argsort(np.argsort(second, kind="stable"), kind="stable")
    bins_first = np.minimum(9, (10 * order_first) // len(first))
    bins_second = np.minimum(9, (10 * order_second) // len(second))
    codes = bins_first * 10 + bins_second
    groups = [np.flatnonzero(codes == code) for code in sorted(np.unique(codes))]
    large = [value for value in groups if len(value) >= 2]
    singletons = [int(value[0]) for value in groups if len(value) == 1]
    if not large:
        raise RobustnessAnalysisError("matched-null strata cannot form derangements")
    for index in singletons:
        distances = [
            abs(float(first[index] - np.mean(first[group])))
            + abs(float(second[index] - np.mean(second[group])))
            for group in large
        ]
        selected = int(np.argmin(distances))
        large[selected] = np.sort(np.append(large[selected], index))
    return [np.asarray(value, dtype=np.int64) for value in large]


def _gene_label_null(
    matrix: np.ndarray,
    eligible: np.ndarray,
    prevalence: np.ndarray,
    target_std: np.ndarray,
    *,
    draws: int = GENE_NULL_DRAWS,
    seed: int = GENE_NULL_SEED,
) -> dict[str, Any]:
    source = np.asarray(matrix, dtype=np.float64)
    mask = np.asarray(eligible)
    prevalence_array = np.asarray(prevalence, dtype=np.float64)
    std_array = np.asarray(target_std, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[0] != source.shape[1]
        or mask.dtype != np.bool_
        or mask.shape != (source.shape[0],)
        or prevalence_array.shape != mask.shape
        or std_array.shape != mask.shape
        or int(mask.sum()) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(prevalence_array).all()
        or not np.isfinite(std_array).all()
        or isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise RobustnessAnalysisError("gene-label-null inputs are invalid")
    selected = source[np.ix_(mask, mask)]
    prevalence_selected = prevalence_array[mask]
    std_selected = std_array[mask]
    count = selected.shape[0]
    absolute = np.abs(selected)
    ranks = np.empty_like(absolute, dtype=np.int32)
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
    threshold = max(1, int(np.ceil(0.01 * count)))
    observed = {
        "median_absolute_diagonal": float(np.median(np.diag(absolute))),
        "row_top1_fraction": float(np.mean(np.diag(ranks) <= 1)),
        "row_top1_percent_fraction": float(np.mean(np.diag(ranks) <= threshold)),
    }
    families = {
        "full": [np.arange(count, dtype=np.int64)],
        "prevalence_sd_decile_matched": _matched_strata(
            prevalence_selected, std_selected
        ),
    }
    distributions: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, Any] = {}
    row_indices = np.arange(count)
    for family_index, (family, strata) in enumerate(families.items()):
        values = {name: np.empty(draws, dtype=np.float64) for name in GENE_NULL_STATISTICS}
        for draw in range(draws):
            mapping = np.empty(count, dtype=np.int64)
            for stratum_index, stratum in enumerate(strata):
                local = deterministic_derangement(
                    len(stratum),
                    seed=seed + family_index * 10_000_019 + draw * 100_003 + stratum_index,
                )
                mapping[stratum] = stratum[local]
            selected_absolute = absolute[row_indices, mapping]
            selected_rank = ranks[row_indices, mapping]
            values["median_absolute_diagonal"][draw] = np.median(selected_absolute)
            values["row_top1_fraction"][draw] = np.mean(selected_rank <= 1)
            values["row_top1_percent_fraction"][draw] = np.mean(selected_rank <= threshold)
        distributions[family] = values
        summaries[family] = {
            "draws": int(draws),
            "base_seed": int(seed),
            "family_index": family_index,
            "derangement_seed_expression": (
                "base_seed + family_index*10000019 + draw_index*100003 + stratum_index"
            ),
            "eligible_gene_count": count,
            "stratum_count": len(strata),
            "minimum_stratum_size": min(len(value) for value in strata),
            "statistics": {
                name: {
                    "observed": observed[name],
                    "null_mean": float(np.mean(distribution)),
                    "null_95": [
                        float(np.quantile(distribution, 0.025)),
                        float(np.quantile(distribution, 0.975)),
                    ],
                    "upper_tail_p": float(
                        (1 + np.sum(distribution >= observed[name])) / (draws + 1)
                    ),
                }
                for name, distribution in values.items()
            },
        }
    return {"summary": summaries, "distributions": distributions}


def _budget_difference_bootstrap(
    selected_morphology: np.ndarray,
    selected_near: np.ndarray,
    anchor_morphology: np.ndarray,
    anchor_near: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    return_draws: bool = False,
) -> dict[str, Any]:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            selected_morphology,
            selected_near,
            anchor_morphology,
            anchor_near,
        )
    ]
    group_array = np.asarray(groups, dtype=np.int64)
    if any(value.shape != group_array.shape for value in arrays):
        raise RobustnessAnalysisError("budget bootstrap inputs differ")
    strata = [np.flatnonzero(group_array // 100 == code) for code in (1, 2)]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = np.concatenate(
            [rng.choice(value, size=len(value), replace=True) for value in strata]
        )
        selected_gain = _relative_gain(arrays[0][indices], arrays[1][indices])
        anchor_gain = _relative_gain(arrays[2][indices], arrays[3][indices])
        samples[draw] = selected_gain - anchor_gain
    point = _relative_gain(arrays[0], arrays[1]) - _relative_gain(arrays[2], arrays[3])
    result: dict[str, Any] = {
        "draws": int(draws),
        "seed": int(seed),
        "point_selected_minus_anchor12_gain": point,
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "positive_draw_fraction": float(np.mean(samples > 0)),
        "slide_stratified": True,
    }
    if return_draws:
        result["samples"] = samples
    return result


def _selected_near_anchor_fold_support(
    selected_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str = "V0",
) -> tuple[dict[str, int], bool]:
    """Apply the frozen 4/5-seed by 3/4-fold near-versus-anchor rule."""

    fold_counts: dict[str, int] = {}
    for seed in MODEL_SEEDS:
        selected_groups, selected_near, _, selected_folds = _prediction_array(
            selected_rows,
            variant=variant,
            arm="observed_near",
            seed=seed,
        )
        anchor_groups, anchor_near, _, anchor_folds = _prediction_array(
            anchor_rows,
            variant=variant,
            arm="observed_near",
            seed=seed,
        )
        if not np.array_equal(selected_groups, anchor_groups) or not np.array_equal(
            selected_folds, anchor_folds
        ):
            raise RobustnessAnalysisError(
                f"selected/anchor component assignment differs: {variant}/{seed}"
            )
        fold_counts[str(seed)] = int(
            sum(
                float(np.mean(selected_near[selected_folds == fold]))
                < float(np.mean(anchor_near[anchor_folds == fold]))
                for fold in FOLDS
            )
        )
    passed = sum(value >= 3 for value in fold_counts.values()) >= 4
    return fold_counts, bool(passed)


def _budget_attribution(
    runs: Sequence[VerifiedRun],
    selected_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    variant_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    v0 = [run for run in runs if run.variant_id == "V0"]
    trajectories = []
    for run in v0:
        for arm in ("morphology_only", "observed_near"):
            history = run.result["arms"][arm]["validation_history"]
            by_epoch = {int(row["epoch"]): float(row["component_equal_mse"]) for row in history}
            gain_96_to_192 = (by_epoch[96] - by_epoch[192]) / by_epoch[96]
            selected_epoch = int(run.result["arms"][arm]["selected_epoch"])
            saturated = selected_epoch < 192 or gain_96_to_192 <= 0.001
            trajectories.append(
                {
                    "run_id": run.run_id,
                    "model_seed": run.model_seed,
                    "fold": run.fold,
                    "arm": arm,
                    "selected_epoch": selected_epoch,
                    "literal_relative_gain_96_to_192": gain_96_to_192,
                    "saturated": bool(saturated),
                    "saturation_margin": max(192 - selected_epoch, 0),
                    "late_gain_margin": 0.001 - gain_96_to_192,
                }
            )
    if len(trajectories) != 40:
        raise RobustnessAnalysisError("budget audit requires exactly 40 V0 trajectories")
    groups, selected_morph, _, folds = _prediction_array(
        selected_rows, variant="V0", arm="morphology_only"
    )
    selected_near = _prediction_array(
        selected_rows, variant="V0", arm="observed_near"
    )[1]
    anchor_morph = _prediction_array(
        anchor_rows, variant="V0", arm="morphology_only"
    )[1]
    anchor_near = _prediction_array(
        anchor_rows, variant="V0", arm="observed_near"
    )[1]
    bootstrap = _budget_difference_bootstrap(
        selected_morph, selected_near, anchor_morph, anchor_near, groups,
        return_draws=True,
    )
    samples = bootstrap.pop("samples")
    selected_seed_fold_success, selected_seed_fold_support = (
        _selected_near_anchor_fold_support(selected_rows, anchor_rows)
    )
    selected_near_gates = variant_payload["consensus_gates"]
    prediction_explained = bool(
        selected_near_gates["near_vs_morphology_prediction"]["passed"]
        and bootstrap["lower_95"] > 0
        and selected_seed_fold_support
    )

    eligible = v0[0].eligible
    _, _, selected_matrix = _matrix_consensus(
        runs, variant="V0", arm="observed_near", label="selected"
    )
    fold_anchor, _, anchor_matrix = _matrix_consensus(
        runs, variant="V0", arm="observed_near", label="anchor12"
    )
    selected_summary = _summary_with_sign(selected_matrix, eligible)
    anchor_summary = _summary_with_sign(anchor_matrix, eligible)
    row_seed_success: dict[str, int] = {}
    fold_selected, _, _ = _matrix_consensus(
        runs, variant="V0", arm="observed_near", label="selected"
    )
    for seed in MODEL_SEEDS:
        improved = 0
        for fold in FOLDS:
            selected_fold_summary = diagonal_summary(
                fold_selected[seed][fold], eligible
            ).as_dict()
            anchor_fold_summary = diagonal_summary(
                fold_anchor[seed][fold], eligible
            ).as_dict()
            improved += int(
                selected_fold_summary["row_top1_fraction"]
                > anchor_fold_summary["row_top1_fraction"]
                and selected_fold_summary["row_top1_percent_fraction"]
                > anchor_fold_summary["row_top1_percent_fraction"]
            )
        row_seed_success[str(seed)] = improved
    row_explained = bool(
        selected_near_gates["strict_row_selectivity"]["passed"]
        and selected_summary["row_top1_fraction"] > anchor_summary["row_top1_fraction"]
        and selected_summary["row_top1_percent_fraction"]
        > anchor_summary["row_top1_percent_fraction"]
        and sum(value >= 3 for value in row_seed_success.values()) >= 4
    )
    all_saturated = all(row["saturated"] for row in trajectories)
    strong = bool(all_saturated and prediction_explained and row_explained)
    verdict = (
        "strong_budget_explanation"
        if strong
        else ("optimization_inconclusive" if not all_saturated else "budget_explanation_not_supported")
    )
    return (
        {
            "verdict": verdict,
            "trajectory_count": len(trajectories),
            "saturated_trajectory_count": int(sum(row["saturated"] for row in trajectories)),
            "all_trajectories_saturated": all_saturated,
            "trajectories": trajectories,
            "selected_prediction_gain": _relative_gain(selected_morph, selected_near),
            "anchor12_prediction_gain": _relative_gain(anchor_morph, anchor_near),
            "selected_minus_anchor12_gain_bootstrap": bootstrap,
            "selected_seed_folds_near_better": selected_seed_fold_success,
            "prediction_explained": prediction_explained,
            "selected_row_summary": selected_summary,
            "anchor12_row_summary": anchor_summary,
            "seed_folds_improving_both_row_metrics": row_seed_success,
            "row_explained": row_explained,
            "strong_budget_explanation": strong,
            "claim_rule": (
                "optimization_inconclusive if any trajectory is unsaturated; otherwise "
                "only the frozen conjunction supports a strong budget explanation"
            ),
        },
        samples,
    )


def _matching_covariates(
    planned: Sequence[PlannedRun], runs: Sequence[VerifiedRun]
) -> tuple[np.ndarray, np.ndarray]:
    job = next(row for row in planned if row.variant_id == "V0")
    variant = train_wrapper._verify_variant_root(job.materialized.variant_root, project_root=PROJECT_ROOT)
    expression_parts = []
    fold_parts = []
    eligible_parts = []
    for slide in ("SO_1", "SO_2"):
        expression_parts.append(
            np.load(variant.root / slide / "expression_log1p.npy", allow_pickle=False)
        )
        fold_parts.append(np.load(variant.root / slide / "fold.npy", allow_pickle=False))
        eligible_parts.append(
            np.load(variant.root / slide / variant.eligibility_file, allow_pickle=False)
        )
    expression = np.concatenate(expression_parts).astype(np.float64, copy=False)
    folds = np.concatenate(fold_parts).astype(np.int8, copy=False)
    eligible_cells = np.concatenate(eligible_parts).astype(bool, copy=False)
    if (
        expression.shape[1] != EXPECTED_GENES
        or folds.shape != (len(expression),)
        or eligible_cells.shape != folds.shape
        or not np.isfinite(expression).all()
    ):
        raise RobustnessAnalysisError("V0 prevalence inputs are invalid")
    prevalence = np.mean(
        [np.mean(expression[(folds != fold) & eligible_cells] > 0, axis=0) for fold in FOLDS],
        axis=0,
    )
    target_std = np.mean(
        [run.target_std for run in runs if run.variant_id == "V0"], axis=0
    )
    if prevalence.shape != (EXPECTED_GENES,) or target_std.shape != prevalence.shape:
        raise RobustnessAnalysisError("matched-null covariate axes differ")
    return prevalence, target_std


def _prepared_graph_audits(
    planned: Sequence[PlannedRun],
) -> dict[str, Any]:
    """Reverify and publish fixed-source and cross-FOV preparation audits."""

    result: dict[str, Any] = {}
    for variant_id in CORE_VARIANTS:
        roots = {
            row.materialized.variant_root.resolve(strict=True)
            for row in planned
            if row.variant_id == variant_id
        }
        if len(roots) != 1:
            raise RobustnessAnalysisError(
                f"variant uses more than one prepared root: {variant_id}"
            )
        variant = train_wrapper._verify_variant_root(
            next(iter(roots)), project_root=PROJECT_ROOT
        )
        integrity = _strict_json(
            variant.integrity_manifest_path,
            label=f"{variant_id} integrity manifest",
        )
        source = integrity.get("slide_graph_audits")
        if not isinstance(source, Mapping) or set(source) != {"SO_1", "SO_2"}:
            raise RobustnessAnalysisError(
                f"prepared graph audits are missing: {variant_id}"
            )
        slides: dict[str, Any] = {}
        total_fixed = 0
        total_cross_fov_near = 0
        total_cross_fov_annular = 0
        for slide in ("SO_1", "SO_2"):
            audit = source[slide]
            if not isinstance(audit, Mapping):
                raise RobustnessAnalysisError(
                    f"prepared graph audit is invalid: {variant_id}/{slide}"
                )
            active = int(audit.get("active_node_count", -1))
            fixed = int(audit.get("permutation_fixed_source_count", -1))
            fixed_by_fov = audit.get("permutation_fixed_sources_by_fov")
            near_cross = int(audit.get("near_cross_fov_edge_count", -1))
            annular_cross = int(audit.get("annular_cross_fov_edge_count", -1))
            changed = _finite_number(
                audit.get("permutation_source_mapping_changed_fraction"),
                label=f"{variant_id}/{slide} changed-source fraction",
            )
            if (
                active <= 0
                or fixed < 0
                or fixed > active
                or not isinstance(fixed_by_fov, Mapping)
                or any(
                    not isinstance(key, str)
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for key, value in fixed_by_fov.items()
                )
                or sum(int(value) for value in fixed_by_fov.values()) != fixed
                or int(audit.get("permutation_fov_count_with_fixed_sources", -1))
                != len(fixed_by_fov)
                or audit.get("permutation_mapping_policy")
                != "receiver_collision_free_fov_bijection_maximizing_changed_sources"
                or audit.get("permutation_degree_preserved") is not True
                or int(audit.get("permutation_receiver_collisions", -1)) != 0
                or near_cross < 0
                or annular_cross < 0
                or not np.isclose(
                    changed,
                    (active - fixed) / active,
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                raise RobustnessAnalysisError(
                    f"prepared fixed-source audit differs: {variant_id}/{slide}"
                )
            slides[slide] = {
                "active_source_states": active,
                "fixed_source_states": fixed,
                "fixed_source_states_by_fov": dict(fixed_by_fov),
                "source_mapping_changed_fraction": changed,
                "receiver_collisions": 0,
                "receiver_degree_preserved": True,
                "near_cross_fov_directed_edges": near_cross,
                "annular_cross_fov_directed_edges": annular_cross,
            }
            total_fixed += fixed
            total_cross_fov_near += near_cross
            total_cross_fov_annular += annular_cross
        expects_cross_fov = variant_id in {"V1", "V5"}
        if expects_cross_fov != (total_cross_fov_near > 0 and total_cross_fov_annular > 0):
            raise RobustnessAnalysisError(
                f"prepared cross-FOV mechanism differs: {variant_id}"
            )
        result[variant_id] = {
            "slides": slides,
            "total_fixed_source_states": total_fixed,
            "total_near_cross_fov_directed_edges": total_cross_fov_near,
            "total_annular_cross_fov_directed_edges": total_cross_fov_annular,
            "cross_fov_within_component_expected": expects_cross_fov,
        }
    return result


def _cross_variant_classification(
    variants: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    gate_names = tuple(variants["V0"]["gate_classification"])
    for gate in gate_names:
        labels = {
            variant: variants[variant]["gate_classification"][gate]["classification"]
            for variant in PRIMARY_VARIANTS
        }
        unique = set(labels.values())
        result[gate] = {
            "classification": (
                "robust_across_preprocessing"
                if len(unique) == 1
                else "preprocessing_sensitive"
            ),
            "shared_gate_classification": (
                next(iter(unique)) if len(unique) == 1 else None
            ),
            "variant_classifications": labels,
            "all_six_identical": len(unique) == 1,
            "majority_vote_used": False,
            "V6_mechanistic_secondary": variants["V6"]["gate_classification"][gate][
                "classification"
            ],
        }
    return result


def _eligible_gene_rows(
    genes: Sequence[str], matrix: np.ndarray, eligible: np.ndarray
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(eligible)
    selected = np.asarray(matrix, dtype=np.float64)[np.ix_(indices, indices)]
    absolute = np.abs(selected)
    rows = []
    for local, global_index in enumerate(indices):
        rank = 1 + int(np.sum(absolute[local] > absolute[local, local]))
        value = float(selected[local, local])
        rows.append(
            {
                "gene_index": int(global_index),
                "gene": genes[global_index],
                "signed_diagonal": value,
                "absolute_diagonal": abs(value),
                "absolute_row_rank": rank,
                "row_top1": int(rank == 1),
                "row_top1_percent": int(rank <= max(1, int(np.ceil(0.01 * len(indices))))),
            }
        )
    return rows


def _build_payload(
    runs: Sequence[VerifiedRun],
    planned: Sequence[PlannedRun],
    *,
    declared: Sequence[PlannedRun],
    superseded_attempts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    launch: Any,
    scientific_ids: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selected_rows = _per_component_rows(runs, anchor=False)
    anchor_rows = _per_component_rows(runs, anchor=True)
    component_axis = _verify_component_coverage(selected_rows)
    if _verify_component_coverage(anchor_rows) != component_axis:
        raise RobustnessAnalysisError("anchor and selected component axes differ")
    variant_payloads: dict[str, Any] = {}
    aggregate_arrays: dict[str, np.ndarray] = {}
    for variant in CORE_VARIANTS:
        payload, arrays = _analyze_variant(
            runs, selected_rows, variant=variant
        )
        variant_payloads[variant] = payload
        aggregate_arrays.update(arrays)
    cross_variant = _cross_variant_classification(variant_payloads)
    budget, budget_samples = _budget_attribution(
        runs, selected_rows, anchor_rows, variant_payloads["V0"]
    )
    aggregate_arrays["V0_budget_selected_minus_anchor12_bootstrap"] = budget_samples.astype(
        np.float64
    )
    prepared_graph_audits = _prepared_graph_audits(planned)
    prevalence, target_std = _matching_covariates(planned, runs)
    v0_matrix = aggregate_arrays[
        "V0_observed_near_selected_total"
    ]
    eligibility = runs[0].eligible
    gene_null = _gene_label_null(
        v0_matrix, eligibility, prevalence, target_std
    )
    for family, values in gene_null["distributions"].items():
        for statistic, distribution in values.items():
            aggregate_arrays[f"gene_label_null_{family}_{statistic}"] = distribution.astype(
                np.float64
            )
    gene_null_summary = gene_null["summary"]
    genes = runs[0].genes
    gene_rows = _eligible_gene_rows(genes, v0_matrix, eligibility)
    checks = []
    for run in sorted(runs, key=lambda value: (value.variant_id, value.model_seed, value.fold)):
        checks.append(
            {
                "variant_id": run.variant_id,
                "model_seed": run.model_seed,
                "fold": run.fold,
                "attempt": run.attempt,
                "run_id": run.run_id,
                "scientific_id": run.scientific_id,
                "artifact_path": str(run.artifact_path),
                "config_sha256": next(
                    row.config_sha256 for row in planned if row.key == (run.variant_id, run.model_seed, run.fold)
                ),
                "checkpoint_sha256": run.check["checkpoint_sha256"],
                "jacobian_npz_sha256": run.check["jacobian_npz_sha256"],
                "maximum_jacobian_reconstruction_error": run.check[
                    "maximum_jacobian_reconstruction_error"
                ],
                "prediction_rows": run.check["metric"]["prediction_rows"],
                "bundle_verified": True,
                "native_manifest_verified": True,
                "registry_hashes_verified": True,
                "marker_verified": True,
            }
        )
    selected_attempt_rows = [
        {
            "variant_id": run.variant_id,
            "model_seed": run.model_seed,
            "fold": run.fold,
            "attempt": run.attempt,
            "selected": True,
            "registry_status": "completed",
            "artifact_status": "success",
            "run_id": run.run_id,
            "artifact_path": str(run.artifact_path),
            "config_sha256": next(
                row.config_sha256
                for row in planned
                if row.key == (run.variant_id, run.model_seed, run.fold)
            ),
            "plan_sha256": sha256_file(
                next(
                    row.plan_path
                    for row in planned
                    if row.key == (run.variant_id, run.model_seed, run.fold)
                )
            ),
        }
        for run in runs
    ]
    attempt_history = sorted(
        [*map(dict, superseded_attempts), *selected_attempt_rows],
        key=lambda row: (
            str(row["variant_id"]),
            int(row["model_seed"]),
            int(row["fold"]),
            int(row["attempt"]),
        ),
    )
    if len(attempt_history) != len(declared):
        raise RobustnessAnalysisError("attempt inventory lost a declared plan entry")
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "report_name": REPORT_NAME,
        "contract_sha256": launch.contract_sha256,
        "launch_manifest_sha256": launch.sha256,
        "coverage": {
            "variant_ids": list(CORE_VARIANTS),
            "model_seeds": list(MODEL_SEEDS),
            "folds": list(FOLDS),
            "expected_jobs": 140,
            "verified_jobs": len(runs),
            "declared_attempts": len(declared),
            "superseded_unsuccessful_attempts": len(superseded_attempts),
            "failed_jobs": 0,
            "duplicate_slots": 0,
            "component_count": len(component_axis),
            "component_axis": list(component_axis),
            "eligible_gene_count": int(eligibility.sum()),
            "gene_count": len(genes),
        },
        "attempt_history": attempt_history,
        "scientific_ids": dict(scientific_ids),
        "design_notes": {
            "V0": (
                "antecedent-exact observed near/annular inputs and primary cohort; "
                "corrected degree-preserving receiver-collision-free permutation control, "
                "so not a fully exact antecedent replication"
            )
        },
        "variants": variant_payloads,
        "cross_variant_classification": cross_variant,
        "prepared_graph_audits": prepared_graph_audits,
        "budget_attribution": budget,
        "gene_label_null": gene_null_summary,
        "fixed_randomization": {
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
            "gene_label_null": {"draws": GENE_NULL_DRAWS, "seed": GENE_NULL_SEED},
        },
        "exact_row_counts": {
            "run_verification_csv": len(checks),
            "component_metrics_csv": len(selected_rows) + len(anchor_rows),
            "eligible_gene_summary_csv": len(gene_rows),
        },
        "claim_limits": [
            "post_hoc robustness analysis, not independent replication",
            "conditional on 27 geometry components from two observed slides",
            "not patient-level inference, mechanism, causality, or cell-cell communication",
            "prediction losses average all 1000 outputs; Jacobian name-alignment uses only the frozen common 932-gene axis",
            "V0 observed arms and primary cohort are antecedent-exact, but its corrected permutation control is not the antecedent permutation",
            "V6 is mechanistic-secondary and cannot rescue a mixed V0--V5 classification",
            "the deterministic V0 gene-label null is secondary and cannot change or rescue a frozen strict gate",
            "no best-variant, best-seed, or favorable-attempt selection is permitted",
        ],
    }
    return payload, aggregate_arrays, checks, selected_rows + anchor_rows, gene_rows


def _csv_text(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> str:
    field_set = set(fields)
    if len(field_set) != len(fields):
        raise RobustnessAnalysisError("CSV schema contains duplicate fields")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    for index, row in enumerate(rows):
        if set(row) != field_set:
            raise RobustnessAnalysisError(
                f"CSV row schema differs at row {index}: "
                f"missing={sorted(field_set.difference(row))}, "
                f"extra={sorted(set(row).difference(field_set))}"
            )
        writer.writerow(dict(row))
    return handle.getvalue()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_csv_text(rows, fields))


def _report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Same-gene robustness multiverse v1",
        "",
        "This immutable report includes all prespecified V0--V6 core jobs: "
        f"{payload['coverage']['verified_jobs']}/140 verified, with no missing, failed, "
        "or duplicate selected coverage slot; "
        f"{payload['coverage']['superseded_unsuccessful_attempts']} earlier failed/pruned "
        "attempts are retained in attempt_history.",
        "",
        f"V0 design boundary: {payload['design_notes']['V0']}.",
        "",
        "## Gate classifications",
        "",
        "| Gate | V0 consensus | V0 seed classification | Across V0–V5 |",
        "|---|---:|---|---|",
    ]
    v0 = payload["variants"]["V0"]
    for gate, item in v0["gate_classification"].items():
        lines.append(
            f"| {gate} | {str(item['consensus_passed']).lower()} | "
            f"{item['classification']} ({item['seed_specific_pass_count']}/5 pass) | "
            f"{payload['cross_variant_classification'][gate]['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Budget attribution",
            "",
            f"Frozen verdict: `{payload['budget_attribution']['verdict']}`. "
            f"Saturated trajectories: {payload['budget_attribution']['saturated_trajectory_count']}/40.",
            "",
            "## Deterministic gene-label null (secondary)",
            "",
            "Computed only after all 140 core jobs passed verification, on the "
            "frozen 932-gene V0 selected observed-near consensus matrix.",
            "",
            "| Family | Statistic | Observed | Upper-tail p |",
            "|---|---|---:|---:|",
        ]
    )
    for family in GENE_NULL_FAMILIES:
        statistics = payload["gene_label_null"][family]["statistics"]
        for statistic in GENE_NULL_STATISTICS:
            item = statistics[statistic]
            lines.append(
                f"| {family} | {statistic} | {item['observed']:.8g} | "
                f"{item['upper_tail_p']:.8g} |"
            )
    lines.extend(
        [
            "",
            "This secondary null cannot change or rescue a frozen strict gate.",
            "",
            "## Prepared permutation audit",
            "",
            "| Variant | Fixed source states | Near cross-FOV edges | Annular cross-FOV edges |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in CORE_VARIANTS:
        audit = payload["prepared_graph_audits"][variant]
        lines.append(
            f"| {variant} | {audit['total_fixed_source_states']} | "
            f"{audit['total_near_cross_fov_directed_edges']} | "
            f"{audit['total_annular_cross_fov_directed_edges']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in payload["claim_limits"])
    lines.extend(
        [
            "",
            "All ratios and classifications in this document are machine-derived from "
            "the float64 aggregate NPZ and checked across JSON/CSV/NPZ before publication.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_provenance(
    *,
    contract_path: Path,
    launch: Any,
    plan_paths: Sequence[Path],
    planned: Sequence[PlannedRun],
    environment_verification: Mapping[str, Any],
) -> dict[str, Any]:
    paths: set[Path] = {
        Path(__file__).resolve(strict=True),
        contract_path.resolve(strict=True),
        launch.path.resolve(strict=True),
        *(path.resolve(strict=True) for path in plan_paths),
        *(row.config_path.resolve(strict=True) for row in planned),
    }
    for row in planned:
        if row.marker_path.is_file() and not row.marker_path.is_symlink():
            paths.add(row.marker_path.resolve(strict=True))
        variant = train_wrapper._verify_variant_root(
            row.materialized.variant_root, project_root=PROJECT_ROOT
        )
        paths.add(variant.manifest_path.resolve(strict=True))
        paths.add((variant.root / "eligible_genes.npy").resolve(strict=True))
        if row.materialized.pilot_receipt is not None:
            paths.add(row.materialized.pilot_receipt.resolve(strict=True))
    for source in launch.sources:
        paths.add((PROJECT_ROOT / str(source["path"])).resolve(strict=True))
    records = []
    for path in sorted(paths):
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise RobustnessAnalysisError(f"analysis provenance path escapes project: {path}") from error
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "analyzer_path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "sources": records,
        "float_policy": {"aggregate_dtype": "float64", "storage_dtype": "float64"},
        "environment_verification": dict(environment_verification),
        "tolerances": {
            "jacobian_max_abs": JACOBIAN_TOLERANCE,
            "metric_relative": METRIC_TOLERANCE,
        },
    }


def _analysis_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.iterdir() if value.is_file()):
        if path.name in {"analysis_manifest.json", "_SUCCESS"}:
            continue
        records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _write_analysis_manifest(root: Path) -> None:
    files = _analysis_inventory(root)
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "files": files,
        "files_sha256": canonical_sha256(files),
    }
    _write_json(root / "analysis_manifest.json", manifest)
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "analysis_manifest_sha256": sha256_file(root / "analysis_manifest.json"),
        "files_sha256": manifest["files_sha256"],
    }
    success = {"payload": payload, "success_sha256": canonical_sha256(payload)}
    (root / "_SUCCESS").write_text(canonical_json(success) + "\n", encoding="utf-8")


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RobustnessAnalysisError(f"CSV is empty: {path}")
    return len(rows) - 1


def _assert_scalar_close(observed: Any, expected: float, *, label: str) -> None:
    value = _finite_number(observed, label=label)
    if not np.isclose(value, expected, rtol=METRIC_TOLERANCE, atol=1e-15):
        raise RobustnessAnalysisError(
            f"JSON/NPZ coherence failed for {label}: {value} != {expected}"
        )


def _verify_analysis_output(
    root: Path, *, expected: PublicationExpectation
) -> dict[str, Any]:
    try:
        output = root.resolve(strict=True)
    except OSError as error:
        raise RobustnessAnalysisError(f"analysis output is missing: {root}") from error
    if output.is_symlink() or not output.is_dir():
        raise RobustnessAnalysisError("analysis output must be a regular directory")
    manifest = _strict_json(output / "analysis_manifest.json", label="analysis manifest")
    if set(manifest) != {"schema_version", "campaign_id", "files", "files_sha256"}:
        raise RobustnessAnalysisError("analysis manifest keys differ")
    files = manifest["files"]
    if (
        manifest["schema_version"] != 1
        or manifest["campaign_id"] != CAMPAIGN_ID
        or not isinstance(files, list)
        or canonical_sha256(files) != manifest["files_sha256"]
    ):
        raise RobustnessAnalysisError("analysis manifest identity/digest differs")
    declared = set()
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "size_bytes", "sha256"}:
            raise RobustnessAnalysisError("analysis manifest file record differs")
        name = str(item["path"])
        if name in declared or Path(name).name != name:
            raise RobustnessAnalysisError("analysis manifest path is duplicate/unsafe")
        declared.add(name)
        path = output / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(item["size_bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise RobustnessAnalysisError(f"analysis payload hash differs: {name}")
    observed_files = {
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name not in {"analysis_manifest.json", "_SUCCESS"}
    }
    if observed_files != declared:
        raise RobustnessAnalysisError("analysis directory has undeclared/missing files")
    required_publication_files = {
        "aggregate_results.json",
        "aggregate_jacobians.npz",
        "run_verification.csv",
        "component_metrics.csv",
        "eligible_gene_summary.csv",
        "analysis_provenance.json",
        "report.md",
    }
    if declared != required_publication_files:
        raise RobustnessAnalysisError("analysis publication file inventory differs")
    success = _strict_json(output / "_SUCCESS", label="analysis success marker")
    if set(success) != {"payload", "success_sha256"} or not isinstance(success["payload"], Mapping):
        raise RobustnessAnalysisError("analysis success marker keys differ")
    if canonical_sha256(success["payload"]) != success["success_sha256"]:
        raise RobustnessAnalysisError("analysis success marker digest differs")
    if (
        success["payload"].get("analysis_manifest_sha256")
        != sha256_file(output / "analysis_manifest.json")
        or success["payload"].get("files_sha256") != manifest["files_sha256"]
    ):
        raise RobustnessAnalysisError("analysis success marker binding differs")

    payload = _strict_json(output / "aggregate_results.json", label="aggregate results")
    if canonical_json(payload) != canonical_json(expected.payload):
        raise RobustnessAnalysisError(
            "aggregate JSON differs from the independently recomputed current "
            "contract/launch/plan/run authority"
        )
    if (
        payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("coverage", {}).get("verified_jobs") != 140
        or payload.get("coverage", {}).get("expected_jobs") != 140
    ):
        raise RobustnessAnalysisError("aggregate coverage is not complete")
    attempt_history = payload.get("attempt_history")
    declared_attempts = payload.get("coverage", {}).get("declared_attempts")
    superseded_count = payload.get("coverage", {}).get(
        "superseded_unsuccessful_attempts"
    )
    if (
        not isinstance(attempt_history, list)
        or isinstance(declared_attempts, bool)
        or not isinstance(declared_attempts, int)
        or len(attempt_history) != declared_attempts
        or sum(row.get("selected") is True for row in attempt_history) != 140
        or sum(row.get("selected") is False for row in attempt_history)
        != superseded_count
    ):
        raise RobustnessAnalysisError("aggregate attempt history is incomplete")
    grouped_attempts: dict[tuple[str, int, int], list[int]] = {}
    for row in attempt_history:
        if not isinstance(row, Mapping):
            raise RobustnessAnalysisError("aggregate attempt history row is malformed")
        key = (
            str(row.get("variant_id")),
            int(row.get("model_seed", -1)),
            int(row.get("fold", -1)),
        )
        grouped_attempts.setdefault(key, []).append(int(row.get("attempt", -1)))
        if row.get("selected") is False and (
            row.get("registry_status") not in {"failed", "cancelled", "pruned"}
            or row.get("artifact_status") not in {"failed", "pruned"}
        ):
            raise RobustnessAnalysisError(
                "aggregate superseded attempt is not terminal unsuccessful"
            )
    if any(
        sorted(values) != list(range(1, max(values) + 1))
        for values in grouped_attempts.values()
    ):
        raise RobustnessAnalysisError("aggregate attempt histories are noncontiguous")
    csv_expectations = {
        "run_verification.csv": _csv_text(
            expected.checks, RUN_VERIFICATION_FIELDS
        ),
        "component_metrics.csv": _csv_text(
            expected.component_rows, COMPONENT_METRIC_FIELDS
        ),
        "eligible_gene_summary.csv": _csv_text(
            expected.gene_rows, ELIGIBLE_GENE_FIELDS
        ),
    }
    for filename, recomputed in csv_expectations.items():
        try:
            with (output / filename).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                published = handle.read()
        except (OSError, UnicodeError) as error:
            raise RobustnessAnalysisError(
                f"published CSV is unreadable: {filename}"
            ) from error
        if published != recomputed:
            raise RobustnessAnalysisError(
                f"{filename} differs from independently recomputed rows"
            )
    expected_counts = payload["exact_row_counts"]
    for name, filename in (
        ("run_verification_csv", "run_verification.csv"),
        ("component_metrics_csv", "component_metrics.csv"),
        ("eligible_gene_summary_csv", "eligible_gene_summary.csv"),
    ):
        if _csv_row_count(output / filename) != int(expected_counts[name]):
            raise RobustnessAnalysisError(f"published CSV row count differs: {filename}")
    with np.load(output / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        keys = set(archive.files)
        expected_arrays = {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in expected.arrays.items()
        }
        if keys != set(expected_arrays):
            raise RobustnessAnalysisError(
                "aggregate NPZ key inventory differs from independent recomputation"
            )
        for key, recomputed in expected_arrays.items():
            published = np.asarray(archive[key])
            if (
                published.dtype != np.float64
                or recomputed.dtype != np.float64
                or published.shape != recomputed.shape
                or not np.isfinite(published).all()
                or not np.isfinite(recomputed).all()
                or published.tobytes(order="C") != recomputed.tobytes(order="C")
            ):
                raise RobustnessAnalysisError(
                    f"aggregate NPZ differs from independent recomputation: {key}"
                )
        required = {
            f"{variant}_{arm}_{label}_{part}"
            for variant in CORE_VARIANTS
            for arm in MATRIX_ARMS
            for label in ("selected", "anchor12")
            for part in MATRIX_PARTS
        }
        required.update(
            f"gene_label_null_{family}_{statistic}"
            for family in GENE_NULL_FAMILIES
            for statistic in GENE_NULL_STATISTICS
        )
        required.add("V0_budget_selected_minus_anchor12_bootstrap")
        if not required.issubset(keys):
            raise RobustnessAnalysisError(
                f"aggregate NPZ omits matrix keys: {sorted(required.difference(keys))}"
            )
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.dtype != np.float64 or not np.isfinite(value).all():
                raise RobustnessAnalysisError(f"aggregate NPZ is not finite float64: {key}")
        for variant in CORE_VARIANTS:
            matrix = np.asarray(
                archive[f"{variant}_observed_near_selected_total"], dtype=np.float64
            )
            if matrix.shape != (EXPECTED_GENES, EXPECTED_GENES):
                raise RobustnessAnalysisError("aggregate Jacobian shape differs")
            eligible_count = int(payload["coverage"]["eligible_gene_count"])
            # The frozen mask is common, and the eligible-gene CSV carries its exact rows.
            gene_csv = output / "eligible_gene_summary.csv"
            with gene_csv.open("r", encoding="utf-8", newline="") as handle:
                gene_records = list(csv.DictReader(handle))
            indices = np.asarray([int(row["gene_index"]) for row in gene_records], dtype=np.int64)
            if (
                len(indices) != eligible_count
                or len(np.unique(indices)) != eligible_count
                or np.any(indices < 0)
                or np.any(indices >= EXPECTED_GENES)
            ):
                raise RobustnessAnalysisError("eligible-gene CSV axis differs")
            mask = np.zeros(EXPECTED_GENES, dtype=bool)
            mask[indices] = True
            summary = diagonal_summary(matrix, mask).as_dict()
            encoded = payload["variants"][variant]["jacobian"]["consensus"][
                "observed_near"
            ]["selected"]["total"]
            for name in (
                "median_absolute_diagonal",
                "median_absolute_offdiagonal",
                "diagonal_offdiagonal_ratio",
                "row_top1_fraction",
                "row_top1_percent_fraction",
                "positive_diagonal_fraction",
            ):
                _assert_scalar_close(encoded[name], float(summary[name]), label=f"{variant}/{name}")
            gates = payload["variants"][variant]["consensus_gates"]
            _assert_scalar_close(
                gates["same_name_diagonal_enrichment"]["observed"],
                float(summary["diagonal_offdiagonal_ratio"]),
                label=f"{variant}/gate diagonal",
            )
            _assert_scalar_close(
                gates["strict_row_selectivity"]["row_top1_fraction"],
                float(summary["row_top1_fraction"]),
                label=f"{variant}/gate top1",
            )
        fixed = payload.get("fixed_randomization")
        if fixed != {
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
            "gene_label_null": {"draws": GENE_NULL_DRAWS, "seed": GENE_NULL_SEED},
        }:
            raise RobustnessAnalysisError("published randomization authority differs")
        null_payload = payload.get("gene_label_null")
        if not isinstance(null_payload, Mapping) or tuple(null_payload) != GENE_NULL_FAMILIES:
            raise RobustnessAnalysisError("published gene-label-null families differ")
        for family in GENE_NULL_FAMILIES:
            family_payload = null_payload[family]
            if (
                not isinstance(family_payload, Mapping)
                or family_payload.get("draws") != GENE_NULL_DRAWS
                or family_payload.get("base_seed") != GENE_NULL_SEED
                or family_payload.get("eligible_gene_count")
                != int(payload["coverage"]["eligible_gene_count"])
            ):
                raise RobustnessAnalysisError(
                    f"published gene-label-null authority differs: {family}"
                )
            statistics = family_payload.get("statistics")
            if not isinstance(statistics, Mapping) or tuple(statistics) != GENE_NULL_STATISTICS:
                raise RobustnessAnalysisError(
                    f"published gene-label-null statistics differ: {family}"
                )
            for statistic in GENE_NULL_STATISTICS:
                distribution = np.asarray(
                    archive[f"gene_label_null_{family}_{statistic}"],
                    dtype=np.float64,
                )
                encoded = statistics[statistic]
                if distribution.shape != (GENE_NULL_DRAWS,) or not isinstance(
                    encoded, Mapping
                ):
                    raise RobustnessAnalysisError(
                        f"published gene-label-null distribution differs: {family}/{statistic}"
                    )
                _assert_scalar_close(
                    encoded.get("null_mean"),
                    float(np.mean(distribution)),
                    label=f"{family}/{statistic}/null_mean",
                )
                for index, quantile in enumerate((0.025, 0.975)):
                    _assert_scalar_close(
                        encoded.get("null_95", [None, None])[index],
                        float(np.quantile(distribution, quantile)),
                        label=f"{family}/{statistic}/null_95/{index}",
                    )
                _assert_scalar_close(
                    encoded.get("upper_tail_p"),
                    float(
                        (1 + np.sum(distribution >= float(encoded["observed"])))
                        / (GENE_NULL_DRAWS + 1)
                    ),
                    label=f"{family}/{statistic}/upper_tail_p",
                )
        budget_draws = np.asarray(
            archive["V0_budget_selected_minus_anchor12_bootstrap"],
            dtype=np.float64,
        )
        budget_bootstrap = payload.get("budget_attribution", {}).get(
            "selected_minus_anchor12_gain_bootstrap"
        )
        if budget_draws.shape != (BOOTSTRAP_DRAWS,) or not isinstance(
            budget_bootstrap, Mapping
        ):
            raise RobustnessAnalysisError("published budget bootstrap differs")
        for name, recomputed in (
            ("lower_95", float(np.quantile(budget_draws, 0.025))),
            ("upper_95", float(np.quantile(budget_draws, 0.975))),
            ("positive_draw_fraction", float(np.mean(budget_draws > 0))),
        ):
            _assert_scalar_close(
                budget_bootstrap.get(name), recomputed, label=f"budget/{name}"
            )
    provenance = _strict_json(output / "analysis_provenance.json", label="analysis provenance")
    if canonical_json(provenance) != canonical_json(expected.provenance):
        raise RobustnessAnalysisError(
            "analysis provenance differs from the current CLI authority"
        )
    if provenance.get("float_policy") != {
        "aggregate_dtype": "float64",
        "storage_dtype": "float64",
    }:
        raise RobustnessAnalysisError("analysis provenance float policy differs")
    environment = provenance.get("environment_verification")
    if not isinstance(environment, Mapping):
        raise RobustnessAnalysisError(
            "analysis provenance environment verification is missing"
        )
    if (
        environment.get("verified") is not True
        or environment.get("visibility_mode") != "analysis"
        or environment.get("environment_lock_sha256")
        != sha256_file(PROJECT_ROOT / ENVIRONMENT_LOCK_RELATIVE_PATH)
    ):
        raise RobustnessAnalysisError(
            "analysis provenance environment verification differs"
        )
    environment_digest = environment.get("verification_sha256")
    if (
        not isinstance(environment_digest, str)
        or len(environment_digest) != 64
        or canonical_sha256(
            {
                key: value
                for key, value in environment.items()
                if key != "verification_sha256"
            }
        )
        != environment_digest
    ):
        raise RobustnessAnalysisError(
            "analysis provenance environment verification digest differs"
        )
    for source in provenance.get("sources", []):
        path = (PROJECT_ROOT / source["path"]).resolve(strict=True)
        if (
            path.stat().st_size != int(source["size_bytes"])
            or sha256_file(path) != source["sha256"]
        ):
            raise RobustnessAnalysisError(f"analysis source changed: {source['path']}")
    try:
        report = (output / "report.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RobustnessAnalysisError("analysis report is unreadable") from error
    if report != expected.report_markdown:
        raise RobustnessAnalysisError(
            "analysis report differs from the independently recomputed aggregate"
        )
    return {
        "verified": True,
        "output": str(output),
        "manifest_sha256": sha256_file(output / "analysis_manifest.json"),
        "payload_sha256": sha256_file(output / "aggregate_results.json"),
        "file_count": len(files),
    }


def _publish(
    output: Path,
    *,
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    checks: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    gene_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    destination = output.absolute()
    report_markdown = _report_markdown(payload)
    expectation = PublicationExpectation(
        payload=payload,
        arrays=arrays,
        checks=checks,
        component_rows=component_rows,
        gene_rows=gene_rows,
        provenance=provenance,
        report_markdown=report_markdown,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RobustnessAnalysisError(f"analysis output already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        _write_json(staging / "aggregate_results.json", payload)
        np.savez_compressed(
            staging / "aggregate_jacobians.npz",
            **{name: np.asarray(value, dtype=np.float64) for name, value in sorted(arrays.items())},
        )
        _write_csv(
            staging / "run_verification.csv",
            checks,
            RUN_VERIFICATION_FIELDS,
        )
        _write_csv(
            staging / "component_metrics.csv",
            component_rows,
            COMPONENT_METRIC_FIELDS,
        )
        _write_csv(
            staging / "eligible_gene_summary.csv",
            gene_rows,
            ELIGIBLE_GENE_FIELDS,
        )
        _write_json(staging / "analysis_provenance.json", provenance)
        (staging / "report.md").write_text(report_markdown, encoding="utf-8")
        _write_analysis_manifest(staging)
        _verify_analysis_output(staging, expected=expectation)
        os.rename(staging, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _verify_analysis_output(destination, expected=expectation)


def _default_output() -> Path:
    paths = current_paths()
    return paths.report_root / "analyses" / REPORT_NAME


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    contract, launch, environment_verification = _verify_campaign_inputs(
        arguments.contract, arguments.launch_manifest, project_root=project_root
    )
    declared = _collect_plan_entries(arguments.plan, project_root=project_root)
    planned = _validate_coverage(declared)
    registry = Registry(current_paths().state_root / "tracking" / "bagm.sqlite3")
    _verify_registry_attempt_inventory(
        declared,
        registry=registry,
        project_root=project_root,
    )
    superseded_attempts = _verify_superseded_attempts(
        declared,
        planned,
        registry=registry,
        project_root=project_root,
    )
    # This entire loop precedes staging creation: one missing/failed/tampered job
    # therefore produces no partial scientific output.
    runs = [
        _verify_one_run(
            row,
            contract=contract,
            launch=launch,
            registry=registry,
            project_root=project_root,
        )
        for row in planned
    ]
    scientific_ids = _verify_common_scientific_id(runs)
    output = _default_output() if arguments.output is None else arguments.output
    payload, arrays, checks, component_rows, gene_rows = _build_payload(
        runs,
        planned,
        declared=declared,
        superseded_attempts=superseded_attempts,
        contract=contract,
        launch=launch,
        scientific_ids=scientific_ids,
    )
    provenance = _source_provenance(
        contract_path=arguments.contract,
        launch=launch,
        plan_paths=arguments.plan,
        planned=declared,
        environment_verification=environment_verification,
    )
    expectation = PublicationExpectation(
        payload=payload,
        arrays=arrays,
        checks=checks,
        component_rows=component_rows,
        gene_rows=gene_rows,
        provenance=provenance,
        report_markdown=_report_markdown(payload),
    )
    if arguments.verify_only:
        return _verify_analysis_output(output, expected=expectation)
    return _publish(
        output,
        payload=payload,
        arrays=arrays,
        checks=checks,
        component_rows=component_rows,
        gene_rows=gene_rows,
        provenance=provenance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(_arguments(argv))
    except (RobustnessAnalysisError, train_wrapper.RobustnessRunError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
